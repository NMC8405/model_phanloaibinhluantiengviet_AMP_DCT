# -*- coding: utf-8 -*-
"""
train.py
========
Script huan luyen mo hinh PhoBERT-AMP (Adaptive Metadata-aware PhoBERT).

Cach chay (tu thu muc goc cua project):
    python -m src.train
    python -m src.train --epochs 20 --batch_size 16 --freeze_phobert

Cac ky thuat duoc dung (phu hop GPU VRAM han che nhu RTX 4060 8GB):
    - Discriminative learning rate: PhoBERT dung LR nho (fine-tune nhe
      nhang), phan CNN+metadata+gate+FC (head) dung LR lon hon.
    - Gradient accumulation: mo phong batch size lon hon.
    - Mixed precision (AMP): giam VRAM + tang toc tren GPU.
    - Warmup + linear decay learning rate schedule.
    - Gradient clipping: chong no gradient.
    - Class weights: bu dap mat can bang giua 3 lop.
    - Early stopping theo macro-F1 tren tap validation.
    - Adaptive Metadata Gate: hoc trong so metadata + L1 regularization.
"""

import argparse
import math
import os
import sys

# Dam bao import config thanh cong bat ke chay tu dau
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Khac phuc UnicodeEncodeError tren Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import config as cfg
from src.dataset import dataframe_to_dataset, make_collate_fn
from src.evaluate import compute_metrics
from src.model import PhoBertAMP
from src.preprocessing.metadata_processor import MetadataProcessor
from src.utils import (
    AverageMeter,
    get_device,
    save_checkpoint,
    save_json,
    set_seed,
    setup_logger,
)


def parse_args():
    p = argparse.ArgumentParser(description="Huấn luyện mô hình PhoBERT-AMP (Adaptive Metadata-aware PhoBERT)")
    p.add_argument("--train_path", default=cfg.TRAIN_PATH)
    p.add_argument("--val_path", default=cfg.VAL_PATH)
    p.add_argument("--phobert_model_name", default=cfg.MODEL_CONFIG.phobert_model_name)
    p.add_argument("--max_length", type=int, default=cfg.MODEL_CONFIG.max_seq_length)

    p.add_argument("--cnn_num_filters", type=int, default=cfg.MODEL_CONFIG.cnn_num_filters)
    p.add_argument("--per_field_embed_dim", type=int, default=cfg.MODEL_CONFIG.per_field_embed_dim)
    p.add_argument("--gate_hidden_dim", type=int, default=cfg.MODEL_CONFIG.gate_hidden_dim)
    p.add_argument("--gate_dropout", type=float, default=cfg.MODEL_CONFIG.gate_dropout)
    p.add_argument("--fc_hidden_dim", type=int, default=cfg.MODEL_CONFIG.fc_hidden_dim)
    p.add_argument("--dropout", type=float, default=cfg.MODEL_CONFIG.dropout)
    p.add_argument("--freeze_phobert", action="store_true", default=cfg.MODEL_CONFIG.freeze_phobert)

    p.add_argument("--epochs", type=int, default=cfg.TRAIN_CONFIG.num_epochs)
    p.add_argument("--batch_size", type=int, default=cfg.TRAIN_CONFIG.batch_size)
    p.add_argument("--eval_batch_size", type=int, default=cfg.TRAIN_CONFIG.eval_batch_size)
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=cfg.TRAIN_CONFIG.gradient_accumulation_steps,
    )
    p.add_argument("--lr_phobert", type=float, default=cfg.TRAIN_CONFIG.lr_phobert)
    p.add_argument("--lr_head", type=float, default=cfg.TRAIN_CONFIG.lr_head)
    p.add_argument("--weight_decay", type=float, default=cfg.TRAIN_CONFIG.weight_decay)
    p.add_argument("--warmup_ratio", type=float, default=cfg.TRAIN_CONFIG.warmup_ratio)
    p.add_argument("--max_grad_norm", type=float, default=cfg.TRAIN_CONFIG.max_grad_norm)
    p.add_argument("--patience", type=int, default=cfg.TRAIN_CONFIG.patience)
    p.add_argument("--seed", type=int, default=cfg.TRAIN_CONFIG.seed)
    p.add_argument("--num_workers", type=int, default=cfg.TRAIN_CONFIG.num_workers)
    p.add_argument(
        "--no_class_weights", action="store_true", help="Tắt trọng số lớp (class weights)"
    )
    p.add_argument("--no_amp", action="store_true", help="Tắt mixed precision (AMP)")
    p.add_argument("--gate_reg_lambda", type=float, default=cfg.TRAIN_CONFIG.gate_reg_lambda,
                   help="Lambda cho L1 regularization trên gate score (0 = tắt)")
    p.add_argument("--gate_max_val", type=float, default=cfg.MODEL_CONFIG.gate_max_val,
                   help="Giá trị lớn nhất có thể của gate score (hard cap)")
    p.add_argument("--metadata_dropout", type=float, default=cfg.MODEL_CONFIG.metadata_dropout,
                   help="Tỉ lệ dropout metadata khi training (0 = tắt)")
    return p.parse_args()



def build_optimizer_and_scheduler(model, args, num_training_steps):
    optimizer = torch.optim.AdamW(
        [
            {"params": model.phobert_parameters(), "lr": args.lr_phobert},
            {"params": model.head_parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    return optimizer, scheduler


@torch.no_grad()
def evaluate_loop(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        metadata = batch["metadata"].to(device)
        labels = batch["label"].to(device)

        # Eval mode -> forward tra ve logits only
        logits = model(input_ids, attention_mask, metadata)
        loss = criterion(logits, labels)
        loss_meter.update(loss.item(), n=labels.size(0))

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = loss_meter.avg
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def plot_training_history(history, save_path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # Loss
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train loss", marker="o")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss theo epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy / F1
    axes[1].plot(epochs, [h["val_accuracy"] for h in history], label="Val accuracy", marker="o")
    axes[1].plot(epochs, [h["val_macro_f1"] for h in history], label="Val macro-F1", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Accuracy / Macro-F1 theo epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Gate score
    axes[2].plot(epochs, [h["avg_gate_score"] for h in history], label="Avg Gate Score",
                 marker="s", color="purple")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Gate Score")
    axes[2].set_title("Mức độ sử dụng Metadata (Gate Score)")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def _create_grad_scaler(use_amp: bool, device_type: str):
    """Tao GradScaler tuong thich ca PyTorch 2.1 va 2.3+."""
    try:
        # PyTorch >= 2.3
        return torch.amp.GradScaler(device=device_type, enabled=use_amp)
    except TypeError:
        # PyTorch 2.1 - 2.2: API cu, chi ho tro CUDA
        if device_type == "cuda":
            return torch.cuda.amp.GradScaler(enabled=use_amp)
        else:
            # CPU khong can GradScaler, tao dummy
            return torch.cuda.amp.GradScaler(enabled=False)


def main():
    args = parse_args()
    cfg.ensure_dirs()
    set_seed(args.seed)
    device = get_device()

    logger = setup_logger("train", log_file=os.path.join(cfg.LOG_DIR, "train.log"))
    logger.info(f"Thiết bị sử dụng: {device}")
    logger.info(f"Tham số chạy: {vars(args)}")

    # ------------------------------------------------------------------ Dữ liệu
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    logger.info(f"Train: {len(train_df)} dòng | Val: {len(val_df)} dòng")
    logger.info(f"Phân bố nhãn (train):\n{train_df[cfg.LABEL_COLUMN].value_counts()}")

    tokenizer = AutoTokenizer.from_pretrained(args.phobert_model_name)
    metadata_processor = MetadataProcessor(schema=cfg.METADATA_SCHEMA, columns=cfg.METADATA_COLUMNS)

    train_dataset = dataframe_to_dataset(
        train_df,
        tokenizer,
        metadata_processor,
        text_column=cfg.TEXT_COLUMN,
        label_column=cfg.LABEL_COLUMN,
        label2id=cfg.LABEL2ID,
        max_length=args.max_length,
        fit_metadata=True,  # CHỈ fit trên train
    )
    metadata_processor.save(cfg.METADATA_SCALER_PATH)
    logger.info(f"Đã lưu metadata scaler tại: {cfg.METADATA_SCALER_PATH}")

    val_dataset = dataframe_to_dataset(
        val_df,
        tokenizer,
        metadata_processor,
        text_column=cfg.TEXT_COLUMN,
        label_column=cfg.LABEL_COLUMN,
        label2id=cfg.LABEL2ID,
        max_length=args.max_length,
        fit_metadata=False,
    )

    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------ Model
    model_config = dict(
        phobert_model_name=args.phobert_model_name,
        num_classes=cfg.MODEL_CONFIG.num_classes,
        cnn_filter_sizes=cfg.MODEL_CONFIG.cnn_filter_sizes,
        cnn_num_filters=args.cnn_num_filters,
        metadata_dim=cfg.MODEL_CONFIG.metadata_dim,
        per_field_embed_dim=args.per_field_embed_dim,
        gate_hidden_dim=args.gate_hidden_dim,
        gate_dropout=args.gate_dropout,
        fc_hidden_dim=args.fc_hidden_dim,
        dropout=args.dropout,
        freeze_phobert=args.freeze_phobert,
        gate_max_val=args.gate_max_val,
        metadata_dropout=args.metadata_dropout,
    )
    model = PhoBertAMP(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Tổng tham số: {n_params:,} | Có thể huấn luyện: {n_trainable:,}")
    logger.info(f"Kiến trúc: PhoBERT-AMP (Adaptive Metadata-aware PhoBERT)")
    logger.info(f"  - Multi-Kernel CNN: filters={cfg.MODEL_CONFIG.cnn_filter_sizes}, num_filters={args.cnn_num_filters}")
    logger.info(f"  - Per-Field Embedding: {cfg.MODEL_CONFIG.metadata_dim} fields × {args.per_field_embed_dim}-dim = {cfg.MODEL_CONFIG.metadata_dim * args.per_field_embed_dim}-dim")
    logger.info(f"  - Adaptive Gate: hidden={args.gate_hidden_dim}, dropout={args.gate_dropout}")
    logger.info(f"  - Gate regularization: lambda={args.gate_reg_lambda}")

    # ------------------------------------------------------------------ Loss
    if not args.no_class_weights:
        train_labels = train_df[cfg.LABEL_COLUMN].map(cfg.LABEL2ID).values
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.arange(len(cfg.LABEL_LIST)), y=train_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
        logger.info(
            "Class weights: "
            + ", ".join(f"{lbl}={w:.3f}" for lbl, w in zip(cfg.LABEL_LIST, class_weights.tolist()))
        )
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ------------------------------------------------------------------ Optimizer / scheduler
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    num_training_steps = num_update_steps_per_epoch * args.epochs
    optimizer, scheduler = build_optimizer_and_scheduler(model, args, num_training_steps)

    use_amp = (not args.no_amp) and device.type == "cuda"
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = _create_grad_scaler(use_amp, autocast_device)
    logger.info(f"Mixed precision (AMP): {use_amp}")

    # ------------------------------------------------------------------ Vòng lặp huấn luyện
    best_val_f1 = -1.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_meter = AverageMeter()
        gate_score_meter = AverageMeter()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                # Training mode -> forward tra ve (logits, gate_scores)
                logits, gate_scores = model(input_ids, attention_mask, metadata)
                ce_loss = criterion(logits, labels)

                # L1 regularization tren gate score: khuyen khich model
                # dung it metadata khi text da du ro rang
                gate_reg = args.gate_reg_lambda * gate_scores.mean()
                loss = ce_loss + gate_reg

                loss_to_backward = loss / args.gradient_accumulation_steps

            scaler.scale(loss_to_backward).backward()
            train_loss_meter.update(ce_loss.item(), n=labels.size(0))
            gate_score_meter.update(gate_scores.mean().item(), n=labels.size(0))

            is_last_step_in_epoch = (step + 1) == len(train_loader)
            if (step + 1) % args.gradient_accumulation_steps == 0 or is_last_step_in_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % cfg.TRAIN_CONFIG.log_every_n_steps == 0 or is_last_step_in_epoch:
                logger.info(
                    f"Epoch {epoch}/{args.epochs} | batch {step + 1}/{len(train_loader)} "
                    f"| loss={train_loss_meter.avg:.4f} | gate={gate_score_meter.avg:.3f} "
                    f"| lr_head={scheduler.get_last_lr()[-1]:.2e}"
                )

        val_metrics = evaluate_loop(model, val_loader, criterion, device)
        logger.info(
            f"[Epoch {epoch}] train_loss={train_loss_meter.avg:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"avg_gate={gate_score_meter.avg:.3f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_meter.avg,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "avg_gate_score": gate_score_meter.avg,
            }
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            save_checkpoint(cfg.BEST_MODEL_PATH, model, model_config, epoch, val_metrics)
            logger.info(f"  -> Lưu best model mới (macro-F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  -> Không cải thiện ({patience_counter}/{args.patience})")

        save_checkpoint(cfg.LAST_MODEL_PATH, model, model_config, epoch, val_metrics)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if patience_counter >= args.patience:
            logger.info(f"Early stopping tại epoch {epoch} (hết kiên nhẫn).")
            break

    history_path = os.path.join(cfg.REPORT_DIR, "training_history.json")
    save_json(history, history_path)
    logger.info(f"Đã lưu lịch sử huấn luyện tại: {history_path}")

    # Lưu training config (model config + hyperparams + dataset stats)
    training_config_info = {
        "model_architecture": "PhoBERT-AMP (Adaptive Metadata-aware PhoBERT)",
        "model_config": model_config,
        "train_args": {k: v for k, v in vars(args).items()},
        "dataset_stats": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "train_label_dist": train_df[cfg.LABEL_COLUMN].value_counts().to_dict(),
            "val_label_dist": val_df[cfg.LABEL_COLUMN].value_counts().to_dict(),
        },
        "total_params": n_params,
        "trainable_params": n_trainable,
        "best_epoch": max(history, key=lambda h: h["val_macro_f1"])["epoch"],
        "best_val_macro_f1": best_val_f1,
        "final_avg_gate_score": history[-1]["avg_gate_score"],
        "device": str(device),
    }
    config_path = os.path.join(cfg.REPORT_DIR, "training_config.json")
    save_json(training_config_info, config_path)
    logger.info(f"Đã lưu training config tại: {config_path}")

    try:
        curves_path = os.path.join(cfg.REPORT_DIR, "training_curves.png")
        plot_training_history(history, curves_path)
        logger.info(f"Đã lưu biểu đồ training curves tại: {curves_path}")
    except Exception as e:
        logger.warning(f"Không vẽ được biểu đồ training curve: {e}")

    logger.info(f"HOÀN TẤT. Best validation macro-F1 = {best_val_f1:.4f}")
    logger.info(f"Chạy `python -m src.evaluate` để đánh giá trên tập TEST.")


if __name__ == "__main__":
    main()
