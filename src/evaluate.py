# -*- coding: utf-8 -*-
"""
evaluate.py
===========
(4) "Danh gia ket qua" trong so do: Accuracy, Precision, Recall, F1-score,
Confusion Matrix.

File nay co 2 vai tro:
  1. Ham `compute_metrics()` duoc `src/train.py` goi lai moi epoch de danh
     gia tren tap validation (early stopping theo macro-F1).
  2. Chay truc tiep (`python -m src.evaluate`) de danh gia mo hinh DA TRAIN
     tren tap TEST (giu rieng, khong dung khi train/early-stop) va xuat
     bao cao + bieu do confusion matrix ra outputs/reports/.
"""

import argparse
import os
import sys
from typing import Dict, List

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

matplotlib.use("Agg")  # khong can man hinh (chay duoc tren server / CLI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
import config as cfg
from src.dataset import dataframe_to_dataset, make_collate_fn
from src.model import PhoBertAMP
from src.preprocessing.metadata_processor import MetadataProcessor
from src.utils import get_device, load_checkpoint, save_json, setup_logger, load_tokenizer


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """Tinh Accuracy/Precision/Recall/F1 (macro & weighted) + theo tung lop.

    Macro-F1 la chi so dung de early-stopping / chon best checkpoint vi no
    coi trong TAT CA cac lop nhu nhau - phu hop bai toan co the mat can
    bang (vd it binh luan "negative" hon "normal").
    """
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    labels_present = sorted(set(y_true) | set(y_pred))
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_present, average=None, zero_division=0
    )
    per_class = {
        cfg.ID2LABEL.get(lbl, str(lbl)): {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for lbl, p, r, f, s in zip(
            labels_present, per_class_p, per_class_r, per_class_f1, per_class_support
        )
    }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(f1_macro),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "weighted_f1": float(f1_weighted),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "per_class": per_class,
    }


def plot_confusion_matrix(
    y_true: List[int], y_pred: List[int], labels: List[str], save_path: str
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix (số lượng)")
    axes[0].set_xlabel("Dự đoán")
    axes[0].set_ylabel("Thực tế")

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Confusion Matrix (chuẩn hoá theo hàng)")
    axes[1].set_xlabel("Dự đoán")
    axes[1].set_ylabel("Thực tế")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def run_inference(model, loader, device) -> tuple:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        metadata = batch["metadata"].to(device)

        logits = model(input_ids, attention_mask, metadata)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        if "label" in batch:
            all_labels.extend(batch["label"].cpu().tolist())

    return all_preds, all_labels, all_probs


def load_model_for_inference(checkpoint_path: str, device):
    """Tra ve (model, model_config) tu 1 checkpoint. `model_config` chua
    ca `phobert_model_name` - dung de tai dung tokenizer tuong ung."""
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model = PhoBertAMP(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["model_config"]


def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình trên tập test")
    parser.add_argument("--test_path", default=cfg.TEST_PATH)
    parser.add_argument("--checkpoint", default=cfg.BEST_MODEL_PATH)
    parser.add_argument("--metadata_scaler", default=cfg.METADATA_SCALER_PATH)
    parser.add_argument("--batch_size", type=int, default=cfg.TRAIN_CONFIG.eval_batch_size)
    parser.add_argument("--max_length", type=int, default=cfg.MODEL_CONFIG.max_seq_length)
    parser.add_argument(
        "--report_path", default=os.path.join(cfg.REPORT_DIR, "test_evaluation.json")
    )
    parser.add_argument(
        "--cm_path", default=os.path.join(cfg.REPORT_DIR, "confusion_matrix.png")
    )
    args = parser.parse_args()

    logger = setup_logger("evaluate")
    device = get_device()
    logger.info(f"Thiết bị: {device}")

    test_df = pd.read_csv(args.test_path)
    logger.info(f"Tập test: {len(test_df)} dòng")

    model, model_config = load_model_for_inference(args.checkpoint, device)
    logger.info(f"Đã tải checkpoint: {args.checkpoint}")

    tokenizer = load_tokenizer(model_config["phobert_model_name"])
    metadata_processor = MetadataProcessor.load(args.metadata_scaler)

    test_dataset = dataframe_to_dataset(
        test_df,
        tokenizer,
        metadata_processor,
        text_column=cfg.TEXT_COLUMN,
        label_column=cfg.LABEL_COLUMN,
        label2id=cfg.LABEL2ID,
        max_length=args.max_length,
        fit_metadata=False,
    )
    collate_fn = make_collate_fn(tokenizer)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    preds, labels, probs = run_inference(model, test_loader, device)
    metrics = compute_metrics(labels, preds)

    logger.info("=" * 60)
    logger.info(f"Accuracy       : {metrics['accuracy']:.4f}")
    logger.info(f"Macro F1       : {metrics['macro_f1']:.4f}")
    logger.info(f"Macro Precision: {metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall   : {metrics['macro_recall']:.4f}")
    logger.info(f"Weighted F1    : {metrics['weighted_f1']:.4f}")
    for label_name, m in metrics["per_class"].items():
        logger.info(
            f"  [{label_name:>8}] precision={m['precision']:.4f} recall={m['recall']:.4f} "
            f"f1={m['f1']:.4f} support={m['support']}"
        )
    logger.info("=" * 60)

    report_text = classification_report(
        labels, preds, target_names=[cfg.ID2LABEL[i] for i in range(len(cfg.LABEL_LIST))],
        zero_division=0,
    )
    logger.info("\n" + report_text)

    # Lưu classification report text ra file
    report_text_path = os.path.join(cfg.REPORT_DIR, "classification_report.txt")
    with open(report_text_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  CLASSIFICATION REPORT - PhoBERT-AMP (Adaptive Metadata-aware PhoBERT)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy       : {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1       : {metrics['macro_f1']:.4f}\n")
        f.write(f"Macro Precision: {metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall   : {metrics['macro_recall']:.4f}\n")
        f.write(f"Weighted F1    : {metrics['weighted_f1']:.4f}\n\n")
        f.write(report_text + "\n")
    logger.info(f"Đã lưu classification report text tại: {report_text_path}")

    save_json(metrics, args.report_path)
    logger.info(f"Đã lưu báo cáo JSON tại: {args.report_path}")

    plot_confusion_matrix(labels, preds, cfg.LABEL_LIST, args.cm_path)
    logger.info(f"Đã lưu biểu đồ confusion matrix tại: {args.cm_path}")


if __name__ == "__main__":
    main()
