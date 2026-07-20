# -*- coding: utf-8 -*-
"""
scripts/generate_report.py
===========================
Tạo báo cáo toàn diện về kết quả huấn luyện mô hình PhoBERT+CNN+Metadata.

Bao gồm:
  1. Bảng tổng quan Dataset (số mẫu, phân bố nhãn)
  2. Bảng cấu hình Model & Hyperparameters
  3. Bảng Training History (loss, accuracy, F1 theo epoch)
  4. Bảng Classification Report (Precision/Recall/F1 theo từng lớp)
  5. Biểu đồ Training Curves nâng cao (loss, accuracy, F1, precision, recall)
  6. Biểu đồ phân bố dataset (pie chart + bar chart)
  7. Biểu đồ so sánh per-class metrics
  8. Biểu đồ Confusion Matrix (giữ nguyên từ evaluate.py)
  9. Tổng hợp tất cả vào 1 file ảnh summary

Cách chạy:
    python -m scripts.generate_report
"""

import json
import os
import sys

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

import config as cfg


# ============================================================================
# Tiện ích đọc dữ liệu
# ============================================================================

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv_stats(path):
    """Đọc CSV và trả về (num_rows, label_counts_dict)."""
    if not os.path.exists(path):
        return 0, {}
    df = pd.read_csv(path)
    counts = df[cfg.LABEL_COLUMN].value_counts().to_dict() if cfg.LABEL_COLUMN in df.columns else {}
    return len(df), counts


# ============================================================================
# 1. Bảng tổng quan Dataset
# ============================================================================

def plot_dataset_overview(ax, train_stats, val_stats, test_stats):
    """Vẽ bảng tổng quan dataset lên axes."""
    ax.axis("off")
    ax.set_title("[1] Tong quan Dataset", fontsize=14, fontweight="bold", pad=15)

    headers = ["Tập dữ liệu", "Số mẫu", "Negative", "Normal", "Positive"]
    rows = []
    for name, (n, counts) in [("Train", train_stats), ("Val", val_stats), ("Test", test_stats)]:
        rows.append([
            name, str(n),
            str(counts.get("negative", 0)),
            str(counts.get("normal", 0)),
            str(counts.get("positive", 0)),
        ])

    total_n = sum(s[0] for s in [train_stats, val_stats, test_stats])
    rows.append(["TỔNG", str(total_n), "", "", ""])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Tô màu header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Tô màu dòng tổng
    for j in range(len(headers)):
        table[len(rows), j].set_facecolor("#ecf0f1")
        table[len(rows), j].set_text_props(fontweight="bold")


# ============================================================================
# 2. Bảng cấu hình Model
# ============================================================================

def plot_model_config_table(ax):
    """Vẽ bảng cấu hình model."""
    ax.axis("off")
    ax.set_title("[2] Cau hinh Mo hinh & Hyperparameters", fontsize=14, fontweight="bold", pad=15)

    mc = cfg.MODEL_CONFIG
    tc = cfg.TRAIN_CONFIG

    rows = [
        ["PhoBERT backbone", mc.phobert_model_name],
        ["Max sequence length", str(mc.max_seq_length)],
        ["CNN filter sizes", str(mc.cnn_filter_sizes)],
        ["CNN num filters", str(mc.cnn_num_filters)],
        ["Metadata dim", str(mc.metadata_dim)],
        ["Metadata embed dim", str(mc.metadata_embed_dim)],
        ["FC hidden dim", str(mc.fc_hidden_dim)],
        ["Dropout", str(mc.dropout)],
        ["Freeze PhoBERT", str(mc.freeze_phobert)],
        ["Batch size (effective)", f"{tc.batch_size} × {tc.gradient_accumulation_steps} = {tc.batch_size * tc.gradient_accumulation_steps}"],
        ["Num epochs (max)", str(tc.num_epochs)],
        ["Patience (early stop)", str(tc.patience)],
        ["LR PhoBERT", str(tc.lr_phobert)],
        ["LR Head", str(tc.lr_head)],
        ["Weight decay", str(tc.weight_decay)],
        ["Warmup ratio", str(tc.warmup_ratio)],
        ["Mixed precision (AMP)", str(tc.use_amp)],
        ["Class weights", str(tc.use_class_weights)],
    ]

    headers = ["Tham số", "Giá trị"]
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#2980b9")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows) + 1):
        color = "#f8f9fa" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i, j].set_facecolor(color)


# ============================================================================
# 3. Bảng Training History
# ============================================================================

def plot_training_history_table(ax, history):
    """Vẽ bảng training history."""
    ax.axis("off")
    ax.set_title("[3] Lich su Huan luyen theo Epoch", fontsize=14, fontweight="bold", pad=15)

    headers = ["Epoch", "Train Loss", "Val Loss", "Val Accuracy", "Val Macro-F1"]
    rows = []
    best_f1_epoch = max(history, key=lambda h: h.get("val_macro_f1", 0))["epoch"]

    for h in history:
        marker = " *" if h["epoch"] == best_f1_epoch else ""
        rows.append([
            f"{h['epoch']}{marker}",
            f"{h['train_loss']:.4f}",
            f"{h['val_loss']:.4f}",
            f"{h['val_accuracy'] * 100:.2f}%",
            f"{h['val_macro_f1'] * 100:.2f}%",
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#27ae60")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best epoch row
    best_idx = best_f1_epoch  # 1-indexed epoch = row index in table
    for j in range(len(headers)):
        table[best_idx, j].set_facecolor("#d5f5e3")
        table[best_idx, j].set_text_props(fontweight="bold")


# ============================================================================
# 4. Bảng Classification Report (Test)
# ============================================================================

def plot_classification_report_table(ax, test_metrics):
    """Vẽ bảng classification report."""
    ax.axis("off")
    ax.set_title("[4] Ket qua Danh gia tren Tap Test", fontsize=14, fontweight="bold", pad=15)

    headers = ["Lớp", "Precision", "Recall", "F1-Score", "Support"]
    rows = []

    per_class = test_metrics.get("per_class", {})
    for label_name in cfg.LABEL_LIST:
        if label_name in per_class:
            m = per_class[label_name]
            rows.append([
                label_name.capitalize(),
                f"{m['precision'] * 100:.2f}%",
                f"{m['recall'] * 100:.2f}%",
                f"{m['f1'] * 100:.2f}%",
                str(m['support']),
            ])

    rows.append(["", "", "", "", ""])
    rows.append([
        "Macro Avg",
        f"{test_metrics['macro_precision'] * 100:.2f}%",
        f"{test_metrics['macro_recall'] * 100:.2f}%",
        f"{test_metrics['macro_f1'] * 100:.2f}%",
        "",
    ])
    rows.append([
        "Weighted Avg",
        f"{test_metrics['weighted_precision'] * 100:.2f}%",
        f"{test_metrics['weighted_recall'] * 100:.2f}%",
        f"{test_metrics['weighted_f1'] * 100:.2f}%",
        "",
    ])
    rows.append([
        "Accuracy",
        "",
        "",
        f"{test_metrics['accuracy'] * 100:.2f}%",
        "",
    ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#8e44ad")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Tô đậm các dòng avg
    for i in [len(rows) - 2, len(rows) - 1, len(rows)]:
        for j in range(len(headers)):
            table[i, j].set_facecolor("#f4ecf7")
            table[i, j].set_text_props(fontweight="bold")


# ============================================================================
# 5. Biểu đồ Training Curves nâng cao
# ============================================================================

def plot_enhanced_training_curves(fig, gs_row, history):
    """Vẽ biểu đồ training curves nâng cao."""
    epochs = [h["epoch"] for h in history]

    ax1 = fig.add_subplot(gs_row[0])
    ax1.plot(epochs, [h["train_loss"] for h in history], "o-", color="#e74c3c",
             label="Train Loss", linewidth=2, markersize=6)
    ax1.plot(epochs, [h["val_loss"] for h in history], "s-", color="#3498db",
             label="Val Loss", linewidth=2, markersize=6)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Loss theo Epoch", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(epochs)

    # Annotate min val loss
    min_val_loss = min(history, key=lambda h: h["val_loss"])
    ax1.annotate(
        f"Min: {min_val_loss['val_loss']:.4f}",
        xy=(min_val_loss["epoch"], min_val_loss["val_loss"]),
        xytext=(10, 15), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#3498db"),
        fontsize=9, color="#3498db", fontweight="bold",
    )

    ax2 = fig.add_subplot(gs_row[1])
    ax2.plot(epochs, [h["val_accuracy"] * 100 for h in history], "o-", color="#2ecc71",
             label="Val Accuracy", linewidth=2, markersize=6)
    ax2.plot(epochs, [h["val_macro_f1"] * 100 for h in history], "s-", color="#e67e22",
             label="Val Macro-F1", linewidth=2, markersize=6)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score (%)", fontsize=11)
    ax2.set_title("Accuracy & Macro-F1 theo Epoch", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(epochs)

    # Annotate best F1
    best_f1 = max(history, key=lambda h: h["val_macro_f1"])
    ax2.annotate(
        f"Best: {best_f1['val_macro_f1'] * 100:.2f}%",
        xy=(best_f1["epoch"], best_f1["val_macro_f1"] * 100),
        xytext=(10, -20), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#e67e22"),
        fontsize=9, color="#e67e22", fontweight="bold",
    )


# ============================================================================
# 6. Biểu đồ phân bố Dataset
# ============================================================================

def plot_dataset_distribution(fig, gs_row, train_stats, val_stats, test_stats):
    """Biểu đồ phân bố dataset: pie chart + stacked bar."""
    # Pie chart - overall
    ax1 = fig.add_subplot(gs_row[0])
    all_counts = {}
    for _, counts in [train_stats, val_stats, test_stats]:
        for k, v in counts.items():
            all_counts[k] = all_counts.get(k, 0) + v

    labels_pie = [k.capitalize() for k in cfg.LABEL_LIST if k in all_counts]
    sizes = [all_counts.get(k, 0) for k in cfg.LABEL_LIST if k in all_counts]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    explode = [0.03] * len(labels_pie)

    ax1.pie(sizes, explode=explode, labels=labels_pie, colors=colors, autopct="%1.1f%%",
            shadow=True, startangle=90, textprops={"fontsize": 10})
    ax1.set_title("Phân bố nhãn (toàn bộ)", fontsize=13, fontweight="bold")

    # Grouped bar chart - per split
    ax2 = fig.add_subplot(gs_row[1])
    x = np.arange(len(cfg.LABEL_LIST))
    width = 0.25

    train_vals = [train_stats[1].get(l, 0) for l in cfg.LABEL_LIST]
    val_vals = [val_stats[1].get(l, 0) for l in cfg.LABEL_LIST]
    test_vals = [test_stats[1].get(l, 0) for l in cfg.LABEL_LIST]

    ax2.bar(x - width, train_vals, width, label="Train", color="#3498db", alpha=0.85)
    ax2.bar(x, val_vals, width, label="Val", color="#e67e22", alpha=0.85)
    ax2.bar(x + width, test_vals, width, label="Test", color="#2ecc71", alpha=0.85)

    ax2.set_xlabel("Nhãn", fontsize=11)
    ax2.set_ylabel("Số mẫu", fontsize=11)
    ax2.set_title("Phân bố nhãn theo tập", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.capitalize() for l in cfg.LABEL_LIST])
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [ax2.containers[0], ax2.containers[1], ax2.containers[2]]:
        ax2.bar_label(bars, fontsize=8, padding=2)


# ============================================================================
# 7. Biểu đồ per-class metrics
# ============================================================================

def plot_per_class_metrics(ax, test_metrics):
    """So sánh Precision/Recall/F1 theo từng lớp."""
    per_class = test_metrics.get("per_class", {})
    labels_bar = [l.capitalize() for l in cfg.LABEL_LIST if l in per_class]

    precision_vals = [per_class[l]["precision"] * 100 for l in cfg.LABEL_LIST if l in per_class]
    recall_vals = [per_class[l]["recall"] * 100 for l in cfg.LABEL_LIST if l in per_class]
    f1_vals = [per_class[l]["f1"] * 100 for l in cfg.LABEL_LIST if l in per_class]

    x = np.arange(len(labels_bar))
    width = 0.25

    ax.bar(x - width, precision_vals, width, label="Precision", color="#9b59b6", alpha=0.85)
    ax.bar(x, recall_vals, width, label="Recall", color="#1abc9c", alpha=0.85)
    ax.bar(x + width, f1_vals, width, label="F1-Score", color="#e74c3c", alpha=0.85)

    ax.set_xlabel("Lớp", fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("So sánh Precision / Recall / F1 theo lớp (Test)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=max(0, min(precision_vals + recall_vals + f1_vals) - 5))

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)


# ============================================================================
# MAIN: Tổng hợp tất cả
# ============================================================================

def main():
    cfg.ensure_dirs()
    report_dir = cfg.REPORT_DIR

    print("=" * 60)
    print("  GENERATE TRAINING REPORT - PhoBERT+CNN+Metadata")
    print("=" * 60)

    # --- Load data ---
    history_path = os.path.join(report_dir, "training_history.json")
    eval_path = os.path.join(report_dir, "test_evaluation.json")

    if not os.path.exists(history_path):
        print(f"❌ Không tìm thấy {history_path}. Hãy chạy train trước!")
        return
    if not os.path.exists(eval_path):
        print(f"❌ Không tìm thấy {eval_path}. Hãy chạy evaluate trước!")
        return

    history = _load_json(history_path)
    test_metrics = _load_json(eval_path)

    train_stats = _load_csv_stats(cfg.TRAIN_PATH)
    val_stats = _load_csv_stats(cfg.VAL_PATH)
    test_stats = _load_csv_stats(cfg.TEST_PATH)

    print(f"  ✓ Training history: {len(history)} epochs")
    print(f"  ✓ Test metrics: accuracy={test_metrics['accuracy']:.4f}")
    print(f"  ✓ Dataset: train={train_stats[0]}, val={val_stats[0]}, test={test_stats[0]}")

    # ================================================================
    # REPORT 1: Bảng tổng hợp (tables_summary.png)
    # ================================================================
    print("\n[1/4] Tạo bảng tổng hợp...")
    fig_tables = plt.figure(figsize=(16, 28))
    fig_tables.suptitle(
        "BÁO CÁO HUẤN LUYỆN - PhoBERT + CNN + Metadata\nPhân loại cảm xúc bình luận thương mại điện tử",
        fontsize=16, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(4, 1, figure=fig_tables, hspace=0.35, top=0.95, bottom=0.03)

    ax1 = fig_tables.add_subplot(gs[0])
    plot_dataset_overview(ax1, train_stats, val_stats, test_stats)

    ax2 = fig_tables.add_subplot(gs[1])
    plot_model_config_table(ax2)

    ax3 = fig_tables.add_subplot(gs[2])
    plot_training_history_table(ax3, history)

    ax4 = fig_tables.add_subplot(gs[3])
    plot_classification_report_table(ax4, test_metrics)

    tables_path = os.path.join(report_dir, "tables_summary.png")
    fig_tables.savefig(tables_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_tables)
    print(f"  ✓ Đã lưu: {tables_path}")

    # ================================================================
    # REPORT 2: Biểu đồ training curves nâng cao
    # ================================================================
    print("[2/4] Tạo biểu đồ training curves...")
    fig_curves = plt.figure(figsize=(14, 5.5))
    gs_curves = gridspec.GridSpec(1, 2, figure=fig_curves, wspace=0.3)
    plot_enhanced_training_curves(fig_curves, [gs_curves[0], gs_curves[1]], history)
    fig_curves.suptitle("Biểu đồ Quá trình Huấn luyện", fontsize=15, fontweight="bold", y=1.02)

    curves_path = os.path.join(report_dir, "enhanced_training_curves.png")
    fig_curves.savefig(curves_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_curves)
    print(f"  ✓ Đã lưu: {curves_path}")

    # ================================================================
    # REPORT 3: Phân bố dataset
    # ================================================================
    print("[3/4] Tạo biểu đồ phân bố dataset...")
    fig_dist = plt.figure(figsize=(14, 5.5))
    gs_dist = gridspec.GridSpec(1, 2, figure=fig_dist, wspace=0.3)
    plot_dataset_distribution(fig_dist, [gs_dist[0], gs_dist[1]], train_stats, val_stats, test_stats)
    fig_dist.suptitle("Phân bố Dữ liệu", fontsize=15, fontweight="bold", y=1.02)

    dist_path = os.path.join(report_dir, "dataset_distribution.png")
    fig_dist.savefig(dist_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_dist)
    print(f"  ✓ Đã lưu: {dist_path}")

    # ================================================================
    # REPORT 4: Per-class metrics
    # ================================================================
    print("[4/4] Tạo biểu đồ per-class metrics...")
    fig_pc, ax_pc = plt.subplots(figsize=(10, 5.5))
    plot_per_class_metrics(ax_pc, test_metrics)
    fig_pc.tight_layout()

    pc_path = os.path.join(report_dir, "per_class_metrics.png")
    fig_pc.savefig(pc_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_pc)
    print(f"  ✓ Đã lưu: {pc_path}")

    # ================================================================
    # Tổng kết
    # ================================================================
    print("\n" + "=" * 60)
    print("  ✅ HOÀN TẤT TẠO BÁO CÁO!")
    print(f"  📁 Thư mục: {report_dir}")
    print("  📄 Các file đã tạo:")
    print(f"     - tables_summary.png          (bảng tổng hợp)")
    print(f"     - enhanced_training_curves.png (biểu đồ training)")
    print(f"     - dataset_distribution.png     (phân bố dữ liệu)")
    print(f"     - per_class_metrics.png        (metrics theo lớp)")
    print(f"     - confusion_matrix.png         (đã có từ evaluate)")
    print(f"     - training_curves.png          (đã có từ train)")
    print("=" * 60)


if __name__ == "__main__":
    main()
