# -*- coding: utf-8 -*-
"""
config.py
=========
Cấu hình tập trung cho toàn bộ project.
Mọi hyperparameter, đường dẫn, tên cột... đều khai báo ở đây để dễ chỉnh sửa
mà không phải sửa rải rác trong nhiều file.
"""

import os
import platform
from dataclasses import dataclass, field
from typing import List, Dict

# ---------------------------------------------------------------------------
# Đường dẫn thư mục
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

RAW_DATA_PATH = os.path.join(DATA_RAW_DIR, "comments.csv")
TRAIN_PATH = os.path.join(DATA_PROCESSED_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_PROCESSED_DIR, "val.csv")
TEST_PATH = os.path.join(DATA_PROCESSED_DIR, "test.csv")
METADATA_SCALER_PATH = os.path.join(DATA_PROCESSED_DIR, "metadata_scaler.joblib")

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pt")


# ---------------------------------------------------------------------------
# Cột dữ liệu (schema)
# ---------------------------------------------------------------------------
TEXT_COLUMN = "comment"
LABEL_COLUMN = "label"

# Nhãn phân loại - PHẢI khớp thứ tự index dùng khi train (0,1,2)
LABEL_LIST = ["negative", "normal", "positive"]  # tiêu cực, bình thường, tích cực
LABEL2ID: Dict[str, int] = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(LABEL_LIST)}

# Metadata đi kèm mỗi bình luận: rating, chất lượng sản phẩm
# (num_likes đã được loại bỏ theo yêu cầu để tránh nhiễu và không ảnh hưởng mô hình)
# type: "numeric" (co scale) hoac "binary" (0/1, khong can scale)
METADATA_SCHEMA: Dict[str, str] = {
    "rating": "numeric",             # so sao danh gia 1-5
    # "num_likes": "numeric_log",    # da bo luot thich
    "quality": "numeric",            # diem chat luong san pham 1-5 (tu cot 'Chat luong san pham')
}
METADATA_COLUMNS: List[str] = list(METADATA_SCHEMA.keys())
METADATA_DIM: int = len(METADATA_COLUMNS)


# ---------------------------------------------------------------------------
# Cau hinh mo hinh
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    phobert_model_name: str = "vinai/phobert-base"
    max_seq_length: int = 256  # Tang len 256 cho du lieu 15k+ (binh luan co the dai hon)

    # Nhanh CNN (tren output cua PhoBERT) - Multi-Kernel CNN voi BatchNorm
    cnn_filter_sizes: List[int] = field(default_factory=lambda: [2, 3, 4, 5])
    cnn_num_filters: int = 128  # So filter moi kernel size

    # Nhanh metadata - Per-Field Embedding
    metadata_dim: int = METADATA_DIM           # so truong metadata (2: rating, quality)
    per_field_embed_dim: int = 32              # moi truong metadata embed thanh 32-dim
    
    @property
    def metadata_embed_dim(self) -> int:
        return self.metadata_dim * self.per_field_embed_dim

    # Adaptive Metadata Gate
    gate_hidden_dim: int = 128                 # hidden dim cua gate MLP
    gate_dropout: float = 0.2                  # dropout trong gate MLP
    gate_max_val: float = 0.5                  # Hard cap cho gate_score, ep model luon dung it nhat 50% text
    metadata_dropout: float = 0.3              # Ti le dropout metadata de buoc model phai hoc text khi train

    # Fusion head
    fc_hidden_dim: int = 256  # Tang tu 128 len 256 cho du lieu lon hon
    dropout: float = 0.3

    num_classes: int = len(LABEL_LIST)
    freeze_phobert: bool = False  # True neu VRAM khong du de fine-tune toan bo PhoBERT


# ---------------------------------------------------------------------------
# Cau hinh huan luyen - toi uu cho dataset 15k+
# ---------------------------------------------------------------------------
_IS_WINDOWS = platform.system() == "Windows"


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4  # batch hieu dung = batch_size * step nay = 32

    num_epochs: int = 20  # Tang len 20 epoch de model hoc ky hon
    patience: int = 4     # Tang tu 3 len 4 cho du lieu lon hon (it overfit hon)

    lr_phobert: float = 2e-5      # learning rate rieng cho PhoBERT (fine-tune)
    lr_head: float = 2e-4         # learning rate cho CNN + metadata + FC head (giam tu 1e-3 cho dataset lon 56k)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1     # Tang tu 0.06 len 0.1 cho du lieu lon hon
    max_grad_norm: float = 1.0

    # Adaptive Metadata Gate regularization
    gate_reg_lambda: float = 0.015  # L1 penalty tren gate score - gia tri thiet ke chuan
                                    # giup gate dao dong can bang (0.15 - 0.35) tren tap 56k

    use_class_weights: bool = True
    use_amp: bool = True          # mixed precision - quan trong voi GPU 8GB (RTX 4060)

    # Windows: num_workers > 0 co the gay loi multiprocessing neu khong co
    # freeze_support(). Tu dong dat 0 tren Windows de tranh loi.
    num_workers: int = 0 if _IS_WINDOWS else 2

    log_every_n_steps: int = 10   # In tien trinh moi 10 batch de theo doi de hon


MODEL_CONFIG = ModelConfig()
TRAIN_CONFIG = TrainConfig()


def ensure_dirs() -> None:
    """Tao cac thu muc can thiet neu chua ton tai."""
    for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, CHECKPOINT_DIR, LOG_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)
