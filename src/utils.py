# -*- coding: utf-8 -*-
"""
utils.py
========
Cac ham tien ich dung chung: set seed (tai lap ket qua), thiet lap logger,
luu/doc checkpoint.
"""

import json
import logging
import os
import random
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Co dinh seed cho random/numpy/torch (CPU+GPU) de ket qua tai lap duoc."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger  # tranh add handler trung khi goi lai nhieu lan

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    model_config: Dict[str, Any],
    epoch: int,
    metrics: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str, map_location=None) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint tại '{path}'. Hãy chạy src/train.py trước."
        )
    return torch.load(path, map_location=map_location, weights_only=False)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class AverageMeter:
    """Theo doi trung binh chay (running average) cua 1 dai luong (vd loss)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


def load_tokenizer(model_name: str) -> Any:
    """Tải AutoTokenizer ưu tiên cache local để không bị delay/timeout mạng."""
    from transformers import AutoTokenizer
    try:
        # Thử tải trực tiếp từ cache local trước để phản hồi tức thì
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        # Nếu chưa có trong cache thì mới tải từ HuggingFace Hub
        try:
            return AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Không thể tải tokenizer '{model_name}' từ HuggingFace Hub lẫn local cache: {e}"
            ) from e

