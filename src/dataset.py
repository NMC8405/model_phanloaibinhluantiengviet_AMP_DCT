# -*- coding: utf-8 -*-
"""
dataset.py
==========
PyTorch Dataset cho bai toan phan loai binh luan 3 nhan (PhoBERT + CNN +
Metadata). Nhan mot van ban DA duoc tien xu ly + tach tu (Word Segmentation)
san (xem `scripts/prepare_data.py`), tokenize bang PhoBERT tokenizer ngay
tai __getitem__, va ghep voi vector metadata da chuan hoa san.

Dung "dynamic padding" (pad theo do dai lon nhat TRONG TUNG BATCH thay vi
pad co dinh theo max_length) de tiet kiem tinh toan/VRAM - quan trong voi
GPU 8GB nhu RTX 4060.
"""

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        metadata: np.ndarray,
        tokenizer,
        max_length: int = 128,
        labels: Optional[Sequence[int]] = None,
    ):
        assert len(texts) == metadata.shape[0], (
            f"So dong text ({len(texts)}) va metadata ({metadata.shape[0]}) khong khop"
        )
        if labels is not None:
            assert len(labels) == len(texts), "So dong labels va texts khong khop"

        self.texts = list(texts)
        self.metadata = metadata.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = list(labels) if labels is not None else None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx] if self.texts[idx] else ""
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        item = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "metadata": torch.tensor(self.metadata[idx], dtype=torch.float),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def make_collate_fn(tokenizer):
    """Tra ve mot collate_fn dung `tokenizer.pad` de dynamic-pad input_ids/
    attention_mask theo do dai lon nhat trong batch."""

    def collate_fn(batch: List[dict]):
        features = [
            {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
            for item in batch
        ]
        padded = tokenizer.pad(features, padding=True, return_tensors="pt")

        result = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "metadata": torch.stack([item["metadata"] for item in batch]),
        }
        if "label" in batch[0]:
            result["label"] = torch.stack([item["label"] for item in batch])
        return result

    return collate_fn


def dataframe_to_dataset(
    df: pd.DataFrame,
    tokenizer,
    metadata_processor,
    text_column: str,
    label_column: Optional[str],
    label2id: dict,
    max_length: int = 128,
    fit_metadata: bool = False,
) -> SentimentDataset:
    """Tien ich: tao SentimentDataset truc tiep tu 1 DataFrame da xu ly.

    Args:
        fit_metadata: True CHI khi day la tap TRAIN (goi fit_transform).
            Voi val/test/inference luon de False (chi transform bang scaler
            da hoc tu train) de tranh ro ri du lieu (data leakage).
    """
    texts = df[text_column].fillna("").tolist()

    if fit_metadata:
        metadata = metadata_processor.fit_transform(df)
    else:
        metadata = metadata_processor.transform(df)

    labels = None
    if label_column is not None and label_column in df.columns:
        labels = df[label_column].map(label2id).tolist()
        if any(l is None for l in labels):
            bad = df[label_column][df[label_column].map(label2id).isna()].unique()
            raise ValueError(f"Có nhãn không nằm trong LABEL_LIST: {bad}")

    return SentimentDataset(
        texts=texts,
        metadata=metadata,
        tokenizer=tokenizer,
        max_length=max_length,
        labels=labels,
    )
