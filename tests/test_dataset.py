# -*- coding: utf-8 -*-
"""
tests/test_dataset.py
=====================
Unit tests cho PyTorch Dataset va Collate Function:
  - SentimentDataset
  - make_collate_fn
  - dataframe_to_dataset
"""

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src.dataset import SentimentDataset, dataframe_to_dataset, make_collate_fn
from src.preprocessing.metadata_processor import MetadataProcessor


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()

    def fake_call(text, truncation=True, max_length=128, padding=False):
        words = text.split() if text else ["dummy"]
        ids = [hash(w) % 1000 + 1 for w in words][:max_length]
        mask = [1] * len(ids)
        return {"input_ids": ids, "attention_mask": mask}

    def fake_pad(features, padding=True, return_tensors="pt"):
        max_l = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        for f in features:
            cur_ids = f["input_ids"].tolist() if isinstance(f["input_ids"], torch.Tensor) else list(f["input_ids"])
            cur_mask = f["attention_mask"].tolist() if isinstance(f["attention_mask"], torch.Tensor) else list(f["attention_mask"])
            pad_len = max_l - len(cur_ids)
            input_ids.append(cur_ids + [0] * pad_len)
            attention_mask.append(cur_mask + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    tok.side_effect = fake_call
    tok.pad.side_effect = fake_pad
    return tok


def test_sentiment_dataset(mock_tokenizer):
    texts = ["áo đẹp", "giao hàng chậm", "bình thường"]
    metadata = np.random.randn(3, cfg.METADATA_DIM).astype(np.float32)
    labels = [2, 0, 1]

    ds = SentimentDataset(texts=texts, metadata=metadata, tokenizer=mock_tokenizer, labels=labels)

    assert len(ds) == 3
    item0 = ds[0]
    assert "input_ids" in item0
    assert "attention_mask" in item0
    assert "metadata" in item0
    assert "label" in item0
    assert item0["metadata"].shape == (cfg.METADATA_DIM,)
    assert item0["label"].item() == 2


def test_collate_fn(mock_tokenizer):
    texts = ["áo đẹp", "giao hàng rất chậm và sản phẩm bị rách"]
    metadata = np.random.randn(2, cfg.METADATA_DIM).astype(np.float32)
    labels = [2, 0]

    ds = SentimentDataset(texts=texts, metadata=metadata, tokenizer=mock_tokenizer, labels=labels)
    collate = make_collate_fn(mock_tokenizer)

    batch = collate([ds[0], ds[1]])
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "metadata" in batch
    assert "label" in batch
    assert batch["metadata"].shape == (2, cfg.METADATA_DIM)
    assert batch["label"].shape == (2,)
    # Kiem tra ca hai mau duoc pad cung chieu dai
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] == batch["attention_mask"].shape[1]


def test_dataframe_to_dataset(mock_tokenizer):
    df = pd.DataFrame({
        "comment": ["sản phẩm tốt", "hàng tệ", "ổn"],
        "label": ["positive", "negative", "normal"],
        "comment_length": [12, 7, 2],
        "word_count": [3, 2, 1],
        "has_image": [1, 0, 0],
    })

    processor = MetadataProcessor(cfg.METADATA_SCHEMA, cfg.METADATA_COLUMNS)
    ds = dataframe_to_dataset(
        df,
        tokenizer=mock_tokenizer,
        metadata_processor=processor,
        text_column="comment",
        label_column="label",
        label2id=cfg.LABEL2ID,
        max_length=64,
        fit_metadata=True,
    )

    assert len(ds) == 3
    item = ds[0]
    assert item["metadata"].shape == (cfg.METADATA_DIM,)
    assert item["label"].item() == cfg.LABEL2ID["positive"]
