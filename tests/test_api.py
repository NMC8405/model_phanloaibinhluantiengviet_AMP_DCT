# -*- coding: utf-8 -*-
"""
tests/test_api.py
=================
Unit tests cho FastAPI endpoints:
  - GET  / (trang giao dien Web HTML)
  - GET  /health (kiem tra trang thai service)
  - POST /predict (phan loai 1 binh luan voi 5 truong metadata)
  - POST /predict/batch (phan loai batch binh luan)
  - Kiem tra auto-fill comment_length, word_count
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api.app as api_module
from api.app import app


@pytest.fixture(scope="module")
def client():
    mock_predictor = MagicMock()
    mock_predictor.device = "cpu"
    mock_predictor.model_config = {"phobert_model_name": "vinai/phobert-base"}

    def fake_predict(text, metadata=None):
        return {
            "text": text,
            "segmented_text": text,
            "label": "positive",
            "confidence": 0.985,
            "probabilities": {"negative": 0.005, "normal": 0.010, "positive": 0.985},
            "gate_score": 0.25,
            "attention_weights": {
                "comment_length": 0.40,
                "word_count": 0.30,
                "has_image": 0.30,
            },
        }

    def fake_predict_batch(texts, metadata_list=None):
        return [fake_predict(t, m if metadata_list else None) for t, m in zip(texts, metadata_list or [{}] * len(texts))]

    mock_predictor.predict.side_effect = fake_predict
    mock_predictor.predict_batch.side_effect = fake_predict_batch

    with patch("api.app.SentimentPredictor", return_value=mock_predictor):
        with TestClient(app) as test_client:
            api_module.predictor = mock_predictor
            yield test_client


def test_root_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Phân Tích Cảm Xúc Bình Luận" in response.text


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["phobert_model_name"] == "vinai/phobert-base"
    assert "positive" in data["labels"]


def test_predict_single(client):
    payload = {
        "text": "Sản phẩm dùng rất tốt, giao hàng nhanh",
        "has_image": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "positive"
    assert data["confidence"] > 0.9
    assert "gate_score" in data
    assert "attention_weights" in data
    assert "has_image" in data["attention_weights"]


def test_predict_auto_fill_lengths(client):
    """Kiem tra comment_length va word_count duoc tu dong tinh neu khong truyen."""
    payload = {
        "text": "Đóng gói đẹp",
        "has_image": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "positive"


def test_predict_batch(client):
    payload = {
        "comments": [
            {"text": "Hàng đẹp", "has_image": 1},
            {"text": "Kém chất lượng", "has_image": 0},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["label"] == "positive"


def test_predict_empty_batch(client):
    response = client.post("/predict/batch", json={"comments": []})
    assert response.status_code == 400
