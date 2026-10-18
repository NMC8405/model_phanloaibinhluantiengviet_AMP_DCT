# -*- coding: utf-8 -*-
"""
api/app.py
==========
FastAPI microservice phuc vu mo hinh PhoBERT-AMP (Adaptive Metadata-aware
PhoBERT) - CUNG MO HINH kien truc (FastAPI microservice, goi tu backend
chinh qua HTTP) nhu cach tich hop mot AI service voi ung dung web
(vd ASP.NET Core MVC goi sang service Python qua REST API).

Chay (tu thu muc goc cua project):
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Sau khi chay, xem tai lieu API tu dong tai: http://localhost:8000/docs

Vi du goi tu C# (HttpClient) - tham khao doan sau trong README.md.
"""

import os
import sys
from contextlib import asynccontextmanager

# Dam bao import config thanh cong bat ke chay tu dau
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config as cfg
from api.schemas import (
    BatchCommentInput,
    BatchPredictionResult,
    CommentInput,
    HealthResponse,
    PredictionResult,
)
from src.predict import SentimentPredictor

from fastapi.responses import HTMLResponse

# Instance predictor dung chung, khoi tao 1 LAN DUY NHAT luc startup (nap
# model + tokenizer + scaler vao RAM/VRAM 1 lan, tranh nap lai moi request
# gay cham va ton tai nguyen).
predictor: SentimentPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = SentimentPredictor(
        checkpoint_path=cfg.BEST_MODEL_PATH,
        metadata_scaler_path=cfg.METADATA_SCALER_PATH,
    )
    print(f"[startup] Đã tải mô hình lên thiết bị: {predictor.device}")
    yield
    predictor = None


app = FastAPI(
    title="API phân loại cảm xúc bình luận (PhoBERT-AMP)",
    description="Phân loại bình luận tiếng Việt thành 3 nhãn: negative, normal, positive. "
                "Sử dụng Adaptive Metadata Gate để tự động quyết định mức độ dùng metadata.",
    version="2.0.0",
    lifespan=lifespan,
)

# Cho phep goi tu frontend/backend khac domain (vd ASP.NET Core MVC chay
# port khac). Trong production nen gioi han allow_origins ve dung domain
# cua backend thay vi "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa sẵn sàng")
    return HealthResponse(
        status="ok",
        device=str(predictor.device),
        phobert_model_name=predictor.model_config["phobert_model_name"],
        labels=cfg.LABEL_LIST,
    )


@app.post("/predict", response_model=PredictionResult)
def predict(item: CommentInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa sẵn sàng")

    metadata = item.model_dump(exclude={"text", "rating", "quality"}, exclude_none=True)
    # Tu dong tinh comment_length va word_count tu text neu chua truyen
    if "comment_length" not in metadata:
        metadata["comment_length"] = len(item.text)
    if "word_count" not in metadata:
        metadata["word_count"] = len(item.text.split())
    result = predictor.predict(item.text, metadata)
    return PredictionResult(
        text=result["text"],
        label=result["label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        gate_score=result.get("gate_score"),
        attention_weights=result.get("attention_weights"),
    )


@app.post("/predict/batch", response_model=BatchPredictionResult)
def predict_batch(payload: BatchCommentInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa sẵn sàng")
    if not payload.comments:
        raise HTTPException(status_code=400, detail="Danh sách 'comments' rỗng")

    texts = [c.text for c in payload.comments]
    metadata_list = [c.model_dump(exclude={"text", "rating", "quality"}, exclude_none=True) for c in payload.comments]
    # Tu dong tinh comment_length va word_count cho tung binh luan neu chua truyen
    for text, meta in zip(texts, metadata_list):
        if "comment_length" not in meta:
            meta["comment_length"] = len(text)
        if "word_count" not in meta:
            meta["word_count"] = len(text.split())

    results = predictor.predict_batch(texts, metadata_list)
    return BatchPredictionResult(
        results=[
            PredictionResult(
                text=r["text"],
                label=r["label"],
                confidence=r["confidence"],
                probabilities=r["probabilities"],
                gate_score=r.get("gate_score"),
                attention_weights=r.get("attention_weights"),
            )
            for r in results
        ]
    )


@app.get("/", response_class=HTMLResponse)
def root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>API phân loại cảm xúc bình luận đang chạy.</h3>"


if __name__ == "__main__":
    import uvicorn
    # Chạy uvicorn trực tiếp khi thực thi file này bằng python api/app.py
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)

