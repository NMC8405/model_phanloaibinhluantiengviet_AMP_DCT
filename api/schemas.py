# -*- coding: utf-8 -*-
"""
schemas.py
==========
Pydantic schema cho request/response cua FastAPI service.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CommentInput(BaseModel):
    text: str = Field(..., min_length=1, description="Nội dung bình luận cần phân loại")
    rating: Optional[float] = Field(None, ge=1, le=5, description="Số sao đánh giá (1-5)")
    num_likes: Optional[float] = Field(None, ge=0, description="Số lượt thích của bình luận")
    quality: Optional[float] = Field(
        None, ge=1, le=5, description="Điểm chất lượng sản phẩm (1-5)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Giao hàng nhanh, đóng gói cẩn thận, sản phẩm dùng rất tốt!",
                    "rating": 5,
                    "num_likes": 12,
                    "quality": 5.0,
                }
            ]
        }
    }


class PredictionResult(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    gate_score: Optional[float] = Field(
        None, description="Mức độ mô hình sử dụng metadata (0=chỉ text, 1=dùng đầy đủ metadata)"
    )


class BatchCommentInput(BaseModel):
    comments: List[CommentInput]


class BatchPredictionResult(BaseModel):
    results: List[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    device: str
    phobert_model_name: str
    labels: List[str]
