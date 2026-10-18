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
    rating: Optional[float] = Field(
        None, ge=1, le=5, description="Số sao đánh giá (1-5) [Không dùng trong mô hình để tránh Target Leakage]"
    )
    quality: Optional[float] = Field(
        None, ge=1, le=5, description="Điểm chất lượng sản phẩm (1-5) [Không dùng trong mô hình để tránh Target Leakage]"
    )
    comment_length: Optional[int] = Field(
        None, ge=0, description="Độ dài bình luận (số ký tự). Nếu không truyền, tự tính từ text."
    )
    word_count: Optional[int] = Field(
        None, ge=0, description="Số từ trong bình luận. Nếu không truyền, tự tính từ text."
    )
    has_image: Optional[int] = Field(
        None, ge=0, le=1, description="Bình luận có đính kèm ảnh không (1=có, 0=không)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Giao hàng nhanh, đóng gói cẩn thận, sản phẩm dùng rất tốt!",
                    "has_image": 0,
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
    attention_weights: Optional[Dict[str, float]] = Field(
        None, description="Trọng số Cross-Attention cho từng metadata field "
                          "(cho biết metadata nào liên quan hơn đến nội dung bình luận)"
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

