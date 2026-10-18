# -*- coding: utf-8 -*-
"""
predict.py
==========
Suy luan (inference) tren binh luan MOI, CHUA qua xu ly - dung mo hinh
PhoBERT-AMP v2 (Adaptive Metadata-aware PhoBERT + Cross-Attention).

QUAN TRONG: predict.py tu chay lai DUNG pipeline tien xu ly da dung luc
chuan bi du lieu train (TextCleaner.clean -> WordSegmenter.segment) truoc
khi dua vao tokenizer. Neu bo qua buoc nay, ket qua se sai lech do
"train-serve skew".

Ket qua du doan bao gom:
  - `gate_score` - muc do mo hinh su dung metadata cho tung binh luan
    (0 = chi dung text, 1 = dung toan bo metadata)
  - `attention_weights` - trong so cross-attention cho tung metadata field
    (cho biet metadata nao lien quan hon voi noi dung binh luan)

Dung nhu 1 module (import SentimentPredictor) hoac chay truc tiep tu CLI:
    python -m src.predict --text "Giao hàng nhanh, sản phẩm dùng tốt" --has_image 1
    python -m src.predict --input_csv data/raw/new_comments.csv --output_csv outputs/reports/predictions.csv
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Union

# Dam bao import config thanh cong bat ke chay tu dau
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Khac phuc UnicodeEncodeError tren Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import torch

import config as cfg
from src.evaluate import load_model_for_inference
from src.preprocessing.metadata_processor import MetadataProcessor
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.word_segmenter import WordSegmenter
from src.utils import get_device, load_tokenizer


class SentimentPredictor:
    def __init__(
        self,
        checkpoint_path: str = cfg.BEST_MODEL_PATH,
        metadata_scaler_path: str = cfg.METADATA_SCALER_PATH,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.model, self.model_config = load_model_for_inference(checkpoint_path, self.device)
        self.tokenizer = load_tokenizer(self.model_config["phobert_model_name"])
        self.metadata_processor = MetadataProcessor.load(metadata_scaler_path)
        self.metadata_columns = getattr(self.metadata_processor, "columns", cfg.METADATA_COLUMNS)

        self.cleaner = TextCleaner()
        self.segmenter = WordSegmenter()
        self.max_length = cfg.MODEL_CONFIG.max_seq_length

    def _preprocess_text(self, raw_text: str) -> str:
        cleaned = self.cleaner.clean(raw_text)
        segmented = self.segmenter.segment(cleaned)
        return segmented

    @torch.no_grad()
    def predict(self, text: str, metadata: Optional[Dict[str, Union[int, float]]] = None) -> Dict:
        """Du doan cho MOT binh luan.

        Args:
            text: van ban THO (chua xu ly gi ca) - vd nguoi dung nhap truc tiep.
            metadata: dict cac truong metadata (vd {"has_image": 0}).
                Thieu truong nao se duoc dien gia tri hoc tu tap train (xem MetadataProcessor).
                comment_length va word_count se duoc TU DONG tinh tu text
                neu khong truyen.

        Returns:
            dict: {"label": str, "confidence": float, "probabilities": {label: prob}}
        """
        if metadata is None:
            metadata = {}
        # Tu dong tinh comment_length va word_count tu text goc neu chua co
        if "comment_length" not in metadata:
            metadata["comment_length"] = len(text)
        if "word_count" not in metadata:
            metadata["word_count"] = len(text.split())
        results = self.predict_batch([text], [metadata])
        return results[0]

    @torch.no_grad()
    def predict_batch(
        self, texts: List[str], metadata_list: Optional[List[Dict]] = None
    ) -> List[Dict]:
        if metadata_list is None:
            metadata_list = [{} for _ in texts]
        assert len(texts) == len(metadata_list)

        # Tu dong tinh comment_length va word_count cho tung binh luan neu chua co
        for i, (text, meta) in enumerate(zip(texts, metadata_list)):
            if "comment_length" not in meta:
                meta["comment_length"] = len(text)
            if "word_count" not in meta:
                meta["word_count"] = len(text.split())

        segmented_texts = [self._preprocess_text(t) for t in texts]
        metadata_df = pd.DataFrame(metadata_list)
        metadata_array = self.metadata_processor.transform(metadata_df)

        encodings = self.tokenizer(
            segmented_texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        metadata_tensor = torch.tensor(metadata_array, dtype=torch.float)

        self.model.eval()
        # Dung predict_with_gate de lay xac suat, gate_score va attention_weights
        probs, gate_scores, attn_weights = self.model.predict_with_gate(
            encodings["input_ids"].to(self.device),
            encodings["attention_mask"].to(self.device),
            metadata_tensor.to(self.device),
        )
        probs = probs.cpu().numpy()
        gate_scores = gate_scores.cpu().numpy()
        attn_weights = attn_weights.cpu().numpy()

        results = []
        num_attn_fields = attn_weights.shape[1]
        if hasattr(self, "metadata_columns") and len(self.metadata_columns) == num_attn_fields:
            field_names = self.metadata_columns
        elif len(cfg.METADATA_COLUMNS) == num_attn_fields:
            field_names = cfg.METADATA_COLUMNS
        else:
            field_names = [f"field_{k}" for k in range(num_attn_fields)]

        for i in range(len(texts)):
            row_probs = {cfg.ID2LABEL[j]: float(probs[i, j]) for j in range(len(cfg.LABEL_LIST))}
            pred_id = int(probs[i].argmax())
            # Chuyen attention weights thanh dict {field_name: weight}
            row_attn = {
                field_names[j]: float(attn_weights[i, j])
                for j in range(num_attn_fields)
            }
            results.append(
                {
                    "text": texts[i],
                    "segmented_text": segmented_texts[i],
                    "label": cfg.ID2LABEL[pred_id],
                    "confidence": float(probs[i, pred_id]),
                    "probabilities": row_probs,
                    "gate_score": float(gate_scores[i, 0]),
                    "attention_weights": row_attn,
                }
            )
        return results


def _cli():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Dự đoán nhãn cảm xúc cho bình luận mới")
    parser.add_argument("--checkpoint", default=cfg.BEST_MODEL_PATH)
    parser.add_argument("--metadata_scaler", default=cfg.METADATA_SCALER_PATH)

    parser.add_argument("--text", type=str, help="Một bình luận duy nhất cần dự đoán")
    parser.add_argument("--comment_length", type=int, default=None,
                       help="Độ dài bình luận (số ký tự). Tự tính nếu không truyền.")
    parser.add_argument("--word_count", type=int, default=None,
                       help="Số từ trong bình luận. Tự tính nếu không truyền.")
    parser.add_argument("--has_image", type=int, default=None, choices=[0, 1],
                       help="Bình luận có ảnh hay không (1=có, 0=không)")

    parser.add_argument("--input_csv", type=str, help="File CSV chứa nhiều bình luận cần dự đoán")
    parser.add_argument("--text_column", type=str, default=cfg.TEXT_COLUMN)
    parser.add_argument("--output_csv", type=str, default=os.path.join(cfg.REPORT_DIR, "predictions.csv"))

    args = parser.parse_args()

    predictor = SentimentPredictor(
        checkpoint_path=args.checkpoint, metadata_scaler_path=args.metadata_scaler
    )

    if args.text:
        metadata = {}
        if args.comment_length is not None:
            metadata["comment_length"] = args.comment_length
        if args.word_count is not None:
            metadata["word_count"] = args.word_count
        if args.has_image is not None:
            metadata["has_image"] = args.has_image

        result = predictor.predict(args.text, metadata)
        print(f"Bình luận : {result['text']}")
        print(f"Nhãn dự đoán: {result['label']}  (độ tin cậy: {result['confidence']:.2%})")
        print(f"Gate Score : {result['gate_score']:.3f} (mức sử dụng metadata: 0=chỉ text, 1=dùng đầy đủ)")
        if result.get("attention_weights"):
            attn_str = ", ".join(f"{k}={v:.3f}" for k, v in result["attention_weights"].items())
            print(f"Attention  : [{attn_str}] (trọng số cross-attention cho từng metadata field)")
        print("Chi tiết xác suất:")
        for label, prob in result["probabilities"].items():
            print(f"  - {label:>8}: {prob:.2%}")

    elif args.input_csv:
        df = pd.read_csv(args.input_csv)
        texts = df[args.text_column].fillna("").tolist()
        metadata_cols = [c for c in cfg.METADATA_COLUMNS if c in df.columns]
        metadata_list = df[metadata_cols].to_dict("records") if metadata_cols else None

        results = predictor.predict_batch(texts, metadata_list)
        out_df = pd.DataFrame(
            [
                {
                    "text": r["text"],
                    "predicted_label": r["label"],
                    "confidence": r["confidence"],
                    **{f"prob_{k}": v for k, v in r["probabilities"].items()},
                }
                for r in results
            ]
        )
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        out_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        print(f"Đã dự đoán {len(out_df)} bình luận. Kết quả lưu tại: {args.output_csv}")

    else:
        parser.error("Cần truyền --text HOẶC --input_csv")


if __name__ == "__main__":
    _cli()
