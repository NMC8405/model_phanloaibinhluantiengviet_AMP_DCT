# -*- coding: utf-8 -*-
"""
predict.py
==========
Suy luan (inference) tren binh luan MOI, CHUA qua xu ly - dung mo hinh
PhoBERT-AMP (Adaptive Metadata-aware PhoBERT).

QUAN TRONG: predict.py tu chay lai DUNG pipeline tien xu ly da dung luc
chuan bi du lieu train (TextCleaner.clean -> WordSegmenter.segment) truoc
khi dua vao tokenizer. Neu bo qua buoc nay, ket qua se sai lech do
"train-serve skew".

Ket qua du doan bao gom `gate_score` - muc do mo hinh su dung metadata
cho tung binh luan (0 = chi dung text, 1 = dung toan bo metadata).

Dung nhu 1 module (import SentimentPredictor) hoac chay truc tiep tu CLI:
    python -m src.predict --text "Giao hàng nhanh, sản phẩm dùng tốt" --rating 5 --num_likes 12
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
from src.utils import get_device
from transformers import AutoTokenizer


class SentimentPredictor:
    def __init__(
        self,
        checkpoint_path: str = cfg.BEST_MODEL_PATH,
        metadata_scaler_path: str = cfg.METADATA_SCALER_PATH,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.model, self.model_config = load_model_for_inference(checkpoint_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config["phobert_model_name"])
        self.metadata_processor = MetadataProcessor.load(metadata_scaler_path)

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
            metadata: dict cac truong metadata (vd {"rating": 5, "num_likes": 3,
                "quality": 4.0}). Thieu truong nao se duoc dien gia
                tri hoc tu tap train (xem MetadataProcessor).

        Returns:
            dict: {"label": str, "confidence": float, "probabilities": {label: prob}}
        """
        results = self.predict_batch([text], [metadata or {}])
        return results[0]

    @torch.no_grad()
    def predict_batch(
        self, texts: List[str], metadata_list: Optional[List[Dict]] = None
    ) -> List[Dict]:
        if metadata_list is None:
            metadata_list = [{} for _ in texts]
        assert len(texts) == len(metadata_list)

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
        # Dung predict_with_gate de lay ca xac suat va gate_score
        probs, gate_scores = self.model.predict_with_gate(
            encodings["input_ids"].to(self.device),
            encodings["attention_mask"].to(self.device),
            metadata_tensor.to(self.device),
        )
        probs = probs.cpu().numpy()
        gate_scores = gate_scores.cpu().numpy()

        results = []
        for i in range(len(texts)):
            row_probs = {cfg.ID2LABEL[j]: float(probs[i, j]) for j in range(len(cfg.LABEL_LIST))}
            pred_id = int(probs[i].argmax())
            results.append(
                {
                    "text": texts[i],
                    "segmented_text": segmented_texts[i],
                    "label": cfg.ID2LABEL[pred_id],
                    "confidence": float(probs[i, pred_id]),
                    "probabilities": row_probs,
                    "gate_score": float(gate_scores[i, 0]),
                }
            )
        return results


def _cli():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Dự đoán nhãn cảm xúc cho bình luận mới")
    parser.add_argument("--checkpoint", default=cfg.BEST_MODEL_PATH)
    parser.add_argument("--metadata_scaler", default=cfg.METADATA_SCALER_PATH)

    parser.add_argument("--text", type=str, help="Một bình luận duy nhất cần dự đoán")
    parser.add_argument("--rating", type=float, default=None)
    parser.add_argument("--num_likes", type=float, default=None)
    parser.add_argument("--quality", type=float, default=None)

    parser.add_argument("--input_csv", type=str, help="File CSV chứa nhiều bình luận cần dự đoán")
    parser.add_argument("--text_column", type=str, default=cfg.TEXT_COLUMN)
    parser.add_argument("--output_csv", type=str, default=os.path.join(cfg.REPORT_DIR, "predictions.csv"))

    args = parser.parse_args()

    predictor = SentimentPredictor(
        checkpoint_path=args.checkpoint, metadata_scaler_path=args.metadata_scaler
    )

    if args.text:
        metadata = {}
        if args.rating is not None:
            metadata["rating"] = args.rating
        if args.num_likes is not None:
            metadata["num_likes"] = args.num_likes
        if args.quality is not None:
            metadata["quality"] = args.quality

        result = predictor.predict(args.text, metadata)
        print(f"Bình luận : {result['text']}")
        print(f"Nhãn dự đoán: {result['label']}  (độ tin cậy: {result['confidence']:.2%})")
        print(f"Gate Score : {result['gate_score']:.3f} (mức sử dụng metadata: 0=chỉ text, 1=dùng đầy đủ)")
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
