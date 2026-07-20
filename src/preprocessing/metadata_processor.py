# -*- coding: utf-8 -*-
"""
metadata_processor.py
======================
Trien khai nhanh "Metadata" trong so do luong xu ly (phan "Tien xu ly" +
"Vector metadata" TRUOC khi qua Dense layer):

    Metadata -> Tien xu ly (dien missing, log-transform, clip) -> Vector metadata

Ho tro 3 loai cot (khai bao trong config.METADATA_SCHEMA):
    - "numeric"     : gia tri so binh thuong (vd rating 1-5) -> chuan hoa
                      bang StandardScaler (hoc tren tap train).
    - "numeric_log" : gia tri so phan phoi lech / co outlier (vd luot thich,
                      co the tu 0 den hang nghin) -> log1p roi moi scale.
    - "binary"      : gia tri 0/1 (vd da mua hang hay chua) -> giu nguyen,
                      KHONG scale (scale co the lam mat y nghia 0/1).

QUAN TRONG: goi fit() DUY NHAT tren tap TRAIN. Tap val/test/inference chi
goi transform() de tranh data leakage (ro ri thong tin thong ke tu
val/test vao qua trinh chuan hoa).
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


class MetadataProcessor:
    def __init__(self, schema: Dict[str, str], columns: List[str] = None):
        self.schema = schema
        self.columns = columns or list(schema.keys())
        self.scaler = StandardScaler()

        self._fill_values: Dict[str, float] = {}
        self._constant_cols: Dict[str, float] = {}
        self._scale_cols: List[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    def _extract_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lay cac cot metadata tu DataFrame, ep ve kieu so.

        Neu DataFrame dau vao THIEU mot cot (vd luc inference nguoi dung
        khong truyen `verified_purchase`), cot do duoc dien NaN va se duoc
        xu ly o buoc dien missing ngay sau (dung fill_value da hoc luc fit).
        """
        out = pd.DataFrame(index=df.index)
        for col in self.columns:
            if col in self._constant_cols:
                # Neu cot do la hang so trong tap train, cuong che ve gia tri hang so do
                out[col] = self._constant_cols[col]
            elif col in df.columns:
                out[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                out[col] = np.nan
        return out

    def _fill_and_prescale(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Dien missing (theo gia tri hoc duoc luc fit) + log1p/clip.
        Chua thuc hien buoc scale (StandardScaler)."""
        out = pd.DataFrame(index=raw.index)
        for col in self.columns:
            col_type = self.schema[col]
            values = raw[col].fillna(self._fill_values.get(col, 0.0))
            if col_type == "numeric_log":
                values = np.log1p(values.clip(lower=0))
            elif col_type == "binary":
                values = values.clip(0, 1)
            # "numeric": giu nguyen, se duoc StandardScaler xu ly
            out[col] = values
        return out

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "MetadataProcessor":
        # extract raw first without constant_cols since it is not filled yet
        out = pd.DataFrame(index=df.index)
        for col in self.columns:
            if col in df.columns:
                out[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                out[col] = np.nan
        raw = out

        # Phát hiện các cột là hằng số (không biến đổi / variance = 0)
        for col in self.columns:
            unique_vals = raw[col].dropna().unique()
            if len(unique_vals) <= 1:
                # Nếu cột toàn rỗng hoặc chỉ có 1 giá trị duy nhất
                self._constant_cols[col] = float(unique_vals[0]) if len(unique_vals) == 1 else 0.0

        # Trích xuất lại raw với cột hằng số đã được cưỡng chế
        raw = self._extract_raw(df)

        # Hoc gia tri dien missing tren TAP TRAIN (median cho numeric/
        # numeric_log, 0 cho binary)
        for col in self.columns:
            if self.schema[col] == "binary":
                self._fill_values[col] = 0.0
            else:
                median = raw[col].median()
                self._fill_values[col] = float(median) if pd.notna(median) else 0.0

        prescaled = self._fill_and_prescale(raw)

        # Chỉ scale những cột không phải binary và không phải hằng số
        self._scale_cols = [
            c for c in self.columns 
            if self.schema[c] != "binary" and c not in self._constant_cols
        ]
        if self._scale_cols:
            self.scaler.fit(prescaled[self._scale_cols])

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "MetadataProcessor chưa được fit(). Hãy gọi fit() trên tập "
                "train trước khi transform()."
            )
        raw = self._extract_raw(df)
        prescaled = self._fill_and_prescale(raw)
        if self._scale_cols:
            prescaled[self._scale_cols] = self.scaler.transform(
                prescaled[self._scale_cols]
            )
        return prescaled[self.columns].values.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def transform_single(self, metadata: dict) -> np.ndarray:
        """Tien ich: transform cho MOT binh luan duy nhat (dung khi inference).

        VD: transform_single({"rating": 5, "num_likes": 12, "quality": 4.0})
        """
        df = pd.DataFrame([metadata])
        return self.transform(df)[0]

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(
            {
                "schema": self.schema,
                "columns": self.columns,
                "scaler": self.scaler,
                "fill_values": self._fill_values,
                "constant_cols": self._constant_cols,
                "scale_cols": self._scale_cols,
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "MetadataProcessor":
        data = joblib.load(path)
        obj = cls(data["schema"], data["columns"])
        obj.scaler = data["scaler"]
        obj._fill_values = data["fill_values"]
        obj._constant_cols = data.get("constant_cols", {})
        obj._scale_cols = data["scale_cols"]
        obj._fitted = data["fitted"]
        return obj

