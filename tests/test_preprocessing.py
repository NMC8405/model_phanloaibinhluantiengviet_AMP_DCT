# -*- coding: utf-8 -*-
"""
tests/test_preprocessing.py
============================
Unit test cho cac module tien xu ly / augmentation - khong can GPU, khong
can tai PhoBERT, chay nhanh (vai giay).

Chay: pytest tests/ -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.augmentation.eda_augmenter import EDAAugmenter
from src.preprocessing.metadata_processor import MetadataProcessor
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.word_segmenter import WordSegmenter


@pytest.fixture(scope="module")
def cleaner():
    return TextCleaner()


@pytest.fixture(scope="module")
def segmenter():
    return WordSegmenter()


# ---------------------------------------------------------------------- TextCleaner
class TestTextCleaner:
    def test_lowercase(self, cleaner):
        assert cleaner.clean("SẢN PHẨM TỐT") == "sản phẩm tốt"

    def test_html_removed(self, cleaner):
        out = cleaner.clean("<b>đẹp</b> lắm <br/>")
        assert "<b>" not in out and "</b>" not in out and "<br/>" not in out
        assert "đẹp" in out

    def test_url_replaced(self, cleaner):
        out = cleaner.clean("xem tại https://shop.vn/abc?id=1 nhé")
        assert "<url>" in out
        assert "https://" not in out

    def test_repeated_chars_kept_at_two(self, cleaner):
        assert cleaner.clean("đẹpppppp") == "đẹpp"
        # tu tieng Anh co chu kep hop le khong bi anh huong
        assert "hello" in cleaner.clean("hello bạn")

    def test_repeated_punctuation_collapsed(self, cleaner):
        assert cleaner.clean("tốt!!!!!") == "tốt!"
        assert cleaner.clean("sao vậy?????") == "sao vậy?"

    def test_ellipsis_normalized_not_over_collapsed(self, cleaner):
        # >=2 dau cham -> chuan hoa thanh dung 3 dau (dau ba cham), KHONG bi
        # buoc chuan-hoa-ky-tu-lap nen tiep tuc rut xuong con 2 dau.
        out = cleaner.clean("thôi rồi.....")
        assert out.endswith("...")
        assert not out.endswith("....")

    def test_teencode_normalized(self, cleaner):
        out = cleaner.clean("sp dc lắm, ko chê được")
        assert "được" in out
        assert "không" in out
        assert "sản phẩm" in out

    def test_ambiguous_word_qua_not_corrupted(self, cleaner):
        # "qua" (di qua) la tu hop le, KHONG duoc doi thanh "quá"
        out = cleaner.clean("đi qua đường thấy shop này")
        assert " qua " in f" {out} "

    def test_emoji_mapped_to_sentiment_word(self, cleaner):
        pos = cleaner.clean("sản phẩm này 😊")
        neg = cleaner.clean("sản phẩm này 😡")
        assert "tích cực" in pos
        assert "tiêu cực" in neg

    def test_icon_case_sensitive_before_lowercase(self, cleaner):
        # "TT" (icon buon) phai duoc nhan dien TRUOC khi ha chu thuong
        out = cleaner.clean("buồn quá TT")
        assert "tiêu cực" in out

    def test_whitespace_collapsed(self, cleaner):
        assert cleaner.clean("nhiều   khoảng    trắng") == "nhiều khoảng trắng"

    def test_empty_and_none_input(self, cleaner):
        assert cleaner.clean("") == ""
        assert cleaner.clean(None) == ""

    def test_clean_with_trace_returns_all_steps(self, cleaner):
        trace = cleaner.clean_with_trace("Sản phẩmmmm dc lắmmm ạ!!!!")
        assert isinstance(trace, list)
        assert trace[0][0] == "Gốc (input)"
        assert trace[-1][1] == cleaner.clean("Sản phẩmmmm dc lắmmm ạ!!!!")
        assert len(trace) > 5


# ---------------------------------------------------------------------- WordSegmenter
class TestWordSegmenter:
    def test_segment_joins_syllables(self, segmenter):
        out = segmenter.segment("sản phẩm rất tốt")
        assert "sản_phẩm" in out

    def test_segment_empty(self, segmenter):
        assert segmenter.segment("") == ""


# ---------------------------------------------------------------------- MetadataProcessor
class TestMetadataProcessor:
    SCHEMA = {"rating": "numeric", "num_likes": "numeric_log", "verified_purchase": "binary"}
    COLUMNS = ["rating", "num_likes", "verified_purchase"]

    def test_fit_transform_shape(self):
        df = pd.DataFrame(
            {"rating": [5, 4, 1, 3], "num_likes": [10, 0, 100, 5], "verified_purchase": [1, 0, 1, 1]}
        )
        mp = MetadataProcessor(self.SCHEMA, self.COLUMNS)
        out = mp.fit_transform(df)
        assert out.shape == (4, 3)
        assert out.dtype == np.float32

    def test_binary_column_not_scaled(self):
        df = pd.DataFrame(
            {"rating": [5, 4, 1, 3], "num_likes": [10, 0, 100, 5], "verified_purchase": [1, 0, 1, 0]}
        )
        mp = MetadataProcessor(self.SCHEMA, self.COLUMNS)
        out = mp.fit_transform(df)
        # cot cuoi (verified_purchase) phai van la 0/1, khong bi StandardScaler bien doi
        assert set(np.unique(out[:, 2]).tolist()) <= {0.0, 1.0}

    def test_transform_before_fit_raises(self):
        df = pd.DataFrame({"rating": [5], "num_likes": [1], "verified_purchase": [1]})
        mp = MetadataProcessor(self.SCHEMA, self.COLUMNS)
        with pytest.raises(RuntimeError):
            mp.transform(df)

    def test_missing_column_filled_with_learned_value(self):
        train_df = pd.DataFrame(
            {"rating": [5, 3, 1], "num_likes": [10, 5, 0], "verified_purchase": [1, 0, 1]}
        )
        mp = MetadataProcessor(self.SCHEMA, self.COLUMNS)
        mp.fit(train_df)
        # luc inference nguoi dung KHONG truyen verified_purchase
        single = mp.transform_single({"rating": 4, "num_likes": 2})
        assert single.shape == (3,)
        assert not np.isnan(single).any()

    def test_save_and_load_roundtrip(self, tmp_path):
        df = pd.DataFrame(
            {"rating": [5, 4, 1, 3], "num_likes": [10, 0, 100, 5], "verified_purchase": [1, 0, 1, 1]}
        )
        mp = MetadataProcessor(self.SCHEMA, self.COLUMNS)
        mp.fit(df)
        path = str(tmp_path / "scaler.joblib")
        mp.save(path)

        mp2 = MetadataProcessor.load(path)
        np.testing.assert_allclose(mp.transform(df), mp2.transform(df))


# ---------------------------------------------------------------------- EDAAugmenter
class TestEDAAugmenter:
    def test_augment_returns_requested_count(self):
        aug = EDAAugmenter(seed=1)
        variants = aug.augment("sản phẩm rất tốt đóng gói cẩn thận giao hàng nhanh", num_aug=5)
        assert len(variants) == 5

    def test_augment_does_not_crash_on_short_text(self):
        aug = EDAAugmenter(seed=1)
        assert aug.augment("tốt", num_aug=2) is not None
        assert aug.augment("", num_aug=2) == ["", ""]

    def test_random_deletion_keeps_at_least_one_word(self):
        aug = EDAAugmenter(seed=1, alpha_rd=0.99)
        out = aug.random_deletion("một hai ba bốn năm", p=0.99)
        assert len(out.strip()) > 0

    def test_synonym_replacement_no_duplication_artifact(self):
        # Hoi quy cho bug da fix: "nhanh" -> "nhanh chóng" tung gay lap tu
        # khi van ban da co san "chóng" ngay sau "nhanh".
        aug = EDAAugmenter(seed=0)
        text = "giao hàng nhanh chóng lắm"
        for _ in range(20):
            out = aug.synonym_replacement(text, n=3)
            assert "chóng chóng" not in out
            assert "bỉ bỉ" not in out

    def test_deterministic_with_seed(self):
        aug1 = EDAAugmenter(seed=123)
        aug2 = EDAAugmenter(seed=123)
        text = "sản phẩm chất lượng tốt giao hàng nhanh đóng gói cẩn thận"
        assert aug1.augment(text, num_aug=3) == aug2.augment(text, num_aug=3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
