# -*- coding: utf-8 -*-
"""
eda_augmenter.py
=================
Buoc "Data Augmentation" (buoc 17 trong bang tien xu ly): "Tang du lieu cho
lop it -> can bang du lieu".

Ap dung ky thuat EDA - Easy Data Augmentation (Wei & Zou, 2019), gom 4 phep
bien doi giu nguyen nhan (label-preserving), moi lan augment chon NGAU
NHIEN mot trong bon phep:

    - SR (Synonym Replacement) : thay mot vai tu/cum tu bang tu dong nghia
    - RI (Random Insertion)    : chen them mot tu dong nghia vao vi tri
                                  ngau nhien
    - RS (Random Swap)         : hoan doi vi tri hai tu ngau nhien
    - RD (Random Deletion)     : xoa ngau nhien mot vai tu

VI TRI TRONG PIPELINE: module nay chay tren van ban DA `TextCleaner.clean()`
nhung CHUA `WordSegmenter.segment()`, va CHI duoc ap dung cho tap TRAIN
(khong bao gio dung cho val/test - xem `scripts/prepare_data.py`). Sau khi
augment, van ban sinh them se di qua buoc Word Segmentation nhu binh thuong.

SR/RI dung SYNONYM_DICT (co the la cum nhieu tu, vd "tuyệt vời") nen duoc
cai dat bang regex khop CA CUM (khong tach theo tung ky tu trang), tranh
truong hop tu dien khong khop voi tu da qua Word Segmentation (co dau '_').
"""

import random
import re
from typing import Dict, List, Optional

from src.preprocessing.dictionaries import SYNONYM_DICT


class EDAAugmenter:
    def __init__(
        self,
        synonym_dict: Optional[Dict[str, List[str]]] = None,
        alpha_sr: float = 0.15,
        alpha_ri: float = 0.10,
        alpha_rs: float = 0.10,
        alpha_rd: float = 0.10,
        seed: Optional[int] = None,
    ):
        self.synonym_dict = synonym_dict or SYNONYM_DICT
        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.alpha_rd = alpha_rd
        self.rng = random.Random(seed)

        self._synonym_keys = sorted(self.synonym_dict.keys(), key=len, reverse=True)
        if self._synonym_keys:
            escaped = [re.escape(k) for k in self._synonym_keys]
            self._synonym_pattern = re.compile(
                r"\b(" + "|".join(escaped) + r")\b", re.UNICODE
            )
        else:
            self._synonym_pattern = None

    # ------------------------------------------------------------------
    def _random_synonym_for(self, word: str) -> Optional[str]:
        options = [w for w in self.synonym_dict.get(word, []) if w != word]
        if not options:
            return None
        return self.rng.choice(options)

    # ------------------------------------------------------------------ SR
    def synonym_replacement(self, text: str, n: int) -> str:
        """Thay toi da `n` cum tu (khop trong SYNONYM_DICT) bang dong nghia."""
        if self._synonym_pattern is None or n <= 0:
            return text

        matches = list(self._synonym_pattern.finditer(text))
        if not matches:
            return text
        self.rng.shuffle(matches)
        chosen = matches[:n]
        # Thay theo thu tu VI TRI GIAM DAN de khong lam lech index cac match con lai
        chosen.sort(key=lambda m: m.start(), reverse=True)

        for m in chosen:
            syn = self._random_synonym_for(m.group(0))
            if syn is not None:
                text = text[: m.start()] + syn + text[m.end() :]
        return text

    # ------------------------------------------------------------------ RI
    def random_insertion(self, text: str, n: int) -> str:
        """Chen `n` tu dong nghia (lay ngau nhien tu SYNONYM_DICT) vao vi
        tri ngau nhien trong cau (dua tren danh sach tu tach boi khoang
        trang)."""
        if not self._synonym_keys or n <= 0:
            return text
        words = text.split(" ")
        for _ in range(n):
            key = self.rng.choice(self._synonym_keys)
            syn = self._random_synonym_for(key)
            if syn is None:
                continue
            pos = self.rng.randint(0, len(words))
            words.insert(pos, syn)
        return " ".join(words)

    # ------------------------------------------------------------------ RS
    def random_swap(self, text: str, n: int) -> str:
        """Hoan doi vi tri ngau nhien `n` cap tu."""
        words = [w for w in text.split(" ") if w]
        if len(words) < 2 or n <= 0:
            return text
        for _ in range(n):
            i, j = self.rng.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        return " ".join(words)

    # ------------------------------------------------------------------ RD
    def random_deletion(self, text: str, p: float) -> str:
        """Xoa ngau nhien tung tu voi xac suat `p` (luon giu lai it nhat 1 tu)."""
        words = [w for w in text.split(" ") if w]
        if len(words) <= 1:
            return text
        kept = [w for w in words if self.rng.random() > p]
        if not kept:
            kept = [self.rng.choice(words)]
        return " ".join(kept)

    # ------------------------------------------------------------------
    def augment(self, text: str, num_aug: int = 1) -> List[str]:
        """Sinh `num_aug` bien the moi cua `text`.

        Moi bien the chon NGAU NHIEN 1 trong 4 ky thuat (SR/RI/RS/RD) —
        cach lam nay dung theo de xuat goc cua EDA (Wei & Zou, 2019) de
        da dang hoa loai nhieu duoc them vao, thay vi luon ap dung ca 4.
        """
        text = (text or "").strip()
        if not text:
            return [text] * num_aug

        num_words = max(1, len(text.split(" ")))
        techniques = ["sr", "ri", "rs", "rd"]
        results = []

        for _ in range(num_aug):
            technique = self.rng.choice(techniques)
            if technique == "sr":
                n = max(1, int(self.alpha_sr * num_words))
                new_text = self.synonym_replacement(text, n)
            elif technique == "ri":
                n = max(1, int(self.alpha_ri * num_words))
                new_text = self.random_insertion(text, n)
            elif technique == "rs":
                n = max(1, int(self.alpha_rs * num_words))
                new_text = self.random_swap(text, n)
            else:
                new_text = self.random_deletion(text, self.alpha_rd)
            results.append(new_text)

        return results
