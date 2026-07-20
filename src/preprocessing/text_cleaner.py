# -*- coding: utf-8 -*-
"""
text_cleaner.py
================
Trien khai cac buoc "Tien xu ly" (nhanh Van ban) trong so do luong xu ly:

    Loai bo HTML -> Chuan hoa Unicode -> Chuan hoa emoji -> Chuan hoa icon
    -> Thay the URL -> Chuan hoa hoa/thuong -> Chuan hoa khoang trang
    -> Chuan hoa ky tu lap -> Chuan hoa dau cau -> Chuan hoa teencode
    -> Chuan hoa tu viet tat -> Chuan hoa khoang trang (lan cuoi)

Hai buoc con lai cua nhanh van ban (Word Segmentation, PhoBERT Tokenizer)
nam o cac module rieng: `word_segmenter.py` va `dataset.py`.

Thu tu cac buoc DUOC CHON CO CHU DICH, khong tuy tien:
  - Loai bo HTML PHAI truoc Thay the URL (vi HTML co the chua the <a href=...>
    boc quanh URL; ngoai ra token placeholder "<URL>" ma buoc sau sinh ra
    cung co dang "<...>" nen phai sinh SAU khi da loai bo HTML de tranh bi
    chinh buoc loai HTML xoa mat).
  - Chuan hoa Unicode (NFC) PHAI truoc cac buoc dua tren regex ky tu (chuan
    hoa ky tu lap, ...) de moi ky tu tieng Viet duoc tinh la MOT codepoint.
  - Chuan hoa icon (":)", "T_T", "TT", ...) PHAI truoc buoc ha chu thuong,
    vi mot so icon dung chu hoa co chu dich (vd "TT") - ha thuong truoc se
    lam icon do bien mat / bi nham voi tu thuong "tt".
  - Chuan hoa KY TU LAP phai truoc chuan hoa DAU CAU: neu lam nguoc lai,
    dau "..." vua duoc chuan hoa o buoc dau cau se bi chinh buoc ky tu lap
    (quy tac chung cho MOI ky tu) nen lai lan nua, thanh "..". Lam ky tu
    lap truoc dam bao dau cau la buoc "chot" cuoi cung tren dau cham/hoi/
    than.
  - Chuan hoa teencode/tu viet tat PHAI sau buoc ha chu thuong (dict key la
    chu thuong).
"""

import re
import unicodedata
from typing import List, Tuple

from .dictionaries import (
    TEENCODE_DICT,
    ABBREVIATION_DICT,
    EMOJI_SENTIMENT_DICT,
    ICON_SENTIMENT_DICT,
)


class TextCleaner:
    """Pipeline lam sach van ban binh luan tieng Viet, tung buoc mot.

    Vi du:
        >>> cleaner = TextCleaner()
        >>> cleaner.clean("Sản phẩmmmm dc lắmmm ạ!!!! :)) <b>tốt</b> ghê")
        'sản phẩm được lắm ạ! tích cực tốt ghê'
    """

    URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
    HTML_PATTERN = re.compile(r"<[^>]+>")
    REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")
    MULTI_DOT_PATTERN = re.compile(r"\.{2,}")
    MULTI_EXCLAIM_QUESTION_PATTERN = re.compile(r"([!?])[!?]+")
    MULTI_COMMA_PATTERN = re.compile(r",{2,}")
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(self):
        # Icon/emoji: sap theo do dai key GIAM DAN de tranh match nham
        # pattern con (vd ":)" la chuoi con cua ":))").
        self._sorted_icons: List[Tuple[str, str]] = sorted(
            ICON_SENTIMENT_DICT.items(), key=lambda kv: len(kv[0]), reverse=True
        )
        self._sorted_emojis: List[Tuple[str, str]] = sorted(
            EMOJI_SENTIMENT_DICT.items(), key=lambda kv: len(kv[0]), reverse=True
        )

        self._teencode_pattern = self._build_dict_pattern(TEENCODE_DICT)
        self._abbreviation_pattern = self._build_dict_pattern(ABBREVIATION_DICT)

    @staticmethod
    def _build_dict_pattern(dictionary: dict):
        """Ghep 1 tu dien thanh 1 regex duy nhat (khop theo tu, word-boundary)."""
        if not dictionary:
            return None
        sorted_keys = sorted(dictionary.keys(), key=len, reverse=True)
        escaped = [re.escape(k) for k in sorted_keys]
        pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.UNICODE)
        return pattern

    # ------------------------------------------------------------------
    # Tung buoc rieng le (public de co the goi doc lap / viet unit test)
    # ------------------------------------------------------------------
    def remove_html(self, text: str) -> str:
        """(1) Loai bo the HTML con sot lai."""
        return self.HTML_PATTERN.sub(" ", text)

    def normalize_unicode(self, text: str) -> str:
        """(2) Dong nhat mã hoá tiếng Việt ve dang NFC (precomposed)."""
        return unicodedata.normalize("NFC", text)

    def normalize_emoji(self, text: str) -> str:
        """(3) Emoji -> tu the hien cam xuc. VD: 😊 -> 'tích cực'."""
        for emoji_char, replacement in self._sorted_emojis:
            if emoji_char in text:
                text = text.replace(emoji_char, replacement)
        return text

    def normalize_icons(self, text: str) -> str:
        """(4) Icon go bang ky tu (":)", "T_T", "-_-") -> tu the hien cam xuc."""
        for icon, replacement in self._sorted_icons:
            if icon in text:
                text = text.replace(icon, replacement)
        return text

    def replace_url(self, text: str) -> str:
        """(5) Thay the duong link bang token <URL> de tranh spam / nhieu."""
        return self.URL_PATTERN.sub(" <URL> ", text)

    def lowercase(self, text: str) -> str:
        """(6) Chuan hoa chu hoa/thuong -> giam so luong tu khac nhau."""
        return text.lower()

    def normalize_whitespace(self, text: str) -> str:
        """(7) Loai bo khoang trang thua."""
        return self.WHITESPACE_PATTERN.sub(" ", text).strip()

    def normalize_repeated_chars(self, text: str) -> str:
        """(8) Ky tu lap >=3 lan -> giu lai 2 lan.

        VD: 'xấuuuuuuu' -> 'xấuu'. Giu 2 lan (khong rut ve 1) de khong lam
        mat cac tu tieng Anh co chu cai kep hop le nhu 'hello', 'good'.
        Chay TRUOC buoc chuan hoa dau cau (xem ghi chu dau file).
        """
        return self.REPEATED_CHAR_PATTERN.sub(r"\1\1", text)

    def normalize_punctuation(self, text: str) -> str:
        """(9) Chuan hoa dau cau lap: '?????' -> '?', '!!!!' -> '!'.

        Rieng chuoi dau cham (>=2 dau) duoc chuan hoa thanh dau ba cham
        '...' (giu y nghia "bo lung", khac voi mot dau cham don thuan).
        """
        text = self.MULTI_DOT_PATTERN.sub("...", text)
        text = self.MULTI_EXCLAIM_QUESTION_PATTERN.sub(r"\1", text)
        text = self.MULTI_COMMA_PATTERN.sub(",", text)
        return text

    def normalize_teencode(self, text: str) -> str:
        """(10) Teencode -> tu chuan. VD: 'ko' -> 'không'."""
        if self._teencode_pattern is None:
            return text
        return self._teencode_pattern.sub(lambda m: TEENCODE_DICT[m.group(0)], text)

    def normalize_abbreviation(self, text: str) -> str:
        """(11) Tu viet tat -> tu day du. VD: 'sp' -> 'sản phẩm'."""
        if self._abbreviation_pattern is None:
            return text
        return self._abbreviation_pattern.sub(
            lambda m: ABBREVIATION_DICT[m.group(0)], text
        )

    # ------------------------------------------------------------------
    # Pipeline day du
    # ------------------------------------------------------------------
    def clean(self, text: str) -> str:
        """Chay toan bo pipeline tien xu ly tren MOT chuoi van ban."""
        if text is None:
            return ""
        text = str(text)

        text = self.remove_html(text)
        text = self.normalize_unicode(text)
        text = self.normalize_emoji(text)
        text = self.normalize_icons(text)
        text = self.replace_url(text)
        text = self.lowercase(text)
        text = self.normalize_whitespace(text)
        text = self.normalize_repeated_chars(text)
        text = self.normalize_punctuation(text)
        text = self.normalize_teencode(text)
        text = self.normalize_abbreviation(text)
        text = self.normalize_whitespace(text)  # lan cuoi, sau khi thay the tu

        return text

    def clean_with_trace(self, text: str) -> List[Tuple[str, str]]:
        """Giong `clean()` nhung tra ve KET QUA SAU TUNG BUOC.

        Rat huu ich khi lam slide / bao cao do an de minh hoa tac dung cua
        tung buoc tien xu ly tren mot vi du cu the.

        Returns:
            List cac tuple (ten_buoc, van_ban_sau_buoc_do).
        """
        if text is None:
            text = ""
        text = str(text)
        trace = [("Gốc (input)", text)]

        steps = [
            ("Loại bỏ HTML", self.remove_html),
            ("Chuẩn hoá Unicode", self.normalize_unicode),
            ("Chuẩn hoá emoji", self.normalize_emoji),
            ("Chuẩn hoá icon", self.normalize_icons),
            ("Thay thế URL", self.replace_url),
            ("Chuẩn hoá hoa/thường", self.lowercase),
            ("Chuẩn hoá khoảng trắng", self.normalize_whitespace),
            ("Chuẩn hoá ký tự lặp", self.normalize_repeated_chars),
            ("Chuẩn hoá dấu câu", self.normalize_punctuation),
            ("Chuẩn hoá teencode", self.normalize_teencode),
            ("Chuẩn hoá từ viết tắt", self.normalize_abbreviation),
            ("Chuẩn hoá khoảng trắng (cuối)", self.normalize_whitespace),
        ]
        for name, fn in steps:
            text = fn(text)
            trace.append((name, text))

        return trace

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Ap dung `clean()` cho mot danh sach van ban."""
        return [self.clean(t) for t in texts]
