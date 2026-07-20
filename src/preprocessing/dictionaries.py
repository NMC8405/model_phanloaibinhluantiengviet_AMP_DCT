# -*- coding: utf-8 -*-
"""
dictionaries.py
================
Cac tu dien dung cho buoc "Tien xu ly" trong so do luong xu ly:
  - TEENCODE_DICT       : "ko" -> "khong"
  - ABBREVIATION_DICT   : "sp" -> "san pham"
  - EMOJI_SENTIMENT_DICT: icon Unicode (emoji) -> tu the hien cam xuc
  - ICON_SENTIMENT_DICT : icon go bang ky tu (":)", "T_T", ...) -> tu the hien cam xuc
  - SYNONYM_DICT        : dung cho module augmentation (Synonym Replacement)

Ghi chu: day la tap tu dien MAU, duoc chon loc de bao quat cac truong hop
pho bien nhat trong binh luan tieng Viet tren san thuong mai dien tu / mang
xa hoi. Nhom co the/nen mo rong them theo du lieu thuc te cua minh (xem
phan "Mo rong tu dien" trong README.md).
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# 1) TEENCODE: cac kieu viet tat / bien am pho bien khi nhan tin, binh luan
#    Key PHAI la chu thuong, se duoc match theo tung "tu" (word boundary).
# ---------------------------------------------------------------------------
#
# LUU Y THIET KE: tu dien nay KHONG "kiem duyet" (censor) cac tu tuc tiu /
# thieu van hoa bang cach thay bang chuoi rong. Ly do: day la tin hieu QUAN
# TRONG de mo hinh nhan biet binh luan tieu cuc / xuc pham (dung muc tieu
# cua bai toan: "Phat hien cac binh luan tieu cuc co noi dung xuc pham").
# Neu xoa cac tu nay o buoc tien xu ly se lam MAT tin hieu, khien mo hinh
# kho phan biet negative voi normal hon. Nen de nguyen (khong dua vao dict)
# va de PhoBERT tokenizer (subword) tu xu ly.
#
TEENCODE_DICT: Dict[str, str] = {
    "ko": "không",
    "k": "không",
    "kh": "không",
    "hok": "không",
    "hong": "không",
    "khong": "không",
    "dc": "được",
    "đc": "được",
    "duoc": "được",
    "j": "gì",
    "z": "vậy",
    "zậy": "vậy",
    "v": "vậy",
    "r": "rồi",
    "rùi": "rồi",
    "roi": "rồi",
    "bt": "bình thường",
    "bth": "bình thường",
    "ok": "ổn",
    "oke": "ổn",
    "okie": "ổn",
    "okay": "ổn",
    "nch": "nói chung",
    "ntn": "như thế nào",
    "nt": "nhắn tin",
    "vs": "với",
    "mng": "mọi người",
    "mn": "mọi người",
    "ny": "người yêu",
    "bít": "biết",
    "bik": "biết",
    "biet": "biết",
    "iu": "yêu",
    "yeu": "yêu",
    "thik": "thích",
    "thich": "thích",
    "thjk": "thích",
    "hqua": "hôm qua",
    "h": "giờ",
    "bh": "bây giờ",
    "bgio": "bây giờ",
    "ck": "chồng",
    "vk": "vợ",
    "cx": "cũng",
    "cung": "cũng",
    "vc": "vợ chồng",
    "add": "thêm",
    "sml": "rất tệ",
    "qá": "quá",
    "wá": "quá",
    "nhìu": "nhiều",
    "nhieu": "nhiều",
    "ntnay": "như thế này",
    "dt": "điện thoại",
    "mk": "mình",
    "mik": "mình",
    "tui": "tôi",
}
# Ghi chu: KHONG dua "qua" -> "quá" vao day vi "qua" la mot tu tieng Viet
# hop le (nghia "di qua/luot qua"), de tranh thay the sai nghia.
# Ghi chu: "T_T" duoc xu ly rieng trong ICON_SENTIMENT_DICT (buoc chuan hoa
# icon chay TRUOC buoc teencode trong pipeline) nen khong lap lai o day.

# ---------------------------------------------------------------------------
# 2) TU VIET TAT (van phong / san pham thuong mai dien tu)
# ---------------------------------------------------------------------------
ABBREVIATION_DICT: Dict[str, str] = {
    "sp": "sản phẩm",
    "sph": "sản phẩm",
    "shop": "cửa hàng",
    "ship": "giao hàng",
    "sz": "kích cỡ",
    "size": "kích cỡ",
    "cty": "công ty",
    "nv": "nhân viên",
    "kh": "khách hàng",
    "khg": "không",
    "đt": "điện thoại",
    "sdt": "số điện thoại",
    "sđt": "số điện thoại",
    "đk": "đăng ký",
    "dk": "đăng ký",
    "tg": "thời gian",
    "gg": "google",
    "fb": "facebook",
    "cskh": "chăm sóc khách hàng",
    "vd": "ví dụ",
    "vs": "với",
    "hd": "hướng dẫn",
    "hdsd": "hướng dẫn sử dụng",
    "bh": "bảo hành",
    "ncc": "nhà cung cấp",
    "đh": "đơn hàng",
    "dh": "đơn hàng",
    "cod": "thanh toán khi nhận hàng",
}

# ---------------------------------------------------------------------------
# 3) EMOJI -> tu the hien cam xuc (dua theo vi du trong tai lieu: 😊 -> tich
#    cuc, 😡 -> tieu cuc). Chen tu tu nhien (chua segment) de tiep tuc di qua
#    buoc Word Segmentation nhu binh thuong.
# ---------------------------------------------------------------------------
POSITIVE_EMOJIS: List[str] = [
    "😊", "😀", "😁", "😃", "😄", "😆", "🙂", "🙃", "😉", "😍", "🥰", "😘",
    "😻", "🤩", "😇", "👍", "👏", "🙌", "💪", "❤️", "❤", "💕", "💖", "💗",
    "✨", "🎉", "🔥", "💯", "☺️", "☺", "😙", "😚", "👌", "🥳",
]
NEGATIVE_EMOJIS: List[str] = [
    "😡", "😠", "🤬", "😢", "😭", "😞", "😔", "😟", "😣", "😖", "😩", "😫",
    "💔", "👎", "🖕", "😤", "😒", "🙄", "😑", "😐", "🤢", "🤮", "😨", "😰",
    "😱", "☹️", "☹", "😥", "😪",
]

EMOJI_SENTIMENT_DICT: Dict[str, str] = {}
for _e in POSITIVE_EMOJIS:
    EMOJI_SENTIMENT_DICT[_e] = " tích cực "
for _e in NEGATIVE_EMOJIS:
    EMOJI_SENTIMENT_DICT[_e] = " tiêu cực "

# ---------------------------------------------------------------------------
# 4) ICON go bang ky tu (emoticon kieu cu) -> tu the hien cam xuc.
#    Cac pattern nay PHAN BIET HOA/THUONG (vd "TT" khac "tt"), nen buoc
#    chuan hoa icon phai chay TRUOC buoc ha chu thuong trong pipeline.
#    Sap xep theo do dai giam dan de tranh match nham pattern con.
# ---------------------------------------------------------------------------
ICON_SENTIMENT_DICT: Dict[str, str] = {
    # positive truoc
    ":)))": " tích cực ",
    ":))": " tích cực ",
    ":)": " tích cực ",
    "=))": " tích cực ",
    "=)))": " tích cực ",
    ":D": " tích cực ",
    ":3": " tích cực ",
    "^^": " tích cực ",
    "^_^": " tích cực ",
    ":v": " tích cực ",
    ":P": " tích cực ",
    "<3": " tích cực ",
    # negative
    ":((( ": " tiêu cực ",
    ":((": " tiêu cực ",
    ":(": " tiêu cực ",
    ":'(": " tiêu cực ",
    "T_T": " tiêu cực ",
    "T.T": " tiêu cực ",
    "TT_TT": " tiêu cực ",
    "TT": " tiêu cực ",
    "-_-": " tiêu cực ",
    ":-(": " tiêu cực ",
}

# ---------------------------------------------------------------------------
# 5) SYNONYM_DICT: dung cho module augmentation (Synonym Replacement / EDA).
#    Tap trung vao cac tinh tu / cum tu mang tinh cam xuc de tang du lieu ma
#    khong lam doi nhan (label-preserving).
# ---------------------------------------------------------------------------
SYNONYM_DICT: Dict[str, List[str]] = {
    "tốt": ["ổn", "được"],
    "tuyệt vời": ["xuất sắc", "tuyệt hảo", "hoàn hảo"],
    "đẹp": ["xinh", "bắt mắt"],
    "nhanh": ["lẹ", "mau"],
    "chậm": ["trễ", "ì ạch"],
    "tệ": ["dở", "kém", "tồi"],
    "xấu": ["tệ", "kém chất lượng"],
    "hài lòng": ["ưng ý", "vừa ý"],
    "thất vọng": ["chán", "hụt hẫng"],
    "rẻ": ["giá tốt", "hời"],
    "đắt": ["mắc", "chát"],
    "chất lượng": ["phẩm chất"],
    "thích": ["ưng", "mê"],
    "ghét": ["không ưa", "không thích"],
    "ổn": ["được", "tạm được"],
    "kém": ["tệ", "dở"],
    "tuyệt": ["đỉnh", "xuất sắc"],
    "dễ thương": ["đáng yêu", "cute"],
    "bền": ["chắc chắn", "chắc"],
    "uy tín": ["đáng tin cậy", "tin cậy"],
}
# Ghi chu: cac gia tri dong nghia duoc chon de KHONG chua chinh tu khoa lam
# tien to/hau to (vd tranh "nhanh" -> "nhanh chóng" vi cau goc thuong da co
# san "chóng" ngay sau "nhanh", thay the se tao ra "nhanh chóng chóng").
