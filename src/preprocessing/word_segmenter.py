# -*- coding: utf-8 -*-
"""
word_segmenter.py
==================
Buoc "Word Segmentation" trong nhanh van ban: tach tu tieng Viet truoc khi
dua vao PhoBERT Tokenizer (PhoBERT duoc pretrain tren du lieu DA tach tu,
noi cac am tiet cua mot tu bang dau gach duoi, vd "san_pham").

Mac dinh dung `pyvi` (nhe, thuan Python, mo hinh CRF di kem san trong goi
pip - khong can tai gi them tu internet). Neu nhom muon do chinh xac cao
hon (dung dung bo tach tu ma PhoBERT dung khi pretrain), co the thay bang
VnCoreNLP (xem ghi chu ben duoi).
"""

from pyvi import ViTokenizer


class WordSegmenter:
    """Wrapper tach tu tieng Viet.

    Vi du:
        >>> ws = WordSegmenter()
        >>> ws.segment("sản phẩm rất tốt")
        'sản_phẩm rất tốt'
    """

    def segment(self, text: str) -> str:
        if not text:
            return ""
        try:
            return ViTokenizer.tokenize(text)
        except Exception:
            # pyvi hiem khi loi voi chuoi rong/ky tu la, fallback an toan:
            # tra ve nguyen van ban (khong tach tu) thay vi lam crash pipeline.
            return text

    def segment_batch(self, texts):
        return [self.segment(t) for t in texts]


# ---------------------------------------------------------------------------
# Ghi chu mo rong: doi sang VnCoreNLP (RDRSegmenter) - chinh la bo tach tu
# ma PhoBERT dung khi pretrain, co the cho ket qua khop tot hon mot chut,
# nhung can Java (JDK 8+) va nang hon:
#
#   pip install py_vncorenlp
#   import py_vncorenlp
#   model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/path/to/vncorenlp')
#   segmented = model.word_segment(text)[0]
#
# Neu doi, chi can sua class WordSegmenter o file nay, phan con lai cua
# project khong can thay doi gi (chi goi qua interface `.segment(text)`).
# ---------------------------------------------------------------------------
