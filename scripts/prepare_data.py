# -*- coding: utf-8 -*-
"""
scripts/prepare_data.py
========================
Ghep toan bo cac buoc "Tien xu ly" (1)-(17) trong tai lieu thanh MOT pipeline
duy nhat, tu file CSV tho -> 3 file train/val/test san sang de `src/train.py`
su dung.

THU TU XU LY (quan trong, xem giai thich trong tung buoc):
    1. Loại bỏ dữ liệu trùng lặp
    2. Loại bỏ dữ liệu rỗng
    3-13. Làm sạch văn bản (TextCleaner - HTML, unicode, emoji, icon, URL,
          hoa/thường, khoảng trắng, dấu câu, ký tự lặp, teencode, viết tắt)
    -> Chia tập train/val/test (stratify theo nhãn) TRƯỚC augmentation
    17. Data Augmentation (CHỈ trên train, CHỈ lớp thiểu số, để cân bằng)
    14. Word Segmentation (tách từ - áp dụng SAU augmentation để các dòng
        sinh thêm cũng được tách từ nhất quán với dữ liệu gốc)

    (Bước 15 "PhoBERT Tokenization" và 16 "Chuẩn hoá metadata" KHÔNG nằm ở
    file này: tokenization diễn ra ngay tại lúc train/predict (xem
    `src/dataset.py`), còn chuẩn hoá metadata được `MetadataProcessor` học
    (fit) ngay trong `src/train.py` để đảm bảo CHỈ dùng thống kê của tập
    train, không rò rỉ từ val/test.)

Cách chạy (từ thư mục gốc project):
    python -m scripts.prepare_data
    python -m scripts.prepare_data --input_csv data/raw/comments.csv --no_augment
"""

import argparse
import os
import sys

# Fix encoding cho Windows console (tieng Viet)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src.augmentation.eda_augmenter import EDAAugmenter
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.word_segmenter import WordSegmenter
from src.utils import set_seed, setup_logger

tqdm.pandas()


def parse_args():
    p = argparse.ArgumentParser(
        description="Chuẩn bị dữ liệu: dedup, làm sạch, chia tập, cân bằng lớp, tách từ"
    )
    p.add_argument("--input_csv", default=cfg.RAW_DATA_PATH)
    p.add_argument("--text_column", default=cfg.TEXT_COLUMN)
    p.add_argument("--label_column", default=cfg.LABEL_COLUMN)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment", dest="augment", action="store_true", default=True)
    p.add_argument("--no_augment", dest="augment", action="store_false")
    p.add_argument(
        "--augment_target_ratio",
        type=float,
        default=1.0,
        help="Tỷ lệ (số mẫu lớp thiểu số) / (số mẫu lớp nhiều nhất) muốn đạt "
        "sau augment. 1.0 = cân bằng hoàn toàn.",
    )
    p.add_argument("--train_out", default=cfg.TRAIN_PATH)
    p.add_argument("--val_out", default=cfg.VAL_PATH)
    p.add_argument("--test_out", default=cfg.TEST_PATH)
    return p.parse_args()


def augment_minority_classes(df, text_column, label_column, target_ratio, seed, logger):
    """(17) Data Augmentation: tăng dữ liệu cho lớp ít -> cân bằng dữ liệu.

    Áp dụng EDA lên cột `text_column` (lúc này là cleaned_text, CHƯA tách
    từ) của TỪNG dòng thuộc lớp thiểu số, lặp tuần hoàn qua các dòng gốc
    của lớp đó cho tới khi đạt số lượng mục tiêu.
    """
    augmenter = EDAAugmenter(seed=seed)
    counts = df[label_column].value_counts()
    max_count = int(counts.max())
    target_count = int(max_count * target_ratio)

    logger.info(f"Phân bố lớp trước augment:\n{counts}")
    logger.info(f"Số lượng mục tiêu mỗi lớp (target_ratio={target_ratio}): {target_count}")

    new_rows = []
    for label, count in counts.items():
        count = int(count)
        if count >= target_count:
            continue
        n_needed = target_count - count
        label_rows = df[df[label_column] == label].reset_index(drop=True)
        logger.info(f"  Lớp '{label}': {count} mẫu -> augment thêm {n_needed} mẫu")

        for i in tqdm(range(n_needed), desc=f"Augment '{label}'", leave=False):
            src_row = label_rows.iloc[i % len(label_rows)]
            variant_text = augmenter.augment(src_row[text_column], num_aug=1)[0]
            new_row = src_row.copy()
            new_row[text_column] = variant_text
            if "raw_text" in new_row:
                new_row["raw_text"] = variant_text
            new_rows.append(new_row)

    if not new_rows:
        logger.info("Các lớp đã đủ cân bằng, không cần augment thêm.")
        return df

    aug_df = pd.DataFrame(new_rows)
    result = pd.concat([df, aug_df], ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(f"Số dòng train: {len(df)} -> {len(result)} (sau augment)")
    logger.info(f"Phân bố lớp sau augment:\n{result[label_column].value_counts()}")
    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg.ensure_dirs()
    logger = setup_logger("prepare_data", log_file=os.path.join(cfg.LOG_DIR, "prepare_data.log"))

    logger.info(f"Đọc dữ liệu thô từ: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    logger.info(f"Số dòng ban đầu: {len(df)}")

    missing_cols = [c for c in [args.text_column, args.label_column] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"File CSV thiếu cột bắt buộc: {missing_cols}. Các cột hiện có: {list(df.columns)}"
        )

    invalid_labels = sorted(set(df[args.label_column].dropna().unique()) - set(cfg.LABEL_LIST))
    if invalid_labels:
        raise ValueError(
            f"Có nhãn không hợp lệ: {invalid_labels}. Nhãn hợp lệ theo config.LABEL_LIST: {cfg.LABEL_LIST}"
        )

    metadata_cols_present = [c for c in cfg.METADATA_COLUMNS if c in df.columns]
    missing_meta = sorted(set(cfg.METADATA_COLUMNS) - set(metadata_cols_present))
    if missing_meta:
        logger.warning(
            f"Thiếu cột metadata {missing_meta} trong file thô - các giá trị này sẽ được "
            f"MetadataProcessor tự điền mặc định lúc train/inference."
        )

    # -------------------------------------------------------------- (1)
    before = len(df)
    df = df.drop_duplicates(subset=[args.text_column]).reset_index(drop=True)
    logger.info(f"(1) Loại bỏ dữ liệu trùng lặp: {before} -> {len(df)} dòng")

    # -------------------------------------------------------------- (2)
    before = len(df)
    df[args.text_column] = df[args.text_column].astype(str)
    df = df[df[args.text_column].str.strip().astype(bool)]
    df = df.dropna(subset=[args.label_column]).reset_index(drop=True)
    logger.info(f"(2) Loại bỏ dữ liệu rỗng: {before} -> {len(df)} dòng")

    logger.info(f"Phân bố nhãn (toàn bộ, trước chia tập):\n{df[args.label_column].value_counts()}")

    # -------------------------------------------------------------- (3)-(13)
    logger.info("(3-13) Đang làm sạch văn bản (TextCleaner)...")
    cleaner = TextCleaner()
    df["raw_text"] = df[args.text_column]
    df["cleaned_text"] = df[args.text_column].progress_apply(cleaner.clean)

    before = len(df)
    df = df[df["cleaned_text"].str.strip().astype(bool)].reset_index(drop=True)
    if before != len(df):
        logger.info(
            f"  Loại bỏ thêm {before - len(df)} dòng trở thành rỗng sau khi làm sạch "
            f"(vd bình luận chỉ toàn ký tự đặc biệt/emoji không mang nghĩa)"
        )

    # -------------------------------------------------------------- Chia tập
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio phải nhỏ hơn 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - args.train_ratio),
        stratify=df[args.label_column],
        random_state=args.seed,
    )
    relative_val_ratio = args.val_ratio / (args.val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_val_ratio),
        stratify=temp_df[args.label_column],
        random_state=args.seed,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    logger.info(
        f"Chia tập (stratify theo nhãn): train={len(train_df)}, val={len(val_df)}, "
        f"test={len(test_df)}"
    )

    # -------------------------------------------------------------- (17)
    if args.augment:
        logger.info("(17) Đang cân bằng dữ liệu (Data Augmentation) trên tập TRAIN...")
        train_df = augment_minority_classes(
            train_df,
            text_column="cleaned_text",
            label_column=args.label_column,
            target_ratio=args.augment_target_ratio,
            seed=args.seed,
            logger=logger,
        )
    else:
        logger.info("(17) Bỏ qua Data Augmentation (--no_augment).")

    # -------------------------------------------------------------- (14)
    logger.info("(14) Đang tách từ tiếng Việt (Word Segmentation) cho cả 3 tập...")
    segmenter = WordSegmenter()
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df["segmented_text"] = split_df["cleaned_text"].progress_apply(segmenter.segment)
        logger.info(f"  Đã tách từ xong tập {name} ({len(split_df)} dòng)")

    # -------------------------------------------------------------- Hoàn thiện & lưu
    def finalize(split_df: pd.DataFrame) -> pd.DataFrame:
        out = split_df.copy()
        out[cfg.TEXT_COLUMN] = out["segmented_text"]  # cột chính, sẵn sàng cho tokenizer
        # Tinh lai comment_length va word_count tu raw_text (vi augmentation co the tao text moi)
        if "raw_text" in out.columns:
            out["comment_length"] = out["raw_text"].fillna("").str.len().astype(int)
            out["word_count"] = out["raw_text"].fillna("").str.split().str.len().fillna(0).astype(int)
        else:
            out["comment_length"] = out[cfg.TEXT_COLUMN].fillna("").str.len().astype(int)
            out["word_count"] = out[cfg.TEXT_COLUMN].fillna("").str.split().str.len().fillna(0).astype(int)
        # Dam bao has_image ton tai (mac dinh 0 neu khong co trong du lieu goc)
        if "has_image" not in out.columns:
            out["has_image"] = 0
        ordered_cols = (
            [cfg.TEXT_COLUMN, args.label_column]
            + [c for c in cfg.METADATA_COLUMNS if c in out.columns]
            + ["raw_text", "cleaned_text"]
        )
        return out[ordered_cols]

    train_final = finalize(train_df)
    val_final = finalize(val_df)
    test_final = finalize(test_df)

    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    train_final.to_csv(args.train_out, index=False, encoding="utf-8-sig")
    val_final.to_csv(args.val_out, index=False, encoding="utf-8-sig")
    test_final.to_csv(args.test_out, index=False, encoding="utf-8-sig")

    logger.info("=" * 60)
    logger.info(f"Đã lưu train ({len(train_final)} dòng) -> {args.train_out}")
    logger.info(f"Đã lưu val   ({len(val_final)} dòng) -> {args.val_out}")
    logger.info(f"Đã lưu test  ({len(test_final)} dòng) -> {args.test_out}")
    logger.info("Tiếp theo: chạy `python -m src.train` để huấn luyện mô hình.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
