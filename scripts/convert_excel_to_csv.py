# -*- coding: utf-8 -*-
"""
scripts/convert_excel_to_csv.py
================================
Chuyển đổi file Excel dữ liệu gốc sang định dạng CSV phù hợp
với pipeline của project (comments.csv).

Hỗ trợ cả hai format:
  - update.xlsx (15k dữ liệu cũ)
  - DataCrawl.xlsx (56k+ dữ liệu mới, sheet 'Du lieu train')

Mapping cột từ Excel sang CSV:
    Excel "Cmt"                    → CSV "comment"
    Excel "Phan loai"              → CSV "label"  (Negative→negative, Normal→normal, Positive→positive)
    Excel "rating"                 → CSV "rating"
    Excel "Like"                   → CSV "num_likes" (cột mới trong DataCrawl.xlsx)
    Excel "Chất lượng sản phẩm"   → CSV "quality" (mã hoá text → điểm 1-5)
    (Tự tính từ comment)           → CSV "comment_length" (số ký tự)
    (Tự tính từ comment)           → CSV "word_count" (số từ)
    Excel "Hình ảnh" / "Image"      → CSV "has_image" (1=có ảnh, 0=không)

Cách chạy (từ thư mục gốc project):
    python -m scripts.convert_excel_to_csv
    python -m scripts.convert_excel_to_csv --input path/to/DataCrawl.xlsx --output data/raw/comments.csv
"""

import argparse
import os
import sys

# Fix encoding cho Windows console (tieng Viet)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Chuyển đổi Excel sang CSV")
    default_excel = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "train_model.xlsx",
    )
    if not os.path.exists(default_excel):
        default_excel = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "DataCrawl.xlsx",
        )

    parser.add_argument(
        "--input",
        default=default_excel,
        help="Đường dẫn file Excel đầu vào (mặc định: train_model.xlsx hoặc DataCrawl.xlsx)",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Tên sheet cần đọc (mặc định: tự động phát hiện)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Đường dẫn file CSV đầu ra (mặc định: data/raw/comments.csv)",
    )
    args = parser.parse_args()

    import config as cfg

    if args.output is None:
        args.output = cfg.RAW_DATA_PATH

    # -------------------------------------------------------------- Đọc Excel
    print(f"[1/5] Đọc file Excel: {args.input}")

    # Tự động phát hiện sheet name
    sheet_name = args.sheet
    if sheet_name is None:
        xl = pd.ExcelFile(args.input, engine="openpyxl")
        available_sheets = xl.sheet_names
        print(f"       Các sheet có sẵn: {available_sheets}")

        # Ưu tiên sheet 'Du lieu train' nếu có, nếu không lấy 'Sheet1' hoặc sheet đầu tiên
        if "Du lieu train" in available_sheets:
            sheet_name = "Du lieu train"
        elif "Sheet1" in available_sheets:
            sheet_name = "Sheet1"
        else:
            sheet_name = available_sheets[0]
        print(f"       Sử dụng sheet: '{sheet_name}'")

    df = pd.read_excel(args.input, sheet_name=sheet_name, engine="openpyxl")
    print(f"       Số dòng đọc được: {len(df)}")
    print(f"       Các cột: {list(df.columns)}")

    # -------------------------------------------------------------- Mapping cột
    print("[2/5] Mapping cột và chuẩn hoá nhãn...")

    # Tạo DataFrame output
    out = pd.DataFrame()

    # comment
    comment_col = None
    for cand in ["Comment", "Cmt", "comment", "cmt"]:
        if cand in df.columns:
            comment_col = cand
            break
    if comment_col is not None:
        out["comment"] = df[comment_col].astype(str)
        print(f"  ✓ Đã mapping cột '{comment_col}' → comment")
    else:
        raise ValueError(f"Không tìm thấy cột bình luận ('Comment' hoặc 'Cmt') trong file Excel. Các cột: {list(df.columns)}")

    # label: chuẩn hoá về lowercase
    label_col = None
    for cand in ["Class", "Phan loai", "Phân loại", "label", "class"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is not None:
        label_map = {
            "Negative": "negative",
            "Normal": "normal",
            "Positive": "positive",
            "negative": "negative",
            "normal": "normal",
            "positive": "positive",
        }
        out["label"] = df[label_col].map(label_map)
        unmapped = out["label"].isna().sum()
        if unmapped > 0:
            bad_labels = df[label_col][out["label"].isna()].unique()
            print(f"  ⚠ {unmapped} dòng có nhãn không hợp lệ: {bad_labels} → sẽ bị loại bỏ")
            out = out.dropna(subset=["label"])
        print(f"  ✓ Đã mapping cột '{label_col}' → label")
    else:
        raise ValueError(f"Không tìm thấy cột nhãn ('Class' hoặc 'Phan loai') trong file Excel. Các cột: {list(df.columns)}")

    # rating
    rating_col = None
    for cand in ["rating", "Rating"]:
        if cand in df.columns:
            rating_col = cand
            break
    if rating_col is not None:
        out["rating"] = pd.to_numeric(df[rating_col], errors="coerce")
        print(f"  ✓ Đã mapping cột '{rating_col}' → rating")
    else:
        print("  ℹ Không có cột 'rating' trong Excel → để trống")

    # num_likes: Sử dụng cột 'Like' từ file Excel nếu có
    like_col = None
    for cand in ["Like", "like", "num_likes", "Num_Likes"]:
        if cand in df.columns:
            like_col = cand
            break
    if like_col is not None:
        out["num_likes"] = pd.to_numeric(df[like_col], errors="coerce").fillna(0).astype(int)
        print(f"  ✓ Đã mapping cột '{like_col}' → num_likes (min={out['num_likes'].min()}, "
              f"max={out['num_likes'].max()}, mean={out['num_likes'].mean():.1f})")
    else:
        out["num_likes"] = 0
        print("  ℹ Không có cột 'Like' → num_likes mặc định = 0")

    # quality: ma hoa tu cot 'Chat luong san pham' hoac 'Quality' thanh diem so 1-5
    quality_map = {
        "tốt": 5.0,
        "Tot": 5.0,
        "ok": 4.0,
        "bình thường": 3.0,
        "bình thường/hỗn hợp": 2.5,
        "kém": 2.0,
        "kém/cần cải thiện": 1.5,
        "dịch vụ sàn cần cải thiện": 1.0,
    }
    quality_col = None
    for cand in ["Quality", "quality", "Chất lượng sản phẩm", "Chất lượng SP"]:
        if cand in df.columns:
            quality_col = cand
            break
    if quality_col is not None:
        out["quality"] = df[quality_col].map(quality_map).fillna(3.0)
        print(f"  ✓ Đã mã hoá cột '{quality_col}' → quality (1-5)")
        print(f"    Phân bố quality: min={out['quality'].min()}, max={out['quality'].max()}, "
              f"mean={out['quality'].mean():.2f}")
        unmapped_quality = set(df[quality_col].dropna().unique()) - set(quality_map.keys())
        if unmapped_quality:
            print(f"  ⚠ Các giá trị quality chưa được map (sẽ dùng mặc định 3.0): {unmapped_quality}")
    else:
        out["quality"] = 3.0
        print("  ℹ Không có cột chất lượng sản phẩm → quality mặc định = 3.0")

    # comment_length: do dai binh luan (so ky tu)
    len_col = None
    for cand in ["Comment_Length", "comment_length", "length"]:
        if cand in df.columns:
            len_col = cand
            break
    if len_col is not None:
        out["comment_length"] = pd.to_numeric(df[len_col], errors="coerce").fillna(out["comment"].str.len()).astype(int)
    else:
        out["comment_length"] = out["comment"].str.len().fillna(0).astype(int)
    print(f"  ✓ Đã tính/mapping comment_length (min={out['comment_length'].min()}, "
          f"max={out['comment_length'].max()}, mean={out['comment_length'].mean():.1f})")

    # word_count: so tu trong binh luan (dem theo khoang trang)
    wc_col = None
    for cand in ["Word_Count", "word_count"]:
        if cand in df.columns:
            wc_col = cand
            break
    if wc_col is not None:
        out["word_count"] = pd.to_numeric(df[wc_col], errors="coerce").fillna(out["comment"].str.split().str.len()).astype(int)
    else:
        out["word_count"] = out["comment"].str.split().str.len().fillna(0).astype(int)
    print(f"  ✓ Đã tính/mapping word_count (min={out['word_count'].min()}, "
          f"max={out['word_count'].max()}, mean={out['word_count'].mean():.1f})")

    # has_image: binh luan co dinh kem anh hay khong (1=co, 0=khong)
    image_col = None
    for candidate in ["HasImage", "has_image", "hasimage", "Hình ảnh", "Hinh anh", "Image", "image"]:
        if candidate in df.columns:
            image_col = candidate
            break
    if image_col is not None:
        out["has_image"] = df[image_col].apply(
            lambda x: 1 if pd.notna(x) and str(x).strip() not in ["", "0", "nan", "None", "False", "false"] else 0
        ).astype(int)
        print(f"  ✓ Đã mapping cột '{image_col}' → has_image "
              f"(có ảnh: {out['has_image'].sum()}, không ảnh: {(out['has_image']==0).sum()})")
    else:
        out["has_image"] = 0
        print("  ℹ Không có cột hình ảnh trong Excel → has_image mặc định = 0")

    # -------------------------------------------------------------- Loại bỏ dòng rỗng / None
    print("[3/5] Lọc dữ liệu...")
    before = len(out)
    # Loại bỏ dòng comment là 'None' (từ ô trống Excel chuyển thành string 'None')
    out = out[~out["comment"].isin(["None", "none", "nan", "NaN", ""])].reset_index(drop=True)
    out = out[out["comment"].str.strip().astype(bool)].reset_index(drop=True)
    print(f"       Loại bỏ dòng comment rỗng/None: {before} → {len(out)}")

    # -------------------------------------------------------------- Loại bỏ trùng lặp
    print("[4/5] Loại bỏ dữ liệu trùng lặp...")
    before = len(out)
    out = out.drop_duplicates(subset=["comment"]).reset_index(drop=True)
    print(f"       Loại bỏ trùng lặp: {before} → {len(out)}")

    # -------------------------------------------------------------- Thống kê
    print(f"\n[5/5] Thống kê dữ liệu cuối cùng:")
    print(f"       Tổng số dòng: {len(out)}")
    print(f"       Phân bố nhãn:")
    for label, count in out["label"].value_counts().items():
        print(f"         {label}: {count} ({count/len(out)*100:.1f}%)")
    print(f"       Rating: min={out['rating'].min()}, max={out['rating'].max()}, "
          f"mean={out['rating'].mean():.2f}")
    print(f"       Num likes: min={out['num_likes'].min()}, max={out['num_likes'].max()}, "
          f"mean={out['num_likes'].mean():.1f}")
    print(f"       Quality: min={out['quality'].min()}, max={out['quality'].max()}, "
          f"mean={out['quality'].mean():.2f}")
    print(f"       Comment length: min={out['comment_length'].min()}, max={out['comment_length'].max()}, "
          f"mean={out['comment_length'].mean():.1f}")
    print(f"       Word count: min={out['word_count'].min()}, max={out['word_count'].max()}, "
          f"mean={out['word_count'].mean():.1f}")
    print(f"       Has image: {out['has_image'].sum()} có ảnh / {(out['has_image']==0).sum()} không ảnh")

    # -------------------------------------------------------------- Lưu file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n✓ Đã lưu file CSV tại: {args.output}")
    print(f"  Tiếp theo chạy: python -m scripts.prepare_data")


if __name__ == "__main__":
    main()
