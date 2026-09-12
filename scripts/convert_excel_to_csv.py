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
    parser.add_argument(
        "--input",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "DataCrawl.xlsx",
        ),
        help="Đường dẫn file Excel đầu vào (mặc định: DataCrawl.xlsx)",
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

        # Ưu tiên sheet 'Du lieu train', nếu không thì dùng sheet đầu tiên
        if "Du lieu train" in available_sheets:
            sheet_name = "Du lieu train"
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
    if "Cmt" in df.columns:
        out["comment"] = df["Cmt"].astype(str)
    else:
        raise ValueError("Không tìm thấy cột 'Cmt' trong file Excel")

    # label: chuẩn hoá về lowercase
    if "Phan loai" in df.columns:
        label_map = {
            "Negative": "negative",
            "Normal": "normal",
            "Positive": "positive",
            "negative": "negative",
            "normal": "normal",
            "positive": "positive",
        }
        out["label"] = df["Phan loai"].map(label_map)
        unmapped = out["label"].isna().sum()
        if unmapped > 0:
            bad_labels = df["Phan loai"][out["label"].isna()].unique()
            print(f"  ⚠ {unmapped} dòng có nhãn không hợp lệ: {bad_labels} → sẽ bị loại bỏ")
            out = out.dropna(subset=["label"])
    else:
        raise ValueError("Không tìm thấy cột 'Phan loai' trong file Excel")

    # rating
    if "rating" in df.columns:
        out["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    else:
        print("  ℹ Không có cột 'rating' trong Excel → để trống")

    # num_likes: Sử dụng cột 'Like' từ DataCrawl.xlsx nếu có
    if "Like" in df.columns:
        out["num_likes"] = pd.to_numeric(df["Like"], errors="coerce").fillna(0).astype(int)
        print(f"  ✓ Đã mapping cột 'Like' → num_likes (min={out['num_likes'].min()}, "
              f"max={out['num_likes'].max()}, mean={out['num_likes'].mean():.1f})")
    else:
        out["num_likes"] = 0
        print("  ℹ Không có cột 'Like' → num_likes mặc định = 0")

    # quality: ma hoa tu cot 'Chat luong san pham' thanh diem so 1-5
    # De tranh model chi dua vao quality (tuong quan cao voi label),
    # ma hoa thanh diem so lien tuc thay vi 3 gia tri trung nhan.
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
    if "Chất lượng sản phẩm" in df.columns:
        out["quality"] = df["Chất lượng sản phẩm"].map(quality_map).fillna(3.0)
        print(f"  ✓ Đã mã hoá cột 'Chất lượng sản phẩm' → quality (1-5)")
        print(f"    Phân bố quality: min={out['quality'].min()}, max={out['quality'].max()}, "
              f"mean={out['quality'].mean():.2f}")
        # Hiển thị các giá trị chưa được map
        unmapped_quality = set(df["Chất lượng sản phẩm"].dropna().unique()) - set(quality_map.keys())
        if unmapped_quality:
            print(f"  ⚠ Các giá trị quality chưa được map (sẽ dùng mặc định 3.0): {unmapped_quality}")
    elif "Chất lượng SP" in df.columns:
        out["quality"] = df["Chất lượng SP"].map(quality_map).fillna(3.0)
        print(f"  ✓ Đã mã hoá cột 'Chất lượng SP' → quality (1-5)")
    else:
        out["quality"] = 3.0  # mac dinh trung binh
        print("  ℹ Không có cột chất lượng sản phẩm → quality mặc định = 3.0")

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

    # -------------------------------------------------------------- Lưu file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n✓ Đã lưu file CSV tại: {args.output}")
    print(f"  Tiếp theo chạy: python -m scripts.prepare_data")


if __name__ == "__main__":
    main()
