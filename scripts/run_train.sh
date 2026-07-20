#!/usr/bin/env bash
# run_train.sh
# =============
# Chay toan bo pipeline: chuan bi du lieu -> train -> danh gia tren tap test.
#
# Cach dung:
#   bash scripts/run_train.sh
#
# Truoc khi chay: dam bao da (1) cai dat requirements.txt, (2) co file
# data/raw/comments.csv dung schema (xem README.md).

set -e  # dung ngay neu co loi

cd "$(dirname "$0")/.."

echo "============================================"
echo "BUOC 1/3: Chuan bi du lieu (prepare_data.py)"
echo "============================================"
python -m scripts.prepare_data

echo ""
echo "============================================"
echo "BUOC 2/3: Huan luyen mo hinh (train.py)"
echo "============================================"
python -m src.train

echo ""
echo "============================================"
echo "BUOC 3/3: Danh gia tren tap test (evaluate.py)"
echo "============================================"
python -m src.evaluate

echo ""
echo "HOAN TAT! Xem ket qua tai outputs/reports/"
