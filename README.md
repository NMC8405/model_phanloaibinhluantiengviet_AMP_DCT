# Phân loại cảm xúc bình luận tiếng Việt — PhoBERT-AMP

Hệ thống AI phân loại bình luận thương mại điện tử tiếng Việt thành 3 nhãn
**negative (tiêu cực) / normal (bình thường) / positive (tích cực)**, sử dụng
kiến trúc **PhoBERT-AMP (Adaptive Metadata-aware PhoBERT)** — kết hợp ngữ
nghĩa văn bản (PhoBERT + Multi-Kernel CNN) với thông tin metadata đi kèm
bình luận (rating, chất lượng sản phẩm) thông qua cơ chế
**Adaptive Metadata Gate** tự động quyết định mức độ sử dụng metadata.

## Kiến trúc mô hình đề xuất

```
Văn bản → Tiền xử lý → Word Segmentation → PhoBERT Tokenizer → PhoBERT
   (Contextual Embedding) → Multi-Kernel CNN (Conv1D + BatchNorm + ReLU)
   → Global Max Pooling → Vector văn bản (512d) ─────────────────────────┐
                                                                          │
                                                          ┌──Adaptive ────┤
                                                          │  Metadata     │
                                                          │   Gate ───────┤
                                                          │  (σ ∈ 0~1)   │
Metadata → Tiền xử lý → Per-Field Embedding ─────────────┘               │
  (rating→32d, quality→32d = 64d)                                         │
                                                                          │
                     Gated Metadata (gate × meta_vec) ────────────────────┘
                                                                          │
                     Concatenate(text_vec, gated_meta) → LayerNorm (576d) │
                     → FC(256) + ReLU → Dropout → Classifier → 3 lớp
```

**Điểm khác biệt so với mô hình cũ (PhoBERT + CNN + Metadata):**

| Thành phần | Mô hình cũ | PhoBERT-AMP (v2) |
|---|---|---|
| CNN | Conv1D → ReLU → MaxPool | Conv1D → **BatchNorm** → ReLU → MaxPool |
| Metadata Embedding | Dense chung (3→64) | **Per-Field Embedding** (mỗi trường riêng 1→32, bỏ num_likes để giảm nhiễu) |
| Fusion | Concat cứng | **Adaptive Metadata Gate** (học trọng số ∈ 0-1) |
| Normalization | Không có | **LayerNorm** trước FC |
| Kích thước fusion | 576 (512+64) | **576** (512 + 64) |


---

## 1. Cấu trúc project

```
├── config.py                    # Cấu hình tập trung (đường dẫn, hyperparameter, schema)
├── requirements.txt
├── setup.py                     # Hỗ trợ pip install -e . (stable imports)
├── docs/                        # Báo cáo học thuật và tài liệu cải tiến mô hình
│   ├── Báo_cáo_chi_tiết_Dự_án_PhoBERT_AMP.docx
│   ├── cải tiến model.docx      # Nghiên cứu cơ chế Cross-Attention
│   └── Tóm tắt mô hình đề xuất.docx
├── data/
│   ├── raw/comments.csv         # Dữ liệu 56,878 dòng (chuyển đổi từ DataCrawl.xlsx)
│   └── processed/                # train.csv / val.csv / test.csv (sinh ra bởi prepare_data.py)
├── src/
│   ├── preprocessing/
│   │   ├── dictionaries.py       # Từ điển teencode, viết tắt, emoji, icon, đồng nghĩa
│   │   ├── text_cleaner.py       # TextCleaner: 11 bước làm sạch văn bản
│   │   ├── word_segmenter.py     # Tách từ tiếng Việt (pyvi)
│   │   └── metadata_processor.py # Chuẩn hoá rating / chất lượng sản phẩm
│   ├── augmentation/
│   │   └── eda_augmenter.py      # EDA augmentation (cân bằng lớp thiểu số)
│   ├── dataset.py                # PyTorch Dataset + dynamic padding collate_fn
│   ├── model.py                  # Kiến trúc PhoBERT-AMP (Adaptive Metadata Gate)
│   ├── train.py                  # Script huấn luyện + gate regularization
│   ├── evaluate.py               # Đánh giá: Accuracy/Precision/Recall/F1/Confusion Matrix
│   ├── predict.py                # Suy luận (CLI + class SentimentPredictor) + gate_score
│   └── utils.py
├── scripts/
│   ├── convert_excel_to_csv.py   # Chuyển đổi file Excel gốc sang CSV (DataCrawl.xlsx / update.xlsx)
│   ├── generate_report.py        # Sinh bộ biểu đồ, ma trận nhầm lẫn và bảng tổng hợp ảnh
│   ├── generate_docx_report.py   # Tự động xuất báo cáo học thuật chuẩn format Word (.docx)
│   ├── prepare_data.py           # Pipeline chuẩn bị dữ liệu đầy đủ (17 bước tiền xử lý)
│   ├── run_train.bat             # Windows: chạy trọn pipeline 1 lệnh
│   └── run_train.sh              # Linux/Mac: chạy trọn pipeline 1 lệnh
├── api/
│   ├── app.py                    # FastAPI microservice (/predict, /predict/batch, /health)
│   ├── schemas.py                # Pydantic schemas (có gate_score)
│   └── index.html                # Giao diện web demo (2 trường metadata)
├── tests/
│   └── test_preprocessing.py     # 25 unit test (không cần GPU/PhoBERT)
└── outputs/
    ├── checkpoints/              # best_model.pt, last_model.pt (loại trừ trên Git)
    ├── logs/                     # log huấn luyện
    └── reports/                  # confusion_matrix.png, training_curves.png, metrics JSON
```

---

## 2. Cài đặt

Yêu cầu Python **3.10+**. Khuyến nghị dùng GPU (project đã tối ưu riêng cho
VRAM hạn chế như **RTX 4060 8GB**: mixed precision, gradient accumulation,
dynamic padding — xem mục 4).

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

> **Torch + CUDA:** nếu muốn train trên GPU, cài `torch` bản có CUDA phù hợp
> với driver máy bạn **trước** khi chạy `pip install -r requirements.txt`
> (lấy đúng lệnh tại https://pytorch.org/get-started/locally/). Ví dụ Windows
> + CUDA 12.1: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

Kiểm tra cài đặt đúng:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
pytest tests/ -v
```

---

## 3. Chuẩn bị dữ liệu

### 3.0. Chuyển đổi file Excel (nếu cần)

Project hỗ trợ chuyển đổi từ file Excel `DataCrawl.xlsx` (56,878+ dòng) hoặc `update.xlsx`:

```bash
# Mặc định tự động tìm và đọc file DataCrawl.xlsx (hoặc chỉ định đường dẫn)
python -m scripts.convert_excel_to_csv --input ../DataCrawl.xlsx
```

Script tự động:
- Đọc file Excel, tự động nhận diện sheet dữ liệu (`Du lieu train`)
- Mapping cột (Cmt → comment, Phan loai → label, rating, Chất lượng sản phẩm → quality)
- Chuẩn hoá nhãn (Negative → negative, Normal → normal, Positive → positive)
- Lọc bỏ dòng comment rỗng hoặc text 'None', loại bỏ hoàn toàn bình luận trùng lặp
- Lưu ra `data/raw/comments.csv`

### 3.1. Định dạng file CSV đầu vào

File `data/raw/comments.csv` với các cột:

| Cột                  | Bắt buộc | Ý nghĩa                                      |
|-----------------------|:--------:|-----------------------------------------------|
| `comment`             | ✅       | Nội dung bình luận (văn bản thô, chưa xử lý)  |
| `label`                | ✅       | `negative` \| `normal` \| `positive`          |
| `rating`               | khuyến nghị | Số sao đánh giá (1–5)                     |
| `quality`              | khuyến nghị | Điểm chất lượng sản phẩm (1–5)            |

> **Ghi chú về `num_likes`**: Trong phiên bản tối ưu hóa mới nhất, trường `num_likes` đã được loại bỏ khỏi `METADATA_SCHEMA` để tránh gây nhiễu và làm mất cân bằng gate score. Hệ thống tập trung tối ưu vào 2 thuộc tính cốt lõi: `rating` và `quality`.


### 3.2. Chạy pipeline chuẩn bị dữ liệu

```bash
python -m scripts.prepare_data
```

Script thực hiện tuần tự đúng 17 bước trong bảng tiền xử lý của nhóm:

1–2. Loại bỏ trùng lặp / dữ liệu rỗng
3–13. Làm sạch văn bản (`TextCleaner`): HTML → Unicode NFC → emoji → icon →
URL → hoa/thường → khoảng trắng → **ký tự lặp** → **dấu câu** → teencode →
viết tắt *(ghi chú: ký tự lặp chạy trước dấu câu — xem giải thích trong
`text_cleaner.py`, nếu không dấu "..." sẽ bị chính bước ký tự lặp nén tiếp
thành "..")*
→ **Chia tập train/val/test** (stratify theo nhãn, **trước** khi augment)
17. **Data Augmentation** — chỉ áp dụng cho tập train, chỉ cho lớp thiểu số,
đưa các lớp về cân bằng (EDA: Synonym Replacement / Random Insertion /
Random Swap / Random Deletion)
14. Word Segmentation (tách từ, áp dụng sau cùng để dữ liệu augment cũng
được tách từ nhất quán)

Kết quả: `data/processed/{train,val,test}.csv`, mỗi dòng gồm văn bản đã tách
từ (sẵn sàng cho PhoBERT tokenizer) + nhãn + metadata + văn bản gốc (để đối
chiếu).

Tuỳ chọn:
```bash
python -m scripts.prepare_data --no_augment                    # tắt augmentation
python -m scripts.prepare_data --augment_target_ratio 0.7      # cân bằng 70% thay vì 100%
python -m scripts.prepare_data --train_ratio 0.8 --val_ratio 0.1
```

---

## 4. Huấn luyện mô hình

```bash
python -m src.train
```

Hoặc chạy trọn cả 3 bước (chuẩn bị dữ liệu → train → đánh giá) bằng 1 lệnh:
```bash
scripts\run_train.bat      REM Windows
bash scripts/run_train.sh  # Linux/Mac
```

### Các kỹ thuật đã tích hợp (tối ưu cho GPU VRAM hạn chế, vd RTX 4060 8GB)

| Kỹ thuật | Vì sao cần |
|---|---|
| **Discriminative learning rate** | PhoBERT dùng LR nhỏ (2e-5, fine-tune nhẹ nhàng), phần CNN+metadata+gate+FC (khởi tạo ngẫu nhiên) dùng LR lớn hơn (1e-3) để học nhanh hơn |
| **Gradient accumulation** | Mô phỏng batch size lớn hơn mà không cần thêm VRAM |
| **Mixed precision (AMP)** | Giảm VRAM + tăng tốc trên GPU (tự động bật khi có CUDA) |
| **Dynamic padding** | Pad theo độ dài lớn nhất *trong từng batch* thay vì cố định — tiết kiệm tính toán |
| **Warmup + linear decay LR schedule** | Chuẩn khi fine-tune BERT, tránh phá vỡ trọng số pretrained ở đầu quá trình train |
| **Class weights** | Bù đắp mất cân bằng lớp còn sót lại sau augmentation |
| **Early stopping theo macro-F1** | Chống overfit, chọn checkpoint tốt nhất trên tập validation |
| **Adaptive Metadata Gate + L1 Reg** | Khuyến khích model tự quyết định không dùng metadata khi text đã đủ rõ ràng (λ=0.01) |

### Tham số dòng lệnh thường dùng

```bash
python -m src.train --epochs 20 --batch_size 16 --gradient_accumulation_steps 2
python -m src.train --freeze_phobert          # đóng băng PhoBERT nếu VRAM quá hạn chế / data quá nhỏ
python -m src.train --gate_reg_lambda 0.02    # tăng L1 penalty cho gate (ép model dùng ít metadata hơn)
python -m src.train --per_field_embed_dim 64  # tăng kích thước embedding mỗi trường metadata
python -m src.train --phobert_model_name vinai/phobert-large   # nếu muốn thử bản large
```

Nếu gặp lỗi **CUDA out of memory**: giảm `--batch_size` (vd 8), tăng
`--gradient_accumulation_steps` tương ứng để giữ nguyên effective batch
size, hoặc thêm `--freeze_phobert`.

Kết quả: `outputs/checkpoints/best_model.pt` (checkpoint tốt nhất theo
validation macro-F1), `outputs/reports/training_curves.png` (3 biểu đồ:
Loss, Accuracy/F1, **Gate Score theo epoch**),
`outputs/reports/training_history.json`.

---

## 5. Đánh giá mô hình

```bash
python -m src.evaluate
```

In ra Accuracy / Precision / Recall / F1 (macro + theo từng lớp) trên tập
**test** (giữ riêng, không dùng lúc train/early-stopping), đồng thời lưu:
- `outputs/reports/test_evaluation.json` — số liệu chi tiết
- `outputs/reports/confusion_matrix.png` — confusion matrix (số lượng +
  chuẩn hoá theo hàng)
- `outputs/reports/classification_report.txt` — báo cáo đầy đủ

---

## 6. Dự đoán bình luận mới (Inference)

### Dùng trực tiếp trong Python

```python
from src.predict import SentimentPredictor

predictor = SentimentPredictor()  # tự nạp best_model.pt + tokenizer + scaler
result = predictor.predict(
    "Giao hàng nhanh, đóng gói cẩn thận, sản phẩm dùng rất tốt!",
    metadata={"rating": 5, "num_likes": 12, "quality": 5.0},
)
print(result["label"], result["confidence"])
print(f"Gate Score: {result['gate_score']:.3f}")  # mức độ dùng metadata
print(result["probabilities"])
```

`predict()`/`predict_batch()` nhận **văn bản thô** (chưa xử lý gì) — hàm tự
động chạy lại đúng pipeline `TextCleaner → WordSegmenter` như lúc train để
tránh lệch phân phối dữ liệu (train-serve skew).

**Gate Score** cho biết mức độ mô hình sử dụng metadata cho bình luận đó:
- `gate_score ≈ 0`: model chỉ dựa vào text (văn bản đã đủ rõ ràng)
- `gate_score ≈ 1`: model dùng đầy đủ metadata (text mơ hồ, cần rating/likes)

### Dùng qua CLI

```bash
python -m src.predict --text "Sản phẩm tệ, không đáng tiền" --rating 1
python -m src.predict --input_csv data/raw/new_comments.csv --output_csv outputs/reports/predictions.csv
```

### Dùng qua API (FastAPI microservice)

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Xem tài liệu API tự động tại `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Giao hàng nhanh, sản phẩm tốt!", "rating": 5, "quality": 5}'
```

Response mẫu:
```json
{
  "text": "Giao hàng nhanh, sản phẩm tốt!",
  "label": "positive",
  "confidence": 0.96,
  "probabilities": {"negative": 0.01, "normal": 0.03, "positive": 0.96},
  "gate_score": 0.28
}
```

**Gọi từ ASP.NET Core MVC (C#)** — cùng mô hình microservice như service
gợi ý sản phẩm hiện có của nhóm:

```csharp
public class SentimentClient
{
    private readonly HttpClient _http;
    public SentimentClient(HttpClient http) => _http = http; // base URL: http://localhost:8000

    public async Task<SentimentResult> PredictAsync(string text, double? rating = null, double? quality = null)
    {
        var payload = new { text, rating, quality };
        var response = await _http.PostAsJsonAsync("/predict", payload);
        response.EnsureSuccessStatusCode();

        return await response.Content.ReadFromJsonAsync<SentimentResult>();
    }
}

public record SentimentResult(
    string Text, string Label, double Confidence,
    Dictionary<string, double> Probabilities, double? GateScore);
```

---

## 7. Kiến trúc mô hình chi tiết (`src/model.py`)

### Nhánh văn bản — Multi-Kernel CNN

PhoBERT (`vinai/phobert-base`, lấy `last_hidden_state`) →
CNN nhiều kích thước bộ lọc `[2,3,4,5]` × 128 filter mỗi loại (theo TextCNN,
Kim 2014 — bắt các cụm từ 2-5 từ liên tiếp mang tín hiệu cảm xúc) →
**BatchNorm1d** (chuẩn hoá output sau mỗi Conv1d để ổn định gradient) →
ReLU → Global Max Pooling **có mask** (loại bỏ ảnh hưởng của các vị trí
padding) → **vector 512 chiều**.

### Nhánh metadata — Per-Field Embedding

Mỗi trường metadata (`rating`, `quality`) có **riêng** 1 embedding
layer: `Linear(1 → 32) + ReLU + Dropout`. Sau đó concat lại →
**vector 64 chiều** (2 × 32). Trường `num_likes` đã được loại bỏ để tập trung
vào các đặc trưng có tính phân loại cao và không gây nhiễu.

Lợi ích: model học biểu diễn độc lập cho từng trường metadata trước khi kết hợp với văn bản.

### Adaptive Metadata Gate — Cơ chế trọng tâm

Gate quyết định **mức độ sử dụng metadata** cho **từng sample**:

```
gate_input = concat(text_vector, metadata_vector)     # (batch, 576)
gate_score = σ(W₂ · ReLU(W₁ · gate_input + b₁) + b₂)  # scalar ∈ (0,1)
gated_metadata = gate_score × metadata_vector           # (batch, 64)
```

- **gate ≈ 0**: text đã rõ ràng (vd "sản phẩm rất tệ") → bỏ qua metadata
- **gate ≈ 1**: text mơ hồ (vd "ok", "dùng được") → dựa thêm metadata (rating, quality)

Gate được học end-to-end cùng model, có L1 regularization (`λ=0.015`) để
khuyến khích model tự quyết định không phụ thuộc metadata khi không cần.

### Fusion

Concatenate(text_vector 512d + gated_metadata 64d = 576d) → **LayerNorm** →
Linear(256) + ReLU → Dropout(0.3) → Linear(3) → **logits thô**.

Toàn bộ kích thước khai báo tập trung tại `ModelConfig` trong `config.py`.

---

## 8. Dữ liệu & Kết quả thực nghiệm

### 8.1. Phân bố tập dữ liệu (56,878 bình luận)

| Nhãn | Số lượng | Tỷ lệ |
|------|----------|--------|
| Positive (Tích cực) | 22,616 | 39.8% |
| Normal (Bình thường) | 19,175 | 33.7% |
| Negative (Tiêu cực) | 15,087 | 26.5% |
| **Tổng cộng** | **56,878** | **100.0%** |

### 8.2. Kết quả kiểm thử trên Test Set (8,533 mẫu)

Đánh giá thực nghiệm chi tiết sau khi huấn luyện (xem thêm tại `outputs/reports/`):

| Chỉ số | Giá trị |
|---|---|
| **Accuracy (Độ chính xác toàn cục)** | **90.80%** |
| **Macro F1-Score** | **91.54%** |
| **Weighted F1-Score** | **90.80%** |

**Chi tiết theo từng lớp (Per-class Metrics):**

| Lớp | Precision | Recall | F1-Score | Số mẫu test (Support) |
|---|---|---|---|---|
| **Negative (Tiêu cực)** | **99.12%** | **99.56%** | **99.34%** | 2,263 |
| **Normal (Bình thường)** | 86.02% | 87.45% | 86.73% | 2,877 |
| **Positive (Tích cực)** | 89.33% | 87.80% | 88.56% | 3,393 |


---

## 9. Mở rộng dự án

- **Thêm teencode/viết tắt/emoji/từ đồng nghĩa:** sửa trực tiếp các dict
  trong `src/preprocessing/dictionaries.py`.
- **Đổi bộ tách từ** (vd sang VnCoreNLP/RDRSegmenter — đúng bộ PhoBERT dùng
  khi pretrain): sửa `src/preprocessing/word_segmenter.py` (đã ghi chú sẵn
  cách đổi trong file).
- **Thêm trường metadata mới:** khai báo trong `METADATA_SCHEMA` (`config.py`)
  với loại `"numeric"` / `"numeric_log"` / `"binary"`, đảm bảo cột đó có mặt
  trong CSV thô — `MetadataProcessor` tự xử lý phần còn lại. Per-Field
  Embedding sẽ tự động tạo thêm 1 embedding layer cho trường mới.
- **Đổi kích thước PhoBERT:** `--phobert_model_name vinai/phobert-large`
  (cần nhiều VRAM hơn — cân nhắc bật `--freeze_phobert`).
- **Điều chỉnh gate behavior:** tăng `--gate_reg_lambda` để ép model dùng
  ít metadata hơn, giảm hoặc đặt `0` để model tự do quyết định.

---

## 10. Câu hỏi thường gặp

**"Không tìm thấy checkpoint" khi chạy evaluate/predict** → chạy
`python -m src.train` trước.

**Cảnh báo `VisibleDeprecationWarning` từ `pyvi` lúc import** → vô hại
(do phiên bản `numpy` mới hơn bản `pyvi` được build cùng lúc phát hành),
không ảnh hưởng kết quả tách từ.

**Muốn train nhanh thử nghiệm trước khi chạy full** → dùng
`--epochs 2 --cnn_num_filters 32` để có vòng lặp nhanh, kiểm tra pipeline
chạy thông suốt trước khi train đầy đủ.

**Lỗi encoding trên Windows** → project đã xử lý tự động (`chcp 65001` +
`sys.stdout.reconfigure(encoding='utf-8')`).

**Gate Score trong API/CLI** → mỗi prediction có thêm `gate_score` (0~1)
cho biết mức độ model dùng metadata cho bình luận đó. Hữu ích để debug và
giải thích kết quả cho người dùng.

---

## 11. Tài liệu tham khảo

- Nguyen, D. Q., & Nguyen, A. T. (2020). *PhoBERT: Pre-trained language
  models for Vietnamese.* Findings of EMNLP.
- Kim, Y. (2014). *Convolutional Neural Networks for Sentence
  Classification.* EMNLP.
- Wei, J., & Zou, K. (2019). *EDA: Easy Data Augmentation Techniques for
  Boosting Performance on Text Classification Tasks.* EMNLP-IJCNLP.
- Arevalo, J., Solorio, T., Montes-y-Gómez, M., & González, F. A. (2017).
  *Gated Multimodal Units for Information Fusion.* ICLR Workshop.
