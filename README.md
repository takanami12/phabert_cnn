# PhaBERT-CNN

**PhaBERT-CNN** là một hệ thống ứng dụng Deep Learning để phân loại lối sống của thực khuẩn thể (phage lifestyle) từ các đoạn bộ gen ngắn (contigs). Mô hình kết hợp sức mạnh của ngôn ngữ học DNA với **DNABERT-2** và **Mạng Nơ-ron Tích chập đa tầng (Multi-Scale CNN)**.

Đặc biệt, dự án hỗ trợ cấu trúc **Gene-Gated**, cho phép kết hợp các đặc trưng sinh học truyền thống như danh mục gen (thông qua HMM) và điểm số pathway sinh học, qua đó nâng cao độ chính xác dự đoán.

## 🚀 Tính Năng Nổi Bật
- **Lõi DNABERT-2 Nhúng Đặc Trưng**: Rút gọn chuỗi DNA tự nhiên thành các vector nhúng (embeddings) giàu ngữ nghĩa.
- **Kiến Trúc CNN & Attention**: Trích xuất các mô hình ngữ cảnh cục bộ và tích hợp chúng thành thông tin toàn cục để dự đoán độc lực hoặc ôn hoà.
- **Gene Gating System**: Tích hợp các đặc trưng gen (Mật độ gen, xu hướng chiều mã hóa, cấu hình HMM theo chuẩn hệ VOG/EBI).
- **Tối Ưu Hóa Bộ Nhớ Cache & LoRA**: Hỗ trợ xuất cache `hidden states` cho giai đoạn tập huấn warm-up, cũng như kỹ thuật LoRA của **PEFT** để tinh chỉnh siêu tốc.

## 📁 Cấu Trúc Dự Án
```text
phabert_cnn/
├── data_annotation/        # Xử lý CSDL HMM, quét và tiên đoán các nhóm gen đặc trưng
├── models/               # Khởi tạo thuật toán phân loại (Baseline và Gated)
├── scripts/              # Chứa mã nguồn chính (Chuẩn bị Dữ liệu, Huấn luyện, Đánh giá)
├── utils/                # Các tiện ích (Trích xuất các phép đo mét-rích, Xử lý tính hiệu chuẩn chéo)
```
Dữ liệu được lưu ở link drive sau: https://drive.google.com/drive/folders/1EHFSOyCk38UqWvrX9kerKo2eTNSoe4bp?usp=drive_link

## ⚙️ Cài Đặt
```bash
# Khởi tạo môi trường ảo (Khuyến khích)
# Python >= 3.9
pip install -r requirements.txt
```

## 🛠️ Hướng Dẫn Sử Dụng
### 1. Khởi Tạo Dữ Liệu
Chạy script dưới đây để tải về, làm sạch, và tạo ra các bộ mẫu contig theo nhiều tỉ lệ khác nhau (Groups A, B, C, D tượng trưng cho độ dài contig tăng dần):
```bash
python scripts/prepare_data.py --output_dir data/processed --groups A,B,C,D
```

*(Lưu ý: Nếu bật tính năng `--no_features`, mô hình sẽ thuần túy tập trung vào chuỗi)*

### 2. Quá Trình Huấn Luyện (Training)
Huấn luyện cho Nhóm A, Nếp gấp (Fold) 0, áp dụng kiến trúc chuẩn baseline:
```bash
python scripts/train.py \
    --group A --fold 0 \
    --data_dir data/processed --output_dir results 
```

**Sử dụng cơ chế Gated + LoRA (Bản nâng cao)**:
```bash
python scripts/train.py \
    --group A --fold 0 --gated --lora \
    --data_dir data/processed --output_dir results 
```

### 3. Đánh Giá (Evaluation)
Sử dụng các đối tượng trọng số tốt nhất được ghi lại từ quá trình huấn luyện:
```bash
# Áp dụng cho mô hình cơ sở
python scripts/evaluate.py --group A

# Áp dụng cho mô hình Gated
python scripts/evaluate.py --group A --gated
```

## 🛡️ Giấy Phép & Bảo Mật
Mã nguồn đã được rà soát không chứa bất kỳ khóa bảo mật, tài nguyên máy cục bộ và token tư nhân. Tất cả phụ thuộc tài nguyên đều hoàn toàn độc lập và an toàn cho sử dụng công cộng mã nguồn mở.

## 🔬 Góp Ý Và Khắc Phục (Issues)
Vui lòng gửi vé khắc phục vấn đề tại mục Issue. 
Cảm ơn vì đã sử dụng và đồng hành cùng dự án PhaBERT-CNN!
