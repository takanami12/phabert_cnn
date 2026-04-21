#!/bin/bash
# ============================================================
# PhaBERT-CNN: Quy trình Thực nghiệm Toàn diện
#
# Kịch bản thực thi toàn bộ pipeline:
#   0. (Tùy chọn) Quy trình gán nhãn lược đồ HMM (HMM annotation pipeline)
#   1. (Tùy chọn) Tiền xử lý và chuẩn bị dữ liệu (Data preparation)
#   2. Huấn luyện (Training) — trên toàn bộ các nhóm (groups) × các nếp gấp (folds)
#   3. Đánh giá (Evaluation) — trên toàn bộ các phân nhóm
#
# Hướng dẫn sử dụng:
#   # Thực thi toàn bộ (từ thư mục gốc hoặc từ thư mục scripts/):
#   bash phabert_cnn/scripts/run_all.sh
#
#   # Lược bỏ bước gán nhãn và chuẩn bị dữ liệu (khi dữ liệu đã sẵn sàng):
#   bash phabert_cnn/scripts/run_all.sh --skip_annotate --skip_prepare
#
#   # Huấn luyện không sử dụng LoRA (thời gian huấn luyện lâu hơn, tinh chỉnh toàn bộ tham số):
#   bash phabert_cnn/scripts/run_all.sh --no_lora
#
#   # Lựa chọn thiết bị cấu hình GPU:
#   bash phabert_cnn/scripts/run_all.sh --gpu=1
#
#   # Chỉ định các phân nhóm / nếp gấp cụ thể:
#   bash phabert_cnn/scripts/run_all.sh --groups=A,B --folds=0,1,2
# ============================================================

set -e   # Chấm dứt ngay khi gặp lỗi đầu tiên
set -u   # Xử lý các biến chưa được gán giá trị như một lỗi hệ thống

# ---- Luôn thiết lập đường dẫn tương đối tới thư mục gói phabert_cnn/ ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # Thư mục phabert_cnn/
ROOT_DIR="$(cd "$PKG_DIR/.." && pwd)"            # Thư mục gốc dự án
cd "$PKG_DIR"
echo "Working directory: $(pwd)"

# ============================================================
# Thiết lập các tham số mặc định (Defaults)
# ============================================================
SKIP_ANNOTATE=false
SKIP_PREPARE=false
USE_LORA=true
GPU_ID=0
GROUPS="A B C D"
FOLDS=5
N_FAMILIES=26
BATCH_SIZE=64
NUM_WORKERS=4
WARMUP_EPOCHS=1
FINETUNE_EPOCHS=10
PATIENCE=3
HMM_STRATEGY="pfam"    # pfam | vog | combined

# ============================================================
# Phân tích cú pháp đối số đầu vào (Parse arguments)
# ============================================================
for arg in "$@"; do
    case $arg in
        --skip_annotate)   SKIP_ANNOTATE=true  ;;
        --skip_prepare)    SKIP_PREPARE=true   ;;
        --no_lora)         USE_LORA=false       ;;
        --gpu=*)           GPU_ID="${arg#*=}"   ;;
        --groups=*)        GROUPS="${arg#*=}"; GROUPS="${GROUPS//,/ }" ;;
        --folds=*)         FOLDS="${arg#*=}";  FOLDS="${FOLDS//,/ }"  ;;
        --hmm=*)           HMM_STRATEGY="${arg#*=}" ;;
        *)                 echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ============================================================
# Cấu hình biến môi trường (Environment Config)
# ============================================================
export CUDA_VISIBLE_DEVICES=$GPU_ID
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LORA_FLAG=""
if [ "$USE_LORA" = true ]; then
    LORA_FLAG="--lora"
fi

echo "========================================================"
echo "Quy trình Thực nghiệm PhaBERT-CNN"
echo "  GPU:           $GPU_ID"
echo "  Groups:        $GROUPS"
echo "  Folds:         $FOLDS"
echo "  LoRA:          $USE_LORA"
echo "  n_families:    $N_FAMILIES"
echo "  Số kỷ nguyên tối ưu hóa sơ bộ (Warmup epochs): $WARMUP_EPOCHS"
echo "  Số kỷ nguyên tinh chỉnh (Finetune epochs): $FINETUNE_EPOCHS"
echo "========================================================"

# ============================================================
# Bước 0: Quy trình gán nhãn (HMM profiles + đặc trưng gene)
# ============================================================
if [ "$SKIP_ANNOTATE" = false ]; then
    echo ""
    echo "========================================================"
    echo "Bước 0: Khởi chạy quy trình gán nhãn (Annotation pipeline)"
    echo "========================================================"

    HMM_DIR="data/hmm"
    ANNOT_DIR="data/annotations/raw"
    RAW_DIR="data/raw"

    echo "  [0a] Xây dựng cơ sở dữ liệu HMM (Chiến lược bảo lưu: $HMM_STRATEGY)..."
    python data_annotation/prepare_hmm_profiles.py \
        --strategy "$HMM_STRATEGY" \
        --output_dir "$HMM_DIR"

    echo "  [0b] Trích xuất đặc trưng chú giải hệ gen → $ANNOT_DIR"
    mkdir -p "$ANNOT_DIR"
    python data_annotation/preprocess_gene_features.py \
        --data_dir "data/processed" \
        --hmm_db   "$HMM_DIR/gene_families.hmm" \
        --vocab    "$HMM_DIR/vocabulary.json" \
        --output_dir "$ANNOT_DIR" \
        --complete_genome

    echo "  Hoàn tất Bước 0."
fi

# ============================================================
# Bước 1: Chuẩn bị dữ liệu (các đoạn contig + vector đặc trưng)
# ============================================================
if [ "$SKIP_PREPARE" = false ]; then
    echo ""
    echo "========================================================"
    echo "Bước 1: Khởi tạo quy trình chuẩn bị dữ liệu..."
    echo "========================================================"

    HMM_DIR="data/hmm"
    ANNOT_DIR="$ROOT_DIR/data/annotations/raw"

    python scripts/prepare_data.py \
        --data_dir   data/raw \
        --output_dir data/processed \
        --annot_dir  "$ANNOT_DIR" \
        --vocab      "$HMM_DIR/vocabulary.json" \
        --skip_download

    echo "  Hoàn tất Bước 1."
fi

# ============================================================
# Bước 2: Huấn luyện (Training) — trên toàn bộ các nhóm × các nếp gấp
# ============================================================
echo ""
echo "========================================================"
echo "Bước 2: Tiến hành huấn luyện mô hình..."
echo "========================================================"

TRAIN_START=$(date +%s)

for group in A B C D; do
    for ((fold=0; fold < $FOLDS; fold++)); do
        echo ""
        echo "--------------------------------------------------------"
        echo "Tiến trình Huấn luyện: Group $group, Fold $fold  [Sử dụng LoRA=$USE_LORA]"
        echo "--------------------------------------------------------"

        python scripts/train.py \
            --group           "$group"           \
            --fold            "$fold"            \
            --gated                              \
            --n_families      "$N_FAMILIES"      \
            --batch_size      "$BATCH_SIZE"      \
            --num_workers     "$NUM_WORKERS"     \
            --warmup_epochs   "$WARMUP_EPOCHS"   \
            --finetune_epochs "$FINETUNE_EPOCHS" \
            --patience        "$PATIENCE"        \
            $LORA_FLAG

        echo "  Hoàn thành Huấn luyện: Group $group, Fold $fold"
    done
done

TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$(( TRAIN_END - TRAIN_START ))
echo ""
echo "  Tổng thời gian huấn luyện thực thi: $(( TRAIN_ELAPSED / 3600 ))h $(( (TRAIN_ELAPSED % 3600) / 60 ))m"

# ============================================================
# Bước 3: Đánh giá mô hình (Evaluation) — trên toàn bộ các nhóm
# ============================================================
echo ""
echo "========================================================"
echo "Bước 3: Thực thi đánh giá phân loại hình thái..."
echo "========================================================"

for group in A B C D; do
    echo ""
    echo "--------------------------------------------------------"
    echo "Đánh giá hiệu suất: Nhóm $group"
    echo "--------------------------------------------------------"

    python scripts/evaluate.py \
        --group       "$group"       \
        --gated                      \
        --n_families  "$N_FAMILIES"  \
        --num_workers "$NUM_WORKERS" \
        --eval_split  val            \
        $LORA_FLAG

    echo "  Hoàn tất Đánh giá: Nhóm $group"
done

# ============================================================
# Tổng kết báo cáo (Summary)
# ============================================================
echo ""
echo "========================================================"
echo "Hoàn thành toàn bộ Thực nghiệm!"
echo "========================================================"
echo ""
echo "  Metrics: $PKG_DIR/results/metrics/"
echo "  Checkpoints: $PKG_DIR/results/group_*/fold_*/"
