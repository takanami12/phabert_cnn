#!/bin/bash
# ============================================================
# PhaBERT-CNN: End-to-End Experiment Pipeline
#
# Runs the full pipeline in order:
#   0. (Optional) HMM annotation pipeline
#   1. (Optional) Data preparation (contigs + feature vectors)
#   2. Training  — across all groups × folds
#   3. Evaluation — across all groups
#
# Usage:
#   # Full run (from project root or scripts/ directory):
#   bash phabert_cnn/scripts/run_all.sh
#
#   # Skip annotation and data preparation (data already ready):
#   bash phabert_cnn/scripts/run_all.sh --skip_annotate --skip_prepare
#
#   # Train without LoRA (full fine-tune, slower):
#   bash phabert_cnn/scripts/run_all.sh --no_lora
#
#   # Select GPU:
#   bash phabert_cnn/scripts/run_all.sh --gpu=1
#
#   # Specific groups / folds:
#   bash phabert_cnn/scripts/run_all.sh --groups=A,B --folds=0,1,2
# ============================================================

set -e   # Exit immediately on first error
set -u   # Treat unset variables as errors

# ---- Always resolve paths relative to the phabert_cnn/ package directory ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # phabert_cnn/ package dir
ROOT_DIR="$(cd "$PKG_DIR/.." && pwd)"            # project root dir
cd "$PKG_DIR"
echo "Working directory: $(pwd)"

# ============================================================
# Default parameters
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
# Parse arguments
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
# Environment config
# ============================================================
export CUDA_VISIBLE_DEVICES=$GPU_ID
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LORA_FLAG=""
if [ "$USE_LORA" = true ]; then
    LORA_FLAG="--lora"
fi

echo "========================================================"
echo "PhaBERT-CNN Experiment Pipeline"
echo "  GPU:             $GPU_ID"
echo "  Groups:          $GROUPS"
echo "  Folds:           $FOLDS"
echo "  LoRA:            $USE_LORA"
echo "  n_families:      $N_FAMILIES"
echo "  Warmup epochs:   $WARMUP_EPOCHS"
echo "  Finetune epochs: $FINETUNE_EPOCHS"
echo "========================================================"

# ============================================================
# Step 0: Annotation pipeline (HMM profiles + gene features)
# ============================================================
if [ "$SKIP_ANNOTATE" = false ]; then
    echo ""
    echo "========================================================"
    echo "Step 0: Running annotation pipeline"
    echo "========================================================"

    HMM_DIR="data/hmm"
    ANNOT_DIR="data/annotations/raw"
    RAW_DIR="data/raw"

    echo "  [0a] Building HMM database (strategy: $HMM_STRATEGY)..."
    python data_annotation/prepare_hmm_profiles.py \
        --strategy "$HMM_STRATEGY" \
        --output_dir "$HMM_DIR"

    echo "  [0b] Extracting genome annotation features → $ANNOT_DIR"
    mkdir -p "$ANNOT_DIR"
    python data_annotation/preprocess_gene_features.py \
        --data_dir "data/processed" \
        --hmm_db   "$HMM_DIR/gene_families.hmm" \
        --vocab    "$HMM_DIR/vocabulary.json" \
        --output_dir "$ANNOT_DIR" \
        --complete_genome

    echo "  Step 0 done."
fi

# ============================================================
# Step 1: Data preparation (contigs + feature vectors)
# ============================================================
if [ "$SKIP_PREPARE" = false ]; then
    echo ""
    echo "========================================================"
    echo "Step 1: Preparing data..."
    echo "========================================================"

    HMM_DIR="data/hmm"
    ANNOT_DIR="$ROOT_DIR/phabert_cnn/data/annotations/raw"

    python scripts/prepare_data.py \
        --data_dir   data/raw \
        --output_dir data/processed \
        --annot_dir  "$ANNOT_DIR" \
        --vocab      "$HMM_DIR/vocabulary.json" \
        --skip_download

    echo "  Step 1 done."
fi

# ============================================================
# Step 2: Training — all groups × folds
# ============================================================
echo ""
echo "========================================================"
echo "Step 2: Training..."
echo "========================================================"

TRAIN_START=$(date +%s)

for group in A B C D; do
    for ((fold=0; fold < $FOLDS; fold++)); do
        echo ""
        echo "--------------------------------------------------------"
        echo "Training: Group $group, Fold $fold  [LoRA=$USE_LORA]"
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

        echo "  Done: Group $group, Fold $fold"
    done
done

TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$(( TRAIN_END - TRAIN_START ))
echo ""
echo "  Total training time: $(( TRAIN_ELAPSED / 3600 ))h $(( (TRAIN_ELAPSED % 3600) / 60 ))m"

# ============================================================
# Step 3: Evaluation — all groups
# ============================================================
echo ""
echo "========================================================"
echo "Step 3: Evaluating..."
echo "========================================================"

for group in A B C D; do
    echo ""
    echo "--------------------------------------------------------"
    echo "Evaluating: Group $group"
    echo "--------------------------------------------------------"

    python scripts/evaluate.py \
        --group       "$group"       \
        --gated                      \
        --n_families  "$N_FAMILIES"  \
        --num_workers "$NUM_WORKERS" \
        --eval_split  val            \
        $LORA_FLAG

    echo "  Done: Group $group"
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================================"
echo "All experiments complete!"
echo "========================================================"
echo ""
echo "  Metrics:     $PKG_DIR/results/metrics/"
echo "  Checkpoints: $PKG_DIR/results/group_*/fold_*/"
