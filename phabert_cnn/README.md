# PhaBERT-CNN: Deep Learning Based Methods for Phage Contig Prediction

A hybrid deep learning framework integrating DNABERT-2 pre-trained foundation model with multi-scale CNN and attention-based pooling for bacteriophage lifestyle classification from metagenomic contigs.

## Project Structure

```
phabert_cnn/
├── configs/
│   └── config.yaml          # All hyperparameters and settings
├── data/
│   └── raw/                  # Place raw genome FASTA files here
├── models/
│   ├── phabert_cnn.py        # PhaBERT-CNN model architecture
│   └── attention.py          # Attention-based pooling module
├── utils/
│   ├── dataset.py            # Dataset & DataLoader utilities
│   ├── contig_generator.py   # Contig generation with sliding window
│   ├── metrics.py            # Evaluation metrics (sn, sp, acc)
│   └── data_download.py      # Download raw data from DeePhage/DeepPL
├── scripts/
│   ├── prepare_data.py       # Full data preparation pipeline
│   ├── train.py              # Training script with 2-phase fine-tuning
│   ├── evaluate.py           # Evaluation script
│   └── run_all.sh            # Run full experiment pipeline
├── results/                  # Saved results and checkpoints
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Download and prepare the dataset (DeePhage + DeepPL combined dataset):

```bash
python scripts/prepare_data.py
```

This will:
- Download phage genome data from DeePhage and DeepPL sources
- Generate contigs in 4 length groups (A: 100-400bp, B: 400-800bp, C: 800-1200bp, D: 1200-1800bp)
- Create reverse complement augmentations
- Apply stratified 5-fold cross-validation splits
- Apply random undersampling for class balance

### 3. Train

Train PhaBERT-CNN with 5-fold cross-validation:

```bash
# Train on a specific group (A/B/C/D) and fold (0-4)
python scripts/train.py --group A --fold 0

# Train all groups and folds
bash scripts/run_all.sh
```

### 4. Evaluate

```bash
python scripts/evaluate.py --group A
```

## Key Results (from paper)

| Method      | Metric | Group A (100-400bp) | Group B (400-800bp) | Group C (800-1200bp) | Group D (1200-1800bp) |
|-------------|--------|---------------------|---------------------|----------------------|-----------------------|
| PhaBERT-CNN | sn     | 82.00%              | 89.91%              | 91.12%               | 88.47%                |
| PhaBERT-CNN | sp     | 80.15%              | 80.44%              | 85.93%               | 90.95%                |
| PhaBERT-CNN | acc    | 81.59%              | 87.91%              | 90.01%               | 90.69%                |

## Model Architecture

1. **DNABERT-2 backbone**: Pre-trained genome foundation model (768-dim embeddings)
2. **Multi-scale CNN branch**: 3 parallel Conv1d pathways (kernel sizes 3, 5, 7), each with 768→256→128 channels
3. **Attention-based pooling branch**: Self-attention mechanism for global sequence representation (768→128)
4. **Classification head**: Concatenated 512-dim features → LayerNorm → Linear → Binary output

## Training Strategy

- **Phase 1 (Warm-up)**: Freeze DNABERT-2, train CNN + attention layers for 1 epoch (lr=2e-3)
- **Phase 2 (Full fine-tuning)**: Unfreeze all, discriminative learning rates (DNABERT-2: 1e-5, task layers: 1e-4), up to 10 epochs with early stopping (patience=3)

## Dataset

- 2,241 complete phage genomes (707 temperate, 1,534 virulent)
- Sources: DeePhage + DeepPL datasets
- 5-fold stratified cross-validation (train:val = 8:2 per fold)
- Data augmentation: reverse complement sequences
- Class balancing: random undersampling on training set

## Citation

```
Son Vu Quang and Diep Thi Hoang. "Deep Learning Based Methods for Phage Contig Prediction."
University of Engineering and Technology, Vietnam National University, Hanoi, Vietnam.
```
