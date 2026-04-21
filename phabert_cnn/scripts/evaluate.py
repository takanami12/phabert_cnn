import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phabert_cnn import PhaBERTCNN
from models.phabert_cnn_gated import PhaBERTCNN_GeneGated
from utils.dataset import PhageContigDataset, load_features
from utils.metrics import compute_metrics, print_metrics, aggregate_fold_metrics


# ================================================================
# Định nghĩa tham số cấu hình (Arguments)
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Khung đánh giá hiệu suất mô hình PhaBERT-CNN")

    # --- Định danh Dữ liệu / Mô hình (Data / model identity) ---
    parser.add_argument("--group", type=str, required=True, choices=['A', 'B', 'C', 'D'])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Đường dẫn đầu ra, bắt buộc tương đồng với thư mục đã thiết lập trong giai đoạn huấn luyện (--output_dir)")
    parser.add_argument("--output_dir", type=str, default="results/metrics")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--eval_split", type=str, default="val",
                        choices=['val', 'test'])

    # --- Kích hoạt tham số mô hình (áp dụng logic tương tự tiến trình huấn luyện để khôi phục tệp trọng số) ---
    parser.add_argument("--gated", action="store_true")
    parser.add_argument("--no_gate", action="store_true")
    parser.add_argument("--no_gene_stats", action="store_true")
    parser.add_argument("--no_pathway_scores", action="store_true")
    parser.add_argument("--lora", action="store_true",
                        help="Xác định mô hình đã được huấn luyện bằng phương pháp tối ưu hóa tham số cục bộ (LoRA), ảnh hưởng tới tên cấu trúc thư mục lưu trữ")
    parser.add_argument("--n_families", type=int, default=26)

    # --- Tham số cho quá trình suy luận (Inference parameters) ---
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--model_name", type=str, default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


# ================================================================
# Các hàm bổ trợ (Helper functions)
# ================================================================

def get_mode_suffix(args) -> str:
    """Định dạng hậu tố của thư mục trọng số (checkpoint) — tuân thủ logic từ tệp train.py nhằm bảo đảm sự nhất quán."""
    suffix = "gated" if args.gated else "baseline"
    if args.gated:
        if args.no_gate:       suffix += "_nogate"
        if args.no_gene_stats: suffix += "_nostats"
        if args.lora:          suffix += "_lora"
    return suffix


def get_checkpoint_path(args, fold_idx: int) -> Path:
    suffix = get_mode_suffix(args)
    return (Path(args.results_dir)
            / f"group_{args.group}"
            / f"fold_{fold_idx}_{suffix}"
            / "best_model.pt")


@torch.no_grad()
def evaluate_fold(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    gated: bool,
) -> dict:
    """Thực thi suy luận (inference) và trả về các thang đo đánh giá kết hợp cùng xác suất nhận dạng phage độc lực (virulent) trên từng mẫu."""
    model.eval()
    all_preds, all_labels, all_scores = [], [], []
    use_amp = device.type == "cuda"

    for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"]

        with torch.amp.autocast("cuda", enabled=use_amp):
            if gated:
                input_ids  = batch["input_ids"].to(device, non_blocking=True)
                activation = batch["activation"].to(device, non_blocking=True)
                gene_stats = batch["gene_stats"].to(device, non_blocking=True)
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    activation_vector=activation,
                    gene_stats=gene_stats,
                )
            else:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

        probs = torch.softmax(logits.float(), dim=-1)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_scores.extend(probs[:, 1].cpu().tolist())   # Xác suất P(virulent)

    return compute_metrics(all_labels, all_preds, y_score=all_scores)


# ================================================================
# Luồng thực thi chính (Main execution)
# ================================================================

def main():
    args = parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    mode_suffix = get_mode_suffix(args)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Evaluating Group {args.group} | mode={mode_suffix} | split={args.eval_split}")
    print(f"Device: {device}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    all_fold_metrics = []

    for fold_idx in range(args.n_folds):
        print(f"\n--- Fold {fold_idx} ---")

        # ---- Tải trạng thái mô hình (Checkpoint) ----
        ckpt_path = get_checkpoint_path(args, fold_idx)
        if not ckpt_path.exists():
            print(f"  WARNING: checkpoint not found: {ckpt_path}  — skipping")
            continue

        # ---- Thiết lập Dữ liệu (Data) ----
        fold_dir = Path(args.data_dir) / f"group_{args.group}" / f"fold_{fold_idx}"
        pkl_path = fold_dir / f"{args.eval_split}.pkl"
        if not pkl_path.exists():
            print(f"  WARNING: data not found: {pkl_path}  — skipping")
            continue

        with open(pkl_path, "rb") as f:
            eval_data = pickle.load(f)

        sequences = eval_data["sequences"]
        labels    = eval_data["labels"]
        print(f"  Samples: {len(sequences)}")

        # ---- Truy xuất Đặc trưng (chỉ áp dụng đối với mô hình Gated) ----
        acts = stats = None
        if args.gated:
            feat_path = fold_dir / f"{args.eval_split}_features.pt"
            if not feat_path.exists():
                print(f"  WARNING: features not found: {feat_path}  — skipping fold")
                continue
            acts, stats = load_features(str(feat_path), normalize=True)

        # ---- Cấu trúc Bộ nạp dữ liệu (Dataset / DataLoader) ----
        dataset = PhageContigDataset(
            sequences, labels, tokenizer, args.max_seq_length,
            activations=acts, gene_stats=stats,
        )
        _extra = dict(
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=(2 if args.num_workers > 0 else None),
        )
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, **_extra,
        )

        # ---- Khởi tạo và khôi phục Mô hình (Model Initialization) ----
        if args.gated:
            model = PhaBERTCNN_GeneGated(
                dnabert2_model_name=args.model_name,
                n_families=args.n_families,
                use_gate=not args.no_gate,
                use_gene_stats=not args.no_gene_stats,
                use_pathway_scores=not args.no_pathway_scores,
            )
        else:
            model = PhaBERTCNN(dnabert2_model_name=args.model_name)

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if missing:
            print(f"  WARNING: missing keys in checkpoint: {missing[:5]}")
        model = model.to(device)
        model.eval()

        # ---- Tiến hành Đánh giá (Evaluate) ----
        metrics = evaluate_fold(model, dataloader, device, gated=args.gated)
        all_fold_metrics.append(metrics)
        print_metrics(metrics, prefix="  ")

        # Giải phóng bộ nhớ GPU sau mỗi nếp gấp (fold) để ngăn ngừa tràn VRAM
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ================================================================
    # Tổng hợp các chỉ số đánh giá (Aggregate Metrics)
    # ================================================================
    if not all_fold_metrics:
        print("\nNo folds evaluated — check checkpoint paths.")
        return

    print("\n" + "=" * 60)
    print(f"Aggregated Results — Group {args.group}  "
          f"[{len(all_fold_metrics)}/{args.n_folds} folds]")
    print("=" * 60)

    agg = aggregate_fold_metrics(all_fold_metrics)
    print(f"  Sensitivity: {agg['sensitivity_mean']:.2f}% ± {agg['sensitivity_std']:.2f}%")
    print(f"  Specificity: {agg['specificity_mean']:.2f}% ± {agg['specificity_std']:.2f}%")
    print(f"  Accuracy:    {agg['accuracy_mean']:.2f}% ± {agg['accuracy_std']:.2f}%")
    if 'auc_mean' in agg:
        print(f"  AUC-ROC:     {agg['auc_mean']:.2f}% ± {agg['auc_std']:.2f}%")

    # ---- Lưu lại cấu hình và hiệu suất (Save results) ----
    results = {
        "group":        args.group,
        "mode":         mode_suffix,
        "eval_split":   args.eval_split,
        "n_folds_run":  len(all_fold_metrics),
        "fold_metrics": all_fold_metrics,
        "aggregated":   agg,
    }
    result_path = Path(args.output_dir) / f"group_{args.group}_{mode_suffix}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved → {result_path}")

    # ================================================================
    # So sánh với tham chiếu (Paper reference comparison)
    # ================================================================
    paper_results = {
        'A': {'sn': 82.00, 'sp': 80.15, 'acc': 81.59},
        'B': {'sn': 89.91, 'sp': 80.44, 'acc': 87.91},
        'C': {'sn': 91.12, 'sp': 85.93, 'acc': 90.01},
        'D': {'sn': 88.47, 'sp': 90.95, 'acc': 90.69},
    }
    if args.group in paper_results:
        ref = paper_results[args.group]
        print(f"\n  Paper baseline (Group {args.group}): "
              f"sn={ref['sn']:.2f}%  sp={ref['sp']:.2f}%  acc={ref['acc']:.2f}%")
        print(f"  Δ accuracy: {agg['accuracy_mean'] - ref['acc']:+.2f}%")


if __name__ == "__main__":
    main()
