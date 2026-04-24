import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_download import (
    download_deephage_data,
    download_deeppl_data,
    prepare_genome_dataset,
)
from utils.contig_generator import (
    generate_dataset_contigs,
    CONTIG_GROUP_CONFIGS,
)


# ============================================================
# Cấu hình Giao diện Dòng lệnh (CLI Parsing)
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Khởi tạo tập dữ liệu (dataset) PhaBERT-CNN v2 tích hợp đặc trưng mức độ gen (gene features)"
    )
    parser.add_argument("--data_dir", type=str, default="data/raw",
                        help="Thư mục cấu hình chứa tệp tin cấu trúc FASTA của bộ gen thô")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Thư mục đầu ra lưu trữ tệp .pkl sau tiền xử lý")
    parser.add_argument("--annot_dir", type=str, default="data/annotations/raw",
                        help="Thư mục chứa định dạng vector đặc trưng _features.pt cấp độ bộ gen")
    parser.add_argument("--vocab", type=str, default="data/hmm/vocabulary.json",
                        help="Đường dẫn đến tệp JSON chứa từ vựng (vocabulary) kiến trúc nhóm gen (gene family)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Số lượng nếp gấp (fold) phân vùng đánh giá chéo (cross-validation)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Tỉ lệ phân bổ tập huấn luyện/xác thực (train/val) nội bộ từng nếp gấp")
    parser.add_argument("--seed", type=int, default=42,
                        help="Hạt giống khởi tạo (Seed) cho quá trình sinh số ngẫu nhiên")
    parser.add_argument("--max_contigs", type=int, default=50,
                        help="Ngưỡng tải số lượng contig tối đa trên mỗi bộ gen (giảm thiểu bùng nổ dữ liệu cục bộ)")
    parser.add_argument("--overlap_min", type=float, default=0.5,
                        help="Tỷ lệ chồng chéo tối thiểu (overlap ratio) để một gen được ánh xạ vào contig")
    parser.add_argument("--skip_download", action="store_true",
                        help="Loại bỏ chu trình truy xuất và tải xuống dữ liệu")
    parser.add_argument("--no_features", action="store_true",
                        help="Vô hiệu hóa tiến trình trích xuất đặc trưng (chế độ khảo sát baseline thuần)")
    parser.add_argument("--groups", type=str, default="A,B,C,D",
                        help="Định danh nhóm khảo sát, phân lớp qua dấu phẩy")
    return parser.parse_args()


# ============================================================
# Hàm hỗ trợ
# ============================================================

def load_genomes_with_ids(data_dir: str):
    """
    Nạp dữ liệu bộ gen tổng hợp dưới dạng cấu trúc danh sách tuần tự (tuple): (genome_id, sequence, label).

    Gói đóng (Wrap) hàm prepare_genome_dataset() và tích hợp định danh bộ gen (genome ID)
    nhằm phục vụ quá trình truy xuất đối chiếu nhóm đặc trưng (features) thông qua aggregator.
    """
    genomes = prepare_genome_dataset(data_dir)

    # Tiêu chuẩn hoá dữ liệu đầu ra: chấp nhận khuôn dạng (seq, label) hoặc đầy đủ định danh (gid, seq, label)
    if genomes and len(genomes[0]) == 2:
        raise RuntimeError(
            "Cảnh báo: Hàm prepare_genome_dataset() trả về cấu trúc mảng (seq, label). "
            "Cần chỉnh sửa module utils/data_download.py để trả về đầy đủ tuple (gid, seq, label) "
            "nhằm đảm bảo liên kết dữ liệu định danh (genome ID) truyền tới khối feature aggregator. "
            "Tham khảo thêm tài liệu di trú (migration notes) ở cuối mã nguồn."
        )
    return genomes


def save_fold_split(
    fold_dir: Path,
    split_name: str,
    sequences,
    labels,
    activations=None,
    gene_stats=None,
    codon_features=None,
    n_families=None,
    codon_feature_dim=None,
):
    """Tiến hành lưu một phân vùng đánh giá (fold, split) — bao gồm định dạng dữ liệu tuần tự chuẩn pkl và dữ liệu nạp bổ sung .pt (nếu được kích hoạt cấu hình module)."""
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Định dạng tuần tự đối tượng chuẩn pkl (Cấu trúc tương thích xuôi để duy trì code huấn luyện kế thừa)
    pkl_path = fold_dir / f"{split_name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"sequences": sequences, "labels": labels}, f)

    # Nhóm đặc trưng tích hợp (Tính năng MỚI — Đồng bộ cấu trúc ánh xạ với thành phần chuỗi (sequences) phía trên)
    if activations is not None:
        pt_path = fold_dir / f"{split_name}_features.pt"
        acts = np.stack(activations) if activations else np.zeros((0, n_families or 24), dtype=np.float32)
        stats = np.stack(gene_stats) if gene_stats else np.zeros((0, 4), dtype=np.float32)
        n_genes = stats[:, 0].astype(np.int64) if stats.size else np.zeros(0, dtype=np.int64)

        payload = {
            "activations": torch.from_numpy(acts.astype(np.float32)),
            "gene_stats":  torch.from_numpy(stats.astype(np.float32)),
            "n_genes":     torch.from_numpy(n_genes),
            "n_families":  int(n_families),
        }
        if codon_features is not None:
            codon_arr = (
                np.stack(codon_features) if codon_features
                else np.zeros((0, codon_feature_dim or 65), dtype=np.float32)
            )
            payload["codon_features"] = torch.from_numpy(codon_arr.astype(np.float32))
            payload["codon_feature_dim"] = int(codon_feature_dim or codon_arr.shape[-1])
        torch.save(payload, pt_path)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Bước 1: Nạp hoặc khởi tạo truy xuất nguồn dữ liệu gen thô
    # ------------------------------------------------------------
    print("=" * 60)
    print("Bước 1: Quá trình phân giải và tải thể thực khuẩn (Phage) về bộ nhớ")
    print("=" * 60)

    if not args.skip_download:
        download_deephage_data(args.data_dir)
        download_deeppl_data(args.data_dir)

    genomes = load_genomes_with_ids(args.data_dir)

    if not genomes:
        print("\nKhông tải được bộ gen nào. Vui lòng đặt file FASTA vào:")
        print(f"  {args.data_dir}/")
        return

    n_vir = sum(1 for _, _, l in genomes if l == 1)
    n_temp = sum(1 for _, _, l in genomes if l == 0)
    print(f"Đã tải {len(genomes)} bộ gen "
          f"(độc lực={n_vir}, ôn hoà={n_temp})")

    # ------------------------------------------------------------
    # Bước 2: Tích hợp module tổng hợp đặc trưng (Feature Aggregator) - Tuỳ chọn
    # ------------------------------------------------------------
    aggregator = None
    n_families = 26  # Ngưỡng giới hạn kích thước từ vựng tiêu chuẩn (tối ưu hóa ở mức 26 module ontology)

    if not args.no_features:
        print("\n" + "=" * 60)
        print("Bước 2: Khởi tạo phân luồng ContigFeatureAggregator")
        print("=" * 60)

        if not Path(args.annot_dir).exists():
            print(f"CẢNH BÁO TỪ HỆ THỐNG: Đường dẫn {args.annot_dir} không gắn kết hợp lệ")
            print("  Vui lòng thực thi quá trình `preprocess_gene_features.py` trên bộ cấu trúc FASTA để sinh đặc trưng khởi đầu,")
            print("  hoặc cung cấp cờ bổ sung `--no_features` để vô hiệu hóa chuỗi phân tích cấu trúc đặc trưng.")
            return
        if not Path(args.vocab).exists():
            print(f"CẢNH BÁO TỪ HỆ THỐNG: Tệp từ vựng {args.vocab} bị thiếu")
            print("  Tiến hành khởi chạy `prepare_hmm_profiles.py` trước, hoặc chuyển sang trạng thái `--no_features`.")
            return

        from utils.contig_feature_aggregator import ContigFeatureAggregator
        aggregator = ContigFeatureAggregator.from_directory(
            annot_dir=args.annot_dir,
            vocab_path=args.vocab,
        )
        n_families = aggregator.n_families

        # Giai đoạn rà soát tính toàn vẹn: Đảm bảo nhóm định danh ID bộ gen tham chiếu được ánh xạ tương ứng kết quả annotations.
        missing = [gid for gid, _, _ in genomes if not aggregator.has_genome(gid)]
        if missing:
            print(f"\nCẢNH BÁO: {len(missing)}/{len(genomes)} mẫu bộ gen hệ thống báo thiếu "
                  f"nguồn thông tin định hướng dạng annotations. Dữ liệu tham khảo: {missing[:5]}")
            print("  Hệ thống xử lý bảo mật cho phép contig tại các bộ gen trên phản hồi giá trị zero-vector thay vì crash.")

    # ------------------------------------------------------------
    # Bước 3: Phân rã dữ liệu phân tầng (Stratified K-fold CV ở mức độ cấu trúc bộ gen)
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Bước 3: Thực thi cơ chế phân chia {args.n_folds}-Fold Stratified Cross-Validation")
    print("=" * 60)

    labels_arr = np.array([l for _, _, l in genomes])
    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.seed,
    )

    fold_splits = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(genomes)), labels_arr)
    ):
        # Tiếp tục chia train_val thành train / val (8:2)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            train_size=args.train_ratio,
            stratify=labels_arr[train_val_idx],
            random_state=args.seed + fold_idx,
        )

        fold_splits.append({
            "train": [genomes[i] for i in train_idx],
            "val":   [genomes[i] for i in val_idx],
            "test":  [genomes[i] for i in test_idx],
        })
        print(f"  Fold {fold_idx}: "
              f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ------------------------------------------------------------
    # Bước 4: Chuyển đổi và trích xuất contig tích hợp vector đặc trưng theo quy trình phân nhóm
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Bước 4: Sinh mẫu ngẫu nhiên chuỗi đặc trưng contig")
    print("=" * 60)

    groups_to_process = [g.strip() for g in args.groups.split(",") if g.strip()]

    for group_name in groups_to_process:
        if group_name not in CONTIG_GROUP_CONFIGS:
            print(f"  Bỏ qua nhóm không xác định: {group_name}")
            continue

        group_config = CONTIG_GROUP_CONFIGS[group_name]
        group_dir = Path(args.output_dir) / f"group_{group_name}"
        print(f"\n  Nhóm {group_name} "
              f"(độ dài {group_config['min_length']}-{group_config['max_length']}, "
              f"overlap {int(100*group_config['overlap_pct'])}%)")

        for fold_idx, fold_data in enumerate(fold_splits):
            print(f"    Fold {fold_idx}:")
            fold_dir = group_dir / f"fold_{fold_idx}"

            for split_name in ("train", "val", "test"):
                # RC augmentation CHỈ cho train — val/test phải đánh giá
                # trên sample độc lập, không paired fwd/rc.
                out = generate_dataset_contigs(
                    genomes=fold_data[split_name],
                    group_config=group_config,
                    aggregator=aggregator,
                    use_reverse_complement=(split_name == "train"),
                    seed=args.seed + fold_idx * 100 + ord(group_name),
                    max_contigs_per_genome=args.max_contigs,
                    overlap_min=args.overlap_min,
                )

                codon_features = None
                if aggregator is not None:
                    if len(out) == 5:
                        sequences, labels, activations, gene_stats, codon_features = out
                    else:
                        sequences, labels, activations, gene_stats = out
                else:
                    sequences, labels = out
                    activations, gene_stats = None, None

                save_fold_split(
                    fold_dir=fold_dir,
                    split_name=split_name,
                    sequences=sequences,
                    labels=labels,
                    activations=activations,
                    gene_stats=gene_stats,
                    codon_features=codon_features,
                    n_families=n_families,
                    codon_feature_dim=getattr(aggregator, "codon_feature_dim", None),
                )

                # Tổng hợp thống kê phản hồi màn hình stdout
                n = len(sequences)
                n_v = sum(1 for l in labels if l == 1)
                n_t = n - n_v
                summary = f"{split_name:5s}: {n:6d} contig (vir={n_v}, temp={n_t})"
                if activations:
                    acts = np.stack(activations)
                    n_hits = int((acts.sum(axis=1) > 0).sum())
                    summary += f" | có HMM hits: {n_hits} ({100*n_hits/n:.1f}%)"
                print(f"      {summary}")

    # ------------------------------------------------------------
    # Bước 5: Kết xuất Metadata thông số quy trình cấu trúc
    # ------------------------------------------------------------
    metadata = {
        "n_genomes": len(genomes),
        "n_virulent": int(n_vir),
        "n_temperate": int(n_temp),
        "n_folds": args.n_folds,
        "max_contigs_per_genome": args.max_contigs,
        "groups": groups_to_process,
        "seed": args.seed,
        "features_enabled": aggregator is not None,
        "n_families": n_families if aggregator else None,
        "overlap_min": args.overlap_min if aggregator else None,
    }
    with open(Path(args.output_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Hoàn tất chu trình phân phối dữ liệu phân tích học thuật!")
    print(f"  Thư mục đầu ra cấu trúc: {args.output_dir}")
    print(f"  Trạng thái Vector đặc trưng (Features): {'Khả dụng' if aggregator else 'Hủy kích hoạt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()


# ============================================================
# Quản lý sự phụ thuộc Di trú mã nguồn (Migration Policy Guidelines)
# ============================================================
# Cảnh báo nội bộ: Nếu phương thức `utils/data_download.py` thiết bị 
# tại phiên bản hiện tại trả về bộ tuple (seq, label) bị tiêu lược mất ID định danh
# thay cho cấu trúc (genome_id, seq, label), hãy tiến hành tái cấu trúc hàm `prepare_genome_dataset()`
# để duy trì mã ID trích lọc từ phần nhãn FASTA:
#
#     def prepare_genome_dataset(data_dir):
#         genomes = []
#         for fasta_path in Path(data_dir).glob("*.fasta"):
#             label = 1 if "virulent" in fasta_path.name else 0
#             for record in SeqIO.parse(fasta_path, "fasta"):
#                 genomes.append((record.id, str(record.seq).upper(), label))
#         return genomes
#
# Mã thông báo hệ gen (Genome ID) bắt buộc duy trì đối chiếu đồng bộ với thẻ tiêu đề (header tag)
# khởi tạo khi khởi chạy bộ quy trình gán nhãn lược đồ `preprocess_gene_features.py` (cơ chế FASTA mode).
