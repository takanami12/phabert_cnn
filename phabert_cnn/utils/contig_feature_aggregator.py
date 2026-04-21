"""
contig_feature_aggregator.py
=============================
Tổng hợp features theo luồng — cắt contig và tính features đồng thời,
không cần provenance matching hay Aho-Corasick automaton.

Thay thế toàn bộ workflow map_genome_features_to_contigs.py bằng cách
tích hợp vào prepare_data.py (quá trình sinh contig).

Workflow:
    1. Tải annotation gene một lần → khởi tạo aggregator
    2. Trong sliding window loop, với mỗi (genome_id, start, end):
         activation, gene_stats = aggregator.get_features(genome_id, start, end)
    3. Lưu (sequences, labels, activations, gene_stats) cùng vào pkl

Bộ nhớ: O(1 genome + ~80 gene tại một thời điểm) thay vì O(all contigs + automaton).

Sử dụng:
    from contig_feature_aggregator import ContigFeatureAggregator

    # Khởi tạo một lần, trước khi cắt contig
    aggregator = ContigFeatureAggregator(
        annotation_paths=[
            "data/annotations/raw/Dataset-1_temperate_features.pt",
            "data/annotations/raw/Dataset-1_virulent_features.pt",
            # ... hoặc dùng from_directory() để tự động phát hiện
        ],
        vocab_path="data/hmm/vocabulary.json",
    )

    # Hoặc gọn hơn:
    aggregator = ContigFeatureAggregator.from_directory(
        annot_dir="data/annotations/raw/",
        vocab_path="data/hmm/vocabulary.json",
    )

    # Bên trong vòng lặp sinh contig (trong prepare_data.py):
    for genome_id, genome_seq in genomes:
        for w_start, w_end in sliding_windows(genome_seq):
            contig_seq = genome_seq[w_start:w_end]
            activation, gene_stats = aggregator.get_features(
                genome_id, w_start, w_end, overlap_min=0.5
            )

            # Contig thuận
            contigs["sequences"].append(contig_seq)
            contigs["activations"].append(activation)
            contigs["gene_stats"].append(gene_stats)

            # Reverse complement — features GIỐNG HỆT
            # (cùng vùng genomic nhìn từ strand ngược lại)
            contigs["sequences"].append(reverse_complement(contig_seq))
            contigs["activations"].append(activation)
            contigs["gene_stats"].append(gene_stats)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch


class ContigFeatureAggregator:
    """
    Aggregator in-memory để tính features cấp contig ngay trong quá trình sinh contig.

    Thiết kế: giữ annotation gene được lập chỉ mục theo genome_id, với gene
    được sắp xếp theo tọa độ start để truy vấn overlap nhanh.
    """

    def __init__(
        self,
        annotation_paths: List[Union[str, Path]],
        vocab_path: Union[str, Path],
    ):
        """
        Args:
            annotation_paths: danh sách file .pt từ preprocess_gene_features.py
                              ở FASTA mode (phải chứa 'gene_details')
            vocab_path:       đường dẫn tới vocabulary.json
        """
        self.genome_to_genes: Dict[str, List[dict]] = {}
        self.n_families: Optional[int] = None

        # Tải tất cả annotations
        for ap in annotation_paths:
            data = torch.load(str(ap), weights_only=False)
            if "gene_details" not in data:
                raise ValueError(
                    f"{ap} thiếu 'gene_details' — phải là đầu ra ở FASTA mode"
                )

            local_n_families = data["n_families"]
            if self.n_families is None:
                self.n_families = local_n_families
            elif self.n_families != local_n_families:
                raise ValueError(
                    f"n_families không nhất quán: {self.n_families} vs "
                    f"{local_n_families} trong {ap}"
                )

            for gid, genes in data["gene_details"].items():
                if gid in self.genome_to_genes:
                    continue  # lần xuất hiện đầu tiên được ưu tiên
                # Sắp xếp theo start để truy vấn nhanh hơn
                sorted_genes = sorted(genes, key=lambda g: g["start"])
                self.genome_to_genes[gid] = sorted_genes

        # Tải vocabulary
        with open(vocab_path) as f:
            vocab = json.load(f)
        self.family_name_to_idx: Dict[str, int] = vocab["name_to_idx"]

        # Thống kê
        n_genomes = len(self.genome_to_genes)
        n_total_genes = sum(len(g) for g in self.genome_to_genes.values())
        n_with_hits = sum(
            sum(1 for g in genes if g.get("hmm_hits"))
            for genes in self.genome_to_genes.values()
        )
        print(f"ContigFeatureAggregator đã khởi tạo:")
        print(f"  Bộ gen:  {n_genomes}")
        print(f"  Gene:    {n_total_genes} tổng, "
              f"{n_with_hits} có HMM hits "
              f"({100*n_with_hits/max(n_total_genes,1):.1f}%)")
        print(f"  Family:  {self.n_families}")

    @classmethod
    def from_directory(
        cls,
        annot_dir: Union[str, Path],
        vocab_path: Union[str, Path],
        pattern: str = "*_features.pt",
    ) -> "ContigFeatureAggregator":
        """Tự động phát hiện tất cả file annotation trong một thư mục."""
        annot_dir = Path(annot_dir)
        paths = sorted(annot_dir.glob(pattern))
        if not paths:
            raise ValueError(
                f"Không tìm thấy file nào khớp với '{pattern}' trong {annot_dir}"
            )
        print(f"Tìm thấy {len(paths)} file annotation trong {annot_dir}")
        return cls(annotation_paths=paths, vocab_path=vocab_path)

    def has_genome(self, genome_id: str) -> bool:
        return genome_id in self.genome_to_genes

    def get_features(
        self,
        genome_id: str,
        fwd_start: int,
        fwd_end: int,
        overlap_min: float = 0.5,
        coords_are_one_based: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tổng hợp features cho một cửa sổ contig.

        Args:
            genome_id:           ID bộ gen nguồn (khớp với header FASTA)
            fwd_start:           vị trí start 0-based trên strand thuận
            fwd_end:             vị trí end 0-based (exclusive)
            overlap_min:         tỷ lệ overlap tối thiểu của gene HOẶC cửa sổ
            coords_are_one_based: đặt True nếu fwd_start theo 1-based

        Returns:
            activation: np.ndarray[N_FAMILIES] — max HMM bit-score per family
            gene_stats: np.ndarray[4] — count, density(gene/kb),
                                        coding_fraction, strand_bias

        Lưu ý: features GIỐNG HỆT cho contig và reverse complement của nó
               (cùng vùng genomic nhìn từ strand ngược lại).
        """
        activation = np.zeros(self.n_families, dtype=np.float32)
        gene_stats = np.zeros(4, dtype=np.float32)

        genes = self.genome_to_genes.get(genome_id)
        if not genes:
            return activation, gene_stats

        # Căn chỉnh tọa độ với gene coords (Pyrodigal dùng 1-based inclusive)
        contig_start = fwd_start + (0 if coords_are_one_based else 1)
        contig_end = fwd_end
        contig_len = fwd_end - fwd_start
        if contig_len <= 0:
            return activation, gene_stats

        coding_bp = 0
        n_fwd = 0
        n_rev = 0
        n_overlapping = 0

        for gene in genes:
            g_start = gene["start"]
            g_end = gene["end"]

            # Bỏ qua sớm: gene được sắp xếp theo start, nên khi gene bắt đầu
            # sau contig, có thể dừng vòng lặp
            if g_start > contig_end:
                break

            # Tính giao (tọa độ 1-based inclusive)
            ovl_start = max(contig_start, g_start)
            ovl_end = min(contig_end, g_end)
            if ovl_start > ovl_end:
                continue

            ovl_len = ovl_end - ovl_start + 1
            gene_len = g_end - g_start + 1

            overlap_frac_gene = ovl_len / gene_len
            overlap_frac_contig = ovl_len / contig_len
            if overlap_frac_gene < overlap_min and overlap_frac_contig < overlap_min:
                continue

            n_overlapping += 1
            coding_bp += ovl_len

            if gene.get("strand", 1) == 1:
                n_fwd += 1
            else:
                n_rev += 1

            # Tổng hợp HMM hits (max-pool per family)
            for family_name, score in gene.get("hmm_hits", {}).items():
                fam_idx = self.family_name_to_idx.get(family_name)
                if fam_idx is not None:
                    activation[fam_idx] = max(activation[fam_idx], score)

        # Thống kê gene
        if n_overlapping > 0:
            density = n_overlapping / (contig_len / 1000)
            coding_frac = min(coding_bp / contig_len, 1.0)
            strand_bias = abs(n_fwd - n_rev) / n_overlapping
        else:
            density = 0.0
            coding_frac = 0.0
            strand_bias = 0.0

        gene_stats = np.array(
            [n_overlapping, density, coding_frac, strand_bias],
            dtype=np.float32,
        )
        return activation, gene_stats


INTEGRATION_EXAMPLE = """
# ============================================================
# Ví dụ tích hợp: chỉnh sửa prepare_data.py để lưu features
# ============================================================

from contig_feature_aggregator import ContigFeatureAggregator
import pickle
import torch

# --- BƯỚC 1: Khởi tạo aggregator MỘT LẦN ở đầu main() ---
aggregator = ContigFeatureAggregator.from_directory(
    annot_dir="data/annotations/raw/",
    vocab_path="data/hmm/vocabulary.json",
)


# --- BƯỚC 2: Sửa generate_dataset_contigs() ---
# Thêm aggregator và trả về cả features:

def generate_dataset_contigs_with_features(
    genomes, group_config, aggregator,
    use_reverse_complement=True, seed=0,
    max_contigs_per_genome=None,
):
    rng = np.random.RandomState(seed)
    contig_seqs = []
    contig_labels = []
    activations = []   # MỚI
    gene_stats = []    # MỚI

    for genome_id, (genome_seq, label) in genomes.items():
        windows = _sliding_windows(genome_seq, group_config, rng)
        if max_contigs_per_genome:
            windows = windows[:max_contigs_per_genome]

        for w_start, w_end in windows:
            contig_seq = genome_seq[w_start:w_end]

            # MỚI: tính features cho cửa sổ này
            act, stats = aggregator.get_features(
                genome_id, w_start, w_end, overlap_min=0.5,
            )

            # Contig thuận
            contig_seqs.append(contig_seq)
            contig_labels.append(label)
            activations.append(act)
            gene_stats.append(stats)

            # Reverse complement — features GIỐNG HỆT
            if use_reverse_complement:
                contig_seqs.append(reverse_complement(contig_seq))
                contig_labels.append(label)
                activations.append(act)
                gene_stats.append(stats)

    return contig_seqs, contig_labels, activations, gene_stats


# --- BƯỚC 3: Lưu features cùng với pkl ---

contig_seqs, contig_labels, acts, stats = generate_dataset_contigs_with_features(
    genomes=fold_data,
    group_config=group_config,
    aggregator=aggregator,
    ...
)

# Lưu pkl (định dạng không đổi, code huấn luyện cũ vẫn hoạt động)
with open(fold_dir / f"{split_name}.pkl", "wb") as f:
    pickle.dump({"sequences": contig_seqs, "labels": contig_labels}, f)

# MỚI: lưu features cạnh pkl
torch.save({
    "activations": torch.tensor(np.stack(acts)),
    "gene_stats":  torch.tensor(np.stack(stats)),
    "n_genes":     torch.tensor(
        [int(s[0]) for s in stats], dtype=torch.long
    ),
    "n_families":  aggregator.n_families,
}, fold_dir / f"{split_name}_features.pt")
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--annot_dir", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--test_genome", default=None,
                        help="Genome ID để kiểm tra tra cứu features")
    args = parser.parse_args()

    agg = ContigFeatureAggregator.from_directory(
        args.annot_dir, args.vocab
    )

    if args.test_genome:
        if not agg.has_genome(args.test_genome):
            print(f"Genome '{args.test_genome}' không tìm thấy")
            print(f"Có sẵn: {list(agg.genome_to_genes.keys())[:5]}...")
        else:
            # Mô phỏng một vài cửa sổ contig ngẫu nhiên
            genes = agg.genome_to_genes[args.test_genome]
            max_end = max(g["end"] for g in genes)
            print(f"\nTest windows trên {args.test_genome} (genome kết thúc tại {max_end}):")
            for s, e in [(0, 400), (1000, 1800), (max_end//2, max_end//2 + 1200)]:
                act, stats = agg.get_features(args.test_genome, s, e)
                print(f"  [{s}:{e}] n_genes={int(stats[0])}, "
                      f"density={stats[1]:.2f}/kb, "
                      f"activation_sum={act.sum():.2f}")

    print("\n" + "="*60)
    print("Ví dụ tích hợp:")
    print("="*60)
    print(INTEGRATION_EXAMPLE)
