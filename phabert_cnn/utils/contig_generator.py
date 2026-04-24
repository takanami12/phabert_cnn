"""
utils/contig_generator.py
=========================
Tạo contig từ bộ gen thực khuẩn thể hoàn chỉnh bằng phương pháp sliding window
với tùy chọn tổng hợp gene/protein features đồng thời.

Thay đổi so với v1:
  - generate_contigs_from_genome trả về (contig_seq, window_start, window_end)
  - generate_dataset_contigs nhận ContigFeatureAggregator và trả về
    (sequences, labels, activations, gene_stats) nếu được cung cấp
  - Genome IDs được truyền xuyên suốt pipeline
"""

import random
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from Bio.Seq import Seq


# ============================================================
# Cấu hình nhóm contig (giữ nguyên theo bài báo gốc)
# ============================================================

CONTIG_GROUP_CONFIGS = {
    'A': {'min_length': 100,  'max_length': 400,  'overlap_pct': 0.1},
    'B': {'min_length': 400,  'max_length': 800,  'overlap_pct': 0.2},
    'C': {'min_length': 800,  'max_length': 1200, 'overlap_pct': 0.3},
    'D': {'min_length': 1200, 'max_length': 1800, 'overlap_pct': 0.4},
}


# ============================================================
# Tiện ích xử lý chuỗi
# ============================================================

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(seq: str) -> str:
    """Tính reverse complement nhanh bằng str.translate."""
    return seq.translate(_COMPLEMENT)[::-1]


# ============================================================
# Sinh contig
# ============================================================

def generate_contigs_from_genome(
    genome_seq: str,
    min_length: int,
    max_length: int,
    overlap_pct: float,
    seed: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    """
    Sinh contig từ một bộ gen bằng sliding window.

    Args:
        genome_seq:  chuỗi bộ gen hoàn chỉnh (chữ hoa)
        min_length:  độ dài contig tối thiểu (bp)
        max_length:  độ dài contig tối đa (bp)
        overlap_pct: tỷ lệ chồng lấp giữa các cửa sổ liên tiếp (0.0-1.0)
        seed:        seed RNG để tái tạo kết quả

    Returns:
        Danh sách các tuple (contig_seq, start, end) với start/end là
        vị trí 0-based trên strand thuận của bộ gen gốc.
    """
    rng = np.random.RandomState(seed)
    genome_len = len(genome_seq)
    contigs = []

    # Phân phối chuẩn cho độ dài contig
    mean_length = (min_length + max_length) / 2
    std_length = (max_length - min_length) / 6  # ~99.7% trong khoảng cho phép

    pos = 0
    while pos < genome_len:
        # Lấy mẫu độ dài
        contig_len = int(rng.normal(mean_length, std_length))
        contig_len = max(min_length, min(max_length, contig_len))

        end_pos = min(pos + contig_len, genome_len)
        contig = genome_seq[pos:end_pos]

        if len(contig) >= min_length:
            contigs.append((contig.upper(), pos, end_pos))

        # Tiến vị trí với overlap
        step = max(int(contig_len * (1 - overlap_pct)), 1)
        pos += step

    return contigs


def generate_dataset_contigs(
    genomes: List[Tuple[str, str, int]],
    group_config: Dict,
    aggregator=None,
    use_reverse_complement: bool = True,
    seed: int = 42,
    max_contigs_per_genome: Optional[int] = 50,
    overlap_min: float = 0.5,
) -> Union[
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int], List[np.ndarray], List[np.ndarray]],
]:
    """
    Sinh contig cho toàn bộ dataset, tùy chọn tổng hợp gene/protein features.

    Args:
        genomes:                danh sách tuple (genome_id, sequence, label)
        group_config:           dict với min_length, max_length, overlap_pct
        aggregator:             ContigFeatureAggregator tùy chọn. Nếu cung cấp,
                                features được tính per-contig và trả về cùng sequences.
        use_reverse_complement: nếu True, nhân đôi dataset với contig reverse complement
        seed:                   seed RNG gốc (seed per-genome được suy ra từ đây)
        max_contigs_per_genome: giới hạn số contig trên mỗi genome
        overlap_min:            tỷ lệ overlap tối thiểu để một gene được tính là có mặt

    Returns:
        Nếu aggregator là None:  (sequences, labels)
        Nếu có aggregator:       (sequences, labels, activations, gene_stats)

        Features cho contig reverse complement GIỐNG HỆT contig thuận
        (cùng vùng genomic, nhìn từ strand ngược lại).
    """
    rng = np.random.RandomState(seed)
    all_sequences: List[str] = []
    all_labels: List[int] = []
    all_activations: List[np.ndarray] = []
    all_gene_stats: List[np.ndarray] = []
    all_codon: List[np.ndarray] = []

    use_features = aggregator is not None
    codon_dim = getattr(aggregator, "codon_feature_dim", None) if use_features else None
    has_codon = use_features and codon_dim is not None

    for i, (genome_id, genome_seq, label) in enumerate(genomes):
        # Sinh cửa sổ cho bộ gen này: danh sách (seq, start, end)
        windows = generate_contigs_from_genome(
            genome_seq=genome_seq,
            min_length=group_config['min_length'],
            max_length=group_config['max_length'],
            overlap_pct=group_config['overlap_pct'],
            seed=seed + i,
        )

        # Lấy mẫu con nếu quá nhiều cửa sổ
        if max_contigs_per_genome and len(windows) > max_contigs_per_genome:
            indices = rng.choice(
                len(windows), max_contigs_per_genome, replace=False
            )
            windows = [windows[j] for j in indices]

        # Codon vector per-genome (share cho mọi contig của cùng genome)
        if has_codon:
            codon_vec = aggregator.get_codon(genome_id)
            if codon_vec is None:
                codon_vec = np.zeros(codon_dim, dtype=np.float32)

        # Phát contig thuận (+ RC) cho mỗi cửa sổ
        for contig_seq, w_start, w_end in windows:
            if use_features:
                activation, gene_stats = aggregator.get_features(
                    genome_id, w_start, w_end, overlap_min=overlap_min,
                )

            # Contig thuận
            all_sequences.append(contig_seq)
            all_labels.append(label)
            if use_features:
                all_activations.append(activation)
                all_gene_stats.append(gene_stats)
                if has_codon:
                    all_codon.append(codon_vec)

            # Reverse complement — features giống hệt (cùng vùng genomic)
            if use_reverse_complement:
                all_sequences.append(reverse_complement(contig_seq))
                all_labels.append(label)
                if use_features:
                    all_activations.append(activation)
                    all_gene_stats.append(gene_stats)
                    if has_codon:
                        all_codon.append(codon_vec)

    if use_features:
        if has_codon:
            return all_sequences, all_labels, all_activations, all_gene_stats, all_codon
        return all_sequences, all_labels, all_activations, all_gene_stats
    return all_sequences, all_labels
