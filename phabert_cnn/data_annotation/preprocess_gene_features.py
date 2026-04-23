import json
import argparse
import warnings
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import torch


# ============================================================
# Codon usage constants (standard genetic code)
# ============================================================

CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}
ALL_CODONS = sorted(CODON_TABLE.keys())
CODON_IDX = {c: i for i, c in enumerate(ALL_CODONS)}
AA_TO_SYN: Dict[str, List[str]] = {}
for _c, _aa in CODON_TABLE.items():
    AA_TO_SYN.setdefault(_aa, []).append(_c)

CODON_FEATURE_DIM = 65  # 64 RSCU + 1 GC3


def compute_codon_features(cds_sequences: List[str]) -> np.ndarray:
    """
    Tính vector đặc trưng codon usage 65-d từ tập hợp các CDS DNA của một contig.

    Cấu phần:
      [0..63]  RSCU (Relative Synonymous Codon Usage) cho 64 codon (theo thứ tự ALL_CODONS).
               RSCU(c) = obs(c) / (sum(obs trong nhóm synonymous của a) / N_synonymous(a))
               Giá trị 1.0 = không thiên lệch; > 1 = ưu tiên; < 1 = né tránh.
      [64]     GC3 — tỷ lệ G+C ở vị trí codon thứ 3 (proxy cho codon adaptation).

    Khi không có codon hợp lệ → trả vector mặc định: RSCU = 1.0 (uniform), GC3 = 0.5.
    """
    feat = np.zeros(CODON_FEATURE_DIM, dtype=np.float32)
    feat[:64] = 1.0
    feat[64] = 0.5

    if not cds_sequences:
        return feat

    counter: Counter = Counter()
    third_gc = 0
    third_total = 0
    valid_chars = set("ACGT")

    for cds in cds_sequences:
        cds = cds.upper()
        # Chỉ duyệt các codon đầy đủ và toàn ACGT
        for i in range(0, len(cds) - 2, 3):
            codon = cds[i:i + 3]
            if len(codon) != 3:
                continue
            if not all(ch in valid_chars for ch in codon):
                continue
            counter[codon] += 1
            third_total += 1
            if codon[2] in ("G", "C"):
                third_gc += 1

    if sum(counter.values()) == 0:
        return feat

    # Tổng codon trong từng nhóm amino acid
    aa_totals: Dict[str, int] = {aa: 0 for aa in AA_TO_SYN}
    for codon, count in counter.items():
        aa = CODON_TABLE.get(codon)
        if aa is not None:
            aa_totals[aa] += count

    for codon, idx in CODON_IDX.items():
        aa = CODON_TABLE[codon]
        aa_total = aa_totals[aa]
        n_syn = len(AA_TO_SYN[aa])
        if aa_total > 0 and n_syn > 0:
            expected = aa_total / n_syn
            feat[idx] = counter[codon] / expected if expected > 0 else 1.0

    if third_total > 0:
        feat[64] = third_gc / third_total

    return feat

# ============================================================
# Imports with availability checks
# ============================================================

try:
    import pyrodigal
except ImportError:
    raise ImportError("pyrodigal required: pip install pyrodigal --break-system-packages")

try:
    import pyhmmer
    from pyhmmer.easel import TextSequence, Alphabet
    from pyhmmer.plan7 import HMMFile
except ImportError:
    raise ImportError("pyhmmer required: pip install pyhmmer --break-system-packages")

try:
    from Bio import SeqIO
    from Bio.Seq import Seq
except ImportError:
    raise ImportError("biopython required: pip install biopython --break-system-packages")


# ============================================================
# Data structures
# ============================================================

@dataclass
class GeneInfo:
    """Thông tin về một gen dự đoán được."""
    start: int
    end: int
    strand: int          # +1 hoặc -1
    partial: bool        # True nếu gen trải dài qua giới hạn của contig
    translation: str     # chuỗi axit amin
    cds_seq: str = ""    # chuỗi DNA mã hóa (đã reverse-complement nếu strand = -1)
    hmm_hits: Dict[str, float] = field(default_factory=dict)  # family_name → bit-score


@dataclass
class ContigAnnotation:
    """Toàn bộ chú thích cho một contig đơn lẻ."""
    contig_id: str
    contig_length: int
    genes: List[GeneInfo]

    # Các vector đặc trưng (được điền sau tiến trình quét HMM)
    activation_vector: Optional[np.ndarray] = None   # [N_FAMILIES]
    gene_stats: Optional[np.ndarray] = None          # [4]
    codon_features: Optional[np.ndarray] = None      # [65] (64 RSCU + 1 GC3)

    @property
    def gene_count(self) -> int:
        return len(self.genes)

    @property
    def gene_density(self) -> float:
        """Số gen trên một kilobase."""
        if self.contig_length == 0:
            return 0.0
        return self.gene_count / (self.contig_length / 1000)

    @property
    def coding_fraction(self) -> float:
        """Tỷ lệ contig bị bao phủ bởi các vùng mã hóa."""
        if self.contig_length == 0:
            return 0.0
        total_coding = sum(abs(g.end - g.start) for g in self.genes)
        return min(total_coding / self.contig_length, 1.0)

    @property
    def strand_bias(self) -> float:
        """Sự bất đối xứng của mạch (Strand asymmetry): |thuận - nghịch| / tổng. Khoảng [0, 1]."""
        if self.gene_count == 0:
            return 0.0
        fwd = sum(1 for g in self.genes if g.strand == 1)
        rev = self.gene_count - fwd
        return abs(fwd - rev) / self.gene_count

    def compute_gene_stats(self) -> np.ndarray:
        """Compute 4-dimensional gene statistics vector."""
        self.gene_stats = np.array([
            self.gene_count,
            self.gene_density,
            self.coding_fraction,
            self.strand_bias,
        ], dtype=np.float32)
        return self.gene_stats

    def compute_codon_features(self) -> np.ndarray:
        """Tổng hợp codon usage 65-d (64 RSCU + GC3) trên toàn bộ CDS của contig."""
        cds_list = [g.cds_seq for g in self.genes if g.cds_seq]
        self.codon_features = compute_codon_features(cds_list)
        return self.codon_features


# ============================================================
# Step 1: Gene Prediction with Pyrodigal
# ============================================================

class GenePrediction:
    """
    Dự đoán gen trong các contig bằng Pyrodigal (Prodigal bao bọc).

    Hai chế độ:
      - meta=True  (mặc định): mô hình hệ gen lai (metagenome) huấn luyện sẵn, 
                               thích hợp cho đoạn ngắn khi huấn luyện từng genome không khả thi.
      - complete_genome=True:  huấn luyện Prodigal trên mỗi genome độc lập trước khi dự đoán,
                               giúp độ chính xác cao hơn trên genome hoàn chỉnh.
    """

    # Độ dài chuỗi tối thiểu cần thiết để tiến hành quá trình huấn luyện ở mức đơn hệ gen 
    MIN_TRAIN_LEN = 20_000

    def __init__(self, min_gene_len: int = 60, closed: bool = False,
                 complete_genome: bool = False):
        """
        Args:
            min_gene_len:     Độ dài gen tối thiểu theo nucleotide (mặc định 60 = 20 aa)
            closed:           Nếu True, không cho phép gen vượt qua các biên của contig
            complete_genome:  Nếu True, tiến hành huấn luyện Prodigal trên mỗi genome trước khi quyết định
                              (đề nghị cho trường hợp genome phage hoàn chỉnh ≥ 20 kb)
        """
        self.min_gene_len = min_gene_len
        self.closed = closed
        self.complete_genome = complete_genome

        if complete_genome:
            # Single-genome mode: gene_finder is created fresh per genome in predict()
            self._meta_finder = pyrodigal.GeneFinder(meta=True, closed=closed)
        else:
            self.gene_finder = pyrodigal.GeneFinder(meta=True, closed=closed)

    def predict(self, contig_id: str, sequence: str) -> ContigAnnotation:
        """
        Dự đoán các gen trên một contig duy nhất hoặc trên toàn bộ hệ gen.

        Trong chế độ complete_genome, Prodigal được huấn luyện trên chuỗi chính nó 
        (chế độ single-genome). Sẽ lùi về chế độ meta nếu chuỗi đó quá
        ngắn và không đủ cho một đợt đào tạo ổn định (< 20 kb).

        Args:
            contig_id: mã định danh duy nhất của contig / bộ gen
            sequence:  chuỗi DNA (ACGT)

        Returns:
            ContigAnnotation chứa những gen đã dự đoán
        """
        genes = []

        try:
            if self.complete_genome and len(sequence) >= self.MIN_TRAIN_LEN:
                # Train on this genome, then predict.
                # Suppress pyrodigal's "< 100000 chars" warning — phage genomes
                # are inherently short; the warning targets bacterial use cases.
                gene_finder = pyrodigal.GeneFinder(meta=False, closed=self.closed)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="sequence should be at least")
                    gene_finder.train(sequence.encode())
                predicted = gene_finder.find_genes(sequence.encode())
            elif self.complete_genome:
                # Sequence too short for training → fall back to meta mode
                warnings.warn(
                    f"{contig_id}: sequence too short ({len(sequence)} bp) for "
                    f"single-genome training, using meta mode"
                )
                predicted = self._meta_finder.find_genes(sequence.encode())
            else:
                predicted = self.gene_finder.find_genes(sequence.encode())
        except Exception as e:
            warnings.warn(f"Pyrodigal failed on {contig_id} ({len(sequence)}bp): {e}")
            return ContigAnnotation(
                contig_id=contig_id,
                contig_length=len(sequence),
                genes=[],
            )

        for gene in predicted:
            gene_len = abs(gene.end - gene.begin)
            if gene_len < self.min_gene_len:
                continue

            # Translate to amino acid sequence
            try:
                translation = gene.translate()
                # Remove trailing stop codon '*'
                if translation and translation[-1] == "*":
                    translation = translation[:-1]
            except Exception:
                # Fallback: manual translation
                try:
                    dna_seq = sequence[gene.begin - 1 : gene.end]
                    if gene.strand == -1:
                        dna_seq = str(Seq(dna_seq).reverse_complement())
                    translation = str(Seq(dna_seq).translate())[:-1]
                except Exception:
                    translation = ""

            if len(translation) < 20:  # skip very short proteins
                continue

            # Trích xuất CDS DNA (đã reverse-complement nếu strand = -1) cho codon
            try:
                dna_cds = sequence[gene.begin - 1 : gene.end]
                if gene.strand == -1:
                    dna_cds = str(Seq(dna_cds).reverse_complement())
            except Exception:
                dna_cds = ""

            genes.append(GeneInfo(
                start=gene.begin,
                end=gene.end,
                strand=1 if gene.strand == 1 else -1,
                partial=gene.partial_begin or gene.partial_end,
                translation=translation,
                cds_seq=dna_cds,
            ))

        return ContigAnnotation(
            contig_id=contig_id,
            contig_length=len(sequence),
            genes=genes,
        )


# ============================================================
# Step 2: HMM Scanning with pyhmmer
# ============================================================

class HMMScanner:
    """
    Quét chuỗi protein dự báo và so sánh với cấu hình HMM của các họ gen.

    Dùng pyhmmer để tăng tốc độ xử lý HMM ngay trong tiến trình (không cần file
    thực thi hmmscan từ bên ngoài).
    """

    def __init__(self, hmm_path: str, vocab_path: str,
                 e_value_threshold: float = 1e-5,
                 use_bitscore: bool = True):
        """
        Args:
            hmm_path:           Đường dẫn của tệp chứa tệp HMM nối
            vocab_path:         đường dẫn của vùng lưu vocabulary.json
            e_value_threshold:  giá trị e-value lớn nhất để chấp nhận một hit
            use_bitscore:       nếu True, sử dụng giá trị bit-score trong vector kích hoạt (activation vector)
                                nếu False, kích hoạt nhị phân thay thế (0/1)
        """
        self.e_value_threshold = e_value_threshold
        self.use_bitscore = use_bitscore

        # Load vocabulary
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        self.n_families = self.vocab["n_families"]
        self.name_to_idx = self.vocab["name_to_idx"]
        self.families = self.vocab["families"]

        # Load HMM profiles
        print(f"Loading HMM profiles from {hmm_path}...")
        self.alphabet = Alphabet.amino()
        self.hmms = []
        self.hmm_name_to_family = {}

        # Try to load a pre-built mapping file (e.g. from VOG strategy)
        mapping_path = Path(hmm_path).with_suffix(".hmm2family.json")
        prebuilt_mapping = {}
        if mapping_path.exists():
            with open(mapping_path) as f:
                prebuilt_mapping = json.load(f)
            print(f"  Loaded pre-built HMM→family mapping ({len(prebuilt_mapping)} entries)")

        with HMMFile(hmm_path) as hmm_file:
            for hmm in hmm_file:
                self.hmms.append(hmm)
                hmm_name = hmm.name.decode() if isinstance(hmm.name, bytes) else hmm.name

                # Use pre-built mapping first (covers VOG-named HMMs)
                if hmm_name in prebuilt_mapping:
                    self.hmm_name_to_family[hmm_name] = prebuilt_mapping[hmm_name]
                    continue

                hmm_accession = ""
                if hmm.accession:
                    hmm_accession = hmm.accession.decode() if isinstance(hmm.accession, bytes) else hmm.accession

                # Match by Pfam accession or by name/keywords
                for family in self.families:
                    pfam_acc = family.get("pfam", "")
                    if (pfam_acc and pfam_acc in hmm_accession) or \
                       (pfam_acc and pfam_acc in hmm_name):
                        self.hmm_name_to_family[hmm_name] = family["idx"]
                        break
                    # Keyword matching fallback
                    for kw in family.get("keywords", []):
                        hmm_desc = ""
                        if hmm.description:
                            hmm_desc = hmm.description.decode() if isinstance(hmm.description, bytes) else hmm.description
                        if kw.lower() in hmm_name.lower() or kw.lower() in hmm_desc.lower():
                            self.hmm_name_to_family[hmm_name] = family["idx"]
                            break

        print(f"  Loaded {len(self.hmms)} HMMs, "
              f"{len(self.hmm_name_to_family)} mapped to gene families")

    def scan_contig(self, annotation: ContigAnnotation) -> np.ndarray:
        """
        Quét mọi protein được phát hiện từ một contig chéo ngang các cấu hình HMM.

        Returns:
            activation_vector: np.ndarray với kích thước [N_FAMILIES]
                - Nếu use_bitscore=True: bit-score tối ưu trên mỗi family
                - Nếu use_bitscore=False: nhị phân (1 nếu có trúng, thay vì vậy thì 0)
        """
        activation = np.zeros(self.n_families, dtype=np.float32)

        if not annotation.genes or not self.hmms:
            annotation.activation_vector = activation
            return activation

        # Build digital sequence block for pyhmmer
        sequences = []
        for i, gene in enumerate(annotation.genes):
            if not gene.translation or len(gene.translation) < 20:
                continue
            try:
                seq = TextSequence(
                    name=f"gene_{i}".encode(),
                    sequence=gene.translation,
                )
                ds = seq.digitize(self.alphabet)
                sequences.append((i, ds))
            except Exception:
                continue

        if not sequences:
            annotation.activation_vector = activation
            return activation

        # Run hmmsearch: each HMM against all protein sequences
        try:
            for hits in pyhmmer.hmmsearch(
                self.hmms,
                [ds for _, ds in sequences],
                bit_cutoffs="trusted" if False else None,
                E=self.e_value_threshold * 10,  # pre-filter, tighter filter below
            ):
                hmm_name = hits.query.name.decode() if isinstance(hits.query.name, bytes) else hits.query.name

                if hmm_name not in self.hmm_name_to_family:
                    continue

                family_idx = self.hmm_name_to_family[hmm_name]

                for hit in hits:
                    if hit.evalue > self.e_value_threshold:
                        continue

                    score = hit.score
                    hit_name = hit.name.decode() if isinstance(hit.name, bytes) else hit.name
                    gene_idx = int(hit_name.split("_")[1])

                    # Record HMM hit on the gene
                    family_name = self.vocab["idx_to_name"][str(family_idx)]
                    annotation.genes[gene_idx].hmm_hits[family_name] = float(score)

                    # Update activation vector (keep max score per family)
                    if self.use_bitscore:
                        activation[family_idx] = max(activation[family_idx], score)
                    else:
                        activation[family_idx] = 1.0

        except Exception as e:
            warnings.warn(f"HMM scan failed for {annotation.contig_id}: {e}")

        # ── Xác thực CI repressor dùng Dual-Pfam ─────────────────────────
        # Repressor CI CHỈ được xác nhận lại khi CÙNG MỘT protein thoả mãn cả 2 điều điện:
        #   • idx 3  (PF07022) — miền bắt DNA HTH ở đầu N
        #   • idx 24 (PF00717) — miền tự phân giải protein Peptidase_S24 ở đầu C
        # Nếu chỉ có PF07022 thì chưa đủ: vì khá nhiều các protein của phage chứa 
        # cấu trúc hình xoắn HTH chung (giống như Cro) và sẽ cho ra dương tính CI giả. 
        # Sự cắt tỉa lại đoạn protein ở đầu phân cắt C-Terminal của RecA-cleavage
        # (Peptidase_S24 / nhóm LexA) chính là tính chất đặc hiệu riêng chỉ có mặt ở dòng
        # CI-class, tạo nên sự nghiêm ngặt trên.
        #
        # Nếu idx 24 chưa nằm trong các danh sách từ vựng (do bản cập nhật cũ - 24 family),
        # khối lượng bước này sẽ được thông quan tự động do idx 3 cũng hoạt động lại như ban đầu.
        ci_name      = "CI_repressor"
        ci_ctail_name = "CI_repressor_Ctail"
        ci_idx       = self.name_to_idx.get(ci_name)
        ci_ctail_idx = self.name_to_idx.get(ci_ctail_name)

        if ci_idx is not None and ci_ctail_idx is not None:
            confirmed_ci = any(
                ci_name in g.hmm_hits and ci_ctail_name in g.hmm_hits
                for g in annotation.genes
            )
            if not confirmed_ci:
                # Downgrade: HTH-only hit is ambiguous — clear CI activation
                activation[ci_idx] = 0.0
                activation[ci_ctail_idx] = 0.0
                for gene in annotation.genes:
                    gene.hmm_hits.pop(ci_name, None)
                    gene.hmm_hits.pop(ci_ctail_name, None)
        # ──────────────────────────────────────────────────────────────────

        annotation.activation_vector = activation
        return activation


# ============================================================
# Step 3: Batch Processing
# ============================================================

def process_fasta(
    fasta_path: str,
    gene_predictor: GenePrediction,
    hmm_scanner: Optional[HMMScanner],
    n_families: int,
) -> Dict[str, dict]:
    """
    Xử lý tất cả các contig trong tệp tin FASTA.

    Returns:
        dict thiết lập bản đồ contig_id → {
            "activation": np.ndarray[N],
            "gene_stats": np.ndarray[4],
            "n_genes": int,
            "gene_details": list của {start, end, strand, hmm_hits}
        }
    """
    results = {}
    contigs = list(SeqIO.parse(fasta_path, "fasta"))
    n_total = len(contigs)
    n_with_genes = 0
    n_with_hits = 0

    t_start = time.time()
    print(f"Processing {n_total} contigs from {fasta_path}...")

    for i, record in enumerate(contigs):
        contig_id = record.id
        sequence = str(record.seq).upper()

        # Skip very short sequences
        if len(sequence) < 30:
            zero_codon = np.zeros(CODON_FEATURE_DIM, dtype=np.float32)
            zero_codon[:64] = 1.0
            zero_codon[64] = 0.5
            results[contig_id] = {
                "activation": np.zeros(n_families, dtype=np.float32),
                "gene_stats": np.zeros(4, dtype=np.float32),
                "codon_features": zero_codon,
                "n_genes": 0,
                "gene_details": [],
            }
            continue

        # Step 1: Gene prediction
        annotation = gene_predictor.predict(contig_id, sequence)

        if annotation.gene_count > 0:
            n_with_genes += 1

        # Step 2: HMM scan (if scanner available)
        if hmm_scanner and annotation.gene_count > 0:
            hmm_scanner.scan_contig(annotation)
            if annotation.activation_vector is not None and annotation.activation_vector.sum() > 0:
                n_with_hits += 1
        else:
            annotation.activation_vector = np.zeros(n_families, dtype=np.float32)

        # Step 3: Compute gene statistics + codon usage
        annotation.compute_gene_stats()
        annotation.compute_codon_features()

        # Package results
        gene_details = []
        for g in annotation.genes:
            gene_details.append({
                "start": g.start,
                "end": g.end,
                "strand": g.strand,
                "partial": g.partial,
                "translation_len": len(g.translation),
                "hmm_hits": g.hmm_hits,
            })

        results[contig_id] = {
            "activation": annotation.activation_vector,
            "gene_stats": annotation.gene_stats,
            "codon_features": annotation.codon_features,
            "n_genes": annotation.gene_count,
            "gene_details": gene_details,
        }

        # Progress
        if (i + 1) % 500 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{n_total}] {rate:.1f} contigs/sec | "
                  f"genes found: {n_with_genes} | HMM hits: {n_with_hits}")

    elapsed = time.time() - t_start
    print(f"  Finished in {elapsed:.1f}s | "
          f"{n_with_genes}/{n_total} contigs have genes "
          f"({100*n_with_genes/max(n_total,1):.1f}%) | "
          f"{n_with_hits} have HMM hits")

    return results


def process_pkl(
    pkl_path: str,
    gene_predictor: GenePrediction,
    hmm_scanner: Optional[HMMScanner],
    n_families: int,
) -> List[dict]:
    """
    Xử lý danh sách contig thu nhận qua đối tượng thu thập tin học (pkl file) từ mô hình PhaBERT-CNN.

    Cấu trúc của pkl sẽ bao gồm: { 'sequences': list[str], 'labels': list[int] }
    Giá trị kết xuất sẽ có định dạng (danh sách - không phải thư viện từ điển) 
    và giống với trình tự ở phần dữ liệu mở rộng cho quá trình test/tiền quá trình huấn luyện:
    khi đó features[i] ánh xạ song song sequences[i].

    Returns:
        danh sách đại diện cho { "activation": np.ndarray[N],
                  "gene_stats": np.ndarray[4],
                  "n_genes": int }
    """
    import pickle as pkl

    with open(pkl_path, "rb") as f:
        data = pkl.load(f)

    sequences = data["sequences"]
    n_total = len(sequences)
    n_with_genes = 0
    n_with_hits = 0

    results = []

    t_start = time.time()
    print(f"Processing {n_total} contigs from {pkl_path}...")

    for i, sequence in enumerate(sequences):
        sequence = sequence.upper()

        # Very short sequences → zero features
        if len(sequence) < 30:
            zero_codon = np.zeros(CODON_FEATURE_DIM, dtype=np.float32)
            zero_codon[:64] = 1.0
            zero_codon[64] = 0.5
            results.append({
                "activation": np.zeros(n_families, dtype=np.float32),
                "gene_stats": np.zeros(4, dtype=np.float32),
                "codon_features": zero_codon,
                "n_genes": 0,
            })
            continue

        # Step 1: Gene prediction
        contig_id = f"seq_{i}"
        annotation = gene_predictor.predict(contig_id, sequence)

        if annotation.gene_count > 0:
            n_with_genes += 1

        # Step 2: HMM scan
        if hmm_scanner and annotation.gene_count > 0:
            hmm_scanner.scan_contig(annotation)
            if annotation.activation_vector is not None and annotation.activation_vector.sum() > 0:
                n_with_hits += 1
        else:
            annotation.activation_vector = np.zeros(n_families, dtype=np.float32)

        # Step 3: Gene statistics + codon usage
        annotation.compute_gene_stats()
        annotation.compute_codon_features()

        results.append({
            "activation": annotation.activation_vector,
            "gene_stats": annotation.gene_stats,
            "codon_features": annotation.codon_features,
            "n_genes": annotation.gene_count,
        })

        # Progress
        if (i + 1) % 500 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{n_total}] {rate:.1f} contigs/sec | "
                  f"genes found: {n_with_genes} | HMM hits: {n_with_hits}")

    elapsed = time.time() - t_start
    print(f"  Finished in {elapsed:.1f}s | "
          f"{n_with_genes}/{n_total} have genes "
          f"({100*n_with_genes/max(n_total,1):.1f}%) | "
          f"{n_with_hits} have HMM hits")

    return results


def save_results_from_pkl(results: List[dict], output_path: str, n_families: int):
    """
    Lưu đặc trưng thu được qua tiến trình xử lý pkl ra làm tệp PyTorch .pt.

    Cấu trúc (sử dụng dựa trên khóa chỉ mục, có cùng trật tự với pkl):
        {
            "activations": Tensor [n_contigs, N_FAMILIES],
            "gene_stats":  Tensor [n_contigs, 4],
            "n_genes":     Tensor [n_contigs],
            "n_families":  int,
        }

    Trong lúc huấn luyện, chỉ cần áp dụng cơ chế đánh dấu vị trí:
        features = torch.load("train_features.pt")
        activation_i = features["activations"][i]  # được khớp với sequences[i]
    """
    n = len(results)
    activations = np.zeros((n, n_families), dtype=np.float32)
    gene_stats  = np.zeros((n, 4), dtype=np.float32)
    codon_features = np.zeros((n, CODON_FEATURE_DIM), dtype=np.float32)
    n_genes_arr = np.zeros(n, dtype=np.int64)

    for i, r in enumerate(results):
        activations[i] = r["activation"]
        gene_stats[i]  = r["gene_stats"]
        codon_features[i] = r.get(
            "codon_features",
            np.concatenate([np.ones(64, dtype=np.float32),
                            np.array([0.5], dtype=np.float32)]),
        )
        n_genes_arr[i] = r["n_genes"]

    data = {
        "activations": torch.from_numpy(activations),
        "gene_stats":  torch.from_numpy(gene_stats),
        "codon_features": torch.from_numpy(codon_features),
        "n_genes":     torch.from_numpy(n_genes_arr),
        "n_families":  n_families,
        "codon_feature_dim": CODON_FEATURE_DIM,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)

    has_genes = (n_genes_arr > 0).sum()
    has_hits  = (activations.sum(axis=1) > 0).sum()
    avg_genes = n_genes_arr[n_genes_arr > 0].mean() if has_genes > 0 else 0

    print(f"  Saved → {output_path}")
    print(f"    Contigs: {n} | With genes: {has_genes} ({100*has_genes/n:.1f}%) | "
          f"HMM hits: {has_hits} ({100*has_hits/n:.1f}%) | "
          f"Avg genes: {avg_genes:.1f}")


def save_results(results: Dict[str, dict], output_path: str, n_families: int):
    """
    Lưu bảng chú giải kết quả bằng đối tượng tensor PyTorch trong tệp .pt.

    Cấu trúc định dạng:
        {
            "contig_ids": danh sách str,
            "activations": Tensor [n_contigs, N_FAMILIES],
            "gene_stats":  Tensor [n_contigs, 4],
            "n_genes":     Tensor [n_contigs],
            "gene_details": dict { contig_id → danh sách thông tin mã gen },
            "n_families":  int,
        }
    """
    contig_ids = sorted(results.keys())
    n = len(contig_ids)

    activations = np.zeros((n, n_families), dtype=np.float32)
    gene_stats  = np.zeros((n, 4), dtype=np.float32)
    codon_features = np.zeros((n, CODON_FEATURE_DIM), dtype=np.float32)
    n_genes_arr = np.zeros(n, dtype=np.int64)
    gene_details = {}

    default_codon = np.concatenate(
        [np.ones(64, dtype=np.float32), np.array([0.5], dtype=np.float32)]
    )

    for i, cid in enumerate(contig_ids):
        r = results[cid]
        activations[i] = r["activation"]
        gene_stats[i]  = r["gene_stats"]
        codon_features[i] = r.get("codon_features", default_codon)
        n_genes_arr[i] = r["n_genes"]
        gene_details[cid] = r["gene_details"]

    data = {
        "contig_ids": contig_ids,
        "activations": torch.from_numpy(activations),
        "gene_stats":  torch.from_numpy(gene_stats),
        "codon_features": torch.from_numpy(codon_features),
        "n_genes":     torch.from_numpy(n_genes_arr),
        "gene_details": gene_details,
        "n_families":  n_families,
        "codon_feature_dim": CODON_FEATURE_DIM,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)

    # Print summary statistics
    has_genes = (n_genes_arr > 0).sum()
    has_hits  = (activations.sum(axis=1) > 0).sum()
    avg_genes = n_genes_arr[n_genes_arr > 0].mean() if has_genes > 0 else 0

    print(f"\nSaved → {output_path}")
    print(f"  Contigs:        {n}")
    print(f"  With genes:     {has_genes} ({100*has_genes/n:.1f}%)")
    print(f"  With HMM hits:  {has_hits} ({100*has_hits/n:.1f}%)")
    print(f"  Avg genes/contig (when > 0): {avg_genes:.1f}")
    print(f"  Activation sparsity: {100*(activations == 0).mean():.1f}%")

    # Per-family hit counts
    print(f"\n  Per-family detection rate:")
    for j in range(n_families):
        n_hit = (activations[:, j] > 0).sum()
        if n_hit > 0:
            print(f"    [{j:2d}] {n_hit:5d} hits ({100*n_hit/n:.2f}%)")


# ============================================================
# Normalization utilities (for use during training)
# ============================================================

def compute_normalization_stats(features_path: str) -> dict:
    """
    Tính thông số tính giá trị trung bình/tiêu chuẩn độ lệch (mean/std) đối với toàn bộ gene_stats.
    Thực hiện 1 lần duy nhất, tiếp đên đưa dữ liệu ngầm cho mô hình tại thời điểm đào tạo để chuẩn hóa đầu vào.

    Returns:
        { "gene_stats_mean": Tensor[4], "gene_stats_std": Tensor[4],
          "activation_max": Tensor[N] }
    """
    data = torch.load(features_path, weights_only=False)
    gene_stats = data["gene_stats"]   # [n, 4]
    activations = data["activations"]  # [n, N]

    # Gene stats: z-score normalization
    mean = gene_stats.mean(dim=0)
    std  = gene_stats.std(dim=0).clamp(min=1e-6)

    # Activations: max-normalize per family (bit-scores vary widely)
    act_max = activations.max(dim=0).values.clamp(min=1e-6)

    stats = {
        "gene_stats_mean": mean,
        "gene_stats_std": std,
        "activation_max": act_max,
    }

    print(f"Normalization stats from {features_path}:")
    print(f"  gene_stats mean: {mean.tolist()}")
    print(f"  gene_stats std:  {std.tolist()}")
    print(f"  activation max:  {act_max.tolist()}")

    return stats


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Annotate contigs with gene and protein features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # === PKL mode (PhaBERT-CNN existing data) ===
  # Process entire data/processed/ directory:
  python preprocess_gene_features.py \\
      --data_dir data/processed/ \\
      --hmm_db data/hmm/gene_families.hmm \\
      --vocab data/hmm/vocabulary.json

  # Process a single pkl file:
  python preprocess_gene_features.py \\
      --pkl data/processed/group_A/fold_0/train.pkl \\
      --hmm_db data/hmm/gene_families.hmm \\
      --vocab data/hmm/vocabulary.json

  # === FASTA mode ===
  python preprocess_gene_features.py \\
      --contig_fasta data/contigs/group_A.fasta \\
      --hmm_db data/hmm/gene_families.hmm \\
      --vocab data/hmm/vocabulary.json \\
      --output data/annotations/group_A_features.pt

  # Gene prediction only (no HMM, for testing):
  python preprocess_gene_features.py \\
      --data_dir data/processed/ --no_hmm
        """,
    )
    # Input — PKL mode (PhaBERT-CNN data)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="PhaBERT-CNN data dir (data/processed/) with group_*/fold_*/*.pkl")
    parser.add_argument("--pkl", type=str, default=None,
                        help="Path to a single .pkl file")

    # Input — FASTA mode
    parser.add_argument("--contig_fasta", type=str, default=None,
                        help="Path to a single FASTA file with contigs")
    parser.add_argument("--contig_dir", type=str, default=None,
                        help="Directory containing FASTA files (*.fasta, *.fa, *.fna); searched recursively")
    parser.add_argument("--complete_genome", action="store_true",
                        help="Complete-genome mode: train Prodigal on each sequence "
                             "(recommended for full phage genomes ≥ 20 kb)")

    # HMM database
    parser.add_argument("--hmm_db", type=str, default="data/hmm/gene_families.hmm",
                        help="Path to HMM database file")
    parser.add_argument("--vocab", type=str, default="data/hmm/vocabulary.json",
                        help="Path to gene family vocabulary JSON")
    parser.add_argument("--no_hmm", action="store_true",
                        help="Skip HMM scanning (gene stats only)")

    # Output (for FASTA mode)
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pt file (for single FASTA)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (for FASTA batch mode)")

    # Parameters
    parser.add_argument("--min_gene_len", type=int, default=60,
                        help="Minimum gene length in nucleotides (default: 60)")
    parser.add_argument("--e_value", type=float, default=1e-5,
                        help="E-value threshold for HMM hits (default: 1e-5)")
    parser.add_argument("--binary_activation", action="store_true",
                        help="Use binary (0/1) instead of bit-scores")
    parser.add_argument("--compute_norm", type=str, default=None,
                        help="Compute normalization stats from this .pt file")

    args = parser.parse_args()

    # Handle normalization stats mode
    if args.compute_norm:
        stats = compute_normalization_stats(args.compute_norm)
        out_path = Path(args.compute_norm).with_suffix(".norm.pt")
        torch.save(stats, out_path)
        print(f"Saved normalization stats → {out_path}")
        return

    # Validate inputs
    has_pkl = args.data_dir or args.pkl
    has_fasta = args.contig_fasta or args.contig_dir
    if not has_pkl and not has_fasta:
        parser.error("Provide --data_dir / --pkl (for .pkl) or --contig_fasta / --contig_dir (for FASTA)")

    # Initialize gene predictor
    gene_predictor = GenePrediction(
        min_gene_len=args.min_gene_len,
        complete_genome=getattr(args, "complete_genome", False),
    )

    # Initialize HMM scanner (optional)
    hmm_scanner = None
    n_families = 24  # default vocabulary size

    if not args.no_hmm:
        if not Path(args.hmm_db).exists():
            print(f"WARNING: HMM database not found at {args.hmm_db}")
            print("  Run prepare_hmm_profiles.py first, or use --no_hmm")
            print("  Proceeding with gene prediction only (no HMM)...")
        elif not Path(args.vocab).exists():
            print(f"WARNING: Vocabulary not found at {args.vocab}")
            print("  Proceeding with gene prediction only (no HMM)...")
        else:
            hmm_scanner = HMMScanner(
                hmm_path=args.hmm_db,
                vocab_path=args.vocab,
                e_value_threshold=args.e_value,
                use_bitscore=not args.binary_activation,
            )
            n_families = hmm_scanner.n_families

    print(f"\n{'='*60}")
    print(f"PhaBERT-CNN v2 — Gene Feature Preprocessing")
    print(f"{'='*60}")
    complete_genome = getattr(args, "complete_genome", False)
    print(f"  Mode:           {'PKL' if has_pkl else 'FASTA'}")
    print(f"  Gene prediction:{'complete-genome' if complete_genome else 'meta (metagenome)'}")
    print(f"  Gene families:  {n_families}")
    print(f"  Min gene len:   {args.min_gene_len} nt")
    print(f"  HMM scanning:   {'ON' if hmm_scanner else 'OFF'}")
    print(f"  E-value cutoff: {args.e_value}")
    print(f"  Score type:     {'binary' if args.binary_activation else 'bit-score'}")
    print(f"{'='*60}\n")

    # ============================================================
    # PKL mode: process PhaBERT-CNN .pkl files
    # ============================================================
    if has_pkl:
        pkl_files = []

        if args.pkl:
            pkl_files.append(Path(args.pkl))
        elif args.data_dir:
            # Walk data/processed/group_*/fold_*/*.pkl
            data_dir = Path(args.data_dir)
            pkl_files = sorted(data_dir.glob("**/fold_*/*.pkl"))
            if not pkl_files:
                # Also try flat structure
                pkl_files = sorted(data_dir.glob("**/*.pkl"))

        if not pkl_files:
            parser.error(f"No .pkl files found in {args.data_dir or args.pkl}")

        print(f"Found {len(pkl_files)} pkl files to process\n")

        for pkl_path in pkl_files:
            # Output: train.pkl → train_features.pt (same directory)
            out_path = pkl_path.with_name(pkl_path.stem + "_features.pt")
            if out_path.exists():
                print(f"  SKIP {pkl_path} (features already exist)")
                continue

            print(f"{'─'*50}")
            print(f"  {pkl_path.relative_to(Path(args.data_dir) if args.data_dir else pkl_path.parent)}")

            results = process_pkl(
                pkl_path=str(pkl_path),
                gene_predictor=gene_predictor,
                hmm_scanner=hmm_scanner,
                n_families=n_families,
            )

            save_results_from_pkl(results, str(out_path), n_families)

    # ============================================================
    # FASTA mode: process FASTA files
    # ============================================================
    else:
        fasta_files = []
        if args.contig_fasta:
            fasta_files.append(Path(args.contig_fasta))
        elif args.contig_dir:
            contig_dir = Path(args.contig_dir)
            for ext in ["*.fasta", "*.fa", "*.fna"]:
                fasta_files.extend(sorted(contig_dir.rglob(ext)))

        if not fasta_files:
            parser.error(f"No FASTA files found")

        for fasta_path in fasta_files:
            # Determine output path early so we can skip if it already exists
            if args.output and len(fasta_files) == 1:
                skip_check_path = Path(args.output)
            elif args.output_dir:
                skip_check_path = Path(args.output_dir) / f"{fasta_path.stem}_features.pt"
            else:
                skip_check_path = fasta_path.with_suffix(".features.pt")

            if skip_check_path.exists():
                print(f"  SKIP {fasta_path.name} (features already exist: {skip_check_path})")
                continue

            print(f"\n{'─'*50}")
            print(f"File: {fasta_path.name}")
            print(f"{'─'*50}")

            results = process_fasta(
                fasta_path=str(fasta_path),
                gene_predictor=gene_predictor,
                hmm_scanner=hmm_scanner,
                n_families=n_families,
            )

            if args.output and len(fasta_files) == 1:
                out_path = args.output
            elif args.output_dir:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = str(out_dir / f"{fasta_path.stem}_features.pt")
            else:
                out_path = str(fasta_path.with_suffix(".features.pt"))

            save_results(results, out_path, n_families)

    print(f"\n{'='*60}")
    print("All done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()