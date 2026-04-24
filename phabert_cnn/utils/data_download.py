"""
utils/data_download.py
======================
Hệ thống tiện ích hỗ trợ phân giải và truy xuất bộ dữ liệu (dataset) chuyên biệt cho mô hình PhaBERT-CNN.

Các thay đổi chiến lược so với phiên bản v1 (Thay đổi kiến trúc API):
  - Phương thức `prepare_genome_dataset()` hiện trả về cấu trúc liên kết 3 thành tố: (genome_id, sequence, label)
    thay vì bộ số liệu tĩnh (sequence, label). Cải tiến này cho phép đối chiếu mã định danh genome ID với
    ma trận đặc trưng gen (annotations/features) đã được trích xuất ưu tiên từ `preprocess_gene_features.py`.
  - Thuật toán khử nhiễu trùng lặp (Deduplication): Tự động phát hiện genome ID trùng lặp xuyên suốt nhiều tập
    kết xuất FASTA phân tán (ví dụ: Dataset-1 đan xen với các phân mảnh sequences_*), và chỉ bảo lưu biên bản ghi nhận đầu tiên.
    Trạng thái báo cáo chuẩn tắc của PhaBERT-CNN ghi nhận 2.241 cá thể bộ gen duy nhất (707 chủng ôn hoà + 1.534 chủng độc lực).

Tổng hợp định lượng tập mẫu (Dataset Profile): 2.241 bộ gen thực khuẩn thể hoàn chỉnh:
  - 707 thực khuẩn thể ôn hoà (nhãn 0)
  - 1.534 thực khuẩn thể độc lực (nhãn 1)

Nguồn dữ liệu:
  - DeePhage (Dataset-1, Dataset-2): https://github.com/shufangwu/DeePhage
  - DeepPL   (sequences_*):          Zhang et al. 2024, PLOS Comput Biol
"""

import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

from Bio import SeqIO


# ============================================================
# Trình tải dữ liệu (tùy chọn — bỏ qua với --skip_download)
# ============================================================

def download_deephage_data(output_dir: str) -> str:
    """Tự động truy xuất và tải xuống tập cơ sở DeePhage thông qua repository GitHub nếu dữ liệu chép cục bộ chưa tồn tại."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    deephage_dir = output_dir / "deephage"
    if deephage_dir.exists():
        print("Trạng thái: Tập dữ liệu DeePhage tồn tại hợp lệ trong phân vùng lưu trữ cục bộ. Tiến hành bỏ qua giao thức tải xuống.")
        return str(deephage_dir)

    print("Quy trình: Khởi động đường truyền tải xuống tập mẫu DeePhage...")
    try:
        subprocess.run(
            ["git", "clone",
             "https://github.com/shufangwu/DeePhage.git",
             str(deephage_dir)],
            check=True,
        )
        print("Xác nhận: Quy trình truy xuất dữ liệu DeePhage hoàn tất thành công.")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi truy nhập mạng: Tiến trình tải xuống bị gián đoạn: {e}")
    return str(deephage_dir)


def download_deeppl_data(output_dir: str) -> str:
    """Vùng giữ chỗ (Placeholder) đại diện nhóm mẫu DeepPL. Theo quy ước chuẩn, người dùng phải đáp ứng quá trình tải xuống thủ công."""
    output_dir = Path(output_dir)
    deeppl_dir = output_dir / "deeppl"
    deeppl_dir.mkdir(parents=True, exist_ok=True)

    if not (deeppl_dir / "data").exists():
        print("=" * 60)
        print("KHUYẾN CÁO: Tài nguyên bộ mẫu DeepPL yêu cầu thao tác giải nén và tải về thủ công trực tiếp từ")
        print("trang phụ lục dữ liệu của ấn phẩm nghiên cứu.")
        print(f"Đường dẫn định vị yêu cầu: {deeppl_dir}")
        print("=" * 60)
    return str(deeppl_dir)


# ============================================================
# Đọc bộ gen kèm ID + khử trùng lặp
# ============================================================

def _load_fasta_with_ids(fasta_path: Path, label: int) -> List[Tuple[str, str, int]]:
    """
    Tiến hành phiên dịch các trình tự gen từ tệp FASTA, đặc biệt bảo toàn thuộc tính nhận diện (ID) của chuỗi khối.

    Tham số mô tả (Args):
        fasta_path: Tuyến đường chỉ định đích đến tập tin FASTA đích
        label:      Nhãn giá trị nguyên (0 chỉ định hình thái ôn hoà, 1 biểu thị hình thức độc lực)

    Kết quả mong đợi (Returns):
        Chuỗi mảng cấu trúc (Danh sách List của các tuple đại lượng: (genome_id, sequence, label)).
    """
    loaded = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        genome_id = record.id  # Đọc phân tách ID theo cơ chế xác định token tiên phong sau ký hiệu tiêu đề '>'
        sequence = str(record.seq).upper()
        if len(sequence) < 100:
            continue  # Lược bỏ các phân mảnh gen khuyết hoặc quá ngắn (Tối thiểu phải ≥ 100 bp)
        loaded.append((genome_id, sequence, label))
    return loaded


def prepare_genome_dataset(
    data_dir: str,
    virulent_patterns: Optional[List[str]] = None,
    temperate_patterns: Optional[List[str]] = None,
) -> List[Tuple[str, str, int]]:
    """
    Nạp đối tượng dataset cấu thành từ đa nguồn tập lệnh FASTA phân lập.

    Hệ thống phân định bộ phân lớp tự động, quét nhãn file dựa theo cụm cú pháp định dạng: "virulent"
    sẽ được tự động biên phiên thành biến nhãn độc lực (label=1), trong khi "temperate" quy đổi thành dạng ôn hoà (label=0).

    Loại trừ điểm nút dư thừa đa nguyên: thuật toán nhận biết trùng lặp ID cục bộ của hệ gen, trường hợp
    số hiệu được tái định danh (giao thoa thư mục xuất bản của DeePhage và DeepPL), hệ thống chốt lưu vết hiện diện đầu tiên.

    Kết xuất (Returns):
        Chuỗi danh sách tổ hợp đối chiếu 3 tham biến (genome_id, sequence, label).
    """
    data_dir = Path(data_dir)

    # Thu thập file FASTA
    if virulent_patterns is None:
        virulent_patterns = ["*virulent*.fasta", "*virulent*.fa", "*virulent*.fna"]
    if temperate_patterns is None:
        temperate_patterns = ["*temperate*.fasta", "*temperate*.fa", "*temperate*.fna"]

    virulent_files = []
    temperate_files = []
    for pattern in virulent_patterns:
        virulent_files.extend(sorted(data_dir.glob(pattern)))
    for pattern in temperate_patterns:
        temperate_files.extend(sorted(data_dir.glob(pattern)))

    # Dedup theo (1) record.id, (2) hash chuỗi (catch cùng genome tên khác),
    # (3) hash k-mer minhash thô để bắt near-duplicate (>95% identity).  Bước
    # (3) dùng sketch đơn giản: lấy min 128 hash của 21-mer — tương tự MinHash
    # độ tương đồng Jaccard, rẻ hơn CD-HIT nhiều.
    import hashlib

    def _seq_hash(seq: str) -> str:
        return hashlib.md5(seq.encode("ascii", errors="ignore")).hexdigest()

    def _minhash_sketch(seq: str, k: int = 21, n_hash: int = 128) -> frozenset:
        if len(seq) < k:
            return frozenset()
        # Chỉ lấy min-hash của k-mer; đủ để check Jaccard thô giữa 2 genome.
        hashes = set()
        for i in range(0, len(seq) - k + 1, max(1, (len(seq) - k + 1) // (n_hash * 8))):
            kmer = seq[i:i + k]
            if "N" in kmer:
                continue
            hashes.add(int(hashlib.md5(kmer.encode()).hexdigest()[:8], 16))
        # Lấy n_hash giá trị nhỏ nhất làm sketch
        return frozenset(sorted(hashes)[:n_hash])

    def _jaccard(a: frozenset, b: frozenset) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    seen_ids = set()
    seen_hashes = set()
    sketches: List[Tuple[str, frozenset, int]] = []  # (gid, sketch, label)
    NEAR_DUP_THRESHOLD = 0.90   # Jaccard của MinHash sketch
    genomes: List[Tuple[str, str, int]] = []

    def _try_add(gid: str, seq: str, label: int) -> Tuple[bool, str]:
        if gid in seen_ids:
            return False, "id"
        h = _seq_hash(seq)
        if h in seen_hashes:
            return False, "exact-seq"
        sketch = _minhash_sketch(seq)
        for prev_gid, prev_sketch, prev_label in sketches:
            if _jaccard(sketch, prev_sketch) >= NEAR_DUP_THRESHOLD:
                return False, f"near-dup({prev_gid},label={prev_label})"
        seen_ids.add(gid)
        seen_hashes.add(h)
        sketches.append((gid, sketch, label))
        genomes.append((gid, seq, label))
        return True, "ok"

    skip_counts = {"id": 0, "exact-seq": 0, "near-dup": 0}

    for fp in virulent_files:
        print(f"Trạng thái nạp: Giải mã tệp cơ sở hệ gen chủng độc lực tại vị trí: {fp}")
        raw = _load_fasta_with_ids(fp, label=1)
        added = 0
        for gid, seq, label in raw:
            ok, reason = _try_add(gid, seq, label)
            if ok:
                added += 1
            else:
                key = "near-dup" if reason.startswith("near-dup") else reason
                skip_counts[key] = skip_counts.get(key, 0) + 1
        print(f"  -> Tổng {len(raw)} thô, thêm mới {added}")

    for fp in temperate_files:
        print(f"Trạng thái nạp: Giải mã tệp cơ sở hệ gen chủng ôn hoà tại vị trí: {fp}")
        raw = _load_fasta_with_ids(fp, label=0)
        added = 0
        for gid, seq, label in raw:
            ok, reason = _try_add(gid, seq, label)
            if ok:
                added += 1
            else:
                key = "near-dup" if reason.startswith("near-dup") else reason
                skip_counts[key] = skip_counts.get(key, 0) + 1
        print(f"  -> Tổng {len(raw)} thô, thêm mới {added}")

    print(f"\nDedup report — bỏ qua: id={skip_counts['id']}, "
          f"exact-seq={skip_counts['exact-seq']}, "
          f"near-dup(Jaccard≥{NEAR_DUP_THRESHOLD})={skip_counts['near-dup']}")

    # Tóm tắt
    n_vir = sum(1 for _, _, l in genomes if l == 1)
    n_temp = sum(1 for _, _, l in genomes if l == 0)
    print(f"\nKiểm kê tích lũy bộ gen chuẩn tắc: {len(genomes)} "
          f"({n_vir} xác nhận phổ độc lực, {n_temp} thuộc phổ ôn hoà)")

    if not genomes:
        print("\nCảnh báo: Khối dữ liệu FASTA nguồn phản hồi giá trị rỗng. Vui lòng kiểm tra định vị các tệp nền tảng tại không gian:")
        print(f"  {data_dir}/")
        print("Lưu ý quy tắc đánh dấu tĩnh: Cú pháp tên tệp tin chỉ định nhóm bắt buộc có tiền tố phân nhóm 'virulent' (độc lực) hoặc 'temperate' (ôn hòa).")

    return genomes
