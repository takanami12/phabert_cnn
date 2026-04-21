#!/usr/bin/env python3
"""
prepare_hmm_profiles.py
=======================
Tải xuống và chuẩn bị các profile HMM để nhận diện họ gen sinh tan/tiềm tan (lysogenic/lytic).

Hỗ trợ hai chiến lược:
  1. Dựa trên Pfam: tải từng profile Pfam HMM cho các họ gen đã được giám tuyển.
  2. Dựa trên VOG:  tải cơ sở dữ liệu VOG và lọc ra các họ gen liên quan.

Cách dùng:
    python prepare_hmm_profiles.py --strategy pfam --output_dir data/hmm/
    python prepare_hmm_profiles.py --strategy vog  --output_dir data/hmm/

Đầu ra:
    data/hmm/gene_families.hmm       — ghép các cấu hình HMM (sẵn sàng để press)
    data/hmm/gene_families.hmm.h3*   — định dạng CSDL đã press (cho pyhmmer)
    data/hmm/vocabulary.json         — ánh xạ từ vựng của họ gen
"""

import json
import argparse
from pathlib import Path

# ============================================================
# Từ vựng các họ gen
# ============================================================
# Mỗi mục: { pfam_accession, name, category, keywords }
# - pfam:     ID của Pfam để tải xuống (nếu strategy=pfam)
# - keywords: các từ khóa để khớp với chú thích của VOG (nếu strategy=vog)
# - category: "lusogenic" (tiềm tan) hoặc "lytic" (sinh tan) — vai trò sinh học

GENE_VOCABULARY = [
    # ------------------------------------------------------------------ #
    # LYSOGENIC ĐỘC QUYỀN (idx 0-2, 8, 25)                               #
    # CHỈ có mặt trong thực khuẩn thể ôn hòa — Tín hiệu phân biệt mạnh nhất
    # ------------------------------------------------------------------ #
    {"idx":  0, "name": "Integrase_tyrosine",  "category": "exclusive_lysogenic",
     "pfam": "PF00589", "keywords": ["integrase", "tyrosine recombinase"]},
    {"idx":  1, "name": "Integrase_serine",    "category": "exclusive_lysogenic",
     "pfam": "PF07508", "keywords": ["serine recombinase", "resolvase"]},
    {"idx":  2, "name": "Excisionase",         "category": "exclusive_lysogenic",
     "pfam": "PF06806", "keywords": ["excisionase", "xis"]},

    # ------------------------------------------------------------------ #
    # ĐIỀU HÒA LYSOGENY (idx 3, 24)                                      #
    # Chất ức chế CI yêu cầu CẢ HAI domain trên cùng protein (luật dual-Pfam) #
    # PF07022 = HTH đầu N; PF00717 = phân giải protein đầu C             #
    # ------------------------------------------------------------------ #
    {"idx":  3, "name": "CI_repressor",        "category": "lysogeny_regulatory",
     "pfam": "PF07022", "pfam_secondary": "PF00717", "ci_require_both": True,
     "keywords": ["CI repressor", "phage repressor", "cI protein"]},

    # ------------------------------------------------------------------ #
    # ĐIỀU HÒA LYTIC (idx 4, 9)                                          #
    # Đẩy mạnh chu trình sinh tan; được phân loại lại từ "lysogenic"     #
    # Cro: bám vào OR3 → ức chế tổng hợp CI → thúc đẩy quyết định sinh tan. #
    # Antirepressor: vô hiệu hóa CI → làm mất ổn định trạng thái lysogeny.  #
    # ------------------------------------------------------------------ #
    {"idx":  4, "name": "Cro_antirepressor",   "category": "lytic_regulatory",
     "pfam": "PF01381", "keywords": ["cro", "antirepressor"]},

    # ------------------------------------------------------------------ #
    # KHÔNG RÕ RÀNG (idx 5, 10, 11)                                      #
    # Độ đặc hiệu thấp — có mặt trong phage độc lực hoặc MGE của vi khuẩn #
    # ------------------------------------------------------------------ #
    {"idx":  5, "name": "Transposase",         "category": "ambiguous",
     "pfam": "PF01609", "keywords": ["transposase"]},

    # ------------------------------------------------------------------ #
    # LYSOGENIC PLASMID (idx 6, 7)                                       #
    # Các phage giống P1 được duy trì dưới dạng plasmid ngoài nhiễm sắc thể  #
    # ------------------------------------------------------------------ #
    {"idx":  6, "name": "ParA_partitioning",   "category": "plasmid_lysogenic",
     "pfam": "PF10609", "keywords": ["parA", "partitioning ATPase"]},
    {"idx":  7, "name": "ParB_partitioning",   "category": "plasmid_lysogenic",
     "pfam": "PF02195", "keywords": ["parB", "partition"]},

    {"idx":  8, "name": "Superinfection_imm",  "category": "exclusive_lysogenic",
     "pfam": "PF13657", "keywords": ["superinfection", "immunity"]},

    {"idx":  9, "name": "Phage_antirepressor", "category": "lytic_regulatory",
     "pfam": "PF03374", "keywords": ["antirepressor", "anti-repressor"]},

    {"idx": 10, "name": "Resolvase",           "category": "ambiguous",
     "pfam": "PF00239", "keywords": ["resolvase", "recombinase"]},
    {"idx": 11, "name": "DNA_methylase",       "category": "ambiguous",
     "pfam": "PF01555", "keywords": ["methyltransferase", "methylase", "DNA methylation"]},

    # ------------------------------------------------------------------ #
    # CHIA SẺ THỰC THI PHÂN GIẢI (LYSIS) (idx 12-15)                     #
    # Được yêu cầu bởi CẢ CÁC phage độc lực VÀ ôn hòa để phân giải vật chủ #
    # Không thể dùng độc lập để phân biệt — phân loại lại từ "lytic"      #
    # ------------------------------------------------------------------ #
    {"idx": 12, "name": "Holin_class_I",       "category": "shared_lysis",
     "pfam": "PF01510", "keywords": ["holin"]},
    {"idx": 13, "name": "Holin_class_II",      "category": "shared_lysis",
     "pfam": "PF05105", "keywords": ["holin class II"]},
    {"idx": 14, "name": "Endolysin",           "category": "shared_lysis",
     "pfam": "PF00959", "keywords": ["endolysin", "lysin", "lysozyme", "muramidase"]},
    {"idx": 15, "name": "Spanin_Rz",           "category": "shared_lysis",
     "pfam": "PF06605", "keywords": ["spanin", "Rz"]},

    # ------------------------------------------------------------------ #
    # CHIA SẺ CẤU TRÚC / LẮP RÁP VIRION (idx 16-23)                      #
    # Có mặt trong TẤT CẢ các phage dsDNA bất kể kiểu hình                #
    # Giúp xác nhận định danh phage nhưng KHÔNG phân biệt được độc/ôn hòa #
    # Phân loại lại từ "lytic" (trước đây là không đúng về mặt sinh học)    #
    # ------------------------------------------------------------------ #
    {"idx": 16, "name": "Terminase_large",     "category": "shared_structural",
     "pfam": "PF03354", "keywords": ["terminase large", "TerL"]},
    {"idx": 17, "name": "Terminase_small",     "category": "shared_structural",
     "pfam": "PF04466", "keywords": ["terminase small", "TerS"]},
    {"idx": 18, "name": "Portal_protein",      "category": "shared_structural",
     "pfam": "PF05065", "keywords": ["portal"]},
    {"idx": 19, "name": "Major_capsid",        "category": "shared_structural",
     "pfam": "PF11651", "keywords": ["major capsid", "MCP", "HK97"]},
    {"idx": 20, "name": "Tail_fiber",          "category": "shared_structural",
     "pfam": "PF09693", "keywords": ["tail fiber", "tail spike", "receptor binding"]},
    {"idx": 21, "name": "Baseplate",           "category": "shared_structural",
     "pfam": "PF04865", "keywords": ["baseplate"]},
    {"idx": 22, "name": "Tail_tape_measure",   "category": "shared_structural",
     "pfam": "PF09718", "keywords": ["tape measure", "tail length"]},
    {"idx": 23, "name": "Neck_protein",        "category": "shared_structural",
     "pfam": "PF04883", "keywords": ["neck", "head-tail connector"]},

    # ------------------------------------------------------------------ #
    # CÁC HỌ MỚI (idx 24-25) — mở rộng lấy cảm hứng từ BACPHLIP         #
    # ------------------------------------------------------------------ #

    # Domain phân giải tự xúc tác đầu C của chất ức chế CI (Peptidase_S24 / họ LexA)
    # Được dùng làm phát hiện bổ sung để nhận diện CI với độ tin cậy cao.
    # Chất ức chế CI thực sự = gen chứa CẢ PF07022 (idx 3) + PF00717 (idx 24).
    {"idx": 24, "name": "CI_repressor_Ctail",  "category": "lysogeny_regulatory",
     "pfam": "PF00717",
     "keywords": ["LexA", "repressor C terminal", "Peptidase_S24", "CI cleavage"]},

    # Phage integrase kiểu DDE (bao phủ các integrase không bị bắt bởi 
    # PF00589/PF07508 — cải thiện mức độ phát hiện ở các phage ôn hòa mới theo BACPHLIP)
    {"idx": 25, "name": "Integrase_DDE",       "category": "exclusive_lysogenic",
     "pfam": "PF02022",
     "keywords": ["DDE integrase", "retroviral integrase",
                  "phage integrase DDE", "rve integrase"]},
]

N_FAMILIES = len(GENE_VOCABULARY)


def save_vocabulary(output_dir: Path):
    """Lưu từ vựng gen dưới dạng chuỗi JSON phục vụ cho quá trình huấn luyện."""
    # Tuần tự hóa các họ gen: chuyển đổi tham số bool ci_require_both → JSON bool
    families_json = []
    for v in GENE_VOCABULARY:
        entry = dict(v)
        families_json.append(entry)

    def _indices(cat):
        return [v["idx"] for v in GENE_VOCABULARY if v["category"] == cat]

    vocab = {
        "n_families": N_FAMILIES,
        "families": families_json,
        "idx_to_name": {str(v["idx"]): v["name"] for v in GENE_VOCABULARY},
        "name_to_idx": {v["name"]: v["idx"] for v in GENE_VOCABULARY},
        # Phân loại nhóm sinh học tinh chỉnh (đã được làm chuẩn)
        "exclusive_lysogenic_indices": _indices("exclusive_lysogenic"),
        "lysogeny_regulatory_indices": _indices("lysogeny_regulatory"),
        "plasmid_lysogenic_indices":   _indices("plasmid_lysogenic"),
        "lytic_regulatory_indices":    _indices("lytic_regulatory"),
        "shared_lysis_indices":        _indices("shared_lysis"),
        "shared_structural_indices":   _indices("shared_structural"),
        "ambiguous_indices":           _indices("ambiguous"),
        # Các nhóm aggregated cho tính tương thích ngược
        # tiềm tan (lysogenic) = tất cả họ gen định tính chỉ báo lối sống tiềm tan
        "lysogenic_indices": sorted(
            _indices("exclusive_lysogenic") +
            _indices("lysogeny_regulatory") +
            _indices("plasmid_lysogenic")
        ),
        # sinh tan (lytic) = họ gen tích cực đẩy mạnh sự lựa chọn phát triển lytic 
        # (thuộc cấu trúc/shared-lysis đã bị loại: nó xuất hiện trong CẢ HAI lối sống)
        "lytic_indices": _indices("lytic_regulatory"),
    }
    vocab_path = output_dir / "vocabulary.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Đã lưu từ vựng ({N_FAMILIES} họ gen) → {vocab_path}")
    return vocab


def _read_hmm_file(path) -> str:
    """Đọc tệp HMM, có thể ở định dạng root text hoặc nén gzip."""
    import gzip
    with open(path, "rb") as fb:
        magic = fb.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(path, "rt") as f:
            return f.read()
    with open(path, "r") as f:
        return f.read()


def prepare_pfam_profiles(output_dir: Path):
    """
    Tải từng cấu hình Pfam HMM và nối chúng thành một tệp đơn duy nhất.

    Yêu cầu truy cập mạng. Mỗi cấu hình tầm khoảng ~5-50KB.
    Tổng dung lượng tải về: Khoảng ~500KB cho 24 cấu hình.
    """
    import urllib.request

    hmm_dir = output_dir / "individual"
    hmm_dir.mkdir(parents=True, exist_ok=True)

    pfam_base_url = "https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/{accession}?annotation=hmm"
    # Cách thay thế: Pfam FTP
    # pfam_base_url = "https://pfam.xfam.org/family/{accession}/hmm"

    all_hmm_text = []
    seen_pfam = set()

    for family in GENE_VOCABULARY:
        acc = family["pfam"]
        name = family["name"]

        if acc in seen_pfam:
            print(f"  [{name}] Sử dụng lại {acc} (đã được tải)")
            continue
        seen_pfam.add(acc)

        hmm_file = hmm_dir / f"{acc}.hmm"
        if hmm_file.exists():
            print(f"  [{name}] {acc} đã có sẵn trong bộ nhớ đệm")
            content = _read_hmm_file(hmm_file)
            if content:
                all_hmm_text.append(content)
            continue

        # Thử tải từ Pfam/InterPro
        url = pfam_base_url.format(accession=acc)
        print(f"  [{name}] Đang tải {acc} từ {url} ...")
        try:
            urllib.request.urlretrieve(url, hmm_file)
            content = _read_hmm_file(hmm_file)
            if content and content.startswith("HMMER"):
                all_hmm_text.append(content)
                print(f"    HOÀN TẤT ({len(content)} bytes)")
            else:
                print(f"    CẢNH BÁO: định dạng không mong đợi, bỏ qua")
                if hmm_file.exists():
                    hmm_file.unlink()
        except Exception as e:
            print(f"    THẤT BẠI: {e}")
            print(f"    Bạn có thể cần tải xuống {acc}.hmm thủ công từ Pfam")

    # Nối tất cả các cấu hình
    concat_path = output_dir / "gene_families.hmm"
    with open(concat_path, "w") as f:
        f.write("\n".join(all_hmm_text))
    print(f"\nĐã nối {len(all_hmm_text)} cấu hình HMM → {concat_path}")

    # Lưu ánh xạ Pfam để preprocess_gene_features.py giải quyết tên HMM
    mapping = _build_pfam_mapping(output_dir)
    mapping_path = output_dir / "gene_families.hmm2family.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f)
    print(f"Đã lưu ánh xạ HMM→family ({len(mapping)} mục) → {mapping_path}")

    # Nhấn cơ sở dữ liệu bằng công cụ hmmpress (thông qua pyhmmer)
    press_hmm_database(concat_path)


def _smart_open(path, mode="rt"):
    """Mở tệp với khả năng tự động phát hiện kiểu nén (gzip, bzip2, hoặc chưa nén)."""
    import gzip
    import bz2

    # Phát hiện nhờ việc đọc magic bytes
    with open(path, "rb") as f:
        magic = f.read(3)

    if magic[:2] == b"\x1f\x8b":        # gzip
        return gzip.open(path, mode)
    elif magic[:2] == b"BZ":             # bzip2
        return bz2.open(path, mode)
    else:                                 # plain text
        return open(path, mode)


def prepare_vog_profiles(output_dir: Path):
    """
    Tải CSDL HMM VOG và trích xuất các cấu hình liên quan.

    CSDL VOG: tải về ~200MB, trích xuất ~600MB.
    Trang web: https://vogdb.org/

    Lưu ý: VOG phân phối tệp dưới dạng .tar.bz2 (nén bzip2).
    """
    import tarfile
    import urllib.request

    vog_dir = output_dir / "vog_raw"
    vog_dir.mkdir(parents=True, exist_ok=True)

    # Các URL của VOG — tệp lưu trữ HMM được nén bzip2
    vog_url = "https://fileshare.csb.univie.ac.at/vog/latest/vog.hmm.tar.bz2"
    vog_annot_url = "https://fileshare.csb.univie.ac.at/vog/latest/vog.annotations.tsv.gz"

    # Tải các chú thích
    annot_file = vog_dir / "vog.annotations.tsv.gz"
    if not annot_file.exists():
        print(f"Đang tải chú thích VOG...")
        urllib.request.urlretrieve(vog_annot_url, annot_file)

    # Phân tích chú thích để tìm các VOG liên quan
    print("Đang phân tích chú thích VOG để tìm các họ gen sinh tan/tiềm tan...")
    relevant_vogs = {}  # VOG_id → family_idx

    with _smart_open(annot_file, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            vog_id = parts[0]
            description = parts[4].lower() if len(parts) > 4 else ""

            for family in GENE_VOCABULARY:
                for kw in family["keywords"]:
                    if kw.lower() in description:
                        relevant_vogs[vog_id] = family["idx"]
                        break

    print(f"Tìm thấy {len(relevant_vogs)} VOG khớp với các họ gen")

    # Tải tệp cấu hình HMM (tệp lưu trữ bzip2 tar)
    # Kiểm tra cả hai định dạng (tệp .tar.gz cũ và tệp .tar.bz2 chính xác)
    hmm_tar = vog_dir / "vog.hmm.tar.bz2"
    hmm_tar_old = vog_dir / "vog.hmm.tar.gz"
    if hmm_tar_old.exists() and not hmm_tar.exists():
        # Lần chạy trước tải định dạng sai — tái sử dụng lại
        print(f"  Đã tìm thấy tệp {hmm_tar_old.name} trong bộ nhớ rỗng (đổi tên lại thành .tar.bz2)")
        hmm_tar_old.rename(hmm_tar)
    if not hmm_tar.exists():
        print(f"Đang tải các cấu hình HMM của VOG (~200MB)...")
        urllib.request.urlretrieve(vog_url, hmm_tar)

    print("Đang trích xuất các cấu hình HMM liên quan...")
    concat_path = output_dir / "gene_families.hmm"
    extracted = 0

    # Sử dụng tùy chọn "r:*" để cho phép tự phát hiện định dạng mã hóa (xử lý bz2, gz, xz)
    with open(concat_path, "w") as out_f:
        with tarfile.open(hmm_tar, "r:*") as tar:
            for member in tar.getmembers():
                vog_id = member.name.replace(".hmm", "").split("/")[-1]
                if vog_id in relevant_vogs:
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode()
                        out_f.write(content + "\n")
                        extracted += 1

    print(f"Đã trích xuất được {extracted} cấu hình HMM → {concat_path}")

    # Lưu ánh xạ VOG→family map để preprocess_gene_features.py sử dụng nó
    mapping_path = output_dir / "gene_families.hmm2family.json"
    with open(mapping_path, "w") as f:
        json.dump(relevant_vogs, f)
    print(f"Đã lưu ánh xạ HMM→family (với {len(relevant_vogs)} mục) → {mapping_path}")

    press_hmm_database(concat_path)


def _build_pfam_mapping(output_dir: Path) -> dict:
    """
    Xây dựng ánh xạ hmm2family cho các cấu hình Pfam.

    Các tệp Pfam HMM sử dụng hệ lưu trữ trường model NAME (ví dụ "PF00589"),
    vì vậy chúng ta thiết lập ánh xạ accession → family_idx trực tiếp từ GENE_VOCABULARY.
    Đầu ra dict: {pfam_accession: family_idx}
    """
    mapping = {}
    for family in GENE_VOCABULARY:
        acc = family["pfam"]
        mapping[acc] = family["idx"]
        # Một số tệp Pfam HMM có thể sử dụng accession.version (ví dụ "PF00589.27") làm tên
        mapping[acc.split(".")[0]] = family["idx"]
        # Bao gồm accesion phụ nếu có (ví dụ: CI_repressor_Ctail)
        if "pfam_secondary" in family:
            sec = family["pfam_secondary"]
            mapping[sec] = family["idx"]
            mapping[sec.split(".")[0]] = family["idx"]
    return mapping


def prepare_combined_profiles(output_dir: Path):
    """
    Tải CẢ Pfam VÀ VOG profiles, và gộp chúng vào cơ sở dữ liệu chung.

    Chiến lược:
      1. Tải các cấu hình Pfam cho tất cả 26 họ (nguồn đánh giá precision)
      2. Tải VOG + trích xuất bằng cách tìm kiếm từ khoá (nguồn đánh giá recall)
      3. Gộp tất cả vào gene_families.hmm
      4. Tạo tệp hmm2family.json đồng bộ cho cả Pfam accessions và VOG IDs

    Khi một protein vừa khớp Pfam và VOG tương tự đối với cùng một họ,
    preprocess_gene_features.py sẽ dùng max(bit_score) — để tránh tính kép.
    """
    import urllib.request

    hmm_dir = output_dir / "individual"
    hmm_dir.mkdir(parents=True, exist_ok=True)

    pfam_base_url = "https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/{accession}?annotation=hmm"
    all_hmm_text = []
    seen_pfam = set()

    # ---- Giai đoạn 1: Pfam ----
    print("\n[Kết hợp] Giai đoạn 1: Đang tải cấu hình Pfam...")
    for family in GENE_VOCABULARY:
        for acc in [family["pfam"]] + ([family.get("pfam_secondary")] if family.get("pfam_secondary") else []):
            if acc in seen_pfam:
                continue
            seen_pfam.add(acc)
            name = family["name"]
            hmm_file = hmm_dir / f"{acc}.hmm"
            if hmm_file.exists():
                print(f"  [{name}] {acc} đã tồn tại trong cache")
                content = _read_hmm_file(hmm_file)
                if content:
                    all_hmm_text.append(content)
                continue
            url = pfam_base_url.format(accession=acc)
            print(f"  [{name}] Đang tải {acc} ...")
            try:
                urllib.request.urlretrieve(url, hmm_file)
                content = _read_hmm_file(hmm_file)
                if content and content.startswith("HMMER"):
                    all_hmm_text.append(content)
                    print(f"    HOÀN TẤT ({len(content)} bytes)")
                else:
                    print(f"    CẢNH BÁO: định dạng không mong đợi, bỏ qua")
                    if hmm_file.exists():
                        hmm_file.unlink()
            except Exception as e:
                print(f"    THẤT BẠI: {e}")

    pfam_count = len(all_hmm_text)
    print(f"  Pfam: Đã thu thập được {pfam_count} cấu hình")

    # Xây dựng ánh xạ Pfam
    combined_mapping = _build_pfam_mapping(output_dir)

    # ---- Giai đoạn 2: VOG ----
    print("\n[Kết hợp] Giai đoạn 2: Đang tải cấu hình VOG...")
    import tarfile

    vog_dir = output_dir / "vog_raw"
    vog_dir.mkdir(parents=True, exist_ok=True)

    vog_url = "https://fileshare.csb.univie.ac.at/vog/latest/vog.hmm.tar.bz2"
    vog_annot_url = "https://fileshare.csb.univie.ac.at/vog/latest/vog.annotations.tsv.gz"

    annot_file = vog_dir / "vog.annotations.tsv.gz"
    if not annot_file.exists():
        print("  Đang tải tệp chú thích VOG...")
        urllib.request.urlretrieve(vog_annot_url, annot_file)

    print("  Đang phân tích chú thích VOG...")
    vog_mapping = {}  # VOG_id → family_idx
    with _smart_open(annot_file, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            vog_id = parts[0]
            description = parts[4].lower()
            for family in GENE_VOCABULARY:
                for kw in family["keywords"]:
                    if kw.lower() in description:
                        vog_mapping[vog_id] = family["idx"]
                        break

    print(f"  Đã tìm thấy {len(vog_mapping)} VOGs khớp các họ gen")

    hmm_tar = vog_dir / "vog.hmm.tar.bz2"
    if not hmm_tar.exists():
        print("  Đang tải cấu hình VOG HMM (~200MB)...")
        urllib.request.urlretrieve(vog_url, hmm_tar)

    print("  Đang trích xuất cấu hình VOG khớp...")
    vog_hmm_texts = []
    with tarfile.open(hmm_tar, "r:*") as tar:
        for member in tar.getmembers():
            vog_id = member.name.replace(".hmm", "").split("/")[-1]
            if vog_id in vog_mapping:
                f = tar.extractfile(member)
                if f:
                    vog_hmm_texts.append(f.read().decode())

    print(f"  VOG: Đã thu thập được {len(vog_hmm_texts)} cấu hình")
    all_hmm_text.extend(vog_hmm_texts)
    combined_mapping.update(vog_mapping)

    # ---- Giai đoạn 3: Ghi các DB đã hợp nhất ----
    concat_path = output_dir / "gene_families.hmm"
    with open(concat_path, "w") as f:
        f.write("\n".join(all_hmm_text))
    print(f"\n[Kết hợp] Tổng cộng: {pfam_count} Pfam + {len(vog_hmm_texts)} VOG "
          f"= {len(all_hmm_text)} cấu hình → {concat_path}")

    mapping_path = output_dir / "gene_families.hmm2family.json"
    with open(mapping_path, "w") as f:
        json.dump(combined_mapping, f)
    print(f"Đã lưu ánh xạ HMM→family gộp chung "
          f"(gồm {len(combined_mapping)} mục) → {mapping_path}")

    press_hmm_database(concat_path)


def press_hmm_database(hmm_path: Path):
    """Nén ("press") cơ sở dữ liệu HMM để tìm kiếm nhanh bằng pyhmmer."""
    import pyhmmer

    print(f"Đang press tệp cơ sở dữ liệu HMM: {hmm_path}")
    with pyhmmer.plan7.HMMFile(str(hmm_path)) as hmm_file:
        hmms = list(hmm_file)
    print(f"  Đã tải được {len(hmms)} tệp cấu hình HMM")

    pyhmmer.hmmer.hmmpress(hmms, str(hmm_path))
    print(f"  Đã press tệp cơ sở dữ liệu → {hmm_path}.h3{{m,i,f,p}}")


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị các cấu hình HMM phục vụ cho việc chú thích gen")
    parser.add_argument("--strategy", choices=["pfam", "vog", "combined"], default="pfam",
                        help="Nguồn HMM: 'pfam' (chính xác), 'vog' (toàn diện), "
                             "hoặc 'combined' (Pfam precision + VOG recall)")
    parser.add_argument("--output_dir", type=str, default="data/hmm/",
                        help="Thư mục ghi kết quả cho các cấu hình HMM")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lưu từ vựng
    save_vocabulary(output_dir)

    # Tải và chuẩn bị các cấu hình HMM
    if args.strategy == "pfam":
        prepare_pfam_profiles(output_dir)
    elif args.strategy == "vog":
        prepare_vog_profiles(output_dir)
    else:
        prepare_combined_profiles(output_dir)

    print("\nHoàn tất! Cơ sở dữ liệu HMM đã sẵn sàng sử dụng cho việc chú thích.")
    print(f"  Từ vựng: {output_dir / 'vocabulary.json'}")
    print(f"  CSDL HMM: {output_dir / 'gene_families.hmm'}")


if __name__ == "__main__":
    main()