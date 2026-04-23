"""
utils/dataset.py
================
Cung cấp tiện ích đóng gói dữ liệu cấu trúc (Dataset) và bộ điều phối chuẩn (DataLoader),
tích hợp cơ chế hỗ trợ điều phối các đặc trưng di truyền (gene features).

Kiến trúc thay đổi so với biên bản thiết kế v1:
  - Khối `PhageContigDataset` tiếp nhận tham số truyền tùy chọn `activations` và `gene_stats`.
    Sau khi đáp ứng trạng thái đầu vào, hàm nội hàm `__getitem__` trích trả các key cấu trúc 'activation'
    và 'gene_stats', song hành với định dạng chuỗi token `input_ids/attention_mask/label`.
  - Khối định tuyến `create_dataloaders()` thực nhận địa chỉ hệ file `_features.pt` tương tác cho vòng lặp train và val.
  - Thuật toán tái biểu diễn tập mẫu (Undersampling) tuân thủ tính nhất quán đối chiếu index giữa bộ ba dữ liệu đầu vào sequences, labels và features.
"""

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from imblearn.under_sampling import RandomUnderSampler


class PhageContigDataset(Dataset):
    """
    Quy trình Tokenize tuyến tính (Mã hóa phân đoạn chuỗi DNA) thực thi đồng bộ trên toàn bộ tập tại thời gian khởi tạo.
    Mô-đun duy trì quyền bảo lưu biểu diễn tính chất gen (gene features) riêng biệt theo mẫu, căn đối định chỉ số chính xác tuyệt đối.
    """

    def __init__(
        self,
        sequences: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        activations: Optional[torch.Tensor] = None,
        gene_stats: Optional[torch.Tensor] = None,
        codon_features: Optional[torch.Tensor] = None,
    ):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.max_length = max_length
        self.has_features = activations is not None and gene_stats is not None
        self.has_codon = codon_features is not None

        if self.has_features:
            assert len(sequences) == activations.shape[0] == gene_stats.shape[0], \
                f"Kích thước không khớp: seqs={len(sequences)}, " \
                f"acts={activations.shape[0]}, stats={gene_stats.shape[0]}"
            self.activations = activations.float()
            self.gene_stats = gene_stats.float()

        if self.has_codon:
            assert len(sequences) == codon_features.shape[0], \
                f"Kích thước codon không khớp: seqs={len(sequences)}, " \
                f"codon={codon_features.shape[0]}"
            self.codon_features = codon_features.float()

        # Phân luồng mã hóa phân đoạn theo tổ hợp lô BATCH = 10000 chuỗi (tối ưu hóa băng thông so với cấu trúc lặp từng mẫu)
        print(f"    Trạng thái: Tiến hành mã hóa chuỗi token (Tokenizing) trên quy mô {len(sequences)} mảng...", end=" ", flush=True)
        BATCH = 10000
        all_ids, all_masks = [], []
        for i in range(0, len(sequences), BATCH):
            enc = tokenizer(
                sequences[i:i+BATCH],
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            all_ids.append(enc['input_ids'])
            all_masks.append(enc['attention_mask'])
        self.input_ids = torch.cat(all_ids, dim=0)
        self.attention_masks = torch.cat(all_masks, dim=0)
        print("Xong!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'label': self.labels[idx],
        }
        if self.has_features:
            item['activation'] = self.activations[idx]
            item['gene_stats'] = self.gene_stats[idx]
        if self.has_codon:
            item['codon_features'] = self.codon_features[idx]
        return item


def load_features(features_path: str,
                  normalize: bool = True
                  ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Kích hoạt quá trình tải các đặc trưng hệ gen (gene features) đã được tính toán tiền xử lý từ các định dạng lưu điện tử _features.pt.

    Returns:
        (activations [N, 26], gene_stats [N, 4], codon_features [N, 65] | None)
        codon_features = None nếu file features cũ chưa có trường này
        (backward-compat — train pipeline sẽ tự fallback).
    """
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file features: {features_path}")

    data = torch.load(str(features_path), weights_only=False)
    activations = data["activations"].float()   # [N, 26]
    gene_stats = data["gene_stats"].float()     # [N, 4]
    codon_features = data.get("codon_features")
    if codon_features is not None:
        codon_features = codon_features.float()  # [N, 65]

    if normalize:
        # Z-score gene_stats
        mean = gene_stats.mean(dim=0)
        std = gene_stats.std(dim=0).clamp(min=1e-6)
        gene_stats = (gene_stats - mean) / std

        # Max-scale activations (bit-scores HMM biến thiên rộng giữa families)
        act_max = activations.max(dim=0).values.clamp(min=1e-6)
        activations = activations / act_max

        # Codon features: log-transform RSCU + z-score GC3
        # RSCU > 0, đuôi nặng → log1p ổn định
        if codon_features is not None:
            rscu = torch.log1p(codon_features[:, :64])
            gc3 = codon_features[:, 64:65]
            gc3_mean = gc3.mean(dim=0)
            gc3_std = gc3.std(dim=0).clamp(min=1e-6)
            gc3 = (gc3 - gc3_mean) / gc3_std
            codon_features = torch.cat([rscu, gc3], dim=-1)

    return activations, gene_stats, codon_features


def apply_undersampling(
    sequences: List[str],
    labels: List[int],
    activations: Optional[torch.Tensor] = None,
    gene_stats: Optional[torch.Tensor] = None,
    codon_features: Optional[torch.Tensor] = None,
    random_state: int = 42,
):
    """Random undersampling đồng bộ chỉ số giữa sequences/labels/features."""
    X = np.arange(len(sequences)).reshape(-1, 1)
    y = np.array(labels)
    rus = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    indices = X_resampled.flatten()

    new_seqs = [sequences[i] for i in indices]
    new_labels = y_resampled.tolist()

    if activations is not None and gene_stats is not None:
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        new_acts = activations[idx_tensor]
        new_stats = gene_stats[idx_tensor]
        new_codon = codon_features[idx_tensor] if codon_features is not None else None
        return new_seqs, new_labels, new_acts, new_stats, new_codon

    return new_seqs, new_labels


def create_dataloaders(
    train_seqs, train_labels, val_seqs, val_labels,
    tokenizer, batch_size=128, max_length=512,
    num_workers=4, use_undersampling=True, random_state=42,
    train_features_path: Optional[str] = None,
    val_features_path: Optional[str] = None,
):
    """
    Triển khai cấu trúc DataLoader dành riêng biệt cho tập huấn luyện (train) và nhóm đối chuẩn xác thực (val), kết hợp quyền tải
    yếu tố features hệ gen nội suy từ các nhánh lưu đối chiếu _features.pt gốc.
    """
    use_features = train_features_path is not None and val_features_path is not None

    # Trích xuất module bộ điều hướng Features
    train_acts = train_stats = val_acts = val_stats = None
    train_codon = val_codon = None
    if use_features:
        print(f"  Thực thi quá trình chép tải đặc trưng khối huấn luyện (training features) từ định vị trung chuyển: {train_features_path}")
        train_acts, train_stats, train_codon = load_features(train_features_path)
        print(f"  Thực thi quá trình chép tải đặc trưng mảng đối chiếu (validation features) từ định vị: {val_features_path}")
        val_acts, val_stats, val_codon = load_features(val_features_path)
        if train_codon is None:
            print("  [!] Codon features không có trong file features (file cũ). "
                  "Bỏ qua codon branch — chạy lại preprocess_gene_features.py để bật.")

    # Chuyển đổi trạng thái dữ liệu rút gọn theo mẫu thử dưới mức thiểu số (Undersampling, độc quyền dành cho nhóm đối chứng train)
    if use_undersampling:
        if use_features:
            train_seqs, train_labels, train_acts, train_stats, train_codon = apply_undersampling(
                train_seqs, train_labels, train_acts, train_stats,
                codon_features=train_codon, random_state=random_state,
            )
        else:
            train_seqs, train_labels = apply_undersampling(
                train_seqs, train_labels, random_state=random_state,
            )
        print(f"  Bảng tổng kết chuẩn Undersampling: Thu được {len(train_seqs)} tập luyện phân bổ")
        unique, counts = np.unique(train_labels, return_counts=True)
        for u, c in zip(unique, counts):
            name = "Trạng thái ôn hoà" if u == 0 else "Phân lớp độc lực"
            print(f"    {name}: {c}")

    # Tập hợp đối tượng dataset mỏ neo chuẩn (Dataset Constructor)
    train_dataset = PhageContigDataset(
        train_seqs, train_labels, tokenizer, max_length,
        activations=train_acts, gene_stats=train_stats,
        codon_features=train_codon,
    )
    val_dataset = PhageContigDataset(
        val_seqs, val_labels, tokenizer, max_length,
        activations=val_acts, gene_stats=val_stats,
        codon_features=val_codon,
    )

    _extra = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=False, **_extra,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        **_extra,
    )
    return train_loader, val_loader
