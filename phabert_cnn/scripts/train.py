import os
import sys
import time
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phabert_cnn import PhaBERTCNN
from models.phabert_cnn_gated import PhaBERTCNN_GeneGated
from utils.dataset import create_dataloaders
from utils.metrics import compute_metrics, print_metrics


# ================================================================
# Phân tích cú pháp cấu hình đầu vào (Argument Parsing)
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình PhaBERT-CNN (cấu trúc cơ sở baseline hoặc kiến trúc cổng gen gene-gated)")

    # --- Dữ liệu (Data) ---
    parser.add_argument("--group", type=str, required=True, choices=['A', 'B', 'C', 'D'])
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results")

    # --- Tham số Huấn luyện Chung (General Training) ---
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="zhihan1996/DNABERT-2-117M")

    # --- Kiến trúc mô hình Gene-Gated ---
    parser.add_argument("--gated", action="store_true",
                        help="Dùng PhaBERTCNN_GeneGated với gene features")
    parser.add_argument("--no_gate", action="store_true",
                        help="[Ablation] tắt gene gate")
    parser.add_argument("--no_gene_stats", action="store_true",
                        help="[Ablation] tắt gene_stats concat")
    parser.add_argument("--no_pathway_scores", action="store_true",
                        help="[Ablation] tắt pathway scores")
    parser.add_argument("--n_families", type=int, default=26)

    # --- Giai đoạn 1 (Phase 1): Tối ưu hóa sơ bộ (Warm-up) ---
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--warmup_lr", type=float, default=2e-3)
    parser.add_argument("--warmup_wd", type=float, default=1e-4)

    # --- Giai đoạn 2 (Phase 2): Tinh chỉnh (Fine-tuning) ---
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--backbone_wd", type=float, default=1e-5)
    parser.add_argument("--task_lr", type=float, default=1e-4)
    parser.add_argument("--task_wd", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)

    # --- Chiến lược tối ưu hóa và tăng tốc (Acceleration / Optimization) ---
    parser.add_argument("--lora", action="store_true",
                        help="Phase 2: chỉ train LoRA adapters (cần thư viện peft)")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="Rank LoRA (mặc định 8)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="Hệ số scaling alpha của LoRA (mặc định 16)")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", type=str, default="Wqkv",
                        help="Tên module attention cho LoRA, phân cách bởi dấu phẩy "
                             "(DNABERT-2 dùng 'Wqkv' — combined QKV projection)")
    parser.add_argument("--cache_embeddings", action="store_true",
                        help="Phase 1: cache hidden states backbone xuống disk. "
                             "Chỉ có lợi khi --warmup_epochs >= 2.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Thư mục lưu cache embeddings (mặc định: <run_dir>/.cache)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile mô hình (~25%% nhanh hơn trên GPU hiện đại)")

    return parser.parse_args()


# ================================================================
# Các hàm bổ trợ tăng tốc (Acceleration Helpers)
# ================================================================

def setup_lora(model, r: int, alpha: int, dropout: float,
               target_modules: list) -> nn.Module:
    """
    Bao bọc (Wrap) mô hình ngôn ngữ lõi (backbone) DNABERT-2 bằng các khối tinh chỉnh (adapters) LoRA thông qua thư viện PEFT.

    Khởi tạo này đảm bảo chỉ những ma trận thay đổi (delta matrices) của quy trình LoRA (chiếm khoảng 1-2% tổng lượng tham số module lõi)
    kết nạp gradient đạo hàm ở Giai đoạn 2, giúp cắt giảm xấp xỉ 5 lần thời gian của truyền ngược (backward pass).
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError(
            "Cần thư viện peft cho --lora.  Cài với:  pip install peft"
        )

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    model.backbone = get_peft_model(model.backbone, config)

    n_lora  = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.backbone.parameters())
    print(f"  LoRA adapters trên {target_modules}  r={r}, α={alpha}")
    print(f"  Backbone trainable: {n_lora:,} / {n_total:,}  "
          f"({100 * n_lora / n_total:.2f}%)")
    return model


class CachedEmbeddingDataset(Dataset):
    """
    Kế thừa Dataset, cho phép cung cấp trực tiếp các véc-tơ trạng thái ẩn (hidden states) đã được trích xuất 
    từ mô hình DNABERT-2 thay vì chuỗi khóa (token IDs) thô. Áp dụng chuyên biệt cho cơ chế bộ nhớ đệm (cache) tại Giai đoạn 1,
    giúp vô hiệu hóa hoàn toàn việc lặp lại tính toán qua mô hình lõi trên mỗi kỷ nguyên.
    """

    def __init__(self, cache: dict):
        self.h_trans    = cache['h_trans']           # [N, L, 768] fp16
        self.masks      = cache['attention_mask']    # [N, L]
        self.labels     = cache['labels']            # [N]
        self.activation = cache.get('activation')    # [N, n_families] hoặc None
        self.gene_stats = cache.get('gene_stats')    # [N, 4] hoặc None
        self.has_features = self.activation is not None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'h_trans':        self.h_trans[idx].float(),
            'attention_mask': self.masks[idx],
            'label':          self.labels[idx],
        }
        if self.has_features:
            item['activation'] = self.activation[idx]
            item['gene_stats'] = self.gene_stats[idx]
        return item


def extract_backbone_cache(
    model: nn.Module,
    loader: DataLoader,
    cache_path: Path,
    device: torch.device,
    gated: bool,
) -> dict:
    """
    Thực hiện truyền tính toán (forward pass) một lần duy nhất qua mô hình DNABERT-2 để cô đọng (extract)
    các trạng thái ẩn (hidden states) của toàn bộ các mẫu huấn luyện.

    Cấu trúc bản ghi lưu chuyển vào <cache_path> định dạng biểu diễn dictionary tensor bao gồm:
        h_trans        [N, L, 768] cấu hình fp16  — Đặc trưng chuỗi thu được (output) từ mô hình lõi
        attention_mask [N, L]
        labels         [N]
        activation     [N, n_families]   (Chỉ khả dụng khi bật tính năng gated)
        gene_stats     [N, 4]            (Chỉ khả dụng khi bật tính năng gated)

    Thước đo ước lượng bộ nhớ: N=60k, L=512, D=768 → Chiếm dụng ~47 GB lưu trữ fp16.
    Có thể cấu hình thiết giảm tham số --max_seq_length 256 đối với cụm contig ngắn để bảo lưu 50% rủi ro bộ nhớ.
    """
    print(f"\n  [Cache] Đang trích xuất hidden states backbone → {cache_path}")
    model.eval()

    all_h, all_mask, all_labels = [], [], []
    all_act, all_stats = ([], []) if gated else (None, None)

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Đang trích xuất"):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            out  = model.backbone(input_ids=ids, attention_mask=mask)
            h    = out[0] if isinstance(out, tuple) else out.last_hidden_state

            all_h.append(h.cpu().half())           # fp16 để giảm bộ nhớ
            all_mask.append(batch['attention_mask'].cpu())
            all_labels.append(batch['label'].cpu())
            if gated:
                all_act.append(batch['activation'].cpu())
                all_stats.append(batch['gene_stats'].cpu())

    cache = {
        'h_trans':        torch.cat(all_h, 0),
        'attention_mask': torch.cat(all_mask, 0),
        'labels':         torch.cat(all_labels, 0),
    }
    if gated:
        cache['activation'] = torch.cat(all_act, 0)
        cache['gene_stats']  = torch.cat(all_stats, 0)

    size_gb = cache['h_trans'].nelement() * 2 / 1e9
    print(f"  [Cache] {len(cache['labels']):,} mẫu  "
          f"h_trans = {size_gb:.1f} GB (fp16)")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"  [Cache] Đã lưu → {cache_path}")
    return cache


# ================================================================
# Các hàm đa dụng (Utilities)
# ================================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _forward(model: nn.Module, batch: dict, device: torch.device, gated: bool):
    """
    Khối truyền thẳng (forward pass) được hợp nhất cho cả hai điều kiện môi trường.

    Xử lý linh động xuyên suốt cấu trúc logic:
      • Chế độ tiêu chuẩn (Standard mode) — Mỗi lô (batch) chứa 'input_ids'; tiến hành lan truyền toàn bộ mô hình (backbone + head).
      • Chế độ bộ nhớ đệm (Cache mode)  — Lô chứa 'h_trans'; bỏ qua mô hình lõi (backbone), chỉ thực thi lớp phân loại học tăng cường (head).
    """
    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
    labels         = batch['label'].to(device, non_blocking=True)

    if 'h_trans' in batch:
        # ---- Chế độ cache ----
        h = batch['h_trans'].to(device, non_blocking=True)
        if gated:
            activation = batch['activation'].to(device, non_blocking=True)
            gene_stats = batch['gene_stats'].to(device, non_blocking=True)
            logits = model.forward_head(h, attention_mask, activation, gene_stats)
        else:
            logits = model.forward_head(h, attention_mask)
    else:
        # ---- Chế độ thường ----
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        if gated:
            activation = batch['activation'].to(device, non_blocking=True)
            gene_stats = batch['gene_stats'].to(device, non_blocking=True)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                activation_vector=activation,
                gene_stats=gene_stats,
            )
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

    return logits, labels


# ================================================================
# Vòng lặp tối ưu và Đánh giá hiệu chuẩn (Training / Evaluation Loops)
# ================================================================

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device,
                    gated, scaler=None):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    use_amp = scaler is not None

    for batch in tqdm(loader, desc="Huấn luyện"):
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits, labels = _forward(model, batch, device, gated)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, labels = _forward(model, batch, device, gated)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, compute_metrics(all_labels, all_preds)


@torch.no_grad()
def evaluate(model, loader, criterion, device, gated):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="Validation"):
        logits, labels = _forward(model, batch, device, gated)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / len(loader.dataset), compute_metrics(all_labels, all_preds)


# ================================================================
# Quá trình cốt lõi (Main execution block)
# ================================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Thiết bị: {device}")
    print(f"Chế độ: {'GENE-GATED' if args.gated else 'BASELINE'}")

    # ---------- Tải dữ liệu pkl ----------
    fold_dir = Path(args.data_dir) / f"group_{args.group}" / f"fold_{args.fold}"
    print(f"Đang tải dữ liệu từ: {fold_dir}")
    with open(fold_dir / "train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(fold_dir / "val.pkl", "rb") as f:
        val_data = pickle.load(f)

    train_features = str(fold_dir / "train_features.pt") if args.gated else None
    val_features   = str(fold_dir / "val_features.pt")   if args.gated else None

    if args.gated:
        for p in (train_features, val_features):
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"Không tìm thấy file features: {p}\n"
                    f"Hãy chạy prepare_data.py trước (với gene annotations)."
                )

    # ---------- Tokenizer ----------
    print(f"Đang tải tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # ---------- DataLoaders ----------
    train_loader, val_loader = create_dataloaders(
        train_data["sequences"], train_data["labels"],
        val_data["sequences"],   val_data["labels"],
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
        num_workers=args.num_workers,
        use_undersampling=True,
        random_state=args.seed,
        train_features_path=train_features,
        val_features_path=val_features,
    )

    # ---------- Khởi tạo mô hình ----------
    print("Đang khởi tạo mô hình...")
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
    model = model.to(device)
    print(f"  Tổng số tham số: {sum(p.numel() for p in model.parameters()):,}")

    # Tùy chọn: torch.compile (nhanh hơn 20-30% sau warmup)
    if args.compile:
        print("  Đã bật torch.compile (batch đầu tiên cần ~2 phút để compile)")
        model = torch.compile(model, mode="reduce-overhead")

    criterion = nn.CrossEntropyLoss()
    use_amp   = device.type == "cuda"
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("  Đã bật Mixed Precision (AMP)")

    # ---------- Đường dẫn đầu ra ----------
    mode_suffix = "gated" if args.gated else "baseline"
    if args.gated:
        if args.no_gate:       mode_suffix += "_nogate"
        if args.no_gene_stats: mode_suffix += "_nostats"
        if args.lora:          mode_suffix += "_lora"
    run_dir = (Path(args.output_dir) / f"group_{args.group}"
               / f"fold_{args.fold}_{mode_suffix}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Thư mục đầu ra: {run_dir}")

    training_log = {"args": vars(args), "phases": []}

    # ================================================================
    # Phase 1: Warm-up (đóng băng backbone)
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 1: Warm-up (Đóng băng DNABERT-2)")
    print("=" * 60)

    model.freeze_backbone()
    n_train_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Tham số trainable (Phase 1): {n_train_p1:,}")

    optimizer_p1 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.warmup_lr, weight_decay=args.warmup_wd, betas=(0.9, 0.999),
    )
    total_steps_p1 = args.warmup_epochs * len(train_loader)
    scheduler_p1   = OneCycleLR(
        optimizer_p1, max_lr=args.warmup_lr, total_steps=total_steps_p1,
        pct_start=0.3, div_factor=5, final_div_factor=10,
    )

    warmup_skip_path = run_dir / "after_warmup.pt"
    if warmup_skip_path.exists():
        print("  [Bỏ qua] Đang tải checkpoint warmup, bỏ qua Phase 1...")
        ckpt = torch.load(warmup_skip_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # --- Tùy chọn: cache hidden states backbone để Phase 1 nhanh hơn ---
        p1_train_loader = train_loader  # mặc định: dùng loader thông thường

        if args.cache_embeddings:
            if args.warmup_epochs < 2:
                print("  [Cache] CẢNH BÁO: --warmup_epochs=1, cache không mang lại lợi ích "
                      "(thêm --warmup_epochs 3 để có hiệu quả).")
            cache_root = Path(args.cache_dir) if args.cache_dir else run_dir / ".cache"
            cache_path = cache_root / "phase1_train_cache.pt"

            if cache_path.exists():
                print(f"  [Cache] Đang tải cache từ {cache_path}")
                cache = torch.load(cache_path, map_location='cpu', weights_only=False)
            else:
                cache = extract_backbone_cache(
                    model, train_loader, cache_path, device, gated=args.gated,
                )

            cached_ds = CachedEmbeddingDataset(cache)
            p1_train_loader = DataLoader(
                cached_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=(2 if args.num_workers > 0 else None),
            )
            # Điều chỉnh scheduler theo độ dài loader mới
            total_steps_p1 = args.warmup_epochs * len(p1_train_loader)
            scheduler_p1   = OneCycleLR(
                optimizer_p1, max_lr=args.warmup_lr, total_steps=total_steps_p1,
                pct_start=0.3, div_factor=5, final_div_factor=10,
            )
            print(f"  [Cache] Phase 1 sẽ dùng embeddings đã cache "
                  f"({len(cached_ds):,} mẫu, {len(p1_train_loader)} batch/epoch)")

        for epoch in range(args.warmup_epochs):
            print(f"\n--- Warm-up Epoch {epoch+1}/{args.warmup_epochs} ---")
            train_loss, train_metrics = train_one_epoch(
                model, p1_train_loader, optimizer_p1, scheduler_p1, criterion,
                device, gated=args.gated, scaler=scaler,
            )
            val_loss, val_metrics = evaluate(
                model, val_loader, criterion, device, gated=args.gated,
            )
            print(f"  Train Loss: {train_loss:.4f}")
            print_metrics(train_metrics, prefix="  Train ")
            print(f"  Val Loss:   {val_loss:.4f}")
            print_metrics(val_metrics, prefix="  Val ")
            training_log["phases"].append({
                "phase": 1, "epoch": epoch + 1,
                "train_loss": train_loss, "train_metrics": train_metrics,
                "val_loss": val_loss, "val_metrics": val_metrics,
            })

        torch.save({"model_state_dict": model.state_dict()}, warmup_skip_path)

    # ================================================================
    # Phase 2: Fine-tuning (discriminative LR, tùy chọn LoRA)
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Fine-tuning Toàn Bộ (Discriminative LR)")
    if args.lora:
        print("  [LoRA] Chỉ train LoRA adapters (backbone còn lại bị đóng băng)")
    print("=" * 60)

    if args.lora:
        # setup_lora đặt tham số non-LoRA của backbone thành requires_grad=False
        # và tham số LoRA thành requires_grad=True.  KHÔNG gọi unfreeze_backbone().
        targets = [t.strip() for t in args.lora_targets.split(",")]
        model   = setup_lora(model, args.lora_r, args.lora_alpha,
                             args.lora_dropout, targets)
        backbone_params = [p for p in model.backbone.parameters()
                           if p.requires_grad]
    else:
        model.unfreeze_backbone()
        backbone_params = list(model.get_backbone_params())

    n_train_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Tham số trainable (Phase 2): {n_train_p2:,}")

    optimizer_p2 = AdamW([
        {"params": backbone_params,
         "lr": args.backbone_lr, "weight_decay": args.backbone_wd},
        {"params": model.get_task_params(),
         "lr": args.task_lr, "weight_decay": args.task_wd},
    ], betas=(0.9, 0.999), eps=1e-6)

    total_steps_p2 = args.finetune_epochs * len(train_loader)
    scheduler_p2   = OneCycleLR(
        optimizer_p2,
        max_lr=[args.backbone_lr, args.task_lr],
        total_steps=total_steps_p2,
        pct_start=0.1, div_factor=5, final_div_factor=50,
    )

    best_val_acc    = 0.0
    epochs_no_improve = 0
    best_epoch      = 0

    for epoch in range(args.finetune_epochs):
        print(f"\n--- Fine-tune Epoch {epoch+1}/{args.finetune_epochs} ---")
        t_start = time.time()

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer_p2, scheduler_p2, criterion,
            device, gated=args.gated, scaler=scaler,
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, gated=args.gated,
        )

        elapsed = time.time() - t_start
        print(f"  Thời gian epoch: {elapsed:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print_metrics(train_metrics, prefix="  Train ")
        print(f"  Val Loss:   {val_loss:.4f}")
        print_metrics(val_metrics, prefix="  Val ")

        training_log["phases"].append({
            "phase": 2, "epoch": epoch + 1,
            "train_loss": train_loss, "train_metrics": train_metrics,
            "val_loss": val_loss, "val_metrics": val_metrics,
            "time": elapsed,
        })

        val_acc = val_metrics["accuracy"]
        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            best_epoch        = epoch + 1
            epochs_no_improve = 0

            # Với LoRA: merge adapters trước khi lưu để checkpoint có thể
            # load được mà không cần peft (state dict sạch cho inference).
            if args.lora and hasattr(model.backbone, 'merge_and_unload'):
                merged          = model.backbone.merge_and_unload()
                _orig_backbone  = model.backbone
                model.backbone  = merged
                state_dict = {
                    k.replace("_orig_mod.", ""): v
                    for k, v in model.state_dict().items()
                }
                model.backbone  = _orig_backbone   # khôi phục adapters để tiếp tục train
            else:
                state_dict = {
                    k.replace("_orig_mod.", ""): v
                    for k, v in model.state_dict().items()
                }

            torch.save({
                "model_state_dict": state_dict,
                "epoch":            epoch + 1,
                "val_metrics":      val_metrics,
                "args":             vars(args),
            }, run_dir / "best_model.pt")
            print(f"  ✓ Val accuracy tốt nhất mới: {val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"  Không cải thiện ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print(f"  Early stopping tại epoch {epoch+1}")
                break

    # ---------- Lưu log huấn luyện ----------
    training_log["best_val_acc"] = best_val_acc
    training_log["best_epoch"]   = best_epoch
    with open(run_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"Huấn luyện hoàn tất. Val accuracy tốt nhất: {best_val_acc:.2f}% "
          f"tại epoch {best_epoch}")
    print(f"Checkpoint: {run_dir / 'best_model.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
