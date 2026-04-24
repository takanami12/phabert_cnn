"""
Kiến trúc tổng thể của mô hình PhaBERT-CNN.

Khung học sâu lai ghép (hybrid deep learning framework), kết hợp giữa mô hình ngôn ngữ lõi (backbone) 
DNABERT-2 được huấn luyện trước (pre-trained) cùng mạng nơ-ron tích chập đa tầng (multi-scale CNN) 
và cơ chế chú ý tổng hợp (attention pooling) nhằm tối ưu hóa bài toán phân loại hình thái vòng đời thực khuẩn thể.

Cấu trúc phân lớp:
    1. Lõi DNABERT-2 (backbone): Trích xuất đặc trưng ngữ nghĩa không gian 768 chiều.
    2. Nhánh CNN đa kích thước (Multi-scale CNN): Gồm 3 nhánh Conv1d hoạt động song song với kích thước bộ lọc (kernel sizes) 3, 5, và 7.
    3. Nhánh tổng hợp cơ chế chú ý (Attention pooling): Trích xuất biểu diễn đặc trưng toàn cục (global sequence representation).
    4. Bộ phân loại tuyến tính (Classification head): Ánh xạ không gian 512 chiều sang không gian quyết định nhị phân.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from .attention import AttentionPooling


def _replace_flash_attn(bert_module) -> None:
    """
    Replace flash_attn_qkvpacked_func in bert_layers' module namespace with a
    standard PyTorch implementation.

    DNABERT-2 revision 7bce263b uses a custom Triton flash-attention kernel
    (flash_attn_triton.py) that relies on the tl.dot(a, b, trans_b=True) API
    removed in Triton 3.0.  Triton captures the kernel AST at @triton.jit
    decoration time (import), so patching the source file after import has no
    effect.  The correct fix is to replace the Python-level function reference
    that bert_layers.BertSelfAttention.forward looks up in its module globals.
    """

    def _std_attn(qkv, bias=None, softmax_scale=None, causal=False, **_kw):
        # qkv: (B, S, 3, H, D) — packed Q/K/V in DNABERT-2's padded format
        dtype = qkv.dtype
        B, S, _, H, D = qkv.shape
        q, k, v = qkv.unbind(dim=2)             # each (B, S, H, D)
        q = q.transpose(1, 2)                   # (B, H, S, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = softmax_scale or (D ** -0.5)
        if bias is None and not causal:
            out = F.scaled_dot_product_attention(q, k, v, scale=scale)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if bias is not None:
                scores = scores + bias
            if causal:
                mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=qkv.device), diagonal=1)
                scores.masked_fill_(mask, float('-inf'))
            out = torch.matmul(F.softmax(scores.float(), dim=-1).to(dtype), v)
        return out.transpose(1, 2).contiguous()  # (B, S, H, D)

    bert_module.flash_attn_qkvpacked_func = _std_attn


class MultiScaleCNNBranch(nn.Module):
    """
    Một nhánh độc lập của mạng CNN với kiến trúc tích chập 1 chiều (Conv1d) xếp chồng kép.

    Tiến trình xử lý tuần tự:
    Conv1d (768 chiều → 256 chiều, kernel_size=k) → Chuẩn hóa theo lô (Batch Normalization) → ReLU →
    Conv1d (256 chiều → 128 chiều, kernel_size=k) → Chuẩn hóa theo lô (Batch Normalization) → ReLU →
    Max Pooling thích ứng 1 chiều (AdaptiveMaxPool1d) triệt tiêu chiều không gian.
    """

    def __init__(self, in_channels: int, kernel_size: int, dropout: float = 0.2):
        super().__init__()

        padding1 = kernel_size // 2
        padding2 = kernel_size // 2

        # GroupNorm thay BatchNorm1d: BN running stats bị lệch vì train có
        # reverse-complement augmentation còn val thì không → train/val
        # distribution mismatch. GroupNorm không có running stats nên miễn
        # nhiễm vấn đề này.
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=kernel_size, padding=padding1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 128, kernel_size=kernel_size, padding=padding2),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tham số đầu vào (Args):
            x: Dạng (batch_size, embedding_dim, seq_len) — Ma trận đặc trưng DNABERT-2 đã trải qua hoán vị chiều (transposed).

        Dữ liệu trả về (Returns):
            out: Dạng (batch_size, 128) — Vector đặc trưng được trích lọc sau quá trình tinh giản (pooling).
        """
        out = self.conv_block(x)  # (B, 128, L')
        out = self.pool(out)       # (B, 128, 1)
        out = out.squeeze(-1)      # (B, 128)
        out = self.out_drop(out)
        return out


class PhaBERTCNN(nn.Module):
    """
    Khuôn khổ mô hình PhaBERT-CNN tích hợp DNABERT-2, mạng nơ-ron CNN đa tầng và cơ chế Attention Pooling.

    Các thành phần cốt lõi:
        1. Base DNABERT-2: Mô hình nền tảng phân tích hệ gen (foundation genome model) dưới dạng pre-train.
        2. CNN đa kích thước: Gồm 3 nhánh xử lý độc lập, khai thác đặc trưng cục bộ thông qua kernel 3, 5, và 7.
           Mỗi khối nhánh kết xuất vector đặc trưng 128 chiều → Cấu thành chuẩn 384 biểu diễn đa hướng.
        3. Attention pooling: Tổng hợp phân phối chú ý nhằm thiết lập đặc trưng tổng quát toàn chuỗi → 128 chiều.
        4. Module phân lớp (Classification head): Tối thiểu hóa không gian 512 chiều về xác suất phân bố nhị phân (Độc lực vs Ôn hòa / Virulent vs Temperate).
    """

    def __init__(
        self,
        dnabert2_model_name: str = "zhihan1996/DNABERT-2-117M",
        embedding_dim: int = 768,
        cnn_kernel_sizes: list = [3, 5, 7],
        attention_hidden_dim: int = 64,
        attention_dropout: float = 0.1,
        classifier_hidden_dim: int = 256,
        classifier_dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # ============================================================
        # Thành phần thứ nhất: Lõi cấu trúc (Backbone) DNABERT-2
        # ============================================================
        from transformers import BertConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config = BertConfig.from_pretrained(dnabert2_model_name)
        if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
            config.pad_token_id = 0

        # Vô hiệu hóa phân bổ flash attention để đảm bảo tính tương thích với Triton phiển bản thực thi hiện tại
        config.use_flash_attn = False

        model_cls = get_class_from_dynamic_module(
            "bert_layers.BertModel",
            dnabert2_model_name,
            trust_remote_code=True,
        )

        import inspect as _inspect
        _bert_module = _inspect.getmodule(model_cls)

        # Fix 1: skip ALiBi rebuild during from_pretrained (meta-device crash)
        _BertEncoder = _bert_module.BertEncoder
        _orig_rebuild = _BertEncoder.rebuild_alibi_tensor
        _BertEncoder.rebuild_alibi_tensor = lambda self, size, device=None: None

        # Fix 2: replace Triton flash-attn with standard PyTorch attention
        _replace_flash_attn(_bert_module)

        self.backbone = model_cls.from_pretrained(
            dnabert2_model_name,
            config=config,
            low_cpu_mem_usage=False,
        )

        _BertEncoder.rebuild_alibi_tensor = _orig_rebuild
        self.backbone.encoder.rebuild_alibi_tensor(size=config.alibi_starting_size)

        self.tokenizer = AutoTokenizer.from_pretrained(
            dnabert2_model_name,
            trust_remote_code=True,
        )

        # ============================================================
        # Thành phần thứ hai: Truyền dẫn đặc trưng CNN đa độ phân giải (Multi-scale CNN)
        # Bao gồm ba nhánh cấu trúc song song định hình kích thước kernel mở rộng 3, 5, 7
        # ============================================================
        self.cnn_branches = nn.ModuleList([
            MultiScaleCNNBranch(embedding_dim, ks) for ks in cnn_kernel_sizes
        ])
        cnn_output_dim = 128 * len(cnn_kernel_sizes)  # 128 * 3 = 384

        # ============================================================
        # Thành phần thứ ba: Cơ chế Gộp Chú ý (Attention Pooling) — Tổng hợp đặc trưng toàn cục
        # ============================================================
        self.attention_pooling = AttentionPooling(
            embedding_dim=embedding_dim,
            hidden_dim=attention_hidden_dim,
        )

        # Ánh xạ chiếu tuyến tính (Linear Projection): Ngưng tụ không gian 768 → 128 chiều kèm hàm kích hoạt ReLU và rdropout giới hạn quá ngưỡng
        self.global_projection = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(max(attention_dropout, 0.3)),
        )

        # ============================================================
        # Thành phần thứ tư: Cụm mạng phân loại quyết định (Classification head)
        # Khối đầu vào tích hợp: 384 (Đặc trưng CNN) + 128 (Đặc trưng Attention) = 512
        # ============================================================
        total_feature_dim = cnn_output_dim + 128  # 512

        self.classifier = nn.Sequential(
            nn.LayerNorm(total_feature_dim),
            nn.Dropout(0.2),
            nn.Linear(total_feature_dim, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        """
        Lan truyền hướng tiến (Forward pass) của cấu trúc PhaBERT-CNN.

        Tham số điều kiện (Args):
            input_ids:      (batch_size, seq_len) — Chuỗi vector mô hình hóa DNA (mã hóa chuẩn tokenize)
            attention_mask: (batch_size, seq_len) — Ma trận che (mask) bỏ qua nhiễu đệm

        Kết quả phân rã (Returns):
            logits: (batch_size, num_classes) — Logit không gian đại diện kết quả phân loại
        """
        # Giai đoạn 1: Nạp xuất chuỗi dữ liệu nhúng (embedding) qua khối trung tâm DNABERT-2
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # hidden_states: (B, L, 768)
        # Trích lọc tham biến ngõ ra: DNABERT-2 tuân thủ phản hồi kiểu cấu trúc tuple cấu hình hẹp thay vì ModelOutput nguyên bản
        if isinstance(backbone_outputs, tuple):
            hidden_states = backbone_outputs[0]
        else:
            hidden_states = backbone_outputs.last_hidden_state

        # Giai đoạn 2: Tiếp nhận thông tin không gian qua CNN đa bậc
        # Hoán vị tương thích lớp Conv1d: (B, L, 768) → (B, 768, L)
        cnn_input = hidden_states.transpose(1, 2)

        cnn_features = []
        for branch in self.cnn_branches:
            feat = branch(cnn_input)  # (B, 128)
            cnn_features.append(feat)

        # Ghép nối tuần tự các đặc trưng khu vực CNN: (B, 384)
        cnn_out = torch.cat(cnn_features, dim=-1)

        # Giai đoạn 3: Tính toán vector tổng thể nhờ gộp module chú ý (Attention pooling)
        context_vector, attn_weights = self.attention_pooling(
            hidden_states, attention_mask
        )
        # Giảm chiều tích hợp (Projection): 768 → 128
        global_out = self.global_projection(context_vector)  # (B, 128)

        # Giai đoạn 4: Hợp nhất song tuyến đặc tính: hình thành vector phân nhóm tổng quát (B, 512)
        combined = torch.cat([cnn_out, global_out], dim=-1)

        # Giai đoạn 5: Đưa vào đầu phân loại phi tuyến và hàm softmax ẩn (Classification)
        logits = self.classifier(combined)  # (B, 2)

        return logits

    def forward_head(
        self,
        h_trans: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Nút truyền thẳng rút gọn chuyên biệt (Head-only forward pass) dành riêng cho khối state đã vào cache sẵn.
        Áp dụng vào Giai đoạn xử lý thứ 1 (Phase 1 warmup) nhằm loại bỏ thao tác định tuyến trùng lặp qua DNABERT-2 trên từng epoch.

        Tham biến (Args):
            h_trans:        Dạng [B, L, 768] — Đặc trưng lưu trạm từ backbone (fp16 / fp32)
            attention_mask: Dạng [B, L]

        Trở lại (Returns):
            logits: Dạng [B, 2]
        """
        h = h_trans.float()
        cnn_input = h.transpose(1, 2)
        cnn_features = [branch(cnn_input) for branch in self.cnn_branches]
        cnn_out = torch.cat(cnn_features, dim=-1)
        context_vector, _ = self.attention_pooling(h, attention_mask)
        global_out = self.global_projection(context_vector)
        combined = torch.cat([cnn_out, global_out], dim=-1)
        return self.classifier(combined)

    def get_backbone_params(self):
        """Khởi xuất tập hợp tham số thuộc lớp module trung tâm (backbone DNABERT-2), thiết kế phục vụ tối ưu learning rate phân biệt (discriminative LR)."""
        return self.backbone.parameters()

    def get_task_params(self):
        """Truy vấn tham số tương tác tầng phân tích chuyên gia (task-specific bao gồm CNN, attention, classifier)."""
        task_modules = [
            self.cnn_branches,
            self.attention_pooling,
            self.global_projection,
            self.classifier,
        ]
        params = []
        for module in task_modules:
            params.extend(module.parameters())
        return params

    def freeze_backbone(self):
        """Khóa cứng không gian tham số trên bộ phận mô hình tiền huấn luyện (freeze backbone)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Mở khóa và khôi phục khả năng thay đổi đạo hàm (unfreeze) cho các cấp lớp DNABERT-2."""
        for param in self.backbone.parameters():
            param.requires_grad = True
