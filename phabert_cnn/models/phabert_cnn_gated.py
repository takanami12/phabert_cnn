"""
PhaBERT-CNN — Phiên bản v3 (Multi-modal Fusion).

So với phiên bản trước (global sigmoid gate + hand-crafted pathway buckets),
phiên bản này khắc phục các vấn đề kiến trúc đã được xác định:

  1. Global gate broadcast cùng vector cho mọi token → không có khả năng
     modulate per-position. Thay bằng **FiLM** (Feature-wise Linear Modulation)
     init = identity (gate=0, bias=0) để day-1 không vỡ phân phối backbone.

  2. PathwayScoreLayer hand-crafted bucket (max-pool tĩnh) → mất khả năng học,
     scale bit-score thô không norm. Thay bằng **LearnableFamilyAggregator**:
     family embeddings + Transformer encoder + attention pooling, có mask
     những family activation = 0.

  3. activation_vector thô (bit-scores HMM) đưa thẳng vào MLP gate → MLP thấy
     thang lệch giữa các family. Thay bằng **ActivationEncoder** (log1p +
     LayerNorm + MLP) → biểu diễn ổn định hơn, kết hợp cùng gene_stats thành
     một vector điều kiện chung `z_cond`.

  4. (Tùy chọn) **FamilyCrossAttention**: coi mỗi family là một token, DNA
     token query qua family token. Cung cấp tín hiệu gen ở mức per-position
     thay vì chỉ global. Init residual scale = 0 → identity day-1.

Khuôn khổ tổng quát:
  z_cond = ActivationEncoder(activation_vector, gene_stats)   # [B, d_cond]
  h0 = DNABERT-2(input_ids)                                   # [B, L, 768]
  h1 = FiLM(h0, z_cond)                                       # FiLM-modulated
  h2 = (FamilyCrossAttention(h1, activation_vector) if use_cross_attn else h1)
  cnn_out  = MultiScaleCNN(h2)                                # [B, 384]
  attn_out = AttentionPooling(h2) → projection                # [B, 128]
  fam_agg  = LearnableFamilyAggregator(activation_vector)     # [B, d_fam]
  combined = cat([cnn_out, attn_out, gene_stats_norm, fam_agg])
  logits = Classifier(combined)
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .phabert_cnn import MultiScaleCNNBranch, _replace_flash_attn
from .attention import AttentionPooling


# ============================================================
# Khối điều kiện (Conditioning blocks)
# ============================================================


class ActivationEncoder(nn.Module):
    """
    Encoder vector kích hoạt HMM + gene_stats → vector điều kiện chung.

    Pipeline:
        activation [B, N]
            → log1p (ổn định bit-score đuôi nặng)
            → LayerNorm per-family
            → concat với gene_stats (đã norm bên ngoài hoặc identity nếu None)
            → MLP (GELU + LayerNorm + Linear)
            → vector điều kiện [B, d_cond]
    """

    def __init__(self, n_families: int = 26, n_gene_stats: int = 4,
                 d_cond: int = 256, hidden_dim: int = 256,
                 use_gene_stats: bool = True):
        super().__init__()
        self.n_families = n_families
        self.n_gene_stats = n_gene_stats
        self.use_gene_stats = use_gene_stats

        self.act_norm = nn.LayerNorm(n_families)
        in_dim = n_families + (n_gene_stats if use_gene_stats else 0)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_cond)
        self.norm2 = nn.LayerNorm(d_cond)

    def forward(self, activation: Optional[torch.Tensor],
                gene_stats: Optional[torch.Tensor],
                batch_size: int, device, dtype) -> torch.Tensor:
        if activation is None:
            activation = torch.zeros(batch_size, self.n_families,
                                     device=device, dtype=dtype)
        # log1p ổn định cho bit-score (không âm sau khi normalize-max)
        z = torch.log1p(activation.clamp(min=0.0))
        z = self.act_norm(z)
        if self.use_gene_stats:
            if gene_stats is None:
                gene_stats = torch.zeros(batch_size, self.n_gene_stats,
                                         device=device, dtype=dtype)
            z = torch.cat([z, gene_stats], dim=-1)
        h = self.fc1(z)
        h = self.norm1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = self.norm2(h)
        return h  # [B, d_cond]


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (Perez et al. 2018).

        h_film = h * (1 + gamma(z)) + beta(z)

    Init: gamma.weight = gamma.bias = beta.weight = beta.bias = 0
        → gamma(z) = beta(z) = 0
        → h_film = h (identity day-1, KHÔNG vỡ phân phối backbone).

    So với global sigmoid gate cũ:
        - Per-token modulation (broadcast theo chiều L tự nhiên).
        - Có thêm bias term beta → modulate cả vị trí lẫn scale.
        - Init đúng identity (sigmoid init = 0.5 không phải 0).
    """

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_dim)
        self.beta = nn.Linear(cond_dim, feat_dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # h: [B, L, D], z: [B, C]
        # tanh-bound gamma/beta → modulation không vỡ distribution backbone
        # khi task_lr kéo weight khỏi 0 sau vài step.
        gamma = torch.tanh(self.gamma(z)).unsqueeze(1)  # [B, 1, D] in [-1, 1]
        beta = torch.tanh(self.beta(z)).unsqueeze(1)    # [B, 1, D] in [-1, 1]
        return h * (1.0 + gamma) + beta


class FamilyCrossAttention(nn.Module):
    """
    Cross-attention: DNA tokens (query) ⇄ family tokens (key/value).

    Mỗi family được biểu diễn như một token = embedding học được + scale theo
    bit-score activation. Family activation = 0 bị mask khỏi attention.

    Output:  h + scale * attn_out   với `scale` là parameter init = 0
        → identity day-1, tránh vỡ phân phối backbone trước khi học.
    """

    def __init__(self, n_families: int = 26, d_model: int = 768,
                 n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.n_families = n_families
        self.fam_emb = nn.Embedding(n_families, d_model)
        self.act_proj = nn.Linear(1, d_model)
        nn.init.zeros_(self.act_proj.weight)
        nn.init.zeros_(self.act_proj.bias)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        # Learnable residual scale init = 0 → identity day-1
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor,
                activation: Optional[torch.Tensor]) -> torch.Tensor:
        if activation is None:
            return h
        key_padding_mask = (activation.abs() < 1e-6)  # [B, N]
        # Sample nào toàn-zero → attention softmax trên toàn -inf = NaN.
        # Chỉ chạy attention cho các sample hợp lệ, giữ nguyên h cho phần còn lại.
        valid = ~key_padding_mask.all(dim=-1)  # [B]
        if not valid.any():
            return h
        h_valid = h[valid]
        act_valid = activation[valid]
        Bv = h_valid.size(0)
        fam_tokens = self.fam_emb.weight.unsqueeze(0).expand(Bv, -1, -1)
        fam_tokens = fam_tokens + self.act_proj(act_valid.unsqueeze(-1))
        kpm_valid = (act_valid.abs() < 1e-6)
        attn_out, _ = self.attn(
            query=h_valid, key=fam_tokens, value=fam_tokens,
            key_padding_mask=kpm_valid, need_weights=False,
        )
        out = h.clone()
        out[valid] = h_valid + self.residual_scale * attn_out
        return out


class CodonBranch(nn.Module):
    """
    Encoder cho codon usage features (RSCU 64-d log-transformed + GC3 z-scored).

    Output: vector [B, d_out] để concat tại classifier.

    Lý do: lytic vs temperate khác biệt về codon adaptation index (lytic có
    xu hướng dùng codon gần host hơn để replicate nhanh). Tín hiệu này hoàn
    toàn độc lập với HMM activation và DNA n-gram của DNABERT-2.
    """

    def __init__(self, codon_dim: int = 65, hidden_dim: int = 128,
                 d_out: int = 64, dropout: float = 0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(codon_dim)
        self.fc1 = nn.Linear(codon_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, d_out)
        self.out_norm = nn.LayerNorm(d_out)

    def forward(self, codon_features: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(codon_features)
        h = self.fc1(h)
        h = self.norm1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return self.out_norm(h)


class LearnableFamilyAggregator(nn.Module):
    """
    Bộ tổng hợp họ gen có khả năng học, thay thế PathwayScoreLayer (hand-crafted).

    Mỗi family được biểu diễn = (family embedding học được, augmented với
    activation magnitude). Một Transformer encoder 1-tầng cho phép các family
    "nói chuyện" với nhau (mô hình hóa pathway implicitly), sau đó attention
    pooling trả về một vector pathway-aware [B, d_out].

    Family với activation = 0 bị mask để tránh nhiễu vào pooling.
    """

    def __init__(self, n_families: int = 26, d_emb: int = 64, d_out: int = 32,
                 n_heads: int = 4, n_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.n_families = n_families
        self.fam_emb = nn.Embedding(n_families, d_emb)
        self.in_proj = nn.Linear(d_emb + 1, d_emb)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_emb, nhead=n_heads, dim_feedforward=d_emb * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.set_enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool_score = nn.Linear(d_emb, 1)
        self.out_proj = nn.Linear(d_emb, d_out)
        self.out_norm = nn.LayerNorm(d_out)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        B = activation.size(0)
        device, dtype = activation.device, activation.dtype
        mask = (activation.abs() < 1e-6)                          # [B, N]
        valid = ~mask.all(dim=-1)                                 # [B]
        # Output mặc định = 0 (sau out_norm, zero vector là identity hợp lệ).
        out_full = torch.zeros(
            B, self.out_proj.out_features, device=device, dtype=dtype,
        )
        if not valid.any():
            return out_full

        fam = self.fam_emb.weight.unsqueeze(0).expand(B, -1, -1)  # [B, N, d]
        fam_aug = torch.cat([fam, activation.unsqueeze(-1)], dim=-1)
        x = self.in_proj(fam_aug)                                 # [B, N, d]

        # Chạy encoder + pooling chỉ cho các sample hợp lệ để tránh NaN
        # softmax khi row mask toàn True.
        x_v = self.set_enc(x[valid], src_key_padding_mask=mask[valid])
        scores = self.pool_score(x_v).squeeze(-1)                 # [Bv, N]
        scores = scores.masked_fill(mask[valid], float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)     # [Bv, N, 1]
        agg = (weights * x_v).sum(dim=1)                          # [Bv, d]
        out_v = self.out_drop(self.out_norm(self.out_proj(agg)))
        out_full[valid] = out_v
        return out_full  # [B, d_out]


# ============================================================
# Mô hình chính
# ============================================================


class PhaBERTCNN_GeneGated(nn.Module):
    """
    PhaBERT-CNN multi-modal v3.

    Args (giữ tương thích cờ ablation cũ):
        use_gate:           Bật FiLM modulation (thay global sigmoid gate cũ).
        use_gene_stats:     Đưa gene_stats vào ActivationEncoder + concat classifier.
        use_pathway_scores: Bật LearnableFamilyAggregator (thay PathwayScoreLayer cũ).
        use_cross_attn:     (Mới) Bật FamilyCrossAttention DNA↔family.
    """

    def __init__(
        self,
        dnabert2_model_name: str = "zhihan1996/DNABERT-2-117M",
        n_families: int = 26,
        n_gene_stats: int = 4,
        codon_dim: int = 65,
        use_gate: bool = True,
        use_gene_stats: bool = True,
        use_pathway_scores: bool = True,
        use_cross_attn: bool = False,
        use_codon: bool = False,
        embedding_dim: int = 768,
        cnn_out_dim: int = 384,
        global_out_dim: int = 128,
        cond_dim: int = 256,
        family_agg_dim: int = 32,
        codon_out_dim: int = 64,
        cross_attn_heads: int = 4,
        num_classes: int = 2,
    ):
        super().__init__()
        self.use_gate = use_gate
        self.use_gene_stats = use_gene_stats
        self.use_pathway_scores = use_pathway_scores
        self.use_cross_attn = use_cross_attn
        self.use_codon = use_codon
        self.embedding_dim = embedding_dim
        self.n_families = n_families
        self.n_gene_stats = n_gene_stats
        self.codon_dim = codon_dim
        self.codon_out_dim = codon_out_dim

        # --- Backbone DNABERT-2 ---
        _revision = "7bce263b15377fc15361f52cfab88f8b586abda0"

        config = BertConfig.from_pretrained(
            dnabert2_model_name, revision=_revision,
        )
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = 0
        config.use_flash_attn = False

        model_cls = get_class_from_dynamic_module(
            "bert_layers.BertModel",
            dnabert2_model_name,
            revision=_revision,
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
            revision=_revision,
            low_cpu_mem_usage=False,
        )

        _BertEncoder.rebuild_alibi_tensor = _orig_rebuild
        self.backbone.encoder.rebuild_alibi_tensor(size=config.alibi_starting_size)

        # --- Conditioning encoder (activation + gene_stats → z_cond) ---
        # Cần khi bất kỳ một trong các injection (gate / cross-attn) bật.
        if self.use_gate or self.use_cross_attn:
            self.cond_encoder = ActivationEncoder(
                n_families=n_families,
                n_gene_stats=n_gene_stats,
                d_cond=cond_dim,
                hidden_dim=cond_dim,
                use_gene_stats=use_gene_stats,
            )

        # --- FiLM injection (thay global gate) ---
        if self.use_gate:
            self.film = FiLM(cond_dim=cond_dim, feat_dim=embedding_dim)

        # --- Family cross-attention (per-token gene injection, optional) ---
        if self.use_cross_attn:
            self.cross_attn = FamilyCrossAttention(
                n_families=n_families, d_model=embedding_dim,
                n_heads=cross_attn_heads,
            )

        # --- CNN multi-scale (kế thừa) ---
        self.cnn_branches = nn.ModuleList([
            MultiScaleCNNBranch(embedding_dim, kernel_size=k)
            for k in (3, 5, 7)
        ])

        # --- Attention pooling toàn cục (kế thừa) ---
        self.attention_pooling = AttentionPooling(embedding_dim)
        self.global_projection = nn.Sequential(
            nn.Linear(embedding_dim, global_out_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # --- Learnable family aggregator (thay PathwayScoreLayer) ---
        if self.use_pathway_scores:
            self.family_aggregator = LearnableFamilyAggregator(
                n_families=n_families, d_emb=64, d_out=family_agg_dim,
            )

        # --- Norm cho gene_stats trước concat classifier ---
        if self.use_gene_stats:
            self.gene_stats_norm = nn.LayerNorm(n_gene_stats)

        # --- Codon usage branch ---
        if self.use_codon:
            self.codon_branch = CodonBranch(
                codon_dim=codon_dim, hidden_dim=128, d_out=codon_out_dim,
            )

        # --- Classifier ---
        classifier_in = cnn_out_dim + global_out_dim                # 512
        if self.use_gene_stats:
            classifier_in += n_gene_stats                            # +4
        if self.use_pathway_scores:
            classifier_in += family_agg_dim                          # +32
        if self.use_codon:
            classifier_in += codon_out_dim                           # +64
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in),
            nn.Dropout(0.2),
            nn.Linear(classifier_in, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    # ============================================================
    # Trục xử lý chính
    # ============================================================

    def _condition(self, activation_vector: Optional[torch.Tensor],
                   gene_stats: Optional[torch.Tensor],
                   batch_size: int, device, dtype) -> Optional[torch.Tensor]:
        """Tạo vector điều kiện z_cond từ activation + gene_stats."""
        if not (self.use_gate or self.use_cross_attn):
            return None
        return self.cond_encoder(
            activation_vector, gene_stats, batch_size, device, dtype,
        )

    def _apply_injections(self, h_trans: torch.Tensor,
                          attention_mask: torch.Tensor,
                          activation_vector: Optional[torch.Tensor],
                          gene_stats: Optional[torch.Tensor]) -> torch.Tensor:
        """Áp FiLM + cross-attn lên h_trans."""
        B = h_trans.size(0)
        device, dtype = h_trans.device, h_trans.dtype
        z_cond = self._condition(
            activation_vector, gene_stats, B, device, dtype,
        )
        h = h_trans
        if self.use_gate:
            h = self.film(h, z_cond)
        if self.use_cross_attn:
            h = self.cross_attn(h, activation_vector)
        return h

    def _classify(self, h: torch.Tensor, attention_mask: torch.Tensor,
                  activation_vector: Optional[torch.Tensor],
                  gene_stats: Optional[torch.Tensor],
                  codon_features: Optional[torch.Tensor]) -> torch.Tensor:
        """Pipeline CNN + Attention pooling + family aggregator + codon + classifier."""
        # CNN branches
        cnn_input = h.transpose(1, 2)
        cnn_feats = [branch(cnn_input) for branch in self.cnn_branches]
        cnn_out = torch.cat(cnn_feats, dim=-1)                    # [B, 384]

        # Attention pooling
        context_vector, _ = self.attention_pooling(h, attention_mask)
        global_out = self.global_projection(context_vector)        # [B, 128]

        combined = torch.cat([cnn_out, global_out], dim=-1)        # [B, 512]
        B = combined.size(0)
        device, dtype = combined.device, combined.dtype

        # gene_stats (norm + concat)
        if self.use_gene_stats:
            stats = gene_stats if gene_stats is not None else torch.zeros(
                B, self.n_gene_stats, device=device, dtype=dtype,
            )
            stats = self.gene_stats_norm(stats)
            combined = torch.cat([combined, stats], dim=-1)

        # Learnable family aggregator
        if self.use_pathway_scores:
            act = activation_vector if activation_vector is not None else torch.zeros(
                B, self.n_families, device=device, dtype=dtype,
            )
            fam_agg = self.family_aggregator(act)
            combined = torch.cat([combined, fam_agg], dim=-1)

        # Codon branch
        if self.use_codon:
            if codon_features is None:
                # Default: RSCU = 0 sau log1p (= 1.0 RSCU thô uniform), GC3 = 0 sau norm
                codon_features = torch.zeros(
                    B, self.codon_dim, device=device, dtype=dtype,
                )
            codon_out = self.codon_branch(codon_features)
            combined = torch.cat([combined, codon_out], dim=-1)

        return self.classifier(combined)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        activation_vector: Optional[torch.Tensor] = None,
        gene_stats: Optional[torch.Tensor] = None,
        codon_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        backbone_outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        if isinstance(backbone_outputs, tuple):
            h_trans = backbone_outputs[0]
        else:
            h_trans = backbone_outputs.last_hidden_state          # [B, L, 768]

        h = self._apply_injections(
            h_trans, attention_mask, activation_vector, gene_stats,
        )
        return self._classify(h, attention_mask, activation_vector,
                              gene_stats, codon_features)

    def forward_head(
        self,
        h_trans: torch.Tensor,
        attention_mask: torch.Tensor,
        activation_vector: Optional[torch.Tensor] = None,
        gene_stats: Optional[torch.Tensor] = None,
        codon_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bypass backbone (cache mode); injection + head giống `forward`."""
        h_trans = h_trans.float()
        h = self._apply_injections(
            h_trans, attention_mask, activation_vector, gene_stats,
        )
        return self._classify(h, attention_mask, activation_vector,
                              gene_stats, codon_features)

    # ============================================================
    # Helpers cho discriminative LR
    # ============================================================

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_task_params(self):
        """Tham số học chuyên biệt task (ngoại trừ backbone)."""
        modules = [
            self.cnn_branches,
            self.attention_pooling,
            self.global_projection,
            self.classifier,
        ]
        if self.use_gate or self.use_cross_attn:
            modules.append(self.cond_encoder)
        if self.use_gate:
            modules.append(self.film)
        if self.use_cross_attn:
            modules.append(self.cross_attn)
        if self.use_pathway_scores:
            modules.append(self.family_aggregator)
        if self.use_gene_stats:
            modules.append(self.gene_stats_norm)
        if self.use_codon:
            modules.append(self.codon_branch)
        params = []
        for m in modules:
            params.extend(m.parameters())
        return params

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def extra_repr(self) -> str:
        return (f"use_gate={self.use_gate}, "
                f"use_gene_stats={self.use_gene_stats}, "
                f"use_pathway_scores={self.use_pathway_scores}, "
                f"use_cross_attn={self.use_cross_attn}, "
                f"use_codon={self.use_codon}, "
                f"n_families={self.n_families}")
