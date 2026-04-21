from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .phabert_cnn import MultiScaleCNNBranch
from .attention import AttentionPooling


class PathwayScoreLayer(nn.Module):
    """
    Tính toán chỉ số định hướng con đường sinh học (biological pathway scores) dựa trên vector đặc trưng hoạt hóa (activation) nhóm gen.

    Lược đồ phân rã con đường sinh lý (pathway index → mô tả danh mục):
        0  exclusive_lysogenic  [0, 1, 2, 8, 25]   Mã hóa nhóm men integrase/excisionase/imm (đặc trưng dung giải độc quyền)
        1  lysogeny_regulatory  [3, 24]            Nhóm yếu tố kìm hãm CI repressor (yêu cầu đồng thời hai tín hiệu hit)
        2  plasmid_lysogenic    [6, 7]             Hệ thống phân vùng (ParA/ParB)
        3  lytic_regulatory     [4, 9]             Yếu tố điều hòa lytic (Cro / antirepressor)
        4  shared_lysis         [12, 13, 14, 15]   Phức hợp dung giải (holin/endolysin/spanin)
        5  shared_structural    [16..23]           Cấu trúc phân tử (virion assembly gene)
    """

    N_PATHWAYS = 6

    _PATHWAY_DEF = [
        ("exclusive_lysogenic", [0, 1, 2, 8, 25]),
        ("lysogeny_regulatory", [3, 24]),
        ("plasmid_lysogenic",   [6, 7]),
        ("lytic_regulatory",    [4, 9]),
        ("shared_lysis",        [12, 13, 14, 15]),
        ("shared_structural",   [16, 17, 18, 19, 20, 21, 22, 23]),
    ]

    def __init__(self, n_families: int = 26):
        super().__init__()
        for name, indices in self._PATHWAY_DEF:
            valid = [i for i in indices if i < n_families]
            self.register_buffer(
                f"_idx_{name}",
                torch.tensor(valid, dtype=torch.long),
            )

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Tham số đầu vào (Args):
            activation: [B, N_FAMILIES] — Điểm kiểm tra dạng chu Bit (HMM bit-scores) đã qua bước chuẩn hóa chuẩn.

        Cấu trúc trả về (Returns):
            pathway_scores: [B, 6] — Ma trận trọng số tương quan theo 6 con đường sinh học.
        """
        scores = []
        for name, _ in self._PATHWAY_DEF:
            idx = getattr(self, f"_idx_{name}")
            if idx.numel() > 0:
                score = activation[:, idx].max(dim=1).values
            else:
                score = torch.zeros(
                    activation.size(0),
                    device=activation.device,
                    dtype=activation.dtype,
                )
            scores.append(score)
        return torch.stack(scores, dim=1)   # [B, 6]


class GeneGateMLP(nn.Module):
    """
    Khối tính toán phi tuyến tính thực hiện phép nội suy từ vector đặc trưng gen (activation vector) thành vector kiểm soát ngưỡng (gate vector) theo từng chiều cụ thể.

    Định dạng dữ liệu:
        Đầu vào:  activation_vector [B, N_FAMILIES]
        Đầu ra:   gate vector       [B, 768] (Bị giới hạn không gian miền giá trị [0, 1] qua hàm Sigmoid)

    Quy hoạch kiến trúc: Theo nguyên lý cổ chai (bottleneck design) nhằm cực tiểu hóa tổng lượng tham số học máy.
        N_FAMILIES → Điểm chạm nội suy 128 → Ngõ ra 768
    """

    def __init__(self, n_families: int = 24, hidden_dim: int = 128,
                 output_dim: int = 768):
        super().__init__()
        self.fc1 = nn.Linear(n_families, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Giai đoạn thiết lập chuẩn định hình trọng số lớp fc2 tiệm cận zero (zero-centered init)
        # Giúp bộ phân phối xác suất kiểm soát (gate distribution) tiệm cận gốc 0.
        # Hệ quả: Hệ thống hội tụ với đặc tính (h_gated ≈ h_trans) tại chu trình huấn luyện ban đầu.
        nn.init.zeros_(self.fc2.bias)

    def forward(self, activation_vector: torch.Tensor) -> torch.Tensor:
        """
        Tham số (Args):
            activation_vector: [B, N_FAMILIES]

        Kết quả (Returns):
            gate: Ngõ ra cấp tín hiệu gating [B, 768] biên độ ngưỡng chuẩn hóa đoạn [0, 1]
        """
        h = self.fc1(activation_vector)
        h = self.layer_norm(h)
        h = self.activation(h)
        h = self.fc2(h)
        return torch.sigmoid(h)


class PhaBERTCNN_GeneGated(nn.Module):
    """
    Kiến trúc mở rộng PhaBERT-CNN tích hợp cổng điều hướng tín hiệu đặc trưng gen (gene gating), số liệu cấu trúc thống kê (gene statistics) và đánh giá lộ trình chức năng sinh học (pathway scores).

    Khuôn khổ hoạt động:
        1. Base mô hình lõi DNABERT-2         → h_trans [B, L, 768]
        2. Kênh tín hiệu cổng gen (chuẩn tiêm dữ liệu thặng dư - residual injection):
               g = GeneGateMLP(activation_vector)
               h_gated = h_trans + h_trans * g.unsqueeze(1)
        3. CNN đa tầng hoạt động trên h_gated → cnn_out [B, 384]
        4. Tích hợp chú ý (Attention) từ h_gated→ global_out [B, 128]
        5. Phối hợp yếu tố gene_stats         → combined [B, 516]
        6. Trọng số lộ trình (không khả huấn) → combined [B, 522]
        7. Không gian quyết định (Classifier) → logits [B, 2]

    Cấu hình đầu vào quy trình phân lớp cuối:
        384 (Đặc trưng định hướng CNN) + 128 (Chú ý Toàn cục - Attn) + 4 (Đặc tính Gen - Stats) + 6 (Biểu thị Pathway) = Tổng kích thước 522 chiều phân loại.
    """

    def __init__(
        self,
        dnabert2_model_name: str = "zhihan1996/DNABERT-2-117M",
        n_families: int = 26,
        gate_hidden_dim: int = 128,
        n_gene_stats: int = 4,
        use_gate: bool = True,
        use_gene_stats: bool = True,
        use_pathway_scores: bool = True,
        embedding_dim: int = 768,
        cnn_out_dim: int = 384,
        global_out_dim: int = 128,
        num_classes: int = 2,
    ):
        super().__init__()
        self.use_gate = use_gate
        self.use_gene_stats = use_gene_stats
        self.use_pathway_scores = use_pathway_scores
        self.embedding_dim = embedding_dim
        self.n_families = n_families

        # --- Khối Kiến Trúc Nền (Backbone) ---
        # Khởi xuất DNABERT-2 thông qua dynamic module class (giải quyết triệt để rủi ro xung đột đăng ký - registry conflict trong họ hàm AutoModel).
        # Đồng nhất cơ chế kỹ thuật gọi gốc (load backbone) của cấu trúc PhaBERTCNN tiêu chuẩn trong mục models/phabert_cnn.py.
        _revision = "7bce263b15377fc15361f52cfab88f8b586abda0"

        config = BertConfig.from_pretrained(
            dnabert2_model_name, revision=_revision,
        )
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = 0
        # Tạm vô hiệu thuật toán bộ nhớ flash attention (nhằm duy trì ổn định hệ thống qua khả năng tương thích của môi trường Triton bản phân phối chuẩn)
        config.use_flash_attn = False

        model_cls = get_class_from_dynamic_module(
            "bert_layers.BertModel",
            dnabert2_model_name,
            revision=_revision,
            trust_remote_code=True,
        )
        self.backbone = model_cls.from_pretrained(
            dnabert2_model_name,
            config=config,
            revision=_revision,
        )

        # --- Cơ chế chốt kiểm soát gen (Giao thức tiêm dữ liệu Injection 1) ---
        if self.use_gate:
            self.gene_gate = GeneGateMLP(
                n_families=n_families,
                hidden_dim=gate_hidden_dim,
                output_dim=embedding_dim,
            )

        # --- Mô hình các cấu trúc chuỗi phân lớp CNN mở rộng (Kế thừa nền tảng baseline) ---
        self.cnn_branches = nn.ModuleList([
            MultiScaleCNNBranch(embedding_dim, kernel_size=k)
            for k in (3, 5, 7)
        ])

        # --- Trích xuất tính chất tập trung bằng Attention pooling (Kế thừa nền tảng baseline) ---
        self.attention_pooling = AttentionPooling(embedding_dim)
        self.global_projection = nn.Sequential(
            nn.Linear(embedding_dim, global_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # --- Giai đoạn lọc đặc tính chuyên môn Pathway score layer (Giao thức tiền giả thiết Inject 3, cơ chế tĩnh không khả huấn) ---
        if self.use_pathway_scores:
            self.pathway_score_layer = PathwayScoreLayer(n_families=n_families)

        # --- Trạm quyết định (Classifier) ---
        # Tổng hòa đầu vào: 384 (mạch CNN) + 128 (mạch attn) = chuẩn hóa 512 chiểu
        # Nếu điều hướng kích hoạt use_gene_stats:         Bổ sung 4 → Mức 516
        # Nếu điều hướng kích hoạt use_pathway_scores:     Bổ sung 6 → Mức 522
        classifier_in = cnn_out_dim + global_out_dim  # Chiều ban đầu 512
        if self.use_gene_stats:
            classifier_in += n_gene_stats              # +4
        if self.use_pathway_scores:
            classifier_in += PathwayScoreLayer.N_PATHWAYS  # +6
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in),
            nn.Linear(classifier_in, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        activation_vector: Optional[torch.Tensor] = None,
        gene_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Giao thức tính suy theo hướng tiến.

        Tham số điều kiện (Args):
            input_ids:         Dạng [B, L]
            attention_mask:    Dạng [B, L]
            activation_vector: Dạng [B, N_FAMILIES] — Ma trận trọng lượng phân giải bit (HMM bit-scores) tiệm cận chuẩn.
            gene_stats:        Dạng [B, 4]          — (Khai báo mẫu: số lượng count, mật độ density, khoảng mã hóa coding_frac, sự chênh lệch chuỗi mã strand_bias)

        Kết quả phản hồi (Returns):
            logits:            Dạng [B, num_classes]
        """
        # Bước 1: Trạm gốc DNABERT-2 tiến hành trích xuất phổ dữ liệu thô
        backbone_outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        if isinstance(backbone_outputs, tuple):
            h_trans = backbone_outputs[0]
        else:
            h_trans = backbone_outputs.last_hidden_state      # [B, L, 768]

        # Bước 2: Kích hoạt màng lọc Cổng gen (Cơ chế Inject 1 — Hiệu chỉnh đường truyền qua tích hợp thặng dư residual)
        if self.use_gate:
            if activation_vector is None:
                gate = torch.zeros(
                    h_trans.size(0), self.embedding_dim,
                    device=h_trans.device, dtype=h_trans.dtype,
                )
            else:
                gate = self.gene_gate(activation_vector)      # [B, 768]
            h_gated = h_trans + h_trans * gate.unsqueeze(1)   # [B, L, 768]
        else:
            h_gated = h_trans

        # Bước 3: Đưa đặc tính thô qua cấu trúc phân tán đa tầng CNN
        cnn_input = h_gated.transpose(1, 2)                   # [B, 768, L]
        cnn_features = [branch(cnn_input) for branch in self.cnn_branches]
        cnn_out = torch.cat(cnn_features, dim=-1)             # [B, 384]

        # Bước 4: Thiết định trọng lượng tầm soát bao trùm nhờ Attention pooling
        context_vector, _ = self.attention_pooling(h_gated, attention_mask)
        global_out = self.global_projection(context_vector)   # [B, 128]

        # Bước 5: Kết nạp chuỗi tham số thực thể từ hệ thống gene_stats (Cơ chế Inject 2)
        combined = torch.cat([cnn_out, global_out], dim=-1)   # [B, 512]
        if self.use_gene_stats:
            stats = gene_stats if gene_stats is not None else torch.zeros(
                combined.size(0), 4,
                device=combined.device, dtype=combined.dtype,
            )
            combined = torch.cat([combined, stats], dim=-1)   # [B, 516]

        # Bước 6: Tập hợp thông số chuẩn đích ngắm (Pathway scores - Cơ chế Inject 3 — Khảo sát tĩnh không tham gia mạng hội tụ trainable)
        if self.use_pathway_scores:
            act = activation_vector if activation_vector is not None else torch.zeros(
                combined.size(0), self.n_families,
                device=combined.device, dtype=combined.dtype,
            )
            pathway_s = self.pathway_score_layer(act)         # [B, 6]
            combined = torch.cat([combined, pathway_s], dim=-1)  # [B, 522]

        # Bước 7: Thâm nhập lớp quy hoạch ngõ ra chuẩn (Classifier)
        logits = self.classifier(combined)
        return logits

    def forward_head(
        self,
        h_trans: torch.Tensor,
        attention_mask: torch.Tensor,
        activation_vector: Optional[torch.Tensor] = None,
        gene_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pha suy luận rút gọn theo cơ chế bypass (tiến trình bỏ qua) áp dụng cho luồng truy xuất cache đặc trưng.
        Thực thi chuyên biệt tại Bước 1 (Phase 1 cache mode) để triệt tiêu thời gian chi phí hoạt động qua trung tâm DNABERT-2 ở mỗi chu trình lặp (epoch).

        Tham số nạp:
            h_trans:           Khoảng [B, L, 768] — Biểu diễn chuỗi nền tảng truy xuất trạng thái cache (fp16 hoặc cấp fp32)
            attention_mask:    Khoảng [B, L]
            activation_vector: Khoảng [B, N_FAMILIES] hoặc tham số bỏ qua (None)
            gene_stats:        Khoảng [B, 4] hoặc tham số rỗng (None)

        Giá trị trả về (Returns):
            logits:            Chuẩn [B, 2]
        """
        h = h_trans.float()

        if self.use_gate:
            if activation_vector is None:
                gate = torch.zeros(
                    h.size(0), self.embedding_dim,
                    device=h.device, dtype=h.dtype,
                )
            else:
                gate = self.gene_gate(activation_vector)
            h_gated = h + h * gate.unsqueeze(1)
        else:
            h_gated = h

        cnn_input = h_gated.transpose(1, 2)
        cnn_features = [branch(cnn_input) for branch in self.cnn_branches]
        cnn_out = torch.cat(cnn_features, dim=-1)

        context_vector, _ = self.attention_pooling(h_gated, attention_mask)
        global_out = self.global_projection(context_vector)

        combined = torch.cat([cnn_out, global_out], dim=-1)
        if self.use_gene_stats:
            stats = gene_stats if gene_stats is not None else torch.zeros(
                combined.size(0), 4, device=combined.device, dtype=combined.dtype,
            )
            combined = torch.cat([combined, stats], dim=-1)

        if self.use_pathway_scores:
            act = activation_vector if activation_vector is not None else torch.zeros(
                combined.size(0), self.n_families,
                device=combined.device, dtype=combined.dtype,
            )
            pathway_s = self.pathway_score_layer(act)
            combined = torch.cat([combined, pathway_s], dim=-1)

        return self.classifier(combined)

    # --- Hàm hỗ trợ điều phối tỉ lệ học phân biệt chiều hướng (Discriminative Learning Rate Helpers) ---

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_task_params(self):
        """Quy định chu kỳ hội tụ dành cho toàn bộ biến tham số chuyên biệt hệ thống (ngoại trừ backbone - thường được chỉ định nhân LR cường độ 10x)."""
        params = []
        for module in [
            self.cnn_branches,
            self.attention_pooling,
            self.global_projection,
            self.classifier,
        ]:
            params.extend(module.parameters())
        if self.use_gate:
            params.extend(self.gene_gate.parameters())
        # Chú thích thiết kế: Tầng PathwayScoreLayer thuần túy là dữ liệu tĩnh đệm (buffer) và không sở hữu các hệ số học tuyến tính trainable → bỏ qua trong trình tính hàm này.
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
                f"n_families={self.n_families}")
