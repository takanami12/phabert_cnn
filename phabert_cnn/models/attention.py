"""
Mô đun tối ưu hóa đặc trưng bằng Cơ chế Chú ý (Attention Pooling Module) cho kiến trúc PhaBERT-CNN.

Triển khai cấu trúc self-attention theo chuẩn mực nghiên cứu của Lin et al. (2017)
"A Structured Self-Attentive Sentence Embedding" để trích xuất và cô đọng
thông tin ngữ nghĩa toàn cục (global level) từ các ma trận đặc trưng tĩnh (embeddings) của mô hình lõi DNABERT-2.

Mô hình phương trình cấu trúc cơ bản ứng với trạng thái ẩn h_t tại vị trí chuỗi t:
  u_t = tanh(W_a * h_t + b_a)
  alpha_t = softmax(u_t^T * u_s)
  v = sum(alpha_t * h_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 64):
        super().__init__()
        
        # Mạng lan truyền xuôi (Feed-forward network) hỗ trợ tối ưu hóa trọng số chú ý cục bộ
        self.W_a = nn.Linear(embedding_dim, hidden_dim)
        self.u_s = nn.Linear(hidden_dim, 1, bias=False)  # Véc-tơ tham chiếu bối cảnh không gian (Context vector)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Tham số đầu vào:
            hidden_states:  Dạng (batch_size, seq_len, embedding_dim) — Tín hiệu đặc trưng thô từ DNABERT-2
            attention_mask: Dạng (batch_size, seq_len) — Vector nhị phân: 1 chỉ định token hợp lệ, 0 biểu thị token đệm (padding)

        Dữ liệu trả về (Returns):
            context_vector:    Dạng (batch_size, embedding_dim) — Cấu trúc biểu diễn nội dung không gian tuyến tính theo phân bố tập trung
            attention_weights: Dạng (batch_size, seq_len) — Ma trận phân phối xác suất chú ý
        """
        # u_t = tanh(W_a * h_t + b_a)
        u = torch.tanh(self.W_a(hidden_states))  # (B, L, hidden_dim)

        # Thước đo điểm chuẩn hóa vô hướng (Scalar attention score)
        scores = self.u_s(u).squeeze(-1)  # (B, L)

        # Vô hiệu hóa phân phối attention cho các token bị đệm (padding masking) để tối ưu tính xác thực hàm softmax
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # alpha_t = softmax(scores)
        attention_weights = F.softmax(scores, dim=-1)  # (B, L)

        # v = sum(alpha_t * h_t)
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # (B, 1, L)
            hidden_states                    # (B, L, D)
        ).squeeze(1)  # (B, D)
        
        return context_vector, attention_weights
