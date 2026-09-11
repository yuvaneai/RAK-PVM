import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class DensePromptForTest(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(1, output_dim))

    def forward(self, batch_size):
        return self.prompt.expand(batch_size, -1)


import torch
import torch.nn as nn
import torch.nn.functional as F

# class AnatomyStudent(nn.Module):
#     """
#     Anatomy-aware Student for patch-based distillation.
#     - Query = content + position
#     - Multi-layer cross-attention with residual + layernorm
#     - Outputs patch embeddings + attention maps for KL distillation
#     """
#     def __init__(
#         self,
#         patch_dim=256,
#         hidden_dim=256,
#         num_queries=50,
#         num_layers=3,
#         nhead=4
#     ):
#         super().__init__()
#
#         self.num_queries = num_queries
#         self.hidden_dim = hidden_dim
#
#         # --------------------------
#         # 1️⃣ Queries: content + position
#         # --------------------------
#         self.query_content = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
#         self.query_pos = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
#
#         # --------------------------
#         # 2️⃣ Patch projection
#         # --------------------------
#         self.patch_proj = nn.Linear(patch_dim, hidden_dim)
#
#         # --------------------------
#         # 3️⃣ Query graph modeling (self-attention among queries)
#         # --------------------------
#         self.query_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
#         self.query_norm = nn.LayerNorm(hidden_dim)
#
#         # --------------------------
#         # 4️⃣ Multi-layer cross-attention
#         # --------------------------
#         self.cross_attn_layers = nn.ModuleList([
#             nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
#             for _ in range(num_layers)
#         ])
#         self.cross_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
#
#         # --------------------------
#         # 5️⃣ Feed-forward network
#         # --------------------------
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 4),
#             nn.GELU(),
#             nn.Linear(hidden_dim * 4, hidden_dim)
#         )
#         self.ffn_norm = nn.LayerNorm(hidden_dim)
#
#         # --------------------------
#         # 6️⃣ Output projection
#         # --------------------------
#         self.out_proj = nn.Linear(hidden_dim, patch_dim)
#
#     def forward(self, patch_emb):
#         """
#         Args:
#             patch_emb: [B, P, D] - patch embeddings from image encoder
#         Returns:
#             out: [B, Q, D] - student patch embeddings
#             final_attn: [B, Q, P] - attention map for KL distillation
#         """
#         B, P, D = patch_emb.shape
#
#         # --------------------------
#         # 1️⃣ Project patches
#         # --------------------------
#         feat = self.patch_proj(patch_emb)  # [B, P, H]
#
#         # --------------------------
#         # 2️⃣ Initialize queries
#         # --------------------------
#         q = self.query_content + self.query_pos
#         q = q.expand(B, -1, -1)  # [B, Q, H]
#
#         # --------------------------
#         # 3️⃣ Query self-attention (graph modeling)
#         # --------------------------
#         q_res = q
#         q_attn, _ = self.query_attn(q, q, q)
#         q = self.query_norm(q + q_attn)
#
#         # --------------------------
#         # 4️⃣ Multi-layer cross-attention
#         # --------------------------
#         attn_maps = []
#         for attn_layer, norm_layer in zip(self.cross_attn_layers, self.cross_norms):
#             q_res = q
#             # cross-attention: queries attend to patches
#             q_attn, attn_map = attn_layer(q, feat, feat, need_weights=True)
#             q = norm_layer(q + q_attn)
#             attn_maps.append(attn_map)  # [B, Q, P]
#
#         # --------------------------
#         # 5️⃣ Feed-forward network
#         # --------------------------
#         q_res = q
#         q = self.ffn_norm(q + self.ffn(q))
#
#         # --------------------------
#         # 6️⃣ Output embeddings
#         # --------------------------
#         out = self.out_proj(q)  # [B, Q, D]
#
#         # --------------------------
#         # 7️⃣ Return final attention map
#         # --------------------------
#         final_attn = attn_maps[-1]  # [B, Q, P]
#
#         return out, final_attn

class AnatomyStudent(nn.Module):
    def __init__(self, patch_dim=256, hidden_dim=128, num_queries=30):
        super().__init__()
        self.num_queries = num_queries

        # 可学习解剖查询向量 🔥 这个必须用！
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        # 图像特征投影
        self.proj = nn.Linear(patch_dim, hidden_dim)

        # 🔥 必须用 TransformerDecoder，不是 Encoder！
        # 因为 queries 要和图像特征做交叉注意力
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=2,
                dim_feedforward=256,
                batch_first=True
            ),
            num_layers=1
        )

        self.out = nn.Linear(hidden_dim, patch_dim)

    def forward(self, patch_emb):
        B = patch_emb.shape[0]

        # 1. 图像特征
        feat = self.proj(patch_emb)  # [B, P, H]

        # 2. 可学习查询向量 🔥 这里才是正确使用！
        q = self.queries.repeat(B, 1, 1)  # [B, Q, H]

        # 3. 🔥 查询向量 + 图像特征交叉注意力（核心！）
        out = self.decoder(tgt=q, memory=feat)  # q 去查 feat

        # 4. 输出
        return self.out(out)

class AnatomyTeacher(nn.Module):
    def __init__(self, INIT_EMBED_VOCAB: int, POS_EMBED_DIM: int):
        super().__init__()
        self.INIT_EMBED_VOCAB = INIT_EMBED_VOCAB
        self.POS_EMBED_DIM = POS_EMBED_DIM
        self.dynamic_pos2idx = {}

        # 🔥 修复：注册 buffer 保存字典
        self.register_buffer('pos_keys', torch.empty(0, dtype=torch.long))
        self.register_buffer('pos_values', torch.empty(0, dtype=torch.long))

        self.pos_embedding_layer = nn.Embedding(
            self.INIT_EMBED_VOCAB, self.POS_EMBED_DIM, padding_idx=0
        )
        self.maha_sim = nn.Parameter(torch.eye(self.POS_EMBED_DIM))

    def get_dynamic_pos_idx(self, pos_list, device):
        for pos in pos_list:
            if pos not in self.dynamic_pos2idx:
                self.dynamic_pos2idx[pos] = len(self.dynamic_pos2idx)
                if len(self.dynamic_pos2idx) + 1 > self.pos_embedding_layer.num_embeddings:
                    # 🔥 修复：先复制权重，再移动设备
                    new_embedding = nn.Embedding(len(self.dynamic_pos2idx) + 10, self.POS_EMBED_DIM, padding_idx=0)
                    new_embedding.weight.data[
                    :self.pos_embedding_layer.num_embeddings] = self.pos_embedding_layer.weight.data
                    new_embedding = new_embedding.to(device)
                    self.pos_embedding_layer = new_embedding

        pos_idx = [self.dynamic_pos2idx[pos] + 1 for pos in pos_list]
        return torch.tensor(pos_idx, dtype=torch.long, device=device)

    def MahalanobisSimilarity(self, x, y):
        xM = torch.matmul(x, self.maha_sim)
        yM = torch.matmul(y, self.maha_sim)
        x_term = (xM * x).sum(dim=-1).unsqueeze(2)
        y_term = (yM * y).sum(dim=-1).unsqueeze(1)
        xy_term = torch.matmul(xM, y.transpose(1, 2))
        dist2 = x_term + y_term - 2 * xy_term
        dist2 = torch.clamp(dist2, min=0.0)
        return torch.exp(-dist2 / 2.0)

    def forward(self, patch_batch, pos_list_batch):
        device = patch_batch.device
        pos_idx_batch = []

        for pos_list in pos_list_batch:
            valid_pos = [pos for pos in pos_list if pos != "none"]
            if not valid_pos:
                pos_idx_batch.append(torch.tensor([0], device=device))
                continue
            pos_idx = self.get_dynamic_pos_idx(valid_pos, device)
            pos_idx_batch.append(pos_idx)

        pad_val = 0
        pos_idx_padded = pad_sequence(pos_idx_batch, batch_first=True, padding_value=pad_val)
        pos_idx_padded = pos_idx_padded.clamp(max=self.pos_embedding_layer.num_embeddings - 1)
        padding_mask = (pos_idx_padded != pad_val).unsqueeze(-1).to(device)

        pos_emb_batch = self.pos_embedding_layer(pos_idx_padded)
        pos_emb_norm = F.normalize(pos_emb_batch, dim=-1)
        patch_emb_norm = F.normalize(patch_batch, dim=-1)

        similarity_batch = self.MahalanobisSimilarity(pos_emb_norm, patch_emb_norm)
        similarity_batch = similarity_batch.masked_fill(~padding_mask, 0.0)

        similarity_soft = F.softmax(similarity_batch, dim=-1)

        # 🔥 修复：sum(dim=-1)
        patch_competition = similarity_soft / (similarity_soft.sum(dim=-1, keepdim=True) + 1e-8)

        best_patch_teacher = torch.matmul(patch_competition, patch_emb_norm)

        return best_patch_teacher, patch_competition, similarity_batch, padding_mask