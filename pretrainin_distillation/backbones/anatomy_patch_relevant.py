import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class DensePromptForTest(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(1, output_dim))

    def forward(self, batch_size):
        return self.prompt.expand(batch_size, -1)

class AnatomyStudent(nn.Module):
    def __init__(self, patch_dim=256, hidden_dim=128, num_queries=30):
        super().__init__()
        self.num_queries = num_queries

        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        self.proj = nn.Linear(patch_dim, hidden_dim)

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
        feat = self.proj(patch_emb)
        q = self.queries.repeat(B, 1, 1)
        out = self.decoder(tgt=q, memory=feat)
        return self.out(out)

class AnatomyTeacher(nn.Module):
    def __init__(self, INIT_EMBED_VOCAB: int, POS_EMBED_DIM: int):
        super().__init__()
        self.INIT_EMBED_VOCAB = INIT_EMBED_VOCAB
        self.POS_EMBED_DIM = POS_EMBED_DIM
        self.dynamic_pos2idx = {}
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

        patch_competition = similarity_soft / (similarity_soft.sum(dim=-1, keepdim=True) + 1e-8)

        best_patch_teacher = torch.matmul(patch_competition, patch_emb_norm)

        return best_patch_teacher, patch_competition, similarity_batch, padding_mask