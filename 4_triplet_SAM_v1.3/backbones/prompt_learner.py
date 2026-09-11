import os

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from typing import List, Tuple, Type

# from backbones.two_way_transformer import TwoWayTransformer
# from backbones.common import LayerNorm2d


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class Seg_Prompt_Learner(nn.Module):
    def __init__(self,
                 *,
                 transformer_dim: int,
                 num_multimask_outputs: int = 3,
                 activation: Type[nn.Module] = nn.GELU,
                 iou_head_depth: int = 3,
                 iou_head_hidden_dim: int = 256,


                 ):
        super(Seg_Prompt_Learner, self).__init__()

        self.transformer_dim = transformer_dim
        # self.transformer = TwoWayTransformer(
        #     depth=2,
        #     embedding_dim=transformer_dim,
        #     mlp_dim=2048,
        #     num_heads=8,
        # )
        #
        # self.num_multimask_outputs = num_multimask_outputs
        #
        # self.iou_token = nn.Embedding(1, transformer_dim)
        # self.num_mask_tokens = num_multimask_outputs + 1
        # self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        #
        # self.output_upscaling = nn.Sequential(
        #     nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
        #     LayerNorm2d(transformer_dim // 4),
        #     activation(),
        #     nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        #     activation(),
        # )
        # self.output_hypernetworks_mlps = nn.ModuleList(
        #     [
        #         MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        #         for i in range(self.num_mask_tokens)
        #     ]
        # )
        #
        # self.iou_prediction_head = MLP(
        #     transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        # )


    def forward(self, cls_tokens, x_img_e, report_feat_q, word_feat_q):  # (48,1,128) (48,78,128) (48,1,128) (48,78,128)

        ################ decoder ################
        img_word_output, img_word_attn = self.cross_attention(cls_tokens.unsqueeze(1), word_feat_q, word_feat_q)  # (48,1,78)
        report_patch_output, report_patch_attn = self.cross_attention(report_feat_q.unsqueeze(1), x_img_e, x_img_e)  # (48,1,78)
        # loss_local = optimal_transport_loss(img_word_output,report_patch_output)
        loss_local = F.mse_loss(img_word_output.squeeze(1),report_patch_output.squeeze(1))
        patch_word_matrix = report_patch_attn.squeeze(1).unsqueeze(2) @ img_word_attn.squeeze(1).unsqueeze(1)  # (48, 78, 78)
        word_patch_matrix = img_word_attn.squeeze(1).unsqueeze(2) @ report_patch_attn.squeeze(1).unsqueeze(1)  # (48, 78, 78)

        ################ Graph Learner ###############
        # 二值化处理
        patch_word_binary_matrix = self.patch_word_binarynet(patch_word_matrix)
        word_patch_binary_matrix = self.word_patch_binarynet(word_patch_matrix)
        binary_matrix = ((patch_word_binary_matrix.bool()) & (word_patch_binary_matrix.permute(0, 2, 1).bool())).float()

        sparsity_ratio = 0.5  # 控制稀疏化程度（例如 0.5 表示保留 50%）
        mask = (torch.rand_like(binary_matrix) < sparsity_ratio).float()
        binary_matrix = binary_matrix * mask

        # 构图
        graphs = []  # 存储每个 batch 的图
        batch_size = binary_matrix.size(0)
        img_nodes = binary_matrix.size(1)
        text_nodes = binary_matrix.size(2)

        for b in range(batch_size):
            # 获取当前 batch 的二值邻接矩阵
            adj_matrix = binary_matrix[b]  # (78,78)
            # 构造节点特征
            x_img = x_img_e[b]  # (78,128)
            x_text = word_feat_q[b]  # (78,128)
            x = torch.cat([x_img, x_text], dim=0)  # (156,128)
            # 构造边（edge_index）
            edge_index = []
            for i in range(img_nodes):
                for j in range(text_nodes):
                    if adj_matrix[i, j] == 1:  # 如果二值矩阵中 i -> j 存在连接
                        edge_index.append([i, j + img_nodes])  # 连接 img 节点 i 和 text 节点 j
                        # edge_index.append([j + img_nodes, i])  # 双向连接（可选）

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # 转置成 (2, num_edges)
            edge_index = edge_index.to(x.device)
            # 创建 PyG 的数据结构
            graph = Data(x=x, edge_index=edge_index)
            graphs.append(graph)

        ################ Prompt Generator ###############
        # 输入GAT
        gat_outputs = []
        for graph in graphs:
            output = self.gat_p(graph)  # GAT 输出形状 (num_nodes, 128)
            gat_outputs.append(output)
        gat_outputs = torch.stack(gat_outputs, dim=0)  # (48, num_nodes, 128)
        # 下采样
        gat_outputs_tran = torch.transpose(gat_outputs, 2, 1)
        prompt_down = self.downsample_p(gat_outputs_tran)
        prompt_down = torch.transpose(prompt_down, 2, 1)
        prompt_down = self.linear_proj(prompt_down)  # 128 -> 768

        return prompt_down, loss_local







