# coding=utf-8
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layers import create_spectral_features, MLP


class AIGerConv(nn.Module):
    # Attention, we will make the source code fully open after the paper is accepted.

    def forward(self, x, edge_indices):
        agg = 0
        for i in range(self.num_relations):
            edge_index = edge_indices[i]
            layer = self.relation_layers[i]
            source = edge_index[0]
            target = edge_index[1]
            messages = layer(x[source])
            agg_per_rel = torch.zeros(x.size(0), messages.size(1), dtype=messages.dtype, device=x.device)
            agg_per_rel.scatter_add_(0, target.unsqueeze(-1).expand(-1, messages.size(1)), messages)
            agg += agg_per_rel

        if self.self_loop:
            agg += self.self_loop_layer(x)

        return agg


class AIGer(nn.Module):
    def __init__(
            self,
            args,
            node_num: int,
            device: torch.device,
            in_dim: int = 64,
            out_dim: int = 64,
            layer_num: int = 2,
            lamb: float = 5,
            norm_emb: bool = False,
            ** kwargs
    ):
        super().__init__(**kwargs)
        self.args = args
        self.node_num = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lamb = lamb
        self.device = device

        self.pos_edge_index = None
        self.neg_edge_index = None
        self.x = None

        self.conv1 = AIGerConv(in_dim, out_dim // 2, num_relations=2)

        self.convs = nn.ModuleList()
        for _ in range(layer_num - 1):
            self.convs.append(AIGerConv(out_dim // 2, out_dim // 2, num_relations=2))

        # 调整维度映射
        self.weight = nn.Linear(out_dim // 2, out_dim)
        self.readout_prob = MLP(out_dim, out_dim, 1, num_layer=3, p_drop=0.2,
                                norm_layer='batchnorm', act_layer='relu')
        self.reset_parameters()

    def forward(self, init_emb, edge_index_s) -> Tuple[Tensor, Tensor]:
        self.get_x_edge_index(init_emb, edge_index_s)

        # 前向传播时传递关系类型的边索引列表
        z = torch.tanh(self.conv1(self.x, [self.pos_edge_index, self.neg_edge_index]))
        for conv in self.convs:
            z = torch.tanh(conv(z, [self.pos_edge_index, self.neg_edge_index]))

        # 维度调整
        z = torch.tanh(self.weight(z))
        prob = torch.sigmoid(self.readout_prob(z))

        return z, prob
