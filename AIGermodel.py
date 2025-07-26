# coding=utf-8
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layers import create_spectral_features, MLP


class AIGerConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations=2, self_loop=True):
        super().__init__()
        self.num_relations = num_relations
        self.self_loop = self_loop

        self.relation_layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)
        ])

        if self_loop:
            self.self_loop_layer = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.self_loop_layer = None

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.relation_layers:
            nn.init.xavier_uniform_(layer.weight)
        if self.self_loop:
            nn.init.xavier_uniform_(self.self_loop_layer.weight)

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

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.weight.reset_parameters()

    def get_x_edge_index(self, init_emb, edge_index_s):
        self.pos_edge_index = edge_index_s[edge_index_s[:, 2] > 0][:, :2].t()
        self.neg_edge_index = edge_index_s[edge_index_s[:, 2] < 0][:, :2].t()
        if init_emb is None:
            init_emb = create_spectral_features(
                pos_edge_index=self.pos_edge_index,
                neg_edge_index=self.neg_edge_index,
                node_num=self.node_num,
                dim=self.in_dim
            ).to(self.device)
        self.x = init_emb

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