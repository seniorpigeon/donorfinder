"""样本编码器 Enc：把图节点特征编码为样本 embedding z。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 可选依赖：若安装 torch_geometric，可用 PyG 版 GraphSAGE
try:
    from torch_geometric.nn import SAGEConv, global_mean_pool

    HAS_PYG = True
except Exception:
    HAS_PYG = False


class _DenseGraphSAGELayer(nn.Module):
    """Dense 实现的 GraphSAGE 单层。

    输入：
    - x: (B, N, F)
    - adj_norm: (N, N)（已归一化）

    输出：
    - h: (B, N, F_out)
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.nei_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # 邻居聚合：adj_norm @ x
        nei = torch.einsum("ij,bjf->bif", adj_norm, x)
        # 自身特征与邻居特征分别线性映射后相加
        out = self.self_linear(x) + self.nei_linear(nei)
        return F.relu(out)


class DenseGraphSAGEEncoder(nn.Module):
    """纯 PyTorch 版本编码器（无需 PyG）。"""

    def __init__(
        self,
        adj_norm: np.ndarray,
        hidden_dim: int,
        z_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        # 邻接矩阵作为 buffer 保存，不参与训练
        adj = torch.tensor(adj_norm, dtype=torch.float32)
        self.register_buffer("adj_norm", adj)

        # 输入节点初始维度为1（每个节点一个标量丰度）
        layers = []
        in_dim = 1
        for _ in range(num_layers):
            layers.append(_DenseGraphSAGELayer(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(dropout)

        # Readout：先对节点做 mean pooling，再映射到 z_dim
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, x_node: torch.Tensor) -> torch.Tensor:
        """x_node: (B, N) -> z: (B, z_dim)"""
        # 扩一维得到节点特征维 F=1
        h = x_node.unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, self.adj_norm)
            h = self.dropout(h)
        pooled = h.mean(dim=1)
        return self.readout(pooled)


class MLPEncoder(nn.Module):
    """无图 fallback 编码器（纯 MLP）。"""

    def __init__(self, num_nodes: int, hidden_dim: int, z_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, x_node: torch.Tensor) -> torch.Tensor:
        return self.net(x_node)


class PygGraphSAGEEncoder(nn.Module):
    """PyG 版本 GraphSAGE 编码器。"""

    def __init__(
        self,
        num_nodes: int,
        edge_index: np.ndarray,
        hidden_dim: int,
        z_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not HAS_PYG:
            raise RuntimeError("torch_geometric is not installed.")

        self.num_nodes = int(num_nodes)
        ei = torch.tensor(edge_index, dtype=torch.long)
        self.register_buffer("edge_index", ei)

        # 多层 SAGEConv
        convs = []
        in_dim = 1
        for _ in range(num_layers):
            convs.append(SAGEConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.convs = nn.ModuleList(convs)

        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )

    def _batch_edge_index(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """把单图 edge_index 平铺扩展成 batch 图 edge_index。"""
        e = self.edge_index
        offsets = torch.arange(batch_size, device=device, dtype=torch.long) * self.num_nodes
        expanded = e.unsqueeze(0) + offsets[:, None, None]
        return expanded.permute(1, 0, 2).reshape(2, -1)

    def forward(self, x_node: torch.Tensor) -> torch.Tensor:
        """x_node: (B, N) -> z: (B, z_dim)"""
        bsz, n = x_node.shape
        if n != self.num_nodes:
            raise ValueError(f"x_node num_nodes mismatch: expected {self.num_nodes}, got {n}")

        # PyG 输入展平为 (B*N, F)
        x = x_node.reshape(bsz * n, 1)
        edge_index = self._batch_edge_index(bsz, x_node.device)
        batch = torch.arange(bsz, device=x_node.device).repeat_interleave(n)

        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = self.dropout(h)

        # 图级池化后输出样本 embedding
        pooled = global_mean_pool(h, batch)
        return self.readout(pooled)


class SampleEncoder(nn.Module):
    """编码器统一包装器。

    backend 选项：
    - dense_gnn: 默认，纯 PyTorch GraphSAGE
    - pyg:       使用 torch_geometric（未安装则自动 fallback 到 mlp）
    - mlp:       纯 MLP
    """

    def __init__(
        self,
        num_nodes: int,
        edge_index: np.ndarray,
        adj_norm: np.ndarray,
        z_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        backend: str = "dense_gnn",
    ) -> None:
        super().__init__()

        if backend == "pyg":
            if HAS_PYG:
                self.model = PygGraphSAGEEncoder(
                    num_nodes=num_nodes,
                    edge_index=edge_index,
                    hidden_dim=hidden_dim,
                    z_dim=z_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            else:
                # PyG 不可用时回退为 MLP，保证可运行
                self.model = MLPEncoder(num_nodes=num_nodes, hidden_dim=hidden_dim, z_dim=z_dim, dropout=dropout)
        elif backend == "mlp":
            self.model = MLPEncoder(num_nodes=num_nodes, hidden_dim=hidden_dim, z_dim=z_dim, dropout=dropout)
        else:
            self.model = DenseGraphSAGEEncoder(
                adj_norm=adj_norm,
                hidden_dim=hidden_dim,
                z_dim=z_dim,
                num_layers=num_layers,
                dropout=dropout,
            )

    def forward(self, x_node: torch.Tensor) -> torch.Tensor:
        """统一前向接口。"""
        return self.model(x_node)
