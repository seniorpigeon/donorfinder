"""逆问题模块 fp：根据 (z_pre, z_target) 生成理想供体 embedding。"""

from __future__ import annotations

import torch
import torch.nn as nn


class InverseGenerator(nn.Module):
    """逆向生成器。

    数学形式：
      z_d_hat = fp([z_pre, z_target, z_target - z_pre])

    这里使用 MLP 实现，输出维度与 embedding 维度 z_dim 一致。
    """

    def __init__(self, z_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()

        # 输入拼接了三个 z 向量，因此输入维度是 3*z_dim
        in_dim = z_dim * 3

        # 两层隐藏层的 MLP
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z_pre: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        # 按论文/需求把 pre、target、差值拼接
        x = torch.cat([z_pre, z_target, z_target - z_pre], dim=-1)
        # 输出理想供体 embedding
        return self.net(x)
