"""前向响应模块 fr：输入 (z_pre, z_donor) 输出 (z_post_pred, y_pred)。"""

from __future__ import annotations

import torch
import torch.nn as nn


class ForwardSimulator(nn.Module):
    """模拟 FMT 后表征与疗效概率。

    关键约束（按你的要求）：
    y_pred 只能依赖 (z_pre, z_donor)，不依赖 z_post_true。
    """

    def __init__(self, z_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()

        # pre 与 donor 拼接后输入维度
        in_dim = z_dim * 2

        # 头1：预测 post-FMT embedding
        self.post_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )

        # 头2：预测疗效 logit（后续 sigmoid 变概率）
        self.y_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_pre: torch.Tensor, z_donor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (z_post_pred, y_logit)。"""
        # 拼接 pre 与 donor embedding
        x = torch.cat([z_pre, z_donor], dim=-1)
        # 分别经过两个任务头
        z_post_pred = self.post_head(x)
        y_logit = self.y_head(x).squeeze(-1)
        return z_post_pred, y_logit

    def predict_proba(self, z_pre: torch.Tensor, z_donor: torch.Tensor) -> torch.Tensor:
        """返回疗效概率 y_pred（sigmoid(logit)）。"""
        _, y_logit = self.forward(z_pre, z_donor)
        return torch.sigmoid(y_logit)
