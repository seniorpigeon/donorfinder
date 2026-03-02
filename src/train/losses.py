"""训练损失函数定义（Stage A/B/C 共用）。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def loss_post(z_post_pred: torch.Tensor, z_post_true: torch.Tensor) -> torch.Tensor:
    """L_post：post 表征回归损失（MSE）。"""
    return F.mse_loss(z_post_pred, z_post_true)


def loss_y(y_logit: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """L_y：结局监督损失（二分类 BCE with logits）。"""
    y_true = y_true.float()
    return F.binary_cross_entropy_with_logits(y_logit, y_true)


def loss_cycle(
    forward_model,
    z_pre: torch.Tensor,
    z_d_hat: torch.Tensor,
    z_target: torch.Tensor,
) -> torch.Tensor:
    """L_cycle：cycle consistency。

    公式：
      z_post_cycle = fr(z_pre, z_d_hat)
      L_cycle = MSE(z_post_cycle, z_target)
    """
    z_post_cycle, _ = forward_model(z_pre, z_d_hat)
    return F.mse_loss(z_post_cycle, z_target)


def loss_donor_supervision(
    z_d_hat: torch.Tensor,
    z_d_true: torch.Tensor,
    y_true: torch.Tensor,
    positive_only: bool = True,
) -> torch.Tensor:
    """L_d：逆向供体监督损失。

    - positive_only=True 时，仅在 y=1 样本上监督 z_d_hat 接近 z_d_true。
    - 若当前 batch 没有正样本，则退化为全样本 MSE，避免梯度为零。
    """
    if not positive_only:
        return F.mse_loss(z_d_hat, z_d_true)

    # y>0.5 认为是治愈样本
    mask = (y_true > 0.5).float().unsqueeze(-1)
    if torch.sum(mask) < 1:
        return F.mse_loss(z_d_hat, z_d_true)

    # 手动按 mask 计算均方误差
    diff2 = (z_d_hat - z_d_true) ** 2
    masked = diff2 * mask
    return torch.sum(masked) / (torch.sum(mask) * z_d_hat.shape[1] + 1e-12)
