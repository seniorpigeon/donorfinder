"""Stage A：预训练 Enc + fr。

目标损失：
  L = L_post + lambda_y * L_y
其中：
  - L_post: MSE(z_post_pred, z_post_true)
  - L_y:    BCE(y_pred, y)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from src.data.dataset_fmt import FMTQuadrupletDataset, load_quadruplet_arrays, make_dataloaders
from src.graphs.build_graph import build_graph_from_config
from src.models.encoder_gnn import SampleEncoder
from src.models.forward_fr import ForwardSimulator
from src.train.losses import loss_post, loss_y
from src.train.utils import (
    ensure_dir,
    get_device,
    load_config,
    save_checkpoint,
    save_json,
    set_global_seed,
    split_train_val_indices,
)


def parse_args() -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="Stage A training for Enc + fr.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def _extract_taxa_names(arrays: Dict[str, np.ndarray]) -> List[str]:
    """从数据中提取 taxon 名称；若没有则自动构造。"""
    if "taxa_names" in arrays:
        return [str(x) for x in arrays["taxa_names"].tolist()]
    t = arrays["R_pre"].shape[1]
    return [f"sp_{i:04d}" for i in range(t)]


def _run_epoch(
    encoder: SampleEncoder,
    forward_model: ForwardSimulator,
    loader,
    optimizer,
    device: torch.device,
    lambda_y: float,
    training: bool,
) -> Dict[str, float]:
    """运行一个 epoch（训练或验证）。"""
    if training:
        encoder.train()
        forward_model.train()
    else:
        encoder.eval()
        forward_model.eval()

    total = 0.0
    total_post = 0.0
    total_y = 0.0
    count = 0

    for batch in loader:
        # 1) 取出 batch 数据
        x_pre = batch["R_pre"].to(device)
        x_d = batch["D"].to(device)
        x_post = batch["R_post"].to(device)
        y = batch["y"].to(device)

        if training:
            optimizer.zero_grad()

        # 2) 先编码到 latent space
        z_pre = encoder(x_pre)
        z_d = encoder(x_d)
        z_post_true = encoder(x_post)

        # 3) 前向模块预测 post 与 outcome
        z_post_pred, y_logit = forward_model(z_pre, z_d)

        # 4) 计算 Stage A 损失
        l_post = loss_post(z_post_pred, z_post_true)
        l_y = loss_y(y_logit, y)
        loss = l_post + lambda_y * l_y

        if training:
            # 5) 反向传播更新参数
            loss.backward()
            optimizer.step()

        # 6) 累计统计指标
        bs = len(y)
        total += float(loss.item()) * bs
        total_post += float(l_post.item()) * bs
        total_y += float(l_y.item()) * bs
        count += bs

    return {
        "loss": total / max(count, 1),
        "loss_post": total_post / max(count, 1),
        "loss_y": total_y / max(count, 1),
    }


@torch.no_grad()
def _compute_healthy_prototype(
    encoder: SampleEncoder,
    x_post_all: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """计算默认推理目标 prototype_healthy。

    规则：
    - 优先使用 y=1 的 post 样本 embedding 均值
    - 若没有 y=1，则退化为全体 post 均值
    """
    encoder.eval()
    pos_idx = np.where(y > 0.5)[0]
    if len(pos_idx) == 0:
        pos_idx = np.arange(len(y))

    zs = []
    for st in range(0, len(pos_idx), batch_size):
        ed = min(st + batch_size, len(pos_idx))
        xb = torch.tensor(x_post_all[pos_idx[st:ed]], dtype=torch.float32, device=device)
        z = encoder(xb)
        zs.append(z.cpu().numpy())

    z_all = np.concatenate(zs, axis=0).astype(np.float32)
    proto = np.mean(z_all, axis=0, keepdims=False).astype(np.float32)
    return proto


def main() -> None:
    """Stage A 主流程。"""
    args = parse_args()
    cfg = load_config(args.config)

    # 1) 基础初始化：随机种子 + 设备
    set_global_seed(int(cfg["seed"]))
    device = get_device(str(cfg["train"]["device"]))

    # 2) 加载四元组数据并做 transform
    arrays = load_quadruplet_arrays(
        data_dir=cfg["paths"]["data_dir"],
        transform=str(cfg["data"]["transform"]),
        clr_eps=float(cfg["data"]["clr_eps"]),
    )

    # 3) 构图（frc/species）并把 species 丰度映射到 node 特征
    taxa_names = _extract_taxa_names(arrays)
    graph = build_graph_from_config(cfg, taxa_names=taxa_names)

    x_pre_node = graph.to_node_features(arrays["R_pre"])
    x_d_node = graph.to_node_features(arrays["D"])
    x_post_node = graph.to_node_features(arrays["R_post"])

    # 4) 组装 Dataset
    ds = FMTQuadrupletDataset(
        r_pre=x_pre_node,
        donor=x_d_node,
        r_post=x_post_node,
        y=arrays["y"],
        recipient_id=arrays.get("recipient_id"),
        donor_id=arrays.get("donor_id"),
    )

    # 5) 划分 train/val（可按 recipient 分组，避免泄漏）
    train_idx, val_idx = split_train_val_indices(
        y=arrays["y"],
        groups=arrays.get("recipient_id"),
        val_ratio=float(cfg["data"]["val_ratio"]),
        seed=int(cfg["seed"]),
        use_group_split=bool(cfg["data"]["use_group_split"]),
    )

    train_loader, val_loader = make_dataloaders(
        dataset=ds,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
    )

    # 6) 实例化 Enc 与 fr
    adj_norm = graph.build_dense_adj(add_self_loop=True)
    encoder = SampleEncoder(
        num_nodes=graph.num_nodes(),
        edge_index=graph.edge_index,
        adj_norm=adj_norm,
        z_dim=int(cfg["encoder"]["z_dim"]),
        hidden_dim=int(cfg["encoder"]["hidden_dim"]),
        num_layers=int(cfg["encoder"]["num_layers"]),
        dropout=float(cfg["encoder"]["dropout"]),
        backend=str(cfg["encoder"]["backend"]),
    ).to(device)

    forward_model = ForwardSimulator(
        z_dim=int(cfg["encoder"]["z_dim"]),
        hidden_dim=int(cfg["forward"]["hidden_dim"]),
        dropout=float(cfg["forward"]["dropout"]),
    ).to(device)

    # 7) 优化器：同时更新 encoder + forward_model
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(forward_model.parameters()),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["stage_a_epochs"])
    patience = int(cfg["train"]["early_stop_patience"])
    lambda_y = float(cfg["loss"]["lambda_y"])

    best_val = float("inf")
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    # 8) 标准训练循环 + 早停
    for epoch in tqdm(range(1, epochs + 1), desc="Stage A"):
        train_metrics = _run_epoch(
            encoder, forward_model, train_loader, optimizer, device, lambda_y=lambda_y, training=True
        )
        val_metrics = _run_epoch(
            encoder, forward_model, val_loader, optimizer, device, lambda_y=lambda_y, training=False
        )

        rec = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_loss_post": train_metrics["loss_post"],
            "train_loss_y": train_metrics["loss_y"],
            "val_loss": val_metrics["loss"],
            "val_loss_post": val_metrics["loss_post"],
            "val_loss_y": val_metrics["loss_y"],
        }
        history.append(rec)

        # 9) 保存最优 checkpoint
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            bad_epochs = 0

            save_checkpoint(
                cfg["paths"]["stage_a_ckpt"],
                {
                    "encoder_state": encoder.state_dict(),
                    "forward_state": forward_model.state_dict(),
                    "graph_mode": graph.mode,
                    "taxa_names": graph.taxa_names,
                    "node_names": graph.node_names,
                    "best_val": best_val,
                    "config": cfg,
                },
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    # 10) 保存日志与 prototype_healthy（供推理默认 target 使用）
    logs_dir = ensure_dir(cfg["paths"]["logs_dir"])
    save_json(logs_dir / "stage_a_metrics.json", {"history": history, "best_val": best_val})

    proto = _compute_healthy_prototype(
        encoder=encoder,
        x_post_all=x_post_node,
        y=arrays["y"],
        device=device,
        batch_size=int(cfg["inference"]["batch_size"]),
    )
    np.save(logs_dir / "prototype_healthy.npy", proto.astype(np.float32))

    print(f"Stage A finished. best_val={best_val:.6f}")
    print(f"Checkpoint: {cfg['paths']['stage_a_ckpt']}")


if __name__ == "__main__":
    main()
