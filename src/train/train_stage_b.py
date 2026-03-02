"""Stage B：冻结 Enc + fr，仅训练逆向模块 fp。

目标损失：
  L = L_cycle + lambda_d * L_d
其中：
  - L_cycle: MSE(fr(z_pre, fp(z_pre, z_post_true)), z_post_true)
  - L_d:     MSE(z_d_hat, z_d_true)（可只在 y=1 上监督）
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import torch
from tqdm import tqdm

from src.data.dataset_fmt import FMTQuadrupletDataset, load_quadruplet_arrays, make_dataloaders
from src.graphs.build_graph import build_graph_from_config
from src.models.encoder_gnn import SampleEncoder
from src.models.forward_fr import ForwardSimulator
from src.models.inverse_fp import InverseGenerator
from src.train.losses import loss_cycle, loss_donor_supervision
from src.train.utils import (
    ensure_dir,
    get_device,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_json,
    set_global_seed,
    split_train_val_indices,
)


def parse_args() -> argparse.Namespace:
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="Stage B training for inverse generator fp.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def _extract_taxa_names(arrays: Dict[str, object]) -> List[str]:
    """提取 taxon 名称。"""
    if "taxa_names" in arrays:
        return [str(x) for x in arrays["taxa_names"].tolist()]
    t = arrays["R_pre"].shape[1]
    return [f"sp_{i:04d}" for i in range(t)]


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    """统一设置模块参数是否可训练。"""
    for p in module.parameters():
        p.requires_grad = flag


def _run_epoch(
    encoder: SampleEncoder,
    forward_model: ForwardSimulator,
    inverse_model: InverseGenerator,
    loader,
    optimizer,
    lambda_d: float,
    positive_only: bool,
    device: torch.device,
    training: bool,
) -> Dict[str, float]:
    """执行一个 Stage B epoch。"""
    # Stage B 中 Enc 和 fr 全程冻结，仅 fp 训练
    encoder.eval()
    forward_model.eval()
    inverse_model.train(mode=training)

    total = 0.0
    total_c = 0.0
    total_d = 0.0
    count = 0

    for batch in loader:
        x_pre = batch["R_pre"].to(device)
        x_d = batch["D"].to(device)
        x_post = batch["R_post"].to(device)
        y = batch["y"].to(device)

        # Enc 冻结后，不需要为其保存梯度图
        with torch.no_grad():
            z_pre = encoder(x_pre)
            z_d_true = encoder(x_d)
            z_post_true = encoder(x_post)

        if training:
            optimizer.zero_grad()

        # fp 生成理想供体 embedding
        z_d_hat = inverse_model(z_pre, z_post_true)

        # Stage B 损失
        l_cycle = loss_cycle(
            forward_model=forward_model,
            z_pre=z_pre,
            z_d_hat=z_d_hat,
            z_target=z_post_true,
        )
        l_d = loss_donor_supervision(
            z_d_hat=z_d_hat,
            z_d_true=z_d_true,
            y_true=y,
            positive_only=positive_only,
        )
        loss = l_cycle + lambda_d * l_d

        if training:
            loss.backward()
            optimizer.step()

        bs = len(y)
        total += float(loss.item()) * bs
        total_c += float(l_cycle.item()) * bs
        total_d += float(l_d.item()) * bs
        count += bs

    return {
        "loss": total / max(count, 1),
        "loss_cycle": total_c / max(count, 1),
        "loss_d": total_d / max(count, 1),
    }


def main() -> None:
    """Stage B 主流程。"""
    args = parse_args()
    cfg = load_config(args.config)

    # 1) 初始化
    set_global_seed(int(cfg["seed"]))
    device = get_device(str(cfg["train"]["device"]))

    # 2) 加载并准备数据
    arrays = load_quadruplet_arrays(
        data_dir=cfg["paths"]["data_dir"],
        transform=str(cfg["data"]["transform"]),
        clr_eps=float(cfg["data"]["clr_eps"]),
    )
    taxa_names = _extract_taxa_names(arrays)
    graph = build_graph_from_config(cfg, taxa_names=taxa_names)

    x_pre_node = graph.to_node_features(arrays["R_pre"])
    x_d_node = graph.to_node_features(arrays["D"])
    x_post_node = graph.to_node_features(arrays["R_post"])

    ds = FMTQuadrupletDataset(
        r_pre=x_pre_node,
        donor=x_d_node,
        r_post=x_post_node,
        y=arrays["y"],
        recipient_id=arrays.get("recipient_id"),
        donor_id=arrays.get("donor_id"),
    )

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

    # 3) 构建模型
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

    inverse_model = InverseGenerator(
        z_dim=int(cfg["encoder"]["z_dim"]),
        hidden_dim=int(cfg["inverse"]["hidden_dim"]),
        dropout=float(cfg["inverse"]["dropout"]),
    ).to(device)

    # 4) 加载 Stage A 最优权重（Enc + fr）
    stage_a_ckpt = load_checkpoint(cfg["paths"]["stage_a_ckpt"], map_location=device)
    encoder.load_state_dict(stage_a_ckpt["encoder_state"])
    forward_model.load_state_dict(stage_a_ckpt["forward_state"])

    # 5) 冻结 Enc + fr，仅训练 fp
    _set_requires_grad(encoder, False)
    _set_requires_grad(forward_model, False)

    optimizer = torch.optim.Adam(
        inverse_model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["stage_b_epochs"])
    patience = int(cfg["train"]["early_stop_patience"])
    lambda_d = float(cfg["loss"]["lambda_d"])
    positive_only = bool(cfg["loss"]["donor_supervision_positive_only"])

    best_val = float("inf")
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    # 6) 训练循环 + 早停
    for epoch in tqdm(range(1, epochs + 1), desc="Stage B"):
        train_metrics = _run_epoch(
            encoder,
            forward_model,
            inverse_model,
            train_loader,
            optimizer,
            lambda_d=lambda_d,
            positive_only=positive_only,
            device=device,
            training=True,
        )
        val_metrics = _run_epoch(
            encoder,
            forward_model,
            inverse_model,
            val_loader,
            optimizer,
            lambda_d=lambda_d,
            positive_only=positive_only,
            device=device,
            training=False,
        )

        rec = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_loss_cycle": train_metrics["loss_cycle"],
            "train_loss_d": train_metrics["loss_d"],
            "val_loss": val_metrics["loss"],
            "val_loss_cycle": val_metrics["loss_cycle"],
            "val_loss_d": val_metrics["loss_d"],
        }
        history.append(rec)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            bad_epochs = 0
            save_checkpoint(
                cfg["paths"]["stage_b_ckpt"],
                {
                    "inverse_state": inverse_model.state_dict(),
                    "best_val": best_val,
                    "config": cfg,
                },
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    # 7) 保存日志
    logs_dir = ensure_dir(cfg["paths"]["logs_dir"])
    save_json(logs_dir / "stage_b_metrics.json", {"history": history, "best_val": best_val})

    print(f"Stage B finished. best_val={best_val:.6f}")
    print(f"Checkpoint: {cfg['paths']['stage_b_ckpt']}")


if __name__ == "__main__":
    main()
