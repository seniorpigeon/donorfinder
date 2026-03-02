"""Stage C：联合微调 Enc + fr + fp，并导出 donor cache。

目标损失：
  L = L_post + lambda_y*L_y + lambda_c*L_cycle + lambda_d*L_d
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.data.dataset_fmt import FMTQuadrupletDataset, load_quadruplet_arrays, make_dataloaders
from src.graphs.build_graph import build_graph_from_config
from src.models.encoder_gnn import SampleEncoder
from src.models.forward_fr import ForwardSimulator
from src.models.inverse_fp import InverseGenerator
from src.models.retrieval import build_or_load_donor_cache
from src.train.losses import loss_cycle, loss_donor_supervision, loss_post, loss_y
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
    parser = argparse.ArgumentParser(description="Stage C joint finetuning.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def _extract_taxa_names(arrays: Dict[str, object]) -> List[str]:
    """提取 taxon 名称。"""
    if "taxa_names" in arrays:
        return [str(x) for x in arrays["taxa_names"].tolist()]
    t = arrays["R_pre"].shape[1]
    return [f"sp_{i:04d}" for i in range(t)]


def _run_epoch(
    encoder: SampleEncoder,
    forward_model: ForwardSimulator,
    inverse_model: InverseGenerator,
    loader,
    optimizer,
    lambdas: Tuple[float, float, float],
    positive_only: bool,
    device: torch.device,
    training: bool,
) -> Dict[str, float]:
    """执行一个 Stage C epoch。"""
    lambda_y, lambda_c, lambda_d = lambdas

    # Stage C 联合训练：三者都参与（验证时切 eval）
    encoder.train(mode=training)
    forward_model.train(mode=training)
    inverse_model.train(mode=training)

    total = 0.0
    total_post = 0.0
    total_y = 0.0
    total_c = 0.0
    total_d = 0.0
    count = 0

    for batch in loader:
        x_pre = batch["R_pre"].to(device)
        x_d = batch["D"].to(device)
        x_post = batch["R_post"].to(device)
        y = batch["y"].to(device)

        if training:
            optimizer.zero_grad()

        # 1) 编码
        z_pre = encoder(x_pre)
        z_d = encoder(x_d)
        z_post_true = encoder(x_post)

        # 2) 前向预测 + 逆向生成
        z_post_pred, y_logit = forward_model(z_pre, z_d)
        z_d_hat = inverse_model(z_pre, z_post_true)

        # 3) 四项损失
        l_post = loss_post(z_post_pred, z_post_true)
        l_y = loss_y(y_logit, y)
        l_c = loss_cycle(forward_model, z_pre, z_d_hat, z_post_true)
        l_d = loss_donor_supervision(z_d_hat, z_d, y_true=y, positive_only=positive_only)

        # 4) 加权总损失
        loss = l_post + lambda_y * l_y + lambda_c * l_c + lambda_d * l_d

        if training:
            loss.backward()
            optimizer.step()

        bs = len(y)
        total += float(loss.item()) * bs
        total_post += float(l_post.item()) * bs
        total_y += float(l_y.item()) * bs
        total_c += float(l_c.item()) * bs
        total_d += float(l_d.item()) * bs
        count += bs

    return {
        "loss": total / max(count, 1),
        "loss_post": total_post / max(count, 1),
        "loss_y": total_y / max(count, 1),
        "loss_cycle": total_c / max(count, 1),
        "loss_d": total_d / max(count, 1),
    }


@torch.no_grad()
def _compute_healthy_prototype(
    encoder: SampleEncoder,
    x_post_all: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """计算 y=1 的 post embedding 均值，作为默认目标。"""
    encoder.eval()
    pos_idx = np.where(y > 0.5)[0]
    if len(pos_idx) == 0:
        pos_idx = np.arange(len(y))

    zs = []
    for st in range(0, len(pos_idx), batch_size):
        ed = min(st + batch_size, len(pos_idx))
        xb = torch.tensor(x_post_all[pos_idx[st:ed]], dtype=torch.float32, device=device)
        zs.append(encoder(xb).cpu().numpy())

    z_pos = np.concatenate(zs, axis=0).astype(np.float32)
    return np.mean(z_pos, axis=0).astype(np.float32)


def _get_donor_bank(arrays: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """从数据中提取 donor bank 与 donor_id。"""
    if "donor_bank" in arrays:
        db = arrays["donor_bank"].astype(np.float32)
        ids = np.arange(len(db), dtype=np.int64)
        return db, ids

    if "donor_id" in arrays:
        donor_id = arrays["donor_id"].astype(np.int64)
        uniq, idx = np.unique(donor_id, return_index=True)
        return arrays["D"][idx].astype(np.float32), uniq.astype(np.int64)

    d = arrays["D"].astype(np.float32)
    return d, np.arange(len(d), dtype=np.int64)


def main() -> None:
    """Stage C 主流程。"""
    args = parse_args()
    cfg = load_config(args.config)

    # 1) 初始化
    set_global_seed(int(cfg["seed"]))
    device = get_device(str(cfg["train"]["device"]))

    # 2) 数据加载 + 构图 + node 特征转换
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

    # 3) 模型实例化
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

    # 4) 加载 Stage A/B 最优权重作为 Stage C 初始点
    stage_a_ckpt = load_checkpoint(cfg["paths"]["stage_a_ckpt"], map_location=device)
    stage_b_ckpt = load_checkpoint(cfg["paths"]["stage_b_ckpt"], map_location=device)

    encoder.load_state_dict(stage_a_ckpt["encoder_state"])
    forward_model.load_state_dict(stage_a_ckpt["forward_state"])
    inverse_model.load_state_dict(stage_b_ckpt["inverse_state"])

    # 5) 联合优化三模块
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(forward_model.parameters()) + list(inverse_model.parameters()),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["stage_c_epochs"])
    patience = int(cfg["train"]["early_stop_patience"])
    lambdas = (
        float(cfg["loss"]["lambda_y"]),
        float(cfg["loss"]["lambda_c"]),
        float(cfg["loss"]["lambda_d"]),
    )
    positive_only = bool(cfg["loss"]["donor_supervision_positive_only"])

    best_val = float("inf")
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    # 6) 训练循环 + 最优模型保存
    for epoch in tqdm(range(1, epochs + 1), desc="Stage C"):
        train_metrics = _run_epoch(
            encoder,
            forward_model,
            inverse_model,
            train_loader,
            optimizer,
            lambdas=lambdas,
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
            lambdas=lambdas,
            positive_only=positive_only,
            device=device,
            training=False,
        )

        rec = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_loss_post": train_metrics["loss_post"],
            "train_loss_y": train_metrics["loss_y"],
            "train_loss_cycle": train_metrics["loss_cycle"],
            "train_loss_d": train_metrics["loss_d"],
            "val_loss": val_metrics["loss"],
            "val_loss_post": val_metrics["loss_post"],
            "val_loss_y": val_metrics["loss_y"],
            "val_loss_cycle": val_metrics["loss_cycle"],
            "val_loss_d": val_metrics["loss_d"],
        }
        history.append(rec)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            bad_epochs = 0
            save_checkpoint(
                cfg["paths"]["stage_c_ckpt"],
                {
                    "encoder_state": encoder.state_dict(),
                    "forward_state": forward_model.state_dict(),
                    "inverse_state": inverse_model.state_dict(),
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

    # 7) 重新加载 best checkpoint，确保导出的是最优参数
    best_ckpt = load_checkpoint(cfg["paths"]["stage_c_ckpt"], map_location=device)
    encoder.load_state_dict(best_ckpt["encoder_state"])
    forward_model.load_state_dict(best_ckpt["forward_state"])
    inverse_model.load_state_dict(best_ckpt["inverse_state"])

    # 8) 保存训练日志
    logs_dir = ensure_dir(cfg["paths"]["logs_dir"])
    save_json(logs_dir / "stage_c_metrics.json", {"history": history, "best_val": best_val})

    # 9) 保存 prototype_healthy，推理默认使用
    proto = _compute_healthy_prototype(
        encoder=encoder,
        x_post_all=x_post_node,
        y=arrays["y"],
        device=device,
        batch_size=int(cfg["inference"]["batch_size"]),
    )
    np.save(logs_dir / "prototype_healthy.npy", proto.astype(np.float32))

    # 10) 编码 donor bank 并保存缓存
    donor_bank_raw, donor_bank_ids = _get_donor_bank(arrays)
    cache = build_or_load_donor_cache(
        cache_file=cfg["paths"]["donor_cache_file"],
        donor_bank_abundance=donor_bank_raw,
        encoder=encoder,
        graph=graph,
        device=device,
        donor_ids=donor_bank_ids,
        batch_size=int(cfg["inference"]["batch_size"]),
        force_rebuild=True,
    )

    print(f"Stage C finished. best_val={best_val:.6f}")
    print(f"Checkpoint: {cfg['paths']['stage_c_ckpt']}")
    print(
        f"Donor cache: {cfg['paths']['donor_cache_file']} "
        f"(N={len(cache['donor_ids'])}, z_dim={cache['z_donor_bank'].shape[1]})"
    )


if __name__ == "__main__":
    main()
