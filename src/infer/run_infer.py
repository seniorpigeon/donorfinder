"""推理脚本：给定新受体 pre-FMT，输出 Top-K 候选供体。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.data.dataset_fmt import load_quadruplet_arrays
from src.graphs.build_graph import build_graph_from_config
from src.models.encoder_gnn import SampleEncoder
from src.models.forward_fr import ForwardSimulator
from src.models.inverse_fp import InverseGenerator
from src.models.retrieval import build_or_load_donor_cache, cosine_similarity_matrix
from src.train.utils import ensure_dir, get_device, load_checkpoint, load_config, set_global_seed


def parse_args() -> argparse.Namespace:
    """解析推理参数。"""
    parser = argparse.ArgumentParser(description="Run donor retrieval inference for a new recipient pre-FMT sample.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--recipient_index", type=int, default=0)
    parser.add_argument("--target_mode", type=str, default=None, choices=[None, "prototype_healthy", "reference_post"])
    parser.add_argument("--reference_post_index", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--force_rebuild_cache", action="store_true")
    return parser.parse_args()


def _extract_taxa_names(arrays: Dict[str, object]) -> List[str]:
    """提取 taxon 名称。"""
    if "taxa_names" in arrays:
        return [str(x) for x in arrays["taxa_names"].tolist()]
    t = arrays["R_pre"].shape[1]
    return [f"sp_{i:04d}" for i in range(t)]


def _get_donor_bank(arrays: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """获取供体库丰度矩阵与 donor_id 列表。"""
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


@torch.no_grad()
def _compute_proto_if_missing(
    encoder: SampleEncoder,
    x_post_node: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """当 prototype 文件不存在时在线计算健康原型。"""
    pos = np.where(y > 0.5)[0]
    if len(pos) == 0:
        pos = np.arange(len(y))

    encoder.eval()
    zs = []
    for st in range(0, len(pos), batch_size):
        ed = min(st + batch_size, len(pos))
        xb = torch.tensor(x_post_node[pos[st:ed]], dtype=torch.float32, device=device)
        zs.append(encoder(xb).cpu().numpy())
    return np.mean(np.concatenate(zs, axis=0), axis=0).astype(np.float32)


def _load_models(cfg, graph, device: torch.device):
    """创建模型并加载 Stage C checkpoint。"""
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

    stage_c_path = Path(cfg["paths"]["stage_c_ckpt"])
    if stage_c_path.exists():
        ckpt = load_checkpoint(stage_c_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state"])
        forward_model.load_state_dict(ckpt["forward_state"])
        inverse_model.load_state_dict(ckpt["inverse_state"])
    else:
        raise FileNotFoundError(
            f"Stage C checkpoint not found: {stage_c_path}. Run train_stage_c.py before inference."
        )

    encoder.eval()
    forward_model.eval()
    inverse_model.eval()

    return encoder, forward_model, inverse_model


def main() -> None:
    """推理主流程。"""
    args = parse_args()
    cfg = load_config(args.config)

    # 1) 初始化
    set_global_seed(int(cfg["seed"]))
    device = get_device(str(cfg["train"]["device"]))

    # 2) 加载数据并构图
    arrays = load_quadruplet_arrays(
        data_dir=cfg["paths"]["data_dir"],
        transform=str(cfg["data"]["transform"]),
        clr_eps=float(cfg["data"]["clr_eps"]),
    )
    taxa_names = _extract_taxa_names(arrays)
    graph = build_graph_from_config(cfg, taxa_names=taxa_names)

    # 3) 加载训练好的 Enc/fr/fp
    encoder, forward_model, inverse_model = _load_models(cfg, graph, device)

    # 4) 加载或构建 donor bank embedding 缓存
    donor_bank_raw, donor_ids = _get_donor_bank(arrays)
    cache = build_or_load_donor_cache(
        cache_file=cfg["paths"]["donor_cache_file"],
        donor_bank_abundance=donor_bank_raw,
        encoder=encoder,
        graph=graph,
        device=device,
        donor_ids=donor_ids,
        batch_size=int(cfg["inference"]["batch_size"]),
        force_rebuild=bool(args.force_rebuild_cache),
    )

    z_bank = cache["z_donor_bank"].astype(np.float32)
    donor_ids = cache["donor_ids"].astype(np.int64)

    # 5) 选择一个“新受体 pre 样本”
    ridx = int(args.recipient_index)
    if ridx < 0 or ridx >= len(arrays["R_pre"]):
        raise IndexError(f"recipient_index out of range: {ridx}")

    x_pre_node = graph.to_node_features(arrays["R_pre"][ridx : ridx + 1])
    x_pre_t = torch.tensor(x_pre_node, dtype=torch.float32, device=device)
    z_pre = encoder(x_pre_t)

    # 6) 选择 target：prototype_healthy 或指定 reference_post
    target_mode = args.target_mode or str(cfg["inference"]["target_mode"])

    if target_mode == "prototype_healthy":
        proto_path = Path(cfg["paths"]["logs_dir"]) / "prototype_healthy.npy"
        if proto_path.exists():
            z_target_np = np.load(proto_path).astype(np.float32)
        else:
            # 若 prototype 文件缺失，临时在线计算
            x_post_node = graph.to_node_features(arrays["R_post"])
            z_target_np = _compute_proto_if_missing(
                encoder=encoder,
                x_post_node=x_post_node,
                y=arrays["y"],
                device=device,
                batch_size=int(cfg["inference"]["batch_size"]),
            )
        z_target = torch.tensor(z_target_np[None, :], dtype=torch.float32, device=device)
    elif target_mode == "reference_post":
        ref_idx = (
            int(args.reference_post_index)
            if args.reference_post_index is not None
            else int(cfg["inference"].get("reference_post_index", 0) or 0)
        )
        if ref_idx < 0 or ref_idx >= len(arrays["R_post"]):
            raise IndexError(f"reference_post_index out of range: {ref_idx}")
        x_ref = graph.to_node_features(arrays["R_post"][ref_idx : ref_idx + 1])
        z_target = encoder(torch.tensor(x_ref, dtype=torch.float32, device=device))
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    # 7) 逆向生成理想供体 embedding
    z_d_hat = inverse_model(z_pre, z_target)
    z_d_hat_np = z_d_hat.detach().cpu().numpy().astype(np.float32)

    # 8) 与 donor bank 做余弦相似度
    sim = cosine_similarity_matrix(z_d_hat_np, z_bank).reshape(-1)

    # 9) 对 donor bank 每个候选跑 forward，得到 y_pred 与 z_post_pred
    y_preds = []
    z_post_preds = []
    batch_size = int(cfg["inference"]["batch_size"])
    z_pre_np = z_pre.detach().cpu().numpy().astype(np.float32)

    for st in range(0, len(z_bank), batch_size):
        ed = min(st + batch_size, len(z_bank))

        # 批量候选 donor embedding
        zd = torch.tensor(z_bank[st:ed], dtype=torch.float32, device=device)
        # 把同一个 z_pre 复制到 batch 大小
        zp = torch.tensor(np.repeat(z_pre_np, repeats=ed - st, axis=0), dtype=torch.float32, device=device)

        z_post_pred, y_logit = forward_model(zp, zd)
        y_prob = torch.sigmoid(y_logit)

        y_preds.append(y_prob.detach().cpu().numpy())
        z_post_preds.append(z_post_pred.detach().cpu().numpy())

    y_preds_np = np.concatenate(y_preds, axis=0).astype(np.float32)
    z_post_preds_np = np.concatenate(z_post_preds, axis=0).astype(np.float32)

    # 10) 组合评分（可选）
    alpha = float(args.alpha if args.alpha is not None else cfg["inference"]["alpha"])
    score = alpha * sim + (1.0 - alpha) * y_preds_np

    # 11) 按 sim 排序输出 Top-K
    top_k = int(args.top_k if args.top_k is not None else cfg["inference"]["top_k"])
    k = min(top_k, len(z_bank))
    top_idx = np.argsort(-sim)[:k]

    result_df = pd.DataFrame(
        {
            "rank": np.arange(1, k + 1),
            "donor_id": donor_ids[top_idx],
            "sim": sim[top_idx],
            "y_pred": y_preds_np[top_idx],
            "score": score[top_idx],
        }
    )

    # 12) 落盘输出
    logs_dir = ensure_dir(cfg["paths"]["logs_dir"])
    out_file = Path(cfg["paths"]["topk_output_file"])
    ensure_dir(out_file.parent)
    result_df.to_csv(out_file, index=False)

    np.save(logs_dir / "topk_z_post_pred.npy", z_post_preds_np[top_idx])
    np.save(logs_dir / "z_d_hat.npy", z_d_hat_np)

    # 13) 终端打印结果
    print("Top-K donors by cosine similarity:")
    for _, row in result_df.iterrows():
        print(
            f"rank={int(row['rank'])} donor_id={int(row['donor_id'])} "
            f"sim={float(row['sim']):.4f} y_pred={float(row['y_pred']):.4f} score={float(row['score']):.4f}"
        )

    print(f"Saved ranking: {out_file}")


if __name__ == "__main__":
    main()
