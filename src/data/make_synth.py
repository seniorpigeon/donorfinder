"""生成 synthetic FMT 四元组数据与 FRN 先验文件。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np


def _dirichlet_matrix(rng: np.random.Generator, n: int, t: int, alpha: float) -> np.ndarray:
    """按 Dirichlet 分布生成 n 行、t 列的丰度矩阵。"""
    return rng.dirichlet(alpha=np.full(t, alpha, dtype=np.float64), size=n).astype(np.float32)


def _row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """对矩阵逐行归一化。"""
    rs = np.sum(x, axis=1, keepdims=True) + eps
    return (x / rs).astype(np.float32)


def _make_symmetric_dist(rng: np.random.Generator, n_taxa: int, latent_dim: int = 12) -> np.ndarray:
    """构造对称距离矩阵 dist_mat（值归一到 0~1）。"""
    # 先在隐空间采样每个 taxon 的坐标
    latent = rng.normal(0.0, 1.0, size=(n_taxa, latent_dim)).astype(np.float32)
    # 计算两两欧氏距离
    diff = latent[:, None, :] - latent[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    # 归一化到 [0,1]
    maxv = float(np.max(dist) + 1e-8)
    dist = (dist / maxv).astype(np.float32)
    np.fill_diagonal(dist, 0.0)
    return dist


def _make_frc_map(rng: np.random.Generator, taxa_names: np.ndarray, n_frc: int) -> Dict[str, str]:
    """随机生成 species->FRC 映射。"""
    assignments = rng.integers(low=0, high=n_frc, size=len(taxa_names))
    return {taxa: f"frc_{int(fid)}" for taxa, fid in zip(taxa_names, assignments)}


def _make_tree_edges(rng: np.random.Generator, n_frc: int) -> np.ndarray:
    """随机生成一个连通树（FRC 节点之间）。"""
    nodes = [f"frc_{i}" for i in range(n_frc)]
    edges = []

    # 通过“每个新节点连到一个旧节点”构造随机生成树
    for i in range(1, n_frc):
        parent = nodes[int(rng.integers(0, i))]
        child = nodes[i]
        edges.append((parent, child, 1.0))
    return np.asarray(edges, dtype=object)


def generate_synthetic_data(
    n_samples: int,
    n_taxa: int,
    n_donors: int,
    n_recipients: int,
    n_frc: int,
    alpha_pre: float,
    alpha_donor: float,
    noise_std: float,
    hidden_scale: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """生成 synthetic 四元组数据。

    生成逻辑：
    1) 采样 donor_bank 与 R_pre
    2) 用隐藏线性规则控制 donor 混入强度 gate
    3) 得到 R_post ≈ mix(R_pre, D) + noise
    4) 用另一条隐藏规则采样 y
    5) 同时生成 dist_mat / frc_map / frc_tree
    """
    rng = np.random.default_rng(seed)

    taxa_names = np.array([f"sp_{i:04d}" for i in range(n_taxa)])

    # 供体库及样本中的 donor 选择
    donor_bank = _dirichlet_matrix(rng, n_donors, n_taxa, alpha=alpha_donor)
    recipient_id = rng.integers(0, n_recipients, size=n_samples, endpoint=False)
    donor_id = rng.integers(0, n_donors, size=n_samples, endpoint=False)

    # pre 与 donor
    r_pre = _dirichlet_matrix(rng, n_samples, n_taxa, alpha=alpha_pre)
    d = donor_bank[donor_id]

    # 隐藏线性函数：控制 donor 对 post 的影响比例 gate
    w_mix = rng.normal(0.0, hidden_scale / np.sqrt(n_taxa), size=(n_taxa,)).astype(np.float32)
    b_mix = float(rng.normal(0.0, 0.2))

    delta = d - r_pre
    gate = 1.0 / (1.0 + np.exp(-(delta @ w_mix + b_mix)))
    gate = gate.astype(np.float32)

    # post = (1-gate)*pre + gate*donor + noise
    noise = rng.normal(0.0, noise_std, size=(n_samples, n_taxa)).astype(np.float32)
    r_post = (1.0 - gate[:, None]) * r_pre + gate[:, None] * d + noise
    r_post = np.clip(r_post, a_min=1e-8, a_max=None)
    r_post = _row_normalize(r_post)

    # 隐藏线性规则采样疗效 y
    w_y = rng.normal(0.0, hidden_scale / np.sqrt(n_taxa), size=(n_taxa,)).astype(np.float32)
    b_y = float(rng.normal(0.0, 0.2))
    p_y = 1.0 / (1.0 + np.exp(-(delta @ w_y + b_y)))
    y = rng.binomial(1, p=np.clip(p_y, 1e-4, 1 - 1e-4)).astype(np.float32)

    # 生成图先验
    dist_mat = _make_symmetric_dist(rng, n_taxa)
    frc_map = _make_frc_map(rng, taxa_names, n_frc)
    frc_tree_edges = _make_tree_edges(rng, n_frc)

    return {
        "R_pre": r_pre.astype(np.float32),
        "D": d.astype(np.float32),
        "R_post": r_post.astype(np.float32),
        "y": y.astype(np.float32),
        "recipient_id": recipient_id.astype(np.int64),
        "donor_id": donor_id.astype(np.int64),
        "donor_bank": donor_bank.astype(np.float32),
        "taxa_names": taxa_names,
        "dist_mat": dist_mat.astype(np.float32),
        "frc_tree_edges": frc_tree_edges,
        "frc_map": frc_map,
    }


def save_synthetic_bundle(bundle: Dict[str, np.ndarray], out_dir: str | Path) -> None:
    """把 synthetic 数据与先验文件写到磁盘。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 保存四元组核心数组
    np.save(out / "R_pre.npy", bundle["R_pre"])
    np.save(out / "D.npy", bundle["D"])
    np.save(out / "R_post.npy", bundle["R_post"])
    np.save(out / "y.npy", bundle["y"])
    np.save(out / "recipient_id.npy", bundle["recipient_id"])
    np.save(out / "donor_id.npy", bundle["donor_id"])
    np.save(out / "donor_bank.npy", bundle["donor_bank"])

    taxa_names = bundle["taxa_names"]
    (out / "taxa_names.txt").write_text("\n".join(taxa_names.tolist()) + "\n", encoding="utf-8")

    # 保存 dist_mat（带行列名）
    dist = bundle["dist_mat"]
    with (out / "dist_mat.tsv").open("w", encoding="utf-8") as f:
        f.write("taxon\t" + "\t".join(taxa_names.tolist()) + "\n")
        for i, name in enumerate(taxa_names):
            row = "\t".join(f"{float(v):.6f}" for v in dist[i])
            f.write(f"{name}\t{row}\n")

    # 保存 species->FRC 映射
    with (out / "frc_map.tsv").open("w", encoding="utf-8") as f:
        f.write("species\tfrc\n")
        for taxon in taxa_names:
            f.write(f"{taxon}\t{bundle['frc_map'][taxon]}\n")

    # 保存 FRC 树边
    with (out / "frc_tree_edges.tsv").open("w", encoding="utf-8") as f:
        f.write("parent\tchild\tweight\n")
        for parent, child, weight in bundle["frc_tree_edges"]:
            f.write(f"{parent}\t{child}\t{float(weight):.6f}\n")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Generate synthetic FMT quadruplets + FRN priors.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional override for output directory.")
    return parser.parse_args()


def main() -> None:
    """脚本主函数：读取配置 -> 生成数据 -> 保存文件。"""
    import yaml

    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    synth = cfg["synthetic"]
    out_dir = args.out_dir or cfg["paths"]["data_dir"]

    bundle = generate_synthetic_data(
        n_samples=int(synth["n_samples"]),
        n_taxa=int(synth["n_taxa"]),
        n_donors=int(synth["n_donors"]),
        n_recipients=int(synth["n_recipients"]),
        n_frc=int(synth["n_frc"]),
        alpha_pre=float(synth["dirichlet_alpha_pre"]),
        alpha_donor=float(synth["dirichlet_alpha_donor"]),
        noise_std=float(synth["noise_std"]),
        hidden_scale=float(synth["hidden_scale"]),
        seed=int(cfg["seed"]),
    )
    save_synthetic_bundle(bundle, out_dir)
    print(f"Synthetic data written to: {out_dir}")


if __name__ == "__main__":
    main()
