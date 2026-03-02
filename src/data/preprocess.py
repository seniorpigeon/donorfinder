"""真实数据预处理工具：读取、对齐、角色配对与变换。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# 统一角色同义词，便于兼容不同数据源命名
ROLE_PRE = {"pre", "recipient_pre", "r_pre", "pre_fmt"}
ROLE_DONOR = {"donor", "d", "d_pre"}
ROLE_POST = {"post", "recipient_post", "r_post", "post_fmt"}


def apply_abundance_transform(x: np.ndarray, method: str = "log1p", clr_eps: float = 1e-6) -> np.ndarray:
    """对丰度矩阵做变换。

    - log1p: log(1+x)，稳健且易用
    - clr:   中心化对数比变换（加入 pseudocount）
    """
    x = np.asarray(x, dtype=np.float32)
    if method == "log1p":
        return np.log1p(x).astype(np.float32)

    if method == "clr":
        # CLR 需要正值，先裁剪并加伪计数
        xp = np.clip(x, a_min=0.0, a_max=None) + float(clr_eps)
        logx = np.log(xp)
        gm = np.mean(logx, axis=1, keepdims=True)
        return (logx - gm).astype(np.float32)

    raise ValueError(f"Unsupported transform: {method}")


def row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """按行归一化，让每个样本和为1。"""
    x = np.asarray(x, dtype=np.float32)
    rs = np.sum(x, axis=1, keepdims=True) + eps
    return (x / rs).astype(np.float32)


def read_table(path: str | Path, index_col: int | str = 0) -> pd.DataFrame:
    """读取 CSV/TSV 表格。"""
    p = Path(path)
    sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(p, sep=sep, index_col=index_col)


def align_taxa_tables(tables: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], List[str]]:
    """多个 abundance 表按共同 taxa 对齐列顺序。"""
    if not tables:
        raise ValueError("No tables provided for taxa alignment.")

    # 求列名交集
    common = set(tables[0].columns)
    for t in tables[1:]:
        common &= set(t.columns)

    taxa = sorted(common)
    aligned = [t.loc[:, taxa].copy() for t in tables]
    return aligned, taxa


def build_pairs_from_metadata(abundance: pd.DataFrame, metadata: pd.DataFrame) -> Dict[str, np.ndarray]:
    """从 metadata 构建 (R_pre, D, R_post, y) 配对样本。

    metadata 必须含列：
    - sample_id
    - pair_id
    - role
    - outcome
    """
    required = {"sample_id", "pair_id", "role", "outcome"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")

    meta = metadata.copy()
    meta["role"] = meta["role"].astype(str).str.lower()

    rows_pre: List[np.ndarray] = []
    rows_d: List[np.ndarray] = []
    rows_post: List[np.ndarray] = []
    ys: List[float] = []
    pair_ids: List[str] = []

    # 对每个 pair_id 查找 pre/donor/post 三种角色
    for pair_id, grp in meta.groupby("pair_id"):
        pre = grp[grp["role"].isin(ROLE_PRE)]
        donor = grp[grp["role"].isin(ROLE_DONOR)]
        post = grp[grp["role"].isin(ROLE_POST)]

        # 任一角色缺失则跳过
        if pre.empty or donor.empty or post.empty:
            continue

        sid_pre = str(pre.iloc[0]["sample_id"])
        sid_donor = str(donor.iloc[0]["sample_id"])
        sid_post = str(post.iloc[0]["sample_id"])

        # 保证 abundance 中存在这些 sample_id
        if sid_pre not in abundance.index or sid_donor not in abundance.index or sid_post not in abundance.index:
            continue

        rows_pre.append(abundance.loc[sid_pre].to_numpy(dtype=np.float32))
        rows_d.append(abundance.loc[sid_donor].to_numpy(dtype=np.float32))
        rows_post.append(abundance.loc[sid_post].to_numpy(dtype=np.float32))

        y_val = float(grp.iloc[0]["outcome"])
        ys.append(y_val)
        pair_ids.append(str(pair_id))

    if not rows_pre:
        raise ValueError("No valid (pre, donor, post) pairs found in metadata.")

    return {
        "R_pre": np.stack(rows_pre).astype(np.float32),
        "D": np.stack(rows_d).astype(np.float32),
        "R_post": np.stack(rows_post).astype(np.float32),
        "y": np.asarray(ys, dtype=np.float32),
        "pair_id": np.asarray(pair_ids),
    }


def load_real_data_interface(
    abundance_path: str | Path,
    metadata_path: str | Path,
    transform: str = "log1p",
    clr_eps: float = 1e-6,
) -> Dict[str, np.ndarray | List[str]]:
    """真实数据入口（占位实现）。

    功能：
    1) 读取 abundance + metadata
    2) 按 metadata 配对成四元组
    3) 应用丰度变换
    4) 输出训练可用数组
    """
    abundance = read_table(abundance_path, index_col=0)
    metadata = read_table(metadata_path, index_col=None)

    # 只保留数值型 taxa 列，避免混入非数值字段
    abundance = abundance.select_dtypes(include=[np.number]).astype(np.float32)

    pairs = build_pairs_from_metadata(abundance, metadata)

    pairs["R_pre"] = apply_abundance_transform(pairs["R_pre"], method=transform, clr_eps=clr_eps)
    pairs["D"] = apply_abundance_transform(pairs["D"], method=transform, clr_eps=clr_eps)
    pairs["R_post"] = apply_abundance_transform(pairs["R_post"], method=transform, clr_eps=clr_eps)
    pairs["taxa_names"] = abundance.columns.tolist()

    return pairs
