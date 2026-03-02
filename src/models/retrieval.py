"""供体库 embedding 缓存与相似度检索工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from src.graphs.build_graph import GraphBundle


@torch.no_grad()
def encode_abundance_to_embedding(
    encoder: torch.nn.Module,
    abundance: np.ndarray,
    graph: GraphBundle,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """把丰度矩阵编码成样本 embedding。

    参数说明：
    - abundance: (N, T) 物种丰度矩阵（已做预处理）
    - graph: 图结构对象，负责 species->node 特征变换
    - 返回值: (N, z_dim)
    """
    encoder.eval()

    # 先把 species 丰度映射到图节点特征空间
    node_x = graph.to_node_features(abundance)
    n = node_x.shape[0]
    outs = []

    # 分批编码，避免一次性占用过多显存/内存
    for st in range(0, n, batch_size):
        ed = min(st + batch_size, n)
        xb = torch.tensor(node_x[st:ed], dtype=torch.float32, device=device)
        z = encoder(xb)
        outs.append(z.detach().cpu().numpy())

    return np.concatenate(outs, axis=0).astype(np.float32)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """按行做 L2 归一化，供余弦相似度计算使用。"""
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return (x / n).astype(np.float32)


def build_or_load_donor_cache(
    cache_file: str | Path,
    donor_bank_abundance: np.ndarray,
    encoder: torch.nn.Module,
    graph: GraphBundle,
    device: torch.device,
    donor_ids: Optional[np.ndarray] = None,
    batch_size: int = 64,
    force_rebuild: bool = False,
) -> Dict[str, np.ndarray]:
    """构建或加载供体 embedding 缓存。

    缓存内容：
    - z_donor_bank: (N_donor, z_dim)
    - donor_ids:    (N_donor,)
    """
    cache_path = Path(cache_file)

    # 默认优先复用已存在缓存，加快推理速度
    if cache_path.exists() and not force_rebuild:
        payload = np.load(cache_path)
        return {
            "z_donor_bank": payload["z_donor_bank"].astype(np.float32),
            "donor_ids": payload["donor_ids"].astype(np.int64),
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 从原始供体丰度编码得到供体 embedding
    z = encode_abundance_to_embedding(
        encoder=encoder,
        abundance=donor_bank_abundance,
        graph=graph,
        device=device,
        batch_size=batch_size,
    )
    # 统一归一化，后续点积可直接视为余弦
    z = _l2_normalize(z)

    if donor_ids is None:
        donor_ids = np.arange(len(z), dtype=np.int64)
    else:
        donor_ids = donor_ids.astype(np.int64)

    np.savez(cache_path, z_donor_bank=z.astype(np.float32), donor_ids=donor_ids)
    return {"z_donor_bank": z, "donor_ids": donor_ids}


def cosine_similarity_matrix(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """计算 query 与 bank 的余弦相似度矩阵。"""
    q = _l2_normalize(np.asarray(query, dtype=np.float32))
    b = _l2_normalize(np.asarray(bank, dtype=np.float32))
    return (q @ b.T).astype(np.float32)


def retrieve_topk(z_query: np.ndarray, z_bank: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """检索每个 query 的 Top-K 候选供体。

    返回：
    - top_idx:    (B, K) 供体下标
    - top_scores: (B, K) 相似度得分
    """
    sims = cosine_similarity_matrix(z_query, z_bank)
    k = min(top_k, z_bank.shape[0])

    # argpartition 先取无序 top-k，再在 top-k 内部排序，效率高于全量 argsort
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    row = np.arange(sims.shape[0])[:, None]
    part_scores = sims[row, idx]

    order = np.argsort(-part_scores, axis=1)
    top_idx = idx[row, order]
    top_scores = part_scores[row, order]
    return top_idx.astype(np.int64), top_scores.astype(np.float32)
