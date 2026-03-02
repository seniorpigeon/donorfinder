"""根据配置构建 FRN 图结构（frc/species 两种模式）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .frn_utils import TreeEdge, load_dist_matrix, load_frc_map, load_tree_edges


@dataclass
class GraphBundle:
    """图结构封装。

    字段说明：
    - edge_index: shape (2, E)
    - edge_weight: shape (E,)
    - taxa_to_node: shape (T, N)，用于 species->node 聚合
    """

    mode: str
    node_names: List[str]
    edge_index: np.ndarray
    edge_weight: np.ndarray
    taxa_names: List[str]
    taxa_to_node: np.ndarray

    def num_nodes(self) -> int:
        """返回图节点数。"""
        return len(self.node_names)

    def to_node_features(self, x_species: np.ndarray) -> np.ndarray:
        """把 species 丰度映射到图节点特征。

        x_species: (B, T)
        taxa_to_node: (T, N)
        输出: (B, N)
        """
        if x_species.ndim != 2:
            raise ValueError(f"x_species must be 2D, got shape={x_species.shape}")
        return (x_species @ self.taxa_to_node).astype(np.float32)

    def build_dense_adj(self, add_self_loop: bool = True) -> np.ndarray:
        """构建并行归一化后的稠密邻接矩阵（供 dense GNN 使用）。"""
        n = self.num_nodes()
        adj = np.zeros((n, n), dtype=np.float32)

        # 累加边权
        for (u, v), w in zip(self.edge_index.T, self.edge_weight):
            adj[int(u), int(v)] += float(w)

        # 可选添加自环
        if add_self_loop:
            adj += np.eye(n, dtype=np.float32)

        # 行归一化
        row_sum = np.sum(adj, axis=1, keepdims=True) + 1e-12
        return (adj / row_sum).astype(np.float32)


def _resolve_path(path: str | Path) -> Path:
    """把相对路径解析为绝对路径。"""
    p = Path(path)
    return p if p.is_absolute() else Path.cwd() / p


def _coerce_undirected(edges: Sequence[Tuple[int, int, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """把边集转成显式无向边（u->v 与 v->u 都保留），并按 key 去重。"""
    d: Dict[Tuple[int, int], float] = {}

    for u, v, w in edges:
        key_uv = (int(u), int(v))
        key_vu = (int(v), int(u))
        # 同一方向重复边时保留更大权重
        d[key_uv] = max(d.get(key_uv, 0.0), float(w))
        d[key_vu] = max(d.get(key_vu, 0.0), float(w))

    idx = np.asarray([[k[0], k[1]] for k in d.keys()], dtype=np.int64).T
    wt = np.asarray([d[k] for k in d.keys()], dtype=np.float32)
    return idx, wt


def _build_frc_graph(
    taxa_names: Sequence[str],
    map_path: str | Path,
    tree_path: str | Path,
    tree_format: str,
    default_weight: float,
) -> GraphBundle:
    """FRC 模式构图：先聚合，再基于 FRC 树连边。"""
    frc_map = load_frc_map(_resolve_path(map_path))
    tree_edges: List[TreeEdge] = load_tree_edges(
        _resolve_path(tree_path),
        tree_format=tree_format,
        default_weight=default_weight,
    )

    # 每个 species 找到所属 FRC；缺失映射时给一个独立 fallback 节点
    species_to_frc: Dict[str, str] = {}
    for i, sp in enumerate(taxa_names):
        species_to_frc[sp] = frc_map.get(sp, f"unmapped_{i}")

    frc_labels = sorted(set(species_to_frc.values()))
    frc_to_idx = {name: i for i, name in enumerate(frc_labels)}

    # 构建 species->FRC 聚合矩阵 (T, N_frc)
    taxa_to_node = np.zeros((len(taxa_names), len(frc_labels)), dtype=np.float32)
    for i, sp in enumerate(taxa_names):
        taxa_to_node[i, frc_to_idx[species_to_frc[sp]]] = 1.0

    # 根据树边构建 FRC 图边
    edges_tmp: List[Tuple[int, int, float]] = []
    for e in tree_edges:
        if e.parent in frc_to_idx and e.child in frc_to_idx:
            edges_tmp.append((frc_to_idx[e.parent], frc_to_idx[e.child], float(e.weight)))

    # 若树边无法对上 FRC 标签，退化为链式图，保证图连通可训练
    if not edges_tmp and len(frc_labels) > 1:
        for i in range(1, len(frc_labels)):
            edges_tmp.append((i - 1, i, 1.0))

    edge_index, edge_weight = _coerce_undirected(edges_tmp)

    return GraphBundle(
        mode="frc",
        node_names=frc_labels,
        edge_index=edge_index,
        edge_weight=edge_weight,
        taxa_names=list(taxa_names),
        taxa_to_node=taxa_to_node,
    )


def _build_species_graph(
    taxa_names: Sequence[str],
    dist_mat_path: str | Path,
    knn_k: int,
    sigma: float | None,
) -> GraphBundle:
    """Species 模式构图：基于 dist_mat 做 kNN 连边。"""
    dist_taxa, dist = load_dist_matrix(_resolve_path(dist_mat_path))

    # 优先按 taxon 名重排 dist 矩阵
    if set(taxa_names).issubset(set(dist_taxa)):
        pos = {name: i for i, name in enumerate(dist_taxa)}
        idx = [pos[t] for t in taxa_names]
        dist = dist[np.ix_(idx, idx)]
    elif len(taxa_names) != dist.shape[0]:
        raise ValueError(
            "Species graph build failed: taxa_names are not aligned with dist matrix and shapes differ."
        )

    n = len(taxa_names)

    # 默认 sigma 取非零距离中位数
    if sigma is None:
        nz = dist[dist > 0]
        sigma = float(np.median(nz)) if nz.size > 0 else 1.0
        sigma = max(sigma, 1e-6)

    edges: List[Tuple[int, int, float]] = []
    k = max(1, min(knn_k, n - 1)) if n > 1 else 1

    for i in range(n):
        di = dist[i].copy()
        di[i] = np.inf  # 排除自己
        nn_idx = np.argpartition(di, kth=k - 1)[:k]
        for j in nn_idx:
            # 权重：exp(-dist/sigma)
            w = float(np.exp(-float(dist[i, j]) / sigma))
            edges.append((i, int(j), w))

    edge_index, edge_weight = _coerce_undirected(edges)

    return GraphBundle(
        mode="species",
        node_names=list(taxa_names),
        edge_index=edge_index,
        edge_weight=edge_weight,
        taxa_names=list(taxa_names),
        # species 模式节点即 taxon，本质是单位映射
        taxa_to_node=np.eye(n, dtype=np.float32),
    )


def build_graph_from_config(cfg: Dict[str, object], taxa_names: Sequence[str]) -> GraphBundle:
    """统一构图入口：按 config 中 graph_mode 选择 frc/species。"""
    graph_cfg = cfg["graph"]
    mode = graph_cfg["graph_mode"]

    if mode == "frc":
        frc_cfg = graph_cfg["frc"]
        return _build_frc_graph(
            taxa_names=taxa_names,
            map_path=frc_cfg["map_path"],
            tree_path=frc_cfg["tree_path"],
            tree_format=frc_cfg.get("tree_format", "edges"),
            default_weight=float(frc_cfg.get("default_weight", 1.0)),
        )

    if mode == "species":
        sp_cfg = graph_cfg["species"]
        return _build_species_graph(
            taxa_names=taxa_names,
            dist_mat_path=sp_cfg["dist_mat_path"],
            knn_k=int(sp_cfg.get("knn_k", 10)),
            sigma=sp_cfg.get("sigma", None),
        )

    raise ValueError(f"Unsupported graph_mode: {mode}")
