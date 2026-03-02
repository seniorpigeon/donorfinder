from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.make_synth import generate_synthetic_data, save_synthetic_bundle
from src.graphs.build_graph import build_graph_from_config
from src.models.encoder_gnn import SampleEncoder
from src.models.retrieval import build_or_load_donor_cache, retrieve_topk


def test_donor_cache_and_retrieval(tmp_path: Path) -> None:
    """验证供体缓存构建与 Top-K 检索流程可运行。"""
    # 1) 生成小型 synthetic 数据
    bundle = generate_synthetic_data(
        n_samples=12,
        n_taxa=20,
        n_donors=6,
        n_recipients=8,
        n_frc=5,
        alpha_pre=0.3,
        alpha_donor=0.2,
        noise_std=0.01,
        hidden_scale=1.0,
        seed=321,
    )
    save_synthetic_bundle(bundle, tmp_path)

    # 2) 使用 species 图模式
    cfg = {
        "graph": {
            "graph_mode": "species",
            "frc": {
                "map_path": str(tmp_path / "frc_map.tsv"),
                "tree_path": str(tmp_path / "frc_tree_edges.tsv"),
                "tree_format": "edges",
                "default_weight": 1.0,
            },
            "species": {
                "dist_mat_path": str(tmp_path / "dist_mat.tsv"),
                "knn_k": 4,
                "sigma": None,
            },
        }
    }

    taxa_names = bundle["taxa_names"].tolist()
    graph = build_graph_from_config(cfg, taxa_names=taxa_names)
    adj_norm = graph.build_dense_adj(add_self_loop=True)

    # 3) 用 MLP backend 构建一个最小编码器
    enc = SampleEncoder(
        num_nodes=graph.num_nodes(),
        edge_index=graph.edge_index,
        adj_norm=adj_norm,
        z_dim=16,
        hidden_dim=12,
        num_layers=2,
        dropout=0.0,
        backend="mlp",
    )

    cache_file = tmp_path / "donor_cache.npz"
    donor_bank = bundle["donor_bank"].astype(np.float32)

    # 4) 构建 donor cache
    payload = build_or_load_donor_cache(
        cache_file=cache_file,
        donor_bank_abundance=donor_bank,
        encoder=enc,
        graph=graph,
        device=torch.device("cpu"),
        donor_ids=np.arange(len(donor_bank), dtype=np.int64),
        batch_size=4,
        force_rebuild=True,
    )

    # 5) 以第0个 donor 向量为 query，检查检索输出
    z_bank = payload["z_donor_bank"]
    idx, scores = retrieve_topk(z_bank[:1], z_bank, top_k=3)

    assert cache_file.exists()
    assert z_bank.shape[0] == donor_bank.shape[0]
    assert idx.shape == (1, 3)
    assert scores.shape == (1, 3)
    assert int(idx[0, 0]) == 0
