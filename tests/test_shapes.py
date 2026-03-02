from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.make_synth import generate_synthetic_data, save_synthetic_bundle
from src.graphs.build_graph import build_graph_from_config
from src.models.encoder_gnn import SampleEncoder
from src.models.forward_fr import ForwardSimulator
from src.models.inverse_fp import InverseGenerator


def test_model_shapes(tmp_path: Path) -> None:
    """验证核心模型在小数据上的张量 shape 正确。"""
    # 1) 生成一个很小的 synthetic 数据集
    bundle = generate_synthetic_data(
        n_samples=16,
        n_taxa=24,
        n_donors=8,
        n_recipients=10,
        n_frc=6,
        alpha_pre=0.3,
        alpha_donor=0.2,
        noise_std=0.01,
        hidden_scale=1.0,
        seed=123,
    )
    save_synthetic_bundle(bundle, tmp_path)

    # 2) 构造最小配置（frc 图模式）
    cfg = {
        "graph": {
            "graph_mode": "frc",
            "frc": {
                "map_path": str(tmp_path / "frc_map.tsv"),
                "tree_path": str(tmp_path / "frc_tree_edges.tsv"),
                "tree_format": "edges",
                "default_weight": 1.0,
            },
            "species": {
                "dist_mat_path": str(tmp_path / "dist_mat.tsv"),
                "knn_k": 5,
                "sigma": None,
            },
        },
        "encoder": {
            "z_dim": 32,
            "hidden_dim": 16,
            "num_layers": 2,
            "dropout": 0.0,
            "backend": "dense_gnn",
        },
        "forward": {"hidden_dim": 32, "dropout": 0.0},
        "inverse": {"hidden_dim": 32, "dropout": 0.0},
    }

    # 3) 构图并准备输入
    taxa_names = bundle["taxa_names"].tolist()
    graph = build_graph_from_config(cfg, taxa_names=taxa_names)

    x_pre = graph.to_node_features(bundle["R_pre"][:4])
    x_d = graph.to_node_features(bundle["D"][:4])
    x_post = graph.to_node_features(bundle["R_post"][:4])

    # 4) 初始化 Enc/fr/fp
    adj_norm = graph.build_dense_adj(add_self_loop=True)
    enc = SampleEncoder(
        num_nodes=graph.num_nodes(),
        edge_index=graph.edge_index,
        adj_norm=adj_norm,
        z_dim=32,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        backend="dense_gnn",
    )
    fr = ForwardSimulator(z_dim=32, hidden_dim=32, dropout=0.0)
    fp = InverseGenerator(z_dim=32, hidden_dim=32, dropout=0.0)

    # 5) 前向并验证 shape
    z_pre = enc(torch.tensor(x_pre, dtype=torch.float32))
    z_d = enc(torch.tensor(x_d, dtype=torch.float32))
    z_post = enc(torch.tensor(x_post, dtype=torch.float32))

    z_post_pred, y_logit = fr(z_pre, z_d)
    z_d_hat = fp(z_pre, z_post)

    assert z_pre.shape == (4, 32)
    assert z_post_pred.shape == (4, 32)
    assert y_logit.shape == (4,)
    assert z_d_hat.shape == (4, 32)
    assert np.isfinite(z_d_hat.detach().numpy()).all()
