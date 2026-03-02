"""训练与运行时通用工具函数。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# 配置字典类型别名，便于类型提示
ConfigDict = Dict[str, object]


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，不存在则递归创建。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path: str | Path) -> ConfigDict:
    """读取 YAML 配置文件。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, payload: Dict[str, object]) -> None:
    """保存 JSON（UTF-8，缩进输出便于查看）。"""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def set_global_seed(seed: int) -> None:
    """设置全局随机种子，提升结果可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_cfg: str) -> torch.device:
    """根据配置字符串解析设备。

    - auto: 优先 CUDA，否则 CPU
    - 其他字符串（如 cpu/cuda:0）直接构造 torch.device
    """
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def split_train_val_indices(
    y: np.ndarray,
    groups: Optional[np.ndarray],
    val_ratio: float,
    seed: int,
    use_group_split: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成训练/验证划分索引。

    - 若 use_group_split=True 且 groups 提供，则按 group 划分（避免泄漏）。
    - 否则普通随机切分（尽量按 y 分层）。
    """
    n = len(y)
    all_indices = np.arange(n)

    if use_group_split and groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(splitter.split(all_indices, y, groups=groups))
        return np.asarray(train_idx), np.asarray(val_idx)

    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    return np.asarray(train_idx), np.asarray(val_idx)


def save_checkpoint(path: str | Path, state: Dict[str, object]) -> None:
    """保存 PyTorch checkpoint（字典格式）。"""
    p = Path(path)
    ensure_dir(p.parent)
    torch.save(state, p)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, object]:
    """从磁盘加载 checkpoint 字典。"""
    return torch.load(Path(path), map_location=map_location)
