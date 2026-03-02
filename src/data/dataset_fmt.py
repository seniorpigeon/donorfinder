"""FMT 四元组数据集定义与本地数组加载逻辑。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .preprocess import apply_abundance_transform


class FMTQuadrupletDataset(Dataset):
    """PyTorch Dataset：返回 (R_pre, D, R_post, y, recipient_id, donor_id)。"""

    def __init__(
        self,
        r_pre: np.ndarray,
        donor: np.ndarray,
        r_post: np.ndarray,
        y: np.ndarray,
        recipient_id: Optional[np.ndarray] = None,
        donor_id: Optional[np.ndarray] = None,
    ) -> None:
        # 三个输入向量与标签，统一为 float32
        self.r_pre = r_pre.astype(np.float32)
        self.donor = donor.astype(np.float32)
        self.r_post = r_post.astype(np.float32)
        self.y = y.astype(np.float32)

        n = len(self.y)
        # 若外部未提供 ID，就按顺序生成一个默认 ID
        self.recipient_id = (
            recipient_id.astype(np.int64) if recipient_id is not None else np.arange(n, dtype=np.int64)
        )
        self.donor_id = donor_id.astype(np.int64) if donor_id is not None else np.arange(n, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        # 返回字典格式，训练代码中可按字段名取值，语义更清晰
        return {
            "R_pre": torch.from_numpy(self.r_pre[index]),
            "D": torch.from_numpy(self.donor[index]),
            "R_post": torch.from_numpy(self.r_post[index]),
            "y": torch.tensor(self.y[index], dtype=torch.float32),
            "recipient_id": torch.tensor(self.recipient_id[index], dtype=torch.int64),
            "donor_id": torch.tensor(self.donor_id[index], dtype=torch.int64),
        }


def _pick_existing(base: Path, candidates: Tuple[str, ...]) -> Path:
    """在候选文件名中找到第一个存在的文件。"""
    for name in candidates:
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the files exist in {base}: {candidates}")


def load_quadruplet_arrays(
    data_dir: str | Path,
    transform: str = "log1p",
    clr_eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """从 data_dir 加载四元组数组并做统一变换。"""
    d = Path(data_dir)

    # 同时兼容两套命名（R_pre / X_pre）
    r_pre_path = _pick_existing(d, ("R_pre.npy", "X_pre.npy"))
    donor_path = _pick_existing(d, ("D.npy", "X_donor.npy"))
    r_post_path = _pick_existing(d, ("R_post.npy", "X_post.npy"))
    y_path = _pick_existing(d, ("y.npy",))

    # 读取原始矩阵
    r_pre = np.load(r_pre_path)
    donor = np.load(donor_path)
    r_post = np.load(r_post_path)
    y = np.load(y_path).astype(np.float32)

    # 可选 ID 文件（不存在则回退为顺序 ID）
    recipient_id_path = d / "recipient_id.npy"
    donor_id_path = d / "donor_id.npy"

    recipient_id = np.load(recipient_id_path) if recipient_id_path.exists() else np.arange(len(y))
    donor_id = np.load(donor_id_path) if donor_id_path.exists() else np.arange(len(y))

    out = {
        "R_pre": apply_abundance_transform(r_pre, method=transform, clr_eps=clr_eps),
        "D": apply_abundance_transform(donor, method=transform, clr_eps=clr_eps),
        "R_post": apply_abundance_transform(r_post, method=transform, clr_eps=clr_eps),
        "y": y.astype(np.float32),
        "recipient_id": recipient_id.astype(np.int64),
        "donor_id": donor_id.astype(np.int64),
    }

    # 可选：加载供体库（用于缓存或推理）
    donor_bank_file = d / "donor_bank.npy"
    if donor_bank_file.exists():
        out["donor_bank"] = apply_abundance_transform(
            np.load(donor_bank_file), method=transform, clr_eps=clr_eps
        )

    # 可选：加载 taxon 名称列表
    taxa_file = d / "taxa_names.txt"
    if taxa_file.exists():
        out["taxa_names"] = np.array(
            [ln.strip() for ln in taxa_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        )

    return out


def make_dataloaders(
    dataset: FMTQuadrupletDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """根据索引切分创建训练/验证 DataLoader。"""
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,  # 训练集打乱
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,  # 验证集不打乱
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader
