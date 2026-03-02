下面是一份**可直接交给 Codex/工程师生成代码文件并在本地/云上跑通**的“脚本清单 + 最小可运行代码框架”（中文，MVP优先）。我假设你已经有：
以下文件为 `FR_Hierarchy_Gut` 项目生成的核心网络与拓扑结构数据，可直接用于您的深度学习模型。/home/data/FMT/DonorFinder/DFtest1/NETWORK_SUMMARY.md


- （可选）已训练好的节点嵌入：`artifacts/embed/V_taxa.npy`（T×d）

> 默认MVP用 **taxa节点嵌入 + 丰度加权聚合** 得到样本embedding（d=64），FRC映射用于解释/可选增强；不要求你先重算SE/nFR。

---

## 一、工程最小目录（建议）
```text
fmt_frn_inverse/
├─ requirements.txt
├─ scripts/
│  ├─ 00_make_fake_data.py
│  ├─ 01_prepare_dataset.py
│  ├─ 02_train_node_embed.py          # 可选：无V_taxa时训练/生成
│  ├─ 03_build_sample_embed.py
│  ├─ 04_train_fr.py
│  ├─ 05_train_g.py
│  ├─ 06_build_cure_prototypes.py
│  ├─ 07_train_fp.py
│  ├─ 08_build_faiss_index.py
│  └─ 09_infer_recommend.py
├─ src/
│  ├─ seed.py
│  ├─ io.py
│  ├─ data.py
│  ├─ embed.py
│  ├─ models.py
│  ├─ losses.py
│  ├─ train.py
│  └─ retrieval.py
├─ data/
│  ├─ raw/            # 可空；用fake数据也可直接写processed
│  └─ processed/
├─ artifacts/
│  ├─ frn/            # frn_edges.csv, frc_map.csv, (optional frc_tree.newick)
│  ├─ embed/          # V_taxa.npy, Z_pre.npy/Z_post.npy/Z_donor.npy
│  ├─ proto/          # cure_prototypes.npy
│  └─ faiss/          # donor.index, donor_ids.npy
└─ checkpoints/
   ├─ fr.pt
   ├─ g.pt
   └─ fp.pt
```

---

## 二、脚本化实现清单（每步：脚本名/功能/IO/默认超参）
> 默认超参：`d=64, batch=32, lr=1e-3, wd=1e-4, epochs=200, early_stop=20, device=cuda(if available)`  
> GroupKFold：按 `recipient_id` 分组，`n_splits=5`

| 步骤 | 脚本 | 功能 | 输入 | 输出 | 关键参数(默认) |
|---|---|---|---|---|---|
| 0 | `00_make_fake_data.py` | 生成假数据（pre/post/donor丰度矩阵 + y + recipient_id + donor_id），可同时生成 FRN/FRC/V_taxa 以便一键跑通 | 无/少量参数 | `data/processed/*.npy` + 可选 `artifacts/frn/*`、`artifacts/embed/V_taxa.npy` | `N=200,T=300,D=40,R=120,K_frc=52,d=64,seed=0` |
| 1 | `01_prepare_dataset.py` | 真实数据预处理：对齐taxa列、过滤、log1p/CLR、保存npy | `data/raw/*.csv/tsv` + meta | `data/processed/X_*.npy,y.npy,recipient_id.npy,donor_id.npy` | `transform=log1p, min_prev=0.05` |
| 2 | `02_train_node_embed.py` | 若无 `V_taxa.npy`：生成/训练节点嵌入（MVP可用随机；升级可用PyG Node2Vec） | `artifacts/frn/frn_edges.csv` | `artifacts/embed/V_taxa.npy` | `mode=random/node2vec, d=64, topk=20` |
| 3 | `03_build_sample_embed.py` | 样本embedding：`z=Σ a_i v_i`；生成 `Z_pre/Z_post/Z_donor` | `X_*.npy` + `V_taxa.npy` | `artifacts/embed/Z_*.npy` | `l2_norm=True` |
| 4 | `04_train_fr.py` | 训练响应模型 `f_r(pre,donor)->post`（MSE） | `Z_pre,Z_donor,Z_post,recipient_id` | `checkpoints/fr.pt` + `outputs/fr_metrics.jsonl` | `batch=32,lr=1e-3` |
| 5 | `05_train_g.py` | 训练疗效头 `g(pre,donor)->y`（BCEWithLogits） | `Z_pre,Z_donor,y,recipient_id` | `checkpoints/g.pt` + `outputs/g_metrics.jsonl` | `pos_weight=auto` |
| 6 | `06_build_cure_prototypes.py` | 用治愈样本 `Z_post(y=1)` 聚类得到目标原型 μ_k | `Z_post,y` | `artifacts/proto/cure_prototypes.npy` | `K=10,n_init=auto` |
| 7 | `07_train_fp.py` | 冻结fr/g，训练生成器 `f_p(pre,target)->donor*`（cycle+cure+prior） | `Z_pre,Z_post,Z_donor,y,fr.pt,g.pt,prototypes` | `checkpoints/fp.pt` | `λ_cycle=1,λ_cure=0.5,λ_prior=0.1,M=10,τ=0.07` |
| 8 | `08_build_faiss_index.py` | 建供体库索引（余弦：先L2再IP） | `Z_donor` + donor_ids | `artifacts/faiss/donor.index, donor_ids.npy` | `index=FlatIP` |
| 9 | `09_infer_recommend.py` | 推理：选目标原型（用g打分）→生成 donor* → FAISS检索TopK供体 | `Z_pre_new` + `fp.pt,g.pt,prototypes,faiss_index` | `outputs/recs.csv` | `TopK=5` |

---

## 三、快速跑通（bash命令顺序）
```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 生成假数据（同时生成FRN/FRC/V_taxa，快速闭环）
python scripts/00_make_fake_data.py --out_dir data/processed --art_dir artifacts --N 200 --T 300 --D 40 --R 120 --d 64

# 3) 生成样本embedding
python scripts/03_build_sample_embed.py --data_dir data/processed --art_dir artifacts --d 64

# 4) 训练fr与g（GroupKFold按recipient分组）
python scripts/04_train_fr.py --data_dir data/processed --art_dir artifacts --ckpt_dir checkpoints --n_splits 5 --fold 0
python scripts/05_train_g.py  --data_dir data/processed --art_dir artifacts --ckpt_dir checkpoints --n_splits 5 --fold 0

# 5) 构建治愈原型
python scripts/06_build_cure_prototypes.py --art_dir artifacts --K 10

# 6) 训练fp（冻结fr/g）
python scripts/07_train_fp.py --data_dir data/processed --art_dir artifacts --ckpt_dir checkpoints --fold 0

# 7) 建FAISS供体库索引 + 推理
python scripts/08_build_faiss_index.py --art_dir artifacts
python scripts/09_infer_recommend.py --data_dir data/processed --art_dir artifacts --ckpt_dir checkpoints --TopK 5
```

> 说明：`--fold 0` 是先跑通一折；工程化版本可循环 fold=0..4 汇总平均指标。

---

## 四、requirements.txt（建议）
> Node2Vec训练可选（默认不强依赖PyG）。FAISS可用CPU版。

```txt
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
tqdm>=4.66
torch>=2.2
faiss-cpu>=1.7.4
# 可选（如果你想用PyG的Node2Vec）
# torch-geometric>=2.5
# torch-scatter
# torch-sparse
# torch-cluster
```

**硬件建议**
- N≈200，小模型：CPU也能跑通；建议有GPU更快（尤其你后续换更复杂表示）。
- FAISS检索：CPU足够；供体库>50k再考虑GPU或ANN索引。

---

## 五、最小可运行代码框架（核心模块）
下面是“最小框架”，Codex可据此生成完整文件。你可把这些内容分别保存到对应路径。

### `src/seed.py`
```python
import os, random
import numpy as np
import torch

def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### `src/io.py`
```python
import json, os
import numpy as np

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_npy(path: str, arr):
    ensure_dir(os.path.dirname(path))
    np.save(path, arr)

def load_npy(path: str):
    return np.load(path, allow_pickle=False)

def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
```

### `src/embed.py`（节点嵌入加载 + 样本聚合）
```python
import numpy as np

def l2_normalize(x: np.ndarray, eps: float = 1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def build_sample_z(X: np.ndarray, V_taxa: np.ndarray, l2_norm: bool = True) -> np.ndarray:
    """
    X: (N,T) abundance (float32, non-negative, row-sum=1 recommended)
    V_taxa: (T,d) node embedding
    return Z: (N,d)
    """
    # 归一化（防止输入不是严格sum=1）
    row_sum = X.sum(axis=1, keepdims=True) + 1e-12
    W = X / row_sum
    Z = W @ V_taxa  # (N,d)
    if l2_norm:
        Z = l2_normalize(Z.astype(np.float32))
    return Z.astype(np.float32)
```

### `src/models.py`（Interaction + f_r + g + f_p）
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class InteractionFeatures(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a,b: (B,d) -> (B,4d)
        return torch.cat([a, b, b - a, a * b], dim=-1)

def make_mlp(in_dim: int, hidden=(256,128), out_dim: int = 64, dropout: float = 0.1):
    layers = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(h)]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)

class ResponseNet(nn.Module):
    """f_r: (z_pre,z_donor)->z_post_hat  using residual delta"""
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.feat = InteractionFeatures()
        self.net = make_mlp(in_dim=4*d, hidden=(256,128), out_dim=d, dropout=dropout)

    def forward(self, z_pre: torch.Tensor, z_donor: torch.Tensor) -> torch.Tensor:
        x = self.feat(z_pre, z_donor)
        delta = self.net(x)
        return z_pre + delta

class OutcomeHead(nn.Module):
    """g: (z_pre,z_donor)->logit"""
    def __init__(self, d: int, dropout: float = 0.2):
        super().__init__()
        self.feat = InteractionFeatures()
        self.net = make_mlp(in_dim=4*d, hidden=(256,128), out_dim=1, dropout=dropout)

    def forward(self, z_pre: torch.Tensor, z_donor: torch.Tensor) -> torch.Tensor:
        x = self.feat(z_pre, z_donor)
        return self.net(x).squeeze(-1)  # (B,)

class DonorGenerator(nn.Module):
    """f_p: (z_pre,z_target)->z_donor_star"""
    def __init__(self, d: int, dropout: float = 0.1, l2_norm: bool = True):
        super().__init__()
        self.feat = InteractionFeatures()
        self.net = make_mlp(in_dim=4*d, hidden=(256,128), out_dim=d, dropout=dropout)
        self.l2_norm = l2_norm

    def forward(self, z_pre: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        x = self.feat(z_pre, z_target)
        z = self.net(x)
        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
        return z
```

### `src/losses.py`（cycle/cure/prior + 监督loss）
```python
import torch
import torch.nn.functional as F

def loss_post(z_hat_post: torch.Tensor, z_post: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(z_hat_post, z_post)

def loss_y(logits: torch.Tensor, y: torch.Tensor, pos_weight=None) -> torch.Tensor:
    # logits:(B,), y:(B,)
    if pos_weight is not None:
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    return F.binary_cross_entropy_with_logits(logits, y)

def loss_cycle(fr, z_pre, z_target, z_donor_star):
    z_hat = fr(z_pre, z_donor_star)
    return F.mse_loss(z_hat, z_target)

def loss_cure(g, z_pre, z_donor_star):
    logits = g(z_pre, z_donor_star)
    ones = torch.ones_like(logits)
    return F.binary_cross_entropy_with_logits(logits, ones)

@torch.no_grad()
def soft_nn_target(z_query: torch.Tensor, bank: torch.Tensor, top_m: int = 10, tau: float = 0.07):
    """
    z_query: (B,d) 已归一化
    bank: (Nb,d) 已归一化
    返回加权邻居均值 z_nn: (B,d)
    """
    sim = z_query @ bank.t()  # (B,Nb) 余弦(因为L2)
    vals, idx = torch.topk(sim, k=min(top_m, bank.size(0)), dim=1)
    w = torch.softmax(vals / tau, dim=1)  # (B,M)
    nn_vecs = bank[idx]                   # (B,M,d)
    z_nn = (w.unsqueeze(-1) * nn_vecs).sum(dim=1)
    return z_nn

def loss_prior(z_donor_star: torch.Tensor, donor_bank: torch.Tensor, top_m: int = 10, tau: float = 0.07):
    z_nn = soft_nn_target(z_donor_star, donor_bank, top_m=top_m, tau=tau)
    return F.mse_loss(z_donor_star, z_nn)
```

### `src/train.py`（最小训练循环：分阶段训练与冻结）
```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold

class FMTDataset(Dataset):
    def __init__(self, Z_pre, Z_donor, Z_post, y, recipient_id):
        self.Z_pre = Z_pre
        self.Z_donor = Z_donor
        self.Z_post = Z_post
        self.y = y
        self.recipient_id = recipient_id

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return (self.Z_pre[i], self.Z_donor[i], self.Z_post[i], self.y[i], self.recipient_id[i])

def to_torch(batch, device):
    z_pre, z_donor, z_post, y, rid = batch
    return (torch.tensor(z_pre, device=device),
            torch.tensor(z_donor, device=device),
            torch.tensor(z_post, device=device),
            torch.tensor(y, device=device),
            torch.tensor(rid, device=device))

def make_loader(dataset, idx, batch=32, shuffle=True):
    subset = torch.utils.data.Subset(dataset, idx)
    return DataLoader(subset, batch_size=batch, shuffle=shuffle, drop_last=False)

def groupkfold_split(recipient_ids, n_splits=5, fold=0, y=None):
    gkf = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros((len(recipient_ids), 1))
    y_dummy = y if y is not None else np.zeros(len(recipient_ids))
    splits = list(gkf.split(X_dummy, y_dummy, groups=recipient_ids))
    return splits[fold]  # (train_idx, test_idx)

def freeze_(model, flag=True):
    for p in model.parameters():
        p.requires_grad = (not flag)

def train_fr(fr, loader_tr, loader_va, optim, loss_fn, device, epochs=200, early_stop=20):
    best = 1e9; bad=0; best_state=None
    for ep in range(1, epochs+1):
        fr.train()
        for batch in loader_tr:
            z_pre, z_donor, z_post, y, _ = to_torch(batch, device)
            optim.zero_grad()
            z_hat = fr(z_pre, z_donor)
            loss = loss_fn(z_hat, z_post)
            loss.backward(); optim.step()
        # val
        fr.eval(); tot=0; n=0
        with torch.no_grad():
            for batch in loader_va:
                z_pre, z_donor, z_post, y, _ = to_torch(batch, device)
                z_hat = fr(z_pre, z_donor)
                tot += loss_fn(z_hat, z_post).item()*len(z_pre); n += len(z_pre)
        val = tot/n
        if val < best:
            best = val; bad=0
            best_state = {k:v.cpu().clone() for k,v in fr.state_dict().items()}
        else:
            bad += 1
            if bad >= early_stop: break
    fr.load_state_dict(best_state)
    return best

def train_g(g, loader_tr, loader_va, optim, loss_fn, device, epochs=200, early_stop=20):
    best = 1e9; bad=0; best_state=None
    for ep in range(1, epochs+1):
        g.train()
        for batch in loader_tr:
            z_pre, z_donor, z_post, y, _ = to_torch(batch, device)
            optim.zero_grad()
            logits = g(z_pre, z_donor)
            loss = loss_fn(logits, y)
            loss.backward(); optim.step()
        # val
        g.eval(); tot=0; n=0
        with torch.no_grad():
            for batch in loader_va:
                z_pre, z_donor, z_post, y, _ = to_torch(batch, device)
                logits = g(z_pre, z_donor)
                tot += loss_fn(logits, y).item()*len(z_pre); n += len(z_pre)
        val = tot/n
        if val < best:
            best = val; bad=0
            best_state = {k:v.cpu().clone() for k,v in g.state_dict().items()}
        else:
            bad += 1
            if bad >= early_stop: break
    g.load_state_dict(best_state)
    return best

def train_fp(fp, fr, g, donor_bank, loader_cure, optim, device,
             lam_cycle=1.0, lam_cure=0.5, lam_prior=0.1, top_m=10, tau=0.07,
             epochs=200, early_stop=20, losses=None):
    # fr,g冻结
    freeze_(fr, True); fr.eval()
    freeze_(g, True); g.eval()

    best=1e9; bad=0; best_state=None
    for ep in range(1, epochs+1):
        fp.train()
        for batch in loader_cure:
            z_pre, z_donor, z_post, y, _ = to_torch(batch, device)
            z_target = z_post
            optim.zero_grad()
            z_star = fp(z_pre, z_target)
            Lc = losses["cycle"](fr, z_pre, z_target, z_star)
            Ly = losses["cure"](g, z_pre, z_star)
            Lp = losses["prior"](z_star, donor_bank, top_m=top_m, tau=tau)
            loss = lam_cycle*Lc + lam_cure*Ly + lam_prior*Lp
            loss.backward(); optim.step()
        # 简单val：用同一loader算一次(可升级为独立val_cure)
        fp.eval(); tot=0; n=0
        with torch.no_grad():
            for batch in loader_cure:
                z_pre, _, z_post, y, _ = to_torch(batch, device)
                z_star = fp(z_pre, z_post)
                Lc = losses["cycle"](fr, z_pre, z_post, z_star)
                tot += Lc.item()*len(z_pre); n += len(z_pre)
        val = tot/n
        if val < best:
            best=val; bad=0
            best_state = {k:v.cpu().clone() for k,v in fp.state_dict().items()}
        else:
            bad += 1
            if bad >= early_stop: break
    fp.load_state_dict(best_state)
    return best
```

### `src/retrieval.py`（FAISS索引与检索示例）
```python
import numpy as np
import faiss

def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)

def build_faiss_flatip(emb: np.ndarray):
    # emb: (N,d) float32, 已L2
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    return index

def search(index, q: np.ndarray, topk: int = 5):
    # q: (B,d) float32, 已L2
    scores, idx = index.search(q, topk)
    return scores, idx
```

---

## 六、假数据生成脚本（必须项）`scripts/00_make_fake_data.py`
> 目标：一键生成能跑完整管线的数据（含X_pre/X_post/X_donor/y/recipient_id/donor_id + FRN edges + FRC map + 可选 V_taxa）。

```python
import argparse, os
import numpy as np
from src.io import ensure_dir, save_npy, save_json

def dirichlet_rows(n, t, alpha=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.dirichlet(alpha*np.ones(t), size=n).astype(np.float32)

def make_random_frn_edges(T, topk=20, seed=0):
    rng = np.random.default_rng(seed)
    edges = []
    # 对每个节点连topk随机邻居，权重>0
    for i in range(T):
        nbr = rng.choice(T, size=topk, replace=False)
        w = rng.uniform(0.1, 1.0, size=topk)
        for j,wij in zip(nbr, w):
            if i==j: continue
            edges.append((i,int(j),float(wij)))
            edges.append((int(j),i,float(wij)))
    return np.array(edges, dtype=object)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--art_dir", default="artifacts")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--D", type=int, default=40)   # donors
    ap.add_argument("--R", type=int, default=120)  # recipients
    ap.add_argument("--K_frc", type=int, default=52)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write_V_taxa", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.art_dir, "frn"))
    ensure_dir(os.path.join(args.art_dir, "embed"))

    rng = np.random.default_rng(args.seed)

    # Donor abundance (D×T)
    donors = dirichlet_rows(args.D, args.T, alpha=0.2, seed=args.seed+1)

    # Recipient id per sample
    recipient_ids = rng.integers(0, args.R, size=args.N)
    donor_ids = rng.integers(0, args.D, size=args.N)

    # pre abundance
    X_pre = dirichlet_rows(args.N, args.T, alpha=0.3, seed=args.seed+2)
    X_donor = donors[donor_ids]

    # compatibility -> mixing coefficient m
    # 简单：用donor与pre的余弦相似度决定m
    def cos(a,b):
        na = np.linalg.norm(a, axis=1, keepdims=True)+1e-12
        nb = np.linalg.norm(b, axis=1, keepdims=True)+1e-12
        return (a/na * b/nb).sum(axis=1)

    sim = cos(X_pre, X_donor)
    m = 1/(1+np.exp(-6*(sim-0.5)))  # sigmoid
    m = m.astype(np.float32)

    # post abundance = (1-m)*pre + m*donor + noise, renorm
    noise = rng.normal(0, 0.01, size=X_pre.shape).astype(np.float32)
    X_post = (1-m[:,None])*X_pre + m[:,None]*X_donor + noise
    X_post = np.clip(X_post, 1e-8, None)
    X_post = (X_post / (X_post.sum(axis=1, keepdims=True)+1e-12)).astype(np.float32)

    # y label: 基于m或sim（可调）
    y = (sim > 0.55).astype(np.float32)

    save_npy(os.path.join(args.out_dir, "X_pre.npy"), X_pre)
    save_npy(os.path.join(args.out_dir, "X_post.npy"), X_post)
    save_npy(os.path.join(args.out_dir, "X_donor.npy"), X_donor)
    save_npy(os.path.join(args.out_dir, "y.npy"), y)
    save_npy(os.path.join(args.out_dir, "recipient_id.npy"), recipient_ids.astype(np.int64))
    save_npy(os.path.join(args.out_dir, "donor_id.npy"), donor_ids.astype(np.int64))

    # FRN edges
    edges = make_random_frn_edges(args.T, topk=20, seed=args.seed+3)
    np.savetxt(os.path.join(args.art_dir, "frn", "frn_edges.csv"),
               edges, fmt="%s", delimiter=",", header="u,v,w", comments="")

    # FRC map（随机映射，真实任务用你的映射替换）
    frc_map = np.column_stack([
        np.arange(args.T),
        rng.integers(0, args.K_frc, size=args.T)
    ]).astype(np.int64)
    np.savetxt(os.path.join(args.art_dir, "frn", "frc_map.csv"),
               frc_map, fmt="%d", delimiter=",", header="taxa_idx,frc_id", comments="")

    # 可选：直接写V_taxa（可跳过node2vec训练）
    if args.write_V_taxa:
        V = rng.normal(0, 1, size=(args.T, args.d)).astype(np.float32)
        V = V / (np.linalg.norm(V, axis=1, keepdims=True)+1e-12)
        save_npy(os.path.join(args.art_dir, "embed", "V_taxa.npy"), V)

    save_json(os.path.join(args.out_dir, "meta.json"), vars(args))
    print("Fake data generated.")

if __name__ == "__main__":
    main()
```

---

## 七、GroupKFold按recipient分组（N≈200）如何用
在 `scripts/04_train_fr.py` / `05_train_g.py` 里用：

```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for fold,(tr,te) in enumerate(gkf.split(Z_pre, y, groups=recipient_id)):
    # te作为test，tr内部再切val（也要按recipient分组）
    ...
```

> 常见坑：`n_splits` 不能大于不同 recipient 数；否则 scikit-learn 会报错。

---

## 八、每步可能报错点与调试建议（按阶段）
### 数据与embedding
- **shape mismatch**：`X_pre` 列数T必须等于 `V_taxa` 行数T；检查 `X_pre.shape[1] == V_taxa.shape[0]`
- **NaN/Inf**：CLR或log变换后出现；先用 `log1p` 跑通，再切换CLR并加 pseudocount
- **分组泄漏**：不要随机split；必须 GroupKFold 按 `recipient_id`

### 训练f_r/g
- **loss不下降**：先把lr降到 `3e-4`；或关掉LayerNorm排查；检查输入是否已L2 normalize
- **y全0/全1**：pos_weight会爆；先打印 `y.mean()` 确认
- **过拟合**：加大dropout(0.3)、early_stop缩短、减少隐藏层宽度

### 训练f_p（cycle/cure/prior）
- **fp生成“骗g”的向量**：提高 `λ_prior`（0.2~0.5），并确保 donor_bank 已L2 normalize
- **prior topk报错**：`top_m > donor_bank.size(0)`；代码里已min处理，但仍建议设置 M<=供体数
- **梯度无效**：确认 fr/g 已冻结而 fp 可训练：`any(p.requires_grad for p in fp.parameters())==True`

### FAISS检索
- **FAISS add/search报错**：必须 `float32` 且连续内存；`np.ascontiguousarray(x.astype(np.float32))`
- **余弦检索不对**：必须先L2 normalize，再用 `IndexFlatIP`（内积=余弦）

---

如果你希望我把上面这些“脚本骨架”进一步扩成**每个 scripts/*.py 的完整可运行版本**（包含CLI参数解析、日志、保存checkpoint、输出metrics/recs.csv），我也可以继续按这份清单把每个脚本补全成工程级模板（仍然保持MVP简洁）。