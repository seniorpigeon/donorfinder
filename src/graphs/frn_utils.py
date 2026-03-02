"""FRN/FRC 先验文件读取工具（dist、map、tree）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TreeEdge:
    """树的父子边定义。"""

    parent: str
    child: str
    weight: float = 1.0


def _read_table(path: str | Path) -> pd.DataFrame:
    """按后缀自动选择 CSV/TSV 分隔符读取表格。"""
    p = Path(path)
    sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(p, sep=sep)


def load_dist_matrix(path: str | Path) -> Tuple[List[str], np.ndarray]:
    """读取物种距离矩阵。

    支持两种常见格式：
    1) 第一列是 taxon 名称，后面是数值矩阵
    2) 纯数值方阵（自动生成 taxon 名）
    """
    p = Path(path)
    sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(p, sep=sep)

    first_col = df.columns[0]
    as_num = pd.to_numeric(df[first_col], errors="coerce")

    # 第一列全是非数值，视为 taxon 名
    if as_num.isna().all():
        taxa = df[first_col].astype(str).tolist()
        mat = df.drop(columns=[first_col]).to_numpy(dtype=np.float32)
    else:
        # 否则整表作为矩阵读取
        mat = df.to_numpy(dtype=np.float32)
        taxa = [f"sp_{i:04d}" for i in range(mat.shape[0])]

    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Distance matrix must be square, got shape={mat.shape}")

    return taxa, mat.astype(np.float32)


def load_frc_map(path: str | Path) -> Dict[str, str]:
    """读取 species -> FRC 映射表。

    支持列名：
    - species + frc
    - species + cluster
    - species + supercluster
    """
    df = _read_table(path)
    cols = {c.lower(): c for c in df.columns}

    species_col = cols.get("species")
    if species_col is None:
        # 兜底：把第一列当 species
        species_col = df.columns[0]

    frc_col = None
    for c in ["frc", "cluster", "supercluster"]:
        if c in cols:
            frc_col = cols[c]
            break

    if frc_col is None:
        raise ValueError(
            f"FRC map must contain one of ['frc','cluster','supercluster'], got columns: {list(df.columns)}"
        )

    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        sp = str(row[species_col])
        frc = str(row[frc_col])
        mapping[sp] = frc
    return mapping


def load_tree_edges_from_table(path: str | Path, default_weight: float = 1.0) -> List[TreeEdge]:
    """从表格读取树边（parent/child/weight）。"""
    df = _read_table(path)
    cols = {c.lower(): c for c in df.columns}

    parent_col = cols.get("parent")
    child_col = cols.get("child")

    # 兼容无标准列名场景：默认前两列是 parent/child
    if parent_col is None or child_col is None:
        if len(df.columns) >= 2:
            parent_col = df.columns[0]
            child_col = df.columns[1]
        else:
            raise ValueError(f"Tree edge table requires at least two columns, got {list(df.columns)}")

    weight_col = cols.get("weight")
    edges: List[TreeEdge] = []

    for _, row in df.iterrows():
        w = float(row[weight_col]) if weight_col is not None else float(default_weight)
        edges.append(TreeEdge(parent=str(row[parent_col]), child=str(row[child_col]), weight=w))
    return edges


@dataclass
class _NewickNode:
    """Newick 解析后的内部节点结构。"""

    name: str
    length: Optional[float]
    children: List["_NewickNode"]


def _parse_newick(newick: str) -> _NewickNode:
    """最小 Newick 解析器（仅用于提取父子关系与分支长度）。"""

    s = newick.strip()
    if not s.endswith(";"):
        s += ";"

    idx = 0
    internal_counter = 0

    def parse_name(i: int) -> Tuple[str, int]:
        """读取节点名称，直到碰到 Newick 控制字符。"""
        chars = []
        while i < len(s) and s[i] not in ",():;":
            chars.append(s[i])
            i += 1
        return "".join(chars).strip(), i

    def parse_length(i: int) -> Tuple[Optional[float], int]:
        """读取 ':length' 分支长度字段。"""
        if i < len(s) and s[i] == ":":
            i += 1
            chars = []
            while i < len(s) and s[i] not in ",();":
                chars.append(s[i])
                i += 1
            txt = "".join(chars).strip()
            if txt:
                try:
                    return float(txt), i
                except ValueError:
                    return None, i
        return None, i

    def parse_subtree(i: int) -> Tuple[_NewickNode, int]:
        """递归解析子树。"""
        nonlocal internal_counter

        # 若以 '(' 开头，说明是内部节点
        if s[i] == "(":
            i += 1
            children: List[_NewickNode] = []
            while True:
                child, i = parse_subtree(i)
                children.append(child)
                if s[i] == ",":
                    i += 1
                    continue
                if s[i] == ")":
                    i += 1
                    break

            # 读内部节点名称与长度
            name, i = parse_name(i)
            length, i = parse_length(i)
            if not name:
                name = f"internal_{internal_counter}"
                internal_counter += 1
            return _NewickNode(name=name, length=length, children=children), i

        # 否则是叶子节点
        name, i = parse_name(i)
        length, i = parse_length(i)
        if not name:
            name = f"leaf_{internal_counter}"
            internal_counter += 1
        return _NewickNode(name=name, length=length, children=[]), i

    root, idx = parse_subtree(idx)
    if idx < len(s) and s[idx] == ";":
        idx += 1
    if idx != len(s):
        raise ValueError("Failed to parse Newick: trailing tokens found.")
    return root


def load_tree_edges_from_newick(path: str | Path, default_weight: float = 1.0) -> List[TreeEdge]:
    """从 Newick 文件提取 parent-child 边。"""
    txt = Path(path).read_text(encoding="utf-8").strip()
    root = _parse_newick(txt)

    edges: List[TreeEdge] = []

    def visit(node: _NewickNode) -> None:
        """DFS 遍历树，记录每条父子边。"""
        for child in node.children:
            w = child.length if child.length is not None else default_weight
            edges.append(TreeEdge(parent=node.name, child=child.name, weight=float(w)))
            visit(child)

    visit(root)
    return edges


def load_tree_edges(
    path: str | Path,
    tree_format: str = "edges",
    default_weight: float = 1.0,
) -> List[TreeEdge]:
    """统一入口：按格式读取树边。"""
    if tree_format == "edges":
        return load_tree_edges_from_table(path, default_weight=default_weight)
    if tree_format == "newick":
        return load_tree_edges_from_newick(path, default_weight=default_weight)
    raise ValueError(f"Unsupported tree format: {tree_format}")
