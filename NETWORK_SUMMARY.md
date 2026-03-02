# 网络与拓扑结构文件索引

以下文件为 `FR_Hierarchy_Gut` 项目生成的核心网络与拓扑结构数据，可直接用于您的深度学习模型。

## 1. 物种-物种关联网络 (Species-Species Network)
**核心用途**: 构建图神经网络 (GNN) 的邻接矩阵，或计算物种间的功能相似度。

*   **路径**: `data/sp_d.tsv`
*   **格式**: TSV (Tab-separated values), 对称矩阵 (2008 x 2008)
*   **内容**: 物种间的**功能距离矩阵** (Distance Matrix)。
    *   **值**: 0 ~ 1。值越小，代表两个物种的功能基因组越相似。
    *   **转换建议**: `Adjacency = 1 - Distance` (作为相似度/权重)。
*   **关键代码参考**: [GCN.py](file:///home/data/FMT/FR_Hierarchy_Gut-main/src/GCN.py) (加载与处理逻辑)

## 2. 物种功能特征网络 (Species-Function Feature Network)
**核心用途**: 为图网络中的节点（物种）提供初始特征向量 (Node Features)。

*   **路径**: `data/gcn2008.tsv`
*   **格式**: TSV, 稀疏矩阵形式 (功能基因 x 物种)
*   **内容**: 记录了每个物种包含哪些 KEGG Orthology (KO) 功能基因。
    *   **行**: 功能基因 ID (如 K00001)
    *   **列**: 物种名称
    *   **值**: 1 (存在) 或 0 (不存在)
*   **模型输入**: 可以直接作为节点的特征矩阵 $X$ (Feature Matrix)。

## 3. 功能层级树结构 (Functional Hierarchy Tree)
**核心用途**: 构建层次化模型 (Hierarchical GNN) 或进行多尺度特征池化 (Pooling)。

*   **路径**: `result/GCN_fix_tree/renamed_GCN_tree.newick`
*   **格式**: Newick (标准树格式)
*   **内容**: 基于功能距离构建的层级聚类树。
    *   **叶子节点**: 具体物种。
    *   **内部节点**: 功能簇 (Cluster) 和 超簇 (Supercluster)。
*   **解析后数据 (方便模型读取)**:
    *   **节点表**: `result/GCN_fix_tree/tree_nodes.csv` (包含节点 ID, 名称, 是否为叶子)
    *   **边表**: `result/GCN_fix_tree/tree_edges.csv` (包含父子连接关系, 边长)

## 4. 物种功能簇划分 (Cluster Partition)
**核心用途**: 用于粗粒度特征聚合，或作为分类任务的辅助标签。

*   **路径**: `result/GCN_fix_tree/leaves_cluster.tsv`
*   **格式**: TSV
*   **内容**: 每个物种所属的功能分组。
    *   `species`: 物种名
    *   `cluster`: 细粒度功能簇
    *   `supercluster`: 粗粒度功能超簇

## 5. 疾病特异性网络 (Disease-Specific Networks)
**核心用途**: 如果您的模型针对特定疾病（如 NAFLD），使用这些子网络会更精准。

*   **NAFLD (非酒精性脂肪肝)**:
    *   距离矩阵: `data/NAFLD/NASH_distance.tsv`
    *   功能网络: `data/NAFLD/NASH_GCN.tsv`
*   **关键物种列表 (Keystone Species)**:
    *   路径: `data/NAFLD/NASH_keystone.list`
    *   用途: 这里的物种在网络中起核心作用，可以在模型中对其特征进行加权 (Attention Mechanism)。
