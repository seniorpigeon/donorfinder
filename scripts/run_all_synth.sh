#!/usr/bin/env bash
# 开启严格模式：
# -e: 任一步失败立即退出
# -u: 使用未定义变量时报错
# -o pipefail: 管道中任何一步失败都算失败
set -euo pipefail

# 计算项目根目录（脚本所在目录的上一级）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 切到项目根目录，保证相对路径都基于仓库根目录
cd "$ROOT_DIR"

# 统一配置文件入口
CONFIG="configs/default.yaml"

echo "[1/5] Generate synthetic data"
# 第1步：生成合成数据（四元组 + FRN/FRC先验）
python -m src.data.make_synth --config "$CONFIG"

echo "[2/5] Train Stage A (Enc + fr)"
# 第2步：预训练编码器与前向模块（L_post + λy*L_y）
python -m src.train.train_stage_a --config "$CONFIG"

echo "[3/5] Train Stage B (fp, freeze Enc+fr)"
# 第3步：冻结 Enc+fr，只训练逆向模块 fp（L_cycle + λd*L_d）
python -m src.train.train_stage_b --config "$CONFIG"

echo "[4/5] Train Stage C (joint finetune + donor cache)"
# 第4步：联合微调 Enc+fr+fp，并构建供体库 embedding 缓存
python -m src.train.train_stage_c --config "$CONFIG"

echo "[5/5] Inference Top-K donors"
# 第5步：对一个新受体（默认索引0）执行推理，输出Top-K供体
python -m src.infer.run_infer --config "$CONFIG" --recipient_index 0 --top_k 5

echo "Pipeline completed."
