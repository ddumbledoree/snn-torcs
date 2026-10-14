#!/usr/bin/env bash
# SNN-TORCS 展示软件 · 接口验收启动脚本
# ----------------------------------------------------------------------------
# 用法:  ./run.sh
# 作用:  激活 conda env → 启动 TORCS(headless, Xvfb)→ 加载 exp013 →
#        闭环跑若干步 → 终端逐行打印 per-step 数据,并写出回放 JSON。
# 验收:  看到 "TORCS terminated." 且无报错、data/replay_example.json 生成 = 通过。
# ----------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/app"

# --- 激活 conda env(尝试常见安装路径,不硬编码)---
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/Conda/etc/profile.d/conda.sh" ]; then
    source "$HOME/Conda/etc/profile.d/conda.sh"
else
    echo "[run.sh] 未找到 conda。请先 'conda activate torcs_RL' 再运行,或修正本脚本中的 conda 路径。" >&2
    exit 1
fi
conda activate torcs_RL

# --- 逐帧 eval(demo + 生成回放数据)---
# --max-steps 200 为快速验收;要完整一圈回放数据改为 1000。
python eval_steptrace.py \
    --model-type snn_baseline \
    --checkpoint "$HERE/model/exp013_best.pth" \
    --config configs/default.yaml \
    --episodes 1 \
    --max-steps 200 \
    --run-mode fast_train \
    --track g-track-1 \
    --report "$HERE/data/replay_example.json"

echo
echo "[run.sh] 完成。回放数据见 data/replay_example.json"
