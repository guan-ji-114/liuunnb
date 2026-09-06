#!/usr/bin/env bash
# AutoDL (Linux) 全量训练启动脚本 (nohup 后台运行 + 断点续训)
# 用法: bash run_train.sh
# 查看进度: tail -f runs/full_train.log
set -euo pipefail
cd "$(dirname "$0")"

# 数据根目录: 默认 /root/autodl-tmp/Renming (AutoDL 数据盘)
# 若上传到别处, 改这里或 export DATA_ROOT=...
export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/Renming}"

# DataLoader 进程数: 0 最稳; CPU 核多可设 4-8 加速数据加载
WORKERS="${WORKERS:-4}"

# ---- 快速验证 (首次上机建议先跑这个, 200张×10ep, 约十几分钟) ----
# python -u train.py --epochs 10 --limit 200 --w-layout 0.5 --w-sym 0.5 --out runs/verify --workers "$WORKERS"

# ---- 全量训练 (40 epoch, 断点续训) ----
nohup python -u train.py \
    --data-root "$DATA_ROOT" \
    --epochs 40 \
    --w-layout 0.5 --w-sym 0.5 \
    --out runs/full \
    --resume \
    --workers "$WORKERS" \
    > runs/full_train.log 2>&1 &

echo "训练已在后台启动 (PID $!)"
echo "日志: runs/full_train.log  ->  tail -f runs/full_train.log"
