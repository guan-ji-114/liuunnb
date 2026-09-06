#!/usr/bin/env bash
# AutoDL (Linux) 环境初始化脚本
# 用法: bash setup_autodl.sh
set -euo pipefail

MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
cd "$(dirname "$0")"

echo "[1/3] 安装 Python 依赖 (清华镜像)..."
pip install -i "$MIRROR" \
    transformers timm torchmetrics faster-coco-eval opencv-python-headless

echo "[2/3] 确认 PyTorch / CUDA 可用..."
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available(),
      "| cuda", torch.version.cuda, "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

echo "[3/3] 检查 DETR 预训练权重..."
WEIGHTS_DIR="$(pwd)/weights/facebook_detr_resnet50"
if [ -f "$WEIGHTS_DIR/model.safetensors" ]; then
    echo "  权重已存在: $WEIGHTS_DIR"
else
    echo "  权重缺失, 从 hf-mirror 下载 facebook/detr-resnet-50 ..."
    HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("facebook/detr-resnet-50", local_dir="weights/facebook_detr_resnet50")
print("download done")
PY
fi

echo ""
echo "完成。下一步:"
echo "  1) 确认数据在 DATA_ROOT (默认 /root/autodl-tmp/Renming) 下"
echo "  2) 快速验证环境: bash run_train.sh  (或先跑小样本见 run_train.sh 注释)"
