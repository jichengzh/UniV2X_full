#!/bin/bash
# 阶段 1.1 — 下载 UniAD-tiny + R50 pretrain (~1GB) 用于 R50 全网剪枝可行性实验
#
# 设计:
#   - 阶段 1 的 L2/L3/L4 实验主要需要权重和模型架构,实际不需要完整 b2d 数据
#   - L1 framework sanity 用一个最小 clip (~150MB) 即可
#   - 总下载量 ~1.2GB,15-30 分钟(视网速)
#
# 使用:
#   bash scripts/phase1/setup_b2d_mini.sh           # 默认下载到 /home/jichengzhi/Bench2DriveZoo_trb
#   bash scripts/phase1/setup_b2d_mini.sh --no-data # 只下权重,不下示例 clip

set -euo pipefail

# ============================================================
# 配置
# ============================================================
B2D_ROOT="${B2D_ROOT:-/home/jichengzhi/Bench2DriveZoo_trb}"
CKPTS_DIR="$B2D_ROOT/ckpts"
DATA_DIR="$B2D_ROOT/data/bench2drive/v1"
SKIP_DATA=0

# parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-data) SKIP_DATA=1; shift ;;
        --root) B2D_ROOT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$CKPTS_DIR" "$DATA_DIR"

# 启用 conda 环境(huggingface_hub 在这里)
source /home/jichengzhi/miniconda3/etc/profile.d/conda.sh
conda activate UniV2X_2.0

# 设置 HF 镜像(国内连原站慢)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT=60

echo "=================================================="
echo "阶段 1.1 — b2d Mini + UniAD-tiny 下载"
echo "=================================================="
echo "  B2D_ROOT:    $B2D_ROOT"
echo "  CKPTS_DIR:   $CKPTS_DIR"
echo "  DATA_DIR:    $DATA_DIR"
echo "  HF_ENDPOINT: $HF_ENDPOINT"
echo "  SKIP_DATA:   $SKIP_DATA"
echo ""

# ============================================================
# 1. 下权重(必下)
# ============================================================
download_weight() {
    local fname="$1"
    local repo="rethinklab/Bench2DriveZoo"
    local out="$CKPTS_DIR/$fname"
    if [ -f "$out" ]; then
        echo "[skip] $fname already exists ($(du -h "$out" | cut -f1))"
        return 0
    fi
    echo "[download] $fname from $repo (mirror: $HF_ENDPOINT)"
    python -c "
import os
os.environ['HF_ENDPOINT'] = '$HF_ENDPOINT'
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='$repo',
    filename='$fname',
    local_dir='$CKPTS_DIR',
)
print(f'  saved: {p}')
"
}

echo "[step 1/3] 下载 UniAD-tiny 权重 (873MB)..."
download_weight "uniad_tiny_b2d.pth"

echo ""
echo "[step 2/3] 下载 ResNet50 pretrain (103MB)..."
download_weight "resnet50-19c8e357.pth"

# ============================================================
# 2. 下载示例 clip (可选,用于 L1 framework sanity)
# ============================================================
if [ $SKIP_DATA -eq 1 ]; then
    echo ""
    echo "[step 3/3] 跳过 b2d 数据下载(--no-data)"
    echo "  L1 framework sanity 时若需真实数据,运行:"
    echo "    bash $0 --root $B2D_ROOT  # 重新跑会下数据"
else
    echo ""
    echo "[step 3/3] 下载 1 个最小示例 clip (~150MB) 用于 framework sanity..."
    SAMPLE_CLIP="Accident_Town03_Route101_Weather23.tar.gz"
    SAMPLE_OUT="$DATA_DIR/$SAMPLE_CLIP"
    if [ -f "$SAMPLE_OUT" ] || [ -d "${SAMPLE_OUT%.tar.gz}" ]; then
        echo "[skip] sample clip already present"
    else
        python -c "
import os
os.environ['HF_ENDPOINT'] = '$HF_ENDPOINT'
from huggingface_hub import hf_hub_download
try:
    p = hf_hub_download(
        repo_id='rethinklab/Bench2Drive',
        repo_type='dataset',
        filename='$SAMPLE_CLIP',
        local_dir='$DATA_DIR',
    )
    print(f'  saved: {p}')
except Exception as e:
    print(f'  [WARN] sample clip 下载失败: {e}')
    print(f'  L1 sanity 可改用 dummy input,不阻塞 L2/L3/L4')
"
        # 解压
        if [ -f "$SAMPLE_OUT" ]; then
            echo "  解压 $SAMPLE_CLIP ..."
            (cd "$DATA_DIR" && tar -xzf "$SAMPLE_CLIP" && rm "$SAMPLE_CLIP")
            echo "  done."
        fi
    fi
fi

# ============================================================
# 3. 校验
# ============================================================
echo ""
echo "=================================================="
echo "下载完成 — 校验"
echo "=================================================="
echo ""
echo "ckpts/:"
ls -la "$CKPTS_DIR" | head
echo ""
echo "data/bench2drive/v1/:"
ls -la "$DATA_DIR" | head
echo ""

# 简单 sanity: 看权重能不能加载
echo "[sanity] 检查 uniad_tiny_b2d.pth 能否被 torch 加载..."
python -c "
import torch
ck = torch.load('$CKPTS_DIR/uniad_tiny_b2d.pth', map_location='cpu', weights_only=False)
keys = list(ck.keys()) if isinstance(ck, dict) else []
print(f'  ckpt keys: {keys[:3]} ... ({len(keys)} total)')
sd = ck.get('state_dict', ck) if isinstance(ck, dict) else ck
if isinstance(sd, dict):
    sample_keys = list(sd.keys())[:5]
    print(f'  state_dict sample keys: {sample_keys}')
    print(f'  total params: {sum(v.numel() for v in sd.values() if hasattr(v, \"numel\")):,}')
"

echo ""
echo "=================================================="
echo "阶段 1.1 下载就绪"
echo "=================================================="
