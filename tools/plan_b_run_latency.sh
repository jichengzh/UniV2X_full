#!/usr/bin/env bash
# Run benchmark_latency.py for one config on one GPU.
#
# For D class (quant-only configs with no prune), use active_baseline.json
# (fake-quant doesn't change latency in PyTorch path).
#
# Usage: bash tools/plan_b_run_latency.sh <CONFIG_ID> [GPU_ID]

set -uo pipefail

CONFIG_ID="${1:?Usage: $0 <CONFIG_ID> [GPU_ID]}"
GPU_ID="${2:-0}"

source /home/jichengzhi/miniconda3/etc/profile.d/conda.sh
conda activate UniV2X_2.0
cd /home/jichengzhi/UniV2X
mkdir -p output/plan_b/latency logs/plan_b/latency

# For configs that come from 1.2 P1 series, prune_configs filename differs
case "$CONFIG_ID" in
    P1_20) PRUNE_CFG="prune_configs/p1_ffn_20pct.json" ;;
    P1_30) PRUNE_CFG="prune_configs/p1_ffn_30pct.json" ;;
    P1_40) PRUNE_CFG="prune_configs/p1_ffn_40pct.json" ;;
    P1_50) PRUNE_CFG="prune_configs/p1_ffn_50pct.json" ;;
    P1_60) PRUNE_CFG="prune_configs/p1_ffn_60pct.json" ;;
    *) PRUNE_CFG="prune_configs/active_${CONFIG_ID}.json" ;;
esac

if [[ ! -f "$PRUNE_CFG" ]]; then
    echo "ERROR: $PRUNE_CFG not found"; exit 1
fi

OUT_JSON="output/plan_b/latency/${CONFIG_ID}.json"
LOG="logs/plan_b/latency/${CONFIG_ID}.log"

echo "=== latency $CONFIG_ID === GPU=$GPU_ID prune=$PRUNE_CFG"
PYTHONPATH=/home/jichengzhi/UniV2X CUDA_VISIBLE_DEVICES="$GPU_ID" python tools/benchmark_latency.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --prune-config "$PRUNE_CFG" \
    --n-warmup 3 --n-runs 10 \
    --output "$OUT_JSON" > "$LOG" 2>&1
echo "$CONFIG_ID exit=$?"
