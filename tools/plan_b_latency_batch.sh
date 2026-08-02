#!/usr/bin/env bash
# Run benchmark_latency.py for all 23 configs across multiple GPUs.
# Usage: bash tools/plan_b_latency_batch.sh "<CONFIG_LIST>" "<GPU_LIST>"

set -uo pipefail

CONFIGS="${1:?Usage: $0 \"<CONFIG_IDS>\" \"<GPU_IDS>\"}"
GPUS="${2:?Need GPU list}"

cd /home/jichengzhi/UniV2X
mkdir -p output/plan_b/latency logs/plan_b/latency

IFS=',' read -ra GPU_ARR <<< "$GPUS"
read -ra CFG_ARR <<< "$CONFIGS"

NUM_GPUS=${#GPU_ARR[@]}
NUM_CFGS=${#CFG_ARR[@]}
echo "=== latency batch: $NUM_CFGS configs on $NUM_GPUS GPUs ==="

declare -A GPU_BUSY
for g in "${GPU_ARR[@]}"; do GPU_BUSY[$g]=""; done

cfg_idx=0
while (( cfg_idx < NUM_CFGS )); do
    free_gpu=""
    for g in "${GPU_ARR[@]}"; do
        pid="${GPU_BUSY[$g]}"
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
            free_gpu="$g"
            break
        fi
    done

    if [[ -z "$free_gpu" ]]; then
        sleep 5
        continue
    fi

    cfg="${CFG_ARR[$cfg_idx]}"
    echo "[dispatch-lat] $cfg -> GPU $free_gpu"
    bash tools/plan_b_run_latency.sh "$cfg" "$free_gpu" &
    GPU_BUSY[$free_gpu]=$!
    cfg_idx=$((cfg_idx + 1))
done

wait
echo "=== latency batch DONE ==="
ls -1 output/plan_b/latency/*.json | wc -l
