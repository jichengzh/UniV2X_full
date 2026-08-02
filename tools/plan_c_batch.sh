#!/usr/bin/env bash
# Plan C batch: distribute tiny configs across multiple GPUs.
# Same pattern as plan_b_batch.sh but for tiny baseline.
#
# Usage: bash tools/plan_c_batch.sh "<CONFIG_LIST>" "<GPU_LIST>"

set -uo pipefail

CONFIGS="${1:?Usage: $0 \"<CONFIG_IDS>\" \"<GPU_IDS>\"}"
GPUS="${2:?Need GPU list}"

cd /home/jichengzhi/UniV2X
mkdir -p logs/plan_c data/phase4/stage5_v4_tiny

IFS=',' read -ra GPU_ARR <<< "$GPUS"
read -ra CFG_ARR <<< "$CONFIGS"

NUM_GPUS=${#GPU_ARR[@]}
NUM_CFGS=${#CFG_ARR[@]}
echo "=== Plan C tiny batch: $NUM_CFGS configs on $NUM_GPUS GPUs ==="

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
    if [[ -z "$free_gpu" ]]; then sleep 5; continue; fi

    cfg="${CFG_ARR[$cfg_idx]}"
    echo "[dispatch] $cfg -> GPU $free_gpu"
    bash tools/plan_c_run_one_config.sh "$cfg" "$free_gpu" &
    GPU_BUSY[$free_gpu]=$!
    cfg_idx=$((cfg_idx + 1))
done

wait
echo "=== Plan C tiny batch DONE ==="
ls -1 logs/plan_c/*.log 2>/dev/null | wc -l
