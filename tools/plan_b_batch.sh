#!/usr/bin/env bash
# Plan B batch fanout: distribute configs across multiple GPUs in parallel.
#
# Usage:
#   bash tools/plan_b_batch.sh "<CONFIG_LIST>" "<GPU_LIST>"
# Example:
#   bash tools/plan_b_batch.sh "B1 B2 B3 C1 C2 C3" "1,2,4,5,6,7"
#
# Each config runs on one GPU; up to N parallel where N = len(GPU_LIST).
# Configs are queued and assigned to free GPUs as they become available.

set -euo pipefail

CONFIGS="${1:?Usage: $0 \"<CONFIG_IDS>\" \"<GPU_IDS>\"}"
GPUS="${2:?Need GPU list, e.g. \"1,2,4,5\"}"

cd /home/jichengzhi/UniV2X
mkdir -p logs/plan_b data/phase4/stage5_v3

IFS=',' read -ra GPU_ARR <<< "$GPUS"
read -ra CFG_ARR <<< "$CONFIGS"

NUM_GPUS=${#GPU_ARR[@]}
NUM_CFGS=${#CFG_ARR[@]}

echo "=== Plan B batch: $NUM_CFGS configs on $NUM_GPUS GPUs ==="
echo "  configs: ${CFG_ARR[*]}"
echo "  gpus:    ${GPU_ARR[*]}"

# Track running PIDs and which GPU is busy
declare -A GPU_BUSY
for g in "${GPU_ARR[@]}"; do GPU_BUSY[$g]=""; done

cfg_idx=0
while (( cfg_idx < NUM_CFGS )); do
    # Find a free GPU
    free_gpu=""
    for g in "${GPU_ARR[@]}"; do
        pid="${GPU_BUSY[$g]}"
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
            free_gpu="$g"
            break
        fi
    done

    if [[ -z "$free_gpu" ]]; then
        # All busy — wait briefly
        sleep 5
        continue
    fi

    cfg="${CFG_ARR[$cfg_idx]}"
    echo "[dispatch] $cfg -> GPU $free_gpu"
    bash tools/plan_b_run_one_config.sh "$cfg" "$free_gpu" &
    GPU_BUSY[$free_gpu]=$!
    cfg_idx=$((cfg_idx + 1))
done

# Wait for all to finish
echo "All dispatched, waiting for completion..."
wait
echo ""
echo "=== Plan B batch DONE ==="

# Extract metrics for all configs
echo ""
echo "=== Extracting metrics ==="
for cfg in "${CFG_ARR[@]}"; do
    python tools/plan_b_extract_amota.py --config-id "$cfg" 2>/dev/null || \
        echo "[skip metrics] $cfg"
done

echo ""
echo "=== Summary ==="
ls -la data/phase4/stage5_v3/
