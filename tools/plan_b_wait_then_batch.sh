#!/usr/bin/env bash
# Wait for the current 4 in-flight configs to complete, then run remaining 13
# in waves of 4-concurrent.
set -uo pipefail

cd /home/jichengzhi/UniV2X

# Phase 1: wait for in-flight configs to finish
WAIT_FOR=("C1" "D3" "E1" "E6")
echo "[wait] Waiting for in-flight: ${WAIT_FOR[*]}"
for cfg in "${WAIT_FOR[@]}"; do
    LOG="logs/plan_b/${cfg}.log"
    while ! grep -q "pts_bbox_NuScenes/amota" "$LOG" 2>/dev/null && \
          ! grep -qE "RuntimeError|AssertionError|Traceback" "$LOG" 2>/dev/null; do
        sleep 15
    done
    if grep -q "pts_bbox_NuScenes/amota" "$LOG" 2>/dev/null; then
        amota=$(grep "pts_bbox_NuScenes/amota" "$LOG" | tail -1 | grep -oE '0\.[0-9]+' | head -1)
        echo "[wait] $cfg done: AMOTA=$amota"
    else
        echo "[wait] $cfg failed"
    fi
    python tools/plan_b_extract_amota.py --config-id "$cfg" 2>&1 | tail -1
done

# Phase 2: launch remaining 13 configs across 4 GPUs
REMAINING="C2 D1 D2 D4 D5 E2 E3 E4 E5 E7 E8 E11 E12"
echo ""
echo "[wave] Launching remaining: $REMAINING"
bash tools/plan_b_batch.sh "$REMAINING" "0,1,4,5"
