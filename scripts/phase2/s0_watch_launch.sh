#!/usr/bin/env bash
# Wait for a genuinely idle 4090 (total mem <= 80 MiB, util <= 5%), then launch S0
# pinned to it. Respects the "latency on fully-idle GPU only" discipline.
set -u
REPO=/home/jichengzhi/V2X
PY=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python
LOG="$REPO/results/S0_run.log"
mkdir -p "$REPO/results"

echo "[watch] $(date) waiting for a clean GPU (mem<=80MiB util<=5%)..." | tee -a "$LOG"
while true; do
  CLEAN=""
  while IFS=, read -r idx util mem; do
    idx=$(echo "$idx" | tr -d ' '); util=$(echo "$util" | tr -d ' '); mem=$(echo "$mem" | tr -d ' ')
    if [ "$mem" -le 80 ] && [ "$util" -le 5 ]; then CLEAN="$idx"; break; fi
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)

  if [ -n "$CLEAN" ]; then
    echo "[watch] $(date) GPU$CLEAN is clean → launching S0" | tee -a "$LOG"
    cd "$REPO" || exit 1
    CUDA_VISIBLE_DEVICES="$CLEAN" "$PY" scripts/phase2/s0_hwsw_coupling_probe.py >> "$LOG" 2>&1
    echo "[watch] $(date) S0 finished (exit $?)" | tee -a "$LOG"
    break
  fi
  sleep 60
done
