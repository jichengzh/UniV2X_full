#!/usr/bin/env bash
# Retry the S0b bench on freshly-picked clean GPUs until the output CSV is produced.
# Engines already built. Robust to fierce/sustained contention.
set -u
REPO=/home/jichengzhi/V2X
PY=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python
LOG="$REPO/results/S0b_bench.log"
OUTCSV="$REPO/results/S0b_coupling_clean_4090.csv"
CACHE="$REPO/models/s0b_cache"
mkdir -p "$REPO/results"

need=( dense_fp16 dense_int8 p50_fp16 p50_int8 trap25_fp16 trap25_int8 p75_fp16 p75_int8 )
all_built() { for n in "${need[@]}"; do [ -f "$CACHE/${n}_opt3_ws4096.engine" ] || return 1; done; return 0; }
clean_gpu() {
  while IFS=, read -r idx util mem; do
    idx=$(echo "$idx"|tr -d ' '); util=$(echo "$util"|tr -d ' '); mem=$(echo "$mem"|tr -d ' ')
    if [ "$mem" -le 80 ] && [ "$util" -le 5 ]; then echo "$idx"; return; fi
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
}

echo "[s0b-watch] $(date) bench-retry loop start" | tee -a "$LOG"
attempt=0
while [ ! -f "$OUTCSV" ]; do
  if all_built; then
    G=$(clean_gpu)
    if [ -n "$G" ]; then
      attempt=$((attempt+1))
      echo "[s0b-watch] $(date) attempt $attempt on GPU$G" | tee -a "$LOG"
      cd "$REPO" || exit 1
      CUDA_VISIBLE_DEVICES="$G" "$PY" scripts/phase2/s0b_coupling_clean.py bench >> "$LOG" 2>&1
      if [ -f "$OUTCSV" ]; then
        echo "[s0b-watch] $(date) SUCCESS — CSV produced" | tee -a "$LOG"; break
      fi
      echo "[s0b-watch] $(date) attempt $attempt incomplete, retrying" | tee -a "$LOG"
    fi
  fi
  sleep 45
done
echo "[s0b-watch] $(date) done" | tee -a "$LOG"
