#!/bin/bash
# E-couple-A driver: wait for a fully-idle H800 GPU, then sweep tile x width.
# argmin_tile(latency|W) shift across W  => HW knob couples with pruning width.
set -u
cd /exdata/jichengzhi/s2_tvm
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path)
PY=/exdata/jichengzhi/tvm310/bin/python
OUT=/exdata/jichengzhi/s2_tvm/s2_2d_couple_tile.csv
LOG=/exdata/jichengzhi/s2_tvm/2d_couple.log
rm -f "$OUT"

# --- wait for an idle GPU (foreign mem <= 50 MiB, util <= 2%) ---
pick_idle() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
  | awk -F',' '{gsub(/ /,"",$1);gsub(/ /,"",$2);gsub(/ /,"",$3); if($2<=50 && $3<=2){print $1; exit}}'
}
echo "[wait] polling for idle GPU $(date)" | tee -a "$LOG"
IDLE=""
for i in $(seq 1 720); do   # up to ~6h at 30s
  IDLE=$(pick_idle)
  if [ -n "$IDLE" ]; then echo "[wait] GPU $IDLE idle at $(date)" | tee -a "$LOG"; break; fi
  sleep 30
done
if [ -z "$IDLE" ]; then echo "[wait] NO idle GPU after timeout $(date)" | tee -a "$LOG"; exit 9; fi
export CUDA_VISIBLE_DEVICES=$IDLE

M=256
WIDTHS="32 48 64 128"
# curated tile set (TM TN BK)
TILES="8,8,8 8,8,16 16,16,8 16,16,16 16,16,32 32,32,16 32,8,16 8,32,16"
for W in $WIDTHS; do
  for T in $TILES; do
    IFS=',' read TM TN BK <<< "$T"
    LBL="W${W}_t${TM}x${TN}x${BK}"
    $PY s2_2d_couple_tile.py $M $W $TM $TN $BK "$LBL" "$OUT" 1000 >> "$LOG" 2>&1
  done
done
echo "[done] sweep complete $(date)" | tee -a "$LOG"
echo "=== CSV ===" | tee -a "$LOG"
cat "$OUT" | tee -a "$LOG"
