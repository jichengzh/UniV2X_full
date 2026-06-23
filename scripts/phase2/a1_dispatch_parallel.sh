#!/usr/bin/env bash
# A.1 phase1 parallel dispatcher: 11 anchors over 6 GPUs (0-5).
# Each (triplet, q_tag) is one job. GPUs serialize their own queue.

set -e
REPO=/home/jichengzhi/UniV2X
PYTHON=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python
LOG_DIR=/tmp/a1_parallel
mkdir -p "$LOG_DIR"

# 11 remaining anchors (T3_p37/Q_mix_s0 done; T1/T2 done)
# format: TRIPLET:Q_TAG:GPU
JOBS=(
  "T3_p37:Q_mix_s2:0"
  "T4_p50:Q_mix_s0:1"
  "T4_p50:Q_mix_s2:2"
  "T5_p62:Q_mix_s0:3"
  "T5_p62:Q_mix_s2:4"
  "T6_p75:Q_mix_s0:5"
  # batch 2 (GPUs free after batch 1)
  "T6_p75:Q_mix_s2:0"
  "T7_wide_shallow:Q_mix_s0:1"
  "T7_wide_shallow:Q_mix_s2:2"
  "T8_narrow_deep:Q_mix_s0:3"
  "T8_narrow_deep:Q_mix_s2:4"
)

# Run batch 1 (first 6 in parallel)
echo "=== batch 1 ($(date +%T)) ===" | tee -a "$LOG_DIR/dispatch.log"
declare -a PIDS=()
for job in "${JOBS[@]:0:6}"; do
  IFS=: read -r T Q G <<< "$job"
  log="$LOG_DIR/${T}_${Q}.log"
  echo "  launch GPU $G: $T/$Q -> $log" | tee -a "$LOG_DIR/dispatch.log"
  ( $PYTHON "$REPO/scripts/phase2/a1_run_one_anchor.py" \
      --triplet "$T" --q-tag "$Q" --gpu "$G" > "$log" 2>&1 ) &
  PIDS+=($!)
done
echo "  batch1 PIDs: ${PIDS[*]}" | tee -a "$LOG_DIR/dispatch.log"
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  PID $pid failed"; done
echo "=== batch 1 done ($(date +%T)) ===" | tee -a "$LOG_DIR/dispatch.log"

# Batch 2 (remaining 5)
echo "=== batch 2 ($(date +%T)) ===" | tee -a "$LOG_DIR/dispatch.log"
PIDS=()
for job in "${JOBS[@]:6}"; do
  IFS=: read -r T Q G <<< "$job"
  log="$LOG_DIR/${T}_${Q}.log"
  echo "  launch GPU $G: $T/$Q -> $log" | tee -a "$LOG_DIR/dispatch.log"
  ( $PYTHON "$REPO/scripts/phase2/a1_run_one_anchor.py" \
      --triplet "$T" --q-tag "$Q" --gpu "$G" > "$log" 2>&1 ) &
  PIDS+=($!)
done
echo "  batch2 PIDs: ${PIDS[*]}" | tee -a "$LOG_DIR/dispatch.log"
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  PID $pid failed"; done
echo "=== batch 2 done ($(date +%T)) ===" | tee -a "$LOG_DIR/dispatch.log"

# Also generate row JSONs for the 5 already-done anchors (cached fast path)
echo "=== compose row JSONs for cached anchors ($(date +%T)) ===" | tee -a "$LOG_DIR/dispatch.log"
PIDS=()
DONE_JOBS=(
  "T1_base:Q_mix_s0:0"
  "T1_base:Q_mix_s2:1"
  "T2_p25:Q_mix_s0:2"
  "T2_p25:Q_mix_s2:3"
  "T3_p37:Q_mix_s0:4"
)
for job in "${DONE_JOBS[@]}"; do
  IFS=: read -r T Q G <<< "$job"
  log="$LOG_DIR/${T}_${Q}.log"
  ( $PYTHON "$REPO/scripts/phase2/a1_run_one_anchor.py" \
      --triplet "$T" --q-tag "$Q" --gpu "$G" > "$log" 2>&1 ) &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
echo "=== all 16 anchors done ($(date +%T)) ===" | tee -a "$LOG_DIR/dispatch.log"
