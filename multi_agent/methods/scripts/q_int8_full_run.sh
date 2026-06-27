#!/bin/bash
# Q_INT8_FULL: Complete INT8 measurement suite on H800 GPU 6
# Runs base gate + all 6 key pairs
# Each width in its own subprocess (TVM feedback-tvm-tune-apply-fresh-workdir lesson)
# Author: hw-optimizer, 2026-06-21

set -o pipefail
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path)
export CUDA_HOME=/usr/local/cuda-12.2

GPU=${CUDA_VISIBLE_DEVICES:-6}
PYTHON=/exdata/jichengzhi/tvm310/bin/python
WORKER=/exdata/jichengzhi/s2_tvm/q1_int8_pairs_worker.py
MODELS=/exdata/jichengzhi/s2_tvm/models
BASE_OUT=/exdata/jichengzhi/s2_tvm/q_int8_base_gate.csv
PAIRS_OUT=/exdata/jichengzhi/s2_tvm/q_int8_pairs.csv
LOG_DIR=/exdata/jichengzhi/s2_tvm/q_int8_logs
TRIALS=300
REPS=200

echo "=== Q_INT8_FULL: GPU=$GPU TRIALS=$TRIALS $(date) ==="
mkdir -p $LOG_DIR
mkdir -p /exdata/jichengzhi/tvm_int8

# Verify GPU is idle
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i $GPU | tr -d ' %')
MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $GPU | tr -d ' MiB')
echo "[CHECK] GPU $GPU: util=${UTIL}% mem=${MEM}MiB"
if [ "$UTIL" -gt 5 ]; then
    echo "WARNING: GPU $GPU util=${UTIL}% > 5%, but proceeding as coordinator approved util=0% check"
fi

# Verify nvcc
which nvcc || { echo "ERROR: nvcc not found, check PATH"; exit 1; }
echo "[CHECK] nvcc: $(which nvcc)"
echo "[CHECK] PYTHON: $PYTHON"

# Remove old result CSVs
rm -f $BASE_OUT $PAIRS_OUT
echo "label,prec,sched,lat_us,source,notes" > $BASE_OUT
echo "num_filters,label,sched,lat_us,source" > $PAIRS_OUT

# ============================================================
# STEP 1: Base gate - [64,128,256] INT8
# ============================================================
echo ""
echo "=== STEP1: Base gate [64,128,256] $(date +%H:%M:%S) ==="
BASE_ONNX=$MODELS/base_backbone.onnx
if [ ! -f "$BASE_ONNX" ]; then
    echo "ERROR: $BASE_ONNX not found"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU $PYTHON $WORKER \
    "base" "$BASE_ONNX" "$BASE_OUT" "$TRIALS" "$REPS" \
    2>&1 | tee $LOG_DIR/base.log
EXIT_BASE=${PIPESTATUS[0]}
echo "=== Base gate done exit=$EXIT_BASE $(date +%H:%M:%S) ==="
sleep 5

# ============================================================
# STEP 2: Key pairs
# trap25 = [48,96,192]   W_g pair1 (misaligned s0=96)
# pad64  = [64,96,192]   P_g pair1 (aligned s0=128)
# mix_b  = [48,64,256]   W_g pair2 (misaligned s1=64)
# s1_64  = [64,64,256]   P_g pair2
# mix_d  = [48,128,128]  W_g pair3 (KEY: misaligned s2=128)
# s2_128 = [64,128,128]  P_g pair3 (KEY)
# ============================================================

declare -A ONNX_MAP
ONNX_MAP[trap25]="$MODELS/trap25_backbone.onnx"
ONNX_MAP[pad64]="$MODELS/trap25_pad64_backbone.onnx"
ONNX_MAP[mix_b]="$MODELS/mix_b_backbone.onnx"
ONNX_MAP[s1_64]="$MODELS/s1_64_backbone.onnx"
ONNX_MAP[mix_d]="$MODELS/mix_d_backbone.onnx"
ONNX_MAP[s2_128]="$MODELS/s2_128_backbone.onnx"

declare -A FILTERS_MAP
FILTERS_MAP[trap25]="[48,96,192]"
FILTERS_MAP[pad64]="[64,96,192]"
FILTERS_MAP[mix_b]="[48,64,256]"
FILTERS_MAP[s1_64]="[64,64,256]"
FILTERS_MAP[mix_d]="[48,128,128]"
FILTERS_MAP[s2_128]="[64,128,128]"

for LABEL in trap25 pad64 mix_b s1_64 mix_d s2_128; do
    ONNX=${ONNX_MAP[$LABEL]}
    FILTERS=${FILTERS_MAP[$LABEL]}
    LOG=$LOG_DIR/${LABEL}.log
    TMP_CSV=$LOG_DIR/${LABEL}_tmp.csv

    echo ""
    echo "=== Pair: $LABEL $FILTERS $(date +%H:%M:%S) ==="

    # Check GPU still idle
    UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i $GPU | tr -d ' %')
    echo "[CHECK] GPU $GPU util=${UTIL}% before $LABEL"

    if [ ! -f "$ONNX" ]; then
        echo "ERROR: ONNX not found: $ONNX — skipping $LABEL"
        echo "$FILTERS,$LABEL,default,-1.0,H800_TVM_int8_SKIP" >> $PAIRS_OUT
        echo "$FILTERS,$LABEL,tuned,-1.0,H800_TVM_int8_SKIP" >> $PAIRS_OUT
        continue
    fi

    # Run worker in subprocess — writes to TMP_CSV
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $WORKER \
        "$LABEL" "$ONNX" "$TMP_CSV" "$TRIALS" "$REPS" \
        2>&1 | tee $LOG
    EXIT_CODE=${PIPESTATUS[0]}
    echo "=== $LABEL exit=$EXIT_CODE $(date +%H:%M:%S) ==="

    # Append rows from TMP_CSV to PAIRS_OUT (with filters column)
    if [ -f "$TMP_CSV" ]; then
        # Skip header line, add filters prefix
        tail -n +2 $TMP_CSV | while IFS=, read -r lbl prec sched lat_us src notes; do
            echo "$FILTERS,$lbl,$sched,$lat_us,H800_TVM_int8_real" >> $PAIRS_OUT
        done
    else
        echo "[ERROR] No TMP_CSV for $LABEL"
        echo "$FILTERS,$LABEL,default,-1.0,H800_TVM_int8_FAIL" >> $PAIRS_OUT
        echo "$FILTERS,$LABEL,tuned,-1.0,H800_TVM_int8_FAIL" >> $PAIRS_OUT
    fi

    sleep 5
done

echo ""
echo "=== ALL DONE $(date) ==="
echo "=== Base gate results: ==="
cat $BASE_OUT
echo ""
echo "=== Pairs results: ==="
cat $PAIRS_OUT

echo ""
echo "=== Q-rank-flip analysis: pair3 (mix_d vs s2_128) ==="
echo "=== Is mix_d[48,128,128] INT8 default > s2_128[64,128,128]? ==="
