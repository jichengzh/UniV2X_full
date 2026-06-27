#!/bin/bash
# Q1: Run INT8 pairs measurements on H800 — subprocess isolated per label
# Must run on H800 with idle GPU
# Usage: CUDA_VISIBLE_DEVICES=<gpu> bash q1_int8_pairs_run.sh

set -e
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path)
export CUDA_HOME=/usr/local/cuda-12.2

PYTHON=/exdata/jichengzhi/tvm310/bin/python
WORKER=/exdata/jichengzhi/s2_tvm/q1_int8_pairs_worker.py
MODELS=/exdata/jichengzhi/s2_tvm/models
OUT_CSV=/exdata/jichengzhi/s2_tvm/q1_int8_pairs_result.csv
LOG_DIR=/exdata/jichengzhi/s2_tvm/q1_logs
TRIALS=500
REPS=200
GPU=${CUDA_VISIBLE_DEVICES:-4}

echo "=== Q1 INT8 pairs: GPU=$GPU TRIALS=$TRIALS ==="
echo "=== $(date) ==="
mkdir -p $LOG_DIR
mkdir -p /exdata/jichengzhi/tvm_int8

# Remove old result CSV so we start fresh
rm -f $OUT_CSV

# Check nvcc
which nvcc || { echo "ERROR: nvcc not found"; exit 1; }

# Define label -> onnx_fp32 mapping
# trap25 = [48,96,192]   -> trap25_backbone.onnx
# pad64  = [64,96,192]   -> trap25_pad64_backbone.onnx  (zero-padded pad64)
# mix_b  = [48,64,256]   -> mix_b_backbone.onnx
# s1_64  = [64,64,256]   -> s1_64_backbone.onnx

declare -A ONNX_MAP
ONNX_MAP[trap25]="$MODELS/trap25_backbone.onnx"
ONNX_MAP[pad64]="$MODELS/trap25_pad64_backbone.onnx"
ONNX_MAP[mix_b]="$MODELS/mix_b_backbone.onnx"
ONNX_MAP[s1_64]="$MODELS/s1_64_backbone.onnx"

# Run each label in its own subprocess
for LABEL in trap25 pad64 mix_b s1_64; do
    ONNX=${ONNX_MAP[$LABEL]}
    LOG="$LOG_DIR/${LABEL}.log"
    echo ""
    echo "=== Starting $LABEL $(date +%H:%M:%S) ==="
    echo "    ONNX: $ONNX"
    echo "    LOG:  $LOG"

    if [ ! -f "$ONNX" ]; then
        echo "ERROR: ONNX not found: $ONNX"
        continue
    fi

    # Run in isolated subprocess (no &, sequential for clean GPU state)
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $WORKER \
        "$LABEL" "$ONNX" "$OUT_CSV" "$TRIALS" "$REPS" \
        2>&1 | tee "$LOG"

    EXIT_CODE=${PIPESTATUS[0]}
    echo "=== $LABEL done exit=$EXIT_CODE $(date +%H:%M:%S) ==="

    # Small pause between subprocesses
    sleep 5
done

echo ""
echo "=== ALL PAIRS DONE $(date) ==="
echo "=== Result CSV: $OUT_CSV ==="
cat $OUT_CSV
