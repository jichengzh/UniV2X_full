#!/bin/bash
# Run q0 and q1 INT8 measurements on GPU 7 when cod_l3_int8_screen finishes.
# Runs in background via nohup. Check results in LOG_FILE.
# Usage: bash run_q0_q1_when_free.sh (runs on GPU 7 by default)

LOG_FILE=/exdata/jichengzhi/s2_tvm/q0_q1_int8_run.log
RESULT_DIR=/home/jichengzhi/V2X/results

export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path)
export CUDA_HOME=/usr/local/cuda-12.2

PYTHON=/exdata/jichengzhi/tvm310/bin/python
GPU=7
WAIT_PID=2330347  # cod_l3_int8_screen main process

log() {
    echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG_FILE
}

log "=== Q0+Q1 INT8 runner started, waiting for PID $WAIT_PID (cod_l3_int8_screen) ==="

# Wait for the cod_l3_int8_screen to finish
while ps -p $WAIT_PID > /dev/null 2>&1; do
    done_count=$(wc -l < /exdata/jichengzhi/s2_tvm/results/cod_int8_screen.csv 2>/dev/null || echo 0)
    gpu7_util=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | awk 'NR==8{print}')
    log "Still waiting... done_rows=$done_count GPU7=$gpu7_util"
    sleep 60
done

log "=== PID $WAIT_PID finished! ==="
log "GPU state after experiment:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tee -a $LOG_FILE

# Wait for GPU 7 memory to clear
log "Waiting 30s for GPU memory to clear..."
sleep 30

log "GPU state after 30s wait:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tee -a $LOG_FILE

# Check if GPU 7 popen_workers are gone
remaining_workers=$(pgrep -a python 2>/dev/null | grep popen_worker | while read pid cmd; do
    cvd=$(cat /proc/$pid/environ 2>/dev/null | tr '\0' '\n' | grep CUDA_VISIBLE_DEVICES 2>/dev/null)
    echo "$cvd"
done | grep "=$GPU" | wc -l)
log "Remaining popen_workers on GPU $GPU: $remaining_workers"

GPU7_UTIL=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | awk 'NR==8{print $2}' | tr -d ' ,%')
GPU7_MEM=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | awk -F',' 'NR==8{gsub(/ /,"",$3); print $3}' | tr -d 'MiB ')

log "GPU $GPU: util=${GPU7_UTIL}% mem=${GPU7_MEM}MiB"

# ===== STEP 1: Run q0 default + full gate =====
log ""
log "=== STEP 1: Q0 INT8 base gate ==="
log "GPU before q0:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=$GPU $PYTHON /exdata/jichengzhi/s2_tvm/q0_int8_gate_fixed.py \
    2>&1 | tee -a $LOG_FILE
Q0_EXIT=$?
log "q0 exit=$Q0_EXIT"

# Read q0 result
log "Q0 result JSON:"
cat /exdata/jichengzhi/s2_tvm/q0_int8_gate_fixed_result.json 2>/dev/null | tee -a $LOG_FILE

# Write to official results
if [ -f /exdata/jichengzhi/s2_tvm/q0_int8_gate_fixed_result.json ]; then
    INT8_DEF=$(python3 -c "import json; d=json.load(open('/exdata/jichengzhi/s2_tvm/q0_int8_gate_fixed_result.json')); print(d.get('int8_default_us',-1))" 2>/dev/null)
    INT8_TUN=$(python3 -c "import json; d=json.load(open('/exdata/jichengzhi/s2_tvm/q0_int8_gate_fixed_result.json')); print(d.get('int8_tuned_us',-1))" 2>/dev/null)
    log "INT8 default=${INT8_DEF}us tuned=${INT8_TUN}us"

    # Write base gate CSV
    cat > $RESULT_DIR/q_int8_base_gate.csv << CSVEOF
width,prec,sched,lat_us,source,notes
"[64,128,256]",int8,default,${INT8_DEF},H800_TVM_int8,q0_int8_gate_fixed GPU$GPU
"[64,128,256]",int8,tuned,${INT8_TUN},H800_TVM_int8,q0_int8_gate_fixed GPU$GPU 300trials
CSVEOF
    log "Written: $RESULT_DIR/q_int8_base_gate.csv"
fi

# ===== STEP 2: Run INT8 pairs =====
log ""
log "=== STEP 2: Q1 INT8 key pairs ==="

# Clear previous results
rm -f /exdata/jichengzhi/s2_tvm/q1_int8_pairs_result.csv
mkdir -p /exdata/jichengzhi/s2_tvm/q1_logs
mkdir -p /exdata/jichengzhi/tvm_int8

declare -A ONNX_MAP
MODELS=/exdata/jichengzhi/s2_tvm/models
ONNX_MAP[trap25]="$MODELS/trap25_backbone.onnx"
ONNX_MAP[pad64]="$MODELS/trap25_pad64_backbone.onnx"
ONNX_MAP[mix_b]="$MODELS/mix_b_backbone.onnx"
ONNX_MAP[s1_64]="$MODELS/s1_64_backbone.onnx"

for LABEL in trap25 pad64 mix_b s1_64; do
    ONNX=${ONNX_MAP[$LABEL]}
    WORKER_LOG="/exdata/jichengzhi/s2_tvm/q1_logs/${LABEL}.log"

    log ""
    log "--- Running $LABEL $(date +%H:%M:%S) ---"
    log "GPU before $LABEL:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tee -a $LOG_FILE

    if [ ! -f "$ONNX" ]; then
        log "ERROR: ONNX not found: $ONNX"
        continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON /exdata/jichengzhi/s2_tvm/q1_int8_pairs_worker.py \
        "$LABEL" "$ONNX" /exdata/jichengzhi/s2_tvm/q1_int8_pairs_result.csv 500 200 \
        2>&1 | tee "$WORKER_LOG" | tee -a $LOG_FILE

    EXIT_CODE=$?
    log "--- $LABEL done exit=$EXIT_CODE $(date +%H:%M:%S) ---"
    sleep 10
done

log ""
log "=== ALL DONE ==="
log "Pairs result:"
cat /exdata/jichengzhi/s2_tvm/q1_int8_pairs_result.csv 2>/dev/null | tee -a $LOG_FILE

# Copy pairs result to official location
cp /exdata/jichengzhi/s2_tvm/q1_int8_pairs_result.csv $RESULT_DIR/q_int8_pairs.csv 2>/dev/null
log "Copied to $RESULT_DIR/q_int8_pairs.csv"
log "=== SCRIPT COMPLETE ==="
