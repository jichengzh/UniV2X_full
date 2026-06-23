#!/bin/bash
# L3 Master: CoDriving multi-point REAL measurement
# STEP 1: Kill extrapolation (p25/p75 FP32 TVM, >=4 real-measured widths)
# STEP 2: s0-mismatch probe (cod_s0_48 FP32 TVM, direct test of s0 rank-flip)
# STEP 3: INT8 alignment screen (synthetic 3x3 conv, misaligned vs aligned Cin)
#
# GPU assignments (verified idle: 4,5,6,7):
#   GPU 4: p75 [16,32,64] FP32 TVM - 1000 trials
#   GPU 5: p25 [48,96,192] FP32 TVM - 1000 trials
#   GPU 6: cod_s0_48 [48,128,256] FP32 TVM - 1000 trials (re-run with clean workdir)
#   GPU 7: INT8 screen (3x3 conv, aligned vs misaligned Cin, ~200 trials each)

set -e
cd /exdata/jichengzhi/s2_tvm
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path)
PY=/exdata/jichengzhi/tvm310/bin/python
V2X_RESULTS=/home/jichengzhi/V2X/results
TVM_RESULTS=/exdata/jichengzhi/s2_tvm/results

mkdir -p $V2X_RESULTS $TVM_RESULTS

echo "=== L3 START $(date) ==="
echo "=== GPU status: ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | head -8
echo "=== nvcc version: ==="
nvcc --version | head -2

# ---- STEP 1a: Export p25 ONNX (quick, ~60s, blocking) ----
echo ""
echo "=== STEP 1a: Export p25 backbone ONNX [48,96,192] ==="
P25_ONNX=models/codriving_cache/p25_backbone.onnx
if [ -f "$P25_ONNX" ]; then
  echo "[SKIP] p25 ONNX already exists"
else
  $PY /home/jichengzhi/V2X/scripts/phase2/cod_l3_p25_export.py
fi
echo "=== p25 export done at $(date +%H:%M:%S) ==="
ls -la $P25_ONNX

# ---- STEP 1b: FP32 TVM for p75 (GPU 4) ----
echo ""
echo "=== STEP 1b: FP32 TVM for p75 [16,32,64] on GPU 4 ==="
rm -rf ms_work_cod_cod_p75
CUDA_VISIBLE_DEVICES=4 $PY s2_codriving_e2e.py \
  models/codriving_cache/p75_backbone.onnx \
  cod_p75 1000 ${TVM_RESULTS}/cod_p75_result.csv 2 500 0 \
  > cod_l3_p75_gpu4.log 2>&1 &
PID_P75=$!
echo "  p75 PID=$PID_P75 started on GPU 4"

# ---- STEP 1c: FP32 TVM for p25 (GPU 5) ----
echo ""
echo "=== STEP 1c: FP32 TVM for p25 [48,96,192] on GPU 5 ==="
rm -rf ms_work_cod_cod_p25
CUDA_VISIBLE_DEVICES=5 $PY s2_codriving_e2e.py \
  models/codriving_cache/p25_backbone.onnx \
  cod_p25 1000 ${TVM_RESULTS}/cod_p25_result.csv 2 500 0 \
  > cod_l3_p25_gpu5.log 2>&1 &
PID_P25=$!
echo "  p25 PID=$PID_P25 started on GPU 5"

# ---- STEP 2: s0-mismatch probe - cod_s0_48 (GPU 6) ----
echo ""
echo "=== STEP 2: s0-mismatch probe cod_s0_48 [48,128,256] on GPU 6 ==="
rm -rf ms_work_cod_cod_s0_48
CUDA_VISIBLE_DEVICES=6 $PY s2_codriving_e2e.py \
  models/codriving_cache/cod_s0_48_backbone.onnx \
  cod_s0_48 1000 ${TVM_RESULTS}/cod_s0_48_result.csv 2 500 0 \
  > cod_l3_s0_48_gpu6.log 2>&1 &
PID_S0_48=$!
echo "  s0_48 PID=$PID_S0_48 started on GPU 6"

# ---- STEP 3: INT8 screen (GPU 7) ----
echo ""
echo "=== STEP 3: INT8 alignment screen on GPU 7 ==="
$PY /home/jichengzhi/V2X/scripts/phase2/cod_l3_int8_screen.py \
  ${TVM_RESULTS}/cod_int8_screen.csv 7 200 \
  > cod_l3_int8_screen_gpu7.log 2>&1 &
PID_INT8=$!
echo "  INT8 screen PID=$PID_INT8 started on GPU 7"

echo ""
echo "=== ALL JOBS LAUNCHED ==="
echo "  PIDs: p75=$PID_P75 p25=$PID_P25 s0_48=$PID_S0_48 int8=$PID_INT8"

# Wait for INT8 screen first (fastest ~20-30 min)
echo ""
echo "=== Waiting for INT8 screen ==="
wait $PID_INT8
INT8_EXIT=$?
echo "=== INT8 screen DONE (exit=$INT8_EXIT) at $(date +%H:%M:%S) ==="
echo "--- INT8 results ---"
cat ${TVM_RESULTS}/cod_int8_screen.csv 2>/dev/null || echo "NO INT8 RESULTS"

# Wait for FP32 jobs
echo ""
echo "=== Waiting for FP32 TVM jobs ==="
wait $PID_P75; echo "=== p75 DONE at $(date +%H:%M:%S) ===" && cat ${TVM_RESULTS}/cod_p75_result.csv
wait $PID_P25; echo "=== p25 DONE at $(date +%H:%M:%S) ===" && cat ${TVM_RESULTS}/cod_p25_result.csv
wait $PID_S0_48; echo "=== s0_48 DONE at $(date +%H:%M:%S) ===" && cat ${TVM_RESULTS}/cod_s0_48_result.csv

# ---- Merge FP32 results + existing base/p50 into unified CSV ----
echo ""
echo "=== Merging all FP32 results ==="
MERGED=${TVM_RESULTS}/cod_l3_fp32_all.csv
echo "label,onnx,trials,batch,default_us,tuned_us,e2e_ratio,tune_s,notes" > $MERGED
# Existing: base, p50
echo "base,base_backbone.onnx,1000,2,16680.47,8057.79,2.070,1581,existing_measured" >> $MERGED
echo "p50,p50_backbone.onnx,1000,2,3643.15,1609.10,2.264,1474,existing_measured" >> $MERGED
# New: p25, p75 (strip header, add notes)
tail -n +2 ${TVM_RESULTS}/cod_p25_result.csv 2>/dev/null | awk -F',' '{print $0",new_l3"}' >> $MERGED
tail -n +2 ${TVM_RESULTS}/cod_p75_result.csv 2>/dev/null | awk -F',' '{print $0",new_l3"}' >> $MERGED

echo "--- Merged FP32 summary ---"
cat $MERGED

# ---- Copy to V2X results ----
cp $MERGED $V2X_RESULTS/codriving_tvm_p25_p75.csv 2>/dev/null || true
cp ${TVM_RESULTS}/cod_int8_screen.csv $V2X_RESULTS/codriving_s0probe_int8.csv 2>/dev/null || true
cp ${TVM_RESULTS}/cod_s0_48_result.csv $V2X_RESULTS/codriving_s0probe_fp16.csv 2>/dev/null || true

echo ""
echo "=== L3 ALL DONE at $(date) ==="
echo "Results saved to:"
echo "  $V2X_RESULTS/codriving_tvm_p25_p75.csv (FP32 p25/p75)"
echo "  $V2X_RESULTS/codriving_s0probe_fp16.csv (s0 mismatch FP32)"
echo "  $V2X_RESULTS/codriving_s0probe_int8.csv (INT8 alignment)"
