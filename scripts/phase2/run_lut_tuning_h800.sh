#!/bin/bash
# B1 LUT Tuning batch script — runs on H800 sequentially, one independent process per width
# Launch via: tmux new-session -d -s lut 'bash /exdata/jichengzhi/s2_tvm/run_lut_tuning.sh 2>&1 | tee /exdata/jichengzhi/s2_tvm/lut_tuning.log'

set -e

export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path 2>/dev/null || echo "")
export CUDA_VISIBLE_DEVICES=0

PYTHON=/exdata/jichengzhi/tvm310/bin/python
SCRIPT=/exdata/jichengzhi/s2_tvm/s2_2e_e2e.py
MODELS=/exdata/jichengzhi/s2_tvm/models
TRIALS=1000
REPS=500
OUT=/exdata/jichengzhi/s2_tvm/lut_results.csv

echo "===== B1 LUT Tuning START $(date) ====="
echo "GPU status (should be idle):"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1

# 0. Verification run — iso_s0_v2 vs known tuned=21752µs
echo ""
echo "--- STEP 0: Verify iso_s0_v2 (fresh work dir, expect tuned~21752µs) ---"
setsid $PYTHON $SCRIPT $MODELS/iso_s0_backbone.onnx iso_s0_v2 $TRIALS $OUT $REPS 0
echo "--- iso_s0_v2 done ---"

# 1. Single-stage sweep variants
echo ""
echo "--- STEP 1: s0_16 [16,128,256] ---"
setsid $PYTHON $SCRIPT $MODELS/s0_16_backbone.onnx s0_16 $TRIALS $OUT $REPS 0

echo "--- STEP 2: s0_32 [32,128,256] ---"
setsid $PYTHON $SCRIPT $MODELS/s0_32_backbone.onnx s0_32 $TRIALS $OUT $REPS 0

echo "--- STEP 3: s1_32 [64,32,256] ---"
setsid $PYTHON $SCRIPT $MODELS/s1_32_backbone.onnx s1_32 $TRIALS $OUT $REPS 0

echo "--- STEP 4: s1_64 [64,64,256] ---"
setsid $PYTHON $SCRIPT $MODELS/s1_64_backbone.onnx s1_64 $TRIALS $OUT $REPS 0

echo "--- STEP 5: s2_64 [64,128,64] ---"
setsid $PYTHON $SCRIPT $MODELS/s2_64_backbone.onnx s2_64 $TRIALS $OUT $REPS 0

echo "--- STEP 6: s2_128 [64,128,128] ---"
setsid $PYTHON $SCRIPT $MODELS/s2_128_backbone.onnx s2_128 $TRIALS $OUT $REPS 0

# 2. Validation combo widths
echo ""
echo "--- STEP 7: mix_a [32,96,192] ---"
setsid $PYTHON $SCRIPT $MODELS/mix_a_backbone.onnx mix_a $TRIALS $OUT $REPS 0

echo "--- STEP 8: mix_b [48,64,256] ---"
setsid $PYTHON $SCRIPT $MODELS/mix_b_backbone.onnx mix_b $TRIALS $OUT $REPS 0

echo "--- STEP 9: mix_c [16,128,128] ---"
setsid $PYTHON $SCRIPT $MODELS/mix_c_backbone.onnx mix_c $TRIALS $OUT $REPS 0

echo ""
echo "===== B1 LUT Tuning ALL DONE $(date) ====="
echo "Results in $OUT"
cat $OUT
