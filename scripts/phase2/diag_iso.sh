#!/bin/bash
# Run the 3 single-stage-misaligned iso widths via the PROVEN s2_2e_e2e.py,
# each in its OWN process (CUDA-crash isolation), fresh _retest work dirs.
# Appends to the same diag_retune.csv (already has base_retest, p75_retest).
cd /exdata/jichengzhi/s2_tvm
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path)
export CUDA_VISIBLE_DEVICES=0
CSV=results/diag_retune.csv
for cfg in "iso_s0_backbone.onnx iso_s0_retest" \
           "iso_s1_backbone.onnx iso_s1_retest" \
           "iso_s2_backbone.onnx iso_s2_retest"; do
  set -- $cfg
  echo "=== RETUNE $2 ($1) $(date +%H:%M:%S) ==="
  /exdata/jichengzhi/tvm310/bin/python s2_2e_e2e.py "models/$1" "$2" 1000 "$CSV" 200 0
  echo "=== exit=$? $2 done $(date +%H:%M:%S) ==="
done
echo "=== ISO ALL DONE ==="
cat "$CSV"
