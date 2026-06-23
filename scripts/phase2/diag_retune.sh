#!/bin/bash
# Diagnostic: does FRESH MetaSchedule tuning still deliver in current TVM env?
# Re-tune base (known 9x from cached db) into a NEW work dir + p75, via the
# trusted s2_2e_e2e.py. Each python invocation = its own process (CUDA isolation).
cd /exdata/jichengzhi/s2_tvm
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path)
export CUDA_VISIBLE_DEVICES=0
CSV=results/diag_retune.csv
rm -f "$CSV"
for cfg in "base_backbone.onnx base_retest" "p75_backbone.onnx p75_retest"; do
  set -- $cfg
  echo "=== RETUNE $2 ($1) $(date +%H:%M:%S) ==="
  /exdata/jichengzhi/tvm310/bin/python s2_2e_e2e.py "models/$1" "$2" 1000 "$CSV" 200 0
  echo "=== exit=$? $2 done $(date +%H:%M:%S) ==="
done
echo "=== DIAG ALL DONE ==="
cat "$CSV"
