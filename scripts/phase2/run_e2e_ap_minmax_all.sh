#!/bin/bash
# Run e2e AP eval for 8 INT8 minmax engines (4 official + 4 self-trained)
# vs subnet AP from class_a_pyramid_full to confirm method 1 is real.
set -uo pipefail

REPO=/home/jichengzhi/UniV2X
PY=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python
OUT_DIR="$REPO/results/e2e_ap_minmax_v1"
mkdir -p "$OUT_DIR"

# (tag, ckpt_dir)
declare -A CKPT_DIR=(
  [T1_base]="/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
  [T2_p25]="/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10"
  [T4_p50]="/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10"
  [T6_p75]="/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10"
  [T3_p37]="$REPO/models/dataset_a_cache/ft_040_080_160"
  [T5_p62]="$REPO/models/dataset_a_cache/ft_024_056_128"
  [T7_wide_shallow]="$REPO/models/dataset_a_cache/ft_048_064_128"
  [T8_narrow_deep]="$REPO/models/dataset_a_cache/ft_024_048_192"
)

ENGINE_DIR="$REPO/models/e2e_cache"

for tag in T1_base T2_p25 T3_p37 T4_p50 T5_p62 T6_p75 T7_wide_shallow T8_narrow_deep; do
  engine="$ENGINE_DIR/${tag}_Q_int8_mm.engine"
  ckpt="${CKPT_DIR[$tag]}"
  report="$OUT_DIR/${tag}_int8_mm.json"
  if [[ -f "$report" ]]; then
    echo "[skip] $tag report exists: $report"
    continue
  fi
  echo "[run ] $tag — engine=$(basename $engine)"
  CUDA_VISIBLE_DEVICES=0 $PY "$REPO/scripts/phase2/e2e_eval_ap.py" \
    --engine "$engine" \
    --ckpt-dir "$ckpt" \
    --max-voxels 32000 \
    --n-samples 500 \
    --tag "${tag}_int8_mm_e2e" \
    --report "$report" 2>&1 | tail -8
  echo "----"
done

echo ""
echo "===== Summary ====="
$PY - <<'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob("/home/jichengzhi/UniV2X/results/e2e_ap_minmax_v1/*.json")):
    d = json.load(open(p))
    rows.append((os.path.basename(p).replace(".json",""),
                 d.get("ap30"), d.get("ap50"), d.get("ap70"),
                 d.get("n_samples"), d.get("n_trt"), d.get("n_pytorch_fallback")))
print(f"{'tag':30s} {'ap30':>6s} {'ap50':>6s} {'ap70':>6s} {'n':>4s} {'trt':>4s} {'fb':>4s}")
for r in rows:
    print(f"{r[0]:30s} {r[1] or 0:6.3f} {r[2] or 0:6.3f} {r[3] or 0:6.3f} {r[4] or 0:4d} {r[5] or 0:4d} {r[6] or 0:4d}")
PY
