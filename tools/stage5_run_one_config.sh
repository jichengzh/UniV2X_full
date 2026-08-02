#!/usr/bin/env bash
# Stage 5.1 single-config end-to-end pipeline.
#
# Steps:
#   1. Apply prune_config.json to sub_vehicle_tiny + stage1 ckpt → pruned ego ckpt
#   2. Reconstruct coop ckpt: prefix ego with 'model_ego_agent.' + merge with stage2 inf
#   3. Run dist_eval on coop_tiny.py + merged ckpt → results_nusc.json
#   4. Run eval_real_recall.py → metrics JSON
#
# Usage:
#   bash tools/stage5_run_one_config.sh <CONFIG_ID> [GPU_IDS]
# Example:
#   bash tools/stage5_run_one_config.sh B1 1
#   bash tools/stage5_run_one_config.sh A1 1,2

set -euo pipefail

CONFIG_ID="${1:?Usage: $0 <CONFIG_ID> [GPU_IDS]}"
GPUS_ARG="${2:-1}"
NUM_GPUS=$(echo "$GPUS_ARG" | awk -F',' '{print NF}')

# shellcheck source=/dev/null
source /home/jichengzhi/miniconda3/etc/profile.d/conda.sh
conda activate UniV2X_2.0

cd /home/jichengzhi/UniV2X
mkdir -p ckpts/stage5_smoke ckpts/stage5_merged data/phase4/stage5_metrics

PRUNE_CFG="prune_configs/active_${CONFIG_ID}.json"
EGO_PRUNED="ckpts/stage5_smoke/${CONFIG_ID}_ego_pruned.pth"
COOP_MERGED="ckpts/stage5_merged/${CONFIG_ID}_coop.pth"
METRICS_OUT="data/phase4/stage5_metrics/${CONFIG_ID}.json"

if [[ ! -f "$PRUNE_CFG" ]]; then
    echo "ERROR: $PRUNE_CFG not found"; exit 1
fi

echo "=== [1/4] Prune ego (sub_vehicle config + stage1 ckpt) ==="
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="${GPUS_ARG%%,*}" python tools/prune_and_eval.py \
    --config projects/configs_e2e_univ2x/univ2x_sub_vehicle_tiny.py \
    --checkpoint projects/work_dirs_e2e_univ2x/univ2x_sub_vehicle_tiny/epoch_30.pth \
    --prune-config "$PRUNE_CFG" \
    --output "$EGO_PRUNED" \
    --gpu-id 0 2>&1 | tail -20

echo ""
echo "=== [2/4] Build coop ckpt: pruned ego + stage2 inf ==="
# NOTE: V2X fusion injection (cross_agent_query_interaction + seg_head.cross_lane_fusion)
# was tested on 2026-05-01 and REVERTED. Injecting trained V2X weights from coop_tiny ep30
# DEGRADED F1 (-1.1pt baseline, +0.05pt B1 violating monotonicity). Cause: V2X module was
# co-trained with coop_tiny ego/inf jointly, but our merge uses stage1_ego + stage2_inf
# (trained independently) → distribution mismatch. Random init effectively zeroes V2X
# output, yielding cleaner ego-only signal. See logs/v2x_inject/batch.log.
PYTHONPATH=$PWD python - <<PYEOF
import torch
ego = torch.load("${EGO_PRUNED}", map_location="cpu")
inf = torch.load("projects/work_dirs_e2e_univ2x/univ2x_sub_inf_tiny/epoch_30.pth", map_location="cpu")
merged = {}
for k, v in ego["state_dict"].items():
    merged[f"model_ego_agent.{k}"] = v
for k, v in inf["state_dict"].items():
    new_k = k.replace("model_ego_agent.", "model_other_agent_inf.", 1)
    merged[new_k] = v
torch.save({
    "state_dict": merged,
    "meta": {"prune_config": "${CONFIG_ID}", "src": "ego_pruned + stage2_inf"},
}, "${COOP_MERGED}")
print(f"Saved {len(merged)} keys to ${COOP_MERGED}")
PYEOF

echo ""
echo "=== [3/4] Run coop eval on cooperative val ==="
CUDA_VISIBLE_DEVICES="${GPUS_ARG}" MASTER_PORT="$((28700 + RANDOM % 100))" \
    bash tools/univ2x_dist_eval.sh \
    projects/configs_e2e_univ2x/univ2x_coop_tiny.py \
    "$COOP_MERGED" \
    "$NUM_GPUS"

# Locate latest results_nusc.json
LATEST_DIR=$(ls -td /home/jichengzhi/UniV2X/test/univ2x_coop_tiny/Fri_May* 2>/dev/null | head -1)
if [[ -z "$LATEST_DIR" || ! -f "$LATEST_DIR/results_nusc.json" ]]; then
    echo "ERROR: no results_nusc.json found in $LATEST_DIR"; exit 2
fi

echo ""
echo "=== [4/4] Compute real recall / F1 ==="
PYTHONPATH=$PWD python tools/eval_real_recall.py \
    --pred "$LATEST_DIR/results_nusc.json" \
    --ann-root datasets/V2X-Seq-SPD-New/cooperative \
    --info data/infos/V2X-Seq-SPD-New/cooperative/spd_infos_temporal_val.pkl \
    --label "stage5_${CONFIG_ID}" \
    --out "$METRICS_OUT"

echo ""
echo "=== DONE: $CONFIG_ID ==="
echo "  ego pruned : $EGO_PRUNED"
echo "  coop ckpt  : $COOP_MERGED"
echo "  metrics    : $METRICS_OUT"
