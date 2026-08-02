#!/usr/bin/env bash
# Plan B single-config pipeline on R101+DCN baseline (stg2.pth).
#
# Routing logic (based on what's in prune_configs and quant_configs):
#   - prune-only configs (B/C with no quant): test_with_pruning.py
#   - quant-only configs (D class):           quick_eval_quant.py
#   - joint prune+quant configs (E class):    quick_eval_quant.py with --prune-config
#
# Skips configs that require backbone pruning (A class, E9/E10) — those need
# extension work and run later via plan_b_run_backbone_quant.sh.
#
# Usage:
#   bash tools/plan_b_run_one_config.sh <CONFIG_ID> [GPU_ID]
# Examples:
#   bash tools/plan_b_run_one_config.sh baseline 0
#   bash tools/plan_b_run_one_config.sh B1 0
#   bash tools/plan_b_run_one_config.sh D1 1
#   bash tools/plan_b_run_one_config.sh E1 2

set -euo pipefail

CONFIG_ID="${1:?Usage: $0 <CONFIG_ID> [GPU_ID]}"
GPU_ID="${2:-0}"

# shellcheck source=/dev/null
source /home/jichengzhi/miniconda3/etc/profile.d/conda.sh
conda activate UniV2X_2.0

cd /home/jichengzhi/UniV2X
mkdir -p output/plan_b logs/plan_b data/phase4/stage5_v3

PRUNE_CFG="prune_configs/active_${CONFIG_ID}.json"
QUANT_CFG="quant_configs/active_${CONFIG_ID}.json"
COOP_CFG="projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py"
COOP_CKPT="ckpts/univ2x_coop_e2e_stg2.pth"
OUT_PKL="output/plan_b/${CONFIG_ID}.pkl"
LOG="logs/plan_b/${CONFIG_ID}.log"

# Determine config category (skip backbone-prune configs)
SKIP_LIST=("A1" "A2" "A3" "A4" "A5" "A6" "C4" "E9" "E10")
for skip_id in "${SKIP_LIST[@]}"; do
    if [[ "$CONFIG_ID" == "$skip_id" ]]; then
        echo "SKIP $CONFIG_ID: requires backbone pruning, deferred to backbone-quant variant"
        echo "{\"label\": \"$CONFIG_ID\", \"skipped\": \"backbone_prune_not_supported\"}" \
            > "data/phase4/stage5_v3/${CONFIG_ID}.json"
        exit 0
    fi
done

# Check pruning status
HAS_PRUNE=$(python -c "import json; d=json.load(open('$PRUNE_CFG')); pruned=any([d.get('encoder',{}).get('ffn_mid_ratio',1.0)<1.0, d.get('decoder',{}).get('ffn_mid_ratio',1.0)<1.0, d.get('encoder',{}).get('attn_proj_ratio',0)>0, d.get('decoder',{}).get('attn_proj_ratio',0)>0, d.get('heads',{}).get('head_mid_ratio',1.0)<1.0]); print('1' if pruned else '0')")

# Check quantization status (any module not FP32/none)
HAS_QUANT="0"
if [[ -f "$QUANT_CFG" ]]; then
    HAS_QUANT=$(python -c "import json; d=json.load(open('$QUANT_CFG')); m=d.get('modules',{}); active=any([m.get(k,{}).get('bits',32)<32 and m.get(k,{}).get('target','none')!='none' for k in ['encoder','decoder','heads']]); print('1' if active else '0')")
fi

echo "=== $CONFIG_ID === GPU=$GPU_ID has_prune=$HAS_PRUNE has_quant=$HAS_QUANT"

if [[ "$HAS_PRUNE" == "1" && "$HAS_QUANT" == "0" ]]; then
    # PRUNE-ONLY (B class, C1-C3)
    echo "Mode: PRUNE-ONLY via test_with_pruning.py"
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU_ID" torchrun --nproc_per_node=1 \
        --master_port="$((29500 + GPU_ID * 10))" \
        tools/test_with_pruning.py "$COOP_CFG" "$COOP_CKPT" \
        --prune-config "$PRUNE_CFG" \
        --out "$OUT_PKL" \
        --eval bbox \
        --launcher pytorch > "$LOG" 2>&1

elif [[ "$HAS_PRUNE" == "0" && "$HAS_QUANT" == "1" ]]; then
    # QUANT-ONLY (D class)
    echo "Mode: QUANT-ONLY via quick_eval_quant.py"
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU_ID" python tools/quick_eval_quant.py \
        --config "$COOP_CFG" \
        --checkpoint "$COOP_CKPT" \
        --quant-config "$QUANT_CFG" \
        --eval-samples 168 > "$LOG" 2>&1

elif [[ "$HAS_PRUNE" == "1" && "$HAS_QUANT" == "1" ]]; then
    # JOINT (E class except E9/E10)
    echo "Mode: PRUNE+QUANT via quick_eval_quant.py --prune-config"
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU_ID" python tools/quick_eval_quant.py \
        --config "$COOP_CFG" \
        --checkpoint "$COOP_CKPT" \
        --quant-config "$QUANT_CFG" \
        --prune-config "$PRUNE_CFG" \
        --eval-samples 168 > "$LOG" 2>&1

else
    # BASELINE (no prune, no quant)
    echo "Mode: BASELINE via test_with_pruning.py + active_baseline.json"
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU_ID" torchrun --nproc_per_node=1 \
        --master_port="$((29500 + GPU_ID * 10))" \
        tools/test_with_pruning.py "$COOP_CFG" "$COOP_CKPT" \
        --prune-config "$PRUNE_CFG" \
        --out "$OUT_PKL" \
        --eval bbox \
        --launcher pytorch > "$LOG" 2>&1
fi

EXIT_CODE=$?
echo "$CONFIG_ID exit=$EXIT_CODE"
exit $EXIT_CODE
