#!/usr/bin/env bash
# Plan C single-config pipeline on UniAD-tiny (R50 no DCN) baseline.
#
# Differences from Plan B:
#   - config: univ2x_coop_tiny.py (instead of coop_e2e_track.py)
#   - ckpt:   univ2x_coop_tiny/epoch_30.pth (instead of stg2.pth)
#   - skip:   NO skip list (R50 + no DCN allows all configs incl. backbone prune)
#   - target: F1 (from eval_real_recall) since AMOTA is truncated by min_recall=0.1
#
# Usage: bash tools/plan_c_run_one_config.sh <CONFIG_ID> [GPU_ID]

set -uo pipefail

CONFIG_ID="${1:?Usage: $0 <CONFIG_ID> [GPU_ID]}"
GPU_ID="${2:-0}"

source /home/jichengzhi/miniconda3/etc/profile.d/conda.sh
conda activate UniV2X_2.0
cd /home/jichengzhi/UniV2X
mkdir -p output/plan_c logs/plan_c data/phase4/stage5_v4

PRUNE_CFG="prune_configs/active_${CONFIG_ID}.json"
QUANT_CFG="quant_configs/active_${CONFIG_ID}.json"
TINY_CFG="projects/configs_e2e_univ2x/univ2x_coop_tiny.py"
TINY_CKPT="projects/work_dirs_e2e_univ2x/univ2x_coop_tiny/epoch_30.pth"
OUT_PKL="output/plan_c/${CONFIG_ID}.pkl"
LOG="logs/plan_c/${CONFIG_ID}.log"

if [[ ! -f "$PRUNE_CFG" ]]; then
    echo "ERROR: $PRUNE_CFG not found"; exit 1
fi

# Determine config category
HAS_PRUNE=$(python -c "import json; d=json.load(open('$PRUNE_CFG')); pruned=any([d.get('encoder',{}).get('ffn_mid_ratio',1.0)<1.0, d.get('decoder',{}).get('ffn_mid_ratio',1.0)<1.0, d.get('encoder',{}).get('attn_proj_ratio',0)>0, d.get('decoder',{}).get('attn_proj_ratio',0)>0, d.get('heads',{}).get('head_mid_ratio',1.0)<1.0, d.get('backbone',{}).get('channel_pruning_ratio',0)>0]); print('1' if pruned else '0')")

HAS_QUANT="0"
if [[ -f "$QUANT_CFG" ]]; then
    HAS_QUANT=$(python -c "import json; d=json.load(open('$QUANT_CFG')); m=d.get('modules',{}); active=any([m.get(k,{}).get('bits',32)<32 and m.get(k,{}).get('target','none')!='none' for k in ['encoder','decoder','heads','backbone']]); print('1' if active else '0')")
fi

echo "=== plan_c $CONFIG_ID === GPU=$GPU_ID has_prune=$HAS_PRUNE has_quant=$HAS_QUANT"

if [[ "$HAS_PRUNE" == "1" && "$HAS_QUANT" == "0" ]]; then
    echo "Mode: PRUNE-ONLY via test_with_pruning.py"
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU_ID" torchrun --nproc_per_node=1 \
        --master_port="$((29700 + GPU_ID * 10))" \
        tools/test_with_pruning.py "$TINY_CFG" "$TINY_CKPT" \
        --prune-config "$PRUNE_CFG" \
        --out "$OUT_PKL" \
        --eval bbox \
        --launcher pytorch > "$LOG" 2>&1

elif [[ "$HAS_PRUNE" == "0" && "$HAS_QUANT" == "1" ]]; then
    echo "Mode: QUANT-ONLY via quick_eval_quant.py"
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU_ID" python tools/quick_eval_quant.py \
        --config "$TINY_CFG" \
        --checkpoint "$TINY_CKPT" \
        --quant-config "$QUANT_CFG" \
        --eval-samples 168 > "$LOG" 2>&1

elif [[ "$HAS_PRUNE" == "1" && "$HAS_QUANT" == "1" ]]; then
    echo "Mode: PRUNE+QUANT via quick_eval_quant.py --prune-config"
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU_ID" python tools/quick_eval_quant.py \
        --config "$TINY_CFG" \
        --checkpoint "$TINY_CKPT" \
        --quant-config "$QUANT_CFG" \
        --prune-config "$PRUNE_CFG" \
        --eval-samples 168 > "$LOG" 2>&1

else
    echo "Mode: BASELINE (no prune, no quant)"
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU_ID" torchrun --nproc_per_node=1 \
        --master_port="$((29700 + GPU_ID * 10))" \
        tools/test_with_pruning.py "$TINY_CFG" "$TINY_CKPT" \
        --prune-config "$PRUNE_CFG" \
        --out "$OUT_PKL" \
        --eval bbox \
        --launcher pytorch > "$LOG" 2>&1
fi

EXIT_CODE=$?
echo "$CONFIG_ID exit=$EXIT_CODE"
exit $EXIT_CODE
