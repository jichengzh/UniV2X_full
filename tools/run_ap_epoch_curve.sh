#!/bin/bash
# AP-vs-epoch curve experiment for Pyramid p50 pruning
# Run on H800 with GPU5, using HEAL inference.py
#
# Usage: bash tools/run_ap_epoch_curve.sh
#
# This script:
# 1. Evals zero-shot pruned ckpt (epoch0)
# 2. Evals finetune epochs 1,2,4,8 from Pyramid_DAIR_m1_pruned50_2026_05_10
# 3. Optionally extends training for epochs 16,31
# 4. Saves results to results/ap_epoch_curve_p50.json

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
V2X_ROOT="${V2X_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CUDA_DEV="${CUDA_DEV:-5}"
HEAL_ROOT="${HEAL_ROOT:?set HEAL_ROOT to the HEAL checkout}"
PRUNED50_DIR="${PRUNED50_DIR:?set PRUNED50_DIR to the pruned checkpoint directory}"
RESULTS_DIR="${V2X_RESULTS_DIR:-${V2X_ROOT}/results}"
ZEROSHOT_CKPT="${ZEROSHOT_CKPT:-${RESULTS_DIR}/zeroshot_p50_exp/net_epoch_bestval_at23.pth}"
CURVE_JSON="${RESULTS_DIR}/ap_epoch_curve_p50.json"
WORK_DIR="/tmp/ap_eval_p50_$$"

PYTHONPATH_HEAL="${PYTHONPATH_HEAL:-${HEAL_ROOT}}"
PY="${V2X_PYTHON:-python3}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEV}"

echo "=== AP-vs-epoch curve experiment ==="
echo "  GPU: ${CUDA_DEV}"
echo "  pruned50_dir: ${PRUNED50_DIR}"
echo "  zeroshot_ckpt: ${ZEROSHOT_CKPT}"
echo "  results: ${CURVE_JSON}"
echo

# ── GPU preflight ────────────────────────────────────────────────────────────
nvidia-smi | head -30
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i ${CUDA_DEV} | tr -d ' ')
echo "GPU${CUDA_DEV} status: ${GPU_UTIL}"

# ── Eval function ─────────────────────────────────────────────────────────────
eval_ckpt() {
    local label="$1"
    local ckpt_src="$2"
    local tmpdir="${WORK_DIR}/${label}"
    mkdir -p "${tmpdir}"

    # Copy config with DAIR paths (pruned50 config has absolute paths)
    cp "${PRUNED50_DIR}/config.yaml" "${tmpdir}/config.yaml"

    # Symlink ckpt as bestval
    # HEAL inference.py uses load_saved_model which finds net_epoch_bestval_atN.pth
    # Use epoch N=999 to ensure it's picked as the highest
    ln -sf "${ckpt_src}" "${tmpdir}/net_epoch_bestval_at999.pth"

    echo "--- Evaling: ${label} ---"
    echo "  ckpt: ${ckpt_src}"

    # Run HEAL inference.py
    local ap_log="${RESULTS_DIR}/ap_eval_${label}.log"
    PYTHONPATH="${PYTHONPATH_HEAL}" \
    ${PY} "${HEAL_ROOT}/opencood/tools/inference.py" \
        --model_dir "${tmpdir}" \
        --fusion_method intermediate \
        --range "102.4,102.4" \
        2>&1 | tee "${ap_log}"

    # Extract AP values from log
    AP50=$(grep -oP '(?<=ap50: )\d+\.\d+' "${ap_log}" | tail -1)
    AP70=$(grep -oP '(?<=ap70: )\d+\.\d+' "${ap_log}" | tail -1)
    AP30=$(grep -oP '(?<=ap30: )\d+\.\d+' "${ap_log}" | tail -1)

    if [ -z "$AP50" ]; then
        # Try alternative grep patterns
        AP50=$(grep -i 'ap50\|AP@0.5' "${ap_log}" | grep -oP '\d+\.\d+' | tail -1)
        AP70=$(grep -i 'ap70\|AP@0.7' "${ap_log}" | grep -oP '\d+\.\d+' | tail -1)
        AP30=$(grep -i 'ap30\|AP@0.3' "${ap_log}" | grep -oP '\d+\.\d+' | tail -1)
    fi

    echo "  RESULT: label=${label} AP30=${AP30} AP50=${AP50} AP70=${AP70}"

    # Append to curve JSON (simple approach)
    echo "{\"label\": \"${label}\", \"ckpt_path\": \"${ckpt_src}\", \"ap30\": ${AP30:-null}, \"ap50\": ${AP50:-null}, \"ap70\": ${AP70:-null}}" \
        >> "${RESULTS_DIR}/ap_epoch_curve_raw_p50.jsonl"

    rm -rf "${tmpdir}"
}

# ── Init results ──────────────────────────────────────────────────────────────
mkdir -p "${WORK_DIR}" "${RESULTS_DIR}"
> "${RESULTS_DIR}/ap_epoch_curve_raw_p50.jsonl"  # clear

# ── Step 1: Zero-shot eval ────────────────────────────────────────────────────
echo "============================================"
echo "STEP 1: Zero-shot (pruned but no finetune)"
echo "============================================"
if [ -f "${ZEROSHOT_CKPT}" ]; then
    eval_ckpt "zeroshot_epoch0" "${ZEROSHOT_CKPT}"
else
    echo "[SKIP] Zero-shot ckpt not found at ${ZEROSHOT_CKPT}"
    echo "  → Run structural pruning first to create it"
fi

# ── Step 2: Finetune epoch 1 (epoch24.pth) ───────────────────────────────────
echo "============================================"
echo "STEP 2: Finetune epochs 1,2,4,8"
echo "============================================"
for ft_ep in 1 2 4 8; do
    abs_ep=$((23 + ft_ep))
    ckpt="${PRUNED50_DIR}/net_epoch${abs_ep}.pth"
    if [ -f "${ckpt}" ]; then
        eval_ckpt "finetune_ep${ft_ep}" "${ckpt}"
    else
        echo "[SKIP] ${ckpt} not found"
    fi
done

# ── Step 3: Converged bestval (epoch29) ──────────────────────────────────────
echo "============================================"
echo "STEP 3: Converged bestval (ep29)"
echo "============================================"
bestval="${PRUNED50_DIR}/net_epoch_bestval_at29.pth"
if [ -f "${bestval}" ]; then
    eval_ckpt "converged_bestval_ep29" "${bestval}"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "=== Curve results (raw) ==="
cat "${RESULTS_DIR}/ap_epoch_curve_raw_p50.jsonl"

echo
echo "Done. Check ${RESULTS_DIR}/ap_epoch_curve_raw_p50.jsonl for results"
echo "Full eval logs: ${RESULTS_DIR}/ap_eval_*.log"

# Cleanup temp
rm -rf "${WORK_DIR}"
