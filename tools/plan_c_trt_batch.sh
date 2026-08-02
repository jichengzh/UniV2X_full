#!/usr/bin/env bash
# Plan C TRT pipeline: ONNX export + TRT FP16 build + benchmark for tiny configs.
# For each config: produces onnx + .trt + benchmark JSON.
#
# Usage: bash tools/plan_c_trt_batch.sh "<CONFIG_LIST>" "<GPU_LIST>"

set -uo pipefail

CONFIGS="${1:?Usage: $0 \"<CONFIG_IDS>\" \"<GPU_IDS>\"}"
GPUS="${2:?Need GPU list}"

cd /home/jichengzhi/UniV2X
mkdir -p logs/plan_c/trt onnx trt_engines output/plan_c/trt
source /home/jichengzhi/miniconda3/etc/profile.d/conda.sh
conda activate UniV2X_2.0

IFS=',' read -ra GPU_ARR <<< "$GPUS"
read -ra CFG_ARR <<< "$CONFIGS"

echo "=== Plan C TRT batch: ${#CFG_ARR[@]} configs on ${#GPU_ARR[@]} GPUs ==="

run_trt_pipeline() {
    local cfg=$1
    local gpu=$2
    local prune_cfg="prune_configs/active_${cfg}.json"
    case "$cfg" in
        P1_20) prune_cfg="prune_configs/p1_ffn_20pct.json" ;;
        P1_30) prune_cfg="prune_configs/p1_ffn_30pct.json" ;;
        P1_40) prune_cfg="prune_configs/p1_ffn_40pct.json" ;;
        P1_50) prune_cfg="prune_configs/p1_ffn_50pct.json" ;;
        P1_60) prune_cfg="prune_configs/p1_ffn_60pct.json" ;;
    esac
    local onnx_path="onnx/univ2x_tiny_${cfg}_bev_200.onnx"
    local trt_path="trt_engines/univ2x_tiny_${cfg}_bev_fp16.trt"
    local bench_json="output/plan_c/trt/${cfg}_bench.json"
    local log="logs/plan_c/trt/${cfg}.log"

    {
        echo "=== $cfg on GPU $gpu ==="
        if [[ ! -f "$onnx_path" ]]; then
            PYTHONPATH=. CUDA_VISIBLE_DEVICES=$gpu python tools/export_onnx_univ2x.py \
                projects/configs_e2e_univ2x/univ2x_coop_tiny_trt.py \
                projects/work_dirs_e2e_univ2x/univ2x_coop_tiny/epoch_30.pth \
                --model ego --backbone-only --bev-size 200 --num-cam 1 \
                --prune-config "$prune_cfg" \
                --out "$onnx_path"
        else
            echo "[reuse] $onnx_path"
        fi
        if [[ ! -f "$trt_path" ]]; then
            PYTHONPATH=. CUDA_VISIBLE_DEVICES=$gpu python tools/build_trt_fp_univ2x.py \
                --onnx "$onnx_path" --out "$trt_path" \
                --plugin plugins/build/libuniv2x_plugins.so \
                --precision fp16
        else
            echo "[reuse] $trt_path"
        fi
        if [[ -f "$trt_path" ]]; then
            PYTHONPATH=. CUDA_VISIBLE_DEVICES=$gpu python tools/benchmark_trt_engine.py \
                --engine "$trt_path" --plugin plugins/build/libuniv2x_plugins.so \
                --n-warmup 10 --n-runs 30 > /tmp/${cfg}_bench.txt 2>&1
            mean=$(grep "^mean" /tmp/${cfg}_bench.txt | awk '{print $3}')
            echo "{\"config_id\": \"$cfg\", \"trt_bev_ms_mean\": $mean}" > "$bench_json"
            echo "[$cfg] DONE: ${mean} ms"
        else
            echo "[$cfg] FAILED to build TRT engine"
        fi
    } > "$log" 2>&1
}

declare -A GPU_BUSY
for g in "${GPU_ARR[@]}"; do GPU_BUSY[$g]=""; done

cfg_idx=0
while (( cfg_idx < ${#CFG_ARR[@]} )); do
    free_gpu=""
    for g in "${GPU_ARR[@]}"; do
        pid="${GPU_BUSY[$g]}"
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
            free_gpu="$g"
            break
        fi
    done
    if [[ -z "$free_gpu" ]]; then sleep 5; continue; fi
    cfg="${CFG_ARR[$cfg_idx]}"
    echo "[dispatch-trt] $cfg -> GPU $free_gpu"
    run_trt_pipeline "$cfg" "$free_gpu" &
    GPU_BUSY[$free_gpu]=$!
    cfg_idx=$((cfg_idx + 1))
done

wait
echo "=== Plan C TRT batch DONE ==="
ls -1 output/plan_c/trt/*.json 2>/dev/null | wc -l
