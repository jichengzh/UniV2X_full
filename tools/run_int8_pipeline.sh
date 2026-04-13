#!/bin/bash
# run_int8_pipeline.sh — Full INT8 quantization pipeline + end-to-end evaluation
# Runs all build and evaluation steps in sequence, writing logs to logs/
# Usage: bash tools/run_int8_pipeline.sh 2>&1 | tee logs/pipeline.log

set -e
cd /home/jichengzhi/UniV2X
mkdir -p logs calibration trt_engines output

CONDA_RUN="conda run -n UniV2X_2.0"
CONFIG=projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py
CKPT=ckpts/univ2x_coop_e2e_stg2.pth
PLUGIN=plugins/build/libuniv2x_plugins.so

# ── Helper: print timestamped section header ────────────────────────────────
section() { echo; echo "════════════════════════════════════════════════════════════════"; echo "  $1"; echo "  $(date '+%Y-%m-%d %H:%M:%S')"; echo "════════════════════════════════════════════════════════════════"; }

# ============================================================================
# STEP 1 — A-2: Build ego BEV encoder INT8 (1-cam ONNX, use existing calib PKL)
# ============================================================================
section "A-2: ego BEV encoder INT8 build"
$CONDA_RUN python tools/build_trt_int8_univ2x.py \
    --onnx  onnx/univ2x_ego_bev_encoder_200_1cam.onnx \
    --out   trt_engines/univ2x_ego_bev_encoder_1cam_int8.trt \
    --target bev_encoder \
    --cali-data calibration/bev_encoder_calib_inputs.pkl \
    --plugin $PLUGIN \
    --workspace-gb 8 \
    2>&1 | tee logs/a2_bev_int8_build.log
echo "[A-2 DONE] $(ls -lh trt_engines/univ2x_ego_bev_encoder_1cam_int8.trt)"

# ============================================================================
# STEP 2 — B-1: Dump downstream head calibration data (ego + infra)
# ============================================================================
section "B-1: Dump downstream calibration data (50 frames)"
$CONDA_RUN python tools/dump_downstream_calibration.py \
    $CONFIG $CKPT \
    --n-frames 50 \
    --out-ego  calibration/downstream_ego_calib_inputs.pkl \
    --out-infra calibration/downstream_infra_calib_inputs.pkl \
    2>&1 | tee logs/b1_dump_downstream_calib.log
echo "[B-1 DONE] $(ls -lh calibration/downstream_ego_calib_inputs.pkl calibration/downstream_infra_calib_inputs.pkl)"

# ============================================================================
# STEP 3 — B-2: Smoke test downstream FP16 TRT build (verify build path)
# ============================================================================
section "B-2: Smoke test downstream FP16 build"
$CONDA_RUN python tools/build_trt_int8_univ2x.py \
    --onnx  onnx/univ2x_ego_downstream.onnx \
    --out   trt_engines/univ2x_ego_downstream_fp16_smoke.trt \
    --target downstream \
    --no-int8 \
    --plugin $PLUGIN \
    --workspace-gb 8 \
    2>&1 | tee logs/b2_downstream_fp16_smoke.log
echo "[B-2 DONE] $(ls -lh trt_engines/univ2x_ego_downstream_fp16_smoke.trt)"

# ============================================================================
# STEP 4 — B-3: Build downstream INT8 engines (ego + infra)
# ============================================================================
section "B-3a: Build ego downstream INT8 engine"
$CONDA_RUN python tools/build_trt_int8_univ2x.py \
    --onnx  onnx/univ2x_ego_downstream.onnx \
    --out   trt_engines/univ2x_ego_downstream_int8.trt \
    --target downstream \
    --cali-data calibration/downstream_ego_calib_inputs.pkl \
    --plugin $PLUGIN \
    --workspace-gb 8 \
    2>&1 | tee logs/b3a_ego_downstream_int8_build.log
echo "[B-3a DONE] $(ls -lh trt_engines/univ2x_ego_downstream_int8.trt)"

section "B-3b: Build infra downstream INT8 engine"
$CONDA_RUN python tools/build_trt_int8_univ2x.py \
    --onnx  onnx/univ2x_infra_downstream.onnx \
    --out   trt_engines/univ2x_infra_downstream_int8.trt \
    --target downstream \
    --cali-data calibration/downstream_infra_calib_inputs.pkl \
    --plugin $PLUGIN \
    --workspace-gb 8 \
    2>&1 | tee logs/b3b_infra_downstream_int8_build.log
echo "[B-3b DONE] $(ls -lh trt_engines/univ2x_infra_downstream_int8.trt)"

# ============================================================================
# STEP 5 — B-4a: Cosine validation (INT8 vs FP32 on 5 random samples)
# ============================================================================
section "B-4a: Cosine validation — downstream INT8 vs FP32"
$CONDA_RUN python - <<'PYEOF' 2>&1 | tee logs/b4a_cosine_validation.log
import sys, pickle, ctypes, torch, numpy as np
sys.path.insert(0, '.')
import tensorrt as trt

PLUGIN = 'plugins/build/libuniv2x_plugins.so'
EGO_FP16 = 'trt_engines/univ2x_ego_downstream.trt'
EGO_INT8 = 'trt_engines/univ2x_ego_downstream_int8.trt'
CALIB_PKL = 'calibration/downstream_ego_calib_inputs.pkl'

ctypes.CDLL(PLUGIN)
trt.init_libnvinfer_plugins(None, '')
logger = trt.Logger(trt.Logger.WARNING)

def load_engine(path):
    with open(path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    output_shapes = {}
    output_names = []
    input_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)
            output_shapes[name] = tuple(ctx.get_tensor_shape(name))
    return ctx, input_names, output_names, output_shapes

def run_engine(ctx, input_names, output_names, output_shapes, sample):
    for name in input_names:
        arr = sample[name]
        if isinstance(arr, np.ndarray):
            if arr.dtype == np.bool_:
                t = torch.from_numpy(arr.astype(np.uint8)).cuda()
            else:
                t = torch.from_numpy(arr.astype(np.float32)).cuda()
        else:
            t = torch.tensor(int(arr), dtype=torch.int64, device='cuda')
        ctx.set_tensor_address(name, t.contiguous().data_ptr())
    out = {}
    for name, shape in output_shapes.items():
        out[name] = torch.zeros(*shape, dtype=torch.float32, device='cuda')
        ctx.set_tensor_address(name, out[name].data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return out

print(f'Loading calibration data...')
with open(CALIB_PKL, 'rb') as f:
    calib = pickle.load(f)
print(f'  {len(calib)} samples')

ctx_fp16, in_fp16, out_fp16, shapes_fp16 = load_engine(EGO_FP16)
ctx_int8, in_int8, out_int8, shapes_int8 = load_engine(EGO_INT8)

print(f'\nFP16 outputs: {out_fp16}')
print(f'INT8 outputs: {out_int8}')

N = min(5, len(calib))
print(f'\nComparing {N} samples...')
for i in range(N):
    s = calib[i]
    o_fp16 = run_engine(ctx_fp16, in_fp16, out_fp16, shapes_fp16, s)
    o_int8 = run_engine(ctx_int8, in_int8, out_int8, shapes_int8, s)
    print(f'\n  Sample {i}:')
    for name in out_fp16:
        if name not in o_int8:
            print(f'    {name}: missing in INT8 output')
            continue
        fp16_t = o_fp16[name].cpu().float()
        int8_t = o_int8[name].cpu().float()
        cos = torch.nn.functional.cosine_similarity(
            fp16_t.flatten().unsqueeze(0),
            int8_t.flatten().unsqueeze(0)).item()
        mae = (fp16_t - int8_t).abs().mean().item()
        print(f'    {name}: cos={cos:.7f}  mean_abs_diff={mae:.4e}')

print('\nCosine validation DONE')
PYEOF

# ============================================================================
# STEP 6 — C-1: Dump detection head calibration data (ego, 50 frames)
# ============================================================================
section "C-1: Dump detection head calibration data (50 frames)"
$CONDA_RUN python tools/dump_heads_calibration.py \
    $CONFIG $CKPT \
    --n-frames 50 \
    --out calibration/heads_ego_calib_inputs.pkl \
    2>&1 | tee logs/c1_dump_heads_calib.log
echo "[C-1 DONE] $(ls -lh calibration/heads_ego_calib_inputs.pkl)"

# ============================================================================
# STEP 7 — C-2: Build detection head INT8 engine (V2X 1101-query)
# ============================================================================
section "C-2: Build V2X detection head INT8 engine (1101-query)"
$CONDA_RUN python tools/build_trt_int8_univ2x.py \
    --onnx  onnx/univ2x_ego_heads_v2x_1101.onnx \
    --out   trt_engines/univ2x_ego_heads_v2x_1101_int8.trt \
    --target heads \
    --cali-data calibration/heads_ego_calib_inputs.pkl \
    --plugin $PLUGIN \
    --workspace-gb 8 \
    2>&1 | tee logs/c2_heads_int8_build.log
echo "[C-2 DONE] $(ls -lh trt_engines/univ2x_ego_heads_v2x_1101_int8.trt)"

# ============================================================================
# STEP 8 — Model size comparison
# ============================================================================
section "Model size comparison"
echo "FP32 (ONNX source):"
ls -lh onnx/univ2x_ego_downstream.onnx onnx/univ2x_infra_downstream.onnx onnx/univ2x_ego_heads_v2x_1101.onnx 2>/dev/null || true
echo ""
echo "FP16 TRT engines:"
ls -lh trt_engines/univ2x_ego_downstream.trt trt_engines/univ2x_infra_downstream.trt trt_engines/univ2x_ego_heads_v2x_1101.trt 2>/dev/null || true
echo ""
echo "INT8 TRT engines:"
ls -lh trt_engines/univ2x_ego_downstream_int8.trt trt_engines/univ2x_infra_downstream_int8.trt trt_engines/univ2x_ego_heads_v2x_1101_int8.trt 2>/dev/null || true

# ============================================================================
# STEP 7 — AMOTA evaluation: Config A (FP16 BEV + FP16 downstream heads)
# ============================================================================
section "B-4b Config-A: Hooks A+B+C (FP16 BEV, PyTorch downstream)"
$CONDA_RUN python tools/test_trt.py \
    $CONFIG $CKPT \
    --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_1cam_int8.trt \
    --bev-engine-inf trt_engines/univ2x_infra_bev_encoder_200_1cam.trt \
    --use-lane-trt --use-agent-trt \
    --eval bbox \
    --out output/eval_hookABC_bev_int8.pkl \
    --plugin $PLUGIN \
    2>&1 | tee logs/eval_hookABC_bev_int8.log
echo "[Config-A DONE]"

# ============================================================================
# STEP 8 — AMOTA evaluation: Config B (FP16 downstream TRT via Hook E)
# ============================================================================
section "B-4b Config-B: Hook E with FP16 downstream engine"
$CONDA_RUN python tools/test_trt.py \
    $CONFIG $CKPT \
    --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_1cam_int8.trt \
    --bev-engine-inf trt_engines/univ2x_infra_bev_encoder_200_1cam.trt \
    --use-lane-trt --use-agent-trt \
    --downstream-int8-ego trt_engines/univ2x_ego_downstream.trt \
    --eval bbox \
    --out output/eval_hookE_fp16_downstream.pkl \
    --plugin $PLUGIN \
    2>&1 | tee logs/eval_hookE_fp16_downstream.log
echo "[Config-B DONE]"

# ============================================================================
# STEP 9 — AMOTA evaluation: Config C (INT8 downstream TRT via Hook E)
# ============================================================================
section "B-4b Config-C: Hook E with INT8 downstream engine"
$CONDA_RUN python tools/test_trt.py \
    $CONFIG $CKPT \
    --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_1cam_int8.trt \
    --bev-engine-inf trt_engines/univ2x_infra_bev_encoder_200_1cam.trt \
    --use-lane-trt --use-agent-trt \
    --downstream-int8-ego trt_engines/univ2x_ego_downstream_int8.trt \
    --eval bbox \
    --out output/eval_hookE_int8_downstream.pkl \
    --plugin $PLUGIN \
    2>&1 | tee logs/eval_hookE_int8_downstream.log
echo "[Config-C DONE]"

# ============================================================================
# STEP 12 — AMOTA evaluation: Config D (Hook D with FP16 heads TRT)
# ============================================================================
section "Config-D: Hook A+B+C+D (FP16 BEV + FP16 heads TRT)"
$CONDA_RUN python tools/test_trt.py \
    $CONFIG $CKPT \
    --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_1cam_int8.trt \
    --bev-engine-inf trt_engines/univ2x_infra_bev_encoder_200_1cam.trt \
    --use-lane-trt --use-agent-trt \
    --heads-engine-ego trt_engines/univ2x_ego_heads_v2x_1101.trt \
    --eval bbox \
    --out output/eval_hookD_fp16_heads.pkl \
    --plugin $PLUGIN \
    2>&1 | tee logs/eval_hookD_fp16_heads.log
echo "[Config-D DONE]"

# ============================================================================
# STEP 13 — AMOTA evaluation: Config E (Hook D with INT8 heads TRT)
# ============================================================================
section "Config-E: Hook A+B+C+D (FP16 BEV + INT8 heads TRT)"
$CONDA_RUN python tools/test_trt.py \
    $CONFIG $CKPT \
    --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_1cam_int8.trt \
    --bev-engine-inf trt_engines/univ2x_infra_bev_encoder_200_1cam.trt \
    --use-lane-trt --use-agent-trt \
    --heads-engine-ego trt_engines/univ2x_ego_heads_v2x_1101_int8.trt \
    --eval bbox \
    --out output/eval_hookD_int8_heads.pkl \
    --plugin $PLUGIN \
    2>&1 | tee logs/eval_hookD_int8_heads.log
echo "[Config-E DONE]"

# ============================================================================
# STEP 14 — AMOTA evaluation: Config F (INT8 heads + INT8 downstream)
# ============================================================================
section "Config-F: Hook A+B+C+D+E (INT8 heads + INT8 downstream)"
$CONDA_RUN python tools/test_trt.py \
    $CONFIG $CKPT \
    --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_1cam_int8.trt \
    --bev-engine-inf trt_engines/univ2x_infra_bev_encoder_200_1cam.trt \
    --use-lane-trt --use-agent-trt \
    --heads-engine-ego trt_engines/univ2x_ego_heads_v2x_1101_int8.trt \
    --downstream-int8-ego trt_engines/univ2x_ego_downstream_int8.trt \
    --eval bbox \
    --out output/eval_hookDE_int8_all.pkl \
    --plugin $PLUGIN \
    2>&1 | tee logs/eval_hookDE_int8_all.log
echo "[Config-F DONE]"

# ============================================================================
# FINAL SUMMARY
# ============================================================================
section "FINAL COMPARISON SUMMARY"
echo ""
echo "═══ Model Sizes ══════════════════════════════════════════════"
for f in \
    onnx/univ2x_ego_downstream.onnx \
    trt_engines/univ2x_ego_downstream.trt \
    trt_engines/univ2x_ego_downstream_int8.trt; do
    sz=$(du -sh $f 2>/dev/null | cut -f1)
    echo "  $sz  $f"
done
echo ""
echo "═══ Latency (from eval logs) ════════════════════════════════"
for log in logs/eval_hookABC_bev_int8.log logs/eval_hookE_fp16_downstream.log logs/eval_hookE_int8_downstream.log logs/eval_hookD_fp16_heads.log logs/eval_hookD_int8_heads.log logs/eval_hookDE_int8_all.log; do
    [ -f "$log" ] || continue
    echo "  --- $log ---"
    grep -E "ms/frame|fps|TRT" $log | tail -10 || true
done
echo ""
echo "═══ AMOTA (from eval logs) ══════════════════════════════════"
for log in logs/eval_hookABC_bev_int8.log logs/eval_hookE_fp16_downstream.log logs/eval_hookE_int8_downstream.log logs/eval_hookD_fp16_heads.log logs/eval_hookD_int8_heads.log logs/eval_hookDE_int8_all.log; do
    [ -f "$log" ] || continue
    echo "  --- $log ---"
    grep -E "AMOTA|amota|NDS|mAP" $log | tail -5 || true
done
echo ""
echo "Pipeline complete: $(date)"
