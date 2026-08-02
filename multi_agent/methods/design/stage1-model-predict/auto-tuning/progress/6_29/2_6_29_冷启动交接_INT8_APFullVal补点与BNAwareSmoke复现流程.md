# 58_6_29_冷启动交接_INT8_APFullVal补点与BNAwareSmoke复现流程

## 0. 本文用途

本文是 INT8 AP full-val 补点的单独冷启动文档。目标是让新窗口直接复用 2026-06-28 已跑通的 BN-aware + int32 Conv bias smoke 流程, 不再重复做已经完成的 5-sample smoke 探索。

下一阶段的具体目标:

```text
1. 不重复 5-sample smoke; 先审阅已有 smoke artifacts。
2. 复用已有 route_dir / tensor_quant_params / TVM graph / runtime_weights。
3. 对 5 个 original60 label 跑 1789-frame INT8 full-val AP。
4. 通过 gate 后导入 rows/native_int8_original60_ap_rows_v1.jsonl measured rows。
5. 刷新 original60_quant_three_metric_summary_latest.* 和 completion review。
```

5 个目标 label:

| label | width csv | existing smoke gpu | full-val 备注 |
|---|---|---:|---|
| s0_024 | `24,128,256` | 7 | 当前已有旧 route full-val row, 但 BN-aware smoke route 仍应重跑 full-val 覆盖旧低 AP 证据 |
| s0_040 | `40,128,256` | 5 | 这是此前 collapse 根因主点 |
| s0_056 | `56,128,256` | 7 | smoke 已通过 |
| s1_048 | `64,48,256` | 7 | smoke 已通过 |
| s2_160 | `64,128,160` | 7 | smoke 已通过 |

## 1. 启动初期必读材料

新窗口启动后先读本文, 再读:

```text
当前 INT8 smoke review:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_bnaware_bias_ap_smoke_5labels_review_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_bnaware_bias_ap_smoke_5labels_review_latest.json

INT8 修复主线:
multi_agent/methods/design/auto-tuning/progress/6_27/55_6_28_冷启动交接_NativeINT8精度崩溃修复与ScaleAwareGate.md

三精度数据审查:
multi_agent/methods/design/auto-tuning/progress/6_27/57_6_29_交接文档_阶段三_FP32Energy异常复盘与冷启动防错计划.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_measurement_source_audit_latest.md

关键脚本:
scripts/stage2_h800_native_int8_real_activation_bridge.py
scripts/stage2_import_native_int8_full_ap_row.py
scripts/stage2_h800_native_int8_op_alignment.py
scripts/stage2_native_int8_tvm_worker.py
framework/stage2/native_int8_full_onnx.py
framework/tests/test_stage2_native_int8_route.py
```

固定规则:

```text
1. 单 agent 执行, 不启动 agent team。
2. 不明文写 H800 密码。只使用 export '<REDACTED_LEGACY_SECRET>')"。
3. 启动 H800 任务前重新审计 nvidia-smi / compute-apps, 不 kill 未知任务。
4. 5-sample smoke 不是 measured AP row; full-val row 必须 processed_samples >= 1789。
5. 所有 latency/energy 仍是 TVM backbone/subnet module 口径; AP bridge 是 TVM INT8 backbone/subnet + PyTorch head/postprocess, full_network_claim=false。
6. 当前 BN-aware route 仍有 float32-heavy TIR 限制; full-val AP 只能证明 AP 可恢复, 不能单独证明 fully integer INT8 speedup 完成。
7. 失败时保存 raw artifact、command、stdout/stderr、GPU id、failure reason, 不因单点失败停止其他 label。
```

## 2. 已完成 smoke 的权威结论

当前 review:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_bnaware_bias_ap_smoke_5labels_review_latest.md
```

结论:

```text
Root cause:
- reference range capture 曾把 ONNX Conv outputs 错配到 PyTorch BN 前 raw Conv outputs。
- TVM route 在 Conv+BN fusion 后丢了 ONNX Conv bias。
- BN-aware reference range + int32 Conv bias accumulation 后, backbone numeric alignment 恢复。

限制:
- 当前 scale-aware route 仍 float32-heavy in TIR。
- smoke 证明 AP recoverability, 不证明最终 native INT8 acceleration speedup。
```

5-sample smoke 结果:

| label | samples | nonempty | AP30 | AP50 | AP70 | numeric corr L0/L1/L2 | latency ms | energy J |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| s0_024 | 5 | 5 | 0.836992 | 0.827401 | 0.751548 | 0.995/0.977/0.973 | 12.285930 | 2.854578 |
| s0_040 | 5 | 5 | 0.826001 | 0.826001 | 0.708244 | 0.992/0.970/0.964 | 16.578358 | 2.838798 |
| s0_056 | 5 | 5 | 0.876831 | 0.867277 | 0.766703 | 0.986/0.981/0.983 | 14.632021 | 3.448992 |
| s1_048 | 5 | 5 | 0.793045 | 0.793045 | 0.711346 | 0.984/0.936/0.968 | 11.162400 | 2.647444 |
| s2_160 | 5 | 5 | 0.859451 | 0.849133 | 0.757073 | 0.989/0.987/0.986 | 12.059082 | 2.869185 |

这些结果已经足够作为 full-val 启动 gate。新窗口不要重新做同一批 5-sample smoke, 除非对应 route/report 文件缺失。

## 3. 远端已有 artifact 结构

远端 H800 路径:

```text
V2X=${V2X_ROOT}
PY=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python
TVMPY=${V2X_DATA_ROOT}/tvm310/bin/python
HEAL=${V2X_HOME}/heal_research/HEAL
RAW_PARENT=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route
ROWS=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl
```

pipeline 控制目录:

```text
$RAW_PARENT/20260628_5label_bnaware_bias_ap_smoke_pipeline_gpu7
```

该目录包含:

```text
pipeline.sh
events.tsv
pipeline_stdout.txt
pipeline_stderr.txt
pipeline.pid
```

`events.tsv` 已显示:

```text
s0_024: export_skip -> bootstrap_route -> range_capture -> calibrated_route -> numeric_sanity -> ap_smoke -> LABEL_DONE
s0_056: export -> bootstrap_route -> range_capture -> calibrated_route -> numeric_sanity -> ap_smoke -> LABEL_DONE
s1_048: export -> bootstrap_route -> range_capture -> calibrated_route -> numeric_sanity -> ap_smoke -> LABEL_DONE
s2_160: export -> bootstrap_route -> range_capture -> calibrated_route -> numeric_sanity -> ap_smoke -> LABEL_DONE
PIPELINE_DONE
```

s0_040 是此前单独主线完成, 不在该 pipeline loop 内, 但 artifact 同样完整。

## 4. 每个 label 应复用的路径

### s0_024

```text
CKPT=${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s0_024_2026_06_26
ROUTE=$RAW_PARENT/20260628_s0_024_scaleaware_bnaware_bias_v1/s0_024
PARAMS=$RAW_PARENT/20260628_s0_024_bnaware_tensor_range_capture_v1_gpu7/tensor_quant_params_calibration_bnaware_v1_to_pyramid_level2.json
ART=$ROUTE/s0_024_native_int8_full_onnx_native_int8_full_onnx_tvm_graph.so
INV=$ROUTE/tvm_operator_inventory.json
WEIGHTS=$ROUTE/runtime_weights_int8.npz
SMOKE=$ROUTE/ap_smoke_5samples_bnaware_bias_v1_gpu7/full_ap_eval_report.json
```

### s0_040

```text
CKPT=${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s0_040_2026_06_26
ROUTE=$RAW_PARENT/20260628_s0_040_scaleaware_bnaware_bias_v1/s0_040
PARAMS=$RAW_PARENT/20260628_s0_040_bnaware_tensor_range_capture_v1_gpu5/tensor_quant_params_calibration_bnaware_v1_to_pyramid_level2.json
ART=$ROUTE/s0_040_native_int8_full_onnx_native_int8_full_onnx_tvm_graph.so
INV=$ROUTE/tvm_operator_inventory.json
WEIGHTS=$ROUTE/runtime_weights_int8.npz
SMOKE=$ROUTE/ap_smoke_5samples_bnaware_bias_v1_gpu5/full_ap_eval_report.json
```

### s0_056

```text
CKPT=${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s0_056_2026_06_26
ROUTE=$RAW_PARENT/20260628_s0_056_scaleaware_bnaware_bias_v1/s0_056
PARAMS=$RAW_PARENT/20260628_s0_056_bnaware_tensor_range_capture_v1_gpu7/tensor_quant_params_calibration_bnaware_v1_to_pyramid_level2.json
ART=$ROUTE/s0_056_native_int8_full_onnx_native_int8_full_onnx_tvm_graph.so
INV=$ROUTE/tvm_operator_inventory.json
WEIGHTS=$ROUTE/runtime_weights_int8.npz
SMOKE=$ROUTE/ap_smoke_5samples_bnaware_bias_v1_gpu7/full_ap_eval_report.json
```

### s1_048

```text
CKPT=${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s1_048_2026_06_26
ROUTE=$RAW_PARENT/20260628_s1_048_scaleaware_bnaware_bias_v1/s1_048
PARAMS=$RAW_PARENT/20260628_s1_048_bnaware_tensor_range_capture_v1_gpu7/tensor_quant_params_calibration_bnaware_v1_to_pyramid_level2.json
ART=$ROUTE/s1_048_native_int8_full_onnx_native_int8_full_onnx_tvm_graph.so
INV=$ROUTE/tvm_operator_inventory.json
WEIGHTS=$ROUTE/runtime_weights_int8.npz
SMOKE=$ROUTE/ap_smoke_5samples_bnaware_bias_v1_gpu7/full_ap_eval_report.json
```

### s2_160

```text
CKPT=${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s2_160_2026_06_26
ROUTE=$RAW_PARENT/20260628_s2_160_scaleaware_bnaware_bias_v1/s2_160
PARAMS=$RAW_PARENT/20260628_s2_160_bnaware_tensor_range_capture_v1_gpu7/tensor_quant_params_calibration_bnaware_v1_to_pyramid_level2.json
ART=$ROUTE/s2_160_native_int8_full_onnx_native_int8_full_onnx_tvm_graph.so
INV=$ROUTE/tvm_operator_inventory.json
WEIGHTS=$ROUTE/runtime_weights_int8.npz
SMOKE=$ROUTE/ap_smoke_5samples_bnaware_bias_v1_gpu7/full_ap_eval_report.json
```

## 5. 已跑通 smoke 的完整实现流程

这部分只用于理解和模仿。不要默认重跑。

### Step 1: checkpoint-consistent multiscale ONNX export

用途:

```text
从 exact-label FP16/AP checkpoint 导出 checkpoint-consistent backbone ONNX。
```

模板:

```bash
cd ${V2X_ROOT}
PYTHONPATH=${V2X_ROOT}:${V2X_HOME}/heal_research/HEAL \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
scripts/stage2_h800_export_checkpoint_multiscale_onnx.py \
  --label ${LABEL} \
  --ckpt-dir ${CKPT} \
  --out ${RAW_PARENT}/checkpoint_consistent_${LABEL}_multiscale_export_v1/${LABEL}_backbone.onnx \
  --report-json ${RAW_PARENT}/checkpoint_consistent_${LABEL}_multiscale_export_v1/export_report.json \
  --gpu-id ${GPU}
```

s0_024 已经 `export_skip`; s0_040 已有:

```text
$RAW_PARENT/checkpoint_consistent_s0_040_multiscale_export_v1/s0_040_backbone.onnx
```

### Step 2: bootstrap route for op records

用途:

```text
先用普通/旧 route 生成 TVM route_dir 和 op records, 供 BN-aware reference range capture 对齐 ONNX tensor 名。
```

模板:

```bash
cd ${V2X_ROOT}
LD_LIBRARY_PATH=${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/tvm/lib \
STAGE2_NATIVE_INT8_MODEL_ROOT=${MODEL_ROOT} \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
${V2X_DATA_ROOT}/tvm310/bin/python \
${RAW_PARENT}/stage2_h800_native_int8_full_onnx_route.py \
  --run-id 20260628_${LABEL}_bootstrap_bias_oprecords_v1 \
  --gpu ${GPU} \
  --labels ${LABEL} \
  --completion-queue ${QUEUE} \
  --allow-non-h800-debug
```

### Step 3: BN-aware reference range capture

用途:

```text
捕获 PyTorch BN 后输出范围, 修复旧流程把 ONNX Conv output 对齐到 BN 前 raw Conv output 的错误。
```

模板:

```bash
cd ${V2X_ROOT}
PYTHONPATH=${V2X_ROOT}:${V2X_HOME}/heal_research/HEAL \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
scripts/stage2_h800_native_int8_op_alignment.py \
  --label ${LABEL} \
  --ckpt-dir ${CKPT} \
  --route-dir ${BOOT_ROUTE} \
  --raw-dir ${CAPTURE_DIR} \
  --gpu-id ${GPU} \
  --collect-reference-ranges \
  --execute-reference-range-capture \
  --reference-range-stop-output pyramid_level2 \
  --reference-range-out tensor_reference_range_targets_to_pyramid_level2_bnaware_v1.json \
  --module-inventory-out pytorch_module_inventory_to_pyramid_level2_bnaware_v1.json \
  --reference-range-plan-out tensor_reference_range_capture_plan_to_pyramid_level2_bnaware_v1.json \
  --reference-ranges-out tensor_reference_ranges_to_pyramid_level2_bnaware_v1.json \
  --reference-range-calibration-out tensor_quant_params_calibration_bnaware_v1_to_pyramid_level2.json
```

关键产物:

```text
${CAPTURE_DIR}/tensor_quant_params_calibration_bnaware_v1_to_pyramid_level2.json
```

### Step 4: calibrated scale-aware route with int32 Conv bias

用途:

```text
用 BN-aware params 重新 build route, 并把 ONNX Conv bias 放入 int32 accumulator 输入。
```

模板:

```bash
cd ${V2X_ROOT}
LD_LIBRARY_PATH=${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/tvm/lib \
STAGE2_NATIVE_INT8_MODEL_ROOT=${MODEL_ROOT} \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
${V2X_DATA_ROOT}/tvm310/bin/python \
${RAW_PARENT}/stage2_h800_native_int8_full_onnx_route.py \
  --run-id 20260628_${LABEL}_scaleaware_bnaware_bias_v1 \
  --gpu ${GPU} \
  --labels ${LABEL} \
  --completion-queue ${QUEUE} \
  --tensor-quant-params-path ${PARAMS} \
  --allow-non-h800-debug
```

通过条件:

```text
native_int8_route_manifest.json:
- status = success
- route_spec = full_onnx_topology_scale_aware_tensor_quant_params_v2
- full_network_claim = false

full_onnx_route_attempt.json:
- status = success
- artifact_path 指向 ${LABEL}_native_int8_full_onnx_native_int8_full_onnx_tvm_graph.so
```

### Step 5: numeric sanity

用途:

```text
只跑 1 sample, 检查 TVM INT8 backbone output 与 PyTorch reference multiscale output 对齐。
```

模板:

```bash
cd ${V2X_ROOT}
PYTHONPATH=${V2X_ROOT}:${V2X_HOME}/heal_research/HEAL \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
scripts/stage2_h800_native_int8_real_activation_bridge.py \
  --label ${LABEL} \
  --ckpt-dir ${CKPT} \
  --raw-dir ${ROUTE}/numeric_sanity_bnaware_bias_v1_gpu${GPU} \
  --route-dir ${ROUTE} \
  --artifact-path ${ART} \
  --inventory-path ${INV} \
  --runtime-weight-archive-path ${WEIGHTS} \
  --gpu-id ${GPU} \
  --num-samples 1 \
  --full-ap-min-samples 1 \
  --ap-row-min-samples 1789 \
  --keep-detailed-samples 1 \
  --numeric-sanity-only \
  --tensor-quant-params-path ${PARAMS}
```

通过条件:

```text
real_activation_bridge_report.json:
- status = passed

output_dequant_calibration_summary.json:
- status = passed
- processed_samples = 1
- full_network_claim = false
```

### Step 6: 5-sample AP smoke

用途:

```text
验证非空预测、AP30/AP50/AP70 非零趋势、head/postprocess 分布可解释。
```

模板:

```bash
cd ${V2X_ROOT}
PYTHONPATH=${V2X_ROOT}:${V2X_HOME}/heal_research/HEAL \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
scripts/stage2_h800_native_int8_real_activation_bridge.py \
  --label ${LABEL} \
  --ckpt-dir ${CKPT} \
  --raw-dir ${ROUTE}/ap_smoke_5samples_bnaware_bias_v1_gpu${GPU} \
  --route-dir ${ROUTE} \
  --artifact-path ${ART} \
  --inventory-path ${INV} \
  --runtime-weight-archive-path ${WEIGHTS} \
  --gpu-id ${GPU} \
  --num-samples 5 \
  --full-ap-min-samples 5 \
  --ap-row-min-samples 1789 \
  --keep-detailed-samples 5 \
  --tensor-quant-params-path ${PARAMS}
```

通过条件:

```text
full_ap_eval_report.json:
- status = success
- processed_samples = 5
- pred_nonempty_count = 5
- AP30/AP50/AP70 finite
- ap_row_allowed = false
- ap_row_block_reason = full_eval_num_samples_5_lt_1789
- full_network_claim = false
```

注意:

```text
ap_row_allowed=false 是正确结果, 因为 smoke 只有 5 samples。
不能把该 smoke row 导入 rows/native_int8_original60_ap_rows_v1.jsonl。
```

## 6. full-val 补点执行方案

### Step 0: GPU 与现有 artifact 审计

远端执行:

```bash
export '<REDACTED_LEGACY_SECRET>')"
ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>
```

进入远端后:

```bash
cd ${V2X_ROOT}
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

确认 artifact:

```bash
RAW_PARENT=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route
for label in s0_024 s0_040 s0_056 s1_048 s2_160; do
  ROUTE=$RAW_PARENT/20260628_${label}_scaleaware_bnaware_bias_v1/${label}
  test -f "$ROUTE/native_int8_route_manifest.json" || echo "missing manifest: $label"
  test -f "$ROUTE/${label}_native_int8_full_onnx_native_int8_full_onnx_tvm_graph.so" || echo "missing graph so: $label"
  test -f "$ROUTE/runtime_weights_int8.npz" || echo "missing weights: $label"
done
```

### Step 1: 运行一个 label 的 full-val

以 s0_040 为模板:

```bash
set -euo pipefail
V2X=${V2X_ROOT}
PY=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python
HEAL=${V2X_HOME}/heal_research/HEAL
RAW_PARENT=$V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route

LABEL=s0_040
GPU=5
CKPT=${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_${LABEL}_2026_06_26
ROUTE=$RAW_PARENT/20260628_${LABEL}_scaleaware_bnaware_bias_v1/${LABEL}
PARAMS=$RAW_PARENT/20260628_${LABEL}_bnaware_tensor_range_capture_v1_gpu${GPU}/tensor_quant_params_calibration_bnaware_v1_to_pyramid_level2.json
ART=$ROUTE/${LABEL}_native_int8_full_onnx_native_int8_full_onnx_tvm_graph.so
INV=$ROUTE/tvm_operator_inventory.json
WEIGHTS=$ROUTE/runtime_weights_int8.npz
FULL_DIR=$ROUTE/ap_fullval_1789_bnaware_bias_v1_gpu${GPU}_20260629
mkdir -p "$FULL_DIR"

CMD="cd $V2X && PYTHONPATH=$V2X:$HEAL CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY scripts/stage2_h800_native_int8_real_activation_bridge.py --label $LABEL --ckpt-dir $CKPT --raw-dir $FULL_DIR --route-dir $ROUTE --artifact-path $ART --inventory-path $INV --runtime-weight-archive-path $WEIGHTS --gpu-id $GPU --num-samples 1789 --full-ap-min-samples 1789 --ap-row-min-samples 1789 --keep-detailed-samples 20 --tensor-quant-params-path $PARAMS"
printf '%s\n' "$CMD" > "$FULL_DIR/command.txt"
printf '%s\n' "$(date -Is)" > "$FULL_DIR/started_at.txt"
nohup bash -lc "$CMD" > "$FULL_DIR/stdout.txt" 2> "$FULL_DIR/stderr.txt" &
echo $! > "$FULL_DIR/runner.pid"
```

其他 label 只替换 `LABEL/GPU/WIDTH_CSV`。注意 s0_040 的 params 在 `gpu5`, 其他四个 smoke params 在 `gpu7`。full-val 运行时 GPU 可以换成空闲 H800 GPU, 但如果替换 GPU, `PARAMS` 仍指向既有 capture 目录, 不要因为运行 GPU 改了就误改 params 路径。

### Step 2: full-val 完成 gate

```bash
FULL_DIR=/path/to/ap_fullval_1789_bnaware_bias_v1_gpuX_20260629
/usr/bin/python3 - <<PY
import json
from pathlib import Path
p=Path("$FULL_DIR/full_ap_eval_report.json")
assert p.exists(), p
r=json.loads(p.read_text())
print({k:r.get(k) for k in ["status","processed_samples","pred_nonempty_count","pred_total_count","ap30","ap50","ap70","ap_row_allowed","ap_row_block_reason","full_network_claim"]})
assert r.get("status") == "success"
assert int(r.get("processed_samples") or 0) >= 1789
assert int(r.get("pred_nonempty_count") or 0) > 0
assert r.get("ap30") is not None and r.get("ap50") is not None and r.get("ap70") is not None
assert r.get("full_network_claim") is False
assert r.get("ap_row_allowed") is True
PY
```

如果 `ap_row_allowed=false`, 不导入 row。先看:

```text
full_ap_eval_report.json
native_int8_s0_024_full_ap_blocker.json
output_dequant_summary.json
postprocess_summary.json
worker_response_summary.json
stderr.txt
stdout.txt
```

### Step 3: 导入 measured AP row

以 s0_040 为模板:

```bash
cd ${V2X_ROOT}
LABEL=s0_040
WIDTH_CSV=40,128,256
GPU=5
RAW_PARENT=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route
ROUTE=$RAW_PARENT/20260628_${LABEL}_scaleaware_bnaware_bias_v1/${LABEL}
PARAMS=$RAW_PARENT/20260628_${LABEL}_bnaware_tensor_range_capture_v1_gpu${GPU}/tensor_quant_params_calibration_bnaware_v1_to_pyramid_level2.json
FULL_DIR=$ROUTE/ap_fullval_1789_bnaware_bias_v1_gpu${GPU}_20260629

python scripts/stage2_import_native_int8_full_ap_row.py \
  --label $LABEL \
  --width $WIDTH_CSV \
  --raw-dir $FULL_DIR \
  --route-dir $ROUTE \
  --tensor-quant-params-path $PARAMS \
  --report-json $FULL_DIR/full_ap_eval_report.json \
  --run-id 20260629_native_int8_${LABEL}_bnaware_bias_fullval_v1
```

导入脚本会检查:

```text
processed_samples >= 1789
output_dequant_summary 存在
route manifest 存在
full_network_claim=false
```

失败时会写:

```text
$FULL_DIR/native_int8_ap_row_import_blocker.json
```

### Step 4: 刷新总表与 completion review

```bash
cd ${V2X_ROOT}
ROOT=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627

python scripts/stage2_generate_original60_quant_state_coverage.py --output-root "$ROOT"
python scripts/stage2_generate_fp16_int8_original60_completion_queue.py --output-root "$ROOT"
```

核对:

```bash
python - <<'PY'
import json, collections
from pathlib import Path
root=Path('${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627')
data=json.loads((root/'exports/original60_quant_three_metric_summary_latest.json').read_text())
rows=[r for r in data['rows'] if r['precision']=='int8']
print('int8 ap status', collections.Counter(r.get('ap_status') for r in rows))
for r in rows:
    if r['label'] in {'s0_024','s0_040','s0_056','s1_048','s2_160'}:
        print(r['label'], r.get('ap_status'), r.get('ap70'), r.get('ap_measurement_source'), r.get('quality_gate_status'))
PY
```

## 7. 常见失败与处理

| failure | 含义 | 处理 |
|---|---|---|
| `processed_samples < 1789` | full-val 没跑完整 | 不导入 row; 查 stdout/stderr 和中断点后 resume/重跑 |
| `pred_nonempty_count = 0` | 又出现空预测崩溃 | 回到 numeric_sanity, 查 output_dequant_summary/postprocess_summary |
| `ap_row_allowed=false` | gate 阻止写正式 row | 看 `ap_row_block_reason`, 修正后重跑 |
| `missing output_dequant_summary.json` | bridge 未完整落盘 | 查 runner stderr, 不手工补 |
| importer 写 blocker | row contract 不满足 | 读 `native_int8_ap_row_import_blocker.json`, 修脚本或输入 artifact 后重导 |
| AP 极低但非空 | route 可运行但精度未恢复 | 对比 5-sample smoke 同 label, 查 full-val postprocess 分布和 sample drift |

## 8. 不要重复的实验

除非文件缺失, 新窗口不要重复:

```text
1. s0_040 root-cause prefix trace。
2. BN-aware reference range capture 的探索版本。
3. 5-sample BN-aware bias smoke。
4. old centered conv route 的 full AP v2 低 AP 导入。
```

可直接复用:

```text
1. 5 个 label 的 checkpoint-consistent ONNX。
2. 5 个 label 的 tensor_quant_params_calibration_bnaware_v1_to_pyramid_level2.json。
3. 5 个 label 的 route_dir。
4. 5 个 label 的 TVM graph .so。
5. 5 个 label 的 runtime_weights_int8.npz。
6. 5 个 label 的 numeric_sanity 与 5-sample AP smoke gate 结果。
```

## 9. /goal 命令

```text
/goal 在 ${V2X_ROOT} 中单 agent 推进 INT8 AP full-val 补点。启动后先阅读 multi_agent/methods/design/auto-tuning/progress/6_27/58_6_29_冷启动交接_INT8_APFullVal补点与BNAwareSmoke复现流程.md, 再读 exports/native_int8_bnaware_bias_ap_smoke_5labels_review_latest.md、55_6_28_冷启动交接_NativeINT8精度崩溃修复与ScaleAwareGate.md、57_6_29_交接文档_阶段三_FP32Energy异常复盘与冷启动防错计划.md。不要重复已完成的 5-sample BN-aware bias smoke; 先在 H800 远端审计 5 个 label 的 route_dir、tensor_quant_params、TVM graph .so、runtime_weights_int8.npz、smoke full_ap_eval_report.json。固定规则: 不明文写 H800 密码; 启动前审计 nvidia-smi/compute-apps; 5-sample smoke 不写 measured AP row; full-val row 必须 processed_samples>=1789、pred_nonempty_count>0、AP30/AP50/AP70 finite、ap_row_allowed=true、full_network_claim=false; 当前 BN-aware route 仍 float32-heavy, full-val AP 只能证明 AP 可恢复, 不能单独宣称 fully integer INT8 acceleration 完成。执行顺序: P0 复核 s0_024/s0_040/s0_056/s1_048/s2_160 既有 smoke artifacts; P1 复用已有 route 和 params, 对 5 个 label 跑 scripts/stage2_h800_native_int8_real_activation_bridge.py --num-samples 1789 --full-ap-min-samples 1789 --ap-row-min-samples 1789; P2 每个 label 通过 gate 后运行 scripts/stage2_import_native_int8_full_ap_row.py 写 rows/native_int8_original60_ap_rows_v1.jsonl; P3 刷新 scripts/stage2_generate_original60_quant_state_coverage.py 与 scripts/stage2_generate_fp16_int8_original60_completion_queue.py; P4 写新的 review md/json, 明确成功 label、失败 blocker、AP 趋势和 raw artifact。遇到失败不停止全局, 保存 raw artifact、command、stdout/stderr、GPU id、failure reason, 写 per-label blocker 后继续其他 label。
```

