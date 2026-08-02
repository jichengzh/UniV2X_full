# 2_6_30 执行记录: FP16 AP-Shape Matcher 修复与 H800 待运行命令

日期: 2026-06-30

## 0. 当前状态

已完成本地代码修复:

```text
scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py
framework/tests/test_stage2_fp16_full_engine_group_conv_matcher.py
```

本地验证已通过:

```bash
python -m unittest framework.tests.test_stage2_fp16_full_engine_group_conv_matcher -v
python -m unittest framework.tests.test_stage2_fp16_h800_gate_check framework.tests.test_stage2_fp16_h800_rewrite_suite_runner -v
python -m py_compile scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py
```

但 H800 远端 build/smoke 尚未执行。当前 shell 没有 `SSH 也失败:

```text
Permission denied (publickey,password).
```

因此本文只记录本地修复和下一步远端命令, 不能写成 AP-shape TensorCore gate 已完成。

---

## 1. Full-val v2 状态

本地 export 仍不存在:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_rewritten_full_engine_ap_fullval_latest.json
```

远端 full-val v2 需要在 H800 上检查:

```bash
RAW=${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260629_lhc07_apshape_fullval_v2
cat "$RAW/pid.txt" 2>/dev/null || true
ps -p $(cat "$RAW/pid.txt" 2>/dev/null) -o pid,ppid,stat,wchan:30,etime,pcpu,pmem,rss,cmd 2>/dev/null || true
ls -l "$RAW"/*report*.json "$RAW"/*blocker*.json 2>/dev/null || true
ls -l ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_rewritten_full_engine_ap_fullval_latest.json 2>/dev/null || true
find "$RAW" -maxdepth 3 -type f -printf "%T@ %TY-%Tm-%Td %TH:%TM:%TS %p %s\n" 2>/dev/null | sort -nr | head -20
tail -80 "$RAW/stdout.txt" 2>/dev/null || true
tail -120 "$RAW/stderr.txt" 2>/dev/null || true
```

---

## 2. AP-shape matcher root cause

旧 `_replace_group_conv_primfuncs_for_full_engine()` 的 matcher 有两类硬编码:

```text
1. classify() 只识别 speed-gate shape:
   input [2,256,64,128] -> output [2,256,32,64]
   input [2,256,32,64] -> output [2,256,32,64]

2. replacement spec 固定使用 GROUP_CONV_CANDIDATES 中的 speed-gate shape。
```

AP runtime shape 是:

```text
input [2,64,256,256]
AP-shape group conv internal shape 预期包含:
  downsample: [2,256,128,128] -> [2,256,64,64]
  repeated:   [2,256,64,64]   -> [2,256,64,64]
```

因此旧代码在 AP-shape 上会得到:

```text
replace_records=[]
wmma=0
tvm_mma_sync=0
tensorcore_gate=false
```

这不是 TVM 不能 build AP-shape INT/FP route, 而是 matcher 没命中。

---

## 3. 已完成修复

新增纯文本结构 matcher:

```text
_classify_full_engine_group_conv_primfunc_text(text)
```

修复点:

1. 不再硬编码 `64x128 -> 32x64` 和 `32x64 -> 32x64`。
2. 从 TIR `T.Buffer(...)` 文本抽取 shape。
3. 只匹配目标权重:

```text
T.Buffer((T.int64(256), T.int64(8), T.int64(3), T.int64(3)), "float16")
```

4. 结构识别:

```text
若存在 [2,256,H,W] 和 [2,256,H/2,W/2] -> fused_conv2d4_add10_relu6 / stride=2 downsample
若 [2,256,H,W] 至少出现两次 -> fused_conv2d6_add10_relu6 / stride=1 repeated
```

5. 根据实际匹配到的 input shape 动态生成 replacement spec。
6. replace record 中新增:

```text
matched_input_nchw
matched_output_nchw
match_rule
```

新增测试:

```text
framework/tests/test_stage2_fp16_full_engine_group_conv_matcher.py
```

测试覆盖:

```text
speed-gate downsample/repeated shape
AP-shape downsample/repeated shape
非目标 96-channel group conv 不匹配
```

---

## 4. H800 待运行命令

拿到远端凭据后, 先同步本地修复:

```bash
rsync -av \
  ${V2X_ROOT}/scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py \
  ${V2X_ROOT}/framework/tests/test_stage2_fp16_full_engine_group_conv_matcher.py \
  -e "ssh -p 30001 -o StrictHostKeyChecking=accept-new" \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST>:${V2X_ROOT}/
```

如果使用 `password-based SSH (disabled; use an SSH key)`:

```bash
export shell 设置, 不要写入文档'
rsync -av \
  -e "ssh -p 30001 -o StrictHostKeyChecking=accept-new" \
  ${V2X_ROOT}/scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST>:${V2X_ROOT}/scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py
rsync -av \
  -e "ssh -p 30001 -o StrictHostKeyChecking=accept-new" \
  ${V2X_ROOT}/framework/tests/test_stage2_fp16_full_engine_group_conv_matcher.py \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST>:${V2X_ROOT}/framework/tests/test_stage2_fp16_full_engine_group_conv_matcher.py
```

远端先跑测试:

```bash
cd ${V2X_ROOT}
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python -m unittest framework.tests.test_stage2_fp16_full_engine_group_conv_matcher -v
${V2X_DATA_ROOT}/tvm310/bin/python -m py_compile scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py
```

然后重建 AP-shape rewritten full-engine:

```bash
cd ${V2X_ROOT}
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/tvm/lib:$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path 2>/dev/null):${LD_LIBRARY_PATH:-}
CUDA_VISIBLE_DEVICES=4 ${V2X_DATA_ROOT}/tvm310/bin/python \
  scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py \
  --mode full-engine-group-conv-rewrite \
  --gpu 4 \
  --reps 10 \
  --full-reps 10 \
  --raw-dir ${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_shape_lhc07_20260630_matcher_fix \
  --export-dir multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_apshape_lhc07_rewrite_20260630_matcher_fix \
  --onnx ${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_shape_lhc07_20260629/lhc_07_apshape_multiscale.onnx \
  --group-conv-rewrite-filter all
```

预期 gate:

```text
replace_records 非空
matched_input_nchw 至少包含 [2,256,128,128] 或 [2,256,64,64]
wmma > 0
tvm_mma_sync > 0
tensorcore_gate=true
```

若 gate 失败:

```text
不要直接跑 AP smoke。
先检查 full_engine_legalize_fuse_before_rewrite.py 中目标 PrimFunc 是否 shape/name 与 matcher 假设不一致,
并保存 failure_traceback/stdout/stderr/TIR。
```

---

## 5. AP smoke 待运行命令

只有 AP-shape TensorCore gate 通过后再运行:

```bash
cd ${V2X_ROOT}
CUDA_VISIBLE_DEVICES=4 ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/stage2_h800_fp16_rewritten_activation_bridge.py \
  --label lhc_07 \
  --ckpt-dir ${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_lhc_07_2026_06_28 \
  --raw-dir ${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260630_lhc07_apshape_tensorcore_smoke_v1 \
  --gpu-id 4 \
  --num-samples 1 \
  --keep-detailed-samples 1 \
  --persistent-worker \
  --rewrite-report multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_apshape_lhc07_rewrite_20260630_matcher_fix/fp16_lhc07_full_engine_group_conv_rewrite_latest.json \
  --export-report-json multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_apshape_tensorcore_ap_smoke_20260630_latest.json
```

smoke 验收:

```text
processed_samples=1
failed_samples=0
pred_nonempty_count>0
AP30/AP50/AP70 非空或 report 中明确说明 1-sample 估计口径
output0/1/2 error 可解释
```

---

## 6. 下一步收口

拿到 H800 凭据后继续:

1. 检查 full-val v2 是否已完成或失败。
2. 同步本地 matcher 修复到 H800。
3. 重建 AP-shape rewritten full-engine。
4. 只有 `tensorcore_gate=true` 后跑 AP smoke。
5. 若 smoke 通过, 再启动 AP-shape TensorCore full-val。

---

## 7. 2026-06-30 H800 执行结果更新

### 7.1 H800 连接与 full-val v2 状态

H800 已恢复, `nvidia-smi` 显示 GPU0-7 均为 H800/sm90。

检查旧 full-val v2:

```text
raw_dir=${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260629_lhc07_apshape_fullval_v2
pid=3698217
```

结果:

```text
pid 已不再运行。
远端 export fp16_lhc07_rewritten_full_engine_ap_fullval_latest.json 存在,
但读取后发现它仍是 v1 误标的 1-sample 报告:
  raw_artifact=${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260629_lhc07_apshape_fullval_v1
  processed_samples=1
  requested_num_samples=1
  ap_measured=false
```

结论:

```text
旧 full-val v2 不能算 full-val 完成。
后续仍需重新启动 AP-shape TensorCore full-val。
```

### 7.2 Matcher 修复同步与远端测试

已同步:

```text
${V2X_ROOT}/scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py
${V2X_ROOT}/framework/tests/test_stage2_fp16_full_engine_group_conv_matcher.py
```

远端测试:

```bash
cd ${V2X_ROOT}
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python framework/tests/test_stage2_fp16_full_engine_group_conv_matcher.py -v
${V2X_DATA_ROOT}/tvm310/bin/python -m py_compile scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py
```

结果:

```text
3 tests OK
py_compile OK
```

### 7.3 第一次 AP-shape rebuild 未命中原因

第一次 rebuild 没加 `--cast-fp16-source`:

```text
export_dir=.../exports/fp16_apshape_lhc07_rewrite_20260630_matcher_fix
default_latency_ms=38.0264825
rewritten_latency_ms=38.021654000000005
replace_records=[]
scheduled_counts={"wmma":0,"tvm_mma_sync":0,"float16":0,"float32":228,"conv2d":204,"matmul":0}
tensorcore_gate=false
```

根因:

```text
AP-shape ONNX 源仍是 FP32, 本轮 matcher 只匹配 FP16 TIR buffer。
因此必须显式加 --cast-fp16-source。
```

### 7.4 AP-shape TensorCore rebuild 成功

第二次 rebuild 命令增加 `--cast-fp16-source`。

审阅文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_apshape_lhc07_rewrite_20260630_matcher_fix_castfp16/fp16_lhc07_full_engine_group_conv_rewrite_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_apshape_lhc07_rewrite_20260630_matcher_fix_castfp16/fp16_lhc07_full_engine_group_conv_rewrite_latest.md
```

H800 raw:

```text
${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_shape_lhc07_20260630_matcher_fix_castfp16/lhc07_full_engine_group_conv_rewrite/
```

关键结果:

| item | value |
|---|---:|
| source_cast_fp16 | `true` |
| default_latency_ms | `36.61666684` |
| rewritten_latency_ms | `24.47697138` |
| latency_delta_ms | `12.139695459999999` |
| speedup_ratio | `1.4959639520565555` |
| tensorcore_gate | `true` |
| wmma | `72` |
| tvm_mma_sync | `2` |
| replace_records | `2` |

命中的 AP-shape replacement:

| PrimFunc | candidate | matched_input_nchw | matched_output_nchw | rule |
|---|---|---:|---:|---|
| `fused_conv2d12_add8_relu6` | downsample | `[2,256,128,128]` | `[2,256,64,64]` | `weight_256x8x3x3_stride2_shape` |
| `fused_conv2d16_add8_relu6` | repeated | `[2,256,64,64]` | `[2,256,64,64]` | `weight_256x8x3x3_stride1_same_shape` |

AP-shape rebuild output error:

| output | shape | max_abs_err | mean_abs_err | mean_abs_err/ref_abs_mean |
|---|---:|---:|---:|---:|
| output0 | `[2,24,256,256]` | `0.0` | `0.0` | `0.0` |
| output1 | `[2,48,128,128]` | `0.0` | `0.0` | `0.0` |
| output2 | `[2,128,64,64]` | `0.806640625` | `0.0017165043391287327` | `0.013215835206210613` |

结论:

```text
AP runtime shape [2,64,256,256] 的 FP16 TensorCore full-engine speed gate 已闭合。
此前 AP-shape tensorcore_gate=false 的问题已经修复。
```

### 7.5 AP-shape TensorCore smoke 成功

审阅文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_apshape_tensorcore_ap_smoke_20260630_latest.json
```

本地 raw summary:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_rewritten_ap_bridge/lhc07_apshape_tensorcore_smoke_v1/
```

H800 raw:

```text
${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260630_lhc07_apshape_tensorcore_smoke_v1/
```

结果:

| item | value |
|---|---:|
| status | `success` |
| processed_samples | `1` |
| failed_samples | `0` |
| pred_nonempty_count | `1` |
| pred_total_count | `18` |
| AP30 | `0.7406249999999999` |
| AP50 | `0.7406249999999999` |
| AP70 | `0.6293154761904762` |
| smoke_gate_passed | `true` |
| elapsed_secs | `4.760504722595215` |

TVM worker artifact:

```text
${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_shape_lhc07_20260630_matcher_fix_castfp16/lhc07_full_engine_group_conv_rewrite/rewritten_full_engine.so
```

Smoke output error:

| output | shape | max_abs_err | mean_abs_err | mean_abs_err/ref_abs_mean |
|---|---:|---:|---:|---:|
| output0 | `[1,24,256,256]` | `0.031064987182617188` | `0.0012152163544669747` | `0.0014728081170246165` |
| output1 | `[1,48,128,128]` | `0.035881638526916504` | `0.0008538038819096982` | `0.002678844667527701` |
| output2 | `[1,128,64,64]` | `1.1837341785430908` | `0.001805933308787644` | `0.0189206381797621` |

Prediction / score / box distribution:

| item | value |
|---|---:|
| pred_count | `18` |
| pred_score min/max/mean/std | `0.26313111186027527 / 0.9875447750091553 / 0.6936538219451904 / 0.28932443261146545` |
| pred_box_tensor min/max/mean/std | `-30.92837905883789 / 99.93756103515625 / 13.12568473815918 / 26.383840560913086` |
| gt_box_tensor min/max/mean/std | `-48.09828567504883 / 101.95936584472656 / 14.554669380187988 / 30.269208908081055` |

结论:

```text
AP-shape TensorCore rewritten backbone/subnet -> PyTorch head/postprocess/AP eval bridge 已完成 1-sample smoke。
prediction 非空, AP30/AP50/AP70 与此前 non-TensorCore AP-shape smoke 一致。
下一步应启动 AP-shape TensorCore full-val, 不能再把旧 v1/v2 1-sample report 当 full-val。
```

### 7.6 AP-shape TensorCore full-val 已启动

smoke 通过后已启动新的 full-val 后台任务。

```text
raw_dir=${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260630_lhc07_apshape_tensorcore_fullval_v1
pid=1285434
export_report_json=multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_apshape_tensorcore_ap_fullval_20260630_latest.json
```

启动命令不含 `--num-samples`, 应按完整验证执行。

启动后约 50 秒检查:

```text
PID=1285434
process state=Rl
elapsed≈00:50
active/progress file reached worker_tmp/bridge_call_000039
report/blocker 尚未产出
stderr 只有 timm/pkg_resources/numpy overflow warnings
```

判断:

```text
full-val 正在推进, 不能写成完成。
后续需要持续监控到 full-val report 或 blocker 产出。
```

监控命令:

```bash
RAW=${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260630_lhc07_apshape_tensorcore_fullval_v1
cat "$RAW/pid.txt" 2>/dev/null || true
ps -p $(cat "$RAW/pid.txt" 2>/dev/null) -o pid,ppid,stat,wchan:30,etime,pcpu,pmem,rss,cmd 2>/dev/null || true
ls -l "$RAW"/*report*.json "$RAW"/*blocker*.json 2>/dev/null || true
ls -l ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_apshape_tensorcore_ap_fullval_20260630_latest.json 2>/dev/null || true
find "$RAW" -maxdepth 3 -type f -printf "%T@ %TY-%Tm-%Td %TH:%TM:%TS %p %s\n" 2>/dev/null | sort -nr | head -30
tail -80 "$RAW/stdout.txt" 2>/dev/null || true
tail -120 "$RAW/stderr.txt" 2>/dev/null || true
```

### 7.7 AP-shape TensorCore full-val v1 完成但缺全量 score/box aggregate

v1 后续完成:

```text
raw_dir=${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260630_lhc07_apshape_tensorcore_fullval_v1
processed_samples=1789
failed_samples=0
AP30=0.7912569013379303
AP50=0.7479978528038091
AP70=0.5923713372074328
pred_nonempty_count=1789
pred_total_count=38528
ap_measured=true
```

但检查发现:

```text
prediction_distribution 仍来自 keep_detailed_samples=1 的 postprocess_summary_items,
因此 samples=1, total_predictions=18。
这不能作为 full-val score/box distribution。
```

修复:

```text
scripts/stage2_h800_fp16_rewritten_activation_bridge.py
新增 StreamingTensorDistribution 和 FullPostprocessDistribution。
full-val loop 对每个样本流式统计 pred_score、pred_box_tensor、gt_box_tensor 和 pred_count。
report 新增 postprocess_distribution。
raw_dir 新增 postprocess_distribution_summary.json。
```

新增测试:

```text
framework/tests/test_stage2_h800_fp16_rewritten_activation_bridge.py
test_streaming_tensor_distribution_accumulates_all_values
test_full_prediction_distribution_uses_all_samples_and_tensor_stats
```

本地和 H800 测试:

```text
本地 16 tests OK:
  framework.tests.test_stage2_h800_fp16_rewritten_activation_bridge
  framework.tests.test_stage2_fp16_full_engine_group_conv_matcher
  framework.tests.test_stage2_fp16_tvm_worker

H800 bridge test 10 tests OK。
```

### 7.8 AP-shape TensorCore full-val v2 aggregate 完成

审阅文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_apshape_tensorcore_ap_fullval_20260630_aggregate_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_apshape_tensorcore_ap_fullval_20260630_aggregate_latest.md
```

本地 raw summary:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_rewritten_ap_bridge/lhc07_apshape_tensorcore_fullval_v2_aggregate/
```

H800 raw:

```text
${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260630_lhc07_apshape_tensorcore_fullval_v2_aggregate/
```

AP full-val:

| item | value |
|---|---:|
| processed_samples | `1789` |
| failed_samples | `0` |
| ap_measured | `true` |
| AP30 | `0.7911407009177379` |
| AP50 | `0.74768319713647` |
| AP70 | `0.5923936354624854` |
| pred_nonempty_count | `1789` |
| pred_total_count | `38524` |
| elapsed_secs | `1761.0960562229156` |

Prediction distribution:

| item | value |
|---|---:|
| samples | `1789` |
| nonempty | `1789` |
| total_predictions | `38524` |
| min_predictions | `2` |
| max_predictions | `47` |
| mean_predictions | `21.53381777529346` |

Score / box aggregate:

| tensor | samples | size | min | max | mean | std |
|---|---:|---:|---:|---:|---:|---:|
| pred_score | `1789` | `38524` | `0.20001035928726196` | `0.9901663064956665` | `0.6176334444312381` | `0.24548097793084023` |
| pred_box_tensor | `1789` | `924576` | `-102.38395690917969` | `102.38613891601562` | `6.17294421699048` | `31.487610601945267` |
| gt_box_tensor | `1789` | `858288` | `-102.35637664794922` | `102.36088562011719` | `4.9729003661967806` | `31.260924787382425` |

Output error aggregate:

| output | samples | shape | max_abs_err_max | max_abs_err_mean | mean_abs_err_mean | mean_abs_err/ref_mean mean | mean_abs_err/ref_mean max |
|---|---:|---:|---:|---:|---:|---:|---:|
| output0 | `1789` | `[1,24,256,256]` | `0.06964302062988281` | `0.03810295524870037` | `0.0011326542078991628` | `0.0013419927204424689` | `0.00149143515039661` |
| output1 | `1789` | `[1,48,128,128]` | `0.05471688508987427` | `0.03382740706588517` | `0.0008376248941177377` | `0.002601515459703793` | `0.002743668666707431` |
| output2 | `1789` | `[1,128,64,64]` | `2.5718517303466797` | `1.0840137661869396` | `0.002253711418625615` | `0.022903898929540298` | `0.17834489527917907` |

结论:

```text
lhc_07 AP runtime shape 的 FP16 TensorCore full-engine 已完成:
  1. speed gate: 36.6167 ms -> 24.4770 ms, speedup≈1.496x, wmma=72, tvm_mma_sync=2。
  2. AP bridge: TVM rewritten backbone/subnet output 已接入 PyTorch head/postprocess/AP eval。
  3. AP full-val: AP30/AP50/AP70 均合理, prediction 全量非空。
  4. full-val score/box distribution 和 output0/1/2 error 已保存。
```
