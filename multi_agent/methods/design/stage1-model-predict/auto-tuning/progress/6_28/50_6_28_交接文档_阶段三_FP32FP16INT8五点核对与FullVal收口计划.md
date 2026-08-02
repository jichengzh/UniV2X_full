# 56_6_28_交接文档_阶段三_FP32FP16INT8五点核对与FullVal收口计划

## 0. 本文用途

本文用于清理上下文后的冷启动续接。当前任务不再停留在“INT8 是否完全崩溃”的判断上, 而是进入:

```text
1. 统一 FP32 / FP16 / INT8 的证据口径;
2. 把 5 个 smoke 点从非塌缩证据推进到同协议 full-val AP 对比;
3. 后续扩展到 original60 的 FP16 与 INT8 大范围补点;
4. 明确不把 5-sample INT8 AP 误写成 full-val measured AP。
```

最新审阅表:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_fp16_int8_5label_speed_ap_crosscheck_latest.md
```

最新 INT8 精度恢复审阅:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_bnaware_bias_ap_smoke_5labels_review_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_bnaware_bias_ap_smoke_5labels_review_latest.json
```

## 1. 启动初期必读材料

新窗口启动后先读以下文件, 再改代码或启动 GPU 任务:

```text
本文:
multi_agent/methods/design/auto-tuning/progress/6_27/56_6_28_交接文档_阶段三_FP32FP16INT8五点核对与FullVal收口计划.md

INT8 修复主线:
multi_agent/methods/design/auto-tuning/progress/6_27/55_6_28_冷启动交接_NativeINT8精度崩溃修复与ScaleAwareGate.md

FP16 AP/checkpoint 主线:
multi_agent/methods/design/auto-tuning/progress/6_27/54_6_28_冷启动交接_FP16_AP大规模补点与CheckpointRecovery.md

FP16/INT8 completion review:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.json

FP32/FP16/INT8 5 点交叉核对:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_fp16_int8_5label_speed_ap_crosscheck_latest.md

当前量化三指标总表:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.csv
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.json
```

建议快速命令:

```bash
cd ${V2X_ROOT}
sed -n '1,220p' multi_agent/methods/design/auto-tuning/progress/6_27/56_6_28_交接文档_阶段三_FP32FP16INT8五点核对与FullVal收口计划.md
sed -n '1,160p' multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_fp16_int8_5label_speed_ap_crosscheck_latest.md
sed -n '1,180p' multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_bnaware_bias_ap_smoke_5labels_review_latest.md
```

## 2. 固定规则与红线

1. latency 指标统一使用 ms; 如果原始字段是 `latency_p50_us`, 对外表格必须转换为 ms。
2. 所有 TVM latency/energy 仍是 backbone/subnet compiled module 口径, 不是 full network end-to-end; `full_network_claim=false`。
3. 不得把 TRT、hybrid engine 或 5-sample smoke AP 伪装成 H800 TVM full-val measured AP。
4. 不得把 INT8 5-sample AP 高于 FP16 full-val AP 解读为 INT8 更准。
5. INT8 AP 行只有 `processed_samples=1789` 且同协议通过后, 才能进入正式 AP measured row。
6. FP32 remap 必须保留原始 artifact/source_files/run_id, 并写明从历史 fp16-tagged row 重映射而来。
7. 找不到 checkpoint 或 AP 源时不能盲找到超时; 必须写 per-label blocker, 然后进入 checkpoint recovery 或重新 finetune 生成。
8. 启动 H800 GPU 任务前必须重新审计 `nvidia-smi` 和 compute-apps; 不 kill 未知任务。
9. 不把 H800 密码写入文档、脚本、日志或提交; 按 RUNBOOK 使用临时环境变量方式连接。
10. 遇到失败不直接停止, 必须落盘 raw artifact、命令、stdout/stderr、失败原因、下一步诊断。

## 3. 当前状态摘要

### 3.1 FP32 状态

目前需要区分“已有证据”和“总表正式声明”:

```text
FP32 latency:
- 5 个核对点均已有 measured row。
- 来源: original60_quant_20260627/rows/latency_original60_quant_rows_v1.jsonl。
- 这是此前确认的 fp16-tagged TVM backbone row 重映射主线中最可靠的一部分。

FP32 energy:
- coverage_pipeline_v1 中有 5 个点的 historical measured energy row。
- 这些 row 原本 quant_policy=fp16, 但 source_files 指向 *_backbone.onnx, 可作为 FP32-remap evidence。
- 当前 original60_quant energy summary 仍未全部正式导入为 FP32 measured row; 不能在总表口径中说已完全收口。

FP32 AP:
- 本地 ap_stability_20260626/raw/<label>/ap_eval_report.json 可读到 4 个点 full-val AP。
- 这些报告 tag/command 中仍有 *_fp16.engine / q_fp16 痕迹, 不能不经审计直接改写为 FP32 AP measured。
- s2_160 在历史 AP-stability 阶段记录为 export/eval quarantined, 本地没有对应 FP32/AP-stability AP 报告。
```

结论:

```text
不能再说“没有 FP32 实测”。更准确的说法是:
FP32 latency 实测明确存在; energy 有历史 remap 证据; AP 有 4 个历史 full-val 报告但需口径审计, s2_160 AP 缺口仍在。
```

### 3.2 FP16 状态

```text
FP16 latency = 60/60 measured
FP16 energy  = 60/60 measured
FP16 AP      = 5/60 measured
```

当前 5 个 FP16 AP 是 full-val true-FP16 AP, `num_samples=1789`:

```text
s0_024, s0_040, s0_056, s1_048, s2_160
```

FP16 AP 后续主线仍是 checkpoint recovery / finetune generation, 不是继续盲找 checkpoint。

### 3.3 INT8 状态

当前 INT8 已完成:

```text
native INT8 latency = 60/60 measured
native INT8 energy  = 60/60 measured
native INT8 AP      = 0/60 formal measured
```

5 个 INT8 AP smoke 已完成, 且精度崩溃问题已从“全空预测/极低 AP”恢复到非空、AP30/AP50/AP70 合理非零趋势:

```text
s0_024, s0_040, s0_056, s1_048, s2_160
```

但这 5 个 AP 是 5-sample smoke, `processed_samples=5`, 不是 full-val measured AP, 不能导入正式 AP 行。

INT8 修复的关键结论:

```text
已修复:
- reference range capture 误把 ONNX Conv output 对齐到 PyTorch BN 前输出的问题。
- TVM native route 在 Conv+BN fusion 后丢失 Conv bias 的问题。
- bias 已按 input_scale * weight_scale 量化为 int32 accumulator scale 并传入 TVM route。

仍未最终完成:
- scale-aware route 为了验证 AP 可恢复, 仍允许 float32 requant/Add/ReLU 实值换算。
- 这证明 AP 不再塌缩, 但还不能宣称最终 fully integer fixed-point INT8 acceleration 完成。
```

## 4. 五点速度/能耗/AP核对表

AP 列格式为 `AP30/AP50/AP70`。INT8 AP 是 5-sample smoke; FP16 AP 是 full-val 1789 samples; FP32 AP 是历史 AP-stability evidence。

| label | FP32 latency ms | FP32 energy J | FP32 AP30/50/70 | FP16 latency ms | FP16 energy J | FP16 AP30/50/70 | INT8 latency ms | INT8 energy J | INT8 AP30/50/70 |
|---|---:|---:|---|---:|---:|---|---:|---:|---|
| s0_024 | 39.523 | 4.955 | 0.833/0.791/0.632 | 37.703 | 8.968 | 0.796/0.755/0.596 | 12.286 | 2.855 | 0.837/0.827/0.752 |
| s0_040 | 44.705 | 4.794 | 0.823/0.781/0.627 | 42.980 | 10.009 | 0.788/0.745/0.590 | 16.578 | 2.839 | 0.826/0.826/0.708 |
| s0_056 | 51.342 | 6.196 | 0.832/0.791/0.624 | 48.298 | 11.197 | 0.795/0.754/0.587 | 14.632 | 3.449 | 0.877/0.867/0.767 |
| s1_048 | 44.503 | 5.500 | 0.822/0.783/0.636 | 42.282 | 9.575 | 0.786/0.747/0.599 | 11.162 | 2.647 | 0.793/0.793/0.711 |
| s2_160 | 49.777 | 4.163 | NA | 47.530 | 10.854 | 0.793/0.751/0.593 | 12.059 | 2.869 | 0.859/0.849/0.757 |

## 5. 为什么 INT8 AP 会高于 FP16

当前不能解释为 INT8 精度优于 FP16。原因是协议不一致:

```text
FP16 AP:
- full-val
- 1789 samples
- 可作为 measured AP row

INT8 AP:
- 5-sample smoke
- 只验证非塌缩和基本趋势
- 样本方差极大
- ap_row_allowed=false, block_reason=full_eval_num_samples_5_lt_1789
```

5 个样本可能偏简单, 或阈值/排序在小样本上刚好有利, 因此 AP70 高于 FP16 full-val 并不异常。正确结论是:

```text
INT8 精度崩溃已被修复到可继续 full-val 的状态;
不能宣称 INT8 比 FP16 更准;
下一步必须做 same-protocol full-val 或至少 same-sample paired FP16-vs-INT8 smoke。
```

## 6. 下一阶段目标

下一阶段必须收口到具体可验收目标:

```text
P0: 将 FP32 三轴口径修正清楚。
    - FP32 latency: 保持 measured。
    - FP32 energy: 审计 coverage_pipeline_v1 historical fp16-tagged energy rows, 生成正式 FP32-remap energy rows 或逐点 blocker。
    - FP32 AP: 审计 ap_stability_20260626 4 个 full-val AP report 是否可 remap; s2_160 明确为缺口或重跑。

P1: 完成 INT8 5 个点 full-val AP。
    - 从当前 BN-aware + int32 Conv-bias route 出发。
    - 先做 same-sample paired FP16-vs-INT8 smoke 解释 5-sample 高 AP。
    - 再跑 1789-frame full-val INT8 AP。
    - 成功后写 native_int8_original60_ap_rows_v1.jsonl measured rows。

P2: 继续 FP16 AP original60 大范围补点。
    - 对有 checkpoint 的 label 立即跑 true-FP16 full-val AP。
    - 对无 checkpoint 的 label 进入 checkpoint recovery 或 finetune generation, 不再盲找。

P3: 扩大 FP16/INT8 original60 coverage。
    - 最终目标仍是 original60 中 FP16 与 INT8 AP/energy/latency 三轴对齐。
    - INT8 AP 初始收口目标先以 5 个 smoke label full-val 成功为准, 再扩展。
```

## 7. 推荐执行顺序

### Step 1: 只读审计当前表与 artifact

```bash
cd ${V2X_ROOT}
python - <<'PY'
import json
from pathlib import Path
base=Path('multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627')
for rel in [
  'rows/latency_original60_quant_rows_v1.jsonl',
  'rows/energy_original60_quant_rows_v1.jsonl',
  'rows/fp16_true_original60_ap_rows_v1.jsonl',
  'exports/native_int8_bnaware_bias_ap_smoke_5labels_review_latest.json',
]:
    p=base/rel
    print(rel, p.exists(), p)
PY
```

### Step 2: FP32 remap 审计

目标是生成一个明确的审计文件:

```text
exports/fp32_three_axis_remap_audit_latest.md
exports/fp32_three_axis_remap_audit_latest.json
```

内容必须逐点标注:

```text
latency: measured / missing
energy: remap_candidate / measured / missing / blocker
AP: remap_candidate / measured / missing / blocker
artifact_path
source_files
是否允许写入正式 FP32 measured row
```

### Step 3: INT8 same-sample paired smoke

对当前 5 个 label 使用同一批 5 samples, 跑:

```text
FP16 paired smoke AP
INT8 paired smoke AP
```

目的不是写正式 AP, 而是回答:

```text
INT8 5-sample AP 为什么高于 FP16 full-val?
同样 5 个样本上 FP16 是否也同样偏高?
```

### Step 4: INT8 full-val AP

只有 5-sample smoke 通过后, 对 5 个 label 跑 1789-frame full-val:

```text
s0_024
s0_040
s0_056
s1_048
s2_160
```

成功门槛:

```text
processed_samples=1789
pred_nonempty_count 合理
AP30 >= AP50 >= AP70 或趋势可解释
head/postprocess 分布与 FP16 baseline 可解释
row importer 生成 measured AP row
```

### Step 5: 刷新 review/summary

成功后刷新:

```text
exports/original60_quant_three_metric_summary_latest.md
exports/original60_quant_three_metric_summary_latest.csv
exports/original60_quant_three_metric_summary_latest.json
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_completion_review_latest.json
```

## 8. 关键 artifact 位置

### FP32 / FP16 / INT8 5点核对

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_fp16_int8_5label_speed_ap_crosscheck_latest.md
```

### FP16 full AP

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_true_s0_024_full_amp_fp16_v1/ap_eval_report.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_true_s0_040_full_amp_fp16_v1/ap_eval_report.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_true_s0_056_full_amp_fp16_v1/ap_eval_report.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_true_s1_048_full_amp_fp16_v1/ap_eval_report.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_true_s2_160_full_amp_fp16_v1/ap_eval_report.json
```

### INT8 5-sample AP smoke

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_s0_024_scaleaware_bnaware_bias_v1/s0_024/ap_smoke_5samples_bnaware_bias_v1_gpu7/full_ap_eval_report.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_s0_040_scaleaware_bnaware_bias_v1/s0_040/ap_smoke_5samples_bnaware_bias_v1_gpu5/full_ap_eval_report.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_s0_056_scaleaware_bnaware_bias_v1/s0_056/ap_smoke_5samples_bnaware_bias_v1_gpu7/full_ap_eval_report.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_s1_048_scaleaware_bnaware_bias_v1/s1_048/ap_smoke_5samples_bnaware_bias_v1_gpu7/full_ap_eval_report.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_s2_160_scaleaware_bnaware_bias_v1/s2_160/ap_smoke_5samples_bnaware_bias_v1_gpu7/full_ap_eval_report.json
```

### INT8 numeric sanity

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_s0_024_scaleaware_bnaware_bias_v1/s0_024/numeric_sanity_bnaware_bias_v1_gpu7/output_dequant_calibration_summary.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_s0_040_scaleaware_bnaware_bias_v1/s0_040/numeric_sanity_bnaware_bias_v1_gpu5/output_dequant_calibration_summary.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_s0_056_scaleaware_bnaware_bias_v1/s0_056/numeric_sanity_bnaware_bias_v1_gpu7/output_dequant_calibration_summary.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_s1_048_scaleaware_bnaware_bias_v1/s1_048/numeric_sanity_bnaware_bias_v1_gpu7/output_dequant_calibration_summary.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_s2_160_scaleaware_bnaware_bias_v1/s2_160/numeric_sanity_bnaware_bias_v1_gpu7/output_dequant_calibration_summary.json
```

## 9. 已验证内容

本轮已完成:

```text
1. 新增 5 点交叉核对 md:
   exports/fp32_fp16_int8_5label_speed_ap_crosscheck_latest.md

2. 明确 INT8 AP smoke 高于 FP16 full-val 的解释:
   协议不一致, 5 samples 高方差, 不能宣称 INT8 更准。

3. 确认 INT8 精度崩溃已解除到可继续 full-val 的状态:
   processed_samples=5
   pred_nonempty_count=5
   AP30/AP50/AP70 非零且趋势可解释
   numeric corrcoef 高, s0_040 从旧 route 约 0.468/0.109/0.374 恢复到约 0.992/0.970/0.964
```

已跑测试:

```bash
cd ${V2X_ROOT}
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route
python -m py_compile framework/stage2/native_int8_full_onnx.py scripts/stage2_native_int8_tvm_worker.py scripts/stage2_h800_native_int8_op_alignment.py multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_full_onnx_route.py
```

结果:

```text
unittest: 57 tests OK
py_compile: OK
```

## 10. /goal 命令

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP32/FP16/INT8 三轴补点收口。启动后先阅读 multi_agent/methods/design/auto-tuning/progress/6_27/56_6_28_交接文档_阶段三_FP32FP16INT8五点核对与FullVal收口计划.md, 再阅读 55_6_28_冷启动交接_NativeINT8精度崩溃修复与ScaleAwareGate.md、54_6_28_冷启动交接_FP16_AP大规模补点与CheckpointRecovery.md、multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_fp16_int8_5label_speed_ap_crosscheck_latest.md 和 native_int8_bnaware_bias_ap_smoke_5labels_review_latest.md。固定规则: latency 统一 ms; 不把 5-sample smoke AP 写成 full-val measured AP; 不把 TRT/hybrid engine 伪装成 H800 TVM measured; 不把 INT8 smoke AP 高于 FP16 full-val 解读为 INT8 更准; 所有 TVM latency/energy 仍是 backbone/subnet scope 且 full_network_claim=false; 启动 H800 前必须重新审计 GPU/compute-apps; 不明文写密码。当前优先级: P0 审计并刷新 FP32 三轴口径, 将 FP32 latency measured、historical energy remap evidence、AP remap_candidate/缺口逐点写入 fp32_three_axis_remap_audit_latest.md/.json, 能合法导入的生成正式 FP32 measured rows, 不能导入的写 blocker。P1 对 s0_024、s0_040、s0_056、s1_048、s2_160 做 same-sample paired FP16-vs-INT8 smoke, 解释当前 INT8 5-sample AP 高于 FP16 full-val 的原因。P2 使用 BN-aware reference range + int32 Conv bias 的 native INT8 route 跑这 5 个 label 的 1789-frame full-val AP, 通过 processed_samples=1789、pred_nonempty_count 合理、AP30/AP50/AP70 趋势合理、head/postprocess 分布可解释后, 再写 native_int8_original60_ap_rows_v1.jsonl measured rows。P3 继续 FP16 AP original60 大范围补点: 有 checkpoint 立即跑 true-FP16 full-val AP; 无 checkpoint 则写 per-label blocker 并进入 checkpoint recovery 或 finetune generation, 不盲找。遇到任何 build/eval/import/adapter/gate 问题不得直接停止, 必须保存 raw artifact、命令、stdout/stderr、GPU 状态、failure reason、numeric/AP summary, 做反思和修复后继续。stop condition: 产出 FP32 三轴 remap audit; 5 个 INT8 label 至少完成 same-sample paired smoke 和 full-val AP 成功或逐点 blocker; summary/review md/csv/json 刷新; 新中文交接文档写明剩余缺口。
```
