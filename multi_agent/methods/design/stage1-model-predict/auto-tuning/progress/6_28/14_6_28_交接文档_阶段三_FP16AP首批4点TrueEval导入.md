# 20_6_28_交接文档_阶段三_FP16AP首批4点TrueEval导入

日期: 2026-06-28

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/19_6_28_交接文档_阶段三_APRunnerBlocker落盘修复与FP16Smoke复现.md`
- `multi_agent/methods/design/auto-tuning/progress/6_27/18_6_28_交接文档_阶段三_FP16INT8_EnergyAP补点收口执行计划.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md`

## 0. 当前权威状态

当前权威状态以 `original60_quant_20260627` 最新 rows/exports 为准。

FP16 + INT8 当前覆盖:

| axis | FP16 | INT8 | 合计 |
|---|---:|---:|---:|
| latency | 60/60 measured | 60/60 measured | 120/120 measured |
| energy | 60/60 measured | 60/60 measured | 120/120 measured |
| AP70 | 4/60 measured | 0/60 measured | 4/120 measured |

completion review:

```text
latency measured = 120/120
energy measured = 120/120
AP measured = 4/120
jobs_requiring_action = 116
```

latency 口径仍保持:

```text
H800 TVM backbone/subnet module latency in ms
full_network_claim = false
```

AP row 口径:

```text
backend = model_eval
measurement_source = true_eval
full_network_claim = false
```

## 1. 本轮解决的 FP16 AP blocker

上一轮 FP16 `amp_fp16` smoke 的失败 traceback 定位到:

```text
${V2X_HOME}/heal_research/HEAL/opencood/data_utils/post_processor/voxel_postprocessor.py:442
boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + anchors_reshaped[..., [0, 1]]
```

根因:

```text
deltas / boxes3d 为 FP16
anchors_reshaped / anchors_d 被 HEAL postprocessor 显式转为 FP32
Half destination 接收 Float RHS, 触发 Index put dtype mismatch
```

本轮修复在 Stage2 runner 中完成, 不改 HEAL 源码:

```text
scripts/stage2_h800_true_fp16_ap_eval.py
```

新增 wrapper:

```text
inference_intermediate_fusion_fp32_postprocess(...)
```

策略:

```text
model/input 仍按 model_half 或 amp_fp16 路线运行
model output 在进入 dataset.post_process 前 cast 到 FP32
postprocess_dtype_policy = model outputs cast to float32 before dataset.post_process
```

这样保留 FP16 model eval 路线, 同时匹配 HEAL postprocessor 的 FP32 anchor math。row 仍保持 `full_network_claim=false`, 不把 AP eval 口径扩大成完整 TVM full-network latency claim。

## 2. 本轮新增测试

新增测试:

```text
test_true_fp16_runner_casts_model_outputs_to_fp32_before_postprocess
```

保留上一轮新增测试:

```text
test_true_fp16_ap_eval_execute_failure_writes_blocker_logs
test_true_fp16_ap_eval_keeps_relative_raw_dir_under_launch_cwd_after_chdir_failure
```

验证:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_original60_quant_completion
python -m py_compile scripts/stage2_h800_true_fp16_ap_eval.py scripts/stage2_generate_original60_quant_ap_true_eval_queue.py framework/tests/test_stage2_original60_quant_completion.py
```

当前验证结果:

```text
Ran 12 tests
OK
py_compile OK
```

## 3. 本轮新增 FP16 AP measured rows

新增 row 文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_ap_rows_v1.jsonl
```

当前 4 行:

| label | width | AP30 | AP50 | AP70 | num_samples | run_id |
|---|---|---:|---:|---:|---:|---|
| `s0_024` | 24x128x256 | 0.796236 | 0.754558 | 0.596145 | 1789 | `20260628_fp16_true_ap_s0_024_full_amp_fp16_v1` |
| `s0_040` | 40x128x256 | 0.787672 | 0.745429 | 0.589736 | 1789 | `20260628_fp16_true_ap_s0_040_full_amp_fp16_v1` |
| `s0_056` | 56x128x256 | 0.795163 | 0.753717 | 0.587067 | 1789 | `20260628_fp16_true_ap_s0_056_full_amp_fp16_v1` |
| `s1_048` | 64x48x256 | 0.786048 | 0.746732 | 0.599228 | 1789 | `20260628_fp16_true_ap_s1_048_full_amp_fp16_v1` |

每行共同字段:

```text
precision = fp16
quant_method = h800_tvm_true_fp16_onnx_relax
backend = model_eval
engine_kind = model_eval
measurement_status = measured
measurement_source = true_eval
metric = AP70
secondary_metrics includes AP30/AP50
full_network_claim = false
```

raw artifact:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_true_s0_024_full_amp_fp16_v1/
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_true_s0_040_full_amp_fp16_v1/
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_true_s0_056_full_amp_fp16_v1/
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_true_s1_048_full_amp_fp16_v1/
```

每个 raw 目录包含:

```text
ap_eval_report.json
eval_<label>_amp_fp16.yaml
gpu_preflight.json
layer_precision_summary.json
runner_command.json
```

## 4. 已刷新 canonical/review/gap/queue

已执行 canonical refresh:

```text
scripts/stage2_generate_original60_quant_state_coverage.py
scripts/stage2_generate_fp16_int8_original60_completion_queue.py
```

刷新产物:

```text
rows/latency_original60_quant_rows_v1.jsonl
rows/energy_original60_quant_rows_v1.jsonl
rows/ap_original60_quant_rows_v1.jsonl
exports/original60_quant_three_metric_summary_latest.md
exports/original60_quant_three_metric_summary_latest.json
exports/original60_quant_three_metric_summary_latest.csv
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_completion_review_latest.json
exports/fp16_int8_original60_gap_report_latest.md
exports/fp16_int8_original60_gap_report_latest.json
jobs/fp16_int8_original60_completion_queue_v1.jsonl
```

当前 queue:

```text
total_jobs = 120
fp16_ap_remaining = 56
int8_ap_remaining = 60
jobs_requiring_action = 116
```

## 5. Checkpoint/source inventory

当前确认可直接重跑 true-eval 的 FP16 checkpoint 来源来自历史 AP finetune smoke manifest。已确认并完成导入:

```text
s0_024
s0_040
s0_056
s1_048
```

历史队列中还有:

```text
s2_160
```

但其历史状态显示:

```text
failure_reason = JobFailure:onnx_export_failed_rc_1
```

下一步应先检查 H800 上:

```text
${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s2_160_2026_06_26/
```

如果 checkpoint/config 存在, 可直接用新的 runner 尝试 FP16 true-eval; 如果不存在或损坏, 写 `ap_eval_blocker.json` 并继续处理其他 labels。

## 6. 下一步

优先顺序:

1. 在 H800 上做 original60 checkpoint inventory, 不再只依赖本地目录。
2. 对存在 checkpoint 的 FP16 labels 继续批量跑 `stage2_h800_true_fp16_ap_eval.py --precision-mode amp_fp16`。
3. 对 checkpoint 缺失的 FP16 labels 写逐点 blocker, 但不要停止其他 labels。
4. 启动 native INT8 AP adapter; 目标是 `rows/native_int8_original60_ap_rows_v1.jsonl`。
5. 每批 AP rows 后刷新 canonical/review/gap/queue。

下一批建议:

```text
s2_160: 先检查 checkpoint/config, 再决定 true-eval 或 blocker
frontier/lhc labels: 需要先定位是否有对应 finetuned checkpoint, 没有则不能硬导入 AP measured
INT8: 先做 s0_024 native INT8 AP adapter 单点
```

## 7. 下一阶段 /goal 命令

```text
/goal 继续在 ${V2X_ROOT} 执行 Stage2 original60 FP16/INT8 AP true-eval 收口, 单 agent 执行, 不启动 agent team。当前权威状态: FP16/INT8 latency=120/120 measured, energy=120/120 measured, AP70=4/120 measured, jobs_requiring_action=116; latency 是 H800 TVM backbone/subnet module latency in ms, full_network_claim=false。本轮已修复 FP16 AP runner 的 HEAL postprocess dtype mismatch: model/input 保持 model_half 或 amp_fp16, model output 在 dataset.post_process 前 cast 到 FP32, 并已完成 s0_024/s0_040/s0_056/s1_048 四个 FP16 full AP true-eval measured rows, 每行 num_samples=1789 且有 ap_eval_report、layer_precision_summary、runner_command、gpu_preflight。下一步先在 H800 做 original60 checkpoint inventory; 对存在 checkpoint 的 FP16 labels 继续批量跑 scripts/stage2_h800_true_fp16_ap_eval.py, 对 checkpoint 缺失或 eval 失败的 label 写 ap_eval_blocker.json/stdout/stderr 并继续其他 label; 同时启动 native INT8 AP adapter, 先做 s0_024 单点, 再扩展 original60。每批完成后用 scripts/stage2_generate_original60_quant_state_coverage.py 和 scripts/stage2_generate_fp16_int8_original60_completion_queue.py 刷新 canonical summary/review/gap/queue。最终验收仍是 FP16/INT8 latency=120/120 measured、energy=120/120 measured、AP70=120/120 measured、jobs_requiring_action=0。
```
