# 6_6_27_交接文档_阶段二_FP32Remap_trueFP16_INT8Smoke结果

日期: 2026-06-27

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/5_6_27_交接文档_阶段二_FP32Smoke与INT8Route.md`
- `multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md`

## 0. 置顶审阅结论

三点 FP32 / true-FP16 / INT8 latency 对齐如下, 单位统一为 `ms`:

| label | width | FP32 smoke ms | true-FP16 smoke ms | INT8 smoke ms | summary scope |
|---|---|---:|---:|---:|---|
| `base` | 64x128x256 | 6.208528 | 53.277002 | 59.078584 | quant anchor only |
| `s0_024` | 24x128x256 | 39.522546 | 37.734585 | 41.040342 | original60 + quant anchor |
| `s1_048` | 64x48x256 | 44.503228 | 42.327026 | 46.150869 | original60 + quant anchor |

关键修正:

1. 历史 FP16-tagged latency 有严重可信度问题: 58 条 historical FP16-tagged measured row 已被 audit 归类为 `remap_to_fp32_candidate`, 很可能是把 FP32 当作 FP16 测量或入表; 这些历史行不能再作为 true-FP16 速度证据。
2. 本轮 true-FP16 证据只来自 `base/s0_024/s1_048` 新生成的 `*_backbone_true_fp16.onnx` H800 TVM smoke, 且只声明 backbone/subnet latency 与 energy smoke, 不声明全网络 FP16 或 AP measured。
3. INT8 三点 smoke 只证明 TVM VM artifact 可 route 并能跑 latency/energy smoke; calibration source 为 synthetic shape smoke, 不声明真实数据集 INT8 校准效果或 AP measured。
4. 三点 latency 本身不能解释“量化对速度不敏感”的根因; 2026-06-27 后续 H800 root-cause profile 支持 INT8 QDQ/float32-heavy lowering 作为候选 contributor, 但不是完整因果闭环。FP16 行为、base schedule 公平性、per-op 耗时占比和 AP 仍未闭环。
5. 2026-06-27 10:16 后, original60/quant anchor 结构化 summary 已按 reviewer notes 收紧: 历史 FP16-tagged rows 默认降级为 no-claim for true-FP16, FP16/INT8 energy 已用 H800 idle baseline + active power telemetry 补齐三点 smoke, FP16/INT8 AP 仍为 no-claim/blocker。
6. 2026-06-27 10:32 后, quant anchor 中 `p50/p75/trap25/s0_040/s0_056` 这 5 条历史 FP16 seed 也已降级为 no-claim; quant anchor 现在只有 `base/s0_024/s1_048` 三条 true-FP16 latency measured。
7. 2026-06-27 11:45 后, 新一轮 agent team 已完成实验/批判闭环: reviewer 指出的 FP32 remap `candidate_id` 精度污染已修正为 `:fp32`, 原始 suspect provenance 保留在 `original_candidate_id`; FP16/INT8 energy 的 `measurement_source/claim_status` 已统一降级为 smoke-only。
8. 新增 per-op/equiv 根因 helper `raw/quant_speed_root_cause/stage2_h800_quant_speed_per_op_or_equiv.py`, 并生成 schema-only evidence `raw/quant_speed_root_cause/20260627_per_op_schema_only_v1`; 该 evidence 只验证输出 schema 和 gating, 不是 H800 实测 per-op timing。
9. 2026-06-27 12:28 主实验 agent 已重试 H800 per-op/equiv helper: `raw/quant_speed_root_cause/20260627_per_op_or_equiv_h800_light_v5`。该 run 在 H800 GPU 3 + TVM Relax VM 上执行, `trt_used=false`, `full_network_claim=false`, 状态为 `partial`: 1 个 cell 成功、8 个 cell 失败。唯一成功是 `base/fp32`, artifact p50 `6.212230 ms`, selected equivalent op p50 sum `0.154017 ms`, coverage ratio `0.024793`; 这不是 root-cause closure。
10. v5 失败模式已定位为 instrumentation blocker: FP16/INT8 和 `s0_024/s1_048` 的 direct PrimFunc microbench 在 memory verification / host memory binding 阶段失败, `vm_overhead_baseline` 也未能计时。下一步必须修 instrumentation, 使用 native TVM profiler、GPU schedule 后的 PrimFunc microbench, 或已调度 lowered functions; 不应原样重跑并宣称闭环。
11. reviewer 新增证据修正已完成: FP16 latency smoke/import 的 `measurement_source` 统一为 `true_measurement_smoke`; summary JSON 新增 `latency/energy/ap_{measurement_source,claim_status,evidence_scope,repeat}` 字段; INT8 digest 已拆为 `engine_digest`、`calibration_manifest_digest`、`quant_recipe_digest`、`layer_precision_summary_digest`, QDQ ONNX digest 仍是 `unknown_remote_only_not_synced`。
12. 新增 structured manual lowered evidence: `exports/quant_speed_manual_lowered_evidence_int8_qdq_v1.json`, 并写入三个 INT8 `tvm_operator_inventory.json` 的 `manual_lowered_evidence` 引用。它支持 QDQ/float32-heavy 作为 candidate contributor, 但仍不是因果闭环。

## 1. 本轮硬约束

1. measured backend 仍限定为 H800 + TVM / TVM VM, 不使用 TRT。
2. 所有新增 latency measured row 均为 backbone/subnet scope, `full_network_claim=false`。
3. latency 对外统一使用 `ms`; JSONL 原始 row 仍保留 schema 规定的 `latency_p50_us`, summary/审阅表转换为 `latency_ms`。
4. 历史 FP16-tagged row 不再作为 true-FP16 evidence; 只能作为 suspect provenance 或 FP32 remap source。
5. H800 登录敏感信息未写入仓库文件、脚本、JSON/JSONL 或 Markdown。

## 2. FP32 original60 remap/audit

新增脚本:

```text
scripts/stage2_audit_fp32_original60_remap.py
```

输入:

```text
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/rows/latency_lut_rows_original60_v1.jsonl
```

输出:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_original60_remap_audit_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_original60_remap_audit_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp32_latency_original60_remapped_rows_v1.jsonl
```

audit 结果:

| classification | count |
|---|---:|
| `remap_to_fp32_candidate` | 58 |
| `needs_remeasure` | 0 |
| `quarantine` | 0 |

说明:

1. 这 58 个 historical FP16-tagged measured label 的 `source_files` 指向 plain `*_backbone.onnx`, 没有 `*_fp16.onnx` / true-FP16 artifact marker。
2. 对可 remap 行生成独立 FP32 rows, `measurement_source=historical_true_measurement_reclassified`, `quality_gate_status=fp32_reclassified_from_suspect_fp16_tagged_row`, 并保留原始 `run_id/source_files/raw_artifact`。
3. `s0_024` / `s1_048` 同时有 remap 候选和本轮直接 FP32 smoke; summary 优先使用直接 FP32 smoke。
4. `lhc_17` 和 `s2_096` 原始 summary 中是 latency quarantine / not-run, 不在 58 个可 remap measured row 内; 当前 FP32 latency 仍是 no-claim。
5. reviewer 发现 remap 产物曾把 `candidate_id` 继续保留为 `:fp16`; 已修正为 row-level `candidate_id=:fp32`, 并用 `original_candidate_id` 保存原始 `:fp16` provenance, 避免后续 join 把 FP32 remap 当作 FP16。

刷新后的 original60 FP32 latency:

| category | count |
|---|---:|
| FP32 precision rows | 60 |
| FP32 latency measured | 58 |
| FP32 latency no-claim | 2 |

no-claim label:

```text
lhc_17, s2_096
```

## 3. true-FP16 三点 smoke

输出 rows:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_latency_smoke_rows_v1.jsonl
```

canonical schedule 采用 `metaschedule_tuned`:

| label | width | true-FP16 latency_ms | evidence |
|---|---|---:|---|
| `base` | 64x128x256 | 53.277002 | input dtype float16, initializer float16=100 |
| `s0_024` | 24x128x256 | 37.734585 | input dtype float16, initializer float16=56 |
| `s1_048` | 64x48x256 | 42.327026 | input dtype float16, initializer float16=56 |

本地 dtype evidence 已同步:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_true_smoke/20260627_075128/base/layer_precision_summary.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_true_smoke/20260627_075128/s0_024/layer_precision_summary.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_true_smoke/20260627_075128/s1_048/layer_precision_summary.json
```

quant anchor summary 已使用这三条 true-FP16 latency, 不再用历史 suspect FP16-tagged latency 覆盖这三点。
original60 summary 中仅 `s0_024/s1_048` 可进入 original60 视角; 其它 historical FP16-tagged latency 已降级为 `historical_fp16_tagged_latency_not_true_fp16_evidence` no-claim。

## 4. INT8 route + latency smoke

本轮扩展了 route helper, 新增 `s1_048`:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_route/stage2_h800_int8_route_attempt.py
scripts/stage2_probe_tvm_int8_artifact_route.py
```

route run:

```text
20260627_083401
```

artifact registry:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/artifacts/tvm_int8_artifact_registry_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/tvm_int8_artifact_status_latest.json
```

artifact 状态:

| label | artifact_status | artifact_digest |
|---|---|---|
| `base` | ready | `8ac8e6896effb4aef3d652e62e46c0139b306bd9003f8e012610e1e9afbae28f` |
| `s0_024` | ready | `d1a0400cce6ee2763d785527d9379612e57b0fc207e6520c409dbad92a870f95` |
| `s1_048` | ready | `01b4d0f14a0d045b36490bfb85d21f0879b6009a44148ff892ac243e6c7c6868` |

INT8 latency smoke helper:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_latency_smoke/stage2_h800_int8_latency_smoke.py
```

latency run:

```text
20260627_083911
```

rows:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/int8_latency_smoke_rows_v1.jsonl
```

INT8 measured latency:

| label | width | INT8 latency_ms | summary scope |
|---|---|---:|---|
| `base` | 64x128x256 | 59.078584 | quant anchor only |
| `s0_024` | 24x128x256 | 41.040342 | original60 + quant anchor |
| `s1_048` | 64x48x256 | 46.150869 | original60 + quant anchor |

注意: calibration source 为 synthetic shape smoke, 只用于 route/smoke; 不声明真实数据集 INT8 校准效果或 AP measured。INT8 energy 只在第 4.1 节按 TVM VM artifact + H800 idle baseline / power telemetry smoke 声明。

## 4.1 FP16/INT8 energy smoke

true-FP16 energy rows:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_ap_energy_smoke_rows_v1.jsonl
```

INT8 TVM VM energy rows:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/int8_ap_energy_smoke_rows_v1.jsonl
```

三点 energy measured:

| label | true-FP16 energy_J | INT8 energy_J | evidence |
|---|---:|---:|---|
| `base` | 9.671490 | 10.497519 | H800 idle baseline + active power telemetry |
| `s0_024` | 6.457136 | 7.480517 | H800 idle baseline + active power telemetry |
| `s1_048` | 7.207481 | 7.090357 | H800 idle baseline + active power telemetry |

关键修正:

1. `scripts/stage2_h800_run_measurement_job.py` 已修正 true-FP16 ONNX 输入 dtype: FP16 ONNX 现在喂 `float16`, 不再喂 `float32`。
2. `scripts/stage2_generate_energy_lut.py` 已同步到 H800, 支持 quant contract 参数。
3. 新增 direct TVM VM `.so` INT8 energy helper:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_energy_profile/stage2_h800_int8_energy_profile.py
```

失败诊断保留:

- `raw/fp16_true_energy/fp16_true_energy_base_20260627_100500/energy_result.json`: generic runner 曾因给 FP16 ONNX 喂 `float32` 失败。
- `raw/fp16_true_energy/fp16_true_energy_base_20260627_101500/energy_result.json`: telemetry 已完成但远端 energy generator 版本过旧, 不接受 quant contract 参数。后续 10:25 run 已修复并成功。

## 5. 刷新后的 summary

original60:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md
```

| precision | latency measured | note |
|---|---:|---|
| `fp32` | 58 / 60 | `lhc_17`, `s2_096` still no-claim |
| `fp16` | 2 / 60 latency, 2 / 60 energy | only `s0_024`, `s1_048` true-FP16 smoke rows; historical FP16-tagged rows downgraded to no-claim |
| `int8` | 2 / 60 latency, 2 / 60 energy | `s0_024`, `s1_048`; `base` not in original60 |

original60 AP 当前全部 no-claim。FP16/INT8 AP 需要下一阶段用 non-TRT true-FP16/TVM eval source 或 real TVM INT8 eval source 补齐, 不能从 TRT/4090/simulated source import 当作本阶段 measured。

quant anchor:

```text
multi_agent/data/stage2_lut_generation_v1/generated/quant_smoke_20260627/exports/quant_three_metric_summary_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/quant_smoke_20260627/exports/quant_three_metric_summary_latest.md
```

quant anchor latency measured count:

| precision | measured labels |
|---|---|
| `fp32` | `base`, `s0_024`, `s1_048` |
| `fp16` | `base`, `s0_024`, `s1_048` true-FP16 ONNX latency + energy smoke |
| `int8` | `base`, `s0_024`, `s1_048` INT8 QDQ TVM VM latency + energy smoke |

quant anchor FP16 latency no-claim labels:

```text
p50, p75, trap25, s0_040, s0_056
```

failure reason:

```text
historical_fp16_tagged_latency_not_true_fp16_evidence
```

quant anchor AP 当前全部 no-claim; 这是 deliberate downgrade, 避免把历史 TRT/AP import 误读为 true-FP16/INT8 TVM AP 完成。

审阅表:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_int8_next_step_review_latest.md
```

根因计划与当前结论:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/quant_speed_root_cause_experiment_plan_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/quant_speed_root_cause_experiment_plan_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/quant_speed_root_cause_results_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/quant_speed_root_cause_results_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/quant_ap_source_compliance_audit_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/quant_ap_source_compliance_audit_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/quant_speed_root_cause/stage2_h800_quant_speed_profile.py
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/quant_speed_root_cause/20260627_tvm_vm_introspection/tvm_vm_module_introspection.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/quant_speed_root_cause/20260627_quant_speed_profile_v1
```

## 6. 本轮代码变更

新增/更新:

```text
scripts/stage2_audit_fp32_original60_remap.py
scripts/stage2_generate_original60_quant_state_coverage.py
scripts/stage2_generate_quant_anchor_smoke.py
scripts/stage2_generate_energy_lut.py
scripts/stage2_h800_run_measurement_job.py
scripts/stage2_audit_quant_ap_source_compliance.py
scripts/stage2_probe_tvm_int8_artifact_route.py
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_route/stage2_h800_int8_route_attempt.py
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_latency_smoke/stage2_h800_int8_latency_smoke.py
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_energy_profile/stage2_h800_int8_energy_profile.py
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/quant_speed_root_cause/stage2_h800_quant_speed_profile.py
framework/tests/test_stage2_lut_productization.py
```

## 7. 已知剩余 gap

1. original60 FP32 latency 若要求 60/60 measured, 只剩 `lhc_17` 和 `s2_096` 需要 H800 TVM 补测或明确 quarantine。
2. original60 INT8 latency 目前只导入 `s0_024` / `s1_048`; 大规模 original60 INT8 队列尚未启动。
3. FP16 original60 历史 measured rows 已在当前 summary 中降级为 no-claim for true-FP16; true-FP16 latency canonical 目前只确认 `base/s0_024/s1_048` 三点, 其中 `s0_024/s1_048` 可进入 original60 视角。
4. true-FP16 和 INT8 当前已有 latency + energy smoke; AP 仍是 no-claim/blocker。`quant_ap_source_compliance_audit_latest.*` 已逐项审计 `base/s0_024/s1_048` x `fp16/int8`: FP16 缺 non-TRT true-FP16/TVM eval source, INT8 缺 real TVM INT8 eval source。下一阶段若要 measured AP, 必须新增合规 AP runner/source; 在此之前只能保留逐点 blocker/quarantine。
5. 三点 latency smoke 显示 FP16/INT8 速度提升非常有限且不稳定; 当前根因结论为 partial: INT8 QDQ/float32-heavy lowering 只是有 evidence 支持的候选 contributor, 不能写成完整因果。FP16 行为、base schedule 公平性、per-op 耗时占比和 AP 仍未闭环。
6. 已保存的 INT8 TVM VM `.so` 只暴露 artifact-level `time_evaluator`; `astext/get_source/per-op profile` 无法直接从 saved `.so` 取得。已通过 rebuild/instrument helper 生成 lowered/operator inventory 与 total-time profile, 但仍缺 per-op timing。
7. H800 profiling helper `raw/quant_speed_root_cause/stage2_h800_quant_speed_profile.py` 已运行 `20260627_quant_speed_profile_v1`; 它只写 root-cause evidence, 不写 measured LUT row, 不作为 AP source。
8. per-op/equiv helper 已就位并通过本地 self-test/schema-only run: `raw/quant_speed_root_cause/stage2_h800_quant_speed_per_op_or_equiv.py` 和 `raw/quant_speed_root_cause/20260627_per_op_schema_only_v1/*`。随后 H800 v5 已真跑但只有 `base/fp32` 成功、8 cells 失败, root-cause verdict 仍保持 `partial`。
9. FP16/INT8 energy machine 字段已经统一为 `measurement_source=true_measurement_smoke` 与 `claim_status=claimable_true_measurement_smoke`; 不能把 energy smoke 写成稳定能耗收益或三指标完成。
10. 2026-06-27 11:40 本轮 H800 per-op/equiv 真跑因认证不可用未启动; 队列与失败诊断见 `plans/h800_quant_completion_queue_latest.json` 和 `raw/h800_access_blocker/20260627_main_experiment_agent_auth_probe_v1/access_probe_result.json`。

## 8. 验证记录

已运行:

```text
python3 -m unittest framework.tests.test_stage2_lut_productization framework.tests.test_stage2_energy_coverage_jobs framework.tests.test_stage2_artifact_task_planner
```

结果:

```text
Ran 45 tests in 6.891s
OK
```

已运行脚本语法检查:

```text
python3 -m py_compile framework/stage2/lut_productization.py scripts/stage2_audit_fp32_original60_remap.py scripts/stage2_generate_quant_anchor_smoke.py scripts/stage2_generate_original60_quant_state_coverage.py scripts/stage2_h800_run_measurement_job.py scripts/stage2_generate_energy_lut.py scripts/stage2_audit_quant_ap_source_compliance.py multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_energy_profile/stage2_h800_int8_energy_profile.py multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/quant_speed_root_cause/stage2_h800_quant_speed_profile.py multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/quant_speed_root_cause/stage2_h800_quant_speed_per_op_or_equiv.py
```

结果: exit 0。

per-op/equiv helper self-test:

```text
PYTHONDONTWRITEBYTECODE=1 python multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/quant_speed_root_cause/stage2_h800_quant_speed_per_op_or_equiv.py --self-test
```

结果: `self_test_passed`。

JSON / Markdown 一致性检查:

```text
quant_speed_root_cause_results_latest.json: ok
quant_speed_root_cause_experiment_plan_latest.json: ok
original60_quant_three_metric_summary_latest.json: ok
quant_three_metric_summary_latest.json: ok
stage2_ap_source_audit_latest.json: ok
raw/quant_speed_root_cause/20260627_tvm_vm_introspection/tvm_vm_module_introspection.json: ok
raw/quant_speed_root_cause/20260627_quant_speed_profile_v1/profile_summary.json: ok
raw/quant_speed_root_cause/20260627_per_op_schema_only_v1/run_manifest.json: ok
raw/quant_speed_root_cause/20260627_per_op_schema_only_v1/matched_schedule_policy_audit.json: ok
raw/quant_speed_root_cause/20260627_per_op_schema_only_v1/vm_overhead_baseline.json: ok
quant_ap_source_compliance_audit_latest.json: ok
handoff/review/root-cause/AP-audit Markdown fences: ok
quant anchor FP16 measured labels == base/s0_024/s1_048: ok
quant anchor FP16 historical seed labels == no_claim: ok
FP32 remap candidate_id ends with :fp32 and original_candidate_id preserves :fp16: ok
energy measured rows/imports use true_measurement_smoke / claimable_true_measurement_smoke: ok
AP compliance audit blocked count == 6: ok
per-op/equiv schema-only outputs have status=schema_only_no_measurement, full_network_claim=false, latency_unit=ms: ok
root-cause JSON records helper status as schema-only/no-measurement: ok
```

敏感信息检查:

1. 本轮相关脚本和 Markdown 已按认证信息、私钥、令牌类模式扫描。
2. 当前扫描无命中, 未发现明文认证信息。
3. `py_compile` 产生的 `__pycache__` 已清理。

## 9. 下一阶段目标修正

上一版 `/goal` 只覆盖 FP32 剩余点和 INT8 扩展, 目标过窄。修正后的下一阶段必须交付两类结果:

| workstream | required outcome | hard gate |
|---|---|---|
| FP16/INT8 AP 补测 + energy 复核 | `base/s0_024/s1_048` 的 true-FP16 与 INT8 AP measured rows, 并复核现有 energy measured rows; AP 不可完成时逐点 blocker/quarantine | AP 必须来自真实 eval source; energy 已有 H800 idle baseline / telemetry evidence, 若复测必须保留同等证据 |
| 速度不敏感根因实验 | 给出 FP16/INT8 相对 FP32 speedup 很小的实验证明结论 | 只能基于实验和 profiling 写结论; 未证实时写 `inconclusive` |
| 追加对比点 | 如三点不足以定位原因, 追加约 8 个 representative labels 的 FP16/INT8 latency/AP/energy 或 latency+profiling 对比 | 每个 label 必须有 provenance, 失败写 blocker, 不伪造 measured |
| summary/review 刷新 | original60 / quant anchor summary、review MD、gap report、交接文档同步更新 | 对外 latency 一律 `ms`; 新增 rows 默认 `full_network_claim=false` |

速度根因实验至少需要区分这些候选原因:

1. 是否只是 backbone/subnet 量化, 并未覆盖全网络, 导致端到端收益不可见。
2. 是否存在 FP32/FP16/INT8 dtype conversion 往返, conversion overhead 抵消 kernel 收益。
3. INT8 QDQ 是否被 TVM VM 编译成大量 fallback / dequantize-quantize 路径, 而不是有效 INT8 kernel。
4. FP16/INT8 是否没有拿到对应 tuned schedule, 或 schedule/shape 差异导致比较不公平。
5. 主要耗时是否落在未量化 op、memory-bound op、layout transform、VM runtime overhead 或数据搬运上。

允许的 evidence 包括但不限于:

```text
layer_precision_summary.json
ONNX op/type count
TVM Relax/VM lowered graph or operator list
TVM profiling / per-op timing
dtype conversion op count
artifact registry + digest
H800 idle gate + power telemetry
AP eval logs / metrics source
```

推荐新增输出:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/quant_speed_root_cause_experiment_plan_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/quant_speed_root_cause_results_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/quant_speed_root_cause_results_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_ap_energy_smoke_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/int8_ap_energy_smoke_rows_v1.jsonl
```

## 10. Agent team 职责与协作

下一阶段关于“量化对速度不敏感”的研究应启动两个 agent 协作, 主线程只整合有证据支撑且经过批判复核的结论。

| agent | role | ownership | required output |
|---|---|---|---|
| 主实验 agent | 负责实验执行、结果汇总、原因初判 | 运行/排队 `base/s0_024/s1_048` 的 true-FP16 与 INT8 latency/energy/AP; 维护 row、summary、artifact/profiling evidence; 验证 dtype、artifact digest、latency 单位与 `full_network_claim` | 3 个 FP16 完整数据 + 3 个 INT8 完整数据; 量化速度不敏感的候选原因排序; 证据不足时标 `inconclusive` |
| 批判 agent | 负责反驳、审计和风险复核 | 默认只读审阅主实验 agent 的命令、日志、row、summary、profiling evidence; 不直接覆盖主实验 agent 的 row/json/csv; 重点检查 FP32 误当 FP16、单位 us/ms、artifact 不一致、QDQ/fallback、AP 缺证、energy telemetry/digest 不足、结论过度外推 | 对主实验结论逐条批判; 给出必须补的实验或可接受/不可接受判定; 最终留下 reviewer notes |

协作规则:

1. 主实验 agent 先提交阶段性结论、证据路径和失败诊断。
2. 批判 agent 再逐条审阅, 默认寻找反例和证据缺口。
3. 若两者结论冲突, 交接文档必须保留冲突点、争议证据和下一步判定实验。
4. 禁止把没有实验或 profiling 支撑的推测写成最终根因。

本轮已整合的批判清单:

| issue | required handling |
|---|---|
| AP source audit 只证明 blocker, 不证明 AP measured 完成 | 六个 `base/s0_024/s1_048` x `fp16/int8` AP target 继续 no-claim/blocker; 只有真实 TVM/TVM VM eval source 才能写 measured AP |
| INT8 QDQ/float32-heavy lowering 不是完整因果证明 | 只能写 candidate contributor; 必须补 per-op/equivalent timing 才能判断 dominance |
| `base` FP32/FP16/INT8 可能有 schedule fairness 风险 | 下一阶段要做 matched schedule policy repeat/audit, 否则不能用 `base` 作因果证据 |
| FP16/INT8 energy 目前是 smoke | 只能声明 H800 idle baseline + telemetry smoke, 不声明稳定 energy benefit |
| FP32 remap `candidate_id` 曾保留 `:fp16` | `rows/fp32_latency_original60_remapped_rows_v1.jsonl` 现在使用 `:fp32`; 原始 provenance 保留在 `original_candidate_id` |
| FP16/INT8 energy machine 字段过强 | 已统一为 `true_measurement_smoke` / `claimable_true_measurement_smoke`; 不再写成普通 `true_measurement` |
| original60 INT8 `candidate_id` 精度污染已修复 | `candidate_id` 现在使用 `:int8`; 原始 provenance 保留在 `original60_candidate_id` |
| inventory 自动字段不是分类器 | QDQ/dequantize 解释以 lowered text/manual evidence 为准, 不把自动 `contains_*` flag 当 proof |
| FP16 latency smoke source 字段过强 | 已统一为 `true_measurement_smoke`; summary 使用 `evidence_scope=smoke_only` |
| summary measured smoke 可能被自动表格误读 | JSON summary 新增 measurement_source/claim_status/evidence_scope/repeat 字段 |
| INT8 digest 语义曾污染 | 已拆分 engine/calibration_manifest/recipe/layer_summary digest; QDQ ONNX digest 仍未闭合 |
| H800 per-op/equiv retry v5 未闭合 | `success=1, failed=8`; blocker 变为 instrumentation/schedule/profiler, 不能声明完整根因 |

本轮 agent team 执行状态:

| agent | status | output |
|---|---|---|
| 主实验 agent | closed | 增加 per-op/equiv helper, 补充 schema-only run; 主线程随后完成 H800 v5 retry, 结果为 partial/instrumentation-blocked |
| 批判 agent | closed | 指出 FP32 remap `candidate_id` 精度污染、energy smoke/latency smoke 字段过强、digest 语义污染、manual evidence 不足; 已修正并纳入验证 |

## 11. 下一阶段 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 LUT 量化维度收尾与根因实验, 并启动 agent team 协作。

先阅读:
- multi_agent/methods/design/auto-tuning/progress/6_27/6_6_27_交接文档_阶段二_FP32Remap_trueFP16_INT8Smoke结果.md
- multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_int8_next_step_review_latest.md

请使用 multi_agent_v1.spawn_agent 启动两个 sub-agent:
- 主实验 agent: agent_type=worker。职责是执行实验、排队/运行 H800 任务、汇总 rows/summary/profiling evidence、给出原因初判。
- 批判 agent: agent_type=reviewer。职责是审阅主实验 agent 的命令、日志、row、summary 和 profiling evidence, 专门寻找 FP32 误当 FP16、latency 单位 us/ms 混用、artifact digest 不一致、INT8 QDQ/fallback 未生效、AP 缺证、energy telemetry/digest 不足、结论过度外推等问题。

启动命令模板:

multi_agent_v1.spawn_agent(
  agent_type="worker",
  message="主实验 agent: 在 ${V2X_ROOT} 中执行 Stage2 true-FP16/INT8 三点完整数据补齐和量化速度不敏感根因实验。负责 base/s0_024/s1_048 的 H800 + TVM/TVM VM AP 补测、必要 latency/energy 复核、row/summary/profiling evidence 更新、原因初判。注意 20260627_per_op_or_equiv_h800_light_v5 已 partial, 下一步要修 instrumentation/TVM profiler/已调度 lowered functions, 不要原样重复失败路径。不要回退他人改动, 不写入敏感信息。"
)

multi_agent_v1.spawn_agent(
  agent_type="reviewer",
  message="批判 agent: 只读审阅主实验 agent 的命令、日志、row、summary、artifact/profiling evidence 与结论。重点攻击 FP32 误当 FP16、us/ms 混用、artifact digest 不一致、QDQ/fallback 未生效、AP 缺证、energy telemetry/digest 不足、v5 partial 被过度解读、结论过度外推。输出 reviewer notes 和必须补的实验。"
)

硬约束:
- measured backend 保持 H800 + TVM / TVM VM, 不是 TRT。
- 不得声明全网络量化; 新增 measured row 默认 full_network_claim=false。
- latency 对外统一使用 latency_ms / ms, 不允许混用 us/ms。
- 历史 FP16-tagged row 不能作为 true-FP16 证据; 这些行可能是 FP32 被当作 FP16 测量或入表。
- AP 必须来自真实 eval source; energy 必须保存 H800 idle baseline 和 power telemetry; 不得伪造 measured。
- AP source audit 只能作为 blocker evidence, 不能当作 measured AP。
- base 只属于 quant anchor 视角, 不要伪装成 original60 row; s0_024/s1_048 可进入 original60 视角。

当前状态:
- base/s0_024/s1_048 已有 FP32 latency smoke、true-FP16 latency smoke、INT8 latency smoke。
- true-FP16 与 INT8 已有三点 energy smoke; AP 仍缺合规 measured source。
- 三点 latency 显示 FP16/INT8 速度提升有限且不稳定; 当前 INT8 QDQ/float32-heavy lowering 只是候选 contributor, 还不能写成完整因果。
- 已有 per-op/equiv helper、schema-only 证据和 H800 v5 partial retry。v5 只有 `base/fp32` 成功, 8 个 FP16/INT8/s0/s1 cell 失败, coverage 仅 `2.48%`; 不能据此宣称 operator dominance。

首要目标:
- 复核并固定 base、s0_024、s1_048 三点 true-FP16 现有 latency_ms / energy_J, 补齐 AP70; 若 AP 无法合规完成, 写逐点 blocker/quarantine。
- 复核并固定 base、s0_024、s1_048 三点 INT8 现有 latency_ms / energy_J, 补齐 AP70; 若 AP 无法合规完成, 写逐点 blocker/quarantine。
- 若某点无法完成, 必须先 diagnose/fix/retry, 保存日志、定位失败阶段并尝试修复或替换 artifact; 只有确认不可恢复或缺少真实 source 后, 才允许写入 per-label blocker/quarantine。

根因研究目标:
- 先写 quant_speed_root_cause_experiment_plan_latest.md/.json。
- 做受控实验区分 backbone-only/未全网络量化、dtype conversion 往返、QDQ/fallback、schedule 不公平、memory-bound/layout、TVM VM overhead、operator coverage 不足、FP16/INT8 kernel 未实际命中等原因。
- 对 `base` 做 matched schedule policy repeat/audit 后, 才能把它用于因果归因。
- 必须保留 layer_precision_summary、ONNX op count、TVM lowered graph/operator list、TVM per-op profiling 或等价 evidence。
- 优先修复 `raw/quant_speed_root_cause/stage2_h800_quant_speed_per_op_or_equiv.py` 的 instrumentation: 使用 TVM native profiler、GPU schedule 后的 PrimFunc microbench, 或已调度 lowered functions。schema-only 输出和 v5 partial 都不能作为完整 latency attribution evidence。
- 没有实验证据时结论写 inconclusive, 禁止推测。
- 若三点不足以定位根因, 再追加约 8 个 representative labels 的 FP16/INT8 对比实验, 每个 label 记录 provenance、artifact digest、latency_ms、profiling evidence 和失败 blocker。

可选收尾:
- original60 FP32 latency 若要 60/60, 只补 lhc_17 和 s2_096 两点或明确 blocker, 不重跑 58 个已 remap label。

stop condition:
- 交付 3 个 true-FP16 完整数据和 3 个 INT8 完整数据, 每条都有 latency_ms、energy_J、AP70 measured 或明确 blocker/quarantine。
- 速度不敏感根因报告有实验证据结论, 或明确 inconclusive + 下一实验。
- 主实验 agent 输出结果已被批判 agent 审阅, reviewer notes 已写入交接或结果文档。
- original60/quant anchor summary、review MD、gap report、交接文档已更新。
- 回归测试通过, 敏感信息扫描为空。
```
