# 5_6_27_交接文档_阶段二_FP32Smoke与INT8Route

日期: 2026-06-27

序号: 5

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/0_6_27_冷启动入口_阶段二LUT.md`
- `multi_agent/methods/design/auto-tuning/progress/6_27/4_6_27_交接文档_阶段二_original60量化状态覆盖与INT8Route预检.md`
- `multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md`

## 1. 本轮硬约束

1. 当前 Stage2 LUT 主路径仍是 **H800 + TVM / TVM VM**, 不是 TRT。
2. 本轮没有声明全网络量化; 新增/刷新 row 均保持 `full_network_claim=false`。
3. FP32 latency smoke 可以写 measured row; INT8 本轮只写 artifact route readiness, 不写 latency/AP/energy measured claim。
4. latency 指标在总表、交接文档和下一阶段验收中统一使用 `ms`; runner 原始输出若为 `us`, 进入 summary 时必须转换为 `latency_ms`。
5. H800 登录敏感信息没有写入 repo 文件、Markdown、JSON/JSONL、脚本或日志。

## 2. H800 read-only probe 与 FP32 preflight

H800 mux 可用, 目标主机:

```text
<PRIVATE_HOST>
```

只读 probe 原始文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/h800_probe/h800_readonly_probe_latest.json
```

结果:

| 项 | 值 |
|---|---|
| status | `succeeded` |
| hostname | `<PRIVATE_HOST>` |
| idle_gpus | `[0,1,2,3,4,5,6,7]` |

preflight 输出:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_latency_smoke_preflight_latest.json
```

当前 `ready_for_manual_fp32_smoke=3`, 覆盖 `base`、`s0_024`、`s1_048`。

## 3. FP32 H800 TVM latency smoke

第一次 smoke 已到达 GPU idle gate, 但 TVM import/build 失败:

```text
undefined symbol: cudaGraphAddDependencies_v2, version libcudart.so.12
```

诊断结论:

1. Python 进程优先解析了 `/usr/local/cuda-12.2/.../libcudart.so.12`。
2. pip TVM CUDA runtime 路径中的 `libcudart.so.12` 才包含所需 symbol。
3. 修复方式是在启动 Python 前设置 runbook 环境, 让 pip CUDA runtime 位于 `LD_LIBRARY_PATH` 前段。

第二次 smoke `retry2` 成功, 结果文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp32_latency_smoke_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/logs/fp32_latency_smoke_retry2.out
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp32_latency_smoke/
```

关键 measured rows:

| label | precision | backend | TVM strategy | latency_ms | full_network_claim |
|---|---|---|---|---:|---|
| `base` | `fp32` | `h800_tvm` | `relax_metaschedule_reuse_existing_ms_db` | 6.208528 | false |
| `s0_024` | `fp32` | `h800_tvm` | `relax_metaschedule_reuse_existing_ms_db` | 39.522546 | false |
| `s1_048` | `fp32` | `h800_tvm` | `relax_metaschedule_reuse_existing_ms_db` | 44.503228 | false |

注意: original60 60 点 summary 只包含 `s0_024` 和 `s1_048`; `base` 只进入 quant anchor smoke summary。

## 4. FP32 smoke row 导入

新增 generator 能力:

```text
scripts/stage2_generate_original60_quant_state_coverage.py --fp32-latency-smoke-rows ...
scripts/stage2_generate_quant_anchor_smoke.py --fp32-latency-smoke-rows ...
```

导入规则:

1. 只接受 `precision=fp32`。
2. 只接受 `measurement_status=measured`。
3. 只接受 `backend` 以 `h800_tvm` 开头。
4. 必须 `full_network_claim=false`。
5. 优先选择 `metaschedule_tuned` / `relax_metaschedule_reuse_existing_ms_db` row。
6. label 可从 `label`、`software_point_id=original60:<label>:...` 或 `candidate_id` 推断。

新增测试覆盖:

```text
Stage2Original60QuantCoverageCliTest.test_original60_quant_state_coverage_imports_measured_fp32_smoke_latency
Stage2QuantAnchorSmokeCliTest.test_quant_anchor_smoke_imports_measured_fp32_smoke_latency
```

## 5. 刷新后的三指标 summary

original60 quant state:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.json
```

| axis | measured | no_claim |
|---|---:|---:|
| latency | 60 | 120 |
| energy | 56 | 124 |
| AP | 5 | 175 |

解释:

1. latency measured 从上一轮 58 提升到 60, 来自 `s0_024` 和 `s1_048` 的 FP32 H800 TVM smoke。
2. original60 中仍有 58 个 FP32 latency gap, 因为只启动了 3 点 smoke 且 `base` 不属于 original60 60 点。
3. INT8 latency/AP/energy 仍全部是 no-claim, 因为本轮只打通 artifact route, 没有启动 INT8 latency/AP/energy 测量。
4. original60 总表字段名为 `latency_ms`; 当前已导入的 FP32 smoke 为 `s0_024=39.522546 ms`、`s1_048=44.503228 ms`。
5. 重要修正: 当前 generated summary 中历史 FP16-tagged 行仍带有历史 `measured` status, 但这不是 FP16 measured evidence; 复查 provenance 后不能把这些行当作可信 true-FP16 证据。尤其 `s0_024` 和 `s1_048` 的历史 FP16-tagged latency 可能实际来自 FP32 backbone 测量或 FP32 artifact 归档。

quant anchor smoke:

```text
multi_agent/data/stage2_lut_generation_v1/generated/quant_smoke_20260627/exports/quant_three_metric_summary_latest.json
```

| axis | measured | no_claim |
|---|---:|---:|
| latency | 11 | 13 |
| energy | 4 | 20 |
| AP | 8 | 16 |

这里 `base`、`s0_024`、`s1_048` 的 FP32 latency smoke 均已导入。

quant anchor smoke 中三条 FP32 smoke 的 `latency_ms` 为:

| label | latency_ms | full_network_claim |
|---|---:|---|
| `base` | 6.208528 | false |
| `s0_024` | 39.522546 | false |
| `s1_048` | 44.503228 | false |

gap report:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_gap_report_latest.json
```

关键 gap:

| failure reason | count |
|---|---:|
| `tvm_fp32_backbone_measurement_not_run` | 58 |
| `tvm_int8_backbone_subnet_not_ready` | 60 |
| `energy_fp32_latency_measurement_not_run` | 60 |
| `tvm_int8_energy_backend_missing` | 60 |
| `ap_fp32_true_eval_not_available` | 60 |
| `tvm_int8_ap_eval_backend_missing` | 60 |

## 6. TVM INT8 backbone/subnet route

新增 H800 route helper:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_route/stage2_h800_int8_route_attempt.py
```

该 helper 只用于本轮 route attempt:

1. 大 artifact 写到 H800 `/exdata`, 不拉回 git repo。
2. repo 只保留小 JSON/日志。
3. calibration source 明确为 `synthetic_shape_smoke`, 不声明真实数据集校准效果。
4. 不产生 latency/AP/energy measured claim。

第一轮 route attempt:

| label | 结果 |
|---|---|
| `base` | synthetic static QDQ ONNX 导出成功; TVM compile/export 失败 |
| `s0_024` | synthetic static QDQ ONNX 导出成功; TVM compile/export 失败 |

失败原因:

```text
RuntimeError("[Errno 2] No such file or directory: 'nvcc'")
```

诊断修复:

1. H800 上存在 `/usr/local/cuda-12.2/bin/nvcc`。
2. 默认 PATH 未包含 nvcc。
3. 第二轮重跑时加入 `/usr/local/cuda-12.2/bin:/usr/local/cuda/bin`。

第二轮 route attempt 成功, run id:

```text
20260627_063158
```

artifact root:

```text
${V2X_DATA_ROOT}/s2_tvm/int8_route_20260627/20260627_063158
```

INT8 artifact registry:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/artifacts/tvm_int8_artifact_registry_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/tvm_int8_artifact_status_latest.json
```

当前 registry 状态:

| label | artifact_status | build_status | validation_status | artifact_digest |
|---|---|---|---|---|
| `base` | `ready` | `tvm_compile_succeeded` | `compiled_artifact_present_no_latency_claim` | `8ac8e6896effb4aef3d652e62e46c0139b306bd9003f8e012610e1e9afbae28f` |
| `s0_024` | `ready` | `tvm_compile_succeeded` | `compiled_artifact_present_no_latency_claim` | `d1a0400cce6ee2763d785527d9379612e57b0fc207e6520c409dbad92a870f95` |

artifact paths:

```text
${V2X_DATA_ROOT}/s2_tvm/int8_route_20260627/20260627_063158/base/base_backbone_int8_qdq_direct_tvm_vm.so
${V2X_DATA_ROOT}/s2_tvm/int8_route_20260627/20260627_063158/s0_024/s0_024_backbone_int8_qdq_direct_tvm_vm.so
```

配套 metadata:

| label | qdq_node_count | int8_initializer_count | metadata |
|---|---:|---:|---|
| `base` | 234 | 167 | calibration manifest, quant recipe, layer precision summary |
| `s0_024` | 192 | 170 | calibration manifest, quant recipe, layer precision summary |

注意: 虽然 TVM VM `.so` artifact 已 ready, 但本轮没有执行 INT8 latency smoke, 所以 original60/quant smoke 中的 INT8 三指标仍保持 no-claim。

## 7. 验证

已运行:

```bash
python3 -m unittest framework.tests.test_stage2_lut_productization framework.tests.test_stage2_energy_coverage_jobs framework.tests.test_stage2_artifact_task_planner
```

结果:

```text
Ran 38 tests in 3.981s
OK
```

单项导入测试也已单独通过:

```text
Stage2Original60QuantCoverageCliTest.test_original60_quant_state_coverage_imports_measured_fp32_smoke_latency
Stage2QuantAnchorSmokeCliTest.test_quant_anchor_smoke_imports_measured_fp32_smoke_latency
```

敏感信息检查:

已对本轮相关脚本、generated 输出和 6/27 交接文档做 H800 登录口令精确边界匹配检查, 结果为空。检查命令不写入交接文档, 避免文档自身包含敏感字符串。

## 8. 下一步计划: true-FP16 + INT8 小规模完整数据优先

### 8.0 FP16 provenance 风险与本轮 true-FP16 smoke

复查发现当前 summary 中的历史 FP16-tagged row 存在 provenance 风险:

1. `base`、`s0_024`、`s1_048` 的历史 FP16-tagged 行来自 `existing_h800_tvm_fp16_seed` 或 normalized historical summary。
2. 这些历史 FP16-tagged row 的 `raw_artifact=null`, 且 `source_files` 指向历史 summary 或 FP32 backbone ONNX, 不是明确的 `*_fp16.onnx` / true-FP16 Relax artifact。
3. 因此不能再把历史 FP16-tagged latency 与 FP32 smoke 的接近解释成“FP16 没收益”; 更合理的判断是历史 FP16-tagged row 可能把 FP32 backbone 当作 FP16 policy row 测量或归档。
4. 从下一阶段开始, 交接和 summary 审阅中不应把历史 `s0_024` / `s1_048` FP16-tagged row 称为 FP16 实测数据; 只能称为 suspect historical FP16-tagged rows。它们很可能本质上是 FP32 backbone 的测量结果, 需要走 FP32 remap/audit, 不能进入 FP16 canonical measured 口径。

为排查该问题, 本轮新增 true-FP16 smoke helper:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_true_smoke/stage2_h800_fp16_true_smoke.py
```

该 helper 对 `base`、`s0_024`、`s1_048` 生成真实 FP16 ONNX:

1. 输入 dtype 改为 `float16`。
2. FLOAT initializers 转为 `float16`。
3. 写出 layer precision summary。
4. 使用 H800 TVM / TVM VM 实测 latency。
5. 结果写入标准 row 文件, 但暂不覆盖现有 original60/quant anchor summary。

输出:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_latency_smoke_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_true_smoke/fp16_true_smoke_latest.json
```

本轮新生成 true-FP16 ONNX 的 smoke 结果:

| label | width | historical suspect FP16-tagged ms, not FP16 evidence | new true-FP16 default ms | new true-FP16 tuned ms | evidence |
|---|---|---:|---:|---:|---|
| `base` | 64x128x256 | 6.319980 | 53.272280 | 53.277002 | input + initializer dtype = float16 |
| `s0_024` | 24x128x256 | 39.503066 | 37.735195 | 37.734585 | input + initializer dtype = float16 |
| `s1_048` | 64x48x256 | 44.452754 | 42.324824 | 42.327026 | input + initializer dtype = float16 |

解释:

1. `s0_024` 和 `s1_048` 的 true-FP16 与 suspect historical FP16-tagged 数值接近, 但这不能证明历史行是 FP16; 只能说明三点需要用 true-FP16 artifact 重新建立 canonical evidence。
2. `base` 的 true-FP16 约 53ms, 明显不同于 suspect historical FP16-tagged 约 6.3ms, 说明历史 `base` FP16-tagged row 不能作为 true-FP16 artifact evidence。
3. true-FP16 tuned 与 default 接近, 说明现有 MetaSchedule DB 可能没有命中 FP16 workload, 或 dtype 改变后无法复用原 tuned schedule; 下一阶段如果要优化 true-FP16, 需要明确 FP16 workload tune DB。

### 8.1 当前总表是否已更新

已更新, 但要区分两个总表:

| 总表 | FP32 smoke 导入状态 | 说明 |
|---|---|---|
| `original60_quant_three_metric_summary_latest.*` | 已导入 `s0_024`、`s1_048` | `base` 不属于 original60 60 点, 所以不会出现在 original60 总表 |
| `quant_three_metric_summary_latest.*` | 已导入 `base`、`s0_024`、`s1_048` | 三条 FP32 smoke 都在 quant anchor smoke 总表中 |

FP32 original60 剩余 58 个 gap 不应立即进入大规模重测。复查 `coverage_pipeline_v1/rows/latency_lut_rows_original60_v1.jsonl` 后发现, 58 个历史 FP16-tagged original60 latency 原始测量的 `source_files` 均指向 `*_backbone.onnx`, 没有 true-FP16 artifact evidence。因此这些历史 FP16-tagged latency 很可能实际就是 FP32 ONNX backbone 的 H800 TVM 测量结果。下一步应先做 provenance remap/audit, 判断能否将这 58 个 suspect FP16-tagged row 重新归档为 FP32 measured evidence, 而不是重复跑 58 点。

### 8.2 下一阶段目标

最终验收目标:

| 类别 | 目标 |
|---|---:|
| true-FP16 small-smoke complete rows | 3 |
| INT8 small-smoke complete rows | 3 |

硬约束:

1. latency 对外一律使用 `latency_ms`。
2. H800 + TVM / TVM VM 是唯一 measured backend; 不得用 TRT 替代。
3. 所有新增 measured row 必须 `full_network_claim=false`。
4. latency job 启动前必须通过 GPU idle gate。
5. 若原始 runner 输出 `latency_p50_us`, 写入总表前必须除以 1000 并进入 `latency_ms`。

### 8.3 FP32 original60 provenance remap/audit, 非重测计划

执行顺序:

1. 读取 `coverage_pipeline_v1/rows/latency_lut_rows_original60_v1.jsonl`, 抽取 58 个 measured original60 latency label 的 default/tuned row。
2. 对每个 row 检查 `source_files` 中的 ONNX 是否为 `*_backbone.onnx`, 且无 `*_fp16.onnx` / true-FP16 layer precision summary。
3. 检查 `software_point_id`、`candidate_id` 是否只是带有 `fp16` 标签, 而实际 ONNX artifact 是 FP32 backbone。
4. 生成 `fp32_original60_remap_audit_latest.md/.json`, 将每个 label 标记为:
   - `remap_to_fp32_candidate`: ONNX 为 FP32 backbone, 可作为 FP32 measured evidence 候选;
   - `needs_remeasure`: source artifact 不清楚或 raw 缺失;
   - `quarantine`: raw/result 不一致或有 CUDA error。
5. 对 `remap_to_fp32_candidate` 行, 不重新跑 H800, 而是生成独立的 FP32 remapped latency rows, `measurement_source=historical_true_measurement_reclassified`, `quality_gate_status=fp32_reclassified_from_suspect_fp16_tagged_row`, 并保留原始 run_id/source_files/raw_artifact。
6. 将 `s0_024` 和 `s1_048` 本轮 FP32 smoke 与 remapped rows 做一致性交叉检查; 如果差异在合理范围内, 可作为 remap sanity evidence。
7. 重新运行 `stage2_generate_original60_quant_state_coverage.py --fp32-latency-smoke-rows ...` 或新增 `--fp32-latency-remap-rows`, 让 original60 FP32 summary 达到 60 行 measured。
8. 若 remap audit 发现 source artifact 不能证明 FP32, 再只对失败 label 启动小规模补测, 不启动 58 点全量重测。

建议优先输出:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_original60_remap_audit_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_original60_remap_audit_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp32_latency_original60_remapped_rows_v1.jsonl
```

### 8.4 FP16/INT8 小规模完整数据计划

目标 label:

```text
base, s0_024, s1_048
```

true-FP16 smoke 当前状态:

说明: 下表只指本轮新生成的 `*_backbone_true_fp16.onnx` smoke, 不包括历史 FP16-tagged row。历史 `s0_024` / `s1_048` FP16-tagged row 仍按 suspect / likely-FP32-remap 处理。

| label | true-FP16 status | 说明 |
|---|---|---|
| `base` | new_true_fp16_measured | true-FP16 ONNX + TVM VM latency row 已生成 |
| `s0_024` | new_true_fp16_measured | true-FP16 ONNX + TVM VM latency row 已生成; 历史 FP16-tagged row 不作为 FP16 证据 |
| `s1_048` | new_true_fp16_measured | true-FP16 ONNX + TVM VM latency row 已生成; 历史 FP16-tagged row 不作为 FP16 证据 |

当前状态:

| label | INT8 route status |
|---|---|
| `base` | TVM VM artifact ready |
| `s0_024` | TVM VM artifact ready |
| `s1_048` | 需要先补 INT8 QDQ + TVM VM artifact 或明确 blocker |

执行顺序:

1. 复用本轮成功环境: pip CUDA runtime 优先, PATH 包含 `/usr/local/cuda-12.2/bin`。
2. 将 `fp16_true_latency_smoke_rows_v1.jsonl` 中每个 label 选定一个 canonical schedule row, 建议先用 `metaschedule_tuned`; 若确认 tuned 未命中 FP16 workload, 可标注 `fp16_tuned_db_miss` 并用 default 作为 canonical。
3. 对 `s1_048` 先执行 INT8 route helper, 生成 calibration manifest、quant recipe、layer precision summary、QDQ ONNX、TVM VM `.so` 和 digest。
4. 刷新 `tvm_int8_artifact_registry_v1.jsonl`, 目标至少 `base`、`s0_024`、`s1_048` 三个 label artifact ready; 若 `s1_048` 不可实现, 写 quarantine/blocker。
5. 启动 3 条 INT8 TVM VM latency smoke, 每条都通过 GPU idle gate。
6. 只写 backbone/subnet 实际 scope, 不写 full-network claim。
7. 输出 3 行 INT8 measured latency smoke, 单位为 `latency_ms`。
8. 刷新 quant anchor smoke summary; 如要进入 original60 summary, 只导入 original60 内存在的 label, 即 `s0_024` 和 `s1_048`; `base` 继续留在 quant anchor smoke summary。

建议优先输出:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/int8_latency_smoke_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_latency_smoke/
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/tvm_int8_artifact_status_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/quant_smoke_20260627/exports/quant_three_metric_summary_latest.json
```

### 8.5 验收与验证

必须检查:

1. 本轮新 true-FP16 small-smoke 有 `base`、`s0_024`、`s1_048` 三个 label 的 canonical measured row, 且 layer summary 能证明 input/initializer dtype 为 `float16`; 历史 FP16-tagged row 不满足该验收项。
2. INT8 small-smoke 有 `base`、`s0_024`、`s1_048` 三个 label 的 canonical measured row, 或每个失败都有 blocker。
3. 所有新增 latency 对外字段为 `latency_ms`, 不用 `us` 做总表字段。
4. 所有新增 measured H800 TVM row 不含 TRT reference 字段。
5. 所有新增 row `full_network_claim=false`。
6. 运行:

```bash
python3 -m unittest framework.tests.test_stage2_lut_productization framework.tests.test_stage2_energy_coverage_jobs framework.tests.test_stage2_artifact_task_planner
```

## 9. 下一次 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 LUT 量化维度工作。先阅读 multi_agent/methods/design/auto-tuning/progress/6_27/0_6_27_冷启动入口_阶段二LUT.md, 再阅读 multi_agent/methods/design/auto-tuning/progress/6_27/5_6_27_交接文档_阶段二_FP32Smoke与INT8Route.md 和 multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md。当前主路径仍是 H800 + TVM / TVM VM, 不是 TRT; 不得声明全网络量化, 新行默认 full_network_claim=false。latency 指标在总表、交接文档和验收中统一使用 ms, 对外字段必须是 latency_ms; 如果 runner 原始输出 latency_p50_us, 写入总表前必须转换为 ms。当前历史 FP16-tagged row 存在 provenance 风险, 可能把 FP32 backbone 当作 FP16 policy row 测量或归档; 不得继续用历史 FP16-tagged row 作为 true-FP16 证据。先不要重跑 original60 的 58 个 FP32 gap; 先审计 coverage_pipeline_v1/rows/latency_lut_rows_original60_v1.jsonl, 因为这些 suspect FP16-tagged rows 的 source_files 指向 *_backbone.onnx, 很可能已经是 FP32 ONNX backbone 的 H800 TVM 实测。目标: 生成 fp32_original60_remap_audit_latest.md/.json, 将 58 个 suspect rows 分类为 remap_to_fp32_candidate / needs_remeasure / quarantine; 对可 remap 的行生成 fp32_latency_original60_remapped_rows_v1.jsonl, 保留原始 run_id/source_files/raw_artifact, 并刷新 original60 FP32 summary。并拿到 base、s0_024、s1_048 三个 label 的 true-FP16 完整数据和三个 label 的 INT8 完整数据。true-FP16 已有初始 smoke 输出 multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_latency_smoke_rows_v1.jsonl, 需要审计 layer_precision_summary 证明 input/initializer dtype=float16, 选定 canonical schedule row, 并刷新审阅表/summary。INT8 以 base、s0_024、s1_048 为目标, 复用 H800 TVM INT8 route; base 和 s0_024 使用已有 TVM VM artifact, s1_048 如缺 artifact 则先生成 calibration manifest、quant recipe、layer_precision_summary、QDQ ONNX、TVM VM artifact 和 digest, 再启动 3 条 INT8 TVM VM latency smoke。FP16/INT8 smoke 都只能写真实 backbone/subnet scope, full_network_claim=false, 不得写 AP/energy measured claim。若任一 FP16/INT8/remap job 失败, 不得终止整批任务; 写 job_state/quarantine/gap reason 后继续后续 label。stop condition: FP32 original60 remap audit 完成且可 remap 行进入 FP32 latency_ms measured summary; true-FP16 small-smoke 确认 3 行 latency_ms canonical measured 数据; INT8 small-smoke 产出 3 行 latency_ms canonical measured 数据或明确每个失败 blocker; tvm_int8_artifact_registry_v1.jsonl 更新 artifact_path/artifact_digest/build_status/validation_status; 新三指标 summary/审阅 MD 和 gap report 刷新; 回归测试通过; 写新的中文交接文档。
```
