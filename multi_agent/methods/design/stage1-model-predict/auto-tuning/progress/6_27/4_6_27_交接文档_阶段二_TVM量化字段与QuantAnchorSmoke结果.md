# 3_6_27_交接文档_阶段二_TVM量化字段与QuantAnchorSmoke结果

日期: 2026-06-27

序号: 3

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/0_6_27_冷启动入口_阶段二LUT.md`
- `multi_agent/methods/design/auto-tuning/progress/6_27/2_6_27_交接文档_阶段二_量化维度全链路LUT计划.md`
- `multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md`

## 1. 本轮硬约束

1. 当前 Stage2 LUT 主加速路线是 **H800 + TVM / TVM VM**, 不是 TensorRT/TRT。
2. 当前可以先做 `backbone_only`、`backbone_subnet` 或真实 `subnet` scope 的量化和测量, 但这不是全网络量化。
3. 新行默认 `full_network_claim=false`; 不得把 TVM micro/stage0 probe 写成 full-backbone 或 full-network measured。
4. INT8 默认方法为 `h800_tvm_int8_backbone_subnet_experimental`; TRT 只能是 historical/reference evidence, 不能替代 H800 TVM measured LUT。

## 2. 本轮完成项

### 2.1 schema / validator / generator

已完成:

1. `framework/stage2/lut_productization.py`
   - 增加 `precision`, `quant_scheme`, `quant_method`, `quant_scope`, `calibrator`, `fallback_policy`, `layer_precision_summary`, `full_network_claim`, `engine_kind`, `measurement_source`, `claim_status`, `quality_gate_status`, `schedule_profile`, `tune_budget` 等字段默认值和校验。
   - `measurement_status` 增加 `no_claim` 和 `quarantine`。
   - measured H800 TVM latency/energy 行禁止使用 TRT reference 字段。
   - 当前默认拒绝 `full_network_claim=true`。
2. `scripts/stage2_generate_latency_lut.py`
3. `scripts/stage2_generate_energy_lut.py`
4. `scripts/stage2_generate_ap_lut.py`
   - 三个 generator 均支持显式传入 TVM-first quant 字段。
5. `scripts/stage2_plan_lut_jobs.py`
   - job plan generator 会把 quant 字段传入 `generate_*_lut` 主路径。
6. `scripts/stage2_h800_run_measurement_job.py`
   - H800 latency/energy runner 会把 quant 字段继续透传到 row generator。
7. `scripts/stage2_generate_quant_anchor_smoke.py`
   - 新增 quant anchor smoke 状态生成器。
   - 只导入已有 H800 TVM FP16 证据和 AP/energy summary; 不伪造 FP32/INT8 measured 数据。

### 2.2 生成的 quant smoke 目录

输出目录:

```text
multi_agent/data/stage2_lut_generation_v1/generated/quant_smoke_20260627/
```

文件:

| 文件 | 行数/状态 | 用途 |
|---|---:|---|
| `plans/quant_anchor_job_plan_v1.json` | 72 jobs | 8 anchors x 3 precision x 3 axis 的状态计划 |
| `rows/latency_quant_rows_v1.jsonl` | 24 | latency measured/no-claim rows |
| `rows/energy_quant_rows_v1.jsonl` | 24 | energy measured/no-claim rows |
| `rows/ap_quant_rows_v1.jsonl` | 24 | AP measured/no-claim rows |
| `jobs/latency_quant_job_state_v1.jsonl` | 24 | latency job state |
| `jobs/energy_quant_job_state_v1.jsonl` | 24 | energy job state |
| `jobs/ap_quant_job_state_v1.jsonl` | 24 | AP job state |
| `quarantine/quant_unclaimable_v1.jsonl` | 52 | no-claim/gap 追踪 |
| `exports/quant_three_metric_summary_latest.md` | 24 rows | 人读三指标状态表 |
| `exports/quant_three_metric_summary_latest.csv` | 24 rows | CSV 三指标状态表 |
| `exports/quant_three_metric_summary_latest.json` | 24 rows | 机器读 summary |
| `exports/quant_gap_report_latest.json` | 7 类 gap | gap 统计 |

## 3. 当前 quant anchor 三指标状态表

说明: 本表是状态覆盖表, 不是全部 measured claim。FP32/INT8 当前不可测项写 no-claim/gap。

| label | width | precision | quant_method | quant_scope | latency ms | latency | energy J | energy | AP70 | AP | failure reasons |
|---|---|---|---|---|---:|---|---:|---|---:|---|---|
| base | 64x128x256 | fp32 | `h800_tvm_relax_fp32` | `backbone_only` |  | no_claim |  | no_claim |  | no_claim | `tvm_fp32_backbone_measurement_not_run;energy_fp32_latency_measurement_not_run;ap_fp32_true_eval_not_available` |
| base | 64x128x256 | fp16 | `h800_tvm_relax_metaschedule_fp16` | `backbone_only` | 6.319980 | measured |  | no_claim | 0.630863623 | measured | `energy_fp16_anchor_not_found_in_current_summary` |
| base | 64x128x256 | int8 | `h800_tvm_int8_backbone_subnet_experimental` | `backbone_only_requested` |  | no_claim |  | no_claim |  | no_claim | `tvm_int8_backbone_subnet_not_ready;tvm_int8_energy_backend_missing;tvm_int8_ap_eval_backend_missing` |
| p50 | 32x64x128 | fp32 | `h800_tvm_relax_fp32` | `backbone_only` |  | no_claim |  | no_claim |  | no_claim | `tvm_fp32_backbone_measurement_not_run;energy_fp32_latency_measurement_not_run;ap_fp32_true_eval_not_available` |
| p50 | 32x64x128 | fp16 | `h800_tvm_relax_metaschedule_fp16` | `backbone_only` | 3.010510 | measured |  | no_claim | 0.564131504 | measured | `energy_fp16_anchor_not_found_in_current_summary` |
| p50 | 32x64x128 | int8 | `h800_tvm_int8_backbone_subnet_experimental` | `backbone_only_requested` |  | no_claim |  | no_claim |  | no_claim | `tvm_int8_backbone_subnet_not_ready;tvm_int8_energy_backend_missing;tvm_int8_ap_eval_backend_missing` |
| p75 | 16x32x64 | fp32 | `h800_tvm_relax_fp32` | `backbone_only` |  | no_claim |  | no_claim |  | no_claim | `tvm_fp32_backbone_measurement_not_run;energy_fp32_latency_measurement_not_run;ap_fp32_true_eval_not_available` |
| p75 | 16x32x64 | fp16 | `h800_tvm_relax_metaschedule_fp16` | `backbone_only` | 0.483630 | measured |  | no_claim | 0.529988343 | measured | `energy_fp16_anchor_not_found_in_current_summary` |
| p75 | 16x32x64 | int8 | `h800_tvm_int8_backbone_subnet_experimental` | `backbone_only_requested` |  | no_claim |  | no_claim |  | no_claim | `tvm_int8_backbone_subnet_not_ready;tvm_int8_energy_backend_missing;tvm_int8_ap_eval_backend_missing` |
| trap25 | 48x96x192 | fp32 | `h800_tvm_relax_fp32` | `backbone_only` |  | no_claim |  | no_claim |  | no_claim | `tvm_fp32_backbone_measurement_not_run;energy_fp32_latency_measurement_not_run;ap_fp32_true_eval_not_available` |
| trap25 | 48x96x192 | fp16 | `h800_tvm_relax_metaschedule_fp16` | `backbone_only` | 21.614800 | measured |  | no_claim | 0.590472382 | measured | `energy_fp16_anchor_not_found_in_current_summary` |
| trap25 | 48x96x192 | int8 | `h800_tvm_int8_backbone_subnet_experimental` | `backbone_only_requested` |  | no_claim |  | no_claim |  | no_claim | `tvm_int8_backbone_subnet_not_ready;tvm_int8_energy_backend_missing;tvm_int8_ap_eval_backend_missing` |
| s0_024 | 24x128x256 | fp32 | `h800_tvm_relax_fp32` | `backbone_only` |  | no_claim |  | no_claim |  | no_claim | `tvm_fp32_backbone_measurement_not_run;energy_fp32_latency_measurement_not_run;ap_fp32_true_eval_not_available` |
| s0_024 | 24x128x256 | fp16 | `h800_tvm_relax_metaschedule_fp16` | `backbone_only` | 39.503066 | measured | 4.954709 | measured | 0.632495206 | measured |  |
| s0_024 | 24x128x256 | int8 | `h800_tvm_int8_backbone_subnet_experimental` | `backbone_only_requested` |  | no_claim |  | no_claim |  | no_claim | `tvm_int8_backbone_subnet_not_ready;tvm_int8_energy_backend_missing;tvm_int8_ap_eval_backend_missing` |
| s0_040 | 40x128x256 | fp32 | `h800_tvm_relax_fp32` | `backbone_only` |  | no_claim |  | no_claim |  | no_claim | `tvm_fp32_backbone_measurement_not_run;energy_fp32_latency_measurement_not_run;ap_fp32_true_eval_not_available` |
| s0_040 | 40x128x256 | fp16 | `h800_tvm_relax_metaschedule_fp16` | `backbone_only` | 44.704673 | measured | 4.794238 | measured | 0.626659483 | measured |  |
| s0_040 | 40x128x256 | int8 | `h800_tvm_int8_backbone_subnet_experimental` | `backbone_only_requested` |  | no_claim |  | no_claim |  | no_claim | `tvm_int8_backbone_subnet_not_ready;tvm_int8_energy_backend_missing;tvm_int8_ap_eval_backend_missing` |
| s0_056 | 56x128x256 | fp32 | `h800_tvm_relax_fp32` | `backbone_only` |  | no_claim |  | no_claim |  | no_claim | `tvm_fp32_backbone_measurement_not_run;energy_fp32_latency_measurement_not_run;ap_fp32_true_eval_not_available` |
| s0_056 | 56x128x256 | fp16 | `h800_tvm_relax_metaschedule_fp16` | `backbone_only` | 51.342031 | measured | 6.196241 | measured | 0.623711042 | measured |  |
| s0_056 | 56x128x256 | int8 | `h800_tvm_int8_backbone_subnet_experimental` | `backbone_only_requested` |  | no_claim |  | no_claim |  | no_claim | `tvm_int8_backbone_subnet_not_ready;tvm_int8_energy_backend_missing;tvm_int8_ap_eval_backend_missing` |
| s1_048 | 64x48x256 | fp32 | `h800_tvm_relax_fp32` | `backbone_only` |  | no_claim |  | no_claim |  | no_claim | `tvm_fp32_backbone_measurement_not_run;energy_fp32_latency_measurement_not_run;ap_fp32_true_eval_not_available` |
| s1_048 | 64x48x256 | fp16 | `h800_tvm_relax_metaschedule_fp16` | `backbone_only` | 44.452754 | measured | 5.499634 | measured | 0.635616211 | measured |  |
| s1_048 | 64x48x256 | int8 | `h800_tvm_int8_backbone_subnet_experimental` | `backbone_only_requested` |  | no_claim |  | no_claim |  | no_claim | `tvm_int8_backbone_subnet_not_ready;tvm_int8_energy_backend_missing;tvm_int8_ap_eval_backend_missing` |

状态统计:

| axis | measured | no_claim |
|---|---:|---:|
| latency | 8 | 16 |
| energy | 4 | 20 |
| AP | 8 | 16 |

## 4. gap 统计

来自 `exports/quant_gap_report_latest.json`:

| failure reason | count | 含义 |
|---|---:|---|
| `tvm_fp32_backbone_measurement_not_run` | 8 | FP32 H800 TVM backbone latency 尚未实测 |
| `energy_fp32_latency_measurement_not_run` | 8 | FP32 energy 不能脱离 latency 单独 claim |
| `ap_fp32_true_eval_not_available` | 8 | FP32 AP true eval/import 尚未建立 |
| `tvm_int8_backbone_subnet_not_ready` | 8 | TVM INT8 backbone/subnet route 尚未打通 |
| `tvm_int8_energy_backend_missing` | 8 | INT8 energy 缺可用 TVM INT8 runtime |
| `tvm_int8_ap_eval_backend_missing` | 8 | INT8 AP 缺合法 TVM INT8 eval path |
| `energy_fp16_anchor_not_found_in_current_summary` | 4 | 历史 base/p50/p75/trap25 只有 latency/AP seed, 当前 summary 无 energy |

## 5. 验证结果

已运行:

```bash
python3 -m unittest framework.tests.test_stage2_lut_productization
python3 -m unittest framework.tests.test_stage2_energy_coverage_jobs
python3 -m unittest framework.tests.test_stage2_artifact_task_planner
python3 scripts/stage2_generate_quant_anchor_smoke.py
```

结果:

| 命令 | 结果 |
|---|---|
| `framework.tests.test_stage2_lut_productization` | 26 tests, OK |
| `framework.tests.test_stage2_energy_coverage_jobs` | 3 tests, OK |
| `framework.tests.test_stage2_artifact_task_planner` | 3 tests, OK |
| `stage2_generate_quant_anchor_smoke.py` | latency/AP/energy 各 24 rows |

额外审查:

1. 三轴 72 行均通过 `validate_lut_row`。
2. 所有新增行 `full_network_claim=false`。
3. measured H800 TVM latency/energy 行中未出现 TRT 字段。
4. INT8 行全部为 `no_claim`, `quant_method=h800_tvm_int8_backbone_subnet_experimental`, `engine_kind=tvm_vm`, `quant_scope=backbone_only_requested`。
5. 新增脚本、测试、quant smoke 产物和 6/27 文档中未写入明文 H800 密码。

## 6. 是否进入 original60 quant coverage

结论分两层:

1. **可以进入 original60 quant state coverage。**
   - 当前 schema、validator、job plan、runner 和 summary 已能承载 `fp32/fp16/int8`。
   - 可以为 original60 生成 60 x 3 precision 的状态表, FP16 继承现有 measured, FP32/INT8 暂时写 no-claim/gap。
2. **还不应进入 original60 FP32/INT8 measured 大规模 claim。**
   - FP32 H800 TVM backbone latency 还没有实际 smoke。
   - TVM INT8 backbone/subnet route、calibration manifest、fallback policy、layer precision inventory 尚未打通。
   - AP 的 FP32/INT8 true eval/import backend 尚未建立。
   - energy 必须跟随同一 runtime 的 latency; 不能脱离 latency 单独 claim。

本轮没有启动新的远端 H800 实测 job, 因此没有新的远端 raw 结果需要 rsync。quant smoke 使用的是本地已同步的 H800 TVM FP16 历史证据和 original60 summary。

## 7. 下一阶段计划

P0: original60 quant state coverage

1. 写 `stage2_generate_original60_quant_state_coverage.py`。
2. 输出 original60 60 x 3 precision 的 latency/AP/energy state rows 和 summary。
3. FP16 使用现有 original60 measured rows; FP32/INT8 缺测时写 no-claim/gap。
4. 不启动大规模 measured claim, 只扩充预测器可读的状态覆盖表。

P1: FP32 H800 TVM smoke

1. 选 `base`、`s0_024`、`s1_048` 三个点做 FP32 backbone latency smoke。
2. latency/energy 必须通过 GPU idle gate。
3. AP 只接受 true eval/import; 缺 source 写 no-claim。
4. 通过后再补 original60 的 FP32 latency/energy follower。

P2: TVM INT8 backbone/subnet route

1. 明确 TVM INT8 quant route: calibration manifest、input list、fallback policy。
2. 输出 TVM module/TIR/op inventory, 填 `layer_precision_summary`。
3. 先跑 `base` 和 `s0_024` INT8 latency smoke。
4. 若仍只能到 micro/stage0, 保持 `quant_scope` 为真实局部 scope, 不写 backbone/subnet measured。

P3: AP/energy 对齐

1. AP FP32/INT8 只写 true eval/import 或 no-claim。
2. energy 必须跟随同一 H800 TVM runtime 和同一 `(label, precision, schedule_profile)`。
3. 任何测不了的配置进入 quarantine/gap, 不停止队列。

## 8. 下一次 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 LUT 量化维度工作。先阅读 multi_agent/methods/design/auto-tuning/progress/6_27/0_6_27_冷启动入口_阶段二LUT.md, 再阅读 multi_agent/methods/design/auto-tuning/progress/6_27/3_6_27_交接文档_阶段二_TVM量化字段与QuantAnchorSmoke结果.md 和 multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md。当前主路径仍是 H800 + TVM / TVM VM, 不是 TRT; 当前允许 backbone_only/backbone_subnet/subnet scope, 但不得声明全网络量化, 新行默认 full_network_claim=false。目标: 先完成 original60 quant state coverage, 生成 original60 60 x 3 precision 的 latency/AP/energy 状态表和 summary。FP16 继承现有 H800 TVM measured rows; FP32/INT8 如果没有合法 H800 TVM measured path, 必须写 no-claim/gap/quarantine, 不得用 TRT 替代。随后只选择 base/s0_024/s1_048 做 FP32 H800 TVM latency smoke 预检; latency/energy 必须通过 GPU idle gate, AP 记录 GPU 状态即可。stop condition: original60 x fp32/fp16/int8 状态表完成并通过 validator; measured H800 TVM 行无 TRT 字段; 所有行 full_network_claim=false; 输出 original60_quant_three_metric_summary_latest.md/.csv/.json 和 original60_quant_gap_report_latest.json; 写新的中文交接文档说明是否进入 FP32 measured smoke 和 TVM INT8 route 实现。
```
