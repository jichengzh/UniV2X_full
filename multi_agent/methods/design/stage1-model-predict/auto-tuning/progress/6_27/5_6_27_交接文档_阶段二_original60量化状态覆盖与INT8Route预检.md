# 4_6_27_交接文档_阶段二_original60量化状态覆盖与INT8Route预检

日期: 2026-06-27

序号: 4

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/0_6_27_冷启动入口_阶段二LUT.md`
- `multi_agent/methods/design/auto-tuning/progress/6_27/3_6_27_交接文档_阶段二_TVM量化字段与QuantAnchorSmoke结果.md`
- `multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md`

## 1. 本轮硬约束

1. 当前 Stage2 LUT 主路径仍是 **H800 + TVM / TVM VM**, 不是 TRT。
2. 当前只允许 `backbone_only`、`backbone_subnet` 或真实 `subnet` scope 的 claim; 不得声明全网络量化。
3. 所有新 row 默认 `full_network_claim=false`。
4. FP32/INT8 没有合法 H800 TVM measured path 时只能写 `no_claim`、`missing` 或 `quarantine`, 不得用 TRT 替代。
5. H800 密码没有写入任何 repo 文件、Markdown、JSON/JSONL、脚本或日志。

## 2. 本轮完成项

### 2.1 original60 quant state coverage

新增脚本:

```text
scripts/stage2_generate_original60_quant_state_coverage.py
```

输出目录:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/
```

核心输出:

| 文件 | 行数/状态 | 说明 |
|---|---:|---|
| `plans/original60_quant_state_plan_v1.json` | 540 jobs | 60 candidates x 3 precision x 3 axis 的状态计划 |
| `rows/latency_original60_quant_rows_v1.jsonl` | 180 | latency state rows |
| `rows/energy_original60_quant_rows_v1.jsonl` | 180 | energy state rows |
| `rows/ap_original60_quant_rows_v1.jsonl` | 180 | AP state rows |
| `quarantine/original60_quant_unclaimable_v1.jsonl` | 421 | no-claim/gap 追踪 |
| `exports/original60_quant_three_metric_summary_latest.md` | 180 rows | 人读三指标状态表 |
| `exports/original60_quant_three_metric_summary_latest.csv` | 180 rows | CSV 三指标状态表 |
| `exports/original60_quant_three_metric_summary_latest.json` | 180 rows | 机器读 summary |
| `exports/original60_quant_gap_report_latest.json` | 9 类 gap | gap 统计 |

### 2.2 TVM INT8 artifact route status

新增脚本:

```text
scripts/stage2_probe_tvm_int8_artifact_route.py
```

输出:

| 文件 | 行数 | 说明 |
|---|---:|---|
| `artifacts/tvm_int8_artifact_registry_v1.jsonl` | 2 | `base`、`s0_024` INT8 artifact status |
| `exports/tvm_int8_artifact_status_latest.json` | 2 | 机器读 artifact status |
| `exports/tvm_int8_artifact_status_latest.csv` | 2 | CSV artifact status |
| `exports/tvm_int8_artifact_status_latest.md` | 2 | 人读 artifact status |

当前状态:

| label | artifact_status | build_status | validation_status | blocker |
|---|---|---|---|---|
| `base` | `quarantine` | `probe_failed` | `not_run_probe_failed` | H800 SSH rate limit, 未能读取远端 artifact |
| `s0_024` | `quarantine` | `probe_failed` | `not_run_probe_failed` | H800 SSH rate limit, 未能读取远端 artifact |

注意: 本轮没有产出 calibration manifest、quant recipe、layer precision summary、quantized model export 或 TVM compiled artifact。因此没有任何 INT8 measured claim。

### 2.3 FP32 H800 TVM latency smoke preflight

新增脚本:

```text
scripts/stage2_record_fp32_latency_smoke_preflight.py
```

输出:

| 文件 | 行数 | 说明 |
|---|---:|---|
| `exports/fp32_latency_smoke_preflight_latest.json` | 3 | `base`、`s0_024`、`s1_048` preflight 状态 |
| `exports/fp32_latency_smoke_preflight_latest.csv` | 3 | CSV preflight 状态 |
| `exports/fp32_latency_smoke_preflight_latest.md` | 3 | 人读 preflight 状态 |
| `raw/h800_probe/h800_readonly_probe_latest.json` | 1 | H800 SSH probe 原始状态 |
| `raw/h800_probe/h800_preflight_stdout_latest.txt` | 1 | H800 probe stdout |
| `raw/h800_probe/h800_preflight_stderr_latest.txt` | 1 | H800 probe stderr |

当前状态:

| label | preflight_status | gpu_idle_gate_status | measurement_status | blocker |
|---|---|---|---|---|
| `base` | `ssh_failed_preflight_blocked` | `not_reached` | `not_launched` | `kex_exchange_identification: Connection closed by remote host` |
| `s0_024` | `ssh_failed_preflight_blocked` | `not_reached` | `not_launched` | `kex_exchange_identification: Connection closed by remote host` |
| `s1_048` | `ssh_failed_preflight_blocked` | `not_reached` | `not_launched` | `kex_exchange_identification: Connection closed by remote host` |

没有启动 FP32 latency measured job。

## 3. original60 quant coverage 总表统计

来自:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.json
```

| 项 | 值 |
|---|---:|
| candidate_count | 60 |
| total_cells | 180 |
| fp16 cells | 60 |
| fp32 cells | 60 |
| int8 cells | 60 |

三轴状态:

| axis | measured | no_claim |
|---|---:|---:|
| latency | 58 | 122 |
| energy | 56 | 124 |
| AP | 5 | 175 |

解释:

1. FP16 latency 继承 original60 已有 H800 TVM measured rows, 当前 58/60。
2. FP16 energy 继承 original60 energy telemetry, 当前 56/60。
3. FP16 AP 只保留已有 `claimable_*` 且 AP70 存在的真实/transfer source, 当前 5/60。
4. FP32 全部 `no_claim`, 原因是 H800 TVM FP32 latency smoke 尚未通过 preflight。
5. INT8 全部 `no_claim`, 原因是 TVM INT8 backbone/subnet route、calibration manifest、fallback policy、layer precision summary 和 artifact 仍未就绪。

## 4. gap 统计

来自:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_gap_report_latest.json
```

| failure reason | count |
|---|---:|
| `ap_fp16_true_source_missing` | 55 |
| `ap_fp32_true_eval_not_available` | 60 |
| `tvm_int8_ap_eval_backend_missing` | 60 |
| `energy_fp32_latency_measurement_not_run` | 60 |
| `fp16_energy_missing_or_quarantined` | 4 |
| `tvm_int8_energy_backend_missing` | 60 |
| `fp16_latency_missing_or_quarantined` | 2 |
| `tvm_fp32_backbone_measurement_not_run` | 60 |
| `tvm_int8_backbone_subnet_not_ready` | 60 |

## 5. H800 状态

本轮尝试了只读 H800 probe, 不启动编译、不启动测量。

结果:

```text
kex_exchange_identification: Connection closed by remote host
Connection closed by <PRIVATE_HOST> port 30001
```

判断:

1. 这符合 H800 runbook 中的 SSH rate limit / MaxStartups 问题。
2. 因 SSH 连接未建立, GPU idle gate 没有实际到达。
3. 因 GPU idle gate 未到达, 不允许写 FP32 latency measured row。
4. 因远端 artifact 未读到, `base`、`s0_024` 的 TVM INT8 artifact route 当前只能写 `quarantine`。

## 6. 验证

已运行:

```bash
python3 -m unittest framework.tests.test_stage2_lut_productization framework.tests.test_stage2_energy_coverage_jobs framework.tests.test_stage2_artifact_task_planner
```

结果:

```text
Ran 36 tests
OK
```

额外审查:

1. `latency_original60_quant_rows_v1.jsonl`、`energy_original60_quant_rows_v1.jsonl`、`ap_original60_quant_rows_v1.jsonl` 均为 180 行。
2. 三轴 540 行均通过 `validate_lut_row`。
3. 所有 row `full_network_claim=false`。
4. measured H800 TVM latency/energy 行中未发现 TRT 字段。
5. INT8 行无 measured claim。
6. 新增脚本、测试、产物和 6/27 文档中未发现明文 H800 密码。

## 7. 是否进入下一阶段

### 7.1 original60 quant state coverage

可以视为完成。当前已具备作为性能预测器基础状态表的 60 x 3 precision 覆盖, 但其中 FP32/INT8 大多是 no-claim/gap 状态, 不能当 measured data。

### 7.2 FP32 measured smoke

当前不能进入 measured smoke, 原因是 H800 SSH preflight 被 rate limit 阻塞, GPU idle gate 没有到达。

恢复条件:

1. H800 SSH 能稳定连接。
2. `base`、`s0_024`、`s1_048` 目标 GPU 通过 idle gate。
3. FP32 TVM runtime 命令明确不依赖 TRT。
4. 只启动这 3 个点的 latency smoke, 成功后再考虑 energy follower。

### 7.3 TVM INT8 route 实现

当前不能进入 INT8 measured route claim, 但可以进入 INT8 route 实现/排障阶段。

下一步需要补齐:

1. `base` 和 `s0_024` 的 TVM INT8 calibration manifest。
2. quant recipe。
3. layer precision summary。
4. quantized model export 或 TVM compiled artifact。
5. artifact digest。
6. validation status。

这些产物就绪前, INT8 继续保持 `artifact_status=quarantine/missing`, 不写 measured LUT。

## 8. 下一次 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 LUT 量化维度工作。先阅读 multi_agent/methods/design/auto-tuning/progress/6_27/0_6_27_冷启动入口_阶段二LUT.md, 再阅读 multi_agent/methods/design/auto-tuning/progress/6_27/4_6_27_交接文档_阶段二_original60量化状态覆盖与INT8Route预检.md 和 multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md。当前主路径仍是 H800 + TVM / TVM VM, 不是 TRT; 不得声明全网络量化, 新行默认 full_network_claim=false。目标: 等 H800 SSH rate limit 恢复后, 先只读重跑 H800 probe, 确认 GPU idle gate 和 base/s0_024/s1_048 artifact 路径; 若 GPU 空闲, 只启动 base/s0_024/s1_048 的 FP32 H800 TVM latency smoke, 不启动大规模队列。并继续为 base/s0_024 打通 TVM INT8 backbone/subnet route: 输出 calibration manifest、quant recipe、layer_precision_summary、quantized model export 或 TVM compiled artifact 与 artifact digest; 若任一步失败, 写 artifact_status=missing/quarantine 和 blocker, 不写 measured claim。stop condition: H800 probe 成功或明确继续 SSH blocked; FP32 smoke 有 measured rows 或 preflight_blocked rows; INT8 artifact registry 更新 artifact_path/artifact_digest/build_status/validation_status; 新三指标和 gap report 刷新; 写新的中文交接文档。
```
