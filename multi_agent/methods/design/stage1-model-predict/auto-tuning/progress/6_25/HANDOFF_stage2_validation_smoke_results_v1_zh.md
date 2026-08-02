# HANDOFF: Stage2 H800 TVM validation replay 与 smoke 结果 v1

日期: 2026-06-25

## 1. 当前结论

Validation replay 已通过。Pyramid base 已从近期 default-like `56 ms` 链路回到历史 tuned 量级, 本次 validation 为 `6.213 ms`; corrected pad64 也已确认使用正确来源 `trap25_pad64_backbone.onnx + ms_work_2e_pad64_retest`, replay 为 `6.154 ms`。

Smoke 已完成基础 gate, 并追加 AP supplement:

- latency: 4/4 measured pass
- AP: 6 条 measured/imported true AP anchors pass
- energy: 2/2 H800 telemetry smoke pass

Calibration latency small 已完成并保留为有效数据:

- 9 个 FP16 width x 2 schedules = 18 条 measured latency rows
- default 与 MetaSchedule tuned 均来自同一批 preflight-clean H800 raw artifacts

Calibration latency medium 已启动并跑完, 但不作为 measured claim:

- 9 个 FP16 width x 2 schedules = 18 条 row 已隔离到 quarantine
- 原因: medium preflight 记录到外部 `serve_qwen_ra_grpo.py` 进程 `pid=3260506` 占用 GPUs 0-7, 违反 target GPU idle 规则

当前判定:

- `GO_LATENCY_CALIBRATION_SMALL = true`
- `GO_FULL_LUT = false`
- `GO_MEDIUM_LATENCY_CLAIM = false`

原因: AP gate 已补充到更多真实 DAIR-1789 anchors, 但 energy pad64 supplement 被外部 GPU 进程阻断; latency medium 也因同一外部进程污染而需要重跑。当前只能继续使用 validation/smoke 与 calibration small 的有效 rows, 不能启动三轴大规模长期 worker。

## 2. Validation replay 结果

主产物:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/validation/h800_tvm_replay_v1/
```

关键文件:

```text
compare/h800_tvm_replay_summary_v1.json
compare/h800_tvm_replay_compare_v1.csv
compare/pad64_combo_debug_matrix_v1.csv
latency/latency_lut_rows_v1.jsonl
jobs/replay_job_state_v1.jsonl
raw/
```

Validation gate:

| model | label | old tuned ms | new tuned ms | gate | status |
|---|---:|---:|---:|---|---|
| Pyramid | base | 6.320 | 6.213 | 5.5-7.5 | pass |
| Pyramid | p50 | 3.011 | 3.003 | 2.6-3.6 | pass |
| Pyramid | trap25 | 21.615 | 21.601 | 18-25 | pass |
| Pyramid | pad64 | 6.152 | 6.154 | 5.4-7.2 | pass |
| CoDriving | base | 8.058 | 8.037 | 7.0-9.5 | pass |
| CoDriving | p50 | 1.609 | 1.500 | 1.3-2.0 | pass |

Pad64 纠错:

- 失败配对: `pad64_backbone_fixed.onnx + ms_work_2e_pad64_retest`, tuned `7.963 ms`, 不作为 corrected pad64 gate。
- 正确配对: `trap25_pad64_backbone.onnx + ms_work_2e_pad64_retest`, tuned `6.154 ms`。
- debug matrix: `compare/pad64_combo_debug_matrix_v1.csv`。

## 3. Smoke 结果

主产物:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/
```

快速审查:

```text
compare/smoke_summary_v1.json
compare/smoke_quick_review_v1.csv
compare/evidence_registry.smoke_updated.json
latency/latency_lut_rows_v1.jsonl
ap/ap_anchor_rows_v1.jsonl
energy/energy_lut_rows_v1.jsonl
jobs/job_plan.jsonl
jobs/job_state.jsonl
raw/
```

Latency smoke:

| model | label | latency ms | gate | status |
|---|---:|---:|---|---|
| Pyramid | base | 6.209 | 5.5-7.5 | pass |
| Pyramid | p50 | 3.006 | 2.6-3.6 | pass |
| Pyramid | pad64 | 6.160 | 5.4-7.2 | pass |
| CoDriving | base | 8.039 | 7.0-9.5 | pass |

AP smoke:

| model | label | AP70 | source | status |
|---|---:|---:|---|---|
| Pyramid | base | 0.630864 | `stage_a_ap_real.csv` true AP anchor | pass |
| Pyramid | p50 | 0.564132 | `stage_a_ap_real.csv` true AP anchor | pass |
| Pyramid | trap25 | 0.590472 | `stage_a_ap_real.csv` true AP anchor | pass |
| Pyramid | p75 | 0.529988 | `stage_a_ap_real.csv` true AP anchor | pass |
| Pyramid | mix_b | 0.636160 | `ap70_depgraph_expansion.json` true AP anchor | pass |
| Pyramid | mix_d | 0.636938 | `ap70_depgraph_expansion.json` true AP anchor | pass |

Energy smoke:

| model | label | J / inference | idle W | active W avg | status |
|---|---:|---:|---:|---:|---|
| Pyramid | base | 2.050199 | 72.668 | 401.955 | pass |
| Pyramid | p50 | 0.776318 | 72.910 | 330.688 | pass |

Registry smoke:

- `evidence_registry.smoke_updated.json` 可读取本次 canonical rows。
- coverage: latency `4/4`, AP `2/2`, energy `2/2` measured。

Smoke v2 supplement:

```text
compare/smoke_summary_v2.json
compare/ap_smoke_supplement_quick_review_v1.csv
compare/energy_smoke_supplement_gate_v1.json
compare/evidence_registry.smoke_v2_updated.json
```

- AP supplement 通过 `generate_ap_lut` 写入 canonical AP rows, 当前 AP row count 为 6。
- energy pad64 supplement 未运行 measured telemetry; `energy_smoke_supplement_gate_v1.json` 记录为 `BLOCKED_BY_FOREIGN_GPU_PROCESS`。
- registry v2 coverage: latency `4/4`, AP `6/6`, energy `2/2` measured。

## 4. Calibration small/medium 结果

主产物:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/
```

有效文件:

```text
latency/latency_lut_rows_v1.jsonl
compare/latency_calibration_progress_summary_v1.json
compare/latency_calibration_valid_quick_review_v1.csv
```

有效 calibration small rows:

- row count: 18
- width count: 9
- schedules: `default`, `metaschedule_tuned`
- corrected pad64: `trap25_pad64_backbone.onnx + ms_work_2e_pad64_retest`, tuned `6.164 ms`

Quarantine 文件:

```text
latency/latency_lut_rows_quarantine_medium_contaminated_20260625_194602.jsonl
compare/latency_calibration_medium_contamination_summary_v1.json
```

Quarantine 内容:

- row count: 18
- labels: `s0_16`, `s0_32`, `s1_32`, `s1_64`, `s2_64`, `mix_a`, `mix_b`, `mix_c`, `mix_d`
- 这些数值可用于诊断, 不能进入 measured claim 或 registry claim。

## 5. 已知问题与约束

1. CoDriving 第一次 smoke 使用通用 ONNX shape reader 失败, 因为 symbolic batch 被读成 0。已用 CoDriving 专用逻辑修复: batch 替换为 2, tuned apply 后接 dlight fallback。
2. S8 INT8 stage0 probe 本轮未跑, 在 job state 中标记为 skipped/deferred; 它不是三轴 smoke gate, 后续机制覆盖时补。
3. energy smoke 的 target GPU 满足 idle preflight, 但同节点 GPU0-2 有其他任务。该数据可用于链路验证和粗粒度 LUT smoke, 不作为 paper-grade energy。
4. AP 当前是已有真实 AP anchor / expansion eval 导入, 不是本轮 fresh rerun; 若要宣布 full tri-axis large LUT, 需要明确接受 cached-real AP eval import 作为正式产品化来源, 或等待 GPU/环境空闲重跑 fresh AP eval。
5. 2026-06-25 19:44 起 H800 上出现外部 `serve_qwen_ra_grpo.py` 进程 `pid=3260506`, 附着 GPUs 0-7。该进程存在期间不得启动 latency/energy measured job。

## 6. 下一步计划

P0:

- 等 `pid=3260506` 释放 GPUs 3-7 后, 重跑 latency medium; 重跑前必须保存 clean preflight, 目标 GPU 无 compute process 且 memory used 回到 idle baseline。
- 补跑 energy pad64 telemetry supplement; 只允许在目标 GPU 无外部进程、显存低于 idle 阈值时运行。
- 继续使用 corrected pad64 source: `trap25_pad64_backbone.onnx + ms_work_2e_pad64_retest`。

P1:

- 若接受 cached-real AP import 作为产品化 AP 来源, 将 `stage_a_ap_real.csv` 与 `ap70_depgraph_expansion.json` 纳入 AP generator 的正式 seed registry; 否则排队 fresh AP rerun。
- 补 energy paper-grade smoke: 尽量整机低负载, 固定 GPU/clock policy, base/p50/pad64 至少各 2 次重复。
- 跑 S8 INT8 stage0 probe, 只作为 scope-limited/mechanism row。

P2:

- 通过 clean medium replay 与 energy supplement gate 后再启动 full latency/AP/energy calibration worker。
- 更新 export repo 与 docs, 将 corrected pad64 配对写入长期 job template, 避免回退到 `pad64_backbone_fixed.onnx`。

## 7. 2026-06-25 21:20 增量进展: GPU0/1/2 释放后的重跑

### 7.1 latency calibration medium clean 重跑

用户释放 GPU0/1/2 后, 先对目标 GPU 做 preflight。实际状态中 GPU0 后续被其他用户进程占用, 因此本轮有效重跑主要使用 GPU1/GPU2。

新增有效 canonical latency rows:

```text
multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/latency/latency_lut_rows_v1.jsonl
```

当前 canonical latency 进度:

| 类别 | row 数 | 说明 |
|---|---:|---|
| calibration small | 18 | 9 个 config x default/tuned, preflight-clean |
| calibration medium clean | 16 | 8 个 medium config x default/tuned, preflight-clean |
| total valid canonical | 34 | schema validation 通过 |
| medium quarantine | 18 | 旧污染批次, 仍只作诊断 |

medium clean 已完成标签:

```text
s0_16, s0_32, s1_32, s2_64, mix_a, mix_b, mix_c, mix_d
```

关键实测值:

| label | default ms | tuned ms | 备注 |
|---|---:|---:|---|
| s0_16 | 39.468 | 6.898 | clean g012 |
| s0_32 | 44.634 | 5.968 | clean g012 |
| s1_32 | 41.887 | 3.761 | clean g012 |
| s2_64 | 43.568 | 5.033 | clean g012 |
| mix_a | 35.793 | 5.658 | clean g012 |
| mix_b | 41.446 | 19.390 | clean retry GPU2 |
| mix_c | 30.730 | 5.103 | clean g012 |
| mix_d | 42.409 | 21.044 | clean g012 |

`s1_64` 当前阻塞:

- 原始 clean g012 批次: preflight 被上一批 GPU 上下文释放延迟拒绝, 未采信。
- env-fixed clean retry on GPU1/GPU2: default 阶段可测, tuned VM 在 `dev.sync()` 触发 `CUDA illegal memory access`。
- 最新失败证据:

```text
multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/raw/calib_h800_tvm_pyramid_s1_64_fp16_medium_clean_retry_g1_20260625_210525/
```

因此 `s1_64` 不写 canonical measured row; 需要诊断 `ms_bumped_s1_64` MetaSchedule DB/kernel 稳定性, 或制定排除/重建 DB 策略。

### 7.2 energy smoke supplement

pad64 energy supplement 已在 GPU1 clean preflight 下重试, 但 tuned VM 运行失败:

```text
multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/raw/smoke_h800_tvm_energy_pyramid_pad64_fp16_tuned_20260625_211139/
```

失败原因:

- target GPU1 preflight clean。
- tuned TVM VM 触发 `CUDA_ERROR_ILLEGAL_ADDRESS`。
- 失败 kernel: `fused_conv2d15_add8_relu6_kernel`。
- 未写 pad64 energy measured row。

当前 energy canonical rows 仍为 2:

| label | J / inference | status |
|---|---:|---|
| base | 2.050199 | measured |
| p50 | 0.776318 | measured |

energy gate 已更新为:

```text
BASE_P50_PASS_PAD64_TUNED_KERNEL_FAILED
```

### 7.3 当前大规模启动判定

结论不是“完全不可启动”, 而是分层启动:

| 启动类型 | 判定 | 原因 |
|---|---|---|
| latency background 补点, 排除/隔离 `s1_64` 同类风险点 | 可以小规模启动 | 产品化链路已能生成 34 条 clean latency rows, 但需保留 preflight + per-job failure isolation |
| AP cached-real import/generator 扩展 | 可以启动 | AP smoke 已有 6 条 true AP anchors, 走 `generate_ap_lut` |
| energy base/p50 类似安全点 supplement | 可以小规模串行启动 | energy generator/validator 可用, 但只允许目标 GPU clean |
| full unattended tri-axis large LUT | 暂不建议 | `s1_64` latency tuned 与 pad64 energy tuned 都暴露 MetaSchedule kernel/DB failure, 需要先加 bad-DB quarantine / retry isolation / exclusion policy |

下一阶段 P0:

1. 给 latency worker 增加 per-config 进程隔离与 bad-DB quarantine: 单点 tuned illegal memory access 不得中断同批其它 job。
2. 对 `s1_64` 和 pad64 energy 相关 DB 做诊断: 重建 DB、换 known-good latency path、或将该类点标记为 blocked/excluded。
3. 若先启动后台补点, 初始 allowlist 应排除 `s1_64` 与 pad64 energy tuned, 并限制 energy 串行。
4. 所有 latency/energy measured job 继续执行 GPU idle 硬门槛: target GPU 无 compute process, util <= 5%, memory <= 1024 MiB。
