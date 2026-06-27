# Stage2 三臂 readiness goal closure audit v1

日期：2026-06-26  
审计对象：

```text
完成 Stage2 三臂大规模生产前收敛：
- 修复 energy 生成链路
- 修复 artifact registry 链路
- 修复 outlier 处理
- 完成 energy runner/salvage 产品化
- 完成 artifact_registry_v1 schema + validator + backfill
- 完成 latency outlier detector + retest gate
- 通过 allowlist dry-run 给出是否 GO large-scale P/Q/S production 的判定
```

## 1. 需求到证据映射

| 要求 | 当前结论 | 权威证据 |
|---|---|---|
| 阅读并继承 readiness plan | 完成 | `HANDOFF_stage2_three_arm_readiness_execution_v1_zh.md` 记录 plan 要求、执行结果和下一步 |
| 阅读并继承 dataset-v2 aligned LUT plan | 完成 | 同一交接文档沿用 GPU idle、generate_*_lut 主路径、分层启动规则 |
| energy 生成链路修复 | 完成机制修复 | `scripts/stage2_generate_energy_lut.py` 写 `measurement_run_id` + `row_source=direct_generation` |
| energy salvage 产品化 | 完成 | `scripts/stage2_salvage_energy_payloads.py`，真实 payload salvage 8 rows |
| energy runner 自动恢复 | 完成 | `scripts/stage2_lut_worker.py` 支持 `resource.energy_salvage`，测试覆盖失败后自动 salvage 到 succeeded |
| artifact registry schema | 完成 | `framework/stage2/artifact_registry.py` |
| artifact registry validator | 完成 | `validate_artifact_registry_row` + `scripts/stage2_validate_artifact_registry.py` |
| artifact registry backfill | 完成 | `scripts/stage2_build_artifact_registry.py` 输出 `artifact_registry_v1.jsonl` |
| artifact registry 当前覆盖 | 完成 | `artifact_registry_summary_v1.json`: total=41, ready=41, missing=0 |
| outlier detector | 完成 | `framework/stage2/outlier_policy.py` + `scripts/stage2_detect_latency_outliers.py` |
| outlier retest/promotion gate | 完成机制修复 | `outlier_report_latest.json` 标记 trap25 unstable；`stage2_readiness_gate.py` 对 outlier 输出 conditional |
| quick review export | 完成 | `scripts/stage2_export_quick_review.py` 输出 `quick_review_latest.csv/json` |
| evidence registry update | 完成 | `evidence_registry_latest.json` 已更新并通过 readiness gate |
| allowlist dry-run | 完成 | `three_arm_allowlist_job_plan_v1.jsonl` + `three_arm_allowlist_readiness_gate_latest.json` |
| GO/NO-GO 判定 | 完成 | full P/Q/S 判定 `NO_GO`，P/S subset 判定 `CONDITIONAL_GO` |

## 2. 当前 gate 输出

### 当前 rows readiness

```text
multi_agent/data/stage2_lut_generation_v1/exports/readiness_gate_latest.json
decision=CONDITIONAL_GO
```

通过：

```text
schema=pass
artifact_registry=pass
energy=pass
evidence_registry=pass
```

条件项：

```text
outlier=conditional
unstable_rows=1
```

### 严格 P/Q/S 三臂 allowlist dry-run

```text
multi_agent/data/stage2_lut_generation_v1/exports/three_arm_allowlist_readiness_gate_latest.json
decision=NO_GO
```

NO_GO 不是机制失败，而是 artifact registry 正确阻断 Q arm：

```text
latency:allowlist_q_base_int8_metaschedule_tuned
ap:allowlist_q_base_int8_metaschedule_tuned
energy:allowlist_q_base_int8_metaschedule_tuned
latency:allowlist_q_p50_int8_metaschedule_tuned
ap:allowlist_q_p50_int8_metaschedule_tuned
energy:allowlist_q_p50_int8_metaschedule_tuned
```

### P/S ready subset allowlist dry-run

```text
multi_agent/data/stage2_lut_generation_v1/exports/ps_ready_subset_readiness_gate_latest.json
decision=CONDITIONAL_GO
```

该 subset 的 artifact/evidence/energy/schema gates 均通过，只受 trap25 outlier conditional gate 影响。

## 3. 不能标记为 GO 的原因

当前不能启动 full unattended P/Q/S large-scale LUT production，原因是数据与 artifact 缺口，不是 P0 机制缺口：

1. Q arm 至少 base INT8 / p50 INT8 artifact 尚未进入 artifact registry ready。
2. trap25 P2 repeat outlier 仍需真实 GPU 空闲 retest 或 quarantine 决策。
3. trap25/mix_a/iso_s1/iso_s2 仍缺 claimable energy row。
4. 真实 worker 6-12 jobs 长跑还未在 GPU idle 窗口执行。

## 4. 后续进入 GO 的必要条件

1. 生成或定位 Q arm ONNX + TVM work dir + MetaSchedule DB，并写入 artifact registry。
2. GPU idle 下完成 trap25 retest，更新 outlier report 和 quick review。
3. 补齐 energy repeats 和缺口 energy rows。
4. 执行真实 6-12 jobs worker allowlist 长跑。
5. 重新生成并确认：
   ```text
   artifact_registry_summary_v1.json
   evidence_registry_latest.json
   outlier_report_latest.json
   quick_review_latest.csv
   three_arm_allowlist_readiness_gate_latest.json
   ```

## 5. 本目标完成边界

本目标要求的是“修复机制并通过 allowlist dry-run 给出是否 GO 判定”。该边界已经完成，判定为：

```text
Stage2 full P/Q/S large-scale production: NO_GO
P/S ready subset allowlist: CONDITIONAL_GO
```

后续真实补点和把 `NO_GO` 推进到 `GO` 是下一阶段实验目标，不属于本轮机制收敛目标的完成条件。
