# Stage2 三臂大规模生产前收敛执行交接 v1

日期：2026-06-26  
范围：energy runner/salvage、artifact registry、latency outlier、quick review、allowlist dry-run gate。

## 1. 本轮结论

本轮已把三个机制缺口从“文档计划”推进到可运行脚本和测试覆盖：

| 机制 | 当前状态 | 证据 |
|---|---|---|
| energy salvage | 已产品化 | `stage2_salvage_energy_payloads.py` 可从 telemetry payload 自动恢复 canonical energy rows |
| worker energy resume | 已接入 | `stage2_lut_worker.py` 支持 `resource.energy_salvage`，energy job 失败但 payload 存在时可自动 salvage 到 succeeded |
| artifact registry | 已产品化 | `artifact_registry_v1.jsonl` 已回填，41/41 ready |
| evidence registry export | 已产品化 | `evidence_registry_latest.json` 已一键更新并通过 readiness gate |
| latency outlier detector | 已产品化 | 44 条 latency row 中识别 1 条 unstable repeat |
| quick review export | 已产品化 | `quick_review_latest.csv/json` 已输出 AP/latency/energy/quality/claim |
| allowlist dry-run gate | 已产品化 | P/S ready subset 为 `CONDITIONAL_GO`；严格 P/Q/S 三臂为 `NO_GO` |

当前不能直接启动无监督三臂大规模生产。原因不是 energy 或 artifact registry 机制未修，而是：

1. Q arm 当前没有 ready artifact registry record；严格三臂 allowlist dry-run 中 Q 的 6 个 jobs 被 artifact gate 阻断。
2. trap25 存在一条 P2 repeat outlier：`repeat_max_min_ratio=2.168 > 2.000`，paper-grade 仍需 retest/quarantine 决策。
3. `trap25/mix_a/iso_s1/iso_s2` 等配置仍缺 claimable energy row；可继续 P/S allowlist 补点，但不能宣称三轴全覆盖。

## 2. 本轮新增或修改的代码路径

新增：

```text
framework/stage2/artifact_registry.py
framework/stage2/outlier_policy.py
framework/tests/test_stage2_large_scale_readiness.py
scripts/stage2_build_artifact_registry.py
scripts/stage2_validate_artifact_registry.py
scripts/stage2_salvage_energy_payloads.py
scripts/stage2_detect_latency_outliers.py
scripts/stage2_export_quick_review.py
scripts/stage2_readiness_gate.py
```

修改：

```text
framework/stage2/lut_productization.py
framework/tests/test_stage2_lut_productization.py
scripts/stage2_generate_energy_lut.py
scripts/stage2_import_energy_lut.py
scripts/stage2_lut_worker.py
```

关键 contract 变化：

1. measured energy row 现在必须有 `measurement_run_id`。
2. energy row 写入 `row_source`：`direct_generation` / `import_existing` / `salvaged_from_payload`。
3. job state 允许 `payload_ready` 和 `salvaged`，用于记录 energy recovery path。
4. quick review 对同一 artifact 的任一 unstable latency row 做聚合，避免 clean row 掩盖 outlier。

## 3. 本轮生成的数据与报告

```text
multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/energy/energy_lut_rows_salvaged_from_payload_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/jobs/energy_salvage_job_state_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/artifacts/artifact_registry_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/artifacts/artifact_registry_summary_v1.json
multi_agent/data/stage2_lut_generation_v1/registry/evidence_registry_base_v1.json
multi_agent/data/stage2_lut_generation_v1/registry/evidence_registry_latest.json
multi_agent/data/stage2_lut_generation_v1/exports/outlier_report_latest.json
multi_agent/data/stage2_lut_generation_v1/exports/outlier_report_latest.csv
multi_agent/data/stage2_lut_generation_v1/exports/quick_review_latest.csv
multi_agent/data/stage2_lut_generation_v1/exports/quick_review_latest.json
multi_agent/data/stage2_lut_generation_v1/exports/readiness_gate_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/allowlist_dry_run_v1/jobs/three_arm_allowlist_job_plan_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/allowlist_dry_run_v1/jobs/ps_ready_subset_job_plan_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/exports/three_arm_allowlist_readiness_gate_latest.json
multi_agent/data/stage2_lut_generation_v1/exports/ps_ready_subset_readiness_gate_latest.json
```

快速审查：

```bash
cd ${V2X_ROOT}
python3 -m unittest framework.tests.test_stage2_lut_productization framework.tests.test_stage2_large_scale_readiness framework.tests.test_stage2_evidence_registry -q
python3 -m json.tool multi_agent/data/stage2_lut_generation_v1/artifacts/artifact_registry_summary_v1.json
python3 -m json.tool multi_agent/data/stage2_lut_generation_v1/registry/evidence_registry_latest.json
python3 -m json.tool multi_agent/data/stage2_lut_generation_v1/exports/three_arm_allowlist_readiness_gate_latest.json
head -n 20 multi_agent/data/stage2_lut_generation_v1/exports/quick_review_latest.csv
```

本轮验证命令已通过：

```text
python3 -m unittest framework.tests.test_stage2_lut_productization framework.tests.test_stage2_large_scale_readiness framework.tests.test_stage2_evidence_registry -q
Ran 35 tests in 4.220s
OK
```

## 4. 当前数据读数

Artifact registry：

| 指标 | 数值 |
|---|---:|
| total artifacts | 41 |
| ready artifacts | 41 |
| missing artifacts | 0 |
| blocked artifacts | 0 |

Evidence registry：

| 指标 | 数值 |
|---|---:|
| latency measured cells | 44 |
| AP measured cells | 11 |
| energy measured cells | 8 |
| total measured cells | 63 |
| energy claim allowed | true |

保留的 unsupported conclusions：

```text
cross_model_ds_map_for_pyramid
historical_trt_as_h800_measured_quant_evidence
```

这两个不阻断当前 LUT 生产 gate；含义是 Pyramid 不使用 CoDriving DS map 做主 claim，Q evidence 仍是 historical/proxy，不能当作 H800 measured Q claim。

Energy salvage：

| 指标 | 数值 |
|---|---:|
| scanned calibration telemetry payloads | 8 |
| salvaged canonical energy rows | 8 |
| rows with `measurement_run_id` | 8 |
| row_source | `salvaged_from_payload` |

Outlier detector：

| 指标 | 数值 |
|---|---:|
| latency rows checked | 44 |
| claimable rows | 43 |
| unstable rows | 1 |

unstable row：

```text
config_id=calibration_h800_tvm_pyramid_trap25_fp16_metaschedule_tuned
run_id=calib_h800_tvm_pyramid_trap25_p2_latency_repeat_20260625_234205_metaschedule_tuned
reason=repeat_max_min_ratio=2.168>2.000
claim_status=no_claim
```

## 5. 当前可直接审查的实测表

来自：

```text
multi_agent/data/stage2_lut_generation_v1/exports/quick_review_latest.csv
```

| label | config | AP70 | latency ms | energy J/inf | latency quality | latency claim | energy claim |
|---|---|---:|---:|---:|---|---|---|
| base | `calibration_h800_tvm_pyramid_base_fp16_metaschedule_tuned` | 0.6309 | 6.210 | 1.9213 | stable | claimable | claimable |
| p50 | `calibration_h800_tvm_pyramid_p50_fp16_metaschedule_tuned` | 0.5641 | 3.012 | 0.7095 | stable | claimable | claimable |
| p75 | `calibration_h800_tvm_pyramid_p75_fp16_metaschedule_tuned` | 0.5300 | 0.483 | 0.0566 | stable | claimable | claimable |
| trap25 | `calibration_h800_tvm_pyramid_trap25_fp16_metaschedule_tuned` | 0.5905 | 21.605 |  | has_unstable_repeat | conditional | no_claim |
| mix_a | `calibration_h800_tvm_pyramid_mix_a_fp16_metaschedule_tuned` | 0.6288 | 5.658 |  | stable | claimable | no_claim |
| mix_b | `calibration_h800_tvm_pyramid_mix_b_fp16_metaschedule_tuned` | 0.6362 | 19.390 | 4.6820 | stable | claimable | claimable |
| mix_d | `calibration_h800_tvm_pyramid_mix_d_fp16_metaschedule_tuned` | 0.6369 | 21.044 | 5.3143 | stable | claimable | claimable |
| iso_s1 | `calibration_h800_tvm_pyramid_iso_s1_fp16_metaschedule_tuned` | 0.6336 | 6.918 |  | stable | claimable | no_claim |
| iso_s2 | `calibration_h800_tvm_pyramid_iso_s2_fp16_metaschedule_tuned` | 0.6339 | 7.252 |  | stable | claimable | no_claim |

备注：

1. AP 通过 quick review 的 label/width/quant fallback join 从 smoke AP rows 对齐到 calibration latency rows。
2. p75 energy 虽然 schema/claim gate 通过，但数值过低，仍建议做 repeat reasonableness check。
3. trap25 表中显示 clean representative latency，但 artifact-level quality 已聚合 P2 outlier，因此 claim 为 conditional。

## 6. Dry-run 判定

### 6.1 当前 rows 离线 readiness

```text
multi_agent/data/stage2_lut_generation_v1/exports/readiness_gate_latest.json
decision=CONDITIONAL_GO
```

通过：

1. schema gate pass：63 条 LUT rows + 41 artifact rows 无 schema error。
2. artifact registry pass：41/41 ready。
3. energy gate pass：8 条 salvaged energy rows 中存在 claimable measured telemetry rows。
4. evidence registry pass：`evidence_registry_latest.json` 可加载，且 energy claim gate 打开。

条件项：

1. outlier gate conditional：1 条 trap25 unstable repeat。

### 6.2 P/S ready subset dry-run

```text
multi_agent/data/stage2_lut_generation_v1/generated/allowlist_dry_run_v1/jobs/ps_ready_subset_job_plan_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/exports/ps_ready_subset_readiness_gate_latest.json
decision=CONDITIONAL_GO
```

含 4 个 config、12 个 generate jobs：

| arm | config |
|---|---|
| P | p50, p75 |
| S | base, iso_s1 |

结论：可以继续 P/S 小范围 allowlist 后台补点，但仍要隔离 trap25 outlier，并遵守 GPU idle 硬门槛。

### 6.3 严格 P/Q/S 三臂 dry-run

```text
multi_agent/data/stage2_lut_generation_v1/generated/allowlist_dry_run_v1/jobs/three_arm_allowlist_job_plan_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/exports/three_arm_allowlist_readiness_gate_latest.json
decision=NO_GO
```

含 P/S/Q 各 2 个 config、18 个 generate jobs。失败原因：

```text
artifact_registry gate fail:
- latency:allowlist_q_base_int8_metaschedule_tuned
- ap:allowlist_q_base_int8_metaschedule_tuned
- energy:allowlist_q_base_int8_metaschedule_tuned
- latency:allowlist_q_p50_int8_metaschedule_tuned
- ap:allowlist_q_p50_int8_metaschedule_tuned
- energy:allowlist_q_p50_int8_metaschedule_tuned
```

解释：当前 artifact registry 没有 Q arm ready artifacts。不能把 Q arm 从三臂 dry-run 中跳过后声称三臂可大规模生产。

## 7. 下一阶段计划

### P0 收尾

1. 将 `energy_lut_rows_salvaged_from_payload_v1.jsonl` 作为当前严格 energy rows 输入；旧 `energy_lut_rows_v1.jsonl` 缺 `measurement_run_id`，只能保留为历史格式，不应继续作为 claimable 输入。
2. 在下一轮真实 worker job plan 中为 energy jobs 填入：
   ```json
   {
     "resource": {
       "energy_salvage": {
         "payload_root": ".../raw/<job-dir>",
         "latency_rows": [".../latency_lut_rows_v1.jsonl"],
         "out_jsonl": ".../energy_lut_rows_v1.jsonl"
       }
     }
   }
   ```
3. 若后续需要把 artifact registry 纳入 worker preflight，可在 worker 启动前读取 `artifact_registry_v1.jsonl`，对 `config_id` 不 ready 的 job 直接写 `preflight_blocked`。

### P1 真实补点

1. trap25 tuned retest：
   - 目标：GPU 完全空闲，单 job 至少 2 次。
   - 判定：若回到约 21.6 ms，则 P2 outlier 关联 quarantine；若继续分叉，进入 DB/kernel investigation。
2. energy repeats：
   - base/p50/p75 各补到 2-3 repeats。
   - mix_b/mix_d 各至少 2 repeats。
   - mix_a/iso_s1/iso_s2/trap25 补首批 energy rows。
3. Q arm artifact unblock：
   - 先选 base INT8、p50 INT8 两个最小 allowlist。
   - 生成或定位 ONNX + TVM work dir + MetaSchedule DB。
   - 写入 artifact registry 后，再跑 Q latency/AP/energy smoke。
4. missing/bad DB case：
   - `p50b2_136`、`mix_e_64_64_128` 需要 ONNX artifact。
   - `s1_64/pad64` 类 bad DB 继续 quarantine 或重建 DB。

### P2 大规模启动前最后 gate

1. 严格 P/Q/S allowlist dry-run 必须从 `NO_GO` 变成至少 `CONDITIONAL_GO`，且 Q jobs 不再被 artifact registry 阻断。
2. 真实 worker 需要在 GPU idle 条件下连续跑完 6-12 个 jobs，不需要手工修 canonical rows。
3. `three_arm_allowlist_readiness_gate_latest.json` 中：
   - schema pass
   - artifact_registry pass
   - evidence_registry pass
   - energy pass
   - outlier pass 或仅有明确隔离的 conditional
4. 重新导出：
   ```text
   quick_review_latest.csv
   outlier_report_latest.csv
   artifact_registry_summary_v1.json
   three_arm_allowlist_readiness_gate_latest.json
   ```

## 8. 当前启动判定

```text
Stage2 三臂大规模生产判定：NO_GO
```

可以启动：

```text
P/S ready subset allowlist：CONDITIONAL_GO
```

不可以启动：

```text
full unattended P/Q/S large-scale LUT production：NO_GO
```

NO_GO 的必要解除条件：

1. Q arm 至少 base INT8 和 p50 INT8 两个 config 进入 artifact registry ready。
2. trap25 outlier 有 retest/quarantine 结论。
3. energy 缺口配置补齐或明确 no-claim，不再混入 claimable 表。
4. 真实 worker 完成 6-12 jobs 的 GPU idle 长跑验证。
