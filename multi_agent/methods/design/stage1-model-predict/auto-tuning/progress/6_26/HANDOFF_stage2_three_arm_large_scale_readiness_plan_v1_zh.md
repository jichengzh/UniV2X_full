# Stage2 三臂大规模生产前交接与补齐计划 v1

日期：2026-06-26  
范围：Stage2 latency/AP/energy 产品化 LUT 生成链路，以及 P/Q/S 三臂进入后台大规模补点前的 readiness gate。

## 0. 启动命令

建议清空上下文后使用：

```text
/goal Stage2 三臂大规模生产前收敛：彻底修复 energy 生成链路、artifact registry 链路和 outlier 处理；完成 energy runner/salvage 产品化、artifact_registry_v1 schema+validator+backfill、latency outlier detector+retest gate，并通过 allowlist dry-run 后给出是否 GO large-scale P/Q/S production 的判定。
```

启动后优先阅读：

1. `multi_agent/methods/design/auto-tuning/progress/HANDOFF_stage2_three_arm_large_scale_readiness_plan_v1_zh.md`
2. `multi_agent/methods/design/auto-tuning/progress/HANDOFF_stage2_lut_p0_p1_p2_start_v1_zh.md`
3. `multi_agent/methods/design/auto-tuning/progress/HANDOFF_stage2_lut_gpu012_progress_v1_zh.md`
4. `multi_agent/methods/design/auto-tuning/progress/PLAN_stage2_dataset_v2_aligned_lut_generation_v1_zh.md`

## 1. 当前结论

当前不建议直接启动三臂大规模生产。可以继续做 allowlist 级别的真实补点，但必须先把三个链路问题收敛到可自动判定、可恢复、可隔离：

1. energy 链路：已有 5 条 calibration energy rows，但生成过程中出现过“测量 payload 已成功、canonical row 生成失败”的情况。目前靠 salvage 整理，尚未产品化为 runner 的正式恢复路径。
2. artifact registry 链路：latency/AP/energy canonical rows 可以生成，但还缺少一等公民级 artifact registry，把每个配置的 ONNX、TVM work dir、MetaSchedule DB、AP 来源、energy raw artifact、canonical row id 串起来。
3. outlier 链路：trap25 tuned replay 出现重复样本大幅分叉，p50 从历史约 21.6 ms 被拉到 30.169 ms。当前已有 quarantine 机制，但还缺少 latency repeat outlier detector 和 paper-grade promotion gate。

当前已有 canonical 数据规模：

| 数据链路 | 当前数量 | 说明 |
|---|---:|---|
| latency LUT | 44 rows | 包含 validation、smoke、calibration 的 H800+TVM 真实链路数据 |
| AP LUT | 11 rows | 已有 smoke/calibration 级 AP rows |
| energy LUT | 5 rows | 来自真实 telemetry payload salvage，仍需产品化 |
| quarantine | 2 rows | 已记录 bad DB / unsupported case |

当前快速审查表的关键实测结果：

| label | AP70 | energy J/inf | default latency ms | tuned latency ms | 备注 |
|---|---:|---:|---:|---:|---|
| base | 0.6309 | 1.9244 | 56.324 | 6.213 | 验证了 56 ms 链路可回到历史 tuned 约 6.3 ms |
| p50 | 0.5641 | 0.6954 | 26.239 | 3.010 | 可作为轻量配置 calibration anchor |
| p75 | 0.5300 | 0.0572 | 12.675 | 0.483 | 需要复核 energy 合理性和 measurement duration |
| mix_b | 0.6362 | 4.6820 | 41.429 | 19.401 | 需要重复 energy/latency |
| mix_d | 0.6369 | 5.3143 | 42.436 | 21.050 | 需要重复 energy/latency |
| trap25 | 0.5905 |  | 42.358 | 30.169 | tuned repeats 分叉，不能 paper-grade claim |
| iso_s1 | 0.6336 |  | 51.624 | 6.918 | energy 待补 |
| iso_s2 | 0.6339 |  | 51.957 | 7.252 | energy 待补 |
| mix_a | 0.6288 |  | 35.793 | 5.658 | energy 待补 |

## 2. 三臂大规模生产的定义

这里的“三臂”指 P/Q/S 三类候选路线进入后台长期补点：

1. P arm：结构/通道/剪枝类候选。
2. Q arm：量化/精度策略类候选。
3. S arm：schedule/deployment/TVM tuning 类候选。

三臂可以大规模生产，不等于只要 worker 能跑起来。必须满足：

1. 每个候选在 measurement 前已经有 artifact registry record。
2. latency/AP/energy 三条 evidence 链路都走 `generate_*_lut` 主路径；`import_existing_*` 只用于历史数据整理和明确标记的 salvage。
3. 对每个 canonical row，可以追溯到 config、artifact、raw log、GPU、command、environment、schema validator、quality flag。
4. 任何 missing artifact、bad DB、GPU busy、outlier、generator failure 都进入 quarantine 或 blocked status，而不是隐式失败。
5. worker 支持长时间后台运行、断点续跑、重复跳过、失败隔离、registry update 和快速审查输出。

## 3. 大规模启动 GO gates

只有全部通过后，才可以给出 `GO large-scale P/Q/S production`：

| Gate | 名称 | 通过标准 |
|---|---|---|
| G0 | Schema gate | latency/AP/energy/quarantine/artifact registry 全部 schema validation 通过 |
| G1 | Artifact registry gate | 所有已测 canonical rows 均能回指 artifact_id；所有待跑 jobs 的 required artifacts 为 ready |
| G2 | Energy gate | energy generator 可从测量到 canonical row 自动闭环；payload-ready 但 row-missing 的情况能自动 salvage |
| G3 | Outlier gate | latency repeat outlier detector 已接入；paper-grade rows 不允许 unstable flag |
| G4 | GPU idle gate | latency 和 energy job 启动前必须确认目标 GPU 空闲；busy 时 job 进入 preflight_blocked |
| G5 | Quarantine gate | bad DB、missing ONNX、unsupported config、outlier run 都有明确 quarantine/block 记录 |
| G6 | Dry-run gate | 至少 6-12 个 allowlist jobs 跨 P/Q/S 跑完，不能需要手工修 row |
| G7 | Registry export gate | artifact registry、evidence registry、quick review table 能一键生成并一致 |

## 4. P0：先把机制补齐

P0 的目标不是补很多点，而是保证以后补点不会产生无法解释的数据。

### 4.1 Energy 链路修复

必须完成：

1. 固化 energy raw artifact contract。
   - 每次 energy job 至少写入：`telemetry_payload.json`、`job_state.json`、`command.json`、`env.json`、`nvidia_smi_preflight.csv`、GPU utilization/power samples、stdout/stderr。
   - canonical energy row 必须包含 raw artifact 路径和 `measurement_run_id`。
2. 产品化 energy salvage。
   - 新增或完善 `scripts/stage2_salvage_energy_payloads.py`，扫描 payload-ready 但 canonical-row-missing 的 job。
   - salvage 生成的 row 必须标记 `row_source=salvaged_from_payload`，不能伪装成正常 direct generation。
   - salvage 成功后更新 job state：`payload_ready -> salvaged -> succeeded`。
3. 修复 generator dependency isolation。
   - energy runner 不能因为远端 python 缺少 project dependency 而丢 canonical row。
   - 保持 `framework.stage2.lut_productization` 轻量可导入；对需要 `yaml` 等额外依赖的模块延迟导入。
   - 明确 runner 的 generator python 策略：使用当前 project python 生成 canonical row，TVM python 只负责 TVM runtime/measurement。
4. 接入 no-claim/claim validator。
   - 只有通过 energy schema、raw artifact、GPU idle、repeat quality 的 row 可以进入 claimable。
   - 缺 repeat 或 CV 异常时保留数据，但 claim status 为 `no_claim`.

P0 energy 验收：

1. 对已有 5 条 energy rows 可以从 raw artifact 重新构建并得到一致 canonical rows。
2. 人工制造一个 generator failure 后，salvage 能自动恢复。
3. `base/p50/p75/mix_b/mix_d` 至少各 1 条 job 通过完整 runner，不需要手工修 row。

### 4.2 Artifact registry 链路修复

必须新增一等公民 artifact registry：

建议路径：

```text
multi_agent/data/stage2_lut_generation_v1/artifacts/artifact_registry_v1.jsonl
```

建议脚本：

```text
scripts/stage2_build_artifact_registry.py
scripts/stage2_validate_artifact_registry.py
scripts/stage2_update_registry_from_luts.py
```

artifact registry row 最小字段：

| 字段 | 说明 |
|---|---|
| `schema_version` | 固定版本，如 `stage2_artifact_registry_v1` |
| `artifact_id` | 稳定 id，建议由 model/label/width/quant/schedule/scope hash 得到 |
| `config_id` | job/config id |
| `model_name` | pyramid/codriving 等 |
| `label` | base/p50/mix_b/iso_s1 等 |
| `width` | backbone width tuple 或对应结构参数 |
| `arm` | P/Q/S 或 baseline/mixed |
| `optimized_scope` | 当前主线为 `backbone_only` |
| `onnx_path` | measurement 所用 ONNX |
| `tvm_work_dir` | TVM build/tuning work dir |
| `database_path` | MetaSchedule DB 路径 |
| `ap_source_path` | AP eval 原始结果路径 |
| `energy_raw_dir` | energy raw artifact 目录 |
| `latency_row_ids` | 对应 latency canonical rows |
| `ap_row_ids` | 对应 AP canonical rows |
| `energy_row_ids` | 对应 energy canonical rows |
| `artifact_status` | ready/missing/blocked/quarantined |
| `missing_artifacts` | 明确缺失项 |
| `quarantine_refs` | 关联 quarantine row |
| `created_at/updated_at` | 时间戳 |

artifact registry 规则：

1. worker 启动前先查 registry；artifact 不 ready 时 job 不启动，写 blocked/quarantine。
2. P1 中 `p50b2_136`、`mix_e_64_64_128` 这类 missing ONNX 配置，必须先 registry ready，再测 latency/energy。
3. evidence registry 只记录证据覆盖和 claim 状态；artifact registry 负责 artifact 可追溯性。两者不能互相替代。
4. quick review table 由 artifact registry + canonical LUT rows join 生成，避免手工整理。

P0 artifact 验收：

1. 当前 44 条 latency、11 条 AP、5 条 energy rows 都能 backfill 到 artifact registry。
2. 每个 measured row 都能查到 artifact_id。
3. missing ONNX / bad DB 不再表现为 runner 运行时随机失败，而是 preflight blocked。

### 4.3 Outlier 链路修复

必须新增 latency outlier detector，建议路径：

```text
scripts/stage2_detect_latency_outliers.py
framework/stage2/outlier_policy.py
```

outlier detector 输入：

1. canonical latency LUT。
2. raw repeat samples。
3. historical anchors，例如 base tuned 约 6.3 ms、trap25 tuned 历史约 21.6 ms。
4. job metadata：GPU、DB path、schedule policy、timestamp、preflight 状态。

最小判定规则：

| 规则 | calibration 级 | paper-grade 级 |
|---|---:|---:|
| 单 run repeat max/min | <= 2.0 | <= 1.5 |
| 多 run p50 spread | <= 15% | <= 10% 或 <= 0.5 ms 二者取宽 |
| historical anchor drift | 可 warning | 超阈值必须 retest |
| GPU busy preflight | blocked | blocked |
| raw repeat 缺失 | no-claim | no-claim |

trap25 当前处理：

1. 当前 30.169 ms tuned row 保留为真实观测，但标记 `unstable_repeat`。
2. 不能用于 paper claim，也不能作为 DS/latency surrogate 的稳定 anchor。
3. 必须在 GPU 完全空闲时单 job retest 至少 2 次。
4. 如果 2/3 run 回到约 21.6 ms，则把异常 run 关联 quarantine，paper table 使用 clean median-of-runs。
5. 如果继续分叉，则检查 MetaSchedule DB、build cache、TVM runtime warmup、GPU clock/persistence 状态。

P0 outlier 验收：

1. detector 能识别 trap25 当前异常。
2. detector 能放行 base tuned 6.213 ms 和 p50 tuned 3.010 ms 等稳定样本。
3. quick review table 增加 `quality_flag` 和 `claim_status`。

## 5. P1：补点与链路压力测试

P1 的目标是从“能跑”推进到“可以无人工修补地跑一批”。

### 5.1 Validation replay 补齐

优先顺序：

1. base tuned/default replay：保持当前 56.324 ms default、6.213 ms tuned anchor。
2. p50 tuned/default replay：验证轻量配置的低延迟稳定性。
3. trap25 tuned retest：专门解决 outlier。
4. iso_s1/iso_s2/mix_a：补 energy 后进入 calibration anchor。

### 5.2 Energy calibration 补齐

第一批 energy repeat：

| label | 目标 repeat | 目的 |
|---|---:|---|
| base | 2-3 | 作为 H800 backbone_only energy anchor |
| p50 | 2-3 | 轻量配置 anchor |
| p75 | 2-3 | 复核 0.057 J/inf 是否过低 |
| mix_b | 2 | 高 AP / 中延迟配置 |
| mix_d | 2 | 高 AP / 中延迟配置 |

第二批：

| label | 目标 repeat | 目的 |
|---|---:|---|
| iso_s1 | 1-2 | 与 base 同 AP 附近的 schedule/structure 对比 |
| iso_s2 | 1-2 | 与 iso_s1 交叉验证 |
| mix_a | 1-2 | 低 latency / 高 AP 附近候选 |

energy 运行规则：

1. latency 和 energy 测试必须在目标 GPU 完全空闲时运行。
2. 同一张 GPU 上不允许并行跑两个 latency/energy measurement。
3. AP eval 可以和 latency/energy 分离调度，但如果 AP 会占用 GPU，也必须走 preflight。
4. GPU0/1/2 可以并行使用，但每张 GPU 内部必须串行 measurement。
5. 每个 job 写明 GPU id；跨 GPU 对比前必须标记 GPU id，不默认混合为同分布。

### 5.3 Artifact unblock

当前已知 blocked 配置：

| config | 问题 | 下一步 |
|---|---|---|
| `p50b2_136` | remote ONNX candidates missing | 先生成/定位 ONNX，再 registry ready 后补测 |
| `mix_e_64_64_128` | remote ONNX candidates missing | 同上 |
| `s1_64/pad64` 类 bad DB | DB evidence 不可用 | 重建 DB 或显式排除 |

P1 验收：

1. 至少 8 条 claimable energy calibration rows。
2. 至少一个 missing artifact case 从 blocked 修复到 ready 并成功生成 latency/AP/energy 中至少两类 row。
3. trap25 有 retest 结论：clean、unstable、或 DB/kernel investigation required。
4. worker 连续完成 6-12 个 allowlist jobs，无需手工改 canonical rows。

## 6. P2：三臂 allowlist dry-run 到大规模启动

P2 不是全量启动，而是三臂各选少量代表配置，验证后台生产能力。

建议 allowlist：

| arm | 第一批 | 第二批 |
|---|---|---|
| P | p50、p75、trap25 retest、p50b2_136 修复后 | 更多 P 候选补齐 |
| Q | 先选择已有 ONNX/DB 的量化候选 | 扩到 Q top-K |
| S | base tuned/default、iso_s1、iso_s2、mix_a | 扩到更多 schedule policy |

每个 allowlist job 必须经过：

1. artifact registry preflight。
2. GPU idle preflight。
3. latency/AP/energy schema validation。
4. raw artifact completeness validation。
5. outlier detection。
6. registry update。
7. quick review export。

P2 验收：

1. 三臂至少各 2 个配置通过完整链路，或明确 blocked 且原因可解释。
2. worker 可以后台长时间运行，支持 resume，不重复生成已成功 row。
3. artifact registry 和 evidence registry 一键更新后没有 orphan row。
4. quick review table 能展示每个配置的 AP、latency、energy、quality flag、claim status。

## 7. 最终大规模生产计划

通过 P0/P1/P2 后，再进入大规模补点。建议目标规模分三层：

| 阶段 | 目标规模 | 目的 |
|---|---:|---|
| Smoke/allowlist | 12-24 configs | 验证三臂链路和异常处理 |
| Calibration | 80-150 configs | 建立 latency/AP/energy cost model 可用覆盖 |
| Paper-grade | 20-40 configs x repeat | 形成论文可声明的核心表格和曲线 |

大规模阶段配置分布建议：

1. P arm：约 40%。覆盖 p50/p75/trap/mix 结构附近，重点看 latency/AP/energy Pareto。
2. Q arm：约 25%。先只选 artifact ready 的量化策略，逐步引入更多组合。
3. S arm：约 25%。base、iso、mix 附近的 tuned/default/schedule policy 对比。
4. Holdout/validation：约 10%。用于复测历史 anchor、outlier retest、跨 GPU 漂移检查。

最终数据必须统一存储到：

```text
multi_agent/data/stage2_lut_generation_v1/
```

建议目录：

```text
generated/
  validation/
  smoke/
  calibration/
  paper/
  quarantine/
artifacts/
  artifact_registry_v1.jsonl
  artifact_registry_summary_v1.json
exports/
  quick_review_latest.csv
  claimable_rows_latest.csv
  no_claim_rows_latest.csv
  outlier_report_latest.csv
```

## 8. 下一轮执行清单

### P0 必做

1. 写 `artifact_registry_v1` schema、validator、backfill builder。
2. 写 energy payload salvage 产品化脚本，并接入 worker resume。
3. 写 latency outlier detector 和 quality report。
4. 将 quick review export 改为 registry join 生成。
5. 为上述机制补单元测试和最小集成测试。

### P1 补点

1. energy repeat：base/p50/p75/mix_b/mix_d。
2. energy 补齐：iso_s1/iso_s2/mix_a。
3. latency retest：trap25 tuned。
4. artifact unblock：p50b2_136、mix_e_64_64_128。
5. bad DB 处理：s1_64/pad64 重建或明确排除。

### P2 补点

1. 三臂 allowlist dry-run：P/Q/S 各 2-4 个配置。
2. 后台 worker 长跑 6-12 jobs。
3. artifact registry + evidence registry + quick review 一键导出。
4. 生成 GO/NO-GO 判定文档。

## 9. 大规模启动判定模板

下一阶段结束时必须给出以下判定：

```text
Stage2 三臂大规模生产判定：GO / CONDITIONAL GO / NO-GO

GO 条件：
- artifact registry 对所有已测 rows 覆盖 100%，对待跑 allowlist jobs preflight 100%。
- energy 链路可以自动完成 measurement -> raw artifact -> canonical row -> registry update。
- outlier detector 已接入，paper-grade rows 无 unstable flag。
- GPU idle gate 生效，busy 时 job 自动 blocked。
- worker 完成 6-12 个 allowlist jobs，无需手工修 row。
- quick review/export/evidence registry 一致。

若任一条件不满足：
- 只能 CONDITIONAL GO 到更小 allowlist。
- 不允许进入无监督大规模补点。
```

