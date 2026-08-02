# Stage2 H800+TVM Existing Evidence Inventory v1

更新时间: 2026-06-25

目的: 将历史 H800+TVM 实测从 `dataset_v2` 的 4090+TRT 证据中分离出来, 作为 Stage2 latency LUT 产品化路线的主要历史 seed 和 validation replay 依据。

归一化数据目录:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/existing_h800_tvm
```

单位约定:

```text
本 inventory 中所有归一化 CSV 的 latency 列统一为 ms。
原始结果文件中很多字段仍是 us, 已在归一化表中转换。
```

---

## 1. 总体结论

`dataset_v2.csv` 的主体是 4090+TRT, 对当前 H800+TVM 产品化路线只能作为 AP/结构/历史对照, 不能作为 H800 latency 的主基线。真正应该继承的 H800+TVM 证据主要来自:

1. Pyramid backbone-only FP16 TVM LUT。
2. Pyramid Gap1 corrected W_g/P_g 复测。
3. Pyramid INT8 stage0 / format 机制证据。
4. CoDriving backbone-only H800 TVM 复测。
5. CoDriving s0 / INT8 micro-probe 和 separable negative anchor。

当前最重要的校正是: 最近产品化 smoke 的 `56.130 ms` 不是 H800 tuned backbone 性能, 而是 **default-like/link-check**。它和历史 Pyramid base default `56.321 ms` 基本一致, 但和历史 Pyramid base tuned `6.320 ms` 相差约 8.9x。因此该 smoke 只能证明链路跑通, 不能作为正式 LUT claim。

---

## 2. 已整理的数据文件

| 文件 | 行数 | 用途 |
|---|---:|---|
| `pyramid_h800_tvm_backbone_fp16_ms.csv` | 19 | Pyramid backbone-only FP16 H800 TVM measured LUT seed |
| `pyramid_h800_tvm_gap1_corrected_ms.csv` | 8 | Gap1 corrected rows, W_g/P_g 机制与 AP70 对齐 |
| `pyramid_h800_tvm_int8_scope_limited_ms.csv` | 5 | Pyramid INT8/stage0 机制证据, 不能当 full-backbone INT8 LUT |
| `codriving_h800_tvm_backbone_ms.csv` | 4 | CoDriving backbone H800 TVM measured rows |
| `codriving_h800_tvm_s0_and_int8_probes_ms.csv` | 11 | CoDriving s0 和 INT8 micro-probes |
| `h800_tvm_existing_summary.json` | 1 | 汇总、结论和废弃源说明 |

快速审查:

```bash
cd ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/existing_h800_tvm
python -m json.tool h800_tvm_existing_summary.json | sed -n '1,120p'
for f in *.csv; do echo "$f"; tail -n +2 "$f" | wc -l; done
```

---

## 3. Pyramid H800+TVM 证据

### 3.1 FP16 backbone-only LUT seed

来源:

```text
results/latency_lut_pyramid.json
results/gap1_grid_corrected.json
results/gap1_grid_retune_h800.csv
```

归一化文件:

```text
multi_agent/data/stage2_lut_generation_v1/existing_h800_tvm/pyramid_h800_tvm_backbone_fp16_ms.csv
multi_agent/data/stage2_lut_generation_v1/existing_h800_tvm/pyramid_h800_tvm_gap1_corrected_ms.csv
```

全量 FP16 backbone measured seed 共 19 行。speedup 分布:

| 指标 | 值 |
|---|---:|
| min speedup | 1.960x |
| median speedup | 7.466x |
| max speedup | 26.237x |

代表性点:

| label | width | default ms | tuned ms | speedup | 说明 |
|---|---|---:|---:|---:|---|
| base | [64,128,256] | 56.321 | 6.320 | 8.912x | 主基线, 说明近期 56ms smoke 是 default-like |
| p50 | [32,64,128] | 26.242 | 3.011 | 8.717x | 与 pruning 50% 对齐的主 seed |
| p75 | [16,32,64] | 12.689 | 0.484 | 26.237x | 最大 tuned/default, 但 AP70 较低 |
| pad64 | [64,96,192] | 47.407 | 6.152 | 7.706x | W_g/P_g 机制中的 tuned Pareto 点 |
| trap25 | [48,96,192] | 42.355 | 21.615 | 1.960x | default-Pareto 吸引子, tuned 后被 pad64 支配 |

### 3.2 Gap1 corrected: W_g/P_g 机制

来源:

```text
results/gap1_grid_corrected.json
```

核心结论:

| 对比 | default ms | tuned ms | AP70 | 结论 |
|---|---:|---:|---:|---|
| trap25 [48,96,192] | 42.355 | 21.615 | 0.590 | default 看起来较优, tuned 后慢 |
| pad64 [64,96,192] | 47.407 | 6.152 | 0.590 | default 较差, tuned 后成为同 AP70 下优点 |

解释:

1. trap25 是 `W_g`: default-Pareto 上会被串行贪心选中, 但 tuned 后掉出前沿。
2. pad64 是 `P_g`: default 下被错过, tuned 后严格支配 trap25。
3. 机制定位到 stage0 对齐: `in_per_g=3` 的 s0 失配导致 tuning headroom 崩塌; 将 s0 从 48 pad 到 64 后, 即使 s1/s2 仍失配, tuned speedup 恢复到 7.706x。

这组数据是“我们的优化方案可带来约 10x 提速”的主要证据之一, 但精确说法应是: **Pyramid grouped-conv backbone 在 H800+TVM MetaSchedule 下存在 7x-9x 常见收益, p75 等特定点可到 26x; 失配点可能只有约 2x。**

### 3.3 INT8 / Q-axis 证据的范围

来源:

```text
results/q_int8_ms_stage0_result.json
results/q_int8_dp4a_pairs.csv
results/q_tvm_int8_verdict.md
results/coupling_map/C2_C3_pyramid.json
results/pqs_ablation_results.json
```

归一化文件:

```text
multi_agent/data/stage2_lut_generation_v1/existing_h800_tvm/pyramid_h800_tvm_int8_scope_limited_ms.csv
```

可以继承的事实:

| scope | 事实 |
|---|---|
| stage0 s0=64 | INT8 WMMA measured 0.150 ms vs FP16 ref 0.218 ms, 1.449x |
| numerical | int8 stage0 correctness PASS, max_rel_error=0.0 |
| format mechanism | NCHWc INT8 可行性受 `IC_BN` / `in_per_g` 约束 |

不能越界使用:

1. 这些 INT8 证据主要是 stage0 / format-level 机制证据, 不是 full-backbone INT8 LUT。
2. `pqs_ablation_results.json` 中 INT8 latency magnitude 使用的是 stage0 speedup proxy; robust 结论是 buildability/format constraint, 不是 full-backbone INT8 绝对延迟。
3. `C2_C3_pyramid.json` 已修正早期结论: trap25 的 INT8 NCHW fallback 与 tuned FP16 接近, padding 到 NCHWc 不一定带来净收益。因此 Q-axis 需要重新实测 full-backbone 才能成为 paper-grade LUT。

---

## 4. CoDriving H800+TVM 证据

### 4.1 Backbone measured rows

来源:

```text
results/codriving_tvm_p25_p75.csv
results/latency_lut_codriving.json
multi_agent/methods/progress/HANDOFF_codriving_tvm_migration_v1.md
```

归一化文件:

```text
multi_agent/data/stage2_lut_generation_v1/existing_h800_tvm/codriving_h800_tvm_backbone_ms.csv
```

实测行:

| label | batch | default ms | tuned ms | speedup | 说明 |
|---|---:|---:|---:|---:|---|
| base | 2 | 16.680 | 8.058 | 2.070x | CoDriving 主基线 |
| p50 | 2 | 3.643 | 1.609 | 2.264x | 剪枝后 backbone 编译级加速明显 |
| p25 | 2 | 11.885 | 10.467 | 1.135x | 非强耦合点 |
| p75 | 1 | 2.832 | 2.451 | 1.155x | batch=1, 绝对值不能和 batch=2 直接横比 |

注意:

1. `latency_lut_codriving.json` 中 p25/p75 曾有 power-law estimated 值, 但现在 `codriving_tvm_p25_p75.csv` 中已有实测恢复行。后续应优先使用 CSV 实测行。
2. CoDriving 相关文件中 precision 元数据存在 `fp16`/`fp32` 表述不完全一致的问题。正式 canonical LUT 前应通过 replay command 固化 precision 字段, 不能只按文件名推断。

### 4.2 CoDriving 的 negative anchor 结论

来源:

```text
results/codriving_coupling_verdict.md
results/b4_codriving_ablation_results.json
results/coupling_map/C0c_codriving_pqs.json
results/coupling_map/C6_codriving_highdim.json
results/codriving_int8_verify.json
```

核心结论:

1. CoDriving 是 standard 3x3 ResNet conv, groups=1, reduction dim `K=Cin*9` 较大。
2. H800 TVM tuning ratio 约 1.1x-2.3x, 明显弱于 Pyramid 的 grouped conv。
3. CoDriving 没有 Pyramid 那种稳定的 alignment-gated W_g/P_g trap。
4. `b4_codriving_ablation_results.json` 中 A-joint 与 A-serial 相等, 是 separable / weak-coupling negative anchor。
5. `C6_codriving_highdim.json` 已撤回早期 high-dim trap 结论, 当前 revised verdict 是 `NO_ROBUST_HIGHDIM_TRAP`。

### 4.3 CoDriving INT8 micro evidence

归一化文件:

```text
multi_agent/data/stage2_lut_generation_v1/existing_h800_tvm/codriving_h800_tvm_s0_and_int8_probes_ms.csv
```

关键实测:

| config | fp16 ms | int8 ms | int8 speedup | 结论 |
|---|---:|---:|---:|---|
| Cin=48 p25_s0 | 0.076 | 0.054 | 1.42x | buildable, no rank flip |
| Cin=64 base_s0 | 0.090 | 0.068 | 1.32x | buildable, no rank flip |

这说明 CoDriving standard conv 的 INT8 路径对宽度主要是 soft scaling, 不像 Pyramid grouped conv 那样形成 categorical alignment gate。

---

## 5. 可作为 LUT seed 的数据

### 5.1 可以直接作为 latency seed

| 数据 | 可用性 | 说明 |
|---|---|---|
| `pyramid_h800_tvm_backbone_fp16_ms.csv` | yes | 19 行 H800 TVM backbone-only FP16 measured LUT |
| `pyramid_h800_tvm_gap1_corrected_ms.csv` | yes | 8 行 corrected measured rows, 可作为 validation replay 首批 |
| `codriving_h800_tvm_backbone_ms.csv` | yes, with caveat | 4 行 CoDriving measured rows; p75 batch=1, precision metadata 需 replay 固化 |

### 5.2 只能作为机制证据

| 数据 | 原因 |
|---|---|
| `pyramid_h800_tvm_int8_scope_limited_ms.csv` | stage0 / format-level, 不是 full-backbone INT8 |
| `codriving_h800_tvm_s0_and_int8_probes_ms.csv` | s0 / micro-probe, 不是 full-model LUT |
| `pqs_ablation_results.json` 中 INT8 latency | 使用 stage0 speedup proxy, robust claim 是结构约束而非 full-backbone latency |

### 5.3 不应继续作为 claim 的旧证据

| 文件/结论 | 处理 |
|---|---|
| `results/gap1_schedule_lut.json` 中 pad64 ratio=1.0 | 已被 `gap1_grid_corrected.json` 的 pad64 ratio=7.706 取代 |
| 旧 C1/C7 中 “FP16 MetaSchedule 0 valid trials” | 已被 `C2_C3_pyramid.json` 修正 |
| 最近产品化 smoke 的 56.130 ms | 只作为 link-check/default-like row, 不作为 tuned LUT |
| CoDriving highdim trap original verdict | 已被 `C6_codriving_highdim.json` revised verdict 撤回 |

---

## 6. 对当前产品化 LUT 路线的影响

### 6.1 validation replay 应优先对齐 H800+TVM 旧实测

第一批 replay 不应只对齐 dataset_v2 的 4090+TRT。建议新增 H800 replay anchors:

| model | label | width | 旧 tuned ms | replay 目的 |
|---|---|---|---:|---|
| Pyramid | base | [64,128,256] | 6.320 | 验证 productized tuned path 是否能从 56ms default-like 回到 tuned |
| Pyramid | p50 | [32,64,128] | 3.011 | 验证 pruning width 与 tuned workdir/replay |
| Pyramid | trap25 | [48,96,192] | 21.615 | 验证失配点确实 low headroom |
| Pyramid | pad64 | [64,96,192] | 6.152 | 验证 W_g/P_g 机制 |
| CoDriving | base | [64,128,256] | 8.058 | 验证 standard conv negative anchor |
| CoDriving | p50 | [32,64,128] | 1.609 | 验证 CoDriving pruning+schedule |

通过条件:

1. 所有 replay 统一输出 ms。
2. 若使用已有 MetaSchedule DB, 必须记录 workdir digest。
3. 若 fresh tune, trials 不能再用 smoke 的 `2`; Pyramid base 至少应复现到 6-7 ms 区间, 否则只能标为 default-like。
4. 若 replay 值接近 56 ms, 应判定 productized path 未加载/未生成有效 tuning DB。

### 6.2 大规模 LUT 的起点应调整

新计划中的已有数据基础应从两层 seed 开始:

1. `existing/dataset_v2_ap_valid_63.csv`: 4090+TRT/AP/energy 历史 seed。
2. `existing_h800_tvm/*.csv`: H800+TVM latency 主 seed。

其中 latency 产品化路线应以 H800+TVM seed 为主, dataset_v2 只做 AP/结构和跨 backend sanity context。

---

## 7. 下一阶段工作计划

本节是清理上下文后直接启动实验的执行计划。目标不是马上补全大规模 LUT, 而是先证明产品化生成链路能复现历史 H800+TVM tuned 事实, 再进入真实 smoke。

### 7.1 阶段 0: 启动前审查

目的: 确认现有 seed、历史基线和 GPU 状态都可用。

输入:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/existing/dataset_v2_ap_valid_63.csv
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/existing_h800_tvm/h800_tvm_existing_summary.json
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/existing_h800_tvm/pyramid_h800_tvm_gap1_corrected_ms.csv
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/existing_h800_tvm/codriving_h800_tvm_backbone_ms.csv
```

审查命令:

```bash
cd ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1

tail -n +2 existing/dataset_v2_ap_valid_63.csv | wc -l
python -m json.tool existing_h800_tvm/h800_tvm_existing_summary.json | sed -n '1,140p'
for f in existing_h800_tvm/*.csv; do echo "$f"; tail -n +2 "$f" | wc -l; done
```

通过条件:

| gate | 通过条件 |
|---|---|
| seed-count | dataset_v2 AP-valid 行数为 63 |
| h800-summary | `h800_tvm_existing_summary.json` 可解析 |
| h800-latency | Pyramid direct latency seed 为 19 行, CoDriving backbone seed 为 4 行 |
| unit | 后续报告和 compare 一律使用 ms |

### 7.2 阶段 1: Validation replay

目的: 验证产品化 `generate_latency_lut` 能复现历史 H800+TVM tuned anchors, 并确认 dataset_v2 历史三轴 seed 没有被破坏。

输出目录:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/validation/h800_tvm_replay_v1/
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/validation/dataset_v2_replay_v1/
```

#### 7.2.1 H800+TVM latency replay anchors

第一批必须先跑 H800+TVM latency, 因为这是当前产品化链路最关键的缺口。

| priority | model | label | width | old default ms | old tuned ms | replay 判定 |
|---:|---|---|---|---:|---:|---|
| 1 | Pyramid | base | [64,128,256] | 56.321 | 6.320 | 必须证明不再只生成 56ms default-like |
| 2 | Pyramid | p50 | [32,64,128] | 26.242 | 3.011 | pruning anchor |
| 3 | Pyramid | trap25 | [48,96,192] | 42.355 | 21.615 | low-headroom / misaligned anchor |
| 4 | Pyramid | pad64 | [64,96,192] | 47.407 | 6.152 | W_g/P_g tuned anchor |
| 5 | CoDriving | base | [64,128,256] | 16.680 | 8.058 | negative anchor |
| 6 | CoDriving | p50 | [32,64,128] | 3.643 | 1.609 | CoDriving prune anchor |

输出文件:

```text
validation/h800_tvm_replay_v1/jobs/replay_job_plan_v1.jsonl
validation/h800_tvm_replay_v1/jobs/replay_job_state_v1.jsonl
validation/h800_tvm_replay_v1/latency/latency_lut_rows_v1.jsonl
validation/h800_tvm_replay_v1/compare/h800_tvm_replay_compare_v1.csv
validation/h800_tvm_replay_v1/compare/h800_tvm_replay_summary_v1.json
validation/h800_tvm_replay_v1/raw/
validation/h800_tvm_replay_v1/logs/
```

容差:

| anchor | 通过条件 |
|---|---|
| Pyramid base | tuned ms 必须落在 5.5-7.5 ms; 若接近 56 ms, 判定 tuned DB/MetaSchedule apply 失败 |
| Pyramid p50 | tuned ms 落在 2.6-3.6 ms |
| Pyramid trap25 | tuned ms 落在 18-25 ms |
| Pyramid pad64 | tuned ms 落在 5.4-7.2 ms |
| CoDriving base | tuned ms 落在 7.0-9.5 ms |
| CoDriving p50 | tuned ms 落在 1.3-2.0 ms |

若 fresh tune 而非复用历史 DB, 不允许使用 smoke 的 `trials=2`; Pyramid/Pyramid-like replay 至少使用可解释的 replay mode:

| mode | 用途 | 是否可进入 smoke |
|---|---|---|
| `reuse_existing_ms_db` | 使用历史 workdir/database 复现旧 tuned ms | yes |
| `fresh_tune_1000_trials` | 重调优, 时间较长, paper-grade 候选 | yes |
| `fresh_tune_smoke_trials` | 小 trials 链路检查 | no, 只能写 default-like/link-check |

#### 7.2.2 dataset_v2 三轴 replay

dataset_v2 replay 继续保留, 但不再作为 H800 latency 主依据。它用于检查 AP/energy/import/registry 兼容性。

| config_id | latency | AP | energy |
|---|---|---|---|
| `pyr_64-128-256_FP16` | 4090 TRT collab2 replay 或 canonical import check | DAIR val 1789 replay/import | E5/NVML replay/import |
| `pyr_64-128-256_INT8` | 4090 TRT collab2 replay 或 canonical import check | DAIR val 1789 replay/import | E5/NVML replay/import |
| `pyr_32-64-128_FP16` | 4090 TRT collab2 replay 或 canonical import check | DAIR val 1789 replay/import | E5/NVML replay/import |
| `pyr_32-64-128_INT8` | 4090 TRT collab2 replay 或 canonical import check | DAIR val 1789 replay/import | E5/NVML replay/import |

输出文件:

```text
validation/dataset_v2_replay_v1/jobs/replay_job_plan_v1.jsonl
validation/dataset_v2_replay_v1/jobs/replay_job_state_v1.jsonl
validation/dataset_v2_replay_v1/compare/dataset_v2_replay_compare_v1.csv
validation/dataset_v2_replay_v1/compare/dataset_v2_replay_summary_v1.json
```

通过条件:

1. 每个 replay row 必须记录 `comparison_mode`。
2. 若是真 replay, 写 `old_value_ms`, `new_value_ms`, `abs_error_ms`, `rel_error_pct`, `pass_fail`。
3. 若只能 import check, 写 `comparison_mode=canonical_import_check`, 不允许把它当成性能复测。
4. AP/energy 仍未真实 replay 时, validation 可以通过 H800 latency gate 进入 smoke, 但 calibration 仍保持 `NO-GO`。

### 7.3 阶段 2: 真实 smoke

目的: 证明产品化三轴 generator 能产出真实 row, 且 latency 已能进入 tuned 口径。

输出目录:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/
```

Smoke 配置:

| job group | model | config | width | axis | expected |
|---|---|---|---|---|---|
| S1 | Pyramid | smoke_base_fp16_tuned | [64,128,256] | latency | tuned ms 5.5-7.5 |
| S2 | Pyramid | smoke_p50_fp16_tuned | [32,64,128] | latency | tuned ms 2.6-3.6 |
| S3 | Pyramid | smoke_pad64_fp16_tuned | [64,96,192] | latency | tuned ms 5.4-7.2 |
| S4 | Pyramid | smoke_base_fp16_ap | [64,128,256] | AP | true eval/import row, no placeholder |
| S5 | Pyramid | smoke_p50_fp16_ap | [32,64,128] | AP | true eval/import row, no placeholder |
| S6 | Pyramid | smoke_base_fp16_energy | [64,128,256] | energy | true telemetry row or failed row |
| S7 | Pyramid | smoke_p50_fp16_energy | [32,64,128] | energy | true telemetry row or failed row |
| S8 | Pyramid | smoke_int8_stage0_probe | s0=64 stage0 | latency/mechanism | scope-limited row only |
| S9 | CoDriving | smoke_codriving_base_fp16 | [64,128,256] | latency | tuned ms 7.0-9.5 |

Smoke 输出:

```text
generated/smoke/h800_tvm_v1/jobs/job_plan.jsonl
generated/smoke/h800_tvm_v1/jobs/job_state.jsonl
generated/smoke/h800_tvm_v1/latency/latency_lut_rows_v1.jsonl
generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl
generated/smoke/h800_tvm_v1/energy/energy_lut_rows_v1.jsonl
generated/smoke/h800_tvm_v1/compare/smoke_summary_v1.json
generated/smoke/h800_tvm_v1/raw/
generated/smoke/h800_tvm_v1/logs/
```

Smoke 通过条件:

| gate | 通过条件 |
|---|---|
| latency-tuned | Pyramid base latency 不再是 56ms default-like, 必须接近 6.320ms |
| schema | latency/AP/energy rows schema validate 通过 |
| no-placeholder | AP/energy 不允许写 `not_run` placeholder 成 measured |
| raw-artifact | 每个 measured row 有 raw artifact |
| failure-policy | 失败必须写 failed row/job_state 和 `failure_reason` |
| registry | smoke rows 能被 evidence registry/coverage summary 读取 |

通过 smoke 后的判定:

```text
GO: 进入 calibration 的 latency-only 小规模补点
NO-GO: 完整 latency/AP/energy 大规模补点, 除非 AP 与 energy smoke 均真实通过
```

### 7.4 GPU 并行使用说明

H800 机器可见 8 张 GPU, 但不能假设都空闲。任何 job 启动前必须先做 preflight。

#### 7.4.1 每个 job 必须记录的 preflight

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate --format=csv
nvidia-smi pmon -c 1
```

每个 measured row 的 raw artifact 目录中必须保存:

```text
nvidia_smi_preflight.csv
nvidia_smi_pmon_preflight.txt
hostname.txt
command.json
env.json
```

#### 7.4.2 空闲判定

| 条件 | 判定 |
|---|---|
| target GPU `utilization.gpu` > 5% | 不启动 latency/energy measured job |
| target GPU 存在非本 job 计算进程 | 不启动 latency/energy measured job |
| target GPU memory used > 1024 MiB 且不是系统保留 | 不启动 latency/energy measured job |
| pstate/power 异常但无进程 | 写 warning, 可运行 validation; paper-grade 需复测 |

#### 7.4.3 并行策略

| job 类型 | 是否可并行 | 规则 |
|---|---|---|
| latency replay / latency smoke | 可跨 GPU 并行 | 每个 GPU 最多 1 个 latency job; 同一 GPU 不共享 |
| MetaSchedule fresh tune | 可跨 GPU 并行但慎用 | CPU/IO 也重, 建议最多 2-3 个并发; paper-grade 逐个复测 |
| AP eval | 可并行 | 不与同 GPU latency/energy 共用; 优先低于 latency replay |
| energy telemetry | 默认串行 | 只在 target GPU 完全空闲时测; paper-grade 建议整机低负载 |
| registry/import/compare | 可在 CPU 并行 | 不占 GPU |

#### 7.4.4 推荐执行顺序

第一轮不要一次性占满 8 张卡。推荐顺序:

1. 单卡验证 Pyramid base replay: 使用空闲 GPU, 先证明 `6.320 ms` tuned path 可复现。
2. 若 base 通过, 并行跑 Pyramid p50/trap25/pad64, 每个 job 独占一张空闲 GPU。
3. 同时或随后跑 CoDriving base/p50, 但不要挤占 Pyramid validation 的 GPU。
4. validation summary 通过后再进入 smoke。
5. smoke latency 可并行, AP 可并行, energy 先串行。

示例 GPU 分配, 仅在 preflight 显示空闲时使用:

| wave | GPU | job |
|---|---:|---|
| validation-0 | 3 | Pyramid base replay |
| validation-1 | 4 | Pyramid p50 replay |
| validation-1 | 5 | Pyramid trap25 replay |
| validation-1 | 6 | Pyramid pad64 replay |
| validation-1 | 7 | CoDriving base replay |
| validation-2 | 3 | CoDriving p50 replay |
| smoke-latency | 3/4/5 | Pyramid base/p50/pad64 latency |
| smoke-ap | 6/7 | AP eval/import |
| smoke-energy | 3 | energy telemetry, 串行 |

如果 GPU 0-2 忙碌, 直接跳过; 不要为追求并行度抢占训练/其他进程。

### 7.5 阶段完成后的交付物

验证阶段完成后必须新增:

```text
validation/h800_tvm_replay_v1/compare/h800_tvm_replay_summary_v1.json
validation/h800_tvm_replay_v1/compare/h800_tvm_replay_compare_v1.csv
validation/dataset_v2_replay_v1/compare/dataset_v2_replay_summary_v1.json
```

Smoke 阶段完成后必须新增:

```text
generated/smoke/h800_tvm_v1/compare/smoke_summary_v1.json
generated/smoke/h800_tvm_v1/latency/latency_lut_rows_v1.jsonl
generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl
generated/smoke/h800_tvm_v1/energy/energy_lut_rows_v1.jsonl
```

最终判定写入:

```text
${V2X_ROOT}/multi_agent/methods/design/auto-tuning/progress/HANDOFF_stage2_validation_smoke_results_v1_zh.md
```

判定格式:

| decision | 含义 |
|---|---|
| `GO_VALIDATION_ONLY` | H800 seed 可审查, 但 replay 未通过 |
| `GO_LATENCY_SMOKE` | H800 tuned latency replay 通过, 可跑 latency smoke |
| `GO_LATENCY_CALIBRATION_SMALL` | latency smoke 通过, 可小规模 latency-only calibration |
| `NO_GO_FULL_LUT` | AP 或 energy 未真实通过, 禁止三轴大规模 |
| `GO_FULL_LUT` | latency/AP/energy smoke 全部真实通过, 可进入 calibration 三轴补点 |

---

## 8. 快速结论给后续接手者

1. **是的, 历史 H800+TVM 已经支持“Pyramid backbone 可有约 10x tuned/default 提速”的初步结论。**
2. 最近产品化 smoke 的 56ms 不是反证, 它只是没有复现 tuned path, 数值与历史 default 56.321ms 对齐。
3. Pyramid 的 co-design 价值来自 grouped-conv alignment 造成的 schedule headroom 差异: trap25 只有 1.96x, pad64/base/p50 可到 7x-9x, p75 可到 26x。
4. CoDriving 是对照: standard conv H800+TVM 只有约 1.1x-2.3x, A-joint≈A-serial, 没有 robust high-dim trap。
5. 下一步真实 smoke 不应只看“能不能生成 row”, 而要把 Pyramid base replay 到 `6.320 ms` 量级作为产品化 tuned path 的硬门槛。
