# Stage2 H800 overnight LUT 实验交接 v1

日期: 2026-06-26

范围: Stage2 latency/AP/energy 产品化 LUT, H800+TVM Pyramid backbone-only / dense-core 过渡实验, 以及下一阶段进入可控大规模补点前的缺口收敛。

清空上下文后启动顺序:

1. 先读 `multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md`。
2. 再读本文档。
3. 再读 `multi_agent/methods/design/auto-tuning/progress/PLAN_stage2_multi_agent_lut_coverage_pipeline_v1_zh.md`。
4. 再读 `multi_agent/methods/design/auto-tuning/progress/PLAN_stage2_dataset_v2_aligned_lut_generation_v1_zh.md` 的章节 1-3。

## 1. 历史文档回顾与重复工作反思

本轮写交接前已回顾:

```text
PLAN_stage2_dataset_v2_aligned_lut_generation_v1_zh.md
HANDOFF_stage2_h800_tvm_existing_evidence_inventory_v1_zh.md
PLAN_stage2_6h_parallel_lut_generation_v1_zh.md
HANDOFF_stage2_three_arm_large_scale_readiness_plan_v1_zh.md
HANDOFF_stage2_three_arm_readiness_execution_v1_zh.md
HANDOFF_stage2_lut_p0_p1_p2_start_v1_zh.md
HANDOFF_stage2_lut_gpu012_progress_v1_zh.md
```

已经反复确认但容易被清空上下文后忘记的事实:

1. `dataset_v2.csv` 主要是 4090+TRT / Orin 历史证据, 不能当 H800 latency measured。
2. H800 产品化 smoke 的 56ms 是 default-like/link-check, 不是 tuned 结论; tuned base 应回到约 6.2-6.3ms。
3. 新增 H800 measured rows 必须走 `generate_*_lut` 主路径。
4. latency/energy 必须 GPU idle preflight; AP 不能用 predicted value 写 measured row。
5. bad DB / illegal memory / missing artifact / outlier 必须显式进入 job_state、quarantine 或 readiness gate。

本轮确实存在重复和执行偏差:

| 问题 | 表现 | 原因反思 | 已采取修正 |
|---|---|---|---|
| H800 连接信息反复丢失 | 清空上下文后重新找 host/port/password | 连接方式没有固定在当前 auto-tuning progress 文档中 | 新增 `RUNBOOK_stage2_h800_server_access_v1_zh.md` |
| AP source 远端缺失 | 第一次 AP job plan 的 true source 为空 | 本地已有 AP anchor, 但没有同步到 H800 | 已同步 true AP anchor 并重生成 AP plan |
| preflight parser 误判 | `4 MiB` / `0 %` 被当作 malformed, latency/energy 首批全 blocked | parser 假设 `nvidia-smi` 输出纯数字, 与实际 CSV 带单位不一致 | 已修复 `scripts/stage2_lut_worker.py` 的数字提取 |
| “6 小时”理解偏差 | 第一批 finite queue 很快跑完, H800 上看不到进程 | `--max-hours` 是上限, 不是保活; queue 太短 | 已追加 continuation queue, 并在 runbook 写明 |
| 已知风险 config 仍被重复触发 | `pad64`/`mix_a` continuation 中多次 illegal memory | continuation 依据 succeeded source job 机械扩展, 没有把历史风险完全纳入前置 artifact gate | 当前已 quarantine; 下一步必须 registry-preflight 后再生成 queue |
| finalizer v1 会清空 quarantine | 初版脚本有 `: > "$Q"` | 收尾脚本设计时把空文件初始化和保留历史混淆 | 已替换为 v2 finalizer, 不清空 quarantine |

## 2. 本轮实验进展

### 2.1 运行位置

H800 原始目录:

```text
H800:/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626
```

本地已同步副本:

```text
/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626
```

关键输出:

```text
exports/current_experiment_report_v1.json
exports/quick_review_v1.csv
exports/latency_outlier_report_v1.json
exports/readiness_gate_latest.json
registry/artifact_registry_summary_v1.json
registry/artifact_registry_v1.jsonl
registry/evidence_registry_latest.json
merged/latency_lut_rows_merged_v1.jsonl
merged/ap_anchor_rows_merged_v1.jsonl
merged/energy_lut_rows_merged_v1.jsonl
```

### 2.2 job plan 与执行

初始三组总 plan:

| plan | jobs | 说明 |
|---|---:|---|
| `latency_gpu012_job_plan.jsonl` | 19 | 实际拆到 GPU0/1/2 |
| `ap_gpu45_job_plan.jsonl` | 19 | 实际拆到 GPU4/5 |
| `energy_gpu3_job_plan.jsonl` | 17 | GPU3 串行 |

continuation plan:

| plan | jobs | 说明 |
|---|---:|---|
| `latency_cont_gpu01245_job_plan.jsonl` | 432 | 24 passes x 18 source jobs, 分布到 GPU0/1/2/4/5 |
| `energy_cont_gpu3_job_plan.jsonl` | 384 | 24 passes x 16 source jobs, GPU3 |

AP 未做 continuation, 因为当前 AP 是 true anchor import, 重复导入不会增加新实测信息。

### 2.3 当前数据规模

本轮新增:

| 数据 | rows | 说明 |
|---|---:|---|
| latency | 844 | H800+TVM, default/tuned, 包含 continuation repeats |
| AP | 11 | true AP anchor import, 非 predicted |
| energy | 400 | H800 telemetry rows |
| quarantine | 9 | `mix_a` / `pad64` bad DB / illegal memory 相关 |

合并历史 seed 后:

| 数据 | rows |
|---|---:|
| merged latency | 888 |
| merged AP | 22 |
| merged energy | 408 |
| merged job plan | 871 |
| artifact registry | 78 artifacts |

Artifact registry summary:

| 指标 | 值 |
|---|---:|
| total artifacts | 78 |
| ready artifacts | 76 |
| quarantined artifacts | 2 |
| missing artifacts | 0 |
| blocked artifacts | 0 |

Readiness gate:

| gate | 状态 |
|---|---|
| schema | pass |
| energy | pass |
| evidence_registry | pass |
| outlier | conditional |
| artifact_registry | fail |
| final decision | `NO_GO` |

解释: `NO_GO` 不是数据未生成, 而是 artifact registry 发现 `mix_a` 和 `pad64` 相关 jobs/config 被 quarantine; outlier gate 仍有历史 `trap25` unstable repeat。

## 3. 当前核心结果表

单位: latency 为 ms, energy 为 J/inference。表中 latency/energy 是本轮 current report 按 label 聚合的 median; AP 是 true anchor。

| label | width | tuned ms | default ms | speedup | AP70 | energy J | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| base | 64/128/256 | 6.211 | 56.305 | 9.06x | 0.6309 | 0.9245 | tuned replay 成功, 与历史 6.3ms 对齐 |
| p50 | 32/64/128 | 3.003 | 26.215 | 8.73x | 0.5641 | 0.2672 | 轻量主锚 |
| p75 | 16/32/64 | 0.482 | 12.678 | 26.30x | 0.5300 | 0.0756 | energy 有极低值, paper 前需 quiet retest |
| trap25 | 48/96/192 | 21.614 | 42.346 | 1.96x | 0.5905 | 2.4361 | low-headroom; 历史 outlier 仍 conditional |
| pad64 | 64/96/192 | 6.157 | 47.396 | 7.70x | - | - | W_g/P_g anchor, 但 continuation 中 bad DB quarantine |
| iso_s0 | 48/128/256 | 21.740 | 51.228 | 2.36x | 0.6299 | 2.2369 | latency/energy 有数据 |
| iso_s1 | 64/96/256 | 6.915 | 51.612 | 7.46x | 0.6336 | 0.9913 | 高 AP 附近 S anchor |
| iso_s2 | 64/128/192 | 7.253 | 51.986 | 7.17x | 0.6339 | 0.9699 | 高 AP 附近 S anchor |
| s0_16 | 16/128/256 | 5.851 | 39.454 | 6.74x | - | 0.6691 | 缺 AP |
| s0_32 | 32/128/256 | 5.976 | 44.658 | 7.47x | - | 0.8192 | 缺 AP |
| s1_32 | 64/32/256 | 3.763 | 41.865 | 11.12x | - | 0.5253 | 缺 AP |
| s2_64 | 64/128/64 | 5.038 | 43.570 | 8.65x | - | 0.6871 | 缺 AP |
| s2_128 | 64/128/128 | 5.507 | 47.438 | 8.61x | - | 0.7915 | 缺 AP |
| mix_a | 32/96/192 | 5.660 | 35.811 | 6.33x | 0.6288 | - | latency/AP 有, energy/bad DB quarantine |
| mix_b | 48/64/256 | 19.404 | 41.446 | 2.14x | 0.6362 | 1.9725 | 高 AP 但 slow tuned |
| mix_c | 16/128/128 | 5.109 | 30.741 | 6.02x | - | 0.6721 | 缺 AP |
| mix_d | 48/128/128 | 21.056 | 42.435 | 2.02x | 0.6369 | 2.3604 | 高 AP 但 slow tuned |
| mix_e | 64/64/128 | 14.663 | 42.759 | 2.92x | - | 1.8663 | 缺 AP |
| p50b2_136 | 32/64/136 | - | - | - | 0.6425 | - | 有 AP, H800 latency artifact 缺失 |

## 4. 当前空白

### 4.1 AP 覆盖空白

已有 true AP:

```text
base, p50, p75, trap25, p50b2_136, iso_s0, iso_s1, iso_s2, mix_a, mix_b, mix_d
```

仍缺 true AP:

```text
pad64, s0_16, s0_32, s1_32, s2_64, s2_128, mix_c, mix_e
```

原则: 缺 AP 只能写 failed/no-claim, 不能用 predicted AP 写 measured AP。

### 4.2 Artifact / bad DB 空白

| config | 当前状态 | 影响 |
|---|---|---|
| `mix_a` | energy 与部分 continuation latency 触发 CUDA illegal memory, artifact registry quarantined | 不能 paper-grade claim energy; 后续需重建/替换 workdir |
| `pad64` | 已有成功 latency row, 但 continuation repeat 多次触发 CUDA illegal memory, artifact registry quarantined | W_g/P_g 机制仍有历史证据, 但当前 productized artifact 需修复 |
| `p50b2_136` | true AP 存在, H800 ONNX/workdir 缺失 | 不能补 H800 latency/energy |
| `s1_64` | 历史 tuned illegal memory 风险 | 继续默认 exclude/quarantine |

### 4.3 Outlier 空白

唯一 unstable latency row:

```text
config_id=calibration_h800_tvm_pyramid_trap25_fp16_metaschedule_tuned
reason=repeat_max_min_ratio=2.168>2.000
claim_status=no_claim
```

本轮 overnight 的 trap25 median 回到约 21.6ms, 但历史 outlier 仍保留在 merged latency 中, 所以 gate 仍是 conditional。下一步需要把 clean trap25 rows 与 unstable run 分开做 promotion/quarantine。

### 4.4 Energy 质量空白

energy 链路已经产品化并能大规模产出, 但 paper-grade claim 还需要更严格的质量控制:

1. 当前 energy 是 calibration_parallel, 不是 paper quiet retest。
2. 部分配置 energy 分布跨度大, 如 base/p75 存在较低 min 值。
3. 需要 per-config energy outlier / repeat quality gate, 类似 latency outlier detector。
4. 关键 paper 表应在整机低负载、固定 GPU/clock/power policy 下复测。

### 4.5 Worker / registry 空白

1. 当前 artifact registry 是收尾后构建, 还没有成为 worker 启动前的强 preflight gate。
2. job_state 用 append-only 记录 `running -> succeeded/failed`, 汇总时容易把历史 `running` 当活动进程; 需要 latest-status summarizer。
3. continuation queue 是人工补救生成, 缺少按 time budget 自动 top-up 的 job plan generator。
4. 若 H800 上看不到进程, 需要先查 row counts/state, 不应默认认为没有启动。

## 5. 下一步计划

### P0: 文档与执行机制收敛

1. 后续所有 Stage2 H800 任务先读 `RUNBOOK_stage2_h800_server_access_v1_zh.md`。
2. 新增的覆盖优先流水线计划见 `PLAN_stage2_multi_agent_lut_coverage_pipeline_v1_zh.md`; 后续不再用 24 轮 continuation 保活。
3. 将 `stage2_lut_worker.py` preflight parser fix 纳入测试, 覆盖 `4 MiB` / `0 %` 格式。
4. 写 latest-status summarizer, 只按每个 `job_id` 最后一条 state 汇总。
5. 把 artifact registry preflight 接入 worker/job generator: quarantined/missing artifact 不进入 queue。
6. 将 finalizer v2 固化, 禁止清空已有 quarantine rows。

### P1: 数据缺口补齐

1. AP fresh eval/import:
   - `pad64`
   - `s0_16`, `s0_32`
   - `s1_32`
   - `s2_64`, `s2_128`
   - `mix_c`, `mix_e`
2. `p50b2_136` artifact unblock:
   - 定位或重新导出 `p50b2_136_backbone.onnx`。
   - 生成/定位对应 TVM workdir。
   - registry ready 后再测 latency/energy。
3. `mix_a` / `pad64` bad DB 处理:
   - 不再用当前 quarantined workdir 批量 repeat。
   - 重建 MetaSchedule DB 或回退到 known-good historical workdir 单点验证。
   - 修复后先单 GPU 单点测试, 不直接进 continuation。
4. trap25 outlier:
   - 用 current clean overnight rows 作为候选 clean evidence。
   - 把历史 unstable run 关联 quarantine/no-claim。
   - 重新跑 outlier gate, 目标由 conditional 变成 pass。

### P2: Paper-grade 复测

只选核心点, 不再盲目扩 repeat:

```text
base, p50, p75, trap25, iso_s1, iso_s2, mix_b, mix_d
```

要求:

1. latency/energy 在目标 GPU 完全空闲时运行。
2. energy 不与 latency 共享同一 GPU; paper-grade energy 尽量整机低负载。
3. 每个关键配置保留 raw repeat samples、preflight、power samples、command、env、hostname。
4. 输出 claimable/no-claim 表, 不只输出 quick_review。

### P3: 三臂大规模启动 gate

当前结论:

```text
full unattended P/Q/S large-scale LUT production: NO_GO
```

解除条件:

1. artifact registry gate pass: `mix_a`/`pad64` 要么修复 ready, 要么从当前 production queue 移除且不阻断。
2. outlier gate pass 或只有明确隔离的 conditional。
3. AP missing source 不再影响主候选: 要么补齐, 要么明确 no-claim。
4. energy repeat quality gate 建立, paper 表只使用 claimable energy。
5. Q arm 至少 base INT8、p50 INT8 两个 config 进入 artifact registry ready, 否则不能称为三臂。

## 6. 快速审查命令

本地副本:

```bash
cd /home/jichengzhi/V2X
BASE=multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626
python3 -m json.tool "$BASE/exports/readiness_gate_latest.json" | sed -n '1,160p'
python3 -m json.tool "$BASE/registry/artifact_registry_summary_v1.json"
head -n 30 "$BASE/exports/quick_review_v1.csv"
wc -l "$BASE"/merged/*.jsonl "$BASE"/registry/artifact_registry_v1.jsonl
```

H800 远端:

```bash
export SSHPASS='<H800_PASSWORD>'
sshpass -e ssh -p 30001 -o StrictHostKeyChecking=no -o ConnectTimeout=60 jichengzhi@222.95.84.215 \
  'cd /home/jichengzhi/V2X && BASE=multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626 && python3 -m json.tool "$BASE/exports/current_experiment_report_v1.json" | sed -n "1,120p"'
unset SSHPASS
```
