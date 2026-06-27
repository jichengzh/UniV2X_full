# Stage2 original60 H800 measured LUT 交接文档 v1

文档编号：`1_6_26_交接文档_阶段二_original60_H800实测LUT进展`

更新时间：2026-06-26

## 0. 后续交接文档命名规则

从下一份交接文档开始，交接文档统一使用中文文件名，不再新增 `HANDOFF_*` 英文命名文件。文件名和文档开头必须包含“序号_月_日_交接文档_阶段_主题”的编号；新的一天序号从 `1` 重新开始，同一天内依次递增。

推荐格式：

```text
1_6_26_交接文档_阶段二_original60_H800实测LUT进展.md
2_6_26_交接文档_阶段二_quarantine配置复测计划.md
1_6_27_交接文档_阶段二_下一批候选工程产物进展.md
```

后续文档正文第一行也必须显式写出同一个文档编号，便于清空上下文后快速定位最新交接状态。

## 1. 当前结论

本轮已经基于 `original60` 已完成工程产物，在 H800 上跑完 measured LUT 的首轮生产与补测。

优先级更新：下一阶段主目标已经切换为 AP 稳定测试与 AP source contract 产品化。启动下一轮前应优先阅读：

`multi_agent/methods/design/auto-tuning/progress/2_6_26_交接文档_阶段二_AP稳定测试优先计划.md`

该文档覆盖 AP 不合理/不可用的根因、AP source registry、replay gate、stable smoke、original60 AP source map 和新的 `/goal` stop 条件。

| 项目 | 当前状态 | 说明 |
| --- | ---: | --- |
| original60 工程产物 | 60/60 ready | 使用 `artifact_registry_original60_v1.jsonl`，未覆盖该 registry |
| latency rows | 116 | 58 个候选，每个候选 `default` + `metaschedule_tuned` 两行 |
| latency 候选覆盖 | 58/60 | 2 个候选两次 CUDA illegal memory access，进入 latency quarantine |
| energy rows | 56 | 56 个 latency measured 候选完成 energy telemetry |
| energy 候选覆盖 | 56/58 | 2 个候选两次 energy CUDA illegal memory access，进入 energy quarantine |
| AP rows | 0 | 当前无 true AP source；60 个候选均为 no-claim gap |
| latency outlier gate | pass | `unstable_rows=0`, `claimable_rows=116` |
| local schema/readiness gate | GO | artifact/schema/outlier/energy gates 均 pass |
| supervisor gate | CONDITIONAL_GO | 条件项为 AP source 缺失和当前无下一批 latency ready queue |

注意：本轮 latency 单位统一为 ms；energy 单位为 J/inference。当前 measured rows 是 H800 + TVM + backbone-only，不是端到端 pipeline latency。

重要修正：当前 original60 的 `metaschedule_tuned` 不应解释为历史 `MS-1000 tuned` 复现。本轮 original60 工程产物的 artifact worker 默认 `max_trials=32`，远端只读复核显示这批 workdir 多数只有约 32 条 tuning record；历史 Pyramid 10x 证据使用的 `ms_work_2e_base/p50/pad64` 约 1030 条 tuning record。因此当前 original60 的 tuned/default 比值只能表示“当前 shallow generated DB 在产品化链路下的实测结果”，不能直接支撑历史 10x tuned/default 结论。

当前 original60 tuned/default 统计：

| 指标 | 数值 | 含义 |
| --- | ---: | --- |
| paired candidates | 58 | 同时有 default 和 tuned latency 的候选 |
| default/tuned median | 1.00x | 当前 shallow tuned 基本未优于 default |
| default/tuned mean | 1.08x | 少数点有 1.2-1.46x 改善 |
| default/tuned max | 1.46x | 远低于历史 Pyramid base 10.34x / p50 8.74x |

与历史证据的关键差异：

| 口径 | 历史 10x 证据 | 当前 original60 |
| --- | --- | --- |
| default 定义 | `relax.build` 默认 GPU lowering / dlight-default 旋钮关基线 | 同类口径，仍需 golden anchor 校验 |
| tuned DB | `ms_work_2e_*`，约 MS-1000 级 tuning | `workdirs/<label>`，默认 32 trials，这是当前最主要差异 |
| 典型结果 | Pyramid base 64.17ms -> 6.21ms | 多数点 tuned/default 约 1.0x |
| 证据等级 | 可用于 10x 调度收益主张 | 仅可用于 shallow DB 覆盖和预测器初始观测 |

后续任何“tuned 加速收益”声明必须先通过 golden anchor gate：用当前产品化测量脚本复跑历史 `base_backbone.onnx + ms_work_2e_base`，确认 tuned 回到约 6.2ms；再对一个 original60 同宽或邻近点做高 budget retune，确认 32 trials 到 1000 trials 是否恢复大幅收益。

## 2. 关键路径

数据根目录：

`/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1`

关键输出：

| 文件 | 作用 |
| --- | --- |
| `rows/latency_lut_rows_original60_v1.jsonl` | H800 measured latency LUT rows，116 行 |
| `rows/energy_lut_rows_original60_v1.jsonl` | H800 measured energy LUT rows，56 行 |
| `exports/original60_measured_lut_summary.md` | 可直接审查的 60 候选汇总表 |
| `exports/original60_measured_lut_summary.csv` | 同上，CSV 版本 |
| `exports/original60_latency_outliers.json` | latency outlier gate，当前 0 unstable |
| `exports/original60_readiness_gate_local.json` | 本地 schema/readiness gate，当前 GO |
| `exports/original60_supervisor_final/readiness_gate_latest.json` | 最终 supervisor dashboard |
| `exports/original60_ap_axis_gap_report.json` | AP no-claim gap，60 条 |
| `exports/original60_energy_axis_gap_report_final.json` | final energy gap/quarantine 状态 |
| `quarantine/bad_db_quarantine_original60_v1.jsonl` | latency quarantine；`frontier_25` 旧 latency 失败已标记 resolved |
| `quarantine/bad_db_quarantine_original60_energy_v1.jsonl` | energy quarantine |

H800 访问：

```bash
ssh -p 30001 jichengzhi@222.95.84.215
cd /home/jichengzhi/V2X
```

密码使用用户在会话中提供的值，不写入文档、脚本或日志。远端偶发 `kex_exchange_identification: Connection closed by remote host`，按 20-30 秒低频重试即可。

## 3. 当前实测表格

完整 60 候选表格见：

`multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/exports/original60_measured_lut_summary.md`

表格字段：

| 字段 | 含义 |
| --- | --- |
| `label` | 候选短名 |
| `candidate_id` | 原始候选点 ID |
| `width` | dense core/backbone width |
| `latency_default_ms` | TVM default schedule backbone-only latency |
| `latency_tuned_ms` | TVM MetaSchedule tuned backbone-only latency |
| `energy_j_per_inference` | H800 power telemetry energy |
| `latency_status` | `measured` 或 `quarantine_cuda_illegal_memory_access` |
| `energy_status` | `measured`、`not_run_latency_quarantine` 或 `quarantine_cuda_illegal_memory_access` |
| `ap_status` | 当前均为 `no_claim_missing_true_source` |

异常/未完成候选：

| label | width | latency | energy | 原因 |
| --- | --- | --- | --- | --- |
| `s2_096` | `64x128x96` | quarantine | not run | latency 初测和 retry2 均 CUDA illegal memory access |
| `lhc_17` | `56x48x256` | quarantine | not run | latency 初测和 retry2 均 CUDA illegal memory access |
| `frontier_25` | `64x96x224` | measured | quarantine | latency retry2 成功；energy 初测和 retry1 均 CUDA illegal memory access |
| `frontier_26` | `56x80x160` | measured | quarantine | latency 成功；energy 初测和 retry1 均 CUDA illegal memory access |

## 4. 本轮执行摘要

1. 确认 `original60_artifact_ready_count=60`、`missing_artifact_count=0`。
2. 生成 latency job plan：GPU0/1/4/5 各 15 个 job，共 60 个。
3. 初跑 latency：57/60 成功；`s2_096`、`lhc_17`、`frontier_25` 初测失败。
4. latency retry2：`frontier_25` 成功；`s2_096`、`lhc_17` 仍失败，保留 latency quarantine。
5. 生成 AP plan：无 true source，0 AP job，60 no-claim gap。
6. 生成 energy follower：基于 58 个 tuned latency measured 候选，生成 58 个 energy job，分片 GPU0-5。
7. energy 初跑：51/58 成功；5 个 GPU2 preflight_blocked，2 个 CUDA crash。
8. energy retry1：5 个 preflight_blocked 补齐成功；`frontier_25`、`frontier_26` 再次 CUDA crash，保留 energy quarantine。
9. 同步 H800 结果回本地，并回写最终 summary/gap/quarantine 到 H800。

## 5. 验证结果

本地复核命令：

```bash
BASE=multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1
python3 scripts/stage2_readiness_gate.py \
  --artifact-registry "$BASE/artifacts/artifact_registry_original60_v1.jsonl" \
  --latency-rows "$BASE/rows/latency_lut_rows_original60_v1.jsonl" \
  --energy-rows "$BASE/rows/energy_lut_rows_original60_v1.jsonl" \
  --outlier-report "$BASE/exports/original60_latency_outliers.json" \
  --out-json "$BASE/exports/original60_readiness_gate_local.json"
```

当前结果：

```text
decision=GO
schema=pass
artifact_registry=pass
outlier=pass
energy=pass
```

Supervisor 当前为 `CONDITIONAL_GO`，原因：

1. `ap_axis_lag`：AP 仍缺 true eval/import source。
2. `latency_queue_low`：本轮 original60 队列已跑完，没有下一批 ready latency queue。

这两个条件不是本轮 original60 measured LUT 链路失败；它们分别指向下一阶段 AP source 接入和下一批候选扩展。

## 6. 下一阶段计划

P0：保留当前 original60 measured LUT 为预测器初始训练/校准数据，但降级其 tuned 证据等级。

1. 使用 `exports/original60_measured_lut_summary.csv` 作为人工审查入口。
2. 使用 `rows/latency_lut_rows_original60_v1.jsonl` 和 `rows/energy_lut_rows_original60_v1.jsonl` 作为训练/拟合输入。
3. AP 不得用预测值填充；在 true AP source 接入前只保留 no-claim gap。
4. 不得把 original60 当前 `metaschedule_tuned` 当成历史 `MS-1000 tuned`；训练时应记录 `tune_budget=32` 或等价 provenance。

P1：处理 4 个 quarantine 配置。

1. 先做 golden anchor gate：当前产品化脚本 + 历史 `ms_work_2e_base` 必须复现 Pyramid base tuned 约 6.2ms。
2. 对一个 original60 代表点执行高 budget retune gate：至少从 32 trials 扩到 1000 trials，并记录 tuned/default 是否显著恢复。
3. 对 `s2_096`、`lhc_17` 检查 ONNX shape、TVM DB tuning record、生成的 TIR 是否存在非法访问风险。
4. 对 `frontier_25`、`frontier_26` 分离 latency 与 energy 代码路径，确认是 energy warmup/active loop 触发还是 VM 本身触发。
5. 修复后只对这 4 个配置建小队列复测，不重跑 56 个已成功配置。

P2：开始下一批候选覆盖。

1. Artifact agent 继续生成下一批候选工程产物。
2. Latency agent 在 artifact ready 后立即跑 coverage-first measured latency。
3. Energy agent 跟随 tuned latency rows，优先补齐同一候选集合。
4. AP agent 只接 true eval/import；没有 source 时继续 no-claim。
5. Supervisor 继续以 `repeat_ratio=0`、outlier pass、quarantine 可解释为启动门槛。

## 7. 下一次 /goal 命令

当前推荐使用 AP 优先版 `/goal`，详见：

`multi_agent/methods/design/auto-tuning/progress/2_6_26_交接文档_阶段二_AP稳定测试优先计划.md`

旧版 latency/tuned 口径 gate 暂不作为下一阶段首要目标，保留为 AP 稳定链路跑通后的后续质量检查。

```text
/goal 在 /home/jichengzhi/V2X 中继续 Stage2 LUT 工作，但下一阶段优先解决 AP 不合理/不可用问题，目标是可以稳定测试不同配置对应的 AP。先阅读 multi_agent/methods/design/auto-tuning/progress/2_6_26_交接文档_阶段二_AP稳定测试优先计划.md，再阅读 RUNBOOK_stage2_h800_server_access_v1_zh.md。不要优先扩展 latency/energy；不要把 predicted/model-fit/interpolated AP 写成 measured。首要任务是建立 ap_source_registry_v1.jsonl，把 stage_a_ap_real.csv、results/ap70_depgraph_expansion.json、results/b2_eval/*.json、results/p0_2_136/p50b2_136_fp16.json 和现有 ap_anchor_rows 统一分类为 true_eval/true_import/weight_identity_transfer/model_fit_only/no_claim。随后执行 AP replay gate：base、p50、trap25、mix_d 至少 3 个 anchor 复现 AP70，absolute diff <= 0.003。通过后执行 Phase C stable smoke：先给 base/p50/p75/trap25/mix_b/mix_d/iso_s0/iso_s1/iso_s2/p50b2_136 生成 canonical AP rows；再启动 original60 新配置 5GPU 并行微调，GPU0-4 分别跑 s0_024、s0_040、s0_056、s1_048、s2_160，每配置默认 1 次 finetune，patch config 为 epoches=31，即 init@23 到 epoch31 的 8 个 finetune epochs；微调不要求 GPU 完全空闲，只要求显存足够且无 OOM 风险。每个新配置完成 structural_prune_pyramid -> flat ckpt check -> train_ddp --half -> ONNX export -> TRT FP16 build -> DAIR-V2X val_1789 AP eval，并写入 ap_stability_20260626/rows/ap_anchor_rows_v1.jsonl。最后生成 original60_ap_source_map_v1.jsonl 和 original60_ap_axis_gap_report_v2.json，明确 60 个 original60 配置中哪些可以测 AP、哪些必须 no-claim、哪些需要训练/finetune。stop 条件：AP source registry 完成；replay gate 有明确 pass/fail；10 个已有 source 配置完成 canonical AP rows；5 个 original60 新配置完成 finetune+AP eval 或形成明确 quarantine reason；至少 8 个不同配置产生 claimable/provisional AP row；original60 AP gap report 更新；所有结果 rsync 回本地并写新的中文交接文档。
```
