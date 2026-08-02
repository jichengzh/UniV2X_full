# 1_6_27_交接文档_阶段二_AP稳定测试PhaseC实测结果

日期: 2026-06-27

序号: 1

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/0_6_27_冷启动入口_阶段二LUT.md`
- `multi_agent/methods/design/auto-tuning/progress/6_26/2_6_26_交接文档_阶段二_AP稳定测试优先计划.md`
- `multi_agent/methods/design/auto-tuning/progress/6_26/5_6_26_交接文档_阶段二_PhaseC_AP稳定产出规范.md`
- `multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md`

## 0. 后续交接文档命名规则
从下一份文档开始，交接文档统一存储进入日期对应的文件夹。例如 `1_6_27_交接文档_阶段二_AP稳定测试PhaseC实测结果.md` 存入 `multi_agent/methods/design/auto-tuning/progress/6_27/`。如果进入新的一天，创建新的日期文件夹并将新交接文档存入其中。清空上下文后的统一入口是 `multi_agent/methods/design/auto-tuning/progress/6_27/0_6_27_冷启动入口_阶段二LUT.md`。

## 1. 本轮结论

本轮完成了 Stage2 AP 稳定测试链路的关键推进:

1. H800 AP 环境恢复完成: HEAL、UniV2X_2.0、DAIR base checkpoint、AP scripts 均可用。
2. DAIR-V2X-C train/val required files 已同步到 H800, 使用 required-files 清单避免同步无关数据。
3. `ap_source_registry_v1.jsonl` 已建立, registry 共 32 行。
4. AP replay gate 有明确结果: import/replay gate `pass`, 4/4 anchor 通过, threshold `abs(AP70 diff) <= 0.003`。
5. 10 个已有 source 配置完成 canonical AP rows。
6. Phase C 5GPU stable smoke 已真实启动并完成:
   - 4 个 original60 新配置完成 `structural_prune -> train_ddp --half -> ONNX -> TRT FP16 -> DAIR val_1789 AP eval`。
   - 1 个 original60 新配置 `s2_160` 完成训练和 ONNX 导出, 但 ONNXRuntime sanity fail, 已 quarantine, 没有写 measured AP。
7. 本地结果已从 H800 rsync 回 `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/`。

重要限制: 本轮新 AP 是真实 measured AP, 但不是 paper-grade final AP。`s0_024` 和 `s1_048` 的 AP70 高于 base, 触发趋势异常复测规则, 下一阶段必须补第二 seed 或延长训练验证, 不得直接写成“剪枝自然提升 AP”的结论。

## 2. 关键产物

根目录:

```text
multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/
```

关键文件:

| 文件 | 状态 | 说明 |
|---|---:|---|
| `registry/ap_source_registry_v1.jsonl` | 32 行 | AP source registry |
| `exports/ap_replay_gate_report.json` | pass | 4/4 import/replay gate pass |
| `rows/ap_anchor_rows_v1.jsonl` | 14 行 | 10 个已有 source + 4 个 Phase C true_eval |
| `jobs/ap_finetune_smoke_job_state_v1.jsonl` | 10 行 | 5 个 start + 4 succeeded + 1 failed |
| `quarantine/ap_unstable_or_unclaimable_v1.jsonl` | 1 行 | `s2_160` ONNX sanity fail |
| `exports/original60_ap_source_map_v1.jsonl` | 60 行 | original60 AP source map 已更新 |
| `exports/original60_ap_axis_gap_report_v2.json` | 已更新 | 4 exact true source, 1 quarantine, 54 no-claim, 1 transfer |
| `exports/ap_stability_summary.md` | 已更新 | 可直接审查 AP30/AP50/AP70 和 claim_status |
| `exports/ap_quality_gate_report.json` | 已生成 | 标注 first-seed、趋势异常待 repeat、quarantine |
| `exports/ap_quality_gate_report.md` | 已生成 | 人读版质量门控表 |
| `raw/<label>/` | 已同步 | ONNX/TRT/AP eval logs 和 raw reports |
| `../coverage_pipeline_v1/exports/original60_three_metric_summary_latest.md` | 已补充 | Original60 AP/Latency/Energy 三指标总表, 人读版 |
| `../coverage_pipeline_v1/exports/original60_three_metric_summary_latest.csv` | 已补充 | Original60 AP/Latency/Energy 三指标总表, CSV 版 |
| `../coverage_pipeline_v1/exports/original60_three_metric_summary_latest.json` | 已补充 | Original60 AP/Latency/Energy 三指标总表, 带 sources/summary |
| `scripts/stage2_export_original60_three_metric_summary.py` | 已新增 | 三指标总表生成器, 用于后续重复刷新 |

## 3. 当前 AP 表

### 3.1 已有 source/import anchor

| label | width | AP30 | AP50 | AP70 | source_kind | claim_status |
|---|---|---:|---:|---:|---|---|
| base | [64,128,256] | 0.833159 | 0.791041 | 0.630864 | true_import | provisional_missing_digest |
| p50 | [32,64,128] | 0.818254 | 0.764428 | 0.564132 | true_import | provisional_missing_digest |
| p75 | [16,32,64] | 0.815909 | 0.756669 | 0.529988 | true_import | provisional_missing_digest |
| trap25 | [48,96,192] | 0.824434 | 0.776939 | 0.590472 | true_import | provisional_missing_digest |
| mix_b | [48,64,256] | 0.818581 | 0.780496 | 0.636160 | true_eval | provisional_missing_digest |
| mix_d | [48,128,128] | 0.825772 | 0.785835 | 0.636938 | true_eval | provisional_missing_digest |
| iso_s0 | [48,128,256] | 0.823978 | 0.784323 | 0.629919 | true_eval | provisional_missing_digest |
| iso_s1 | [64,96,256] | 0.830391 | 0.789614 | 0.633599 | true_eval | provisional_missing_digest |
| iso_s2 | [64,128,192] | 0.821267 | 0.783871 | 0.633890 | true_eval | provisional_missing_digest |
| p50b2_136 | [32,64,136] | 0.827085 | 0.787692 | 0.642530 | true_eval | provisional_missing_digest |

### 3.2 Phase C original60 新配置

| label | width | GPU | finetune_runs | epoches | AP30 | AP50 | AP70 | state | 质量门控 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| s0_024 | [24,128,256] | 0 | 1 | 31 | 0.833141 | 0.791044 | 0.632495 | succeeded | AP70 高于 base, 必须 second seed/repeat |
| s0_040 | [40,128,256] | 1 | 1 | 31 | 0.822729 | 0.781245 | 0.626659 | succeeded | first-seed true_eval, 建议补 repeat |
| s0_056 | [56,128,256] | 2 | 1 | 31 | 0.832072 | 0.790659 | 0.623711 | succeeded | first-seed true_eval, 建议补 repeat |
| s1_048 | [64,48,256] | 3 | 1 | 31 | 0.822220 | 0.782852 | 0.635616 | succeeded | AP70 高于 base, 必须 second seed/repeat |
| s2_160 | [64,128,160] | 4 | 1 | 31 |  |  |  | failed/quarantine | ONNXRuntime sanity fail, 未写 measured AP |

说明:

1. 4 个 succeeded 行均来自 DAIR-V2X-C `val_1789`, `num_samples=1789`。
2. 4 个 succeeded 行均写入 `rows/ap_anchor_rows_v1.jsonl`, `claim_status=claimable_true_eval` 表示“真实 eval 来源可追溯”, 不表示 paper-grade final claim。
3. `s2_160` 训练完成并生成 ONNX, 但 export sanity 输出:

```text
cls max|delta|=3.310e-02 rel_max=0.39%
reg max|delta|=7.463e-03 rel_max=0.19%
dir max|delta|=2.402e-02 rel_max=1.02%
Sanity FAIL
```

因此没有构建 TRT engine, 没有 AP eval report, 没有 measured AP row。

## 4. Original60 AP gap 最新状态

`exports/original60_ap_axis_gap_report_v2.json` 当前统计:

| status | count |
|---|---:|
| exact_true_source_available | 4 |
| finetune_smoke_quarantined | 1 |
| no_claim_missing_source | 54 |
| weight_identity_transfer_available | 1 |

claim 统计:

| claim_status | count |
|---|---:|
| claimable_true_eval | 4 |
| claimable_weight_identity_transfer | 1 |
| no_claim_missing_source | 55 |

没有 predicted/model-fit/interpolated AP measured row。本轮 `PREDICTED_MEASURED_ROWS=[]`。

质量门控见:

```text
exports/ap_quality_gate_report.json
exports/ap_quality_gate_report.md
```

当前质量门控:

| label | quality_gate_status | 原因 |
|---|---|---|
| s0_024 | trend_anomaly_pending_repeat | AP70 比 base 高 0.001632 |
| s0_040 | first_seed_true_eval_pending_repeat | 单 seed true_eval, paper-grade 前建议 repeat |
| s0_056 | first_seed_true_eval_pending_repeat | 单 seed true_eval, paper-grade 前建议 repeat |
| s1_048 | trend_anomaly_pending_repeat | AP70 比 base 高 0.004753 |
| s2_160 | export_or_eval_quarantined | ONNX export sanity fail |

### 4.1 Original60 三指标总表补充更新

已补充刷新 Original60 AP/Latency/Energy 三指标总表:

```text
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/exports/original60_three_metric_summary_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/exports/original60_three_metric_summary_latest.csv
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/exports/original60_three_metric_summary_latest.json
```

总表由以下来源合成, 不改写 raw measurement:

1. latency: `coverage_pipeline_v1/rows/latency_lut_rows_original60_v1.jsonl`
2. energy: `coverage_pipeline_v1/rows/energy_lut_rows_original60_v1.jsonl`
3. AP: `ap_stability_20260626/exports/ap_stability_summary.csv`
4. AP source/claim: `ap_stability_20260626/exports/original60_ap_source_map_v1.jsonl`
5. AP quality gate: `ap_stability_20260626/exports/ap_quality_gate_report.json`
6. fallback quarantine/status: 旧 `coverage_pipeline_v1/exports/original60_measured_lut_summary.csv`

当前统计:

| item | count |
|---|---:|
| total_rows | 60 |
| latency_measured_rows | 58 |
| energy_measured_rows | 56 |
| ap_with_ap70_rows | 5 |
| ap_true_eval_rows | 4 |
| ap_transfer_rows | 1 |
| ap_no_claim_rows | 55 |
| ap_quality_pending_repeat_rows | 4 |
| quarantined_rows | 5 |

关键行已确认进入总表:

| label | width | latency_tuned_ms | energy_j_per_inference | AP30 | AP50 | AP70 | AP 状态 | 质量门控 |
|---|---|---:|---:|---:|---:|---:|---|---|
| s0_024 | 24x128x256 | 39.503066 | 4.954709 | 0.833141389 | 0.791044145 | 0.632495206 | claimable_true_eval | trend_anomaly_pending_repeat |
| s0_040 | 40x128x256 | 44.704673 | 4.794238 | 0.822728829 | 0.781245120 | 0.626659483 | claimable_true_eval | first_seed_true_eval_pending_repeat |
| s0_056 | 56x128x256 | 51.342031 | 6.196241 | 0.832072361 | 0.790658722 | 0.623711042 | claimable_true_eval | first_seed_true_eval_pending_repeat |
| s1_048 | 64x48x256 | 44.452754 | 5.499634 | 0.822220053 | 0.782852439 | 0.635616211 | claimable_true_eval | trend_anomaly_pending_repeat |
| s1_064 | 64x64x256 | 46.633232 | 4.817434 |  |  | 0.636159742 | claimable_weight_identity_transfer |  |
| s2_160 | 64x128x160 | 49.776660 | 4.162604 |  |  |  | no_claim_missing_source | export_or_eval_quarantined |

刷新命令:

```bash
python3 scripts/stage2_export_original60_three_metric_summary.py
```

## 5. 本轮环境修复与执行记录

### 5.1 H800 环境

H800 访问方式仍以 `RUNBOOK_stage2_h800_server_access_v1_zh.md` 为准。不要把密码写进仓库、脚本、日志或文档。

本轮确认:

| 项 | 状态 |
|---|---|
| hostname | `<PRIVATE_HOST>` |
| HEAL | `${V2X_HOME}/heal_research/HEAL -> ${V2X_DATA_ROOT}/heal_research/HEAL` |
| checkpoint root | `${V2X_HOME}/heal_research/checkpoints -> ${V2X_DATA_ROOT}/heal_research/checkpoints` |
| UniV2X python | `${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python` |
| torch | 2.0.1+cu118 |
| numpy | 1.22.4 |
| opencood import | ok |
| AP preflight | pass |

### 5.2 DAIR 数据同步

最初全量同步会传入无关数据, 后改为 required-files:

```text
logs/dair_train_val_required_files_20260626.txt
logs/dair_required_files_rsync_z_to_h800_20260626.log
```

required-files dry-run 校验输出为 0, 表示 required train/val 文件已同步完成。远端 required pcd 计数:

```text
infrastructure-side/velodyne/*.pcd: 12424 present
vehicle-side/velodyne/*.pcd: 6600 present
```

## 6. 当前完成度对照 stop condition

| stop condition | 当前证据 | 状态 |
|---|---|---|
| AP source registry 完成 | `registry/ap_source_registry_v1.jsonl` 32 行 | 完成 |
| replay gate 有明确 pass/fail | `exports/ap_replay_gate_report.json`, pass 4/4 | 完成, 但为 import/replay gate, 非 fresh eval |
| 10 个已有 source 配置 canonical rows | `rows/ap_anchor_rows_v1.jsonl` 前 10 个 source rows | 完成 |
| 5 个 original60 新配置完成 eval 或 quarantine | 4 succeeded, 1 failed/quarantine | 完成 |
| 至少 8 个不同配置 AP row | 当前 14 行 AP rows | 完成 |
| original60 AP gap report 更新 | `exports/original60_ap_axis_gap_report_v2.json` 已更新 | 完成 |
| 所有结果 rsync 回本地 | `ap_stability_20260626/` 已同步到本地 | 完成 |
| 写新的中文交接文档 | 本文档 | 完成 |

未完全满足 paper-grade 的部分:

1. `s0_024` 和 `s1_048` 高于 base, 必须二次 seed 或延长训练复测。
2. `s2_160` 需要修复 ONNX export sanity 或换 ckpt 重导出后再评测。
3. replay gate 当前是 source import/replay gate, 不是 base/p50/trap25/mix_d 的 fresh H800 eval repeat。

## 7. 下一阶段计划

P0:

1. 对 `s0_024` 和 `s1_048` 做 second seed/repeat:
   - 仍使用 DAIR-V2X-C train/val_1789。
   - 默认 `epoches=31`, 若 bestval 再次落在最后 epoch 或趋势仍异常, 延长到 `epoches=48`。
   - 目标: 判断 AP 高于 base 是训练协议收益、随机性、还是评测/导出路径问题。
2. 修复 `s2_160`:
   - 先用 `net_epoch31.pth` 重新 export ONNX。
   - 若 sanity 仍 fail, 对比 PyTorch/ONNX 差异是否来自 `GridSample`/动态路径。
   - sanity pass 前不得构建 TRT/AP eval measured row。
3. 为 4 个 succeeded 新 rows 添加 quality gate 字段或单独 quality report:
   - `first_seed_only`
   - `trend_anomaly_pending_repeat`
   - `paper_claim_blocked_until_repeat`

P1:

1. 给 base/p50/trap25/mix_d 做 fresh H800 eval replay, 不只依赖 import gate。
2. 补 ckpt/config digest 到 10 个历史 source/import rows。
3. 对 original60 继续挑选 5-10 个配置做第二批 Phase C finetune smoke, 但必须先处理 P0 的异常点。

P2:

1. 把 AP measured rows 和 latency/energy candidate bundle 重新对齐。
2. 更新 supervisor gate: AP 轴区分 `raw true_eval`, `repeat-passed true_eval`, `paper-grade claimable`。
3. 继续扩展 original60 AP coverage, 但不允许用 model-fit AP 填 measured。

## 8. 下一次启动入口

推荐 `/goal`:

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 AP stable smoke 后处理。先阅读 multi_agent/methods/design/auto-tuning/progress/6_27/0_6_27_冷启动入口_阶段二LUT.md, 再阅读 multi_agent/methods/design/auto-tuning/progress/6_27/1_6_27_交接文档_阶段二_AP稳定测试PhaseC实测结果.md 和 multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md。目标: 对 s0_024 和 s1_048 做 second seed/repeat, 修复或确认 s2_160 ONNX sanity fail, 并把 AP quality gate 明确为 first_seed_only / trend_anomaly_pending_repeat / repeat_passed / quarantine。不要改写已有真实 AP 数值, 不要把 predicted/model-fit/interpolated AP 写成 measured row。H800 密码不得写入任何文件; 如需登录, 向用户确认密码并只用 rsync 回本地。
```
