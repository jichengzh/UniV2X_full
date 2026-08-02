# HANDOFF: Stage1 Standard-Conv Coupling Plan Status (2026-06-23)

> 新窗口接手先读这份。本文严格按 `multi_agent/methods/design/stage1-model-predict/standard_conv_coupling_deepening_plan_v1.md` 的 S0-S6 阶段定义更新。旧 `HANDOFF_stage1_model_predict_standard_conv_v1.md` 只保留历史背景。

---

## 0. 当前结论

**按 plan 文件定义，S2 低成本 schedule anchor 队列扫描已经完成，但结论不是低风险通过，而是发现标准卷积存在 P x S / Q x S / batch x P 风险信号。S2.5 只作为 S2 收尾，关闭两个 targeted coverage gate；S3 已完成当前可执行的 Q/AP sensitivity 证据绑定；S4/S5/S6 的当前可执行闭环也已完成并落盘。**  
本轮已使用 H800 `${V2X_DATA_ROOT}/tvm310` 跑完两个 `MEASUREMENT_BACKLOG_RUNNER_REQUIRED` 项：

- `std_basebev_backbone_schedule_anchor`
- `standard_neck_deconv_anchor`

cell 维度是 base / moderate / boundary × fp16 / int8 × batch 1/2，共 24 个 cell；每个 cell 内跑 default 和 own-tuned，int8 cell 额外跑 schedule-swap。所有 default、own-tuned、int8 schedule-swap 均产出。9 个探针队列项也都已有状态：2 个实测完成，7 个按 guardrail / existing-evidence binding / unresolved blocker 归档。审计见：

- `results/stage1_model_predict/s2_schedule_anchor_audit_v1.json`
- `results/stage1_model_predict/s2_schedule_anchor_audit_v1.md`
- `results/stage1_model_predict/s2_probe_results/stage1_s2_probe_completion_v1.json`
- `results/stage1_model_predict/s2_probe_results/stage1_s2_probe_completion_v1.md`

当前科学口径：

1. `groups=1` 不是可分离判据，只表示没有已知 grouped-conv IC_BN hard-format cliff。
2. CoDriving 只在已测 `codriving_measured_resnet_backbone_envelope` 内是 scoped low-risk anchor；S2 新 anchor 已经反向说明不能外推到 F-Cooper / AttFuse / V2X-ViT / Pyramid / Where2comm / V2VNet / DiscoNet。
3. Where2comm / V2VNet / DiscoNet 已纳入 auto-scan，但均为 `missing_architecture_scan_only`，不能写成 trained-checkpoint scan。
4. Fusion 不能默认忽略：Where2comm fusion 与 backbone/shrinker 同量级；V2VNet fusion 明显主导；DiscoNet 只在本轮 DAIR 单 agent 样本中较小，不代表多 agent 场景。
5. Who2com 暂不纳入实扫：本地 HEAL 有相关实现分支，但缺少可直接构建的 config/ckpt 组合，不能伪造扫描。
6. S2.5 结论：F-Cooper MaxFusion 已有源码静态 shape-preserving proof + 历史 true-weight timing bound，只能解除 “MaxFusion 未定界” 这一点；routing/fusion gate 仍是模型级 coverage cap。
7. S3 结论：Pyramid 有 historical TRT AP/latency 证据，只能作为历史证据绑定；V2X-ViT 只有 fake-quant QxP + structural routing gate，true TRT INT8 AP/per-channel/Orin latency 仍 blocked。
8. S4 结论：复用 S2 H800/TVM 24-cell 实测矩阵做 mini three-arm latency validation；两个 high-risk anchors 都出现 local-only winner change，pair-search 在这个 mini matrix 内匹配 joint。该结论只能支持“至少需要 pair-level schedule calibration”，不能写成 AP/HV 或 full-model irreducible coupling。
9. S5/S6 结论：calibrated predictor v1 已生成，`no_overpromotion=True`，最终报告明确区分 measured H800 TVM、historical TRT AP/latency、architecture-only、random-init timing、static proof、fake-quant gate、blocked/unresolved。

### 0.1 2026-06-24 classifier closure addendum

已新增端到端 Stage1 模型分类器，默认覆盖 9 个模型：

- implementation: `framework/stage1/model_classifier.py`
- CLI: `scripts/stage1_classify_models.py`
- JSON: `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json`
- Markdown: `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md`
- validation note: `multi_agent/methods/design/stage1-model-predict/model_classifier_validation_v1.md`

分类器契约：

- 新增实测后端统一为 H800 TVM / Relax / MetaSchedule：`allowed_new_measurement_backends=["h800_tvm"]`。
- 模型主分类已集成成三类字段：`CO_ACCELERATION_REQUIRED`=需要协同加速，`SEPARABLE_ACCELERATION`=可分离加速，`SCAN_FAILED`=扫描失败；旧 `classification` 保留为细粒度证据 verdict。
- TRT 只能作为 `historical_evidence_sources` / `historical_trt_evidence`；不得写入 `measured_h800_tvm`，不得作为默认后端或新增 probe backend。
- S2/S4 latency 证据只标为 `measured_h800_tvm_latency`；S4 仍是 latency-only，不含 AP/HV。
- Pyramid TRT AP/latency 只作为 historical evidence 绑定；V2X-ViT true TRT INT8 AP 继续是 blocked / not done。
- Where2comm / V2VNet / DiscoNet 继续是 `missing_architecture_scan_only`，不能写成 trained checkpoint classification，也不写入 `measured_h800_tvm`。
- `groups=1`、`no int8 buildability cliff`、bridge 低 cliff 风险都不能推出模型级 separable。

扫描/manifest 字段已补齐：

- `trace.skipped_modules` legacy 字段保留，新增 typed `trace.skipped_subgraphs`。
- B1 prune-group feature: `cin/cout/groups/ic_bn/kernel/stride/op_types/fanout_buckets`。
- B1 search-group feature: `min_ic_bn/max_groups/op_types/fanout_buckets`。
- `view_latency.coverage` 明确为 `trace_net_only`，不是 full-model coverage。

Bridge/search 文案已替换：不再输出旧的全旋钮可串行/可分离 architecture label，搜索注释改为 “no int8 buildability cliff in traced dense space” 口径。
P×Q×S 搜索中的 historical 4090 TRT ratio fallback 默认禁用；缺 H800 INT8 数据时使用 neutral FP16 latency，除非显式选择 real H800 TVM 或 H800 stage0 proxy。

---

## 1. Plan 对齐阶段表

| Plan 阶段 | 状态 | 已完成 | 未完成 / 阻塞 |
|---|---|---|---|
| S0 Evidence Envelope | done | 已把 `groups=1` 降级为低 hard-format 风险；CoDriving 只作为 scoped measured negative anchor | 无 |
| S1 Static Standard-Conv Census | done with pending manifest upgrade | 9 manifests census 已生成；`standard_conv_census_v1` 和 `standard_conv_probe_queue_v1` 已更新 | per-root `cin/cout/groups/kernel/stride/input_hw/output_hw/fanout` 仍未补齐 |
| S2 Low-Cost Standard-Conv Anchor Probes | done / coupling signals detected | H800 TVM 环境已验证；两个 schedule anchor 共 24 cell 已完成；9 个 probe/gate 队列项均有状态；S2.5 targeted coverage gate 已收口 | 不是 low-risk pass；发现 P x S / Q x S / batch x P 风险信号，不能支持“标准卷积默认可分离” |
| S3 Quantization-Sensitivity Probe | done for executable evidence binding / blockers explicit | Pyramid historical TRT AP/latency 只作为 historical evidence 绑定；V2X-ViT fake-quant QxP + structural C5 routing 已绑定；输出 `stage1_s3_quant_sensitivity_v1` | V2X-ViT true TRT INT8 AP、true TRT per-channel、Orin latency 仍因 fusion ONNX export blocked，不能写成 real TRT closure |
| S4 Mini Three-Arm Validation | done / latency-only | 两个 high-risk anchors 已用 S2 H800/TVM 实测矩阵完成 local-only / pair-search / joint-search 三臂验证；输出 `stage1_s4_three_arm_validation_v1` | 不含 AP/HV；不能推出 full-model irreducible coupling |
| S5 Predictor Rule Update | done / v1 guarded | calibrated predictor v1 已接入 S2/S2.5/S3/S4 证据；`stage1_coupling_predictions_v1` 已生成且 `no_overpromotion=True` | 仍禁止 blocked/gated 项升级为 pass；attention/fusion/custom 未集成模型仍不能 full-model promotion |
| S6 Calibration And Reporting | done current closure | `calibrated_predictor_report_v1` 已生成，汇总 S0-S6、9 模型 verdict、证据等级、禁用说法和下一步最小动作；critic final review 已 `ACCEPT` | 后续仅剩更强证据的下一步动作，不影响本轮 S6 closure |

后续汇报必须使用这张表。不要再把 Safe Predictor v0 完成写成 plan S2 完成，也不要把 S2 队列完成写成“标准卷积低风险通过”。

---

## 2. 当前 9 模型扫描状态

来源：

- manifests: `framework/partitions/*.yaml`, `results/autoscan_*_partition.yaml`
- predictions: `results/stage1_model_predict/stage1_coupling_predictions_v0.json`

| model | scan_status | ckpt_status | B1 groups | B1 knobs | B2 units | verdict |
|---|---|---|---:|---:|---:|---|
| `codriving` | `ok` | `ok` | 20 | 4 | 3 | `ANCHOR_PROBED_LOW_RISK` scoped to CoDriving envelope |
| `fcooper` | `ok` | `ok` | 24 | 4 | 3 | `LOW_CONFIDENCE_NEEDS_TARGETED_PROBE` |
| `attfuse` | `ok` | `ok` | 24 | 4 | 3 | `FUSION_UNCOVERED_UNKNOWN` |
| `v2xvit` | `ok` | `ok` | 23 | 4 | 3 | `JOINT_OR_PAIR_SEARCH_REQUIRED_UNTIL_C4_C5_BOUND` |
| `pyramid_lidar` | `ok` | `ok` | 43 | 5 | 4 | `P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_STANDARD_CONV_PROMOTION` |
| `pyramid_camera` | `ok` | `opv2v_only_no_dair` | 46 | 6 | 5 | `P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_STANDARD_CONV_PROMOTION` |
| `where2comm` | `ok` | `missing_architecture_scan_only` | 24 | 4 | 3 | `FUSION_UNCOVERED_UNKNOWN` |
| `v2vnet` | `ok` | `missing_architecture_scan_only` | 24 | 4 | 3 | `FUSION_UNCOVERED_UNKNOWN` |
| `disconet` | `ok` | `missing_architecture_scan_only` | 24 | 4 | 3 | `FUSION_UNCOVERED_UNKNOWN` |

新增三模型的当前 dense-core 共同事实：

- `entry_shape`: `[1, 64, 256, 256]`
- `params_total`: `8057620`
- `param_dist`: `backbone=0.742`, `neck=0.2574`, `heads=0.0006`
- skipped: `pillar_vfe`, `scatter`, `fusion_net`
- full-model verdict 被 fusion/routing/attention coverage gate 阻断

---

## 3. S2 审计与补充探针

### 3.1 S2 schedule anchor 状态

`standard_conv_probe_queue_v1` 的 9 个 probe/gate 队列项已经全部扫描并落盘。注意这里的“全部扫描”含义是：

- 2 个 `MEASUREMENT_BACKLOG_RUNNER_REQUIRED` 项完成 H800 TVM 实测；
- 7 个非 runner 项按 guardrail / existing-evidence binding / unresolved blocker 给出状态；
- blocked/unresolved 项不能写成 pass，也不能被用来支持 full-model separability。

H800 实测入口：

- runner: `scripts/phase2/stage1_s2_anchor_runner.py`
- remote JSON: `${V2X_DATA_ROOT}/s2_tvm/results/stage1_s2_anchor_scan_v1_20260623_2110.json`
- local JSON: `results/stage1_model_predict/s2_anchor_scan/stage1_s2_anchor_scan_v1_20260623_2110.json`
- local log: `results/stage1_model_predict/s2_anchor_scan/stage1_s2_anchor_scan_v1_20260623_2110.log`
- TVM: `0.20.dev1070+gb628d91fa`
- trials/reps: `4` / `20`
- batches: `1,2`

实测矩阵：

| probe | cells | default ok | own tuned ok | schedule-swap ok | rank flips | gain min/max | swap min/max | batch flip | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `std_basebev_backbone_schedule_anchor` | 12 | 12 | 12 | 6/6 | 4 | 0.669/3.331 | 0.969/3.333 | true | `COUPLING_SIGNAL_DETECTED` |
| `standard_neck_deconv_anchor` | 12 | 12 | 12 | 6/6 | 3 | 0.189/2.025 | 0.365/1.830 | false | `COUPLING_SIGNAL_DETECTED` |

四类 S2 判据：

- default 排序和 tuned 排序翻转：两个 anchor 均命中。
- schedule gain 随 width 剧烈变化：两个 anchor 均命中。
- 同一 int8 图内 own-tuned vs fp16-tuned schedule gap：两个 anchor 均命中。
- batch 改变后最优 width 翻转：BaseBEVBackbone anchor 命中；neck/deconv anchor 未命中。

9 个探针总表：

- `results/stage1_model_predict/s2_probe_results/stage1_s2_probe_completion_v1.json`
- `results/stage1_model_predict/s2_probe_results/stage1_s2_probe_completion_v1.md`

审计文件：

- `results/stage1_model_predict/s2_schedule_anchor_audit_v1.json`
- `results/stage1_model_predict/s2_schedule_anchor_audit_v1.md`

H800 smoke 日志仍保留为环境验证：

- remote: `${V2X_DATA_ROOT}/s2_tvm/results/stage1_model_predict_h800/stage1_s2_h800_p50_smoke_20260623_2031.log`
- local: `results/stage1_model_predict/h800_s2_anchor_probe/stage1_s2_h800_p50_smoke_20260623_2031.log`

### 3.2 已补跑的 fusion timing probe

这不是 S2 schedule anchor，只是回答 “fusion 是否可以不纳入扫描空间” 的 coverage sidecar。

| model | random_init | record_len_mean | fusion mean | fusion stage-sum pct | 结论 |
|---|---:|---:|---:|---:|---|
| `where2comm` | true | 2.0 | 6.563 ms | 26.9% | fusion 与 backbone/shrinker 同量级，不能忽略 |
| `v2vnet` | true | 2.0 | 92.582 ms | 88.7% | fusion 明显主导，不能忽略 |
| `disconet` | true | 1.0 | 1.386 ms | 12.4% | 单 agent DAIR 样本较小，但不能代表多 agent fusion |

文件：

- `results/stage1_model_predict/fusion_timing_probe/stage1probe_where2comm_random.json`
- `results/stage1_model_predict/fusion_timing_probe/stage1probe_v2vnet_random.json`
- `results/stage1_model_predict/fusion_timing_probe/stage1probe_disconet_random.json`

解释边界：

- 这些 timing 是 PyTorch CUDA per-stage timing，不是 TVM schedule anchor。
- 三者都是 random-init latency-only，AP/accuracy 无意义。
- DiscoNet 本轮 DAIR 样本 `record_len_mean=1.0`，不能证明多 agent fusion 成本可忽略。

### 3.3 S2.5 targeted coverage gate 收口

S2.5 不是新增宽泛探针，也不是把 9 个 probe 扩成完整矩阵。它只关闭两个低成本 coverage gate：

| gate | 状态 | 结论 | 边界 |
|---|---|---|---|
| `maxfusion_coverage_anchor` | `STATIC_SHAPE_PROOF_PLUS_EXISTING_TIMING_BOUND` | F-Cooper MaxFusion 已有源码 shape-preserving proof：输入 `(sum(n_cav), C, H, W)`，输出 `(B, C, H, W)`；历史 true-weight timing 为 mean 1.41 ms / e2e 1.6% | 只解除 F-Cooper MaxFusion 未定界，不推出 full-model separable |
| `routing_fusion_coverage_anchor` | `MODEL_LEVEL_COVERAGE_GATE_REMAINS` | Where2comm/V2VNet fusion 不能全局忽略，routing/fusion 仍是模型级 coverage cap | sidecar 是 random-init timing，不是 TVM schedule anchor，也不是 trained AP 证据 |

文件：

- `results/stage1_model_predict/s2_5_coverage_gates/stage1_s2_5_coverage_gate_closure_v1.json`
- `results/stage1_model_predict/s2_5_coverage_gates/stage1_s2_5_coverage_gate_closure_v1.md`

### 3.4 S3 quantization/AP sensitivity 证据绑定

S3 当前可执行部分已经完成：复用已有真 AP/latency 和 C4/C5 证据，明确哪些可以作为 measured evidence，哪些只能作为 gate/blocker。

| model family | S3 状态 | 结论 | 不能外推 |
|---|---|---|---|
| Pyramid | `MEASURED_AP_AND_LATENCY_BOUND` | forced per-stage INT8 在剪枝档暴露 AP sensitivity；但 hand-forced per-stage mixed precision 被 `global_int8_automix` 纳入后不是 Pareto-positive | 不能据此扩张任意 per-stage Q 搜索 |
| V2X-ViT | `WEAK_FAKE_QUANT_GATE_ONLY_TRUE_TRT_BLOCKED` | fake-quant QxP + structural routing blacklist 足以保留 pair/joint gate | 不能写成 true TRT INT8 AP closure；Orin latency 未测 |

文件：

- `results/stage1_model_predict/s3_quant_sensitivity/stage1_s3_quant_sensitivity_v1.json`
- `results/stage1_model_predict/s3_quant_sensitivity/stage1_s3_quant_sensitivity_v1.md`

因此，针对“9 个探针只有两个能用，是否需要更多探针”的当前结论是：

- 不需要新增泛化 S2 schedule probe；这会滑向低质量枚举。
- 需要的两个 targeted S2.5 gate 已完成收口。
- 现在应进入 S4 mini three-arm validation，而不是继续在 S2 增量加同类探针。

### 3.5 S4 mini three-arm validation

S4 复用 S2 的 H800/TVM 24-cell 实测矩阵，不新增 latency 测量，也不扩展 S2 大矩阵。

文件：

- runner/analysis: `scripts/phase2/stage1_s4_three_arm_validation.py`
- raw JSON: `results/stage1_model_predict/s4_three_arm_validation/stage1_s4_three_arm_validation_v1.json`
- analysis MD: `results/stage1_model_predict/s4_three_arm_validation/stage1_s4_three_arm_validation_v1.md`

三臂定义：

- local-only: 先在 `fp16/default` 下选 P，再在该 P/default 下选 Q，最后在该 P/Q 下选 S。
- pair-search: 在 local Q 固定下搜索 P x S，并在 local P 固定下搜索 Q x S，取二者最优。
- joint-search: 在已有 measured P x Q x S mini matrix 中取全局最优。

结果：

| probe | batch 覆盖 | S4 verdict | 结论 |
|---|---:|---|---|
| `std_basebev_backbone_schedule_anchor` | 1,2 | `PAIR_SEARCH_REQUIRED_LOCAL_ONLY_UNSAFE` | 两个 batch 都出现 local-only winner change，pair-search 匹配 joint |
| `standard_neck_deconv_anchor` | 1,2 | `PAIR_SEARCH_REQUIRED_LOCAL_ONLY_UNSAFE` | batch 1 出现 local-only winner change，batch 2 local/pair/joint 一致 |

边界：

- S4 是 latency-only，不含 AP/HV。
- S4 证明 local-only/静态规则不安全，但 pair-search 在当前 mini matrix 内已匹配 joint。
- 因此 S5 只能要求至少 pair-level schedule calibration；不能写成 full-model irreducible coupling。

### 3.6 S5 calibrated predictor v1

文件：

- implementation: `framework/stage1/calibrated_predictor.py`
- CLI: `scripts/stage1_predict_coupling_v1.py`
- JSON: `results/stage1_model_predict/stage1_coupling_predictions_v1.json`
- MD: `results/stage1_model_predict/stage1_coupling_predictions_v1.md`

核心规则：

- `groups=1` 只降低 grouped-conv hard-format cliff 风险，不能推出 separability。
- S2 `COUPLING_SIGNAL_DETECTED` 阻断 static standard-conv overpromotion。
- S2.5 MaxFusion 只解除 F-Cooper MaxFusion 未定界，不能推出 F-Cooper/full-model separable。
- routing/fusion gate 仍是模型级 coverage cap。
- Pyramid 只保留 historical TRT AP/latency evidence，仍受 P-hub context 限制，不能作为新增测量后端。
- V2X-ViT 只能写 fake-quant QxP + structural routing gate；true TRT INT8 AP/per-channel/Orin latency 仍 blocked。
- Where2comm / V2VNet / DiscoNet 只能写 `missing_architecture_scan_only` + fusion uncovered unknown。

最终 v1 verdict 表见 `results/stage1_model_predict/stage1_coupling_predictions_v1.md`。`no_overpromotion=True`。

### 3.7 S6 calibrated report

文件：

- JSON: `results/stage1_model_predict/calibrated_predictor_report_v1.json`
- MD: `results/stage1_model_predict/calibrated_predictor_report_v1.md`

S6 report 包含：

- S0-S6 状态表；
- 9 模型最终 verdict 表；
- S2/S2.5/S3/S4/S5 证据链；
- measured 支撑结论；
- weaker/blocked 结论；
- prohibited claims；
- next minimal actions。

当前 S6 状态是 `done_current_closure`。Critic final review 已给出 `ACCEPT`：未发现 overpromotion、伪实测、blocked/pass 混淆、random-init timing 误用或 architecture-only scan 误写成 checkpoint scan。

---

## 4. 代码和结果产物

### 4.1 代码

- `framework/stage1/auto_trace.py`
  - 新增 `where2comm / v2vnet / disconet` 到 `AUTO_REGISTRY`。
  - 缺 ckpt 时允许 `missing_architecture_scan_only`。
  - 增加 DiscoNet `disco_fuse.py` checkpoint-script fallback。
- `scripts/autoscan_reproduce_check.py`
  - 新增 `--models a2`。
- `framework/stage1/standard_conv_census.py`
  - 默认 manifest 扩展到 9 个。
  - routing representative models 收紧为 non-channel-preserving / attention fusion models。
- `framework/stage1/standard_conv_probe_plan.py`
  - 增加 Where2comm/V2VNet/DiscoNet gates。
- `framework/stage1/coupling_predictor.py`
  - 增加 conservative gate fallback。
  - `ANCHOR_PROBED_LOW_RISK` 纳入 full-model overpromotion guard。
  - prediction report 增加 `ckpt_status`。
  - 接入 S2 H800 audit、S2.5 coverage gate、S3 quant sensitivity evidence。
  - F-Cooper 不再写 “MaxFusion 未证明”；改为 “MaxFusion 已定界，但 S2 schedule coupling + routing gate 仍阻断 full-model”。
  - Pyramid evidence level 收敛为 `historical_trt_evidence_plus_p_hub_context`，但 verdict 仍是 P-hub context blocker。
- `scripts/phase2/m4_9_v2x_baselines_timing.py`
  - 增加 DiscoNet fallback，仅用于 timing probe 构建模型，不修改 HEAL 源码。
- `scripts/phase2/stage1_s2_anchor_runner.py`
  - H800 TVM/Relax MetaSchedule low-cost anchor runner。
  - 覆盖 standard Conv2d 和 Conv2dTranspose 代表 shape 的 P/Q/S/batch 探针。
- `scripts/phase2/stage1_s2_probe_completion_report.py`
  - 将 H800 实测 anchor 与 9 个 probe/gate 队列项汇总为 completion JSON/MD。
- `scripts/phase2/stage1_s2_5_s3_evidence_report.py`
  - 生成 S2.5 coverage gate closure 和 S3 quant sensitivity 证据绑定 JSON/MD。
- `scripts/phase2/stage1_s4_three_arm_validation.py`
  - 从 S2 H800/TVM 实测 24-cell 矩阵生成 S4 local-only / pair-search / joint-search 三臂验证 JSON/MD。
- `framework/stage1/calibrated_predictor.py`
  - 生成 calibrated predictor v1 与 S6 calibrated report。
- `framework/stage1/model_classifier.py`
  - 生成端到端 Stage1 model classification v1；保留 H800 TVM 新测量与 historical TRT evidence 分桶。
- `scripts/stage1_predict_coupling_v1.py`
  - v1 预测与 S6 报告 CLI 入口。
- `scripts/stage1_classify_models.py`
  - 端到端模型分类器 CLI，默认输出 JSON/MD 到 `results/stage1_model_predict/model_classifier/`。

### 4.2 结果

- `results/autoscan_where2comm_partition.yaml`
- `results/autoscan_v2vnet_partition.yaml`
- `results/autoscan_disconet_partition.yaml`
- `results/stage1_model_predict/standard_conv_census_v1.json`
- `results/stage1_model_predict/standard_conv_probe_queue_v1.json`
- `results/stage1_model_predict/stage1_coupling_predictions_v0.json`
- `results/stage1_model_predict/stage1_coupling_predictions_v0.md`
- `results/stage1_model_predict/s2_schedule_anchor_audit_v1.json`
- `results/stage1_model_predict/s2_schedule_anchor_audit_v1.md`
- `results/stage1_model_predict/s2_anchor_scan/stage1_s2_anchor_scan_v1_20260623_2110.json`
- `results/stage1_model_predict/s2_anchor_scan/stage1_s2_anchor_scan_v1_20260623_2110.log`
- `results/stage1_model_predict/s2_probe_results/stage1_s2_probe_completion_v1.json`
- `results/stage1_model_predict/s2_probe_results/stage1_s2_probe_completion_v1.md`
- `results/stage1_model_predict/s2_probe_results/{probe_id}.json`
- `results/stage1_model_predict/s2_probe_results/{probe_id}.md`
- `results/stage1_model_predict/s2_5_coverage_gates/stage1_s2_5_coverage_gate_closure_v1.json`
- `results/stage1_model_predict/s2_5_coverage_gates/stage1_s2_5_coverage_gate_closure_v1.md`
- `results/stage1_model_predict/s3_quant_sensitivity/stage1_s3_quant_sensitivity_v1.json`
- `results/stage1_model_predict/s3_quant_sensitivity/stage1_s3_quant_sensitivity_v1.md`
- `results/stage1_model_predict/s4_three_arm_validation/stage1_s4_three_arm_validation_v1.json`
- `results/stage1_model_predict/s4_three_arm_validation/stage1_s4_three_arm_validation_v1.md`
- `results/stage1_model_predict/stage1_coupling_predictions_v1.json`
- `results/stage1_model_predict/stage1_coupling_predictions_v1.md`
- `results/stage1_model_predict/calibrated_predictor_report_v1.json`
- `results/stage1_model_predict/calibrated_predictor_report_v1.md`
- `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json`
- `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md`
- `results/stage1_model_predict/h800_s2_anchor_probe/stage1_s2_h800_p50_smoke_20260623_2031.log`
- `results/stage1_model_predict/fusion_timing_probe/*.json`
- `multi_agent/methods/design/stage1-model-predict/standard_conv_census_v1.md`
- `multi_agent/methods/design/stage1-model-predict/standard_conv_probe_queue_v1.md`
- `multi_agent/methods/design/stage1-model-predict/model_classifier_validation_v1.md`

目录纪律：`multi_agent/methods/design/stage1-model-predict/` 只放 `.md`；Python 在 `framework/` 或 `scripts/`；JSON/YAML 结果在 `results/`。

---

## 5. 当前闭环与下一步最小动作

当前 S4/S5/S6 可执行闭环已完成：

1. S4: 两个 high-risk anchors 已完成 local-only、pair-search、joint-search 三臂 latency-only 验证，输出 JSON/MD。
2. S5: calibrated predictor v1 已生成，`no_overpromotion=True`，并显式区分 measured/static/fake-quant/random-init/blocked 证据等级。
3. S6: calibrated predictor report 已生成，汇总 S0-S6、9 模型 verdict、证据链、禁用说法和下一步最小动作。

下一步最小动作：

1. 如果要从 “pair-level schedule calibration required” 升级到 “JOINT required”，需要在一个代表 anchor 上做 AP/HV 或真正 joint-vs-serial 验证；当前 S4 不支持该说法。
2. 如果要提升 AttFuse/V2X-ViT full-model 结论，必须先集成 attention/fusion 模块；当前 full-model promotion 仍被 gate 阻断。
3. 如果要降低未来探针成本，应先补 Stage1 manifest per-root structural fields：`cin/cout/groups/kernel/stride/input_hw/output_hw/fanout`。
4. Who2com 只能在有可直接构建的 config/ckpt 组合时纳入；当前不能写成 HEAL 无实现。

---

## 6. 双 agent 执行要求

继续使用两个 agent：

| agent | 角色 | 必须审查 |
|---|---|---|
| Agent A executor | 推进 S3/S4/S5 或补充 S2 复现实验 | 命令、结果 JSON/MD、失败原因 |
| Agent B critic | 批判性验收 | 是否把 S2 completion 误写成 separability pass、是否把 blocked/gated 项误写成 pass、是否把 fusion timing 当 schedule anchor、是否过度外推 CoDriving、是否跳过 ckpt/scope 标注 |

本轮 critic 已审查过 Safe Predictor 与 extension scan：无 CRITICAL/HIGH。已修正两个 MEDIUM：

- `ANCHOR_PROBED_LOW_RISK` 纳入 full-model overpromotion guard。
- prediction JSON 增加 `ckpt_status`。

本轮 H800/S2 状态更新也拉起了 reviewer critic。先前结论：

- CRITICAL: 无。
- HIGH: 旧 “blocked due missing TVM” 口径过期；这条旧审查意见已由本轮 H800 实测关闭。
- HIGH: 旧历史结果覆盖不足；这条旧审查意见已由本轮 24-cell anchor scan 和 9-probe completion report 关闭。
- MEDIUM: smoke 或单点 MetaSchedule probe 只能写 partial evidence。

随后 executor 已补跑完整 H800 S2 anchor scan，并按 critic 要求输出 9 个 probe/gate 的 completion artifacts。下一轮 critic 需要重点复查：

- 两个 runner-required anchor 是否确实 `MEASURED_COMPLETE`。
- 七个非 runner 项是否仍被标为 guardrail / binding / blocker，而不是 pass。
- `COUPLING_SIGNAL_DETECTED` 是否被接入 predictor 规则，阻止 static standard-conv overpromotion。

本轮 S2.5/S3 reviewer critic 新增必须保留的边界：

- `maxfusion_coverage_anchor` 不能写成通用 pass；只能在源码 shape contract + timing bound 后解除 F-Cooper MaxFusion 未定界。
- `routing_fusion_coverage_anchor` 是模型级 coverage cap，不等价于 Q/S feasible-set gate 通过。
- V2X-ViT C4 只能是 fake-quant gate；true TRT INT8 AP、true TRT per-channel、Orin latency 都是 `NOT_DONE`。
- Pyramid per-stage mixed precision 可以说相对 forced-all-INT8 有 AP 正向点，但不能说手工 per-stage mixed precision Pareto-positive；它被 `global_int8_automix` 基线约束。
- 标准卷积不能因 S2 queue scan 完成而升级为 separable；两个实测 anchors 都检测到 coupling signals。

本轮 S4/S5/S6 final critic 结论：

- status: `ACCEPT`
- S4 覆盖两个 high-risk anchors，三臂齐全，JSON/MD 中包含候选、rank、winner、latency。
- S4 没有 overclaim：限定为 latency-only，pair-search 在当前 mini matrix 匹配 joint，不支持 AP/HV 或 full-model irreducible coupling。
- S5 `stage1_coupling_predictions_v1.json/md` 有 9 模型 verdict、evidence categories、`no_overpromotion=True`。
- S6 `calibrated_predictor_report_v1.json/md` 包含 S0-S6 状态、9 模型 verdict、证据链、measured/weaker/blocked conclusions、prohibited claims、next minimal actions。
- 未发现 random-init timing 被写成 trained performance；未发现 routing/fusion timing 被写成 TVM schedule anchor；未发现 architecture-only 被写成 checkpoint scan；未发现 blocked/pass 混淆。

持续约束：Who2com 只能写“当前没有可直接构建的 config/ckpt 组合”，不能写成“HEAL 完全没有 Who2com 实现”。

---

## 7. 已验证命令

```bash
python scripts/phase2/stage1_s4_three_arm_validation.py
python scripts/stage1_predict_coupling_v1.py
```

结果：通过，生成 S4 JSON/MD、v1 prediction JSON/MD 和 calibrated report JSON/MD。

```bash
python -m py_compile \
  framework/stage1/calibrated_predictor.py \
  scripts/stage1_predict_coupling_v1.py \
  scripts/phase2/stage1_s4_three_arm_validation.py
```

结果：通过。

```bash
PATH=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=${V2X_ROOT} \
pytest -q \
  framework/tests/test_calibrated_predictor_v1.py \
  framework/tests/test_stage1_autoscan_extensions.py \
  framework/tests/test_stage1_manifest_predictor_fields.py \
  framework/tests/test_coupling_predictor_static.py
```

结果：`18 passed, 18 warnings`。

```bash
python scripts/phase2/stage1_s2_5_s3_evidence_report.py
python -m py_compile \
  framework/stage1/coupling_predictor.py \
  scripts/phase2/stage1_s2_5_s3_evidence_report.py \
  scripts/stage1_predict_coupling.py
```

结果：通过，生成 S2.5/S3 JSON/MD。

```bash
python scripts/stage1_predict_coupling.py \
  --manifest framework/partitions/codriving_partition.yaml \
  --manifest results/autoscan_fcooper_partition.yaml \
  --manifest results/autoscan_attfuse_partition.yaml \
  --manifest framework/partitions/v2xvit_partition.yaml \
  --manifest framework/partitions/pyramid_lidar_partition.yaml \
  --manifest framework/partitions/pyramid_camera_partition.yaml \
  --manifest results/autoscan_where2comm_partition.yaml \
  --manifest results/autoscan_v2vnet_partition.yaml \
  --manifest results/autoscan_disconet_partition.yaml \
  --out-json results/stage1_model_predict/stage1_coupling_predictions_v0.json \
  --out-md results/stage1_model_predict/stage1_coupling_predictions_v0.md
```

结果：通过，`no_overpromotion=True`。

```bash
PATH=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=${V2X_ROOT} \
pytest -q \
  framework/tests/test_stage1_autoscan_extensions.py \
  framework/tests/test_stage1_manifest_predictor_fields.py \
  framework/tests/test_coupling_predictor_static.py
```

结果：`14 passed, 18 warnings`。

```bash
python -m py_compile \
  scripts/phase2/m4_9_v2x_baselines_timing.py \
  scripts/phase2/stage1_s2_anchor_runner.py \
  scripts/phase2/stage1_s2_probe_completion_report.py \
  framework/stage1/auto_trace.py \
  framework/stage1/coupling_predictor.py \
  framework/stage1/standard_conv_census.py \
  framework/stage1/standard_conv_probe_plan.py
```

结果：exit 0。

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('results/stage1_model_predict/s2_schedule_anchor_audit_v1.json')
d=json.loads(p.read_text())
assert d['s2_schedule_anchor_status']=='completed_queue_scanned_coupling_signals_detected'
assert d['h800_environment_validation']['status']=='passed'
assert d['h800_environment_validation']['observations']['extracted_tasks']==21
assert len(d['fusion_timing_rows'])==3
assert d['s2_completion_summary']['measured_anchor_matrix_complete'] is True
print('s2_audit_ok')
PY
```

结果：`s2_audit_ok`。

```bash
ssh -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST> 'bash -lc "
export PATH=/usr/local/cuda-12.2/bin:\$PATH
export LD_LIBRARY_PATH=\"\$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path):\${LD_LIBRARY_PATH:-}\"
${V2X_DATA_ROOT}/tvm310/bin/python - <<\"PY\"
import tvm
import tvm.relax
import tvm.s_tir.meta_schedule
import tvm.s_tir.dlight
print(tvm.__version__, bool(tvm.cuda(0).exist))
PY
"'
```

结果：H800 TVM `0.20.dev1070+gb628d91fa`，CUDA 可见。

```bash
# 已在 H800 运行；JSON/log 已拉回本地。
CUDA_VISIBLE_DEVICES=6 ${V2X_DATA_ROOT}/tvm310/bin/python \
  ${V2X_DATA_ROOT}/s2_tvm/stage1_s2_anchor_runner.py \
  --out-json ${V2X_DATA_ROOT}/s2_tvm/results/stage1_s2_anchor_scan_v1_20260623_2110.json \
  --work-root ${V2X_DATA_ROOT}/s2_tvm/ms_work/stage1_s2_anchor_scan_v1_20260623_2110 \
  --trials 4 \
  --reps 20 \
  --batches 1,2 \
  --anchors all
```

结果：`DONE`，`cells=24`，local JSON 为 `results/stage1_model_predict/s2_anchor_scan/stage1_s2_anchor_scan_v1_20260623_2110.json`。

```bash
python scripts/phase2/stage1_s2_probe_completion_report.py
```

结果：生成 `results/stage1_model_predict/s2_probe_results/stage1_s2_probe_completion_v1.json` 和 `.md`，并为 9 个 probe 各生成 JSON/MD。

```bash
# 已在 H800 运行；日志已拉回本地。
CUDA_VISIBLE_DEVICES=6 ${V2X_DATA_ROOT}/tvm310/bin/python \
  ${V2X_DATA_ROOT}/s2_tvm/s2_2_probe.py \
  ${V2X_DATA_ROOT}/s2_tvm/models/p50_backbone.onnx \
  ${V2X_DATA_ROOT}/s2_tvm/ms_work/stage1_s2_h800_p50_smoke_20260623_2031 \
  8
```

结果：`extract_tasks -> 21`，`INT8_TENSORIZE_OK`，`tune_relax COMPLETED`，`compile_relax COMPLETED`，`RUN_OK n_out 3`，`PROBE_DONE`。

```bash
find multi_agent/methods/design/stage1-model-predict -maxdepth 1 -type f ! -name '*.md' -print
```

结果：无输出。
