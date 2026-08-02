# HANDOFF: Stage1 Model-Predict / Standard-Conv Coupling Guardrails (v1, 2026-06-23)

> ⚠️ 历史交接文档。本文件记录 Safe Predictor v0 之前的 S0/S1/S2 规划背景。当前进度、9 模型扫描表、S2/S3 状态和下一阶段停止目标以 `multi_agent/methods/progress/HANDOFF_stage1_safe_predictor_v0.md` 为准。

> ★**新窗口接手先读这份**。本阶段目标不是继续证明“标准卷积都可分离”, 而是把这个错误口径收住: **CoDriving 只能作为已测负例锚点; `groups=1` 不能推出模型可分离; Stage1 预测器必须绑定证据、scope 和 coverage**。
> 本文件自包含, 含当前进度、已产出路径、验证命令、下一阶段任务和硬约束。

---

## 0. 一句话现状

> ★★[2026-06-23 更新 — 标准卷积专项 S0/S1/S2 已完成到“证据整理 + census + probe/gate taxonomy + CoDriving 实测绑定”; **framework 预测器代码尚未实现**]

当前最重要的结论:

1. **CoDriving standard conv 是已测阴性锚点**, 但只在 CoDriving 已测 scope 内成立。
2. **不能把 CoDriving 结论外推到 F-Cooper / AttFuse / V2X-ViT / Pyramid mixed grouped+standard**。
3. **Static-only 不得输出 `PREDICTED_SEPARABLE_LOW_RISK`**。没有同 scope measured/anchor 文件时, 只能是 `LOW_CONFIDENCE_NEEDS_TARGETED_PROBE` 或明确 scoped 的 `ANCHOR_PROBED_LOW_RISK`。
4. 设计目录 `multi_agent/methods/design/stage1-model-predict/` **只保留 `.md`**; 代码和 JSON 已迁出到项目框架/结果目录。

---

## 0.5 本阶段产出与验收 (★已完成)

### 0.5.1 目录职责已纠正

**设计目录仅保留 Markdown**:

- `multi_agent/methods/design/stage1-model-predict/stage1_model_predictor_implementation_plan_v1.md`
- `multi_agent/methods/design/stage1-model-predict/standard_conv_coupling_deepening_plan_v1.md`
- `multi_agent/methods/design/stage1-model-predict/standard_conv_census_v1.md`
- `multi_agent/methods/design/stage1-model-predict/standard_conv_probe_queue_v1.md`
- `multi_agent/methods/design/stage1-model-predict/standard_conv_anchor_measured_results_v1.md`

**Stage1 代码放到框架目录**:

- `framework/stage1/standard_conv_census.py`
- `framework/stage1/standard_conv_probe_plan.py`

**机器可读产物放到结果目录**:

- `results/stage1_model_predict/standard_conv_census_v1.json`
- `results/stage1_model_predict/standard_conv_probe_queue_v1.json`
- `results/stage1_model_predict/standard_conv_anchor_measured_results_v1.json`

### 0.5.2 标准卷积 census 已完成一版

命令:

```bash
python framework/stage1/standard_conv_census.py
```

当前摘要:

- scanned manifests: 6
- search groups: 27
- standard-conv candidates: 20
- high-priority anchors: 8
- medium-priority anchors: 11

关键修正:

- `groups=1` 只表示没有 grouped-conv IC_BN hard cliff, **不是 separability verdict**。
- 含 `Conv2d + ConvTranspose2d + head/neck` 的组被标成 `mixed_conv2d_convtranspose_requires_split`, 不能混称纯 standard Conv2d。
- 缺 member / 缺 grouped metadata / unknown op 不得保守地当成标准卷积安全证据。

### 0.5.3 Probe queue 已扩展到 9 个, 但不是 9 个都可执行实测

命令:

```bash
python framework/stage1/standard_conv_probe_plan.py
```

当前 9 个 probe/gate:

1. `attention_fusion_coverage_anchor`
2. `routing_fusion_coverage_anchor`
3. `std_basebev_backbone_schedule_anchor`
4. `v2xvit_qgranularity_p_anchor`
5. `codriving_resnet_completion_anchor`
6. `maxfusion_coverage_anchor`
7. `per_stage_q_ap_sensitivity_anchor`
8. `pyramid_mixed_p_hub_context_anchor`
9. `standard_neck_deconv_anchor`

本轮相对最初 5 个 probe 的关键补充:

- V2X-ViT `Q-granularity × P`
- routing/fusion coverage
- Pyramid mixed grouped+standard 的 P-hub 邻接风险
- pruning 后 per-stage Q/AP sensitivity

★重要订正: 这 9 个不是“下一阶段必须跑完的 9 个实测探针”。其中 `attention_fusion_coverage_anchor` 当前**不存在可执行 attention 实测探针**, 因为 AttFuse/V2X-ViT attention/fusion 模块还没有集成进 Stage1 trace/export/TVM runner。它在 safe predictor v0 中只能作为 **metadata guardrail / blocker**, 不能作为 latency probe。

当前 readiness 分布:

| probe | readiness | 下一阶段角色 |
|---|---|---|
| `attention_fusion_coverage_anchor` | `BLOCKER_NOT_RUNNABLE_UNTIL_ATTENTION_TRACE_OR_EXPORT_EXISTS` | 只做 blocker: typed skip 必须阻断 full-model separability |
| `routing_fusion_coverage_anchor` | `PARTIAL_EXISTING_EVIDENCE_BINDING` | 绑定已有 C5 / typed routing-fusion skip; 不要求新 fusion runner |
| `std_basebev_backbone_schedule_anchor` | `MEASUREMENT_BACKLOG_RUNNER_REQUIRED` | 真正 dense-backbone 实测候选, 但 safe predictor v0 不以它完成为停止条件 |
| `v2xvit_qgranularity_p_anchor` | `EXISTING_EVIDENCE_BINDING` | 绑定已有 C4 证据 |
| `codriving_resnet_completion_anchor` | `COMPLETED_EXISTING_MEASURED_BINDING` | 已满足, 作为 regression/fixture |
| `maxfusion_coverage_anchor` | `STATIC_PROOF_CANDIDATE` | F-Cooper-only 静态 shape/0-param/channel-preserving proof; 缺 shape contract 则继续 low-confidence |
| `per_stage_q_ap_sensitivity_anchor` | `BACKLOG_OR_EXISTING_EVIDENCE_BINDING` | v0 只暴露 AP-sensitive gate; 不启动新 AP 实验 |
| `pyramid_mixed_p_hub_context_anchor` | `EXISTING_EVIDENCE_BINDING` | 绑定 Pyramid P-hub + adjacency, 防止局部 standard conv overpromotion |
| `standard_neck_deconv_anchor` | `MEASUREMENT_BACKLOG_RUNNER_REQUIRED` | backlog 实测; v0 不以它完成为停止条件 |

评估: **9 个足够覆盖当前“避免误判”的 gate taxonomy, 但不足以证明低成本预测器已能准确判断所有模型可分离性。** 若目标是 safe predictor v0, 够用; 若目标是“可准确预测 standard-conv full-model separability”, 还缺 attention/fusion 集成、dense-backbone/neck runner、结构 shape 字段和 AP-sensitive probe。

### 0.5.4 CoDriving 实测证据已绑定

实测结论文件:

- `multi_agent/methods/design/stage1-model-predict/standard_conv_anchor_measured_results_v1.md`
- `results/stage1_model_predict/standard_conv_anchor_measured_results_v1.json`

CoDriving FP16 dense-core H800/TVM 实测:

| label | ONNX | trials | effective batch | default us | tuned us | default/tuned | source |
|---|---|---:|---:|---:|---:|---:|---|
| base | `base_backbone.onnx` | 1000 | 2 | 16680.47 | 8057.79 | 2.070 | existing H800/TVM real |
| p50 | `p50_backbone.onnx` | 1000 | 2 | 3643.15 | 1609.10 | 2.264 | existing H800/TVM real |
| p25 | `p25_backbone.onnx` | 500 | 2 | 11885.00 | 10467.30 | 1.135 | fresh seed42 real |
| p75 | `p75_backbone.onnx` | 500 | 1 | 2831.70 | 2450.80 | 1.155 | fresh seed42 real |

注意: p75 ONNX 的输入维度实际固定为 batch=1, 不能和 p25/base/p50 当同 batch 直接比较吞吐。

### 0.5.5 实验编号已补定义

这些编号来自 `results/coupling_map_matrix.json` 的 coupling-map cell, 不是模型名:

| 编号 | 全称 | 目标 | 结果文件 | 当前判决 |
|---|---|---|---|---|
| `C0c'` | CoDriving P×Q×S 三臂低维消融 | 比较 joint vs serial HV | `results/coupling_map/C0c_codriving_pqs.json` | `SERIAL`: A-joint=A-serial=100% HV, 0 rank-flip |
| `C1cod` | CoDriving Q×S 固定-P 复核 | 检查量化 Q 与调度 S 是否有 width-specific trap | `results/coupling_map/C1_QxS_codriving.json` | `NO_WIDTH_SPECIFIC_TRAP`: p25 soft trap = cast-chain artifact |
| `C6` | CoDriving high-dimensional trap hunt | batch×width / stage0 K-alignment 等高维陷阱搜索 | `results/coupling_map/C6_codriving_highdim.json` | `NO_ROBUST_HIGHDIM_TRAP`: 原 high-dim trap 结论撤回 |

### 0.5.6 验证已跑

已跑:

```bash
python -m py_compile framework/stage1/standard_conv_census.py framework/stage1/standard_conv_probe_plan.py
python framework/stage1/standard_conv_census.py
python framework/stage1/standard_conv_probe_plan.py
```

并确认:

- `multi_agent/methods/design/stage1-model-predict/` 无非 `.md` 文件。
- `results/stage1_model_predict/*.json` 三个 JSON 均可解析。
- probe queue = 9 个, 每个 probe 有 `evidence_level / blocking_condition / minimal_inputs / expected_artifacts / pass_fail_criteria / estimated_cost_class`。
- 两个并行子代理已关闭; 实现代理产物已吸收, 审查代理指出的问题已修正到计划/脚本/产物中。

---

## 1. 当前阶段进度表

| 阶段 | 状态 | 说明 |
|---|---|---|
| S0 Evidence Envelope | ✅ 完成 | 明确 `groups=1` 不是 verdict; CoDriving 只是 measured negative anchor for its scope |
| S1 Static Standard-Conv Census | ✅ 完成一版 | 6 manifests / 27 groups / 20 candidates; mixed Conv2d+ConvTranspose2d 已标注需拆分 |
| S2 Low-Cost Anchor Probe Plan | ✅ 规格完成但已降级解释 | 9 个 probe/gate; 不是 9 个可执行实测, attention 当前只是不集成 blocker |
| S2-CoDriving Completion Evidence | ✅ 已绑定既有实测 | p25/p75 fresh seed42 real 已纳入; p75 effective batch=1 |
| S3 Quantization-Sensitivity Probe | ⏳ 未启动新实验 | 只把 V2X-ViT C4/per-stage Q/AP 风险挂入 gate |
| S4 Mini Three-Arm Validation | ⏳ 未开始 | 仅当 S2/S3 标高风险 shape 后才跑 |
| S5 Predictor Rule Update | ⚠️ 计划修正, 代码未实现 | 下一阶段目标应是 Safe Predictor v0: 先防误判, 不证明可分离 |
| S6 Calibration & Reporting | ⚠️ 部分完成 | 已有 measured-results report; 还没生成真正 batch prediction CLI/report |

---

## 2. 下一阶段任务 = Safe Stage1 Predictor v0

下一阶段不要再写散落的设计脚本; 应按 TDD 进入 `framework/stage1/` 真正实现。**停止目标不是跑完 9 个探针, 也不是集成 attention。停止目标是让 Stage1 predictor v0 安全地产生“不会误判 separable”的模型级报告。**

### 2.0 启动方式: 必须拉起两个 agent

新窗口开始下一阶段时, 先并行拉起两个 agent, 不要单 agent 自写自审:

| agent | 角色 | 输入 | 输出 | 硬约束 |
|---|---|---|---|---|
| Agent A: executor | 方案执行 / 代码实现 / 命令复跑 | 本 handoff + design 目录 3 个核心 md + `results/stage1_model_predict/*.json` + 6 个 manifest | 代码、测试、CLI、prediction JSON/MD、命令记录 | 只能按 Safe Predictor v0 停止目标实现; 不得把 skipped attention/fusion/custom 子图提升成 full-model separable |
| Agent B: critic | 批判性验收 / 实验结果审查 / 纠偏建议 | Agent A 的 diff、测试输出、prediction JSON/MD、manifest/evidence 原文 | 审查报告: findings / blocking issues / overclaim checks / required corrections | 不写实现代码; 重点找判据漏洞、scope 外推、证据错绑、未覆盖子图被误判 |

执行顺序:

1. 主 agent 先给两个子 agent 同一份停止目标和验收方案。
2. Agent A 实现最小 Safe Predictor v0, 生成预测结果。
3. Agent B 在 A 的产物完成后立即审查, 但审查必须回到原始 evidence/manifest, 不能只读 A 的总结。
4. 主 agent 汇总 B 的 findings; 凡 `CRITICAL/HIGH` 或涉及 overpromotion/scope/coverage 的问题, 必须让 A 修正后重新跑验收。
5. 只有当 A 的验收命令通过, 且 B 没有 unresolved `CRITICAL/HIGH` finding, 才能宣布下一阶段完成。

Agent B 必须重点检查这些问题:

- 是否仍把 `groups=1` 当作可分离判据。
- 是否把 CoDriving 阴性锚点外推到 F-Cooper/AttFuse/V2X-ViT/Pyramid。
- 是否有 skipped attention/fusion/custom 子图的模型拿到 full-model `MEASURED_SEPARABLE` 或 `PREDICTED_SEPARABLE_LOW_RISK`。
- 是否把局部 dense-backbone low-risk 提升为模型级 verdict。
- 是否丢失 `scope / evidence_level / blockers / required_next_probe_or_gate`。
- 是否把 9 个 probe 误解成 9 个必须立即跑完的实测实验。
- 是否有 JSON/MD 结果和原始 manifest/evidence 不一致。

### 2.1 明确停止目标

下一阶段做到这里就停:

> `scripts/stage1_predict_coupling.py` 能读取当前 6 个 manifest + `results/stage1_model_predict/*.json` evidence, 生成 `results/stage1_model_predict/stage1_coupling_predictions_v0.json/.md`; 报告中没有任何 skipped attention/fusion/custom 子图的模型拿到 full-model `MEASURED_SEPARABLE` 或 `PREDICTED_SEPARABLE_LOW_RISK`; 每个模型都有 `verdict / scope / evidence_level / blockers / required_next_probe_or_gate`; Agent B 的批判性审查没有 unresolved `CRITICAL/HIGH` finding。

不属于下一阶段停止目标:

- 不集成 AttFuse/V2X-ViT attention/fusion 到 trace/export/TVM。
- 不跑新的 H800 TVM anchor tuning。
- 不跑新的 AP/finetune/per-stage quant 实验。
- 不证明 F-Cooper/AttFuse/V2X-ViT full-model separable。
- 不实现完整 low-cost predictor calibration。

### 2.2 先读文件

1. `multi_agent/methods/design/stage1-model-predict/stage1_model_predictor_implementation_plan_v1.md`
2. `multi_agent/methods/design/stage1-model-predict/standard_conv_anchor_measured_results_v1.md`
3. `multi_agent/methods/design/stage1-model-predict/standard_conv_probe_queue_v1.md`
4. `results/stage1_model_predict/standard_conv_census_v1.json`
5. `results/stage1_model_predict/standard_conv_probe_queue_v1.json`
6. `results/coupling_map_matrix.json`

### 2.3 代码任务优先级

1. **Task 1: Predictor fixtures + static contract tests**
   - 新建/实现 `framework/stage1/coupling_predictor.py`
   - 新建测试 fixture 和 `framework/tests/test_coupling_predictor_static.py`
   - 关键验收: static-only 不得输出 `PREDICTED_SEPARABLE_LOW_RISK`

2. **Task 2: Typed skipped-subgraph metadata**
   - `framework/stage1/auto_trace.py`
   - `framework/stage1/adapters.py`
   - `framework/stage1/graph_scan.py`
   - 关键验收: attention/fusion/custom skip 能阻断 full-model verdict; channel-preserving fusion 也必须有 coverage 说明

3. **Task 3: Structural features**
   - 每个 Conv2d/ConvTranspose2d group 要有 `cin/cout/groups/kernel/stride/input_hw/output_hw/fanout`
   - 当前 census 只能选 probe 候选, 不能校准预测器, 因为 manifests 仍缺这些 shape 字段

4. **Task 4/5: Coverage-aware guardrail + anchor probe interface**
   - 绑定 measured/anchor JSON schema
   - 输出必须带 `evidence_level` 和 `scope`

5. **Task 6: Batch prediction CLI**
   - 新建 `scripts/stage1_predict_coupling.py`
   - 输入 6 个 manifest + stage1_model_predict evidence JSON
   - 输出 `results/stage1_model_predict/stage1_coupling_predictions_v0.json`
   - 输出 `results/stage1_model_predict/stage1_coupling_predictions_v0.md`

### 2.4 验收方案

必须全部通过才算下一阶段完成:

1. **目录约束**

```bash
find multi_agent/methods/design/stage1-model-predict -maxdepth 1 -type f ! -name '*.md' -print
```

Expected: no output.

2. **生成器仍可跑**

```bash
python framework/stage1/standard_conv_census.py
python framework/stage1/standard_conv_probe_plan.py
python -m json.tool results/stage1_model_predict/standard_conv_probe_queue_v1.json >/tmp/probe_queue.check.json
```

Expected: all commands exit 0.

3. **Predictor unit tests**

```bash
PYTHONPATH=${V2X_ROOT} pytest -q framework/tests/test_coupling_predictor_static.py
PYTHONPATH=${V2X_ROOT} pytest -q framework/tests/test_stage1_manifest_predictor_fields.py
```

Expected: PASS. If tests do not exist yet, next stage must create them first, following TDD.

4. **Batch prediction CLI**

```bash
PYTHONPATH=${V2X_ROOT} python scripts/stage1_predict_coupling.py \
  --manifest framework/partitions/codriving_partition.yaml \
  --manifest results/autoscan_fcooper_partition.yaml \
  --manifest results/autoscan_attfuse_partition.yaml \
  --manifest framework/partitions/v2xvit_partition.yaml \
  --manifest framework/partitions/pyramid_lidar_partition.yaml \
  --manifest framework/partitions/pyramid_camera_partition.yaml \
  --evidence-dir results/stage1_model_predict \
  --out-json results/stage1_model_predict/stage1_coupling_predictions_v0.json \
  --out-md results/stage1_model_predict/stage1_coupling_predictions_v0.md
```

Expected model-level gates:

| model | expected v0 verdict/scope |
|---|---|
| `codriving` | `ANCHOR_PROBED_LOW_RISK` or `MEASURED_ENVELOPE_ONLY`, scope must not overclaim beyond measured CoDriving envelope |
| `fcooper` | no stronger than `LOW_CONFIDENCE_NEEDS_TARGETED_PROBE` unless MaxFusion static proof is explicitly attached |
| `attfuse` | `FUSION_UNCOVERED_UNKNOWN`; attention blocker present |
| `v2xvit` | `JOINT_OR_PAIR_SEARCH_REQUIRED_UNTIL_C4_C5_BOUND` or stronger blocker; C4/C5 gates present |
| `pyramid_lidar` | `P_HUB_COUPLED` or `P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_STANDARD_CONV_PROMOTION` |
| `pyramid_camera` | `P_HUB_COUPLED` or `P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_STANDARD_CONV_PROMOTION` |

5. **No overpromotion invariant**

Run:

```bash
python - <<'PY'
import json
p='results/stage1_model_predict/stage1_coupling_predictions_v0.json'
d=json.load(open(p))
bad=[]
for item in d.get('predictions', []):
    blockers=item.get('blockers', [])
    verdict=item.get('verdict')
    scope=item.get('scope')
    if blockers and scope == 'full_model' and verdict in {'MEASURED_SEPARABLE','PREDICTED_SEPARABLE_LOW_RISK'}:
        bad.append((item.get('model'), verdict, blockers))
if bad:
    raise SystemExit(f'overpromotion: {bad}')
print('no_overpromotion')
PY
```

Expected: `no_overpromotion`.

6. **Critic agent 验收**

Agent B 必须输出一份批判性审查记录, 可写入 `results/stage1_model_predict/stage1_coupling_predictions_v0_review.md` 或合并进 `stage1_coupling_predictions_v0.md` 的 review section。

审查记录必须包含:

- `findings`: 按 `CRITICAL/HIGH/MEDIUM/LOW` 分级。
- `overpromotion_check`: 明确列出每个 skipped attention/fusion/custom 模型的 verdict/scope 是否被阻断。
- `scope_check`: CoDriving / F-Cooper / AttFuse / V2X-ViT / Pyramid 是否存在跨模型外推。
- `evidence_binding_check`: 每个 verdict 是否能追溯到 manifest/evidence JSON。
- `required_corrections`: 若有 `CRITICAL/HIGH`, Agent A 必须修正并重新跑第 1-5 项验收。
- `unresolved`: 若非空, 主 agent 不得宣布下一阶段完成。

### 2.5 当前 9 个 probe 是否足够?

结论:

- **足够支撑下一阶段 Safe Predictor v0**: 因为 v0 的目标是防止错误 full-model separability, 不是证明可分离。
- **不够支撑“低成本准确预测所有模型可分离性”**: 还缺 attention/fusion 可执行集成、真实 dense-backbone/neck runner、结构 shape 字段、AP-sensitive probe 和 calibration。
- 因此下一阶段停止时, 如果报告里大量模型仍是 low-confidence / uncovered / pair-required, 这不是失败; 只要没有 overpromotion, 就是 v0 成功。

### 2.6 不要做的事

- 不要把 `groups=1` 写成 separability 判据。
- 不要把 CoDriving 的阴性结论外推到 F-Cooper/AttFuse/V2X-ViT。
- 不要让 static-only 输出 `PREDICTED_SEPARABLE_LOW_RISK`。
- 不要把 skipped attention/fusion/custom 子图下的 dense-backbone low-risk 提升为 full-model verdict。
- 不要在 `multi_agent/methods/design/stage1-model-predict/` 放 `.py` 或 `.json`。

---

## 3. 当前模型级 gate 口径

| 模型 | 当前 gate | 为什么 |
|---|---|---|
| CoDriving | `MEASURED_ENVELOPE_ONLY` / scoped `ANCHOR_PROBED_LOW_RISK` | C0c'/C1cod/C6 支持 CoDriving 自己 scope 的阴性锚点, 但不是 universal standard-conv rule |
| F-Cooper | `LOW_CONFIDENCE_NEEDS_TARGETED_PROBE` | 还缺 BaseBEVBackbone schedule anchor + MaxFusion/routing coverage |
| AttFuse | `FUSION_UNCOVERED_UNKNOWN` | attention/fusion 未覆盖时不能 full-model verdict |
| V2X-ViT | `JOINT_OR_PAIR_SEARCH_REQUIRED_UNTIL_C4_C5_BOUND` | C4 Q-granularity×P 与 C5 routing/fusion 必须绑定到预测器 |
| Pyramid lidar/camera | `P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_STANDARD_CONV_PROMOTION` | 局部 standard Conv2d 不能覆盖 grouped-conv IC_BN P-hub |

---

## 4. 关键文件索引

| 用途 | 路径 |
|---|---|
| 本 handoff | `multi_agent/methods/progress/HANDOFF_stage1_model_predict_standard_conv_v1.md` |
| 实施计划 | `multi_agent/methods/design/stage1-model-predict/stage1_model_predictor_implementation_plan_v1.md` |
| 标准卷积深入计划 | `multi_agent/methods/design/stage1-model-predict/standard_conv_coupling_deepening_plan_v1.md` |
| census Markdown | `multi_agent/methods/design/stage1-model-predict/standard_conv_census_v1.md` |
| probe queue Markdown | `multi_agent/methods/design/stage1-model-predict/standard_conv_probe_queue_v1.md` |
| measured-results Markdown | `multi_agent/methods/design/stage1-model-predict/standard_conv_anchor_measured_results_v1.md` |
| census 脚本 | `framework/stage1/standard_conv_census.py` |
| probe queue 脚本 | `framework/stage1/standard_conv_probe_plan.py` |
| census JSON | `results/stage1_model_predict/standard_conv_census_v1.json` |
| probe queue JSON | `results/stage1_model_predict/standard_conv_probe_queue_v1.json` |
| measured-results JSON | `results/stage1_model_predict/standard_conv_anchor_measured_results_v1.json` |
| coupling map matrix | `results/coupling_map_matrix.json` |
| CoDriving P×Q×S | `results/coupling_map/C0c_codriving_pqs.json` |
| CoDriving Q×S | `results/coupling_map/C1_QxS_codriving.json` |
| CoDriving high-dim | `results/coupling_map/C6_codriving_highdim.json` |

---

## 5. 环境与纪律

- 真仓库: `${V2X_ROOT}`
- conda python: `${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python`
- 常用环境:

```bash
PYTHONPATH=${V2X_ROOT}
```

- H800 远端:

```bash
ssh -p 30001 -o ConnectTimeout=45 -o StrictHostKeyChecking=accept-new ${V2X_REMOTE_USER}@<PRIVATE_HOST>
```

- 若需跑 H800 TVM: 使用 `${V2X_DATA_ROOT}/tvm310/bin/python`; GPU 只允许 4/5/6, 用前 `nvidia-smi` 确认 idle。
- 当前阶段文件均未 commit; 除非用户明确要求, 不要 commit。
- 不要轻信 agent 自报; 凡“已实现/已测/已完成”都要复跑命令或读文件核验。

---

## 6. 最容易犯错的点

1. **`C0c' / C1cod / C6` 是实验 cell 编号, 不是模型。** 当前文档已补对照表, 但新汇报里仍要写全称。
2. **p75 effective batch=1**。不要把 p75 和 p25/base/p50 当同 batch 直接比较吞吐。
3. **CoDriving negative anchor 不等于 standard conv universal theorem**。
4. **V2X-ViT 不能只看 dense backbone**。C4/C5 gate 必须绑定。
5. **Pyramid 局部 standard Conv2d 不能覆盖模型级 P-hub**。
6. **设计目录只放 `.md`**; `.py` 进 `framework/stage1/`, `.json` 进 `results/stage1_model_predict/`。
