# HANDOFF — Stage2 当前空白：Stage1/Stage2 集成 v1

日期: 2026-06-24

本文是 Stage2 软硬件协同优化框架的四个并行空白工作包之一，聚焦 **Stage1 的扫描/分类结果如何成为 Stage2 优化的正式入口、gate 和输出约束**。主 handoff 索引见 `HANDOFF_stage2_codesign_next_phase_plan_v1_zh.md`。

---

## 1. 本工作包边界

本工作包处理:

- Stage1 manifest 到 Stage2 search space 的正向接口。
- Stage2/TVM 新测量结果到 Stage1 evidence 库、calibrated predictor 和 model classifier 的反向更新接口。
- `model_classifier` 结果进入 Stage2 runtime gate。
- Stage2 input/output contract。
- CLI 串联。
- 开源仓库 fresh clone 的 end-to-end 使用路径。

本工作包不处理:

- 具体 latency LUT 如何测。
- 具体 AP anchors 如何训练/评测。
- Pyramid/CoDriving 实验统计重跑。
- 把 `search_policy` 暴露成用户需要手工选择的产品参数。

---

## 2. 当前已经有的接口

### 2.1 正向接口

```text
Stage1 partition manifest
    ↓
framework/stage1_bridge.py
    ↓
SpaceSpec / KnobSpec / QuantUnit / RoutingSegment
    ↓
framework/search_three_arm.py
    ↓
Stage2 P/S or P/Q/S search
```

当前这条链路已经是 Stage2 真实使用的主通道。

### 2.2 反向 evidence 更新接口

```text
Stage2 / TVM measured evidence delta
    ↓
Stage1 evidence store / results/stage1_model_predict/*.json
    ↓
framework/stage1/calibrated_predictor.py
framework/stage1/model_classifier.py
    ↓
三分类 + blocker + next gate + unsupported conclusions
```

当前这条链路主要服务 Stage1 报告和分类器校准。下一步要明确的是: Stage2 不应把完整 evidence registry 当作用户输入重新解释，而应把新测量写成带 backend/scope/provenance 的 evidence delta，再由 Stage1 evidence/classifier 链路重新吸收。

### 2.3 当前分类器三类

目标三类:

- `CO_ACCELERATION_REQUIRED`
- `SEPARABLE_ACCELERATION`
- `SCAN_FAILED`

Stage2 应把这三类解释为:

```text
SCAN_FAILED -> fail closed，不自动优化
SEPARABLE_ACCELERATION -> serial/default low-budget
CO_ACCELERATION_REQUIRED -> joint 或 Stage2SearchSpace.model_search_policy-driven
```

---

## 3. 当前主要空白

### 空白 1: `model_classifier` 尚未成为 Stage2 runtime gate

当前 Stage2 搜索器不会强制读取分类结果。实际运行仍主要依赖 manifest + LUT/AP/Q。

需要补:

- Stage2 启动时读取 classification JSON。
- 若模型 `SCAN_FAILED`，默认拒绝自动优化。
- 若模型缺 ckpt / architecture-only，拒绝 checkpoint optimization。
- 若只有 dense-core evidence，输出不得升级到 full-model claim。
- 分类器的 blocker / unsupported 信息不全量复制到 Stage2 输出，只提取会影响本次优化声明的最小 claim boundary。

### 空白 2: Stage2 正式输入/输出契约未落盘

建议新增:

```text
framework/stage2/contracts.py
scripts/stage2_optimize_model.py
```

修正后的输入契约应保持很薄。Stage2 公共入口只消费 Stage1 已经收敛出的产物:

```text
Stage2Input:
  manifest_path
  model_classification_path
```

字段边界:

- `manifest_path` 是 Stage2 的主入口，必须已经包含 `hw_capability`、`view_b1_search_groups`、`view_b2_quant_units`、`view_d_routing_segments`、`round_to`、`int8_buildable_align` 等 Stage1 收缩后的搜索空间信息。
- `model_classification_path` 是 Stage1 classifier 的 gate 产物。短期可以作为 CLI 参数传入；中长期可由 manifest 中的模型标识自动定位或把 compact gate summary 回写进 manifest。
- `evidence_registry_path` 不应作为 Stage2 用户输入。原始 evidence 应在 Stage1/classifier 或 TVM evidence 更新链路中完成归档、筛选和标注；Stage2 最多读取分类产物中的 `evidence_level`、`backend_policy`、`scope` 等 compact metadata。
- `hardware_target` 不应作为 Stage2 用户输入。当前 Stage1 `run_scan.py --hw`、`hardware_scan.py`、`graph_scan.py` 已把硬件 capability 归一化进 manifest，并用 `round_to` / `int8_buildable_align` 等字段收缩搜索空间。Stage2 只校验 manifest 内硬件与测量 backend 是否一致，不提供覆盖入口。
- `search_policy` 不应作为用户可选参数。它是实验/调度内部变量，应由 `model_classifier.acceleration_class` 和 `Stage2SearchSpace.model_search_policy` 自动派生；如需调试，只放在 hidden/debug flag 或实验脚本中。

建议输出:

```text
Stage2Output:
  schema
  model
  manifest_digest
  optimization_status
  search_space_summary
  arms
  pareto_front
  recommended_configs
  optimized_scope
  claim_boundaries
  evidence_delta
```

输出字段边界:

- `stage1_gate` 不作为正式主字段输出，避免把 Stage1 结果重复包装成 Stage2 结论。若需要审计，可在 `optimization_status` 中保留 `allowed/mode/reason/source_classification`。
- `evidence_summary` 改为 `evidence_delta`。它不是 Stage2 读取 evidence 后的摘要，而是 Stage2 运行中产生的新测量或新结论，供 Stage1 evidence 库回收。
- `blockers` / `unsupported_conclusions` 不默认复制全量 Stage1 列表。Stage2 只在 `claim_boundaries` 中保留会影响本次优化声明的最小约束，例如 `full_model_claim_allowed=false`、`optimized_scope=dense_core_only`、`measurement_backend=historical_trt_context_only`。
- 若 gate 失败或 scope 受限，`optimization_status.reason` 和 `claim_boundaries` 必须足够解释为什么拒绝优化或为什么不能升级为 full-model claim。

### 空白 3: Stage2 evidence 更新链路未定义

目前 Stage1 classifier 读 `results/stage1_model_predict/*`，Stage2 搜索脚本读 `results/latency_lut_*.json` / `results/ap70_model_*.json`。这里容易混淆两类 evidence:

1. **Stage1 gate evidence**: 用于分类、scope、blocker、no-overpromotion。
2. **Stage2 in-run measurement evidence**: TVM/Relax/MetaSchedule 或 benchmark 在 Stage2 运行中产生的 latency/build/AP 记录。

修正后的职责不是让 Stage2 再消费一个完整 `evidence_registry_path`，而是让 Stage2 把新 evidence 更新回 Stage1 evidence 库:

```text
Stage1 manifest + classifier gate
    ↓
Stage2 search / TVM measurement loop
    ↓
stage2_evidence_delta.json
    ↓
Stage1 evidence store
    ↓
calibrated_predictor / model_classifier 重新生成 gate
```

关键要求:

- Stage2 可以在运行中读取本次 measurement cache / LUT，但这不是用户级 `evidence_registry_path`，而是内部测量缓存或实验 fixture。
- Stage2 写出的 evidence delta 必须包含 `backend`、`hardware`、`scope`、`measured/demo/proxy/historical`、`provenance`、`candidate_config` 和 `metric`。
- 同一份 evidence 不能在 Stage2 delta 里说 measured，在 classifier 里说 historical。
- H800 TVM 与 historical TRT 必须全链路隔离，TRT 只能作为 historical context 或独立 backend scope。
- demo/proxy evidence 不允许升级成 pass 结论，也不能触发 full-model claim。

### 空白 4: Stage2 CLI 还不是完整产品入口

当前存在多个脚本:

- `framework/run_b4_ablation.py`
- `framework/run_pqs_ablation.py`
- `framework/run_pqs_codriving.py`
- `scripts/prepare_stage2_demo_data.py`

需要新增或整合:

```text
scripts/stage2_optimize_model.py
```

建议流程:

```text
load Stage1 manifest
load model_classifier output
apply stage2 gate
build search space
derive internal search mode from classifier + Stage2SearchSpace.model_search_policy
run search / optional TVM measurement update
write Stage2Output JSON/MD
write stage2_evidence_delta.json if new measured evidence exists
rerun or schedule calibrated/model_classifier refresh
```

### 空白 5: 开源仓库是 smoke 可运行，不是完整实验复现包

当前 `github/stage1-model-scanner-aaai/` 已经可跑 demo，但还缺:

- 真实模型 config/ckpt 示例。
- 真实 LUT/AP 生成教程。
- Stage2 evidence delta / evidence update 文档。
- 新硬件接入教程。
- CI / pytest。
- Stage1→Stage2 end-to-end 最小真实样例。

---

## 4. 修复计划与终止条件

终止条件:

```text
空白 1-5 全部完成并通过对应验收后，Stage1/Stage2 集成工作包才算修复成功。
不能只完成 CLI smoke 或只完成契约文档就结束 /goal。
```

### 计划 1: 修复空白 1，接入 classifier runtime gate

目标:

- Stage2 启动时读取 `model_classification_path`。
- 从 classification 中定位与 `manifest_path` 对应的 model record。
- `SCAN_FAILED` fail closed。
- `SEPARABLE_ACCELERATION` 自动派生 serial/default low-budget。
- `CO_ACCELERATION_REQUIRED` 自动派生 joint 或 model-search-policy-driven。
- 缺 ckpt、architecture-only、dense-core-only 等限制写入 `optimization_status` / `claim_boundaries`。

代码入口:

- `framework/stage1/model_classifier.py`
- `framework/stage1_bridge.py`
- `framework/search_three_arm.py`
- `scripts/stage2_optimize_model.py`

验收:

- `where2comm` / `v2vnet` / `disconet` 这类 `SCAN_FAILED` 不进入自动优化。
- CoDriving 默认 serial/low-budget。
- Pyramid 默认 joint/P-hub policy。
- 测试不依赖 `if model == "pyramid"`，只依赖 classifier class + dispatch signal。

### 计划 2: 修复空白 2，落盘薄输入/输出契约

目标:

- 新增 `framework/stage2/contracts.py`。
- `Stage2Input` 只保留 `manifest_path` 和 `model_classification_path`。
- 不暴露 `evidence_registry_path`、`hardware_target`、`search_policy`。
- `Stage2Output` 输出优化结果、claim 边界和 evidence delta，而不是重复输出完整 Stage1 gate。

代码入口:

- `framework/stage2/contracts.py`
- `scripts/stage2_optimize_model.py`

验收:

- Contract fixture 覆盖 Pyramid / CoDriving。
- 输出包含 `optimization_status`、`optimized_scope`、`claim_boundaries`。
- 输出不包含用户级 `evidence_registry_path`、`hardware_target`、`search_policy`。
- `stage1_gate` 不作为正式主字段出现；如需审计，只作为 `optimization_status.source_classification`。

### 计划 3: 修复空白 3，建立 Stage2 evidence 更新链路

目标:

- 新增 Stage2 evidence delta schema。
- Stage2 运行中产生的 TVM/benchmark evidence 写入 `stage2_evidence_delta.json`。
- Stage1 evidence store / calibrated predictor / model classifier 负责吸收 delta 并重新生成 gate。
- Stage2 可用本次内部 measurement cache 或实验 fixture，但不把完整 registry 暴露为 CLI 输入。

代码入口:

- `framework/stage2/contracts.py`
- `framework/stage1/calibrated_predictor.py`
- `framework/stage1/model_classifier.py`
- `scripts/stage2_optimize_model.py`
- 后续可新增 `scripts/stage2_update_evidence.py`

验收:

- `stage2_evidence_delta.json` 至少包含 `backend`、`hardware`、`scope`、`evidence_kind`、`provenance`、`candidate_config`、`metric`。
- H800 TVM measured 与 historical TRT 在 delta 和 classifier 中保持隔离。
- demo/proxy evidence 不会触发 pass 结论或 full-model claim。

### 计划 4: 修复空白 4，形成统一 CLI

目标:

- 新增 `scripts/stage2_optimize_model.py`。
- CLI 只要求 Stage1 manifest 和 Stage1 classification。
- 搜索模式由 classifier + `Stage2SearchSpace.model_search_policy` 自动派生。
- 可选输出 Stage2 result JSON/MD 和 evidence delta JSON。

验收命令:

```bash
PYTHONPATH=${V2X_ROOT} python scripts/stage2_optimize_model.py \
  --manifest framework/partitions/pyramid_lidar_partition.yaml \
  --classification results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-json results/stage2/pyramid/stage2_optimization_v1.json \
  --evidence-delta-out results/stage2/pyramid/stage2_evidence_delta_v1.json
```

没有 classification 时默认 fail closed；demo mode 必须显式 `--demo` 并在输出里标注。

### 计划 5: 修复空白 5，补齐开源最小复现路径

目标:

- 开源仓库 README 保留 smoke 路径。
- 新增真实模型替换 demo 数据 checklist。
- 新增 Stage2 evidence delta 格式说明。
- 新增新硬件 Stage1 scan -> manifest -> Stage2 optimize 的接入说明。
- 增加最小 pytest 或 shell smoke。

代码/文档入口:

- `github/stage1-model-scanner-aaai/README.md`
- `github/stage1-model-scanner-aaai/README.zh-CN.md`
- `github/stage1-model-scanner-aaai/docs/stage2-evidence-delta.zh-CN.md`
- `github/stage1-model-scanner-aaai/docs/stage2-new-hardware.zh-CN.md`
- `github/stage1-model-scanner-aaai/scripts/prepare_stage2_demo_data.py`

验收:

- fresh clone 能跑 Stage1→Stage2 demo smoke。
- 文档明确 demo/proxy 不等于论文实测结果。
- 文档明确 manifest 中的 `hw_capability` 是 Stage2 硬件约束来源，不要求用户在 Stage2 再填硬件目标。

---

## 5. 推荐代码入口

Stage1:

- `framework/stage1/model_classifier.py`
- `framework/stage1/calibrated_predictor.py`
- `scripts/stage1_classify_models.py`

Bridge / Stage2:

- `framework/stage2/contracts.py`
- `framework/stage1_bridge.py`
- `framework/search_three_arm.py`
- `framework/run_b4_ablation.py`
- `framework/run_pqs_ablation.py`
- `framework/run_pqs_codriving.py`
- `scripts/stage2_optimize_model.py`
- 后续可选 `scripts/stage2_update_evidence.py`

Export repo:

- `github/stage1-model-scanner-aaai/README.md`
- `github/stage1-model-scanner-aaai/README.zh-CN.md`
- `github/stage1-model-scanner-aaai/docs/stage2-evidence-delta.zh-CN.md`
- `github/stage1-model-scanner-aaai/docs/stage2-new-hardware.zh-CN.md`
- `github/stage1-model-scanner-aaai/scripts/prepare_stage2_demo_data.py`

---

## 6. 建议验收

最小静态验收:

```bash
PYTHONPATH=${V2X_ROOT} python -m py_compile \
  framework/stage1/model_classifier.py \
  framework/stage1/calibrated_predictor.py \
  framework/stage2/contracts.py \
  framework/stage1_bridge.py \
  framework/search_three_arm.py \
  scripts/stage2_optimize_model.py
```

分类器结果快检:

```bash
PYTHONPATH=${V2X_ROOT} python - <<'PY'
import json
from pathlib import Path

p = Path("results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json")
if p.exists():
    d = json.loads(p.read_text())
    assert d["schema"] == "stage1_model_classification_v1"
    assert d["no_overpromotion"] is True
    assert "models" in d
    print("stage1_classifier_present")
else:
    print("stage1_classifier_missing")
PY
```

正式验收应补:

- Stage2 gate fixture: `SCAN_FAILED` fail closed，CoDriving serial/low-budget，Pyramid joint/P-hub。
- missing classification fail-closed test；只有显式 `--demo` 才能进入 demo mode。
- Stage2Input schema test: 不允许用户级 `evidence_registry_path`、`hardware_target`、`search_policy`。
- Stage2Output schema test: 输出 `optimization_status`、`optimized_scope`、`claim_boundaries`、`evidence_delta`。
- Evidence delta schema test: `historical_trt` 不进入 new measured backend，H800 TVM measured 与 historical TRT 隔离。
- No-overpromotion test: dense-core evidence 不得生成 full-model claim。
- CLI smoke: 只传 `--manifest`、`--classification`、`--out-json`、可选 `--evidence-delta-out`。

---

## 7. 本方向 `/goal` 启动指令

```text
/goal 推进 Stage1/Stage2 集成工作包，终止条件是空白 1-5 全部修复并通过验收。将 Stage1 model_classifier 正式接入 Stage2 runtime gate，建立薄 Stage2Input/Stage2Output 契约和 scripts/stage2_optimize_model.py 统一入口，打通 manifest + classification -> gate -> Stage2SearchSpace.model_search_policy -> search/TVM measurement -> Stage2Output + stage2_evidence_delta -> Stage1 evidence/calibrated/model_classifier refresh 链路。严格约束：Stage2 公共输入只包含 manifest_path 与 model_classification_path；evidence_registry_path、hardware_target、search_policy 不作为用户级 Stage2 参数；hardware_target 只作为从 manifest.hw_capability 派生的只读 hardware_context；Stage2SearchSpace.model_search_policy 是模型级策略，不输出 per-knob dispatch_plan；SCAN_FAILED fail closed；dense-core evidence 不外推 full-model；H800 TVM measured 与 historical TRT 全链路隔离；demo/proxy 不升级成 measured 或 pass 结论。验收包括：空白1 gate fixture，空白2 contract/schema，空白3 evidence delta/update，空白4 CLI smoke，空白5 fresh-clone/docs smoke；五个空白任一未通过则 /goal 不结束。
```
