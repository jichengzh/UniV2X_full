# HANDOFF — Stage2 当前空白：Stage1/Stage2 集成 v2

日期: 2026-06-25

本文继承 `HANDOFF_stage2_gap_stage1_stage2_integration_v1_zh.md`。v1 定义了 Stage1/Stage2 集成工作包的五个空白；v2 记录截至目前的解决进展、已经落盘的代码/文档/验证证据，以及下一阶段应继续推进的真实 measured evidence 与产品化收口计划。

当前结论:

```text
空白 1-5 已完成第一版可执行闭环:
Stage1 manifest + model_classifier
  -> Stage2 runtime gate
  -> Stage2SearchSpace.model_search_policy
  -> Stage2Output + stage2_evidence_delta
  -> Stage1 evidence input index / export repo smoke

下一步不是重新定义接口，而是把 demo/smoke 路径推进到真实 H800 TVM measured evidence、classifier refresh 语义和 CI 固化。
```

---

## 1. v2 继承的硬约束

Stage2 公共输入只保留:

```text
Stage2Input:
  manifest_path
  model_classification_path
```

明确不作为 Stage2 用户级参数暴露:

- `evidence_registry_path`
- `hardware_target`
- `search_policy`

边界解释:

- `hardware_target` 只作为从 `manifest.hw_capability` 派生出的只读 `hardware_context`。
- `Stage2SearchSpace.model_search_policy` 是模型级策略，表达 `joint` / `serial` / `noS/default`；当前不输出 per-knob `dispatch_plan` 作为正式产品契约。
- `stage2_evidence_delta` 是 Stage2 运行后产生的 evidence 更新包，不是 Stage2 启动时要求用户传入的 registry。
- demo/proxy/historical evidence 不能升级成 measured 或 pass 结论。
- dense-core evidence 不能外推成 full-model speedup。

---

## 2. 五个空白解决进展

| 空白 | v1 问题 | 当前状态 | 主要证据 |
|---|---|---|---|
| 空白 1 | `model_classifier` 尚未成为 Stage2 runtime gate | 已接入第一版 gate | `framework/stage2/contracts.py::apply_stage1_gate`；`test_classifier_gate_maps_three_classes_to_runtime_modes` |
| 空白 2 | Stage2 输入/输出契约未落盘 | 已落盘薄契约 | `Stage2Input` 仅含 `manifest_path` / `model_classification_path`；`Stage2Output` 不含顶层 `dispatch_plan` |
| 空白 3 | evidence 更新链路未定义 | 已有 delta schema + 归档入口 + Stage1 evidence input 索引 | `Stage2EvidenceDelta`、`scripts/stage2_update_evidence.py`、`model_classifier._evidence_inputs()["stage2_evidence_delta"]` |
| 空白 4 | Stage2 CLI 不是完整入口 | 已新增统一 CLI | `scripts/stage2_optimize_model.py` |
| 空白 5 | 开源仓库只有 smoke，不是 Stage1->Stage2 使用路径 | 已补 fresh-clone demo smoke 与文档 | `github/stage1-model-scanner-aaai/README*.md`、`docs/stage2-evidence-delta.zh-CN.md`、`docs/stage2-new-hardware.zh-CN.md` |

### 2.1 空白 1: runtime gate

已实现:

- `apply_stage1_gate(manifest_path, model_classification_path)` 会从 classification JSON 中定位对应 model record。
- `SCAN_FAILED` -> `fail_closed`。
- `SEPARABLE_ACCELERATION` -> `serial`。
- `CO_ACCELERATION_REQUIRED` -> `joint`。
- 缺 classification 时默认 `fail_closed`，reason 为 `missing_classification`。
- architecture-only / missing checkpoint 类模型不会进入自动优化。

当前代码入口:

- `framework/stage2/contracts.py`
- `scripts/stage2_optimize_model.py`
- `framework/tests/test_stage2_integration_contract.py`

已验证模型:

- `pyramid_lidar` -> allowed + `joint`
- `codriving` -> allowed + `serial`
- `where2comm` -> not allowed + `fail_closed`

### 2.2 空白 2: Stage2Input / Stage2Output contract

已实现:

- `Stage2Input` 是 dataclass，字段严格为:

```text
manifest_path
model_classification_path
```

- `hardware_context` 由 manifest 派生:

```text
source = manifest.hw_capability
read_only = true
```

- CLI 不接受:

```text
--hardware-target
--evidence-registry
--search-policy
```

- `Stage2Output` 当前核心字段:

```text
schema
model
manifest_digest
public_input
hardware_context
optimization_status
search_space_summary
arms
pareto_front
recommended_configs
optimized_scope
claim_boundaries
evidence_delta
```

显式不做:

- 不输出顶层 `stage1_gate`。
- 不输出顶层 per-knob `dispatch_plan`。
- 不复制完整 Stage1 blockers / unsupported conclusions，只在 `claim_boundaries` 中保留影响本次声明的边界。

### 2.3 空白 3: evidence delta / Stage1 回流

已实现:

- `Stage2EvidenceRecord`
- `Stage2EvidenceDelta`
- `scripts/stage2_update_evidence.py`
- `framework/stage1/model_classifier.py` 的 `_evidence_inputs()` 已纳入:

```text
stage2_evidence_delta -> results/stage1_model_predict/stage2_evidence_delta
```

delta 记录至少包含:

```text
backend
hardware
scope
evidence_kind
provenance
candidate_config
metric
```

当前升级规则:

- 只有 `backend=h800_tvm` 且 `evidence_kind=measured` 的 record 可标记为 `promotable_to_classifier=true`。
- `demo` / `proxy` / `historical` / `estimated` 均不可升级为 pass。
- TRT measured 会被拒绝；TRT 只能作为 historical。

当前限制:

- 现在的 CLI smoke 写出的 delta 是 `demo` evidence，不是 measured evidence。
- `stage2_update_evidence.py` 当前负责验证和归档，不负责自动改写 classifier 判定。
- 下一阶段需要定义 measured delta 被 classifier/calibrated predictor 消费后的精确语义。

### 2.4 空白 4: 统一 CLI

已实现:

```bash
PYTHONPATH=/home/jichengzhi/V2X python scripts/stage2_optimize_model.py \
  --manifest framework/partitions/pyramid_lidar_partition.yaml \
  --classification results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-json results/stage2/pyramid/stage2_optimization_v1.json \
  --evidence-delta-out results/stage2/pyramid/stage2_evidence_delta_v1.json
```

CLI 行为:

- 只要求 manifest + classification。
- 若 classification 缺失，则 fail closed。
- 从 `load_stage2_search_space()` 读取模型级 `model_search_policy`。
- 输出 `Stage2Output`。
- 可选输出 `stage2_evidence_delta`。

当前仍是第一版产品化入口:

- `recommended_configs` 是 smoke-level deterministic recommendation，不是完整 NSGA-II / TVM real search 结果。
- `pareto_front` 目前用于承载 smoke 推荐点，下一阶段要接真实 search result。

### 2.5 空白 5: export repo fresh-clone path

已同步到:

```text
github/stage1-model-scanner-aaai/framework/stage2/
github/stage1-model-scanner-aaai/scripts/stage2_optimize_model.py
github/stage1-model-scanner-aaai/scripts/stage2_update_evidence.py
github/stage1-model-scanner-aaai/scripts/prepare_stage2_demo_data.py
github/stage1-model-scanner-aaai/README.md
github/stage1-model-scanner-aaai/README.zh-CN.md
github/stage1-model-scanner-aaai/docs/stage2-evidence-delta.zh-CN.md
github/stage1-model-scanner-aaai/docs/stage2-new-hardware.zh-CN.md
```

已补能力:

- `prepare_stage2_demo_data.py` 生成 demo manifest、LUT/AP、classification JSON。
- README 增加 Stage2 thin CLI 示例。
- 文档明确:
  - demo/proxy 不是论文实测。
  - `manifest.hw_capability` 是 Stage2 硬件上下文来源。
  - 新硬件必须先进入 Stage1 scan。
  - dense-core 结果不能外推 full-model。

---

## 3. 当前验证记录

最近一次验证命令和结果:

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m unittest framework.tests.test_stage2_integration_contract
```

结果:

```text
Ran 8 tests
OK
```

```bash
PYTHONPATH=/home/jichengzhi/V2X python - <<'PY'
from framework.tests.test_stage2_search_space_contract import (
    test_pyramid_stage2_search_space_has_width_anchors_hierarchy_and_joint_policy,
    test_codriving_stage2_search_space_stays_serial_without_groups1_overpromotion,
)
test_pyramid_stage2_search_space_has_width_anchors_hierarchy_and_joint_policy()
test_codriving_stage2_search_space_stays_serial_without_groups1_overpromotion()
print("stage2_search_space_contract_ok")
PY
```

结果:

```text
stage2_search_space_contract_ok
```

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m py_compile \
  framework/stage1/model_classifier.py \
  framework/stage1/calibrated_predictor.py \
  framework/stage2/contracts.py \
  framework/stage2/__init__.py \
  framework/stage1_bridge.py \
  framework/search_three_arm.py \
  scripts/stage2_optimize_model.py \
  scripts/stage2_update_evidence.py
```

结果: exit code 0。

export repo smoke:

```bash
cd /home/jichengzhi/V2X/github/stage1-model-scanner-aaai
rm -rf framework/partitions results
PYTHONPATH=. python scripts/prepare_stage2_demo_data.py
PYTHONPATH=. python scripts/stage2_optimize_model.py \
  --manifest framework/partitions/pyramid_lidar_partition.yaml \
  --classification results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-json results/stage2/pyramid/stage2_optimization_v1.json \
  --evidence-delta-out results/stage2/pyramid/stage2_evidence_delta_v1.json
PYTHONPATH=. python scripts/stage2_optimize_model.py \
  --manifest framework/partitions/codriving_partition.yaml \
  --classification results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-json results/stage2/codriving/stage2_optimization_v1.json
PYTHONPATH=. python scripts/stage2_update_evidence.py \
  --delta results/stage2/pyramid/stage2_evidence_delta_v1.json \
  --out-dir results/stage1_model_predict/stage2_evidence_delta
```

结果:

```text
stage2_demo_data_ready
stage2_optimize_model_ok model=pyramid_lidar allowed=True mode=joint
stage2_optimize_model_ok model=codriving allowed=True mode=serial
stage2_update_evidence_ok model=pyramid_lidar records=1
```

验证后已清理 export repo smoke 生成的 `results/`、`framework/partitions/` 和 `__pycache__`。

---

## 4. 当前文件索引

核心实现:

- `framework/stage2/__init__.py`
- `framework/stage2/contracts.py`
- `scripts/stage2_optimize_model.py`
- `scripts/stage2_update_evidence.py`

Stage1 回流:

- `framework/stage1/model_classifier.py`

测试:

- `framework/tests/test_stage2_integration_contract.py`
- `framework/tests/test_stage2_search_space_contract.py`

export repo:

- `github/stage1-model-scanner-aaai/framework/stage2/__init__.py`
- `github/stage1-model-scanner-aaai/framework/stage2/contracts.py`
- `github/stage1-model-scanner-aaai/scripts/stage2_optimize_model.py`
- `github/stage1-model-scanner-aaai/scripts/stage2_update_evidence.py`
- `github/stage1-model-scanner-aaai/scripts/prepare_stage2_demo_data.py`
- `github/stage1-model-scanner-aaai/docs/stage2-evidence-delta.zh-CN.md`
- `github/stage1-model-scanner-aaai/docs/stage2-new-hardware.zh-CN.md`

计划记录:

- `docs/superpowers/plans/2026-06-25-stage1-stage2-integration.md`

---

## 5. 当前剩余风险

1. **Stage2 CLI 仍是 smoke optimizer，不是完整 real-search optimizer**
   - 已接 gate、search-space summary 和输出契约。
   - 下一步要把 `framework/search_three_arm.py` 或 TVM measurement loop 的真实结果接进 `pareto_front` / `recommended_configs`。

2. **evidence delta 已有 schema 与归档，但 classifier 消费语义还需深化**
   - 当前 classifier 能索引 `stage2_evidence_delta` 目录。
   - 下一步要定义 measured delta 如何影响 `evidence_level`、`measured_h800_tvm`、`required_next_probe_or_gate` 和 no-overpromotion 检查。

3. **export repo smoke 已有，但还没有 CI**
   - 当前以手动 smoke 命令验证。
   - 下一步应增加轻量 shell check 或 unittest，使 fresh-clone smoke 自动化。

4. **demo data 与论文实测必须继续隔离**
   - 当前文档和 delta schema 已标明 demo/proxy 不升级。
   - 后续写论文表格或 README 结果时仍要避免把 demo 输出混入 measured 结果。

5. **Stage2SearchSpace 与 Stage2Output 的边界要保持稳定**
   - `model_search_policy` 是模型级策略。
   - 不要重新把 per-knob `dispatch_plan` 提升为正式 Stage2Output 主字段。

---

## 6. 下一步计划

### 下一步 A: 接真实 search result

目标:

- `scripts/stage2_optimize_model.py` 从 smoke recommendation 升级为可调用真实 Stage2 search kernel。
- Pyramid / CoDriving 路径能把 `A-joint`、`A-serial`、`A-noS` 的结果写入 `arms`、`pareto_front`、`recommended_configs`。

建议动作:

1. 给 `framework/search_three_arm.py` 增加可复用函数接口，避免 CLI 只能 shell 调脚本。
2. 在 `stage2_optimize_model.py` 中按 gate mode 选择 joint / serial / noS/default。
3. 输出中区分 `search_mode=smoke`、`search_mode=measured_lut`、`search_mode=tvm_live`。

验收:

- Pyramid 输出包含 joint 推荐点。
- CoDriving 输出 serial/low-budget 推荐点。
- `SCAN_FAILED` 仍 fail closed，不运行 search。

### 下一步 B: measured evidence delta 闭环

目标:

- H800 TVM measured latency / build evidence 可写入 `stage2_evidence_delta_v1`。
- classifier refresh 能读取 delta 并反映到 model record 中。

建议动作:

1. 定义 measured delta 文件位置和命名:

```text
results/stage2/<model>/stage2_evidence_delta_v1.json
results/stage1_model_predict/stage2_evidence_delta/<model>_stage2_evidence_delta_v1.json
```

2. 在 `model_classifier.py` 中读取 stage2 delta summary。
3. 为 `demo`、`proxy`、`historical`、`measured` 四类 evidence 写单元测试。

验收:

- `backend=trt, evidence_kind=measured` 被拒绝。
- `backend=h800_tvm, evidence_kind=measured` 能进入 measured evidence summary。
- demo/proxy 不改变 pass / fail 结论。

### 下一步 C: CI / fresh-clone smoke

目标:

- export repo 不依赖人工复制命令验证。

建议动作:

1. 增加 `scripts/stage2_smoke_check.py` 或 shell 脚本。
2. 覆盖:
   - `prepare_stage2_demo_data.py`
   - Pyramid `stage2_optimize_model.py`
   - CoDriving `stage2_optimize_model.py`
   - `stage2_update_evidence.py`
   - py_compile
3. README 中把 smoke 命令改成一个入口。

验收:

```bash
PYTHONPATH=. python scripts/stage2_smoke_check.py
```

输出:

```text
stage2_export_smoke_ok
```

### 下一步 D: 文档收口

目标:

- v1/v2 handoff、主 next-phase plan、export README 三者不再出现旧接口冲突。

建议动作:

1. 更新 `HANDOFF_stage2_codesign_next_phase_plan_v1_zh.md` 中 Stage1/Stage2 集成方向的状态。
2. 检查所有 `dispatch_plan` 表述，只保留 legacy / diagnostic 语境。
3. 检查所有 `hardware_target` 表述，确保它是 manifest-derived read-only context。
4. 检查所有 `evidence_registry_path` 表述，确保不是 Stage2 public input。

验收:

```bash
rg -n "evidence_registry_path|hardware_target|search_policy|dispatch_plan" \
  multi_agent/methods/progress github/stage1-model-scanner-aaai/README*.md
```

每个命中都应处在“禁止暴露 / 只读派生 / legacy diagnostic”语境。

---

## 7. 建议下一轮 `/goal`

```text
/goal 推进 Stage1/Stage2 集成 v2 实测闭环阶段。基于已落盘的 framework/stage2/contracts.py、scripts/stage2_optimize_model.py、scripts/stage2_update_evidence.py 和 Stage2SearchSpace.model_search_policy，把当前 smoke optimizer 接入真实 Stage2 search result 与 H800 TVM measured evidence delta。严格约束：Stage2 公共输入仍只包含 manifest_path 与 model_classification_path；hardware_target 只来自 manifest.hw_capability 的只读 hardware_context；不暴露 evidence_registry_path 或 search_policy；不输出 per-knob dispatch_plan 作为正式契约；SCAN_FAILED fail closed；demo/proxy/historical 不升级为 measured 或 pass；dense-core 不外推 full-model。验收包括：Pyramid/CoDriving real-search output fixture、measured h800_tvm delta classifier refresh 测试、TRT measured 拒绝测试、export repo 一键 smoke、文档旧接口词汇审计。
```

---

## 8. 清空上下文后优先读取

1. `multi_agent/methods/progress/HANDOFF_stage2_gap_stage1_stage2_integration_v2_zh.md`
2. `multi_agent/methods/progress/HANDOFF_stage2_gap_stage1_stage2_integration_v1_zh.md`
3. `framework/stage2/contracts.py`
4. `scripts/stage2_optimize_model.py`
5. `scripts/stage2_update_evidence.py`
6. `framework/tests/test_stage2_integration_contract.py`
7. `framework/stage1/model_classifier.py`
8. `framework/stage1_bridge.py`
9. `framework/tests/test_stage2_search_space_contract.py`
10. `github/stage1-model-scanner-aaai/README.zh-CN.md`
11. `github/stage1-model-scanner-aaai/docs/stage2-evidence-delta.zh-CN.md`
12. `github/stage1-model-scanner-aaai/docs/stage2-new-hardware.zh-CN.md`
