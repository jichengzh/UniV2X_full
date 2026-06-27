# HANDOFF — Stage2 当前空白：搜索空间构建 v1

日期: 2026-06-25

本文是 Stage2 软硬件协同优化框架的四个并行空白工作包之一，聚焦 **搜索空间如何从 Stage1 manifest 自动、保守、可验证地构建出来**。主 handoff 索引见 `HANDOFF_stage2_codesign_next_phase_plan_v1_zh.md`。

2026-06-25 修订重点:

- P 轴候选不能继续停留在固定 `25%/50%/75%` 剪枝率，而应从真实层结构、硬件对齐、AP anchor 和 schedule-LUT 可测点生成离散宽度锚点。
- block 类型只是粗分类，dense backbone 内部还必须进一步拆成 stage/block/group 层级，例如 Pyramid 的 stage1/stage2/stage3。
- Q 轴虽然在既有实验中信号不显著，但仍应作为正式成员进入搜索空间；优先考虑把 P/Q 合并为软件配置轴，再与硬件 schedule 轴做模型级 joint/serial 判定。
- 当前 Stage2 判断的是模型级优化策略，即 Pyramid/CoDriving 这类模型走 joint 还是 serial，不是对每个 knob 单独判定 joint/serial。因此原“dispatch_plan 按 knob 驱动预算”暂不作为独立空白。
- dense-core/full-model 边界需要保留，但在当前 RSU/ego 场景中可弱化为 scope 标注与 claim 约束；RSU 侧目前主要按 dense core 优化，ego fusion/attention 作为未来扩展。

2026-06-25 实现状态:

- 已在 `framework/stage1_bridge.py` 新增 `load_stage2_search_space(path)` 和 `SpaceSpec.stage2_search_space()`，旧 `SpaceSpec.dispatch_plan()` 保留用于历史脚本兼容。
- 已新增 `framework/tests/test_stage2_search_space_contract.py`，覆盖 Pyramid 与 CoDriving 两个 fixture。
- 当前实现能输出:
  - `software_candidates`: P width anchors + Q policy。
  - `hardware_candidates`: default/tuned schedule backend 候选。
  - `model_search_policy`: 模型级 `joint` / `serial` / `noS/default` 策略，不做 per-knob joint/serial。
  - `hierarchical_blocks`: `device_scope -> execution_block -> dense_stage -> search_group -> knob`。
  - `claim_scope`: RSU dense-core 当前闭环、full-model claim 禁止、ego fusion/attention future block。
- 已验证:
  - Pyramid 输出 `joint`，并包含 `P_IC_BN_hub` 原因。
  - Pyramid dense backbone 输出 `stage1/stage2/stage3`。
  - CoDriving 输出 `serial`，且 `groups=1` 不被写成模型级可分离证明。
  - legacy `25%/50%/75%` 被映射为合法 width anchor，其中不合法 INT8 快核的点标为 `diagnostic_trap`。

本轮验证命令:

```bash
PYTHONPATH=/home/jichengzhi/V2X /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest framework/tests/test_stage2_search_space_contract.py -q
PYTHONPATH=/home/jichengzhi/V2X /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest framework/tests/test_coupling_predictor_static.py framework/tests/test_stage1_manifest_predictor_fields.py -q
PYTHONPATH=/home/jichengzhi/V2X /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m py_compile framework/stage1_bridge.py framework/search_three_arm.py
```

---

## 1. 本工作包边界

本工作包只处理搜索空间定义与收缩，不处理 latency/AP 证据生成，也不处理论文实验数字收口。

输入边界:

- Stage1 partition manifest。
- 硬件 capability YAML / manifest 中的硬件合法性字段。
- `framework/stage1_bridge.py` 生成的 `SpaceSpec`。

输出边界:

- 可搜索的 P/Q/S/D 旋钮集合。
- 每个旋钮的合法候选值。
- 模型级 joint/serial 搜索策略、预算口径和触发依据。
- 哪些子图被排除、哪些只能作为 blocker 或 coverage caveat。

---

## 2. 当前已经有的基础

当前代码已经具备以下基础:

- `framework/stage1_bridge.py`
  - 读取 partition manifest。
  - 生成 `SpaceSpec` / `KnobSpec` / `QuantUnit` / `RoutingSegment`。
  - 能从 `view_b1_search_groups`、`hw_capability`、`int8_buildable_align` 等字段推导 legal widths。
  - 能输出 `cliff_strength`、`schedule_headroom`、`coupling_score` 等结构信号。
- `framework/search_three_arm.py`
  - 已有 P/S 或 P/Q/S 三臂搜索内核。
  - 已能比较 `A-noS`、`A-serial`、`A-joint`。
- Stage1 manifest 当前能提供:
  - dense-core trace boundary。
  - `skipped_subgraphs` / legacy `skipped_modules`。
  - `view_b1_prune_groups`。
  - `view_b1_search_groups`。
  - `view_b2_quant_units`。
  - `view_d_routing_segments`。
  - hardware capability。

当前原则:

```text
搜索空间来自模型结构扫描 + 硬件合法性约束；
不是来自人工指定某个模型一定协同或一定可分离。
Stage2 的 joint/serial 结论是模型级策略，不是 knob 级策略。
```

---

## 3. 当前主要空白

### 空白 1: 宽度候选生成仍不够通用

Stage2 当前能消费 manifest 中已有的 candidate widths，但还没有完整实现“从每个 dense candidate path 的真实层结构自动生成 width anchors”的通用流程。过去大量实验使用 `25%/50%/75%` 三个固定剪枝率，这只能作为 smoke / coarse ablation，不应该作为正式搜索空间定义。

正式 P 轴应以 **宽度候选** 为一等成员，剪枝率只是由 `1 - candidate_width / baseline_width` 派生出来的报告字段。真实剪枝率近似连续，Stage2 需要把连续空间离散为少量可测、可解释、硬件合法的锚点。

需要补齐:

- 按每层 `cin/cout/groups/kernel/stride` 计算合法宽度。
- grouped conv 使用 `IC_BN = cin / groups` 或等价口径判断 tensor-core / schedule 友好性。
- standard conv 不能因为 `groups=1` 就直接写成模型级可分离，只能说明该 dense group 缺少 grouped-conv 式 IC_BN cliff。
- 最大剪枝率不能写死，应由以下约束共同确定:
  - 结构合法性: residual / concat / neck / head 连接处的通道一致性不能破坏。
  - 硬件合法性: `int8_align`、`fp16_align`、`int8_pack_factor`、tensor-core tile、grouped conv 的 `IC_BN` 下限。
  - AP 风险边界: 没有 AP anchor 或短微调证据时，production 候选默认不超过保守上限；更激进点只能作为 diagnostic trap anchor。
  - schedule 可测边界: 没有 latency LUT 或 TVM schedule 可构建证据的宽度不能进入 production candidate。
- width candidates 应区分:
  - production legal anchors。
  - diagnostic trap anchors。
  - AP anchor widths。
  - schedule-LUT measured widths。
- 候选取点逻辑:
  - 必含 baseline width。
  - 必含硬件 cliff 两侧的邻近合法宽度，例如 align/tile/`IC_BN` plateau 的前后点。
  - 必含已有 AP anchor 和 schedule-LUT measured width。
  - 对每个 dense stage/group，优先保留 5-9 个候选；小空间可少于 5 个，大空间超过 9 个时按“硬件 cliff、AP anchor、latency anchor、覆盖低/中/高剪枝区间”的优先级截断。
  - `25%/50%/75%` 可以继续作为 legacy coarse report，但正式实现应把它们映射到最近合法宽度，并在输出中说明是否为 production anchor 或 diagnostic trap。

### 空白 2: block / knob 边界仍偏 fixture 化

当前 Pyramid / CoDriving 已经能工作，但 block 切分仍高度依赖已有 manifest 与历史实验组织方式。

需要补齐:

- 从 Stage1 search group 自动形成 Stage2 的层级结构，而不是只按粗 block 类型一刀切。
- 建议层级:
  - `device_scope`: RSU / ego / vehicle / infrastructure。
  - `execution_block`: dense perception / fusion / routing / postprocess / skipped custom。
  - `dense_stage`: backbone 内部 stage，例如 Pyramid 的 stage1/stage2/stage3。
  - `search_group`: 实际 P/Q 候选绑定的 group。
  - `knob`: width、quant policy、schedule family 等最低层旋钮。
- block 类型仍需保留，但只能作为粗分类:
  - serial dense block。
  - fused dense block。
  - routing/fusion block。
  - skipped blocker block。
  - unsupported custom block。
- 对于 dense backbone，必须在 serial/fused dense block 内继续细分 stage/block/group。Pyramid 这类模型至少要能表达 stage1、stage2、stage3 的候选差异，不能只把整个 dense backbone 合成一个不可解释的大 block。
- 对于 fusion / attention / routing，不允许简单拆成 dense serial block 后相加。
- 对 RSU / ego 两端，必须保留设备边界和并行/同步语义，不能把两个 agent 放进一个单设备 block 里算。当前可先把 RSU 侧 dense core 作为主优化对象；ego 侧 fusion/attention 进入 future block 或 blocker，不作为本工作包的硬性闭环目标。

### 空白 3: Q/D 轴还没有成为统一搜索空间成员

当前 P/S 主线最清楚，Q 轴和 D/routing 轴仍是部分脚本、部分证据、部分 future note。需要注意: 既有实验显示 Q 轴单独信号并不显著，因此不应把“单独扩大 Q 搜索维度”作为默认路线。

更合理的路线是把 P 和 Q 合并成软件配置轴:

```text
software_config = pruning_width_candidate + quant_policy
hardware_config = schedule/backend/tile policy
model_policy = joint 或 serial
```

这样 Q 轴仍是正式搜索空间成员，但它不一定单独成为一个强信号搜索臂；它可以作为每个 P 候选的可构建/不可构建、精度代价、backend scope 和 latency evidence 属性，随 P 轴一起进入软件候选，再和 S 轴做模型级策略判断。

需要补齐:

- `QuantUnit` 到 `QLookup` 的正式桥接。
- Q 轴候选必须带 backend/scope:
  - `measured_h800_tvm`
  - `historical_trt`
  - `proxy`
  - `not_available`
- `RoutingSegment` 需要进入搜索空间 summary，即使暂时不可优化，也要成为 blocker 或 fixed segment。
- D/routing 不能只在论文叙述中存在。
- Stage2 输出需要说明 Q 轴在当前模型中是:
  - active software candidate。
  - fixed quant policy。
  - evidence-only attribute。
  - unavailable / blocker。
- P/Q 合并后，三臂比较应仍然保持模型级语义: `A-joint`、`A-serial`、`A-noS` 比较的是模型整体搜索策略，而不是每个 knob 的局部策略。

### 已弱化项 4: `dispatch_plan()` 按 knob 驱动预算不是当前空白

原文把 `dispatch_plan()` 理解成“对每个 knob 判断 joint/serial 并分配预算”，这是误解。当前 Stage2 要回答的是模型级问题:

```text
这个模型在当前硬件/搜索空间/证据下，应该走 joint 优化还是 serial 优化？
```

因此:

- Pyramid / CoDriving 这种模型级分类和三臂结果是主判据。
- `coupling_score`、`cliff_strength`、`schedule_headroom` 可以作为模型级判断的特征或解释材料。
- 暂时不要求实现 per-knob dispatch budget，也不把它列为空白阻塞项。
- 后续若要恢复 `dispatch_plan()`，应把它定义为模型级 search policy summary，而不是 knob 级优化器。

### 已弱化项 5: dense-core 与 full-model 边界保留，但不作为当前主空白

Stage1 已经记录 skipped subgraphs，但 Stage2 当前主要优化 traced dense core。

这个边界必须在输出中保留，但当前不要求把 ego fusion/attention 一并纳入本工作包闭环。原因:

- 对当前 RSU/ego 部署理解，RSU 侧主要是 dense core perception，因此先把 RSU dense core 做成可测、可解释、可迁移的优化闭环是合理目标。
- ego 侧 fusion/attention 仍可能决定 full-model latency 和检测效果，但应作为后续 fusion/attention 研究内容，不阻塞本阶段搜索空间收口。
- 如果只测 dense core，论文和报告仍必须标注 scope，不能写成完整 full-model speedup。

当前需要在搜索空间层明确:

- `optimized_scope = dense_core` 时，只能输出 dense-core 优化结论。
- `full_model_claim_allowed = false` 时，不能写 full e2e speedup。
- skipped sparse / fusion / routing / postprocess 应进入 coverage caveat 或后续 block list。

---

## 4. 下一阶段目标

目标 A: 建立 `Stage2SearchSpace` 契约。

建议字段:

```text
schema
model
hardware_target
optimized_scope
software_candidates
hardware_candidates
model_search_policy
hierarchical_blocks
quant_units
routing_segments
skipped_blocks
legal_candidate_policy
unsupported_conclusions
```

其中:

- `software_candidates` 至少包含 P width anchor 和 Q policy。
- `hardware_candidates` 至少包含 schedule/backend/tile policy。
- `model_search_policy` 表达模型级 `joint`、`serial`、`noS/default` 选择依据。
- `hierarchical_blocks` 表达 device -> execution block -> dense stage -> search group -> knob 的层级。

目标 B: 让 `stage1_bridge.py` 产物稳定可测。

验收:

- Pyramid grouped conv knob 被识别为 P(IC_BN)-hub 风险源。
- Pyramid dense backbone 至少能表达 stage1/stage2/stage3 层级，不把整个 backbone 压成单个不可解释 block。
- CoDriving standard conv dense group 不因为 `groups=1` 被外推成 full-model separable。
- Where2comm / V2VNet / DiscoNet 缺 ckpt 时不能生成可优化 full search space。

目标 C: 建立 P 候选生成策略。

验收:

- 每个 dense stage/group 输出 baseline、production anchors、diagnostic anchors。
- 输出 `max_prune_rate` 及其来源: structure bound、hardware bound、AP bound、LUT bound。
- `25%/50%/75%` 不再作为唯一候选来源；若出现，必须说明其映射到哪个合法 width。
- 单个 dense stage/group 默认候选数控制在 5-9 个，除非结构空间天然更小。

目标 D: 将 Q 轴并入正式搜索空间。

验收:

- `QuantUnit` 到 `QLookup` 或 evidence registry 的桥接稳定。
- Q 轴可以作为 `software_candidate` 的一部分与 P 绑定。
- 输出说明 Q 在当前模型中是 active / fixed / evidence-only / unavailable。
- Q 信号不显著时，不强行扩大单独 Q 搜索臂。

目标 E: 保留 scope/claim 约束。

验收:

- RSU dense-core 优化可作为当前闭环目标。
- ego fusion/attention 明确进入 future block 或 blocker。
- 只测 dense core 时，输出 `full_model_claim_allowed = false`。

---

## 5. 填补这些空白的工作计划

停止目标:

```text
Stage2SearchSpace 能从 Stage1 manifest 自动生成层级化搜索空间；
P 候选有可解释的最大剪枝率、取点逻辑和候选数量；
P/Q 合并为正式 software candidate；
模型级 joint/serial/noS 策略可输出；
RSU dense-core scope 与 full-model claim 边界清楚；
搜索空间相关空白在文档和 fixture 验收上全部收口。
```

### 任务 1: 定义 `Stage2SearchSpace` schema

改动:

- 在 `framework/stage1_bridge.py` 增加或稳定导出 search-space summary。
- 字段包含 `software_candidates`、`hardware_candidates`、`model_search_policy`、`hierarchical_blocks`、`quant_units`、`routing_segments`、`skipped_blocks`、`claim_scope`。

验收:

- Pyramid / CoDriving fixture 都能导出 JSON-like summary。
- 输出中没有 per-knob joint/serial 结论，只有模型级策略。

### 任务 2: 实现 P 轴 width anchor 生成策略

改动:

- 从 dense candidate path 的 `cin/cout/groups/kernel/stride` 生成合法 width。
- 计算 `max_prune_rate`，并记录 structure/hardware/AP/LUT 四类约束来源。
- 将 legacy `25%/50%/75%` 映射为最近合法 width，仅作为 coarse anchor 或 diagnostic anchor。

验收:

- 每个 dense stage/group 默认 5-9 个候选。
- Pyramid grouped conv 候选覆盖 `IC_BN` cliff 两侧。
- CoDriving standard conv 不因 `groups=1` 生成模型级可分离结论。

### 任务 3: 实现层级 block 边界

改动:

- 将 block 结构改成 device -> execution block -> dense stage -> search group -> knob。
- Pyramid 至少识别 stage1/stage2/stage3。
- RSU / ego 保留设备边界；ego fusion/attention 暂列 future block 或 blocker。

验收:

- Pyramid 输出中能看到 stage1/stage2/stage3。
- RSU dense core 可单独作为 `optimized_scope`。
- fusion/attention 不被拆成 dense serial block 累加。

### 任务 4: 将 P/Q 合并为 software candidate

改动:

- `QuantUnit` 与 `QLookup` / evidence registry 建立正式桥接。
- 每个 P candidate 可挂载 Q policy: fp16 / int8 / mixed / unavailable。
- Q policy 必须带 backend/scope/provenance。

验收:

- 输出说明 Q 是 active、fixed、evidence-only 或 unavailable。
- Q 信号弱时，仍能作为软件候选属性参与模型级策略，不要求单独搜索臂显著。

### 任务 5: 输出模型级搜索策略

改动:

- 保留 `A-joint`、`A-serial`、`A-noS` 的模型级语义。
- 使用模型分类器、三臂历史结果、coupling/cliff 结构信号共同解释模型级策略。
- 不实现 per-knob dispatch budget。

验收:

- Pyramid 输出为 joint/P(IC_BN)-hub 风险源。
- CoDriving 输出为 serial 或 low-budget serial 倾向。
- 测试不依赖简单 `if model == "pyramid"`，而是依赖结构和分类证据。

### 任务 6: 验证 scope 与 claim 约束

改动:

- search space summary 输出 `optimized_scope`、`full_model_claim_allowed`、`future_blocks`。
- dense-core-only 结果禁止写成 full-model speedup。

验收:

- RSU dense-core 闭环可通过。
- ego fusion/attention 被记录但不阻塞本阶段停止目标。
- 文档和输出中没有 dense-core 结果外推 full-model 的表述。

---

## 6. 推荐代码入口

- `framework/stage1_bridge.py`
- `framework/search_three_arm.py`
- `framework/run_b4_ablation.py`
- `framework/run_pqs_ablation.py`
- `framework/run_pqs_codriving.py`
- `multi_agent/methods/design/auto-tuning/1_design_space_building_v1.md`

---

## 7. 建议验收

最小验收:

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m py_compile \
  framework/stage1_bridge.py \
  framework/search_three_arm.py
```

完成后功能验收建议:

```bash
PYTHONPATH=/home/jichengzhi/V2X python - <<'PY'
from pathlib import Path
from framework.stage1_bridge import load_stage2_search_space

for p in [
    Path("framework/partitions/pyramid_lidar_partition.yaml"),
    Path("framework/partitions/codriving_partition.yaml"),
]:
    if p.exists():
        space = load_stage2_search_space(p)
        assert "software_candidates" in space
        assert "hierarchical_blocks" in space
        assert "model_search_policy" in space
        assert "claim_scope" in space
        print(p.name, space["model_search_policy"], space["claim_scope"])
PY
```

正式验收应补:

- search space JSON schema 测试。
- P width anchor 生成测试。
- Pyramid stage1/stage2/stage3 层级 block 测试。
- P/Q software candidate 测试。
- 模型级 joint/serial/noS policy 测试。
- dense-core/full-model scope 测试。

---

## 8. 本方向 `/goal` 启动指令

```text
/goal 推进 Stage2 搜索空间构建工作包，停止目标是该方向空白全部收口。基于 Stage1 manifest 和 framework/stage1_bridge.py，建立正式 Stage2SearchSpace 契约：从 dense candidate path 自动生成 P 轴 width anchors，明确最大剪枝率由 structure/hardware/AP/LUT 约束共同决定，候选取点以硬件 cliff、AP anchor、schedule-LUT measured width 和低/中/高剪枝区间覆盖为准，单个 dense stage/group 默认保留 5-9 个候选，legacy 25%/50%/75% 只能映射为最近合法 width 或 diagnostic anchor。实现层级 block 边界 device -> execution_block -> dense_stage -> search_group -> knob，Pyramid 至少表达 stage1/stage2/stage3，RSU/ego 保留设备边界，当前以 RSU dense-core 为闭环目标，ego fusion/attention 作为 future block 或 blocker。将 Q 轴作为正式成员并与 P 轴合并为 software candidate，再与硬件 schedule/backend 轴做模型级 joint/serial/noS 策略判断；Q 信号不显著时不强行扩大单独 Q 搜索臂。注意当前 Stage2 判断的是 Pyramid/CoDriving 这样的模型级 joint 或 serial 优化，不做 per-knob joint/serial 分类，也不把 per-knob dispatch_plan 作为当前空白。验收包括：search space schema 测试、P width anchor/max_prune_rate 测试、Pyramid stage1/stage2/stage3 层级 block 测试、P/Q software candidate 测试、模型级 joint/serial/noS policy 测试、RSU dense-core 与 full-model claim scope 测试，以及 CoDriving 不因 groups=1 外推为模型级可分离。
```
