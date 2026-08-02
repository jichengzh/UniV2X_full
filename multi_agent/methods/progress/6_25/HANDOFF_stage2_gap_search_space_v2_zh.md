# HANDOFF — Stage2 当前空白：搜索空间构建 v2

日期: 2026-06-25

本文继承 `HANDOFF_stage2_gap_search_space_v1_zh.md`，只总结 **Stage2 search space** 方向的空白解决进展与下一步计划。

重要纠错:

- `HANDOFF_stage2_gap_cost_model_v2_zh.md` 是一次误建文档，当前已删除，不作为有效交接内容。
- cost model 空白尚未开始修改；本文不声称 cost model 已经解决，也不把 latency/AP 预测能力写入 search space 的完成范围。
- 当前完成的是: 从 Stage1 manifest 生成 Stage2 可消费的搜索空间契约，包括 P/Q 软件候选、硬件候选、层级 block、模型级 joint/serial 策略和 claim scope。

---

## 0. 当前结论

Search space 工作包的核心空白已经在当前仓库内收口。

已经落地:

- `framework/stage1_bridge.py` 中新增 `SpaceSpec.stage2_search_space()`。
- `framework/stage1_bridge.py` 中新增 `load_stage2_search_space(path)`。
- `framework/tests/test_stage2_search_space_contract.py` 中新增 Pyramid / CoDriving search space 契约测试。
- `HANDOFF_stage2_gap_search_space_v1_zh.md` 已补充实现状态与验收命令。
- `HANDOFF_stage2_codesign_next_phase_plan_v1_zh.md` 已把原先容易误导的 per-knob dispatch 叙述调整为模型级 `model_search_policy`。

当前 search space contract 输出:

```text
schema: stage2_search_space_v1
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
claim_scope
unsupported_conclusions
```

阶段判断:

- 搜索空间定义本身: 已完成当前停止目标。
- 搜索空间被 Stage2 搜索器完整消费: 仍需下一阶段 hardening/integration。
- cost model / latency / AP 证据: 尚未开始当前空白修补，不能写成已完成。

---

## 1. v1 空白解决进展

### 空白 1: P 轴剪枝候选不应固定为 25/50/75

状态: 已解决到当前 manifest 能支撑的程度。

当前实现:

- 不再把 `25%/50%/75%` 当作唯一剪枝候选。
- 从 `view_b1_search_groups` / `view_b1_prune_groups` / `hw_capability` 推导合法宽度。
- 对每个 dense search group 生成 `width_anchors`，并展开为 `software_points`。
- 候选点受以下边界共同约束:
  - `structure`: Stage1 prune group/connectivity。
  - `hardware`: `round_to`、`int8_buildable_align` 等硬件合法性。
  - `ap`: 生产级深剪枝仍需要 AP anchor 或短微调证据。
  - `lut`: 生产级部署仍需要 latency LUT 或 schedule buildability 证据。
- legacy 25/50/75 只作为角色标签或最近合法 width 映射；不合法 INT8 快核的点会标成 `diagnostic_trap`，不会被包装成生产候选。

当前策略:

- 每个 search group 默认保留少量代表性 anchor，目标是覆盖 baseline、硬件可构建点、低/中/高剪枝区间和 legacy 映射点。
- 候选数量不是连续变量穷举，而是从近连续剪枝率中抽取结构和硬件有意义的离散点。
- 当前 fixture 中单个 group 的 anchor 数为 2-9 个；Pyramid 的 grouped conv stage 可达到 9 个，简单 stage 可能只有 2 个合法点。

仍需注意:

- 当前 anchor 生成依赖 Stage1 manifest 的 search group 粒度。
- 如果后续 Stage1 scanner 能补充更细的 per-layer `cin/cout/groups/kernel/stride`，search space 可进一步从 group 级扩展到 layer/stage 混合粒度。
- 目前不应把 search space 的 `max_prune_rate` 当作精度安全承诺；它只是结构/硬件/证据边界下的候选上限。

### 空白 2: block / knob 边界不能只按粗 block 类型划分

状态: 当前 Pyramid / CoDriving dense core 已解决。

当前实现:

- 输出 `hierarchical_blocks`，层级为:

```text
device_scope -> execution_block -> dense_stage -> search_group -> knob
```

- Pyramid dense backbone 明确拆为:
  - `stage1`
  - `stage2`
  - `stage3`
  - `neck`
- CoDriving dense backbone 也按 stage 层级输出，不再只写成一个粗粒度 dense block。
- RSU dense perception 与 ego fusion/attention future block 保留在同一 claim scope 中，但当前闭环优化对象只放在 RSU dense core。

仍需注意:

- 当前层级命名主要由 search group suffix 和 manifest 结构推导。
- 新模型接入时需要做 smoke test，确认 Stage1 scanner 输出的 group 名称足以稳定映射 stage。
- attention/fusion 不是本轮 search space 停止目标，但已经作为 future block 被记录，不能被误写为当前已加速范围。

### 空白 3: Q 轴信号弱，但应成为正式软件候选成员

状态: 已解决为 P/Q 软件轴合并表达。

当前实现:

- `software_candidates` 中每个 P width candidate 都挂载 `quant_policies`。
- 当前可用 Q policy 以 `fp16` / `int8` 为主，并带:
  - `policy`
  - `backend_scope`
  - `provenance`
  - `status`
- `software_points` 展开为 `width x quant_policy` 的组合点。
- Q 轴不再被强行做成独立显著搜索臂，而是作为软件候选属性与 P 轴合并，再与硬件 schedule/backend 轴进入模型级策略判断。

仍需注意:

- Q 轴进入 search space 不等于 Q 轴已经有强 latency/AP 证据。
- Q 轴是否真正影响策略，属于后续 cost model / evidence registry 的工作。
- 当前 search space 只保证 Q policy 有位置、有 provenance、有 status，不承诺 quantization speedup 数字。

### 弱化项 4: per-knob joint/serial 不是当前空白

状态: 已澄清并在代码中降级为 legacy diagnostic。

用户已经明确: 当前判断的是 **模型级** 分类，例如 Pyramid / CoDriving 走 joint 还是 serial，不是判断每个 knob 应该怎么优化。

当前实现:

- `SpaceSpec.dispatch_plan()` 保留给历史脚本兼容和诊断解释。
- 正式 Stage2 search space 使用 `model_search_policy`。
- `model_search_policy.decision_level == "model"`。
- `unsupported_conclusions` 中显式包含 `per_knob_joint_serial`，防止后续误读。

当前模型级策略:

- Pyramid: `joint`
  - 依据包括 `P_IC_BN_hub`、`int8_buildability_cliff`、`schedule_headroom_risk`。
- CoDriving: `serial`
  - 依据包括 `standard_conv_dense_envelope`、`no_int8_buildability_cliff_in_dense_core`。

仍需注意:

- 不要在后续文档里恢复“按 knob 判断 joint/serial”的口径。
- 如果后续确实要做 per-region 或 per-subgraph 策略，那是新的研究扩展，不是当前 Stage2 空白修补。

### 弱化项 5: dense core 与 full-model 边界

状态: 当前作为 claim scope 约束处理，已满足本方向停止目标。

当前实现:

- `optimized_scope == "rsu_dense_core"`。
- `claim_scope.full_model_claim_allowed == false`。
- `future_blocks` 中记录 `ego_fusion_attention_future`。
- `unsupported_conclusions` 中记录:
  - `dense_core_speedup_as_full_model_speedup`
  - `ego_fusion_attention_acceleration_in_current_scope`

含义:

- 当前 search space 可以支撑 RSU dense core 的剪枝/量化/硬件调度候选生成。
- 当前 search space 不能直接支撑 full-model speedup claim。
- ego 端 fusion/attention 仍是未来工作，不阻塞本轮 dense-core search space 停止目标。

---

## 2. 当前输出快照

以下快照来自当前代码:

```bash
PYTHONPATH=${V2X_ROOT} ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python
```

入口:

- `framework/partitions/pyramid_lidar_partition.yaml`
- `framework/partitions/codriving_partition.yaml`

### Pyramid LiDAR

```text
model: pyramid_lidar
schema: stage2_search_space_v1
policy: joint
policy reasons: P_IC_BN_hub, int8_buildability_cliff, schedule_headroom_risk
optimized_scope: rsu_dense_core
full_model_claim_allowed: false
dense stages: neck, stage1, stage2, stage3
software_candidates: 5
hardware_candidates: 2
hierarchical_blocks: 8
unsupported:
  dense_core_speedup_as_full_model_speedup
  ego_fusion_attention_acceleration_in_current_scope
  per_knob_joint_serial
```

候选概览:

| candidate | dense_stage | width anchors | software points | Q policies |
|---|---:|---:|---:|---|
| `neck` | `neck` | 5 | 10 | `fp16`, `int8` |
| `bev_encoder.s2` | `stage3` | 9 | 18 | `fp16`, `int8` |
| `bev_encoder.s1` | `stage2` | 7 | 14 | `fp16`, `int8` |
| `backbone.s0` | `stage1` | 2 | 4 | `fp16`, `int8` |
| `bev_encoder.s0` | `stage1` | 4 | 8 | `fp16`, `int8` |

### CoDriving

```text
model: codriving
schema: stage2_search_space_v1
policy: serial
policy reasons: standard_conv_dense_envelope, no_int8_buildability_cliff_in_dense_core
optimized_scope: rsu_dense_core
full_model_claim_allowed: false
dense stages: neck, stage1, stage2, stage3
software_candidates: 4
hardware_candidates: 2
hierarchical_blocks: 8
unsupported:
  cross_model_extrapolation_from_standard_conv
  dense_core_speedup_as_full_model_speedup
  ego_fusion_attention_acceleration_in_current_scope
  groups1_not_model_separable_proof
  per_knob_joint_serial
```

候选概览:

| candidate | dense_stage | width anchors | software points | Q policies |
|---|---:|---:|---:|---|
| `neck` | `neck` | 4 | 8 | `fp16`, `int8` |
| `backbone.s2` | `stage3` | 6 | 12 | `fp16`, `int8` |
| `backbone.s1` | `stage2` | 4 | 8 | `fp16`, `int8` |
| `backbone.s0` | `stage1` | 2 | 4 | `fp16`, `int8` |

---

## 3. 已完成验收

已通过的功能测试:

```bash
PYTHONPATH=${V2X_ROOT} ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python -m pytest framework/tests/test_stage2_search_space_contract.py -q
```

结果:

```text
2 passed
```

已通过的回归测试:

```bash
PYTHONPATH=${V2X_ROOT} ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python -m pytest framework/tests/test_coupling_predictor_static.py framework/tests/test_stage1_manifest_predictor_fields.py -q
```

结果:

```text
11 passed
```

已通过的语法检查:

```bash
PYTHONPATH=${V2X_ROOT} ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python -m py_compile framework/stage1_bridge.py framework/search_three_arm.py
```

---

## 4. 当前剩余风险

### 风险 1: search space 已闭环，但还没有被所有 Stage2 runner 强制消费

`load_stage2_search_space()` 已经存在，测试也覆盖了契约；但后续还要确认 `search_three_arm.py`、B4/PQS runner 和报告脚本是否统一消费该契约，而不是继续从旧 `dispatch_plan()` 或手写候选列表走旁路。

处理建议:

- 下一阶段先做 integration audit。
- 所有正式 Stage2 实验入口都应读取 `stage2_search_space` 或其 summary。
- 旧 `dispatch_plan()` 只保留为 legacy diagnostic，不进入正式 stopping criterion。

### 风险 2: 当前只对 Pyramid / CoDriving 做了强验收

当前测试覆盖了用户本轮最关心的 Pyramid / CoDriving 分类差异，但还没有把所有 V2X 方法都纳入 search space smoke。

处理建议:

- 对已有 manifest 做批量 `load_stage2_search_space()` smoke。
- 至少确认 Pyramid Camera、autoscan Pyramid、autoscan CoDriving、V2X-ViT/AttFuse 相关 manifest 是 active / future / unsupported 中的哪一种。
- 对没有 Stage1 manifest 的模型，不要伪造 search space；应标为 scanner coverage gap。

### 风险 3: cost model 未开始，不能把候选空间写成性能结论

Search space 只解决“有哪些合法候选、如何分层、如何声明 scope”。它不能回答:

- 哪个候选最优。
- latency/AP 下降多少。
- Q 轴是否显著。
- joint 是否端到端优于 serial。

处理建议:

- 后续 cost model 工作包应单独开启。
- cost model 开始前，本文只能作为候选生成与策略口径的输入。

### 风险 4: Stage1 scanner 的结构粒度会影响候选质量

当前能按 stage/search group 输出候选，但更细粒度候选依赖 Stage1 manifest 是否提供更完整的层级结构。

处理建议:

- 若后续需要真实 per-layer pruning candidates，需要 Stage1 scanner 增加更细的 layer feature。
- 当前不要把 group-level candidate 解释成完整 layer-level NAS/pruning space。

---

## 5. 下一步计划

下一阶段不应直接跳到 cost model，除非用户明确启动 cost model 空白。Search space 方向下一步是 hardening 和集成消费。

### 任务 A: Stage2 runner 消费契约

目标:

- `search_three_arm.py` / B4 / PQS runner 不再手写 P/Q/S 候选。
- 正式实验从 `load_stage2_search_space()` 读取 `software_candidates`、`hardware_candidates`、`model_search_policy`。
- 报告里明确区分:
  - dense-core search result。
  - future fusion/attention block。
  - full-model unsupported claim。

验收:

- 运行 Pyramid / CoDriving runner 时能打印或保存 search space summary。
- 实验日志能追踪候选 `id`、width、quant policy、stage、scope。
- 没有把 `dispatch_plan()` 当作正式策略入口。

### 任务 B: 扩展 manifest smoke 覆盖

目标:

- 对当前仓库已有 partition manifest 做批量 smoke。
- 输出哪些模型可生成 dense-core search space，哪些模型只可作为 future/unsupported。

建议覆盖:

- `framework/partitions/pyramid_lidar_partition.yaml`
- `framework/partitions/pyramid_camera_partition.yaml`
- `framework/partitions/codriving_partition.yaml`
- `results/autoscan_pyramid_lidar_partition.yaml`
- `results/autoscan_codriving_partition.yaml`
- V2X-ViT / AttFuse 若当前只有 attention/fusion manifest，应明确标为 future 或 partial scope。

验收:

- 新增一个 smoke 测试或 CLI audit。
- 每个 manifest 至少输出 schema、candidate count、policy、claim scope。
- 对 scanner 缺口使用 unsupported/blocker，不做静默跳过。

### 任务 C: 增加 search space dump/audit 工具

目标:

- 提供一个命令行工具，把 manifest 转成可审阅的 JSON/Markdown summary。
- 后续 cost model 和 reviewer agent 可以直接读取该 summary。

建议接口:

```bash
PYTHONPATH=${V2X_ROOT} python -m framework.stage1_bridge \
  --stage2-search-space framework/partitions/pyramid_lidar_partition.yaml \
  --out results/stage2_search_space/pyramid_lidar.json
```

或新增:

```bash
python framework/tools/dump_stage2_search_space.py \
  --manifest framework/partitions/pyramid_lidar_partition.yaml \
  --format md
```

验收:

- dump 文件包含 candidate table、policy、claim scope、unsupported conclusions。
- dump 输出不包含 latency/AP 伪结论。

### 任务 D: 与 Stage1 scanner 集成检查

目标:

- 确认 Stage1 扫描输出中 dense stage、search group、quant unit、routing segment 字段足以稳定生成 Stage2SearchSpace。
- 如果 Stage1 缺字段，补充 scanner 输出，而不是在 Stage2 写模型名特判。

验收:

- Pyramid stage1/stage2/stage3 由 manifest 结构稳定推出。
- CoDriving 不因为 `groups=1` 被外推成模型级可分离证明。
- 新模型缺字段时能给出明确 unsupported reason。

---

## 6. 建议下一阶段 `/goal`

如果下一步仍继续 search space hardening，可使用:

```text
/goal 推进 Stage2 search space hardening 与 runner 集成。停止目标是：所有正式 Stage2 实验入口统一消费 load_stage2_search_space() 输出，不再使用手写 P/Q/S 候选或 legacy dispatch_plan 作为正式策略；为当前仓库已有 Pyramid、CoDriving、autoscan manifest 增加批量 smoke/audit，输出每个模型的 software_candidates、hardware_candidates、model_search_policy、claim_scope 和 unsupported_conclusions；新增 search space dump/audit 工具，能生成 JSON 或 Markdown summary，供后续 cost model 和 reviewer agent 使用。注意 cost model 空白尚未开始，本阶段不生成 latency/AP 预测结论，只验证搜索空间契约、候选来源、层级边界和 claim scope 是否可追踪。
```

如果用户准备启动 cost model，需要另开 cost model 工作包，不要复用本文作为 cost model 已完成证据。

---

## 7. 新窗口快速恢复阅读顺序

新窗口建议按以下顺序读取:

1. `multi_agent/methods/progress/HANDOFF_stage2_gap_search_space_v2_zh.md`
2. `multi_agent/methods/progress/HANDOFF_stage2_gap_search_space_v1_zh.md`
3. `multi_agent/methods/progress/HANDOFF_stage2_codesign_next_phase_plan_v1_zh.md`
4. `framework/stage1_bridge.py`
5. `framework/tests/test_stage2_search_space_contract.py`
6. `framework/partitions/pyramid_lidar_partition.yaml`
7. `framework/partitions/codriving_partition.yaml`

不要读取或继承 `HANDOFF_stage2_gap_cost_model_v2_zh.md`；该文件不应存在，也不代表有效进度。
