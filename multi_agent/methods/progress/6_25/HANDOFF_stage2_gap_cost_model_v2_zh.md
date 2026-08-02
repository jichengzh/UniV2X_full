# HANDOFF — Stage2 当前空白：Cost Model 与证据构建 v2

日期: 2026-06-25

本文继承 `HANDOFF_stage2_gap_cost_model_v1_zh.md`，记录截至目前 **Stage2 cost model / evidence registry** 工作包的解决进展、已落盘代码、验证证据、剩余风险和下一步计划。

v1 的核心问题是: latency / AP / Q / energy / downstream DS 证据缺少统一 registry，demo / proxy / historical / measured 边界容易混用，搜索器仍可能直接读硬编码路径。v2 的当前状态是:

```text
Stage2EvidenceRegistry 第一版契约已落盘:
  latency_lut / ap_anchors / quant_evidence / energy_lut / downstream_objective
  -> schema + backend + scope + provenance + measurement_status + coverage
  -> LatencyLUT / APModel / QLookup / Stage2CostInputs
  -> Stage2Output.cost_evidence summary

但真实数据生成产品化仍未完成:
  latency/AP builder 还未统一为生产 CLI
  energy LUT 仍无实测数据
  CoDriving DS map 已可查但默认 report-only
  Stage2 runner 尚未强制全量通过 registry 消费 cost evidence
```

---

## 0. 当前结论

当前 cost model 工作包已经完成 **第一版 evidence registry 契约和安全边界**，但还没有完成全部真实 evidence 生产闭环。

已经落地:

- `framework/stage2/evidence_registry.py`
  - `Stage2EvidenceRegistry`
  - `EvidenceSource`
  - `Stage2CostInputs`
  - `DSQueryResult`
  - `build_default_registry_dict()`
  - `write_default_registry()`
- `scripts/stage2_prepare_evidence.py`
  - 生成 Pyramid / CoDriving registry fixture。
- `framework/stage2/contracts.py`
  - `Stage2Output` 增加内部 `cost_evidence` 摘要。
  - 仍保持 Stage2 public input 只来自 manifest / classification，不暴露 `evidence_registry_path`。
- `framework/tests/test_stage2_evidence_registry.py`
  - 覆盖 registry schema、Pyramid/CoDriving fixture、backend 隔离、missing evidence fail-closed、DS scope/cliff/report-only、energy no-claim、coverage 统计。
- `framework/tests/test_stage2_integration_contract.py`
  - 覆盖 `Stage2Output.cost_evidence` 输出边界。

当前严格口径:

- Stage1 manifest 仍只定义结构和硬件合法性，不伪装成 latency / AP / energy / DS 实测。
- 新 measured hardware evidence 只允许 `backend=h800_tvm`。
- TRT 只能作为 `historical_trt` 或历史上下文证据，不能进入 `measured_h800_tvm`。
- demo / proxy / historical / estimated evidence 不升级成论文正式结果或 classifier pass 结论。
- Energy evidence 缺失时输出 no-claim，不声明 energy improvement。
- CoDriving AP x latency -> DS map 只能在 CoDriving / Town05 / clean6 scope 内查询。
- predicted AP + predicted latency -> DS 默认 `report_only`，并标注 `downstream_validation_required`。

---

## 1. v1 空白解决进展

### 空白 1: Stage2 证据没有统一 registry

状态: 已完成第一版代码契约。

当前实现:

- 新增 `framework/stage2/evidence_registry.py`。
- registry schema 固定为:

```text
stage2_evidence_registry_v1
```

- registry 当前覆盖:

```text
model
hardware_target
manifest
latency_lut
ap_anchors
quant_evidence
energy_lut
downstream_objective
unsupported_conclusions
coverage_summary
```

- 每个 evidence source 统一记录:

```text
path
measurement_status
backend
scope
provenance
coverage:
  expected_cells
  measured_cells
  failed_cells
  proxy_cells
```

- `Stage2EvidenceRegistry.from_file()` / `from_dict()` 支持读取 registry。
- `Stage2EvidenceRegistry.to_dict()` 会输出可审阅的 registry 内容和 coverage 汇总。

已解决的边界:

- `historical_trt + measured` 会被拒绝。
- hardware evidence 若标为 `measured`，backend 必须是 `h800_tvm`。
- missing latency / AP 会通过 `require_search_ready()` fail closed。
- energy 缺失会进入 `unsupported_conclusions`，不允许 energy claim。

仍需注意:

- 当前 registry 主要是代码契约和 Pyramid/CoDriving fixture；还不是所有模型的自动 registry 生成器。
- 当前 `Stage2Input` 仍不暴露 `evidence_registry_path`，符合 Stage1/Stage2 集成 v2 的 thin public input 约束。

### 空白 2: latency LUT 生成流程未产品化

状态: 未完全解决；已纳入 registry，可被安全消费。

当前实现:

- `Stage2EvidenceRegistry.load_latency_lut()` 通过 registry path 构造 `framework/search_three_arm.py::LatencyLUT`。
- `Stage2CostInputs.latency_lut` 也来自 registry。
- 默认 Pyramid registry 指向:

```text
results/latency_lut_pyramid.json
```

并标注:

```text
measurement_status = measured
backend = h800_tvm
scope = dense_core
```

- 默认 CoDriving registry 指向:

```text
results/latency_lut_codriving.json
```

并标注为 H800 TVM LUT，但 coverage 中区分 measured / proxy cells:

```text
expected_cells = 4
measured_cells = 2
proxy_cells = 2
```

仍未完成:

- 还没有统一的 `scripts/stage2_build_latency_lut.py`。
- 还没有把 warmup / repeat / p50 / p90 / std / raw trace 全部写成正式 latency LUT schema。
- `search_three_arm.py` 和 B4/PQS runners 还没有被强制要求只从 registry 获取 latency evidence。

### 空白 3: AP anchors 生成流程未产品化

状态: 未完全解决；已纳入 registry，可被安全消费。

当前实现:

- `Stage2EvidenceRegistry.load_ap_model()` 通过 registry path 构造 `framework/search_three_arm.py::APModel`。
- `Stage2CostInputs.ap_model` 也来自 registry。
- Pyramid 默认 AP source:

```text
results/ap70_model_pyramid.json
```

- CoDriving 默认 AP source:

```text
results/ap70_model_codriving.json
```

- registry 中记录 AP evidence 的:

```text
measurement_status
backend = model_eval
scope = model_accuracy
metric = AP70
provenance
coverage
```

仍未完成:

- 还没有统一的 `scripts/stage2_import_ap_anchors.py` 或 `scripts/stage2_build_ap_anchors.py`。
- ckpt / dataset / eval split / finetune protocol 仍主要存在于源 JSON 文本和实验上下文，没有被规范化为强 schema。
- AP predicted / imported / measured 的细粒度状态还需要在后续真实数据导入时强化。

### 空白 4: Q evidence 的 backend / coverage 边界仍需清理

状态: 第一版边界已落盘。

当前实现:

- `Stage2EvidenceRegistry.load_q_lookup()` 通过 registry path 构造 `framework/search_three_arm.py::QLookup`。
- `Stage2CostInputs.q_lookup` 也来自 registry。
- Pyramid 默认 Q evidence 指向:

```text
results/latency_lut_pyramid_q.json
```

但标注为:

```text
measurement_status = historical
backend = historical_trt
scope = backbone_only
```

- `historical_trt` 不能标成 `measured`。
- `historical_trt` 不具备 `promotable_to_measured`。
- `backend=trt, measurement_status=measured` 或 `backend=historical_trt, measurement_status=measured` 会被拒绝。

仍未完成:

- H800 TVM Q evidence 还没有形成完整 measured LUT。
- V2X-ViT true TRT INT8 AP 仍应保持 blocked / not done，不应由 registry 自动补全。
- 若未来允许 historical ratio fallback，必须继续保持显式 opt-in 和 historical 标注。

### 空白 5: DS / closed-loop objective 尚未成为可靠统一目标

状态: 第一版 downstream LUT 查询与风险标注已落盘。

当前实现:

- registry 可记录:

```text
downstream_objective:
  path: multi_agent/real_test/ds_ap_latency_all_measured.csv
  measurement_status: measured
  backend: closed_loop_sim
  scope:
    model: codriving
    town: Town05
    routes: clean6
    traffic: full_traffic_1
  cliff_band_ms: [600, 650]
```

- `Stage2EvidenceRegistry.query_downstream_ds()` 支持 AP50 / latency_ms 查询。
- 查询自动执行 measured rectangle 检查，超域 fail closed。
- scope 不是 CoDriving / Town05 / clean6 时 fail closed。
- 若输入 AP 或 latency 不是 measured，则:

```text
mode = report_only
flags:
  predicted_ds_report_only
  downstream_validation_required
```

- 若 latency uncertainty interval 跨过 600-650ms cliff band，则标注:

```text
uncertain_due_to_cliff
```

- 查询输出:

```text
DS_low
DS_mid
DS_high
mode
flags
prediction_chain
```

仍需注意:

- DS LUT 仍不是主 Pareto 真值。
- Pyramid 默认 registry 不接 CoDriving DS map，并记录 cross-model blocker。
- 当前 DS uncertainty 是基于输入区间采样的第一版机制，还不是完整统计置信区间。
- 推荐配置仍需要 top-K 闭环复测，不能只依赖 DS lookup。

### 空白 6: 新硬件 evidence 需要实机测量闭环

状态: 边界已部分落盘；新硬件实测流程未完成。

当前实现:

- registry 校验把 hardware measured evidence 限定在 `h800_tvm`。
- Stage1 manifest 的硬件 capability 仍不被当作 measured latency / AP / energy / DS。
- `Stage2Output.cost_evidence` 会显示 evidence backend / status，而不是从 manifest 推断性能结论。

仍未完成:

- 新硬件 registry 生成流程未产品化。
- RTX 3090 / Orin / 其他设备的 measured latency / energy 仍需要各自实测 LUT。
- 新硬件无实测 evidence 时，只能 estimated / proxy / not_available，不能输出正式 speedup 或 energy claim。

### 空白 7: energy LUT 生成流程未产品化

状态: no-claim 边界已落盘；实测 energy LUT 未完成。

当前实现:

- registry 中有 `energy_lut` source。
- 默认 Pyramid / CoDriving registry 都显式输出:

```text
energy_lut:
  path: null
  measurement_status: not_available
  backend: not_available
```

- `Stage2EvidenceRegistry.energy_claim_allowed == false`。
- `unsupported_conclusions` 自动包含:

```text
energy_improvement_without_energy_lut
```

- `Stage2Output.cost_evidence.energy_lut.claim_allowed == false`。

仍未完成:

- 还没有 `scripts/stage2_build_energy_lut.py`。
- 没有统一 telemetry source / idle baseline / sampling window / joule_per_inference schema。
- energy LUT 还没有和 latency LUT 的 `config_id` 形成真实数据级对齐。

### 空白 8: LUT 数据集规模估算和生成计划缺失

状态: 第一版 coverage 统计已落盘；真实采样计划仍需扩展。

当前实现:

- 每个 evidence source 均有 coverage 字段:

```text
expected_cells
measured_cells
failed_cells
proxy_cells
```

- `Stage2EvidenceRegistry.coverage_summary()` 输出单项和总计。
- `scripts/stage2_prepare_evidence.py` 生成 fixture 时会统计现有 JSON/CSV 行数。
- 当前 smoke 输出示例:

```text
pyramid_lidar: expected_cells=57 measured_cells=28
codriving: expected_cells=77 measured_cells=71
```

这些数字来自当前 fixture 的 registry coverage:

- Pyramid: latency 19 measured + AP 9 measured + Q 10 proxy + energy 0 + DS blocked。
- CoDriving: latency 4 total，其中 2 measured / 2 proxy；AP 4 measured；DS 65 measured；energy 0。

仍未完成:

- smoke / calibration / paper 三档采样计划还没有变成独立可执行 plan 文件。
- registry 还没有记录每个 cell 的 config id、失败原因、raw repeat 统计。
- coverage 当前是 source-level 汇总，不是完整 per-cell dataset manifest。

---

## 2. 当前输出快照

### 2.1 Registry fixture CLI

新增入口:

```bash
PYTHONPATH=${V2X_ROOT} python scripts/stage2_prepare_evidence.py \
  --model pyramid_lidar \
  --out-json results/stage2/pyramid/evidence_registry.json
```

```bash
PYTHONPATH=${V2X_ROOT} python scripts/stage2_prepare_evidence.py \
  --model codriving \
  --out-json results/stage2/codriving/evidence_registry.json
```

当前 CLI 是 fixture generator，不是全模型 evidence crawler。它会写出 `stage2_evidence_registry_v1` JSON，并打印 total expected / measured cells。

### 2.2 Stage2Output cost evidence summary

`scripts/stage2_optimize_model.py` 当前输出中新增:

```text
cost_evidence:
  schema: stage2_evidence_registry_summary_v1
  registry_status
  latency_lut
  ap_anchors
  quant_evidence
  energy_lut
  downstream_objective
  unsupported_conclusions
  coverage_summary
```

Pyramid 当前应显示:

```text
latency_lut.measurement_status = measured
latency_lut.backend = h800_tvm
quant_evidence.backend = historical_trt
energy_lut.claim_allowed = false
unsupported_conclusions includes energy_improvement_without_energy_lut
```

---

## 3. 当前验证记录

最近一次验证命令:

```bash
PYTHONPATH=${V2X_ROOT} python -m unittest \
  framework.tests.test_stage2_evidence_registry \
  framework.tests.test_stage2_integration_contract
```

结果:

```text
Ran 15 tests
OK
```

Search-space 回归:

```bash
PYTHONPATH=${V2X_ROOT} \
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  -m pytest framework/tests/test_stage2_search_space_contract.py -q
```

结果:

```text
2 passed
```

语法检查:

```bash
PYTHONPATH=${V2X_ROOT} python -m py_compile \
  framework/stage2/evidence_registry.py \
  framework/stage2/contracts.py \
  framework/stage2/__init__.py \
  scripts/stage2_prepare_evidence.py \
  scripts/stage2_optimize_model.py \
  scripts/stage2_update_evidence.py
```

结果: exit code 0。

CLI smoke:

```text
stage2_registry_cli_smoke_ok
```

---

## 4. 当前文件索引

核心实现:

- `framework/stage2/evidence_registry.py`
- `framework/stage2/contracts.py`
- `framework/stage2/__init__.py`
- `scripts/stage2_prepare_evidence.py`

测试:

- `framework/tests/test_stage2_evidence_registry.py`
- `framework/tests/test_stage2_integration_contract.py`
- `framework/tests/test_stage2_search_space_contract.py`

相关输入:

- `results/latency_lut_pyramid.json`
- `results/ap70_model_pyramid.json`
- `results/latency_lut_pyramid_q.json`
- `results/latency_lut_codriving.json`
- `results/ap70_model_codriving.json`
- `multi_agent/real_test/ds_ap_latency_all_measured.csv`
- `multi_agent/methods/progress/HANDOFF_ap_latency_ds_map_v2.md`
- `multi_agent/real_test/README_ds_map.md`

---

## 5. 当前剩余风险

### 风险 1: registry 已有，但正式 Stage2 search runner 未全部强制消费

`Stage2EvidenceRegistry.load_cost_inputs()` 已经能产出:

```text
LatencyLUT
APModel
QLookup
energy_lut EvidenceSource
downstream_objective EvidenceSource
```

但 `search_three_arm.py`、`run_b4_ablation.py`、`run_pqs_ablation.py`、`run_pqs_codriving.py` 仍可能通过旧默认路径直接构造 LUT。下一步需要让正式 runner 接收 registry / cost inputs，避免旁路。

### 风险 2: default registry 只覆盖 Pyramid / CoDriving fixture

当前 `build_default_registry_dict()` 只支持:

```text
pyramid_lidar
codriving
```

其他模型需要显式 blocked、not_available 或新增 fixture。不能因为没有 registry 就从 manifest 推断 performance evidence。

### 风险 3: CoDriving latency LUT 仍含 estimated cells

CoDriving `latency_lut_codriving.json` 内 base / p50 是真实 H800 TVM，p25 / p75 是 estimated。registry coverage 已记录 measured/proxy 区分，但论文正式结论仍需要避免把所有 cells 写成 measured。

### 风险 4: DS map 能查询，但不能替代闭环验证

`query_downstream_ds()` 已做 scope / rectangle / cliff / report-only 标注，但它仍只是 lookup。真实推荐配置仍需要 top-K closed-loop rerun。

### 风险 5: energy 只有 no-claim 边界，没有数据

当前只解决了“不能乱声称 energy improvement”。如果后续论文或报告需要 energy 维度，必须先补 energy measurement pipeline 和 LUT。

### 风险 6: registry schema 还不是文档化 JSON Schema

当前 schema 是 Python dataclass + tests。后续若要给开源仓库或多 agent 使用，应补正式 JSON schema / docs。

---

## 6. 下一步计划

### 任务 A: 让正式 Stage2 runner 通过 registry 消费 cost evidence

目标:

- `search_three_arm.py` / B4 / PQS runner 不再硬编码 latency / AP / Q path 作为正式入口。
- 增加 registry-aware builder，例如:

```text
Stage2EvidenceRegistry.load_cost_inputs()
```

已经存在，可作为正式 runner 的输入。

建议动作:

1. 给 `search_three_arm.py` 增加 registry path 或 cost input object 的函数接口。
2. `run_b4_ablation.py` / `run_pqs_ablation.py` / `run_pqs_codriving.py` 逐步改成从 registry 读取 evidence。
3. 输出日志保存 `cost_evidence` summary 和 coverage。
4. 保留旧默认路径为 legacy smoke，不作为正式实验入口。

验收:

- Pyramid / CoDriving runner 输出中包含 registry schema、source path、backend、scope、coverage。
- missing latency / AP registry 时 fail closed。
- Q historical TRT 不会进入 measured H800 path。

### 任务 B: 产品化 latency LUT builder/importer

目标:

- 建立 `scripts/stage2_build_latency_lut.py` 或 `scripts/stage2_import_latency_lut.py`。
- 输出统一 latency LUT schema。

必须记录:

- model / hardware / backend / scope / config_id。
- default schedule / tuned schedule。
- warmup / repeat / p50 / p90 / mean / std。
- measurement environment。
- failed cells 和失败原因。
- measured / estimated / proxy 状态。

验收:

- Pyramid / CoDriving smoke grid 可生成 registry-compatible latency LUT。
- CoDriving estimated cells 不会被误标 measured。
- 新硬件无 measured LUT 时只能 not_available / estimated / proxy。

### 任务 C: 产品化 AP anchors builder/importer

目标:

- 建立 `scripts/stage2_import_ap_anchors.py` 或 `scripts/stage2_build_ap_anchors.py`。
- AP evidence 与 config_id 对齐。

必须记录:

- ckpt。
- dataset。
- eval split。
- metric。
- finetune / no-finetune protocol。
- imported / measured / predicted status。

验收:

- finetune 与 no-finetune AP 不混用。
- AP predicted 只能 ranking / screening，不能替代 final truth。
- registry coverage 能反映 measured anchors 数量。

### 任务 D: 补 H800 Q evidence 与 Q schema

目标:

- 区分:

```text
measured_h800_tvm
historical_trt
proxy
not_done
```

- 将 Q evidence 的 backend/scope/provenance/schema 固化。

验收:

- `historical_trt` 与 `measured_h800_tvm` 隔离测试继续通过。
- H800 measured Q evidence 可被 registry 消费。
- V2X-ViT true TRT INT8 AP 未做时继续 blocked / not_done。

### 任务 E: 产品化 energy LUT

目标:

- 新增 `scripts/stage2_build_energy_lut.py`，或先作为 latency builder 的同步字段。

必须记录:

- `config_id == latency_lut.config_id`。
- telemetry source。
- sampling window。
- watt_avg / watt_p50 / watt_p90。
- joule_per_inference。
- idle baseline policy。
- repeat / warmup。
- power cap / clock policy。

验收:

- energy measured 时 `energy_claim_allowed == true`。
- energy 缺失时继续 no-claim。
- 不同 telemetry source 不混写。

### 任务 F: DS downstream top-K validation

目标:

- 使用现有 CoDriving 65-cell DS map 做 report-only / rerank。
- 对 Stage2 top-K 候选做真实 closed-loop 复测。

建议动作:

1. 给 `query_downstream_ds()` 增加批量查询 helper。
2. Stage2 output 中保存 DS prediction chain 和 flags。
3. 选择 5-10 个 top-K config 做 closed-loop rerun。
4. 将复测结果作为新 downstream evidence，而不是覆盖原 DS map。

验收:

- predicted AP + predicted latency 查询 DS 默认 report-only。
- 超出 measured rectangle fail closed。
- 跨 600-650ms cliff band 标注 uncertain。
- 真实 closed-loop 复测结果单独 provenance，不与 lookup 混写。

### 任务 G: 文档与开源仓库同步

目标:

- 补正式 evidence registry 格式文档。
- 同步 export repo。
- 增加 fresh-clone smoke。

建议新增:

```text
docs/stage2-evidence-registry.zh-CN.md
docs/stage2-cost-model-evidence.zh-CN.md
```

验收:

- README 不把 demo/proxy 写成 measured。
- export repo 能运行 `stage2_prepare_evidence.py` fixture。
- 文档说明 Stage2 public input 仍不暴露 `evidence_registry_path`。

---

## 7. 建议下一轮 `/goal`

```text
/goal 推进 Stage2 cost model evidence registry v2 后续产品化。基于已落盘的 framework/stage2/evidence_registry.py、Stage2CostInputs、scripts/stage2_prepare_evidence.py 和 Stage2Output.cost_evidence，把正式 Stage2 search runner 改成 registry-aware cost evidence 消费路径，并启动 latency/AP/energy/Q evidence 的产品化生成与导入。严格约束：Stage2 public input 仍只暴露 manifest_path 和 model_classification_path；Stage1 manifest 不伪装成 latency/AP/energy/DS 实测；H800 TVM 是新增 measured 主后端；TRT 只能 historical；CoDriving DS map 只能 scope 内 report-only/rerank，predicted AP + predicted latency 不写成真实闭环；energy 缺失 no-claim；demo/proxy/historical 不升级成论文结果。验收包括：B4/PQS/search_three_arm registry-aware runner 测试、latency/AP importer schema 测试、energy no-claim/energy measured claim 测试、H800 measured Q 与 historical TRT 隔离测试、DS top-K validation plan、coverage per-cell manifest，以及 export repo smoke。
```

---

## 8. 清空上下文后优先读取

1. `multi_agent/methods/progress/HANDOFF_stage2_gap_cost_model_v2_zh.md`
2. `multi_agent/methods/progress/HANDOFF_stage2_gap_cost_model_v1_zh.md`
3. `multi_agent/methods/progress/HANDOFF_stage2_gap_search_space_v2_zh.md`
4. `multi_agent/methods/progress/HANDOFF_stage2_gap_stage1_stage2_integration_v2_zh.md`
5. `framework/stage2/evidence_registry.py`
6. `framework/stage2/contracts.py`
7. `scripts/stage2_prepare_evidence.py`
8. `framework/tests/test_stage2_evidence_registry.py`
9. `framework/tests/test_stage2_integration_contract.py`
10. `framework/search_three_arm.py`
11. `framework/run_b4_ablation.py`
12. `framework/run_pqs_ablation.py`
13. `framework/run_pqs_codriving.py`
14. `multi_agent/methods/progress/HANDOFF_ap_latency_ds_map_v2.md`
15. `multi_agent/real_test/README_ds_map.md`

---

## 9. 关键提醒

- 不要把 registry fixture 写成真实全模型 evidence 生产系统。
- 不要把 CoDriving estimated latency cells 写成 measured。
- 不要把 historical TRT Q evidence 写成 H800 TVM measured。
- 不要把 predicted AP + predicted latency -> DS lookup 写成真实 closed-loop result。
- 不要在 energy LUT 缺失时声明 energy improvement。
- 不要把 Stage2 public input 扩展成 `evidence_registry_path`；registry 仍是内部 cost evidence contract。
- 不要把 dense-core evidence 外推成 full-model speedup。
