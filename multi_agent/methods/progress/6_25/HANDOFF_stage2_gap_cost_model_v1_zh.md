# HANDOFF — Stage2 当前空白：Cost Model 与证据构建 v1

日期: 2026-06-24

本文是 Stage2 软硬件协同优化框架的四个并行空白工作包之一，聚焦 **latency / AP / quant / energy / downstream objective 等证据如何进入 cost model，并被 Stage2 搜索器可靠消费**。主 handoff 索引见 `HANDOFF_stage2_codesign_next_phase_plan_v1_zh.md`。

---

## 1. 本工作包边界

本工作包处理:

- latency LUT。
- AP/accuracy anchors。
- optional quant evidence。
- energy LUT。
- AP x latency -> DS/closed-loop objective LUT 的可选接入和不确定性边界。
- cost model 的 schema、provenance、backend/scope 标注。
- LUT 数据集规模估算、采样计划和生成优先级。
- 新硬件的 measured evidence 接入。

本工作包不负责:

- Stage1 自动 trace。
- Stage2 搜索空间生成。
- 论文实验 seed 统计收口。

---

## 2. 当前已经有的基础

当前 Stage2 代码中已有三个核心证据消费类:

- `framework/search_three_arm.py::LatencyLUT`
- `framework/search_three_arm.py::APModel`
- `framework/search_three_arm.py::QLookup`

当前典型输入:

| 证据 | 典型路径 | 当前用途 |
|---|---|---|
| latency LUT | `results/latency_lut_pyramid.json` / `results/latency_lut_codriving.json` | 给 width / schedule 候选提供 latency |
| AP anchors | `results/ap70_model_pyramid.json` / `results/ap70_model_codriving.json` | 给 width 候选提供 AP70 或 AP50 |
| Q evidence | `results/latency_lut_pyramid_q.json` 或脚本内 `QLookup` | 判断 INT8 buildability、speedup、AP delta |
| downstream LUT | `multi_agent/real_test/ds_ap_latency_all_measured.csv` | CoDriving/Town05/clean6 下 AP50 x latency_ms -> DS/RC/collision 的实测查询表 |
| energy LUT | 尚未形成统一产物 | 目标是给候选提供能耗/功耗维度，目前是空白 |

AP x latency -> DS map 当前已有独立交接:

- `multi_agent/methods/progress/HANDOFF_ap_latency_ds_map_v2.md`
- `multi_agent/real_test/README_ds_map.md`

当前 DS map 已是 `13 latency points x 5 AP points = 65 cells` 的 CoDriving/Town05/clean6/full-traffic `_1` 实测矩形。它可以作为 downstream LUT 接入，但不能直接升级为 Stage2 主 Pareto 真值，因为 Stage2 中 AP 与 latency 可能本身来自预测或低保真估计，再映射到 DS 会形成二级/三级预测链。

当前重要口径:

```text
Stage1 manifest 定义结构和硬件合法性；
latency/AP/Q 必须来自 Stage2 evidence；
energy/DS 也必须来自 Stage2 evidence 或显式标注为未接入；
没有 evidence 时不能把 manifest 当实测结果。
```

---

## 3. 当前主要空白

### 空白 1: Stage2 证据没有统一 registry

当前各脚本直接读固定文件，路径和 schema 分散。

需要补:

```text
results/stage2/<model>/evidence_registry.json
```

或全局:

```text
results/stage2_evidence_registry.json
```

registry 至少记录:

- model。
- hardware target。
- manifest path。
- latency LUT path。
- AP anchor path。
- Q evidence path。
- energy LUT path。
- downstream DS/RC/collision LUT path。
- measurement status: `measured` / `demo` / `estimated` / `historical` / `proxy`。
- backend: `h800_tvm` / `historical_trt` / 其他显式 backend。
- scope: `dense_core` / `backbone_only` / `full_model` / `stage0_micro`。
- uncertainty / confidence policy。
- prediction chain: 例如 `measured_latency+measured_ap->measured_ds_lut` 或 `predicted_latency+predicted_ap->measured_ds_lut`。
- provenance。
- unsupported conclusions。

### 空白 2: latency LUT 生成流程未产品化

当前 Pyramid / CoDriving 的 LUT 来自多轮实验脚本和人工整理，开源仓库中只有 demo generator。

需要补:

```text
scripts/stage2_build_latency_lut.py
```

该脚本至少要明确:

- 输入 manifest / search space。
- 输入 hardware target。
- 测哪些 block。
- 测哪些 width。
- 测 default schedule 还是 MetaSchedule tuned schedule。
- 输出字段和单位。
- raw repeats、warmup、median/p50/p90/std、measurement environment。
- 如何标注失败样本。

### 空白 3: AP anchors 生成流程未产品化

当前 AP anchors 是模型实验结果，不是通用 CLI 产物。

需要补:

```text
scripts/stage2_build_ap_anchors.py
```

或明确外部评测接口:

```text
scripts/stage2_import_ap_anchors.py
```

关键要求:

- AP 是主 accuracy 目标。
- AP anchors 必须绑定 ckpt / dataset / eval split / metric。
- finetune 与不 finetune 的 AP 不能混用。
- AP 预测只能做排序或粗筛，不能替代最终真值。
- AP anchor 要记录 config id，并能与 latency/energy/Q evidence 对齐。

### 空白 4: Q evidence 的 backend / coverage 边界仍需清理

当前 Q 证据混有:

- H800 TVM stage0 INT8 microbenchmark。
- TRT INT8 AP delta 历史数据。
- `QLookup` uniform speedup proxy。

需要严格拆分:

| 类型 | 可用于什么 | 不可用于什么 |
|---|---|---|
| `measured_h800_tvm` | H800 TVM scope 内搜索/报告 | 外推到 TRT 或其他硬件 |
| `historical_trt` | historical/context evidence | Stage1 新实测后端、H800 TVM 默认依据 |
| `proxy` | smoke / sensitivity / ranking hint | 论文正式 latency/AP 结论 |
| `not_done` | blocker | 任何 pass 结论 |

V2X-ViT true TRT INT8 AP 仍应是 blocked / not done，不能在 cost model 中被默默补全。

### 空白 5: DS / closed-loop objective 尚未成为可靠统一目标

当前已有 AP x latency -> DS 的实测地图，可以作为 downstream LUT 接入:

```text
multi_agent/real_test/ds_ap_latency_all_measured.csv
```

但该 LUT 的适用范围是 CoDriving / Town05 / clean6 / full-traffic `_1`。它解决了“是否有真实闭环地图”的一部分问题，没有解决“Stage2 候选的 AP 和 latency 是否真实”的问题。如果 Stage2 候选的 AP 来自 APModel、latency 来自 latency predictor 或 proxy，再查 DS LUT，本质上是:

```text
predicted AP + predicted latency -> measured DS LUT -> predicted DS
```

这类 DS 可以用于分析、候选排序提示和 top-K 复核优先级，但不能作为论文正式主 Pareto 轴，除非 AP、latency 和 DS 查询链路都满足明确置信度要求。

下一阶段应明确:

```text
AP 是主 accuracy 目标；
latency 是主 hardware 目标；
energy 是可选 hardware/efficiency 目标；
DS/RC/collision 是可选下游目标；
没有真实闭环曲面和不确定性边界时，DS 不进入正式主 Pareto。
```

如果要接入 DS LUT，应采用以下机制降低不准确带来的决策风险:

- **证据链分级**: 区分 `measured_ap+measured_latency->ds_lut`、`measured_ap+predicted_latency->ds_lut`、`predicted_ap+predicted_latency->ds_lut`。只有第一类可考虑进入正式 downstream 分析，第三类默认只能 report-only 或 rerank。
- **覆盖域检查**: AP/latency 必须落在 DS map 的 measured rectangle 内。插值可以使用；外推禁止。`predict_ds` 的 clipping 只能作为查询保护，不能作为正式结论。
- **不确定性传播**: APModel 输出 AP 区间，LatencyLUT/latency predictor 输出 latency 区间；DS 查询输出 `DS_low/DS_mid/DS_high`，优化时优先使用保守下界或风险惩罚。
- **cliff-aware 规则**: 当前 DS cliff 支持“600ms 到 650ms 之间”的 50ms 粒度结论。若候选 latency 区间跨过 cliff band，应标注 `uncertain_due_to_cliff`，不能用单点 DS 排名替代闭环验证。
- **top-K 闭环复核**: DS LUT 只负责筛选或排序，最终推荐配置至少需要对 top-K 做真实闭环复测；没有复测时输出 `downstream_validation_required`。
- **安全指标分离**: composed DS、RC、vehicle collision map 分别输出。不能用 composed DS 单独声明 AP 影响安全，也不能把 CoDriving collision map 外推到 Pyramid。
- **模型/环境隔离**: 不能把 CoDriving/Town05/clean6 的 DS LUT 迁移成 Pyramid、其他 town/route、RSU latency 或其他 traffic setting 的 DS 真值。

必须记录:

- DS LUT 来源、raw episode pattern、routes、N、timeout policy。
- AP/latency 输入是 measured、imported、predicted 还是 proxy。
- 插值方式、覆盖域、cliff band、置信区间和 fail-closed 规则。
- DS 是否仅 report-only、rerank、constraint，或进入正式 downstream Pareto。

### 空白 6: 新硬件 evidence 需要实机测量闭环

hardware YAML 可以支持 static capability 判断，但不能替代 measured latency。

如果接入 RTX 3090 这类新硬件:

- Stage1/Stage2 可以先用 YAML 判断合法精度、alignment、op support。
- latency LUT 必须在 3090 实机生成，或显式标注为 estimated/proxy。
- AP anchors 通常与硬件无关，但 quant/backend 可能影响 AP，必须单独标注。

### 空白 7: energy LUT 生成流程未产品化

当前 cost model 还没有与 latency LUT 对齐的 energy/power evidence。若 Stage2 需要讨论 energy、功耗或 energy-delay trade-off，必须补:

```text
scripts/stage2_build_energy_lut.py
```

或先把 energy 作为 `stage2_build_latency_lut.py` 的同步测量字段输出。

energy LUT 至少要明确:

- 与 latency LUT 使用同一组 model / hardware / backend / scope / config id。
- 采样设备和接口: 例如 GPU power telemetry、board-level power、external meter，不能混写。
- 输出单位: `joule_per_inference`、`watt_avg`、`watt_p50/p90`、`sample_window_ms`。
- warmup、repeat、batch size、输入 shape、DVFS/clock policy、环境温度或 power cap。
- idle baseline 是否扣除。
- 与 latency 是否同一次 run 采样；若不是，必须记录 run id 和时间。
- energy 只能在同 hardware/backend/scope 内比较，不能跨设备做绝对结论。

建议第一阶段把 energy LUT 绑定到 latency LUT 候选集合，避免产生额外组合爆炸:

```text
energy_lut.config_id == latency_lut.config_id
```

### 空白 8: LUT 数据集规模估算和生成计划缺失

当前还缺少“需要测多少数据”的可执行计划。下一阶段应把 LUT 生成分成分阶段规模，而不是一次性全量笛卡尔积。

建议规模估算:

| LUT 类型 | 采样单位 | 最小 smoke | 校准版 | 论文/主实验版 |
|---|---|---:|---:|---:|
| latency LUT | block/config/backend/scope | 每模型 12-24 cells | 每模型 60-120 cells | 每模型 120-300 cells |
| energy LUT | 与 latency config 对齐 | 每模型 12-24 cells | 每模型 60-120 cells | 每模型 120-300 cells |
| AP anchors | model-level config | 每模型 6-10 anchors | 每模型 15-30 anchors | 每模型 40-80 anchors |
| Q evidence | quant unit/backend/config | 每模型 6-12 cells | 每模型 20-50 cells | 覆盖所有正式 Q 候选 |
| DS LUT | AP x latency x env | 复用现有 65 cells | top-K 复测 5-10 configs | 新模型/新环境单独建图或明确 blocked |

latency/energy 原始重复次数不应直接等同于 LUT cells。建议每个 cell 至少保留:

- warmup: 20-50 iterations。
- measured repeats: 50-200 iterations，按 backend 稳定性调整。
- aggregate: p50/p90/mean/std/min/max。
- raw trace: 可选压缩保存，用于复核异常点。

推荐执行顺序:

1. **smoke grid**: Pyramid/CoDriving 各 12-24 个 latency+energy cells，验证 schema、registry、失败样本和单位。
2. **calibration grid**: 每模型 60-120 个 latency+energy cells，覆盖 P anchors、Q modes、default/tuned schedule。
3. **AP anchor grid**: 先测 15-30 个代表性 model-level configs，优先覆盖 Pareto frontier 和 cliff band。
4. **DS gated validation**: 用现有 CoDriving DS map 对 Stage2 top-K 做 report-only DS 预测，再选择 5-10 个配置闭环复测。
5. **paper grid**: 只对最终报告需要的模型/硬件/backend 补到 120-300 latency+energy cells 和 40-80 AP anchors。

---

## 4. 下一阶段目标

目标 A: 建立 `Stage2EvidenceRegistry`。

建议字段:

```text
schema
model
hardware_target
manifest
latency_lut
ap_anchors
quant_evidence
energy_lut
downstream_objective
provenance
scope
measurement_status
backend
uncertainty
prediction_chain
unsupported_conclusions
```

目标 B: 让搜索器只通过 registry 读取证据。

验收:

- Pyramid / CoDriving 不再依赖硬编码路径。
- demo evidence 和 measured evidence 在输出中明显不同。
- historical TRT 不会进入 `measured_h800_tvm` 字段。
- DS/energy 缺失时显式输出 `not_available` 或 blocker，而不是静默跳过。

目标 C: 建立 cost model 输出边界。

Stage2 输出必须说明:

- latency 是 measured / estimated / proxy。
- AP 是 measured / imported / predicted。
- Q 证据的 backend 和 scope。
- energy 是 measured / unavailable / proxy。
- DS 是 measured-map lookup / predicted-chain / report-only / blocked。
- 是否允许 full-model claim。

目标 D: 接入 AP x latency -> DS downstream LUT，但默认不作为主 Pareto 轴。

验收:

- registry 能记录 `ds_ap_latency_all_measured.csv` 的 model/env/scope/provenance。
- 查询输出包含 `DS_low/DS_mid/DS_high` 或等价不确定性字段。
- predicted AP + predicted latency 查询 DS 时，输出自动标注 `predicted_ds_report_only`。
- latency 区间跨 600-650ms cliff band 时，输出 `uncertain_due_to_cliff`。
- 超出 measured rectangle 时 fail closed，不做正式外推。

目标 E: 产品化 energy LUT。

验收:

- latency LUT 与 energy LUT 共享 `config_id`。
- energy 输出包含 `joule_per_inference`、`watt_avg`、采样窗口、idle baseline policy 和 telemetry source。
- 没有 energy evidence 时 Stage2 不输出 energy improvement claim。

目标 F: 建立 LUT 数据集规模计划和采样优先级。

验收:

- 每个 registry 记录 LUT coverage: expected cells、measured cells、failed cells、proxy cells。
- Pyramid/CoDriving 至少有 smoke 级 latency+energy plan。
- AP anchors 明确 smoke/calibration/paper 三档规模。
- DS 只复用 CoDriving 65-cell map 或 top-K 闭环复测，不跨模型外推。

---

## 5. 逐空白完善计划

| 空白 | 优先级 | 完善动作 | 交付物 | 验收方式 |
|---|---|---|---|---|
| 1. evidence registry | P0 | 定义 registry schema，把 latency/AP/Q/energy/DS 都纳入统一 evidence contract | `framework/stage2/evidence_registry.py` 或等价 schema；Pyramid/CoDriving fixture | schema 测试；缺字段 fail-closed；demo/measured/historical 隔离 |
| 2. latency LUT 产品化 | P0 | 建立 latency LUT builder/importer，记录 backend、scope、unit、repeat、失败样本 | `scripts/stage2_build_latency_lut.py` 或 importer；latency LUT schema doc | Pyramid/CoDriving smoke grid 可生成；p50/p90/std 可复核 |
| 3. AP anchors 产品化 | P0 | 建立 AP anchor importer/builder，绑定 ckpt/dataset/split/metric/finetune protocol/config_id | `scripts/stage2_import_ap_anchors.py`；AP anchor schema | AP measured/imported/predicted 状态清楚；finetune 与 no-finetune 不混用 |
| 4. Q evidence 边界 | P0 | 拆分 `measured_h800_tvm`、`historical_trt`、`proxy`、`not_done` | Q evidence schema；隔离测试 | historical TRT 不能进入 H800 measured 字段；V2X-ViT true TRT INT8 未做时 blocked |
| 5. DS downstream LUT | P1 | 把现有 65-cell DS map 接入 registry，增加证据链分级、覆盖域检查、不确定性传播和 top-K 复核规则 | downstream LUT metadata；DS query wrapper；report-only/rerank/constraint mode | predicted AP+latency 查询 DS 时标注 report-only；超域 fail-closed；cliff band 标注 uncertain |
| 6. 新硬件 measured evidence | P1 | 把 hardware YAML static capability 与 measured latency/energy evidence 分开 | hardware evidence policy；new hardware checklist | 新硬件无实测 LUT 时只能输出 estimated/proxy，不输出正式 speedup/energy claim |
| 7. energy LUT 产品化 | P1 | 与 latency LUT 共用 config_id，同步或离线采集 power/energy | `scripts/stage2_build_energy_lut.py` 或 latency builder energy fields | energy 单位、telemetry、baseline、repeat 可复核；缺 energy 时 no-claim |
| 8. LUT 数据规模计划 | P0 | 建立 smoke/calibration/paper 三档规模，记录 coverage 和缺口 | `docs/stage2-evidence-format.zh-CN.md` 或 handoff 表格；registry coverage 字段 | 每个模型能说清 expected/measured/failed/proxy cells；下一轮数据生成有排期 |

---

## 6. 推荐代码入口

- `framework/search_three_arm.py`
- `framework/run_b4_ablation.py`
- `framework/run_pqs_ablation.py`
- `framework/run_pqs_codriving.py`
- `scripts/prepare_stage2_demo_data.py`
- `scripts/stage2_update_evidence.py`
- `multi_agent/methods/progress/HANDOFF_ap_latency_ds_map_v2.md`
- `multi_agent/real_test/README_ds_map.md`
- `multi_agent/methods/design/auto-tuning/2_design_cost_model_v1.md`

---

## 7. 建议验收

最小静态验收:

```bash
PYTHONPATH=${V2X_ROOT} python -m py_compile \
  framework/search_three_arm.py \
  framework/run_b4_ablation.py \
  framework/run_pqs_ablation.py \
  framework/run_pqs_codriving.py
```

registry schema 验收建议:

```bash
PYTHONPATH=${V2X_ROOT} python - <<'PY'
import json
from pathlib import Path

p = Path("results/stage2/pyramid/evidence_registry.json")
if p.exists():
    d = json.loads(p.read_text())
    assert d["schema"].startswith("stage2_evidence_registry")
    assert "latency_lut" in d
    assert "ap_anchors" in d
    assert "energy_lut" in d
    assert "downstream_objective" in d
    assert "scope" in d
    print("stage2_evidence_registry_ok")
else:
    print("registry_not_built_yet")
PY
```

正式验收应补:

- registry fixture 测试。
- `historical_trt` 与 `measured_h800_tvm` 隔离测试。
- missing AP / missing LUT fail-closed 测试。
- missing energy 输出 no-claim 测试。
- DS map 超出 measured rectangle fail-closed 测试。
- predicted AP + predicted latency -> DS 时 report-only 标注测试。
- LUT coverage 统计测试。

---

## 8. 本方向 `/goal` 启动指令

```text
/goal 推进 Stage2 cost model 与 evidence registry 工作包。建立 Stage2EvidenceRegistry 契约，让 LatencyLUT/APModel/QLookup/energy/downstream DS LUT 通过 registry 读取证据，补齐 latency LUT、AP anchors、Q evidence、energy LUT、AP x latency -> DS LUT 的 schema、backend、scope、provenance、不确定性和 coverage 计划。严格约束：Stage1 manifest 不伪装成 latency/AP/energy/DS 实测；H800 TVM 是新增实测主后端；TRT 只能标 historical；proxy/demo 不能写成论文正式结果；DS map 仅在 CoDriving/Town05/clean6 scope 内可查，predicted AP + predicted latency -> DS 默认 report-only；energy 缺失时不输出 energy claim。验收包括 registry schema 测试、Pyramid/CoDriving fixture、historical_trt 与 measured_h800_tvm 隔离测试、缺失证据 fail-closed 测试、DS 超域/跨 cliff 不确定性测试、energy no-claim 测试，以及 LUT 数据规模和 coverage 统计。
```
