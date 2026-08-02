# HANDOFF — Stage2 软硬件协同优化框架下一阶段计划 v1

日期: 2026-06-24

本文是清理上下文前的 Stage2 交接文档。它基于目前 Stage1 扫描/分类器、Stage1→Stage2 bridge、auto-tuning 设计稿、Pyramid/CoDriving 既有实验，以及刚刚对开源仓库 `github/stage1-model-scanner-aaai/` 的 Stage2 代码导出工作。

本文目标:

1. 说明当前方法已经做到什么。
2. 说明 Stage2 的输入、证据和代码运行链路。
3. 建立四个当前空白工作包索引。
4. 制定下一阶段要完成的任务目标和验收方式。

---

## 0. 当前空白工作包索引

当前 Stage2 的空白已拆成四个可并行推进的工作包。主文档只保留总览、索引和共同约束；每个方向的具体问题、目标、代码入口和 `/goal` 启动指令写在独立文件中，便于后续多个窗口分别推进。

| 工作包 | 独立交接文件 | 主要负责的问题 | 对应原空白 |
|---|---|---|---|
| 搜索空间构建 | `multi_agent/methods/progress/HANDOFF_stage2_gap_search_space_v1_zh.md` | manifest -> Stage2SearchSpace、P width anchors、层级 block、P/Q software candidate、模型级 joint/serial/noS policy、dense-core/full-model scope | 搜索空间方向空白；原空白 4/5 已弱化为非阻塞 scope/policy 约束 |
| Cost Model 与证据构建 | `multi_agent/methods/progress/HANDOFF_stage2_gap_cost_model_v1_zh.md` | latency LUT、AP anchors、Q evidence、energy LUT、AP×latency→DS LUT、evidence registry、backend/scope/provenance、LUT 数据规模计划、新硬件 measured evidence | 空白 2、3、7、8、9，外加 energy/数据规模子空白 |
| AAAI 实验补充与验证 | `multi_agent/methods/progress/HANDOFF_stage2_gap_experiments_v1_zh.md` | 审稿视角的实验缺口、AP-preserving acceleration、joint vs serial/single-axis 消融、机制图、跨模型/跨硬件泛化、搜索成本、论文 Experiment 组织 | 空白 6、10，以及论文级实验补强 |
| Stage1/Stage2 集成 | `multi_agent/methods/progress/HANDOFF_stage2_gap_stage1_stage2_integration_v1_zh.md` | model_classifier runtime gate、Stage2Input/Stage2Output 契约、统一 CLI、classifier/calibrated 回流、开源仓库 end-to-end 路径 | 空白 1、2、10 |

建议并行方式:

1. 搜索空间窗口先收敛 `Stage2SearchSpace` 和模型级 `model_search_policy`。
2. Cost model 窗口先收敛 `Stage2EvidenceRegistry`。
3. 实验窗口按 AAAI 审稿视角先收敛主 claim 证据链: 主结果、joint 消融、机制实验、跨模型泛化和搜索成本。
4. 集成窗口最后把 classifier gate、registry、search space 和 CLI 串成端到端路径。

共同约束:

- Stage1 manifest 只定义结构、coverage 和硬件合法性，不伪装成 latency/AP 实测。
- H800 TVM/Relax/MetaSchedule 是新增实测主后端。
- TRT 只能按 historical 或单独 backend scope 标注，不能写成 Stage1 新实测后端。
- dense-core evidence 不得外推为 full-model speedup。
- energy 缺失时不得输出 energy improvement claim。
- AP×latency→DS map 只能在其模型/环境/scope 内作为 downstream LUT；predicted AP + predicted latency 查询 DS 默认 report-only。
- Pyramid 写成 P(IC_BN)-hub 耦合，不写成三轴独立不可约。
- CoDriving 可分离只限已测 standard-conv dense envelope，不跨模型外推。

---

## 1. 当前方法现状

### 1.1 Stage1 已形成的能力

Stage1 当前已经能输出 partition manifest，并由模型分类器做保守三分类。manifest 不是简单日志，而是 Stage2 搜索空间的结构化输入。

Stage1 manifest 目前承担两类信息:

1. **软件结构扫描结果**
   - 哪些 dense core 可以 trace。
   - 哪些 sparse / fusion / routing / attention / custom / postprocess 子图被跳过。
   - B1 剪枝组、B1 search group、B2 量化单元、D routing segment。
   - trace boundary、coverage、manual override / review_required。

2. **硬件 capability 约束**
   - `int8_align`
   - `fp16_align`
   - `int8_pack_factor`
   - legal precision / granularity
   - DLA/NPU/GPU IP 信息和 op whitelist 骨架
   - alignment enforcement

所以 Stage1 manifest 可以理解为:

```text
模型 dense-core 结构扫描 + 硬件能力约束 + 可搜索空间定义
```

但它不是:

```text
真实 latency 结果
真实 AP/精度结果
完整模型闭环结果
完整模型可分离证明
```

### 1.2 Stage1 与 Stage2 的当前接口

当前已经存在两条接口。

第一条是正向接口:

```text
Stage1 partition manifest
    ↓
framework/stage1_bridge.py
    ↓
SpaceSpec / KnobSpec / QuantUnit / RoutingSegment
    ↓
framework/search_three_arm.py
    ↓
Stage2 P/S 或 P/Q/S 三臂搜索
```

这条接口已经是 Stage2 真正使用的主通道。它把 manifest 中的 `view_b1_search_groups`、`hw_capability`、`int8_buildable_align` 等字段转成:

- legal widths
- INT8 buildable widths
- `has_int8_buildability_cliff`
- per-knob `coupling_score` 结构信号
- `Stage2SearchSpace` / 模型级 `model_search_policy` 新契约
- 旧 `dispatch_plan` 保留给历史脚本兼容，但不作为当前搜索空间空白的停止目标

第二条是反向证据接口:

```text
Stage2 / S2 / S4 实验产物
    ↓
results/stage1_model_predict/*.json
    ↓
framework/stage1/calibrated_predictor.py
framework/stage1/model_classifier.py
    ↓
三分类 + blocker + next gate + unsupported conclusions
```

这条接口目前主要服务报告和分类器校准，而不是 Stage2 搜索 runtime 的硬 gate。

### 1.3 Stage2 当前核心输入

Stage2 运行联合优化时，需要结构输入和多类 evidence 输入。下一阶段应由 `Stage2EvidenceRegistry` 统一管理这些 evidence，而不是让各脚本硬编码路径。

| 输入 | 典型路径 | 作用 |
|---|---|---|
| Stage1 manifest | `framework/partitions/<model>_partition.yaml` | 定义 P/Q/D 搜索空间和硬件合法性 |
| 延迟 LUT | `results/latency_lut_*.json` | 给每个宽度/schedule 候选提供 latency |
| AP/精度锚点 | `results/ap70_model_*.json` | 给每个宽度候选提供 AP/accuracy，用于 Pareto/HV |
| 可选量化证据 | `results/latency_lut_*_q.json` 或 `QLookup` | 判断 INT8 是否可 build、速度收益和 AP delta |
| 可选 energy LUT | 待产品化 | 给候选提供 energy / power / energy-delay evidence |
| 可选 downstream DS LUT | `multi_agent/real_test/ds_ap_latency_all_measured.csv` | 在 CoDriving/Town05/clean6 scope 内把 AP50×latency 映射到 DS/RC/collision |

这些输入在代码中的对应模块或计划入口:

- `framework/stage1_bridge.py`
- `framework/stage2/evidence_registry.py`
- `framework/search_three_arm.py::LatencyLUT`
- `framework/search_three_arm.py::APModel`
- `framework/search_three_arm.py::QLookup`
- planned `EnergyLUT` / downstream objective lookup
- `framework/run_b4_ablation.py`
- `framework/run_pqs_ablation.py`
- `framework/run_pqs_codriving.py`

### 1.4 Stage2 当前搜索逻辑

当前 Stage2 主要是三臂度量仪:

| 搜索臂 | 含义 |
|---|---|
| `A-noS` | 消掉 schedule 轴，只用 default schedule |
| `A-serial` | 先按局部/default 口径锁定 P，再在锁定宽度上调 schedule 或 Q×S |
| `A-joint` | P×S 或 P×Q×S 联合搜索 |

这套三臂逻辑的作用不是只给一个模型找最优点，而是度量:

```text
这个模型/这个 dense-core/这个硬件约束下，协同搜索是否真的比串行搜索必要？
```

Pyramid 和 CoDriving 当前的核心区别正是由这套度量仪得到:

- Pyramid grouped conv: P(IC_BN)-hub 耦合强，joint 明显优于 serial。
- CoDriving standard conv: P/Q/S 在当前低维空间中可分离，serial 达到 joint。

---

## 2. 当前已有实验结果

### 2.1 Pyramid

主要产物:

- `results/b4_ablation_results.json`
- `results/b5_convergence_verification.json`
- `results/pqs_ablation_results.json`
- `results/q_int8_ms_stage0_result.json`
- `results/ap_cliff_l4.json`
- `results/coupling_map_matrix.json`

P×S 三臂结果:

| arm | mean HV | ref 占比 |
|---|---:|---:|
| `A-joint` | 6984.217 | 99.90% |
| `A-serial` | 5978.342 | 85.51% |
| `A-noS` | 3336.904 | 47.73% |

显著性:

- Wilcoxon `p=0.00048828125`
- rank-biserial = `1.0`

关键 shipped win:

| W_g | P_g | AP70 | W_g tuned us | P_g tuned us | ratio |
|---|---|---:|---:|---:|---:|
| `[48,128,128]` | `[64,128,128]` | 0.6369 | 21071.43 | 5506.57 | 3.827x |

当前科学解释:

```text
Pyramid 的耦合是真实的，但正确表述是 P(IC_BN)-hub。
P 改变 IC_BN，进而改变 schedule/tensorization/INT8 fast path 的可行域和收益。
不要写成 P/Q/S 三轴完全独立不可约。
```

P×Q×S 的当前状态:

- 当前 `results/pqs_ablation_results.json` 是 4 seed。
- 早先交接文字曾写 12 seed 和 p=4.88e-4。
- 这个口径必须下一阶段统一，不能在论文表格里混用。
- 当前 JSON 仍显示结构性趋势: `A-joint-PQS` 100%，`A-serial-PQS` 83.56%，`A-noS-PQS` 56.81%，但显著性因 n=4 只有 `p=0.125`。

### 2.2 CoDriving

主要产物:

- `results/coupling_map/C0c_codriving_pqs.json`
- `results/codriving_int8_verify.json`
- `results/codriving_isobudget_verdict.csv`
- `results/codriving_dair_grid_8point_clean.csv`

P×Q×S 三臂结果:

| arm | mean HV | joint 占比 |
|---|---:|---:|
| `A-joint-PQS` | 1164.826 | 100.0% |
| `A-serial-PQS` | 1164.826 | 100.0% |
| `A-noS-PQS` | 1130.655 | 97.07% |

结论:

- `verdict=SERIAL`
- rank-flip pairs = 0
- all widths INT8 buildable
- 标准 conv 在当前 P×Q×S 空间可分离

CoDriving INT8 验证:

| config | FP16 us | INT8 us | speedup | buildable | rank flip |
|---|---:|---:|---:|---|---|
| Cin=48 | 76.49 | 53.98 | 1.42x | True | False |
| Cin=64 | 89.61 | 67.89 | 1.32x | True | False |

训练 confound:

- `base_isobudget` AP50 = 0.6263，高于所有剪枝档。
- “剪枝提高 AP”不是剪枝本身，而是第二轮训练/退火协议 confound。

部署口径限制:

- CoDriving body-level 剪枝/INT8 有加速。
- 但 full e2e 受 eager / decode / fusion / copy / launch 开销稀释。
- 后续若要把优化兑现到全网，必须先减少这些非 body 开销。

### 2.3 开源导出仓库状态

已更新:

- `github/stage1-model-scanner-aaai/framework/search_three_arm.py`
- `github/stage1-model-scanner-aaai/framework/run_b4_ablation.py`
- `github/stage1-model-scanner-aaai/framework/run_pqs_ablation.py`
- `github/stage1-model-scanner-aaai/framework/run_pqs_codriving.py`
- `github/stage1-model-scanner-aaai/scripts/prepare_stage2_demo_data.py`
- `github/stage1-model-scanner-aaai/scripts/phase2/closedloop_objective_query.py`
- `github/stage1-model-scanner-aaai/scripts/phase2/b5_verify_convergence.py`
- `github/stage1-model-scanner-aaai/scripts/phase2/b4_integrate.py`
- `github/stage1-model-scanner-aaai/README.md`
- `github/stage1-model-scanner-aaai/README.zh-CN.md`

已验证 clone-like smoke:

```bash
python -m py_compile ...
PYTHONPATH=. python scripts/prepare_stage2_demo_data.py
PYTHONPATH=. python -m framework.search_three_arm --seeds 1 --budget 8 --pop 4 --quiet
PYTHONPATH=. python -m framework.run_b4_ablation --seeds 1 --budget 8 --pop 4 --quiet
PYTHONPATH=. python -m framework.run_pqs_ablation --manifest framework/partitions/pyramid_lidar_partition.yaml --seeds 1 --budget 8 --pop 4 --quiet
PYTHONPATH=. python -m framework.run_pqs_codriving --seeds 1 --budget 8 --pop 4 --quiet
PYTHONPATH=. python scripts/phase2/b5_verify_convergence.py
```

验证输出:

```text
stage2_export_smoke_ok
```

验证产生的 `results/`、demo manifest 和 `__pycache__` 已清理。仓库只保留源码和 README 变更。

---

## 3. 当前主要空白索引

原先第 3 节的 10 个空白已经拆成四个方向。后续多个窗口并行推进时，应优先阅读对应方向的独立文件；主文档只保留索引和交叉边界。

### 3.1 搜索空间构建空白

独立文件:

```text
multi_agent/methods/progress/HANDOFF_stage2_gap_search_space_v1_zh.md
```

本方向负责:

- Stage1 manifest 如何变成 Stage2 search space。
- width candidates 如何从 dense candidate path 和硬件合法性自动生成。
- block / knob / quant unit / routing segment 的正式边界。
- P width anchors 如何由结构、硬件、AP/LUT 边界自动取点。
- block 如何表达 device -> execution block -> dense stage -> search group -> knob 层级。
- P/Q 如何合并为 software candidate。
- 模型级 joint/serial/noS policy 如何输出。
- dense-core search space 与 full-model claim 如何分离并弱化为 scope 约束。

该方向主要吸收原空白:

- 原空白 4: 已明确当前不是 per-knob dispatch 预算问题，而是模型级 joint/serial/noS policy。
- 原空白 5: 已弱化为 dense-core/full-model scope 与 claim 约束；RSU dense-core 先闭环，ego fusion/attention 后续处理。
- 空白 9 中 static capability 与硬件合法性判断部分。

### 3.2 Cost Model 与证据构建空白

独立文件:

```text
multi_agent/methods/progress/HANDOFF_stage2_gap_cost_model_v1_zh.md
```

本方向负责:

- latency LUT / AP anchors / Q evidence / energy LUT / downstream DS LUT 的 schema、生成流程和消费方式。
- 统一 `Stage2EvidenceRegistry`。
- H800 TVM measured、historical TRT、proxy、demo evidence 的边界。
- AP×latency→DS LUT 的证据链分级、不确定性传播、cliff-aware 规则和 report-only/rerank/constraint 边界。
- energy LUT 的采样接口、单位、baseline 和与 latency config 的对齐。
- LUT 数据集规模估算、smoke/calibration/paper 三档采样计划和 coverage 统计。
- 新硬件 measured latency evidence 如何接入。

该方向主要吸收原空白:

- 空白 2: Stage2 输入没有统一 evidence registry。
- 空白 3: latency LUT / AP anchors 的生成流程还不是产品化 CLI。
- 空白 7: 量化证据仍有 backend/coverage 边界。
- 空白 8: AP/DS/闭环目标仍未成为可靠统一目标。
- 空白 9 中 measured evidence 与新硬件实测部分。
- 新增 cost-model 子空白: energy LUT 生成流程未产品化。
- 新增 cost-model 子空白: latency/AP/Q/energy/DS LUT 数据集规模估算和生成计划缺失。

### 3.3 AAAI 实验补充与验证空白

独立文件:

```text
multi_agent/methods/progress/HANDOFF_stage2_gap_experiments_v1_zh.md
```

本方向负责:

- 从 AAAI 审稿视角定义论文还缺哪些实验，而不是只收口脚本或 demo。
- AP-preserving acceleration 主结果: 真实 checkpoint、真实硬件、真实 AP/latency。
- joint vs serial/noS/single-axis 公平消融: 证明协同搜索优于单轴和强串行 baseline。
- P(IC_BN)-hub / rank-flip 机制实验: 解释 joint 为什么赢，避免 cherry-pick。
- 跨模型和跨硬件泛化: 证明框架不是 Pyramid 单点特例。
- 搜索成本、cost model/evidence 质量、full-model/energy/downstream 边界。
- 论文 Experiment 部分组织结构和每个实验的技术依赖/缺口。

该方向主要吸收原空白:

- 空白 6: Pyramid P×Q×S seed 口径必须收口。
- 空白 10 中开源 smoke 与论文正式实验复现的区别。
- 第 2 节已有实验结果在论文主 claim 中能支撑什么、不能支撑什么。
- 新增论文级实验空白: 主结果、single-axis baseline、机制图、跨模型、搜索成本、外部 baseline、energy/DS 边界。

### 3.4 Stage1/Stage2 集成空白

独立文件:

```text
multi_agent/methods/progress/HANDOFF_stage2_gap_stage1_stage2_integration_v1_zh.md
```

本方向负责:

- `model_classifier` 如何成为 Stage2 runtime gate。
- `Stage2Input` / `Stage2Output` 正式契约。
- `scripts/stage2_optimize_model.py` 统一入口。
- Stage2 结果如何回流到 calibrated predictor / model classifier。
- 开源仓库 fresh clone 的 Stage1→Stage2 端到端路径。

该方向主要吸收原空白:

- 空白 1: `model_classifier` 尚未成为 Stage2 runtime gate。
- 空白 2 中 Stage2 输入契约部分。
- 空白 10 中开源仓库 end-to-end 使用路径。

### 3.5 原空白编号到新工作包的映射

| 原空白编号 | 原问题 | 主要归属 | 次要关联 |
|---|---|---|---|
| 1 | `model_classifier` 尚未成为 Stage2 runtime gate | Stage1/Stage2 集成 | 搜索空间构建 |
| 2 | Stage2 输入没有统一 evidence registry | Cost Model 与证据构建 | Stage1/Stage2 集成 |
| 3 | latency LUT / AP anchors 生成流程未产品化 | Cost Model 与证据构建 | 实验收口与验证 |
| 4 | `dispatch_plan()` 未驱动内环预算 | 已弱化: 当前按模型级 joint/serial/noS policy，不做 per-knob dispatch | Stage1/Stage2 集成需兼容旧脚本 |
| 5 | Stage2 仍主要优化 traced dense core | 已弱化: RSU dense-core 当前闭环，full-model claim 受 scope 约束 | 实验收口与验证 |
| 6 | Pyramid P×Q×S seed 口径必须收口 | 实验收口与验证 | Cost Model 与证据构建 |
| 7 | 量化证据 backend/coverage 边界 | Cost Model 与证据构建 | 实验收口与验证 |
| 8 | AP/DS/闭环目标未统一 | Cost Model 与证据构建 | 实验收口与验证；DS LUT 只能 scope 内 report-only/rerank/constraint |
| 9 | 新硬件能力分成 static capability 和 measured evidence | 搜索空间构建 / Cost Model 与证据构建 | Stage1/Stage2 集成；energy 也需要 measured evidence |
| 10 | 开源仓库 smoke 可运行但非完整复现包 | 实验收口与验证 / Stage1/Stage2 集成 | Cost Model 与证据构建 |

新增 cost-model 子空白:

| 子空白 | 问题 | 主要归属 | 次要关联 |
|---|---|---|---|
| 11 | energy LUT 生成流程未产品化 | Cost Model 与证据构建 | 实验收口与验证 |
| 12 | latency/AP/Q/energy/DS LUT 数据集规模估算和生成计划缺失 | Cost Model 与证据构建 | Stage1/Stage2 集成；实验收口与验证 |

---

## 4. 下一阶段任务目标

### 目标 A: 建立 Stage2 正式输入/输出契约

新增或明确:

```text
Stage2Input:
  manifest_path
  model_classification_path
  evidence_registry_path
  hardware_target
  search_policy

Stage2Output:
  schema
  model
  backend_policy
  stage1_gate
  search_space_summary
  model_search_policy
  arms
  pareto_front
  recommended_configs
  evidence_sources
  blockers
  unsupported_conclusions
```

验收:

- 有 JSON schema 或 dataclass contract。
- 有至少 2 个 fixture: Pyramid、CoDriving。
- 旧的 `run_b4_ablation.py` / `run_pqs_ablation.py` 能被新入口调用或复用。

### 目标 B: 将 `model_classifier` 接入 Stage2 gate

实现:

```text
SCAN_FAILED -> fail closed
SEPARABLE_ACCELERATION -> serial/default low-budget
CO_ACCELERATION_REQUIRED -> model-level joint or search-space policy driven
```

验收:

- `where2comm/v2vnet/disconet` 这类 `SCAN_FAILED` 不会进入自动优化。
- `codriving` 默认走 serial/low-budget。
- `pyramid` 默认启用 joint/P-hub policy。

### 目标 C: 实现 evidence registry

建议新增:

```text
framework/stage2/evidence_registry.py
scripts/stage2_prepare_evidence.py
```

registry 至少记录:

- manifest
- latency LUT
- AP anchors
- Q evidence
- energy LUT
- downstream DS/RC/collision LUT
- backend
- measured/demo/estimated/historical 状态
- scope
- uncertainty / confidence policy
- prediction chain
- coverage: expected/measured/failed/proxy cells

验收:

- demo data generator 生成 registry。
- Pyramid/CoDriving 可以通过 registry 而不是硬编码路径运行。
- registry 中历史 TRT 和 H800 TVM 明确分开。
- energy 缺失时输出 no-claim，不静默写成优化收益。
- predicted AP + predicted latency 查询 DS 时输出 report-only 或 validation-required。

### 目标 C2: 完成 cost evidence 数据计划

实现:

- latency/energy LUT 使用同一组 `config_id`。
- AP anchors 绑定 ckpt、dataset、eval split、metric、finetune protocol。
- DS LUT 记录 CoDriving/Town05/clean6 scope、cliff band、插值/超域规则。
- registry 输出 smoke/calibration/paper 三档 coverage 计划。

验收:

- Pyramid/CoDriving 各有 smoke 级 latency+energy cells 计划。
- AP anchors 有 6-10 / 15-30 / 40-80 三档规模口径。
- DS 复用现有 65-cell map 时不跨模型外推；新模型/新环境标 blocked 或另建图。
- 超出 DS measured rectangle fail closed。

### 目标 D: 让模型级搜索策略进入 Stage2 输出

实现:

- Stage2 读取 `Stage2SearchSpace.model_search_policy`。
- Pyramid 这类 P(IC_BN)-hub 模型输出 joint。
- CoDriving 这类已测 standard-conv dense envelope 输出 serial/default。
- 输出 policy 依据、unsupported conclusions 和 scope 约束。

验收:

- Pyramid 模型级 policy 为 joint。
- CoDriving 模型级 policy 为 serial/default。
- 单元测试检查不依赖简单 `if model == "pyramid"`，且不输出 per-knob joint/serial 作为当前结论。

### 目标 E: 复核 Pyramid P×Q×S 正式结果

动作:

1. 重跑或找回 12 seed P×Q×S。
2. 更新 `results/pqs_ablation_results.json`。
3. 更新 `coupling_map_matrix.json` 中相关数字。
4. 明确哪些是 real H800 TVM、哪些是 proxy、哪些是 historical TRT。

验收:

- `n_seeds` 与报告一致。
- Wilcoxon / HV 数字统一。
- 文档不再出现 4 seed 和 12 seed 混写。

### 目标 F: full-model coverage 与 dense-core coverage 分离输出

实现:

- Stage2 输出中新增:
  - `optimized_scope`
  - `trace_dense_core_latency_scope`
  - `skipped_subgraph_blockers`
  - `full_model_claim_allowed`
- 若只有 dense-core LUT，不允许输出 full-model speedup claim。

验收:

- Pyramid / CoDriving 都能输出 dense-core 结论和 full-model caveat。
- CoDriving body-level 优化不会被写成 full e2e 优化。

### 目标 G: 开源仓库形成最小可复现实验

动作:

- 增加 Stage2 smoke pytest 或 shell check。
- README 增加“真实模型替换 demo 数据”的最小 checklist。
- 增加 `docs/stage2-evidence-format.zh-CN.md`。
- 增加 `docs/stage2-new-hardware.zh-CN.md`。

验收:

```bash
PYTHONPATH=. python scripts/prepare_stage2_demo_data.py
PYTHONPATH=. python -m framework.search_three_arm --seeds 1 --budget 8 --pop 4 --quiet
PYTHONPATH=. python -m framework.run_pqs_ablation --manifest framework/partitions/pyramid_lidar_partition.yaml --seeds 1 --budget 8 --pop 4 --quiet
PYTHONPATH=. python -m framework.run_pqs_codriving --seeds 1 --budget 8 --pop 4 --quiet
```

在 fresh clone 中可通过。

---

## 5. 下一阶段建议执行顺序

推荐顺序:

1. **先做契约**
   - Stage2 input/output dataclass。
   - evidence registry。
   - classifier gate。

2. **再补 cost evidence 计划**
   - latency/AP/Q/energy/DS 的 registry schema。
   - DS LUT 的 report-only/rerank/constraint 规则。
   - LUT smoke/calibration/paper 三档数据规模和 coverage 字段。

3. **再改搜索入口**
   - 新增统一 CLI，例如 `scripts/stage2_optimize_model.py`。
   - 旧 B4/PQS 脚本保留为实验驱动器。

4. **再接模型级搜索策略**
   - 使用 `Stage2SearchSpace.model_search_policy`。
   - 输出 joint/serial/noS 的模型级选择依据。

5. **再复核实验结果**
   - Pyramid P×Q×S seed 口径。
   - CoDriving separability 结果。

6. **最后整理开源文档**
   - README 保持简练。
   - 新模型、新硬件、Stage2 evidence 格式放 docs 链接。

---

## 6. 下一阶段 `/goal` 启动指令

```text
/goal 进入 Stage2 软硬件协同优化框架产品化阶段。基于当前 Stage1 manifest/model_classifier、stage1_bridge.SpaceSpec/Stage2SearchSpace、search_three_arm 三臂搜索内核、Pyramid/CoDriving 已有实验，以及 CoDriving AP×latency→DS 65-cell measured map，完成 Stage2 正式输入输出契约、evidence registry、classifier gate 接入、模型级 model_search_policy 输出、cost evidence 数据计划和 DS/energy 证据边界。严格约束：Stage1 manifest 只定义结构/硬件合法性，不伪装成实测 latency/AP/energy/DS；H800 TVM/Relax/MetaSchedule 是新增实测主后端；TRT 只能按 historical 或单独 backend scope 标注；AP×latency→DS LUT 只在 CoDriving/Town05/clean6 scope 内可作为 downstream lookup，predicted AP + predicted latency 查询 DS 默认 report-only；energy 缺失时不得输出 energy improvement claim；Pyramid 结论写成 P(IC_BN)-hub 耦合，不写成三轴独立不可约；CoDriving 可分离只限已测标准卷积 dense envelope，不跨模型外推；当前不做 per-knob joint/serial 分类。验收包括：Pyramid/CoDriving 两个 fixture 的 Stage2 gate 测试、evidence registry schema 测试、Stage2SearchSpace/model_search_policy 测试、DS 超域/跨 cliff 不确定性测试、energy no-claim 测试、LUT smoke/calibration/paper 数据规模和 coverage 统计、Pyramid P×Q×S seed 口径复核、fresh-clone smoke，以及中文 Stage2 evidence/new-hardware 文档。
```

---

## 7. 清空上下文后优先读取

1. `multi_agent/methods/progress/HANDOFF_stage2_codesign_next_phase_plan_v1_zh.md`
2. `multi_agent/methods/progress/HANDOFF_stage2_gap_search_space_v1_zh.md`
3. `multi_agent/methods/progress/HANDOFF_stage2_gap_cost_model_v1_zh.md`
4. `multi_agent/methods/progress/HANDOFF_stage2_gap_experiments_v1_zh.md`
5. `multi_agent/methods/progress/HANDOFF_stage2_gap_stage1_stage2_integration_v1_zh.md`
6. `multi_agent/methods/progress/HANDOFF_stage2_codesign_prep_v1_zh.md`
7. `multi_agent/methods/design/auto-tuning/1_design_space_building_v1.md`
8. `multi_agent/methods/design/auto-tuning/2_design_cost_model_v1.md`
9. `multi_agent/methods/design/auto-tuning/3_design_exploring_tvm_integration_v1.md`
10. `multi_agent/methods/design/auto-tuning/4_design_ablation_proof_v1.md`
11. `multi_agent/methods/progress/HANDOFF_ap_latency_ds_map_v2.md`
12. `multi_agent/real_test/README_ds_map.md`
13. `framework/stage1_bridge.py`
14. `framework/search_three_arm.py`
15. `framework/run_b4_ablation.py`
16. `framework/run_pqs_ablation.py`
17. `framework/run_pqs_codriving.py`
18. `results/b4_ablation_results.json`
19. `results/b5_convergence_verification.json`
20. `results/pqs_ablation_results.json`
21. `results/coupling_map/C0c_codriving_pqs.json`
22. `results/coupling_map_matrix.json`
23. `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json`

---

## 8. 关键提醒

- 不要把 demo data 当论文实验结果。
- 不要把 dense-core speedup 写成 full-model speedup。
- 不要把 `groups=1` 写成模型级可分离证明；CoDriving 可分离来自三臂测量结果。
- 不要把 TRT 当 Stage1 新实测后端。
- 不要把 Pyramid 写成三轴独立不可约；目前正确说法是 P(IC_BN)-hub。
- 新硬件没有接入实机测量时，只能做 capability 结构判断，不能声称正确 latency/AP 结论。
- 不要把 predicted AP + predicted latency 查 DS LUT 得到的 DS 写成真实闭环结果；默认只能 report-only 或触发 top-K 复测。
- 不要在 energy LUT 缺失或 telemetry/source 不一致时声明 energy improvement。
