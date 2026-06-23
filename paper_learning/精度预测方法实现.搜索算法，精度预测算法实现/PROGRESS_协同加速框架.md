# 协同加速框架 — 项目进度跟踪 (PROGRESS)

> **维护原则**: 每完成一个 Stage 任务、获得新决策、识别新阻塞项后,**立即更新本文档**.
> **当前阶段**: 整体路径升级为"b2d → UniV2X-tiny" 7 阶段流程. 决策清单 v1.3 定型 + **阶段 1 已启动并初步通过 L2/L4 实验**.
> **创建**: 2026-04-27
> **最后更新**: 2026-04-27 (深夜, 阶段 1 L2+L4 实验初步完成)
> **AAAI deadline**: 估 2026-08 中, 距今约 3.5 个月

---

## 目录

1. [总体目标](#1-总体目标)
2. [阶段路线图与状态](#2-阶段路线图与状态)
3. [已完成阶段详情](#3-已完成阶段详情)
4. [当前阻塞项](#4-当前阻塞项)
5. [资产清单](#5-资产清单)
6. [关键决策记录](#6-关键决策记录)
7. [重要发现](#7-重要发现)
8. [待启动任务](#8-待启动任务)
9. [更新日志](#9-更新日志)

---

## 1. 总体目标

实现 UniV2X 上的 **B1(剪枝) + B2(量化) + D(部署) 联合搜索**,在 Pareto 前沿上找到精度-延迟最优 trade-off.

- **主战场**: NVIDIA RTX 4090 (Ada sm89)
- **目标硬件**: NVIDIA Jetson Orin AGX (Ampere sm87) — 用于关键节点验证
- **次目标硬件**: Jetson Orin Nano (无 DLA) — 跨硬件迁移验证
- **完整时间线**: 8 周 (W1-W8),分 Phase 1A / 1B / Stage 1-5

参考文档:
- 总框架: [`协同加速框架_工作流_v1.5.md`](../协同加速框架_工作流_v1.5.md) (在 paper_learning/ 根目录)
- 实施计划: [`精度预测方法实现.md`](./精度预测方法实现.md) (本目录)

> **文档存放约定** (2026-04-27 确认):
> - 所有 `.md` 文档(报告 / 设计稿 / PROGRESS / 阶段记录) → **本子目录** `paper_learning/精度预测方法实现.搜索算法，精度预测算法实现/`
> - Python 代码 / YAML / JSON 配置 / parquet 数据 / 模型文件 → **仓库根目录**(`framework/`, `configs/`, `data/`, `prune_configs/`, `quant_configs/`, `models/`, `results/`, `scripts/`)
> - 跨目录引用用相对路径: 子目录内文档之间用 `./xxx.md`,引用根目录文档用 `../xxx.md`

---

## 2. 阶段路线图与状态

### 老路线(原 Phase 1A/1B + Stage 1-5)

| 阶段 | 时长 | 状态 | 完成度 | 备注 |
|------|------|------|--------|------|
| **Phase 1A** 4090 脚手架 | W1-W2 | ✅ 完成 | 8/8 任务 | 2026-04-27 完成 |
| **Phase 1B** Orin 健全性 | W2-W3 | ⏸ 改为操作手册模式(v1.3) | 0/7 任务 | Orin 不在同一局域网,改为我做制品+手册,你手动执行 |
| **Stage 1** latency LUT 完善 | W3-W4 | ⏳ 待启动 | 0/3 任务 | 在 UniV2X-tiny 出来后再做 |
| **Stage 2** 数据基础准备 | W4 | 🟡 部分完成 | 22/200+ 行 | 等 tiny 出来后重做 v3 |
| **Stage 3** 精度预测器 | W5-W6 | 🟡 sanity 完成 | 4/7 任务 | sanity ρ=0.59 PASS;待 Stage 2.5 数据后重训 |
| **Stage 4** 搜索算法 | W6-W7 | ⏳ 待启动 | 0/7 任务 | — |
| **Stage 5** 集成验证 | W7-W8 | ⏳ 待启动 | 0/7 任务 | — |

### 新路线(v1.3 升级:b2d → UniV2X-tiny 7 阶段)

| 阶段 | 时长 | 状态 | 完成度 | 备注 |
|------|------|------|--------|------|
| **阶段 0** 立项决策 | 即时 | ✅ 完成 (v1.3 定型) | 13/13 决策 | 见 [完整决策清单](./完整决策清单_b2d到UniV2X-tiny.md) |
| **阶段 1** R50 全网剪枝可行性 + 速度边界 + R101 失败对照 | 2-3 天 | 🟡 **进行中** | 3/4 层完成 (L2 ✅ L4 ✅, L3 partial, L1 待 b2d 数据准备) | **2026-04-27 深夜启动**,见下方阶段详情 |
| **阶段 2** 跨域可行性测试 (UniAD-tiny on V2X-Seq-SPD) | 1-2 天 | ⏳ 待启动 | 0/5 决策 | 阶段 1 完成后启动 |
| **阶段 2.5** 训练路径决策 (X/Y/Z) | 即时 | ⏳ 待启动 | 阶段 2 后 | — |
| **阶段 3** Stage 2.5 v2 + Stage 3/4 重训 (R101 锁 backbone, fallback) | 1 周 | ⏳ 待启动 (条件触发) | 0/9 决策 | 仅当阶段 1 L2 ✗ 时启动 |
| **阶段 4** UniV2X-tiny 训练 | 3-7 天 (6 卡并行) | ⏳ 待启动 | 0/11 决策 | 仅当阶段 2.5 = 路径 Y/Z 时启动 |
| **阶段 5** 在 tiny 上重跑 Stage 2.5/3/4/5 | 5-7 天 | ⏳ 待启动 | 0/5 决策 | 阶段 4 完成后 |
| **阶段 6** 论文实验 + 写作 | 2-3 周 | ⏳ 待启动 | 0/5 决策 | AAAI 8 月中投稿 |
| **阶段 7** Orin 部署验证 (操作手册模式) | 1-2 天 | ⏸ 待网络/制品准备 | 0/5 决策 | 我做制品+手册,你手动执行 |

**图例**: ✅ 完成 | 🟡 进行中/部分完成 | ⏸ 阻塞 | ⏳ 待启动

---

## 3. 已完成阶段详情

### Phase 1A — 4090 脚手架(2026-04-27 完成)

**目标**: 把"协同加速框架"从概念变成"能跑的 Python 包",4090 上能输入 ONNX + capability YAML,输出 50 个合法配置 + 粗略 latency 排序.

**任务完成情况(8/8)**:

| ID | 任务 | 文件 | 行数 | 验收 |
|----|------|------|------|------|
| 1A.1 | capability schema (Pydantic v2) | `framework/capability_schema.py` | 130 | ✅ 严格模式校验通过 |
| 1A.2 | RTX 4090 / Orin AGX / Orin Nano YAML | `configs/hardware/*.yaml` | 101 | ✅ 三份过 schema 校验 |
| 1A.3 | Config dataclass + 序列化 | `framework/config_schema.py` | 209 | ✅ 兼容 1.1/1.2 现有 JSON |
| 1A.4 | 硬约束 DSL (物理/经验/软三类) | `framework/constraints.py` | 227 | ✅ 9/9 单测通过 |
| 1A.5 | D↔B2 双向传播 | `framework/propagation.py` | 101 | ✅ 5/5 单测通过 + 不动点收敛 |
| 1A.6 | 最小搜索器 v0 (随机+Repair) | `framework/searcher_v0.py` | 171 | ✅ 通过率 75.76% / 50 候选无重复 |
| 1A.7 | 1.1/1.2 数据整合 baseline | `data/phase1/baseline_unified.parquet` | 237 | ⚠️ 22 行(目标 200+,Stage 2.5 补) |
| 1A.8 | 4090 端到端 sanity | `scripts/phase1/run_phase1a_sanity.py` | 118 | ✅ 三硬件均跑通 < 1s |

**代码统计**: 1614 行(8 个 Python 模块 + 3 份 YAML + 2 个测试 + 2 个脚本)
**测试覆盖**: 14/14 单元测试通过(constraints 9 + propagation 5)
**性能**: 4090 上 50 候选 < 1 秒 (验收线 5 分钟)

**关键产物 latency 估计区间**(粗略估计器,正式 LUT 在 Stage 1):
- RTX 4090: 8.0 - 74.9 ms
- Jetson Orin AGX: 17.7 - 166.4 ms
- Jetson Orin Nano: 35.4 - 332.8 ms

---

### Phase 2 Stage 3 — LightGBM 精度预测器 sanity(2026-04-27 完成 4/7)

**目标**: 在 22 行 baseline 上跑通流水线,暴露数据缺口,为 Stage 2.5 主动采样指明方向.

**任务完成情况(4/7,S3.4/S3.5/S3.6 因数据少跳过)**:

| ID | 任务 | 文件 | 验收 |
|----|------|------|------|
| S3.1 | 特征工程 (类别 + 数值 + 派生) | `framework/feature_encoder.py` | ✅ 22 行 → 48 列 (26 类别 + 22 数值) |
| S3.2/3.3 | LightGBM LOOCV 训练 + 评估 | `scripts/phase2/train_lgb_predictor.py` | ✅ Spearman ρ=+0.5885 (sanity > 0.50 PASS) |
| S3.7 | AccuracyPredictor 接口固化 | `framework/accuracy_predictor.py` | ✅ 7/7 单测 + 批量预测 180μs/cfg |
| S3.4 | 调参 | — | ⏭ 跳过 (22 行无意义) |
| S3.5 | MLP 升级 | — | ⏭ 跳过 (22 行不够 NN) |
| S3.6 | 不确定性估计 | — | ⏭ 跳过 (22 行 quantile 不可信) |

**核心指标(LOOCV 留一交叉验证)**:

| 指标 | 实测 | sanity 阈值 | 终目标 |
|------|------|------------|--------|
| Spearman ρ | **+0.5885** | 0.50 ✅ | 0.85 ⚠️ |
| Pareto rank ρ | +0.5885 | — | 0.75 |
| MAE | 0.0220 | — | 0.005 |
| 训练 MAE | 0.0179 | — | (OOF/train ≈ 1.23) |

**最关键的发现(暴露数据盲点)**:
- 模型只用了 **2/48 特征** — 仅 `prune_rate__encoder` + `prune_criterion__encoder` 有 gain > 0
- 其他 46 个特征中,大部分**方差为 0** — 1.1/1.2 数据中 backbone/heads/v2x_comm 的 prune_rate 全 0,d_routing 全 GPU(Phase 1A 锁死)
- **直接证明 Stage 2.5 主动采样必须覆盖多模块差异化的配置**(详见 sanity 报告 §四)

**速度验证**: 1000 候选批量预测 180ms (180μs/cfg) — 完全满足 NSGA-II 1000 评估 < 30s 的速度需求

**产物**:
- 详细报告: [`2.0阶段_phase2_sanity报告.md`](./2.0阶段_phase2_sanity报告.md)
- 模型: `models/lgb_predictor_v0.txt` + `lgb_predictor_v0_meta.json`
- 评估结果: `results/phase2_stage3_*.{csv,json}`

---

## 4. 当前阻塞项

| # | 阻塞项 | 影响 | 等待方 | 状态 |
|---|-------|------|--------|------|
| **B1** | **Orin 网络不可达** (10.201.79.25:1234) | Phase 1B 全部 7 项 / Stage 1.2 Top-K 实测 / Stage 5.6 复现率分析 | 用户提供跳板机或路由 | 2026-04-27 已诊断 |
| **B2** | **数据量缺口** | baseline_unified 只 22 行 (目标 200+),直接训预测器会过拟合 | Stage 2.5 主动采样补 30-50 个联合配置 | 待启动 |
| **B3** | **关键决策待用户确认** | 影响后续路径选择 | 用户回答 §6 决策 #1-#5 | 待确认 |

### B1 详情 — Orin 网络诊断 (2026-04-27)

| 检查 | 结果 |
|------|------|
| 本机网卡 | `192.168.0.2/24` (vlan2586),网关 `192.168.0.254` |
| 互联网可达 | ✓ (8.8.8.8 / baidu.com 都通) |
| `ping 10.201.79.25` | 100% 丢包 |
| `ping 10.201.79.1/254` (Orin 网关) | 100% 丢包 |
| TCP 1234 | Connection timed out |

**根因**: 开发机 192.168.0.0/24 与 Orin 10.201.79.0/24 之间无路由.

**解决路径(待用户选择)**:
- A. 跳板机 (`ssh -J jump@<ip> nvidia@10.201.79.25 -p 1234`)
- B. 这台开发机加静态路由 (需 sudo)
- C. Orin 反向 SSH 到 192.168.0.2
- D. VPN
- E. 用户在 Orin 上手动执行,把日志贴回

---

## 5. 资产清单

### 5.1 框架代码 (`framework/`)

```
framework/
├── capability_schema.py       1A.1  硬件 capability Pydantic 模型
├── config_schema.py           1A.3  Config 数据类 + 序列化
├── constraints.py             1A.4  硬约束 DSL (物理/经验/软)
├── propagation.py             1A.5  D↔B2 双向传播
├── searcher_v0.py             1A.6  最小随机搜索器
├── latency_estimator.py       1A.8  粗略 latency 估计器(待 Stage 1 替换)
└── tests/
    ├── test_constraints.py    9/9 通过
    └── test_propagation.py    5/5 通过
```

### 5.2 配置 (`configs/`)

```
configs/hardware/
├── rtx4090.yaml          Ada sm89, GPU only, FP8 支持, 主开发平台
├── orin_agx.yaml         Ampere sm87, GPU + 2 DLA, 目标部署平台
└── orin_nano.yaml        Ampere sm87, GPU only (无 DLA), 跨硬件迁移目标
```

### 5.3 数据 (`data/phase1/`)

```
data/phase1/
├── baseline_unified.parquet     22 行 (config, AMOTA) 真实数据点
└── baseline_unified.csv         同上,可读版本
```

### 5.3.1 模型 (`models/`) — Stage 3 sanity 产物

```
models/
├── lgb_predictor_v0.txt         LightGBM 模型 (Phase 2 Stage 3.2)
└── lgb_predictor_v0_meta.json   训练 metadata + LOOCV 指标
```

| source | 行数 | AMOTA 区间 | 来源 |
|--------|------|-----------|------|
| 1.1_quant | 5 | 0.338-0.381 | 1.1 量化报告 §二 |
| 1.2_prune | 6 | 0.2366-0.3298 | 1.2 剪枝零微调 Phase B §3.1 |
| 1.2_prune_ft | 7 | 0.2826-0.3356 | 1.2 微调矩阵 §3.2 |
| 1.2_joint | 2 | 0.287-0.360 | 1.2 Phase D.2 联合 §五 |
| 1.2_pareto | 2 | 0.337-0.367 | 1.2 全局 Pareto §8.1 |
| **合计** | **22** | **0.2366-0.3810** | **AMOTA 跨度 0.1444 (信噪比合格)** |

### 5.4 实验结果 (`results/`)

```
results/
├── phase1a_sanity_rtx4090.csv          Phase 1A 50 候选,latency 8-75 ms
├── phase1a_sanity_orin_agx.csv         同上,Orin AGX,18-166 ms
├── phase1a_sanity_orin_nano.csv        同上,Orin Nano,35-333 ms
├── phase2_stage3_sanity.csv            Stage 3 OOF 预测,22 行
├── phase2_stage3_metrics.json          Stage 3 LOOCV 指标
└── phase2_stage3_feat_imp.csv          Stage 3 特征重要性 (gain/split)
```

### 5.5 脚本 (`scripts/`)

```
scripts/phase1/
├── build_baseline_unified.py        1A.7 数据整合脚本
└── run_phase1a_sanity.py            1A.8 端到端 sanity 脚本
scripts/phase2/
└── train_lgb_predictor.py           Stage 3.2/3.3 LightGBM 训练 + LOOCV 评估
```

### 5.6 已建立的 Python 环境

- 环境名: `UniV2X_2.0` (conda)
- Python 3.9.25
- 关键包: pydantic 2.13 / pyarrow / yaml 6.0 / pandas 1.2 / tensorrt 10.13 / torch 2.0.1 / onnx 1.19
- **新装**: lightgbm 4.6.0 / scipy 1.13 / scikit-learn 1.6 (Stage 3 完成)
- **待装**: pymoo (Stage 4) / botorch (Stage 4 可选)

---

## 6. 关键决策记录

### 已确定的决策

| # | 决策 | 选择 | 日期 | 理由 |
|---|------|------|------|------|
| D1 | Phase 1A 是否搜 D 路由轴 | **不搜,锁 GPU only** | 2026-04-27 | 4090 没 DLA,DLA 路由要 Orin 实测验证才有意义 |
| D2 | 数据来源优先级 | **优先用 1.1/1.2 报告中的 22 行真实数据**,Stage 2.5 主动采样补 | 2026-04-27 | 22 行真实数据信噪比合格 (AMOTA 跨度 0.14),先跑通流水线 |
| D3 | 硬约束分类策略 | **物理硬过滤 + 经验硬过滤 + 软作 penalty** | 2026-04-26 (v1.4) | 三类区分清楚,Phase 1B.6 拐点验证后可重新分类 |
| D4 | Config 不可变性 | **frozen=True dataclass** | 2026-04-27 | 防搜索过程中隐式修改,符合 v1.5 编码规范 |

### 待用户确认的决策(影响 Phase 2 启动顺序)

| # | 决策 | 选项 | 默认推荐 | 影响 |
|---|------|------|---------|------|
| D5 | Orin 时间预算 | (a) 0 / (b) 1 周 / (c) 2-3 周 | (b) | (a) 只能算法验证;(b) 够 Phase 1B + Top-K + Stage 5.6;(c) 完整 |
| D6 | 数据采样预算 | (a) 仅现有 22 行 / (b) +30 联合 / (c) +50 联合 | (b) | (a) Stage 3 训不出来;(b) 推荐;(c) +1 周工程量 |
| D7 | 预测器选型 | (a) 纯 LightGBM / (b) APQ Transfer Learning | (a) | (a) 工程量 -1 周;(b) 论文创新点 +1-2 周 |
| D8 | 搜索器是否含 D 路由 | (a) Phase 2 仅 B1+B2 / (b) Phase 2 含 D | (a) | (a) 4090 可完成;(b) D 必须 Orin 重搜 |
| D9 | 是否启动 BoTorch | (a) 不做 / (b) Phase 2 也做 | (a) | 留 Phase 3 |

**默认推荐组合** (基于 4090 主战场 + 1 周 Orin 时间假设): D5(b) / D6(b) / D7(a) / D8(a) / D9(a)

---

## 7. 重要发现

### 7.1 数据相关

- **可挖掘性有限**: 1.1/1.2 实验报告中能直接挖出的 (config, AMOTA) 数据点只有 22 个,远低于 Stage 3 训练需求 (200+)
- **联合配置稀缺**: 现有 1.1 全是"剪枝率=0 的纯量化",1.2 全是"FP32 的纯剪枝",B1×B2 联合空间几乎空白 — Stage 2.5 必须补
- **AMOTA 跨度合格**: 0.2366-0.3810,跨度 0.1444 > 0.05 信噪比阈值

### 7.2 框架相关

- **跨层梯度约束最严**: 75.76% 通过率下,16/66 被拦截的几乎都是跨层剪枝率梯度 > 30% — 这条约束在 Phase 1B.6 必须实测验证(否则可能错过好解)
- **Orin AGX vs Nano 退化路径**: capability YAML 切换后,搜索空间自动收窄(Nano 无 DLA → L1 路由轴退化为只有 GPU),工作流不变
- **双向传播 1-2 次收敛**: 实测 DLA + per-channel 这种典型冲突,1-2 次迭代就稳定,无需复杂收敛检测

### 7.3 平台相关

- **4090 latency 估计 vs Orin 估计 ~2× 关系**: 与频率/带宽差异基本一致 — 但绝对数值不可信,Phase 1B.2 必须实测拟合 latency 映射函数 f
- **网络隔离**: 主开发机 192.168.0.0/24 与 Orin 10.201.79.0/24 无路由,需用户解决(详见阻塞项 B1)

### 7.4 技术选型相关

- **Pydantic v2 严格模式 (extra='forbid') 强制**: 字段拼写错误立刻报错,配置 YAML 里所有字段必须显式列出
- **LightGBM 优先于 MLP**: 数据 < 500 时 GBM 稳过 MLP,且原生支持类别特征,feature_importance 可解释 ✅ Stage 3 验证
- **NSGA-II 优先于 BoTorch**: 预测器评估 ms 级,1000 次评估 ~10 秒,NSGA-II 完全够用;BoTorch 留 Phase 3 实测时再上
- **小数据 LightGBM 死循环陷阱**: `bagging_freq>0` + early stopping + 极小 val 集会让 lgb.train 卡死,Stage 3 实测过. **解决: sklearn `LGBMRegressor`,关闭 bagging,固定 n_estimators**(已写入 train_lgb_predictor.py)

### 7.5 Stage 3 sanity 暴露的关键问题

- **22 行下 GBM 只能学到 1-2 维**: 46/48 特征因方差为零被忽略 — 不是特征工程的问题,**是数据本身没覆盖联合空间**
- **encoder 是当前数据的"信息源"**: `prune_rate__encoder` 和 `prune_criterion__encoder` 是仅有的两个 gain > 0 的特征
- **Stage 2.5 主动采样有了具体方向**: 必须采 backbone/decoder/heads 单独剪枝 + 多模块差异化量化 + 剪枝×量化联合,共 ~30 行(详见 sanity 报告 §四)
- **预测器速度完全够用**: 180μs/cfg,Stage 4 NSGA-II 1000 评估只需 180ms(预算 30 秒,余 99%)

---

## 8. 待启动任务

### 8.1 4090 端可独立启动(不阻塞 Orin)

**Stage 3 sanity 完成后的优先级重排(C 已完成,推荐 B → A)**:

| 优先级 | 路径 | 内容 | 工程量 |
|-------|------|------|-------|
| ~~C~~ | ~~Stage 3 LightGBM 预测器 sanity~~ | ✅ 2026-04-27 完成,Spearman ρ=0.59 | — |
| 🥇 B | **Stage 2.5 主动采样补数据** | Latin Hypercube 采 ~30 配置覆盖多模块差异化 (sanity 报告 §四清单) + 4090 跑 PTQ | 2-3 天(含 4090 跑表 ~15h) |
| 🥈 A | **Stage 1 4090 模块级 LUT 建表** | trtexec 在 4090 上对模块 × 配置笛卡儿积(~240 组合)实测 latency | 1-2 天(含 4090 跑表 6-8h) |
| 🥉 D | **Stage 4 NSGA-II 搜索器骨架** | 用 sanity 预测器先 demo 搜索流程,Stage 2.5 后无缝替换为正式预测器 | 1-2 天 |

**最优路径**: B (主动采样) 与 D (搜索器骨架) **可并行** — 主动采样跑 PTQ 时 4090 GPU 空闲不多,但 NSGA-II 实现是纯 CPU 工作不抢资源.

### 8.2 Orin 端阻塞中的任务

| 任务 ID | 内容 | 估时 | 阻塞原因 |
|--------|------|------|---------|
| 1B.1-1B.7 | Orin 健全性 7 项 (共同 baseline + 映射拟合 + cache 复用 + DLA sanity + 拐点) | 3-5 天 Orin | B1 网络 |
| 2.2 | Orin Top-K LUT 实测 | 2-3 天 Orin | B1 网络 |
| 5.6 | 4090→Orin Pareto 复现率分析 | 1 天 Orin | B1 网络 |

### 8.3 需要用户回答后才能启动

- §6 决策 D5-D9 — 影响 Phase 2 启动顺序与论文 motivation

---

## 9. 更新日志

| 日期 | 内容 |
|------|------|
| **2026-04-27** | 初版,记录 Phase 1A 完成 (8/8 任务,1614 行代码,14/14 单测通过). 诊断 Orin 网络不可达. |
| **2026-04-27 (晚)** | Stage 3 sanity 完成 (LightGBM v0,Spearman ρ=0.59 PASS). 暴露 22 行数据缺口:46/48 特征方差为 0. |
| **2026-04-27 (深夜)** | 路径升级:读 UniAD/BEVFormer-tiny + 李星峰 4.25 报告(R101+DCN 不可剪)→ 升级整体路径为 b2d→UniV2X-tiny 7 阶段. 完整决策清单 v1.3 定型(13 决策, AAAI 8 月中投稿,6 卡资源). 阶段 1 升级为 4 层科学验证. |
| **2026-04-27 (深夜阶段 1 启动)** | **阶段 1 实验初步完成**: <br/> ✓ 下载 uniad_tiny_b2d.pth (873M) + R50 pretrain (98M) + 1 个 sample clip <br/> ✓ 写通用 latency 测量工具 (`tools/measure_latency.py`,P50/P95/模块级) <br/> ✓ **L2 实验跑通 — R50 可剪 90.3%,对比 R101 的 12.5% 提升 7.2×** ([results/phase1_L2_r50_prunability.csv](../../results/phase1_L2_r50_prunability.csv)) <br/> ✓ **L4 实验 3/4 现象复活 PASS** — 敏感度分布、子网唯一性、latency 稳定性都"复活"(p3 对齐拐点 4090 GPU 上看不出,需 Orin) ([results/phase1_L4_r101_failure_replay.json](../../results/phase1_L4_r101_failure_replay.json)) <br/> ⚠ L3 PyTorch eager 上 latency 不稳定(剪 50% 反而慢),需要 TRT engine + 通道对齐 32 才看真实加速 |
| **2026-04-28 (阶段 2 完成)** | **阶段 2 跨域可行性测试 — 路径 Y 选定** (实际 30 分钟完成,原估 1-2 天): <br/> ⚠ **架构不兼容**: UniAD-tiny `cams_embeds.shape=(6,256)` 硬编码 6 cam,V2X-Seq-SPD 车端只 1 cam → **完整 inference 路径 X 死** <br/> ✓ **R50 backbone 跨域可迁移**: 3 张 SPD 真实图像 × 4 层特征 = **12/12 OK**(mean/std/zero% 全合理,无 NaN/Inf) <br/> 🎯 **决定路径 Y**(借 UniAD-tiny 的 R50 权重作 UniV2X-tiny 初始化,节省 ~50% 训练时间) <br/> ✓ V2X-Seq-SPD 数据软链建立(`datasets/V2X-Seq-SPD-New/` → `/data/V2X-Seq/...`) <br/> ✓ 阶段 2 报告: [`阶段2_跨域可行性报告.md`](./阶段2_跨域可行性报告.md) |
| **2026-04-28 (阶段 4 第一步完成)** | **阶段 4 第一步 — UniV2X-tiny config + 权重借用 sanity**: <br/> ✓ **Tiny config 改造完成** (`projects/configs_e2e_univ2x/univ2x_tiny_e2e_track.py`): depth 101→50 / DCN 关 / BEVFormer encoder num_layers 6→3 / style→pytorch / backbone+neck 解冻 / BEV 保留 200×200(seg head 不需要适配) <br/> ✓ **权重借用脚本** (`scripts/phase4/borrow_uniad_tiny_weights.py`): 从 uniad_tiny_b2d.pth 抽 318 个 backbone keys + 14 个 neck keys,复制到车端+路侧端,共 664 keys 写入 `ckpts/univ2x_tiny_init.pth` (215MB) <br/> ✓ **Sanity 验证 7/7 PASS** (`scripts/phase4/L4_3_sanity_check.py`): config 加载 / init ckpt 加载 / **模型 build OK 62.0M 参数** / 权重加载 / **借权重生效**(conv1.weight == init ckpt) <br/> 🎯 **关键指标**: UniV2X-base 261.6M → UniV2X-tiny **62.0M**(**缩减 76.3%**) |
| **2026-04-28 (阶段 4 sanity 训练完成)** | **阶段 4.4 sanity 训练 — 1 epoch 完整跑通**: <br/> ✓ 50 sample 子集 (`data/phase4/spd_train_sanity_50.pkl`),1 epoch 50 iter,**6 分钟跑完** <br/> ✓ **Loss 30.5% 下降**: 151.67 → 105.44 (track cls -35%, map cls -50%, track bbox -23%) <br/> ✓ Memory 21.3GB / iter 7.6s / checkpoint 保存到 `work_dirs/univ2x_tiny_sanity/epoch_1.pth` <br/> ⚠ 中间踩了 3 个坑(canvas_size 硬编码 / aux head 不能简单关 / BEV 缩到 100 与 seg head 不兼容)— 最终保留 BEV 200,只在 backbone+encoder 做 tiny <br/> 🎯 **路径 Y 完全跑通,可进 D4.6 子集→全集渐进训练** |
| **2026-04-28 (sanity ckpt eval)** | **阶段 4.5 sanity ckpt eval — 推理 pipeline 跑通,精度未达可评估**: <br/> ✓ 168 sample val 集**全部推理成功**(0.4 task/s,6 分钟),无 NaN/OOM/shape error <br/> ✓ 推理结果 JSON 生成: `test/univ2x_tiny_eval/.../results_nusc.json` <br/> ⚠ **每 sample 0 detection** — 1 epoch + 50 sample 训练远不足,score 全在阈值下,nuScenes eval 因 box=None 崩 <br/> ✓ **结论**: 推理流水线完全 work,只待**全集训练**(D4.6 b 段:全集 365 sample × 3 epoch,6 卡并行 ~30-60 min)就能拿到可评估精度 |
| **2026-04-27 (晚)** | Stage 3 sanity 完成 (S3.1+S3.2+S3.3+S3.7,共 4/7 任务,S3.4-3.6 数据少跳过). LightGBM LOOCV Spearman ρ=+0.5885 (sanity > 0.50 PASS). 暴露关键发现: 模型只用 2/48 特征 (`prune_rate__encoder` + `prune_criterion__encoder`),46 个特征因方差为零被忽略 — 直接证明 Stage 2.5 主动采样必要性. AccuracyPredictor 接口固化,7/7 单测通过,批量预测 180μs/cfg. 撰写 sanity 报告 ([2.0阶段_phase2_sanity报告.md](./2.0阶段_phase2_sanity报告.md)). 修正 PROGRESS 文件错放进 `精度预测方法实现.搜索算法，精度预测算法实现/` 子目录的问题. |

---

> **下次更新触发条件**: (1) 用户回答 D5-D9 决策 / (2) 启动 Stage 1/2.5/3 任意一项 / (3) Orin 网络打通 / (4) 任一 Stage 任务完成
