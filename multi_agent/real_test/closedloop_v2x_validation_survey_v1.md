# 闭环仿真验证 V2X 有效性 — 方法调研 v1

> 编制: sw-optimizer, 2026-06-05  
> 任务: Task #8 [P0-sw]  
> 核心假设: **路侧 e2e 推理 < 200ms 时，V2X 对自驾精度/安全有可证明的提升**  
> 纪律: 每条文献声明带可核 URL/arXiv 号；禁凭记忆引数字；不确定标【待核】  
> 待 supervisor 逐引核验

---

## 一、核心假设的评测框架

用户假设包含**两个可独立验证的子命题**:

| 子命题 | 验证要求 |
|--------|----------|
| A. 路侧 e2e < 200ms 可达 | 真测延迟 (Task #9 同步分析); 与本文§四互补 |
| B. V2X 在可达延迟下对驾驶有可证明提升 | 需闭环仿真; 评测轴: 碰撞率↓ / 驾驶分↑ / 安全余量↑ |

子命题 B 要求**闭环仿真**（非离线 AP 评测）才能回答，这是本文的核心范围。

**方法谱系概览**:

```
评测方式
├── 离线感知评测 (AP / sAP)       ← 现有 DAIR 实验所在层
│   ├── 标准 AP (static)
│   └── 流式 sAP (含延迟惩罚)      ← §三.1–3.2 详述
│
└── 闭环驾驶评测                    ← 本文核心
    ├── 感知+规控闭环 (CARLA-based)  ← V2XVerse / OpenCDA (§二.1–2.2)
    └── 规控闭环 (log-replay)         ← nuPlan / Bench2Drive (§二.3–2.4)
```

---

## 二、闭环仿真平台谱系

### 2.1 V2XVerse + CoDriving ⭐⭐⭐【最高优先级】

| 属性 | 详情 |
|------|------|
| **arXiv** | 2404.09496 |
| **标题** | Towards Collaborative Autonomous Driving: Simulation Platform and End-to-End System |
| **作者** | Liu et al. (CollaborativePerception 组) |
| **发表** | IEEE T-PAMI (accepted 2025); preprint 2024-04 |
| **代码** | https://github.com/CollaborativePerception/V2Xverse |
| **仿真引擎** | CARLA (Town05), 67 test routes, 100s scenario trigger points |

**平台能力**:
- ✅ **V2X 多端协同**: 支持多智能体 (CAV + RSU) 协同感知, 已验证在 CARLA 中运行
- ✅ **驾驶指标**: 明确报告 **驾驶分 (Driving Score)** 和 **行人碰撞率 (Pedestrian Collision Rate)**
- ✅ **闭环评测**: 在线闭环, 平台"enables online closed-loop evaluation by supporting system deployment and driving performance evaluation"（来源: 搜索结果引用 V2XVerse 描述）
- ⚠️ **延迟注入**: 论文提及"dynamic constraint communication conditions"下 CoDriving 性能优越，但具体注入的延迟毫秒数在摘要/官网中**未明确列出**【待核: 读全文 §4.x 实验节确认延迟范围】
- ✅ **路由完成**: 来源官网确认提供 route completion 相关评测

**关键定量结果** (来源: https://collaborativeperception.github.io/V2Xverse/):
- CoDriving vs. SOTA ego-only: **驾驶分提升 +62.49%**
- **行人碰撞率降低 -53.50%**
- 在"dynamic constraint communication conditions"下仍保持优势

**V2X vs ego-only 对照**: CoDriving 核心实验直接是"有 V2X 通信 vs 单车感知"对比; 典型场景 = 遮挡下仅靠路侧视角才能规避行人碰撞.

**与我们资产的适配性**:
- 需要将 HEAL/Pyramid 感知输出接入 CoDriving 驾驶策略 (非 perception-only)
- 适配成本**高** (需要完整 e2e 驾驶策略模块; HEAL 当前只有感知头, 无 planning)
- ✅ 数据格式: CARLA 点云/图像, 可能与 OPV2V 格式兼容
- ⚠️ DAIR-V2X 数据直接喂入: **不可** (V2XVerse 是仿真数据, DAIR 是真实数据, 不可互用)

---

### 2.2 OpenCDA + CARLA ⭐⭐

| 属性 | 详情 |
|------|------|
| **arXiv** | 2107.06260 |
| **标题** | OpenCDA: An Open Cooperative Driving Automation Framework Integrated with Co-Simulation |
| **作者** | Xu et al. (UCLA Mobility) |
| **发表** | IEEE ITSC 2021 |
| **代码** | https://github.com/ucla-mobility/OpenCDA |
| **文档** | https://opencda-documentation.readthedocs.io/en/latest/ |

**平台能力**:
- ✅ **仿真引擎**: CARLA + SUMO 联合仿真 (场景+交通流)
- ✅ **全栈系统**: perception / localization / planning / control / V2X 通信模块均实现
- ✅ **V2X 通信**: 支持 SAE J3216 标准 V2X; 每辆 CAV 交换位置/意图/感知上下文
- ⚠️ **延迟注入**: 搜索结果提及"transmission delays inherent in V2X communication can be simulated"，但官方文档中**具体延迟注入 API 待核**【待核: 检查 opencda-documentation.readthedocs.io V2X communication 模块文档】
- ✅ **驾驶指标**: 支持 collision rate, route completion (来源: 平台设计描述)
- ✅ **模块化**: 高度可替换, 可将感知模块换为 HEAL/Pyramid

**与我们资产适配性**:
- 感知模块可替换性高: 可用 HEAL 输出替换默认 OpenCDA 感知
- 需要 CARLA 场景, 与 DAIR 真实数据不互用
- 适配成本**中等** (感知替换比 V2XVerse 更模块化)
- ⚠️ 闭环驾驶策略已内置, 但 V2X 协同效果实验不如 V2XVerse 丰富

---

### 2.3 Bench2Drive ⭐⭐

| 属性 | 详情 |
|------|------|
| **arXiv** | 2406.03877 |
| **标题** | Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving |
| **发表** | NeurIPS 2024 (Datasets & Benchmarks Track) |
| **代码/数据** | https://arxiv.org/abs/2406.03877 |

**平台能力**:
- ✅ **高质量闭环**: CARLA-based, 44 交互场景 × 5 weather = 220 routes; 2M 训练帧
- ✅ **驾驶指标**: 综合驾驶分、碰撞率、路线完成率
- ❌ **V2X 支持**: **无** (单智能体 e2e-AD 评测, 无 RSU/CAV 协同机制)
- ❌ **延迟注入**: 无内置 V2X 延迟注入

**评价**: 是目前最权威的闭环 e2e-AD benchmark, 但**不支持 V2X**, 只能作为 ego-only 基准参照.

---

### 2.4 nuPlan / nuPlan-R ⭐

| 属性 | 详情 |
|------|------|
| **arXiv** | 2106.11810 (nuPlan 原始); 2511.10403 (nuPlan-R) |
| **发表** | nuPlan: NeurIPS 2021 Workshop; nuPlan-R: 2025-11 (arXiv submitted 2025-11-13) |

**平台能力**:
- ✅ **闭环规划**: 真实驾驶数据 1500h; 支持 reactive simulation (nuPlan-R)
- ❌ **V2X 支持**: **无** (规划 benchmark, 假设感知已完成)
- ❌ **V2X 通信/延迟注入**: 无

**评价**: 优秀的闭环规划 benchmark, 但完全**不涉及 V2X 感知协同**, 不适用本研究.

---

### 2.5 CARLA Leaderboard 2.0 ⭐

**平台能力**:
- ✅ 严苛的单车闭环 benchmark; Town05 等多场景
- ❌ **V2X 支持**: 无 (SENSORS track + MAP track, 均单智能体)
- 来源: arXiv:2412.09602, autonomousvision/carla_garage GitHub

**评价**: 最具挑战性的单车闭环 benchmark, 但**无 V2X 机制**, 只适合 ego-only 对照实验.

---

### 平台对比汇总

| 平台 | V2X | 延迟注入 | 驾驶指标 | HEAL适配 | 优先级 |
|------|-----|---------|---------|---------|--------|
| V2XVerse+CoDriving | ✅ V2I/V2V | ⚠️【待核ms数】| DS+碰撞率 | 高成本 | ★★★ |
| OpenCDA+CARLA | ✅ V2X全栈 | ⚠️【待核API】| 支持 | 中等 | ★★ |
| Bench2Drive | ❌ | ❌ | ✅完整 | N/A | ★★(ego-only基准) |
| nuPlan-R | ❌ | ❌ | ✅规划 | N/A | ★(规划层) |
| CARLA LB2.0 | ❌ | ❌ | ✅ | N/A | ★(ego-only基准) |

---

## 三、时延感知评测方法

### 3.1 sAP — Streaming Average Precision (基础框架)

| 属性 | 详情 |
|------|------|
| **arXiv** | 2005.10420 |
| **标题** | Towards Streaming Perception |
| **作者** | Mengtian Li, Yuning Chai, Deva Ramanan |
| **发表** | ECCV 2020 |
| **代码** | https://github.com/mtli/sAP |

**核心思想**: 在每个时间步 $t$，评测系统当时**最新可用**的预测结果。若推理未完成，则沿用上一帧结果（可能已过时）。将延迟惩罚自然嵌入 AP 计算，称为 **streaming AP (sAP)**。

**关键公式原理**: $\text{sAP} \triangleq \mathbb{E}_t[\text{AP}(\hat{y}_{t^*}(t), y_t)]$，其中 $t^* \leq t$ 是最近完成推理的帧。

**对我们的意义**:
- 提供了"延迟 × AP"联合评测的理论框架
- 若将 Pyramid/V2X-ViT 推理延迟注入 sAP 框架, 可量化: 推理加速（INT8/剪枝）对流式感知精度的提升
- **直接适配性**: sAP 是**离线感知评测**, 不需要闭环; 可在 DAIR-V2X 上复现

---

### 3.2 ASAP — Autonomous-driving StreAming Perception Benchmark

| 属性 | 详情 |
|------|------|
| **arXiv** | 2212.08914 |
| **标题** | Are We Ready for Vision-Centric Driving Streaming Perception? The ASAP Benchmark |
| **作者** | Wang et al. |
| **发表** | CVPR 2023 |
| **代码** | https://github.com/jeffwang987/asap |

**特点**:
- nuScenes-based; 2Hz → 12Hz 标注扩展管线
- SPUR (Streaming Perception Under constRained-computation) 评测协议: 不同算力预算下的流式性能
- 多任务: 3D detection / semantic map / motion forecasting / depth estimation

**对我们的意义**:
- 与我们的 LiDAR-based 3D 检测任务直接相关
- SPUR 协议可以直接对应"不同优化档 (FP32/FP16/INT8/剪枝) 的算力预算"
- **局限**: nuScenes 非 V2X, 无协同感知; 但框架可迁移至 DAIR-V2X

---

### 3.3 SyncNet — 首个时延感知协同感知系统

| 属性 | 详情 |
|------|------|
| **arXiv** | 2207.08560 |
| **标题** | Latency-Aware Collaborative Perception |
| **作者** | Zixing Lei, Shunli Ren, Yue Hu, Wenjun Zhang, Siheng Chen |
| **发表** | ECCV 2022 |
| **代码** | https://github.com/MediaBrain-SJTU/SyncNet |

**核心方法**:
1. **Dual-branch pyramid LSTM**: 从历史帧估计当前时刻特征 (feature-level time prediction)
2. **Time modulation**: 将时间戳差 $\Delta t$ 编码为注意力权重, 平衡估计特征与实际异步特征

**关键结果** (来源: arXiv 摘要):
- 在通信延迟场景下比 SOTA 协同感知方法提升 **+15.6%**
- 在严重延迟下仍优于单车感知

**对我们的意义**:
- 首次在协同感知中直接建模通信延迟
- 主要用 V2V (OPV2V-based) 数据; **DAIR-V2X (V2I) 直接适配性【待核: 检查论文是否包含 DAIR-V2X 实验】**
- SyncNet 框架可作为"延迟感知版 HEAL"的参考实现

---

### 3.4 CoBEVFlow — 异步鲁棒 BEV 流对齐

| 属性 | 详情 |
|------|------|
| **arXiv** | 2309.16940 |
| **标题** | Asynchrony-Robust Collaborative Perception via Bird's Eye View Flow |
| **作者** | Sizhe Wei et al. |
| **发表** | NeurIPS 2023 |
| **项目页** | https://sizhewei.github.io/projects/cobevflow/ |

**核心方法**: 用 BEV flow map 预测各 grid cell 的运动向量, 将异步时刻的 BEV 特征"搬运"对齐到接收方当前时刻。

**关键结果**:
- **测试延迟范围: 0–500ms** (预期时间间隔 100/300/500ms) — 来源: https://sizhewei.github.io/projects/cobevflow/
- **数据集**: IRV2V (合成, 各种异步度) + **DAIR-V2X (真实世界)** ✅ — 来源: arXiv:2309.16940 摘要
- 在预期延迟 500ms 时, 比 baseline 提升 **>18.9%** — ⚠️【待核: 引源未在项目页/摘要文字中找到该数字, 需读全文实验节 Table/正文确认】
- 即使通信量更小, 仍超越其他方法 — 来源: https://sizhewei.github.io/projects/cobevflow/
- ⚠️ AP50 具体数值: 项目页仅显示图表, **具体 AP50 数字未在文字中列出**【待核: 读全文表格 Table 2/3】

**对我们的意义**:
- **直接使用 DAIR-V2X 数据集** ✅ — 与我们的实验资产完全兼容
- 延迟范围 0–500ms 覆盖真实 V2X 场景 (我们实测 e2e 远低于此范围)
- 可以将我们测量的真实延迟 (Pyramid m1: body TRT ~1ms, e2e ~10-30ms) 作为延迟注入的基础
- **最低成本复现路线**: 在 CoBEVFlow 框架上注入我们的真实测量延迟, 对比有/无补偿

---

### 时延评测方法对比

| 方法 | 类型 | 延迟建模 | DAIR-V2X | V2X协同 | 适配成本 |
|------|------|---------|---------|---------|---------|
| sAP (ECCV 2020) | 离线感知 | 隐式 (用旧结果) | 可迁移 | 无 | 低 |
| ASAP (CVPR 2023) | 离线感知 | SPUR协议 | 可迁移 | 无 | 中 |
| SyncNet (ECCV 2022) | 协同感知 | 显式特征同步 | 【待核】 | V2V | 中 |
| CoBEVFlow (NeurIPS 2023) | 协同感知 | BEV flow 对齐 | ✅ **直接支持** | V2I+V2V | 低 |

---

## 四、与我们资产的适配性

### 4.1 当前实测延迟基线 (来源: 本项目结果文件)

以下延迟均为**真测值**，口径标注：

| 指标 | 数值 | 口径 | 来源 |
|------|------|------|------|
| Pyramid m1 body (TRT FP16) | 1.269 ms | `body_subnet_collab2` | `multi_agent/model/pyramid_lidar_structure_audit_v1.md` |
| Pyramid m1 body (TRT INT8) | 0.809 ms | `body_subnet_collab2` | 同上 |
| 引擎完整推理 (TRT FP16) | 2.47 ms | engine-only benchmark | `results/e2e_engine_speed_bench.json` |
| 引擎完整推理 (TRT INT8) | 1.91 ms | engine-only benchmark | `results/e2e_engine_speed_bench.json` |
| 旧版全 e2e (PyTorch FP16) | 26.70 ms | 含 NMS, CPU; ⚠️ **OPV2V 口径**, DAIR e2e 待 Task#9/真测 | CLAUDE.md §〇 M4.6.0 |
| CUDA NMS 加速比 | 3.04× | vs CPU NMS | CLAUDE.md §〇 |
| NMS 占旧 e2e 比例 | 72.4% | PyTorch FP16 口径 | CLAUDE.md §〇 |

**粗估优化后全 e2e 路侧推理** (⚠️ OPV2V 口径外推, 仅供量级参考):
- 3.04× 是**整体 e2e 加速比** → CUDA-NMS 后整体 ≈ 26.70 / 3.04 ≈ 8.8ms
- 其中非-NMS 部分 (含旧 PyTorch body + 预处理): 26.70 × (1 − 72.4%) ≈ 7.4ms
- NMS 残余 ≈ 8.8 − 7.4 ≈ 1.4ms
- TRT INT8 引擎 (替换 PyTorch body): 1.91ms (节省 ~5.5ms vs PyTorch body)
- 点云预处理 (voxelize/PointPillar sparse): 【待核, 估 2-4ms】
- **粗估 e2e: ~5-9ms** (4090 GPU, TRT INT8 + CUDA NMS)
- **与 200ms 比较**: 远低于 200ms 阈值, 剩余 ~191-195ms 可用于 V2X 通信

> ⚠️ 上述"粗估 e2e"是基于各模块独立测量的加法，非端到端真测。端到端真测需要串联管线实跑。

### 4.2 Orin AGX 边缘端延迟 (来源: 项目实测)

| 指标 | 数值 | 口径 | 来源 |
|------|------|------|------|
| Orin DLA0 (stage01) | 14.49 ms | DLA pipeline | `results/E3_orin_dla_pipeline.csv` |
| Orin DLA1 (stage2) | 6.18 ms | DLA pipeline | 同上 |
| Orin 双进程 DLA 并行 | 15.4 ms | 多进程异构 | 同上 |

**Orin 端 e2e 粗估**: ~30-50ms (含预处理+NMS)【待核: Task #9 hw-optimizer 分析】

### 4.3 逐平台适配性评估

#### CoBEVFlow (最高适配性) ✅
- ✅ 直接使用 DAIR-V2X 数据集
- ✅ V2I 场景与我们的 RSU 部署设定一致
- ✅ 延迟注入: 可将我们的真实测量延迟 (~10ms e2e) 作为注入基准
- ✅ 离线评测, 无需 CARLA
- 成本: 需要实现 BEV flow 补偿与 HEAL backbone 的集成

#### V2XVerse (闭环价值最高, 但成本高) ⚠️
- ✅ 唯一支持 V2X + 闭环驾驶评测的平台
- ❌ 需要完整驾驶策略模块 (planning/control), HEAL 只有感知
- ❌ CARLA 环境配置开销
- 建议: 先用 CoBEVFlow 离线证明感知提升, 再考虑 V2XVerse 闭环验证

#### sAP/ASAP (最低成本) ✅
- ✅ 完全离线, DAIR-V2X 可直接用
- 成本极低: 将推理时间注入 sAP 公式即可
- 局限: 不是闭环驾驶评测, 只能证明"感知精度提升"

#### OpenCDA (中等) ⚠️
- 感知模块可替换但需要 CARLA 环境
- 不如 V2XVerse 的 V2X 评测结果丰富

---

## 五、推荐最小可行实验路线

> ⚠️ **授权声明**: MVE-1 和 MVE-2 均为新实验，**启动须 team-lead/用户明确授权原文后方可执行** (MUST-6)。本节仅作方案规划，不代表已获授权。

### MVE-1: 延迟注入 × AP (离线, 最低成本) — 2-3天

**目标**: 证明在真实测量延迟下, V2X 协同感知精度优于 ego-only, 且延迟补偿（CoBEVFlow 风格）可恢复精度。

**方法**:
1. 使用 DAIR-V2X val set (1789 samples, 已有)
2. 选取延迟注入档: 0ms (同步基线) / 50ms / 100ms / 200ms / 500ms
3. 三条曲线对比:
   - **A**: ego-only (无 V2X, AP@50 随延迟无变化)
   - **B**: V2X 无补偿 (直接用异步特征, AP 随延迟下降)
   - **C**: V2X + 延迟补偿 (CoBEVFlow BEV flow 对齐)
4. x轴 = 注入延迟(ms), y轴 = AP50/AP70

**预期曲线形态**: 注入延迟增大时, B 曲线 AP 下降 > A (ego-only 无变化), C 曲线 AP 高于 B 且接近 A。在我们的实测延迟范围 (~10-30ms) 内, B ≈ A (延迟小, 影响有限), 验证"延迟达标时 V2X 安全有效"。

**资产**: DAIR-V2X val, Pyramid m1 TRT ckpt, CoBEVFlow 代码 (开源)  
**GPU 需求**: 需要模型推理 (需宣告)  
**关键数字**: CoBEVFlow 在 DAIR-V2X 上的 AP 数值 (论文 Table 2, 需读全文)

---

### MVE-2: V2XVerse 闭环对照 (高价值, 高成本) — 2-3周

**目标**: 端到端证明 V2X + 延迟优化对驾驶安全的改善（驾驶分、碰撞率）。

**方法**:
1. V2XVerse 平台 (CARLA Town05)
2. 将我们的 TRT 优化感知模型接入 CoDriving 框架替换默认感知
3. latency sweep: 模拟 {0ms, 100ms, 200ms, 500ms} 通信延迟
4. 对照组: ego-only (无 RSU)
5. 评测: Driving Score / Pedestrian Collision Rate / Route Completion

**预期结果**: V2X 在延迟 < 200ms 时驾驶分显著高于 ego-only (基于 +62.49% 的 CoDriving 基准); 超过某阈值后效益下降。

**风险**:
- CARLA 环境配置 / HEAL 与 CoDriving 接口适配 (非 trivial)
- 建议先完成 MVE-1 验证方向性, 再决定是否启动 MVE-2

---

### MVE 选择建议

| | MVE-1 (离线AP) | MVE-2 (闭环DS) |
|--|--|--|
| **论文诉求** | 感知提升证明 | 驾驶安全提升证明 |
| **成本** | ~2-3天 | ~2-3周 |
| **资产利用率** | 高 (DAIR直用) | 中 (需CARLA) |
| **可信度** | 中 (离线,无控制闭环) | 高 (闭环,真驾驶) |
| **建议** | ✅ 先做 | 视资源决定 |

---

## 六、结论与未解问题

### 已确认的结论
1. **V2XVerse + CoDriving** 是目前唯一成熟的 V2X+闭环驾驶仿真平台 (arXiv:2404.09496, T-PAMI)
2. **CoBEVFlow** (arXiv:2309.16940, NeurIPS 2023) 直接支持 DAIR-V2X, 测试延迟 0–500ms, 是最低成本验证路线
3. **sAP/ASAP** 提供成熟的离线"延迟×AP"联合评测框架
4. 我们的 Pyramid m1 TRT e2e 粗估 **~5-9ms** (OPV2V 口径外推), **远低于 200ms 阈值**, 子命题 A 初步成立（需 Task #9 DAIR 口径精确测量确认）
5. V2X 在 CARLA 闭环中 DS +62.49% / 碰撞率 -53.50% (CoDriving 实测, 信息源: 官网)

### 待核/未解问题 (supervisor 逐引核验用)
1. 【待核 §2.1】V2XVerse 具体延迟注入毫秒数: 读 arXiv:2404.09496 全文 §4 实验节
2. 【待核 §2.2】OpenCDA 延迟注入 API: 读 opencda-documentation.readthedocs.io 通信模块文档
3. 【待核 §3.3】SyncNet 是否包含 DAIR-V2X 实验: 读 arXiv:2207.08560 全文
4. 【待核 §3.4】CoBEVFlow DAIR-V2X 具体 AP50/AP70 数值: 读全文 Table 2/3
5. 【待核 §4.1】Pyramid m1 e2e 粗估需端到端真测确认 (与 Task #9 协同)

---

## 附录: 参考文献索引

| 编号 | arXiv | 标题 | 发表 |
|------|-------|------|------|
| [1] | 2404.09496 | Towards Collaborative Autonomous Driving: Simulation Platform and End-to-End System | IEEE T-PAMI, 2025 |
| [2] | 2107.06260 | OpenCDA: An Open Cooperative Driving Automation Framework Integrated with Co-Simulation | IEEE ITSC, 2021 |
| [3] | 2406.03877 | Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop E2E Autonomous Driving | NeurIPS 2024 |
| [4] | 2106.11810 | nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles | NeurIPS 2021 Workshop |
| [5] | 2511.10403 | nuPlan-R: Closed-Loop Planning Benchmark via Reactive Multi-Agent Simulation | 2025-11 |
| [6] | 2005.10420 | Towards Streaming Perception | ECCV 2020 |
| [7] | 2212.08914 | Are We Ready for Vision-Centric Driving Streaming Perception? The ASAP Benchmark | CVPR 2023 |
| [8] | 2207.08560 | Latency-Aware Collaborative Perception (SyncNet) | ECCV 2022 |
| [9] | 2309.16940 | Asynchrony-Robust Collaborative Perception via BEV Flow (CoBEVFlow) | NeurIPS 2023 |
