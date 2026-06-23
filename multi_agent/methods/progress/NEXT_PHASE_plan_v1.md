# 下一阶段多 agent 协作计划 v1 — 破解"AP 平坦 + 硬件维度过窄"

> 写于 2026-06-03。承接 `HANDOFF_rebuild_figures.md`(61 行 dataset_v2 已出图)。本计划针对本阶段两大根因问题(指标信息量低 + 硬件搜索空间窄)给出**调研 + 实验完整方案**,带文献出处,可直接派发给 sw/hw/data/supervisor 四 agent。

---

## 〇、本阶段问题清单(用户补充 + 主控发现,已整合)

| # | 问题 | 来源 | 本计划对策 |
|---|---|---|---|
| P1 | **AP 轴信息量过低**:front 全程 AP70 跨度仅 0.12,成本轴跨 ~5× → Pareto 退化成成本曲线,搜不出精度-成本 trade-off | 主控+用户 | **双管齐下: ①换指标(优先) ②换模型/数据(其次)** — §一 |
| P2 | 能耗跨硬件口径不统一 | 主控 | **用户已澄清: 不需 4090↔Orin 对比, 只需"特定硬件下搜最优"。降级为非问题, 各平台内自洽即可** — §四 |
| P3 | front 有效独立锚点仅 5 个 | 主控 | **§三 详解为何是 5; 通过换指标让同 5 锚点产出更多有效信号(而非盲目堆架构)** |
| P4 | **硬件维度搜索空间太小** | 用户 | **§二 调研定论: 真维度 = 6 维(非伪维度), 不必转 FPGA** |
| P5 | TRT-auto / workspace / tactic 选择"auto 最优"是否成立 | 用户 | **§二.2 给出判据: 主体 auto 够用, 4 个失效边界可搜** |
| P6 | 单 GPU 流水是否真无法实现, 必须 ≥2 计算单元/FPGA? | 用户 | **§二.1 文献证实: 单 GPU 单模型 conv-dense 并发上界 1.0-1.3×, 必须 ≥2 物理单元** |
| P7 | 是否必须转 FPGA 才能扩搜索维度? | 用户 | **§二.4 明确判断: 不必须, 不推荐; FPGA 仅 related-work** |

---

## 一、问题 P1 — AP 平坦的破局: 换指标(优先) + 换模型/数据(其次)

### 1.1 调研定论(Agent 调研, 全部带出处)
**根因不是 bug, 是 mAP 在过参数化模型上的结构盲区**: mAP 是 0/1 阈值化命中计数, **丢弃了 TP 框的连续定位/尺寸/朝向误差, 也丢弃置信度排序与时序一致性** —— 而这三者恰是剪枝/量化最先损害、mAP 测不到的维度。[Encord; nuScenes devkit; PKL CVPR2020 arXiv:2004.08745]

### 1.2 候选指标优先级(信号强度 × 落地成本)

| 优先级 | 指标 | 为何在 AP 高原处仍有信号 | 实现成本 | 换数据/模型? |
|---|---|---|---|---|
| **P0(最高)** | **NDS 的 TP 误差项 mATE/mASE/mAOE**(平移/尺寸/朝向误差) | mAP 命中后仍计**连续**几何误差; **量化最先伤回归头**(连续值对量化噪声敏感)→ mAOE/mASE 最可能率先单调退化 [nuScenes devkit README] | **最低**: 复用现有检测输出 + DAIR 3D 框 GT 算几何误差, ~半天接评估脚本 | **否, 不换** |
| **P1** | **通信量 AB/BPS(log 刻度)** | V2X 一级轴, 对"传什么特征/压多少"极敏感; 与 latency/energy 正交 [DAIR-V2X arXiv:2204.05575; Where2comm arXiv:2209.12836] | 低-中: 搜索空间须触及融合/特征压缩阶段 | 否(DAIR 原生), 但需搜索触及 fusion |
| **P2** | **AMOTP / IDS / sAMOTA**(跟踪) | 计入连续定位误差 + 时序一致性 + 置信度排序; 剪枝/量化的特征抖动→ID 切换, 单帧 mAP 免疫; 对悬崖最敏感 [Weng IROS2020 arXiv:1907.03961] | 中-高: 需接 AB3DMOT 级跟踪器 + 序列数据 | 需序列数据(**V2X-Seq**); 模型加跟踪后处理(无需重训检测) |
| **不做** | 规划指标 L2/collision/PKL | 理论最贴 AD, 但需规划 head + 自车轨迹 GT, DAIR 单帧不具备, 开环口径不统一且有直行先验偏差 [arXiv:2305.10430] | 高 | 需换 nuScenes 类数据 + 规划 head |

### 1.3 决策
- **第一刀(本阶段必做, P0)**: 在现有 DAIR + 现有 5 锚点上**补算 mATE/mASE/mAOE**。零额外数据/模型, 验证"换指标能否在 AP 平坦处拉出剪枝/量化退化信号"。**这是性价比最高、最可能立刻见效的一招。**
- **第二刀(若融合阶段进入搜索, P1)**: 把**通信量(log 刻度)**设为第二条独立成本轴。
- **第三刀(若前两刀仍平坦, P2)**: 引入 **V2X-Seq 序列数据 + 跟踪指标(AMOTP/IDS)**, 这是"换数据集"路线的落点。
- **换模型**(V2X-ViT 等更难模型, 原 Task#13): 作为与"换指标"并行的另一条腿, 但**优先级低于换指标** —— 先确认是不是指标问题, 再决定是否投入换模型的重训成本。

---

## 二、问题 P4-P7 — 硬件维度: 真搜索空间 = 6 维, 不必转 FPGA

### 2.1 单 GPU 流水(P6): 文献证实上界 1.0-1.3×, 必须 ≥2 物理单元
- **Opara(IEEE TC 2024, arXiv:2312.10351)** 是单 GPU 算子并行 SOTA: 最大 1.68× 但**全部来自多分支网络 + batch=1**(GoogLeNet/NASNet); **BERT 仅 +20%("marginal")**, 因大 GEMM 已占满 SM 无重叠空隙; **batch 一升收益就塌**(Inception batch1=1.41× → batch32=1.09×)。
- → **Pyramid 单路 conv-dense backbone = BERT/大batch 情形, 上界 1.0-1.1×**, 精确解释你实测的 1.08×(3 流)/1.106×(collab2)。
- **MPS/Green Contexts/REEF/HFTA 全是多模型或多进程, 不适用单模型 stage 流水**。RTX 4090 是 Ada 非 Hopper, Green Context runtime SM 切分基本不可用。
- **结论**: 单 GPU 流水/multi-stream/CUDA Graph 并发 **正式降级为消融对照(证伪项)**, 论文里作"为什么需要异构"的动机数据。**真并行只保留异构 GPU∥DLA(Orin 跨进程, 1.34× 实测)**。

### 2.2 TRT-auto 是否最优(P5): 主体够用, 4 个失效边界可搜
- TRT `selectAlgorithms` 是**含 reformat 的全局延迟最优**(不是逐层最快)→ 这正是"per-stage 手工混精被 auto 支配"的根因, **逐层精度/逐层 tactic = 伪维度, 不做**。[TRT Best Practices]
- **4 个 auto 失效边界(此时该手调/外部 autotuner)**:
  1. **时钟漂移**: build 机与 run 机 GPU 时钟不同 → tactic 次优。**搜索必须锁频测**(比"空闲 GPU"更强的要求)。
  2. **workspace 太小 → tactic 池被砍**: `--workspace` 直接限制可选 kernel。**这是真维度**(扫 256MB/1GB/2GB/4GB)。
  3. **tactic sources 被关**(cuBLAS/cuDNN/edge mask)。
  4. **外部 autotuner 可超 TRT**: Ansor 最高 1.7×, Bolt 在 Conv2D 再 2-2.5×(但 10-20min/算子, 工程成本高)→ **仅作"性能上界探针", 不进主循环**。
- **判据**: 标准 conv/GEMM + 目标平台=部署平台 + workspace≥2GB + opt level≥3 + 锁频 → **auto 够用**。跨平台(4090 build→Orin run)必须目标机重 build + 锁频。

### 2.3 固定 GPU 上还能扩什么(P4): 真维度 vs 伪维度

**真维度(进主搜索)**: `INT8 量化边界(calibrator/per-channel/AP 悬崖)` · `2:4 structured sparsity(+finetune)` · `batch(空间/时间)` · `workspace/opt-level/timing-cache`
**Orin 加**: `DLA 路由(FP16; Pyramid 上 DLA INT8 build 全失败)` · `nvpmodel 功耗档(15-40W + Super Mode)`

**伪维度(降为证伪对照, 不做)**: 逐层精度/逐层 tactic(被 auto 支配) · 单 GPU stream 并发(1.08× 已证伪) · CUDA Graph(单模型 1.05-1.10×) · TF32(已在 FP16/INT8 路径) · 内存布局 NHWC/NC32HW32(INT8 时 auto 选)。

> ⚠️ **反直觉警告**: 2:4 sparsity 在小 backbone + batch=1 上**预期仅 1.1-1.3×(不是宣传的 2×), 且必须 finetune 恢复 AP**。[NVIDIA Sparsity blog; arXiv:2104.08378] 别按 2× 规划。

### 2.4 是否必须 FPGA(P7): **不必须, 不推荐, 仅 related-work**
- FPGA 确实提供 GPU/DLA 没有的维度(spatial dataflow / PE 阵列形状 / 片上 buffer / 逐层位宽 / dataflow mapping 搜索), PointPillars 2-bit 仅掉 5-9% AP 缩 16×[arXiv:2007.00493]。
- **但代价数量级更高**(月级开发 / FINN-Vitis HLS / 定点重训 / 时序收敛 / 多数 LiDAR-FPGA 先例仅 ~10 FPS), 会把工程量从天级炸到月级, 挤掉算法贡献; 且与 TRT 搜索框架不兼容需另起炉灶。
- **替代路线(足以撑起有意义、可发表的硬件搜索空间)**:
  `INT8 边界 × 2:4 sparsity(+finetune) × batch × workspace/opt × 异构 GPU∥DLA 路由 × nvpmodel 功耗档` —— 六维横跨五指标 Pareto, 有真平台、低/中成本, INT8 边界与 DLA 路由是**真能移动 Pareto**(E4 实测 INT8 省 30-52% J/frame 佐证)。

---

## 三、问题 P3 — 为何 front 有效独立锚点只有 5 个(详解)

### 3.1 "锚点"的定义: 独立训练的骨干架构
- **一个有效锚点 = 一个 distinct 的 `num_filters`(stage planes)架构, 各自需独立 prune+finetune(~10-20h 训练)**。这是昂贵且独立的单元, 因为 **AP 由架构+权重+val 集决定**, 每个架构携带一份独立的 AP 信号。
- **廉价变体 = 同一架构的下列衍生**(都不产生新的独立 AP 信号):
  - **精度**(FP16/INT8/FP32): 只是一次 TRT build(分钟级), 同权重。
  - **硬件**(4090/Orin): 同权重换平台重测。
  - **batch / workspace / tactic / CUDA Graph**: 同引擎不同测量条件。

### 3.2 为什么是 5(机器可查证)
`df[regime=='front']` 按 (stage0,stage1,stage2) planes 去重 → **5 个 distinct 架构**:
`[16,32,64]` · `[32,64,128]` · `[32,64,136]` · `[48,96,192]` · `[64,128,256]`
- 这 5 个架构在 front 池里展开成 **18 个 planes×精度×硬件 配置**, 再加 ablation 共 **61 行**。
- **行数多 ≠ 独立信息量大**: 61 行里真正独立训练的前沿架构只有 5 个; 其余是同 5 锚点的廉价 build/测量变体 + 消融对照。

### 3.3 全表(含 ablation)的独立训练架构 ≈ 7
若把 pathA 消融的深剪架构也算上: 再加 `[6,13,26]`(prune90) · `[4,6,13]`(prune95)(cliff2_c 与 [16,32,64] 同 planes) → **全表独立训练架构 ≈ 7 个**, front-可部署 = 5 个。这与交接文档"≈6-8"一致。

### 3.4 含义与对策
- **盲目堆架构(再训 10 个 planes)边际收益低** —— 因为 AP 在所有架构上都平(P1 根因)。
- **正确做法: 不堆架构, 而是让同 5 锚点产出更多有效信号** = 换指标(§一, 用 mATE/mASE/mAOE 让同 5 锚点的剪枝/量化退化变得可分辨)。这是 P3 与 P1 的耦合解。

---

## 四、问题 P2 — 跨硬件能耗口径(用户已澄清, 降级)
- 用户澄清: **不需要把 Orin 和 GPU 放一起比, 只需在特定硬件下搜到最优方案**。
- → 能耗"口径不可比"不再是问题。各平台内自洽即可: 4090 用 E5(NVML board), Orin 用 E6(module-total VIN_SYS)。**Pareto 按 hardware 分别搜, 不跨平台对比绝对值**。图上保留跨硬件 panel 仅作"延迟数量级差异"的趋势展示, 不做能耗绝对比较。

---

## 五、下一步多 agent 协作工作方案(可直接派发)

> 原则: 真测就是真测(锁频/空闲 GPU) · 区分真测/估算/复用 · 主控对一切"已测/已修"复跑核验 · 不混 latency 口径。

### Phase R — 调研收口(0.5 day, 已基本完成, data+supervisor 归档)
- **R1[data]**: 把本文 §一/§二 调研结论(带出处)归档进 `background/00_*` 与 `dims_*_v2.md`; 更新 `dims_hardware_v2.md` 的伪维度证伪表。
- **R2[supervisor]**: 核验 §三 锚点计数(复跑 `df[regime=='front']` 去重), 把"5 锚点/7 训练架构"写入 issues_log。

### Phase M — 换指标(P0, 2-3 day, **本阶段最高优先**)
- **M1[sw]**: 实现 **mATE/mASE/mAOE 评估脚本**(复用现有 TRT 检测输出 + DAIR 3D 框 GT 算几何误差, 不重训不换数据)。在现有 5 锚点 × {FP16,INT8} 上复算。
- **M2[data]**: 把 mATE/mASE/mAOE 作为新列接入 `dataset_v2`(扁平列, 沿用"有真数据才加列"); 出图: **"AP 平坦 vs TP 误差项是否有信号"对比图**(同 x=剪枝率/精度, y 分别 AP70 与 mAOE/mASE)。
- **M3[supervisor]**: 判定 —— TP 误差项是否在 AP 高原处出现**单调退化信号**? 若是 → 换指标路线成立, P1 部分解决; 若否 → 升级到 M4。
- **M4[条件触发, sw+data]**: 若 TP 误差项仍平 → 引入 **V2X-Seq 序列数据 + AB3DMOT 跟踪 + AMOTP/IDS**(换数据集路线), 或并行启动**换模型(V2X-ViT)**。

### Phase H — 硬件真维度扩展(P1, 与 M 并行, 3-5 day)
> 主搜索锁定 6 真维度; 全部锁频实测, 标口径。
- **H1[hw]**: **2:4 structured sparsity** on Pyramid backbone(剪枝→2:4 pattern→finetune 恢复 AP→TRT 稀疏 build), 4090 实测延迟/能耗。**按 1.1-1.3× 预期, 不按 2×**。配 AP(+M1 的 TP 误差项)真测。
- **H2[hw]**: **workspace/opt-level 扫描**(256MB/1GB/2GB/4GB × opt 3/5) + timing-cache 复用, 量化 tactic 池变化对延迟的真实影响(预期 5-15%)。锁频。
- **H3[hw]**: **batch 轴**(空间=多 agent 并发 / 时间=积帧)真测吞吐, 确认 throughput 脱离 1/lat 的真实增益(RSU 多车场景)。
- **H4[hw]**: **异构 GPU∥DLA 跨进程流水**补齐真正的跨进程帧交接(共享内存 handoff, +~0.5MB/帧), 把当前"各 stage 独立循环吞吐上界"升级为端到端流水真测。
- **H5[hw]**: **Orin nvpmodel 功耗档**(15/30/40W + Super)扫描, 出各档 (latency, energy) Pareto。
- **H6[hw, 可选 P2]**: TVM/Ansor 对 backbone 做 conv autotuning, **仅作 TRT 性能上界探针**(不进主搜索循环)。

### Phase D — 数据整合 + 框架闭环(P1, data+supervisor, 贯穿)
- **D1[data]**: H1-H5 真测点接入 `dataset_v2`, 每点标 latency_kind/regime/ap_reuse_basis/指标口径。
- **D2[data]**: 重出主图(成本 Pareto + 耦合陷阱 + 新指标信号图 + 硬件真维度覆盖)。
- **D3[supervisor]**: 复跑核验所有 H/M 自报真测; 检查口径隔离; 把"伪维度证伪"与"真维度增益"对比落盘。
- **D4[data]**: 用扩展后的 dataset 重训预测器, 检验 **(AP/新指标, latency, throughput, energy, size) 5 轴是否变得可学习**(新指标是否让精度轴重新携带信号)。

### 里程碑判据
1. **M3 通过** = 换指标在 AP 平坦处拉出信号 → P1/P3 解决, 不必盲目换模型。
2. **H1-H5 完成** = 硬件真 6 维有真测 Pareto → P4 解决, FPGA 留 related-work。
3. **D4 通过** = 5 轴预测器可学习 → 协同搜索框架成立。

---

## 六、出处汇总(关键)
- 指标: PKL CVPR2020 [arXiv:2004.08745] · nuScenes NDS devkit · AMOTA/sAMOTA IROS2020 [arXiv:1907.03961] · DAIR-V2X CVPR2022 [arXiv:2204.05575] · V2X-Seq CVPR2023 [arXiv:2305.05938] · Where2comm NeurIPS2022 [arXiv:2209.12836]
- 硬件: Opara IEEE-TC2024 [arXiv:2312.10351] · TRT Best Practices(NVIDIA docs) · Ansor OSDI2020 · Bolt [arXiv:2110.15238] · NVIDIA 2:4 Sparsity blog · [arXiv:2104.08378] · JetPack 6.2 Super Mode · PointPillars-FPGA [arXiv:2007.00493]
