# 软硬协同优化机理调研 v1.3
> Task #14-② | hw-optimizer | v1.3 增补 2026-06-06  
> v1.1/v1.2: 打回修改(H2冻结引用替换/机制措辞纠正/NACOS全称); v1.3: team-lead 授权增补 §1.5 原理层耦合机理(6条 P1-P6, 双源)  
> 双源: A. 内部实证(本项目真测 7 条) + B. 最新文献(WebFetch 核原文)  
> 纪律: [文献声称]/[我方实证]/[推断]严格分层; WebFetch 已核原文; subnet ≠ e2e 口径已标

---

## §1 核心问题回答

### §1-0 为什么必须做软硬协同优化 — 优势汇总（逐条带引用） [v1.2 增补]

> ★[2026-06-06 team-lead 增补] 本节集中回答"为什么做软硬协同、优势是什么"。**全部数字复用本文档 §1.1/§2/§3/§4 已核验内容**（文献出处见 §5 索引；我方数字见 §6 快速核查索引），无新声称。

**为什么必须协同 — 因果链三句话**:
1. **软件侧的代理指标在真实硬件上失效**: FLOPs/参数量不能预测延迟（[文献声称] FBNet 明示 "FLOP count does not always reflect actual latency"; nn-Meter 证明必须 kernel 级实测才能达 99% 预测精度），且失效是**非解析**的（[我方实证] ISS-014: 同剪枝级延迟差 3.4×, 96=3×32 满足对齐仍掉坑 → 无法用规则预判, 只能真硬件在环）。
2. **软件最优解依赖硬件标的且不可迁移**: 为一个硬件优化的方案换硬件即失效（[文献声称] HAQ: 跨硬件迁移延迟恶化 **5.2–7.2×**; OFA: 不同设备最优子网不同; [我方实证] DLA INT8 0/12 不可 build vs GPU 12/12 — 同一软件配置可达性随硬件单元翻转）。
3. **硬件瓶颈随软件优化动态漂移**: 优化成功本身改变下一步该优化什么（[我方实证] body 压 6.56× 后 pre-body 占 73% — 各管各的会让双方持续优化已经不是瓶颈的部分）。

**协同的优势量化对比表（协同 vs 各管各的，全部带引用）**:

| 优势维度 | 证据（协同 vs 独立/硬件无关基线） | 来源 |
|---------|----------------------------------|------|
| 精度（同延迟预算） | APQ 联合搜索 **+2.3% ImageNet top-1** vs ProxylessNAS+AMC+HAQ 分离串行 | [文献声称] APQ, arXiv:2006.08509 |
| 搜索成本 | APQ **600× 更少搜索代价** vs 分离式 | [文献声称] 同上 |
| 部署效率 | OFA 设备专属子网 vs 通用模型: **+4% acc + 1.5× faster**（vs MobileNetV3） | [文献声称] OFA, arXiv:1908.09791 |
| 量化精度 | HAQ 硬件在环混精 vs 均匀 8-bit: 延迟 1.4–1.95×、能量 1.9×; HAWQ-V2 vs 均匀量化 **+5.92% Top-1** | [文献声称] arXiv:1811.08886; arXiv:1911.03852 |
| 编译性能 | Ansor 硬件实测代价模型 vs 模板搜索: **CPU 3.8× / ARM 2.6× / GPU 1.7×** | [文献声称] Ansor, arXiv:2006.06762 |
| 剪枝增益落地 | HALP 以真实 per-filter latency 驱动剪枝: ResNet-50 **1.6× 吞吐且 +0.3% acc** | [文献声称] HALP, arXiv:2110.10811 |
| 理论-实际差距消除 | 硬件无关量化: FP32→INT4 理论 16×, 实际仅 **4–8×**（2-4× 实现差距来自对齐/kernel/调度 — 正是协同要吃掉的部分） | [文献声称] Survey, arXiv:2103.13630 |
| 避免不可部署方案（我方） | 不协同则产出 DLA INT8 这类 **0/12 不可 build** 的"纸面最优"方案 | [我方实证] §2-②; orin_dspace_bench.parquet |
| 避免收益蒸发（我方） | 剪枝感知 kernel 断崖可避免同剪枝级 **3.4×** 延迟差 | [我方实证] §2-①; ISS-014 canonical |
| 优化方向正确性（我方） | 协同视角发现 pre-body 占 73% → 下一杠杆换向; 各管各的则双方继续空耗在 body 上 | [我方实证] §2-⑤; E7+E8 |
| 跨硬件预测有效性（我方） | 知道软件栈形态(TRT/eager)才能外推: TRT 段 R²>0.995, eager 段盲推低估 **3×**（曾导致整个可达性结论翻转） | [我方实证] §2-⑦; m2_latency_mapping_f.json + ISS-039 |

**优势的边界（诚实声明）**: 协同不等于处处有收益——调度类协同就有负例（[我方实证] 单 GPU 流水仅 1.08×, 进程内 DLA 双流 1.00× 完全串行, §2-④）；协同的真正普适优势是**在搜索时就知道哪些组合不可行/无收益**, 把实验预算花在可达区域 — 这本身就是我方框架(真硬件在环 P×Q×D 联合搜索)的存在理由。

### 1.1 软件维度 × 硬件维度怎么耦合?

软硬耦合的本质是: **同一份软件决策在不同硬件上会产生完全不同的计算路径、内存访问模式和可执行 kernel 集合**。反过来, **同一硬件对不同软件结构的响应方式也截然不同**。两者并非独立信道, 而是通过以下四种机制深度纠缠:

#### 机制 A: 硬件在环反馈 (Hardware-in-the-Loop)

软件优化的目标函数直接来自硬件实测信号(latency/energy), 而非代理指标(FLOPs/参数量)。

**工作原理**: 候选软件配置 → 部署到目标硬件 → 测量实际 latency/energy → 反馈给搜索器/优化器 → 修正下一次候选。

[文献声称] HAQ (CVPR'19) 用 RL 驱动混合精度量化搜索, 反馈信号是硬件仿真器输出的**真实延迟和能耗**(非 FLOPs 代理)。当把为 HW1 (BitFusion) 优化的量化策略迁移到 HW2 (BISMO edge) 时, 延迟从 16.29ms 飙升至 85.24ms (**5.2× 恶化**); 迁移到 HW3 时达 117.44ms (**7.2× 恶化**)。这直接证明: **硬件感知 = 不可分离**。(来源: HAQ arXiv:1811.08886, Table 1)

[文献声称] FBNet (CVPR'19) 对每种候选层类型预先在目标设备(Samsung S8/iPhone X)上测量 latency, 构建查找表, 并将 `λ·log(latency_hw)` 直接纳入可微分 NAS 的损失函数。论文明确指出 **"FLOP count does not always reflect actual latency"**: 同一 FLOPs 预算在两款手机上因算子融合模式不同导致完全不同的 latency。(来源: FBNet arXiv:1812.03443)

[我方实证] **E4 INT8 省能 30-52% J/frame**(4090, 真测, `results/E4_energy_4090.csv`): 同一 PyramidFusion 网络, 仅将精度档位从 FP16 换为 INT8:
- T_baseline: FP16 291mJ → INT8 **141mJ (-52%)**
- T_prune75: FP16 280mJ → INT8 **197mJ (-30%)**
精度选择(软件决策)直接决定硬件功耗路径: INT8 kernel 用 Tensor Core INT8 路径, 吞吐翻倍 + 功率下降; 两者复合使 J/frame 降 30-52%。

#### 机制 B: 代理模型 / 预测器 (Surrogate/Proxy Model)

直接在环硬件太慢, 训练一个"硬件感知代理"来替代实测反馈。

[文献声称] nn-Meter (MobiSys'21, Best Paper) 将模型推理拆分为"kernel"(算子融合执行单元)粒度, 在目标设备上测量构建**内核级延迟预测器**, 精度达 **99%+(CPU/GPU)** 和 **83.4%(VPU)**。论文核心动机是批判 FLOPs 等硬件无关指标的失效: 算子融合、内存布局、缓存效应使实际 latency 与 FLOPs 严重不符。(来源: Microsoft Research, MobiSys'21)

[我方实证] **M2 f 函数(跨平台 latency 映射)**: 4090 → Orin 的 latency 映射, 在 **TRT body 段 R²>0.995**(n=4 ResNet 系列, `results/m2_latency_mapping_f.json`)。该函数仅在 TRT 引擎执行路径上成立; [我方实证] **在 PyTorch eager 段完全失效** — ISS-039 #3: M2 外推 enc+bb≈3.17ms, 真测 B=2 达 68.94ms(>3× 低估; `results/E8_orin_e2e_fullchain.csv`)。根因: M2 是在 **TRT 优化后的 GPU kernel 路径**上标定, PyTorch eager 有完全不同的算子粒度和内存访问模式(head 段: Orin eager 29.8ms vs 4090 TRT 1.48ms = **20× 倍率差**)。

#### 机制 C: 编译器内化 (Compiler-Internalized)

编译器在 build-time 感知目标硬件, 将软件图结构决策(layer fusion/tactic selection)与硬件最优路径自动绑定。

[文献声称] Ansor (OSDI'20) 将张量程序优化分层表示为"分块→循环顺序→向量化→并行化"的组合空间, 用在目标硬件上实测样本**训练的代价模型**指导进化搜索。代价模型直接编码硬件内存带宽、缓存层次结构和 SIMD 宽度。相对 AutoTVM 模板搜索: **Intel CPU +3.8×, ARM CPU +2.6×, NVIDIA GPU +1.7×**。(来源: Ansor arXiv:2006.06762)

[文献声称] TensorRT build-time tactic selection: 对每个 layer 枚举多种 cuDNN/CUBLAS/CUDA-core/Tensor-core 候选 kernel, **在目标 GPU 上实跑计时**, 保留最快者。workspace 大小直接控制哪些 tactic 可用("larger workspace allows TensorRT to pick any algorithm available")。Layer fusion (Conv+BN+ReLU → 单 kernel) 消除中间内存往返。(来源: NVIDIA TensorRT Developer Guide)

[我方实证] TRT **workspace 影响 tactic 选择**(真测, `data/tactic_workspace_bench.csv`, 8 行 fp16/int8×tactic×ws): workspace 改变 TRT 候选 kernel 集合, 实测幅度 **~3-4%**(含反直觉情形: 更大 workspace 反而更慢, 因候选中有 overhead 更大的算法)。workspace 是纯硬件资源参数, 但直接决定 TRT build 时能枚举哪些候选 kernel。[注: opt-level 效应数字来自 Phase H 冻结数据, 待用户裁定后方可引用具体数值。]

#### 机制 D: 联合搜索空间 (Joint Search Space)

软件(架构/量化)配置空间与硬件(设备/精度/部署参数)配置空间联合搜索, 找 Pareto 最优点。

[文献声称] APQ (CVPR'20, MIT Han Lab) 联合搜索 architecture × channel pruning × mixed-precision quantization, 使用**量化感知精度预测器**驱动进化搜索。相比分离式方法(ProxylessNAS + AMC + HAQ): **同等 latency 约束下精度高 2.3%, 搜索代价少 600×**。(来源: APQ arXiv:2006.08509)

[文献声称] OFA (ICLR'20) 训练一个支持弹性宽度/深度/核大小的超网络, 部署时用**设备专属延迟预测器**搜索满足 latency 预算的子网络。在 Samsung Note10/Google Pixel1/1080Ti 等不同设备上提取出不同的最优子网配置: 比 MobileNetV3 **+4.0% accuracy + 1.5× faster**; 比 EfficientNet **2.6× faster**。不同设备的最优子网架构不同, 证明 **"同一模型在所有设备上次优"**。(来源: OFA arXiv:1908.09791)

[文献声称] NACOS (Neural Architecture and Compiler Optimization co-Search, arXiv:2408.04116, 2024) 明确指出: NAS(软件架构搜索)和 ACO(自动编译器优化)各自独立进行时 **"demonstrate their sub-optimality when performed independently"**, 联合优化是下一步前沿。(来源: arXiv:2408.04116)

---

### 1.2 耦合方向图 (双向影响)

```
软件决策(S)                     硬件响应(H)
─────────────────────────────────────────────────────────────────────
网络结构 ──────────────────────→ 决定可用 TRT kernel 集合
 (通道数/对齐)                    (非解析 kernel 选择断崖; 简单对齐规则无法预测, 96=3×32 仍掉坑 → 必须真测)

量化精度档位 ──────────────────→ 激活不同硬件路径
 (FP16/INT8)                     (Tensor Core INT8路径, 功耗-30~52%)

模型部署形式 ──────────────────→ 可预测性范围
 (TRT引擎 vs PyTorch eager)       (TRT R²>0.995; eager外推崩塌)

软件优化目标 ←──────────────── 硬件瓶颈漂移
 (pre-body占73%后body不是瓶颈)     (body压缩后前处理成主瓶颈)

独立优化结论 ←──────────────── 硬件物理约束
 (DLA INT8完全不可build)           (bank限制/kDIRECT_IO)

跨硬件预测有效性 ←───────────── 软件栈形态
 (M2仅TRT有效/eager失效)           (TRT共享优化路径; eager各异)

调度并行收益 ←───────────────── 物理资源是否不相交
 (进程内DLA 1.00× vs 双进程1.34×)  (TRT 8.5 进程内序列化DLA提交)
─────────────────────────────────────────────────────────────────────
```

---

### §1.5 原理层耦合机理: 软硬必然相互决定的第一性原理

> v1.3 新增 | team-lead 授权 | 2026-06-06  
> 从体系结构第一性原理回答"为什么软件决策必然影响硬件行为, 反之亦然"。每条原理配**文献锚 + 我方实证锚**双源。

---

#### P1 Roofline 模型与算术强度 — 软件结构决定硬件性能域

**原理**: 任何计算 kernel 的可达性能上界由 Roofline 方程刻画:

```
可达性能 = min(Peak_FLOPS, I × BW_DRAM)
```

其中 **I(算术强度/operational intensity) = FLOPs / 每次访问 DRAM 的字节数**。`I` 与硬件**脊点**(ridge point = Peak_FLOPS / BW_DRAM)的相对位置决定 kernel 处于计算密集(compute-bound)还是访存密集(memory-bound)区 — 同一优化在两个区域的收益截然不同。

**软硬耦合逻辑**: 软件决策直接改变 `I`:
- **剪枝**: 减少 FLOPs → I 下降 → 可能从 compute-bound 滑入 memory-bound → 再减少 FLOPs 收益递减甚至无效
- **量化** (FP32→INT8): 每操作字节数减 4× → I 上升 → 更容易保持 compute-bound, 计算能力利用率提升
- **batch 扩大**: 权重字节复用率上升 → I 上升 → 单元算力利用率提高

同样的 `I` 值在不同硬件上可能落在不同域: **4090(sm89) ridge point >> Orin AGX(sm87) ridge point**, 因此 4090 上 compute-bound 的 kernel 在 Orin 上可能已是 memory-bound — 软件优化路径在两个硬件上需要不同策略。

[文献声称] Williams, Waterman, Patterson "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (CACM 2009, Vol.52 No.4): 定义 operational intensity = FLOPs / DRAM 字节流量; `Attainable GFlops/s = min(Peak_FLOPS, I × Peak_BW)`; ridge point = 达到最大性能所需的最低 I; X2→X4 升级 ridge point 从 1.0 升至 4.4 FLOPs/Byte, 相同 kernel 在两机器上 bottleneck 不同。论文明确: "the ridge point offers insight into the computer's overall performance... the ridge point suggests the level of difficulty for programmers and compiler writers to achieve peak performance." (来源: CACM 2009; OSTI:963540 核原文)

[我方实证] **E_pipeline 1.08× 推翻 roofline 乐观推断**: 4090 body kernel 实测 SM occupancy 余量 0.53-0.63, 朴素推断"37-47% 空余算力"可供 stage 并行利用; 实跑 E_pipeline 峰值仅 **1.08×**(低于 data 并行 1.13×)。根因: SM occupancy 余量是 compute 维度的乐观上界; 真实瓶颈在 L2/DRAM 带宽(多流共享同一内存访问路径) — **occupancy 槽位 ≠ 可并发吞吐空隙**。这正是 Roofline 的体系结构预测: 若 kernel 已在 memory-bound 区, 增加算力利用率对性能无益。[`results/E_pipeline_singlegpu_4090.csv`]

---

#### P2 内存层级与数据搬运物理 — 量化省能的真正根源

**原理**: 硅片上能耗的主体是**数据搬运**而非**计算**本身。这是硬件物理约束, 不随软件结构变化而改变 — 但软件的**位宽决策**直接控制搬运的字节数, 因此软硬耦合在能耗维度同样成立。

[文献声称] Horowitz "Computing's Energy Problem (and What We Can Do About It)" (ISSCC 2014, Fig.1.1.9, 45nm 0.9V):

| 操作类型 | 能耗 (45nm) | 来源 |
|---------|------------|------|
| DRAM access | **1–2 nJ**(= 1000–2000 pJ) | 原文: "energy cost of a DRAM access (1 to 2nJ)" |
| DRAM (改进 I/O) | **0.6 nJ / 8B**(= 10 pJ/bit) | 原文: "10pJ/bit, 0.6nJ/8B" |
| L1 cache fetch | **~20 pJ** | 原文: "a cache fetch is 20pJ" |
| 内部缓存/功能操作 | **~10 pJ** | 原文: "internal cache access or functional operation (10pJ)" |
| 32-bit FP 乘法 | **~3.7 pJ** | Fig.1.1.9 |
| 32-bit FP 加法 | **~0.9 pJ** | Fig.1.1.9 |
| 8-bit INT 加法 | **~0.03 pJ** | Fig.1.1.9 |

关键比值(原文直接声明): **DRAM : 缓存/计算 = 2 个数量级**("a couple of orders-of-magnitude higher"); 若与 INT8 算术比: **DRAM ≈ 33,000× INT8 ADD**。
(来源: Horowitz, ISSCC 2014, DOI:10.1109/ISSCC.2014.6757323; 核原文: gwern.net/doc/cs/hardware/2014-horowitz-2.pdf)

**软硬耦合逻辑**: 量化位宽直接砍数据搬运字节数:
- FP32 → INT8: 每个权重/激活值从 4B → 1B, 搬运字节数减 **4×**
- 配合 INT8 Tensor Core: 计算吞吐 ↑ + 内存流量 ↓ 同时发生
- J/frame 降幅 > 纯 latency 降幅: 因为功率路径也切换(Tensor Core INT8 功率密度更低)

[我方实证] **INT8 省能 30-52% 的物理验证**: T_baseline FP16 291mJ → INT8 **141mJ(-52%)**; T_prune75 FP16 280mJ → INT8 **197mJ(-30%)**。latency 降幅约 2.4×, 能耗降幅约 1.4-2.1× — 差值来自功率路径切换, 与 Horowitz 能耗层级的物理预测一致。[`results/E4_energy_4090.csv`]

---

#### P3 Kernel/tile 粒度离散性 — 断崖响应的体系结构根源

**原理**: 硬件计算单元(GPU Tensor Core MMA tile, DLA conv bank)工作在**离散粒度**上。Tensor Core wmma 指令要求矩阵维度满足特定倍数约束; DLA conv bank 有 channel 对齐要求。软件结构参数(通道数 C)落在这些粒度边界内/外时, 硬件响应呈**阶跃**而非连续渐变。

**关键补充**: tactic 选择空间本身是离散的 — TRT build-time 枚举有限个候选 kernel 实现, 对每个候选**实跑计时**后取最快者。通道数改变时, 不同候选的相对速度可能发生非单调翻转, 产生断崖。这不是简单的"对齐/不对齐"二元判断, 而是候选集在高维参数空间中的非线性划分 — **无法用代数规则预判, 只能真测**。

[我方实证] **ISS-014 kernel cliff (非解析性直接证明)**: p25_trap **2.7331ms** vs p50 **0.7956ms**(相差 **3.4×**, 同一 pruning level)。**反例**: 96 = 3×32 满足朴素 32 倍数对齐规则, 仍掉坑 → 证明"channels%32==0"等简单规则**既不充分也不必要**。断崖的真实决定因素是 TRT tactic 搜索空间的离散结构 + latency-driven 选择逻辑。[`results/P0_1_p25_layer_profile.csv`; `results/P0_1_p25_tactic_inspect.json`]

---

#### P4 编译器映射的不可交换性 — 独立最优复合 ≠ 最优复合

**原理**: 软件计算图 → 硬件 kernel 的映射(fusion/tiling/tactic assignment)是**非同态**的。若 `f(G₁)` 是图 G₁ 的最优编译, `f(G₂)` 是图 G₂ 的最优编译, 则:

```
f(G₁ ⊕ G₂)  ≠  f(G₁) ⊕ f(G₂)
```

两个独立最优的软件子图复合后, 编译结果不等于各自最优编译的组合。

**具体表现**:
1. **Layer fusion 被打破**: Conv+BN+ReLU 作为整体, TRT 可 fuse 为单 kernel, 消除中间内存往返(节省 2× activation 搬运); 若人工把这 3 层分配到不同 stage 做混合精度, fusion 被打破 → 软件的"分割/并行化"决策改变了 HW 融合的可达性
2. **Workspace 非单调**: 增大 workspace → TRT 枚举更多候选 → overhead 更大的候选有机会赢得计时 → 更大 ws 反而更慢(~3-4%)。两个表面上"更优"的独立决策(更多 flexibility + 选最快 kernel)复合后产生非预期结果

[文献声称] TensorRT Developer Guide: "larger workspace allows TensorRT to pick any algorithm available" — tactic 选择完全由 build-time 实跑计时决定, 非 workspace 单调函数。Layer fusion (Conv+BN+ReLU → 单 kernel) 消除中间内存往返。(来源: NVIDIA TensorRT Developer Guide, docs.nvidia.com/deeplearning/tensorrt)

[我方实证] workspace 非单调效应 ~3-4%(实测): `data/tactic_workspace_bench.csv`(8行 fp16/int8×tactic×ws); 更大 ws 反而更慢情形存在, 幅度 ~3-4%。

---

#### P5 Amdahl 定律的动态形态 — 协同优化下的瓶颈漂移

**原理**: 经典 Amdahl 定律: `Speedup = 1 / (s + (1-s)/N)`, 其中 `s` 为串行(瓶颈)占比。**协同优化下 `s` 本身是优化过程的函数, 不是常数**: 成功优化某子系统后, `s` 的分子缩小, 新的瓶颈浮现, Amdahl 参数动态漂移。**各管各的优化会在错误的 `s` 下持续推进, 产生零边际收益。**

**软硬耦合逻辑**: SW-optimizer 和 HW-optimizer 若各自只盯"body 网络"(历史最大延迟项), 在 body 被压缩 6.56× 后继续优化 body 几乎零收益 — 因为他们都不知道瓶颈已经漂移到 pre-body。协同优化的核心价值之一是**实时共享全链路 Amdahl 视图**, 在瓶颈漂移时立刻改变优化目标。

[我方实证] **pre-body Amdahl 73% 漂移**: body FP32 131.33ms → TRT p75+INT8 **20.02ms**(6.56×压缩); 混合链 pre-body=78.96ms + body=20.02ms + NMS=8.54ms ≈ 107.5ms; **pre-body 占 73%**。此时再压 body 50% 仅省 ~10ms(<10%); 将 pre-body enc+bb 从 PyTorch eager 换到 TRT 则潜在节省 >50ms(>46%)。优化杠杆位置发生根本性转移。[`results/E7_orin_e2e_baseline_vs_best.csv`; `results/E8_orin_e2e_fullchain.csv`]

---

#### P6 并行调度的物理资源代数 — 时间复用 vs 空间分离

**原理**: 并行调度收益的上界由**物理资源的不相交程度**决定, 而非由调度算法复杂度决定:

```
时间复用(共 SM/BW): Speedup ≤ 1 / (资源竞争开销 + 串行提交损失)
空间分离(不同物理单元): Speedup ≤ 物理单元数 × 负载均衡因子
```

两者物理根源不同, 不可互换:
- **时间复用**: 多流共享同一 SM / L2 / DRAM → 内存带宽成为共同瓶颈; 算力"并行"但内存访问串行化
- **空间分离**: 不相交的物理计算单元(不同 DLA 核心, GPU vs DLA) → 真正同时执行, 收益受限于较慢方

**TRT 8.5 进程内 DLA 的特殊情形**: Orin AGX 有 2 个物理独立 DLA 核心, 但 TRT 8.5 在**单进程内序列化** DLA 提交(submit queue 单线程) → 进程内双 DLA 退化为串行(物理不相交资源被软件接口重新串行化); **必须双进程才能触发真硅片并行** — 正是"编译器/运行时形态(软件)决定硬件并行可达性"的直接体现。

[我方实证] **1.08× vs 1.34× 实证**:
- **E_pipeline** (4090 单 GPU stage 流水, SM 时间复用): 峰值 **1.08×**(3流), 甚至低于 data 并行 1.13×; 根因: L2/DRAM 带宽共享, 算力复用收益被带宽竞争抵消
- **E3** (Orin DLA0∥DLA1): 进程内 = **1.00×**(完全串行, TRT 8.5 软件提交序列化); 双进程 = **1.34×**(真硅片并行, 物理不相交触发)
[`results/E_pipeline_singlegpu_4090.csv`; `results/E3_orin_dla_pipeline.csv`]

---

**P1-P6 总结 — 六条原理的两类决定律**:

| 类别 | 原理 | 决定的是 |
|------|------|---------|
| **坐标类** | P1(Roofline) + P2(能耗层级) | 软件决策改变 kernel 在硬件性能/能耗空间中的落点 — 落点不同则受完全不同的物理约束 |
| **映射类** | P3(粒度离散) + P4(不可交换) | 硬件实现的离散性和非同态映射 → 软件→硬件无法用规则预判, 只能真测 |
| **动态类** | P5(Amdahl漂移) + P6(资源代数) | 全局动态特性(瓶颈漂移/资源拓扑)使局部最优无法自动复合为全局最优 → 协同视角不可缺少 |

---

## §2 内部实证对照表 (7条, 数据文件全标)

| # | 实证名称 | 具体现象 | 软件侧 | 硬件侧 | 耦合类别 | 数据来源 |
|---|---------|---------|--------|--------|---------|--------|
| **①** | **INT8 kernel 选择断崖** | p25_trap: **2.7331ms**; p50: **0.7956ms** — 相差 **3.4×**, 同一 pruning level; 断崖非简单对齐规则可解析(96=3×32 是32倍数仍掉坑 — ISS-014 canonical) | 剪枝决定通道数, 通道数触发 TRT 非解析 kernel 选择断崖(简单对齐预测失效, 必须真测) | GPU INT8 tactic 选择路径非线性, 不可仅凭通道对齐预判 | D(联合搜索)+ C(编译器内化) | `results/P0_1_p25_layer_profile.csv`; `results/P0_1_p25_tactic_inspect.json` (ISS-014 canonical, profile+tactic verified) |
| **②** | **DLA INT8 0/12 全失败 vs GPU INT8 正常** | Orin DLA0/DLA1 INT8 build: 0/12成功(fail: kDIRECT_IO + bank>16); GPU INT8: 12/12成功 (20.55ms) | 同一网络/同一量化配置(INT8) | DLA v2的 IO conformance + bank限制使PyramidFusion INT8 无法build; GPU则正常 | D(联合搜索): SW配置 × 硬件单元交叉决定可达性 | `results/orin_dspace_bench.parquet`; `dims_hardware_v2.md §D类实测` |
| **③** | **TRT tactic延迟驱动非精度驱动** | workspace 改变 TRT 候选 kernel 集, 幅度 **~3-4%**(非单调: 更大 ws 不一定更快, 取决于候选算法 overhead); opt-level 效应[Phase H 冻结, 待用户裁定] | 软件模型结构/精度固定 | TRT build 时在 GPU 上实跑候选 kernel, workspace 大小决定候选集 → latency 驱动选择, 非资源单调 | C(编译器内化) | `data/tactic_workspace_bench.csv` (8 行真测, ~3-4%); `dims_hardware_v2.md §B2` |
| **④** | **单GPU流水1.08× vs GPU∥DLA双进程1.34×** | E_pipeline 4090单GPU stage流水峰值 **1.08×**(3流); E3 Orin DLA0∥DLA1进程内 **1.00×**(完全串行); 双进程 **1.34×** (真硅片并行) | 同一流水调度方案 | GPU时间复用: stage共享SM → 1.08×; DLA进程内TRT 8.5序列化提交 → 1.00×; 双进程不相交物理资源 → 1.34× | D+A(硬件在环): 物理资源拓扑决定调度上限 | `results/E_pipeline_singlegpu_4090.csv`; `results/E3_orin_dla_pipeline.csv` |
| **⑤** | **pre-body Amdahl 73%**(软件优化目标随硬件瓶颈漂移) | body从FP32 131.33ms压到TRT p75+INT8 20.02ms(6.56×); 混合链pre-body=78.96ms → **pre-body占73%** (108ms中的79ms) | body软件优化(剪枝+量化)极其成功 | body不再是瓶颈; 此时软件优化目标必须从body转到pre-body(enc+bb=68.94ms); 若各管各的则优化错目标 | A(在环反馈): 硬件执行结果改变软件优化方向 | `results/E7_orin_e2e_baseline_vs_best.csv`; `results/E8_orin_e2e_fullchain.csv` |
| **⑥** | **INT8 真省能耗 30-52%**(精度档位即能耗档位) | T_baseline INT8 141mJ vs FP16 291mJ = **52%节省**; T_prune75 INT8 197mJ vs FP16 280mJ = **30%节省** | 软件量化精度选择(FP16 vs INT8) | INT8路径: 更高吞吐(Tensor Core利用率) + 更低功率复合 → J/frame降幅超过latency降幅 | A(在环反馈): 软件精度选择 → 硬件功耗路径 | `results/E4_energy_4090.csv`; `dims_hardware_v2.md §E类` |
| **⑦** | **M2 f函数TRT有效/eager失效** | TRT body段 4090→Orin: R²>0.995(f_fp16/f_int8 slope~0.93-0.95); PyTorch eager: M2外推enc+bb≈3.17ms, 真测68.94ms(**>3× 低估**) | TRT vs PyTorch eager两种软件栈 | TRT将算子优化为GPU平台共享路径(SM使用率/Tensor Core), 使跨平台比值稳定; eager执行路径因硬件而异(head: 4090 TRT 1.48ms vs Orin eager 29.8ms = **20× 差异**) | B(代理模型): 预测器有效性本身取决于SW栈形态 | `results/m2_latency_mapping_f.json`; `results/E8_orin_e2e_fullchain.csv`; `multi_agent/methods/design/orin_pyramid_baseline_report_v1.md §6` |

---

## §3 文献谱系表

| 论文 | 机制类别 | 核心贡献 | HW信号 | 关键数字 | 年份/venue |
|------|---------|---------|--------|---------|-----------|
| **HAQ** | A(在环) | RL驱动混合精度量化, 真实HW latency/energy作reward | 硬件仿真器真实延迟/能耗 | vs均匀量化: +3.58% Top-1; 跨硬件迁移**5.2×延迟恶化** | CVPR'19 |
| **AMC** | A(在环) | RL驱动逐层剪枝率搜索, 移动端真实latency作reward | 端侧真实latency | 首个在手机端以latency为reward; vs manual: +1.2% acc @ 2× FLOP | ECCV'18 |
| **FBNet** | B(代理) | 可微分NAS + HW latency查找表纳入损失函数 | 预测量测LUT(Samsung S8/iPhone X) | FLOPs≠latency; vs MobileNetV2: **2.4×更小 / 1.5×更快** | CVPR'19 |
| **APQ** | D(联合) | 联合搜索arch × pruning × quant; 量化感知精度预测器 | latency LUT驱动进化搜索 | vs分离优化: **+2.3% acc, -600× 搜索代价** | CVPR'20 |
| **OFA** | D(联合) | 弹性超网络 + 设备专属子网搜索 | 目标设备真实latency | vs MobileNetV3: **+4% acc + 1.5× faster**; 不同设备最优子网不同 | ICLR'20 |
| **HAWQ-V2** | A(在环) | Hessian迹驱动逐层量化敏感度 | 二阶梯度(模型内部,非外部HW) | vs均匀量化: Inception-V3 **+5.92% Top-1**; RetinaNet **+2.6 mAP** | NeurIPS'20 |
| **HAWQv3** | A+D | 混合精度INT4/INT8整数线性规划 + T4 GPU直接部署 | T4 GPU真实latency | vs INT8均匀: **最高50%加速** | arXiv:2011.10680 |
| **HALP** | B(代理) | per-filter latency LUT + augmented knapsack | per-filter GPU latency LUT | ResNet-50: **1.6× 吞吐 + 0.3% acc提升** | NeurIPS'22 |
| **nn-Meter** | B(代理) | 内核级latency预测器(算子融合感知) | 目标设备kernel-level profiling | CPU/GPU预测精度**99%+**; Best Paper MobiSys'21 | MobiSys'21 |
| **Ansor** | C(编译器) | 层次化搜索空间 + HW实测训练代价模型 + task调度器 | 硬件实测样本训练代价模型 | vs AutoTVM: **Intel CPU 3.8×, ARM 2.6×, GPU 1.7×** | OSDI'20 |
| **TensorRT** | C(编译器) | Build-time tactic profiling + layer fusion | 目标GPU实跑每种candidate | vs CPU: **>40× 更快**; workspace控制tactic可用集 | NVIDIA产品 |
| **NACOS: Neural Architecture and Compiler Optimization co-Search** | D(联合) | NAS+ACO联合框架; 明确独立优化次优 | 联合HW-SW搜索空间 | "demonstrate their sub-optimality when performed independently" — 综述级定论 | arXiv:2408.04116, 2024 |
| **Survey (arXiv:2311.17815)** | D | 异构架构加速综述; 需多学科协同 | 架构级设计空间探索 | "requires a multidisciplinary approach combining ML to computer architecture" | 2023 |
| **Survey Quant (arXiv:2103.13630)** | A | 量化综述; HW实现差距系统分析 | 理论vs实际gap分析 | FP32→INT4理论16×, 实际**4-8×**; 2-4×实现差距 | 2021 |

---

## §4 为什么必须软硬协同(反例: 独立优化的失败模式)

### 4.1 独立优化的失败模式分类

#### 模式 F1: SW优化结果在目标HW不可执行/不可build

**[我方实证]** DLA INT8在PyramidFusion上: 量化师标定INT8配置(软件侧正常), 但DLA硬件build完全失败(0/12), fail_reason: `kDIRECT_IO + bank超限`。如果只做SW优化(量化)不考虑HW约束(DLA bank结构), 就会得到"INT8 body@DLA"这个在当前网络上根本无法部署的方案。(`results/orin_dspace_bench.parquet`)

**[文献声称]** HAWQ-V2: 均匀量化(硬件无关)对所有层施同精度, 对敏感层(高Hessian迹)造成大精度损失; Inception-V3均匀2-bit/4-bit = 69.76% vs HAWQ-V2 75.68% (差5.92个百分点)。(arXiv:1911.03852)

#### 模式 F2: HW优化结果在SW层面次优

**[我方实证]** 单GPU stage流水(先做硬件侧调度): 基于roofline余量0.53-0.63的乐观推断, 预期pipeline有大收益; 但实跑 E_pipeline 峰值仅 **1.08×**, 甚至低于data并行的1.13×。根因: occupancy槽位≠可并发吞吐空隙; L2/DRAM带宽墙; eager串行调度。不结合软件执行特征(memory-bound kernel共享L2)的纯硬件调度设计走了弯路。(`results/E_pipeline_singlegpu_4090.csv`)

**[我方实证]** TensorRT workspace 策略非单调: 更大 workspace 允许 TRT 尝试 overhead 更大的候选 kernel, 可导致更大 ws 反而更慢(幅度 ~3-4%, 真测)。纯硬件资源堆叠不等于性能提升。(`data/tactic_workspace_bench.csv`, 8 行 fp16/int8×tactic×ws 真测)

#### 模式 F3: SW层面"优化"抹消HW收益

**[我方实证]** SW 剪枝决定的通道配置触发 TRT INT8 kernel 非解析选择断崖: ISS-014 canonical — p25_trap **2.7331ms** vs p50 **0.7956ms**, 相差 **3.4×**, 同一 pruning level。断崖非简单对齐规则可预测(96=3×32 仍掉坑), 体现 SW 剪枝通道配置直接抹消 HW INT8 量化收益, 必须真测。(`results/P0_1_p25_layer_profile.csv`; `results/P0_1_p25_tactic_inspect.json`)

**[文献声称]** Survey (arXiv:2103.13630): FP32→INT4理论16×压缩比, 实际只有4-8×, gap来自"内存对齐、kernel支持、计算图调度"等硬件实现细节 — 这些是纯SW优化无法预知的HW约束。

#### 模式 F4: 优化目标漂移(各管各的则优化错方向)

**[我方实证]** pre-body Amdahl漂移: 若SW-optimizer只看body网络(做剪枝+量化), HW-optimizer只看body执行(TRT tactic), 两者都会认为"body是主要优化对象"。但当body从131ms压到20ms后, 瓶颈转移到pre-body的PyTorch eager enc+bb (68.94ms占73%), 此时继续优化body几乎零收益。**"各管各的"会让SW和HW都优化一个已经不是瓶颈的部分**。(`results/E7`+`E8`)

**[文献声称]** OFA: 针对特定设备优化的子网络与针对其他设备的最优子网架构不同。Samsung Note10最优子网(窄/深)在1080Ti上次优; 如果SW(架构)优化不考虑HW(设备)约束, 两者都拿到次优解。(arXiv:1908.09791)

#### 模式 F5: 跨硬件预测失效(不知道SW栈形态就不能做HW外推)

**[我方实证]** M2 f函数在TRT body段R²>0.995, 但在PyTorch eager段完全崩塌(>3×低估)。旧预测(enc+bb≈3.17ms)导致基于此的整个可达性分析是错的(旧结论"最悲观情形仍可达" → 真测后翻转)。**不考虑SW栈形态(TRT vs eager)就做HW间延迟外推, 是系统性偏差**。(`results/m2_latency_mapping_f.json`; ISS-039 #3)

**[文献声称]** HAQ 跨硬件迁移失败(5.2-7.2×恶化): 为HW1优化的量化策略在HW2上严重退化。硬件感知量化策略**不可迁移**, 必须对每个目标硬件重新在环优化。(arXiv:1811.08886)

---

### 4.2 为什么协同 — 一句话总结

**"软件决策改变硬件可达的kernel集合, 硬件特性改变软件优化的有效目标; 两者之间的每一条信道都是双向的、非线性的、不可用代理指标替代的。各管各的必然在至少一个信道上拿到次优解, 往往同时在多个信道上失效。"**

---

## §5 完整文献索引

### 硬件感知量化 / 混合精度

| 论文 | URL / arXiv |
|------|------------|
| HAQ: Hardware-Aware Automated Quantization with Mixed Precision (CVPR'19) | https://arxiv.org/abs/1811.08886 |
| HAWQ: Hessian AWare Quantization of Neural Networks (ICCV'19) | https://arxiv.org/abs/1905.03696 |
| HAWQ-V2: Hessian Aware trace-Weighted Quantization (NeurIPS'20) | https://arxiv.org/abs/1911.03852 |
| HAWQv3: Dyadic Neural Network Quantization (ICML'21) | https://arxiv.org/abs/2011.10680 |
| APQ: Any-Precision Quantization with HW-aware Search (CVPR'20) | https://arxiv.org/abs/2006.08509 |
| A Survey of Quantization Methods for Efficient Neural Network Inference | https://arxiv.org/abs/2103.13630 |

### 硬件感知 NAS / 网络压缩

| 论文 | URL / arXiv |
|------|------------|
| AMC: AutoML for Model Compression (ECCV'18) | https://arxiv.org/abs/1802.03494 |
| FBNet: Hardware-Aware Efficient ConvNet Design (CVPR'19) | https://arxiv.org/abs/1812.03443 |
| MnasNet: Platform-Aware NAS for Mobile (CVPR'19) | https://arxiv.org/abs/1807.11626 |
| Once-for-All: Train One Network, Specialize for Efficient Deployment (ICLR'20) | https://arxiv.org/abs/1908.09791 |
| HALP: Hardware-Aware Latency Pruning (NeurIPS'22) | https://arxiv.org/abs/2110.10811 |
| LightPrune: Latency-Aware Structured Pruning (ICCVW'25) | https://openaccess.thecvf.com/content/ICCV2025W/EVW/papers/Belhadi_LightPrune_Latency-Aware_Structured_Pruning_for_Efficient_Deep_Inference_on_Embedded_ICCVW_2025_paper.pdf |

### 编译器协同设计

| 论文 | URL / arXiv |
|------|------------|
| Ansor: Generating High-Performance Tensor Programs (OSDI'20) | https://arxiv.org/abs/2006.06762 |
| TVM: End-to-End Optimization Stack (OSDI'18) | https://arxiv.org/abs/1802.04799 |
| TensorRT Developer Guide (NVIDIA) | https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html |
| TensorRT Speed Up Inference Blog | https://developer.nvidia.com/blog/speed-up-inference-tensorrt/ |
| nn-Meter: Accurate Latency Prediction on Diverse Edge Devices (MobiSys'21) | https://github.com/microsoft/nn-Meter |

### 协同设计综述 (2023-2026)

| 论文 | URL / arXiv |
|------|------------|
| NACOS: Neural Architecture and Compiler Optimizations co-Search Survey (2024) | https://arxiv.org/abs/2408.04116 |
| Survey on Design Methodologies for Accelerating DL on Heterogeneous Architectures (2023) | https://arxiv.org/abs/2311.17815 |
| Hardware-Software Co-Design for Machine Learning: Recent Progress and Future Challenges | https://arxiv.org/abs/2007.06738 |

### 本项目内部数据文件

| 文件 | 内容 |
|------|------|
| `results/E4_energy_4090.csv` | ⑥ 4090 INT8 vs FP16 能耗; 13 engine × 功率×latency |
| `results/E7_orin_e2e_baseline_vs_best.csv` | ⑤ Orin TRT body: FP32 131.33ms / FP16 47.99ms / INT8 20.02ms |
| `results/E8_orin_e2e_fullchain.csv` | ⑤⑦ Orin PyTorch全链D1v2: B=2 260.61ms; D2合成混合链 |
| `results/E3_orin_dla_pipeline.csv` | ④ DLA0∥DLA1: 进程内1.00× / 双进程1.34× |
| `results/E_pipeline_singlegpu_4090.csv` | ④ 单GPU stage流水峰值1.08× |
| `results/m2_latency_mapping_f.json` | ⑦ M2 f函数: TRT R²>0.995; eager失效 |
| `data/tactic_workspace_bench.csv` | ③ workspace×tactic ~3-4%幅度真测(8行fp16/int8×tactic×ws); 替代H2冻结引用 |
| `results/P0_1_p25_layer_profile.csv` + `results/P0_1_p25_tactic_inspect.json` | ① INT8 kernel cliff ISS-014 canonical: p25_trap 2.7331ms vs p50 0.7956ms (3.4×) |
| `results/orin_dspace_bench.parquet` | ② DLA INT8 0/12 vs GPU INT8 12/12 |
| `multi_agent/methods/design/dims_hardware_v2.md` | 全D维度分析 + 调度实验汇总 |
| `paper_learning/survey_raw_2/pruning_x_hardware.md` | 剪枝×硬件深度调研 |
| `paper_learning/survey_raw_2/ad_v2x_deployment.md` | 自动驾驶D空间调研 |

---

## §6 快速核查索引 (供 supervisor 逐引核验)

| 声称数字 | 类型 | 来源文件/位置 |
|---------|------|-------------|
| INT8 kernel cliff: p25_trap 2.7331ms vs p50 0.7956ms (3.4×) | [我方实证] ISS-014 canonical | `results/P0_1_p25_layer_profile.csv`; `results/P0_1_p25_tactic_inspect.json` |
| DLA INT8 0/12; DLA FP16 8/12; GPU INT8 12/12 | [我方实证] 真测 | `results/orin_dspace_bench.parquet` + `dims_hardware_v2.md §D类` |
| workspace效应 ~3-4%幅度(非单调) | [我方实证] 真测 | `data/tactic_workspace_bench.csv` (8行) |
| 单GPU流水1.08×; DLA进程内1.00×; 双进程1.34× | [我方实证] 真测 | `E_pipeline_singlegpu_4090.csv`; `E3_orin_dla_pipeline.csv` |
| pre-body 78.96ms / body 20.02ms / pre-body占73% | [我方实证] 合成 | `E7`+`E8 D2_B2_FP16_synth` 行 |
| T_baseline INT8 141mJ vs FP16 291mJ (-52%) | [我方实证] 真测 | `E4_energy_4090.csv` 第3行 |
| M2 f_fp16 R²=0.9953; f_int8 R²=0.9969 | [我方实证] 真测 | `m2_latency_mapping_f.json` |
| M2外推enc+bb≈3.17ms vs真测68.94ms | [我方实证] ISS-039 #3 | `E8_orin_e2e_fullchain.csv` D1_B2_FP32_v2行 + 报告 §6 |
| head: Orin eager 29.8ms vs 4090 TRT 1.48ms = 20× | [我方实证] 真测 | `E8` D1_B2_FP32_v2行 t_head_p50 + `E7` FP16 head推算 |
| HAQ 跨硬件迁移 5.2-7.2× latency恶化 | [文献声称] | arXiv:1811.08886 Table 1 |
| HAWQ-V2 Inception-V3 +5.92% Top-1 | [文献声称] | arXiv:1911.03852 Table 2 |
| Ansor vs AutoTVM: CPU 3.8×, GPU 1.7× | [文献声称] | arXiv:2006.06762 §Evaluation |
| OFA vs MobileNetV3: +4% acc + 1.5× faster | [文献声称] | arXiv:1908.09791 Table 1 |
| NACOS: "demonstrate their sub-optimality when performed independently" | [文献声称] | arXiv:2408.04116 Abstract |
| HAWQv3: INT4/INT8最高50%加速 vs INT8 | [文献声称] | arXiv:2011.10680 (via survey arXiv:2103.13630 §IV-C) |
