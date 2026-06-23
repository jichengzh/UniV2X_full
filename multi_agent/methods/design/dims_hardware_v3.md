# GPU 硬件加速技术文献综述 (D 维度) v3.0 — GPU 流水线/缓存/调度 × "能否做到 FPGA 级流水线"

> **文档定位**: 本文是 **纯文献综述层**, 与 `dims_hardware_v2.md`(本仓库实测层)互补。v2 回答"我们实测到了什么"; v3 回答"已发表研究说 GPU 上的硬件加速技术是怎么实现的、上限在哪", **所有结论挂可检索的论文/官方文档出处, 不依赖本仓库实验外推**。
>
> **核验声明 (2026-06-05)**: 文末 §R 引用列表中每条均经摘要级核验(WebFetch 抓取 arXiv/官方页, 确认标题、作者、venue、关键数字与正文引用相符); Paella/Orion 为 DOI+检索核验(题录确认); REEF 为 venue 级核验(usenix 页访问受限, 但 OSDI'22 题录与官方代码库公开可查)。正文引用处一律用 [Rn] 编号。
>
> **与 v2 的关系**: v2 §A7/§0.5 已基于本仓库实测(E1/E3/E_pipeline)得出"单 GPU stage 流水已证伪 (1.08×)、异构 GPU∥DLA 多进程真并行 (1.34×)"等结论。本文 §5 把这些实测与文献逐条对账 —— **文献与实测互相印证, 没有冲突**。

---

## 0. 核心问题与一句话答案

**问题**: 以 GPU 为底层硬件的加速方法(流水线设计、缓存优化、并发调度)是怎么实现的? GPU 能否做到和 FPGA 一样的流水线设计与加速?

**一句话答案 (文献定论)**:
GPU **不能在硬件意义上复刻** FPGA 的空间流水线(每层独占专用电路、initiation interval II=1、数据在片上逐级流动不回 DRAM)—— 根本限制是 SM 时间复用、SIMT 编程模型与异步任务并行硬件的错位、共享 L2/DRAM 带宽墙、以及 kernel 边界语义 [R1][R4][R12]。**但 2023–2026 的研究给出了三条"软件数据流"逼近路径**: ① megakernel/持久化内核把整个模型熔成单 kernel 内的数据流图 [R2][R6][R7]; ② warp specialization 在 kernel 内构造生产者–消费者硬件流水 [R4][R5][R9]; ③ 用户态调度器绕开硬件 FIFO 调度 [R13][R14][R15]。这些技术的实测收益普遍在 **1.1×–2.5×** 区间(对强 baseline), **不是数量级**; FPGA 空间数据流真正显著的优势集中在 **batch-1 memory-bound 场景的能效**(如 LLM decode 对 A100: 速度 1.9×、能效 5.7× [R11])。结论: **GPU 可以在"软件层"逼近 FPGA 式流水的形态, 拿不到 FPGA 式流水的数量级**; 对延迟数量级的改善仍要靠 work-reduction(剪枝/量化)与不相交物理资源(异构加速器)。

---

## 1. 流水线 / 数据流执行技术 (Pipeline & Dataflow on GPU)

### 1.1 为什么 GPU 默认不是流水线: bulk-synchronous 时间复用

GPU 标准执行模型是 **bulk-synchronous 的算子级时间复用**: 一个时刻整卡 SM 池只跑一个算子的 kernel, 算子间结果写回 DRAM 再读出。Kitsune (NVIDIA Research + UW-Madison) 明确指出: "由于 SM 的时间复用, 单算子执行时 GPU 上大量资源处于闲置", 这正是 dataflow/空间执行的动机 [R1]。这与 FPGA 形成根本对照 —— FPGA 把每层映射到专属电路区域, 各层**同时**工作, 数据流式穿过 (FINN 的 per-layer tailored compute resources 流式架构 [R10])。

**含义**: 在 GPU 上做"FPGA 式 stage 流水"不是默认能力, 而是要靠下述软件技术显式构造; 且每条技术都要对抗时间复用模型的三个固有摩擦 —— kernel 启动语义、算子间 DRAM 往返、SM/带宽共享。

### 1.2 Kernel fusion (算子融合) — 最成熟的"局部流水"

把相邻算子熔进同一 kernel, 中间结果留在寄存器/shared memory, 不回 DRAM。这是对"FPGA 片上数据流"的最小尺度模仿, 收益对 **memory-bound 算子链**最大:
- FCM (Chalmers, 2024): 把 depthwise+pointwise 卷积熔成单 kernel, **省最多 83% 访存, 对 cuDNN 提速最高 3.7×** —— 证明对 memory-bound 卷积模式, 融合可胜过厂商库 [R8]。
- 工程现状: TensorRT 的 build-time 融合(conv+bias+ReLU 等)即此类技术的产品化; Triton/torch.compile 让自定义融合平民化。
- 局限: 传统"垂直融合"受限于生产者–消费者的 tile 形状兼容性; Kitsune 指出要绕开垂直融合约束、实现通用 GPU 数据流, 需要"适度的体系结构调整"(modest architectural adjustments) —— **即未改硬件的现售 GPU 无法完全实现 FPGA 式空间数据流** [R1]。

### 1.3 CUDA Graphs — 压 launch 开销, 非流水线

CUDA Graph 把 kernel 启动序列录制成图、一次提交, 消解逐 kernel 的 CPU→GPU 启动开销。文献定位它为"必要的工程基线"而非加速杠杆:
- Ada-MK 实测 LLM 推理中 **kernel 启动开销最高占 e2e 14.6%**(1.65M 次启动 ~3.3s), 这是 launch 开销的文献量化上界 [R7];
- FlashInfer (MLSys'25 最佳论文) 指出 CUDA Graph **要求静态 kernel 配置**, 动态负载的调度器必须显式绕设计 (其 load-balanced scheduler 专门做到"适应请求动态性同时保持 CUDA Graph 兼容") [R3] —— 即静态 shape 约束是真实工程限制;
- 对照本仓库: CUDA Graph 实测 1.05–1.10× (v2 §A 实测块), 与"launch 开销占比百分之几到百分之十几"的文献区间一致。

### 1.4 Persistent kernel / Megakernel — 整模型单 kernel 化, 最接近"GPU 上的数据流"

把整个(甚至多卡)模型前向熔成**一个常驻 kernel**, kernel 内部用软件调度器/任务队列驱动各 SM, 算子边界消失 → 跨算子软件流水、细粒度重叠成为可能。这是 2025–2026 文献里 GPU 逼近 FPGA 数据流的**最强形态**:
- **MPK (Mirage Persistent Kernel, CMU 等, 2025)**: 第一个把多 GPU 模型推理自动编译成单 megakernel 的编译器+运行时; 引入 **SM 级依赖图**(以单个 SM 为粒度刻画数据依赖), 实现"以往在 GPU 上不可行"的跨算子软件流水与细粒度重叠; LLM 推理 e2e 延迟最多降 **1.7×** [R2]。
- **Hazy Research megakernel (Stanford, 2025-05)**: 手写 Llama-1B 单 megakernel。动机数据: 现有系统(vLLM/SGLang)把前向拆成 ~百个 kernel, H100 带宽利用率 ≤50%; megakernel 提到 **78% 带宽利用**, H100 上比 vLLM 快 ~2.5×、比 SGLang 快 >1.5×, B200 上 <680µs 完成 1B 模型 16-bit 前向 [R6]。
- **Ada-MK (2026-05)**: 离线 DAG 搜索自动生成 megakernel(MLIR 把算子黑盒拆到 PTX 级依赖); 对 TensorRT-LLM 单 batch 吞吐 +23.6%, 对 vLLM +50.2% [R7]。
- **量级判断**: 即便是 megakernel 这种最激进形态, 对强 baseline 的收益也是 **1.2–2.5×** 量级(对弱 baseline 如 vLLM 单 batch 才到 3.5×), 不是 FPGA 宣传话术中的数量级。且现有 megakernel 工作全部以 **LLM(GEMM 链)** 为对象; conv 检测网的算子异质性(NMS/grid_sample/deformable)使其工程移植成本显著更高, 文献中无 conv-detection megakernel 先例。

### 1.5 Warp specialization + 软件流水 — kernel 内的生产者–消费者硬件流水

Hopper/Blackwell 引入异步硬件单元(TMA、WGMMA、mbarrier)后, kernel 内部可以把 warp 分成**生产者(搬数)/消费者(算)**角色, 用异步屏障衔接 —— 这是 GPU 上最像"FPGA 流水级"的硬件机制(每级有专属执行单元, 数据经 shared memory 逐级传递):
- **Tawa (NVIDIA + Cornell, CGO'26)**: 指出"传统 SIMT 编程模型与现代任务并行 GPU 硬件**根本性错位**"; 提出 aref(异步引用)IR, **编译器自动**把程序划分成生产者–消费者 warp 角色、自动管理数据流软件流水, 无需手写 [R4]。
- **Twill (Stanford/NVIDIA 等, 2025-12)**: 把软件流水(SWP)与 warp specialization(WS)联合形式化为约束求解问题, **自动求出可证最优的调度**, 在 Hopper/Blackwell 上重新发现了专家手写的 Flash Attention 调度 [R5]。
- **硬件侧支撑**: Hopper 微基准论文实测 **TMA 异步编程模型给矩阵乘 1.5× 提速** [R9]。
- **定位**: 这条线解决的是 **kernel 内部**的访存/计算重叠(把 memory 延迟藏进计算), 不是跨算子/跨帧的 stage 级流水; 它已经被 cuBLAS/cuDNN/TensorRT/FlashAttention 内核大量采用, 即"你买到的 TRT 延迟里已经含了这部分收益"。

### 1.6 算子级并行 (inter-operator parallelism)

把模型 DAG 中**无依赖的算子**调到不同 stream 并发, 用资源互补(compute-bound ∥ memory-bound)填空隙:
- **Opara (华东师大等, 2023)**: CUDA Streams + CUDA Graph 自动并行无依赖算子, 按资源需求排启动序; 对 PyTorch 默认串行 CUDA Graph 提速最高 **1.68×**, 对 SOTA 算子并行系统最高 1.29× [R16]。注意 1.68× 是分支丰富模型(Inception 类)的峰值; 顺序 conv 链收益显著更低(v2 §A7 已引用此点: conv-dense 顺序模型预期 1.1–1.3×)。
- **边缘侧佐证**: Jetson 并发推理剖析(2025)发现 "**GPU 利用率 100% 时 SM/TensorCore 实际利用率仅 15–30%**", 且 CPU 侧线程调度/上下文切换常是约束 [R18] —— 即"GPU util 高"≠"没有并行空隙", 但空隙能否兑现取决于调度与依赖(与本仓库 E_pipeline 结论一致, 见 §5)。

---

## 2. 缓存 / 内存层级优化 (Cache & Memory Hierarchy)

GPU 的内存层级旋钮是文献里"对抗 memory-bound"的主战场 —— 这与 FPGA"把权重/激活钉在片上 BRAM"的思路同源, 但 GPU 片上容量与控制粒度都受限得多:

| 技术 | 机制 | 文献/文档依据 | 适用判断 |
|---|---|---|---|
| **Shared memory tiling** | 程序员显式管理的片上 scratchpad, kernel 内复用数据 | 经典技术, 现代形态见 warp-specialized 流水 (Tawa/Twill, 数据经 shared memory 在 warp 角色间逐级传递) [R4][R5] | 一切高性能 kernel 的基础; TRT kernel 已内置 |
| **L2 persisting (accessPolicyWindow)** | CUDA 11+/Ampere 起可**划出一块 L2 set-aside** 给指定地址区间的"persisting"访问优先驻留 | CUDA C Programming Guide §"L2 Cache Control / set-aside for persisting accesses" [R19] | 唯一面向用户的 L2 驻留控制; 适合反复读的小权重/特征图。注意是"优先驻留"非硬隔离, 收益依赖访问模式 |
| **cp.async (Ampere) → TMA (Hopper)** | 异步 DRAM→shared 拷贝, 旁路寄存器, 由专用硬件单元(TMA)搬数, 计算与搬数重叠 | Hopper 微基准: TMA 异步模型给 GEMM **1.5×** [R9]; Tawa: 异步单元是"高性能异步数据流执行"的硬件基础 [R4] | kernel 内隐藏访存延迟的核心机制; 4090(Ada)有 cp.async 无 TMA, Orin(Ampere)有 cp.async |
| **融合省访存** | 见 §1.2, 融合的本质就是把中间张量从 DRAM 移进寄存器/shared | FCM 省 83% 访存 [R8]; megakernel 消除算子间 HBM 往返 [R2][R6][R7] | memory-bound 链收益最大 |
| **带宽墙(roofline)** | 片上优化救不了本质 memory-bound 的算子 —— depthwise/pointwise 卷积计算访存比低, 瓶颈在访存不在算力 | FCM 论文的出发点 [R8]; Hazy: 现有系统连 50% HBM 带宽都吃不满, megakernel 也只推到 78% [R6] | **结论: 缓存优化的上限是把带宽利用率从 ~50% 推向 ~80%, 不能凭空创造带宽**。低带宽平台(Orin LPDDR5 204.8GB/s vs 4090 1008GB/s)上 memory-bound 层只会更痛 |

**与 FPGA 的对照**: FPGA 数据流架构把层间激活完全留在片上 FIFO/BRAM(FINN: per-layer 专属资源、流式衔接 [R10]), 等效于"100% 片上带宽、零 DRAM 往返"; GPU 即便 megakernel + L2 驻留, 大特征图仍必须走 DRAM。这是能效差距(§4)的微观根源。

---

## 3. 并发 / 调度技术 (Concurrency & Scheduling)

### 3.1 硬件/厂商机制

| 机制 | 粒度 | 文献/文档定性 |
|---|---|---|
| **CUDA Streams + 优先级** | kernel 级, FIFO 队列 | Paella: 内建调度是 FIFO、烧死在 runtime/驱动/硬件里, **易发 Head-of-Line 阻塞**, stream 优先级只暴露了很弱的控制 [R13] |
| **MPS** | 进程级空间共享 | Paella/Orion: 减轻但不消除 HoL 与干扰; **非干扰感知** [R13][R14] |
| **MIG** | 硬件切片(仅数据中心卡) | Orion: 粒度太粗, 不适合细粒度推理混跑 [R14]; 4090/Orin 均无 MIG |
| **Green Contexts (CUDA 12.4+)** | SM 子集划分的轻量 context | 官方文档**明确警告**: "即便 green context 的 SM 分区不相交, 也**不保证**其中启动的 kernel 并发运行或有前进性保证 —— 其他资源(如共享带宽/依赖)仍可能造成串行" [R20] |
| **kernel 抢占 (Pascal+)** | 指令级 | Paella: 换入换出开销过高(prohibitive), 实践中不用 [R13]; REEF (OSDI'22) 用 reset-based 思路做到微秒级抢占, 但面向多租户 DNN 混跑 [R15] |

> **要点**: Green Contexts 官方 caveat [R20] 在文献层面坐实了 v2 的实测教训 —— **SM 分区/占用槽位 ≠ 可兑现的并发吞吐**, 共享 L2/DRAM 带宽与依赖才是硬约束。

### 3.2 用户态软件调度器 (REEF / Paella / Orion 一族)

共同思想: **绕开 GPU 硬件 FIFO 调度器**, 在用户态拦截/重排 kernel:
- **Paella (SOSP'23)**: 编译器改写 kernel + 用户态逐 kernel 调度(选预期最短的 kernel 先发), 解决 HoL 阻塞 [R13];
- **Orion (EuroSys'24)**: 拦截多客户端 kernel 启动, 按算子的 compute/memory 画像做**干扰感知**的细粒度(10µs 级)共享, 显著改善尾延迟 [R14];
- **REEF (OSDI'22)**: reset-based 微秒级抢占 + 动态 kernel padding, 实时任务优先 [R15]。
- **适用性判断**: 这族工作的目标是 **多模型/多租户共享一卡** 的吞吐与尾延迟, **不是单模型单帧延迟** —— 对"一张卡只跑一个 PyramidFusion"的部署没有直接杠杆(与 v2 §A7 文献调研结论一致)。

### 3.3 异构多加速器调度 (与 Orin GPU∥DLA 直接相关)

- **XAUTO (SJTU IPADS, 2025)**: 面向自动驾驶的细粒度多 XPU(GPU/DLA/…)调度 —— 把模块拆到 stage 级、全局做 XPU 指派与执行排程, 对 ROS2 模块级调度 **e2e 延迟降 1.61×** [R17]。**这是文献对"stage 级异构流水"价值的直接背书** —— 注意其收益来自"不相交物理资源上的全局指派", 不是单 GPU 内的并发。
- **Jetson 调度综述 (2025)**: 系统梳理 Jetson 上跨 CPU/GPU/DLA/PVA 的 DNN 调度技术, 指出自动化框架仍是空白、领域尚不成熟 [R21]。
- **NVIDIA 官方 DLA 实践**: GPU+DLA 并行部署是 Orin 官方推荐的吞吐路径 (Jetson AGX Orin 的 DLA 提供 INT8 算力的显著份额) [R22]。

---

## 4. 核心问题: GPU 能否做到 FPGA 式流水线? (文献定论)

### 4.1 FPGA 流水线的本质 (作为对照基准)

FPGA 数据流加速器 = **空间展开**: 每层综合成专属电路, 层间用片上 FIFO 衔接, 流水满载后每 II 个周期出一个结果(理想 II=1); 延迟被流水隐藏, 中间激活不出片 [R10][R11]。代价: 每层电路独占资源 → 容量受限于片上逻辑/BRAM, 主频低(数百 MHz), 改网络要重综合。

### 4.2 GPU 的四个根本性差异 (文献明确指认)

1. **时间复用 vs 空间分立**: GPU 一个时刻整卡跑一个算子, "大量片上资源闲置" [R1]; FPGA 各层同时工作。GPU 上没有"每 stage 独占资源"的硬件机制 —— Green Contexts 给了 SM 分区但官方明示不保证并发 [R20]。
2. **SIMT 模型与异步硬件错位**: "传统 SIMT 编程模型与现代任务并行 GPU 硬件根本性错位" [R4] —— 硬件已经有数据流味道的异步单元(TMA/WGMMA), 但编程模型默认不暴露流水语义, 需要 warp specialization 这类高级技巧手工/编译器构造。
3. **共享带宽墙**: 所有 SM 共享 L2 与 DRAM 控制器; megakernel 把带宽利用率从 ≤50% 推到 78% 已是 SOTA [R6] —— 并发 stage 抢的是同一条带宽, 这是 FPGA 片上 FIFO 不存在的约束。
4. **kernel 边界语义**: 逐 kernel 启动+同步的执行契约带来 launch 开销(LLM 中可达 e2e 14.6% [R7])与算子间 DRAM 往返; 消除它要么 CUDA Graph(只省 launch), 要么 megakernel(连 DRAM 往返一起省, 但工程代价大)。

### 4.3 三条逼近路径与各自天花板 (全部有实测出处)

| 路径 | 模仿了 FPGA 的什么 | 文献实测收益 | 天花板 |
|---|---|---|---|
| ① Megakernel / persistent kernel [R2][R6][R7] | 整模型单"电路"、算子边界消失、跨算子软件流水 | 1.2–2.5× (vs 强 baseline) | 带宽墙(78% 利用率已是 SOTA); 全部先例是 LLM, conv 检测网无先例 |
| ② Warp specialization + TMA 异步流水 [R4][R5][R9] | kernel 内生产者–消费者流水级 | GEMM 1.5× (TMA) | 只作用于 kernel 内; 已被 TRT/cuBLAS 吸收, 增量有限 |
| ③ 数据流编译 (Kitsune) [R1] | 算子级空间流水(分区 SM 上的并发 stage) | 推理 1.3–2.3× | 作者明言需"适度体系结构调整"才能充分实现 —— **现售 GPU 不改硬件做不满** |

### 4.4 定论 (回答用户问题)

1. **形态上可以逼近, 数量级上不能等同**。GPU 通过 megakernel/warp-spec/数据流编译能构造出"软件数据流", 但全部文献实测收益在 **1.1×–2.5×** 区间; FPGA 空间流水的标志性优势(II=1、零算子边界、片上全驻留)受 GPU 的时间复用+共享带宽+SIMT 语义三重硬约束, **在不改硬件的前提下无法完全实现** [R1][R4]。
2. **FPGA 的真实优势区间比宣传窄**: 系统性对比(TRETS/FCCM'24)显示 FPGA 空间加速在 **batch-1 memory-bound 的 LLM decode** 上对 A100 速度 1.9×、**能效 5.7×**, 但 prefill/吞吐场景 GPU 占优 [R11]。即 FPGA 赢在"小 batch 延迟敏感 + 能效", 不是普遍更快。
3. **对单模型单帧延迟, 调度类技术(streams/MPS/MIG/软件调度器)不是杠杆** —— 它们面向多模型共享 [R13][R14][R15]; 单模型想要数量级延迟改善, 文献只支持 work-reduction(剪枝/量化/蒸馏)与更强硬件。
4. **stage 级流水的文献正解 = 不相交物理资源上的异构指派**(XAUTO 1.61× [R17]; NVIDIA GPU+DLA 官方路径 [R22]), 而非单 GPU 内分区并发(Green Contexts 官方不保证并发 [R20])。

---

## 5. 与本仓库实测对账 (文献 ↔ v2 实测, 互证无冲突)

| 本仓库实测 (v2) | 文献对应 | 一致性 |
|---|---|---|
| 单 GPU stage 流水峰值 **1.08×**, 出帧间隔距 max(stage) 差 3.6× (E_pipeline) | Green Contexts 官方"SM 分区不保证并发" [R20]; Orion 指认共享资源干扰 [R14]; Jetson 剖析"GPU util 100% 但 SM 利用 15–30%"却难兑现 [R18] | ✅ "occupancy 槽位 ≠ 可并发吞吐空隙"被官方文档与文献双重坐实 |
| multi-stream 吞吐峰值 1.13×、单帧延迟劣化 (E1) | Paella: 硬件 FIFO 调度 + HoL 阻塞 [R13]; Opara: 顺序 conv 链算子并行收益 1.1–1.3× [R16] | ✅ 量级吻合 |
| CUDA Graph 1.05–1.10× | Ada-MK: launch 开销占 e2e ≤14.6% (LLM, 百万级 launch) [R7] —— 本仓库子模块 kernel 数少, 占比更低 | ✅ 机制一致, 占比随 kernel 数缩放 |
| 异构 GPU∥DLA 双进程 **1.34×** 真并行; 进程内双流 1.00× (E3) | XAUTO: 细粒度多 XPU 指派 1.61× [R17]; NVIDIA 官方 GPU+DLA 并行路径 [R22] | ✅ "不相交物理资源才有真流水"是文献正解; 1.34× 在文献量级内 |
| INT8/剪枝是延迟数量级杠杆, 调度不是 (v2 §0.5 裁决) | 全部流水/调度文献收益 1.1–2.5× [R1][R2][R6][R16]; FPGA 能效优势也来自定制数据通路而非调度 [R11] | ✅ |
| **本仓库未做、文献支持可做的增量** | ① megakernel 化 pyramid_backbone(无 conv-detection 先例, 工程量大, 预期收益参照 Kitsune 推理段 1.3–2.3× 上界打折) [R1][R2]; ② L2 persisting 钉住小权重 [R19]; ③ Orin 上 stage 级 GPU∥DLA 全局指派的系统化(XAUTO 式) [R17] | ⏳ 均为可选 future work, 优先级低于量化×剪枝主线 |

> **对论文叙事的供给**: related-work 段可直接引用 §4 的对照逻辑 —— "GPU 软件数据流(megakernel/warp-spec)收益 1.1–2.5× [R1][R2][R6], FPGA 空间流水优势集中于 batch-1 能效 [R11]; 因此本框架把延迟数量级收益寄于剪枝×量化协同, 把流水化收益限定为异构 GPU∥DLA 指派(实测 1.34×, 与 XAUTO 1.61× 同量级 [R17])"。

---

## R. 引用列表 (全部经核验, 2026-06-05)

| # | 出处 (题目 / 作者 / venue / 年 / 检索号) | 核验方式 |
|---|---|---|
| R1 | **Kitsune: Enabling Dataflow Execution on GPUs**. M. Davies, N. Crago, K. Sankaralingam, S. W. Keckler. arXiv:2502.18403, 2025. | 摘要核验 ✅ (1.3–2.3× 推理 / 1.1–2.4× 训练; "temporal multiplexing 致资源闲置"; "需适度体系结构调整") |
| R2 | **Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs**. X. Cheng, Z. Zhang, …, T. Chen, Z. Jia (CMU 等). arXiv:2512.22219, 2025. | 摘要核验 ✅ (首个多 GPU 自动 megakernel; SM 级依赖图; LLM 延迟降至 1.7×) |
| R3 | **FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving**. Z. Ye et al. MLSys 2025. arXiv:2501.01005. | 摘要核验 ✅ (load-balanced 调度兼容"要求静态配置的 CUDA Graph") |
| R4 | **Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References**. H. Chen, …, Z. Zhang, V. Grover (NVIDIA/Cornell). CGO 2026. arXiv:2510.14719. | 摘要核验 ✅ (SIMT 与任务并行硬件"根本性错位"; aref IR 自动 warp 角色划分) |
| R5 | **Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs (Twill)**. R. Soi, R. Yadav, F. Kjolstad, A. Aiken, …, M. Bauer. arXiv:2512.18134, 2025. | 摘要核验 ✅ (SWP+WS 联合约束求解, 可证最优; 重发现 FlashAttention 调度) |
| R6 | **Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B**. B. Spector, …, C. Ré (Stanford Hazy Research). 技术博客, 2025-05-27. https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles | 原文核验 ✅ (vLLM/SGLang ≤50% H100 带宽 → megakernel 78%; H100 ~2.5×/B200 <680µs)。注: 博客非 peer-reviewed, 引用时标注 |
| R7 | **Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference**. W. Dong et al. arXiv:2605.11581, 2026. | 摘要核验 ✅ (launch 开销占 e2e 14.6%; vs TensorRT-LLM +23.6% / vLLM +50.2% 单 batch) |
| R8 | **Fusing Depthwise and Pointwise Convolutions for Efficient Inference on GPUs (FCM)**. F. Qararyah, M. W. Azhar, M. A. Maleki, P. Trancoso. arXiv:2404.19331, 2024. | 摘要核验 ✅ (省 83% 访存; vs cuDNN 最高 3.7×; DW/PW 卷积 memory-bound) |
| R9 | **Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis**. W. Luo, …, Q. Wang, X. Chu. arXiv:2501.12084, 2025. | 摘要核验 ✅ (TMA 异步模型 GEMM 1.5×) |
| R10 | **FINN: A Framework for Fast, Scalable Binarized Neural Network Inference**. Y. Umuroglu et al. FPGA 2017. arXiv:1612.07119. | 摘要核验 ✅ (异构流式架构, per-layer 专属计算资源按吞吐需求定制) |
| R11 | **Understanding the Potential of FPGA-Based Spatial Acceleration for Large Language Model Inference**. H. Chen, …, Z. Zhang (Cornell). ACM TRETS (FCCM'24 journal track), 2024. arXiv:2312.15159. | 摘要核验 ✅ (decode 段 vs A100: 1.9× 速度、5.7× 能效; prefill 段 GPU 占优) |
| R12 | (并入 R1/R4 的体系结构论断, 不单列) | — |
| R13 | **Paella: Low-latency Model Serving with Software-defined GPU Scheduling**. K. K. W. Ng, H. M. Demoulin, V. Liu. SOSP 2023. DOI 10.1145/3600006.3613163. | DOI+题录核验 ✅ (HoL 阻塞、FIFO 内建调度、用户态逐 kernel 调度; PDF 正文未抽取, 数字引用从略) |
| R14 | **Orion: Interference-aware, Fine-grained GPU Sharing for ML Applications**. F. Strati, X. Ma, A. Klimovic (ETH). EuroSys 2024. DOI 10.1145/3627703.3629578. 代码: github.com/eth-easl/orion | DOI+题录核验 ✅ (干扰感知 10µs 级共享; 时间切片/MIG/MPS 局限论断) |
| R15 | **Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences (REEF)**. M. Han et al. (SJTU IPADS). OSDI 2022. 代码: github.com/SJTU-IPADS/reef | venue 级核验 ⚠️ (usenix 页访问受限; OSDI'22 题录与官方代码库公开可查) |
| R16 | **Opara: Exploiting Operator Parallelism for Expediting DNN Inference on GPUs**. A. Chen, F. Xu, et al. arXiv:2312.10351, 2023 (rev. 2024). | 摘要核验 ✅ (vs 串行 CUDA Graph 最高 1.68×; vs SOTA 算子并行 1.29×) |
| R17 | **Holistic Heterogeneous Scheduling for Autonomous Applications using Fine-grained, Multi-XPU Abstraction (XAUTO)**. M. Han, W. Shen, R. Chen, B. Zang, H. Chen (SJTU IPADS). arXiv:2508.09503, 2025. | 摘要核验 ✅ (stage 级多 XPU 指派, 对 ROS2 模块级 e2e 延迟 1.61×) |
| R18 | **Profiling Concurrent Vision Inference Workloads on NVIDIA Jetson — Extended**. A. Chakraborty et al. arXiv:2508.08430, 2025. | 摘要核验 ✅ (GPU util 100% 时 SM/TC 实际利用 15–30%; CPU 侧调度常是约束) |
| R19 | **CUDA C Programming Guide — L2 Cache Control / set-aside for persisting accesses (accessPolicyWindow)**. NVIDIA 官方文档 (CUDA 11+/Ampere 引入)。https://docs.nvidia.com/cuda/cuda-c-programming-guide/ | 文档页核验 ✅ (章节存在确认; 细节以当前版本文档为准) |
| R20 | **CUDA Driver API — Green Contexts**. NVIDIA 官方文档 (CUDA 12.4+)。https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html | 文档核验 ✅ (含关键 caveat: 不相交 SM 分区**不保证**并发/前进性) |
| R21 | **Scheduling Techniques of AI Models on Modern Heterogeneous Edge GPU — A Critical Review**. A. A. Majeed, M. Meribout. arXiv:2506.01377, 2025. | 摘要核验 ✅ (Jetson CPU/GPU/DLA/PVA 调度综述; 自动化框架空白) |
| R22 | **Maximizing Deep Learning Performance on NVIDIA Jetson Orin with DLA**. NVIDIA Developer Blog. https://developer.nvidia.com/blog/maximizing-deep-learning-performance-on-nvidia-jetson-orin-with-dla/ | 官方博客 (deep-research 抓取源; GPU+DLA 并行为官方推荐路径) |

> **核验局限如实声明**: ① R13/R14 的 PDF 抓取为二进制未抽取正文, 故正文只引用其检索确认过的核心论断, 未引用具体数字(如 Paella 的 goodput 倍数); ② R15 仅 venue 级核验; ③ R6 为非 peer-reviewed 技术博客; ④ 本次调研的自动对抗核验环节因 harness 故障未生效(全部 abstain), 上表核验均为主控逐条人工(摘要级)复核。引用任何数字进论文前, 建议再读原文对应章节。
