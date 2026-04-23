# 硬件感知联合搜索调研：NAS + 部署配置

> 本调研聚焦一个关键问题：**除了把硬件延迟作为 NAS 约束的传统 HW-NAS 之外，哪些工作真正把"部署配置维度"（D 空间）纳入了自动化搜索？**
>
> 参考体系：A(网络架构) / B1(压缩策略) / B2(数值表示) / C(硬件架构) / D(部署配置) / E(编译映射)
>
> 我们方案的定位是 B1(剪枝) + B2(量化，标准格式) ↔ D(部署配置) 在固定硬件（Orin/GPU）上的联合搜索。本调研核查"D 空间入搜"在业界的覆盖情况。

---

## 1. 联合搜索的层次分类

把"硬件纳入搜索"这件事按耦合紧密度划分，至少有四层：

| 层次 | 形式 | 搜索变量 | 评估方式 | 代表工作 |
|------|------|---------|----------|----------|
| L0 硬件代理约束 | 仅把延迟/能耗作为外部约束（单向） | A 子空间 | 延迟 LUT / 代理模型 | ProxylessNAS, HAT, MCUNet, OFA |
| L1 压缩+架构联合 | A ↔ B1/B2 在同一搜索器里 | A + B1 + B2 | 量化感知精度预测器 | APQ, BatchQuant |
| L2 算法+硬件架构联合 | 同时搜网络和加速器设计 | A + B + C + E | Timeloop / 自定义模拟器 | NAHAS, NASAIC, DANCE, Auto-NBA, JAQ |
| L3 部署配置入搜 (D) | 搜索运行时参数/调度/内存/精度策略 | D (+E) | 真实/仿真 profile | Vidur-Search, FlexGen, DistServe, Ansor |

**关键区分线：**
- L0/L1 只改变"模型长什么样"，硬件侧只是评估函数；
- L2 必须触及硬件架构参数（PE 数、buffer 容量、NoC 拓扑），要求定制芯片可塑；
- **L3 才是"D 空间入搜"**：硬件物理架构已定，搜索的是"如何在固定硬件上部署"，包含精度执行策略、算子实现选择、内存管理、调度、图分割、批处理、时序缓存、异构卸载、存储放置、服务调度等运行时决策。

我们方案位于 L3 且与 L1 耦合——既搜剪枝/量化，也搜部署配置，但不搜 A 和 C。

---

## 2. 经典 HW-NAS 工作回顾（L0，作为参照系）

这类工作只把硬件作为 NAS 的约束/评估函数，搜的都是 A（架构），**D 空间不出现**。

### 2.1 ProxylessNAS (Cai et al., ICLR 2019)
- 将架构搜索目标直接设为目标硬件上的测量延迟（Pixel 手机/GPU）。
- 贡献：消除代理数据集、代理网络，直接在目标任务+目标硬件上搜索。
- D 空间覆盖：**无**。硬件只是外部黑盒 latency oracle。

### 2.2 HAT: Hardware-Aware Transformers (Wang et al., ACL 2020)
- 针对 Transformer 设计大搜索空间（任意 encoder-decoder attention、异构层）。
- 用延迟预测器替代真实测量。
- D 空间覆盖：**无**。

### 2.3 MCUNet (Lin et al., NeurIPS 2020)
- TinyNAS + TinyEngine 协同。TinyNAS 两阶段：先优化搜索空间以匹配资源约束，再搜网络；TinyEngine 做 memory scheduling 优化。
- 接近 D 空间：TinyEngine 的 memory scheduling 是 D 的一部分，但 MCUNet 的搜索器本身并不把 memory schedule 作为决策变量一起搜索，TinyEngine 的优化是给定网络之后的后处理。
- D 空间覆盖：**部分触及但未入搜**。

### 2.4 Once-for-All (OFA, Cai et al., ICLR 2020)
- 一次训练支持 >1e19 个子网，部署时按硬件约束快速挑选子网。
- 硬件平台覆盖广（Pixel、Note、1080Ti、V100、TX2、CPU）。
- D 空间覆盖：**无**。deployment 只是从训练好的 super-net 里选一个子网，没有部署参数决策。

### 2.5 HELP (Lee et al., NeurIPS 2021)
- 延迟预测器的元学习方法，减少新硬件上训练 latency predictor 的样本量。
- 改进的是 L0 的评估工具，**D 空间无涉**。

**小结：** L0 这一批工作的共同特征是——硬件可感知，但硬件完全是被动评估的对象，搜索决策只关乎网络架构。这正是我们方案要超越的范式。

---

## 3. 架构+压缩联合搜索（L1）

### 3.1 APQ (Wang et al., CVPR 2020) — 最接近我们 B1+B2 联合的先驱

APQ = Architecture + Pruning + Quantization 联合搜索。

- **搜索空间：** A（宽度/深度/kernel）+ B1（通道剪枝率）+ B2（逐层 INT 位宽）。
- **核心贡献：** 训练一个"量化感知精度预测器"，通过从 FP32 精度预测器迁移知识大幅降低样本成本。
- **评估：** 目标硬件延迟 LUT（与 HAQ 同构）。
- **D 空间覆盖：** **无**。D 不在搜索空间里，硬件仍是外部约束。
- **与我们差异：** APQ 搜 A+B1+B2，我们搜 B1+B2+D。APQ 的"D 维度"被视为不可变的运行时环境。

### 3.2 HAQ (Wang et al., CVPR 2019 Oral)
- 用 DDPG 强化学习搜索每层量化位宽（1–8 bit），把硬件加速器的 feedback 作为 reward。
- 延迟减少 1.4–1.95x、能耗减少 1.9x，相比固定 8bit。
- **关键发现：** 不同硬件架构（云/边）下最优量化策略差异巨大——这给出了"策略需与硬件绑定"的经验证据，但 HAQ 本身硬件固定，不搜 D。
- **D 空间覆盖：** **无**（位宽在 B2，延迟是 reward 不是决策变量）。

### 3.3 BRECQ / LSQ / HAWQ / MixQuant — 纯量化搜索
- BRECQ：PTQ 把位宽极限下推到 INT2，块级重建。
- HAWQ / HAWQ-v2：用 Hessian trace 作为敏感度指标自动分配位宽，Pareto 前沿法。
- LSQ：学习步长的量化。
- MixQuant：用 roundoff error 在 {int2..int8} 中搜索每层位宽。
- **D 空间覆盖：** **无**。全部聚焦 B2 单侧。

### 3.4 BatchQuant (Bai et al., NeurIPS 2021)
- Quantized-for-all：一次训练产生鲁棒量化的 super-net，部署时按位宽约束选子网。
- **D 空间覆盖：** **无**。与 OFA 思想相同，只把"位宽"加入可选维度。

---

## 4. 算法+硬件架构联合搜索（L2）

这类工作搜索包含了 C（硬件架构），不是我们的方向（我们硬件固定），但值得对比——它们也**不搜 D**，因为 D 在硬件设计阶段还无意义。

### 4.1 NAHAS (Zhou et al., ICLR 2022 workshop / arXiv 2102.08619)
- 联合搜索 neural architecture 与 accelerator configuration（PE 阵列、buffer 大小）。
- 在所有 latency target 上比 EfficientNet 高 0.5–1% top-1，同时延迟降 20%。

### 4.2 NASAIC (Yang et al., DAC 2020)
- 多任务同时搜多个 DNN 架构 + 对应的异构 ASIC 加速器设计。
- 结果：延迟降 17.77%、能耗降 2.49x、面积降 2.32x。

### 4.3 DANCE (Choi et al., DAC 2021)
- 可微化 accelerator/network co-exploration，用 evaluator network 作为连续代理。

### 4.4 Auto-NBA (Fu et al., ICML 2021)
- Networks + Bitwidths + Accelerators 三元联合。
- 挑战三连：搜索空间爆炸、加速器的离散性、网络-加速器的鸡生蛋问题。
- 解法：异构采样 + 可微 accelerator search engine。

### 4.5 JAQ (Wang et al., AAAI 2025) — 最新 L2 代表
- Architecture + ultra-low-bit mixed precision + accelerator 三元联合。
- 新增 **Channel-wise Sparse Quantization (CSQ)** 解决 QAT 的显存爆炸。
- **BatchTile** 编码所有可能 tiling 模式，加速编译映射搜索（接近 E 空间）。
- ImageNet Top-1 比以往方法高约 7%，硬件搜索每次迭代 0.15s。
- **D 空间覆盖：** BatchTile 触及 E（tiling 映射），但 D 仍然不是决策维度——硬件是被设计出来的、不是被"配置"的。

**小结：** L2 工作统一特征是 C 参与搜索、E 可能部分参与，但 **D 从不参与**——因为它们都是"设计新芯片"范式，D 在这个范式里无意义。

---

## 5. LLM 推理搜索（首次大规模出现 D 空间入搜）

LLM 服务的复杂度迫使研究者真正把部署参数当作决策变量，这是 D 空间入搜最活跃的领域。

### 5.1 Vidur & Vidur-Search (Agrawal et al., MLSys 2024)

**最接近我们方法论的工作之一。**

- Vidur 本身：LLM 推理的大规模仿真框架（算子级 profiling + 预测模型），延迟/吞吐估计误差 <5%。
- **Vidur-Search：** 自动搜索"数百种部署配置"找 throughput/cost 最优点。LLaMA-70B 的最优部署搜索在 CPU 机器上 1 小时完成，而真实部署 sweep 需要 42K GPU 小时（约 $218K）。
- **搜索的 D 维度：** 并行度 (TP/PP)、GPU 类型选择、batch size、chunk size、副本数。
- **D 空间覆盖：** **高**。但搜索空间仍围绕 LLM 服务，不覆盖精度执行策略、kernel 选择。
- **与我们共性：** "仿真驱动搜索"的方法论高度一致——都用低代价仿真替代高代价真实部署。
- **与我们差异：** Vidur 不搜 B（量化/剪枝），搜索单元是"部署配置"；我们在 B↔D 之间建立反馈。

### 5.2 FlexGen (Sheng et al., ICML 2023)

- 在单 GPU 上做高吞吐 LLM 推理，关键挑战是存储层次（GPU/CPU/Disk）的数据放置。
- **用线性规划（LP）搜索 offloading 策略**，搜索空间正式定义为 {computation schedule, tensor placement, computation delegation}；证明搜到的方案 I/O 复杂度在最优 2× 内。
- 附带把权重和 KV cache 压到 4bit（B2）。
- **D 空间覆盖：** **高**。存储放置、计算调度是 D 的核心子空间。
- 是最早把"存储放置 + 调度"规约为可搜索数学问题的工作之一。

### 5.3 DistServe (Zhong et al., OSDI 2024)

- 把 prefill 和 decode 分配到不同 GPU，消除 prefill-decode 干扰。
- **搜索的 D 维度：**
  1. prefill / decode 各自的并行策略 (TP/PP)
  2. 每种实例的副本数
  3. 集群物理放置（对齐带宽以最小化 KV cache 传输）
- 给定模型/workload/SLO，自动决定上述三者。
- 结果：7.4× 更多请求 or 12.6× 更紧 SLO。
- **D 空间覆盖：** **中高**。"图分割 + 异构放置 + 服务调度"三个子空间都入搜。

### 5.4 Splitwise (Patel et al., ISCA 2024)

- 与 DistServe 思想相近：prefill 重计算、decode 重内存。
- **搜索/决策的 D 维度：** prefill 阶段用 H100、decode 用 A100（**异构卸载**），KV cache 跨机传输。
- 1.4× 更高吞吐 with 20% 成本降低；或 2.35× 吞吐 同成本。
- **D 空间覆盖：** **中**。主要是"异构硬件分工"决策，搜索粒度较粗（阶段级）。

### 5.5 Sarathi / Sarathi-Serve (Agrawal et al., OSDI 2024)

- Chunked prefill + decode-maximal batching。
- **搜索的 D 维度：** chunk size (128/256/512)、batch size、P:D 比例。
- 发现峰值性能在 P:D ≈ C/(B-1) 及 tile-size 倍数处——**可搜索公式化**的部署参数关系。
- **D 空间覆盖：** **中**。主要是"批处理策略"子空间。

### 5.6 vLLM 参数调优（工程实践层面）

- 非研究论文，但重要：PagedAttention + continuous batching 的核心参数 --gpu-memory-utilization、max_num_batched_tokens、enable_chunked_prefill 是典型的 D 维度。
- vLLM 官方提供 benchmarks/auto-tuning 工具做参数扫描。
- **D 空间覆盖：** **高**，但**不是联合搜索**——与模型结构/量化解耦。

### 5.7 Llumnix (Sun et al., OSDI 2024)

- "LLM 里的进程调度器"：跨多个模型实例动态重调度请求（负载均衡、碎片整理、优先级、自动扩缩容）。
- 基于实时迁移机制。
- **D 空间覆盖：** **中**。主要在"服务调度"子空间，偏运行时而非离线搜索。

---

## 6. 编译器级搜索（E 空间，对我们 D 空间是邻居）

这些是 E 空间的代表工作，部分涉及 D（如内存复用）。

### 6.1 Ansor / TVM Auto-scheduler (Zheng et al., OSDI 2020)

- 自动为 tensor expression 生成高性能代码，去掉 AutoTVM 的手写 template 依赖。
- **分层搜索空间：** sketch（高层结构） + annotation（tile size、parallel、unroll）。
- 进化搜索 + 学习的 cost model。
- **D vs E：** Ansor 搜 E（循环结构、分块、并行），但 tile size 之类的决策同时也是部署侧的（memory footprint、并行度）。
- **与我们的关系：** 我们的 D 空间"算子实现选择"可以看作 Ansor/MetaSchedule 生成结果集合上的二次选择——但 Ansor 不跨模型做联合优化。

### 6.2 MetaSchedule (Shao et al., NeurIPS 2022)

- TVM 第三代自动调度系统，统一 AutoTVM 和 Ansor。
- 概率调度 DSL：composite schedule → design space → stochastic search。
- 支持 tensorization、loop partitioning 等高级 primitive。
- **D 空间覆盖：** 仍然是 E 空间主体。

### 6.3 Mirage (Wu et al., OSDI 2025)

- **Multi-Level Superoptimizer for Tensor Programs**——最新且最野心勃勃的 E 空间搜索。
- 核心：µGraph 抽象统一 kernel / thread block / thread 三层 GPU 计算层次。
- 发现"跨层融合优化"：代数变换 + 调度变换 + 新 custom kernel 生成。
- 基于抽象的剪枝 + 概率等价性验证。
- 对 LLM 大模型显著超越现有方案。
- **D 空间覆盖：** **触及**——生成 megakernel 本身就是 E+D 的联合（kernel 边界 = 执行调度策略）。
- 启发：我们的 D 空间"算子实现选择"可以被 Mirage 的 µGraph 抽象进一步延伸为"跨层融合策略"。

### 6.4 MLGO (Trofin et al., 2021)

- 用 ML 策略（policy gradient + evolution strategies）替换 LLVM inliner 的启发式。
- 二进制大小降低 7%（相对 LLVM -Oz）。
- **D 空间覆盖：** **无**，是传统编译优化的 ML 化，与 DNN 部署关系弱，但方法论（策略网络 + RL-on-compiler）值得借鉴。

---

## 7. 多模型 / Pipeline 调度搜索

这些工作处理"多模型共享硬件"场景，D 空间的"服务调度"子空间。

### 7.1 Nexus (Shen et al., SOSP 2019)

- 视频分析 GPU 集群引擎。
- **搜索/决策：** squishy bin packing（哪些模型共置到同一 GPU）、prefix batching（模型变体共享前缀）、batch 大小。
- 比 SOTA 快 1.8–12.7×（99% SLO 满足前提下）。
- **D 空间覆盖：** **中**。"异构卸载 + 批处理 + 图共享"多维综合。

### 7.2 Clockwork (Gujarati et al., OSDI 2020)

- 核心思想：DNN 推理延迟可预测 → 用确定性换 SLO。
- 集中式 scheduler，消除反应式、best-effort 机制。
- 千模型规模下 99.997% 请求满足 100ms SLO。
- **D 空间覆盖：** **低**。是"调度策略"本身不做搜索，而是用可预测性替代搜索。

### 7.3 REEF (Han et al., OSDI 2022)

- GPU 并发 DNN 推理的 µs 级 preemption。
- **决策：** RT vs BE 任务分类、Reset-based Preemption、Dynamic Kernel Padding（填充未占用 CU）。
- **D 空间覆盖：** **中**。"执行调度"的高级形态，但 REEF 本身是运行时机制，不是离线搜索。

### 7.4 AlpaServe (Li et al., OSDI 2023)

- 发现：即使模型能放进单 GPU，模型并行也能通过统计复用降低多模型服务延迟。
- **搜索的 D 维度：** 多模型的 placement 策略（哪些模型切到哪些 GPU）、并行方案。
- 10× 更高请求率 or 6× 更 bursty。
- **D 空间覆盖：** **高**。是 D 空间多模型版本的代表。

### 7.5 Sia (Subramanya et al., SOSP 2023)

- 异构 GPU 集群上的 goodput-optimal 调度。
- **搜索变量：** elastic/hybrid parallel job 的配置与 GPU 类型/数量的匹配。
- 首个支持 hybrid parallel job 的弹性调度器。
- JCT 降低 30–93%。
- **D 空间覆盖：** **高**（针对训练任务，但思想可迁移）。

---

## 8. 自动驾驶/边缘部署搜索（较稀缺）

调研发现：**自动驾驶/BEV 感知领域做自动化部署搜索的工作非常少**，主要是人工调优 + 固定 pipeline。

### 8.1 RT-BEV (Liu, Lee, Shin, RTSS 2024)

- **实时 BEV 感知的第一个 co-optimization 工作**。
- 组件：Camera Synchronizer（ROI-aware 同步）+ ROIs Generator（上下文感知 ROI）+ Feature Split & Merge（关键区域高精度、非关键区域复用时序）+ Time Predictor（TTC 驱动的优先级调整）。
- 端到端延迟降 1.5×、最坏情况 19.3×、FES 2.9×。
- **D 空间覆盖：** **中**。"时序缓存 + 批处理（ROI 级）+ 执行调度"的人工协同；**不是搜索，是规则化的运行时决策**。

### 8.2 BEVFusion (Liu et al., ICRA 2023) + TensorRT 部署

- NVIDIA 提供 BEVFusion 的 TensorRT 方案，Jetson Orin 上 25 FPS。
- 部署决策全部是 NVIDIA/社区的人工优化——典型的 D 空间"手动调优"。
- **D 空间覆盖：** 手动优化，非搜索。

### 8.3 Edge-Based VLM 融合（arXiv 2025, Bhaveshkumar et al.）

- Jetson Orin Nano / AGX Xavier 上做离线量化 + attention 增强 + token 长度削减。
- 同样是人工调优。

**关键结论：** **自动驾驶/BEV 领域的 D 空间搜索几乎是空白**。这正是我们方案的稀缺定位——**第一个在 V2X 协同感知场景把 B1+B2 ↔ D 纳入自动化联合搜索**。

---

## 9. 各工作的 D 空间覆盖对比表

D 空间内部子维度（引用我们体系化总结）：
- D1 精度执行策略 (force_fp16/fp32/follow_quant)
- D2 算子实现选择 (trt_native/flash_attn/custom)
- D3 内存管理 (isolated/shared)
- D4 执行调度 (pipeline/branch_parallel)
- D5 图分割 (monolithic/split)
- D6 批处理 (pad/dynamic/bucketed)
- D7 时序缓存 (frames × precision)
- D8 异构卸载 (DLA/CPU/GPU)
- D9 存储放置 (KV/weight placement)
- D10 服务调度 (batch/phase 调度)

| 工作 | A | B1 | B2 | C | D 子空间覆盖 | E | 耦合形态 |
|------|:-:|:-:|:-:|:-:|:------------|:-:|---------|
| ProxylessNAS | 搜 | - | - | - | — | - | L0 单向 |
| OFA | 搜(super-net) | - | - | - | — | - | L0 单向 |
| HAT | 搜 | - | - | - | — | - | L0 单向 |
| MCUNet | 搜 | - | 固定INT8 | - | 接近 D3 (memory schedule) | 搜(TinyEngine) | L0+A↔E |
| **APQ** | **搜** | **搜** | **搜** | - | — | - | L1 A+B1+B2 |
| HAQ | - | - | 搜 | - | — | - | B2 单侧 |
| BRECQ/HAWQ/MixQuant | - | - | 搜 | - | — | - | B2 单侧 |
| **NAHAS/DANCE** | **搜** | - | - | **搜** | — | - | L2 A+C |
| **NASAIC** | **搜** | - | - | **搜** | — | 部分 | L2 多任务 A+C |
| **Auto-NBA** | **搜** | - | **搜** | **搜** | — | - | L2 A+B2+C |
| **JAQ (2025)** | **搜** | - | **搜** | **搜** | — | 搜(BatchTile) | L2+E |
| **Vidur-Search** | - | - | - | - | **D5/D6/D8/D10** (TP/PP/batch/副本) | - | L3 纯 D 搜索 |
| **FlexGen** | - | - | 固定 4bit | - | **D3/D4/D9** (placement+schedule, LP 求解) | - | L3 纯 D 搜索 |
| **DistServe** | - | - | - | - | **D5/D8/D10** (prefill/decode 分离) | - | L3 纯 D 搜索 |
| **Splitwise** | - | - | - | - | **D8** (异构 GPU 分工) | - | L3 纯 D |
| **Sarathi** | - | - | - | - | **D6** (chunk size + P:D) | - | L3 纯 D |
| Llumnix | - | - | - | - | **D10** (动态 reschedule) | - | L3 运行时 |
| vLLM tuning | - | - | - | - | **D3/D6** (gpu-mem-util/chunk) | - | L3 工程调参 |
| Ansor/MetaSchedule | - | - | - | - | 触及 D2 | **搜** | E 主体 |
| Mirage (OSDI 25) | - | - | - | - | 触及 D2/D4 (megakernel) | **搜** | E+部分 D |
| Nexus | - | - | - | - | **D6/D8** | - | L3 多模型 D |
| Clockwork | - | - | - | - | **D10** (确定性调度，非搜索) | - | 运行时 |
| REEF | - | - | - | - | **D4/D10** (preemption) | - | 运行时 |
| **AlpaServe** | - | - | - | - | **D5/D8/D10** (placement 搜索) | - | L3 多模型 D |
| **Sia** | - | - | - | - | **D8/D10** (异构匹配) | - | L3 (训练) |
| **RT-BEV** | - | - | - | - | **D7/D6/D4** (手工协同) | - | 人工 D 协同 |
| **我们的方案** | - | **搜** | **搜(标准格式)** | - | **D1-D10 多子空间入搜** | 外层固定 | **L1+L3 紧耦合** |

---

## 10. 关键洞察与差异化定位

### 10.1 D 空间入搜的"两个出口"

调研发现，**真正把 D 当成决策变量**的工作集中在两个场景：

1. **LLM 服务**（Vidur, FlexGen, DistServe, Splitwise, Sarathi, AlpaServe, Llumnix）
   - 驱动力：模型太大，单机部署有大量权衡空间（TP/PP/chunk/placement）。
   - 特点：几乎都**只搜 D**，不联合 B1/B2。量化/剪枝被视为与服务策略解耦的独立模块。

2. **多模型 GPU 集群调度**（Nexus, Clockwork, REEF, AlpaServe, Sia）
   - 驱动力：服务级别的资源复用和 SLO。
   - 特点：粒度更粗（任务/请求级），模型内部 D 参数通常不搜。

### 10.2 "B ↔ D 联合"是未被占领的高地

从对比表可以清晰看到一个空白区域：

```
                    B1/B2 单独搜索          B+D 联合搜索           D 单独搜索
  传统 CV 模型     HAQ, BRECQ, MixQuant    【空白】              【空白】
  LLM                AWQ, GPTQ             【空白】              Vidur, FlexGen, ...
  自动驾驶/BEV      QD-BEV, Q-PETR         【空白】              【空白】
  V2X 协同感知      QuantV2X               【我们的方案】        【空白】
```

**业界没有在传统感知模型/V2X 场景把 B1+B2 和 D 在同一个搜索器里联合优化的先例。** APQ 虽然搜 A+B1+B2，但仍是固定 D；Vidur 等虽然搜 D，但只面向 LLM 服务、不搜 B。

### 10.3 我们的独特贡献

从搜索空间视角看，我们的方案填补了这些空白：

1. **第一个在固定硬件（非 LLM 场景）把 D 多子空间 (D1-D10) 入搜的工作**：
   - D1（精度执行策略）：业界普遍由 TensorRT 自动选择，没人搜。
   - D2（算子实现选择）：vLLM 有 flash_attn vs naive 的选项，但不是联合搜索维度。
   - D7（时序缓存 frames × precision）：RT-BEV 做了手工版本，我们做自动化版本。
   - D8（异构卸载 DLA）：Jetson Orin 的 DLA offload 集合从无人自动搜。

2. **第一个 B1+B2+D 联合**：
   - APQ 是 A+B1+B2，但 D 缺席；我们砍掉 A（沿用固定 backbone）、补上 D。
   - 相比 L2 工作（NAHAS/JAQ）：我们不需要设计硬件，搜索成本低一个数量级。
   - 相比 LLM-only 工作（Vidur 等）：我们覆盖了他们不触及的感知域。

3. **第一个在 V2X 协同感知做自动化联合搜索**：
   - 通信特征的压缩（B1/B2 的特殊形式，Comm-B）与 D 侧 timing/bandwidth 耦合，是纯 LLM 或纯边缘 NAS 都没遇到的场景。

### 10.4 未来可融合的方向

从调研可借鉴的机制：

| 我们可以借鉴的 | 来源 | 应用在哪 |
|---------------|------|---------|
| 仿真 + 搜索的分离 | Vidur-Search | D 空间的快速评估器 |
| 可微评估器 | DANCE | B↔D 搜索的梯度化 |
| 迁移精度预测器 | APQ | B+D 联合空间的 sample-efficient 预测 |
| LP 求解 placement | FlexGen | D9（存储放置）子空间的显式建模 |
| Phase splitting | DistServe/Splitwise | 是否把感知 pipeline 的 backbone/head 做 prefill/decode 式拆分？ |
| µGraph 抽象 | Mirage | 提升 D2（算子实现选择）的表达能力 |
| Hessian sensitivity | HAWQ | 用于 B2 位宽的预筛选，减小 B+D 联合空间 |

---

## 11. 结论

硬件感知 NAS 这 6 年的发展可以简述为 L0 → L1 → L2 的演进（架构 → +压缩 → +硬件）。**L3（部署配置 D 入搜）是 LLM 服务时代才真正兴起的新范式，但目前只在 LLM/集群调度场景成熟，传统感知模型特别是自动驾驶/V2X 场景是空白。**

我们方案的学术定位可以概括为：

> **在固定硬件 + 标准数值格式的前提下，首次把部署配置 (D) 与模型压缩 (B1+B2) 放入统一的自动化联合搜索，面向 V2X 协同感知场景验证"第三条紧耦合路径" B↔D。**

这条路径相比已有两条（B2↔C 新格式、E↔C 编译映射）的独特优势是：**不需要设计新硬件、不需要发明新格式、搜索评估可以完全基于仿真和真实 profile 的混合**——这对工业界特别是自动驾驶领域的落地价值是决定性的。

---

## 参考文献（关键 URL）

- APQ: https://arxiv.org/abs/2006.08509
- HAQ: https://arxiv.org/abs/1811.08886
- HAWQ-V2 / BRECQ / MixQuant: https://arxiv.org/abs/2309.17341
- ProxylessNAS: https://hanlab.mit.edu/projects/proxylessnas
- HAT: https://hanlab.mit.edu/projects/hat
- MCUNet: https://arxiv.org/abs/2007.10319
- OFA: https://arxiv.org/abs/1908.09791
- NAHAS: https://openreview.net/forum?id=fgpXAu8puGj
- NASAIC: https://arxiv.org/abs/2002.04116
- DANCE: https://arxiv.org/abs/2009.06237
- Auto-NBA: https://arxiv.org/abs/2106.06575
- JAQ (AAAI 2025): https://arxiv.org/abs/2501.05339
- Vidur (MLSys 2024): https://arxiv.org/abs/2405.05465
- FlexGen (ICML 2023): https://arxiv.org/abs/2303.06865
- DistServe (OSDI 2024): https://arxiv.org/abs/2401.09670
- Splitwise (ISCA 2024): https://arxiv.org/abs/2311.18677
- Sarathi / Sarathi-Serve (OSDI 2024): https://arxiv.org/abs/2308.16369 , https://arxiv.org/abs/2403.02310
- Llumnix (OSDI 2024): https://arxiv.org/abs/2406.03243
- Ansor (OSDI 2020): https://www.usenix.org/system/files/osdi20-zheng.pdf
- MetaSchedule: https://github.com/apache/tvm-rfcs/blob/main/rfcs/0005-meta-schedule-autotensorir.md
- Mirage (OSDI 2025): https://arxiv.org/abs/2405.05751
- MLGO: https://arxiv.org/abs/2101.04808
- Nexus (SOSP 2019): https://homes.cs.washington.edu/~arvind/papers/nexus.pdf
- Clockwork (OSDI 2020): https://www.usenix.org/system/files/osdi20-gujarati.pdf
- REEF (OSDI 2022): https://www.usenix.org/system/files/osdi22-han.pdf
- AlpaServe (OSDI 2023): https://arxiv.org/abs/2302.11665
- Sia (SOSP 2023): https://www.pdl.cmu.edu/PDL-FTP/BigLearning/sia_sosp23-final.pdf
- RT-BEV (RTSS 2024): https://rtcl.eecs.umich.edu/rtclweb/assets/publications/2024/rtss24-liu.pdf
