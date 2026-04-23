# 自动驾驶 / V2X 部署 D 空间调研

> 本文聚焦"实际部署工作中可配置的硬件维度（D 空间）"，不重复 `survey_raw/agent_5_driving.md` 已覆盖的算法侧论文（Q-PETR / QD-BEV / PlanKD / QuantV2X / SPADE / LiFT 等）。
> 目标：为 UniV2X（ego/infra 双模型、Jetson Orin 部署）补齐工程层面的可配置维度知识。
> 日期：2026-04，覆盖 2023-2026 论文、NVIDIA 官方文档、开源代码库、国内主机厂/芯片厂公开资料。

---

## 0. TL;DR — 自动驾驶 D 空间的主维度

| 维度类别 | 可配置项 | 对 UniV2X 的相关性 |
|---|---|---|
| 硬件型号 | Orin Nano 8GB / NX 16GB / AGX 32/64GB / DRIVE Thor | 决定 ego/infra 节点可放入的模型大小 |
| 算力分配 | GPU-only / DLA0 / DLA1 / GPU+DLA 协同 | 双模型并行时 DLA 可吸收 BEV backbone 负载 |
| 精度 | FP32 / FP16 / INT8 / INT4 / 混合精度 | DLA 仅支持 FP16+INT8；TRT-10 显式量化迁移中 |
| 功耗/时钟 | nvpmodel 15W/30W/50W/MAXN、jetson_clocks | 直接决定峰值算力与热降频 |
| 内存层级 | LPDDR5 容量、EMC 带宽 204GB/s、UMA、pinned/zero-copy | 多相机 + LiDAR + V2X 特征同驻内存的关键 |
| 推理引擎 | TensorRT 8.x / 10.x、DeepStream、DriveOS、Triton | TRT 10 后 DLA INT8 显式量化是拦路虎 |
| 调度 | CUDA Streams、TRT Profile、多进程/多上下文 | 感知→预测→规划的流水化 |
| 通信 | V2X feature 压缩率、ROI、频率、时间戳对齐 | V2X 特有 D 空间 |
| 平台 | Jetson Orin / DRIVE Thor / 征程 5/6 / MDC / A1000 | 国产生态差异 |

---

## 1. Jetson Orin 部署可配置维度

### 1.1 硬件型号差异（Nano / NX / AGX / Thor）

| 型号 | GPU 核 | Tensor Cores | DLA | 内存 | 带宽 | TDP | 典型用途 |
|---|---|---|---|---|---|---|---|
| Orin Nano 4/8GB | 512–1024 Ampere | 16–32 | **无 DLA** | 4–8GB LPDDR5 | 34/68 GB/s | 7–15W | 轻量 BEV、V2X 路侧验证 |
| Orin NX 8/16GB | 1024 Ampere | 32 | 1× DLA (Nano 无) | 8–16GB LPDDR5 | 102 GB/s | 10–25W | 车端次旗舰 |
| Orin AGX 32/64GB | 2048 Ampere | 64 | **2× DLA v2** | 32/64GB LPDDR5 | **204.8 GB/s** | 15–60W（MAXN） | 车端主算力 / 路侧 |
| DRIVE Thor | Blackwell GPU + Transformer Engine | — | — | 128GB | ~273 GB/s+ | ~130W | 下一代 L3+ 中央舱 |

**关键点**
- **Orin Nano 无 DLA**：所有 DLA 部署技巧在 Nano 上全部失效，只能靠 GPU INT8/FP16。UniV2X 若打算 Nano 验证 infra 端，需接受纯 GPU 路线。
- **AGX 是唯一 2×DLA 平台**：双相机/Lidar 双 backbone 可各占一个 DLA，留 GPU 给 Transformer 头、BEVPool、planning。
- **JetPack 6.2 引入 "Super Mode"**：Orin Nano/NX 解锁高档位 nvpmodel（Nano 25W、NX 40W + MAXN SUPER），生成式 AI 性能提升 1.6×，但功耗涨 1.7×（RidgeRun/NVIDIA 技术博客，2025 年初）。

### 1.2 DLA 使用情况实测

DLA 是 Orin 的"第二/第三类加速器"，其工程细节决定 D 空间中 `[0, 1, 2]` 的真实含义：

1. **性能量级**：NVIDIA 官方博客显示 AGX Orin DLA 上运行 BI3D（3 DNN 机器人感知）可达 ~46 fps、30 ms 延迟；DLA 能效 **3–5× 高于 GPU**（同工作负载、同功耗模式）。
2. **精度限制**：DLA 仅支持 **FP16 和 INT8**，不支持 FP32、BF16、FP8；DLA v2（Orin）相对 v1（Xavier）扩展了结构稀疏和更多层类型。
3. **层回退（Fallback）是常态**：
   - `Shuffle / Reshape / Transpose` 的任意非连续 permutation 通常回退到 GPU（论坛大量报告）；
   - `LeakyReLU`、`GlobalAveragePool`、动态维度、`batch > 4096` 均不支持；
   - 官方做法：`trtexec --allowGPUFallback --useDLACore=0 --fp16`，TRT 会把网络切成多个 DLA loadable，未支持层自动放 GPU。
4. **稀疏化在 DLA 上无收益**：NVIDIA 论坛确认 structured sparsity 在 DLA 上不加速（仅 GPU Tensor Core 有收益），这是 D 空间`[稀疏开关]`上很反直觉的结论。
5. **BEV/感知实战**：
   - **CUDA-BEVFusion**（NVIDIA-AI-IOT/Lidar_AI_Solution）：ResNet-50 PTQ + TRT FP16+INT8，Orin AGX **25 FPS**，nuScenes 6019 样本平均；未把 BEVPool/图像 backbone 放 DLA，整体仍是 GPU 主导。
   - **DeployFusion**（Sensors 2024）：EdgeNeXt + 两阶段融合，Orin NX **138 ms/frame**。
   - **FastPillars**：Orin AGX **18 FPS**，快于 CenterPoint。
6. **TRT 10 的痛点**：TRT-10 弃用 `IInt8Calibrator` 等隐式量化 API（保留到 2025/03），但 **DLA INT8 当前仍需隐式量化** — 社区工程师被迫在两套体系间切换；NVIDIA 表示 DLA 显式量化还在开发中。对 UniV2X：若坚守 TRT-10 + DLA INT8，需要 TensorRT Model Optimizer 工具链，且不是所有 BEV 算子都走得通。

### 1.3 Power mode / 时钟频率 / EMC

- **nvpmodel 档位**：
  - Orin AGX 支持 15W / 30W / 50W / MAXN（无功耗上限）四档；在 50W 以下 CPU 核数、GPU/DLA 频率均被限；
  - MAXN 在热降频触发前才展示峰值，连续推理场景实际吞吐**远低于**瞬时 TOPS 数。
- **`jetson_clocks`**：锁住当前档位的最高频率，关闭 DVFS。量化基准时必开；部署到车端常**关闭** — 否则风扇噪声、热失控、能效比都恶化。
- **EMC（内存控制器时钟）**：Orin LPDDR5 标称 204.8 GB/s，实测（RidgeRun/论坛数据）随机访问 ~70–90%。多相机流 + LiDAR + V2X 特征同时常驻 GPU 内存时 **EMC 成为首要瓶颈**，表现为 GPU 算力空转。
- **统一内存（UMA）优化**：
  - CPU/GPU 共享同一物理 DRAM，`cudaMemcpy(H2D/D2H)` 在 Jetson 上是**零拷贝**（底层只换指针所有权），但显式 memcpy 仍付出 driver 开销；
  - 推荐 `cudaHostAllocMapped` + zero-copy 或 `__managed__` 统一内存；预处理在 CPU 侧就地写入 GPU 可读 buffer，省去一次拷贝；
  - 读回 CPU 时「Array-of-Struct」模式比「基本类型」慢很多（论坛实测），V2X 特征用 SoA。

### 1.4 TensorRT 版本影响

| TRT 版本 | 附带 JetPack | 主要影响 |
|---|---|---|
| 8.5 / 8.6 | JetPack 5.x | 隐式量化（calibrator）为主；BEVFusion、CenterPoint 多数示例基于此 |
| 10.x | JetPack 6.x | 显式量化（Q/DQ 节点）主推；TRT-LLM on Jetson 上线；DLA INT8 仍需隐式（冲突点） |
| 10.10 (DRIVE OS 7.0.3) | DRIVE OS | 面向 Thor，加入 Transformer Engine 专属优化（FP8 kernels） |

**实战坑**：CUDA-BEVFusion 等官方样例多在 TRT 8 环境打磨；升到 TRT 10 时 BEVPool/自定义 plugin 常失效，plugin IO 接口从 `IPluginV2DynamicExt` 向 `IPluginV3` 迁移；很多团队**冻在 JetPack 5.1.2 + TRT 8.6**。

---

## 2. BEV / 感知模型部署工作

### 2.1 StreamPETR / Far3D / BEVFormer（多视图 Transformer 系）

- **NVIDIA `DL4AGX` 仓库**是官方参考实现，集中了 StreamPETR TensorRT、Far3D TensorRT、BEVFormer INT8 显式量化、Sparsity INT8 训练+TRT 推理等。
- **BEVFormer_tensorrt**（DerryHub/社区）：实现 FP32/FP16/INT8 三档；与 PyTorch 精度差 < 0.3 NDS 的前提下，**推理 4× 加速、引擎 90% 缩小、显存 80% 节省**；大量自定义 TRT plugin（Deformable Attention、Temporal Self-Attention）。
- **部署痛点**：
  - Deformable Attention 的 `grid_sample` 离散索引在 DLA 不支持 → 该子图必然 GPU；
  - 时序队列（StreamPETR/Sparse4D）的 recurrent state 需持久化 GPU buffer，TRT engine **不管理跨调用状态**，需外层 runtime 自持 feature bank；
  - 动态 shape（可变 queries、可变历史窗口）和 DLA 静态维度要求冲突。

### 2.2 UniAD / VAD / HOP / PARA-Drive 端到端

- **UniAD** 原生 7.2 FPS（A100），**VAD-Base** 更慢，UniAD 的 perception+prediction 占 71.8% 运行时间（Det&Track 31.2% + Map 19.8% + Motion 10.9% + Occupancy 9.9%）。直接部署到 Orin 不现实，需要 PlanKD 类蒸馏或模块裁剪。
- **PARA-Drive (CVPR 2024)**：把 perception/prediction/planning 改为**并行同训**（原本串行依赖），推理时各模块可同时跑，对**多 CUDA stream 调度友好**，是端到端模型对 D 空间更友好的例子。
- **Hydra-MDP**（NVIDIA 博客 2024）：工业界一个端到端多决策头框架，NVIDIA 自研，DRIVE 平台首发。
- **ExpertAD (2025)**：Mixture-of-Experts 路线，动态路由在 Orin 上带来额外调度复杂度。
- **实际部署共识**：端到端模型普遍**先拆回模块化**（BEV encoder + tracker + MapHead + Planner），各模块独立 TRT engine，runtime 拼管线 — 这是 UniV2X 当前采用的形态。

### 2.3 CenterPoint / PointPillars / FastPillars（经典 LiDAR）

- **FastPillars**：Orin AGX **18 FPS**，无稀疏算子，纯 dense conv，对 DLA 和 INT8 都友好；CenterPoint 对比更慢（体素化 + sparse conv，DLA 完全不支持）。
- **CUDA-Pointpillars / TAO-PointPillars**：NVIDIA 官方 ROS2 管线，Orin AGX 上 ~10 Hz 输入点云实时；体素化（scatter）用 CUDA kernel，不走 TRT。
- **Mixed Precision PointPillars (arXiv 2601.12638)**：混合精度相对 FP32 **2.538× 加速**。
- **部署模板**：LiDAR 管线通常被拆成 [CUDA 预处理 (体素化/scatter)] → [TRT engine (backbone+head)] → [CUDA 后处理 (NMS)]，三段之间用 pinned memory + CUDA stream 同步。

### 2.4 HD Map 在线向量化（MapTR / VectorMapNet）

- **MapTR-nano 25 FPS on RTX 3090**（学术报；未见公开 Orin 实测数据）；MapTRv2-VoVNet99 相对 VectorMapNet 4× 加速、19.7 mAP 提升。
- 部署挑战：hungarian matching、permutation group 在 TRT 上需自写 plugin；实车上多数团队**改用局部 HD map 缓存** + 在线微调，不做完整 MapTR 实时推理。

---

## 3. 多模型流水线调度

### 3.1 感知 → 预测 → 规划 的流水化

- **PARA-Drive**（CVPR 2024）：把原本串行的 perception→prediction→planning 改为并行同训，实车推理时各支并行，对 **CUDA streams 多路并发**友好。
- **Prophet (IROS 2022, Shi 组)**：profiler + time predictor + coordinator，预测当前帧是否来得及，来不及则提前 exit 或丢帧；工业级预测性调度框架的代表。
- **DART (RTSS 2019)**：管线化 CPU/GPU 调度多 DNN 实时推理，把大模型拆成子阶段分发到 CPU / GPU / DLA，pipeline fill 提高吞吐。
- **Holistic Heterogeneous Scheduling for Autonomous Driving**（arXiv 2508.09503，2025）：把 CPU、GPU、DLA、PVA 看成异构资源池联合调度，对 UniV2X 这种 ego+infra 双模型很有启发。

**UniV2X 可借鉴的 D 空间配置**：
- 感知 backbone 走 DLA0（BEV encoder）；
- 预测头（motion / occupancy）走 GPU 主流；
- 规划头走 GPU 次流；
- V2X 通信压缩 encoder 走 DLA1。
  四路并行 + CPU-DRAM 上的 feature bank，是目前公开文献里最优的 Orin AGX 分配。

### 3.2 多相机 / 多传感器并行

- **多相机**：nuScenes 6 相机在 Orin 上典型做法是 concatenate 到 batch 维（N=6）一次前向；内存允许的话 **每相机独立 engine + 6 CUDA streams**（Tesla AI Day 披露 HydraNet 风格）。
- **图像/LiDAR 双分支**：CUDA-BEVFusion 是 NVIDIA 官方参考，LiDAR 分支和 Camera 分支分别进 TRT，BEV-pool 融合点在 CUDA 侧手写；两分支**共享内存但各自 stream**。
- **传感器时间对齐**：车端通常 PTP 硬件同步 + 软件 buffer 重排；EMC 带宽是关键（多 raw stream + CUDA memcpy）。

---

## 4. V2X 协同驾驶专属 D 空间

### 4.1 通信精度 / 频率 / ROI

| 维度 | 取值范围 | 代表工作 |
|---|---|---|
| 融合类型 | Early / Intermediate / Late | V2X-ReaLO (2025) 首个三者都实现的实车框架 |
| 带宽 | Early ~660 Mbps（CoBEVT/V2X-ViT） → Late ~0.09 Mbps（object-level） | Where2Comm 通过 spatial confidence map 做稀疏传输 |
| 通信精度 | FP32 → FP16 → INT8 → 码本/VQ | **QuantV2X**（INT4 权重 + INT8 激活 + 码本通信量化，99.8% 精度保留） |
| 频率 | 10 Hz / 20 Hz / 50 Hz | DSRC/C-V2X 典型 e2e 延迟 ~100 ms |
| ROI | 全 BEV / confidence top-k / feature uncertainty | Where2Comm、BEVCooper |
| 时间戳对齐 | 当前帧 / history / 预测补偿 | SyncNet（双支 LSTM 预测）、Latency-Aware CP (ECCV 2022) |

**关键洞察**（UniV2X 相关）：
- **延迟常常 > 1s**：CAV 链路差的情况下可以秒级；必须有预测式补偿（如 SyncNet / V2X-ReaLO 的 async 融合）。
- **BEVCooper（arXiv 2512.19082）** 和 **Latency Robust CP（WACV 2025）**是最近两篇直接针对「通信延迟+带宽」双约束的工作，可以作为 UniV2X 通信 D 空间的基线。
- **CooperDrive**：报告端到端 89 ms 延迟 + KB/s 级通信量，证明 ROI + 压缩组合在实车可行。

### 4.2 端-路-云任务分工

- **端侧（车）**：BEV encoder、局部 tracker、规划；必须在 Orin AGX 内 100 ms 完成一帧。
- **路侧（RSU）**：全局 tracker、V2X 广播、HD map 增量；算力典型 Orin AGX / 双 Orin NX 集群。
- **云侧**：离线 HD map 构建、路径规划长时预测、模型热更新；不要求实时。

UniV2X 的 ego+infra 双模型结构正对应端+路；云侧目前不是 D 空间优化重点。

---

## 5. 边缘推理实战（延迟 / 内存 / 功耗）

### 5.1 延迟

- 端到端驾驶一帧预算：10 Hz → 100 ms，L2/L3 要求 30–50 ms；
- CUDA-BEVFusion 在 Orin AGX 25 FPS = 40 ms，可用但没有余量；
- **UniAD 原生 138 ms** 不可直接上车，必须蒸馏+裁剪；
- PlanKD 将 52.9M 教师压到 26.3M 学生，推理 39.7 ms（A100），Orin 上估计 ~80–100 ms。

### 5.2 内存

Orin AGX 32GB 典型分配（自己对 UniV2X 场景的推算）：
- 多相机原始帧 ring buffer：~1 GB（6×1920×1080×12bit×N_frames）；
- BEV encoder engine + 中间激活：~2–3 GB；
- LiDAR 分支：~1.5 GB；
- 时序 feature bank（streaming）：~1 GB；
- V2X 特征接收 buffer：~0.5 GB；
- 其他 CUDA context + OS：~3 GB；
- 合计 ~10 GB 稳态，留余量给偶发 allocator 碎片。
  **16 GB NX 上同样配置会频繁 OOM**，需要激活重算或分块 encoder。

### 5.3 功耗

- 乘用车域控一般电源预算 **30–50W**（对应 Orin AGX 30W / 50W 档）；
- 路侧 RSU 可放 60W MAXN，但需主动散热；
- **纯 GPU 方案在 15W 档几乎跑不动 BEVFusion**；必须 DLA 分担 + INT8。

---

## 6. 国产自动驾驶芯片生态

### 6.1 Horizon BPU（征程 5 / 6）

- **Journey 5**：BPU 2.0 "Bayes"，128 TOPS，双核 BPU；8× Cortex-A55，26k DMIPS；2021 年设计，2022 量产。
- **Journey 6P**：BPU 3.0 "Nash" **4 核**，560 TOPS；18 核 ARM Cortex-A78AE（乱序），410k DMIPS；LPDDR5x-8533 共 204 GB/s 带宽（与 Orin AGX 相同量级）；Mali-G78AE GPU；TSMC 7nm、370 亿晶体管。
- **工具链**：Horizon Open Explorer / HBDK（开发包），编译器自动优化 + 可定制算子，专为 Transformer 和大模型设计（Nash 架构）。
- **生态**：比亚迪、理想、蔚来、大众等量产车型；特征是"算法公司自研 BPU + 工具链私有"，对标 Mobileye 模式。

### 6.2 华为 MDC / Ascend

- **Ascend 310 / 310P**：L4 自动驾驶 ~200W 功耗级别 MDC 主力；
- **CANN 软件栈**：Ascend 芯片算子库 + 自动算子开发工具（类似 TensorRT）；
- **MindSpore**：训练+推理框架，与 Ascend 强绑定；
- 主要客户：北汽 ARCFOX、广汽、上汽，以及商用车、园区物流车。

### 6.3 黑芝麻 A1000 / A1000 Pro / A2000

- **A1000**：首款单 SoC 支持"行泊一体"的车规级芯片，ISO26262 ASIL-B + AEC-Q100；
- **Shanhai（山海）工具链**：覆盖量化、优化、编译、仿真、部署、调试全流程，内置参考模型库；
- **A2000**（2026 年初）：美国放行用于全球，是国产车规 AI 芯片出海的第一块；
- 客户：江淮、东风、合众哪吒。

### 6.4 NVIDIA DRIVE（对标参照）

- **DRIVE Orin**：占据 **全球 30 大 EV 厂中 20 家** 市场；蔚来、小鹏、理想、比亚迪、Lucid 等量产；
- **DRIVE Thor**（Zeekr 2025 年 CES 首发套件）：下一代 L3+ 中央舱，自研 Blackwell + Transformer Engine；
- **DriveOS + DriveWorks + TensorRT + NvMedia + NvStreams** 软件栈。

---

## 7. 关键洞察

### 7.1 自动驾驶 D 空间的特殊性

1. **实时性硬约束**：100 ms 是门槛，30–50 ms 是目标；所有 D 选择都必须在此预算内收敛。通用推理引擎研究中「更大 batch 更高吞吐」的常规 D 在车端**无效**。
2. **多加速器异构是常态**：GPU + 2×DLA + PVA + VIC + NVENC/NVDEC 同时工作，D 空间是**多维 tuple** `(device_choice, precision, batch, stream_id)` 每层都要独立决策。
3. **内存（EMC 带宽）常常比算力更紧**：多相机 + LiDAR + V2X + HD map 同驻，带宽 204 GB/s 是真正瓶颈；学术论文里只报 FLOPs/TOPS 是不够的。
4. **DLA 的甜蜜点很窄**：支持层受限、稀疏无增益、INT8 量化在 TRT-10 迁移期有 API 撕裂；只有 backbone conv 主体能充分享受 DLA。
5. **V2X 引入独立通信 D 空间**：压缩率、ROI、频率、时间戳补偿 — 这 4 个维度在纯车端推理里不存在，需要 UniV2X 自建调研基线。

### 7.2 与通用推理引擎的差异

| 通用推理（LLM/CV 服务） | 自动驾驶 |
|---|---|
| 优化 throughput (req/s) | 优化 tail latency (p99 < 100ms) |
| 动态 batch | 固定 batch=1 / 固定 N 相机 |
| 云 GPU (A100/H100) 显存大 | Jetson 16–64GB LPDDR 共享 |
| 允许突发高功耗 | 电源预算硬上限 |
| 单模型主导 | 3–10 个子模型并行 |
| 无 V2X | 有通信 D 空间 |
| 精度容忍度高（ppl） | 精度容忍度低（mAP/NDS 掉 1% 即不可用） |

### 7.3 UniV2X 的定位与建议

- **UniV2X 是 ego+infra 双 Orin AGX 典型配置**，天然落在 D 空间最富的地方：2 台 AGX × (GPU + 2×DLA) + V2X 链路 = 真实的 **6+ 加速单元调度问题**。
- **调研空白**：目前公开文献几乎没有「双 Orin + V2X 协同」的完整 D 空间 benchmark；QuantV2X 是最接近的（但只做 INT4/INT8 量化，没有硬件分配层面的 D）。
- **建议实验矩阵**（UniV2X 1.3/1.4 后续）：
  1. ego 端 BEV encoder：{GPU-FP16, DLA0-INT8, DLA0-FP16+GPU-head} × nvpmodel {30W, 50W, MAXN} = 9 组；
  2. infra 端对称矩阵；
  3. V2X 通信：{无压缩, INT8-FP16, QuantV2X INT4+码本} × {full BEV, Where2Comm ROI} × {10Hz, 20Hz};
  4. 联合指标：p99 延迟、NDS、通信带宽、总功耗、热降频发生率；
- **核心差异化**：现有工作几乎都只优化「单车单模型」的 D，UniV2X 的独特价值是把「双节点 + 通信 + 端到端规划」的联合 D 空间第一次系统刻画出来。

---

## 参考文献与链接

### Jetson Orin / DLA / TensorRT
- NVIDIA Technical Blog, "Maximizing Deep Learning Performance on NVIDIA Jetson Orin with DLA": https://developer.nvidia.com/blog/maximizing-deep-learning-performance-on-nvidia-jetson-orin-with-dla/
- NVIDIA TensorRT Docs, "Working with DLA": https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html
- NVIDIA Technical Blog, "NVIDIA JetPack 6.2 Brings Super Mode to Jetson Orin Nano/NX": https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/
- NVIDIA Forums, "Memory bandwidth on Orin": https://forums.developer.nvidia.com/t/memory-bandwidth-on-orin/277639
- NVIDIA Forums, "Is there a plan to support DLA on the next TensorRT version?": https://forums.developer.nvidia.com/t/is-there-a-plan-to-support-dla-on-the-next-tensorrt-version/313130
- GitHub Issue 4095, "INT8EntropyCalibrator2 implicit quantization superseded by explicit": https://github.com/NVIDIA/TensorRT/issues/4095
- RidgeRun, "Jetson Orin AGX Performance Tuning by Tuning Power": https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin/JetPack_5.0.2/Performance_Tuning/Tuning_Power
- NVIDIA Jetson Linux Developer Guide, Orin Power/Performance: https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html
- arXiv 2508.08430, "Profiling Concurrent Vision Inference Workloads on NVIDIA Jetson": https://arxiv.org/html/2508.08430v1

### BEV / 感知部署
- GitHub NVIDIA/DL4AGX (StreamPETR / Far3D / BEVFormer TRT): https://github.com/NVIDIA/DL4AGX
- GitHub DerryHub/BEVFormer_tensorrt: https://github.com/DerryHub/BEVFormer_tensorrt
- GitHub NVIDIA-AI-IOT/Lidar_AI_Solution CUDA-BEVFusion: https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/CUDA-BEVFusion/README.md
- GitHub mit-han-lab/bevfusion: https://github.com/mit-han-lab/bevfusion
- arXiv, FastPillars (2302.02367): https://arxiv.org/html/2302.02367v6
- arXiv, Mixed Precision PointPillars TRT (2601.12638): https://arxiv.org/abs/2601.12638
- GitHub hustvl/MapTR: https://github.com/hustvl/MapTR
- arXiv, MapTRv2 (2308.05736): https://arxiv.org/html/2308.05736v2
- NVIDIA Technical Blog, "Detecting Objects in Point Clouds with NVIDIA CUDA-Pointpillars": https://developer.nvidia.com/blog/detecting-objects-in-point-clouds-with-cuda-pointpillars/
- MDPI Sensors (2023), "Run Your 3D Object Detector on NVIDIA Jetson Platforms: Benchmark": https://www.mdpi.com/1424-8220/23/8/4005
- arXiv, Fast-BEV on-vehicle BEV (ML4AD 2022): https://ml4ad.github.io/files/papers2022/Fast-BEV:%20Towards%20Real-time%20On-vehicle%20Bird's-Eye%20View%20Perception.pdf

### 端到端驾驶
- GitHub OpenDriveLab/UniAD: https://github.com/OpenDriveLab/UniAD
- PARA-Drive (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/papers/Weng_PARA-Drive_Parallelized_Architecture_for_Real-time_Autonomous_Driving_CVPR_2024_paper.pdf
- NVIDIA Technical Blog, "End-to-End Driving at Scale with Hydra-MDP": https://developer.nvidia.com/blog/end-to-end-driving-at-scale-with-hydra-mdp/
- arXiv, Perception in Plan (2508.11488): https://arxiv.org/html/2508.11488
- arXiv, ExpertAD MoE (2511.11740): https://arxiv.org/html/2511.11740

### 调度与并行
- RTSS 2019, DART pipelined CPU/GPU scheduling: https://intra.ece.ucr.edu/~hyoseung/pdf/rtss19-dart.pdf
- Ming Yang PhD dissertation, Sharing GPUs for RT Autonomous Driving: https://www.cs.unc.edu/~anderson/diss/mingdiss.pdf
- Prophet realtime perception pipeline (Shi 组): https://weisongshi.org/papers/liu22-prophet.pdf
- arXiv 2508.09503, Holistic Heterogeneous Scheduling for AD: https://www.arxiv.org/pdf/2508.09503

### V2X 协同感知
- GitHub Little-Podi/Collaborative_Perception 调研 digest: https://github.com/Little-Podi/Collaborative_Perception
- arXiv 2310.03525 V2X CP Recent Advances: https://arxiv.org/html/2310.03525v5
- arXiv 2503.10034, V2X-ReaLO 实车框架: https://arxiv.org/html/2503.10034
- arXiv 2512.19082, BEVCooper: https://arxiv.org/pdf/2512.19082
- WACV 2025, Latency Robust Cooperative Perception via Async Fusion: https://openaccess.thecvf.com/content/WACV2025/papers/Wang_Latency_Robust_Cooperative_Perception_using_Asynchronous_Feature_Fusion_WACV_2025_paper.pdf
- ECCV 2022, Latency-Aware Collaborative Perception (SyncNet): https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920315.pdf
- CVPR 2024, TUMTraf V2X Cooperative Perception Dataset: https://openaccess.thecvf.com/content/CVPR2024/papers/Zimmer_TUMTraf_V2X_Cooperative_Perception_Dataset_CVPR_2024_paper.pdf

### 国产芯片生态
- Horizon Robotics Journey 5 介绍（Open Explorer 文档）: https://developer.d-robotics.cc/api/v1/fileData/horizon_j5_open_explorer_en_doc/oe_mapper/source/appendix/j5_introduction.html
- Horizon 官网 Superdrive + J6 发布: https://www.horizon.cc/horizon-robotics-launches-next-generation-autonomous-driving-solution-superdrivetm-and-journey-6r-series/
- 42how, Horizon BPU Nash 架构: https://en.42how.com/2023/04/18/horizon-unveils-new-bpu-architecture-for-smart-driving-bpu-nash-built-for-transformer-and-large-models/
- Huawei Ascend 310 产品页: https://actfornet.com/products/intelligent-computing/atlas/huawei-ai/ai-chips/Ascend_310/features
- Black Sesame Huashan A1000: https://www.blacksesame.com/en/huashan-series-a1000/
- Black Sesame Huashan A2000 (2026 年海外获批): https://www.trendforce.com/news/2026/01/06/news-us-reportedly-clears-chinas-black-sesame-huashan-a2000-auto-chip-for-global-launch/

### NVIDIA DRIVE
- NVIDIA DRIVE AGX Developer: https://developer.nvidia.com/drive/agx
- NVIDIA Newsroom, DRIVE Orin Production: https://nvidianews.nvidia.com/news/nvidia-enters-production-with-drive-orin-announces-byd-and-lucid-group-as-new-ev-customers-unveils-next-gen-drive-hyperion-av-platform
- CnEVPost, Zeekr NVIDIA Thor CES 2025: https://cnevpost.com/2025/01/07/zeekr-nvidia-thor-smart-driving-kit-ces-2025/
- Edge AI Vision, NVIDIA DRIVE AGX Thor Developer Kit: https://www.edge-ai-vision.com/2025/09/accelerate-autonomous-vehicle-development-with-the-nvidia-drive-agx-thor-developer-kit/
