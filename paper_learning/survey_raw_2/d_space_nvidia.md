# NVIDIA 栈 D 空间深度调研：固定硬件可配置维度

> **范围**：本报告聚焦"硬件选定后，软件层仍可调的编译/运行时维度（D 空间）"，重点在 NVIDIA GPU / Jetson / DLA 栈。
>
> **与已有报告的边界**：Tensor Core 代际差异与 INT8/FP8 tactic 细节见 `quantization_x_hardware.md`；2:4 稀疏 Tensor Core 与 cuSPARSELt 细节见 `pruning_x_hardware.md`；Jetson Orin 硬件型号、nvpmodel、EMC 带宽、DLA 实战见 `ad_v2x_deployment.md`。本文不重复，仅在必要处交叉引用。
>
> **日期**：2026-04

---

## 1. GPU 资源划分：CUDA Streams vs MPS vs MIG

在"单卡多任务/多模型"场景下，NVIDIA 提供三种层次的并发/隔离机制，粒度由细到粗依次是 **Streams → MPS → MIG**。

### 1.1 CUDA Streams

**机制**：Stream 是一条 GPU 异步命令队列；同一 context 内不同 stream 上的 kernel 和 memcpy 可**并发执行**（只要 SM / copy engine 资源够用）。Stream 不提供内存隔离，所有 stream 共享同一 context 的显存和 L2。

**关键维度**：

- **默认流行为**：`cudaStreamDefault` 是 "legacy default stream"，它与**所有**其它非阻塞流同步（全局屏障）；`cudaStreamNonBlocking` 不参与隐式同步；编译时带 `--default-stream per-thread` 可让每线程各自拥有独立默认流。在多线程 / 多 CUDA context 应用里，**一定要显式使用非阻塞流**，否则 legacy default 会吞掉所有并发。
- **Stream priority**：`cudaStreamCreateWithPriority(&s, flags, prio)`，范围由 `cudaDeviceGetStreamPriorityRange` 决定，通常 `[-2, 0]` 或更宽（Hopper 扩展到 `[-5, 0]`）。高优先级流的 kernel **抢占**低优先级流的 kernel 在 SM 上的调度（不是抢占执行中的 CTA，是抢占调度）。典型用法：感知主路径走最高优先级，预取 / 后处理走次高。
- **Event 依赖**：`cudaEventRecord(evt, streamA)` + `cudaStreamWaitEvent(streamB, evt)` 构建跨流 DAG。比 `cudaStreamSynchronize` 轻得多（不回 CPU）。
- **数量选择**：经验值 2–8 条；继续增加基本不会带来更多并行度（H100/Orin 的 copy engine 2 条，SM 并发受 occupancy 限制）。多于 16 条会导致调度开销（每次 submit 都要查询所有流状态）抵消收益。
- **与 TRT 的交互**：`IExecutionContext::enqueueV3(stream)` 把一个 engine 的所有 kernel submit 到指定 stream。**多实例并发**推荐：每个 engine 建自己的 `IExecutionContext`，分别绑定独立 stream；`IRuntime`/`ICudaEngine` 线程安全可共享。

### 1.2 MPS (Multi-Process Service)

**机制**：MPS 是用户态 daemon (`nvidia-cuda-mps-control` + `nvidia-cuda-mps-server`)，把多个**不同进程**的 CUDA context 合并到**同一 GPU context** 里。本质是把"进程间时间片切换"变为"同一 context 下的 stream 级并发"。

**解决的痛点**：不用 MPS 时，GPU 同一时刻只能跑一个 context（依靠时间切片轮询），导致 SM 利用率低 + 切换开销。MPS 合并后，多进程 kernel 可真正空间并发。

**Volta 及之后的增强**：

- **Volta MPS**：把 client 从一个 server 进程拆分为每客户端独立 service（隔离更好，故障不串扰）。
- **Percentage-based partition**（Volta MPS / Ampere+）：`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 限制单个 client 能用多少 SM 百分比；`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` 限显存。这是**软性限制**（不是 MIG 那种硬隔离），GPU 故障会影响所有 client。
- **Jetson 支持**：JetPack 6.1 + CUDA 12.5 起，MPS 正式支持 **Jetson AGX Orin / Orin NX**（见 NVIDIA Jetson Forum "Proper use of CUDA MPS"、Antmicro 2025 博客）。这是 V2X 多模型（感知+预测+规划+V2X 融合）同卡部署的重要能力。

**典型启动**：
```bash
nvidia-cuda-mps-control -d          # 启 daemon
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
# 每个 client 进程启动前：
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=30   # 该进程最多用 30% SM
```

**不适用场景**：

- 需要硬性 QoS / 故障隔离 → 上 MIG。
- 单进程内多模型 → 用 stream 就够了，MPS 帮不上。
- 显存压力大、需要严格分配 → MPS 软限制不保证。

### 1.3 MIG (Multi-Instance GPU)

**机制**：硬件层把 A100 / H100 / B200 的 GPU 划分为**最多 7 个隔离实例**（GI, GPU Instance）+ 更细粒度的 CI（Compute Instance）。每个实例拥有独立 SM 子集、独立 L2 切片、独立显存 channel、独立故障域。

**代际支持**：

| GPU 架构 | MIG | 最大实例数 | 备注 |
|---|---|---|---|
| V100 (Volta) | 否 | — | 只支持 MPS |
| A100 (Ampere) | 是 | 7 | 1g.5gb/2g.10gb/3g.20gb/4g.20gb/7g.40gb 组合 |
| H100 (Hopper) | 是 | 7 | Profile 略调整，支持 MIG + confidential compute |
| L40S / RTX 6000 Ada | 否 | — | Ada Lovelace 消费/工作站卡无 MIG |
| B200 (Blackwell) | 是 | 7 | 更多 profile，保留 Tensor Core 分配 |
| **Jetson Orin (Ampere iGPU)** | **否** | — | 见下 |
| **Jetson Thor (Blackwell iGPU, 2025)** | **是 / ready** | — | MIG + green contexts + MPS 都可用 |

**Orin 为什么不支持 MIG**：
1. Orin 是 iGPU（与 CPU 共享 LPDDR5 的 SoC），没有独立 HBM 的 memory channel 分区；
2. SM 数量少（Orin AGX 2048 cores = 16 SM），7 分切没意义；
3. SoC 上本就用 DLA / PVA / VIC 做硬件级工作负载隔离，iGPU 走 stream + MPS 足够。
Thor 在 Blackwell iGPU 上引入 MIG 主要面向多路 ADAS 融合和生成式 AI 车内应用。

**Profile 选择**：常见是 1g.10gb 跑小模型、3g.20gb 跑中等模型、7g.80gb 跑大模型；profile 组合受硬件 slice 对齐限制（A100 只能按 [1,1,1,1,2,2,7] 之类组合选一种）。切换 profile 需要**停掉所有 CUDA 工作** + `nvidia-smi mig -i 0 -cgi ... -C`。

### 1.4 三者选择决策树

```
需要硬件级 QoS / 故障隔离 / 多租户 SLA？
├─ 是 → 硬件支持 MIG？
│       ├─ 是（A100/H100/B200/Thor）→ MIG，按 profile 切
│       └─ 否（Orin/消费卡） → 退化到 MPS + cgroups
└─ 否 → 多进程 or 单进程？
        ├─ 多进程，希望空间并发     → MPS（percentage 限制）
        ├─ 多进程，允许时间切片     → 什么都不做（默认即可）
        └─ 单进程，多模型流水线     → CUDA Streams + priority
```

对 UniV2X：车端 Orin 场景下 **MIG 不可用**，方案是 Stream + MPS 组合 —— 感知主路径独占高优先级 stream，预测/规划/V2X 融合各自独立进程，MPS 配 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 粗分。

---

## 2. TensorRT Builder 配置详表

TensorRT Builder 阶段的 flag 是 D 空间最大的维度来源，几乎每个 flag 都会改变 kernel 选取和 engine 大小。

### 2.1 精度与类型约束

| Flag | 含义 | 默认 | 注意 |
|---|---|---|---|
| `FP16` | 允许 FP16 tactic | off | 打开不等于强制 FP16，只是让 builder 可选 |
| `BF16` | 允许 BF16 tactic（Ampere+） | off | 动态范围大于 FP16，Transformer 友好 |
| `INT8` | 允许 INT8 tactic | off | 必须同时提供 calibrator 或显式 QDQ |
| `FP8` | 允许 FP8 tactic（Hopper+, TRT 9+） | off | 要求 explicit QDQ，不是 PTQ |
| `INT4` (TRT 10+) | 权重 INT4（block-wise） | off | 主要给 LLM 权重量化 |
| `OBEY_PRECISION_CONSTRAINTS` | 严格遵守 layer-wise 精度 | off | layer 级 setPrecision 生效的硬约束 |
| `PREFER_PRECISION_CONSTRAINTS` | 尽量遵守，允许 fallback | off | 比 OBEY 宽松，性能更好但未必按你想要的精度跑 |
| `STRONGLY_TYPED` (TRT 10) | 按 ONNX 原生类型严格推导 | off | 打开后禁止 `setPrecision`/`setOutputType`，精度由网络图静态决定，避免隐式类型推断的意外 |
| `TF32` | 允许 TF32 tactic（Ampere+） | **on** | 默认打开；对精度敏感任务可能要显式关掉 |

**组合陷阱**：
- 同时开 FP16 + INT8，builder 会在 **per-layer** 粒度挑最快 tactic；某些 layer 可能挑 FP32 fallback，导致 precision 图不是你预期的。加 `OBEY_PRECISION_CONSTRAINTS` 配合 `ILayer::setPrecision` 强制。
- `STRONGLY_TYPED` 与 `setPrecision` 互斥，已成为 TRT 10+ 推荐做法（更可预测，尤其配合 PyTorch/ONNX 导出的 QDQ 图）。

### 2.2 Tactic sources

Tactic source = builder 选 kernel 的"算法库来源"，按位图 OR 组合：

| Source | 典型 kernel | 默认 | 何时关闭 |
|---|---|---|---|
| `CUBLAS` | GEMM（legacy） | **on** | TRT 10 起逐步废弃，存粹依赖 cuBLASLt |
| `CUBLAS_LT` | GEMM（Lt）支持 epilogue 融合 | on | —— |
| `CUDNN` | Conv / RNN / Norm | **on (TRT <10)** / **off (TRT 10+)** | TRT 10 默认不用 cuDNN，减少 runtime 依赖；Conv 全部走内置 |
| `EDGE_MASK_CONVOLUTIONS` | 特殊 Conv 模式 | off | 大部分场景不需要 |
| `JIT_CONVOLUTIONS` | 运行时 JIT 生成 Conv kernel | off | builder 阶段多花时间换更小 kernel 选择，engine 稍慢 |

关闭某些源可以显著**缩小 engine 的运行时依赖**（不链接 cuDNN = -300MB）以及**缩短 builder 时间**。但可能错过某些 tactic 导致吞吐下降。

配置方式：
```python
config.set_tactic_sources(
    1 << int(trt.TacticSource.CUBLAS_LT)
)  # 只保留 cuBLASLt
```

### 2.3 Workspace / Profile / Cache

**Workspace（memory pool）**：TRT 10 改名 `MemoryPoolType`：

| Pool | 作用 | 建议 |
|---|---|---|
| `WORKSPACE` | builder 尝试 tactic 时的临时显存 | 越大越好，但有上限。Orin 8GB 给 1–2GB 就够；A100 可给 4GB+ |
| `DLA_MANAGED_SRAM` | DLA 本地 SRAM | Orin 默认 1MB，可调 |
| `DLA_LOCAL_DRAM` / `DLA_GLOBAL_DRAM` | DLA 可用 DRAM 上限 | 一般保持默认 |
| `TACTIC_DRAM` (TRT 9+) | tactic 本身运行时的工作区 | 影响 peak memory |
| `TACTIC_SHARED_MEMORY` (TRT 10) | per-CTA shmem 上限 | Hopper 可给到 228KB，Orin 只能 48–64KB |

Workspace 太小 → builder 会自动拒绝一些 tactic（在 verbose 日志里能看到 `Skipping tactic X because workspace ... exceeds ...`），结果是挑到更慢的 tactic。

**Optimization Profile**：动态 shape 必备。每个 profile 定义 `min / opt / max`：

```python
profile = builder.create_optimization_profile()
profile.set_shape("input", (1,3,H,W), (1,3,H,W), (1,3,H,W))  # 纯静态：三个相等
# 或 (1,3,540,960), (1,3,720,1280), (1,3,1080,1920)         # 动态
config.add_optimization_profile(profile)
```

- `opt` 是 builder tuning 的目标，选错会让大部分实际 shape 性能差 20%+。
- 可添加**多个** profile（每个 shape 范围不同），运行时通过 `context.set_optimization_profile_async(idx, stream)` 切换。engine 大小近似乘以 profile 数。
- `--separateProfileRun`（trtexec）：对每个 profile 单独 benchmark，避免 opt shape 之间的 tactic 干扰。

**Timing Cache**：`config.create_timing_cache()` + `set_timing_cache()`，把每个 (tactic, shape, pool) 的实测耗时持久化。后续重建 engine 可以跳过重测，**builder 时间从 10min → 30s**。缓存与硬件型号 + 驱动 + TRT 版本强绑定，换机器失效。CI 里强烈建议持久化到对象存储。

**Refit**：`kREFIT` flag。engine 保留权重"占位符"，推理前 `Refitter` 可热替换权重，常用于**同架构不同权重**的场景（A/B 测试、finetune 后快速替换）。代价是 engine 更大、少数 tactic 不可用。

**Sparsity**：`kSPARSE_WEIGHTS` flag + 权重满足 2:4 结构稀疏 → builder 可挑 Sparse Tensor Core tactic。详见 `pruning_x_hardware.md`。

### 2.4 DLA 相关 flags

（硬件 DLA 细节见 `ad_v2x_deployment.md`，这里只说 builder 配置）

| trtexec flag / API | 作用 |
|---|---|
| `--useDLACore=0` / `config.default_device_type = DLA`, `config.DLA_core = 0` | 指定 DLA core（Orin AGX 有 DLA0/DLA1） |
| `--allowGPUFallback` / `config.set_flag(GPU_FALLBACK)` | DLA 不支持的 layer 退回 GPU；**不开会直接 build 失败** |
| `--fp16` + `--int8` on DLA | Orin DLA 支持 FP16 / INT8；FP32 不支持 |
| `ReformatFreeIO` | 让 DLA 直接读写 GPU tensor 格式，省去 reformat kernel |

**典型陷阱**：

1. DLA 对算子支持有白名单（Conv2D / Pool / Activation / Deconv / SoftMax 等），遇到 LayerNorm / GELU / Attention 就必须 fallback，fallback 多了 DLA 价值 ≈ 0（往返拷贝开销 > 省的 SM）。先用 `trtexec --verbose --useDLACore=0 --allowGPUFallback` 看日志里有多少 `run on GPU`。
2. DLA engine 不能跨卡 / 跨 core serialize，需要对每个 DLA core 各 build 一次。

### 2.5 Verbose 输出解读

`trtexec --verbose` 或 `config.profiling_verbosity = DETAILED` 的日志里关键字段：

- `Tactic 0x...`：TRT 内部 tactic hash，可跨 run 对比。
- `Timing: 0.123 ms`：该 tactic 实测耗时。
- `Fastest Tactic: 0x... Time: 0.082 ms`：最终选中的。
- `Skipping tactic X due to insufficient workspace`：增大 workspace 可能解锁。
- `Cudnn NOT available`：tactic source 关了 CUDNN 或 runtime 没链接；没问题，只是少一个候选。
- `Engine size: 123MB, Memory: Peak activation 456MB`：engine 序列化大小 + 运行时激活峰值；Peak activation 是部署显存估算的关键。
- `Layer name [TRT format] -> [TRT format]`：层输入输出的 tensor format（如 `Linear/fp16/NCHW` vs `Linear/fp16/NC4HW4`）。format 转换 kernel 通常由 `Reformat ...` 行表示，太多说明有明显 quantization boundary 开销。

对调参的**实操 loop**：`--verbose > build.log`，grep `Fastest Tactic` 统计 kernel 分布；grep `Skipping` 看被拒 tactic 原因；grep `Reformat` 看是否有可融合的 precision 边界。

---

## 3. CUDA Graph

### 3.1 捕获 / 重放机制

CUDA Graph 让一组 kernel + memcpy 的调用序列先**被捕获**为 DAG，之后通过 `cudaGraphLaunch` 一次性提交。核心开销模型：

- **捕获期**：一次；走 `cudaStreamBeginCapture(s, mode)` → 正常调用 → `cudaStreamEndCapture(s, &graph)` → `cudaGraphInstantiate(&exec, graph)`。TRT 可通过 `context.enqueueV3(stream)` 在 capture mode 下捕获整个 engine。
- **重放期**：每次 `cudaGraphLaunch(exec, stream)` 的 CPU 开销 ≈ 单次 `cudaLaunchKernel` 的 1/10 – 1/50。对**小 kernel 密集**的网络（每 kernel < 50μs，层数 > 50）通常整体 **+10–30%** 吞吐。

### 3.2 加速场景 vs 反模式

**显著加速**：

- Transformer / BEV 感知这种 launch-bound 小 kernel 密集网络（每 kernel 几十 μs）。
- Batch=1 低延迟推理（CPU launch 开销占比大）。
- 固定 shape、固定 stream 拓扑的流水线。

**反而变慢 / 不适用**：

- 单个大 kernel 占主导（LLM prefill 阶段、大 Conv） → launch 开销已经被摊薄，Graph 捕获开销可能得不偿失。
- **动态 shape**：每个 shape 要重新 capture + instantiate，开销大。TRT 的解决方案是 per-profile graph 缓存，或用 `cudaGraphExecUpdate` 增量更新（仅在拓扑一致时）。
- Host 逻辑干预多（每层间有 CPU 判断） → 无法整体捕获；需要拆成多个子图。
- 内存分配在 capture 内 → 禁止（`cudaMalloc` 不能在 stream capture 中调）；要提前 workspace 预分配。

**框架集成**：

- **TRT**：官方示例 `cuda_graph` 模式；TRT 8.5+ 的 `setOptimizationProfileAsync` 捕获内也支持。
- **PyTorch**：`torch.cuda.graph(graph)` context manager + `graph.replay()`；配合 `torch.cuda.make_graphed_callables`。训练侧也能用（需固定 shape）。
- **JAX**：`jax.jit` 开 `jax_use_cuda_graph_for_pure_xla_calls`。

对 UniV2X：V2X BEV + 检测头都是小 kernel 多，建议对"固定 shape 的主推理路径"用 CUDA Graph，对"动态 shape 的融合层"保留正常 stream launch。

---

## 4. cuDNN / cuBLAS / cuBLASLt tactic

### 4.1 Heuristics vs exhaustive

| 模式 | 含义 | 典型耗时 | 适用 |
|---|---|---|---|
| `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM` 等固定 ID | 直接指定算法 | 0 tuning | 老 API；不推荐 |
| **Heuristics**（v8 API）`cudnnGetConvolutionForwardAlgorithm_v7` | 基于 shape / arch 的启发式推荐 | ms 级 | 默认；90% 情况足够 |
| **Exhaustive** `cudnnFindConvolutionForwardAlgorithm` | 真跑每个候选算法取最快 | 秒级（per shape） | 训练或长寿命 engine 时值得；缓存到文件 |
| cuBLAS **workspace-aware** | 给定 workspace 大小，挑最优 | 同 heuristics | Lt API 推荐 |
| cuBLASLt **autotune** | Lt 的 exhaustive | ms–s | 对 GEMM 形状敏感时用 |

### 4.2 PyTorch cuDNN benchmark 机制

```python
torch.backends.cudnn.benchmark = True
```

打开后，PyTorch 对**每个 unique (input shape, dtype, layout)** 在首次 forward 时跑 exhaustive search，结果 cache 到进程内。

- 优点：典型 +10–30% 吞吐（尤其 CNN）。
- 陷阱：**输入 shape 经常变**（如动态 batch / 可变分辨率）时，每个新 shape 都会 cache miss → 反而拖慢。此时应 `benchmark = False` + `deterministic = True`。
- 相关：`torch.backends.cudnn.deterministic = True` 禁用非确定 kernel，复现性好但慢 5–15%。
- `torch.backends.cuda.matmul.allow_tf32` / `torch.backends.cudnn.allow_tf32`：Ampere+ 默认 True，对精度敏感任务（如坐标回归）可能要关掉。

### 4.3 Workspace 大小影响

cuBLASLt / cuDNN 的 workspace 类似 TRT：

- 太小 → 只能选小 workspace 的 kernel（通常慢 10–20%）。
- 太大 → 浪费显存，对多实例不友好。

经验值：
- cuBLASLt workspace：**32MB** 在 Hopper 上是甜点；Ampere 用 4–8MB 够；Orin 1MB 以内。
- cuDNN：Conv workspace 按 shape 自适应，给到 `max_size = 512MB` 是安全上限。
- PyTorch `CUBLASLT_WORKSPACE_SIZE` 环境变量（KB）可调。

---

## 5. ONNX Runtime

### 5.1 Execution Provider 选择

ORT 通过 EP 抽象多后端，常见 EP + 选择场景：

| EP | 硬件 | 用途 |
|---|---|---|
| `CPUExecutionProvider` | 任何 CPU | 兜底；小模型 / preprocessing |
| `CUDAExecutionProvider` | NVIDIA GPU | 快启动、完全图支持；比 TRT EP 慢 10–50% |
| `TensorrtExecutionProvider` | NVIDIA GPU (+ TRT runtime) | 最快，但冷启动慢（需 build engine）|
| `TensorrtRTXExecutionProvider` (2025 新) | RTX 卡 + TRT-RTX runtime | 针对消费卡 + Windows 延迟优化 |
| `ROCmExecutionProvider` | AMD | — |
| `CoreMLExecutionProvider` | Apple Silicon | iOS/macOS |
| `QNNExecutionProvider` | Qualcomm (Snapdragon) | 边缘 NPU |
| `OpenVINOExecutionProvider` | Intel CPU/iGPU/Myriad | — |
| `DirectMLExecutionProvider` | Windows D3D12 | 跨厂商 |

### 5.2 Session Options

| 选项 | 取值 | 影响 |
|---|---|---|
| `graph_optimization_level` | `DISABLE_ALL` / `BASIC` / `EXTENDED` / `ALL` | BASIC = constant folding + redundant elim；EXTENDED = BASIC + layout 变换；ALL = EXTENDED + 运行时依赖的激进融合（可能影响 EP 兼容性） |
| `intra_op_num_threads` | int | 单 op 内并行线程（主要影响 CPU EP）|
| `inter_op_num_threads` | int | 节点间并行 |
| `execution_mode` | `SEQUENTIAL` / `PARALLEL` | 子图间并行；多 EP 场景小心 race |
| `enable_mem_pattern` | bool | 内存复用规划；动态 shape 建议关 |
| `enable_cpu_mem_arena` | bool | CPU 内存池 |

TRT EP 专属环境变量（部分）：
- `ORT_TENSORRT_FP16_ENABLE`
- `ORT_TENSORRT_INT8_ENABLE` + `ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME`
- `ORT_TENSORRT_ENGINE_CACHE_ENABLE` + `ORT_TENSORRT_CACHE_PATH`
- `ORT_TENSORRT_MIN_SUBGRAPH_SIZE`：小于此节点数的子图退回 CUDA/CPU。默认 1；增大可减少 engine 数量但损失 TRT 覆盖。
- `ORT_TENSORRT_MAX_PARTITION_ITERATIONS`：分区迭代上限，达到后整个模型 fallback 到其它 EP。

### 5.3 vs TRT 的权衡

| 维度 | 纯 TRT | ORT + TRT EP | ORT + CUDA EP |
|---|---|---|---|
| 启动时间 | 慢（build 一次 cache 后快）| 慢（同上）| 快 |
| 推理延迟 | 最低 | 接近 TRT（有 ORT 调度开销 <5%）| 比 TRT 慢 10–50% |
| 算子覆盖 | TRT 算子集（窄）| TRT 覆盖 + ORT fallback 到 CUDA EP，**算子全量覆盖** | 全量 |
| 动态 shape | profile 配置 | 通过 ORT 透传；跨 profile 切换无感 | 天然支持 |
| 部署复杂度 | 高 | 中（多一层 ORT）| 低 |
| 跨硬件可移植 | 差 | 好（换 EP 即可切到 CPU/其它 GPU）| 好 |

**ORT-TRT subgraph 拆分行为**：ORT 先做图优化 → 把可被 TRT 吃掉的子图切出（每个子图至少 `min_subgraph_size` 个节点）→ 每个子图独立 build 一个 engine → 运行时 TRT 子图走 TRT EP，其余节点走 CUDA / CPU EP。如果模型里有**一个 TRT 不支持的算子穿插在中间**，会把图劈成碎片，每段都要单独 engine，性能往往不如纯 CUDA EP。解决：升级 TRT 版本 / 改写该算子 / 用 ORT custom op 替换。

**什么时候用 ORT 而不是 TRT**：
- 模型有 TRT 不支持的算子但改不动（如某些 custom attention）。
- 需要跨硬件部署（同一代码切 EP）。
- 动态 shape 变化非常频繁，不想维护多 profile。
- 快速原型 / 多模型串流水线（ORT 的 IO binding 比 TRT 的 context 管理容易）。

---

## 6. TVM / Ansor / MetaSchedule

### 6.1 搜索空间演进

| 框架 | 时期 | 搜索方式 | 搜索空间 |
|---|---|---|---|
| **AutoTVM** | 2018– | 人写 template（schedule primitives）→ XGBoost cost model + 演化算法挑参数 | 受 template 表达力限制；每个 op 要写 template |
| **Ansor (Auto-Scheduler)** | 2020– | 无模板；自动生成 sketch → 采样填充 → cost model 排序 | 大幅扩展；跨 op 融合 |
| **MetaSchedule** | 2022– | TIR 层统一抽象；tunable scheduling rule 模块化 + python 可扩展 | 最灵活；支持 tensorization（Tensor Core） |
| **Relax + Unity (2024+)** | — | dynamic shape 支持 + BYOC | 与硬件后端（CUTLASS / TensorRT / TRT-LLM）协同 |

### 6.2 Tuning 预算

实测经验（Resnet50 / BERT-base 级别网络，A100）：

| Trials / op | 相对 TRT 性能 | 耗时 |
|---|---|---|
| 100 | 60–80% | 分钟级 |
| 1000 | 85–95% | 小时级 |
| 10000 | 95–105%（偶尔超 TRT） | 半天–一天 |
| 20000+ | 边际收益小 | 1–2 天 |

**何时值得用 TVM 而不是直接 TRT**：

- 目标硬件 **TRT 不支持**（RISC-V GPU、某些 NPU、AMD）。
- 模型里有**奇形怪状的算子**（非标 attention、稀疏算子）TRT 不覆盖。
- 需要**极致的 kernel 融合**（TVM 可跨多层融合 Conv+BN+ReLU+Add+Pool，TRT 的融合规则固定）。
- 批量研究**硬件抽象**（同一 IR 多后端）。

**何时别用**：
- 目标是 NVIDIA GPU + 常见 CNN/Transformer → 直接 TRT，tuning 时间节省 10–100x。
- CI 严格，每次改网络要重跑 tuning，tuning 不稳定（方差大）。

对 UniV2X：BEV + Transformer 主线在 NVIDIA 栈上，**不建议引入 TVM**；如果未来要支持非 NVIDIA 边缘设备（如 V2X 路侧国产 NPU），再考虑 TVM / MetaSchedule。

---

## 7. 汇总表：NVIDIA 栈 D 空间维度

| 层级 | 维度 | 典型取值 | 默认 | 代表资源 | 说明 |
|---|---|---|---|---|---|
| 资源分配 | CUDA Stream 数 | 1 – 8 | 1 | 调度 | 多路并发基础 |
| 资源分配 | Stream priority | [-5, 0] | 0 | SM 调度 | 高优先抢占调度 |
| 资源分配 | MPS active thread % | 10 – 100 | 100 | SM 软分区 | 多进程并发（Jetson Orin 支持自 JP 6.1） |
| 资源分配 | MIG profile | 1g/2g/3g/4g/7g | 无 | SM + L2 + mem 硬分区 | A100/H100/B200/Thor 专有 |
| TRT Builder | precision flags | FP32/FP16/BF16/INT8/FP8/INT4 | FP32 only | Tensor Core | 组合决定 kernel 集 |
| TRT Builder | tactic sources | CUBLAS / CUBLAS_LT / CUDNN / EDGE_MASK / JIT | TRT<10 全开，TRT10+ 不含 CUDNN | kernel 库 | 影响 engine 依赖和候选 |
| TRT Builder | workspace pool | 64MB – 8GB | 平台相关 | 显存 | 太小会错失快 tactic |
| TRT Builder | opt profile 数 | 1 – 4 | 1 | engine size | 多 profile 覆盖 shape 范围 |
| TRT Builder | min/opt/max shape | 任意 | 必填 | tuning 目标 | opt 决定主 tactic |
| TRT Builder | STRONGLY_TYPED | on/off | off | 类型推导 | TRT 10 推荐 |
| TRT Builder | REFIT | on/off | off | 权重热替换 | engine 变大 |
| TRT Builder | SPARSE_WEIGHTS | on/off | off | 2:4 sparse TC | 见 pruning 报告 |
| TRT Builder | DLA core | 0 / 1 / none | none | DLA | Orin AGX 有两个 |
| TRT Builder | GPU fallback | on/off | off | DLA | DLA 不支持的层走 GPU |
| TRT Builder | timing cache | 路径 | 无 | 复用 | 加速重复 build |
| CUDA Graph | 是否启用 | on/off | off | CPU launch | 小 kernel 密集网络 +10–30% |
| cuDNN / cuBLAS | algo mode | heuristics / exhaustive | heuristics | kernel 选择 | PyTorch `benchmark=True` 激活 |
| cuBLASLt | workspace | 1 – 32MB | 4MB | 显存 | 32MB 在 Hopper 是甜点 |
| PyTorch | cudnn.benchmark | on/off | off | kernel cache | shape 稳定时开 |
| PyTorch | cudnn.deterministic | on/off | off | 复现 | 慢 5–15% |
| PyTorch | allow_tf32 | on/off | on (Ampere+) | TF32 | 精度敏感关 |
| ORT | graph_optimization_level | DISABLE / BASIC / EXTENDED / ALL | ALL | 图优化 | 按 EP 可能降级 |
| ORT | EP 组合 | TRT / CUDA / CPU / ... | CPU | 后端 | EP 顺序决定 fallback 链 |
| ORT-TRT | MIN_SUBGRAPH_SIZE | 1 – 20 | 1 | 分区粒度 | 大减 engine 数 |
| ORT-TRT | engine cache | on/off + 路径 | off | 冷启动 | 生产必开 |
| TVM | scheduler | AutoTVM / Ansor / MetaSchedule | — | 搜索 | MetaSchedule 当前推荐 |
| TVM | trials / op | 100 – 20000 | — | tuning 时间 vs 性能 | 1000 是常用折中 |

---

## 8. 给 UniV2X 的建议

结合 UniV2X "车端 Orin + 路侧 V2X + 云端训练" 的三端特征：

1. **车端 Orin（主推理）**：
   - **不要指望 MIG**（iGPU 不支持），走 "每模型独立进程 + MPS + percentage 限制" 的软分区方案；主路径独占高优先级 stream。如果未来升级到 Jetson Thor，可切到 MIG + MPS 混合。
   - TRT Builder：**FP16 + INT8 混合** + `OBEY_PRECISION_CONSTRAINTS` + `STRONGLY_TYPED`（TRT 10），engine 与 timing cache 纳入版本管理。
   - DLA：**只对"纯 Conv + 简单 pooling 的子模型"用 DLA**（如早期特征提取），避免 attention/norm-heavy 子图上 DLA 后大量 fallback。
   - CUDA Graph：对"静态 shape 的主感知+检测路径"启用，对动态融合层不用。
2. **路侧 / 云端**（数据中心卡）：
   - 多路摄像头融合场景用 **MIG + MPS** 组合，MIG 做硬隔离（不同路口/不同任务），MIG 实例内再 MPS 多模型。
   - 批量离线推理：大 workspace（4–8GB）+ exhaustive tactic search + 持久化 timing cache。
3. **D 空间实验表设计**：每个候选点应包含完整维度标签 —— `(GPU 型号, TRT version, precision flags, tactic sources, workspace, profile, cuda graph, stream count, MPS %, MIG profile)`，避免对比时漏掉隐式维度；否则两次实验差 30% 可能只是 cudnn.benchmark 是否打开。
4. **研究优先级**：
   - 先搞定 TRT precision + profile + timing cache（收益最大，90% 性能差距在这）。
   - 再上 CUDA Graph + stream priority（主路径延迟尾部优化）。
   - 最后才考虑 TVM / MetaSchedule（投入产出比在 NV 栈上不佳）。
5. **可观测**：在部署脚本里记录 `trtexec --verbose` 日志中的 tactic 分布 + peak activation + engine size；作为回归基线。每次 TRT 版本升级都可能换 tactic，不可假设不变。

---

## 参考来源

- NVIDIA TensorRT Developer Guide (10.x) — IBuilderConfig, BuilderFlags, TacticSource, MemoryPoolType。
- NVIDIA Multi-Instance GPU User Guide — Deployment Considerations。
- NVIDIA CUDA MPS Documentation + Jetson Forum 讨论"Proper use of CUDA MPS"（Orin 支持自 JP 6.1）。
- NVIDIA Blog "What's New in CUDA Toolkit 13.0 for Jetson Thor"（Thor MIG + green contexts + MPS）。
- Antmicro 博客 2025-02 "Jetson Orin Baseboard Super performance boost"。
- ONNX Runtime Docs — TensorRT Execution Provider / TensorRT RTX EP。
- Apache TVM / MetaSchedule / Relax-Unity 官方文档。
- PyTorch cuDNN / cuBLAS benchmark 相关文档与源码注释。
