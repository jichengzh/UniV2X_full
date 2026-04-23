# 非 NVIDIA 栈 D 空间调研

> 本文聚焦各家非 NVIDIA AI 加速器在"部署时有哪些可调旋钮 (D 空间维度)"。
> 与 `ad_v2x_deployment.md` 的定位区别：该文讲"这颗芯片是什么、算力/接口/场景"，
> 本文讲"同一颗芯片面向同一个模型，能配置出多少种部署变体"。
> 目标是为 UniV2X 的跨硬件部署抽象出统一的配置维度坐标系。

---

## 1. Qualcomm 栈（Hexagon / HTP / QNN / SNPE）

Qualcomm 面向车载 (Snapdragon Ride / SA8650P / SA8775P) 和端侧 (骁龙 8 Gen 3 / X Elite) 的 AI 栈有两条并行线：
**SNPE**（老 API，偏遗留）和 **QNN SDK**（新 API，Qualcomm AI Engine Direct）。当前车载量产项目基本都是 QNN。

### 1.1 Backend 选择（最粗粒度的 D 维度）

QNN 通过"Backend Library"抽象不同执行单元，一个模型可在运行时切换：

| Backend library | 目标 IP | 典型用途 |
|----------------|--------|----------|
| `libQnnCpu.so` | ARM CPU | 调试、兜底、小算子 |
| `libQnnGpu.so` | Adreno GPU (OpenCL) | 中等精度流式负载 |
| `libQnnDsp.so` | Hexagon DSP (Scalar/Vector) | 老一代（SM8250 之前） |
| `libQnnHtp.so` | Hexagon Tensor Processor (HTP) | **当前主力，INT8/INT16 为主** |
| `libQnnSaver.so` | 离线序列化 | 导出 context binary |
| `libQnnHtpMcp.so` | HTP on automotive MCP | Ride 平台专用 |

维度枚举：`backend ∈ {CPU, GPU, HTP, HTP-FP16, HTP-QUANT}`。对每个 subgraph 可单独指定。

### 1.2 精度与图划分

**精度 (precision)**：
- HTP 量化路径：`INT4`（SM8650 起）/ `INT8` / `INT16` / `UINT8`。权重与激活可分开设置。
- HTP FP 路径（新 HTP）：`FP16` 权重+激活，不需要校准。
- `quantization_overrides.json` 允许对每个张量覆盖默认量化参数（scale/offset，per-channel/per-tensor）。

**图划分 (graph partitioning)**：
- QNN converter (`qnn-onnx-converter` / `qnn-pytorch-converter`) 产出 `.cpp`+`.bin` → `qnn-model-lib-generator` 编译为 `.so`。
- `qnn-context-binary-generator` 再把 `.so` 序列化为 context binary（离线编译好的 HTP 指令）。
- 不支持算子自动回退给 CPU backend，也可用 `--op_package_lib` 注册用户自定义算子 (UDO / OpPackage)。
- `--vtcm_mb` 控制 VTCM（HTP 内部 tightly-coupled memory）占用，通常 4/8 MB。
- **HTP graph partitioning**：当模型大于单次可常驻的 VTCM，可拆多个 context，对应 `partition` 粒度；也可利用 multi-PD（Protection Domain）并发多个 graph。

**性能 hint**：
- `QnnHtpGraph_setConfig`：`QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION_TYPE` (1/2/3 → 编译时 vs 运行时权衡)。
- `QnnHtpPerfInfrastructure`：`powerConfig` 设置 DCVS voltage corner / sleep latency / RPC polling。Ride 平台有额外的 `AutomotiveSafetyConfig`。

### 1.3 自动驾驶平台（Snapdragon Ride）

SA8650P / SA8775P 在单颗 SoC 内同时给 QNX 安全域和 Linux/Android 应用域调度 HTP 时间片：
- **QoS / 优先级**：`QNN_PRIORITY_{LOW, NORMAL, HIGH}`，Ride 上 AEB / 感知等安全关键图占 HIGH，APA/泊车占 NORMAL。
- **Multi-PD**：一个 HTP 可暴露给多个进程，通过 `QnnDevice` 的 PD 绑定配置。
- **FSI/MCG 副核**：Safety Island / Multi-Core GPU 可做 fallback 冗余。
- **Context caching**：Ride 出厂镜像会预生成 context binary 避开首帧编译延迟（对实时 SLA 很重要）。

**D 空间小结**：`{backend, precision(W,A), vtcm, opt_level, power_corner, priority, partition_count, PD_id}` 8 维。

---

## 2. Apple 栈（ANE / CoreML / MLX）

Apple 不做车，但 ANE 是当下唯一大规模部署的"专用 NPU + 成熟编译器栈"的对照样本，值得记一笔。

### 2.1 MLComputeUnits 配置

CoreML 的运行时指定三选项之一（或 ALL）：

| MLComputeUnits | 含义 |
|---|---|
| `.cpuOnly` | 纯 CPU（调试 & 确定性）|
| `.cpuAndGPU` | CPU + Metal GPU |
| `.cpuAndNeuralEngine` | CPU + ANE，iOS 14+ 默认推荐 |
| `.all` | CPU + GPU + ANE 自动调度 |

ANE 不独立暴露，CoreML compiler 会把算子按支持度切片：不支持的算子会自动 fallback 给 GPU 或 CPU。
**关键不可配置点**：ANE 的 op list、tiling、内存布局都由 compiler 决定，开发者只能影响"输入/输出"层面。

### 2.2 精度选项

- **权重量化**：`coremltools` 提供 `linear_quantize_weights`（INT8 / INT4 per-channel）、`palettize_weights`（权重调色板，2~8 bit）、`prune_weights`（稀疏）。
- **激活精度**：iOS 17 起可在 MIL 层指定 `compute_precision = FP16 / FP32`；ANE 内部普遍 FP16。
- **Stateful model** (iOS 18)：允许 KV cache 持久化，减少 LLM 场景重复拷贝。
- **MLModel 优化级别**：`.cpuOnly` 下保留 FP32 参考；`.all` 下 compiler 会做 ANE-friendly 重排（如把 LayerNorm 融合到 Conv 尾部）。

### 2.3 MLX on Apple Silicon

MLX 是 Apple 2023 年发布的数值框架，直接在 Unified Memory 上运行：
- 粒度细于 CoreML：显式 `mx.array(dtype=mx.float16 / bfloat16 / int8)`。
- `mx.set_default_device(mx.gpu / mx.cpu)`。无 ANE 后端（截至 2025）。
- 量化 API：`mx.quantize(weights, bits=4, group_size=64)`。
- MLX 对 LLM 部署有用，但对实时感知/规控用得少。

**D 空间小结**：`{compute_units, compute_precision, weight_quant(bits, per-channel), palettize, prune, stateful}` 6 维。ANE 细节对开发者不透明，可调空间相对小。

---

## 3. 华为 Ascend 栈（CANN / TBE / GE）

Ascend 910B/910C 做训练与大模型推理，Atlas 300I Pro / Atlas 200 DK A2 做推理卡，MDC 810 做车载。CANN 是统一栈。
补充 `ad_v2x_deployment.md` 未深入的可配置维度：

### 3.1 算子 tiling（TBE / Ascend C）

Ascend 的达芬奇核心有三级存储：`GM (Global) → L1 → L0A/L0B/L0C (Cube)` + UB (Unified Buffer)。
算子实现两条路径：
- **TBE (Tensor Boost Engine)**：Python DSL，类似 Halide / TVM，`tbe.dsl.auto_schedule`。
- **Ascend C**（2023 起主推）：C++ 显式编程 Cube / Vector 单元，对 Transformer 性能是主路径。

可配置 tiling 参数（每个算子都有 `op_tiling` 回调）：
- `M/N/K` 在 Cube 上的 tile 尺寸（典型 16×16×16 FP16 Cube 原子）。
- `L1 buffer` 分配 (byte)、`UB buffer` 分配、`double-buffer on/off`（流水隐藏访存）。
- `workspace_size`：动态 shape 下的 scratch 内存。
- `op_select_implmode`：`high_performance` / `high_precision`，影响是否启用 FP16 快速路径（牺牲 ULP）。

ATC (Ascend Tensor Compiler) 命令行：`--precision_mode={force_fp16, allow_fp32_to_fp16, must_keep_origin_dtype, allow_mix_precision}`，这一项是 Ascend 独有的精度粒度控制。

### 3.2 AI Core vs AI CPU 分配

Ascend SoC 同时含 **AI Core (Cube+Vector, 达芬奇)** 和 **AI CPU (通用 ARM 核做 control flow / 少量 elementwise)**。

GE (Graph Engine) 在图编译时决定每个算子落在哪个执行器：
- `engine="AiCoreEngine"` / `"AiCpuEngine"` / `"VectorEngine"` (910B 起独立 Vector) / `"HcclEngine"`。
- `AOE (Ascend Optimization Engine)`：`aoe_mode` 取 `1`(算子调优) / `2`(子图调优) / `4`(梯度切分调优)。跑一次 AOE 会生成 `aoe_config.json`。
- **融合规则**：`fusion_switch.cfg` 可按名称开关融合 pass（例如 `LayerNormFusionPass: off`）。调试不稳定算子必用。

### 3.3 HCCL 集合通信

910 系列做多卡训练/推理，HCCL (Huawei Collective Communication Library) 对应 NCCL：
- 通信算法：`HCCL_ALGO=AllReduce:ring / AllReduce:double-binary-tree`。
- `HCCL_BUFFSIZE_MB`：通信缓冲区大小。
- `HCCL_EXEC_TIMEOUT`：超时阈值，训练长尾。
- 拓扑感知：8 卡 HCCS 全互联 vs 跨 Node RoCE，算法默认值不同。

### 3.4 混合精度

- `mix_precision` in MindSpore / `npu_apex` in PyTorch+Ascend。
- 算子黑白名单：`modify_mixlist`，可把 softmax / layernorm 保持 FP32。

### 3.5 MDC 810 与 Atlas 推理卡

MDC 810（昇腾 610 × N）面向 L4 / Robotaxi：
- 多 Die 之间通过 HCCS-lite 共享。
- 安全隔离：每个 AI Core cluster 可分配给不同 ASIL 分区（VOS 之上的隔离）。
- 离线模型格式 `.om`，同一个 ONNX 导出 `atc` 时可按目标 SoC 指定：`--soc_version={Ascend310B1, Ascend310P3, Ascend910B1, Ascend610}`，同一个模型在不同 SoC 上二进制不同。

**D 空间小结**：`{precision_mode, aoe_mode, fusion_switch, tile(M,N,K,L1,UB), double_buffer, engine_assignment, hccl_algo, mix_precision_list, soc_version}` 9+ 维。Ascend 可调面最广也最复杂。

---

## 4. Intel 栈（OpenVINO / NNCF / Gaudi）

OpenVINO 是 x86 推理栈事实标准，2024 起也支持 NPU（Meteor Lake / Lunar Lake 的 Movidius 派生 IP）。

### 4.1 Device plugins 与 hint 配置

设备 plugin：`CPU` / `GPU` (iGPU + dGPU Arc) / `NPU` / `GNA`（低功耗语音）/ `AUTO` / `MULTI` / `HETERO`。

- `AUTO`：首选 GPU，首次推理期间自动在 CPU 上热身，对冷启动延迟友好。
- `MULTI:GPU,CPU`：并发跑两设备做 throughput 叠加。
- `HETERO:GPU,CPU`：按算子切图，类似 Qualcomm partitioning。

**Hints（OpenVINO 2022.3 起的声明式配置）**：
```
ov::hint::performance_mode   = {LATENCY, THROUGHPUT, CUMULATIVE_THROUGHPUT}
ov::hint::num_requests       = N                 # 预估并发请求数
ov::hint::inference_precision = {f32, f16, bf16, i8}
ov::hint::execution_mode     = {ACCURACY, PERFORMANCE}
ov::hint::model_priority     = {LOW, MEDIUM, HIGH}
ov::hint::scheduling_core_type = {PCORE_ONLY, ECORE_ONLY, ANY_CORE}   # Alder/Raptor Lake 大小核
```
hint 之下 runtime 会自动反推底层的 streams / threads / 亲和性。

### 4.2 Streams / threads 调优（精细档）

如果 hint 不够用，可直接指定：
```
ov::streams::num           = N   # 推理 stream 数，决定并发 infer_request 上限
ov::inference_num_threads  = M   # 每个 stream 内的线程数
ov::affinity               = {NONE, CORE, NUMA, HYBRID_AWARE}
ov::enable_cpu_pinning     = true/false
```
经验规则：**LATENCY 模式** → `streams=1, threads=物理核数`；**THROUGHPUT 模式** → `streams ≈ 核数/4, threads=4`（OpenVINO 默认启发式就是这样）。
GPU plugin 上对应：`ov::num_streams` + `GPU_THROUGHPUT_AUTO`。

### 4.3 NNCF（Neural Network Compression Framework）

训练时量化/剪枝入口：
- **PTQ**：`nncf.quantize(model, calibration_dataset, preset={PERFORMANCE, MIXED}, target_device={ANY, CPU, GPU, NPU, VPU}, subset_size)` —— `target_device` 会改变可用的量化 op 白名单。
- **QAT**：`NNCFConfig` YAML 描述 `compression: [{algorithm: quantization, preset: mixed, ignored_scopes: [...]}]`。
- **Sparsity**：`magnitude_sparsity` / `rb_sparsity`，2:4 / 4:8 结构化。
- **AWQ / GPTQ**：LLM 权重压缩新入口（`nncf.compress_weights(mode=INT4_ASYM, group_size=128, ratio=0.8)`）。

### 4.4 Model Optimizer / OVC（离线配置）

MO 已废弃，统一用 `ovc`（OpenVINO Converter）：
- `--compress_to_fp16=True`（默认）/ False
- `--input`、`--output`：图裁剪
- 保存为 `.xml + .bin` IR；`ov::CompiledModel::export_model` 可为目标设备生成 blob 缓存。

### 4.5 Intel Gaudi 简介

Gaudi2 / Gaudi3（Habana，Intel 2019 收购）：
- 栈叫 **SynapseAI**（不是 OpenVINO），框架侧通过 `habana_frameworks.torch` + `hpu` 设备接入 PyTorch。
- 关键可调维度：
  - `PT_HPU_LAZY_MODE=1` / `0`（lazy vs eager）。Lazy 模式下会聚合图做 Graph Compiler 优化。
  - `HCL`（Habana Collective Lib）：拓扑算法类似 HCCL/NCCL。
  - BF16 是默认精度；FP8 在 Gaudi3 上通过 `--fp8_format=E4M3/E5M2` 启用。
  - RecipeCache：`PT_HPU_RECIPE_CACHE_CONFIG` 控制编译产物缓存。
- 相对 NVIDIA 最大 gap：算子覆盖度 & 生态，不是硬件可调维度。

**D 空间小结**：`{device, streams, threads, affinity, precision_hint, perf_mode, priority, compress_bits, sparsity_ratio, gaudi_lazy_mode}` 10 维。OpenVINO 的 hint 抽象是业界最清晰的声明式配置样板。

---

## 5. AMD 栈（ROCm / MIGraphX）

AMD 的服务器 GPU 线（MI250X / MI300X / MI300A）靠 ROCm，客户端 Ryzen AI 用 XDNA NPU（Vitis AI 栈，偏 Xilinx 传承）。

### 5.1 MIGraphX 配置

MIGraphX 是 AMD 官方推理引擎，对应 TensorRT/OpenVINO：
- 离线编译：`migraphx-driver compile model.onnx --gpu --fp16 --exhaustive-tune`。
- **`--exhaustive-tune`**：对每个算子尝试多个 MIOpen kernel 组合并缓存最佳选项（类似 TensorRT 的 builder）。结果落到 `~/.composable_kernel/` 或 MIOpen 的 `~/.config/miopen/`。
- 精度：`--fp16` / `--int8`（需 calibration 数据）/ `--bf16`（MI300 起）。
- Dynamic shape：`--input-dim @input 1 3 224 224`；runtime 可 re-shape。
- Provider：同一个 ONNX 也能走 **ONNX Runtime + ROCmExecutionProvider** 或 **MIGraphXExecutionProvider**，后者会走 MIGraphX 整图编译。

MIOpen（cuDNN 对标品）的调优：
- `MIOPEN_FIND_MODE={1,2,3,4,5}`：1=Normal（默认）, 2=Fast, 3=Hybrid, 4=DynamicHybrid, 5=FastHybrid。等价 cuDNN benchmark 模式。
- `MIOPEN_USER_DB_PATH` / `MIOPEN_SYSTEM_DB_PATH`：tuning 数据库持久化位置。
- Composable Kernel (CK)：MI300 上的 matmul / attention 主路径，`CK_TIME_KERNEL=1` 可 profile tile size。

### 5.2 MI300 推理生态

- **vLLM-ROCm**：vLLM 上游已原生支持 ROCm，`--dtype=bfloat16/float16`；PagedAttention 走 CK。
- **Triton Inference Server**：ROCm backend 可用，但模型仓 instance 的 gpu-id 调度与 NVIDIA 版本一致。
- **FP8**：MI300X 支持 E4M3/E5M2，走 `torch.float8_e4m3fn`（PyTorch ROCm fork）。
- **rocBLAS / hipBLASLt**：GEMM 选 tile 的环境变量 `HIPBLASLT_TENSILE_LIBPATH`。
- **HIP Graph**：和 CUDA Graph 语义一致，减少启动开销。

### 5.3 Ryzen AI / XDNA NPU

笔记本端侧 NPU（Phoenix / Strix Point），栈是 **Vitis AI + ONNX Runtime VitisAIExecutionProvider**：
- 量化：`vai_q_onnx`，INT8 对称权重+激活。
- Partitioning：不支持算子自动切回 CPU。
- Subgraph `xclbin` 二进制在设备上加载。

**D 空间小结**：`{dtype, exhaustive_tune, miopen_find_mode, hip_graph, fp8_mode, provider_choice}` 6 维。相对 OpenVINO / QNN 可调面更少，但与 NVIDIA 最接近（cuDNN↔MIOpen、TRT↔MIGraphX 语义同构），迁移成本低。

---

## 6. 其他厂商（简要）

### 6.1 Google TPU / XLA

- TPU 只通过 XLA 编程，不存在"手写算子"的常规路径。
- 主要可调维度：
  - `jax.jit` 下的 `donate_argnums`（原地更新减少拷贝）、`mesh` + `PartitionSpec`（SPMD 切分）。
  - `xla_flags`：`--xla_gpu_enable_triton_softmax_fusion`、`--xla_tpu_spmd_rng_bit_generator_unsafe`、auto-sharding。
  - 精度：TPU v5e/v5p 上 `bf16` 默认，`int8` 通过 AQT (Accurate Quantized Training)。
- 对 V2X 不适用（TPU 不在车端），只作训练平台。

### 6.2 昆仑芯（Baidu）

- XPU R200 / P800，栈 `XTDK`（类 CUDA） + `XDNN`（类 cuDNN） + `PaddlePaddle xpu backend`。
- 可调：`FLAGS_selected_xpus`（设备号）、`XPUAPI_DEFAULT_SIZE`（workspace）、`xpu::Context::set_l3_size`（L3 cache 划分，对 LLM 推理很关键）。
- 推理引擎：Paddle Inference / XPU 插件，精度 `FP16 / INT8 / INT16`。

### 6.3 寒武纪（Cambricon）

- MLU370-X8 / 590，栈 `CNToolkit`（Neuware）：`CNRT`（runtime）+ `CNNL`（类 cuDNN）+ `CNCL`（集合通信）+ `MagicMind`（推理引擎）。
- MagicMind 构建时可调：`precision={qint8_mixed_float16, qint8_mixed_float32, force_float16}`，`cluster_num`（MLU 的 cluster 粒度并行），`dynamic_shape_range`。
- 算子扩展：`BangC`（类 CUDA C 的 DSL）。

### 6.4 摩尔线程（Moore Threads）

- MTT S3000 / S4000 (MUSA 栈)：`musaMalloc` / `mcBLAS` / `mcDNN`，API 与 CUDA 兼容度高，`MUSIFY` 工具做源码移植。
- 推理引擎 `MTIR`。可调维度公开资料少，集中在 precision 和 workspace。

### 6.5 天数智芯（Iluvatar CoreX）

- BI-V100 / V150，提供 **CUDA 兼容层**（`ixCUDA`），PyTorch/TF 通过标准 CUDA API 调用。
- 等同于"替换 nvcc"，可调维度大体跟随 NVIDIA 原生栈（cudnn_benchmark、amp、dtype），对迁移友好。

---

## 7. 汇总对比表：各厂商 D 空间维度

| 厂商 | 主栈 SDK | 设备/Backend 选择 | 精度 | Op 级调优 | 并发/调度 hint | 典型 V2X 场景 |
|---|---|---|---|---|---|---|
| **Qualcomm** | QNN SDK / SNPE | CPU/GPU/HTP/HTP-MCP | INT4/8/16, FP16, 混合 | UDO, VTCM size, opt_level | priority, multi-PD, power_corner | Ride 车端 (SA8650/8775) |
| **Apple** | CoreML + MLX | CPU/GPU/ANE (via MLComputeUnits) | INT4/8, FP16, palettize | 不开放 | compiler-only | 端侧 (非车载参考) |
| **华为昇腾** | CANN (ATC/GE/TBE) | AiCore/AiCpu/Vector | precision_mode 4 档 | **tiling (M,N,K,L1,UB,DB)**, AOE, fusion_switch | HCCL algo, priority | MDC 810 / Atlas 300I |
| **Intel** | OpenVINO + NNCF | CPU/GPU/NPU/GNA + AUTO/MULTI/HETERO | FP32/FP16/BF16/INT8/INT4 | NNCF quantize/prune/AWQ | **hint::perf_mode, streams, threads, affinity** | 路侧 RSU x86 / 边缘服务器 |
| **Intel Gaudi** | SynapseAI | HPU | BF16, FP8 (Gaudi3) | Graph Compiler (封闭) | HCL, recipe cache | 训练侧 |
| **AMD** | ROCm + MIGraphX | GPU (gfx90a/gfx942) | FP16/BF16/INT8/FP8 | exhaustive_tune, MIOPEN_FIND_MODE | HIP graph, ORT provider 选择 | 数据中心推理、仿真 |
| **Google** | XLA / JAX | TPU | BF16, INT8 (AQT) | auto-sharding, xla_flags | SPMD mesh | 训练集群 |
| **昆仑芯** | XTDK + XDNN | XPU | FP16/INT8/INT16 | L3 size, workspace | PaddleInference 线程 | 国产替代路侧 |
| **寒武纪** | Neuware + MagicMind | MLU cluster | INT8 mixed FP16/32 | cluster_num, dynamic_shape_range | BangC 自定义 | 国产替代路侧 |
| **摩尔/天数** | MUSA / ixCUDA | GPU (CUDA 兼容) | FP16/INT8 | 跟随 CUDA 语义 | CUDA-like streams | 训练/仿真 |

---

## 8. 关键洞察

### 8.1 共性：D 空间的"四层旋钮"抽象

不管哪家栈，部署时可配置维度都可归为四层：
1. **设备路由 (device routing)**：哪颗 IP 跑哪个 subgraph。QNN backend / CoreML MLComputeUnits / OpenVINO AUTO/HETERO / CANN engine assignment 都是一回事。
2. **精度 (numeric precision)**：权重 × 激活 × 累加器三条独立轴，再加量化粒度 (per-tensor vs per-channel vs per-group)。
3. **算子内部 (intra-op)**：tile size、workspace、kernel 选择（MIOpen find mode、AOE、exhaustive_tune）、融合开关。
4. **并发/调度 (inter-op)**：streams / threads / priority / 多 PD / 多实例。

这四层可以作为 UniV2X 硬件抽象层的维度骨架，把 TensorRT 特定配置和 QNN/CANN/OpenVINO 对齐。

### 8.2 差异：封闭 vs 开放

- **最封闭**：Apple ANE、Intel Gaudi Graph Compiler、Google TPU —— 几乎没有 intra-op 旋钮。
- **中等**：Qualcomm QNN、AMD MIGraphX、NVIDIA TensorRT —— 开放精度/设备/tuning 开关，但不暴露 tile。
- **最开放**：华为 CANN（Ascend C / AOE / fusion_switch）、Intel OpenVINO（streams/threads/NNCF）—— 连 tile 都能手写。

V2X 部署策略建议：**把 D 空间设计成"声明式 + 可降级"**，即优先用 OpenVINO-hint 风格的高层意图（LATENCY vs THROUGHPUT），必要时再落到底层 tile/streams。这既能覆盖 Apple/NVIDIA 这种封闭栈，也能利用 Ascend 这种开放栈的上限。

### 8.3 对 UniV2X 的启示

1. **车端感知 (Qualcomm Ride / MDC 810 / Horizon J6)** —— D 空间要把 **priority / multi-PD / 安全分区** 纳入维度，这是 ad_v2x_deployment.md 没覆盖的关键一块，因为它直接决定 ASIL 与混合负载（感知+规控+V2X）下的 tail latency。
2. **路侧 RSU (Intel NUC / 国产 x86 + 昆仑芯/寒武纪卡)** —— D 空间的核心是 **streams/threads/batch**，单台 RSU 通常并发 4~16 个交叉口摄像头流，用 OpenVINO `THROUGHPUT` hint + MULTI plugin 是直接可用样板。
3. **云端训练与仿真 (NVIDIA / AMD MI300 / 昇腾 910)** —— D 空间偏集合通信（NCCL / HCCL / RCCL / HCL）和混合精度（FP8/BF16），与部署 D 空间差异明显，需要在架构上隔离。
4. **跨厂商可移植性** —— 至少 Qualcomm/Intel/Ascend/AMD 的量化都支持 INT8 对称 + per-channel，这一"最小公倍数"应作为 UniV2X 的默认量化契约；高级特性（INT4、palettize、FP8）按目标硬件开启分支。
5. **编译期缓存是必调项** —— QNN context binary、CoreML `.mlmodelc`、OpenVINO `export_model` blob、MIGraphX tuning DB、AOE json 都是"首次启动冷/之后热"的典型。部署流水线必须把缓存目录纳入制品管理。

### 8.4 风险点

- 非 NVIDIA 栈的算子覆盖度普遍弱于 cuDNN/TRT，UniV2X 如果依赖非常规 op（自定义 Transformer、非标 attention mask），要预留 **自定义算子通道**：QNN UDO / CoreML MIL / Ascend C / MIOpen user kernel / VitisAI custom op。
- 量化 calibration 数据的跨厂商复用性差：各家量化参数格式不同（QNN 的 `quant_overrides.json` ≠ OpenVINO 的 `nncf.IgnoredScope` ≠ Ascend 的 `quant_config.json`）。UniV2X 应保存 **float32 校准集 + per-layer scale 中间表示**，而不是直接保存某一家的量化产物。
