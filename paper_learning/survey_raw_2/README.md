# 调研 Round 2：D 空间深度 + B ↔ D 交互

> **目标**：补齐对「固定硬件上的可配置维度 (D 空间)」和「量化/剪枝 ↔ 硬件维度」交互的认知。
> **背景**：Round 1 已产出 `搜索空间体系化总结.md`，定位 UniV2X 方案在 A/B1/B2/C/D/E 六子空间中的 B1+B2 ↔ D 紧耦合。但对 D 空间和 B↔D 的具体技术细节不够深入。本轮补足。
> **方法**：6 个并行代理深度调研，产出 6 份专题文件（本 README 之外），再做跨簇综合。
> **日期**：2026-04-23

---

## 1. 子报告导航

| 文件 | 聚焦 | 核心贡献 |
|------|------|----------|
| [`d_space_nvidia.md`](./d_space_nvidia.md) | NVIDIA 栈深度 | CUDA Streams/MPS/MIG、TRT Builder 全配置、CUDA Graph、cuDNN/cuBLASLt tactic、ORT、TVM |
| [`d_space_non_nvidia.md`](./d_space_non_nvidia.md) | 非 NVIDIA 栈 | Qualcomm QNN、Apple CoreML/ANE/MLX、华为 CANN/TBE/GE、Intel OpenVINO/NNCF/Gaudi、AMD ROCm/MIGraphX，+ TPU/昆仑芯/寒武纪 |
| [`quantization_x_hardware.md`](./quantization_x_hardware.md) | 量化 × 硬件交互 | 量化格式 ↔ tactic、DLA fallback、KV cache 量化的并发效应、TRT Model Optimizer / AWQ / SmoothQuant |
| [`pruning_x_hardware.md`](./pruning_x_hardware.md) | 剪枝 × 硬件交互 | 2:4 Sparse Tensor Core、cuSPARSELt、Block-SpMM 交叉点、DLA 稀疏、动态剪枝流水 |
| [`hw_aware_nas_and_joint_search.md`](./hw_aware_nas_and_joint_search.md) | 硬件感知联合搜索 | L0-L3 层次划分、APQ/HAQ/JAQ/NAHAS/Vidur/FlexGen/DistServe/REEF 等按「D 入搜」归类 |
| [`ad_v2x_deployment.md`](./ad_v2x_deployment.md) | 自动驾驶/V2X 部署 | Jetson Orin 型号差异、DLA 实战、nvpmodel/EMC、BEV 部署、Horizon/华为 MDC/黑芝麻、V2X 通信维度 |

---

## 2. D 空间的分层抽象：四层旋钮

跨所有厂商综合，**D 空间可归结为四层旋钮**（本轮调研的主要抽象贡献）：

```
┌─────────────────────────────────────────────────────────────────┐
│ D 空间四层旋钮                                                   │
├─────────────────────────────────────────────────────────────────┤
│ L1. 设备路由  (device routing)                                   │
│     "哪颗 IP / 哪条 subgraph 跑哪里"                              │
│     ─ NVIDIA: CUDA stream / MPS / MIG / DLA 卸载                 │
│     ─ Qualcomm: QNN backend (CPU/GPU/HTP/HTP-MCP)                │
│     ─ Apple: MLComputeUnits                                      │
│     ─ Ascend: engine={AiCore, AiCpu, Vector, Hccl}               │
│     ─ OpenVINO: AUTO / MULTI / HETERO                            │
├─────────────────────────────────────────────────────────────────┤
│ L2. 精度    (numeric precision)                                  │
│     "权重/激活/累加器用什么格式、粒度如何"                           │
│     ─ 权重位宽 × 激活位宽 × 累加位宽                               │
│     ─ 粒度：per-tensor / per-channel / per-group / per-block     │
│     ─ 混合精度的层级分配（layer-wise / subgraph-wise）            │
├─────────────────────────────────────────────────────────────────┤
│ L3. 算子内部 (intra-op)                                          │
│     "单个算子怎么实现"                                            │
│     ─ tile size (M, N, K, L1, UB, double-buffer)                 │
│     ─ kernel 选择 (tactic / algorithm / find mode)               │
│     ─ workspace 大小                                              │
│     ─ 融合开关 (fusion_switch / graph optimization level)         │
├─────────────────────────────────────────────────────────────────┤
│ L4. 调度    (inter-op / concurrency)                             │
│     "多算子/多模型怎么编排执行"                                    │
│     ─ streams / threads / batch / pipeline / CUDA Graph          │
│     ─ priority / multi-PD / QoS                                  │
│     ─ memory pool / KV cache pool / timing cache                 │
│     ─ 连续批处理 / prefill-decode 分离 / 卸载                      │
└─────────────────────────────────────────────────────────────────┘
```

这个分层与 Round 1 `搜索空间体系化总结.md` 列出的 D 空间 10 维相容，但提供了更清晰的归类骨架，也便于跨厂商对齐。

---

## 3. D 空间维度全目录

> 补充 Round 1 的表，加入跨厂商等价项；**加粗**为 Round 2 新认知。

### L1. 设备路由

| 维度 | NVIDIA | Qualcomm | Apple | Ascend | Intel | AMD |
|------|--------|----------|-------|--------|-------|-----|
| Backend/IP 选择 | CUDA / TensorRT EP / **DLA core 选择** | QNN backend | MLComputeUnits | engine= | AUTO/MULTI/HETERO | ROCm EP / MIGraphX |
| GPU 资源划分 | **Stream / MPS / MIG / Green Context** | Multi-PD | — | AI Core cluster | streams | HIP stream |
| **硬件级隔离** | **MIG (A100/H100/B200/Thor)** | 安全岛 (Ride) | — | ASIL 分区 (MDC) | — | — |
| **软件级并发** | **MPS (Jetson Orin 自 JP 6.1 支持)** | Multi-PD | ALL 模式 | engine 并发 | MULTI | HIP context |

### L2. 精度

| 维度 | 具体取值 |
|------|----------|
| 权重位宽 | FP32 / TF32 / BF16 / FP16 / FP8 (E4M3/E5M2) / INT8 / INT4 |
| 激活位宽 | 同上 (通常 ≥ 权重位宽) |
| 累加位宽 | FP32 / FP16 / INT32 (Tensor Core 隐式决定) |
| 量化粒度 | per-tensor / per-channel / per-group(128) / per-block(32) |
| 量化方案 | 对称/非对称 × MinMax/Entropy/Percentile |
| **分层分配策略** | **layer-wise / subgraph-wise / head-wise (attention)** |
| **边界 Q/DQ 策略** | **GPU 少 Q/DQ 保 fusion ↔ DLA 多 Q/DQ 降延迟（相反方向）** |

### L3. 算子内部

| 维度 | NVIDIA | Ascend | Intel | AMD |
|------|--------|--------|-------|-----|
| tile 大小 | 内置（不可调） | **M/N/K/L1/UB/double-buffer (可手写)** | 内置 | CK tile |
| tactic/algo 选择 | tactic source + sparse flag | AOE | heuristic | MIOPEN_FIND_MODE / exhaustive_tune |
| workspace | WORKSPACE MemoryPool (TRT) / cuBLASLt ws | workspace_size | — | — |
| 融合开关 | tactic source | **fusion_switch.cfg** | graph_optimization_level | — |
| **kernel autotune** | **cuDNN benchmark mode** | **AOE mode 1/2/4** | — | **exhaustive_tune** |

### L4. 调度

| 维度 | 说明 | 代表资源 |
|------|------|----------|
| Stream 数 + 优先级 | 主路径独占高优先级 | CUDA Streams, QNN priority, OpenVINO `ov::streams::num` |
| Pipeline 策略 | 单进程多段流水 | TRT + CUDA Graph, Sarathi, DistServe |
| 批处理 | 静态/动态/连续 | vLLM continuous batching, Orca |
| **Prefill/Decode 分离** | **LLM 专有，通过进程分离释放 GPU 资源互斥** | **DistServe, Splitwise** |
| CUDA/HIP Graph | 捕获 + 重放，降 launch 开销 | PyTorch `torch.cuda.graph` |
| **Timing/Engine/Recipe Cache** | **首次编译缓存，下次秒加载** | **TRT timing cache, QNN context binary, OpenVINO blob, MIGraphX tuning DB, Ascend AOE json** |
| **KV cache 池化** | **LLM 推理容量/碎片率** | **vLLM PagedAttention, TRT-LLM** |
| **连续卸载策略** | **权重/KV 在 GPU↔CPU↔Disk 间的放置** | **FlexGen LP 求解** |

---

## 4. B ↔ D 交互矩阵

Round 1 只指出「B ↔ D 紧耦合是第三条路径」，Round 2 把交互的**具体表现**列出来。

### 4.1 量化 × D 的 7 条可落地交互规则

| 量化决策 | 改变的 D 维度 | 方向与幅度 | 来源 |
|----------|--------------|-----------|------|
| **INT4 权重 on Hopper** | L1 后端 → **实际走 "FP8 Tensor Core + 预 dequant"**（非 INT4 TC）| Ampere 逻辑在 Hopper 失效；TRT Model Optimizer 的 W4A16 AWQ 被标为 Ampere only | `quantization_x_hardware.md` §1 |
| **per-channel activation** | L3 tactic → **INT8 GEMM tile 结构破坏，几乎不可用** | SmoothQuant 的 motivation 就是"迁回权重" | `quantization_x_hardware.md` §1 |
| **DLA fallback** | L1 路由 + L2 粒度约束 → **DLA 要求 INT8 per-tensor + HWC4**；GPU 相反 | 同 ONNX 不能同时最优 GPU+DLA，需 subgraph 分裂 | `quantization_x_hardware.md` §4, `d_space_nvidia.md` §2.4 |
| **混合精度 fallback** | L3 tactic → **开 `--int8` 不保证每层 INT8**，小 shape 会挑 FP16 | 判断量化是否真生效只能看 SASS 指令 | `quantization_x_hardware.md` §2 |
| **KV cache 量化** | L4 并发 → **不是降单 query 延迟，是拉高 context 长度/并发数** | vLLM TQ4 把 Llama-3.1-8B 32K KV 从 FP8 2GB 压到 0.51GB，blocks ×1.94 | `quantization_x_hardware.md` §4 |
| **FP8 KV** | L1 backend → **FA-3 要求 Q 也 FP8**，强制 attention backend | 精度和 backend 耦合 | `quantization_x_hardware.md` §4 |
| **Q/DQ 插入密度** | L3 融合 → **GPU：少 Q/DQ 保 fusion；DLA：多 Q/DQ 降延迟** | 方向相反 | `quantization_x_hardware.md` §3 |

### 4.2 剪枝 × D 的 7 条可落地交互规则

| 剪枝决策 | 改变的 D 维度 | 方向与幅度 | 来源 |
|----------|--------------|-----------|------|
| **剪枝率跨 kernel 切换点** | L3 tactic → **<30%:dense/channel; 30-50%:2:4+cuSPARSELt; 50-70%:Block-SpMM; >70%:Sputnik; >90%:DeepSparse** | cuSPARSE Block-SpMM block=32 时 A100 需密度 <50%；V100 需 <40% | `pruning_x_hardware.md` §2 |
| **非结构化稀疏** | L1/L3 → **GPU 上几乎无加速**，只有 >90% + Sputnik/SparseRT（绕过 Tensor Core）才有效 | warp divergence + uncoalesced + 无 TC + metadata 反噬 | `pruning_x_hardware.md` §1.3 |
| **通道对齐** | L3 tactic → **剪枝后通道数必须对齐** | FP16 dense 对齐 8，FP16 sparse 对齐 16，INT8 对齐 32 | `pruning_x_hardware.md` §1.4 |
| **跨层剪枝梯度** | L3 融合 → **层间剪枝率差 >30% 会打断 fusion** | 显式 reformat kernel 抵消增益 | `pruning_x_hardware.md` §7.3 |
| **DLA 稀疏** | L1 路由 → **DLA 支持 2:4 但仅 math-bound 层收益 1.36×** | memory-bound 层无收益；低功耗模式放大收益 | `pruning_x_hardware.md` §5 |
| **动态剪枝流水友好度** | L4 调度 → **token(DynamicViT) >> MoE block-sparse(MegaBlocks) >> head+多流 >> 层级/空间动态(warp div)** | 形状 reduce vs 条件分支的本质区别 | `pruning_x_hardware.md` §3 |
| **cuSPARSELt 的 MatmulSearch** | L3 tactic → **即使同样 2:4 剪枝，matmul algorithm 差 30%** | 应作为 D 搜索维度，不是黑盒 | `pruning_x_hardware.md` §2.2 |

---

## 5. 相比 Round 1《搜索空间体系化总结》的新增认知

| 方面 | Round 1 认知 | Round 2 补充 |
|------|-------------|--------------|
| D 空间规模 | 列出 10 个子维度 | 按 L1-L4 分层归类，跨 10 家厂商对齐，维度数扩到 30+ |
| 硬件 fallback | 未提 | **DLA fallback 成本、ORT-TRT subgraph 碎片化、QNN 不支持 op 切回 CPU 的场景** |
| 精度 × kernel | B2 分为标准 vs 新格式 | **同一标准格式在不同架构的 tactic 路径都不同**（Hopper 的 INT4 故事） |
| 剪枝 × kernel | 提过 N:M / block / 非结构化 | **每种稀疏模式的 kernel 切换交叉点与对齐约束具体量化** |
| KV cache | 未讨论 | **LLM 部署独有 D 维度，量化→并发数而非延迟** |
| 多模型调度 | 提 Orca/DistServe/Splitwise/FlexGen | **只在 LLM 与多模型 serving 场景有搜索实现；感知/V2X 几乎空白** |
| Jetson Orin | 提"DLA offload" | **MPS 自 JP 6.1 支持、MIG 不支持的根因、Thor 引入 MIG** |
| 缓存可复用 | 未强调 | **TRT timing cache / QNN context binary / OpenVINO blob / Ascend AOE json 是所有栈的共性必调项** |
| 调度抽象 | 具体按栈列 | **OpenVINO hints 是最清晰的声明式抽象样板，值得复用** |
| 跨厂商联合抽象 | 未讨论 | **"L1-L4 四层旋钮" 是本轮提炼的统一坐标系** |

---

## 6. 对 UniV2X 的设计启示

### 6.1 D 空间搜索维度的起步集（建议）

按 L1-L4 分层，优先级由高到低：

```
L2 精度 (收益最大)
  ├─ 权重位宽 ∈ {INT4, INT8, FP16}
  ├─ 激活位宽 ∈ {INT8, FP16}
  ├─ 量化粒度 ∈ {per-tensor (DLA 子图), per-channel (GPU 子图)}
  └─ 混合精度分配 (分层)

L4 调度 (与我们方案的时序缓存/多流已对齐)
  ├─ CUDA Stream 数 + priority
  ├─ CUDA Graph on/off
  ├─ Timing cache 持久化 (必开)
  └─ 时序特征缓存 (frames × precision)

L3 算子内部 (进阶)
  ├─ TRT tactic source (Hopper 时关 CUDNN)
  ├─ workspace pool 大小
  ├─ cuDNN benchmark mode
  └─ cuSPARSELt MatmulSearch (若剪枝 ≥ 50%)

L1 设备路由 (硬约束)
  ├─ DLA offload 子图选择 (只选纯 Conv+Pool 子图)
  ├─ MPS % 分配 (多进程多模型场景)
  └─ ego / infra 双 Orin 的 AI Core 跨设备分配
```

### 6.2 关键硬过滤清单（搜索前置）

搜索无效组合会浪费算力，建议在搜索器前加硬约束：

- `channels % 32 == 0` （INT8 TC 对齐）
- `K_dim >= 64` （2:4 有效门槛）
- Hopper 架构：禁止 `W4 + INT4 TC` 组合 → 改走 `W4A8 + FP8 TC`
- DLA 子图：强制 `per-tensor + INT8 + HWC4`
- 跨层剪枝率梯度 `< 30%`
- MPS `active_thread_%` 总和 `≤ 100%`

### 6.3 可复用的机制（从调研中挑出）

| 机制来源 | 可复用为 UniV2X 的 |
|----------|------------------|
| OpenVINO `ov::hint::*` | **声明式 D 空间 API**（LATENCY/THROUGHPUT 作为高层意图）|
| APQ 的精度预测器迁移 | 减少 D 空间每候选的评估开销 |
| DANCE 的可微评估器 | 梯度驱动的 D 空间搜索 |
| Vidur 的仿真+搜索分离 | 先跑仿真筛候选，再实测前 N 个 |
| HAWQ-V3 的 ILP | 混合精度搜索的启动点 |
| FlexGen 的 LP placement | 多设备（ego/infra/路侧）任务分配 |
| Mirage µGraph (OSDI 25) | D3 算子选择的扩展抽象 |

### 6.4 产物制品化

所有栈的编译缓存都需纳入版本管理：
- `TRT timing cache` + `engine cache`
- `QNN context binary` / `OpenVINO export_model blob` / `AOE json` / `MIOpen tuning DB`
- `cuDNN benchmark cache` (进程内即可)
- 校准集保持 float32 + per-layer scale 中间表示，不直接存某家格式

---

## 7. 下一步延伸方向（本轮未覆盖）

如果还要继续深挖，以下方向仍是空白：

1. **能耗 × D 空间**：本轮几乎没覆盖 energy dimension。LLM 推理的 token/J 指标、自动驾驶的 mW/frame 指标和 D 空间维度的量化关系几乎无公开数据。
2. **安全/确定性 × D 空间**：ASIL-B/D 场景下 CUDA Graph 的确定性保证、Ascend ASIL 分区的量化边界、DLA 与 ISO 26262。
3. **Compiler-in-the-loop 的搜索**：MLGO / Reasoning Compiler 级别的 LLM-guided compilation，这类工作正在兴起（2024-2026）但国内 NPU 侧几乎没人做。
4. **多 Agent 联邦的 D 空间**：V2X 场景下 ego/infra/cloud 三端各自有 D 空间，**联邦优化**是一个全新的外层搜索，本轮没找到直接对标工作。
5. **硬件监控接入**：nvidia-smi / dcgm / tegrastats / SNPE profiling 提供的实时 metrics 如何反馈到 D 空间动态调整 —— 目前我们的方案是"离线搜 + 固定部署"，未覆盖在线自适应。

---

## 附：本轮调研统计

- 总产出：6 份专题 + 1 份 README = 约 13 万字 / 1800 行
- 覆盖论文/产品文档：约 80 条直接引用
- 主要来源：MLSys/OSDI/SOSP/ASPLOS/ISCA/HPCA/MICRO 2024-2026 + NVIDIA/高通/Intel/华为/AMD 官方开发者文档
