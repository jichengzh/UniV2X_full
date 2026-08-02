# 33_7_14 冷启动文档：Lane B 已有软硬件协同结果的 CPU 固定回放 v1

> 日期：2026-07-23  
> 总计划：[32 号三线并行计划](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/32_7_14_交接文档_FCooper_Table1_TVM_CPU与Orin边缘部署计划_v1.md)  
> 并行窗口：[Lane C Orin 冷启动文档](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/34_7_14_冷启动文档_LaneC_Orin快速真测与映射表_v1.md)  
> 本窗口只负责 Lane B，不修改 F-Cooper/Table 1，不运行新的候选搜索。

## 1. 单一目标

本轮研究对象是**软硬件协同优化方案产生的已有配置**，不是 CPU 推理后端本身。为节省时间，不在 CPU profile 上重新运行外层搜索，而是把 31、32 号文档已经冻结的 `original`、非联合方法配置和 `joint/GEAR winner` 原样迁移到 CPU，在同一执行条件下比较它们的 latency、energy 和精度合同。

本轮直接回答：

> H800 profile 上已经产生的软硬件协同优化结果，在 TensorRT 不可用的 CPU 平台上进行固定回放时，是否仍相对原始配置和已有非联合优化方案保持优势？

CPU runtime 只是统一测量工具，不是搜索变量，也不研究“哪个后端更强”。不得运行 cost model、acquisition、候选池扩展或人工补点。

## 2. 固定硬件与环境口径

原计划使用 4090 服务器的 CPU：

- Intel Xeon Gold 6530，双 socket、每 socket 32 个物理核；
- AVX-512、AVX-VNNI、AVX512-FP16、AMX-INT8/BF16；
- 两个 NUMA node；
- 当前 Python 有 ONNX Runtime 1.27.0 和 PyTorch 2.12.1，但没有 TVM。

但第 10.2 节已确认该主机所有物理核都被外部任务持续占满，现阶段不能产生正式 timing。正式回放因此转移到第 10.3 节记录的 H800 主机 **CPU**：双路 Xeon Platinum 8480+，固定 NUMA 1 的 8 个空闲物理核和 8 线程。启动前必须重新验证这些物理核及 sibling 均空闲。

任何 GPU 都不参与计算。所有 runner 必须满足：

- TVM target 为 `llvm`；
- ONNX Runtime provider 仅为 `CPUExecutionProvider`；
- PyTorch tensor/model 保持 CPU；
- 日志记录 `CUDA_VISIBLE_DEVICES`、provider list 和 TVM target，证明没有 GPU fallback。

优先复用第 10.4 节已验证的 H800 CPU-TVM 环境。安装和编译环境准备耗时不计入 inference latency。

## 3. 固定回放点

### 3.1 第一停止点：Pyramid 已有方法结果

所有配置取自 31 号 `H800 + TVM、DeltaAP_max=0.10` 正式结果，必须从其 manifest/SHA 读取，不根据 CPU 结果重新挑选：

| 来源方法 | 冻结配置 | CPU 回放角色 |
|---|---|---|
| original/default | `(64,128,256,fp32)` | 未做软硬件协同优化的共同起点 |
| compression-only | `(24,48,96,fp16)` | 不读取硬件反馈的压缩方案 |
| compress -> tune | `(32,48,128,fp16)` | 串行软件后硬件优化方案 |
| joint SHCoSearch / GEAR | `(16,32,64,int8)` | 本文软硬件联合方案 |

`schedule-only` 的网络结构与 original 相同，不形成新的 genome；它在 CPU 表中由 original 配置经过统一 CPU tuning 后的结果表达。`tune -> compress` 在已有正式实验中为 100% policy-transfer failure，没有合法 winner，不得人工补造 CPU 配置。

### 3.2 第二停止点：CoDriving

Pyramid 闭环后，按相同规则从 CoDriving 已有正式产物中读取 original、compression-only、serial 和 joint winner，禁止手工复用 Pyramid 宽度。F-Cooper 只有在 32 号正式重跑产生可信 winner 后才能加入。

### 3.3 dtype 回放合同

对每个模型的全部方法配置统一选择一种回放口径，禁止各臂混用：

1. `exact_genome_replay`：CPU 存在对应 FP16/INT8 真实 lowering，所有配置按原始 `q_mode` 回放；
2. `structure_replay_dtype_normalized_fp32`：若任一关键低精度路径不成立，则所有配置统一使用各自冻结结构的 FP32 图，只研究结构选择的跨平台迁移效果。

禁止实际运行 FP32 却标记为 INT8，也禁止只给 joint 保留 INT8、却把其他臂改成 FP32 后直接比较。

## 4. 固定 CPU 执行合同，不研究后端优劣

本轮冻结一条主执行合同，避免把 runtime 变成额外实验轴：

| 层级 | 固定执行路径 | 用法 |
|---|---|---|
| 主结果 | TVM LLVM MetaSchedule tuned | 四个冻结配置使用完全相同的 target、线程和 tuning 预算；用于比较搜索方法效果 |
| 外部参照 | ORT CPU EP | 对四个冻结配置使用相同 graph optimization；低成本检查配置收益能否在成熟 CPU runtime 上复现 |
| 内部诊断 | TVM LLVM default | 只用于解释 schedule 增量；不作为方法对手，不阻塞主停止点 |

主要比较始终是主结果列内部的 `joint vs original/compression-only/serial`。ORT 只检查这种排序是否为 TVM 特有；`TVM tuned/default` 只能作为附加内部诊断，不能作为软硬件协同方法的主结论。

正式 tuning 不再使用“最多 30 分钟或 512 trials、先到者停止”的不足预算。四个冻结配置统一执行第 12.5.1 节的逐 task 覆盖、热点加权和收敛门禁；预算规则相同，但必须按各配置实际 task 数保证覆盖。编译/tuning 时间单独报告，不混入 inference latency。不得增加 DLight、DNNL、OpenVINO、CUTLASS 或 cuDNN 等额外后端研究臂。

## 5. 数值与性能合同

### 5.1 数值合同

- 使用固定 seed 和同一输入 tensor；
- ORT 作为部署参考，同时保存 PyTorch 输出；
- FP32 至少报告 max absolute error、mean absolute error 和 cosine similarity；
- INT8/FP16 使用适合量化的逐输出容差，并保存失败详情；
- 数值失败时不测性能，不允许 silent fallback。

### 5.2 CPU 隔离

- 先扫描空闲物理核，不占用 sibling hyperthreads；
- 固定单一 NUMA node并绑定内存；
- 固定 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、TVM runtime threads；
- 关闭动态线程调整；
- benchmark 前记录 CPU governor、频率、温度和系统负载；
- 避开同机正式 GPU latency/energy 复测窗口。

### 5.3 计时

- batch=1；
- 每个 runtime 至少 warmup 100 次；
- 每次测量至少 500 次，3 个独立进程重复；
- 报告 median、p90、p99、mean、std 和 CV；
- 若可读取 RAPL，则以相同采样合同报告 package energy/frame；否则统一标记 unavailable，不估造；
- 原始逐次 timing 保存为 CSV/JSON，不只保留汇总值。

## 6. 产物目录和表格

统一根目录：

```text
${V2X_ROOT}/results/lane_b_cpu_tvm_fixed_replay_20260723/
```

最低产物：

```text
capability/cpu_profile.json
manifests/fixed_replay_manifest.json
artifacts/{model}/{runtime}/
measurements/{model}_runtime_repeats.csv
audits/numerical_contract_audit.json
audits/no_gpu_fallback_audit.json
lane_b_cpu_tvm_pilot.csv
lane_b_cpu_tvm_summary.md
```

Pilot 表：

| Model | Source method | Frozen config | Replay status | Source AP70 | TVM tuned ms | Speedup vs original | ORT reference ms | Numerical |
|---|---|---|---|---:|---:|---:|---:|---|

不得把 CPU 表并入 H800 TensorRT Table 1。

## 7. 停止条件

第一停止点：

- Pyramid 的 original、compression-only、compress->tune、joint 四个已有配置完成统一回放；
- 在冻结的 TVM CPU 执行合同内完成 joint 相对 original 和非联合方法的 latency/energy 比较；
- ORT 完成同四配置的低成本外部参照；TVM default 不阻塞停止点；
- 明确标注 exact replay 或全臂统一 dtype-normalized replay；
- 生成 pilot CSV、summary 和审计 JSON；
- 给出“已有 joint winner 在 CPU 上保持优势”或“已有 H800 winner 不具备 CPU zero-shot 优势”的判断。

是否继续 CoDriving 由**方法配置排序和收益是否稳定**决定，不由 TVM tuned/default 的差值单独决定。无论结果正负，都不得扩大外层搜索预算或更换 winner。

## 8. 禁止事项

1. 不重新搜索 genome；
2. 不使用 CUDA EP 或 TVM CUDA；
3. 不关闭 ORT graph optimization 制造弱基线；
4. 不混用不同线程数或 NUMA node 选优；
5. 不把 tuning 时间算入 latency，也不能隐藏 tuning 成本；
6. 不把 dtype-normalized replay 写成原始 H800 genome 的 exact transfer；
7. 不因为任何 CPU runtime 上结果不理想而手工替换已有方法配置；
8. 不把运行时之间的快慢写成软硬件协同方案的效果；
9. 不把 H800 winner 的 CPU 固定回放写成“CPU-aware joint search”；本轮只验证 zero-shot transfer。

## 9. Lane B 启动 `/goal`（历史方案，已被第 12.6 节替代）

```text
/goal 执行 Lane B“已有软硬件协同结果的 CPU 固定回放”。严格依据 ${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/33_7_14_冷启动文档_LaneB_4090服务器CPU_TVM固定点回放_v1.md，不运行 cost model、acquisition、候选生成或任何新的外层搜索，不修改 F-Cooper 或论文 Table 1。正式 timing 转移到 H800 主机 CPU，固定 NUMA 1 的8个空闲物理核和8线程，启动前重新验证这些物理核及 sibling 均空闲；任何 GPU 都不得参与。第一停止点只回放 Pyramid 31号正式产物中冻结的四个配置：original/default=(64,128,256,fp32)、compression-only=(24,48,96,fp16)、compress->tune=(32,48,128,fp16)、joint SHCoSearch/GEAR=(16,32,64,int8)；必须从正式 manifest/checkpoint/ONNX/SHA 定位产物，不得依据 CPU 结果换点。先做 CPU capability 与数值 gate：只有全部关键低精度路径均真实成立时才执行 exact_genome_replay，否则四个配置全部保持各自结构并统一FP32，标记 structure_replay_dtype_normalized_fp32。主结果统一使用 TVM LLVM MetaSchedule tuned，对四个配置采用完全相同的 target、线程、NUMA、输入、batch、dtype 和最多30分钟或512 trials的tuning合同；主要统计是 joint 相对 original、compression-only、compress->tune 的 latency/energy/AP 改善。ORT CPU EP 对同四配置做低成本外部参照；TVM default 只作可选内部诊断，不阻塞停止点。不研究后端谁更强，不增加 DLight、DNNL、OpenVINO、CUTLASS 或 cuDNN 实验。设置 CUDA_VISIBLE_DEVICES=''，禁止任何GPU fallback；每项warmup至少100次、测量至少500次、3个独立进程，保存CPU profile、固定回放manifest、逐次timing、数值与no-GPU审计。停止条件是得到Pyramid四配置完整小表并判断已有H800 joint winner是否在CPU上保持zero-shot优势；不得把结果写成CPU-aware search。Pyramid闭环后再按同一规则读取并回放CoDriving自身已有配置，禁止套用Pyramid宽度。
```

## 10. 2026-07-23 资源审计与首轮 pilot 边界

### 10.1 结论边界

已完成的首轮 pilot 只测了 Pyramid joint winner `(16,32,64)`，尚未同步测量 original、compression-only 和 serial 配置，因此**不能用于判断软硬件协同方法的效果**。它只提供以下工程信息：

1. H800 INT8 winner 在 Xeon CPU 上没有形成真实 INT8 Conv lowering。51 个 Conv 全部在 FP32 执行，因此只能标记为 `structure_replay_dtype_normalized_fp32`，不是原始 H800 genome 的 exact transfer。
2. 512 global trials 没有覆盖完整网络。最终 scheduler 中只有 8 个 Conv 类任务各获得 64 trials，另有 2 个 reshape 各获得 1 trial；剩余 11 个任务为 0 trials，其中包含 weight=8、7、5 等高权重热点。证据：
   `${V2X_ROOT}/results/lane_b_cpu_tvm_fixed_replay_20260723/artifacts/pyramid/tvm_tuned/metaschedule_db/logs/tvm.s_tir.meta_schedule.logging.task_scheduler.log`。
3. 4090 服务器的性能 timing 与严重外部 CPU 负载重叠，已降级为诊断证据，不能作为最终正式性能结论。诊断结果中 TVM tuned 相对 TVM default 约改善 38.3%，但仍约为 ORT latency 的 11.6 倍；这个倍数必须在空闲 CPU 上重测后才可正式引用。

下一次运行必须补齐四个冻结方法配置，在同一 CPU 执行层内比较；不得继续单独攻击 winner 的 schedule，也不得根据该单点决定方法成败。

### 10.2 4090 服务器 CPU 占用归因

主机 `<PRIVATE_HOST>` 为双路 Xeon Gold 6530，共 64 个物理核、128 个逻辑 CPU。2026-07-23 审计时：

| 来源 | 约占用 |
|---|---:|
| 两个 `neuroncap_server/server.py` | 96.7 个逻辑核 |
| 两组共 20 个 Materials Studio `MatServer.exe` | 19.8 个逻辑核 |
| Pylance、编辑器和其他进程 | 约 2–3 个逻辑核 |

两个 `neuroncap_server` 各有 198 个线程，affinity 均覆盖 `0-127`，因此 Linux 调度器会把压力铺到两个 NUMA 节点。连续 15 秒、5 轮、同时检查两个 sibling 的审计中，两个 NUMA 节点均没有一个物理核能稳定低于 20%；最低物理核仍接近 98%。在这些进程结束或释放 CPU 前，不得在 4090 服务器生成正式 CPU benchmark。

不得擅自停止、修改 affinity 或降低其他用户进程优先级。

### 10.3 H800 服务器 CPU 与 GPU 资源

H800 主机：

```text
host = <PRIVATE_HOST>
cpu = 2 × Intel Xeon Platinum 8480+
physical_cores = 112
logical_cpus = 224
numa_nodes = 2
```

连续 10 秒 sibling-aware 采样结果：

| NUMA | 持续低于 10% 的物理核 | 持续低于 20% 的物理核 |
|---|---:|---:|
| node 0 | 33 | 33 |
| node 1 | 37 | 37 |

H800 CPU 足够开展固定核研究。建议优先使用 NUMA 1 的 8 个物理核：

```text
benchmark_cpus = 59,62,63,72,73,75,100,101
sibling_cpus   = 171,174,175,184,185,187,212,213
numa_node      = 1
threads        = 8
```

正式启动前仍须重新扫描；只有 benchmark CPU 及其 sibling 均保持低负载时才允许计时。

H800 GPU 审计时，GPU 3、4 为空闲卡；GPU 5、6 正在运行 F-Cooper recovery，GPU 0 有其他用户任务。本研究不把 H800 GPU 作为下一阶段目标；这些信息仅保留为资源审计记录，不得据此自动启动 NVIDIA GPU 实验，也不得干扰 GPU 5/6 上的 F-Cooper。

### 10.4 H800 现有 CPU 执行环境

H800 历史专用环境：

```text
python = ${V2X_DATA_ROOT}/tvm310/bin/python
tvm = 0.20.dev1070+gb628d91fa
```

该环境需要把其自带 CUDA runtime 放在 `LD_LIBRARY_PATH` 首位，否则会出现 `cudaGraphAddDependencies_v2` 符号缺失。正确 runtime 路径为：

```text
${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
```

已确认 LLVM runtime、MetaSchedule 入口和 ONNX Runtime 1.23.2 可用，足以按第 3–4 节执行固定配置回放。本轮不以 DLight、DNNL、cuDNN 或 CUTLASS 是否存在作为实验门禁，也不为接入这些后端建立新环境。

必须保持历史 `tvm310` 只读；如因四配置公平回放需要新增脚本、数据库或结果，统一写入 `/exdata` 下独立工作目录。H800 根分区已使用 93%，不得写入根分区。

## 11. 科学解释与论文结论边界

### 11.1 本轮能够回答的问题

本轮是**已有搜索结果的 CPU zero-shot 固定回放**。它可以比较：

- joint/GEAR 已有 winner 相对 original 的迁移收益；
- joint/GEAR 相对 compression-only 和 compress->tune 已有配置的迁移收益；
- 上述配置排序在 TVM tuned 主结果与 ORT 外部参照中是否一致；TVM default 只提供可选内部诊断。

这里的“现有方法”指 31 号六臂中的非联合优化方法，而不是 ORT、TVM、DNNL 或其他推理后端。运行时只负责公平执行同一组冻结配置。

### 11.2 本轮不能回答的问题

因为没有使用 CPU 真实反馈重新运行外层搜索，本轮不能声称：

- joint 方法已经在 CPU 上重新学到了 CPU-specific winner；
- H800 winner 就是 CPU 最优配置；
- 本轮已经完整证明 profile-conditioned hardware-aware search 在 CPU 上优于所有方法。

若 H800 joint winner 在 CPU 上仍优于原始和非联合配置，可以表述为“已有协同结果具有 zero-shot 跨平台迁移收益”。若排序发生变化，则说明硬件 profile 会改变最优配置，支持未来进行 CPU 小预算 recalibration，但本轮不自动启动新搜索。

### 11.3 执行入口

后续以第 12.6 节的新实验 `/goal` 为唯一执行入口；第 9 节旧 `/goal` 降级为历史方案，不再执行。此前以 DLight、DNNL、OpenVINO 或高性能后端接入为中心的计划全部作废，不得继续执行。

## 12. TVM 与 ORT 的角色澄清、五臂表格和新实验计划

### 12.1 为什么不能因为“跨硬件”而强制使用 TVM

TVM 的价值是提供可编程 IR、目标相关 schedule 搜索和可扩展 codegen，适合承载本项目的内层 `S` 优化；但“支持多种硬件”不等于“在每种硬件上都应作为最快部署后端”。部署选择和方法研究必须分开：

- **方法研究**：在冻结的 TVM CPU 合同下复现五臂，回答联合软硬件协同是否优于非联合方法；
- **实际部署**：在相同模型配置和资源合同下比较 ORT 与 TVM。若 ORT 在当前 Intel CPU 上稳定更快，则该 CPU 的生产执行器应选择 ORT；
- **跨平台价值**：TVM 的正面价值必须在缺少成熟 ORT Execution Provider、需要自定义 codegen/schedule，或 TVM 实测具有竞争力的非 NVIDIA/国产硬件上证明，不能仅由 Intel CPU 上的理论可移植性推出。

后端不进入外层 genome。每个平台先通过 capability 和小规模真实性能 gate 冻结执行器，再让所有方法臂在相同执行合同下比较。不得为了统一叙事强制选择显著更慢的执行器。

### 12.2 当前 `11.58x` 差距的已知证据与待验证原因

4090 服务器受污染 pilot 中，Pyramid `(16,32,64)` 得到：

| 路径 | representative median |
|---|---:|
| ORT CPU EP | `142.302 ms` |
| TVM LLVM default | `2670.715 ms` |
| TVM LLVM MetaSchedule tuned | `1648.080 ms` |

`TVM tuned / ORT = 11.5815x`，但该绝对倍数必须在空闲 H800 CPU 上重新测量。当前已确认：

1. 源 INT8 winner 的 51/51 个 Conv 在 CPU TVM 中均回退到 FP32，当前只完成结构迁移；
2. 21 个 MetaSchedule task 中只有 8 个 Conv 和 2 个 reshape 获得 trials，剩余 11 个 task 为 0 trials，其中包含高权重热点；
3. ORT 使用完整 graph optimization；当前 TVM 路径为 Relax `LegalizeOps/FuseOps/FuseTIR + LLVM + 部分 MetaSchedule database`；
4. 当前尚未定量区分未调优 kernel、layout/materialization、内存规划和 Relax VM/end-to-end overhead 的贡献。

本轮只做最小根因诊断，不新增 DLight、DNNL、OpenVINO、CUTLASS、cuDNN 或其他后端接入研究：

- 对 original 与 GEAR 两个结构分别保存 ORT profile；
- 对同两点保存 TVM 各 PrimFunc/task latency、database 命中和 0-trial task；
- 比较 end-to-end latency 与已计时 kernel/task 之和，分离 kernel 与 graph/runtime 税；
- 统计 layout transform、显式 pad/cast/reshape/materialization 数量；
- 所有 profiling 运行与正式 timing 分离，profiling 数值不得填入正式表。

### 12.3 表 A：CPU 五臂软硬件协同方法表

表 A 沿用 31、32 号文档当前论文五臂语义，不重新执行外层搜索：

| 臂 | 冻结模型配置 | CPU 执行合同 | 作用 |
|---|---|---|---|
| Original/default | `(64,128,256,fp32)` | PyTorch eager CPU，正常启用 CPU 优化 | 真实原始部署起点 |
| Compression-only | `(24,48,96,fp16)` | TVM LLVM default，不导入 tuning database | hardware-blind 压缩结果 |
| Schedule-only | `(64,128,256,fp32)` | TVM LLVM MetaSchedule tuned | 只引入硬件调度优化 |
| Compress -> Tune | `(32,48,128,fp16)` | TVM LLVM MetaSchedule tuned | 串行软件后硬件优化 |
| GEAR / joint SHCoSearch | `(16,32,64,int8)` | TVM LLVM MetaSchedule tuned | 本文联合软硬件协同方案 |

配置来源固定为 31 号 `H800 + TVM、DeltaAP_max=0.10` 正式 manifest。`tune -> compress` 不属于当前论文五臂，且历史为 100% policy-transfer failure，本表不补造该臂。

若 CPU 不存在统一可用的真实 FP16/INT8 lowering，则五臂全部保留各自结构并统一 FP32，表名和每行必须标记 `structure_replay_dtype_normalized_fp32`。此时表 A 只能证明结构选择和调度组合的迁移效果，不能声称完整 `P x Q x S` 精度迁移。

正式输出：

| Method | Frozen config | Replay status | Source AP70 | CPU latency p50/p99 | Energy/frame | Speedup vs Original | Tuning valid trials / task coverage / time / convergence | CV | Numerical |
|---|---|---|---:|---:|---:|---:|---:|---:|---|

### 12.4 表 B：ORT–TVM 执行器与配置排序诊断表

表 B 不把执行器当作搜索方法，只用于回答“当前 CPU 最终为什么使用 TVM 或 ORT”。四个唯一模型配置均使用相同 ONNX/dtype/线程/NUMA/batch 合同：

| Source method/config | ORT CPU EP p50 | TVM default p50 | TVM tuned p50 | TVM tuned / ORT | Speedup vs Original under ORT | Speedup vs Original under TVM | Best executor |
|---|---:|---:|---:|---:|---:|---:|---|
| Original `(64,128,256)` |  |  |  |  | `1.0x` | `1.0x` |  |
| Compression-only `(24,48,96)` |  |  |  |  |  |  |  |
| Compress -> Tune `(32,48,128)` |  |  |  |  |  |  |  |
| GEAR `(16,32,64)` |  |  |  |  |  |  |  |

表 B 的结论规则：

1. 同一执行器列内的配置排序用于判断已有协同配置是否具有 zero-shot CPU 迁移收益；
2. 同一配置行内的 ORT/TVM 比值用于决定当前 Intel CPU 的实际部署执行器；
3. 如果 GEAR 只在 TVM 内部优于其他配置，但 TVM 绝对性能仍显著落后 ORT，不得写成“CPU 部署性能优于现有方案”；
4. 如果 ORT 明显更快，则 Intel CPU 部署选 ORT，TVM 结果仅作为可移植编译路径和失败机制证据；
5. 只有在真实非 NVIDIA/国产目标上证明 ORT 无高性能可用路径、或 TVM 达到更优/可接受性能，才能把使用 TVM 写成该平台的部署优势。

### 12.5 统一测量与停止条件

- 正式服务器：H800 主机 **CPU**，GPU 完全禁用；
- NUMA 1，启动前重新扫描并选择 8 个空闲物理核，排除 sibling；
- `batch=1`，ORT/TVM/PyTorch 均为 8 线程，固定 governor、线程亲和和内存绑定；
- 每项 `warmup >= 100`、`measurements >= 500`、3 个独立进程；
- TVM tuned 对四个唯一配置使用第 12.5.1 节相同的覆盖和收敛规则、相同 seed 和 target；禁止退回 `512 trials / 30 min` pilot 合同；
- 先通过逐输出数值合同再计时；禁止 silent fallback；
- 若可读取 RAPL，则报告同口径 package energy/frame，否则统一标记 unavailable；
- 保存逐次 timing、ORT profile、TVM task/PrimFunc profile、database 命中、0-trial task、编译与 tuning 成本；
- 不修改 F-Cooper、H800 TensorRT Table 1 或历史 `tvm310`，新脚本和结果写入 `/exdata` 独立 v2 目录。

#### 12.5.1 MetaSchedule tuned 预算与充分性门禁

上一轮 `512 global trials` 使 21 个 task 中 11 个得到 0 trials，因此本轮不得把“调优进程正常退出”直接记为 `tuned`。每个唯一配置必须使用独立的新数据库，按以下顺序执行：

1. **重新抽取并登记 task**：保存 task 名称、结构哈希、调用权重、算子类型和 TVM 是否支持调优；不得继承上一轮不完整数据库。确实无法调优的 task 标记为 `not_tunable` 并记录默认实现，不能混记为 0-trial。
2. **覆盖阶段**：采用逐 task 或强制 round-robin 调度，保证每个 `tunable` task 至少获得 `64` 个有效 measured trials；Conv 以及 default profile 中占端到端时间至少 `1%` 的热点 task 至少获得 `256` 个有效 measured trials。compile error、run error 和重复无效候选不计入有效 trials。
3. **首轮总预算目标**：每个唯一配置以 `4096` 个有效 measured trials 为目标；覆盖阶段以后的预算按 default profile 的 task 时间占比分配。每配置首轮搜索墙钟时间硬上限为 **6 小时**，到时必须停止并保存可恢复数据库，未经新的人工决策不得自动续跑。
4. **自适应扩展**：若在 6 小时内达到 `4096` trials，则重建并测量端到端程序。如果最近 `512` 个有效 trials 仍使加权 task 最优延迟或端到端 p50 改善至少 `1%`，则每次增加 `2048` trials，最多扩展到 `8192` 个有效 trials，但任何扩展都不得突破 6 小时硬上限。
5. **6 小时终态语义**：若时间耗尽时尚未满足逐 task 覆盖门槛，标记 `tuning_incomplete`，不得把结果填为正式 TVM tuned 数值；若覆盖门槛全部满足但尚未达到收敛门槛，标记 `tuned_6h_budget_limited`，允许作为“统一 6 小时预算”结果填表，但不得称为 TVM 性能上限或充分收敛结果。
6. **最终 tuned gate**：所有 `tunable` task 均有有效记录且最终 build 的 database hit rate 为 `100%`；不存在 0-trial 的可调 task；热点最低预算全部满足；加权 task 最优延迟和端到端 p50 在最后检查窗口内的改善均小于 `1%`（达到 `4096` 时检查最近 `512` 个有效 trials，发生扩展后检查最近一个 `2048`-trial 扩展区间）。四项同时满足后才能标记 `tuned_converged`；否则按第 5 条标记预算受限或调优失败。
7. **公平性与报告**：四个配置使用相同规则，而不是强行使用相同的绝对耗时。逐配置报告 extracted/tunable/not_tunable task 数、有效/失败 trials、每 task trials、database hit rate、预算扩展轨迹、总调优时间和最终 convergence 状态。

停止条件：

1. 表 A 五臂全部获得可信成功或明确 failure 终态；
2. 表 B 四配置三执行路径闭合，并完成三次独立重复；TVM tuned 单元格必须明确标记 `tuned_converged`、`tuned_6h_budget_limited` 或 `tuning_incomplete`，其中只有前两种可以填写性能值，且预算受限结果不得用于声称 TVM 性能上限；
3. 明确回答 GEAR 相对 Original、Compression-only、Compress -> Tune 的 CPU 迁移效果；
4. 明确回答 `11.58x` 差距主要来自 FP32 回退、0-trial kernel、kernel 质量还是 graph/runtime 税；
5. 给出当前 Intel CPU 的部署建议：`ORT` 或 `TVM`，不得以跨平台叙事覆盖真实性能结果；
6. 不自动启动 CPU 外层新搜索；若配置排序变化，只记录为未来 CPU recalibration 的动机。

### 12.6 新实验 `/goal`

```text
/goal 执行 33号文档第12章“TVM与ORT角色澄清、CPU五臂方法表和执行器诊断表”实验。严格依据 ${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/33_7_14_冷启动文档_LaneB_4090服务器CPU_TVM固定点回放_v1.md 第12章，不运行cost model、acquisition、候选生成或任何CPU外层新搜索，不修改F-Cooper、H800 TensorRT Table 1和历史${V2X_DATA_ROOT}/tvm310。正式运行只使用H800主机CPU，设置CUDA_VISIBLE_DEVICES=''并审计禁止任何GPU fallback；启动前重新扫描NUMA 1，选择8个持续空闲的物理核并同时确认其sibling空闲，固定8线程、内存绑定、governor、batch=1、输入和模型scope。所有配置必须从31号H800+TVM、DeltaAP_max=0.10正式manifest/checkpoint/ONNX/SHA读取，不得按CPU结果换点：Original=(64,128,256,fp32)、Compression-only=(24,48,96,fp16)、Schedule-only使用Original结构、Compress->Tune=(32,48,128,fp16)、GEAR/joint=(16,32,64,int8)。先做统一低精度capability与数值gate；只有各臂所需FP16/INT8 Conv都存在真实lowering并通过数值合同时才做exact replay，否则所有臂保留各自结构并统一FP32，标记structure_replay_dtype_normalized_fp32，禁止混合口径。生成表A五臂正式结果：Original走PyTorch eager CPU正常优化，Compression-only走TVM LLVM default，Schedule-only、Compress->Tune、GEAR走TVM LLVM MetaSchedule tuned。四个唯一结构的TVM tuned必须严格执行第12.5.1节充分性门禁：每配置使用独立新数据库和相同mcpu/seed/target；先重新抽取全部task，采用逐task或强制round-robin，使每个tunable task至少获得64个有效measured trials，Conv及default profile占端到端时间至少1%的热点task至少获得256个有效trials；每配置以4096个有效trials为首轮目标，compile/run error及重复无效候选不计入有效数；若在6小时内达到4096 trials，则重建并测量端到端程序，最近512个有效trials仍带来至少1%改善时按2048 trials扩展，最多8192个有效trials，但不得突破每配置6小时硬上限；到时立即停止并保存可恢复数据库，禁止自动续跑。若6小时结束时逐task覆盖未满足，标记tuning_incomplete且不得填写正式tuned结果；若覆盖已满足但尚未收敛，标记tuned_6h_budget_limited，允许作为统一6小时预算结果填表，但不得称为TVM性能上限；只有全部tunable task均有有效记录、最终build数据库命中率100%、不存在0-trial可调task、热点最低预算满足，且加权task最优延迟和端到端p50在最后检查窗口内改善均小于1%（达到4096时检查最近512个有效trials，发生扩展后检查最近一个2048-trial扩展区间），才能标记tuned_converged。生成表B四配置执行器诊断：每个配置均测ORT CPUExecutionProvider、TVM LLVM default和TVM LLVM tuned，ORT开启正常graph optimization，不得人为弱化。每项先通过逐输出数值合同，再warmup至少100次、测量至少500次、3个独立进程，报告p50/p90/p99、mean/std/CV；可用时记录RAPL energy/frame，否则标记unavailable；编译和tuning时间单列。对Original和GEAR另做与正式timing隔离的ORT profile及TVM PrimFunc/task profile，保存每task有效/失败trials、database命中、0-trial/not_tunable task、预算扩展轨迹、layout/pad/cast/reshape/materialization计数，并比较end-to-end与kernel/task计时之和，定位当前TVM/ORT差距。新脚本、数据库和结果写入${V2X_DATA_ROOT}/lane_b_cpu_five_arm_v2_20260723独立目录。停止条件是表A五臂均有可信成功或明确failure终态，表B四配置三路径和三次重复闭合，TVM tuned单元格必须明确标记tuned_converged、tuned_6h_budget_limited或tuning_incomplete，其中只有前两种可以填写性能值，生成raw CSV、JSON、Markdown、数值/no-GPU审计、MetaSchedule覆盖与收敛审计和root-cause summary，明确GEAR相对Original/Compression-only/Compress->Tune的CPU zero-shot迁移效果，并依据同配置ORT/TVM真实性能给出当前Intel CPU应使用ORT还是TVM的部署建议；不得把预算受限结果写成TVM性能上限，不得把H800 winner回放写成CPU-aware search，也不得以TVM跨平台性覆盖ORT更快的结果。
```

## 13. H800 CPU 原生 INT8 编译与深度调度诊断 pilot（非正式表格）

本节是为了尽快闭合 Pyramid GEAR `(16,32,64)` 的原生 INT8 TVM 全流程并获得粗略比例，不替代第 12 章正式五臂实验。4090 不参与；第 12 章正式 runner 继续暂停；本节所有 latency 和 energy 一律标记为 `untrusted_due_to_cpu_contention`，不得写入表 A、表 B或论文正式性能表。

截至 2026-07-23，本 pilot 已完成原生 INT8 Relax/TIR 建图、默认编译、VM 加载、两 trial MetaSchedule 冒烟和三输出执行。已抽取 51 个 INT8 Conv 与 51 个 pack task，汇编观察到 `vpdpwssd`/`vpmaddwd`，没有 FP32 Conv silent fallback；另对 6 个近零权重卷积执行确定性的 int32 偏置可表示性 weight-scale floor 修复，不使用精度反馈或 CPU 性能反馈搜索。当前数值 gate 未通过，故任何粗略比例只能作为实现诊断，不是有效性能结论。

当前可恢复目录为 `${V2X_DATA_ROOT}/lane_b_cpu_gear_int8_pilot_v1_20260723`。执行器必须先读取 `status/pilot_orchestrator.pid`、`status/pilot_stage.txt`、`status/tuning_state.json` 和 `databases/int8_deep_tuned`：若现有 orchestrator 仍存活，只监控并接管，禁止重复启动；若异常停止，先保存 failure audit，只有 target、seed、源码 SHA、量化修复合同和 CPU 亲和完全一致时才允许从同一数据库恢复。曾使用错误 worker affinity 的数据库已隔离为 `int8_deep_tuned_invalid_affinity_*`，禁止重新导入。

### 13.1 H800 INT8 诊断 pilot `/goal`

```text
/goal 接管并完成33号文档第13章“H800 CPU原生INT8编译与深度调度诊断pilot”。本目标只是尽快闭合Pyramid GEAR/joint冻结结构(16,32,64)的原生INT8 TVM全流程和获得粗略比例，不运行第12章正式五臂，不运行cost model、acquisition、候选生成或任何CPU外层搜索，不使用4090，不修改或停止F-Cooper、H800 TensorRT Table 1、历史${V2X_DATA_ROOT}/tvm310和第12章正式数据。只使用H800主机CPU和独立目录${V2X_DATA_ROOT}/lane_b_cpu_gear_int8_pilot_v1_20260723；首先检查status/pilot_orchestrator.pid、status/pilot_stage.txt、status/tuning_state.json、databases/int8_deep_tuned和实际进程，若现有orchestrator仍存活则只接管监控、禁止重复启动，若异常停止则先保存failure audit，只有源码/ONNX/量化参数SHA、target、seed、量化修复合同和亲和一致时才从原数据库恢复，禁止导入任何int8_deep_tuned_invalid_affinity_*目录。冻结输入为31号正式Pyramid GEAR多尺度backbone ONNX `${V2X_ROOT}/results/stage35_gold32_supplement_v1_20260713/pyramid_sources_shape_repaired_v1/016x032x064/pyramid_016x032x064_multiscale.onnx`、SHA256=`8f09b5256f1856cc79ebbebf0d6c26e6dc63fba2ac3994552a690eaacd3be2a5`，量化参数读取`${V2X_ROOT}/results/stage35_gold32_supplement_v1_20260713/ap_v1/tvm_int8_repair/pyramid/16x32x64/tensor_quant_params.json`，校准输入读取`${V2X_ROOT}/results/stage35_gold32_supplement_v1_20260713/pyramid_calibration/016x032x064/spatial_features_train16.npz`并核对SHA；不得按CPU结果更换结构、输入或模型。设置CUDA_VISIBLE_DEVICES=''、TVM_BIND_THREADS=0、OMP/MKL/OPENBLAS/TVM_NUM_THREADS=8、OMP_DYNAMIC=FALSE、MKL_DYNAMIC=FALSE，使用LLVM mcpu=sapphirerapids、8线程、NUMA 1内存策略和当前冻结物理核60,71,73,88,89,93,100,102；审计主进程、LocalBuilder、LocalRunner及其popen worker的实际affinity均只能落在这8核，审计所有相关进程均无/dev/nvidia*文件句柄和GPU fallback，任何affinity漂移都必须立即停止、隔离该数据库并从新数据库重启。必须保存原生INT8证据：51个uint8/int8->int32 Conv task、51个pack/requant task、全部task名称/结构哈希、TIR、导出ELF共享库和objdump整数指令计数；禁止把QDQ后回退FP32写成INT8。保留确定性的6个退化卷积int32偏置可表示性weight-scale floor修复，保存原始量化参数SHA、逐层base/selected scale、quantized-bias范围和修复审计；该修复不得读取延迟或精度反馈，不得扩展成新的量化参数搜索。MetaSchedule只调102个Conv/pack热点task，使用独立JSONDatabase、相同seed/target和round-robin，4096个有效measured trials为目标、6小时墙钟为硬上限，compile/run error与无效结果不计入有效数；6小时到时立即停止并保存可恢复数据库，不自动扩展、不伪称收敛或TVM性能上限。调优结束后必须用数据库重新编译int8_tuned.so，报告有效/失败trials、逐task覆盖、0-trial task、database命中、编译警告、调优时间及default/tuned整数指令审计。使用同一实际输入逐输出比较ORT CPU EP FP32参考、TVM INT8 default和TVM INT8 tuned，报告shape、cosine、NRMSE、MAE和max_abs，并验证default与tuned原始量化输出是否逐元素一致；若数值gate失败，不得隐藏、放宽阈值或称为有效INT8加速，但为了本诊断目标允许继续完成隔离的粗略计时。粗略计时至少执行3个独立进程，每进程分别测TVM FP32 default、INT8 default和INT8 tuned；当前CPU存在外部争用，因此所有latency、energy、speedup字段都必须标记untrusted_due_to_cpu_contention、may_enter_formal_table=false、rough_speedup_claim_valid=false，energy暂填value=null，不得写入第12章正式表。分别计算INT8-default/FP32、INT8-tuned/INT8-default和INT8-tuned/FP32的比例；若比值小于1必须明确写成slowdown而不是加速。停止条件是：现有或恢复后的调优获得6小时/4096-trial明确终态，default/tuned共享库均可加载执行，三次粗略复测闭合，数值、no-GPU、worker-affinity、偏置修复、task覆盖和汇编审计完整，并在pilot目录生成raw JSON/CSV、Markdown summary和root-cause summary，同时把最终状态、粗略比例、数值gate结论和后续建议追加到本节；不得因此自动恢复第12章正式五臂runner。
```

### 13.2 Pilot 最终结果（2026-07-24）

本 pilot 已闭合，终态为 `tuned_6h_budget_limited_untrusted`。它证明 Pyramid GEAR/joint 冻结结构 `(16,32,64)` 可以在 H800 主机 CPU 上完成真实原生 INT8 建图、默认编译、MetaSchedule 数据库回编译、共享库加载和端到端执行；但本轮**没有获得可信的 INT8 加速结论**。数值 gate 失败，受外部 CPU 争用影响的三次粗测也均表现为 slowdown。该结果不得进入第 12 章正式表，也不得称为 TVM 性能上限。

冻结源合同全部通过：

- ONNX SHA256：`8f09b5256f1856cc79ebbebf0d6c26e6dc63fba2ac3994552a690eaacd3be2a5`；
- 量化参数 SHA256：`883ebb222742365c0e9e58a1461451c80d6854e8e3a6f09aa63b76b96fa323fe`；
- 校准输入 SHA256：`1907efa51ad800ab7e7d1b3652401f5c085544d57b7d17d11970f4f5854c3ca4`；
- target 为 LLVM `mcpu=sapphirerapids,num-cores=8`，运行核固定为 `60,71,73,88,89,93,100,102`，NUMA 1，`CUDA_VISIBLE_DEVICES=''`；
- 没有运行 CPU 外层搜索，没有更换结构、输入或模型，没有导入 `int8_deep_tuned_invalid_affinity_*`，也没有恢复第 12 章正式五臂 runner。

原生 INT8 与调度证据：

- 抽取结果为 `51` 个 `uint8/int8 -> int32` Conv task 和 `51` 个 pack/requant task，共 `102` 个深调 task；
- 102 个 task 的名称、结构哈希和逐 task TVM JSON TIR 均已保存并逐项核对，结构哈希与 JSONDatabase workload `102/102` 匹配；
- 6 个退化卷积使用确定性的 weight-scale floor 修复；修复审计包含逐层 base/selected scale 和 quantized-bias 范围，未读取延迟或精度反馈；
- `int8_default.so` SHA256 为 `8149ab9ae9060b44dbb5f591d791ae92c99d9e0faa1a662514f4badea724cca3`；
- `int8_tuned.so` SHA256 为 `e50c130cbec88713acb1745c324421510fe0f065612061ae506a2d7c2783a3f3`；
- default 与 tuned 的 objdump 均观察到 `vpdpwssd=31`、`vpmaddwd=144`，整数向量 fast path 审计通过；因此这里不是把 QDQ 后的 FP32 Conv 回退误写成 INT8。

MetaSchedule 在累计 6 小时硬上限处停止，没有自动扩展。数据库含 `1337` 条原始记录，其中 `1333` 条为有效 measured trial，`4` 条为 TVM 错误哨兵记录；失败类型细分在恢复后不可可靠重建，因此只报告数据库可证实的失败总数，不伪造 compile/run 分类。102 个 task 全部命中，0-trial task 为 `0`，逐 task 有效 trial 的最小值/中位数/最大值为 `2/13/25`，database hit rate 为 `100%`，最终回编译无 error、无 warning。由于只达到 `1333/4096`，该终态仅表示“6 小时预算受限”，不表示收敛。

数值结果如下。TVM INT8 default 与 tuned 的三个原始量化输出均逐元素一致，说明 MetaSchedule 没有改变结果；但 output 1 和 output 2 未通过相对 ORT CPU EP FP32 的合同，所以整体 gate 失败。

| Output | Shape | Cosine | NRMSE | MAE | Max abs | Default=tuned raw | Gate |
|---:|---|---:|---:|---:|---:|---|---|
| 0 | `(2,16,128,256)` | `0.997655` | `0.069362` | `0.080313` | `0.806122` | true | pass |
| 1 | `(2,32,64,128)` | `0.939581` | `0.367430` | `0.155576` | `3.322182` | true | fail |
| 2 | `(2,64,32,64)` | `0.881684` | `0.524212` | `0.103989` | `4.717876` | true | fail |

三次独立进程粗测的 representative p50 和跨进程 CV 如下。所有 latency、energy 和比例均为 `untrusted_due_to_cpu_contention`，`may_enter_formal_table=false`，`rough_speedup_claim_valid=false`，energy 为 `null`。

| 路径 | Representative p50 | 三次 p50 | 跨进程 CV |
|---|---:|---|---:|
| TVM FP32 default | `2492.868 ms` | `2241.586 / 2492.868 / 2566.215 ms` | `5.712%` |
| TVM INT8 default | `3420.328 ms` | `3205.703 / 3420.328 / 4231.173 ms` | `12.202%` |
| TVM INT8 tuned | `3801.288 ms` | `3414.884 / 3963.884 / 3801.288 ms` | `6.178%` |

粗略比例必须按实际方向解释：

- INT8 default / FP32：baseline/candidate=`0.7288`，即 INT8 default 为 `1.3720x slowdown`；
- INT8 tuned / INT8 default：baseline/candidate=`0.8998`，即深度调度后为 `1.1114x slowdown`；
- INT8 tuned / FP32：baseline/candidate=`0.6558`，即完整 INT8 tuned 路径为 `1.5249x slowdown`。

运行隔离审计共保存 `335` 个 orchestrator/LocalBuilder/LocalRunner/popen worker 快照，全部 affinity 均位于冻结 8 核，相关进程 `/dev/nvidia*` 文件句柄计数为 `0`；三次测量进程各自的 8 个运行线程也全部通过 affinity 与 no-GPU 审计。接管期间出现过两次控制面异常：一次为监控器读取瞬态 `/proc/<pid>/fd` 权限竞态，另一次为 TVM FFI 包装六小时 `SIGALRM`。两次均先保存 failure audit，再在 SHA、target、seed、量化修复和 affinity 合同一致的前提下恢复；终态恢复阶段没有新增 trial。累计 elapsed 字段包含约 287 秒清理/回编译开销，不能解释为超过 6 小时继续搜索。

当前根因只能限定为：自定义 im2col/pack/materialization 成本、MetaSchedule 深度不足（`1333/4096`，每 task `2–25` 条）和外部 CPU 争用的组合，本 pilot 尚不能分离三者贡献。下一步应优先修复 output 1/2 的 activation/weight 量化数值合同，再在持续空闲的物理核上重新测时；数值修复前不应追加调度预算，也不应把本轮 slowdown 写成有效 INT8 性能结论。

最终证据位于：

- `${V2X_DATA_ROOT}/lane_b_cpu_gear_int8_pilot_v1_20260723/reports/final_summary.json`
- `${V2X_DATA_ROOT}/lane_b_cpu_gear_int8_pilot_v1_20260723/reports/summary.md`
- `${V2X_DATA_ROOT}/lane_b_cpu_gear_int8_pilot_v1_20260723/reports/raw_measurements.csv`
- `${V2X_DATA_ROOT}/lane_b_cpu_gear_int8_pilot_v1_20260723/reports/root_cause_summary.json`
- `${V2X_DATA_ROOT}/lane_b_cpu_gear_int8_pilot_v1_20260723/audits/completion_audit.json`
- `${V2X_DATA_ROOT}/lane_b_cpu_gear_int8_pilot_v1_20260723/audits/task_tir/selected_task_tir_manifest.json`
- `${V2X_DATA_ROOT}/lane_b_cpu_gear_int8_pilot_v1_20260723/audits/failures/20260724T033924_monitor_proc_permission_race.json`
- `${V2X_DATA_ROOT}/lane_b_cpu_gear_int8_pilot_v1_20260723/audits/failures/20260724T052350_ffi_wrapped_timeout.json`
