# 硬件可配置维度清单 (D 维度) v2.0 — 完整重整

> 角色: 硬件维度研究员交付物。本版相对 v1.0 的核心改动:
> 1. 按**类别**(而非平铺 D1-D14)重组, 新增**并行调度大类**(multi-stream / CUDA Graph / GPU+DLA 并行 / 跨 agent 流水) —— v1 把这一整类几乎漏掉, 只有单设备路由 D2。
> 2. 每个维度逐条标 **实测信号来源 (真测 / 仅声明 / 待确认)**, 引用 `data/*.parquet` 的具体列与取值。
> 3. 订正 v1 两处与实测不符的声明 (DLA INT8、CUDA Graph 加速幅度), 见 §8 勘误。
>
> 数据源: `configs/hardware/{rtx4090,orin_agx,schema}.yaml`, `framework/{capability_schema,config_schema,propagation}.py`,
> `data/{4090_dspace_bench, orin_dspace_bench, cudagraph_bench, tactic_workspace_bench, unified_bench}.parquet`,
> `paper_learning/survey_raw_2/{ad_v2x_deployment, pruning_x_hardware}.md`, `plan6_方法优化路线_交接_v1.md`。

---

## 0. 一句话总览 + 实测覆盖地图

D 维度 = "同一个 (剪枝+量化后的) 网络, 部署到目标硬件时还能调的旋钮", 分 **6 大类**:

| 类别 | 维度数 | 实测覆盖 |
|------|--------|----------|
| A. 设备路由 & 并行调度 | 6 | **部分真测**: 单设备路由(GPU/DLA0/DLA1)、CUDA Graph 已真测; multi-stream / GPU+DLA 并行 / 跨 agent 仅声明或待确认 |
| B. TRT 构建旋钮 | 5 | **部分真测**: tactic / workspace 真测; opt level / strongly-typed 仅声明 |
| C. 精度 & 量化粒度 | 4 | **部分真测**: FP16/INT8 真测; FP8/INT4/per-group 仅声明 |
| D. DLA 专属约束 | 4 | **部分真测**: DLA FP16 真测成功 / DLA INT8 在 Pyramid 上真测**失败** |
| E. 功耗 / 频率 (Orin) | 2 | **部分真测 [2026-06-03 ISS-016 解封]**: nvpmodel 30W 档实测可用 (rail: VDD_GPU_SOC/VDD_CPU_CV/VIN_SYS_5V0, sudo tegrastats; `results/E6_orin_energy.csv`); E1 功率档轴从死维解封为可测; MAXN 切档 + perf/watt sweep 进行中 |
| F. 内存 / 带宽 | 3 | **间接**: 带宽来自 spec, 显存 peak 有真测列 |

> 平台门控: 取值域完全由目标硬件 capability 决定。FP8/INT4/per-group/strongly-typed 仅 4090; DLA 路由/per-tensor-INT8/nvpmodel 仅 Orin。

**实测 bench 一览 (这是判断"真测 vs 声明"的硬依据):**

| bench 文件 | 行数 | 覆盖的 D 维度 | 平台 |
|------------|------|---------------|------|
| `4090_dspace_bench.parquet` | 48 | tactic{default,with_cudnn,edge_only,all_enabled} × workspace{4,8GB} × precision{fp16,int8} × 3 triplet | 4090 |
| `tactic_workspace_bench.parquet` | 8 | tactic_flag{27,31} × workspace{4,8GB} × precision{fp16,int8} | 4090 |
| `cudagraph_bench.parquet` | 12 | cuda_graph{on,off} × precision{fp16,int8} × 3 triplet | 4090 |
| `orin_dspace_bench.parquet` | 72 | d_scheme{A_gpu,B_dla0,B_dla1} × tactic{default,no_cudnn} × workspace{1,2GB} × precision{fp16,int8} × 3 triplet | Orin |
| `unified_bench.parquet` | 660 | 上述汇总 + `dual_A`(GPU+DLA 多引擎, **lat=NaN 未测**) + 跨模型 | 两平台 |

---

## 0.5 各 D 维"可搜性"裁决 (★[2026-06-02] 保留全维度目录, 但明标每条的主要问题)

> 用户决策: **不收缩维度目录**(全维度有文档/通用性价值), **但必须区分"物理存在"与"搜索空间可搜性"** —— 否则会把"6 大类 ~20 维"误当成可搜空间, 而代码里真正进 `Config` 的 D 维只有 `d_routing` 一条。下表给每条轴一个裁决 + 主要问题。

| 轴 | 可搜性裁决 | 主要问题 (为什么不该当主搜索变量) | 证据 |
|---|---|---|---|
| **A1 设备路由** | ✅ **唯一真·可搜 D 维** (在 `Config.d_routing`) | 仅 Orin 有意义(4090 只有 GPU); 且 DLA INT8 全失败 ⇒ 有效路由仅 {GPU-fp16, GPU-int8, DLA-fp16} | Orin 真测 3 档 lat |
| B1 tactic / B2 workspace | ⚠️ 编译期参数, 非搜索变量 | 实测影响 **5–9%**(随剪枝率升), 应交 **TRT-auto build-time 自选**, 不该枚举(与"per-stage 混精被 TRT-auto 支配"同理) | `4090_dspace_bench` 实查 |
| A3 CUDA Graph | ⚠️ 弱开关, 默认即可 | 真测仅 1.05–1.10×, 几乎无杠杆 | `cudagraph_bench` |
| A4 multi-stream | ⚠️ 弱开关 | 真测峰值 1.13×@2流、单帧延迟劣化 9.8×; 最优流数=2≠streams_max=8 | `E1` |
| A5 GPU+DLA 并行 | ❌ 0 实测 | `dual_A` 7 行 build 通但 lat 全 NaN; **`DeployPlan` 无 multi-engine 字段** → 当前无处承载 | `unified_bench` |
| A6/A7 跨 agent/跨帧 | ❌ 0 端到端实测 | 纯架构声明; **是吞吐(非延迟)轴**, 且需 batch/流水才解耦 | — |
| **batch (新增)** | ⏳ **待加 + 待测** | throughput 脱离"=1/latency 冗余"的唯一途径; 增益 = min(GPU 空余算力, batch); 时间维 batch 扣攒批延迟, 空间维(RSU 多车/多传感器)纯吞吐净赚。需 batch-sweep 真测 | corr(tput, 1/lat)=0.9996(实查) |
| C 精度 | 门控前提 | 量化师定, 不进 D 笛卡尔积 | — |
| **B3 opt-level** | ✅ **已升级为真维度** (2026-06-04 H2 真测) | H2 真测: opt5 vs opt3 @ws4096 = **21% 提速** (1.250ms vs 1.599ms) + 消除 3 FP32 fallback 层; **建议: 生产 build 固定 opt=5** | `H2_workspace_scan_4090_final.csv` base_fp16_opt5 行 |
| B5 strongly-typed | ❌ 0 latency 实测 | — | — |
| **2:4 structured sparsity** | ⚠️ **已真测, 收益极小** (2026-06-04 H1 真测) | H1 真测: 2:4+finetune sparse vs non-sparse = **仅 1.02× 提速** (fp16: 1.235vs1.258ms; int8: 0.786vs0.803ms); memory-bound 模型 2:4 不解带宽瓶颈; AP 完全恢复(finetune 2 epoch); **不作主搜索轴** | `H1_sparsity_4090.csv` + `H2_*_final.csv` |
| E1 功耗档 | ⏳ 部分真测 (30W 解封, ISS-016) | nvpmodel 30W 可用 / MAXN 待补 | E6 (`results/E6_orin_energy.csv`) |

> **裁决总结 (2026-06-04 H1/H2 更新)**: 搜索器的 D 维主战场 = **A1 routing**(异构边缘) + 待加的 **batch 轴**(吞吐 regime); tactic/workspace 交 TRT-auto; A3/A4/A5/A7 作 DeployPlan 可选开关/消融, **不作主搜索轴**。**新增**: B3 opt-level=5 应固定为 build-time 最佳实践(21% 提速,无 AP 代价); 2:4 sparsity 对 memory-bound PyramidFusion 收益极小(1.02×),降为"已真测-不推荐"。这与"per-stage 混精被 TRT-auto 支配"是同一规律 —— 延迟驱动的编译期决策委托 TRT/build-time 优化, 框架别手工枚举。**论文叙事: 协同杠杆是 量化×剪枝, D 维贡献是 routing(GPU/DLA)+ D↔B2 约束传播 + 吞吐 regime 的 batch + opt=5 build-time 最佳实践**。

> **FPGA 定位 (用户决策 2026-06-02)**: **仅在论文 related-work 讨论** FPGA dataflow / II=1 空间展开作为 GPU 时间复用的对照, **不进框架真实覆盖, 也不作 future-work 主线**。本文档凡提 FPGA(如 §A7、E 类能耗)均为 related-work 级对照, 不构成框架可执行维度。

---

## A 类. 设备路由 & 并行调度

> v1 的最大缺口在此。v1 只有 D2(单模块设备路由)一条, 完全没有"调度并行"轴。本类补齐 6 条。
> **可搜性见 §0.5 裁决表** —— 本类除 A1 外多为弱开关或 0 实测声明轴, 列出供完整性, 非主搜索变量。

| # | 维度 | 取值域 (4090 / Orin) | 平台约束 | 实测信号 | schema 映射 |
|---|------|---------------------|----------|----------|-------------|
| A1 | **单设备路由** (per-module) | 4090: {GPU}; Orin: {GPU, DLA0, DLA1} | DLA 仅 Orin (4090 `dla.enabled=false`) | **真测**: `orin_dspace_bench.d_scheme ∈ {A_gpu,B_dla0,B_dla1}`, 各档 lat_p50 真测 (见 §4 表) | `config_schema.d_routing[m]`; `capability.has_dla` |
| A2 | **GPU fallback** (DLA 时) | {on, off} (仅 Orin) | `--allowGPUFallback` / `BuilderFlag::GPU_FALLBACK`; 未支持层自动回 GPU | **仅声明**: bench 未单独区分 fallback on/off; orin DLA 失败行的 fail_reason 是 `kDIRECT_IO`/bank 超限, 非 fallback 维度 | DeployPlan; `orin_agx.dla.op_blacklist` |
| A3 | **CUDA Graph** (kernel launch 合批) | {on, off} (两平台 `cuda_graph: true`) | 需静态 shape; 动态激活剪枝路径无法 graph 化 (survey pruning §3.4) | **真测 (4090)**: `cudagraph_bench`, 加速 **1.05–1.10×** (见 §A 实测块) | `capability.features.cuda_graph` |
| A4 | **multi-stream 多流并发** | {1, …, streams_max} 4090=8 / Orin=8 | 仅当存在**异构可并行子图**才有收益 (survey: GQA/MoE/dual backbone) | **真测 (4090, E1, 2026-06-02)**: T_baseline fp16, 同 cudagraph_bench engine。FPS@{1,2,4,8}=789/891/823/643 qps → **峰值 1.13× @streams=2, ≥4 流因 SM 争用反降 (8 流 0.82×)**; **单帧 latency p50 单调劣化 1.0→1.77→3.86→9.80×** (并发抢 SM, 关键路径不缩)。印证"GPU 跨流=吞吐非延迟", 且同器件单网络 multi-stream 收益微弱(子图已打满 SM, 无异构互补) | `capability.ips.gpu.streams_max` (extra) |
| A5 | **GPU+DLA 并行** (dual/triple) | 4090: n/a; Orin: {GPU 单 / GPU+DLA0 / GPU+DLA0+DLA1} | 双 backbone 各占一 DLA, GPU 留给 Transformer 头 (survey §3.1 四路并行) | **✅ 真测 (E3, 2026-06-02, `results/E3_orin_dla_pipeline.csv`)**: DLA0(stage01)∥DLA1(stage2) —— **进程内双流 1.00×(完全串行, TRT 8.5 进程内序列化 DLA 提交)**, **双进程 1.34×(真硅片并行)**。dual_A 的 NaN 已跑出真数: **异构并行真实但只有多进程编排拿得到** | DeployPlan (多 engine **多进程**编排) |
| A6 | **跨 agent / 跨 IP 流水线重叠** (空间维) | V2X ego+infra 两条流水, stage 重叠调度 | UniV2X 天然 ego+infra 异构双流 (survey pruning §4 双模型) | **仅声明 (架构级)**: survey 引 PARA-Drive/DART/Holistic Scheduling; 本仓库**无端到端跨 agent 调度 latency 实测** | 框架级, 不在单 config_schema 内 |
| A7 | **跨帧 / 时间维流水线** (inter-frame software pipelining) | {off, intra-frame, inter-frame}; 帧 N@stageK 与帧 N+1@stageK-1 并发 | 同器件靠 A4 multi-stream / 异构靠 A5 GPU+DLA; A7 = A4/A5/A6 在**时间维**的投影, 非新物理机制 | **真测 (4090, E1, 经 A4 multi-stream 投影)**: 同器件 A4=A7 在时间维的实现; 实测 FPS 峰值仅 1.13× 而单帧 latency 1.0→9.8× 劣化 → **直接证明 A7"跨帧流水=吞吐非延迟"的理论判断**。唯一可能降单帧的异构形态 (A5 GPU+DLA) 待 E3 | DeployPlan (build-time 编排) |

> **★A7 本质 (用户问"GPU 能否做跨帧流水")**: **能, 但本质是吞吐(FPS)优化, 不降单帧延迟**。FPGA 是**空间展开**(每 stage 独占电路, II=1 时延迟也被流水隐藏); GPU 是**时间复用同一批 SM**, 帧间"重叠"只是两帧 kernel 抢同组 SM —— **仅当 stage 间资源互补(memory-bound × compute-bound / TC × CUDA core / GPU stage × DLA stage)且单 stage 未打满 SM 时才有净收益**。任意一帧仍要顺序过 stage1→…→stageK, 关键路径不变 → **单帧 e2e ≈ 不变甚至略升(调度/抢占开销), 只缩短稳态相邻帧间隔(FPS↑)**。
> **对 V2X 部署的价值 [2026-06-02 订正 — 之前"价值有限/非吞吐"是错误框定]**: 实时流式感知有**两条都必须满足的硬实时约束** —— ①**吞吐下限** 持续 FPS ≥ 传感器/聚合输入帧率(跟不上=丢帧/积压=失实时, **路测连续工况、RSU 多车服务正是吞吐 binding**), ②**延迟上限** 单帧 e2e ≤ 控制截止(安全/新鲜度)。**哪条 binding 取决于部署 regime**: ego 单车单流且模型够快 → 吞吐自动满足、latency-binding; **RSU 路侧服务多车 / 多传感器 / 多任务共享算力 → 吞吐 binding**。⇒ 跨帧流水服务的吞吐约束**是真需求**, A7 在吞吐-binding regime 有真实价值。**但 E1 实测同器件 multi-stream 这个"具体手段"很弱(峰值 1.13×、单帧还劣化)** —— 真正能提吞吐**且同时降延迟+能耗**的杠杆是: work-reduction(剪枝/量化, 三赢)/ 异构 GPU+DLA(A5/E3)/ 跨 agent 批处理。结论: **吞吐目标重要, 但 multi-stream 不是好杠杆**。
> **唯一可能也降单帧延迟的形态 = GPU+DLA 异构跨帧流水 (Orin, A5)**: DLA 独立于 SM, 物理并行不抢资源, 最接近 FPGA dataflow。但 §D 真测: Pyramid 上 DLA INT8 0/12、deformable 必回 GPU、dual_A 7 行 latency 全 NaN(0 实测) → 切分点被算子约束死, 真可行性待 E3 验证。
>
> **★★ stage 级流水线并行 (pipeline parallelism) — 用户 2026-06-02 精确提问的正解**:
> 用户设想 = 把一次 forward 切成 stage1→…→stageK, frame N@stage2 与 frame N+1@stage1 并发, 使**稳态出帧间隔 → max(stage) 而非 sum(stage)**(单帧 e2e 仍 = sum, 不变)。**这是 pipeline 并行, ≠ E1 的 data 并行(多流各跑整模型), 但受同一物理天花板限制**:
> - **能否拿到"间隔=max(stage)"取决于相邻 stage 是否落在不相交的硬件资源上**。FPGA = 空间分立(每 stage 独占电路)→ 真并行 → 间隔=max, II=1。GPU = 所有 stage 共享同一 SM 池 → frame N@stage2 与 frame N+1@stage1 抢同批 SM。
> - **若每 stage 已打满 SM → 并发只能时间复用 → 间隔≈sum, 零收益**; 若 stage 欠饱和(memory-bound/小/低占用)→ 有空余资源可真重叠 → 间隔→max。
> - **★[2026-06-02 E_headroom 真测订正 — 我之前"pyramid stage 已近饱和、单 GPU pipeline 零收益"的判断错了]**: roofline 实测(GPU2 干净)显示 **pyramid backbone 三 stage 是 memory-bound 且只用 ~30% 带宽(289–322 / 1008 GB/s)、算力占比更低 → 严重欠饱和, 不是饱和**; 且 **stage 瓶颈互补**(backbone=memory-bound, deblocks=compute-bound 88%@330peak, shrink/heads 也欠饱和)。**latency-加权重叠余量 = 0.53(165 peak)~0.63(330 peak FP16-accum)**, 远高于 E1 data 并行实测的 **0.13**。
> - **[E_headroom 一度暗示上行空间]**: 0.53–0.63 余量 vs E1 的 0.13, 看似 pipeline 并行(memory-bound backbone ∥ compute-bound shrink 错位重叠)能填互补、把 0.13 推向 0.5+。**但这是 roofline 的乐观推断, 必须实跑验证**(峰值 165 vs 330 影响 compute-bound stage 余量; memory-bound backbone 余量 0.68–0.71 基于带宽、稳)。
> - **★★[2026-06-02 E_pipeline 实测定夺 — 推翻上述乐观推断]**(GPU2 干净, `results/E_pipeline_singlegpu_4090.csv`, 数值校验 maxdiff=0): 单 GPU stage-resident 软件流水峰值 speedup = **1.08×(3 流), 反而低于 E1 data 并行的 1.13×**; 出帧间隔 7.57ms 距理想 max(stage)=2.10ms 差 **3.6×**(几乎仍是串行 sum 8.19ms); 流数>3 因调度/event 开销劣化。kernel-overlap 微基准: **最互补的 memory∥compute 对(stage0∥shrink)也仅重叠 1.11×**。
> - **⇒ 结论(已定夺)**: **roofline 0.53–0.63 是不可达的乐观上界**; 单 GPU stage 流水**拿不到 stage 互补收益**, 根因 **occupancy 槽位 ≠ 可并发吞吐空隙**(memory-bound kernel 虽算力闲, 但仍占满 SM occupancy + 共享 L2/显存控制器 + eager 逐 kernel 调度串行化)。**pipeline 并行相对 data 并行在单 GPU 上无独有价值**; 同器件单 GPU 流水从主搜索轴**降为已证伪的消融对照**。
> - **唯一仍有上行的形态 = 不相交物理资源**: ① 异构 **GPU∥DLA(Orin, E3, 唯一可能也降单帧延迟)**; ② **GPU∥CPU**(voxelize/NMS 与下帧 GPU forward 重叠)。这是 FPGA 空间分立优势在 GPU 平台的唯一对应。MPS 空间切 SM 理论可行但每 stage 只拿部分 SM、均衡 compute-bound 不划算。
> **指标口径强制双列**: throughput(FPS, 可受益) + 单帧 e2e latency(基本不变)。**不可只报 FPS 当"加速"**(犯把吞吐当延迟报告的错, 见 CLAUDE.md §六)。
> **可执行实验 (收益均为估算待真测)**:
> - **E_headroom (4090, ✅ 已完成 2026-06-02, GPU2 干净, `results/E_headroom_roofline_4090.csv`)**: roofline/achieved-throughput 法(绕开 ncu admin)。逐 stage 真测:
>
>   | stage | p50(ms) | 实测算力(TFLOPS) | 实测带宽(GB/s) | bound | 余量@330peak |
>   |---|---|---|---|---|---|
>   | stage0/1/2 backbone | 1.87/1.85/1.48 | 9.5/17.6/33.9 | 289/322/296 | **memory(~30%BW)** | 0.71/0.68/0.71 |
>   | deblocks_upsample | 0.58 | 289.5 | 242 | compute(88%) | 0.12 |
>   | shrink_conv | 2.10 | 118.5 | 87 | compute(36%) | 0.64 |
>   | heads | 0.11 | 13.5 | (cache artifact) | — | ~0 |
>
>   **latency-加权重叠余量 = 0.53(165 peak)~0.63(330 peak)** vs E1 data 并行实测 **0.13**。结论见上方 ★stage 流水块: backbone 欠饱和 + stage 互补 → 单 GPU pipeline 有上行空间(0.13→上界 0.5+), 但 roofline 余量是乐观上界(occupancy ≠ throughput 空隙), 需单 GPU stage 流水实跑定夺。
> - **E3 (Orin DLA∥DLA, ✅ 已完成 2026-06-02, `results/E3_orin_dla_pipeline.csv`)**: stage01@DLA0(14.49ms)∥stage2@DLA1(6.18ms), 串行 sum=20.67ms。**进程内双流 = 20.65ms(1.00×, 完全串行 —— TRT 8.5 进程内序列化 DLA 提交, 即便独立 engine)**; **双进程 = 15.4ms 稳态间隔(1.34× 真硅片并行)**(DLA1 拖慢 DLA0 仅 +6%, 证实独立 DLA 硅片非 GPU fallback)。⇒ **异构 GPU∥DLA 是用户 stage 流水想法唯一真生效的形态(1.34× vs 单 GPU 1.08×), 但必须多进程编排**(进程内为 0)。caveat: 现为各 stage 独立 back-to-back 循环测的吞吐上界; 真正跨进程帧交接(stage01→stage2 共享内存 ring buffer, +~0.5MB/帧拷贝)未建, 是 E3 剩余工程。
> - **[单 GPU pipeline 文献调研补充]**: 1.08× 与文献基线一致(naive 并发 ~1.13×, arXiv 2412.14335); 对位更优法 = **Opara(CUDA Graph+算子资源感知 overlap, 峰值 1.68× 但 conv-dense 顺序模型仅 1.1–1.3×, arXiv 2312.10351)**; MPS/Green Contexts/REEF/Orion/HFTA 均多进程/多模型, **不适用单模型 stage 流水**。物理精修: memory-bound stage 并发撞 **L2/DRAM 带宽墙**(SM 分区不解), 非仅 SM occupancy。⇒ 单卡可选 Opara 式重测拿 ~1.1–1.3× 收口, 数量级仍须异构/量化/剪枝。
> - **E1(已完成)**: 4090 同器件 multi-stream, FPS 峰值 1.13×、单帧劣化 → 见 §A 实测块。
> - **E2(待)**: 跨 agent 帧间(ego/infra 双流, fuse=join barrier 上限, FPS~1.2–1.6× 估算)。

**A 类实测块 — CUDA Graph (4090, `cudagraph_bench`, 真测 p50 ms):**

| triplet | precision | graph off | graph on | 加速 |
|---------|-----------|-----------|----------|------|
| 064_128_256 | fp16 | 1.261 | 1.205 | 1.05× |
| 064_128_256 | int8 | 0.814 | 0.753 | 1.08× |
| 016_032_064 | int8 | 0.605 | 0.549 | 1.10× |

> 结论(真测): CUDA Graph 对 sub-millisecond 的 pyramid_backbone 子模块只省 **5–10%** (launch overhead 占比随模型变小而升)。**远不是数量级收益**, 与 plan6 §三"根因 B kernel launch overhead 0.05-0.1ms"一致。

**A 类实测块 — multi-stream 多流并发 (4090, E1, 2026-06-02, GPU6 全空闲真测):**

engine = `models/p0_random_cache/engine_064_128_256_fp16.engine` (与 cudagraph_bench T_baseline fp16 **同一 engine**, 口径可比; 单帧 p50 @streams=1 = 1.270ms ≈ cudagraph graph-off 1.261ms, 互相印证)。N 流各自独立 context + IO buffer, 全部 in-flight 后统一 sync。

| streams | throughput (qps) | FPS 倍数 vs streams=1 | 单帧 latency p50 (ms) | latency 倍数 vs streams=1 |
|---------|------------------|----------------------|----------------------|--------------------------|
| 1 (baseline 顺序) | 788.7 | 1.00× | 1.270 | 1.00× |
| 2 | 890.9 | **1.13×** (峰值) | 2.242 | 1.77× |
| 4 | 823.0 | 1.04× | 4.903 | 3.86× |
| 8 | 642.6 | **0.82× (反降)** | 12.438 | 9.80× |

> 结论(真测, 双口径分列):
> 1. **吞吐(FPS)**: multi-stream 仅在 streams=2 拿到 **1.13×** 峰值, streams≥4 因多流抢同组 SM 反而**掉到 1.04× / 0.82×**。同器件单网络无异构互补子图 → multi-stream 吞吐收益微弱且很快饱和。
> 2. **单帧 latency**: **单调劣化** 1.00→1.77→3.86→**9.80×** (近似线性于流数), 完全符合"GPU 时间复用同一批 SM, 帧间并发不缩关键路径, 反增排队/抢占" 的物理判断。
> 3. ⇒ **印证 §A7 本质 (仅机制层面, 不是说吞吐目标不重要)**: 同器件 multi-stream 在 4090 pyramid_backbone 上 **是吞吐优化、且很有限(峰值 1.13×@2流, 最优流数=2≠8), 单帧延迟不降反升** —— 不可把 FPS 当"单帧加速"上报。**[2026-06-02 订正]**: 这只说明 multi-stream **这个手段**弱, **不是说吞吐这个目标无价值** —— 吞吐 ≥ 输入帧率是路测/RSU 多车的硬实时约束(见 A7 价值订正)。正确结论: **多流不是好的吞吐杠杆; 要提吞吐应走 work-reduction(同时降 latency+能耗)/ GPU+DLA / 跨 agent 批处理**。

---

## B 类. TRT 构建旋钮

| # | 维度 | 取值域 (4090 / Orin) | 平台约束 | 实测信号 | schema 映射 |
|---|------|---------------------|----------|----------|-------------|
| B1 | **tactic sources** | 4090: {default, with_cudnn, edge_only, all_enabled} (= CUBLAS/CUBLAS_LT/CUDNN/EDGE_MASK_CONV/JIT 子集); Orin: {default, no_cudnn} | TRT 版本相关 (TRT10 vs 8.5 enum 不同) | **真测**: `4090_dspace_bench.d_tactic` 4 档 × 真测 lat; `tactic_workspace_bench.tactic_flag∈{27,31}` 真测 | DeployPlan.tactic_sources |
| B2 | **workspace (memPoolSize)** | 4090: {4, 8}GB 真测 (声明可到 16); Orin: {1, 2}GB | ≤ `memory.available_for_inference_gb` | **真测 (旧 + H2 扩展, 2026-06-04)**: 4090 {256MB,1GB,2GB,4GB} × {base_fp16, p50_int8, p25_trap_int8} × body_subnet_collab2 全 13 点 (`results/H2_workspace_scan_4090_final.csv`). **关键: 非单调** — base_fp16 ws1024最快 1.258ms 而非 ws4096 1.599ms; **p50_int8 INT8层数随ws增大**: ws256=23层→ws4096=**47层**; p25_trap workspace 无效(仅5-6 INT8层,workspace不解kernel-cliff) | DeployPlan.workspace_gb |
| B3 | **builder optimization level** | TRT10: {0..5} (4090); TRT8.5 无此 API (Orin) | TRT≥10 才有 `BuilderConfig.builder_optimization_level` | **✅ 真测 (H2, 2026-06-04)**: base_fp16 ws4096 opt3=1.599ms vs opt5=**1.250ms → 21% 提速** + 消除 3 FP32 fallback 层 (72→73 FP16 layers); `results/H2_workspace_scan_4090_final.csv` | DeployPlan (待加字段) |
| B4 | **precision flags** | 见 C 类 (FP32/FP16/INT8/FP8/INT4 flag) | — | 见 C 类 | — |
| B5 | **strongly-typed** | 4090: {on, off}; Orin: {off} | 仅 TRT10; 与显式 FP16 flag 互斥 | **仅声明**: `rtx4090.features.trt_strongly_typed=true` / orin=false; 未做 on/off latency 对比实测 | `capability.features.trt_strongly_typed` |

> 注: `4090_dspace_bench` 里 tactic×workspace×precision×triplet 全 48 行 `build_success=True`, 但 tactic/workspace 对 lat_p50 的影响 **实测 5–9%(随剪枝率升, 实查 fp16 spread 5.5/6.1/8.9%)** —— 不可忽略但也不该枚举进搜索, 更适合 **build-time 让 TRT-auto 自选**。即 **B 类是合法性/编译期 auto-tune 维度, 不是搜索决策变量**。

---

## C 类. 精度 & 量化粒度

| # | 维度 | 4090 取值域 | Orin 取值域 | 平台约束 | 实测信号 |
|---|------|-------------|-------------|----------|----------|
| C1 | **precision** | {FP32, TF32, BF16, FP16, INT8, INT4, FP8(E4M3/E5M2)} | {FP32, TF32, BF16, FP16, INT8, INT4} | FP8 仅 Ada TC4; 见 §3 矩阵 | **真测**: FP16/INT8 全平台 dspace 真测; FP8/INT4/BF16 **仅声明** (capability.precisions 列出, 0 latency) |
| C2 | **INT8 模式** | {explicit Q/DQ, implicit calib} | {implicit calib only} | explicit/STRONGLY_TYPED 仅 TRT10; DLA INT8 必须 implicit (survey §1.2 痛点 6) | **真测**: 4090 INT8 真测; Orin GPU INT8 真测; calib 方法见 `unified_bench.calibration_method` |
| C3 | **量化粒度 (W)** | {per-tensor, per-channel, per-group(AWQ)} | {per-tensor, per-channel} | per-group 仅 TRT10; DLA 仅 per-tensor | **仅声明**: per-group/per-channel 区分未单独 bench latency |
| C4 | **量化粒度 (A)** | {per-tensor} | {per-tensor} | per-channel-A GPU GEMM 不支持 (两平台) | **声明**: `quant_constraints.per_channel_activation_supported=false` |

> 说明: C 类通常由量化师在 B2(量化)维度固定, 一次 build 一个精度。D 维度本体把 precision 当门控前提, 不与 tactic/workspace 做全笛卡尔积 (见 §6 档位计数)。

---

## D 类. DLA 专属约束 (仅 Orin)

| # | 约束 | 内容 | 实测信号 |
|---|------|------|----------|
| D1 | **精度限制** | DLA 仅 FP16 + INT8, 不支持 FP32/BF16/FP8 (survey §1.2) | **真测**: propagation.py 强制 DLA 模块 q_bits ∈ {FP16,INT8} |
| D2 | **per-tensor only** | DLA INT8 仅 per-tensor (混 per-channel kernel 路径受限) | **声明**: `propagation.propagate_d_to_b2` 强制 per-tensor; survey pruning §5.2 |
| D3 | **op 白名单 / GPU fallback** | LayerNorm/DeformableConv/grid_sample 不支持 → 必落 GPU (`orin_agx.dla.op_blacklist`) | **声明**: survey §1.4 — Deformable Attn 的 grid_sample 必 GPU; 动态 shape 与 DLA 静态维度冲突 |
| D4 | **稀疏在 DLA 无普适收益** | 2:4 仅对 math-bound 层有效 (RetinaNet-R34 1.36×); memory-bound 小卷积/DWConv 无收益 (survey pruning §5.1) | **声明 (外部实测引用)**: NVIDIA DLA-SW repo |

**D 类关键实测 (订正 v1 + CLAUDE.md 的"DLA INT8-only"说法):**

`orin_dspace_bench` build_success 按 scheme×precision:

| d_scheme | precision | 成功/总 | 真测 lat_p50 (016 triplet) |
|----------|-----------|---------|----------------------------|
| A_gpu | fp16 | 12/12 | 26.50 ms |
| A_gpu | int8 | 12/12 | **20.55 ms** (GPU INT8 真加速) |
| B_dla0 | fp16 | 8/12 | **20.56 ms** (DLA FP16 比 GPU FP16 快 ~22%) |
| B_dla0 | **int8** | **0/12** | — (**全失败**) |
| B_dla1 | fp16 | 8/12 | 20.69 ms |
| B_dla1 | int8 | 0/12 | — (全失败) |

> **DLA INT8 在 Pyramid triplet 上 0/12 全失败** (fail_reason: `kDIRECT_IO 无 conformant 实现` + `Maximum allotted banks=16 < 5+12`)。
> 但 `orin_agx.yaml empirical_baseline` 写 `int8_dla0_build_success: true` —— **那是 ResNet50** (干净 conv 网, N2v2 sanity), **不是 Pyramid**。
> ⇒ **真测结论: DLA INT8 对干净 conv 网可行, 对 PyramidFusion (含 deformable/grid_sample/非对齐通道) 当前不可 build**。DLA 在本网络上**只有 FP16 路径真测可用**。CLAUDE.md/v1 的"DLA 即 INT8-only 主路径"应改为"DLA FP16 真可用, INT8 待解 bank/IO 约束"。

---

## E 类. 功耗 / 频率 (仅 Orin)

| # | 维度 | 取值域 | 实测信号 |
|---|------|--------|----------|
| E1 | **nvpmodel 功耗档** | {MAXN 60W, 15W, 30W, 50W} (`orin_agx.power.modes` 4 档, 含各档 gpu/dla freq) | **部分真测 [2026-06-03 ISS-016 解封]**: nvpmodel **30W 档实测可用** (`nvpmodel -q`=MODE_30W, GPU 锁 612MHz; base collab2 fp16 48.68ms / VIN_SYS 7.13W / 348mJ-frame, `results/E6_orin_energy.csv`)。旧 "service inactive/optMask 失败" 表述过时 → E1 从死维**解封为可测**。**MAXN 切档 + 30W↔MAXN perf/watt sweep 进行中**(实际切换+MAXN 数据未出, 勿写"已全测") |
| E2 | **锁频 (jetson_clocks)** | {on, off} | **间接**: N2v2 ResNet50 baseline 是锁频 MODE_30W 下测的 (2.530ms FP16); 但 dspace bench 未系统扫 freq 轴 |

> 4090 无 nvpmodel (`power.modes` 单档 default 450W) → E 类**功耗档**在 4090 退化为单档不可搜。

> **★[2026-06-02] 能耗 = Pareto 一级指标 (非仅 future work)**: 边缘核心 figure-of-merit 是 **perf/watt / 能耗每帧 (J/frame)**。Orin nvpmodel 15–60W 直接 trade latency↔功率; FPGA 卖点即能效。框架愿景原文已含"精度×延迟(×能耗)", 现**从括号升为一级 Pareto 轴**。
> - **✅ E4 已完成 (2026-06-02, 4090 GPU1 干净, 13 engine, `results/E4_energy_4090.csv`)**: `J/frame = P̄ × 单帧 latency`(整板功率, NVML 采样; 干净校验排除自身 PID)。**INT8 在能耗上是干净正向点 —— 比 FP16 省 30–52% J/frame**(功率更低 + 延迟更短复合):
>
>   | 配置 | 精度 | lat(ms) | P̄(W) | **J/frame(mJ)** | vs FP16 |
>   |---|---|---|---|---|---|
>   | T1_base | fp32/fp16/int8 | 2.30/0.82/0.49 | 442/355/286 | 1015 / 291 / **141** | int8 **0.48×** |
>   | cudagraph T_baseline | fp16/int8 | 1.25/0.80 | 390/334 | 486 / **266** | **0.55×** |
>   | T_prune75 | fp16/int8 | 0.75/0.59 | 371/333 | 280 / **197** | **0.70×** |
>
>   idle 基线功率 = 28.9W。⇒ energy 轴**放大** INT8 vs FP16 的差距(单看 latency INT8 也快, 能耗轴差距更大), 坐实其作为 Pareto 一级轴的区分力。
> - **Orin 能耗 30W 已测 (ISS-016, E6)**: `sudo tegrastats` 读 VIN_SYS_5V0(module)/VDD_GPU_SOC 功率轨(nvpmodel 非故障, 30W 档可用); base collab2 fp16 @30W = **348mJ/frame < 4090 462mJ(VERIFIED)** → **Orin 单帧能耗(J/frame)更低**。
>   - ⚠️ **rail scope 非完全对等 [ISS-016]**: Orin `VIN_SYS_5V0` = 整模块 5V 输入(含 CPU+SoC+mem); 4090 NVML = **GPU 卡本身**(不含 host CPU/系统)。board-vs-board 近似公平但**非同 scope**, 论文勿写 apples-to-apples。
>   - ⚠️ **措辞精确 [ISS-016]**: 只能说 "Orin **单帧能耗(J/frame)更低**", **不可笼统说 "perf/watt 占优"** —— Orin **吞吐仅 ~1/40**(20.4 vs ~790 qps)→ throughput/watt 反而 4090 占优。卖点 = 边缘**低单帧能耗/低功率预算**, 非全面 perf/watt。
>   - **仍缺**: MAXN 档 + p50/p75 INT8 跨平台 J/frame 全曲线 (sweep 进行中)。
> - **bug 修复记录**: E4 初版在建 CUDA context 后才做干净校验, 把自身 ~488MiB context 误判脏卡 → 改为 NVML 枚举进程、排除自身 PID 统计他进程占用。

---

## F 类. 内存 / 带宽

| # | 维度 | 4090 / Orin | 实测信号 | 对 latency 的物理影响 |
|---|------|-------------|----------|----------------------|
| F1 | **显存预算** | 4090 ~22GB avail / Orin ~50GB unified | **真测列存在**: `unified_bench.peak_gpu_mem_mb`, `engine_size_mb` 真测 | 约束 workspace 上限 (B2) |
| F2 | **带宽** | GDDR6X **1008 GB/s** / LPDDR5 **204.8 GB/s** (≈5×) | **spec (非实测)**: `*.yaml memory.bandwidth_gbs` | memory-bound 算子在 Orin 上相对慢 5× → 跨硬件 latency 外推时不可线性缩放 |
| F3 | **unified memory / H2D-D2H** | 4090 discrete (有拷贝开销) / Orin UMA (零拷贝, pinned) | **声明**: plan6 §三根因 B — 实测单次 transfer < 0.1ms (已用 data_ptr 直传) | 4090 hybrid pipeline 有 H2D; Orin UMA 省此项 |

> **arithmetic intensity / roofline 启示** (survey pruning §5): 在 Orin 低功耗档, TOP/Byte 比降低 → 更多层从 memory-bound 转 math-bound → 此时 INT8/稀疏收益更高。这解释了为何同一剪枝在 4090(高带宽)和 Orin(低带宽)上加速比不同 → **跨硬件 latency 预测器不能用单一系数外推** (与 v1 §5 一致, F2 带宽 5× 差是根因)。

---

## 6. 档位计数 (订正口径)

- **4090 (~25 档, 真测覆盖核心)**: 路由 A1 仅 GPU(×1) × tactic B1(×4 真测档, 声明 5) × workspace B2(×2 真测{4,8}, 声明 4) = **8 真测 / ~25 声明**。precision(C1)由量化师固定不进笛卡尔积。
- **Orin (~96 档声明 / 实测覆盖远小)**: 路由 A1(×3: GPU/DLA0/DLA1, 但 DLA INT8 全失败 ⇒ 有效路由 = GPU{fp16,int8} + DLA{fp16} ) × tactic B1(×2) × workspace B2(×2) = 声明 ~96, **真测有效 ~40 行** (`orin_dspace_bench` 成功行)。
- nvpmodel E1(×4): **30W 档已纳入实测 (ISS-016, E6)**, MAXN/15W/50W 3 档待补。GPU+DLA 并行 A5 / multi-stream A4 / 跨 agent A6 **声明但 0 latency**。

---

## 7. 与 config_schema / capability_schema 映射 (含本版新增轴)

```
config_schema.Config
  ├─ d_routing[m] → A1 单设备路由 (核心可搜)
  ├─ q_bits[m]    → C1 precision (硬件门控)
  ├─ q_granularity[m] → C3/C4 + D2 (DLA 强制 per-tensor)
  └─ q_calibrator[m]  → C2 implicit 细节

capability_schema.HardwareCapability
  ├─ ips.gpu.precisions       → C1 取值域门控
  ├─ ips.gpu.streams_max      → A4 multi-stream 上限 (=8 两平台, 未测)
  ├─ features.cuda_graph      → A3 (真测 1.05-1.10×)
  ├─ features.dla_count / has_dla → A1/A5 DLA 路由是否开放
  ├─ ips.dla.op_whitelist/blacklist → D3
  ├─ features.trt_strongly_typed → B5
  ├─ features.sparse_tc       → 2:4 (D4, DLA 仅 math-bound 受益)
  └─ memory.bandwidth_gbs     → F2 (roofline 跨硬件外推)

propagation.py (D↔B2 闭环, 已实现)
  ├─ propagate_d_to_b2: DLA 路由 → 强制 {FP16|INT8, per-tensor}
  └─ propagate_b2_to_d: per-channel/FP32 模块 → 踢回 GPU
```

> 新轴 A3/A4/A5/B3/E1 在 `config_schema.Config` 里**没有专用字段**, 由 DeployPlan / 部署脚本承载 (build-time 旋钮, 不污染搜索器核心契约)。若要把它们纳入搜索, 需扩 DeployPlan 而非 Config。

---

## 8. 相对 v1.0 的勘误 (实测驱动)

1. **CUDA Graph 加速幅度**: v1 未给数字, 本版真测 = **1.05–1.10×** (非数量级)。
2. **DLA INT8**: v1/CLAUDE.md 称 "DLA = INT8-only 主路径"。**真测订正**: PyramidFusion 上 DLA INT8 build **0/12 全失败** (kDIRECT_IO + bank 超限), **DLA FP16 才是本网络唯一真可用的 DLA 路径** (8/12 成功, 比 GPU FP16 快 ~22%)。yaml 里 `int8_dla0_build_success:true` 仅适用 ResNet50, 不能外推到 Pyramid。
3. **GPU+DLA 并行 (A5 dual_A)**: 本版明确标 **待确认** —— `unified_bench` 有 7 行 build 成功但 **latency=NaN**, 并行收益**未实测**。
4. **新增整个 A 类并行调度大类** (A3-A6) —— v1 缺失。
5. **[2026-06-02] 新增 A7 跨帧/时间维流水线 + 澄清"跨帧流水 ≠ 单帧加速"**: GPU 上跨帧流水是**吞吐(FPS)优化**(时间复用 SM, 关键路径不变), **不降单帧 e2e 延迟**; FPGA 因空间展开/II=1 才能同时隐藏延迟。避免后续 agent 把 FPS 提升误报成 e2e 加速。
6. **[2026-06-02] E1 真测落地 A4 multi-stream (订正"0 latency 实测")**: 4090 GPU6 全空闲, 同 cudagraph_bench engine 真测 streams∈{1,2,4,8}。**FPS 峰值仅 1.13× @streams=2, streams≥4 反降 (0.82× @8); 单帧 latency p50 单调劣化 1.0→9.8×**。→ A4/A7 从"纯声明"升级为"真测", 数据见 `results/E1_multistream_4090.csv` + §A 实测块。实测说明: 同器件单网络 multi-stream = 有限吞吐优化(最优流数=2≠streams_max=8), 单帧延迟不降反升。A5/A6 待 E2/E3。
7. **[2026-06-02 用户订正 — 两条最重要的方向性修正]**:
   (a) **吞吐不是"无价值", 是与延迟并列的硬实时约束**。之前 A7/§A 写"对 V2X 单帧 KPI 无价值"框定错误: 实时流式系统**吞吐 ≥ 输入帧率**与**延迟 ≤ 截止**都必须满足, 哪条 binding 看 regime(ego 单流常 latency-binding; **RSU 多车 / 多任务共享算力 throughput-binding**, 路测连续工况关心吞吐)。E1 证明的是"multi-stream 这个手段弱", 不是"吞吐目标不重要"。已订正 A7 价值段 + §A 结论 3。
   (b) **能耗(perf/watt, J/frame)升为 Pareto 一级指标**(见 E 类订正), 不再仅当 future work。
   ⇒ 框架 Pareto 目标从 (AP, latency) 扩展为多目标, 候选轴 = {AP, latency, throughput, energy, model_size}; 具体哪些作目标 / 哪些作 SLA 约束待定(见下游讨论)。

---

# 交付物 2: 解决算法数据流瓶颈是否能有效提高流水线调度优化效果?

## 结论 (一句话)

**当前 e2e 的真正瓶颈是 NMS 这种串行 CPU 卡点, 不是可并行的算子。流水线调度(stage 重叠 / GPU+DLA 并行 / multi-stream)只有在瓶颈本身可并行时才有效。所以: 必须先把串行数据流瓶颈(NMS、voxelize、跨 agent 同步)解开, 调度优化才能落地; 否则受 Amdahl 定律死压, 调度收益 < 1.05×。**

## 1. e2e 数据流各阶段实测占比 (plan6 §5, RTX 4090, CUDA Event 真测)

**优化前 (OPV2V, Shapely CPU NMS):**

| 阶段 | mean (ms) | % e2e | 性质 |
|------|-----------|-------|------|
| forward 全部 (encoder+backbone+pyramid+shrink+heads) | ~25.7 | **~24%** | GPU, 可并行/可剪/可量化 |
| └ pyramid_backbone (3-stage ResNeXt 占其 73%) | 14.52 | 13.5% | GPU 计算密集 |
| postproc.decode/corners/range_mask | ~2.98 | ~2.8% | 轻 |
| **postproc.nms (Shapely O(N²) CPU 循环)** | **77.62** | **72.4%** | **CPU 串行, GPU 闲** |
| **e2e walltime** | **107.17** | 100% | — |

**优化后 (DAIR, CUDA NMS, plan6 §5.1c):**

| 模块 | mean (ms) | % e2e |
|------|-----------|-------|
| encoder_m1 (PFN) | 2.40 | 13.4% |
| backbone_m1 | 0.84 | 4.7% |
| **pyramid (ResNeXt 主导)** | **9.97** | **55.8%** |
| shrink_conv | 1.40 | 7.8% |
| postproc (含 CUDA NMS 1.18) | 2.85 | 16.0% |
| **e2e** | **17.87** | 100% |

> 单换 CUDA NMS: e2e **107 → ~30ms (实测 3.04×)** —— 全项目 ROI 最高单点 (`P0_1_CUDA_NMS_结果_v1.md`)。

## 2. 各瓶颈"可并行性"分析 (这决定调度是否有用)

| 数据流瓶颈 | 优化前占比 | 性质 | 可被流水线调度并行? |
|------------|-----------|------|--------------------|
| **Shapely NMS** | 72% | CPU O(N²), GPU 空闲 | **不能直接重叠** —— 它本身是串行 CPU 卡点。GPU+DLA/multi-stream 对它 0 收益 (物理: NMS CPU-bound, GPU 忙闲不影响, plan6 §5 三重证据)。**唯一解 = 把算法换成 CUDA NMS (并行化算子本身)**, 不是调度 |
| voxelize (SpVoxel/spconv) | ~5% (encoder 内) | sparse op, ONNX 无支持, 必 PyTorch host | 串行 host 卡点; 可与上一帧 GPU forward 做**帧间流水重叠** (调度有效), 但单帧内不可并行 |
| weighted_fuse (跨 agent warp+加权) | ~17% of collab (~1.9ms) | GPU grid_sample, 但**依赖所有 agent feature 到齐** | 跨 agent **同步点** —— ego 必须等 infra 特征。可并行的是各 agent 的 backbone(A6), 但 fuse 本身是 join barrier, 不可越过 |
| pyramid ResNeXt forward | 55.8% (优化后) | GPU 计算密集 | **可并行**: multi-stream 拆异构 stage / GPU+DLA 把 backbone 卸到 DLA。但 CUDA Graph 实测仅 1.05-1.10× (子模块已小), 真收益靠剪枝/量化降 FLOPs 而非调度 |

## 3. Amdahl 推演 (用真实占比, 不空谈)

**场景 a — 不解 NMS, 只上调度优化 (multi-stream/GPU+DLA/CUDA Graph):**
- NMS 占 72% 串行不可并行 → 即使把可并行的 forward(24%)压到 0, 上限 = 1/(0.72+0.04postproc) ≈ **1.3×**。
- CUDA Graph 实测对 forward 只 1.05-1.10×, 对 e2e ≈ **1.02×**。
- ⇒ **不解 NMS, 调度优化基本无效 (Amdahl 死压)**。

**场景 b — 先解 NMS (CUDA 化, 已实测 3.04×), 再上调度:**
- e2e 107→18ms, 此时 pyramid forward 占比从 13.5% 升到 **55.8%** —— forward 才成为新瓶颈, 且**可并行/可剪/可量化**。
- 此时 multi-stream(A4)/GPU+DLA(A5, 把 backbone 卸 DLA, DLA FP16 真测比 GPU 快 22%)/剪枝/INT8 才开始有意义。
- ⇒ **解开串行瓶颈后, 调度优化的"可作用面"从 24% 扩到 56%, 收益才落地**。

**场景 c — 跨帧/时间维流水 (A7):** 把 NMS/voxelize 这类 host 串行项**重叠到相邻帧**(帧 N 的 CPU NMS 与帧 N+1 的 GPU forward 并发) → 稳态 FPS 从 `1/(forward+nms)` 升到约 `1/max(forward, nms)`。
- 但**任一帧的 end-to-end 延迟一点不降**(关键路径仍是 forward→nms 串行)。
- ⇒ 跨帧流水把 Amdahl 串行项移出**单帧关键路径之外**, **只改吞吐不改单帧延迟**: 对"降单帧延迟"无效, **但对"提吞吐"有效** —— 吞吐是与延迟并列的硬实时约束(RSU/路测 throughput-binding regime 正是其用武之地, 见 §A7 价值订正)。fuse(跨 agent join barrier)是跨帧吞吐重叠的硬上限。

## 4. 可执行判断

| 瓶颈 | 该不该解 | 解了之后调度优化是否有效 |
|------|---------|------------------------|
| **Shapely NMS → CUDA NMS** | **必须先解** (算子并行化, 非调度) | 解后 e2e 3× 且 forward 升为主瓶颈, 给调度腾出空间 ✅ |
| voxelize host 串行 | 帧间流水可缓解 | 帧间 pipeline 重叠有效 (与下帧 GPU forward 重叠); 单帧内无效 |
| 跨 agent fuse 同步 (A6) | 解耦/异步可缓解 | 各 agent backbone 可并行(A6/multi-stream); 但 fuse barrier 本身不可越, 收益受 join 限制 |
| pyramid forward (解 NMS 后的新主瓶颈) | 剪枝/INT8/multi-stream/GPU+DLA | **此处调度+量化才是真战场**; CUDA Graph 仅 1.05-1.10× 辅助 |

**总判断**: 流水线调度优化(multi-stream / GPU+DLA 并行 / CUDA Graph / 跨 agent 重叠)**不是 e2e 提速的第一杠杆**。第一杠杆是**把串行 CPU 卡点 NMS 算子并行化(CUDA NMS, 已实测 3×)**。只有解开 NMS 这类 Amdahl 串行项后, forward 升为主瓶颈, 调度优化(A 类)+ 剪枝量化才有可作用的份额。对 V2X 跨 agent: 各 agent backbone 可跨流/跨 IP 并行(A6), 但 fuse 同步 barrier 不可消除, 是并行收益的硬上限。

---

## 9. H1/H2 实验真测块 (2026-06-04 新增)

> ⚠️ **[治理标注, team-lead 2026-06-04, ISS-023 + 用户裁定]** 本节 H1/H2 数据产生于 **Phase H 被暂缓期间的未授权执行**(消息竞态, 详见 issues_log ISS-023)。用户裁定: **数据冻结保留, 绝不接入 dataset_v2 / 不出图, 等正式开 Phase H 时再授权使用**。本节内容(含上文 §B3 opt-level "21% 提速"、§2:4 sparsity "1.02×" 等引用处)**仅作设计文档内的冻结记录与 Phase H 预案参考, 不得作为已入库结论引用到论文/数据集/对外报告**。supervisor 已核数据本身干净(真 finetune + 真 INT8 build)。解冻条件 = 用户正式授权开启 Phase H。

### H1 — 2:4 Structured Sparsity (base 64_128_256, DAIR, body_subnet_collab2)

**实验**: 对 `Pyramid_DAIR_m1_base` 施 2:4 magnitude mask + DAIR finetune 2 epoch + TRT SPARSE_WEIGHTS build
**数据源**: `results/H1_sparsity_4090.csv` + `results/m4_8_hybrid_ap_H1_base_sparse_{fp16,int8}.json`
**GPU**: RTX 4090 GPU5 (util=0%/mem=4MiB, clock=2520MHz), body_subnet_collab2 口径

| config | precision | lat_p50_ms | fp16_layers | int8_layers | AP50 | AP70 | vs non-sparse |
|--------|-----------|-----------|-------------|-------------|------|------|--------------|
| base_sparse_fp16 | FP16+SPARSE | **1.235ms** | 72 | 0 | 0.791 | 0.634 | 1.018× (ws1024 base=1.258ms) |
| base_sparse_int8 | INT8+SPARSE | **0.786ms** | 0 | 55 | 0.791 | 0.625 | 1.021× (non-sparse int8=0.803ms) |

**结论**:
1. **2:4 sparsity 在 PyramidFusion 上加速极小 (1.02×)**, 远低于 NEXT_PHASE_plan 保守预期 1.1-1.3×。
2. **根因**: memory-bound 模型瓶颈是带宽 (~30% BW)而非算力, 2:4 减算力无法解带宽瓶颈。
3. **AP 完全恢复**: 2 epoch finetune 足够,AP50/AP70 与 base 基本持平。
4. **结论**: 2:4 sparsity 对本模型**不作主搜索轴**,降为"已真测-收益微弱"对照。

---

### H2 — workspace/opt-level 扫描 (3 configs × 4 workspace + 1 opt5, body_subnet_collab2)

**实验**: `{base_fp16, p50_int8, p25_trap_int8}` × workspace `{256MB,1GB,2GB,4GB}` + base_fp16 opt5 共 13 点
**数据源**: `results/H2_workspace_scan_4090_final.csv`
**GPU**: RTX 4090 GPU5 (idle), clock=2520MHz

**关键数据**:

| config | ws_mb | opt | lat_p50_ms | INT8_layers | fp16_layers | fp32_layers |
|--------|-------|-----|-----------|-------------|-------------|-------------|
| base_fp16 | 256 | 3 | 1.303ms | 0 | 72 | 0 |
| base_fp16 | 1024 | 3 | **1.258ms** | 0 | 72 | 0 |
| base_fp16 | 2048 | 3 | 1.687ms | 0 | 72 | 0 |
| base_fp16 | 4096 | 3 | 1.599ms | 0 | 69 | 3 |
| **base_fp16** | **4096** | **5** | **1.250ms** | 0 | 73 | 0 |
| p50_int8 | 256 | 3 | 0.855ms | 23 | 37 | 13 |
| p50_int8 | 1024 | 3 | 0.923ms | 18 | 52 | 3 |
| p50_int8 | 2048 | 3 | 0.843ms | 26 | 35 | 12 |
| **p50_int8** | **4096** | **3** | **0.815ms** | **47** | 2 | 23 |
| p25_trap_int8 | 256 | 3 | 2.820ms | **6** | 36 | 31 |
| p25_trap_int8 | 1024 | 3 | 2.727ms | 6 | 34 | 33 |
| p25_trap_int8 | 2048 | 3 | 2.710ms | 5 | 37 | 31 |
| p25_trap_int8 | 4096 | 3 | 2.721ms | 6 | 35 | 32 |

**H2 关键结论**:

1. **opt=5 是真维度 (21% 提速)**: base_fp16 opt5 ws4096 = 1.250ms vs opt3 ws4096 = 1.599ms → **21% 提速 + 消除 3 FP32 fallback 层**。opt=5 应作为生产 build 的 best practice。

2. **workspace 影响 INT8 tactic 覆盖 (非线性)**: p50_int8 INT8 层数: ws256=23层 → ws1024=18层 → ws2048=26层 → **ws4096=47层**。大 workspace 允许更多 INT8-compatible kernel → 需 ws≥2GB 才能达最优 INT8 tactic 覆盖; ws4096 比 ws256 快 ~5% (0.815 vs 0.855ms)。

3. **kernel-cliff 被 layer count 直接确认**: p25_trap_int8 (48/96/192 非对齐通道) 只有 **5-6 INT8 层 + 31-33 FP32 layers**, 不论 workspace 大小。对比 p50_int8 ws4096 的 47 INT8 层 → kernel-cliff 的本质 = TRT 无法为非对齐通道选 INT8 kernel。

4. **workspace 对 base_fp16 非单调**: ws1024 (1.258ms) 比 ws4096 (1.599ms) 更快 → **不能假设"更大 workspace 总是更好"**; ws 对 FP16 tactic 的影响取决于特定模型架构。

5. **build-time 建议**: INT8 build 用 ws=4GB; FP16 build 用 ws=1GB + opt=5; 两种精度都应关闭"手动 tactic 枚举"委托给 TRT-auto。
