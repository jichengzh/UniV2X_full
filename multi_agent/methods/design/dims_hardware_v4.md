# dims_hardware_v4 — route2 / TVM 调度可搜硬件维度 + 实验计划 + 集成闸门 (v1, 2026-06-17)

> 本文 = route2(以 TVM 把编译调度变成"可搜硬件轴"挣 co-design 标签)的**完整作战计划**:
> ① 选哪些 TVM 调度维度去证明软硬耦合, ② 每个维度从体系结构原理上与软件(剪枝×量化)的耦合关系 + 针对的耦合陷阱 + **文献来源**, ③ 针对什么模型做实验, ④ 整个实验流程 + 决定"是否把 TVM 集成进我们软硬件协同优化框架"必须先过的 smoke 闸门, ⑤ 集成判定标准(含负结果路径)。
>
> **定位 (v4 vs v2)**: `dims_hardware_v2.md` 定义的是 **TRT 上的 D 维**(引擎路由/精度/workspace/DLA), 已证在固定硅片上几乎全塌缩成 TRT-auto。v4 是其延伸: 当 TRT 太薄(已证)时, **改用 TVM 把 tile/layout/tensorize/fusion 这些 TRT 不暴露的调度旋钮变成显式可搜的硬件维**。两文互补, 不重复。
> 纪律: [文献声称](WebSearch/原文核) / [我方实证](本项目真测) / [启发](推断, 明确标注) 严格分层。
> 关联: [survey_compiler_schedule_search_v1.md](../references/survey_compiler_schedule_search_v1.md) · [survey_hwsw_codesign_v1.md](../references/survey_hwsw_codesign_v1.md) · [HANDOFF_tvm_migration_route2_v1.md](../progress/HANDOFF_tvm_migration_route2_v1.md) · memory `project-hwsw-codesign-route2`

---

## §0 一句话目标

证明命题: **"把编译调度(tensorize/layout/tile/fusion)当成与剪枝率×量化档联合搜索的硬件维, 能发现并缓解 TRT-auto 看不见、抹不平的多个软硬耦合陷阱。"** 通过一串 go/no-go smoke 验证此命题成立(→ 把 TVM 集成进框架)或被证伪(→ 诚实记负结果, 退回 route1)。

---

## §1 出发点 (已证事实, 非推断)

[我方实证] **TRT 太薄已证** (S0/S0b, `results/S0b_coupling_clean_4090.csv`): 固定硅片上 D 维几乎全塌缩成 TRT-auto, 唯一立得住的耦合只剩"通道对齐×INT8"(trap25 48_96_192 非÷32 → fp16 比 p50 慢 2.9× / INT8 仅 1.08×, 跨 build 旋钮稳定)。但 S0b 是**固定 HW 只变软件**, 只证"耦合存在", 未证"硬件维度被协同地配置"。

[我方实证] **TVM 工具链已通** (S2.0/S2.1/S2.2a, 2026-06-17, H800): TVM 0.20 relax 导入 Pyramid backbone + 数值对齐(maxdiff 2.4e-6) + MetaSchedule 全链路(21 tasks→tune→compile→run) + INT8 tensor-core tensorize 可表达且实测真启用(scheduled-TIR `tvm_mma_sync` 复核, `results/H800_s2_2_int8_*.log`)。

[我方实证 + 文献核 ★关键发现] **对齐悬崖是"路径依赖"的, 而 intrinsic 粒度本身是可搜的硬件映射决策**:
- [文献声称] **TRT 的 INT8 走 NCHW32 向量化格式**(32 通道打包, 内存布局 `[N][(C+31)/32][H][W][32]`, 通道按 32 对齐填充; 且为 DLA INT8 的 native feature format)。(NVIDIA TensorRT Data Format docs) ⇒ 这正是 S0b ÷32 悬崖的根因: TRT INT8 内核锁死在 32 通道对齐, 48 通道被迫 pad/fallback。
- [文献声称] **CUDA WMMA int8 支持 m16n16k16 / m8n32k16 / m32n8k16(K=16, ÷16 对齐)**; **PTX mma.sync int8 支持 m16n8k32(K=32, ÷32)**。(CUDA C Programming Guide / NVIDIA Developer Forums) ⇒ 两条 tensor-core 路径的对齐粒度不同。
- [我方实证] S2.2a 合成 dense int8 1×1 conv 三臂(Cin=64/48/48→pad64)tune 后**全部真上 WMMA(16×16×16, K=16)** tensor-core(TIR builtin 复核); Cin=48 因 48÷16=3 对齐 → **在 WMMA 路径上无悬崖**。
- [启发] ⇒ 同一个"48 通道", 在 TRT 的 NCHW32(÷32)路径上撞墙, 在 TVM 默认选的 WMMA(÷16)路径上不撞。**对齐悬崖不是物理铁律, 而是 intrinsic/layout 映射选择的函数** —— TRT 把这个选择固定死, TVM 把它暴露成可搜轴。这就是 route2 的技术核心抓手。
- ⚠️ [诚实边界] S2.2a 是**合成 dense conv**, 未含 trap25 真实的 **grouped conv(groups=32)** 约束(groups=32 要求通道 ÷32, 与 tensor-core 的 K 对齐是两个不同机制); 故尚未在真实 backbone 上复现 S0b 悬崖 → 见 §4 S2.2b。

---

## §2 被搜的 TVM 调度维度 (逐维: 原理耦合 + 陷阱 + 文献源)

> 选维原则: ① 与软件维(剪枝率/量化档)有体系结构层面的真耦合; ② TRT 不暴露该旋钮(否则 TVM 无增量); ③ 有文献先例支撑机理与陷阱形态; ④ 我方有实测锚或素材就绪。
> 维度来自 HANDOFF List B(B1–B7)的精炼 + 文献加固 + `s_tir.Schedule` 全原语自省(2026-06-17)。
>
> **★维度按"贴硬件程度 × 与剪枝/量化耦合强度"分三层(整合 v4 补充, 2026-06-17, 把底层与中层做明确区分)**:
> - **T1 ISA / 内存层级原语(最底层, 耦合最深, TRT 完全不暴露)** = 主攻层: D1 tensorize/intrinsic + L1–L5(内存暂存 / 线程绑定 / bank-conflict / 软件流水 / 跨线程 reduction)。
> - **T2 中层结构原语** = D2 layout+pad / D3 tiling-loop。
> - **T3 图级原语** = D4 fusion×Q/DQ。
> 为何这样分: 剪枝改变的通道宽度**同时流入** tile 填充率、SMEM staging 效率、bank 冲突模式、reduction 宽度、occupancy —— 一处软件决策牵动整条硬件执行链, 故 T1 是耦合最深、最该主攻的一层(这层正是 v4 初稿偏中层时漏列的)。
>
> **★关键勘误(整合 v4 补充)— 这些原语是"跨编译器共享的调度搜索范式", 不都是 TVM**: TVM 系 = Ansor(=TVM auto-scheduler)/AMOS(扩展 TVM)/Bolt(TVM 前端+CUTLASS)/CHaNAS(TVM auto code opt); **非 TVM** = Roller(nnfusion)/Hidet(独立编译器)/FAST(Google 加速器栈); ALT 自有系统(对比 Ansor/TVM)。我们选 TVM 是因它**把这批原语(含 T1 最底层的)都暴露且开源可改**, 不是"论文都用 TVM"。论文可移植的是**原语层面的机理**(对齐 / staging / intrinsic 映射), 非具体框架。
>
> **★实操澄清**: T1 的 L1–L5 在 MetaSchedule 里**由 schedule rules 自动生成、由搜索采样参数**(`MultiLevelTilingTensorCore` 自动插 tensorize+cache_read→fragment+软件流水; `AutoBind` 自动 bind+vthread; `sample_perfect_tile`/`sample_categorical` 采样因子), 我们能配置的**维度 = 哪些规则进 space generator + 各自参数范围**; 同时 `s_tir.Schedule` 支持**手工逐原语构造**(Roller 式确定性 aligned-vs-misaligned 对照)。两模式: 机理/陷阱复现走手工构造(确定、便宜), 可搜性/Pareto 走规则采样(autotune)。"够不够底层"的回答 = 底层原语都在规则展开里被搜到。

---

### 【T1】ISA / 内存层级原语 — 最贴硬件, 耦合最深, TRT 完全不暴露 ★主攻层

#### TVM-D1 ★ Tensorize / MMA intrinsic 选择 (WMMA K16 ↔ PTX-MMA K32 ↔ DP4A ↔ 标量)

- **可搜内容**: 一个层映射到哪条 tensor-core 路径(WMMA 16×16×16 / mma.sync 16×8×32 / dp4a / 无 TC 标量), 由 `s_tir.Schedule.tensorize` + intrinsic 选择决定。
- **与软件的耦合(原理)**: **剪枝宽度(通道数) × 量化档(精度) × intrinsic 三方耦合**。层能否上某条 TC 路径, 取决于其 GEMM 维度(M=spatial, N=Cout, K=Cin×kh×kw)是否整除该 intrinsic 的 tile 形状; 而 N=Cout、K=Cin 正是剪枝直接决定的量, 精度(INT8/FP16)决定可用 intrinsic 集合(INT8 用 i8 MMA, FP16 用 f16 MMA)。[原理: §1.5 P3 kernel/tile 粒度离散性]
- **针对的耦合陷阱**: 剪枝到某宽度(如 48)后, 在**粒度更粗的 intrinsic(NCHW32/IMMA, ÷32)上失配 → fallback → 丢失量化加速**(=S0b trap25 1.08×)。陷阱形态 = "欠剪但失配 远差于 激进但对齐"(延迟+可量化性双输)。
- **TVM 能做、TRT 做不到**: TVM 把 intrinsic 选择暴露为可搜——失配宽度可改走**粒度更细的 WMMA(÷16)**绕过 ÷32 墙, 或 pad-to-fit(配合 D2)。TRT INT8 锁死 NCHW32(÷32), 撞了只能 fallback。
- **文献源**:
  - [文献声称] **AMOS (ISCA'22)**: 用硬件抽象把张量计算**经 intrinsic(CUDA WMMA / PTX MMA)自动映射到 tensor core**, TC 上 2.50× over 手优库 — 证明 intrinsic 映射是可自动搜索/生成的轴。(ZhengETAL22AMOS)
  - [文献声称] **Bolt (MLSys'22)**: BYOC + CUTLASS 模板化 TC GEMM, hardware-native 模板搜索 2.5× over auto-tuner — 证明 TC 模板/intrinsic 选择可与编译栈联合。
  - [文献声称] **Roller (OSDI'22)**: rTile = 与硬件 MMA 单元对齐的 tile 抽象; 对齐是第一性设计原则。trap25 = rTile 失配实例。
  - [文献声称] WMMA int8 = m16n16k16(K16) / PTX mma int8 = m16n8k32(K32)(CUDA docs); TRT INT8 = NCHW32(÷32, NVIDIA docs)。
  - [我方实证] S0b trap25(÷32 失配 1.08×) + S2.2a(WMMA K16 下 48 不失配, TIR builtin 复核)。

#### TVM-L1 内存层级暂存 (`set_scope` / `cache_read`→shared/register/`wmma.matrix_a/b`)

- **可搜内容**: 显式管理 global→L2→shared(SMEM)→register→TC fragment 的数据搬运路径与暂存粒度。
- **与软件的耦合(原理)**: **剪枝**→通道窄→单 tile 的 SMEM/寄存器占用变→改变 occupancy(SMEM/寄存器是 occupancy 限制资源); **量化**→位宽减→SMEM 等效容量翻倍(INT8 vs FP16)→最优 staging 深度变。[原理: §1.5 P1 Roofline / P2 Horowitz 搬运能耗主体]
- **针对的耦合陷阱**: 剪枝后 staging 不变 → SMEM/寄存器浪费或 occupancy 掉; 错误 cache 放置 → 撞 L2/DRAM 带宽墙(=E_pipeline 1.08× 同源: occupancy 余量 ≠ 可用吞吐)。
- **TVM 能做、TRT 做不到**: TVM 暴露内存暂存放置; TRT 不暴露。**(原 v4 D5 cache/compute_at 已并入本维)**
- **文献源**: [文献声称] Ansor "Add cache_read" rule(local→shared, 多线程并行取数, OSDI'20 §4.1); AMOS memory mapping; [我方实证] E_pipeline/E_headroom(`results/E_pipeline_singlegpu_4090.csv`)。

#### TVM-L2 线程/块绑定 (`bind` blockIdx / threadIdx / **virtual-thread**)

- **可搜内容**: 把循环显式映射到 SM/warp 网格; virtual-thread 错开访问减 bank conflict。
- **与软件的耦合(原理)**: **per-stage 剪枝**改变各 stage 计算量 → 线程网格最优配置变; 量化改变每线程处理元素数。
- **针对的耦合陷阱**: 固定 binding 在剪枝后 → SM 负载不均 / occupancy 不足。
- **TVM 能做、TRT 做不到**: TVM 显式搜 bind 网格; TRT 不暴露。
- **文献源**: [文献声称] Ansor GPU sketch = **SSSRRSRS, 前三层空间 tile 分别 bind 到 BlockIdx / virtual-thread(减 bank conflict) / ThreadIdx**(OSDI'20 §4.1 原文核); v4 List B7。

#### TVM-L3 Shared-memory bank-conflict padding (`storage_align`)

- **可搜内容**: 给 SMEM buffer 加 padding 避免 32-way bank conflict。
- **与软件的耦合(原理)**: 通道宽度(剪枝)/位宽(量化)直接决定 SMEM 访问的 bank 冲突模式。
- **针对的耦合陷阱**: 剪枝到某宽度恰好触发 bank conflict → SMEM 吞吐暴跌。**比 ÷32 对齐更隐蔽的"隐形悬崖"**(规则不可预判, 必须真测 = §1.5 P3 粒度离散的更深层例子)。
- **TVM 能做、TRT 做不到**: TVM 显式控制 SMEM padding; TRT 不暴露。
- **文献源**: [文献声称] Ansor virtual-thread for bank conflict(原文核); CUDA C Best Practices(Shared Memory Bank Conflicts)。

#### TVM-L4 软件流水 / 双缓冲 (经 `annotate` 的 software_pipeline)

- **可搜内容**: 把 global load 与 compute 重叠(prefetch 下一 tile 同时算当前)隐藏访存延迟; 流水深度。
- **与软件的耦合(原理)**: **剪枝**→更 memory-bound→流水隐藏访存的收益更关键; **量化**→访存量变→最优流水深度变。[原理: §1.5 P1 Roofline]
- **针对的耦合陷阱**: memory-bound kernel(剪枝后更甚)不做流水 → 访存延迟暴露, 量化/剪枝收益蒸发; Hopper TC GEMM 达峰需配合 MMA tensorize。
- **TVM 能做、TRT 做不到**: TVM 暴露流水 annotation; TRT 不暴露。
- **文献源**: [文献声称] CUTLASS multistage / Bolt 持久化模板流水; MetaSchedule TC 规则内含 software-pipeline postproc。

#### TVM-L5 跨线程 reduction (`rfactor` / `decompose_reduction`)

- **可搜内容**: reduction 维跨线程分解 + warp shuffle/atomic。
- **与软件的耦合(原理)**: 收缩维 **K=Cin×kh×kw(剪枝直接决定)** + 是否够并行 → reduction 策略。
- **针对的耦合陷阱**: 窄 reduction 维(剪枝过度)→ 并行不足 / 同步开销主导。
- **TVM 能做、TRT 做不到**: TVM 显式搜跨线程 reduction; TRT 不暴露。
- **文献源**: [文献声称] Ansor "Cross thread reduction" rule(OSDI'20 §4.1 原文核)。

> **T1 文献佐证**: Ansor 真实 GPU 搜索空间就在 L1–L5 这一层 —— 多级 tiling **SSSRRSRS** + `cache_read`→shared(L1) + bind BlockIdx/**vthread 防 bank conflict**(L2/L3) + cross-thread reduction(L5)(OSDI'20 §4.1 原文核)。证明案例论文确实搜到这一最底层, T1 不是我们臆造的"够底层"包装。

---

### 【T2】中层结构原语 — 数据打包 + 循环结构

#### TVM-D2 ★ Data layout (NCHW / NHWC / NCHWc / NC32HW32) + repack/pad 联合搜

- **可搜内容**: 张量打包布局 + 是否插入 repack/pad pass(`relax.transform.ConvertLayout` / `s_tir.Schedule.transform_layout` / `pad_einsum`)。
- **与软件的耦合(原理)**: A1/D1 的"对齐陷阱"本质 = **layout 打包粒度与剪枝后通道数的整除关系**。最优 layout 随剪枝宽度×精度变(INT8 偏好 32 通道向量化, FP16 偏好 16); 失配宽度需 repack/pad 才能填满向量道。
- **针对的耦合陷阱**: 失配通道(48)在固定向量化布局(NCHW32)下被 pad 或 fallback, TRT 不暴露 pad 决策 → 撞墙。**缓解动作 = pad 48→64 / repack, 用 FLOPs 换对齐恢复 TC 资格**。这是"硬件维度被协同地配置"的真 demo(正面回应"S0b 只证耦合存在"的质疑)。
- **TVM 能做、TRT 做不到**: TVM 把 **layout + pad 提进搜索空间, 与 loop/precision 联合搜**; TRT auto 选 layout, 撞了只能 fallback, 无 pad 旋钮可暴露给搜索器。
- **文献源**:
  - [文献声称] **ALT (EuroSys'23)**: 打破"先定 layout 再调 loop", **联合搜 layout+loop**, 单算子平均 1.5× over Ansor — 直接背书"layout 必须进搜索空间, 固定 layout(=TRT/Ansor)不够"。
  - [文献声称] **FAST (ASPLOS'22, Google)**: 把 **tensor padding 当显式编译器搜索决策**(与 datapath 平级)— 即我们 pad 48→64 的文献依据。
  - [文献声称] TRT INT8 NCHW32 固定布局(NVIDIA docs); Roller rTile(对齐=构造原则)。
  - [我方实证] S0b 对齐悬崖; S2.2a pad 臂已验证可构造(48→64 eff)。

#### TVM-D3 Tiling / loop-order (tile 形状 ↔ 通道数 × MMA 形状)

- **可搜内容**: 各 loop 的 split/reorder/tile 因子(`s_tir.Schedule.split/reorder`)。
- **与软件的耦合(原理)**: 最优 tile 随**通道数(剪枝)**与 **MMA 形状(精度)**变; 剪枝后的小通道在 TC fragment 里留碎片 → 占用率(occupancy)与有效算力下降。[原理: P1 Roofline / P3 粒度离散]
- **针对的耦合陷阱**: 剪枝把通道压到不能填满 tile/warp → tensor-core 道浪费, 加速不及预期(剪枝宽度×tile 占用率耦合)。
- **TVM 能做、TRT 做不到**: TVM 显式搜 tile 因子; TRT 内部定 tile, 不暴露。
- **文献源**: [文献声称] **Ansor (OSDI'20)**(分层搜 tiling/loop, GPU 1.7× over AutoTVM); **Roller**(rTile 构造式定 tile); **TLP (ASPLOS'23)**(调度原语即特征的代价模型, 搜索提速 GPU 3.0×, 留 S3 预测器用)。

---

### 【T3】图级原语 — 算子图融合

#### TVM-D4 [探索] Operator fusion × 量化 Q/DQ 放置

- **可搜内容**: 融合分组 + Q/DQ 节点放置(`relax.transform.FuseOps/FuseOpsByPattern`)。
- **与软件的耦合(原理)**: Q/DQ(量化粒度决策)打断融合; 最优融合分组取决于量化边界 × 剪枝后残差结构。[原理: §1.5 P4 编译器映射不可交换 — Conv+BN+ReLU 融合被 Q/DQ 破坏 → 多一次 activation 访存]
- **针对的耦合陷阱**: 逐 stage 混精/量化插入 Q/DQ → 破坏融合 → 访存流量增加, 抵消量化收益(我方 P4 已观察 TRT 上此现象)。
- **TVM 能做、TRT 做不到**: TVM 可显式控制融合分组与 Q/DQ 放置并联合搜; TRT auto-fuse 不可覆盖。
- **文献源**: [我方实证] §1.5 P4(TRT 融合, `data/tactic_workspace_bench.csv`); [文献声称] FAST(算子融合作搜索 pass); TVM FuseOps。⚠️ 此维需先解决 relax 无现成 INT8 量化 pass 的工程前提(见 §3)。

---

### 维度汇总表 (按层 T1/T2/T3)

| 层 | 维度 | TVM 旋钮 | 与软件耦合 | 陷阱(TRT 上的表现) | TRT 能否搜 | 文献锚 | 我方锚 | 优先级 |
|---|---|---|---|---|---|---|---|---|
| **T1** | **D1 tensorize/intrinsic** | `tensorize` + intrinsic 选 | 剪枝宽度×精度×intrinsic 粒度 | 失配宽度 fallback 丢 INT8 加速(S0b 1.08×) | ❌(锁 NCHW32) | AMOS/Bolt/Roller/CUDA-WMMA/TRT-docs | S0b trap25 + S2.2a | ★主 |
| **T1** | **L1 内存暂存** | `set_scope`/`cache_read`→shared/reg/frag | 剪枝→occupancy×量化→SMEM 容量 | staging 浪费 / 撞带宽墙(E_pipeline) | ❌(不暴露) | Ansor/AMOS | E_headroom | ★主 |
| **T1** | **L2 线程绑定** | `bind` blockIdx/threadIdx/vthread | per-stage 剪枝×SM 网格 | SM 负载不均 / occupancy 不足 | ❌(不暴露) | Ansor(SSSRRSRS) | (待 S2.2b) | ★主 |
| **T1** | **L3 bank-conflict** | `storage_align` | 通道宽度×SMEM bank 冲突 | 隐形悬崖: SMEM 吞吐暴跌 | ❌(不暴露) | Ansor/CUDA-BestPractices | (待 S2.2b) | ★主 |
| **T1** | **L4 软件流水** | `annotate` software_pipeline | 剪枝→更 memory-bound×流水深度 | 不流水→访存延迟暴露 | ❌(不暴露) | CUTLASS/Bolt | (待 S2.2b) | 次 |
| **T1** | **L5 跨线程 reduction** | `rfactor`/`decompose_reduction` | 收缩维 K=Cin(剪枝定)×并行度 | 窄 reduction→并行不足 | ❌(不暴露) | Ansor | (待 S2.2b) | 次 |
| **T2** | **D2 layout+pad** | `ConvertLayout`/`transform_layout`/`pad_einsum` | 剪枝宽度×精度×打包粒度 | 失配通道 pad/fallback, 无 pad 旋钮 | ❌(auto 选 layout) | ALT/FAST/Roller/TRT-docs | S0b + S2.2a pad 臂 | ★主(缓解 demo) |
| **T2** | D3 tiling/loop | `split`/`reorder` | 剪枝通道数×MMA tile 占用 | 小通道填不满 tile → TC 碎片 | ❌(内部定 tile) | Ansor/Roller/TLP | (待 S2.2b) | 次 |
| **T3** | D4 fusion×Q/DQ | `FuseOps` + Q/DQ 放置 | 量化边界×剪枝残差×融合 | Q/DQ 破融合 → 多访存(P4) | ❌(auto-fuse) | FAST/TVM/我方 P4 | P4 | 探索 |

> 注: 原 v4 D5(cache/compute_at)已并入 **T1-L1**(同属内存层级暂存)。

> **★S2.2c 实测每旋钮延迟效应(2026-06-18, H800, 确定性手写 schedule 消融, 详见 HANDOFF §2.5c)**: **D1 tensorize 2.3×**(TC vs 非TC, conv128 int8) / **L1 shared 暂存 1.46×**(GEMM2048³, naive→+shared) / **L2 线程绑定 = GPU 必需**(无绑定不可运行) / **L5 跨线程 reduction 67×**(reduction-heavy GEMV)。⇒ **4/6 T1 旋钮单独施加即对延迟有可测显著影响**, 证实这些维度是真旋钮。**L3 storage_align / L4 software_pipeline = 参数依赖**(手设参数未显效, 收益需 per-knob 调参; autotuner 会选)。方法学: rule-drop 消融被搜索预算混淆(弃), 手写消融需 `tirx.is_scheduled` attr 绕过 dlight + shared 必配 cooperative-fetch。T2/T3(D2/D3/D4)待做。

---

## §3 实验模型与口径

- **主目标函数 = PyramidFusion backbone 子图(DAIR 金标准)**: 对齐 **p50(32_64_128÷32)** vs 失配 **trap25(48_96_192✗÷32)**, 两者已 S2.1 数值对齐 PASS(`models/stage_a_cache/{p50,trap25}_backbone.onnx`)。这是 S0b 的同一对模型, 直接对齐已证 TRT 悬崖。
- **忠实合成单 conv(补 INT8 量化缺口)**: relax 0.20 **无现成 INT8 量化 pass**(`ToMixedPrecision` 仅 fp16), 全模型 INT8 受阻。⇒ 用 `BlockBuilder` 按 backbone **真实 conv 形状(含 grouped conv groups=32, 1×1 与 3×3)**逐 conv 建 int8 合成, tune 后查 per-conv TC 命中。这是绕过量化 pass 缺口、又忠实于真实结构的折中(S2.2a 已验证合成 int8 conv 工具链通)。
- **第二目标函数 = CoDriving**(用户此前要求并行推进; H800 已有剪枝×量化 pilot 副本): S2.3+ 把同套 D1/D2 实验复用到 CoDriving backbone, 验证耦合机理跨模型成立(非 Pyramid 特例)。
- **硬件口径**:
  - **机理(TC 是否启用 / intrinsic 选择 / 调度 trace)= H800 跑**(架构无关, GPU6/7; 对脏 GPU 噪声稳健)。
  - **延迟 Pareto = 4090 / Orin 跑**(边缘标的; 必须空闲 GPU util≤2%/foreign≤50MiB 实测)。
  - latency 口径标注 sub-module/e2e; 区分 真测/估算/仅声明。

---

## §4 实验流程 — smoke 闸门阶梯 + 逐维度耦合验证协议 (最终版)

> 闸门哲学: 最低成本先验, 高不确定性的 go/no-go 单 agent 串行不 fan-out; 任一关红灯则停并回报。
> **核心动作 = 逐维度隔离测**(§4.2): 每个 §2 的可调控维度(D1/L1–L5/D2/D3/D4), 控制软件变量(剪枝宽度/量化档), 测该维度对应的**调度产物**随软件变量的变化, 判定耦合是否成立。

### §4.1 smoke 闸门阶梯

| 关 | 名称 | 做什么 | go/no-go 判据 | 状态 |
|---|---|---|---|---|
| **S2.0** | 工具链 go/no-go | TVM-on-H800 env + MetaSchedule 全链路跑通 | tune→compile→run 成功 + INT8 tensorize 可表达 | ✅ **PASS**(2026-06-17) |
| **S2.1** | 导入+数值对齐 | relax 导入 backbone, vs ORT 对齐 | maxdiff<1e-3 | ✅ **PASS**(2.4e-6) |
| **S2.2a** | INT8 TC 启用核验 | 合成 int8 conv tune, TIR builtin 复核真上 TC | TC 真启用(非关键词假阳) | ✅ **PASS** + 得 WMMA-K16 发现 |
| **S2.2b** | **耦合发现 screening**(★全维度而非单陷阱) | 按 §4.3 campaign: 对代表性 conv 扫 **软件轴 S1 宽度 × S3 精度** 网格(故意覆盖 ÷32/仅÷16/仅÷8/奇质数边界), 每点记 **延迟 + 全维度调度产物**(D1/L1–L5/D2/D3/D4), 用信号探测器筛候选 | 产出"维度×软件轴→{有耦合/无/噪声}"候选表; 至少把全 9 维各扫到(负结果也记) | ✅ **DONE**(2026-06-17, 39 tune 点): dense 1×1 全宽度无 D1 悬崖(R2 确认, TVM-WMMA-K16 消解 TRT ÷32 悬崖); L3/L4 平坦负结果; L2-grid/延迟=跨seed证实噪声③; **grouped g=32 全掉 SCALAR=候选耦合#1(层结构×量化, 非TVM专属)**。详见 HANDOFF §2.5 |
| **S2.2c** | **逐维度深探 + 新颖性分类** | 对 screening 标出的每个候选, 按 §4.2 协议隔离深探(固定其余、强制该维度选择、跨 ≥2 配置验稳、给体系结构解释); 三分类: 已知对齐重现 / **新耦合** / auto 噪声 | 确认 **≥1 个"新耦合"**(非 D1/D2 对齐), 跨配置稳定 + 有机理; 或诚实记"除对齐外无新耦合" | ✅ **DONE(诚实负)**: grouped(#1)深探+控制隔离(g=1 3×3=WMMA 证 grouping 是因)但**非 TVM 专属**; dense 上"除对齐外无新耦合"成立; force-K32 反事实 blocked(工具链)。⇒ **判据1(新耦合)未达** |
| **S2.2d** | **缓解 demo("硬件被协同配置")** | 对确认的耦合: ① 串行 vs ② 协同(配置对应维度旋钮把劣势臂救回); 出 调度产物 + 延迟对比 | **协同配置能逆转劣势**, 且该旋钮 TRT 不暴露 | 🔶 部分 DONE(选项①②): ①真实 trap conv 16/16 全留 WMMA(TVM intrinsic 粒度旋钮消解 TRT ÷32 悬崖); ②Q/DQ 被 fuse 进 epilogue(TVM 融合消解 TRT P4 断裂)。⇒ **判据2(缓解)强支持**。绝对边缘延迟坐实(4090/Orin)留 S2.3 |
| **S2.3** | 干净 GPU 延迟 + 跨模型 | S2.2b–d 的 demo 点在 **4090/Orin 空闲实测延迟**; 同套 campaign 复用到 **CoDriving** | 延迟数 std 小/口径一致; 耦合机理跨模型复现(非 Pyramid 特例) | ⬜ |
| **S2.4** | **联合搜可行性 + 集成判定** | 把 剪枝率×量化档×{已证耦合维度子集} 做小规模联合搜(笛卡尔或进化), 看搜索空间是否可处理 + Pareto 是否扩张 | 见 §5 四条判定标准 | ⬜ |

### §4.2 逐维度耦合验证协议 (S2.2b 的具体做法 — 每维度怎么测耦合)

> 通用范式(对每个维度): **固定 conv 其余参数 → 变软件变量(剪枝宽度: 对齐 p50-类 64 vs 失配 trap25-类 48; 或量化档 INT8/FP16)→ 抽该维度对应的调度产物 → 看产物是否随软件变量变 → 跨配置验稳定**。
> **两类测量, 优先静态**: (a) **静态**=从 scheduled-TIR 读(SMEM bytes / bind extent / storage_align factor / 流水 stage / tile 因子 / intrinsic)——**对脏 GPU 噪声免疫, 首选**(`MetaScheduleApplyDatabase` 后 `mod.script()` 解析, 复用 `scripts/phase2/s2_2_verify_tc.py` 套路); (b) **动态**=ncu(Nsight Compute)读 occupancy/bank-conflict/mem-stall——需相对空闲 GPU + profiling 权限(H800 共享, 风险见 §6, 退化时只靠静态)。

| 维度 | 控制的软件变量 | 抽取的调度产物 | 耦合判据(随软件变量是否变) | 测量法 |
|---|---|---|---|---|
| **D1 intrinsic** | 剪枝宽度(64↔48)× 精度(INT8) | 选中 intrinsic(WMMA-K16 / mma-K32 / dp4a / 标量) | 失配宽度是否改变 intrinsic 命中或降级 | 静态: TIR builtin (`tvm_mma_sync` 等) |
| **L1 内存暂存** | 剪枝宽度 | cache scope(shared/reg/fragment)+ 每 tile SMEM bytes | SMEM 占用/occupancy 随宽度变 | 静态(SMEM bytes)+ 动态(ncu occupancy) |
| **L2 线程绑定** | per-stage 剪枝(各 stage 通道) | bind 网格(blockIdx/threadIdx/vthread extent) | 网格配置随通道数变 | 静态(bind extent)+ 动态(ncu achieved occ) |
| **L3 bank-conflict** | 剪枝宽度 | `storage_align` padding 因子 | padding 是否随宽度变 / bank 冲突是否触发 | 静态(align factor)+ 动态(ncu shared bank conflicts) |
| **L4 软件流水** | 剪枝(memory-bound 程度) | software_pipeline stage 数 | 流水深度随访存量变 | 静态(stage 数)+ 动态(ncu mem stall) |
| **L5 跨线程 reduction** | 剪枝(K=Cin) | rfactor / reduction split 因子 | reduction 策略随 K 变 | 静态(TIR reduction 结构) |
| **D2 layout+pad** | 剪枝宽度 × 精度 | 选中 layout + 是否插 pad | layout/pad 随宽度变; pad 能否救回 TC | 静态(layout/pad)+ TC builtin |
| **D3 tiling** | 剪枝宽度 × MMA shape | tile split 因子 | tile 占用率随宽度变 | 静态(tile 因子)+ 动态(ncu tensor active) |

> **判据强度分级**(防 R2/§3.1 单配置假象): 一个维度的"耦合成立"= 调度产物在 aligned vs misaligned 间有差异 **且** 跨 ≥2 trial 预算/seed 稳定 **且** 有体系结构解释。"协同可缓解"(S2.2c)= 主动配置该维度能逆转失配劣势。两者都需 supervisor 复核。

### §4.3 耦合发现 campaign — 主动发现新耦合, 不止验证已知陷阱 ★

> 动机(用户 2026-06-17 指正): §2 对每个维度都给了**假设的**耦合/陷阱, 但已实证的只有 D1/D2 的"对齐×INT8"一个。本 campaign = **用 TVM 的大配置空间主动筛查 L1/L2/L3/L4/L5/D3/D4 上是否存在尚未发现的 prune/quant×schedule 耦合**, 把"假设"变成"实证或证伪"。这是 route2 价值的关键放大器: 发现得越多, "TVM 看得见 TRT 看不见的耦合"越立得住。

**软件轴(S, 要扫的自变量)**:
- **S1 剪枝宽度(通道数)**: 细粒度扫, **故意覆盖各对齐边界** —— ÷32(32/64/96/128) / 仅÷16(48/80/112) / 仅÷8(24/40/56) / 奇·质数(33/50/67), 以暴露**不同粒度**的悬崖(不只 ÷32)。
- **S2 per-stage 剪枝分布**(哪些 stage 变窄): 触发 L2 负载均衡 / D4 跨 stage 融合。
- **S3 量化精度**(INT8 / FP16 / 逐层混精): 触发 D1 intrinsic / L1 SMEM 容量 / L4 memory-bound。
- **S4 量化粒度 / Q-DQ 放置**(per-tensor vs per-channel; DQ 位置): 触发 D4 融合断点。
- **S5 剪枝诱导 roofline 漂移**(FLOPs/byte): 触发 L1/L4。

**两阶段**: ① **screening(宽而廉)** = 对代表性 conv 扫 **S1×S3 网格**, 每点 dlight 快速默认调度或小预算 tune, 记 **延迟 + 全维度调度产物**; ② **deepen(窄而深)** = 对标出的候选按 §4.2 协议隔离深探。

**耦合信号探测器(什么算"发现一个耦合")**:
- **D-a 延迟悬崖/非单调**: 延迟 vs S1 宽度在**非显然宽度**(非 ÷32)跳变/非单调 → 候选新陷阱。
- **D-b 调度产物切换**: 某维度选择(intrinsic/layout/tile/staging/bind/pipeline)在某宽度阈值突变 → 该维度与剪枝耦合。
- **D-c prune×quant 不可分**: (W,INT8) 的最优调度结构 ≠ (W,FP16) 的, 超平凡差异 → 交互耦合。
- **D-d occupancy/bank 悬崖**: 静态 SMEM bytes 跨 occupancy 边界, 或 ncu 测到 bank conflict 在某宽度触发(L1/L3)。

**新颖性分类(每候选必分三类, 防 overclaim)**: ① 已知对齐陷阱重现(D1/D2 ÷32) / ② **新耦合**(其他维度/其他粒度/其他软件轴) / ③ auto-tuner 噪声或单配置假象(跨配置不稳)。**只有 ② 且跨配置稳 + 有机理解释**才算"发现新耦合"。

**S × 维度 探针矩阵(每 cell = 一个设计好的探针; 确保每维都被探, 含负结果)**:

| 维度 \ 软件轴 | S1 宽度 | S2 per-stage | S3 精度 | S4 Q/DQ | S5 roofline |
|---|---|---|---|---|---|
| D1 intrinsic | ★对齐粒度(K16/K32/dp4a) | – | ★精度定 intrinsic 集 | – | – |
| L1 内存暂存 | ★SMEM/tile×occupancy | – | SMEM 等效容量 | – | memory-bound |
| L2 线程绑定 | grid extent | ★负载均衡 | – | – | – |
| L3 bank-conflict | ★SMEM stride→bank | – | 位宽×bank | – | – |
| L4 软件流水 | – | – | 访存量×流水深度 | – | ★memory-bound→流水 |
| L5 reduction | ★K=Cin×并行度 | – | – | – | – |
| D2 layout+pad | ★向量化粒度 | – | INT8 NCHW32 vs FP16 | – | – |
| D3 tiling | ★tile×MMA fit | – | MMA shape | – | – |
| D4 fusion×Q/DQ | – | 跨 stage 融合 | – | ★Q/DQ 断融合 | – |

(★=主探针; –=弱/不探)

**负结果也是发现**: 若某维度在全 S1×S3 扫描下调度产物/延迟都平滑无耦合(TVM 灵活性消解了它), 明确记"该维度在本模型/本硬件上无可观测 prune/quant 耦合" —— 直接喂 §5 集成判定(候选池收窄)。

**产出**: "维度 × 软件轴 → {有耦合(新/已知) / 无 / 噪声}" 总表 + 每个确认耦合的机理 + 证据日志。把 §5 的"发现 ≥2 耦合"从 D1/D2 两候选, 扩成**全 9 维系统筛查**的结果。

---

**关键纪律(违反=数据作废)**: ① 延迟必空闲 GPU 实测(util≤2%/foreign≤50MiB); 调度产物/TC 静态分析对噪声免疫可带噪做。② TVM 绝对延迟常追不上 TRT —— 价值在**发现/量化/缓解耦合**非刷 SOTA, 别把"TVM 更慢"当失败。③ 任何"发现耦合/已缓解"必**跨 build/schedule 配置验稳定**(防 §3.1 单配置假象)。④ supervisor 对每个"已 build/对齐/加速/发现耦合"复跑+读文件+数值 diff 核验。⑤ 区分 真测/估算/仅声明。

---

## §5 把 TVM 集成进框架的判定标准 (S2.4 决策门)

> 整个 route2 的终点决策: **TVM 是否值得作为一条"调度可搜硬件轴"集成进我们的 P(剪枝)×Q(量化)×D(硬件)联合搜索框架?** 必须由实验回答, 不能靠猜测/推断(纪律)。

**集成(GO) 需同时满足**:
1. **[发现]** TVM 暴露 ≥2 个 TRT 看不见的 prune/quant×schedule 耦合(候选池 = T1: D1 intrinsic / L1 内存暂存 / L2 线程绑定 / L3 bank-conflict / L4 流水 / L5 reduction; T2: D2 layout / D3 tiling; T3: D4 fusion×Q/DQ), 且跨配置稳定。
2. **[缓解]** TVM 能**协同地配置** ≥1 个耦合(pad/repack/intrinsic 切换/storage_align/staging/流水), 恢复 TRT 因失配而损失的性能 —— 即真正展示"硬件维度被协同配置"(S2.2c)。
3. **[可搜]** 这些调度轴能与 P×Q **在可处理预算内联合搜**(tune 时间有界; 空间不爆炸; cost model 可用, 必要时引 TLP 思路)。
4. **[Pareto]** 在真实边缘硬件(4090/Orin)上, 联合搜点到达 TRT-auto 不可达的 Pareto 区(latency/energy), **或** 至少绝对延迟输但有清晰可发表的机理增量(承认 TVM 可能追不上 TRT 绝对延迟)。

**不集成(NO-GO / 负结果)路径**: 若标准 1 或 2 失败(TVM 发现不了新耦合, 或发现了但缓解不了/缓解后仍被 TRT 支配) ⇒ **诚实记负结果**: "TRT 太薄已证, 但 TVM 在固定硅片上也未能把调度变成有协同增益的可搜轴" ⇒ 按纪律退回 **route1**(重锚 Orin 异构 GPU∥DLA 真并行 1.34× + 谦逊措辞), 并把 route2 的负证据作为论文"为何不走纯调度协同"的论证。**这一退回必须由 S2.2c/S2.4 的实验结论触发, 而非中途猜测。**

> **★S2.2b 实证后的判定现状(2026-06-17, 诚实更新)**: 61 tune 点跑下来 ——
> - **判据2(缓解)= 强支持**: TVM-WMMA-K16 让所有 TRT 会撞 ÷32 墙的失配 conv 都保住 TC(16/16 零 fallback); Q/DQ 被 fuse 进 epilogue 不破融合。**两个 TRT 耦合(对齐/Q-DQ)在 TVM 都被消解**。
> - **判据1(发现≥2 新耦合)= 未达**: 反复发现的是 **TVM 去掉 TRT 的耦合**, 而非"TVM 揭示 TRT 看不见的新耦合"。唯一候选(grouped g=32→无 TC)**非 TVM 专属**(TRT 也无法 TC grouped)。L2/L3/L4/延迟在 dense 上均负结果或噪声。
> - **⇒ 命题需诚实改写**: route2 的真实价值不是"TVM 揭示新耦合", 而是 **"TVM 的调度/融合灵活性是一层去耦合器, 移除 TRT 锁死的对齐×量化、Q/DQ×融合耦合"**。这本身可发表(co-configuration 缓解 demo, 判据2), 但**若坚持原"发现新耦合"命题则触发 NO-GO**。下一步抉择(HANDOFF §0 A/B/C): A 接受缓解器论点收口 / B 继续找判据1 新耦合(D3 大conv/L4 流水/全图INT8) / C 修通 force-K32 加强反事实。**待用户拍板, 不中途猜测退 route1。**

> **★★E-couple 完整逐旋钮扫描 + E-e2e 判定更新(2026-06-18, H800 空闲实测; 模型=PyramidFusion; 详见 HANDOFF_route2_e2e_coupling_v1.md §6)** —— 回应用户两缺口(per-op≠e2e / 每旋钮耦合未证):
>
> **(1) 逐旋钮 × 软件轴耦合表(每旋钮只变该旋钮、扫剪枝宽度/输出通道, 看 argmin 是否随软件漂移; 全部真实测)**:
>
> | 旋钮 | 软件轴 | argmin 随软件漂移? | 效应 | 耦合判定 | 证据 |
> |---|---|---|---|---|---|
> | **D1** tensorize | 宽度×精度; 结构 | 宽度否(WMMA 全 24–128/int8+fp16); 结构是(grouped→SCALAR) | 类别型 | 宽度可分离; **与层结构耦合**(TRT 共享, 非 TVM 专属) | s2_2b screen |
> | **D3** tiling(含 BK) | 宽度 | **是**(16×16×16 W48/64 最优→W128 最差) | **+21%**(折中 tile 7.3%) | **中等干净耦合** | s2_2d_rep(3×空闲) |
> | **L1** staging | 宽度 | 弱是(shared 仅大宽度回本) | ~+13%@W128 | 弱耦合 | s2_2f_rep |
> | **L2** thread-tile | 宽度 | 部分(16 稳健; 32 在 W48 不可行/W128 +48%) | ≤+48% | 弱耦合 | s2_2f_rep |
> | **L3** bank-align | 宽度 | **否** | ≤15% 噪声地板 | 可分离/不可判(需 bank-conflict-heavy workload) | s2_2f_rep |
> | **L4** sw-pipeline | 宽度 | **否**(无单调) | 9–25% | 不可判(需 deep-K) | s2_2f_rep |
> | **L5** reduction | Cout | **是(强)**(cross@Cout≤32↔serial@Cout≥64) | 4.6×→翻转→11.6× | **强干净耦合** | s2_2g(空闲) |
>
> **⇒ 旋钮不等价耦合**: 强干净 = **L5 reduction + D3 tiling**; 结构型 = D1; 弱 = L1/L2; 不可判 = L3/L4。co-design 价值**集中在 归约策略+tiling+intrinsic↔结构**, 非均匀。(判据1 由此**部分达成**: L5/D3 是宽度×schedule 的真耦合且跨配置稳, 但仍是"调度内部最优随剪枝漂移"而非"TVM 看见 TRT 看不见的物理陷阱"。)
>
> **(2) E-e2e(答 per-op≠e2e)**: backbone 子图 MS-tuned/dlight-default = **base 10.34× / p50 8.74×**(真实测), per-op 旋钮**确实**折端到端(backbone=50conv 旋钮全覆盖)。**但 Amdahl 真相(真实测 E8 全链路 Orin 直算)**: backbone 仅占全协同 pipeline 13.6%, **fusion neck 占 55%且 TVM 当前未调** ⇒ backbone 10.34× 折成**全 pipeline 净 1.14×**。⇒ **加速集中在 RSU 感知段(backbone 占 RSU 链 ~45%, 净 ~1.68×); 车端因 fusion 主导被 Amdahl 稀释**。真 headroom = fusion neck(需 torch 前端导入后才能调, 当前是导入工具链受阻**非**技术不可行)。

---

## §6 风险与诚实边界

- **R1 量化 pass 缺口**: relax 0.20 无现成 INT8 量化 pass → 全模型 INT8 受阻, 当前用忠实合成单 conv 折中。风险: 合成 conv 可能漏掉全图级耦合(融合/layout 传播)。缓解: S2.3 若需全图 INT8, 评估 relax torch 前端 + 手写 QDQ 或外部量化导入。
- **R2 WMMA-K16 可能"消解"主悬崖**: S2.2a 已显示 TVM 默认 WMMA(÷16)可能让 48 不失配 → D1 在 dense conv 上"无悬崖"。若 S2.2b 在 grouped conv / N=Cout÷32 / 全图 NCHW32 传播上也无悬崖, 则 D1 的"发现"价值削弱 → 重心移到 **D2(layout/pad 的协同配置)**与 **D4(fusion×Q/DQ)**。这本身是合法发现(TVM 的 intrinsic 灵活性=对 TRT ÷32 悬崖的天然缓解), 但要诚实区分"发现新陷阱" vs "证明 TVM 绕过了旧陷阱"。
- **R3 绝对延迟**: conv-dense backbone 上 TVM 大概率追不上 TRT 绝对延迟(Roller/ALT 也强调价值在被忽视维度而非峰值)。判定标准 §5.4 已显式允许"输延迟但赢机理"。
- **R4 单配置假象**: 历史已犯 overclaim(S0 "INT8 层数单调塌缩"被全网格证伪)。所有耦合结论必跨配置 + supervisor 核验。

---

## §7 引用索引

**编译调度 / tensor-core 映射**
- AMOS (ISCA'22) — Enabling Automatic Mapping for Tensor Computations on Spatial Accelerators with HW Abstraction — https://cs.stanford.edu/~anjiang/papers/ZhengETAL22AMOS.pdf
- Bolt (MLSys'22) — Bridging the Gap between Auto-tuners and Hardware-native Performance — https://proceedings.mlsys.org/paper_files/paper/2022/file/1f8053a67ec8e0b57455713cefdd8218-Paper.pdf
- Roller (OSDI'22) — Fast and Efficient Tensor Compilation (rTile) — https://www.usenix.org/conference/osdi22/presentation/zhu
- ALT (EuroSys'23) — Breaking the Wall between Data Layout and Loop Optimizations — https://dl.acm.org/doi/10.1145/3552326.3587440
- FAST (ASPLOS'22) — Full-Stack Search incl. tensor padding — https://dl.acm.org/doi/10.1145/3503222.3507767
- Ansor (OSDI'20) — Generating High-Performance Tensor Programs — https://www.usenix.org/conference/osdi20/presentation/zheng
- TLP (ASPLOS'23) — DL-based Cost Model for Tensor Program Tuning — https://dl.acm.org/doi/abs/10.1145/3575693.3575737
- CHaNAS (TECS'22) — NN Architecture & Compile Co-optimization — https://dl.acm.org/doi/full/10.1145/3533251
- Hidet (ASPLOS'23) — Task-Mapping Programming Paradigm — https://dl.acm.org/doi/10.1145/3575693.3575702

**硬件格式 / 对齐 (一手 NVIDIA / CUDA)**
- TensorRT Data Format (NCHW32, 32-channel vectorized, DLA INT8 native) — https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/data-format-desc.html
- CUDA WMMA int8 shapes (m16n16k16/m8n32k16/m32n8k16) & PTX mma int8 (m16n8k32) — CUDA C Programming Guide §WMMA / NVIDIA Developer Forums

**软硬协同原理 (详见 survey_hwsw_codesign_v1 §1.5)**
- Roofline (Williams et al., CACM'09); Horowitz energy (ISSCC'14); HAQ/APQ/OFA/HALP/NACOS — 见 survey_hwsw_codesign_v1.md §5

**本项目实测锚**
- `results/S0b_coupling_clean_4090.csv` — 对齐×INT8 悬崖(trap25 1.08×)
- `results/H800_s2_1_backbone.log` — S2.1 数值对齐 2.4e-6
- `results/H800_s2_2_int8_{aligned64,misaligned48,mitigated48to64}.log` — S2.2a WMMA-K16 三臂
- `scripts/phase2/s2_2_{probe,int8_cliff,verify_tc}.py` — 复跑脚本
