# HANDOFF — 协同加速下一阶段 v2 (2026-06-21, 取代 v1)

> **取代 `HANDOFF_codesign_nextstage_v1.md`**(v1 的 Option B 可行性判断与 DS/CARLA 现状已过时, 见下)。
> 现状入口仍含 `HANDOFF_three_arm_ablation_results_v1.md`(三臂 B1–B5)+ `auto-tuning/4_design_ablation_proof_v1.md §9`(命题 + 真数据 + §9.8 跨阶段更新)。
> **本页**: nextstage 执行后的真实状态 + 下一阶段**该并行开启哪些 agent** + 分工。
> **新窗口阅读顺序**: §B 背景 → §0 状态 → §1 关键事实 → §2 教训 → §3 计划(从 L1 起) → §4 文件命令。深挖看 `HANDOFF_three_arm_ablation_results_v1.md` + `auto-tuning/4_design_ablation_proof_v1.md §9`。

---

## §B 项目背景(新窗口先读这一段)
**这是什么**: 软硬件协同加速框架的论文实证。核心论点 = **对某些网络, 剪枝(P)×编译调度(S)×量化(Q)三个优化轴互相耦合, 必须联合搜索(co-design)才能找到最优配置; 串行/分步优化会系统性错过最优**。主载体 = **PyramidFusion**(HEAL, V2X 协同 3D 检测, grouped/ResNeXt bottleneck conv); 对照 = **CoDriving**(标准 conv); 数据 = **DAIR-V2X**; 延迟 = **H800 上 TVM(relax+MetaSchedule)真测**。

**核心机理(已实证)**: 剪枝宽度决定 TVM 能否选到高效核 —— Pyramid grouped conv 中, stage 通道对齐(s0=64, 每组 4 通道)时 TVM tuned 余量 ~7.9×, 失配(s0=48, 每组 3)仅 ~2× → **同一对 (延迟,AP) 等价配置, default 下 W_g 更快、tuned 后 P_g 远更快 = 排序翻转(rank-flip)** → 串行先按 default 锁宽度会锁错(锁 W_g), 永远到不了 P_g 的 tuned 高速点。机理 iso 消融定位在 stage0(只 s0 失配就塌, s1/s2 失配不塌)。

**三臂 ablation(主证据, 12 seed)**:
- **A-joint** 联合搜 (prune, schedule[, quant]) → **99.6%** ref HV
- **A-serial** 先按 default 锁剪枝宽度, 再只调 schedule → **86.0%**(系统性劣于联合)
- **A-noS** 无调度(全 default) → **48.0%**
- Wilcoxon A-joint vs A-serial **p=4.9e-4** = framing-independent 主结果。

**关键术语**: **W_g**=贪心陷阱宽度(default 快, 被串行锁); **P_g**=Pareto 优宽度(tuned 快, 被串行结构性丢弃); **iso-AP 倍率**=同 AP 下 W_g/P_g 的 tuned 延迟比(pair1 3.51×/pair2 3.29×/pair3 3.83×); **shipped 协同赢**=P_g 在全局 Pareto 上(真出货价值, 仅 pair2 是) vs **仅机理**=P_g 被更高 AP 宽度全局支配(只示机理, pair1); **HV**=hypervolume; **同 AP 由零填充权重恒等保证**(P_g = W_g 的 s0 补齐到对齐宽度, 输出恒等故 AP 相等)。

**论文目标 / 当前张力(决定下一阶段)**: 要把"co-design 必要"做成**普适洞察**而非"只对 Pyramid 这一个网络"。当前 Pyramid 显耦合、CoDriving 疑似可分离(但欠实测)→ 需 ① **Q 量化轴证三轴强耦合**(L1)、② **CoDriving 多点真测 + 架构原理定论**(L3, 量化下标准 conv 可能也耦合 → 普适)、③ **网格扩到"够"**(L4 有明确停止判据)。

**仓库 / 环境(直接开工必备)**:
- 真仓库 = **`/home/jichengzhi/V2X`**(⚠️ `/home/jichengzhi/UniV2X` 是断裂符号链接空壳, 别用)。
- 跑 framework 脚本(`run_b4_ablation.py`/`b5_verify_convergence.py`, 需 matplotlib/numpy/scipy)用 conda: **`/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python`**(系统 python3 缺 matplotlib 会报错)。
- 硬件: **4090 本机**(剪枝/TRT/部分实验, GPU 0-7 有他人占用, 跑前 nvidia-smi 确认 util 0%/mem≤50MiB)+ **H800**(TVM 调优/量化/闭环, ssh 见 §4)。
- 主表 `multi_agent/data/dataset_v2.{csv,parquet}`。

---

## §0 一句话状态(2026-06-21 接管后)
三臂 ablation 主结论坐实(A-joint 99.6%/A-serial 86.0%/A-noS 48.0%, p=4.9e-4)。本轮 nextstage 已**真完成**: ① **pair3 补全**(s2_128 修通, latency rank-flip **3.83×**=三对最大); ② **CoDriving 对照**(三臂脚本跑出 100%=100%, 但**仅 2 真测点+2 估算点, 用户不认可此"可分离"结论 → L3 需多点真测定论**); ③ **RSU 闭环 smoke PASS**(DS=100/RC=100); ④ **Q 轴 Option B 可行性确认**(TVM int8 WMMA 在 sm90 已工作)。**所有 agent 已停**, 结果**尚未 commit**, pair3/CoDriving/Q/RSU 尚未并入 headline ablation 重跑。

> **★全部 agent 当前 = 停止状态**(用户指令)。本页是"下一阶段恢复时该开哪些 agent"的蓝图, 不是"现在在跑"。

> **[2026-06-21 接管恢复 update]** 下一阶段已启动: **L2 已完成并核验**(pair3 s2_128 并入 → 3 对 ablation 重跑: A-joint 99.9%/A-serial 85.5%/A-noS 47.7%, Wilcoxon p=4.88e-4, STRUCTURAL PASS; **shipped headline 转 pair3 = 3.83× @ iso-AP70=0.6369**, 因 s2_128 全局支配旧 shipped winner s1_64; B5 审计通过; doc4 §9.1–9.5 已回写)。**L1(Q 轴 int8, data-orchestrator)+ L3(CoDriving 多点, sw-optimizer)已作为后台 agent 启动**, 跑 H800 TVM。尚未 commit。
>
> **[2026-06-21 16:35 L1-hw-optimizer update]** Q 轴 INT8 测量推进报告: ① **全 7 个 default INT8 延迟已测**(H800 util=0%/mem≤50MiB, 口径=H800_TVM_int8_real, 本地 `/home/jichengzhi/V2X/results/q_int8_{base_gate,pairs}.csv`); pair3: mix_d[48,128,128]=52138µs vs s2_128[64,128,128]=53413µs, default 下差异极小(artifact); ② **7 条 MetaSchedule tuning 任务同步运行**(GPU 0/2/3/4/5/6/7, 300-500 trials, ETA ~7-8h, 约 midnight); ③ **★关键方法论发现**: QDQ-ONNX→TVM relax 编译后 conv buffer 类型=float32, TVM relax 对 QDQ 格式做 FP32-GEMM + QDQ overhead, **不产生真 INT8 TC 计算**(TIR 已确认). 若要真 INT8 TC 需 cutlass/dp4a 路线; ④ **screenA 单 conv INT8 基准(s2_2b_screenA_int8.csv, 13 宽度)**: H800 上 w24–w128 全 WMMA, **w48(3.172µs)快于 w64(4.437µs)** — 无 kernel-cliff, Q-rank-flip 在 H800 TVM 单 conv 层面不成立(与 4090 TRT 方向相反); ⑤ tuned 结果到位后补 PENDING → 真值, 届时更新 latency_lut_pyramid_q_tvm.json 并重跑 detect_q_rank_flip_pairs.

---

## §1 接管期核实的关键事实(纠正 v1 的过时判断)
1. **Option B(真 TVM int8 统一 P×Q×S 单口径)可行** —— v1/接管初期"TVM 无 int8 pass、需数周"**判断错, 已推翻**。证据: `/exdata/jichengzhi/s2_tvm/s2_2b_screenA_int8.csv` 13 conv 宽度全 `WMMA/int8` 真测(sm90)。路线 = **QDQ-ONNX → relax ONNX frontend(`from_onnx`)→ MetaSchedule int8 WMMA**。spike `q0_int8_spike.py` 流程跑通, 唯一卡点 = **nvcc 不在 PATH**(环境, 非能力)。
2. **pair3 已补全**: s2_128 [64,128,128] TVM 崩(seed=0 选出 OOB kernel, 完整图 compile 时 illegal-access)用 **seed=42 + fresh workdir + subprocess 前置验证** 修通。tuned 5506µs / mix_d 21071µs = **3.83×**。(`results/s2_128_fresh_tune_seed42.csv`)。iso-AP 值待校(应≈mix_d 0.637, 非 T2 报的 0.590)。
3. **闭环 RSU 可行**: 移植 P1–P5 **实际已完成**(v1 "CARLA 未装/不可运行" 是 stale)。Pyramid+RSU r0: DS=100/RC=100/Completed。
4. **DS 计算勘误仍在**(doc4 §9.5): DS 须 (AP, latency) 共同决定; β=0 投影到 CoDriving 曲线仅"同 AP 对内"有效。正解 = 真闭环建 Pyramid 专属 DS(AP, τ) 2D 曲面。
5. **TVM 环境正确启动**(直驱必备): `export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path)`; python `/exdata/jichengzhi/tvm310/bin/python`; **CUDA codegen 还需 nvcc 在 PATH**(prior int8 work 有, 须定位并加入)。

---

## §2 ★教训(防再踩, 接手必读)
- **"GPU 上没进程" ≠ "agent stalled"**: 本轮多次把"已完成、进程已退出"误判成"卡死"(T2 12:10 完成, 13:11 看无进程=误判)。**核验要读输出文件/结果 JSON, 不能只看 `nvidia-smi`**。
- **agent 会留孤儿/不汇报**: T4 smoke 完成后把 CARLA+process_b 孤儿留在 GPU5 空转 1h。**用完即清孤儿**(按端口/world-port 精准 kill, 别误杀他人 V2Xverse_apknob 进程)。
- **别过早断"不可能"**: Option B "需数周"是不完整探针(只查 `relay.quantize`)得的错论。穷尽路线(QDQ-ONNX + 已有 int8 WMMA)再下结论。见 [[feedback-no-premature-impossible]]。
- **核验自报**: 一切"已测/已修/已build"主控复跑/读文件/git diff。本轮抓到误判也靠它。

---

## §3 下一阶段计划(用户 2026-06-21 重定向)
> 顺序铁律: **维度先于网格** —— Q 维(L1)建全并证强耦合 → 再 L4 三维扩网格。L1/L2/L3 可即刻并行起;L4 gated on L1。**L5 真闭环已委托其他 agent, 本团队不安排。**

### L1 — Q 量化轴接入 → 证 **三轴(prune×quant×schedule)强耦合**(P0, 关键路径)【data-orchestrator(lead)+sw+hw】
**目标(明确)**: 拿**真 TVM int8 数据填入 headline**, 并在三轴 ablation 上证 **A-joint(三轴联合)≫ A-serial(锁任一轴), Wilcoxon 显著** = 三轴强耦合。
- **第一步(解 blocker)**: 定位 **nvcc 加入 PATH** → 重跑 `q0_int8_spike.py`(GPU6, `LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path)`)拿 **base TVM int8 真延迟 gate**(vs FP32 tuned 6320µs)。
- **第二步**: 关键对 trap25/pad64(共享 s1=96)、mix_b/s1_64(共享 s1=64)的 TVM int8 真延迟 → 验 s0 效应 >1.2×?(TRT 仅 ~5%=null; TVM 显著 → Q-rank-flip TVM-specific 真发现)。
- **第三步**: 真 int8 **AP**(DAIR val + HEAL, per-channel weight 校准, **绝不 simulated/fp16 代**)。
- **第四步**: 真 TVM int8 填入 `framework/search_three_arm.py` 的 P×Q×S 内核 → `detect_q_rank_flip_pairs` + **三轴 ablation** → **判据: 三轴 A-joint ≫ 锁单轴的串行臂**(强耦合成立则放行 L4 的 3 轴扩点)。
- **铁律**: TVM(prune×sched×quant)同口径; TRT 10/10 仅佐证层不进 headline; 空闲 GPU; fresh workdir 进程隔离。

### L2 — pair3 并入 + 3 对 ablation 重跑(P0, 纯 Python, 无 GPU)【data-orchestrator/主控】
- 确认 pair3 **iso-AP 恒等**(s2_128 = mix_d s0 零填充? AP≈mix_d 0.637, 核 T2 报的 0.590)→ 重跑 `run_b4_ablation`+`b5_verify_convergence` → 报 **3 对** HV/Wilcoxon/shipped-vs-mechanism → 更新 doc4 §9.2–9.4(当前仍 2 对)。

### L3 — ★CoDriving **多点真实测**(P0, 用户**不认可**当前可分离结论)【sw-optimizer(lead)+data-orchestrator】
> 现状: "可分离" 仅靠 **2 真测点(base 2.07×/p50 2.26×)+ 2 估算点 + 物理**, 欠实测。用户要求**多点真测**才能下结论。
- **第一步(杀估算)**: p25/p75 跑**真 TVM**(H800, fresh workdir 逐宽度), 补足 ≥4–5 真测宽度, 去掉幂律外推。
- **第二步(s0 失配直接探针)**: 造固定 (s1,s2)、仅变 s0 的 CoDriving 宽度对(镜像 Pyramid 几何)→ 真测 → 直接证 FP 下**有无** rank-flip(不靠物理外推)。
- **第三步(★INT8 态)**: 测 CoDriving 的 **INT8 TVM** 多点 —— 标准 conv 量化后(WMMA K 需 32 整除, 比 fp16 严)**是否变耦合**? 这是把框架从"1:1"变普适的关键实验。
- **判据**: 用真测多点 + 架构原理(对齐敏感度)给 CoDriving 的耦合/可分离**定论**;无论结论是可分离还是(INT8 下)耦合, **都要数据 + 原理双支撑**, 不硬凑。

### L4 — Pyramid 网格扩充【gated on L1, 有**明确停止判据**】【data-orchestrator(lead)+hw+sw】
> 用户要求: 必须有"扩到什么程度就够"的明确目标。**满足下列 4 条即停**:
1. **AP 轴有真 trade-off(找到悬崖)**: 补 aggressive 剪枝(80/87/93%)直到 AP70 真正陡降 → Pareto 的 AP 轴是真曲线非高原(消"AP 轴弱"caveat)。
2. **rank-flip 对充分且跨 AP 分布**: ≥5 对 W_g/P_g, 分布 ≥4 个 AP70 档(~0.55–0.63), 其中 **≥3 对 shipped**(P_g 在全局 Pareto)。
3. **结论饱和**: 再加一批宽度后 A-joint vs A-serial 的 HV gap 相对变化 **<2%** 且**无新 AP 档新对** → 网格已足, 停。
4. **config 数 ≥ 3× budget**(支撑"真搜索非枚举")。
- **★3 轴扩点(gated on L1 证强耦合)**: 若 L1 证三轴强耦合, 则把现有 **2 轴(prune×sched)点全部补 quant 维 → 3 轴点**(每点 ×{fp16,int8} 真测), 在完整 P×Q×S 上扩网格 + 重跑 ablation。若 L1 未证强耦合(Q 与 prune×sched 可分), 则 Q 维保持稀疏, 不强行 3 轴扩点。
- **策略(控成本)**: 廉价延迟筛(含 int8)选 rank-flip 候选 → 门控 finetune(只真测翻转对的 AP)→ 并入重跑。

### L5 — 真闭环 DS ★【已委托其他 agent, 本团队不安排】
- 闭环 DS 实验(τ sweep / DS(AP,τ) 曲面)由其他 agent 负责。**延迟口径 = Orin 真测**(非 H800→Orin 估算), DS 须 (AP,latency) 共同决定。本团队仅按需**提供** trap25/pad64 等 body 延迟 + 剪枝/量化档 AP。G1 RSU smoke 已 PASS 背书可行性。

### 横向角色
- **supervisor**: 全程核验"已测/已修/已build"自报(复跑/读文件)。
- **doc-curator**: 把 §9.8 已核验结论整合进 doc4 主体(非追加)+ dataset_v2, 并 commit(用户单独授权 commit)。

---

## §4 关键文件 / 命令速查
- 内核 `framework/search_three_arm.py`(`detect_wg_pg_pairs` + `CostModelPQS`/`run_*_pqs`/`detect_q_rank_flip_pairs`); 驱动 `framework/run_b4_ablation.py`; B5 `scripts/phase2/b5_verify_convergence.py`。
- 数据: `results/{latency_lut_pyramid.json(含s2_128), latency_lut_pyramid_q.json(TRT佐证10/10), b4_ablation_results.json, b5_convergence_verification.json, b4_codriving_ablation_results.json, s2_128_fresh_tune_seed42.csv}`。
- H800 TVM int8: `/exdata/jichengzhi/s2_tvm/{q0_int8_spike.py, s2_2b_screenA_int8.csv, models/base_backbone.onnx}`; 环境 `LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path)` + **nvcc on PATH**(待补)。
- H800 闭环: `/exdata/jichengzhi/V2Xverse_pyramid`, eval GPU5, τ config `pnp_config_pyramid_sweep_d{0,108,219,500}.yaml`, 设计 `sim_closedloop_pyramid_ds_surface_v1.md`。
- 设计 `auto-tuning/4_design_ablation_proof_v1.md §9`(§9.8 = 本轮新结果)。
- H800 ssh: `ssh -p 30001 jichengzhi@222.95.84.215`(pw 12345678, sshpass ConnectTimeout≥45, 注意连太频会被 rate-limit)。
