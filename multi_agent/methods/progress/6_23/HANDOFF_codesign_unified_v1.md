# HANDOFF — 协同加速统一交接: TVM 跨模型实证 + 论文叙事/补实验 (v1, 2026-06-18)

> **本页是单一权威入口**, 整合原两份文档:
> - 技术现状(原 `HANDOFF_tvm_codesign_crossmodel_v1.md`): TVM route2 在 Pyramid[HEAL]+CoDriving 两模型的适配/耦合/端到端实测。
> - 论文叙事(原 `HANDOFF_paper_story_synthesis_v1.md`): 把全部结论组织成顶会论文的故事架构 + 自洽性检验 + gap/补实验。
>
> 两份原文已并入本页, 仅作历史留存; **接手只读本页**。
> 细节子交接(仍有效): Pyramid 线 [HANDOFF_route2_e2e_coupling_v1.md](HANDOFF_route2_e2e_coupling_v1.md) · CoDriving 线 [HANDOFF_codriving_tvm_migration_v1.md](HANDOFF_codriving_tvm_migration_v1.md)。
> 维度定义 [dims_hardware_v4.md](../design/dims_hardware_v4.md) · 文献支撑 [survey_compiler_schedule_search_v1.md](../../references/survey_compiler_schedule_search_v1.md) + [study_joint_search_methods_v1.md](../../references/study_joint_search_methods_v1.md)(ALT/CHaNAS/AutoTVM 联合搜索研读) · memory `project-hwsw-codesign-route2` + `project-codriving-optimization-pilot`。

---

## §0 一句话现状

route2 = 用 TVM 把编译调度变成"可搜硬件轴"挣 co-design 标签。两个 V2X 协同感知模型都已完成 **TVM 适配 + 耦合/端到端实测**(平台主体 H800;边缘绝对值待 4090/Orin)。

**★可发表的核心科学结论(两模型对比得出)**: **co-design 的 HW×SW 耦合强度是 model-dependent 的** —— 非标算子结构(Pyramid grouped bottleneck)显**耦合**;标准算子(CoDriving 深归约 3×3 conv)显**可分离**。同一套 TVM 调度轴, 不同模型上 co-design 价值天差地别。

**★论文叙事现状**: 原"耦合普遍存在→必须联合搜"主线被本阶段证据**既加固也削弱**(对齐硬约束加固, 耦合弱且 model-dependent 削弱) ⇒ 已诊断须升级为**条件化故事 B**("何时该联合搜, 何时可分离"), 否则不自洽。**最关键缺腿 = Gap1(联合搜 vs 串行搜 Pareto 对比)尚未做** —— 它是"co-design 有用"的唯一直接证据, 也是决定论文定位(框架 vs characterization)的分水岭。

---

# Part I — 技术现状 (TVM route2 跨模型实证)

## §1 共性方法论 (两模型通用工具链, 可复用)

1. **TVM env(H800)**: `${V2X_DATA_ROOT}/tvm310/bin/python`(0.20.dev1070 改版 Unity: `tvm.tir`→`tvm.s_tir`/`tvm.tirx`; MetaSchedule 在 `tvm.s_tir.meta_schedule`; dlight 在 `tvm.s_tir.dlight`; `get_block`→`get_sblock`)。跑前必 `export PATH=/usr/local/cuda-12.2/bin:$PATH; export LD_LIBRARY_PATH=$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path); export CUDA_VISIBLE_DEVICES=<idle>`。SSH 非交互每命令显式 export 代理 7897。
2. **导入+对齐**: `from_onnx`→relax→`relax.build(cuda)`→VM, vs ORT 数值对齐(两模型 maxdiff 均 ~e-6 PASS)。
3. **tune**: `Sequential([LegalizeOps,AnnotateTIROpPattern,FuseOps,FuseTIR])` → `ri.tune_relax(max_trials_global=,seed=)` → `MetaScheduleApplyDatabase` → `tvm.compile` → VM `time_evaluator`。
4. **手写 schedule 消融**: 编译前必打 `func.with_attr("tirx.is_scheduled", True)` 否则 dlight 重排报错; shared 暂存必配 **cooperative fetch**(shared-load 循环 fuse+split+bind threadIdx)否则 24× 假慢。
5. **共同限制**: relax **无 INT8 量化 pass**(全 fp32/fp16, INT8 须 BYOC-TRT); **fusion/neck 导入 BLOCKED**(grid_sample/where2comm/scatter 数据依赖 shape 推断, 两模型同病)→ 都退**抽 backbone-only 子图**做耦合/旋钮实测。
6. **模型专属坑**: Pyramid = grouped conv(g=32→SCALAR 无 TC); CoDriving = ①动态 batch(ONNX dim=0→读 shape 时替 2)②384→1 单类 cls 头触发 reduction "compact dataflow" 崩→抽 backbone-only ③有 BatchNorm(Pyramid 无)→MS apply DB 后须 dlight Fallback 补剩余块。

## §2 ★跨模型对比 (技术整合核心)

| 维度 | **Pyramid (HEAL)** | **CoDriving** |
|---|---|---|
| backbone 结构 | grouped bottleneck conv(g=32)+ 1×1 | 标准 ResNet 3×3 conv(g=1, 深归约 **K=9W ≫ N=W**) |
| fusion neck | pyramid attention(grid_sample warp) | where2comm attention(grid_sample + scatter) |
| **旋钮价值**(backbone tuned/default) | base **10.34×** / p50 8.74× | base **2.07×** / p50 2.26× |
| ↑根因 | grouped 非标 conv → dlight 默认差 → 搜索收益大 | 标准 conv → dlight 默认已好 → 收益小 |
| **HW×SW 耦合**(argmin 随剪枝漂移?) | **显耦合**: L5 翻转 + D3 tiling 21% rank-reversal(方阵 proxy) | **显可分离**: 6 旋钮 argmin 无一随剪枝真漂移 |
| ↑根因 | grouped + 方阵 K=N=W → 宽度依赖 tile 偏好 | 深归约 K=9W ≫ N → 普适 tile(16×16×32)冲掉宽度依赖 |
| 剪枝 backbone 编译级加速 | p50/base 2.07× | base→p50 ~5× |
| fusion neck TVM 导入 | BLOCKED | BLOCKED |
| RSU/车端 backbone 占比 | RSU ~45% / 车端 ~12%(eager 口径) | RSU 50% / 车端 50%(fusion 一半是 backbone 重跑) |
| 残留硬瓶颈 | fusion neck 55%(车端) | fusion_attn 16%(车端) |

**⇒ 统一结论**: ① **旋钮价值 = conv 越非标越大**(Pyramid grouped 10× vs CoDriving 标准 2×); ② **耦合强度 model-dependent**(Pyramid 显耦合 = route2 有 co-design 价值; CoDriving 显可分离 = 可先定 HW 再独立剪, route2 co-design 价值≈0); ③ **共同负证据**: 两模型 fusion neck 都导不进 TVM(grid_sample), 都按 Amdahl 被前后处理稀释, 真加速都需"backbone TVM/TRT + 前端 CUDA 化 + INT8 BYOC-TRT"。

## §3 Pyramid(HEAL)线现状 (细节见子交接)
- **E-couple**(7 旋钮): D1 结构型(grouped→SCALAR, 宽度可分离)/ D3 中等耦合(21% rank-reversal, 折中 tile 7.3%)/ L1·L2 弱 / L3·L4 不可判 / **L5 强耦合**(cross↔serial 翻转 Cout32↔64)。
- **E-e2e**: backbone tuned/default base 10.34× / p50 8.74×。
- **Amdahl(真实测 E8 Orin)**: backbone 占全协同 pipeline 13.6%, fusion neck 55% ⇒ backbone 10× 折成全 pipeline **净 1.14×**。
- **RSU 段(Orin 实测)**: VFE 计算编译化 **>200×**(eager 18ms→编译 85µs, 全是 eager 开销)/ scatter NCHW 向量化 **2–4×**(数值 maxdiff=0, 无需 .cu; 边缘卡须避免 permute 布局转置)/ backbone 10×。
- **全流程剪枝量化**: 剪枝 p50 **2.07×** / trap25(非÷32 失配)**0.29× 反而慢**; 量化(TRT 参考)对齐 1.32–1.60× / trap25 仅 1.08× ⇒ **÷32 对齐纪律**(失配剪枝量化双输)。

## §4 CoDriving 线现状 (细节见子交接)
- **P1**: S2.1 导入+对齐 PASS(3 专属坑解)。
- **P2**: backbone 旋钮 base 2.07× / p50 2.26×(≪ Pyramid); 剪枝 backbone 编译级 ~5× 可见。
- **P3(RSU/车端分段)**: backbone 实占 RSU 50% / 车端 50%(fusion 一半是 backbone 重跑)。**RSU 段** TVM 1.35× / +CUDA 前端 **2.9×**(无 fusion 残留 = 最易拿满加速); **车端段** TVM 1.35× / +CUDA **2.14×**, 残留 fusion_attn 16%。decode CPU-meshgrid 修复全网 **5–6×**(bit 级等价)。
- **P4(6 旋钮)**: HW↔剪枝耦合**弱/可分离**(D3/L1/L2/L3/L4/L5 无一随剪枝漂移); 耦合是算子结构型(深归约 / 头 Cout=1→cross)非剪枝率函数。

## §5 技术线下一步 (真工程, 兑现加速主路)
1. **整段 RSU TRT/编译实测**(两模型同路): voxelize+VFE+backbone, 消 eager + scatter 向量化/CUDA。RSU 无 fusion 残留 = 最易拿满(Pyramid/CoDriving 估算分别待实测、2.9×)。**首要**: 把分段估算变 Orin 实测真 e2e(= Part II 的 Gap4)。
2. **车端 fusion neck**: 两模型都 BLOCKED 在 grid_sample/where2comm。路径 = relax torch 前端 / BYOC-TRT GridSample plugin 碰这 16–55%。
3. **INT8 e2e**: relax 无量化 pass, 两模型都须 BYOC-TRT。
4. **边缘绝对竞争力**: TVM/TRT-tuned vs eager, 4090/Orin 实测(H800 只坐实相对/机理)。

## §6 环境/路径/脚本 (两线共用)
- **H800** `ssh -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>`(密码每会话确认; 代理 7897); GPU 常被外部 DDP 抢, 微秒 kernel 必空闲卡(mem≤50MiB)实测, 偶发竞争产 11–22µs 假尖峰(min-of-N 兜)。
- **Orin**(RSU 边缘卡)`ssh ${V2X_REMOTE_USER}@<PRIVATE_HOST>`(内网不走代理); torch 1.12+CUDA; Tegra 用 tegrastats 查 GR3D。
- **素材(H800 `${V2X_DATA_ROOT}/s2_tvm/`)**: `models/{base,p50,trap25}_backbone.onnx`(Pyramid)+ `models/codriving_cache/{base,p50,p75}_backbone.onnx`(CoDriving)。
- **脚本(本地 `scripts/phase2/` + `scripts/phase1/`)**: Pyramid = `s2_2{d_couple_tile,e_e2e,f_gemm_knob,g_reduce,h_vfe}.py`+`scatter_microbench.py`+`vfe_forward_microbench.py`; CoDriving = `s2_1_codriving_align.py`+`s2_codriving_e2e.py`+`s2_{cod_couple_conv,cod_knob,2g_reduce}.py`+`codriving_segment_timing.py`。
- **结果**: Pyramid `results/s2_2{d,e,f,g,h}*.csv`+`S0b_coupling_clean_4090.csv`+`E8_orin_e2e_fullchain.csv`; CoDriving `results/cod_{e2e,couple_conv,couple_rep,reduce,knob}.csv`+`E_codriving_*`。

---

# Part II — 论文叙事 + 自洽性 + 补实验

## §7 一句话诊断

既有故事主线 = **"P(剪枝)×Q(量化)×D(部署)存在耦合陷阱 → 串行优化次优 → 必须联合搜"**。本阶段新证据对它**既加固也削弱**:
- **加固**: 对齐(÷32)是跨 P/Q/D 三轴的硬约束(trap25 在剪枝 0.29×/量化 1.08×/调度全输); kernel-cliff 真实。
- **削弱**: 耦合强度其实**弱且 model-dependent** —— Pyramid tile 耦合仅"中等有界"(折中 tile 吃掉大部分), CoDriving 6 旋钮**全可分离**; 且过参数化使 AP 轴近塌缩。
- **⇒ 原"耦合普遍存在→必须联合搜"命题硬撑会被审稿人一个反例(CoDriving)证伪。须升级为条件化版本(§9 故事 B)。**

## §8 资产盘点 (所有已确立结论, 正负都列)

**A. 耦合证据(故事核心轴)**
- ✅强: 对齐×INT8 硬耦合(S0b: trap25 非÷32 → INT8 仅 1.08× vs p50 1.32×, 4090 真测)。
- ✅强: prune×quant 耦合(CoDriving: INT8 ΔAP50 随剪枝加深放大 base −0.003→p75 −0.062; 同时 INT8 加速随剪枝缩水 base 1.61×→p50 1.25×)。
- ⚠️中: tile×剪枝宽度耦合(Pyramid E-couple: 16×16×16 在 W48/64 最优 W128 最差, +21% rank-reversal; **但折中 tile 7.3% 吃掉大部分**)。
- ⚠️弱/负: L1/L2 弱, L3/L4 不可判; CoDriving 6 旋钮**全可分离**。
- ★发现: **co-design 耦合强度 model-dependent**(非标算子=grouped bottleneck 显耦合; 标准深归约 conv 可分离)。

**B. 加速证据(系统轴)** — 详见 Part I §2–§4。要点: backbone 编译 Pyramid 10.34×/CoDriving 2.07×; 剪枝 p50 2.07×/CoDriving ~5×, trap25 0.29×; 量化 1.32–1.60×/trap25 1.08×; **Amdahl: backbone 仅 13.6%, fusion neck 55%/16% 是真瓶颈且 TVM 导入 BLOCKED**; RSU 段 VFE >200× / scatter 2–4× / decode 5–6×; 能耗 INT8 省 30–52% J/frame, Orin 异构 GPU∥DLA 1.34×。

**C. 精度证据(AP 轴)★2026-06-19 定稿: AP 轴不塌缩, 公平对比下剪枝平滑伤 AP(2 模型主载体)**
> 演进: 早先"AP 轴塌缩/剪枝无损"是**过度断言, 已推翻**(用户质疑正确)。根因 = 盯 AP50(span 小, 2 位小数看着平)+ 剪枝对比未 iso-finetune。换 **AP70 + 公平协议**后, 平滑单调下降清晰。**主载体 = Pyramid + CoDriving 两条公平曲线**(V2X-ViT 待 iso-budget 重训补)。

- ✅**Pyramid(HEAL)公平 AP-latency Pareto**(stage_a finetuned, base=官方收敛 ckpt 已在 ceiling, DAIR val 1789; 延迟=TRT fp16 clean 空闲卡):
  | anchor | planes | AP70 | AP50 | lat(ms) | ÷32 | Pareto |
  |---|---|---|---|---|---|---|
  | base | 64/128/256 | 0.631 | 0.791 | 1.27 | ✓ | 前沿 |
  | p25 | 48/96/192 | 0.590 | 0.777 | **2.87** | ✗(48) | **被支配**(AP↓+延迟↑) |
  | p50 | 32/64/128 | 0.564 | 0.764 | 1.01 | ✓ | 前沿 |
  | p75 | 16/32/64 | 0.530 | 0.757 | 0.78 | 部分 | 前沿 |
  AP70 平滑单调 0.631→0.530(span 0.10), base 从不被超越。★**p25(48 破 ÷32)被 base 严格支配**(AP 更低+延迟 2.26× 慢=tensor-core 失配)= 对齐耦合在 (AP,lat) 平面的实证 + Gap1"串行踩坑/协同避坑"真实案例。文件 `data/stage_a_ap_real.parquet` + `results/m4_8_dair_*_trt_fp16_clean.json`。
- ✅**CoDriving 公平 iso-budget 对照**(`results/codriving_isobudget_verdict.csv`): **base_isobudget(0% 剪枝 + 同样第二轮 anneal)AP50 0.626 > 所有剪枝档(p25 0.586 / p50 0.61 / p75 0.618)**。⇒ 公平对比下剪枝**轻微伤 AP**; 早先"剪枝↑AP"100% 是训练协议 confound(base_orig 欠训 scratch-30ep, 第二轮退火白捡 +0.066)。曲线浅且中段有噪声(±0.005), 但方向明确(剪枝 < 公平 base)。
- ⇒**"finetune 超越原网络"是 baseline 公平性问题, 非 AP harness bug**: base 训练到位(Pyramid)→ 剪枝永不超越→干净下降; base 欠训(CoDriving/V2X-ViT)→ 第二轮退火白捡→假"超越", iso-budget 修正即消失。AP 计算框架本身没坏(n_tp 随配置变 / 权重验证真加载 missing=0 / 标准 opencood 路径; 原始剪枝 p50 测出 AP50 0.025 崩 = harness 能正确捕捉崩溃)。
- ❌**INT8 轴真实代价仍未测定**(独立于上述剪枝结论): simulated fake-quant 3 变体全近无损但根本失真(FP32 累加 / 注意力 matmul proxy scale 粗糙 / 12 层未覆盖)→ 不可信; 需真 TRT INT8 / TensorRT-ModelOpt。旁证 CoDriving 真 TRT INT8 掉 AP 且随剪枝放大(ΔAP50 base −0.003→p75 −0.062)。
- ⇒ **定稿结论**: **剪枝轴有真 AP-latency trade-off(平滑、公平、2 模型实测)= Gap1 的 AP 载体已成立**; 对齐(÷32)在 Pareto 上制造被支配点(p25); INT8 轴代价待真量化测。

**D. 闭环/驾驶证据(end-task 轴, 未接入)**
- ✅ τ_perc 感知延迟→驾驶分有单调退化曲线; τ_ego(规划延迟)是 null。
- ❌ **加速后的延迟从未接进闭环看驾驶分** —— co-design 加速→驾驶收益的闭环未打通(=Gap5)。

**E. 工具链/方法学(支撑, 非卖点)**: TVM relax 导入+对齐+MS tune 全通; fusion neck 两模型都 BLOCKED(grid_sample); relax 无 INT8 pass(须 BYOC-TRT); 确定性手写 schedule 消融方法。

## §9 三个候选故事线 + 推荐

**故事 A(原线)— "耦合驱动的 P×Q×D 联合搜索框架"**: 耦合陷阱使串行次优, 联合搜达更优 Pareto。❌问题: CoDriving 可分离 + Pyramid 弱耦合 + 过参数化 AP 塌缩 ⇒ "耦合普遍且强"前提站不住, 一个反例就动摇全文。**当前证据不支持强版本。**

**故事 B(推荐)— "条件化 co-design: 何时该联合搜, 何时可分离"**: co-design 价值**条件化于模型算子结构 + 对齐约束**。贡献 = 给出**判据**(何时联合、何时分离)+ 一个 **coupling-aware 搜索器**(检测到耦合→联合, 否则分离省算力), 对齐(÷32)作跨轴硬约束。✅自洽: 把所有负结果(CoDriving 可分离/Pyramid 弱耦合/AP 塌缩)从"缺陷"变成"判据的数据点"。✅顶会卖点: "we characterize WHEN co-design pays off" 比 "yet another co-design framework" 更难被单反例推翻。

**故事 C — "V2X 边缘部署: 瓶颈再定位 + RSU/车端分段加速"**: 真瓶颈是 fusion neck(车端)+ eager 前后处理, 不是大家优化的 backbone。⚠️问题: 手段偏工程, fusion neck 最大瓶颈仍 BLOCKED ⇒ 单独撑不起顶会。

**★推荐 = B 为骨, C 为肉, A 的强证据(对齐)作支柱**
统一标题候选: **"When Does Hardware–Software Co-Design Pay Off for V2X Collaborative Perception? A Coupling Characterization and a Conditional Search Framework"**
一句话贡献: 系统刻画 V2X 协同感知 P×Q×D 三轴耦合, 发现耦合**条件化于算子结构**且**对齐是跨轴硬约束**, 据此提出按需联合/分离的 coupling-aware 搜索, 多模型/多边缘平台真测 Pareto(理想上接闭环驾驶分)。

## §10 论文骨架 (章节 → 用哪些结论)

1. **Intro**: V2X 协同感知边缘部署的 P×Q×D 优化; 既有工作各自孤立优化或假设可分离; 我们问"何时该联合搜"。
2. **Motivation/Background**: 协同感知 pipeline(RSU 感知→车端融合); 边缘约束; QuantV2X/V2X-ViT INT8 崩(崩溃-规避 motivation)。
3. **Coupling Characterization(核心①)**: 三轴耦合系统刻画(对齐×INT8 / prune×quant / tile×宽度)+ **跨模型对比表** → model-dependent 判据。【资产 A】
4. **The Alignment Constraint(核心②)**: ÷32 失配在 P/Q/D 三轴全输(trap25)= 跨方法硬约束。
5. **Bottleneck Relocation(核心③)**: Amdahl 真相, 真瓶颈=fusion/eager 非 backbone; RSU/车端分段。
6. **Conditional Co-Design Search(框架)**: 判据驱动联合/分离搜索 + 对齐硬约束 + 目标对准真瓶颈; searcher_v0。【框架 + Gap1 待补】
7. **Evaluation**: 多模型(Pyramid/CoDriving/+V2X-ViT 待)× 多平台(4090/Orin/H800)Pareto; 联合搜 vs 串行搜增益(Gap1); 理想上闭环驾驶分(Gap5)。
8. **Conclusion**: co-design 价值条件化; 判据 + 框架。

## §11 自洽性检验 (核心 claim × 证据 / gap)

| # | 论文要 claim 的 | 现有证据 | 状态 | 缺口 |
|---|---|---|---|---|
| C1 | 三轴存在耦合 | S0b 对齐×INT8 / CoDriving prune×quant / Pyramid tile | ✅ 够 | — |
| C2 | 耦合 model-dependent(判据) | Pyramid 显耦合 vs CoDriving 可分离(2 模型) | ⚠️ 样本少 | **Gap2**: 需 3+ 模型验判据 |
| C3 | 对齐是跨 P/Q/D 硬约束 | trap25 三轴全输 | ✅ 强 | — |
| C4 | 真瓶颈是 fusion/eager 非 backbone | Amdahl(13.6%)+ RSU/车端分段 | ✅ 够 | fusion 未加速(划界 or 补) |
| C5 | **联合搜 > 串行搜(co-design 真 payoff)** | ✅**首轮机制实证(2026-06-19, H800)**: 对齐感知联合(选 p50)比串行(剪到 25%→trap25 kernel-cliff)快 **7.18×**@−0.026 AP70; pad 局部救援负结果证对齐是全网属性须整网联合搜。`results/gap1_schedule_lut.json` + gap1 design §10.5 | ⚠️ 部分(仅剪枝×schedule 轴, backbone-only/Amdahl, 量化轴待并) | 补: 量化轴 + 全 Pareto + e2e |
| C6 | 精度轴有真 trade-off | Pyramid/CoDriving/**V2X-ViT 全部** DAIR 过参数化 AP 塌缩(2026-06-18 实测) | ❌ **DAIR 任何模型都不支持** | **Gap3 重定义**: 须换非饱和数据集(V2XSet/OPV2V), 非换模型 |
| C7 | 加速→驾驶收益 | τ_perc 曲线存在, 但未接加速 | ❌ 未打通 | **Gap5**: 接闭环 |
| C8 | 边缘真 e2e 加速 | 跨平台分段, **拼不出单一数** | ⚠️ 部分 | **Gap4**: Orin 整段实测 |

**自洽性判断**: 故事 B 骨架(C1/C3/C4)证据已足; 但 **C5(联合搜 payoff)、C6(精度轴)、C2(判据普适)是三根缺腿** —— 不补, 论文停在"刻画了耦合但没证明 co-design 值得做"。

## §12 补实验清单 (按对故事完整性的必要性排序)

**MUST(不补则故事不成立)**
1. **[Gap1] 联合搜 vs 串行搜 Pareto 对比** —— 在显耦合模型(Pyramid)上, 真跑①联合搜 P×Q×D ②串行(先 P 后 Q 后 D), 比最终 (AP,lat,energy) Pareto。证联合搜拿到串行够不到的点(哪怕仅在对齐边界附近)。**这是 co-design "有用"的唯一直接证据, 当前完全缺。** 复用 searcher_v0。⚠️ 设计要点(见 §13): 须在有真 AP trade-off 的载体上做才有区分度, 否则单 latency 轴上联合≈串行。**★完整可执行设计已落盘**: [`../design/gap1_joint_vs_serial_design_v1.md`](../design/gap1_joint_vs_serial_design_v1.md)(三臂消融 S0/S1/S2 + 内外双环 + CHaNAS 式块分解 LUT + iso-AP latency 度量 + 对齐边界 sweet spot + rank-loss 修预测器 + 决策规则), 文献依据 [`study_joint_search_methods_v1.md`](../../references/study_joint_search_methods_v1.md)。
2. **[Gap3 ★2026-06-19 重定义] 把精度轴做实 = 先做真 INT8 测量, 而非先换数据集** —— 复测后认知更新(§8-C): **剪枝侧 AP trade-off 已知弱**(原始崩但 finetune 可恢复=过参数化, 这点 DAIR 上跨模型成立); **真正未解 = INT8 真实 AP 代价**(simulated 不可信)。⇒ Gap3 优先级 = ① **真 INT8 PTQ 测量**(TRT INT8 build / TensorRT-ModelOpt / pytorch-quantization, 真 INT8 GEMM + 静态标定 + 覆盖注意力 matmul) —— V2X-ViT 全模型 TRT 被 fusion 不可 trace 阻塞, 可走 backbone TRT INT8 或 PyTorch 真 PTQ; ② **仅当真 INT8 在 DAIR 也近无损**, 才升级到"换非饱和数据集"(V2XSet 不在盘需下载 / OPV2V 须重训)。**这是 Gap1 的 AP 载体前提**(见 §13)。

**SHOULD(显著加固)**
3. **[Gap2] 第三/四个模型验判据** —— F-Cooper(标准 conv, 预测可分离)/ UniV2X。把"算子结构→耦合强度"从 2 点变成趋势。复用 E-couple/P4 脚本。
4. **[Gap4] Orin 整段 RSU/车端 e2e** —— TRT 串 VFE+scatter+backbone, 拿单一边缘 e2e(消除跨平台拼接的诚实窟窿)。工作量大(VFE/scatter plugin)。

**NICE(顶会竞争力跃升)**
5. **[Gap5] 加速→闭环驾驶分** —— co-design 加速后的延迟接进 CARLA 闭环, 沿 τ_perc 曲线看驾驶分提升。把"快了 X×"升级成"驾驶安全/通过率提升 Y"。基础设施已有。
6. **fusion neck 编译** —— BYOC-TRT GridSample plugin 碰车端 55%/16% 瓶颈; 或诚实划 future work + 量化影响上界。

## §13 ★Gap1 实验设计讨论 (本会话与用户讨论的关键洞察, 未拍板)

> 用户已暂停, 要求先讨论后续实验如何进行再执行。以下是讨论中确立的设计要点, 作为接手依据。

1. **Gap1 与 Gap3 强绑定, 不能独立排**: Gap1 要在 (AP, latency, energy) Pareto 上比联合 vs 串行。但 Pyramid/CoDriving **AP 近常数**(过参数化), 联合与串行的解在 AP 轴挤成一条线, 差异只落 latency 单轴 —— 单轴上"先 P 再 Q"和"联合"很可能几乎一样。**没有真 AP trade-off, Gap1 会得"差不多"的弱结论** ⇒ Gap1 须在有真 AP-latency trade-off 的载体上做。
   - ★**2026-06-19 勘误**(早先"DAIR 无 AP 载体"过度断言已推翻, §8-C): 复测显示 **剪枝真崩 AP(p50 −68.6%)但 finetune 可恢复**(过参数化), **INT8 真实代价未测定**(simulated 不可信)。⇒ Gap1 的 AP 载体前提 = **先做真 INT8 测量**(TRT/ModelOpt); 若真 INT8 显著掉 AP → INT8 轴给 Gap1 真 trade-off, 可在 DAIR 做; 若真 INT8 也近无损 → 才需换非饱和数据集。**剪枝轴**因可恢复, trade-off 弱(联合≈串行=故事 B 数据点)。
2. **联合赢串行的机理前提**: 联合搜赢的唯一来源 = 耦合点, 且必须**串行贪心会踩坑、联合会避坑/利用坑**。但若 P 维单独搜本就不会选非÷32(trap25 本来就慢且 AP 不更好), 串行天然避坑 → 联合无优势。**联合真正能赢的场景 = "单维次优、组合最优"**(如 P 维稍欠剪留对齐余量, 使 Q 维 INT8 能上, 整体更快)。Gap1 成败关键 = 搜索空间里存不存在这种点; 须先确认。
3. **从已有证据预判**: 所有信号(Pyramid 弱耦合/CoDriving 可分离/TVM 消解耦合)指向 Gap1 大概率"联合≈串行, 仅对齐边界小赢"。若如此, **故事 B 反而被直接支撑**(Gap1 价值=标定 co-design 适用边界, 而非证明"必须联合")。这也意味着结果若是联合≈串行, 论文诚实退为 characterization 仍可发表, 但 framing 要随结果调整。
4. **待用户拍板的三件事**(暂停讨论的焦点): ① 论文定位偏"有效 co-design 框架"(系统贡献)还是"刻画 co-design 何时有用/判据"(洞察贡献)? ② V2X-ViT 迁移/训练就绪度(决定 Gap3→Gap1 排序)? ③ 投稿 deadline/目标会议(决定是否冲 NICE 档 Gap5 闭环驾驶分)?

## §14 风险 + 诚实呈现策略

- **负结果是资产不是负债**: CoDriving 可分离 / 过参数化 AP 塌缩 / TVM 消解 TRT 耦合 —— 在故事 B 下全是"判据的数据点"。**切勿硬凑成"耦合很强"**。
- **不夸大 co-design 增益**: 即便补 Gap1, 增益可能仅在对齐边界/特定模型显著。诚实表述"条件化收益"更可信。
- **跨平台口径纪律**: H800(TVM)/4090(TRT)/Orin(边缘)**分平台报, 不拼单一 e2e**; 量化全是 TRT 参考(relax 无 INT8 pass)。
- **AP 轴诚实**: 当前模型 AP 近常数须明说"前沿沿精度轴塌缩", 靠 V2X-ViT(Gap3)才做实。
- **最大单点风险 = Gap1**: 联合≈串行则故事退 characterization。**先跑 Gap1 探明, 再定最终 framing。**

## §15 与既有 method 稿的衔接 / 需修正处
- `stage2_method_zh_v1.md` "耦合陷阱使串行次优"动机段 **需从绝对命题改为条件化** —— 补 model-dependent 判据 + 对齐硬约束, 把 CoDriving 可分离作为"分离适用区"诚实写入。
- `background/00` 的"换 V2X-ViT 做实精度轴(进行中)"= Gap3, 升为 MUST。
- 新增章节(§10 骨架 3/5): Coupling Characterization 跨模型表 + Bottleneck Relocation(RSU/车端 Amdahl)是本阶段新增, 应正式纳入 method 稿。

---

## §16 接手第一步
1. 与用户确认 framing: 故事 B(条件化 co-design)是否接受为主线(§13.4 的三件事)。
2. **优先跑 Gap1**(联合搜 vs 串行搜 Pareto)—— 但先按 §13.1 确认载体(很可能须先把 Gap3/V2X-ViT 就绪)。探明后再定最终 framing。
3. 并行启动 Gap3(V2X-ViT 精度轴)+ Gap2(第三模型验判据)。
4. Gap4/Gap5 视投稿 deadline 与人力排期。
5. 文献支撑: 读 `references/study_joint_search_methods_v1.md`(ALT/CHaNAS/AutoTVM 联合搜索研读), 直接指导 Gap1 的联合搜算法 + searcher_v0 设计。

## §17 纪律 / 诚实边界 (两线共用)
- H800 是 Hopper 数据中心卡 ≠ 边缘标的; 相对/机理结论此坐实, 边缘绝对延迟须 4090/Orin。
- DEFAULT=dlight 是"旋钮关"基线, default/tuned 比 = 开启搜索价值, **非 vs TRT/强手写**。
- backbone-only ≠ 全 pipeline(VFE/scatter/decode/NMS/fusion 非 TVM 调度, Amdahl 稀释)。
- 量化全是 TRT 参考(relax 无 INT8 pass), 部分跨平台(4090 collab2 口径), 不与 H800 TVM 数混加。
- 跨平台/口径**不拼单一 e2e**。
- 任何"加速/对齐/耦合"结论必跨配置(seed/trial)验稳; agent 自报必复核(复跑/读文件/git diff)。
