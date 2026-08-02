# 12_7_14 交接文档 — 专项攻坚:让 TVM 在性能上逼近 TRT(闭合 grouped-conv gap)

> 目的: **专门解决"TVM 无法在性能上逼近 TensorRT"的问题**。攻坚计划(§0–§8)+ **执行结果收口(§9,已完成 2026-07-06)**。
> 日期: 2026-07-06。
> 上游: `11_7_14`(§2.2 = 多 agent 探索 + 批判者复评的问题定位)。
> **★一句话结果(2026-07-06 执行完毕,批判者已复现复核,详见 §9)**: Route B 修好 auto-tune 组合崩(pre-split grouped-conv 的 floordiv 越界)→ **拿到第一个"正确 + 组合真测"的自动后端数 = 3.53ms**(比 TVM-default 15.1× / 比 buggy hand-rewrite 又对又快 1.84× / output_2 干净);距 **TRT-FP16 真 p50 0.907ms = gap 3.89×**(且是**保守上界**,非 TVM 地板)。Route A(CUTLASS BYOC)证否。**结论: 正确 ✅ / 逼近 TRT ❌(未达,但差距定位清晰且可继续缩小)。**

---

## §0 接手须知(自包含)

- **workload**: pyramid_backbone base [64,128,256] fp16,3 输出,**groups=32 的 3×3 grouped conv** 是瓶颈算子(per-group GEMM 的 K 小=36/72/144)。ONNX `${V2X_DATA_ROOT}/s2_tvm/smbo/models/smbo_64x128x256_backbone.onnx`(输入 [2,64,128,256])。
- **口径**: backbone-subnet / input [2,64,128,256] / fp16 / p50 latency / .so 隔离进程 measure。TRT-FP16 baseline=**0.88ms**(★[§9 纠正] 批判者复测真 p50=**0.907ms**,0.88 是历史低采样;下文计划段的 0.88 均以 §9 的 0.907 为准)。
- **TVM env(H800)**: `${V2X_DATA_ROOT}/tvm310/bin/python`(fork 0.20.dev1070,namespace `tvm.s_tir`);LD 必 bash 预设 `export LD_LIBRARY_PATH=$SITE/nvidia/cuda_runtime/lib:$SITE/tvm/lib:$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path):$LD`。
- **纪律**: 真测非 proxy;latency 独占空闲 GPU(GPU4/5/6,util0%/mem≤50MiB,别碰 GPU0-3);tune/build 与 measure .so 隔离;fresh workdir per config；H800 SSH 八进制密码永不明文;不 reboot H800;别过早断言不可能;正确性必须与 TVM-default 对齐(output_2 是已知易错点)。

---

## §1 问题精确定义(from 11_7_14 §2.2,批判者复评后)

- **确立事实**: ① 测量口径公平(证伪"VM overhead 假 gap",纯 kernel gap 更大 8.79×);② 当前手写 im2col rewrite = **6.51ms 但 output_2 错 18%**(既不快也不对);③ auto-tune 被真实、可复现、**非 TensorCore-specific** 的 infra 崩挡住(`MetaScheduleApplyDatabase` 组合 full engine 崩 CUDA_ILLEGAL_ADDRESS;SIMT 也崩);④ **非后端能力上限**——Ansor(OSDI'20)在 MobileNetV2(depthwise=比 groups=32 更极端 grouped conv)已匹配/超 TRT。
- **⇒ 要解决的问题**: 拿到一个 **"正确 + auto-tuned/vendor 级、逼近 TRT 0.88ms"** 的 TVM 数;或诚实定出"某路线到 X ms 触顶"的边界。

---

## §2 已核实的 TVM 能力盘点(2026-07-06 实测这个 fork)

| 能力 | 状态(硬证据) | 对攻坚的意义 |
|------|--------------|-------------|
| **cuDNN BYOC** | ❌ **未编入**(`nm -D libtvm.so \| grep cudnn` = 0 符号;`tvm.contrib.cudnn.conv2d.forward` global func 缺) | 要用须重建 TVM `USE_CUDNN=ON`(H800 无网+fork,高成本高风险,列为备选 Route E) |
| **cuBLAS** | ✅ linked(`tvm.contrib.cublas.matmul` global func 存在) | im2col 后 per-group GEMM 可交 cuBLAS(Route D) |
| **CUTLASS BYOC** | ✅ **linked**(`relax.ext.cutlass` global func 存在;`tvm.relax.backend.pattern_registry`/`get_patterns_with_prefix` 在) | ★**主路线 Route A**:vendor 级 TensorCore conv/GEMM,开源,**不用重建**即可 offload |
| metaschedule/Ansor auto-schedule | ⚠️ 能 tune 但组合崩(Route B 要修) | 纯 TVM 逼近 TRT 的最强证据路线,保 S 轴 |
| 手写 im2col TC rewrite | ⚠️ 6.51ms + output_2 bug(Route C 修) | 正确性基线 |

> 注: relax BYOC 的模块路径在此 fork 与上游不同(`tvm.relax.backend.contrib.*` 直接 import 不到,但 `relax.ext.cutlass` C++ 侧已 linked + `pattern_registry` 在)——接手第一步先摸清此 fork 的 CUTLASS BYOC 正确调用入口(见 Route A 步骤1)。

---

## §3 攻坚路线(按 ROI 排序,每条:假设 / 步骤 / 验收 / 风险)

### ★Route A — CUTLASS BYOC:把 grouped conv offload 给 vendor TensorCore(首选,已 linked)
- **假设**: CUTLASS 的 TensorCore grouped-conv/GEMM kernel 能达到 cuDNN/TRT 级速度;Relax 能 partition 出 grouped conv 子图交 CUTLASS,其余留 TVM。
- **步骤**: ① 摸清此 fork CUTLASS BYOC 入口(`relax.ext.cutlass` + `pattern_registry.get_patterns_with_prefix("cutlass")`,找 conv2d/matmul pattern + `partition_for_cutlass` 等价物;查 fork 源码 `python/tvm/relax/backend/`)。② 对 base backbone ONNX→relax,partition grouped conv(或先 im2col→GEMM 再交 CUTLASS matmul)→ CUTLASS。③ build+run,量 latency + **3 输出正确性 vs TVM-default**。
- **验收**: 正确(output_2 rel err <2%)且 latency 逼近 TRT(目标 <1.5× = <1.3ms)。
- **风险**: CUTLASS grouped-conv 支持可能有限(CUTLASS 强在 GEMM,conv 需 im2col 或 implicit-gemm);grouped/small-K 的 CUTLASS 模板可能未 instantiate;partition pattern 可能匹配不到 grouped conv。
- **框架定位(见 §5 纠正)**: offload 给 CUTLASS = 把 S(schedule)委托给 vendor 自动优化器,**不失框架贡献**(P×Q co-design 正交叠在上面);且经 TVM BYOC 接入,TVM 仍是框架层。**非妥协。**

### ★Route B — 修 auto-tune 组合崩(纯 TVM 逼近 TRT,保 S 轴,最强论文证据)
- **假设**: 崩是 Relax 静态内存 planner 与 auto-tiled grouped-conv footprint 不匹配(11_7_14 §2.2);修了就能跑出 auto-tuned 快 schedule。
- **步骤**(E2/批判者已指): ① 约束 `sample_perfect_tile` 到 **group-divisible 因子**(尊重 group-of-8/32 边界,避免 floor-div 索引越界)。② 或 **手动用 `tvm.tir.Schedule` apply 中选 trace,绕开 `MetaScheduleApplyDatabase`**(就像 hand-rewrite 绕开 Relax planner)。③ 或修 Relax 内存 planner 让它按 scheduled kernel 签名定 buffer。④ build+run 组合 full engine,量 **正确 auto-tuned latency**。
- **验收**: 组合不崩 + 正确 + latency(拿到真数,不管快慢——**先把"未测"变"已测"**)。
- **风险**: ★**用 6_29 铁证警惕过度乐观**——6_29 孤立层 7-14× 但组合后仅 1.5×;auto-tune 的 1224us 孤立和**极可能**在组合后大幅回退。修了崩不等于就快。

### Route C — 修 hand-rewrite output_2 bug + 优化(正确性基线)
- **步骤**: 定位 im2col/restore 里 output_2(最深、spatial 最小)的 18% 漂移(6_29 callsite-aware 未闭);修正后拿"**正确的** 6.5ms"参照;再攻 im2col/layout 转换 overhead(诊断:5 个 group-conv primfunc 占 75%)。
- **验收**: output_2 rel err <2% 且 latency ≤6.5ms。
- **风险**: 修正确性可能反而更慢;6.5ms 本就远离 TRT,优化空间未知。

### Route D — cuBLAS offload im2col-GEMM(中间态,已 linked)
- im2col 后的 per-group batched GEMM 交 **cuBLAS**(已 linked),省手写 schedule。验收同 A。风险:小-K batched GEMM 的 cuBLAS 效率 + im2col/restore overhead 仍在。

### Route E(重成本备选)— 重建 TVM `USE_CUDNN=ON`
- 直接 offload cuDNN(TRT 底层同款)。**风险高**: H800 无网、自定义 fork、重建可能破坏现有 `tvm.s_tir`/tensorize 功能。**仅当 A/B/D 都失败再考虑。**

---

## §4 建议执行顺序 + 并行分工(多 agent + 批判者,沿用本轮成功模式)

1. **并行起 2-3 个探索 agent**: Route A(CUTLASS BYOC)+ Route B(修 auto-tune 崩);Route C(修 bug 拿正确基线)可作第三线或串在 A/B 后。
2. **维护一个批判者 agent**: 每个探索回来后 adversarial 复评(专抓:孤立测量冒充组合、正确性没对齐、latency 口径、把"修了崩"说成"就快")。
3. **主控**: 逐产物独立 stat + 真读 json + 抽查数值核验(不信自报);综合。
- GPU: A/B/C 各占一张空闲卡(GPU4/5/6),measure 独占;tune 重负载注意隔离。

---

## §5 ★框架定位纠正(2026-07-06 用户纠错 — 之前"vendor offload 失 S 轴"是错的)

**先前错误**: 我把"S 轴"当成"我们手写/独占的 schedule 搜索",于是说 offload vendor 库就"失去 S 轴"。**错。**

**正确框架(用户拍板)**:
- **S(schedule/loop 优化)从来就是委托给自动优化器的**,不是我们手搓的(ALT 里 loop 轴也交给 TVM auto-tune)。**cuDNN/CUTLASS/cuBLAS 本身就是自动 schedule 优化器**(cuDNN 有 algorithm selection、CUTLASS profiler 选 tile),只是在"固定 vendor kernel 库"里自动选 vs TVM auto-tune 在"生成搜索空间"里自动选。**两者都是自动调度,offload 没有失去任何东西。**
- **我们的研究 = 筛选/确定最适合"自动调优后端"的 剪枝(P)× 量化(Q)方案**。P×Q 我们搜;S 委托自动后端(TVM-auto **或** vendor,皆可)。
- **co-design 新意 = 搜 P×Q 时把后端自动调度后的真实 latency 当目标**(哪个剪枝/量化配置好,取决于自动后端能多好地调度它)。三臂消融(joint/serial/compression-only)测的是"要不要把后端自动调度 latency 反馈进 P×Q 搜索",**与后端是 TVM-auto 还是 vendor 无关**。
- ⇒ **框架不需要 TVM 自己打赢 TRT**:S 轴用**最好的自动后端**,我们的 P×Q co-design **正交叠在上面**。CUTLASS/cuBLAS 经 **TVM BYOC** 接入 → **TVM 仍是图编排/框架层**(不违"框架在 TVM"铁律),vendor 只作被委托的 conv kernel。**这是 sweet spot,不是妥协。**

**唯一保留的诚实增量**: TVM auto-tune 比 per-op vendor 库多覆盖**跨算子 fusion / 剪枝产生的非常规 shape**——所以 TVM-auto 或 "BYOC+fusion" 有**额外**价值,但这是增量,非"用 vendor 就丢核心"。

**⇒ 对攻坚的影响**: Route A(CUTLASS BYOC)和 Route B(TVM auto-tune)**不是"妥协 vs 纯粹"的对立**,而是**"哪个自动后端"的并列选项**——都合法,都保留框架贡献(P×Q co-design)。目标就是拿到"正确 + 逼近 TRT"的自动后端 latency,喂进 P×Q 搜索。

---

## §6 验收标准(本攻坚何时算成功)

- **成功**: 拿到一个 **正确(3 输出 rel err <2%)+ 逼近 TRT(<1.5× ≈ <1.3ms)** 的 TVM 数(A 或 B 任一)。
- **诚实收口**: 若某路线触顶,给"该路线到 X ms、瓶颈在 Y"的真测结论(不虚报)。
- **底线**: 在拿到"正确 + 快"的 TVM 数前,**7.4× 不可作干净数据引用**(11_7_14 §2.2 已定)。

---

## §7 关键资源 / 路径 / 命令

- 现成脚本: probe `scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py`;metaschedule worker `scripts/stage2_original60_tvm_artifact_worker.py`;full-tune harness `framework/trt_baseline/tvm_fulltune_fp16.py`;TRT profiler `framework/trt_baseline/trt_profile_v1.py`。
- 本轮产物(H800 `${V2X_DATA_ROOT}/s2_tvm/la_fulltune/`): `tvm_groupconv_rewrite_base_result.json`(6.51ms)、`fairness_audit_result.json`、`tvm_autotune_feasibility_result.json`、`critic_review_result.json`、`full_base_fp16_4000/tuned_fp16.so`(tuned DB,组合崩)、`raw/base_64_128_256_full_engine_group_conv_rewrite/*.so`。
- CUTLASS BYOC 摸入口: `nm -D libtvm.so | grep -i cutlass`;`tvm.get_global_func("relax.ext.cutlass")`;fork 源码 `${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/tvm/relax/backend/`。
- 6_29 rewrite 前史: `progress/6_29/{6,7,8,9}_6_29_*.md`。

---

## §8 一句话给接手 agent

**先摸清此 fork 的 CUTLASS BYOC 入口(Route A)并同时修 auto-tune 组合崩(Route B),两条并行、批判者复评,目标=拿到第一个"正确+逼近 TRT"的自动后端 latency 喂进 P×Q 搜索;警惕孤立测量冒充组合(6_29 铁证:7-14×→1.5×)。★框架定位(§5):S 是委托给自动后端(TVM-auto 或 vendor,皆经 TVM BYOC),offload 不失框架贡献,Route A/B 是"哪个自动后端"的并列选项非对立。**

---

## §9 执行结果收口(2026-07-06,两路已跑完 + 批判者已复现复核)

> 主控独立核验(读 json 真值 + 亲跑 measure)+ adversarial 批判者(独立复现)双重把关。grouped-conv backbone,输入 [2,64,128,256] fp16,H800,**全部组合 .so 真测**。

### §9.1 同 backbone latency 谱(唯一可比口径)

| 后端 | latency (p50) | 正确性 | 口径 | 相对 |
|---|---|---|---|---|
| TVM-default fp16 | 53.2 ms | ✅ | 组合 | 1.00× |
| hand-rewrite(im2col TC, 6_29) | 6.51 ms | ❌ **output_2 错 17-18%** | 组合 | 8.2× |
| **Route B auto-tune(split-fix)** | **3.53 ms** | ✅ **output_2 干净(mean 0.34%/p99 3.4%)** | **组合 .so 真测** | **15.1×** |
| **TRT-FP16(对手 baseline,非我们后端)** | **0.907 ms**(真 p50) | — | — | 58.7× |

→ **gap(our-TVM auto-tune vs TRT)= 3.89×**(不是旧写的 0.88ms/4.0×;0.88 是历史低采样,批判者纠正)。

### §9.2 Route B(决胜路,纯 TVM auto-tune)—— 成功拿到第一个"正确"自动后端数

- **崩根因**:auto-tune 组合 build 崩(CUDA_ILLEGAL_ADDRESS)= grouped-conv `pad_temp[.., v_ff//4*4+v_rc, ..]` 的 **floordiv 通道索引**,MetaSchedule 通用 tile 不尊重 group 边界 → 静态 buffer-region 推断出欠尺寸 scratch → 运行期越界。
- **修法 = pre-split**:tune 前对 5 个 grouped-conv PrimFunc 在 `group_conv2d_nchw` block 的 ff loop 上 `sch.split(ff,[None,C])`,把 group 边界从 floordiv 变成 **affine loop-split**。
- **结果**:4000-trial 全量 tune + **full-engine 组合 build 无崩**(crash_resolved=True,**不是孤立核**,是组合尺度)→ 组合 .so 真测 **3.53ms**,3 输出对齐 TVM-default,**真修好了 hand-rewrite 的 output_2 17% bug**。
- 产物:`route_b_autotune_fixed_result.json` + `route_b_work/full_split_fixed.so` + measure 脚本 `route_b_work/route_b_measure_fixed.py`(★跑前须 `export LD_LIBRARY_PATH=$(ls -d .../site-packages/nvidia/*/lib|tr '\n' ':')$LD` 否则 libcudart.so.12 undefined symbol)。

### §9.3 Route A(CUTLASS BYOC)—— 证否,负结果成立

- build 不出可跑 engine,两个独立 blocker:**B1 架构级**=CUTLASS conv2d codegen **不读 groups 属性**,groups=32 瓶颈会被当 dense 算错 → 没法正确 offload;**B4 fork 链接缺陷**=编译好的 CUTLASS 函数进不了 Relax VM(穷尽 5 种打包变体)。
- **关键对照**:**cuBLAS BYOC 同一条 Relax-VM 路径正确跑通(l2_rel_err 2e-4)** → 证 BYOC→VM 通路本身好、失败是 CUTLASS 特有;但 im2col→cuBLAS 的小 per-group GEMM(K=36/72/144, M=N=4/8/16)远低 TensorCore tile 粒度,**物理上追不上 0.907ms**。
- 产物:`route_a_cutlass_byoc_result.json` + `route_a_cublas_smoke.json`。

### §9.4 批判者裁定(adversarial,亲自在空闲 GPU7 复现 3.5427ms,差 0.35%)

**3.53ms「正确 + 组合真测」站得住、可发表,无 pro-TVM 夸大。** 6 攻击点:①孤立冒充组合→**CONFIRMED 组合**(跑 `vm["main"]` 全图 34 funcs 返 3 输出);②latency 口径→**CONFIRMED**(p99/p50=1.02 稳);③正确性 oracle→**CONFIRMED**;④收敛→**PLAUSIBLE-有保留**;⑤TRT 公平→**CONFIRMED + 数值纠正**;⑥CUTLASS 负结果→**CONFIRMED**。

**两处修正(★都对 TVM 不利 = 真 gap 更小,反而利好"TVM 能逼近 TRT"论点)**:
1. **TRT 真 p50 = 0.907ms → gap 3.89×**(非 0.88/4.0×)。
2. **3.53ms 是 gap 上界、非 TVM 地板**:5 个 grouped-conv workload(最慢核)因 split-fix **schedule 空间坍缩、每个只 tune 8 trials**(非真 4000-trial 富搜索)+ **组合税 3.3×**(per-workload 最优核之和才 1.07ms,组合后 3.53ms 全耗在未 tune 的 cast/pad/reformat glue + TRT 会 fuse 掉的 layout transform)→ 更好的 rewrite/fusion 大概率把 gap 继续缩小。
- 产物:`critic_review_route_ab.json`。

### §9.5 定论 + 下一步

- **差距定位**:不在测量口径(组合真测,批判者复现)、不在正确性(output_2 干净),**在 grouped-conv 的 TensorCore 利用**(小 per-group GEMM 撞物理墙,Route A cuBLAS 与 Route B 同因)+ **组合税**(未 tune 的 glue/layout)。
- **对论文**:纯 TVM auto-tune = 唯一拿到"正确 + 大幅加速"的自动后端(3.53ms,15.1× over default);距 TRT 3.89×**且是保守上界**;**别把"未达 TRT"手搓成"TVM 追不上 TRT"定论**——这是"grouped-conv TC 利用 + 组合税未解",非"不可能"(见 memory `feedback-no-premature-impossible`)。逼近 TRT 需专门解 grouped-conv 张量化 + 消组合税(fusion),非通用 tune。
- **停止条件已满足**:两路跑完、批判者复现复核、结果落 memory `project-tvm-vs-trt-two-route-result` + HANDOFF.md + 本节。**/goal(专项攻坚)已清。**

---

## §10 下一步专项:消除组合税(让 TVM 真逼近 TRT 的最大杠杆)

> 背景:批判者拆解 3.89× gap(H800 backbone-only,TVM 3.53ms vs TRT 0.907ms)= **组合税 3.3×**(per-kernel 最优之和 1.07ms → 组合 .so 3.53ms)+ 残余 grouped-conv 张量化不足。**组合税是最大且最可能拿到的一块**,单这一项目标把 3.53 → ~1.2–1.5ms。

### §10.1 组合税从哪来(诊断锚点)
组合 .so 比"各核最优之和"多出的 2.46ms(3.53−1.07)几乎全在**核之间的胶水**:
- **cast**:ToMixedPrecision 在算子边界插的 fp16↔fp32 转换,未融进 conv epilogue。
- **pad**:grouped-conv 前的 pad_temp 显式物化(split-fix 后更显)。
- **reformat / layout_transform**:相邻 kernel layout 不一致(NCHW↔NCHWc↔NHWC)时 TVM 插的重排,TRT 会 fuse 掉。
- **bias-add / relu / residual-add**:未融进上游 conv 的 epilogue,单独起 kernel。

### §10.2 执行计划(三步,H800,fresh workdir per width per process)
1. **算子融合(P0,最大头)**:
   - Relax 层:`FuseOps` + `FuseTIR` 后检查融合边界;对 conv→bias→relu→(residual add) 写/启用 **epilogue fusion pattern**,把 elementwise 尾巴并进 conv。
   - 消 cast:让 ToMixedPrecision 的 cast 落在 conv 输入/输出 epilogue 内(或改用 conv 直接吃 fp16 累加 fp32 的 fused 形态),不单独起核。
2. **layout 统一(P0)**:
   - 全网锁一个一致 layout(优先 NCHWc / NHWC,配合 TensorCore 友好)使 kernel 间**零 reformat**;`ConvertLayout` 一次性转好,把 layout_transform 从热路径挪到边界。
   - 不可避免的 transform 也纳入 MetaSchedule tune(别留 default)。
3. **重测 + 归因**:组合 .so 隔离进程 measure p50(同口径 [2,64,128,256] fp16,GPU 独占);用 nsys kernel-sum vs wall 复算新的"组合税"倍数;对齐正确性 vs TVM-default(★output_2)。

### §10.3 判据 / 停止条件
- [ ] 组合 .so p50 从 3.53ms 降到 **≤1.5ms**(组合税从 3.3× 压到 ≤1.4×),正确性 output_2 仍干净(mean<1%)。
- [ ] nsys 证实 cast/reformat/pad 核数量显著下降(融合真发生,非数字凑)。
- [ ] 若拿到 ≤1.5ms → gap 收窄到 ~1.6×(vs TRT 0.907),连同 grouped-conv 张量化(第二杠杆)一起再逼近。
- [ ] 全程 fresh workdir per width per process、tune/build 与 measure 分进程(.so 隔离防 CUDA_ILLEGAL_ADDRESS)。

### §10.4 风险 / 注意
- 融合可能触发新的 build 崩(融合后的 fused-grouped-conv 又碰 floordiv/buffer-region)→ 沿用 §9.2 的 pre-split 修法,融合与 split-fix 需共存验证。
- layout 统一可能改变 grouped-conv 的 group 边界表达 → 融合后仍须对齐 TVM-default 正确性(不是只看不崩)。
- **口径纪律(本轮教训)**:所有 latency 只在 {同 device=H800、同 scope=backbone-only、同 shape=[2,64,128,256]、同 precision} 下比;**严禁跨 scope(collab2-body 含融合)/跨设备(4090)混比**(见 §9 与 audit §2.2:4090 collab2-body FP16=1.269ms 是含融合的更大 scope,≠ H800 backbone-only 0.907ms)。
