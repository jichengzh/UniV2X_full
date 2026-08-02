# HANDOFF — route2 / S2.2b 逐维度耦合验证 (v1, 2026-06-17)

> **接手先读**: 本页 + [dims_hardware_v4.md](../design/dims_hardware_v4.md)(维度定义 §2 / 闸门 §4.1 / **逐维度验证协议 §4.2** / 集成判定 §5)+ memory `project-hwsw-codesign-route2` + [HANDOFF_tvm_migration_route2_v1.md](HANDOFF_tvm_migration_route2_v1.md)(TVM-on-H800 env 配方 + S2.0/2.1/2.2a 全细节)。
> 本页自足: 含现状、已 PASS 前置、conv 清单、S2.2b 锐化后的实验设计与具体做法、环境命令、纪律、接手第一步。

---

## §0 一句话现状 + 下一步

route2(用 TVM 把编译调度变成可搜硬件轴, 挣 co-design 标签)。**S2.0/2.1/2.2a 已 PASS** + **S2.2b 全部完成**(2026-06-17: screening 26 + grouped 5 + 跨seed 9 + 选项①16/②2/③3 = 61 tune 点, 见 §2.5/§2.5b)。

**已得定论**: ① dense 1×1 全宽度无 D1 悬崖(TVM-WMMA-K16 消解 TRT ÷32 悬崖); ② L3/L4 平坦负结果, L2-grid/延迟=跨seed证实噪声③; ③ grouped g=32 全掉 SCALAR=候选耦合#1(量化加速层结构依赖, 非TVM专属); ④ force-K32 blocked(工具链)。**选项①②③收口(§2.5b)**: ①真实 trap conv **16/16 全留 WMMA 零 fallback**, mitigation CONFIRMED(定性, 对照 TRT 1.08×悬崖); ②Q/DQ 被 fuse 进 epilogue, TRT P4 融合断裂耦合在 TVM **不复现**(NEGATIVE); ③matmul 重试仍 BLOCKED(连对齐 K=64 都 engage 不了 MMA, 工具链非硬件)。

**★跨实验诚实综合(核心, 影响 §5)**: ①对齐悬崖、②Q/DQ 融合——**两个 TRT 耦合在 TVM 都被"消解"而非"复现"** ⇒ route2 价值正浮现为 **"TVM 调度/融合灵活性=缓解器, 移除 TRT 锁死的耦合"(§5判据2强)**, 而非原假设 **"TVM 揭示 TRT 看不见的新耦合"(§5判据1弱: 反复发现 TVM 去耦合, 未发现新耦合)**。合法但**不同于原命题**, 已写进 §5。

**★★下一步 = 解决用户 2026-06-18 指出的两个核心缺口(见 §8, 当前最高优先)**: ① S2.2c 的旋钮加速是 **per-operator 微基准**(2.3×–67×), **未反映到端到端**(Amdahl: 小 op 上的 67× e2e 几乎不可见); ② S2.2c **脱离了耦合主线** —— 只证"旋钮 X 加速算子 Y", 未证"**最优旋钮配置依赖软件决策(剪枝/量化)**"(=co-design 耦合的本质)。⇒ 转做 **E-e2e(端到端旋钮价值+Amdahl 归因)** + **E-couple(HW×SW 耦合: 旋钮×软件交互扫描 / transfer-penalty 矩阵)**。E-couple 是 route2 GO/NO-GO 的决定性实验。详见 §8。

旧选项(降级, E-couple 出结论后再回看):
- A. "缓解器"论点收口(①②mitigation 写成可发表机理 + 4090/Orin 延迟坐实)。
- B. 找 §5判据1 新耦合(D3 大conv/L4 流水/全图 INT8)。
- C. 修通 force-K32(参考 TVM testing MMA schedule)。

---

## §1 已 PASS 前置 (别重做) + 环境

- **S2.0 工具链 go/no-go PASS**: MetaSchedule 全链路(`tvm.s_tir.meta_schedule.relax_integration.tune_relax`/`compile_relax`)跑通; INT8 tensorize 可表达(`MMA_i8i8i32`/WMMA i8)。
- **S2.1 数值对齐 PASS**: relax 导入 backbone vs ORT maxdiff 2.4e-6。
- **S2.2a INT8 TC 启用核验 PASS**: 合成 int8 dense 1×1 conv(Cin 64/48/48→pad64)tune 后**全部真上 WMMA(16×16×16, K=16)** tensor-core(scheduled-TIR `tvm_mma_sync` 复核, 非关键词假阳)。
- **环境(H800, 全坑已解, 别重走 — 细节见 route2 handoff §①)**: env=`${V2X_DATA_ROOT}/tvm310`(系统 py3.10 venv); TVM **0.20.dev1070 改版 Unity**(`tvm.tir`→`tvm.s_tir`/`tvm.tirx`; MetaSchedule 在 `tvm.s_tir.meta_schedule`); 跑前必 `export PATH=/usr/local/cuda-12.2/bin:$PATH` + `export LD_LIBRARY_PATH=$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path)`; xgboost 已装。SSH 非交互每命令显式 export 代理 7897。

---

## §2 关键已知 — conv 清单 + WMMA-K16 发现 → 锐化的实验问题

### §2.1 真实 backbone conv 清单(2026-06-17 从 onnx 抽, 实验直接照此建合成 conv)

**p50_backbone(对齐组): 所有 conv 都 ÷32 且 ÷16 对齐**, 无失配。grouped(g=32)= 64/128/256。

**trap25_backbone(失配组): 失配仅集中在 48 通道的 dense 1×1 conv**:
| 失配 conv (g=1, 1×1) | 失配处 | ÷32 | ÷16 | 数量 |
|---|---|---|---|---|
| Cin=48 → Cout=96 | **Cin=48** | ✗ | ✓ | 3 |
| Cin=48 → Cout=192 | **Cin=48** | ✗ | ✓ | 1 |
| Cin=64 → Cout=48 | **Cout=48** | ✗ | ✓ | 1 |
| Cin=96 → Cout=48 | **Cout=48** | ✗ | ✓ | 3 |

**关键**: trap25 的 grouped conv(g=32)= 96/192/384, **全部 ÷32 对齐** ⇒ **groups=32 不是悬崖所在**(纠正旧 R2 担心)。悬崖**纯粹**来自 48 在 Cin 或 Cout 的 dense 1×1 conv。
对齐对照(从 p50 取同角色): `64→128`/`128→64`/`64→64`/`64→32`(全 ÷32)。

### §2.2 WMMA-K16 发现 → 锐化的核心实验问题

- [文献] WMMA int8 = **K=16(÷16)**; PTX mma.sync int8(Hopper 高性能路径)= **K=32(÷32)**; TRT INT8 = NCHW32(÷32)。
- **48 是 16 的整数倍(48=3×16)但非 32 的整数倍** ⇒ 在 **K16 路径上零碎片**(S2.2a 已证 48 在 WMMA 上无悬崖), 悬崖**只在 32-粒度路径(mma-K32 / NCHW32 / IMMA)出现**。
- ⇒ **S2.2b 的核心问题(锐化)**: TVM tuner 默认给这些 int8 conv 选 **K16(无悬崖)还是高性能 K32(48 撞墙)**? **强制走 K32 路径**时, 48 是否必须 pad→64 才能 tensorize(而 64 不用)? 这才是"剪枝宽度(48)× 量化(INT8)× intrinsic 粒度"耦合的干净显形, 以及"pad 48→64 协同缓解"的真 demo。
- **诚实预期(R2)**: D1 在默认 WMMA-K16 下很可能**无悬崖**(TVM 的 intrinsic 灵活性天然绕过 TRT 的 NCHW32 ÷32 悬崖)。若如此, 这本身是合法发现("TVM 选更细粒度 intrinsic 缓解了 TRT 的硬悬崖"), 但**必须诚实区分"发现新陷阱" vs "证明 TVM 绕过旧陷阱"**; 同时把重心放到: ① 强制 K32 路径复现悬崖 + pad 缓解(可控 demo); ② 更细的 L1/L3/D3 在 48 vs 64 上的调度产物差异。

---

## §2.5 S2.2b 已得结果 (2026-06-17, 39 tune 点, supervisor 可复核)

**资产**: 脚本 `scripts/phase2/s2_2b_{screen,detect,forcek32}.py`(本地+H800 `${V2X_DATA_ROOT}/s2_tvm/`); 数据 `results/s2_2b_screenA_{int8,fp16}.csv`(26) + `s2_2b_{seed,grouped}.csv`(9+5); scheduled-TIR dump 在 H800 `ms_work_2b_*/tir_*.txt`。
**工具**: screen 单点 = tune(固定seed)→`MetaScheduleApplyDatabase`→解析全9维调度产物(改版Unity文本: `T.sblock_alloc_buffer`/`T.sblock_attr`/`T.thread_binding(T.int64(N),thread=)`/`T.axis.reduce`/`scope="shared.dyn"`)+ DB `run_secs` 干净延迟(弃VM循环噪声)。

**① Role A screening — dense 1×1, Cin=Cout=width, 13宽度×{int8,fp16}**(宽度覆盖 ÷32/仅÷16/仅÷8/奇质数):
- **D1 无悬崖(强负结果)**: 26 点**全 WMMA 16×16×16**, 零 fallback。TVM 把 K 补到 ceil(width/16)×16(red_ext 列)照常上 TC。"48 掉 TC"假设全宽度证伪 ⇒ TVM intrinsic 灵活性天然绕过 TRT ÷32 悬崖(=R2, 属"绕过旧陷阱"非新陷阱)。
- **L3 storage_align 平坦**: int8 恒 `32,16` / fp16 恒 `32,8`(随 dtype, 不随宽度)。**L4 software_pipeline 全 0**(1×1 K 小)。负结果。
- **D-a 延迟"悬崖"= 探测器假阳**(小核 launch 开销主导, per-FLOP 单调降, 非悬崖)。

**② 跨 seed 复核(int8 w48/64/112 × seed1/2/3)→ 锁定 class③ 噪声**: grid_sig **3/3 全不同**(100% seed 不稳), 延迟 CoV 6–12%(≥多数宽度间 delta)。⇒ L2-grid + 延迟非单调 = tuner 噪声, 非耦合。

**③ grouped 角色(候选耦合#1)— g=32 3×3, 宽度32/64/96/128 + dense3×3 g=1 控制**: grouped **全 SCALAR(n_mma=0, 不上 TC)**; dense 3×3 g=1(K=576)= WMMA。根因=每组 GEMM N=Cout/g<16 填不满 fragment。⇒ INT8 TC 加速**层结构依赖**(grouped/depthwise 量化无 TC 收益, dense 有), prune 改 dense/grouped 占比即改量化收益。⚠️**TRT 同样无法 TC grouped → 非 TVM 专属, 属框架须编码的共享硬件现实(弱满足§5判据1)**。

**④ force-K32 反事实 demo = blocked(timebox 停)**: `s2_2b_forcek32.py` 建仅 MMA-K32 组的 space(手注册缺失的 `mma_store_16x16_i32_shared_simple_`)。结果连对齐 w64 都掉 SCALAR(11µs vs WMMA 4.4µs)→ 手配 MLT-TC(MMA 组)未 engage tensorize(MMA 需特定 warp/loop+postproc, WMMA 默认全 wired)。非硬件不可能, 是插桩未通。退路见 §0-选项③。

### §2.5b 选项①②③ 收口结果(2026-06-17 夜, 用户拍板①后直推②③, H800)

**资产**: `results/s2_2b_opt1_mitigation.csv`(16) + `results/opt{1,2,3}.log`; 脚本 `scripts/phase2/s2_2b_{screen,mm_k32,d4_fusion}.py`。

**① WMMA-K16 mitigation 坐实 = CONFIRMED(定性)**: 真实 trap25 失配 conv(48 在 Cin 或 Cout) vs 对齐对照, int8+fp16 共 16 点。**16/16 全留 WMMA, 零 fallback**(`n_mma=1` 全程)。trap/aligned 延迟比散在 0.71–1.57×(单 seed+tiling 噪声, 无系统性悬崖), 对照 **TRT S0b trap25 INT8 锁死 1.08×悬崖**。⇒ **TVM 选 WMMA-K16(÷16)让所有 TRT 会撞墙的 48 通道 conv 都保住 TC** = intrinsic 粒度是被协同配置的硬件旋钮(§5 判据2 缓解)。**caveat**: per-conv 合成(非全图 INT8, R1); H800 非边缘(4090/Orin); 单 seed 延迟噪声~10% ⇒ 论点是**定性"无 fallback"**, 不是具体加速数。

**② D4 fusion×Q/DQ = NEGATIVE(TVM 也消解此耦合)**: conv→relu→conv vs conv→relu→[quant→dequant]→conv。两 regime **都 2 kernel**; QDQ 的 quant/dequant elementwise 被 **fuse 进 conv epilogue**(`fused_conv2d_relu_multiply_round_clip_cast_cast_multiply`), 延迟仅 +5%(16.45→17.29µs)。⇒ **TRT 上 Q/DQ 打断融合(P4)的耦合在 TVM relax 不复现**——TVM 融合更激进, 把 QDQ 吸进 epilogue 省掉额外访存。**caveat**: 简化 elementwise QDQ proxy(无真 per-channel 量化 pass)。

**③ force-K32(matmul 重试)= 仍 BLOCKED(工具链, 非硬件)**: 换干净 int8 matmul(GEMM 是 MMA 标准场景), 仍 **连对齐 K=64 都掉 SCALAR**(6.12µs), K=48 SCALAR(9.25µs), K=48→pad64 SCALAR(5.79µs)。⇒ 手配 MLT-TC(MMA-K32 组)在本 build 根本 engage 不了 tensorize(连最易的 GEMM 都不行)。**真因 = MMA 的 warp 级绑定(32线程协作 ldmatrix+mma)+ RewriteTensorize postproc 需要 WMMA 默认提供而手配缺的设置**。反事实未能演示。**侧观察(非 TC)**: scalar 路径上 K=48-pad-64(5.79)比 K=48(9.25)快 = pad 在 layout 层也有益, 但非 TC 悬崖。

**★跨①②③ 的诚实综合(影响 §5 判定)**: ① 对齐悬崖、② Q/DQ 融合断裂——**两个 TRT 耦合在 TVM 上都"被消解"而非"被复现"**。⇒ route2 的真实价值正浮现为 **"TVM 的调度/融合灵活性是一层缓解器, 移除了 TRT 锁死的耦合"(强支持 §5 判据2 缓解)**, 而非原假设的 **"TVM 揭示 TRT 看不见的新耦合"(§5 判据1 被削弱: 反复发现 TVM 去耦合, 没发现新耦合)**。这是合法但**不同于原命题**的论点, 必须诚实写进集成判定。

---

## §2.5c T1 旋钮逐个延迟消融 (2026-06-18, 用户拍板"证明每个硬件旋钮对推理速度的影响", 先 T1 6 维)

**资产**: `scripts/phase2/s2_2c_{knob_ablation,manual_gemm}.py`; `results/s2_2c_manual_gemm.csv` + `results/2c_w1_ruledrop.log`。

**方法学(关键, 接手必读)**:
- ❌ **rule-drop 消融(去一条 schedule rule 重 tune)被搜索预算混淆**: cuda_full(全规则)在 64 trials 下**反而最慢**(151µs), 去规则更快——加规则=扩搜索空间, 固定预算采样更稀, "best-of-64"更差。**delta≠旋钮价值, 弃用**(除非给每 spec 大到收敛的预算)。
- ✅ **确定性手写 schedule 消融(无 tuner)= 正解**: 手搭一个 GEMM schedule, 每次只加一个旋钮, 编译实测, 边际延迟降=该旋钮价值。无搜索=无预算混淆。
- 🔑 **两个工具解锁**: ① 手写 schedule 编译前必给 PrimFunc 打 **`tirx.is_scheduled=True`** attr(`func.with_attr`+`mod.update_func`), 否则 `tvm.compile` 默认 relax pipeline 的 **dlight** 会重排你的 schedule 报 `transform_block_layout` 错; ② shared 暂存(`cache_read`+`compute_at`)**必须配 cooperative fetch**(shared-load 循环 `fuse`+`split`+bind 到 threadIdx.y/x), 否则每线程串行 load = **24× 慢的假象**。

**T1 旋钮延迟效应表(4/6 干净证实)**:
| 旋钮 | 方法 | workload | OFF | ON | 加速 | 状态 |
|---|---|---|---|---|---|---|
| **D1 tensorize** | rule TC vs 非TC | conv 128ch int8 | 55µs(非TC) | 23.5µs(WMMA) | **2.3×** | ✅ 证实 |
| **L1 shared 暂存** | 手写累积 | GEMM 2048³ | 3488µs(naive) | 2391µs | **1.46×** | ✅ 证实 |
| **L2 线程绑定** | 手写 | GEMM | 无绑定=GPU 不可运行 | 411/3488µs | foundational | ✅ 必需 |
| **L5 跨线程 reduction** | 手写 | GEMV M=N=8 K=32768 | 673µs(serial-k) | 9.99µs(cross-thread) | **67×** | ✅ 证实 |
| **L3 storage_align** | 手写累积 | GEMM 2048³ | 2391µs | 2697µs | 0.89×(略伤) | ⚠️ 参数依赖 |
| **L4 software pipeline** | 手写累积 | GEMM 2048³ | 2391µs | 2709µs | ~1× | ⚠️ 参数依赖 |

**诚实结论**: D1/L1/L2/L5 **单独施加都对延迟有可测显著影响**(2.3×–67×), 证实"调硬件旋钮影响推理速度"。L3/L4 旋钮可施加但**延迟收益依赖 per-knob 调参/workload**(手设 align factor=32/offset=8、2-stage pipeline 未显效; autotuner 会选这些参数)。**待补**: L3 用 bank-conflict-heavy + 正确 pad; L4 用 memory-bound 深 K + 多级流水; **T2/T3(D2 layout/D3 tiling/D4 fusion)三维未做**。

---

## §3 S2.2b 具体做法 (耦合发现 campaign + 逐维度深探)

> ★**核心: 发现优先, 不是只验证已知对齐陷阱。** 完整方法 = dims_hardware_v4 §4.3(campaign)+ §4.2(单维度验证协议)。两阶段: A. screening 宽扫筛候选; B. 逐候选深探+三分类。**优先静态测**(scheduled-TIR 解析, 对脏 GPU 免疫), ncu 动态为辅(需相对空闲 + 权限, 见 §6)。

### 阶段 A — screening 宽扫(主动找候选, 覆盖全 9 维)

1. **建合成 conv 网格**(`BlockBuilder`, 复用 `s2_2_int8_cliff.py` 的 `build_int8_conv`; 扩成参数化扫描): 代表性 conv 形状(1×1 dense + 3×3 grouped g=32, 取 backbone 真实角色)× **软件轴**:
   - **S1 剪枝宽度**: 故意覆盖各对齐边界 —— ÷32(32/64/96/128) / 仅÷16(48/80/112) / 仅÷8(24/40/56) / 奇·质数(33/50/67)。
   - **S3 精度**: INT8 / FP16。
   - (S2 per-stage / S4 Q-DQ / S5 roofline 留阶段 B 或 S2.3, 因需多 conv/全图。)
2. **每点记录**: 延迟(脏 GPU 仅作粗信号) + **全维度调度产物**(`MetaScheduleApplyDatabase`+`mod.script()` 解析: D1 intrinsic / L1 SMEM bytes / L2 bind extent / L3 storage_align factor / L4 流水 stage / L5 reduction 结构 / D2 layout+pad / D3 tile 因子)。脚本基座 = `s2_2_verify_tc.py`(已能解析 `tvm_mma_sync` 等 builtin)。
3. **跑信号探测器**(dims_hardware_v4 §4.3): D-a 延迟悬崖/非单调(尤其**非÷32 的宽度**) / D-b 调度产物在某宽度突变 / D-c (W,INT8) vs (W,FP16) 结构不同 / D-d SMEM 跨 occupancy 边界。**输出候选耦合表: 维度 × 软件轴 → {疑似耦合 / 平滑无 / 噪声}**。

### 阶段 B — 逐候选深探 + 三分类

4. 对阶段 A 标出的每个候选, 按 §4.2 隔离深探: 固定其余、强制该维度选择(`s_tir.Schedule` 手工构造)、跨 ≥2 配置(trial 64/256 或 seed)验稳、给体系结构解释。**三分类**: ① 已知对齐重现(D1/D2 ÷32) / ② **新耦合** / ③ auto 噪声。
5. **已知对齐陷阱深探(候选之一, 非全部)**: 用 `s_tir.Schedule.tensorize` 显式指定 —— K16 路径(期望 48/64 都成功) vs K32 路径(`MMA_i8i8i32`, 期望 64 成功 / 48 失败或需 pad)⇒ 悬崖显形; + pad 48→64(`pad_einsum`)缓解。
6. **诚实预期(R2)**: D1 默认 K16 很可能无悬崖(TVM 灵活性天然绕过 TRT NCHW32)→ 这是合法发现但属"绕过旧陷阱"; **真正的增量在 screening 是否在 L1/L3/L4/D3 等维度、或非÷32 的其他宽度上, 找到 TRT 看不见的新耦合**。负结果(某维度无耦合)也明确记录(喂 §5 集成判定)。

### go/no-go(S2.2b/c 通过条件)
screening 把全 9 维各扫到并出候选表; 深探确认 **≥1 个"新耦合"(非 D1/D2 对齐)**, 跨配置稳定 + 有机理; 或诚实记"除对齐外无新耦合"(同样是 §5 判定的有效输入)。

### 产出
`results/H800_s2_2b_*.{log,csv}` + "维度 × 软件轴 → {新耦合/已知/无/噪声}" 总表 + 每确认耦合的机理。脚本落 `scripts/phase2/s2_2b_*.py`。

---

## §4 后续关 (S2.2b 后)

- **S2.2c 缓解 demo**: 串行(prune-then-quant, 失配→fallback)vs 协同(TVM 配置 D1 重选 intrinsic + D2 pad + L3 storage_align + L4 流水 + L1 staging 救回失配臂); 出调度产物 + 干净 GPU 延迟对比。判据: 协同能逆转失配劣势且 TRT 不暴露该旋钮。
- **S2.3 干净延迟 + 跨模型**: demo 点在 **4090/Orin 空闲实测延迟**(util≤2%/foreign≤50MiB); 同套协议复用到 **CoDriving**(H800 已有剪枝×量化 pilot 副本)验跨模型。
- **S2.4 联合搜可行性 + 集成判定**: 剪枝率×量化档×{已证耦合维度子集}小规模联合搜; 按 dims_hardware_v4 §5 四条标准(发现≥2耦合 / 缓解≥1 / 可联合搜 / Pareto 扩张或机理增量)决定**是否把 TVM 集成进框架**, 否则诚实记负结果退 route1。

---

## §5 环境 / 路径 / 命令速查

- **H800**: `ssh -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>`(password-based SSH (disabled; use an SSH key) 密码见私存; 非交互每命令显式 export 代理 7897)。GPU6/7 常空闲; 机理实验对噪声免疫可任意卡。
- **TVM env**: `${V2X_DATA_ROOT}/tvm310/bin/python`; 跑前 `export PATH=/usr/local/cuda-12.2/bin:$PATH; export LD_LIBRARY_PATH=$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path); export CUDA_VISIBLE_DEVICES=<idle>`。
- **素材(H800 `${V2X_DATA_ROOT}/s2_tvm/`)**: `models/{base,p50,trap25}_backbone.onnx`(S2.1 PASS); 脚本 `s2_2_{probe,int8_cliff,verify_tc}.py`(go/no-go + 合成 int8 conv + TIR-builtin 复核)。
- **本地 repo(`${V2X_ROOT}`)**: 脚本镜像 `scripts/phase2/s2_2_*.py`; 日志 `results/H800_s2_2_*.log`; onnx `models/stage_a_cache/{p50,trap25}_backbone.onnx`。
- **关键 API(改版 Unity, 别用旧路径)**: target=`tvm.target.Target.from_device(tvm.cuda(0))`; tune 前必 `Sequential([LegalizeOps,AnnotateTIROpPattern,FuseOps,FuseTIR])`; tune=`from tvm.s_tir.meta_schedule import relax_integration as ri; ri.tune_relax(...)`; 调度后查=`relax.transform.MetaScheduleApplyDatabase(work_dir=...)` + `mod.script()`; TC intrins=`tvm.s_tir.tensor_intrin.cuda`(导入即注册, verify 脚本必 import)。

---

## §6 纪律 + 风险

**纪律**: ① 延迟必空闲 GPU 实测; 调度产物/TC 静态分析对噪声免疫可带噪做。② TVM 绝对延迟常追不上 TRT, 价值在**发现/量化/缓解耦合**非刷 SOTA。③ 任何"发现耦合/已缓解"必跨配置验稳(防 §3.1 单配置假象, 历史已犯 S0 overclaim)。④ supervisor 对每个"已 build/对齐/加速/发现耦合"复跑+读文件+数值 diff 核验。⑤ 区分 真测/估算/仅声明。

**风险**:
- **R-量化pass**: relax 0.20 无现成 INT8 量化 pass(`ToMixedPrecision` 仅 fp16)→ 全模型 INT8 受阻, 故 S2.2b 用忠实合成单 conv。全图级耦合(融合/layout 传播)留 S2.3 评估 torch 前端 + 手写 QDQ。
- **R-WMMA绕过(最可能)**: 见 §2.2 — D1 默认 K16 很可能无悬崖; 重心备选 = 强制 K32 复现 + pad 缓解(可控 demo)+ L1/L3/D3 细粒度差异。
- **R-ncu权限**: 动态 profiling(occupancy/bank-conflict)需权限 + 相对空闲, H800 共享可能受限 → 退化只靠静态 TIR 分析(足够判耦合, 缺的是绝对 occupancy 数)。
- **R-tensorize手工构造工程量**: 强制 K32 路径手工 tensorize schedule 较 fiddly(需按 intrin 期望 loop 结构 blockize+ldmatrix+mma+store), 可能多次迭代, timebox; 退化用"改 space generator 的 intrin_group"让 tuner 在指定路径内搜。

---

## §7 接手第一步

1. 读本页 + dims_hardware_v4 **§4.3(耦合发现 campaign)**/§4.2(单维度协议)/§2(维度定义)/§5(集成判定) + memory + route2 handoff。
2. 确认 H800 env(`${V2X_DATA_ROOT}/tvm310` + nvlibs 路径 + nvcc PATH)仍在; 跑一次 `s2_2_int8_cliff.py 64 64 32 64 <wd> sanity 32` 确认工具链未腐。
3. **起 S2.2b 阶段 A(screening)**: 把 `s2_2_int8_cliff.py` 扩成扫描器, 对代表性 conv 扫 **S1 宽度(32/48/64/80/96/112/128 + 24/40/56 + 33/50/67)× S3 精度(INT8/FP16)**, 每点抽**全维度调度产物 + 延迟**, 跑 §4.3 信号探测器 → 出候选耦合表(维度×软件轴)。**先回答: 除已知对齐外, 哪些维度/哪些宽度出现了延迟悬崖或调度产物突变。**
4. **阶段 B(深探)**: 对候选逐个隔离验证 + 三分类(已知/新/噪声), 含已知对齐 K16-vs-K32 深探; 跨配置验稳; supervisor 核验后进 S2.2c→d。
5. 全程记负结果(无耦合的维度也写进总表, 喂 §5)。

---

## §8 用户 2026-06-18 指出的两个核心缺口 + 解决方案 (当前最高优先)

> 用户原话: "现在的优化方案好像都取得了非常明显的效果, 但似乎没有办法反映到端到端的提速能力上, 主要都是对算子的加速效果。并且似乎没有证明目前这些硬件维度和软件优化的耦合关系。"
> 两条都成立, 是真缺口。S2.2c 把方向带偏成"纯算子旋钮价值", 丢了 e2e 与耦合两条主线。下面是诊断 + 可执行解决方案(工具都已就绪)。

### 缺口①: per-operator ≠ end-to-end
- **诊断**: S2.2c 的 2.3×/1.46×/67× 全是**单算子合成微基准**(方阵 GEMM / 极端 GEMV)。e2e 延迟受 **Amdahl** 限制: 67× 落在占 e2e 1% 的小 op 上 ⇒ e2e 几乎无感; 真实 backbone 还含 memory-bound 层、非 GEMM 部分、kernel launch 开销。**从未实测 e2e**。
- **解决 = 实验 E-e2e(端到端旋钮价值 + Amdahl 归因)**:
  1. 目标函数 = **PyramidFusion backbone 子图全图**(已 S2.1 对齐, `models/stage_a_cache/{p50,trap25}_backbone.onnx`), fp16(全图 INT8 受 R1 限)。
  2. 三臂 e2e 实测(idle GPU): ① TVM **dlight-default**(旋钮最小/默认调度) ② TVM **MetaSchedule-tuned**(全旋钮, 大 budget≥1000) ③ 参考 TRT fp16(S0b 已有 backbone 数)。
  3. **Amdahl 归因**: profile tuned backbone 各 op 延迟占比, 指出哪些 op 主导 e2e、旋钮在主导 op 上的实际加速、折算到 e2e 的净加速。
  4. 判据: 报 **e2e tuned/default 比 = 旋钮栈的真 e2e 价值**(诚实预期 ≪ per-op 极值, 且可能追不上 TRT 绝对值——价值在机理/耦合非 SOTA)。
  - 工具就绪: tune_relax 全图(screening 已验证可 tune backbone); dlight-default 走 `tvm.compile` 默认 pipeline; time_evaluator 实测。

### 缺口②: 未证明 HW 维度 × SW 优化的耦合
- **诊断**: S2.2c 只证"旋钮 X 加速算子 Y"(HW 单独价值), **没碰软件轴**。co-design 耦合的本质命题 = **"最优 HW 旋钮配置是软件决策(剪枝宽度/量化档)的函数"** —— 即不能先定 SW 再独立调 HW, 必须联合搜。S2.2c 完全没测这个依赖关系。
- **解决 = 实验 E-couple(决定 route2 GO/NO-GO 的核心实验)**, 两个层次:
  - **E-couple-A(逐旋钮交互扫描, 主推, 最贴"这些 HW 维度")**: 对每个旋钮, 扫 **旋钮设置 × 软件配置** 网格, 看**最优旋钮设置是否随软件配置移动**(argmin 漂移)。例:
    - D3 tiling: tile T∈{16,32,64} × 剪枝宽度 {32,48,64,128} → best-T 是否随宽度变?
    - L1 staging 深度 × 宽度; D1 intrinsic(K16/K32/scalar)× 精度(int8/fp16)× 宽度。
    - 判据: **argmin(旋钮|SW) 随 SW 漂移 ⇒ 该旋钮与 SW 耦合**(联合搜>可分离)。复用 `s2_2c_manual_gemm.py`(TM/TN/BK 已是参数, 加 M/N/K=SW 维度即可扫)。
  - **E-couple-B(transfer-penalty 矩阵, 整体性, 决定性)**: 对每个 SW 配置 tune 出最优 schedule; **交叉应用**(把 SW-A 的最优 schedule 用到 SW-B 的 workload)实测, 构 N×N 矩阵。
    - 判据: **对角(自身最优) ≪ 非对角(借用) ⇒ HW 最优依赖 SW ⇒ 耦合成立 ⇒ co-design 有价值(route2 GO)**; 非对角≈对角 ⇒ 可分离 ⇒ co-design 无价值(route1 NO-GO)。
    - 工程注意: 不同剪枝宽度 conv 维度不同, schedule trace 维度相关不能直接套 ⇒ 用**同一 schedule 结构(tile 倍数/staging 深度参数化)实例化到两个 shape**, 或比 best-T 漂移(=E-couple-A 的等价表述)。
- **★诚实风险(必须预期)**: S2.2b 已显示 **TVM 倾向"消解"耦合**(WMMA-K16 让对齐无关、Q/DQ fuse 进 epilogue)。⇒ E-couple 很可能测出 **transfer penalty 小 / argmin 不随 SW 漂移 = 可分离 = route2 NO-GO**。但这正是用户质疑要求的诚实测试: **若可分离, 就按纪律退 route1**(重锚 Orin 异构并行), 把"调度可分离于剪枝/量化"作为负证据写进论文"为何不做纯调度协同"。若耦合显著则 route2 GO。**无论哪个结果都是对用户两个问题的正面回答。**

### 执行顺序建议
E-couple-A 先(最便宜、最直接回答耦合、复用现成脚本)→ 若见耦合再 E-couple-B 加固 + E-e2e 给端到端数 → 喂 §5 GO/NO-GO 终判。
