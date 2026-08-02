> ⚠️ **已被 [HANDOFF_codesign_unified_v1.md](HANDOFF_codesign_unified_v1.md) 取代(2026-06-18)** —— 本文档的技术内容已并入该统一交接的 Part I。接手只读 unified 版; 本页仅作历史留存。

# HANDOFF — route2 / TVM 适配统一交接 (Pyramid[HEAL] + CoDriving) v1, 2026-06-18

> **接手先读**: 本页(两模型整合总览 + 跨模型对比)。细节按模型分两份子交接:
> - Pyramid(HEAL): [HANDOFF_route2_e2e_coupling_v1.md](HANDOFF_route2_e2e_coupling_v1.md)(E-couple/E-e2e/RSU 段/全流程剪枝量化)
> - CoDriving: [HANDOFF_codriving_tvm_migration_v1.md](HANDOFF_codriving_tvm_migration_v1.md)(P1–P4 整管线迁移)
> - 维度定义: [dims_hardware_v4.md](../design/dims_hardware_v4.md); memory: `project-hwsw-codesign-route2` + `project-codriving-optimization-pilot`。

---

## §0 一句话现状

route2 = 用 TVM 把编译调度变成"可搜硬件轴"挣 co-design 标签。**两个 V2X 协同感知模型都已完成 TVM 适配 + 耦合/端到端实测**(平台主体 H800,有 TVM;边缘绝对值待 4090/Orin):
- **Pyramid(HEAL,协同检测,grouped bottleneck + pyramid attention neck)**: S2.x 全程 + E-couple/E-e2e/RSU 段/全流程剪枝量化。
- **CoDriving(协同检测+规划基座,标准 ResNet 3×3 conv + where2comm fusion)**: P1–P4 整管线迁移闭合。

**★整合后的核心科学结论(两模型对比得出,可发表)**: **co-design 的 HW×SW 耦合强度是 model-dependent 的** —— 非标算子结构(Pyramid grouped bottleneck)显**耦合**,标准算子(CoDriving 深归约 3×3 conv)显**可分离**。同一套 TVM 调度轴,在不同模型上 co-design 价值天差地别。

---

## §1 共性方法论 (两模型通用工具链, 可复用)

1. **TVM env(H800)**: `${V2X_DATA_ROOT}/tvm310/bin/python`(0.20.dev1070 改版 Unity: `tvm.tir`→`tvm.s_tir`/`tvm.tirx`; MetaSchedule 在 `tvm.s_tir.meta_schedule`; dlight 在 `tvm.s_tir.dlight`; `get_block`→`get_sblock`)。跑前必 `export PATH=/usr/local/cuda-12.2/bin:$PATH; export LD_LIBRARY_PATH=$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path); export CUDA_VISIBLE_DEVICES=<idle>`。SSH 非交互每命令显式 export 代理 7897。
2. **导入+对齐**: `from_onnx`→relax→`relax.build(cuda)`→VM, vs ORT 数值对齐(两模型 maxdiff 均 ~e-6 PASS)。
3. **tune**: `Sequential([LegalizeOps,AnnotateTIROpPattern,FuseOps,FuseTIR])` → `ri.tune_relax(max_trials_global=,seed=)` → `MetaScheduleApplyDatabase` → `tvm.compile` → VM `time_evaluator`。
4. **手写 schedule 消融**: 编译前必打 `func.with_attr("tirx.is_scheduled", True)` 否则 dlight 重排报错; shared 暂存必配 **cooperative fetch**(shared-load 循环 fuse+split+bind threadIdx)否则 24× 假慢。
5. **共同限制**: relax **无 INT8 量化 pass**(全 fp32/fp16,INT8 须 BYOC-TRT); **fusion/neck 导入 BLOCKED**(grid_sample/where2comm/scatter 数据依赖 shape 推断,两模型同病)→ 都退**抽 backbone-only 子图**做耦合/旋钮实测。
6. **模型专属坑**: Pyramid = grouped conv(g=32→SCALAR 无 TC); CoDriving = ①动态 batch(ONNX dim=0→读 shape 时替 2)②384→1 单类 cls 头触发 reduction "compact dataflow" 崩→抽 backbone-only ③有 BatchNorm(Pyramid 无)→MS apply DB 后须 dlight Fallback 补剩余块。

---

## §2 ★跨模型对比 (整合核心)

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
| RSU/车端 backbone 占比 | RSU ~45% / 车端 ~12%(eager 口径)| RSU 50% / 车端 50%(fusion 一半是 backbone 重跑)|
| 残留硬瓶颈 | fusion neck 55%(车端) | fusion_attn 16%(车端) |

**⇒ 统一结论**: ① **旋钮价值 = conv 多非标则越大**(Pyramid grouped 10× vs CoDriving 标准 2×); ② **耦合强度 model-dependent**(Pyramid 显耦合 = route2 有 co-design 价值; CoDriving 显可分离 = 可先定 HW 再独立剪,route2 co-design 价值≈0); ③ **共同负证据**: 两模型 fusion neck 都导不进 TVM(grid_sample),都按 Amdahl 被前后处理稀释,真加速都需"backbone TVM/TRT + 前端 CUDA 化 + INT8 BYOC-TRT"。

---

## §3 Pyramid(HEAL)线现状 (细节见子交接)
- **E-couple**(7 旋钮): D1 结构型(grouped→SCALAR,宽度可分离)/ D3 中等耦合(21% rank-reversal, 折中 tile 7.3%)/ L1·L2 弱 / L3·L4 不可判 / **L5 强耦合**(cross↔serial 翻转 Cout32↔64)。
- **E-e2e**: backbone tuned/default base 10.34× / p50 8.74×。
- **Amdahl(真实测 E8 Orin)**: backbone 占全协同 pipeline 13.6%,fusion neck 55% ⇒ backbone 10× 折成全 pipeline **净 1.14×**。
- **RSU 段(Orin 实测)**: VFE 计算编译化 **>200×**(eager 18ms→编译 85µs,全是 eager 开销)/ scatter NCHW 向量化 **2–4×**(数值 maxdiff=0,无需 .cu;边缘卡须避免 permute 布局转置)/ backbone 10×。
- **全流程剪枝量化**: 剪枝 p50 **2.07×** / trap25(非÷32 失配)**0.29× 反而慢**; 量化(TRT 参考)对齐 1.32–1.60× / trap25 仅 1.08× ⇒ **÷32 对齐纪律**(失配剪枝量化双输)。

## §4 CoDriving 线现状 (细节见子交接)
- **P1**: S2.1 导入+对齐 PASS(3 专属坑解)。
- **P2**: backbone 旋钮 base 2.07× / p50 2.26×(≪ Pyramid); 剪枝 backbone 编译级 ~5× 可见。
- **P3(RSU/车端分段)**: backbone 实占 RSU 50% / 车端 50%(fusion 一半是 backbone 重跑)。**RSU 段** TVM 1.35× / +CUDA 前端 **2.9×**(无 fusion 残留 = 最易拿满加速); **车端段** TVM 1.35× / +CUDA **2.14×**,残留 fusion_attn 16%。decode CPU-meshgrid 修复全网 **5–6×**(bit 级等价)。
- **P4(6 旋钮)**: HW↔剪枝耦合**弱/可分离**(D3/L1/L2/L3/L4/L5 无一随剪枝漂移); 耦合是算子结构型(深归约 / 头 Cout=1→cross)非剪枝率函数。

---

## §5 两线共同下一步 (真工程, 兑现加速主路)
1. **整段 RSU TRT/编译实测**(两模型同路): voxelize+VFE+backbone,消 eager + scatter 向量化/CUDA。RSU 无 fusion 残留 = 最易拿满(Pyramid/CoDriving 估算分别待实测、2.9×)。**首要**: 把分段估算变 Orin 实测真 e2e。
2. **车端 fusion neck**: 两模型都 BLOCKED 在 grid_sample/where2comm。路径 = relax torch 前端 / BYOC-TRT GridSample plugin 碰这 16–55%。
3. **INT8 e2e**: relax 无量化 pass,两模型都须 BYOC-TRT。
4. **边缘绝对竞争力**: TVM/TRT-tuned vs eager,4090/Orin 实测(H800 只坐实相对/机理)。

## §6 环境/路径/脚本 (两线共用)
- **H800** `ssh -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>`(密码每会话确认; 代理 7897); GPU 常被外部 DDP 抢,微秒 kernel 必空闲卡(mem≤50MiB)实测,偶发竞争产 11–22µs 假尖峰(min-of-N 兜)。
- **Orin**(RSU 边缘卡)`ssh ${V2X_REMOTE_USER}@<PRIVATE_HOST>`(内网不走代理); torch 1.12+CUDA; Tegra 用 tegrastats 查 GR3D。
- **素材(H800 `${V2X_DATA_ROOT}/s2_tvm/`)**: `models/{base,p50,trap25}_backbone.onnx`(Pyramid)+ `models/codriving_cache/{base,p50,p75}_backbone.onnx`(CoDriving)。
- **脚本(本地 `scripts/phase2/` + `scripts/phase1/`)**: Pyramid = `s2_2{d_couple_tile,e_e2e,f_gemm_knob,g_reduce,h_vfe}.py`+`scatter_microbench.py`+`vfe_forward_microbench.py`; CoDriving = `s2_1_codriving_align.py`+`s2_codriving_e2e.py`+`s2_{cod_couple_conv,cod_knob,2g_reduce}.py`+`codriving_segment_timing.py`。
- **结果**: Pyramid `results/s2_2{d,e,f,g,h}*.csv`+`S0b_coupling_clean_4090.csv`+`E8_orin_e2e_fullchain.csv`; CoDriving `results/cod_{e2e,couple_conv,couple_rep,reduce,knob}.csv`+`E_codriving_*`。

## §7 纪律 / 诚实边界
- H800 是 Hopper 数据中心卡 ≠ 边缘标的; 相对/机理结论此坐实,边缘绝对延迟须 4090/Orin。
- DEFAULT=dlight 是"旋钮关"基线,default/tuned 比 = 开启搜索价值,**非 vs TRT/强手写**。
- backbone-only ≠ 全 pipeline(VFE/scatter/decode/NMS/fusion 非 TVM 调度,Amdahl 稀释)。
- 量化全是 TRT 参考(relax 无 INT8 pass),部分跨平台(4090 collab2 口径),不与 H800 TVM 数混加。
- 跨平台/口径**不拼单一 e2e**(VFE/backbone=H800 TVM,scatter=Orin torch,量化=4090 TRT)。
- 任何"加速/对齐/耦合"结论必跨配置(seed/trial)验稳; agent 自报必复核(复跑/读文件/git diff)。
