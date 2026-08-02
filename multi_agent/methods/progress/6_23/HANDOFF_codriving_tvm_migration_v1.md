# HANDOFF — CoDriving 整管线 TVM 迁移 (v1, 2026-06-18)

> ★ 跨模型整合视角(Pyramid+CoDriving 对照 + 统一结论)见 [HANDOFF_tvm_codesign_crossmodel_v1.md](HANDOFF_tvm_codesign_crossmodel_v1.md); 本页是 CoDriving 单模型执行细节。

> 全新交接(独立工作流): 把 route2 的 **Pyramid TVM 迁移方法学**迁到 **CoDriving**。
> 接手先读: 本页 + `HANDOFF_route2_e2e_coupling_v1.md`(Pyramid 侧方法/env/已解坑) + `HANDOFF_codriving_optimization_v2.md`(CoDriving 优化现状, 尤其 §8 decode 主瓶颈)+ memory `project-hwsw-codesign-route2` + `project-codriving-optimization-pilot`。

---

## 0. 为什么做这个 (动机, 用户拍板)
CoDriving 优化(v2)发现: 修 decode CPU-meshgrid 后全网 ~15-17ms, 但 **剪枝/量化在全网不可见** —— 整网被 eager 前后处理(VFE/scatter/fusion/NMS + 5460 launch/帧)主导, backbone 只占 ~3ms。要让 prune/quant 在**全网部署口径**兑现, 唯一路 = **整管线编译**消掉 eager 开销。

route2 已证: **TRT 是封闭 auto-tuner(给不了生成式调度轴), TVM relax+MetaSchedule 才是可搜的硬件轴**。Pyramid 侧 TVM 迁移成熟(S2.1 import+对齐 PASS → MetaSchedule 通 → E-e2e backbone tuned/default **10.34×/8.74×**)。**本工作流 = 把这套迁到 CoDriving**, 既服务 CoDriving 全网提速, 又把 route2 协同结论推广到第二个模型。

---

## 1. ★已完成 (2026-06-18, 实测核验)

### 1.1 P1 — S2.1 闸门 PASS: CoDriving 导入 TVM relax + fp32 数值对齐
- **body core(backbone→cls/reg 头)**: relax 导入 OK; llvm CPU build 数值对齐 **maxdiff 4.3e-6 ≪ 1e-3** (cls_pred 4.29e-6 / reg_pred 2.38e-6)。
- **backbone-only 子图**(抽 `spatial_features → /backbone/Concat_output_0`, 27 Conv/3 ConvTranspose deblock/12 Add/3 BN/concat→384ch): **cuda GPU build 对齐 PASS, maxdiff 4.05e-6**。
- ⇒ **TVM 能正确吃下 CoDriving dense 核并数值复现** = 整管线迁移去风险闸门通过。

### 1.2 已解的 3 个 CoDriving 专属坑(Pyramid 没踩到)
1. **动态 batch**: CoDriving ONNX 把 batch 导成符号维(ONNX `dim_value=0`)→ relax build 零长度循环崩。**修: 读 input shape 时把 0 维替换成 BATCH(=2, 2-agent collab)**。Pyramid 是 concrete batch=2 故无此坑。
2. **384→1 单类 cls 头**: out_channel=1 的 1×1 conv → dlight/MS 的 reduction(rfactor cross-thread)schedule 撞 "compact dataflow not satisfied" 崩。**修: 抽 backbone-only 子图(头是可忽略计算, 且协同/tiling/tensorize 信号全在 backbone, 同 Pyramid 只抽 backbone)**。全 pipeline 真 e2e 时头可留 PyTorch 或单独处理。
3. **BatchNormalization 剩余块**: CoDriving backbone 有 3 个 BN(Pyramid backbone 是纯 conv/relu/add 无 BN)→ MS 只 tune conv/matmul tasks, BN/transpose/elementwise 剩余块没 GPU thread 绑定 → 编译 `Memory verification failed: param directly accessed by host memory`。**修: MS apply DB 后再 `dl.ApplyDefaultSchedule(Matmul,GEMV,Reduction,GeneralReduction,Fallback)` 给剩余块补 GPU schedule(dlight 跳过已 MS 调度块)**。

---

### 1.3 P2 — backbone TVM 旋钮栈 e2e 价值 (DONE, H800 GPU5/6, 1000 trials, fp32 batch=2)
| 档 | DEFAULT(dlight 旋钮关) | TUNED(MS-1000 全旋钮) | tuned/default | tune_s |
|---|---|---|---|---|
| base (64_128_256) | 16.68ms | **8.06ms** | **2.07×** | 1581 |
| p50 (32_64_128) | 3.64ms | **1.61ms** | **2.26×** | 1474 |
(`results/cod_e2e.csv` ← H800 `${V2X_DATA_ROOT}/s2_tvm/cod_e2e.csv`)
- **① CoDriving 旋钮价值 ~2.1-2.3× ≪ Pyramid 8.74×/10.34×**: 根因 = CoDriving backbone 是**标准 ResNet BasicBlock conv**, dlight 默认 schedule 已较好(剩余空间小); Pyramid 是 **grouped bottleneck conv**, dlight 默认差→调度收益大。**⇒ "TVM 旋钮价值"是"conv 有多非标准"的函数 = 一个有用的跨模型发现**(标准 conv 模型从编译调度搜索得益有限)。
- **② 剪枝在 backbone 编译级清晰可见**: base→p50 DEFAULT 16.68→3.64ms(**4.6×**)/ TUNED 8.06→1.61ms(**5.0×**)。⇒ 剪枝**真加速 backbone 计算**(两种调度下都 ~5×), 印证它只是**在全网被 eager 前后处理稀释**(CoDriving v2 §8), 非无效。
- caveat: fp32(relax 无量化 pass, 无 tensor core)+ H800(非边缘)+ DEFAULT=dlight 旋钮关基线(非 vs TRT)。绝对 ms 跨卡不可直比 4090 eager。

### 1.4 P3-(a) — RSU/车端分段 Amdahl 折算 (★2026-06-18 用户纠错后重做; 旧"19%/1.11×"已作废)
> **★用户纠错(完全正确, 已实测验证)**: 旧 §1.4 用 per-stage hook 数据把 backbone 算成"全管线 19% → 1.11×", **错在两点**: ① **fusion_net 一半是 backbone 重跑**(`center_point_codriving.py` L140-145 把 `self.backbone` 传进 fusion; `codriving_attn.py` L264 `feats = backbone.resnet(x)` 重跑整个 ResNet)——backbone 实际出现两次, TVM 可调量被严重低估; ② **vfe/scatter/heads 不是永久地板, 已证可 CUDA 化压缩**(同 Pyramid route2 scatter/NMS CUDA 化), 不该算进不可优化部分。

**实测分段(脚本 `scripts/phase1/codriving_segment_timing.py`, 4090 GPU0, per-module CUDA-Event, p10 抗竞争地板, decode 已用 cached-grid 修; `output/codriving_pilot/logs/segment_timing_{base,p50,p75}.json`)。fusion_backbone 占 fusion 比例 base 0.50/p50 0.43/p75 0.45 = 实锤"fusion 一半是 backbone"。**

**按物理部署分两段(base p10, ms):**

| 段 | 模块 | ms | 占段 | 可优化路径 |
|---|---|---|---|---|
| **RSU(路端)** | backbone | 4.57 | **50%** | **TVM**(2.07×) |
| | vfe+scatter+heads | 4.59 | 50% | **CUDA 化** |
| | (RSU 合计) | **9.16** | 100% | **无 fusion** |
| **车端(vehicle)** | backbone ×2(独立+fusion 内重跑) | 7.10 | **50%** | **TVM**(2.07×) |
| | fusion_attn(warp/grid_sample+where2comm) | 2.25 | **16%** | 硬残留(导入 blocked, comm-bound) |
| | vfe+scatter+heads+decode | 4.86 | 34% | **CUDA 化** |
| | (车端合计) | **14.21** | 100% | |

- **★P3-(a) 修正结论(答"TVM 收益有限是不对的"——用户对)**: backbone(TVM 可调)占 **RSU 50% / 车端 50%**(非旧称 19%)。
  - **RSU 段**: TVM backbone 单独 **1.35×**; + CUDA 化 vfe/scatter → **2.9×**(RSU **无 fusion 残留 = 几乎全可加速**)。
  - **车端段**: TVM backbone 单独 **1.35×**(非旧 1.11×); + CUDA 化前端 → **2.14×**; 唯一硬残留 = **fusion_attn(warp+attention)2.25ms = 车端 16% / RSU 0%**。
- **★真瓶颈重新定位**: 不是"fusion_net 39% 整块不可优化", 而是 **fusion 里只有 ~16%(warp grid_sample + where2comm attention)是硬的**, 另一半(backbone 重跑)TVM 可调。decode-fix(5-6×)+ backbone TVM + 前端 CUDA 化叠加, 车端 ~2× / RSU ~2.9× 真可达。
- caveat: TVM 2.07× 是 H800 fp32 default→tuned; 分段是 4090 GPU0(0%util 但 18G 驻留)p10 抗竞争地板; CUDA 化收益是按 Pyramid 先例的估算(vfe~0.6/scatter~0.3ms), 未在 CoDriving 实测 CUDA kernel。绝对 ms 跨卡不直比; 但"backbone 占半 / fusion_attn 才是硬残留 / RSU 无残留"的结构跨 HW 稳健, 3 档一致。

**★1.4-bis 精修(2026-06-18 用户问"为何车端 backbone 与路端 backbone 时长不同")**: 因为两者**不是同一操作**。完整 backbone 模块 = **ResNet 3 stage + deblocks(ConvTranspose 上采样)+ concat**(`base_bev_backbone_resnet.py` L90-102)。
- **RSU backbone(4.57ms)= 完整 backbone**(resnet + deblocks + concat)。
- 我先前 §1.4 表的 "fusion_backbone(2.53)" 只单独计了 fusion 里的 `backbone.resnet(x)`(纯 ResNet stage), **没算 fusion 同样会跑的 deblocks**(codriving_attn L309/318 也调 `backbone.deblocks`)——那部分被错记进了 "fusion_attn"。
- **clean 三分实测**(`scripts/phase1/codriving_backbone_decomp.py`, `output/.../bb_decomp_base.json`, p10): **resnet 3.89 / deblocks+concat 0.93 / warp+attn(grid_sample+where2comm)仅 0.90ms**。⇒ ① 两个 backbone 数不同是因 RSU 数含 deblocks+concat、fusion 数只算了 resnet(+测量竞争噪声 2.5–3.9 抖动); ② **真正唯一不可 TVM 调度的硬残留 = warp+attn ≈ 0.9ms(原 "16%/2.25ms" 偏大, 因把可调的 ConvTranspose deblocks 混入)**。车端两次 backbone(resnet+deblocks)~9.6ms 全 TVM 可调 conv + 前端 CUDA 可化, 仅 ~0.9ms warp/attn 硬核 ⇒ **TVM/编译收益受限的说法被进一步推翻**。⚠️ 4090 GPU0 竞争重(18/30 干净帧, 绝对 ms 不稳), 绝对值待空闲 GPU 复测; "几乎全 conv + 极小 warp/attn 残留"的结构稳。

### 1.5 P3-(b) — 更大子图(where2comm fusion)TVM 导入 = BLOCKED(实测, 2026-06-18 H800)
脚本 `scripts/phase2/s2_codriving_fusion_probe.py`。对象 `collab_export/codriving_collab_base_fp32.onnx`(1005 节点)。
- **op inventory(实测)**: Conv×29 / ConvTranspose×3 / BatchNorm×3 / **GridSample×3**(特征 warp)/ **ScatterND×12**(稀疏 comm)/ **Einsum×3 + Softmax×3**(where2comm attention)/ Where×48 / Equal×48 / Expand×66 / Gather×75。输入 `spatial_features(2,64,256,512)` + `pairwise_t_matrix(1,2,2,4,4)`; 输出 `cls_preds/reg_preds`。
- **导入结果: `from_onnx` IMPORT_BLOCKED** —— 在 multi-scale fusion 的 **Concat 形状推断**崩(`dim2 = 2 vs 4`, three relu 输入拼接)。**注: GridSample/ScatterND 在该 TVM relax onnx frontend 实际已注册**(grep 确认)⇒ blocker 不是缺 op, 而是 relax 前端对 fusion warp/attention 的**符号形状推断弱**(同 ONNX 在 ONNXRuntime 跑得通并产出参考 npz ⇒ ONNX 本身有效, 是前端能力限制)。
- **★P3-(b) 结论(诚实, 已按 §1.4 纠错收窄)**: 注意 collab ONNX 里的 29 Conv **就是 fusion 重跑的 backbone**(导进去也是 backbone 那半)。真正导不进/调不动的硬核 = **GridSample warp + ScatterND comm + Einsum/Softmax attention = §1.4 的 fusion_attn 16%**。这部分与 **Pyramid attention neck 导入 blocked 同构**, 是 comm/数据依赖非 dense conv ⇒ TVM-schedule 路对它关闭。加速它需 ① torch 前端绕 shape 推断 ② BYOC-TRT(GridSample plugin)③ 手写 CUDA。但它只占车端 16%/RSU 0% ⇒ **不深挖也不致命**: 车端 ~2×、RSU ~2.9× 的主力(backbone TVM + 前端 CUDA)不依赖它。

### 1.5 P3-(b) — 更大子图(where2comm fusion)TVM 导入 = BLOCKED(实测, 2026-06-18 H800)
脚本 `scripts/phase2/s2_codriving_fusion_probe.py`。对象 `collab_export/codriving_collab_base_fp32.onnx`(1005 节点)。
- **op inventory(实测)**: Conv×29 / ConvTranspose×3 / BatchNorm×3 / **GridSample×3**(特征 warp)/ **ScatterND×12**(稀疏 comm)/ **Einsum×3 + Softmax×3**(where2comm attention)/ Where×48 / Equal×48 / Expand×66 / Gather×75。输入 `spatial_features(2,64,256,512)` + `pairwise_t_matrix(1,2,2,4,4)`; 输出 `cls_preds/reg_preds`。
- **导入结果: `from_onnx` IMPORT_BLOCKED** —— 在 multi-scale fusion 的 **Concat 形状推断**崩(`dim2 = 2 vs 4`, three relu 输入拼接)。**注: GridSample/ScatterND 在该 TVM relax onnx frontend 实际已注册**(grep 确认)⇒ blocker 不是缺 op, 而是 relax 前端对 fusion warp/attention 的**符号形状推断弱**(同 ONNX 在 ONNXRuntime 跑得通并产出参考 npz ⇒ ONNX 本身有效, 是前端能力限制)。
- **★P3-(b) 结论(诚实, 答"更大子图")**: CoDriving fusion neck **TVM(ONNX 路)导不进**, 与 **Pyramid attention neck 导入 blocked 同构**(route2 handoff §6) —— 跨模型一致的负结果。结合 §1.4 Amdahl: 即便幻想 fusion 全部 TVM 调到 2.07×, 全 module 也仅 **1.43×**; 且 fusion 主体是 GridSample/ScatterND/Einsum 等**数据依赖/comm-bound** 算子(非 TVM 擅长的 dense conv), 真实可调收益 ≪ 2× ⇒ **P3 真 headroom(39% fusion)对 TVM-schedule 路基本关闭**。加速 fusion 需 ① torch 前端/手改图绕过 shape 推断 ② BYOC-TRT(GridSample plugin)③ 手写 CUDA(同 Pyramid scatter/NMS 结论)。**不投入深挖前端**(Amdahl 上界已封顶 1.43×, 性价比低)。⇒ CoDriving 全网真加速主力仍 = **decode-fix(5-6×)+ backbone 编译(占 19%)**, 与 Pyramid 殊途同归。

### 1.6 P4 — HW×SW 耦合 in CoDriving(真 conv shape, DONE 2026-06-18 H800 GPU6/7 空闲, 3× rep 验稳)
脚本 `scripts/phase2/s2_cod_couple_conv.py`(tile×宽度, 真 conv-GEMM N=W/K=9W) + `s2_2g_reduce.py`(L5 reduction×Cout)。数据 `results/cod_couple_conv.csv` / `cod_couple_rep.csv`(3×) / `cod_reduce.csv`。**与 Pyramid 的关键差异: 用真 CoDriving conv 形状(标准 3×3, g=1, im2col K=9W ≫ N=W 的深归约非对称), 非 Pyramid 的 model-agnostic 方阵 proxy(K=N=W)。**

**① D3 tiling 旋钮 — argmin(tile|剪枝宽度)(3× rep min, 真 conv-GEMM)**
| 宽度 W | 16×16×32 | 32×32×16 | 16×16×16 | argmin | 固定 16×16×32 损失 |
|---|---|---|---|---|---|
| 64 (p75) | **21.9µs** | 23.9 | 24.0 | 16×16×32 | 0.0% |
| 128 (p50) | **77.7µs** | 78.2 | 83.4 | 16×16×32 | 0.0% |
| 256 (base) | **297.4µs** | 298.5 | 320.3 | 16×16×32 | 0.0% |
- **★结论: tiling 旋钮对 CoDriving 剪枝宽度 = 完全可分离**(单一固定 tile 16×16×32 在所有剪枝档都最优, 0.0% 损失)。**对比 Pyramid 方阵 proxy 的 21% rank-reversal —— CoDriving 没有**。根因: 标准 3×3 conv 的**深归约 K=9W ≫ N=W**, 深 reduction tile(BK=32)+ 均衡 16×16 thread tile 普适最优, 与宽度无关; Pyramid 方阵(K=N=W)出现的宽度依赖 tile 偏好在真 conv 的 K/N=9 非对称下被冲掉。(单跑 W256 曾见 32×32×16 险胜 1%, 3× rep 证实是噪声, 16×16×32 全胜。)

**② L5 reduction 旋钮 — argmin(serial/cross|Cout)**
| Cout | serial | cross | winner | 比 |
|---|---|---|---|---|
| 8 | 50.6 | **11.2** | cross | 4.53× |
| 32 | 46.2 | **39.3** | cross | 1.18× |
| 64 | **48.1** | 71.2 | serial | 1.48× |
| 128 | **49.4** | 137.3 | serial | 2.78× |
| 256 | **45.9** | 270.3 | serial | 5.89× |
- **翻转边界 = Cout 32↔64**(cross@≤32, serial@≥64), **与 Pyramid s2_2g 完全一致**。
- **★CoDriving 专属解读: 剪枝宽度 {64,128,256} 全 ≥64 ⇒ 全在 serial 区**(翻转边界 32 在最小宽度 64 之下)⇒ reduction 旋钮对**剪枝可分离**(剪 256→128→64 最优策略恒 serial)。**但有强结构耦合**: CoDriving cls 头 Cout=1 深在 cross 区(4.5×+)⇒ reduction 策略须**按层角色(头 vs backbone)选**, 非全局 —— 这是架构固有, 非剪枝率函数。

**③ L1–L4 旋钮(真 conv-GEMM, min of 2 rep, 脚本 `s2_cod_knob.py`, `results/cod_knob.csv`)** — 用户要求"每个旋钮都扫", 补齐:
| 旋钮 | 轴 | W64 / W128 / W256 argmin | 漂移? | 耦合 |
|---|---|---|---|---|
| **L1 shared-staging** | global/shared | shared / shared / shared(恒 1.4× 赢) | 否 | 可分离 |
| **L2 thread-tile** | 8/16/32 | 32 / 32 / 32(恒, 深 K 偏好大 tile) | 否 | 可分离 |
| **L3 bank-align** | off/4/8 | off / off / off(align 反伤 ~1%) | 否 | 可分离 |
| **L4 sw-pipeline** | off/2-stage | off / on / on(差异 ≤0.5%) | 名义漂移=噪声 | 可分离 |
- 加上 §1.6① D3 tiling + ② L5 reduction = **6 旋钮全测**, **全部对剪枝可分离**(L4 名义漂移 <0.5% 在噪声地板)。第 7 个 D1 tensorize(WMMA): CoDriving 标准 conv 的 K=9W 大且 N=W(64/128/256)全 ÷16 ⇒ WMMA 在所有宽度都成立(同 Pyramid "宽度可分离, TVM 补 K 到 ×16 消 ÷32 悬崖"), 按构造可分离(fp32 无 TC 故未实测, 标注估算)。

**★★P4 综合(诚实, 真实验依据, 6 旋钮实测)**: **CoDriving 的 HW 旋钮↔剪枝决策耦合 = 弱/可分离**(6 旋钮 argmin 无一随剪枝宽度真漂移), **显著弱于 Pyramid**。存在的耦合是**算子结构型**(conv 深归约 / 头 vs backbone), 是架构固定属性, 不随剪枝率变。⇒ **跨模型结论: co-design 耦合强度 model-dependent** —— Pyramid(grouped bottleneck + 方阵 proxy)显耦合; CoDriving(标准深归约 3×3 conv)显可分离。对 CoDriving 而言 route2 在这些旋钮上 co-design 价值≈0(可先定 HW 再独立剪)。**caveat**: tile 用真 conv 的 im2col-GEMM 代理(单算子, 非整 conv2d 调度 + fusion epilogue); reduction 用 M=64/K=2048 合成。结构性结论(深归约→tile 普适 / 头 Cout=1→cross)噪声免疫且 3× 验稳。

## 2. 已完成 (P1/P2/P3/P4 全收口) — 见 §3 终判

---

## 3. 终判 (P1–P4 全完成) + 残留可选项

**CoDriving 整管线 TVM 迁移 = 已闭合**, 四阶段真实验结论:
- **P1**(§1.1-1.2): S2.1 导入+对齐 PASS(3 专属坑解)。
- **P2**(§1.3): backbone 旋钮栈 tuned/default = base 2.07× / p50 2.26×(≪ Pyramid 8.7×, 因标准 conv dlight 默认已好); 剪枝 backbone 编译级 ~5× 可见。
- **P3**(§1.4-1.5, ★用户纠错后重做): backbone(TVM 可调)实占 **RSU 50% / 车端 50%**(旧"19%"错——fusion 一半是 backbone 重跑, 实测 fusion_backbone 占 fusion 0.43-0.50)。**RSU 段**: TVM backbone 1.35× / +CUDA 前端 **2.9×**(无 fusion 残留)。**车端段**: TVM backbone 1.35× / +CUDA 前端 **2.14×**, 硬残留只 fusion_attn(warp+attention)=车端 16%/RSU 0%(导入 blocked, comm-bound)。⇒ "TVM 收益有限"被推翻; 真加速 = decode-fix(5-6×)+ backbone TVM + 前端 CUDA 化。
- **P4**(§1.6, 6 旋钮全测): HW 旋钮↔剪枝耦合 **弱/可分离**(D3/L1/L2/L3/L4/L5 argmin 无一随剪枝真漂移), 显著弱于 Pyramid; 耦合是算子结构型(深归约/头 vs backbone)非剪枝率函数。⇒ **co-design 耦合强度 model-dependent**。

**★口径约定(用户 2026-06-18 拍板)**: 此后 CoDriving 所有耗时统计**一律分 RSU(路端=感知, 无 fusion)/ 车端(vehicle=感知+fusion+头+decode)两段汇报**(见 §1.4 分段表)。fusion_net 是车端专属。

**下一步真工程(已不是边际可选, 是兑现加速的主路)**:
1. **整段 RSU TRT/编译实测**(voxelize+VFE+backbone, 消 eager + scatter CUDA 化): RSU 50% 是 backbone(TVM/TRT)、50% 是 CUDA 化前端, **无 fusion 残留 ⇒ RSU 是最易拿满加速的段**(估算 2.9×, 待实测坐实)。与 Pyramid route2 §6 RSU 段同路。**这是把 §1.4 估算变实测的首要任务**。
2. **车端 TRT 整管线**: backbone×2(TRT)+ 前端 CUDA, 残留 fusion_attn(grid_sample warp + where2comm attention)16% —— 用 BYOC-TRT GridSample plugin 或手写 CUDA 碰这 16%。估算车端 2.14×(不碰 fusion_attn)。
3. **绝对竞争力**: CoDriving TVM/TRT-tuned vs eager 同卡(边缘数留 4090/Orin)。
4. **INT8 e2e**: relax 无量化 pass, 须 BYOC-TRT。

---

## 4. 环境 / 路径 / 脚本
- **H800**: `ssh -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>`(密码每会话确认; 非交互每命令 `export https_proxy=http://<PRIVATE_HOST>:7897 http_proxy=同`)。GPU5/6/7 常空闲(本轮 4MiB/0% 干净), 跑前 nvidia-smi 确认。
- **TVM env**: `${V2X_DATA_ROOT}/tvm310/bin/python`(0.20.dev1070 Unity; `tvm.tir`→`tvm.s_tir`; MetaSchedule 在 `tvm.s_tir.meta_schedule`; dlight 在 `tvm.s_tir.dlight`)。跑前必 `export PATH=/usr/local/cuda-12.2/bin:$PATH; export LD_LIBRARY_PATH=$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path); export CUDA_VISIBLE_DEVICES=<idle>`。
- **脚本(本地 `scripts/phase2/` + H800 `${V2X_DATA_ROOT}/s2_tvm/`)**:
  - `s2_1_codriving_align.py <onnx> [batch=2] [target=llvm]` — relax 导入 + fp32 对齐(通用, 处理动态 batch + target 切换)。
  - `s2_codriving_e2e.py <onnx> <label> <trials> <out_csv> [batch] [reps] [seed]` — DEFAULT(dlight, 含 384→1 头时 relax.build 崩→dlight+Fallback 兜底) vs TUNED(MS + apply DB + dlight Fallback 补剩余块) e2e。
  - 驱动 `run_cod_e2e.sh <tag> <gpu> <onnx>`(H800)。
- **素材(H800)**: `models/codriving_cache/{base,p50,p75}_backbone.onnx`(全 S2.1 align PASS); CoDriving body/collab ONNX 在 `${V2X_DATA_ROOT}/V2Xverse_pyramid/output/codriving_pilot/{base,p25,p50,p75}/` + `collab_export/`。
- **结果**: `cod_e2e.csv`(P2)。

---

## 5. 纪律 / caveat
- H800 是 Hopper 数据中心卡 ≠ 边缘标的; 相对/机理结论(旋钮价值/耦合)在此坐实, **边缘绝对延迟最终须 4090/Orin 实测**。
- DEFAULT=dlight 是"旋钮关"基线, default/tuned 比 = 开启搜索的价值, **非 vs TRT/强手写**。
- fp32(relax 无量化 pass); INT8 e2e 须量化 pass 或 BYOC-TRT。
- **backbone-only ≠ 全 pipeline**: VFE/scatter/decode/NMS 非 conv、TVM 不调度, 按 Amdahl 稀释 backbone 内加速(这正是 CoDriving v2 §8 的全网瓶颈)。全 pipeline 真收益看 P3。
- 任何"加速/对齐/耦合"结论必跨配置(seed/trial)验稳; 延迟空闲 GPU 实测; agent 自报必复核。
