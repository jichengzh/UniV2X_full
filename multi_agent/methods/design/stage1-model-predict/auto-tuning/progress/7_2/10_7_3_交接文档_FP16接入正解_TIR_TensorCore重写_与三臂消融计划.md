# 10_7_3 交接文档 — FP16 接入正解(TIR TensorCore 重写)+ 三臂消融计划

**时间**: 2026-07-03
**承接**: `9_7_3_P2实施...md`(P2.0-P2.2 + int8/energy 修复)+ 用户指路(读 6_29 doc6-12 / 6_30 doc1-2 的 FP16 加速工作)
**用途**: compact 前落盘。记录 **① FP16 接入正确路线(我之前走错了)② 三臂消融执行计划**。restart 后按本文直接开工。

---

## 〇、★核心纠错(我之前 fp16 失败的真因)

我在 `9_7_3` 里试图用 **MetaSchedule tune** 复现 fp16, 得到 47.2ms(32-trial)/39.5ms(128-trial, 且 CUDA 崩)—— **这是走错了路**。

**真相(读 6_29/6_30 文档 + 核验 180 数据后确认)**: original60 的 180 条里 fp16 权威口径**不是 metaschedule**, 而是 **TIR 组卷积 TensorCore 重写**。核验:
| fp16 路线 | s2_160 latency | schedule_policy | 是否权威 |
|---|---|---|---|
| **fp16_rewritten_tensorcore_full60** | **11.642ms** | apshape_tensorcore_rewritten_full60 | ✅ **训练表用的就是这个** |
| fp16_true(metaschedule) | 47.530ms | metaschedule_tuned | ✗ 慢路线(=我复现出的 47ms) |

⇒ **我复现的 47ms 恰好就是那条被废弃的 metaschedule 慢路线**; 正确的 11.64ms 来自 TIR 重写。`fp16_true_original60_latency_rows_v1.jsonl` 是历史失败路线, 别再用。

---

## 一、FP16 加速真因与正解(来自 6_29 doc10-11 / 6_30 doc1-2)

### 1.1 根因
- 默认 FP16 lowering **不把主耗时的 3×3 组卷积(groups=32)送进 TensorCore** → 默认 fp16 ≈ fp32(~19ms 级)。int8 有 dp4a 直接快, fp16 默认无 tensorcore 故不快。这是 fp16/int8 加速根因的**区别**。

### 1.2 正解 = 全图内 TIR 组卷积重写 + 选择性 MatmulTensorization
- **脚本**: `scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py`(H800 已在), mode=`full-engine-group-conv-rewrite`。
- **方法**: 定位热点组卷积 PrimFunc(如 `fused_conv2d12_add8_relu6`/`fused_conv2d16_add8_relu6`)→ 同签名 im2col/matmul/restore 替换 → 选择性 `MatmulTensorization` → 生成 WMMA。
- **★两个必须的关键点**(否则 wmma=0):
  1. **`--cast-fp16-source` 必须加**: 否则 matcher 匹配 fp16 buffer 但源是 fp32 → replace_records=[] → wmma=0。
  2. **`s_tir` attr bug 已修**: `_apply_selective_matmul_tensorization` 原硬依赖 `func.attrs["s_tir"]`, H800 TVM 0.20 不设该 attr → 全 PrimFunc 跳过 → wmma=0。已改为**文本检测**(`_classify_full_engine_group_conv_primfunc_text`, 认 `T.Buffer((256,8,3,3),"float16")` 权重 + stride 模式)。
- **实测闸门(lhc_07, H800 sm90)**: speed-gate shape `[2,64,128,256]` default **18.63→12.80ms, 1.4559×, wmma=72, tvm_mma_sync=2, tensorcore_gate=true**。AP-shape `[2,64,256,256]` default 36.62→24.48ms 1.496×。

### 1.3 权威 latency 测量口径(full60 已 rollout)
- **rollout 产物**: `rows/fp16_rewritten_tensorcore_full60_latency_rows_v1.jsonl`(60 行)+ `..._energy_rows_v1.jsonl`。route=`apshape_tensorcore_rewritten_full60`, 每 label 有重写 artifact 在 `${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_original60_20260630_fullval/<label>_gpu0/rewrite_exports/`。
- **两步 recipe**(见 `scripts/stage2_launch_fp16_rewritten_full60_measure_lanes.sh`):
  1. **产 artifact**: probe 脚本 `--cast-fp16-source` → 输出 rewrite 后的 TensorCore engine(.so)+ `fp16_<label>_full_engine_group_conv_rewrite_latest.json`(rewrite_report)。
  2. **测 latency/energy**: `scripts/stage2_measure_fp16_rewritten_artifact.py --kind latency --artifact <engine> --rewrite-report <json> --route apshape_tensorcore_rewritten_full60 --width <> --config-id <> --warmup-iters 20 --measure-iters 300 --repeat 5 --out-jsonl <>`(engine_kind=tvm_vm, build_status=loaded_exported_artifact)。`--kind energy` 出能耗。

### 1.4 ★遗留缺口(必读, 影响 fp16 AP 而非 latency)
- **latency 闸门已关**(TensorCore 真加速已证)。但 **AP-safe 未关**: 重写 im2col/matmul 的 reduction/rounding 顺序 ≠ 默认 TOPI group_conv → fp16 非结合性 → **output2 漂移**(max_abs_err~0.727, mean_err/orig~0.19)。故 **fp16 重写路线的 AP 需专门 AP-safe bridge**(6_30 doc1-2 的 APShape matcher 修复在推进, `fp16_rewritten_ap_safe` 曾 missing)。
- ⇒ **对本阶段 SMBO/三臂: fp16 的 latency+energy 可信直接用**; fp16 的 **AP** 走已测 60 点或待 AP-safe bridge, 不要用重写引擎现算 AP(会漂)。

---

## 二、如何把 FP16 接入 measure_config(restart 后 P1 收尾)

现状: `framework/measure_config.py` 里 int8 已接(pack route, `measure_config_int8`), fp16 还是 `NotImplementedError`。**正确接法 = 复刻 §1.3 两步 recipe**, 不是 metaschedule:

1. **`measure_config_fp16(width, gpu, log_dir)`**:
   - export fp32 ONNX(复用 `export_onnx`)。
   - **step1 rewrite**: 调 `stage2_fp16_tensorcore_convblock_and_engine_probe.py`(mode full-engine-group-conv-rewrite, **`--cast-fp16-source`**, `--onnx <fp32>`, `--label`, `--export-dir`, `--raw-dir`, speed-gate `--batch 2` + shape `[2,64,128,256]`)→ 得 engine + rewrite_report。
   - **step2 measure**: 调 `stage2_measure_fp16_rewritten_artifact.py --kind latency --artifact <engine> --rewrite-report <json> --route apshape_tensorcore_rewritten_full60 --width --config-id --measure-iters 300 --out-jsonl <>`; 再 `--kind energy`。
   - 解析 latency_p50_us/1000 → lat_ms; energy 用**总功率×延迟**(与 int8/fp32 一致, 见 9_7_3 Problem2 修复; parse watt_avg)。
2. **口径自检**: 先在已测 label(如 s2_160)复跑, 断言 lat≈11.64ms(±ε)才算接对。**警惕**: 若得 ~47ms = 又落回 metaschedule 慢路线 = 没走 TIR 重写。
3. **env**: 与 int8 同 —— 预设 `LD_LIBRARY_PATH`(bundled cudart)+ tvm310 python(见 9_7_3 坑)。probe/measure 均在 H800 跑。
4. **注意 shape 口径**: latency 用 speed-gate `[2,64,128,256]`(= fp32/int8 同 ONNX 输入)。AP-shape `[2,64,256,256]` 只在 AP eval 用, 别混。

**验收标准**: 三精度(fp32 74 真测 / int8 pack route / fp16 TIR 重写)对同一新宽度都能出 lat+energy, 且已测宽度口径复现(fp32 s2_160 49.74≈49.78 / int8 8.75≈8.70 / fp16 ≈11.64)。

---

## 三、三臂消融(Gap1)执行计划

### 3.1 目标与判据
证 **联合搜索(P×Q×S)> 串行/默认**。三臂(同预算):
- **S0** = 默认调度(不 tune, 不重写)。
- **S1** = 串行: 先剪枝定形 → 再单独调度/量化(prune-then-tune)。
- **S2** = 联合: prune×quant×schedule 同时搜。
判据: S2 的 Pareto HV(超体积)> S1 > S0; **且 S2 命中 S1/S0 错过的非支配点**(单维次优组合最优)。工具 `framework/run_pqs_ablation.py`(已存, 见记忆: B4 真数据 A-joint/A-serial/A-noS, Wilcoxon p=4.88e-4 已复现)。

### 3.2 ★关键前提: 必须有真 AP-trade-off 载体
- 现状: Pyramid+DAIR 的 AP 是**高原**(近平)→ AP×latency Pareto **退化**(缩宽度近免费直到崖口)→ **单精度单轴联合搜索无 payoff**(9_7_3 三轮 SMBO 已证 latency 收敛在 9.83ms, 无 trade-off)。
- ⇒ 三臂消融的 trade-off **必须来自精度轴 / AP 崖口**:
  1. **精度轴(现在可用)**: fp32/fp16/int8 三档已全有真口径。**int8 是真 trade-off 轴**(快但可能掉 AP = Problem3)。三臂在 **(宽度 × 精度)** 联合空间跑, 让 S2 能同时选"窄+int8"组合, S1 只能先定宽再定精度。
  2. **AP 崖口**: 补 80/87/93% 剪枝率 + int8 边界找崖(记忆 project-dair-ap-axis-collapse: 原始剪枝真崩需 finetune)。
- **Problem3 是三臂的前置**: 先定"低比特+低宽度能否保 AP"(int8 真 finetune AP), 有了真 AP 代价, 三臂才有可分的前沿。建议顺序: **先 int8 AP(Problem3)→ 再三臂**。

### 3.3 数据资产(三臂可直接用)
- fp32: 60(original)+14(SMBO round1-3 真测)= 74 latency 真测点。
- int8: 60 latency(pack route 口径)+ energy; 新宽度可扩(measure_config_int8 已通)。
- fp16: 60 latency(TIR 重写口径)+ energy; 新宽度待 §2 接完。
- AP: fp32 60 真测(finetune); int8/fp16 AP 待补(Problem3 + AP-safe bridge)。

### 3.4 执行步骤(restart 后)
1. **收尾 fp16 接入**(§2)→ 三精度新宽度可测。
2. **Problem3 = int8 AP 真测**: 选 Pareto 前沿 Top-K 的 (窄宽度 × int8) 组合, 真 finetune(≤31 epoch)+ 1789 全 val AP, 定 int8 精度代价。用 int8 AP bridge(`stage2_h800_native_int8_real_activation_bridge.py`, 已在 9_7_3 前序优化过 persistent worker)。
3. **三臂**: 在 (宽度×精度) 联合空间跑 S0/S1/S2, 内环 schedule(int8 pack route / fp16 TIR 重写 / fp32 metaschedule), 外环 NSGA-II 3 目标(AP,lat,energy)+ feasibility 门。比 HV + 找 S2 独有非支配点。`run_pqs_ablation.py` 扩到三精度。

---

## 四、关键文件与命令索引

| 项 | 路径 |
|---|---|
| FP16 重写 probe | `scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py`(mode full-engine-group-conv-rewrite, **--cast-fp16-source**) |
| FP16 measure | `scripts/stage2_measure_fp16_rewritten_artifact.py`(--kind latency/energy --artifact --rewrite-report --route apshape_tensorcore_rewritten_full60) |
| FP16 full60 launcher(recipe 范本) | `scripts/stage2_launch_fp16_rewritten_full60_measure_lanes.sh` |
| FP16 权威 latency 行 | `rows/fp16_rewritten_tensorcore_full60_latency_rows_v1.jsonl`(s2_160=11.642) |
| int8 native route(pack, 权威) | `${V2X_DATA_ROOT}/s2_tvm/native_int8_pack/.../int8_native_route/stage2_h800_native_int8_full_onnx_route.py` |
| measure_config(int8 已接) | `framework/measure_config.py`(`measure_config_int8` ✅; `measure_config_fp16` 待接=§2) |
| 三臂消融 | `framework/run_pqs_ablation.py` |
| 6_29 fp16 根因/完成 | `progress/6_29/{10,11}_6_29_*.md` |
| 6_30 fp16 AP-shape/full-val | `progress/6_30/{1,2}_6_30_*.md` |

## 五、restart 首步
1. 读本文 §0(纠错)+ §2(fp16 接法)+ §3(三臂)。
2. 收尾 `measure_config_fp16`(§2), 口径自检 s2_160≈11.64ms(得 47ms=又走错 metaschedule)。
3. Problem3(int8 AP)→ 三臂(§3.4)。

---

## §6 [2026-07-03 实测收尾] 三精度 measure_config 接入完成 + 关键口径 bug

### §6.1 measure_config_fp16 已接入并验证
`framework/measure_config.py` 现三精度全路由: fp32=metaschedule / int8=pack route /
**fp16=TIR TensorCore 重写**(probe `--mode full-engine-group-conv-rewrite
--cast-fp16-source` → `stage2_measure_fp16_rewritten_artifact.py`)。artifact 路径从
rewrite report 的 `export_library.path` 读取(probe 把 .so 落在 raw_dir 的子目录, 不是
raw_dir 根)。

### §6.2 已测宽度口径复现 s2_160=[64,128,160] (全 ✅)
| precision | measure_config 实测 | 冻结 LUT | Δ |
|-----------|-----|-----|-----|
| fp32 | 49.858 ms | 49.78 | +0.08 (噪声内) ✅ |
| int8 | 8.725 ms | 8.70 | +0.03 ✅ |
| fp16 | 11.577 ms | 11.642 | -0.065 ✅ (gate=True, wmma=180) |

### §6.3 同一新宽度 [48,64,160] 三精度都出 lat+energy (✅)
| precision | lat_ms | energy_j | watt_avg | build |
|-----------|--------|----------|----------|-------|
| fp32 | 45.528 | 9.736 | 213.8 | True |
| int8 | 5.195 | 1.749 | — | True |
| fp16 | 9.094 | 2.942 | 323.5 | True |

### §6.4 ★★★ 关键口径 bug — 精度轴输入分辨率不一致(必须裁决)
排查中发现: **冻结 fp16 rewrite LUT 建在 AP-shape [2,64,256,256], 而 fp32/int8
original60 LUT 建在 [2,64,128,256]**(`stage2_original60_export_onnx.py` 硬编码
`INPUT_SHAPE=(2,64,128,256)`; probe 从 ONNX 读空间维)。后果:
- 同 shape [128,256] 下 fp16 s2_160 = **6.06ms** < int8 8.70ms(fp16 本应更快);
- 混 shape 下 fp16 11.64@256² > int8 8.70@128² → **fp16 被系统性地显得更慢**, 把精度
  轴的真实序 (int8<fp16<fp32) 颠倒, 直接污染联合 Pareto / SMBO / 三臂消融。

**已做的临时对策**: 给 `stage2_original60_export_onnx.py` 加 `--input-hw`(默认 128,256
保 fp32/int8 不变; fp16 走 256,256 → 后缀文件名 `_ap256x256.onnx`), measure_config_fp16
用 `FP16_INPUT_HW=(256,256)` 复现冻结 11.64。**这只是"复现冻结数"**, 没消除不一致。

**待用户裁决的分叉(改动量不同)**:
- 方案A(推荐, 便宜): fp16 LUT 重建在 128×256, 与 fp32/int8 对齐。60 条 fp16 行重测
  (每条 rewrite+measure ~3min); fp16 数值整体≈折半。改 `FP16_INPUT_HW=(128,256)`。
- 方案B(贵, 更贴真实 AP eval): fp32/int8 74+60 行重建在 256×256(改 export 默认 +重跑
  metaschedule/pack)。
- 方案C: 保持混 shape 但在 cost model/Pareto 里按 shape 归一(不推荐, 埋雷)。

### §6.5 改动文件
- `framework/measure_config.py`: +measure_config_fp16, +FP16_* 常量, export_onnx +input_hw
- `scripts/stage2_original60_export_onnx.py`: +--input-hw(默认不变, 后缀隔离 AP-shape)
两者 H800 已同步; ast.parse + import OK。结果落
`${V2X_DATA_ROOT}/s2_tvm/smbo/results/{label}_{prec}/measure_config_result.json`。

---

## §7 [2026-07-03 方案A 执行] fp16 口径统一到 128×256 + int8 未张量化真因

### §7.1 用户裁决 = 方案A(fp16 对齐 128×256), 附两条件均已满足
1. **备份 256×256 fp16 LUT 作证据**: `backup_fp16_apshape256_preharmonize_20260703/`
   (60 行 latency+energy+ap rows + README_evidence.md), 原始不动。
2. **回答"fp16 折半后为何快过 int8"**: 见 §7.3(真查 kernel, 非猜测)。

改动: `FP16_INPUT_HW=(128,256)`。measure_config_fp16 现三精度同 shape 导出。

### §7.2 方案A 验证(2 宽度, 128×256, gate=True/wmma=180)
| width | fp16@128 lat_ms | fp16 energy_j |
|-------|------|------|
| s2_160 [64,128,160] | 6.076 | 1.827 |
| [48,64,160] | 4.782 | 1.482 |
→ s2_160 fp16 从 11.64(256²) 降到 **6.08(128×256)**, 与 fp32/int8 同 shape。

### §7.3 ★真因: fp16 快过 int8 = 张量化层级不对等, 不是精度算得快
真查 int8 pack route 的调度器 `schedule_block`(在
`stage2_h800_native_int8_capability_probe.py`): **只做 fuse→split→bind(blockIdx/
threadIdx), 无 tensorize / wmma / dp4a / mma**。即 int8 conv 是
**一个输出元素一个线程 + 串行归约的朴素 SIMT**(topi conv2d_nchw/group_conv2d_nchw,
int8×int8→int32), 跑在普通 CUDA 核, 连 shared-mem tiling / 向量化都没有。

fp16 rewrite 则是 im2col→dense matmul→**WMMA m16n16k16 张量核**(wmma=180)。

| | int8 pack route | fp16 rewrite |
|-|-|-|
| 硬件单元 | SIMT CUDA 核 | fp16 Tensor Core |
| 张量化 | ❌ | ✅ |

Tensor Core matmul 吞吐 ≫ 一元素一线程 SIMT(约一个数量级), 远超 int8 对 fp16 的 2×
算术密度优势 → 即便同 shape, fp16(6.08) 仍 < int8(8.73)。**结论: 当前 int8 LUT
低估 int8 真实速度**; 精度轴有两层不公平 —— ①shape(方案A 已修) ②优化层级(int8
未张量化, 待修)。若 int8 张量化到 dp4a/INT8-TC(Hopper INT8-TC≈2×fp16-TC), int8
应反超 fp16。这是"fp16 快过 int8"反直觉现象的根因, 也是下一步公平精度轴的必修项。

### §7.4 进行中 + 待办
- **进行中**: 60 行 fp16 LUT 在 128×256 重建 (`rebuild_fp16_128.sh`, 6 GPU 轮转,
  结果落 `results/{label}_fp16/measure_config_result.json`, 进度
  `rebuild_logs/PROGRESS.txt`)。
- **待办(批次完成后)**: ①聚合 60 行 → 新 `fp16_rewritten_tensorcore_full60_latency/
  energy_rows_128x256_v1.jsonl`; ②cost model 用新 fp16 列重训; ③int8 张量化(dp4a/
  INT8-TC)以修复优化层级不对等 = 真公平精度轴的前提(先于三臂消融)。

---

## §8 [2026-07-03] int8 张量化可行性闸门 — 机制已通, 需全引擎融合

### §8.0 用户问答: fp32 要不要张量化 → 不要(硬件层证据)
`get_mma_intrin_group` 的 `in_dtype` 只接受 `{float16, int8, float8_*}`, **不含 float32**。
NVIDIA 张量核不吃真 IEEE fp32(硬张量化=降 TF32=另一个精度点)。故 **fp32 保持 FFMA
参考基准, 不张量化**。不对等只在 fp16-vs-int8(两者都该上张量核, 之前只 fp16 上了)。

### §8.1 关键发现: 用错了 dlight 规则
int8 matmul 用 fp16 的 `MatmulTensorization` → 不张量化(post_counts 全 0)。this TVM fork
(`tvm.s_tir`, tir 被改名 s_tir) 有**专门的 `MatmulInt8Tensorization`**(在
`tvm/s_tir/dlight/gpu/matmul.py:707`, 内部 `get_wmma_intrin_group(in_dtype="int8",
out_dtype="int32")`)。换成它 → **int8 成功张量化**(wmma=36, tvm_mma_sync=1)。

### §8.2 闸门结果(`scripts/stage2_int8_tensorcore_gate_v1.py`, 单 group-conv 块)
| 版本 | tensorized | 单块 lat_ms |
|------|-----------|------|
| int8 naive (fallback schedule) | no | 1.088 |
| int8 MatmulInt8Tensorization | **yes(wmma=36)** | 1.260 |

★单块张量化反而略慢 = 微基准把 **im2col+cast 物化到 global memory** → memory-bound,
matmul 只占小头, 张量化省的算力被 im2col 访存淹没 + ldmatrix 开销。**这不是否定** —— fp16
的加速也**不是单块拿到的, 是全引擎融合**(im2col 折进 matmul 的 shared-mem load)后网络级
拿到的。闸门证明的是**机制通**(int8 能张量化), 不是单块加速。

### §8.3 剩余工作(真正的网络级 int8 张量化)
把 fp16 probe 的 **full-engine-group-conv-rewrite** 复制出 int8 变体:
1. TE builder `_make_te_group_conv_full_im2col_same_signature_primfunc`: 加 int8 支路
   (x_col/w_mat cast int8, matmul int32 累加, 输出 cast 回 fp16 保签名);
2. relax wrapper 同签名(fp16 I/O, 内部 int8 matmul);
3. `_apply_selective_matmul_tensorization` 换用 **MatmulInt8Tensorization**;
4. 全引擎 legalize→fuse→rewrite→build→measure, 对比 naive 8.73ms(网络级)。
预期: 与 fp16 6.08ms 同机制, int8 应 ≤ fp16(Hopper INT8-TC≈2×fp16-TC), 修复优化层级
不对等。工程量: 中(复用已验证管线, 主要是 dtype 参数化 + 换规则), 但需网络级 build 验证。

---

## §9 [2026-07-03] ★全引擎 INT8 张量化重写 — 已实现并验证(精度轴修复)

### §9.1 实现: 给 fp16 probe 加 `--rewrite-dtype int8`(复用全引擎管线)
改 `scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py`:
- TE builder `_make_te_group_conv_full_im2col_same_signature_primfunc`: 加 `accum_dtype
  ="int32"` 支路 —— 物化 int8 buffer(x_col_q/w_mat_q), matmul int8×int8→int32, 输出
  cast 回 fp16 保签名(dummy scale=1, 纯延迟, AP 不受影响)。
- `_replace_group_conv_primfuncs_for_full_engine` + `_apply_selective_matmul_tensorization`
  加 `accum_dtype`; int8 时用 **MatmulInt8Tensorization**(非 fp16 的 MatmulTensorization)。
- 新增 `--rewrite-dtype {float16,int8}`(默认 float16, fp16 路径零改动)。

### §9.2 验证 s2_160=[64,128,160] @128×256(全引擎 build + 权威 measure)
| 精度/路径 | lat_ms | 张量化 |
|-----------|--------|--------|
| fp32 (FFMA 参考) | 49.86 | — (fp32 无 TC 路径) |
| int8 naive (旧 pack route schedule_block) | 8.73 | ❌ SIMT |
| **int8 tensor-core (新)** | **5.871** | ✅ wmma=180, mma_sync=5, 5/5 group conv |
| fp16 tensor-core | 6.076 | ✅ |

→ **int8-TC(5.87) < fp16-TC(6.08) < int8-naive(8.73)**。① 张量化让 int8 从 8.73 提到
5.87 = **1.49×**; ② int8 现在正确地略快于 fp16(Hopper INT8-TC>fp16-TC), 精度轴的**第②层
不对等(优化层级)已修复**。probe 内部计时 5.865 与权威 measure 5.871 一致。artifact:
`int8_fullengine_s2_160/raw/.../rewritten_full_engine.so`。

### §9.3 剩余(rollout)
1. measure_config 接入 int8-TC(measure_config 加 int8-tensorcore 路径, 复用 fp16 rewrite 流程 + --rewrite-dtype int8)。
2. 60 宽度 int8-TC LUT 重建(替换 naive int8 列)+ energy。
3. cost model 用 fair 的 fp16/int8-TC 列重训 → 三臂消融。

### §9.4 [已接入 measure_config + 验证] int8_tc 走统一接口
`measure_config(width, "int8_tc", gpu)` = measure_config_fp16(rewrite_dtype="int8"),
复用全引擎 rewrite→measure 流程, 128×256 口径。验证 s2_160: **lat 5.859ms(≈standalone
5.871)/ energy 1.680J / gate=True / wmma=180 / mma_sync=5**。fair 精度轴(s2_160 @128×256):
fp32 49.86/9.04 · int8_naive 8.73/~3.1 · **int8_tc 5.86/1.68** · fp16 6.08/1.83 —— int8_tc
延迟+能耗双双优于 fp16(int8-TC 应然)。改动: `framework/measure_config.py`(measure_config_fp16
加 rewrite_dtype 参数; precision "int8_tc" 路由; --precision choices +int8_tc)+
`scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py`(--rewrite-dtype int8, TE builder
accum_dtype=int32 支路, MatmulInt8Tensorization)。H800 已同步, ast+import OK。

### §9.5 rollout 待办(与 fp16 60 行同法)
用 `rebuild_fp16_128.sh` 同款批处理跑 `--precision int8_tc` × 60 宽度 → int8_tc LUT 列
(替换 naive int8 或作新列, 待定) → cost model 重训 → 三臂消融。

---

## §10 [2026-07-03] fair-axis rollout: 聚合 + cost model 重训 + 三臂消融

### §10.1 聚合(替换方案已执行)
训练表 `cost_model/train/original60_training_table_latest.{csv,json}`(180 行):fp16 列
换成 128×256 值, int8 列换成 int8_tc(tensor-core)值, fp32 不变, ap70 不变(纯延迟/能耗
替换)。旧表归档 `backup_.../original60_training_table_PREFAIR*`。naive int8 保留作 ablation。

### §10.2 cost model 重训(leave-one-width-out CV)
| target | PREFAIR (256²/naive) | FAIR (128²/int8_tc) |
|--------|------|------|
| latency spearman | 0.9872 | **0.9906** |
| latency MAE(ms) | 1.250 | 1.436 |
| energy spearman | 0.9743 | 0.9554 |
| energy MAE(J) | 0.389 | 0.599 |
排名保持优秀(lat spearman 0.99); MAE 略升 = fair 轴把 fp16/int8_tc 压进更窄重叠带(int8_tc≈fp16)
绝对拟合更难, 但这是诚实现实。模型 `cost_model/models/stage2_cost_v2_*_logval_v2.txt`。

### §10.3 ★三臂消融 fair-axis — joint>serial 稳健(不是 buildability 假象)
先证: **int8_tc(im2col+MMA)不受 dp4a pack-4 buildability 墙约束** —— 60 measured 宽度里
46 个是 dp4a-unbuildable(in_per_g=w0//16 %4≠0), int8_tc **全部 build+tensorize**(含 10 个
w0=48/in_g=3, mma_sync=5)。故 dp4a"墙"是 NCHWc-dp4a kernel 的属性, 不是 int8 本身。

三配置 × 12 seed(`framework/run_pqs_ablation.py`, 新增 `--int8-buildable-all --int8-speedup`):
| arm | 墙ON dp4a ×1.449 | 墙OFF int8_tc ×1.449 | 墙OFF ×1.095(真网络级) |
|-----|------|------|------|
| A-joint | 100% | 100% | 100% |
| A-serial | 85.3% | **90.2%** | 87.4% |
| A-noS | 56.7% | 63.6% | 53.7% |
| Wilcoxon p | 4.9e-4 | 4.9e-4 | 4.9e-4 |

**三者全 PASS(p=4.9e-4)**。分解: buildability 墙只贡献 ~5pp(去墙后 serial 85.3→90.2);
robust 的 ~10pp 残差 = **argmin drift**(fp16-最优宽度 ≠ int8-最优宽度, CHaNAS 型耦合)。
⇒ **joint>serial 在更公平更强的 int8_tc kernel 下依然成立**, 不是 buildability quirk 的假象。
★遗留: 去墙时 run_pqs_ablation 的耦合诊断打印文字仍是旧 "structurally cannot reach"(stale),
HV/Wilcoxon 数值才是证据; 诊断文案待更新为 argmin-drift 口径。

### §10.4 待办
- AP 轴(Problem3): int8_tc 用 dummy scale 纯延迟, **AP 仍是 plateau/未测真 int8 AP**;
  三臂用的 ap70 是 naive-int8 finetune 值 + 0.008 penalty 代理。真 int8_tc AP-trade-off
  仍是"真载体"缺口(见 §3 三臂前提)。
