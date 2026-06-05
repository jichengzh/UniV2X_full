# Orin Pyramid 部署实测报告 + HEAL 无 TRT 部署方案 v1

> 2026-06-05–06 | Task #11 + Task #12 (用户授权实验) 详细报告 + D1v2 全链路真测(含NMS+wallclock)
> 原始数据: `results/E7_orin_e2e_baseline_vs_best.csv`(TRT body) + `results/E8_orin_e2e_fullchain.csv`(PyTorch全链v2)
> 执行: hw-optimizer(Task#11 body+NMS; Task#12 D1v2 PyTorch全链含NMS) → team-lead 亲测收尾(Task#11 fp16/int8复测)
> 状态: §3 TRT body 双执行者真测 ✅; §4 D1v2 全链含NMS真测 ✅; **v1.4 结论级修订 2026-06-06**(ISS-039 #3: enc+bb估算被推翻)

---

## §0 摘要

Orin AGX 30W 模式下, Pyramid(DAIR m1) 双层真测: ①**TRT body**: FP32 131.33ms → p75+INT8 20.02ms = 6.56×; ②**PyTorch FP32 全链路 v2**: B=2 e2e_wallclock=**260.61ms > 200ms** (首次真测坐实"未优化不可达"); ③**混合链合成(B=2)**: base FP16 **~136ms** ✅, p75+INT8 **~108ms** ✅。⚠️ **结论级勘误(ISS-039 #3)**: 旧估 enc+bb 10-25ms → 真测 B=2 **68.94ms(>3×差距)**; 可达性矩阵翻转 — **base FP16+LTE≈236ms ❌, p75+INT8+LTE≈208ms ⚠️临界**; 旧"最悲观情形仍可达"不再成立, 压缩必要性维持但须诚实标注 LTE 情形下的临界状态。

---

## §1 背景与授权

- 用户问题: "Pyramid 在 Orin 上端到端优化前速度是多少, 最优方案能优化到多少?"
- 授权: 用户拍板 + team-lead 授权原文(Task #11), Orin 密码本 session 已向用户确认。
- 关联: 补 `edge_latency_budget_v1.md`(#9) 的估算缺口; 支撑 200ms 闭环验证故事(#8)。

## §2 测量环境与协议

| 项 | 值 | 留痕 |
|----|----|------|
| 设备 | Orin AGX 32GB, **MODE_30W (nvpmodel ID=2), GPU 612MHz** | 启动宣告快照 |
| 软件 | TRT 8.5.2.2, JetPack (torch 1.12.0a0+nv22.3) | trtexec log 头 |
| 测量 | `trtexec --loadEngine --warmUp=200 --duration=3 --avgRuns=200 --noDataTransfers`, 取 **GPU Compute median** | build/复测 log |
| 设备空闲 | 测前 GR3D 0% / RAM 11.7G/62.8G / 无他人进程; 测后 tegrastats 快照 GPU@49.1°C | 宣告 + 收尾快照 |
| 口径 | **body_subnet_collab2** = pyramid_backbone+deblocks+shrink_conv+heads, B=2 双 agent 协同; **≠ e2e**(不含 voxelize/encoder/NMS) | ONNX 输入 2×64×128×256 |

## §3 真测结果

### 3.1 body 三档对照 (全部 Orin 本机真测)

| 配置 | body median | min–max | vs FP32 | 来源 |
|------|------------|---------|---------|------|
| **TRT FP32 (零优化)** | **131.330 ms** ★正式值 | 131.17–139.36(p99) | 1.00× | hw loadEngine 200-run (`E7` AUTHORITATIVE); build 内嵌 133.651ms(24-run) 为参考值, 1.7% 系方法差非冲突 |
| TRT base FP16 | 47.986 / 47.932 ms | 47.92–51.6(p99) | 2.74× | 双执行者(hw/team-lead)同协议独立测, drift 0.11%; 与 E6 48.645 一致 |
| TRT p50+INT8 | 32.967 ms | — | 3.98× | E6 历史真测(本次未复测) |
| TRT p75+FP16 | 26.372 ms | — | 4.98× | E6 历史真测 |
| **TRT p75+INT8 (最优)** | **20.020 / 20.050 ms** | 19.99–20.05 | **6.56×** | 双执行者独立测, drift 0.15%; 与 E6 20.311 一致 |

注: 30W 锁频下测量质量高; fp16/int8 两档**双执行者同协议独立测量 drift ≤0.15%**, 加上与 6/3 E6 历史值 ≤1.5% 偏差, 数据可信度获双重交叉确认。完整 7 行(含双执行者+drift 标注)见 `results/E7_orin_e2e_baseline_vs_best.csv`。
[版本注 2026-06-06] 本表为 v1.1(E7 双执行者整合版); 若与旧 133.651 头条版冲突, 以本版为准(双迁移覆盖事故已订正)。

### 3.2 NMS (Orin 真测, 本次新增)

| 档位 | p50 | mean | p99 | 说明 |
|------|-----|------|-----|------|
| N=500 现实档(阈后) | **0.417 ms** | 4.27 | 15.2 | 重尾显著, mean 被拉高 |
| N=7388 极端档 | 19.385 ms | 17.44 | 26.4 | 全图密集框上界 |

- 实现: `torch.ops.torchvision.nms` (CUDA), warmup 200 / measure 500, MODE_30W。
- **caveat**: 2D BEV proxy ≠ HEAL 3D NMS(box 表示不同), 量级代表性成立、精确值待 HEAL 原生部署后替换。

### 3.3 引擎资产清单 (Orin `~/m4_8_orin/p03_build/`)

base/p50/p75 × FP16/INT8 共 6 个(6/3 build) + 本次新增 base_fp32(23MB)。ONNX 源(base/pruned50/pruned75)在盘, 可复 build。

## §4 e2e 全链路与 200ms 对照 [v1.3 Task#12 D1v2 真测 — 含NMS+独立wallclock]

### 4.1 D1 v2 PyTorch FP32 全链路真测 (含NMS, 2026-06-06, 真测)

**测量口径**: CUDA Event v2协议, 分段计时 + 独立外层wallclock双轨; mmcv GPU旋转NMS; warmup=30, n=30; GR3D 0%实查。**冷启注意**: 首测量帧(第1/30帧)encoder仍显JIT冷启spike(B=1 ~40ms/B=2 ~112ms vs稳态17/33ms); p50(中位)不受1/30离群点影响, mean略偏高; wallclock权威值取p50。

| 段 | B=1 单agent p50 | B=2 双agent p50 | 性质 | 来源 |
|----|-----------------|-----------------|------|------|
| voxelize | **5.09 ms** | **10.02 ms** | **Orin 真测** | D1v2, CUDA Event |
| encoder_m1 | **17.49 ms** | **33.49 ms** | **Orin 真测** | D1v2, 稳态 |
| backbone_m1 | **18.46 ms** | **35.45 ms** | **Orin 真测** | D1v2, CUDA Event |
| fusion(body) | **72.88 ms** | **142.72 ms** | **Orin 真测** | D1v2 |
| head | **29.81 ms** | **29.73 ms** | **Orin 真测** | D1v2 |
| NMS (3D mmcv GPU) | **8.55 ms** | **8.54 ms** | **Orin 真测** | D1v2; ~14-28 dets |
| glue | 0.56 ms | 0.57 ms | 真测 | wallclock−Σstages |
| **t_e2e_wallclock p50** | **152.85 ms** | **260.61 ms** | **真测·权威** | E8 D1v2 CSV |
| **t_e2e_sum p50** | 152.28 ms | 260.05 ms | 真测·分段Σ | E8 D1v2 CSV |

**NMS对账(ISS-041 标尺#7)**: 3D mmcv GPU旋转NMS = **8.54ms** vs E7 2D torchvision proxy mean = 4.27ms → **真3D NMS约2× 2D代理**; E7 proxy已替换为真3D测量值。

### 4.2 D2 混合链合成 (PyTorch pre-body + TRT body + 3D NMS, 合成口径)

pre-body(B=2 D1v2真测): vox(10.02)+enc(33.49)+bb(35.45)=**78.96ms**; NMS使用D1v2真测8.54ms

| 链路 | pre-body | body(E7真测) | NMS | e2e合成 | 200ms对照 |
|------|----------|-------------|-----|---------|-----------|
| PyTorch pre + TRT FP32 + 3D NMS | 78.96 ms | 131.33 ms | 8.54 ms | **219 ms** | ❌ 超 |
| PyTorch pre + TRT FP16 + 3D NMS | 78.96 ms | 47.99 ms | 8.54 ms | **136 ms** | ✅ 余64ms |
| **PyTorch pre + TRT p75+INT8 + 3D NMS** | 78.96 ms | **20.02 ms** | 8.54 ms | **108 ms** | ✅ **余92ms** |

**口径声明**: D2 为组件合成(pre-body+NMS=D1v2真测 + TRT body=E7真测); 张量接口开销未独立实测(<1ms估算); 三分量均为 Orin MODE_30W 真测值。

#### 4.2.1 V2X通信延迟叠加可达性矩阵 ⚠️(ISS-039 #3 结论翻转)

通信延迟取固定加法模型: C-V2X≈20ms, LTE≈100ms(典型值, 见 edge_latency_budget_v1.md §2)

| 通信情形 | base FP16 合成 136ms | p75+INT8 合成 108ms | 备注 |
|---------|---------------------|---------------------|------|
| 无通信  | **136ms ✅** 余64ms | **108ms ✅** 余92ms | 两档均安全 |
| +C-V2X 20ms | **156ms ✅** 余44ms | **128ms ✅** 余72ms | 两档均安全 |
| **+LTE 100ms** | **236ms ❌ 超36ms** | **208ms ⚠️ 超8ms 临界** | **旧结论翻转** |

> **⚠️ 结论翻转说明**: 旧报告估算 enc+bb≈10-25ms 导致混合链 base FP16≈66-84ms, 配 LTE 后仍预测"最悲观 184ms 仍可达"。真测 enc+bb=68.94ms, base FP16 混合链=136ms, 配 LTE 后 236ms ❌。**旧"最悲观仍可达"不再成立。**

### 4.3 结论

①**原生 PyTorch FP32 e2e wallclock(B=2, 含3D NMS): 260.61ms > 200ms** — 无优化部署超预算30%; ②**最优方案(TRT p75+INT8): 合成 ~108ms**, 余92ms; ③**压缩优化是 200ms 预算必要条件**, 但 LTE 情形下 p75+INT8 临界(208ms, 超8ms), 并非"任意通信条件稳过" — **base FP16+LTE超线, p75+INT8+LTE临界**; ④叙事须诚实标注: 只有无通信或 C-V2X 情形下两档均可达, LTE 情形下即便最优压缩仍在红线附近。

## §5 与历史数据对账

- ~~#9 估算 "base FP16 RSU≈84ms" → 本次分段合成 61–97ms, 84 落在区间内 ✓~~ **[ISS-039 #3 勘误]**: 真测 enc+bb=68.94ms → base FP16 混合链合成=**136ms**, 旧估 84ms 低估 **1.6×**; 旧区间 "61-97ms" **不再有效**; 该估算被真测推翻, 勘误痕保留。
- ~~#9 "p75+INT8 RSU≈36ms" → 本次 37–54ms 下沿吻合 ✓~~ **[ISS-039 #3 勘误]**: 真测 p75+INT8 混合链合成=**108ms**, 旧估 36ms 低估 **3×**; 旧区间同样失效。
- E6 body 历史值 → 本次复测偏差 ≤1.5% ✓ (TRT body 段本身精度不变; 误差来源是 enc+bb 段估算, 非 body 段)
- **根因**: M2 f 函数在 TRT body 段标定(R²>0.995), 对 PyTorch eager enc+bb 外推无效 — 见 §6 ISS-039 #3 机制说明。

## §6 caveats 与欠账

1. **AP**: p75+INT8 的 AP50≈0.74-0.75(近无损)为 **4090 参考值**; Orin 端 AP 未验(ISS-017: INT8 跨 TRT 版本输出发散)。
2. ~~encoder/voxelize 段为估算~~ → **已由 Task#12 D1v2 真测替换**(见§4.1); NMS 已替换为真3D mmcv GPU旋转NMS真测(8.54ms), 旧2D proxy(4.27ms)作废。
3. **流程欠账**: Task#11 Step3 复测由 team-lead 一手完成, 欠 supervisor 独立复核; dataset_v2 未入库(新 latency_kind 待定义)。Task#12 D1v2 由 hw-optimizer 单执行者完成, 欠第二执行者交叉确认。
4. Task#12 D2 混合链为组件合成(非全链真测); 张量传递接口耗时未独立实测(估计<1ms)。
5. ~~TRT FP32 代理是唯一"未优化"数据点~~ → Task#12 D1v2 已提供 **PyTorch FP32 全链路真测基准**(B=2 e2e wallclock p50=**260.61ms**, 含3D NMS, 双轨计时)。
6. **[ISS-039 #3] enc+bb估算失效机制(结论级勘误痕)**: 旧估 enc+bb≈3.17ms(4090 M2 f 函数外推)严重低估。根因: **M2 f 函数在 TRT body 段标定**(GPU kernel优化路径, R²>0.995), **对 PyTorch eager 阶段外推无效**。PyTorch eager 在 Orin 上开销极大: head 段 Orin eager=29.8ms vs 4090 TRT=1.48ms = **20× 倍率差**, 远超 TRT body 比值(~6.56×); encoder/backbone 各~33-35ms(B=2)同属 eager 低效区。M2 f 函数仅对 TRT 引擎跨平台预测有效, **切勿外推至 PyTorch eager 段**。今后跨平台估算: TRT 段用 M2 f, eager 段须真测或标注"估算无效"。

---

## §7 HEAL 无 TRT 部署方案 (Orin 原生 PyTorch)

### 7.1 环境实查结论 (2026-06-05 SSH 实查)

| 项 | 现状 | 对部署的意义 |
|----|------|-------------|
| Python | 3.8.10 (`~/uniad_venv/`) | HEAL 兼容(4090 侧 3.9, 纯 py 代码无碍) |
| torch | **1.12.0 NVIDIA 版, CUDA 可用**(Orin GPU 识别 ✓) | 核心依赖已就位, 免装 |
| torchvision | 0.13.0 (`torch.ops.torchvision.nms` 实测可用) | NMS 路径已验证 |
| 已有 | numpy 1.22 / scipy / shapely / numba / opencv | HEAL 主要依赖大半已在 |
| **缺** | HEAL/opencood 代码本体; **spconv**(voxelizer 依赖); open3d; easydict | 见 7.3 处置 |
| 磁盘 | 833G 可用 | 数据集传输无忧 |

### 7.2 三条实现路线

| 路线 | 内容 | 工期 | 推荐度 |
|------|------|------|--------|
| **A. 最小推理提取(推荐)** | 只搬 `opencood/models` m1 路径子集 + 自写 voxelizer + standalone 推理脚本(复用 4090 侧 profiling 脚本骨架) | **1–2 天** | ★★★ |
| B. 完整 HEAL 安装 | git clone + requirements 裁剪安装 + `setup.py develop` | 2–4 天(含踩坑) | ★★ 仅当需要完整 eval 管线 |
| C. ONNX Runtime CUDA EP | body 走 ORT(非 TRT), 其余 PyTorch | 1–2 天 | ★ "不用 TRT 但要加速"的折中, 性能介于两者间 |

### 7.3 路线 A 工程步骤 (推荐)

1. **代码子集打包**(4090 侧): `opencood/models/` 中 m1 推理链(pillar_vfe / point_pillar_scatter / base_bev_backbone(resnet) / pyramid fusion / shrink / head)+ `utils` 最小闭包; easydict 用 pip 装(纯 py)。
2. **voxelizer 替换**(关键工程点): HEAL 的 `SpVoxelPreprocessor` 依赖 spconv(aarch64+torch1.12 编译成本高, 不值)。DAIR m1 是 **PointPillar**: pillar 化 = 网格散列+截断采样, **用 numba(已在 Orin)或纯 torch 实现 ~50-80 行**, 与 spconv 输出等价性在 4090 侧先做 diff 验证(同输入 voxel 输出逐元素比对)再上 Orin。
3. **资产传输**: DAIR ckpt(★必须 `Pyramid_DAIR_m1_base_..._11_42_29`, 勿取 04_28_12 OPV2V 版, ISS-012)+ yaml + val 样本(先 50 帧 smoke, 后可全 1789)。
4. **计时脚本**: CUDA Event + `torch.cuda.synchronize`, warmup≥50, 分段 hook(voxelize/encoder/backbone/fusion/NMS), 口径标 `e2e_pytorch_fp32_orin`; 30W 锁定+tegrastats 留痕(json metadata 字段, ISS-039 要求)。
5. (可选) FP16: `model.half()` 或 autocast, 标 `e2e_pytorch_fp16_orin`。
6. (可选, 高价值) **AP eval**: 跑 DAIR val 1789 → **Orin 端 PyTorch FP32 AP 真测**, 与 4090 AP 对账 → 建立"Orin 端精度基准", 后续可扩展到喂 TRT 引擎输出做 Orin INT8 AP 实测, **关闭 ISS-017 缺口**。

### 7.4 实测性能 [Task#12 D1**v2** 真测更新, v1已废弃]

Task#12 D1v2 (v2协议, 含NMS+独立wallclock双轨)已完成 Orin PyTorch FP32 全链路真测(2026-06-06):

| 指标 | B=1 p50 | B=2 p50 | v1旧值(无NMS) | 口径差 |
|------|---------|---------|--------------|-------|
| e2e_wallclock (含3D NMS, 权威) | **152.85ms** | **260.61ms** | 143.05 / 251.02 (无NMS, 已废弃) | +NMS 8.54ms + glue 0.57ms |
| body(fusion+head) | 102.60ms | 172.45ms | 同 | 分段Σ |
| enc+bb(Orin eager真测) | 35.95ms | **68.94ms** | M2估算3.17ms(无效) | **>3× 估算低估 ISS-039 #3** |

**v1废弃声明**: D1_B1/B2_FP32_v1 缺NMS+缺独立wallclock, 已在 E8 CSV 中标注 `_SUPERSEDED`; 本行所有引用以 **v2 权威值**为准。

**修正**: TRT FP32 body(131.33ms) vs PyTorch FP32 body(~172ms B=2) ≈ 1.31× TRT 加速; TRT 加速比低于预期, 因 Orin 上 TRT 对 PointPillar 型网络的优化效果受 memory-bound 限制。

**结论**: **原生 PyTorch B=2 e2e wallclock=260.61ms > 200ms 红线** — 无优化部署超预算30%; 混合链(TRT p75+INT8)合成 108ms, 余92ms。

### 7.5 风险清单

| 风险 | 等级 | 缓解 |
|------|------|------|
| voxelizer 等价性(自写 vs spconv) | 中 | 4090 侧 diff 验证后再上 Orin |
| torch 1.12 vs 4090 侧版本差(算子行为) | 低 | FP32 下逐层输出 diff 抽查 |
| ckpt 格式(flat vs wrapped, ISS ckpt 陷阱) | 低 | 用金标准 flat ckpt; 加载后跑 1 帧 sanity |
| eager 显存(B=2, 62G RAM 共享) | 低 | 实查 ~数 GB, 余量大 |
| open3d 缺失 | 无 | 仅可视化用, 推理链不需要, 直接不装 |

## §8 建议

1. **批准路线 A**(1-2 天): 产出 `e2e_pytorch_fp32_orin` 真 baseline + 替换本报告 §4 的 encoder/voxelize/NMS 估算段为真测。
2. 顺手做 7.3-6 的 Orin AP eval(增量半天), 为后续 Orin INT8 AP 实测(ISS-017 关闭)铺路。
3. 数据入库: 本报告 §3 真测行入 dataset_v2(新 latency_kind: `trt_body_orin_30w` 已有口径对齐 / NMS 行新口径), 待团队重拉后由 data + supervisor 走正常核验入库流程。
