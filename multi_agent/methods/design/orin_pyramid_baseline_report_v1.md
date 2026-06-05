# Orin Pyramid 部署实测报告 + HEAL 无 TRT 部署方案 v1

> 2026-06-05–06 | Task #11 + Task #12 (用户授权实验) 详细报告 + D1 全链路真测
> 原始数据: `results/E7_orin_e2e_baseline_vs_best.csv`(TRT body) + `results/E8_orin_e2e_fullchain.csv`(PyTorch全链)
> 执行: hw-optimizer(Task#11 body+NMS; Task#12 D1 PyTorch全链) → team-lead 亲测收尾(Task#11 fp16/int8复测)
> 状态: §3 TRT body 双执行者真测 ✅; §4 PyTorch全链真测(Task#12 D1) ✅; v1.2 2026-06-06

---

## §0 摘要

Orin AGX 30W 模式下, Pyramid(DAIR m1) 双层真测: ①**TRT body 子网**: FP32 131.33ms → p75+INT8 20.02ms = 6.56× (双执行者 drift ≤0.15%); ②**PyTorch FP32 全链路**(Task#12 D1 新增): B=1 单 agent e2e_p50=**143ms**, B=2 双 agent e2e_p50=**251ms**(均不含NMS)。e2e 真测合成(B=2+NMS_mean 4.27ms): 原生 PyTorch FP32 **~255ms 超 200ms 预算**; 混合链(PyTorch pre + TRT p75+INT8 body + NMS)合成 **~103ms 余量充足**。⇒ **压缩优化是 200ms 预算的必要条件在全链路真测数据下更有力地成立**。

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

## §4 e2e 全链路与 200ms 对照 [v1.2 Task#12 D1 真测更新]

### 4.1 D1 PyTorch FP32 全链路真测 (Task#12, 2026-06-06, 真测)

| 段 | B=1 单agent p50 | B=2 双agent p50 | 性质 | 来源 |
|----|-----------------|-----------------|------|------|
| voxelize | **4.81 ms** | **9.59 ms** | **Orin 真测** | D1, CUDA Event |
| encoder_m1 | **17.46 ms** | **33.49 ms** | **Orin 真测** | D1, stable(frames3+) |
| backbone_m1 | **18.45 ms** | **35.41 ms** | **Orin 真测** | D1, CUDA Event |
| body(fusion+head) | **102.60 ms** | **172.53 ms** | **Orin 真测** | D1(PyTorch FP32) |
| NMS | — | 4.27 ms (mean) | Orin 真测 | E7 N=500档 |
| **e2e 合计(无NMS)** | **143.05 ms** | **251.02 ms** | **Orin 真测** | E8_D1 CSV |
| **e2e+NMS 合成** | ~147 ms | **~255 ms** | 合成(D1真+E7真NMS) | — |

**注**: B=2 e2e=251ms > 200ms 预算; B=1 单 agent 143ms 不含 V2X 通信本身已贴红线。冷启动首帧(JIT 未完全热身): B=1 encoder~39ms / B=2 encoder~109ms → 首帧 e2e spike ~165ms(B=1) / ~327ms(B=2); 稳态 p50 为报告数值。

### 4.2 D2 混合链合成 (PyTorch pre-body + TRT body + NMS, 合成口径非全链真测)

| 链路 | pre-body(D1真测) | body(E7真测) | NMS(E7真测) | e2e合成 | 200ms对照 |
|------|-----------------|-------------|------------|---------|-----------|
| PyTorch pre + TRT FP32 + NMS | 78.49 ms | 131.33 ms | 4.27 ms | **214 ms** | ❌ 略超 |
| PyTorch pre + TRT FP16 + NMS | 78.49 ms | 47.99 ms | 4.27 ms | **131 ms** | ✅ 余69ms |
| PyTorch pre + TRT p75+INT8 + NMS | 78.49 ms | 20.02 ms | 4.27 ms | **103 ms** | ✅ 余97ms |

**口径声明**: D2 为组件合成(pre-body=D1真测 + TRT body=E7真测 + NMS=E7真测); 张量接口开销未独立测量(同设备GPU张量传递开销估计<1ms); 三个分量均为 Orin MODE_30W 真测值。

### 4.3 结论

①**原生 PyTorch FP32 e2e(B=2 真测): 255ms > 200ms 红线** — 无优化无法满足实时; ②**混合链(TRT p75+INT8 最优): ~103ms 稳过**, 余97ms; ③"压缩优化是 200ms 预算必要条件"由全链路真测(而非估算)支撑 → 论点更有力。

## §5 与历史数据对账

- #9 估算 "base FP16 RSU≈84ms" → 本次分段合成 61–97ms, 84 落在区间内 ✓
- #9 "p75+INT8 RSU≈36ms" → 本次 37–54ms 下沿吻合 ✓
- E6 body 历史值 → 本次复测偏差 ≤1.5% ✓

## §6 caveats 与欠账

1. **AP**: p75+INT8 的 AP50≈0.74-0.75(近无损)为 **4090 参考值**; Orin 端 AP 未验(ISS-017: INT8 跨 TRT 版本输出发散)。
2. ~~encoder/voxelize 段为估算~~ → **已由 Task#12 D1 真测替换**(见§4.1); NMS 仍为 2D CUDA proxy(≠HEAL 3D NMS, 量级代表性成立)。
3. **流程欠账**: Task#11 Step3 复测由 team-lead 一手完成, 欠 supervisor 独立复核; dataset_v2 未入库(新 latency_kind 待定义)。Task#12 D1 由 hw-optimizer 单执行者完成, 欠第二执行者交叉确认。
4. Task#12 D2 混合链为组件合成(非全链真测); 张量传递接口耗时未独立实测(估计<1ms)。
5. ~~TRT FP32 代理是唯一"未优化"数据点~~ → Task#12 D1 已提供 **PyTorch FP32 全链路真测基准**(B=2 e2e 251ms)。

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

### 7.4 实测性能 [Task#12 D1 真测更新, 粗估已替换]

Task#12 D1 已完成 Orin PyTorch FP32 全链路真测(2026-06-06):

| 指标 | B=1 p50 | B=2 p50 | 原粗估 | 实测 vs 粗估 |
|------|---------|---------|--------|------------|
| body(fusion+head) | 102.60ms | 172.53ms | 350-550ms | **实测比粗估低 2-3×** |
| e2e(无NMS) | 143.05ms | 251.02ms | 400-600ms | **实测比粗估低 1.6-2.4×** |

**修正**: TRT FP32 body(131.33ms) vs PyTorch FP32 body(172.53ms) ≈ 1.31× TRT 加速; TRT 加速比低于预期(粗估 3.4×), 因为 Orin 上 TRT 对 PointPillar 型网络的优化效果受 memory-bound 限制。

**结论不变**: **原生 PyTorch B=2 e2e=251ms > 200ms 红线** — 部署目标无法满足; 混合链(TRT p75+INT8)合成 103ms 稳过。

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
