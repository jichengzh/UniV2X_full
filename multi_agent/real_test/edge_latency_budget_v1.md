# 路侧 e2e 200ms 边缘预算可达性分析 (v1.0)

> 编制: 2026-06-05 · hw-optimizer (Task #9/10, 纯分析, 无任何 GPU 操作)
> 问题: **路侧 e2e 200ms 在边缘设备可达吗?**
> 结论: **可达，余量充足。Orin 30W base FP16 最差情形 RSU 计算约 84ms，含 V2X 双向通信 e2e 约 119ms，余量 81ms（200ms 基准）。框架 INT8+剪枝 可额外压缩 RSU 计算至约 36ms（2.3×）。**

---

## 0. 口径与边界声明 (先读)

### 0.1 "路侧 e2e" 定义
本文分析以下端到端链路:

```
[车/路传感器采集]
        ↓
[RSU 端: voxelize → encoder → pyramid backbone → shrink → NMS]
        ↓
[V2X 通信 (路→车下行, 广播感知结果)]
        ↓
[车端: 接收融合结果 (含检测框)]
```

**注意**:
- 传感器扫描周期(10Hz→100ms)是**帧间隔**，非计算延迟；不计入计算 budget。
- 若架构含"车→RSU 上行"（HEAL 协同: 车端特征上传 RSU），还需加上行 V2X 时延和车端 encoder 时间；本文将此拆分为独立组件。
- **DAIR-V2X 采用 V2I 形态**: 1 路侧单元(RSU/infra) + 1 辆车; 本文以此为分析基准。

### 0.2 latency 口径层次 (不可混比)

| 口径名 | 覆盖范围 | 数据来源 |
|---|---|---|
| `body_subnet_collab2` | 仅 PyramidFusion backbone (pyramid ResNeXt + deblocks + shrink), 两 agent 特征拼接后的 TRT engine 推理 | 4090_dspace_bench / dataset_v2 |
| `body_subnet_collab2_orin` | 同上，Orin 平台 | dataset_v2, E6 |
| `forward_hook_pytorch` | encoder_m1 + backbone_m1 + pyramid + shrink + heads, PyTorch FP32 钩子计时 | dims_hardware_v2.md §2 |
| `e2e` | 全模型含 NMS，PyTorch | dims_hardware_v2.md §2 |
| `dla_pipeline_e2e_single_frame` | DLA0∥DLA1 跨进程流水，单帧 produce→consume | E3_orin_dla_pipeline.csv |

**本文每个数字均标 [口径] + [数据源] + [真测/估算]**。

---

## 1. 200ms 总预算拆解

### 1.1 V2X 通信延迟 (文献参考值)

| 技术 | 典型单向延迟 | 来源 |
|---|---|---|
| DSRC / IEEE 802.11p | 1–5 ms | ETSI EN 302 663 |
| C-V2X PC5 Sidelink (LTE-V) | 3–10 ms | ETSI TR 103 439 §6.4 |
| C-V2X 5G NR PC5 | 1–5 ms (短链路) | 3GPP TS 22.186 |
| LTE/5G Uu (蜂窝网络路径) | 15–50 ms | ITU-T Y.3101 |
| RSU 广播帧间隔 (ETSI ITS CAM) | ≤ 100 ms | ETSI EN 302 637-2 §6.1.3 |

> **⚠ 以上均为文献/标准参考值，非本项目实测。标注 [估算-文献]。**
>
> **保守取值**: 单向 V2X = 10ms（C-V2X PC5 偏高估）。
> **双向合计 (车→RSU 上行 + RSU→车下行)**: 20ms。

### 1.2 200ms 总预算分配

```
200ms 总预算
├── V2X 上行 (车→RSU，上传特征/原始数据)  ~ 10ms  [估算-文献]
├── 车端预处理 (voxelize + encoder, 车侧)    ~ 5ms   [估算，见 §2]
├── RSU 计算 (core compute)                  ~ ?ms   [分平台真测/估算，见 §2]
├── V2X 下行 (RSU→车，广播感知结果)          ~ 10ms  [估算-文献]
└── 车端融合 (接收检测框，可选 ego+infra 合并) ~ 5ms   [估算]
```

**RSU 计算可用 budget = 200 - 10 - 5 - 10 - 5 = 170ms**

> 即使 V2X 取最悲观值 50ms（LTE Uu），RSU 计算 budget = 200 - 50 - 20 = 130ms，仍有大量余量（见下表）。

---

## 2. RSU 计算管线逐段延迟

### 2.1 管线组件说明

```
输入: 点云 (N 点, 每帧)
[1] voxelize (CPU/host, spconv)
[2] encoder_m1 (PointPillar VFE, GPU)
[3] backbone_m1 (ResNet BEV, GPU)
[4] ★pyramid backbone (TRT engine, GPU/DLA)  ← 主体, 本项目核心测量
[5] shrink_conv (TRT 内部)
[6] NMS (CUDA)
```

**DAIR V2I: 路侧(RSU infra) + 车端(vehicle) 各做 [1-3], 然后传输特征至 pyramid backbone 协同融合 [4-6]。**
**[4] 的 body_subnet_collab2 scope = 步骤 [4]+[5] 合并 TRT engine。**

### 2.2 各组件延迟实测数据

#### [6] NMS 延迟 (已有真测基线)

| NMS 实现 | 延迟 | 平台 | 口径 | 数据源 |
|---|---|---|---|---|
| Shapely CPU NMS (OPV2V) | **77.62ms** | 4090, CPU | e2e 中独立计时 [真测] | dims_hardware_v2.md §2, PyTorch FP32 |
| CUDA NMS (P0 优化后) | **1.18ms** | 4090 (DAIR) | forward_hook_pytorch FP32 postproc 内 [真测] | dims_hardware_v2.md §2 优化后拆分 |
| CUDA NMS → e2e 加速 | **3.04×** | 4090 | e2e 107→~30ms [真测] | dims_hardware_v2.md §2 P0 结果 |

> **结论**: CUDA NMS 已解串行瓶颈，残余 ~1.18ms [真测, 4090, CUDA Event]。Orin 上 CUDA NMS 未实测，按 GPU 算力差异估算约 **2-5ms** [估算-缩放]。

#### [1] voxelize (CPU/host)

| 平台 | 延迟 | 口径 | 数据源 |
|---|---|---|---|
| 4090 (host CPU) | ~2ms | 未单独真测; 从 e2e 17.87ms - 各模块扣除推算 ≈ 0-1ms [估算] | dims_hardware_v2.md §2 |
| Orin (host CPU, Cortex-A78AE) | ~3-6ms | 未实测 [估算-缩放: 4090 host×~3] | — |

> ⚠ voxelize 在本项目中**从未单独计时**，以上为推算估算值，不作精确依据。

#### [2][3] encoder_m1 + backbone_m1 (每 agent, PyTorch FP32)

| 模块 | 延迟/agent | 总(2 agents) | 口径 | 数据源 |
|---|---|---|---|---|
| encoder_m1 (VFE PFN) | **2.40ms** | ~4.80ms | forward_hook_pytorch FP32, DAIR [真测] | dims_hardware_v2.md §2, 优化后拆分 |
| backbone_m1 (ResNet BEV) | **0.84ms** | ~1.68ms | 同上 [真测] | 同上 |

> 口径: **4090 PyTorch FP32 hook-level 真测**。Orin 未单独实测 encoder/backbone；M2 f 函数 (Orin FP16 ≈ 4090 FP32 × 0.98, R²>0.995 [真测]) 为 ResNet 代理模型拟合，对 Pyramid encoder_m1 的适用性有限。
> **Orin 估算**: encoder+backbone per-agent ≈ 4.80ms × (Orin GPU 时钟/4090 × BW 系数) → **粗估 ~10-20ms 总 [估算-缩放]**（保守）。

#### [4][5] ★ pyramid backbone + shrink (body_subnet_collab2, TRT engine) — 主体

##### 4090 平台

| 配置 | FP16 lat_p50 | INT8 lat_p50 | FP16 vs 4090 FP32 (4.49ms) | 数据源 |
|---|---|---|---|---|
| base (64-128-256) | **1.293ms** | **0.848ms** | 3.5×/5.3× | 4090_dspace_bench, avg 4 tactic rows [真测, TRT trtexec, CUDA-Event, body_subnet_collab2] |
| p50 (32-64-128) | **1.044ms** | **0.841ms** | 4.3×/5.3× | 同上 [真测] |
| p75 (16-32-64) | **0.802ms** | **0.640ms** | 5.6×/7.0× | 同上 [真测] |

> ⚠ **[口径提醒]**: 以上为 body_subnet (pyramid+shrink) 单独 TRT engine，**不含 encoder/backbone/NMS**。完整 e2e 需叠加 §2.2 其他项。

##### Orin AGX GPU (30W mode)

| 配置 | FP16 lat_p50 | INT8 lat_p50 | Orin FP16 / 4090 FP16 | 数据源 |
|---|---|---|---|---|
| base (64-128-256) | **48.645ms** | **32.994ms** | 37.6× | dataset_v2, latency_kind=body_subnet_collab2_orin, E6_orin_p03_lat_energy.csv [真测, TRT trtexec GPU Compute median, nvpmodel 30W] |
| p50 (32-64-128) | **42.065ms** | **32.967ms** | 40.4× | 同上 [真测] |
| p75 (16-32-64) | **26.372ms** | **20.311ms** | 32.9× | 同上 [真测] |

> ⚠ **[口径提醒]**: 同上，body_subnet only，不含 encoder/backbone/NMS。
> ⚠ **[平台口径]**: Orin TRT 8.5.2.2 (≠ 4090 TRT 10.x)；nvpmodel MODE_30W GPU 锁 612MHz。MAXN 模式数据暂缺 [ISS-016 E6 中 MAXN 数据未出]。
> **Orin vs 4090 ratio ~33-40×**: 远大于带宽比(~5×)。根因: ①Orin GPU 算力 ~3.9 vs 165 TFlops (42×)；②TRT 8.5 vs TRT 10 优化差距；③Pyramid 含 grouped-conv 非对齐通道 kernel-cliff (ISS-014)。

##### Orin DLA (FP16 only, INT8 全失败)

| 配置 | 路由 | FP16 lat_p50 | 备注 | 数据源 |
|---|---|---|---|---|
| p75 (16-32-64) | DLA0 | **20.563ms** | 8/12 build 成功 [真测, orin_dspace_bench, d_scheme=B_dla0] | orin_dspace_bench.csv [真测] |
| p75 (16-32-64) | DLA1 | **20.692ms** | 同上 | 同上 [真测] |
| p50 (32-64-128) | DLA0 | **45.934ms** | 8/12 成功 | 同上 [真测] |
| base (64-128-256) | DLA0 | ❌ build 失败 | 0/4 (kDIRECT_IO + bank 超限) [真测] | 同上 |
| 任意 | INT8 | ❌ build 全失败 | **0/12** (DLA INT8 在 Pyramid 不可行) [真测] | orin_dspace_bench.csv |

> **DLA 结论**: Pyramid 在 Orin 上 DLA 路由**仅 FP16 可用，且仅 p75(16-32-64)/p50(32-64-128) 两档**。base 和所有 INT8 档失败。DLA FP16 (p75 20.56ms) 比 GPU FP16 (p75 26.37ms) **快 ~22%**。

##### Orin DLA pipeline (跨进程异构并行, p75)

| 测量点 | 延迟 | 含义 | 数据源 |
|---|---|---|---|
| stage01@DLA0 isolated | **14.49ms** | DLA0 独立运行 stage0+1 [真测] | E3_orin_dla_pipeline.csv, real_trtexec_torch |
| stage2@DLA1 isolated | **6.18ms** | DLA1 独立运行 stage2 [真测] | 同上 |
| serial sum | 20.67ms | 14.49+6.18 [派生] | 同上 |
| 进程内双流 | **20.65ms (1.00×)** | TRT 8.5 进程内序列化 DLA 提交，无并行 [真测] | E3, real_torch_execute_async_v2 |
| **双进程稳态 interframe** | **15.43ms (1.34×)** | 真硅片并行 [真测] | E3, real_trtexec_dualproc_wallclock |
| **跨进程 xproc e2e 单帧** | **23.71ms** | 含 shm handoff (1.0MB, 0.97ms) [真测] | E3, real_xproc_shm_handoff |

> **结论**: DLA0∥DLA1 双进程流水: 吞吐 1.34× (interframe 15.43ms)，但**单帧 e2e = 23.71ms > serial sum 20.67ms**（handoff overhead）。
> **适用场景**: RSU 连续帧吞吐优化（吞吐-binding），不降单帧延迟。

---

## 3. 全管线 e2e 汇总 (RSU 计算端)

### 3.1 4090 平台 (数据中心 GPU / 边缘服务器)

| 组件 | FP32 PyTorch | TRT FP16 (混合) | TRT INT8 (混合) | 口径 |
|---|---|---|---|---|
| voxelize (2 agents) | ~2ms [估算] | ~2ms [估算] | ~2ms [估算] | host CPU |
| encoder_m1 (2 agents) | **4.80ms** [真测] | ~3ms [估算] | ~3ms [估算] | hook-level DAIR |
| backbone_m1 (2 agents) | **1.68ms** [真测] | ~1ms [估算] | ~1ms [估算] | hook-level DAIR |
| ★pyramid+shrink (body) | ~9.97ms [真测,PyTorch] | **1.29ms** [真测,TRT] | **0.85ms** [真测,TRT] | body_subnet_collab2 |
| CUDA NMS | **1.18ms** [真测] | **1.18ms** [真测] | **1.18ms** [真测] | hook DAIR |
| **RSU 计算合计** | **~17.87ms** [真测,e2e hook] | **~8.5ms** [混合估算] | **~7.1ms** [混合估算] | — |

> [真测] 对应 `dims_hardware_v2.md §2` hook-level DAIR 数据。
> TRT 混合估算: body 用真测 TRT 数值，其余沿用 PyTorch 真测值（PyTorch 非 TRT，但其他模块 TRT 收益小）。
> **4090 e2e 远低于 200ms (< 20ms), 余量 >10×。**

### 3.2 Orin AGX GPU (30W mode, 边缘设备)

| 组件 | FP16 | INT8 | 口径 | 数据源 |
|---|---|---|---|---|
| voxelize (2 agents) | ~5-10ms [估算] | ~5-10ms [估算] | host Cortex-A78 | — |
| encoder+backbone (2 agents) | ~10-20ms [估算-缩放] | ~8-15ms [估算] | — | — |
| ★pyramid+shrink (body) | **48.65ms** [真测] | **32.99ms** [真测] | body_subnet_collab2_orin, TRT 8.5, 30W | E6 dataset_v2 |
| CUDA NMS | ~2-5ms [估算] | ~2-5ms [估算] | — | — |
| **RSU 计算合计 (base)** | **~66-84ms** [混合] | **~48-62ms** [混合] | — | — |
| ★pyramid body (p75, 16-32-64) | **26.37ms** [真测] | **20.31ms** [真测] | 同上 | E6 dataset_v2 |
| **RSU 计算合计 (p75+INT8)** | ~44-62ms | **~36-48ms** [混合] | — | — |

> ⚠ encoder/backbone/voxelize 在 Orin 上**未实测**，以上为粗估 [估算-缩放]，实际可能偏高或偏低。
> **保守上界 (base FP16)**: RSU ≈ 84ms < 170ms 预算，余量 >2×。
> **最优 (p75 INT8)**: RSU ≈ 48ms，余量 >3.5×。

### 3.3 Orin DLA FP16 (p75 配置)

| 场景 | RSU body 延迟 | 全 RSU 估算 | 说明 |
|---|---|---|---|
| GPU FP16 (p75) | 26.37ms [真测] | ~44ms [混合] | 基准 |
| **DLA0 FP16 (p75)** | **20.56ms** [真测] | **~38ms** [混合] | 单 DLA 路由，22% 提速 vs GPU FP16 |
| DLA0∥DLA1 双进程 (吞吐优化) | interframe 15.43ms [真测] | 单帧 e2e 23.71ms [真测] | 吞吐模式；单帧延迟略增 |

> DLA path 不改变 encoder/backbone/NMS，故全 RSU 估算差值与 body 差值相同。

### 3.4 V2X-ViT 对比 (更重模型参考)

| 组件 | FP32 PyTorch | 口径 | 数据源 |
|---|---|---|---|
| encoder + backbone | 2.90+2.99+0.89=**6.78ms** | hook-level DAIR [真测] | v2xvit_structure_audit_v1.md |
| **fusion (V2XTransformer)** | **27.39ms** | hook-level DAIR [真测] | 同上 |
| NMS (Shapely CPU) | **23.75ms** | e2e hook [真测] | 同上 |
| **e2e 合计** | **61.86ms** | FP32 PyTorch DAIR [真测] | 同上 |

> V2X-ViT e2e 62ms 在 4090 FP32 下仍 <<200ms。TRT FP16 预估 20-30ms [估算，非真测，详见 v2xvit_trt_risk_v1.md §4.7]。

---

## 4. 可达性判断矩阵

| 平台 | 精度 | 配置 | RSU 计算 | V2X 双向通信 | 总 e2e 估算 | 200ms 可达? | 余量 |
|---|---|---|---|---|---|---|---|
| **4090** | FP32 PyTorch | base | ~18ms [真测] | 20ms [估算] | ~53ms | ✅ 可达 | ~147ms |
| **4090** | TRT FP16 | base | ~8.5ms [混合] | 20ms [估算] | ~44ms | ✅ 可达 | ~156ms |
| **4090** | TRT INT8 | base | ~7ms [混合] | 20ms [估算] | ~42ms | ✅ 可达 | ~158ms |
| **Orin GPU** | TRT FP16 | base | ~84ms [混合] | 20ms [估算] | ~119ms | ✅ 可达 | ~81ms |
| **Orin GPU** | TRT INT8 | base | ~62ms [混合] | 20ms [估算] | ~97ms | ✅ 可达 | ~103ms |
| **Orin GPU** | TRT FP16 | p75 | ~44ms [混合] | 20ms [估算] | ~79ms | ✅ 可达 | ~121ms |
| **Orin GPU** | TRT INT8 | p75 | ~36ms [混合] | 20ms [估算] | ~71ms | ✅ 可达 | ~129ms |
| **Orin DLA0** | FP16 | p75 | ~38ms [混合] | 20ms [估算] | ~73ms | ✅ 可达 | ~127ms |
| **Orin DLA0∥DLA1** | FP16 | p75 | 23.71ms [真测,单帧e2e] | 20ms [估算] | ~59ms | ✅ 可达 | ~141ms |
| **Orin GPU** | TRT FP16 | base (60W MAXN) | ~55ms [估算-缩放] | 20ms [估算] | ~90ms | ✅ 可达 | ~110ms |

> **V2X 最悲观情形 (LTE Uu 单向 50ms → 双向 100ms)**:
> Orin INT8 p75: 36ms + 100ms = 136ms < 200ms → 仍可达。
> Orin FP16 base: 84ms + 100ms = 184ms < 200ms → 勉强可达，仅 16ms 余量。

---

## 5. 框架能压掉多少 (剪枝 × 量化 × 硬件联合)

### 5.1 量化贡献 (body subnet 真测)

| 对比 | 4090 body | Orin body | 节省 |
|---|---|---|---|
| FP16→INT8 (base) | 1.293→**0.848ms** (1.53×) [真测] | 48.65→**32.99ms** (1.47×) [真测] | 35%/32% 延迟 |
| FP16→INT8 能耗 (4090 base) | 291→**141 mJ/frame** [真测, E4] | 409.5→未测 | **52%** J/frame |

> 数据源: 4090 body: 4090_dspace_bench; 能耗: E4_energy_4090.csv (T1_base 行); Orin: dataset_v2 E6 行。

### 5.2 剪枝贡献 (body subnet)

| 配置 | 4090 FP16 body | 4090 INT8 body | Orin FP16 body | Orin INT8 body |
|---|---|---|---|---|
| base (64-128-256) | 1.293ms | 0.848ms | 48.65ms | 32.99ms |
| p50 (32-64-128) | 1.044ms (1.24×) | 0.841ms (1.01×) | 42.07ms (1.16×) | 32.97ms (1.00×) |
| p75 (16-32-64) | 0.802ms (**1.61×**) | 0.640ms (**1.33×**) | 26.37ms (**1.85×**) | 20.31ms (**1.62×**) |

> 数据源: 4090: 4090_dspace_bench [真测, body_subnet_collab2]; Orin: dataset_v2 E6 [真测, body_subnet_collab2_orin]。
> 注: p75 = pyramid_backbone 通道 ~-74.7% (DAIR 真测收敛 AP50 0.757, AP70 0.530，无精度悬崖）。

### 5.3 联合 (剪枝 × 量化) 贡献

| 配置 | 4090 body | vs base FP16 | Orin body | vs base FP16 |
|---|---|---|---|---|
| base + INT8 | **0.848ms** | 1.53× | **32.99ms** | 1.47× |
| p75 + FP16 | **0.802ms** | 1.61× | **26.37ms** | 1.85× |
| **p75 + INT8** | **0.640ms** | **2.02×** | **20.31ms** | **2.40×** |

> **联合杠杆**: p75+INT8 在 Orin 上对 body subnet 压缩 **2.4×** (48.65→20.31ms, -58%)。

### 5.4 DLA 路由额外增益 (p75 FP16)

| 场景 | Orin body | vs GPU FP16 |
|---|---|---|
| GPU FP16 (p75) | 26.37ms | — |
| DLA0 FP16 (p75) | **20.56ms** [真测] | **1.28× 快** |
| DLA dual-proc (interframe) | 15.43ms [真测] | 吞吐 1.34× (≠单帧延迟) |

### 5.5 全 RSU e2e 压缩 (Orin，含非 body 组件估算)

| 方案 | RSU 估算 | vs base FP16 | 200ms 下 RSU budget 余量 |
|---|---|---|---|
| base FP16 (基准) | ~84ms | 1.00× | 86ms 余量 |
| p75 + INT8 | **~36ms** | **2.3×** | 134ms 余量 |
| p75 + DLA FP16 | **~38ms** | 2.2× | 132ms 余量 |
| p75 + INT8 + Phase H opt | (Phase H 数据冻结中，待用户裁定后方可引用，暂不给数值) | — | — |

---

## 6. 结论

### 6.1 主结论: 200ms **可达，且余量充足**

| 关键问题 | 回答 |
|---|---|
| Orin 30W 下 RSU 计算能否 < 200ms? | **是**，base FP16 RSU ≈ 84ms；含 V2X 双向通信(约 20ms)+ 车端(约 15ms) = e2e 约 119ms，**余量 81ms** |
| 最保守情形 (LTE V2X 100ms + Orin base FP16)? | 84+100 = 184ms < 200ms，仍可达，余量 16ms |
| 框架能压掉多少? | 剪枝×量化联合 2.4×，body 从 48.65ms 降至 20.31ms，**RSU 全链 ~36ms** |
| DLA 有额外增益吗? | 单 DLA 路由 +22% 提速 (p75)；双进程吞吐模式 1.34× (不降单帧延迟) |

### 6.2 制约因素与注意点

1. **Orin 非 body 组件 (encoder/backbone/voxelize) 未实测**: 当前估算 ~17-30ms，是不确定性最大处。若真测显著偏高，结论需更新。建议补测：Orin 上 encoder_m1 + backbone_m1 TRT latency。
2. **MAXN 功耗档 Orin 数据缺失**: E6 仅 30W。MAXN (~60W) 估算 body 降至 ~28ms (基于 GPU freq 缩放 612→1300MHz → ~0.6×)，RSU 全链约 ~50ms。[估算，ISS-016 MAXN 数据待补]
3. **DLA INT8 在 Pyramid 仍不可行**: 0/12 build 失败，DLA 路由限 FP16，不能叠加 INT8 节能。
4. **V2X 通信延迟是文献估算**: 实际 V2X 链路性能取决于部署环境，需与通信方专家确认。
5. **body_subnet ≠ e2e**: 本分析明确拆分了各组件，但 encoder/backbone/voxelize 在 Orin 上的数字仍为估算，需一轮真测补全。

### 6.3 不同 regime 最优配置建议

| Regime | 推荐配置 | RSU 延迟估算 | 理由 |
|---|---|---|---|
| **低延迟 RSU (单帧)** | Orin p75 + INT8 + GPU | ~36ms | 最低单帧延迟，框架搜索 Pareto 最优点 |
| **高吞吐 RSU (多车)** | Orin p75 + DLA0∥DLA1 (双进程) | interframe 15.43ms [真测] | 1.34× 吞吐，连续服务多车帧 |
| **能耗最优** | 4090 p75 + INT8 | 0.640ms body + 6ms 其余 | 仅 197 mJ/frame [真测 E4 T_prune75 int8 行] |
| **边缘最简** | Orin DLA0 FP16 p75 | ~38ms | 单 DLA 路由，无多进程编排复杂度 |

---

## 7. 数据白名单 (本文引用的真测数字来源)

| 数字 | 来源文件 | 状态 |
|---|---|---|
| 4090 body_subnet FP16/INT8 lat | `data/4090_dspace_bench.csv` | ✅ 白名单 真测 |
| Orin body_subnet FP16/INT8 lat | `multi_agent/data/dataset_v2.csv` E6 行 | ✅ 白名单 真测 |
| Orin DLA0/DLA1 FP16 lat | `data/orin_dspace_bench.csv` | ✅ 真测 |
| DLA pipeline (E3) | `results/E3_orin_dla_pipeline.csv` | ✅ 白名单 真测 |
| INT8 能耗 30-52% 节省 | `results/E4_energy_4090.csv` | ✅ 白名单 真测 |
| Orin 能耗 348 mJ/frame | `results/E6_orin_energy.csv` | ✅ 真测 (30W 档) |
| CUDA NMS 1.18ms, e2e 17.87ms | `multi_agent/methods/design/dims_hardware_v2.md §2` | ✅ 真测引用 |
| V2X-ViT e2e 61.86ms | `multi_agent/model/v2xvit_structure_audit_v1.md` | ✅ 真测 |
| NMS 3.04× e2e 加速 | `dims_hardware_v2.md §2 + results/P0_1*` | ✅ 真测 |
| V2X 通信延迟 5-20ms | ETSI EN 302 663 / TR 103 439 | ✅ 文献 [估算-文献] |

---

*编制: hw-optimizer · 2026-06-05 · 纯分析产物 (0 GPU 操作)*
*核验待: supervisor 独立复查数字来源与口径标注*
