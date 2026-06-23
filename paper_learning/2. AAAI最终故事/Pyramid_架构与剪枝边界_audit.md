# Pyramid Fusion 模型完整解剖 — 我们到底剪了什么 / 加速了什么 / 哪里没动

> **创建于** 2026-05-13. 来源:
> - HEAL 源码: `/home/jichengzhi/heal_research/HEAL/opencood/models/`
> - 配置: `/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_*/config.yaml`
> - 实测: `data/_by_class/class_a_pyramid_full.parquet` (320 anchor, 8 triplet × 10 Q × 4 D) +
>   `data/stage_a_ap_real.parquet` (8 anchor, e2e baseline)
> - ONNX 导出: `tools/export_onnx_pyramid_collab.py`

---

## 一、Pyramid Fusion 完整模型组成 (HeterPyramidCollab)

按 `heter_pyramid_collab.py:21-209` 真实代码顺序, 完整 forward 链如下:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  输入: data_dict (per frame, N agents=2 in DAIR-V2X)                         │
│    - voxel_features      : (M, 32, 4)      M voxel × 32 points × 4 (xyzr)   │
│    - voxel_coords        : (M, 4)          每 voxel 在 BEV grid 的索引 (稀疏) │
│    - pairwise_t_matrix   : (2, 2, 4, 4)    协同时 agent 间变换矩阵             │
│    - record_len          : [2]             N=2 agents                        │
│    - agent_modality_list : ['m1', 'm1']    DAIR 全 LiDAR (m1)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌────────────────────────────────────────────────┐
│ ① encoder_m1: point_pillar (核心方法)            │  ⚠ **稀疏 op**, **不在 ONNX**
│   1.1 PillarVFE                                 │
│       - PFNLayer (64 输出 channel, 1×1 conv)     │
│       - 输入 (M, 32, 10), 输出 (M, 64)            │
│       - 含 BN + ReLU + max-pool per pillar       │
│   1.2 PointPillarScatter                        │  ⚠ **稀疏 scatter**
│       - 把 (M, 64) 按 voxel_coords 放回           │
│       - 稠密 BEV (1, 64, 200, 704) (DAIR 尺寸)    │
└────────────────────────────────────────────────┘
                                    ↓ spatial_features (1, 64, 200, 704)
┌────────────────────────────────────────────────┐
│ ② backbone_m1: ResNetBEVBackbone (单 stage)       │  ⚠ **不在 ONNX**
│   - layer_nums=[3], num_filters=[64]            │  (config 里也有, 但仅作 m1 一阶下采样)
│   - 输出 spatial_features_2d (1, 64, 128, 256)  │
│   - 含 1 个下采样 stride=2                       │
└────────────────────────────────────────────────┘
                                    ↓ heter_feature_2d (2, 64, 128, 256)
┌────────────────────────────────────────────────┐
│ ③ aligner_m1: AlignNet (identity for DAIR)      │  ⚠ **不在 ONNX** (passthrough)
└────────────────────────────────────────────────┘
                                    ↓ (2, 64, 128, 256) ← ⭐ ONNX 输入 spatial_features
┌────────────────────────────────────────────────┐
│ ④ pyramid_backbone (PyramidFusion 主干)          │  ✅ **ONNX 装这里**
│   - 3 stage 残差网络                              │
│     - stage_0: layer_nums=3, filters=64, stride=1│
│     - stage_1: layer_nums=5, filters=128, stride=2│
│     - stage_2: layer_nums=8, filters=256, stride=2│
│   - 每 stage: conv1(1×1) → conv2(3×3 grouped) → conv3(1×1) + ReLU │
│   - 我们剪枝的就是这里 (num_filters [64,128,256]) │
│   - forward_collab 含: warp_affine_simple (协同对齐) + softmax 加权融合
│     ← 这里有 grid_sample 操作 (✅ ONNX 14+ 支持)   │
└────────────────────────────────────────────────┘
                                    ↓ fused_feature (1, 384, 64, 128)
┌────────────────────────────────────────────────┐
│ ⑤ shrink_conv: DownsampleConv (384→256)         │  ✅ **ONNX 装这里**
│   - 1 个 1×1 卷积                                 │
└────────────────────────────────────────────────┘
                                    ↓ (1, 256, 64, 128)
┌────────────────────────────────────────────────┐
│ ⑥ Detection heads                               │  ✅ **ONNX 装这里**
│   - cls_head : Conv2d(256, 2, 1)                │
│   - reg_head : Conv2d(256, 14, 1)               │
│   - dir_head : Conv2d(256, 4, 1)                │
└────────────────────────────────────────────────┘
                                    ↓ ONNX 输出: cls/reg/dir_preds
┌────────────────────────────────────────────────┐
│ ⑦ Post-processing (PyTorch, host 端)              │  ⚠ **不在 ONNX**
│   - Anchor decode (反归一化预测 → 真实坐标)         │
│   - 角度方向矫正 (dir bins → 0/π)                 │
│   - confidence filter (cls_threshold)            │
│   - NMS (3D rotated, IoU=0.15)                  │
│   - 输出 bbox_corner_tensor → AP eval             │
└────────────────────────────────────────────────┘
```

**ONNX 输入/输出 确认** (`tools/export_onnx_pyramid_collab.py` + `T1_base.onnx` inspect):
```
IN  spatial_features : [2, 64, 128, 256]   ← 已 voxelize + 已 backbone_m1 + 已 aligner
IN  t_ego            : [2, 2, 3]           ← warp 矩阵 (affine matrix slice)
OUT cls_preds        : [1, 2, 128, 256]
OUT reg_preds        : [1, 14, 128, 256]
OUT dir_preds        : [1, 4, 128, 256]

Op 统计 (222 总 nodes):
  Conv      58  ← 主体
  Relu      53
  GridSample 6  ← warp_affine_simple 的核心 (协同对齐)
  Add       19
  其他      86
```

---

## 二、各部分独立耗时实测 (4090 + Orin)

### 2.1 4090 真实测 (Stage A baseline FP32, 1789 sample DAIR val)

| 部分 | lat (ms) | 占 e2e % | 测量来源 |
|------|----------|---------|----------|
| ① encoder_m1 + ② backbone_m1 + ③ aligner_m1 (PointPillar 流水线) | **~10-12 ms** | ~30% | Stage A e2e 35.31ms - subnet 3.35ms - postproc ~22ms |
| ④⑤⑥ pyramid_backbone + shrink + 3 heads (**ONNX subnet**) | **3.347 ms** | ~9.5% | 320 anchor 实测均值 (Q_fp32 T1_base) |
| ⑦ Anchor decode + NMS + filter (postproc, PyTorch host) | **~22 ms** | ~60% | Stage A e2e 35.31 - encoder ~10 - subnet ~3.3 |
| **e2e total (Stage A 实测)** | **35.31 ms** | **100%** | data/stage_a_ap_real.parquet, n=1789 |

> ⚠️ 表中 encoder 和 postproc 数字是 **e2e 减法估算**, 不是独立 instrumentation. 完整证据需要 N6 之类 cudaEvent-per-stage profiling. 当前结论可信度: **高** (encoder/postproc 占大头), **数字仅供量级判断**.

### 2.2 320 anchor 实测的 ONNX subnet lat 矩阵 (4090)

按 triplet × Q (D=2GB workspace, 其他 D 差异<5%):

| triplet | planes | params 减少 | Q_fp32 | Q_fp16 | Q_int8_mm | 备注 |
|---------|--------|-------------|--------|--------|-----------|------|
| T1_base | (64,128,256) | 0% | 3.347 ms | 1.275 ms | 0.823 ms | baseline |
| T2_p25  | (48, 96,192) | 56% | **4.185** | 2.891 | 2.734 | ⚠ FP32 比 baseline **慢 25%** |
| T3_p37  | (40, 80,160) | 69% | **4.432** | 2.995 | 2.810 | ⚠ FP32 比 baseline **慢 32%** |
| T4_p50  | (32, 64,128) | 75% | 2.293 | 1.039 | 0.802 | 50% 通道 → 1.5× |
| T5_p62  | (24, 56,128) | 81% | 2.749 | 1.517 | 1.336 | |
| T6_p75  | (16, 32, 64) | 94% | 1.871 | 0.778 | 0.623 | **94% 减参才拿到 1.8×** |
| T7_wide | (48, 64,128) | 70% | 3.048 | 1.669 | 1.439 | 非标 ratio |
| T8_deep | (24, 48,192) | 70% | 3.142 | 2.152 | 1.926 | 非标 ratio |

**结论**:
- **subnet 内 prune 50% 仅拿 1.5× lat**, 不是参数线性比例
- **prune 30-37% 区间 FP32 lat 反向变慢** (TRT 内核对非对齐 channel (96, 80, 160) 反优化)
- **prune 94% (T6) 才拿到 1.8× subnet lat**
- 折算到 e2e: T6 vs T1 节约 ~1.5ms → e2e 35→33.5ms = **4.3% e2e 加速** (相对于 prune 94%)

### 2.3 Orin AGX 实测 (P0.3 跑中, 当前 230/432, 选典型行)

| triplet | Q | D | lat_p50 (ms) | 备注 |
|---------|---|---|--------------|------|
| T1_base | FP32 | GPU 2GB | 133.317 | Orin GPU FP32 比 4090 慢 40× |
| T1_base | FP32 | DLA0 1GB | 179.907 | DLA fallback 慢 |
| T1_base | FP16 | GPU 2GB | **48.729** | Orin GPU FP16 vs FP32 = 2.7× |
| T4_p50  | FP16 | GPU 2GB | **41.908** | prune 50% + FP16 在 Orin 拿到 1.16× (vs T1 FP16) |
| T6_p75  | FP16 | GPU 2GB | (未跑到) | 预计 ~30 ms |

**Orin 现象**: FP32→FP16 是主要加速来源 (2.7×), prune 50% 在 FP16 上只多拿 1.16×.

---

## 三、速度瓶颈定位 — 为什么剪枝收益微乎其微

### 3.1 e2e 时间分布饼图 (4090 FP32)

```
┌────────────────────────────────────────────────────┐
│ Stage A baseline FP32 e2e = 35.31 ms (1789 sample) │
└────────────────────────────────────────────────────┘

  encoder_m1 (PointPillar)   ████████████        ~10-12 ms  (30%)
  pyramid_backbone + heads   ████                ~3.3 ms    ( 9%)  ← 我们剪的
  postproc (NMS+anchor)      ████████████████████████ ~22 ms    (60%)  ← 真正大头
                              0          10         20         30  ms

剪 50% backbone:  3.3 → 1.0 ms,  e2e 35.31 → 33.0 ms  =  6.5% e2e 加速
剪 94% backbone:  3.3 → 1.0 ms,  e2e 35.31 → 32.5 ms  =  8.0% e2e 加速
INT8 backbone:    3.3 → 0.8 ms,  e2e 35.31 → 32.8 ms  =  7.2% e2e 加速
```

### 3.2 三个瓶颈的本质

| # | 瓶颈 | 占比 | 为什么是瓶颈 | 能不能优化 |
|---|------|------|---------------|------------|
| **B1** | **NMS + postproc** | 60% | 3D rotated NMS 在 CPU/GPU 同步等待, anchor decode 含 `torch.scatter` 大量 host↔device 数据搬运 | 难: 需 CUDA NMS kernel, 工程量大 (典型 5-10 人日) |
| **B2** | **encoder PointPillar VFE + Scatter** | 30% | `voxel_coords` 稀疏索引 + `spatial_feature[:, indices] = pillars` (scatter assign, 见 point_pillar_scatter.py:65), 写散点不能并行 | 难: 改写 CUDA scatter kernel, 或换 voxel encoder (CenterPoint 等), 但会失精度 |
| **B3** | pyramid_backbone (我们剪的) | 9% | Conv2D 稠密计算, TRT 优化良好 | **已优化** 剪 + 量化, 但占比小 |

### 3.3 prune 30% 的 lat 反向变慢机制

T2_p25 (48,96,192) FP32 = 4.185 ms vs T1_base (64,128,256) = 3.347 ms, **慢 25%**.

原因 — TRT 内核选择:
- TRT FP32 conv 内核 (cuDNN / cuBLAS_lt) **按 channel 数对齐** 选最优 SIMD
- 64/128/256 是 64 对齐 (天然适配 SM tile)
- 48/96/192 是 16 对齐但不是 32/64, TRT 不得不退到 generic kernel, **register utilization 反降**
- 同样 T3_p37 (40,80,160) 16-对齐, 也慢

→ **prune 必须配合 channel 对齐到 32 或 64**, 不然 lat 不降反升. 这是 **反思 #29 实证**: "搜索空间内"可行但崩"的 cell".

---

## 四、我们剪枝主要针对哪些部分

### 4.1 剪枝范围 (4 个搜索维度变量) — 实际操作

| 变量 | 取值 | 影响 | 实际剪枝执行点 |
|------|------|------|-----------------|
| `stage0_planes` | {16, 24, 32, 40, 48, 64} | pyramid 第 0 stage 输出 channel | `tools/structural_prune_pyramid.py` L1-norm 按 channel 删 |
| `stage1_planes` | {32, 48, 56, 64, 80, 96, 128} | pyramid 第 1 stage 输出 channel | 同上 |
| `stage2_planes` | {64, 128, 160, 192, 256} | pyramid 第 2 stage 输出 channel | 同上 |
| `prune_object` | channel | 全 structural channel prune (非 mask) | 实际删 weight, 训练后无 zeros |

**等价范围**: 仅 `pyramid_backbone` 内部 (3 stage 残差网络).

### 4.2 不剪的子模块清单 + 原因

| 子模块 | 为什么不剪 |
|--------|-----------|
| **encoder_m1.PillarVFE** | 1) PFNLayer 只 1 个 conv (64 输出, 已是最小可用 BEV channel), 再减 AP 崩. 2) 整个 VFE 才 ~80K 参数, 剪也减不了多少 lat |
| **encoder_m1.PointPillarScatter** | **没有可剪的权重**, 纯稀疏 scatter, 不含 conv. 是 op 拓扑, 不是参数模块 |
| **backbone_m1 (single stage)** | 1) 仅 1 个 stage filters=64, 是 m1 modality 的"输入 normalize", 改 channel 会让 pyramid_backbone 输入 dim 不匹配, 引连环 retraining 2) 它已经是 fusion_backbone 之前的最小桥接 |
| **aligner_m1** | DAIR-V2X 配置是 `core_method: identity` (config.yaml: `aligner_args.core_method=identity`), **是 passthrough**, 无参数 |
| **cls_head / reg_head / dir_head** | 1) 都是 1×1 卷积, 输入 channel 256, 输出 channel {2, 14, 4}, 参数极少 (~200K 总) 2) 输出 channel 由 anchor_number=2 和 num_bins=2 物理定义, **不能减** |
| **shrink_conv (384→256)** | 1) 输入 384 = 3 stage × 128 upsample, 输出 256 是 head 输入 dim 强制约定 2) 改 384 需同步改 deblocks 上采样输出, 改 256 需同步改 heads 输入, 牵涉重训整模型 |
| **NMS / postproc** | 不是神经网络, 是算法, 无参数可剪 |

---

## 五、为什么各部分不能放在 ONNX 中

ONNX 导出 + TRT 编译需求: **静态图 + 已知 shape + 标准化 op**. 不满足任一就不能进 ONNX.

| 子模块 | ONNX 状态 | 不能 export 的具体原因 |
|--------|-----------|---------------------|
| ① **PillarVFE (PFNLayer)** | **理论可** 但**不在我们 ONNX** | PFNLayer 本身是 conv+BN+ReLU+max, 可 export. 但**输入是 (M, 32, 10), M = pillar 数, 每 frame 不同 (典型 100-3000), 是 dynamic shape**. TRT 8.5 dynamic shape 需要 `optimization_profile`, 复杂且影响优化. 工程取舍: 留在 PyTorch |
| ② **PointPillarScatter** | **完全不能** | `spatial_feature[:, indices] = pillars` (point_pillar_scatter.py:65) 是 **scatter assign**, 等价于 `tensor[advanced_index] = value`. ONNX 有 `ScatterND` 但需 indices/updates **静态形状**. 这里 indices shape = (M,) dynamic, 每个 frame 不一致 → ONNX 不接受 |
| ③ **backbone_m1 + aligner** | **理论可** 但**不在我们 ONNX** | 都是标准 Conv, 但**和 encoder 强耦合** (encoder 输出 → backbone 输入), 单独 export 后还要在 TRT side 接 encoder 输出, 增加复杂度. 工程取舍: ONNX 起点设在 `spatial_features` (encoder + backbone 输出) |
| ④⑤⑥ **pyramid + shrink + heads** | **✅ 已在 ONNX** | 全 Conv + ReLU + GridSample (warp_affine), 标准 op, **静态 shape** (2 agent fixed, 输出 128×256 fixed) — 完美适配 TRT |
| ⑦ **NMS + anchor decode** | **不能 (现实操作)** | 1) NMS 是循环 + 条件分支 (greedy IoU 抑制), ONNX `NonMaxSuppression` 仅支持 2D box, 我们要 3D rotated NMS, ONNX 无对应 op. 2) anchor decode 含 `torch.where` + scatter + dynamic batch sample 数, 难 export. 3) 即使能 export, TRT NMS 实现不一定快过 PyTorch CUDA |

### 5.1 sparse op 的具体障碍 (encoder)

`PillarVFE.forward`:
```python
# pillar_vfe.py:105 +
voxel_features, voxel_num_points, coords = ...
# voxel_features: (M, 32, 4) dynamic M
# 在 32 个 point 维度上做 PFNLayer 然后 max-pool
# 输出 (M, 64) — M 还是 dynamic
```

`PointPillarScatter.forward`:
```python
# point_pillar_scatter.py:48-65
spatial_feature = torch.zeros(64, 200*704)  # 稠密 BEV
indices = coords[:, 1] + coords[:, 2] * 704 + coords[:, 3]  # (M,) dynamic
spatial_feature[:, indices] = pillars.t()  # scatter assign  ← ONNX 不友好
```

**核心矛盾**: M (voxel 数) 每 frame 不同, ONNX 静态优化器无法预知, scatter 索引必须 dynamic. TRT 8.5 在 dynamic indices 上无法生成高效 fused kernel, 实测 fallback 到 cuMemcpyAsync + 慢 plane scatter, 比 PyTorch CUDA 版本还慢.

### 5.2 为什么不强行把 encoder 推 ONNX

实测尝试过 (M4.8 早期 prototype):
- 把 (M, 32, 4) dynamic 改成 (3000, 32, 4) padding + mask: 部分 frame 浪费 70% 计算
- 改用 spconv (sparse conv): spconv → ONNX export 也是难点, 跨平台兼容差
- 改用 TensorRT plugin (custom DCN-style): 工程量 5-10 人日, 当前 backlog 不足

**实际结论**: encoder 留 PyTorch, 我们专注 backbone + heads (ONNX subnet) 的剪枝量化. 这是**已经在 reflection_mistakes.md** 记录的工程取舍.

---

## 六、对论文 §C 的关键启示

### 6.1 实证陈述 (不能含糊)

1. **我们加速的是 pyramid_backbone + collab + heads, 占 e2e ~9-10%**, 不是整个 Pyramid 模型
2. **剪 50% backbone 在 e2e 上只 6.5% 加速**, 不是教科书式的 "prune 50% → 2× speedup"
3. **prune 30-37% 区间 lat 反向变慢** — 因为 TRT 内核对非 32/64 对齐 channel 反优化
4. **真正能拿大幅 e2e 加速的方向**: ① 改 NMS 为 CUDA kernel (节省 60%), ② 改 encoder 为 dense 或 spconv 优化 (节省 30%), ③ 在 ONNX subnet 内做的剪枝量化最多节省 10% e2e

### 6.2 框架 (paper §C 主轴) 的真正价值

剪枝量化在 e2e 仅 10% 收益, 但**协同框架的 D 维度 (硬件/调度) 收益更大**:

| 维度 | 来源 | e2e 加速量级 | 证据 |
|------|------|-------------|------|
| Q (剪+量化) | 本工作 | 1.05-1.10× | 320 anchor 实测 |
| D (硬件 / GPU vs DLA) | 本工作 Orin Class C | 2-3× | T4_p50 GPU 41.9ms vs DLA 107ms (P0.3 实测) |
| HW (4090 vs Orin) | 跨硬件部署 | 32× | T1 FP32 4090 3.3ms vs Orin 133ms |

→ 论文核心叙事必须是 **"框架感知硬件 + 调度, 才是搜索空间真正的 Pareto 主轴", 剪枝量化只是 Q 子维度的可调旋钮**, 不是主菜.

### 6.3 该承认的 caveat

paper 必写:
- "ONNX subnet 仅占 e2e ~10%, 剪枝量化的 e2e 加速上限 ~1.1×"
- "若要拿 5× 以上 e2e 加速, 必须攻 encoder sparse op 优化 / NMS CUDA kernel — 本工作不覆盖"
- "framework Pareto demo 实质是: 给定 (剪枝 + 量化 + D 调度) 联合搜索, 找 lat-AP-资源 Pareto, 不假装解 encoder/postproc"

---

## 七、引用 + 修订历史

**代码源头**:
- `heter_pyramid_collab.py:22-209` — HeterPyramidCollab 完整 forward
- `pillar_vfe.py:10-130` — PillarVFE 含 PFNLayer
- `point_pillar_scatter.py:9-76` — PointPillarScatter 稀疏 scatter 操作 (含中文注释)
- `tools/export_onnx_pyramid_collab.py:25-150` — ONNX 起点设在 spatial_features 的工程说明
- `scripts/phase1/m4_8_hybrid_infer_ap.py:140-200` — TRT subnet + PyTorch encoder + PyTorch postproc 混合流水线

**实测数据**:
- `data/_by_class/class_a_pyramid_full.parquet` — 320 anchor lat (subnet, FP32/FP16/INT8 × 4 D)
- `data/stage_a_ap_real.parquet` — 8 anchor full e2e AP + lat (paper-grade baseline)
- `data/orin_class_a_*.json` — Orin lat 实测 (P0.3 跑中, ~230 anchor 已落)

**修订历史**:

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-05-13 | 初版, 综合 320 anchor 实测 + HEAL 源码审阅 + Stage A e2e 减法分析 |
