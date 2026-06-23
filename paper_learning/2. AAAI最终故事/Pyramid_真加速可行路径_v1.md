# Pyramid V2X 真实 e2e 加速可行路径 — 工程级评估

> **创建**: 2026-05-13
> **背景**: 用户在 `Pyramid_架构与剪枝边界_audit.md` 看到 ONNX subnet 仅占 e2e 9%, 剪 backbone 50% 实测 e2e 仅 6.5% 加速后, 提问 "**车路协同算法剪枝加速到底可不可能**". 本文给出工程可行答案 + 落地路径分级.
> **核心结论 (TL;DR)**: 可能, 但不是当前路径. 当前剪 `pyramid_backbone` (e2e 9%) 受 Amdahl 限制上限 1.10×. 已发表 V2X / 车端 e2e 真加速案例 (UPAQ 1.97× / QuantV2X 99.8% AP / SPADE 4-28× 能效) 全部攻 e2e 真瓶颈 (encoder + postproc), 不是单纯剪 backbone. 我们要拿 e2e 2-3× 必须扩工程范围.

---

## 一、问题复述: 为什么当前路径加速微乎其微

### 1.1 Amdahl 定律的死锁

来自 `Pyramid_架构与剪枝边界_audit.md` §3 实测分解:

```
Stage A baseline FP32 e2e = 35.31 ms

  encoder_m1 (PointPillar VFE + Scatter)  ~10 ms  (28%)
  pyramid_backbone + collab + heads        3.3 ms ( 9%)  ← 我们剪 + 量化
  postproc (3D rotated NMS + decode)      ~22 ms  (62%)
```

Amdahl 推导:
- 剪 + 量化把 subnet 砍到 0 → e2e 35→32 ms = **1.10×**
- 剪 + 量化拿到 4× subnet 加速 → e2e 35→33 ms = **1.07×**
- **= 上限 1.10×, 跟纯 FP16 vs FP32 等价**

### 1.2 实证: 实测 e2e 加速 vs 剪枝率

来自 320 anchor (Class A 4090) + Stage A baseline:

| 配置 | subnet lat | e2e lat | e2e vs baseline |
|------|-----------|---------|----------------|
| T1 FP32 baseline | 3.347 ms | 35.31 ms | 1.00× |
| T1 FP16 | 1.275 ms | 26.70 ms | 1.32× |
| T6 (16,32,64, prune 94%) FP16 | 0.778 ms | ~32 ms (估) | ~1.10× |
| T6 INT8 | 0.623 ms | ~32 ms (估) | ~1.10× |

→ **剪 94% 参数 + INT8 量化, e2e 也只 1.1×**.

---

## 二、调研里 V2X / 车端真实 e2e 加速案例

### 2.1 直接对位 (V2X / 自动驾驶, agent_5_driving.md)

| 论文 | 攻什么 | 实测 e2e 加速 | 与我们 Pyramid 对位 |
|------|--------|--------------|---------------------|
| **UPAQ** (2025) | 半结构化 prune + 混合精度, **全 3D 检测流水线** | **Jetson Orin Nano e2e 1.97× + 5.62× 压缩** | ✅ 最直接对位 |
| **QuantV2X** (2025) | **全栈量化** (encoder + backbone + 通信码本) | INT4-W/INT8-A, **99.8% AP 保留** | ✅ V2X 专属, 量化范围扩到 encoder + 通信 |
| **SPADE** (HPCA'24) | **PointPillars encoder ASIC 加速器** (我们的 30% 瓶颈) | 4.1-28.8× 能效 | △ ASIC, GPU 不可直接复用 |
| **LiFT** (DASIP'25) | 替换 3D voxel → 2D cell, 绕过 sparse op | AMD Kria K26 实时 | △ 架构替换, 失精度 |
| **Q-PETR / QD-BEV** | 量化整个 BEV 模型 | mAP↓<1%, 8× 压缩 W4A6 | ✅ 全栈量化 (我们对应 pyramid + heads) |
| **DeployFusion** (Sensors'24) | EdgeNeXt 轻骨干 + TRT 端到端 | Jetson Orin NX 138ms/帧 | △ 整模型换骨干 |
| **PTQAT** (ICCV'25-W) | PTQ + QAT 混合, 冻 50% 层 | 4-bit 多架构 | ✅ 量化效率优化 |

**核心发现**: 所有报真 e2e 加速的案例都**扩大了优化范围**, 没有一篇是 "仅剪 backbone".

### 2.2 通用 GPU/DLA 剪枝硬件交互 (pruning_x_hardware.md 关键点)

| 维度 | 实测加速量级 | 对 Pyramid 适用度 |
|------|-------------|-------------------|
| **通道剪枝 (我们做的)** | 线性正比通道压缩 | 已做, 但 subnet 仅 9% e2e |
| **2:4 N:M 稀疏 (Sparse Tensor Core)** | 单 GEMM 1.4-1.8×, vLLM E2E Llama 1.5-1.7× | △ Pyramid backbone 太小 (K<256) 收益不确定 |
| **DLA 2:4 稀疏** | Orin DLA RetinaNet 1.36× 实测 | △ Pyramid 在 DLA 上 GPU fallback 频繁 |
| **CUDA Graph + 静态形状** | 消 launch overhead, 小 kernel 多场景显著 | ✅ 我们 backbone kernel 数多 (222 ops), 适用 |
| **multi-stream + MPS** | ego/infra 并行天然 2× | ✅ V2X 协同天然异构, 但我们没跑 |
| **动态剪枝 (token / 空间稀疏)** | DynamicViT 40% 吞吐, SkipNet/CoDeNet 无收益 | ❌ 破坏 CUDA Graph, 不推荐 |

---

## 三、Pyramid V2X 真实可行加速路径 (5 条)

按 e2e 加速量级 × 工程量 排序:

### 🌟 Path A — UPAQ 风全栈剪枝量化 (推荐, ROI 高)

**做什么**:
- ① encoder_m1.PillarVFE → 量化 (PFNLayer 1×1 conv 是标准 conv, 直接 INT8)
- ② backbone_m1 (single stage, 64 filter) → channel prune + INT8
- ③ pyramid_backbone → 维持已剪 (已做)
- ④ shrink_conv + heads → INT8 (已做)
- ⑤ 全部走一个 TRT engine (encoder + backbone + pyramid + heads), 消除 PyTorch fallback

**ONNX 扩展**:
- 输入从 `spatial_features (2, 64, 128, 256)` 改为 `pillar_features (M, 64)` + `voxel_coords (M, 4)`
- PillarScatter 难 export (动态 M + scatter assign), **两选一**:
  - (a) 把 PillarScatter 拆为 TRT plugin (~3 人日)
  - (b) Padding 到固定 M_max=3000 + mask 走 dense (~1 人日, 但浪费计算)

**预期 e2e 加速**:
- encoder 10ms → INT8 后 ~5-6ms
- subnet 3.3ms → 已 INT8 ~0.8ms
- postproc 22ms → 不变
- **e2e: 35.3 → ~28 ms = 1.26×**

**工程量**: 5-7 人日
**风险**: 中 — encoder INT8 校准需重新做; PillarScatter ONNX export 是难点

### 🌟 Path B — NMS CUDA kernel 集成 (单点最高 ROI, 强烈推荐)

**做什么**: 集成开源 MMCV `nms_rotated_cuda` 替换 HEAL `box_utils.compute_iou` + Python NMS 循环.

**实施**:
- pip install mmcv-full (已有 CUDA 算子)
- 改 `HEAL/opencood/utils/postprocess_utils.py` 调用 `mmcv.ops.nms_rotated`
- 不动其他代码

**预期 e2e 加速**:
- postproc 22ms → ~3ms (MMCV CUDA NMS 比 Python 快 10×)
- **e2e: 35.3 → ~17 ms = 2.07×**

**工程量**: 1-2 人日 (主要是 box 格式对齐 + AP 一致性验证)
**风险**: 极低 — MMCV 已在 1000+ 仓库验证, 数值精度 ε<1e-6

### 🌟 Path C — 2:4 Sparse Tensor Core (与 channel prune 叠加)

**做什么**:
- 对 pyramid_backbone 启 `trtexec --sparsity=force`
- 训练时 mask 强制每 4 个权重内 2 个零
- 跟通道剪枝叠加 (先剪通道, 再 2:4)

**实施**:
- HEAL train.py 加 2:4 mask 训练步骤 (5 epoch finetune)
- TRT build 加 `--sparsity=force`
- 验证 2:4 kernel 实际被调用 (TRT verbose log)

**预期 e2e 加速**:
- subnet 内 1.4-1.8× 额外 (但 subnet 仅 9% e2e)
- **e2e: 35.3 → ~34 ms = 1.03× — 跟纯 channel prune 差别不大**

**工程量**: 2-3 人日
**风险**: 中 — 我们 channel 数 (16/32/64/128) 满足 K%4=0 ✓, 但层规模小 (K<256), 实际加速可能 <1.4×; AP 可能再降 1-2pp

### 🌟 Path D — CUDA-PointPillars encoder 替换

**做什么**: 用 NVIDIA-AI-IOT/CUDA-PointPillars 库的 voxelize + scatter CUDA kernel 替换 HEAL PointPillar VFE + PillarScatter.

**实施**:
- clone NVIDIA-AI-IOT/CUDA-PointPillars
- 适配 HEAL 数据 dict 接口
- 编译 + 替换 HEAL encoder forward 路径
- 验证 spatial_features 输出与原 PyTorch 版本数值等价 (ε<1e-3)

**预期 e2e 加速**:
- encoder 10ms → ~2ms (CUDA 实测 3-5×)
- **e2e: 35.3 → ~27 ms = 1.31×**

**工程量**: 5-10 人日
**风险**: 中-高 — 需对齐 voxel parameter / range / point cloud schema; CUDA 库可能不支持我们 lidar_range 配置 (DAIR-V2X cav_range)

### 🌟 Path E — Anchor-free heads (CenterPoint 风)

**做什么**: 替换 pyramid 后的 anchor-based cls/reg/dir heads 为 CenterPoint 风 heatmap heads, 用 max-pool peak finding 替代 NMS.

**实施**:
- 改模型: 3 个 anchor head → 1 个 center heatmap + 1 个 regression
- 重训 30 epoch
- postproc 用 max-pool 找 peak, 不做 NMS

**预期 e2e 加速**:
- postproc 22ms → ~2ms (跟 Path B 接近, 但走 architecture)
- **e2e: 35.3 → ~16 ms = 2.21×**

**工程量**: 5-7 人日 + 训练 30 epoch wall (~8h DDP)
**风险**: 中 — 改模型架构, AP 可能差 baseline 1-3pp (CenterPoint 论文报告)

---

## 四、组合策略 ROI 评估

| 组合 | 工程量 | 预期 e2e 加速 | 论文叙事强度 |
|------|--------|--------------|--------------|
| **保守 (当前路径完成 720 anchor)** | 0 (剩余工作) | 1.10× | "框架方法论 + Pareto 搜索" — 故事弱 |
| **+ Path B (NMS CUDA)** | +1-2 人日 | **2.07×** | "V2X 真实 e2e 加速 2×" — 强 |
| **+ Path A (UPAQ 全栈)** | +5-7 人日 | 1.26× 单独 / **B+A = 2.6×** | "INT8 全栈 + CUDA NMS" — 论文级 |
| **B + A + C** | +10 人日 | **~2.8×** | "剪枝 + 量化 + 2:4 + CUDA NMS 联合" — 顶会级 |
| **B + A + D (全栈攻完)** | +20 人日 | **~3.5×** | "encoder + backbone + postproc 全攻" — SOTA 级 |
| **B + E (anchor-free + CUDA NMS 一致性)** | +10 人日 | 2.2× | "架构 + 推理双重优化" — 中等 |

**推荐 ROI 最高 = "B + A"** (~7-9 人日, e2e 2.6×, 论文叙事完整). 仅 Path B 单独也是单点最高 ROI (~2 人日, 2×).

---

## 五、对当前 720 anchor 计划的影响

### 5.1 维持当前计划的代价

如果不扩工程, 720 anchor 完整数据集**能交付**但论文只能报:
- "framework Pareto search 演示" (Q × D × HW 联合)
- 加速绝对值 1.05-1.10× (剪枝 + 量化 e2e 上限)
- 强调 "搜索空间方法论", 不强调 "e2e 加速数字"

### 5.2 扩 Path B 的成本/收益

- 数据集层面: 跑完 720 anchor 后**单独** run 一次 Path B 集成 + 重测 8 个 Stage A 锚 (FP32/FP16/INT8 × baseline/p50) = ~16 个 anchor 重测
- 论文影响: paper §C 主表加列 "with CUDA NMS" → 数字从 1.32× (FP16) 跳到 2.6× (FP16 + CUDA NMS)
- 工程: 2 人日

**ROI 极高, 强烈建议至少做 Path B**.

### 5.3 扩 Path A 的成本/收益

- 数据集层面: 全栈 INT8 是 Q 维度新增 cell, **需扩 320 anchor 到 480** (新 INT8 全栈 Q × 4 D × 8 T = 160 anchor)
- 论文影响: §C 主表 INT8 列从 "subnet only" 改为 "full pipeline", 数字更可信
- 工程: 7 人日

ROI 中等, 看 paper 截稿时间是否容纳.

### 5.4 Path C/D/E 在当前 paper 周期不建议

- Path C: subnet 改善对 e2e 仅 3%, 投入产出比差
- Path D/E: 需改模型 + 重训, 时间风险高, 留给下一篇 paper

---

## 六、决策矩阵

| 假设 | 推荐路径 |
|------|---------|
| **paper 截稿 < 2 周** | 完成 720 anchor + 只加 Path B (CUDA NMS) |
| **paper 截稿 2-4 周** | 720 anchor + B + A (NMS + 全栈 INT8 + encoder 量化) |
| **paper 截稿 > 1 个月** | B + A + C + D 全套, e2e 3-3.5×, 论文叙事 SOTA |
| **不在乎 paper, 想给真实部署用** | 必须 B + D, 拿到 e2e 2-3× 实测 |
| **完全放弃当前路径, 重写 §C** | Path E (anchor-free) + B, 算 BEV detection 系列优化 |

---

## 七、立即可执行的下一步选项

### 选项 1: 保守完成
- 继续等 P0.3 跑完 (~3-4h)
- 启动 P0.3.b INT8 patched cache 补 288 anchor (~4-6h)
- 启 P1 Class B 4090 + Class C Orin
- 启 P2 Class D 跨模型 (~10h)
- 启 P3 LGB v5 重训
- **耗时: ~30h wall**
- **paper 数字: e2e 1.1×**

### 选项 2: 保守 + Path B (推荐)
- 选项 1 的所有
- 加 2 人日做 Path B (集成 MMCV CUDA NMS, 重测 Stage A 8 anchor)
- **耗时: +2 人日**
- **paper 数字: e2e 2.0-2.6×**

### 选项 3: 激进重定位
- 暂停 P0.3.b 之后所有数据扩展
- 立即启动 Path A + B
- 重写 §C 为 "全栈优化", 重新选 paper anchor (~100 anchor 真测 + Pareto 演示)
- **耗时: ~10 人日**
- **paper 数字: e2e 2.6-2.9×, 全栈 INT8 + CUDA NMS**

---

## 八、引用 + 修订历史

**代码 / 工具**:
- NVIDIA-AI-IOT/CUDA-PointPillars (Path D)
- MMCV `mmcv.ops.nms_rotated_cuda` (Path B)
- TensorRT `--sparsity=force` 文档 (Path C)
- HEAL `opencood/utils/postprocess_utils.py` (Path B 替换点)
- HEAL `opencood/models/sub_modules/{pillar_vfe,point_pillar_scatter}.py` (Path A 量化目标)

**调研依据**:
- `paper_learning/survey_raw/agent_5_driving.md` — 10 篇 V2X / 自动驾驶部署加速论文
- `paper_learning/survey_raw_2/pruning_x_hardware.md` — 剪枝 × 硬件 交互调研 (506 行)
- `paper_learning/2. AAAI最终故事/Pyramid_架构与剪枝边界_audit.md` — Pyramid 7 子模块解剖

**实测数据**:
- `data/_by_class/class_a_pyramid_full.parquet` — 320 anchor subnet lat
- `data/stage_a_ap_real.parquet` — 8 anchor e2e baseline (35.31ms FP32, 26.70ms FP16)
- `data/unified_bench.parquet` — 660 行整合数据

**修订历史**:

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-05-13 | 初版, 综合 agent_5_driving + pruning_x_hardware 调研 + 320 anchor 实测 + Pyramid 架构 audit, 给出 5 条可行路径 + ROI 评估 + 3 个决策选项 |
