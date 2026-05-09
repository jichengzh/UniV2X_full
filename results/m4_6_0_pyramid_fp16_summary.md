# M4.6.0: Pyramid_m1_base 完整 e2e FP32 vs FP16 实测

**日期**: 2026-05-09
**硬件**: RTX 4090
**模型**: HEAL Pyramid_m1_base (LiDAR-only, 5.50M params total / 3.79M PyramidFusion)
**数据集**: OPV2V test split (2170 samples, 自下载 21GB from HF gqk/opv2v)
**timing**: model.forward 单独, 不含 dataset.post_process (NMS, ~80ms CPU overhead)
**FP16 实现**: torch.cuda.amp.autocast(dtype=fp16) — model 内 fp16, output 转 fp32 再 post-process

## 核心结果

| precision | AP30 | AP50 | AP70 | lat_mean | lat_p50 | lat_p99 |
|-----------|------|------|------|----------|---------|---------|
| FP32 | 0.9691 | 0.9635 | 0.9272 | 43.00 ms | 35.31 ms | 119.97 ms |
| FP16 | 0.9690 | 0.9631 | 0.9268 | 30.96 ms | 26.70 ms | 80.23 ms |
| Δ (FP16 vs FP32) | -0.01% | -0.04% | -0.05% | **-28.0%** | **-24.4%** | **-33.1%** |

## 解读

### 真实加速倍率: 1.32× (p50)

- 这是 framework 在 Pyramid 上的**第一个真实 measured 加速点** (此前都是 KNN 估算)
- 比 4090 Tensor Cores 理论 2× FP16 加速保守 — 因为:
  1. PointPillar voxelization (encoder_m1) IO-bound, 不受 FP16 影响
  2. shrink_conv + cls/reg/dir heads 都很小 (合 < 2M params), 加速 marginal
  3. 主要加速来自 pyramid_backbone (3.79M, ResNeXt blocks)

### AP 几乎无损

- AP30/50/70 三个阈值的损失都 < 0.05%, FP16 在 Pyramid 上**安全**
- 这跟 BEV-based detection 通常对 FP16 鲁棒一致 (vs Transformer 模型容易 FP16 数值不稳)

### 子模块 vs 完整 e2e 的差距 (印证之前分析)

之前 M4.3/M4.5_6 用 dummy (1, 64, 256, 256) 输入测 PyramidFusion 子模块:
- FP32: 4.49 ms
- FP16: 3.27 ms

本次完整 e2e (含 voxelization + backbone_m1 + shrink_conv + heads):
- FP32: 35.3 ms  (~8× 子模块)
- FP16: 26.7 ms  (~8× 子模块)

差距 ~30 ms 全部来自 voxelization + 上游/下游处理. 这意味着:
- 单独优化 PyramidFusion (e.g. M4.7 的 50 候选剪枝) 上限 = ~3-5 ms 节省
- 真要拿到论文级别加速, 还得动 voxelization (spconv 升级 / lidar input 抽稀)

## 加入 baseline_4090.parquet

新增 2 行 (替换之前 PyramidFusion 子模块 latency 占位):
- pyramid_baseline_full_fp32: lat_e2e_ms=43.00, amota=0.9635, params_after_M=5.50
- pyramid_baseline_full_fp16: lat_e2e_ms=30.96, amota=0.9631, params_after_M=5.50

## 下一步 (M4.6.1+)

1. **INT8 PTQ** (M4.6.1): 期望 ~1.5-2× 多于 FP16 = 13-18 ms p50, AP drop 待测
2. **Channel pruning + finetune** (M4.6.2): 选 M4.7 Pareto-optimal 候选 (e.g. 80% encoder pruning) 实测
3. **抽样验证** (M4.6.3): 5-10 个 M4.7 Pareto-near 候选 → framework Pareto 前沿可信度
