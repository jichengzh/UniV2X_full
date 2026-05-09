# M4.6 修复 6 小时长线计划

**目标**: 把 framework 从 "predicted Pareto 但实测崩" 升级到 "v6 跨模型预测 + real channel reduction 实测 + 抽样验证可信".

## Timeline

| Phase | 内容 | 估时 | 关键产出 |
|-------|------|------|----------|
| 1 | 用 v6 LGB 重跑 M4.7 → v3 (跨模型 lat+amota 双预测) | 30min | `phase2_pareto_pyramid_v3.csv` |
| 2 | M4.6.2 v2 真 channel reduction (重 build smaller PyramidFusion + truncate ckpt + BN recal) | 2.5h | `phase1/m4_6_2_v2_real_channel_reduction.py` + 5 数据点 |
| 3 | 抽样 5 个 M4.7 v3 Pareto 候选用 M4.6.2 v2 实测 | 2h | `m4_6_3_v2_pareto_validation.csv` |
| 4 | Phase 2.5 v2 三模型联合 Pareto 重跑 + 综合报告 commit | 1h | `phase2_5_three_models_v2.parquet` + `m4_6_repair_report.md` |

**核心目标**: 让 framework 的 "predicted Pareto" 跟 "实测 Pareto" 一致 (i.e. predicted Pareto 候选实测真的 dominate 非 Pareto).

## 关键技术决策

### Phase 2 简化策略 (避开 1 天工程量)

不重 build 整个 Pyramid_m1_base, 只重 build **pyramid_backbone (PyramidFusion 子模块)**, 因为:
- pyramid_backbone (3.79M params) 占模型 70% 计算
- 外层 (encoder_m1 voxelization + backbone_m1 + shrink_conv + heads) 不动 → 不用处理 channel 一致性
- shrink_conv 输入 = pyramid_backbone 输出 = 384 (= 3 stages × 128 upsample) — 这个 invariant 需保持; 我们只剪每 stage 内部 num_filters, 不动 num_upsample_filter

实际剪枝维度: `num_filters [64, 128, 256]` → `[32, 64, 128]` (50%) 等.

### Phase 3 候选抽样

从 M4.7 v3 输出选:
- 2 个 v3 Pareto 候选 (high amota + medium amota)
- 2 个 v3 非 Pareto 但 channel pruning (negative control)
- 1 个 v3 非 Pareto 且 quant-only (baseline check)

测试方式: 用 M4.6.2 v2 (real channel reduction) 替代 mask-based, lat 真减小可观测.

## 风险 & fallback

| 风险 | 应对 |
|------|------|
| Phase 2 truncate ckpt 维度不匹配 (pyramid_backbone 内部 ResNeXt block 复杂) | fallback: 仍用 mask-based, 但 num_filters 等价缩减后再加 mask |
| Phase 3 实测 AP 跟 v6 预测严重偏差 | 写 honest 报告: "v6 in-sample 准但 OOD 跨模型 generalization 仍有限" |
| Phase 4 三模型整合发现新数据问题 | 不阻塞 commit, 标 TBD |

## 不在本次 6h 范围

- finetune (需 OPV2V train 50GB)
- INT8 真 GPU 推理 (需 TRT)
- uniad_tiny / univ2x_full 真实测 (本次只 Pyramid)
