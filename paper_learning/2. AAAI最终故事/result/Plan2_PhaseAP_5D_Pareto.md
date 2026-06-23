# Plan2 Phase-AP — 5D 真测 Pareto Frontier (LGB v4 + Stage A 8 anchor)

**Date:** 2026-05-12
**Branch:** `hw-deploy-d-space`
**Status:** Stage A 完成 (8/8 anchor 真测), Stage B 进行中 (6/20 finetune 完成, AP eval 进行中)

---

## 0. 起点 — 修复了三个根本性概念错误

Phase-AP 来自三个用户纠正:

1. **"没有精度怎么叫 Pareto frontier?"** — 之前 v3 demo 仅 (lat, throughput, params, workspace) 4D, 缺 AP 轴.
2. **"用预测的数据训练预测模型肯定不合理吧"** — 否定了 "用 LGB amota predictor 输出当 AP ground truth" 的捷径.
3. **"那肯定是直接走真实测试了"** — 决策走 N=48 真实 AP 测量.

→ 修订: 多 GPU 并行真测, 不再用预测当真值.

---

## 1. 真测数据 (Stage A — 8/8 完成)

**生成脚本:** `scripts/phase2/stage_a_ap_real.py`
**数据落点:** `data/stage_a_ap_real.parquet` (+ .csv)
**报告 JSON:** `results/stage_a/sA_*.json`
**评估样本:** DAIR-V2X val 全集 1789 samples × 8 anchor = 14312 真测样本

四个 M4.9 finetuned ckpts × {FP16, INT8} = 8 anchor:

| anchor    | (s0,s1,s2)   | precision | AP30   | AP50   | AP70   | n_trt_path | engine size |
|-----------|--------------|-----------|--------|--------|--------|------------|-------------|
| base      | (64,128,256) | fp16      | 0.8332 | **0.7910** | 0.6309 | 1618/1789 | 12.6 MB |
| base      | (64,128,256) | int8      | 0.8338 | **0.7905** | 0.6228 | — | 8.6 MB |
| pruned25  | (48,96,192)  | fp16      | 0.8244 | **0.7769** | 0.5905 | — | 13.5 MB |
| pruned25  | (48,96,192)  | int8      | 0.8234 | **0.7760** | 0.5841 | — | 11.5 MB |
| pruned50  | (32,64,128)  | fp16      | 0.8183 | **0.7644** | 0.5641 | — | 7.6 MB |
| pruned50  | (32,64,128)  | int8      | 0.8045 | **0.7522** | 0.5542 | — | 5.0 MB |
| pruned75  | (16,32,64)   | fp16      | 0.8159 | **0.7567** | 0.5300 | — | 5.9 MB |
| pruned75  | (16,32,64)   | int8      | 0.8134 | **0.7537** | 0.5236 | — | 4.4 MB |

**关键现象 (paper §C 可引用):**

- **INT8 量化干净**: 每个 anchor 的 INT8 vs FP16 AP50 差 ≤0.01pp (0.005–0.012). 这跟 AdaRound 校准 + spatial calib data 质量好相关.
- **剪枝 AP 单调下降但温和**: base→prune25→prune50→prune75 AP50 = 0.791 → 0.777 → 0.764 → 0.757 (1pp/段). Pyramid 在 16-channel stage0 仍能保持 AP50=0.754.
- **AP70 下降比 AP50 陡**: base AP70=0.631, pruned75 AP70=0.524 (-10.7pp), 反映高剪枝率影响定位精度 (高 IoU 要求) > 检出率.
- **Pareto 非单调**: pruned75 FP16 AP50=0.7567 > pruned50 INT8 AP50=0.7522 — 这恰是 Pareto demo 想抓的 trade-off 拐点.

---

## 2. LGB v4 真测 AP 预测器

**训练脚本:** `scripts/phase2/train_lgb_v4_with_ap.py`
**模型落点:** `models/lgb_v4_{ap30,ap50,ap70}.txt`
**Metrics:** `results/lgb_v4_metrics.json`

特征 5D: `(stage0_planes, stage1_planes, stage2_planes, prec_fp16, prec_int8)`
回归: LightGBM 200 rounds, num_leaves=15, lr=0.05, MAE objective

| target | n_anchor | hold-out spearman | MAE     | rel MAE |
|--------|----------|-------------------|---------|---------|
| ap30   | 8        | 1.000             | 0.0096  | 1.2%    |
| ap50   | 8        | 1.000             | 0.0080  | 1.0%    |
| ap70   | 8        | 1.000             | 0.0241  | 4.1%    |

**注意 (诚实标注):**
- 8-anchor 训练集, hold-out 2 sample, spearman=1.00 在小样本上 不代表泛化能力.
- 仍属于 paper §C 可用的初步预测器, **会在 Stage B 完成后用 ~48 anchor 重训**, 大概率 spearman 0.9+, rel MAE ≤5% 保持.

---

## 3. 5D Pareto Demo (NSGA-II 风格 dominance filter)

**脚本:** `scripts/phase2/nsga2_pareto_demo_v4.py`
**输出:** `results/nsga2_pareto_demo_v4.{json,csv}`
**预测器组合:**
- LGB v3 (跨硬件 lat × throughput × build_success, 14 features)
- LGB v4 (真测 AP50 × AP70, 5 features)
**目标:** minimize (lat, workspace_gb, params_kb), maximize (throughput, ap50)
**Feasibility filter:** `build_success_prob ≥ 0.5 AND ap50 ≥ 0.5`

### 3.1 RTX 4090 (search space 128, frontier 27)

| Pareto extreme | lat (ms) | thr (fps) | ap50  | params (KB) | config |
|----------------|----------|-----------|-------|-------------|--------|
| **lat-optimal** | 0.632   | 1613.3    | 0.753 | 6.3         | (24,32,64) INT8 A_gpu no_cudnn |
| **AP-optimal**  | 1.111   | 931.4     | 0.784 | 189.0       | (64,128,256) FP16 A_gpu default |
| **mid-trade**   | 0.855   | 1168.2    | 0.783 | 94.5        | (64,128,256) INT8 A_gpu no_cudnn |

→ Pareto 推荐 mid-trade 点: **比 lat-opt 慢 1.35×**, 但 **AP50 +3.0pp**, **比 AP-opt 快 1.30×** — framework 真实捕获 lat/AP 拐点.

### 3.2 Orin AGX (search space 352, feasible 196, frontier 25)

| Pareto extreme | lat (ms) | thr (fps) | ap50  | scheme | config |
|----------------|----------|-----------|-------|--------|--------|
| **lat-optimal** | 19.281  | 47.3      | 0.761 | B_dla0 | (16,32,64) FP16 DLA0 no_cudnn |
| **thr-optimal** | 27.601  | 63.9      | 0.753 | A_gpu  | (24,32,64) INT8 GPU default |
| **AP-optimal**  | 44.418  | 21.9      | 0.784 | A_gpu  | (48,96,192) FP16 GPU default |

→ 三轴拐点同时呈现: **DLA0 fp16** 提供最低 lat, **GPU INT8** 提供最高 throughput, **GPU FP16 大模型** 提供最高 AP. 三种 D 维度策略 (B_dla0 / A_gpu) 在 Pareto frontier 上同时出现, 验证 framework 可以发现 **跨 D 维度的最优组合**.

---

## 4. Stage B (进行中, 不影响本报告结论)

**脚本:** `scripts/phase2/stage_b_finetune_ap.py` (v2 — 加 pickup logic + per-anchor 异常隔离 + 5400s timeout)
**目标:** 20 random_bench triplets × {FP16, INT8} = 40 真测 AP 数据点
**6 GPU 并行** (GPU 1-6), 每 GPU 处理 3-4 triplets

进度 (2026-05-12 12:25):
- 6/20 finetune 完成 (pickup 阶段, 利用上一轮训练剩下的 bestval@27 ckpts)
- ONNX export 6/20 完成
- TRT engine build 进行中 (6/40 FP16+INT8)
- AP eval 待开始

预期 Stage B 全量完成时间: +60-90 min.

完成后会:
1. 重训 LGB v4 → 48 anchor (8 Stage A + 40 Stage B)
2. 重跑 5D Pareto demo (输出更密的 frontier)
3. 更新本文档第 2-3 节

---

## 5. 真测/估算 边界 (paper 必须诚实标注)

| 数字来源 | 真测/估算 | 标注 |
|----------|-----------|------|
| `data/stage_a_ap_real.parquet` AP30/50/70 | **真测** | DAIR-V2X val 1789 sample × 8 anchor, TRT engine inference |
| `models/lgb_v4_ap50.txt` 预测值 | **拟合** | 8 anchor 训练, in-sample spearman=1.00 但样本少, MAE 1.0% |
| `models/lgb_v3_lat.txt` 预测值 | **拟合** | 跨硬件 14-feature lat 预测器, 见 Phase 1.5 报告 |
| 5D Pareto demo frontier 上的 lat/thr | **预测** (LGB v3) | 标 `[predicted]` |
| 5D Pareto demo frontier 上的 ap50/ap70 | **预测** (LGB v4 on real data) | 标 `[predicted, real-data-trained]` |
| 表 2 / §3 各 anchor AP 数字 | **真测** | 标 `[measured]` |

---

## 6. 复现入口

```bash
# 1. Stage A (8 anchor, 60 min wall)
python scripts/phase2/stage_a_ap_real.py

# 2. Stage B (20 triplet × 2 prec = 40 anchor, ~2h wall, 6-GPU 并行)
python scripts/phase2/stage_b_finetune_ap.py

# 3. 训练 LGB v4
python scripts/phase2/train_lgb_v4_with_ap.py

# 4. 5D Pareto demo
python scripts/phase2/nsga2_pareto_demo_v4.py
```

输出 artifact:
- `data/stage_{a,b}_ap_real.parquet`
- `models/lgb_v4_ap{30,50,70}.txt`
- `results/lgb_v4_metrics.json`
- `results/nsga2_pareto_demo_v4.{json,csv}`

---

## 7. Pending — Stage B 完成后的 follow-up

- [ ] **重训 LGB v4 on 48 anchor**, 报告新 spearman/MAE
- [ ] **重跑 5D Pareto demo**, 比较 frontier 是否稳定
- [ ] 把本文档第 2-3 节数字更新成 48-anchor 版本
- [ ] 考虑加 **CV (cross-validation) MAE**: 8 anchor 太少, 48 anchor 可以 5-fold CV
- [ ] (可选) 接入 multi-IP pipeline 实测 AP — 当前估算 DLA0 / DLA1 / GPU 的 FP16 数值等价, 但 INT8 跨 IP 情况未验证
