# 预测器训练 plan (未来工作)

> 仅记录待做工作. 已完成内容见 `00_故事评估与实验路线_v1.md` §3.6.

---

## P0 — Pyramid 单模型 LGB 闭环 (~6-8 h)

**目标**: 训出 Pyramid 单模型 lat predictor, CV MAE < 5pp.

| Step | 内容 | 工具 / 文件 |
|---|---|---|
| P0.1 | 简化 Pyramid 为 1-module schema | `framework/config_schema.py` 加 `PYRAMID_M1_MODULES = ("model",)` |
| P0.2 | random_search 200 candidates → 经硬约束过滤 → 取 86 unique legal | `framework/searcher_v0.py::random_search(modules=PYRAMID_M1_MODULES)` |
| P0.3 | 对 86 candidates 实测 lat (build engine + bench, **不跑 AP**) | 复用 `framework/measure_pyramid.py` 接口, AP 用 rule-based 估 |
| P0.4 | 训 Pyramid LGB lat predictor (LightGBM regression, 81-feature 同 v6) | 仿 `scripts/phase2/train_lgb_v6.py` 写 `train_lgb_pyramid.py` |
| P0.5 | hold-out 10% 做 CV, 报 spearman + MAE | `models/lgb_pyramid_lat.txt` + `results/lgb_pyramid_metrics.json` |
| P0.6 | 把 LGB 接到 `m4_9_closed_loop_v2.py` 替换 rule-based predictor | 验证 LGB predicted vs measured 误差 < rule-based |

**输出**: `models/lgb_pyramid_lat.txt`, `data/pyramid_random_86.parquet`, P0 报告

---

## P1 — 跨模型 LGB (~3-5 day)

**目标**: 训出 cross-model LGB, 三模型 hold-out CV MAE < 7pp.

| Step | 内容 | 工具 / 文件 |
|---|---|---|
| P1.1 | UniV2X full 跑 ~80 random candidates 实测 lat | M4 主线 + adapter `framework/adapters/univ2x_full.py` |
| P1.2 | uniad_tiny 跑 ~80 random candidates 实测 lat | M5 主线 + adapter `framework/adapters/uniad_tiny.py` |
| P1.3 | 合并 240 anchor (Pyramid 86 + UniV2X 80 + uniad_tiny 80) | 统一 schema 写到 `data/baseline_4090_v2.parquet` |
| P1.4 | 训 cross-model LGB (含 model_class one-hot) | `scripts/phase2/train_lgb_v7_crossmodel.py` |
| P1.5 | leave-one-model-out CV (训 2 模型, 测 1 模型) | 报真正的 OOD MAE |

**输出**: `models/lgb_v7_crossmodel.txt`, P1 报告

---

## P2 — NSGA-II + LGB 真 framework demo (~半天)

**目标**: 替换 `m4_9_closed_loop_v2.py` 的 enumeration 为真正搜索.

| Step | 内容 | 工具 / 文件 |
|---|---|---|
| P2.1 | random_search 1000-3000 candidates → cross-model LGB 评估 | `framework/searcher_v0.py` + `models/lgb_v7_crossmodel.txt` |
| P2.2 | NSGA-II 多目标 Pareto search | `framework/nsga2_pareto_search.py` (已写好, 接 LGB) |
| P2.3 | Top-10 Pareto 候选实测验证 | 复用 `framework/measure_pyramid.py` 接口 |
| P2.4 | predicted vs measured 报告 | `results/m4_9_closed_loop_v3_demo.md` |

**输出**: 1000-candidate Pareto frontier + Top-10 真实测量验证

---

## P3 — 制品化 (Step 5, ~半天)

| Step | 内容 | 输出 |
|---|---|---|
| P3.1 | 部署 manifest yaml (engine 路径 + TRT 版本 + sm + sha256) | `deploy/pyramid_dair_m1_manifest.yaml` |
| P3.2 | Pareto SLA 决策表 (lat<X / AP>Y → 推荐 anchor) | `deploy/pyramid_dair_m1_decision_table.md` |
| P3.3 | TRT timing cache 持久化 | `models/pyramid_dair_m1_timing_cache.bin` |
| P3.4 | Orin 部署脚本 (jetson_clocks + nvpmodel) | `deploy/orin_setup.sh` (已有 v2) |

---

## 顺序约束

```
P0 → P1 → P2 → P3
P0 完成才能开始 P1 (跨模型方法论先在单模型上调通)
P1 完成才能开始 P2 (NSGA-II 必须接跨模型 LGB 才有意义)
P3 可独立做, 不阻塞 P2
```

## 时间表

| 周 | 任务 | 累计工作 |
|---|---|---|
| Week 1 | P0 (~1 day) | Pyramid LGB 闭环 |
| Week 1-2 | P1 UniV2X full (~2 day) | 加上 cross-model anchor |
| Week 2 | P1 uniad_tiny (~2 day) + 训 LGB v7 | cross-model LGB |
| Week 2 | P2 (~半天) + P3 (~半天) | framework demo + 制品化 |

总计 ~7 working day 到 paper level minimum.
