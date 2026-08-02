# 6_7_2_P0+P1 完成 — cost model 就绪 — P2 入口

## 0. 状态
- **P0 完成**: original60 三精度三指标 180/180 全 measured, validator PASS, 已冻结为 cost-model 训练数据。
- **P1 完成**: AP/latency/energy 三指标 cost model 训练完毕, 指标见下。
- 下一步: **P2 搜索闭环 smoke**。

## 1. P1 cost model 结果(LOO-CV, GBM, 权威)
| target | 变体 | MAE | RMSE | Spearman | Kendall | 说明 |
|---|---|---|---|---|---|---|
| AP70 | allpts | 0.0195 | — | 0.413 | — | frontier_01 三零点污染 |
| AP70 | **excl_anom** | **0.0020** | — | **0.858** | — | 排除 frontier_01 后优秀 |
| latency_ms | allpts | ~1.41 | — | **0.977** | — | 排序极好 |
| energy_j | allpts | ~0.48 | — | **0.941** | — | 排序好 |

(完整指标: `cost_model/reports/stage2_cost_model_metrics_latest.json`)

**关键结论**: AP 排除 frontier_01 后 Spearman 0.41→0.86 / MAE 0.0195→0.0020, **数据验证了异常策略正确**。latency/energy 排序 Spearman 0.94-0.98, 适合搜索内层排序用途(设计文档强调 ranking 优先于绝对值)。

## 2. 产物清单(output-root = .../original60_quant_20260627/)
```
cost_model/train/original60_training_table_latest.{csv,json}      # 180行训练表
cost_model/schema/stage2_cost_model_feature_schema_v1.{md,json}   # 特征定义
cost_model/models/stage2_cost_{ap70,latency_ms,energy_j}_{allpts,excl_anom}_v1.joblib  # 6模型
cost_model/reports/stage2_cost_model_training_report_latest.md
cost_model/reports/stage2_cost_model_metrics_latest.json
exports/original60_cost_model_training_freeze_latest.{json,md}    # 机器可读冻结
exports/original60_quant_anomaly_review_20260702.{json,md}        # frontier_01 blocked / frontier_31 normal
```
复现: `python3 scripts/stage2_train_cost_model_v1.py`

## 3. 特征与口径
- 特征(15): w0/w1/w2 + w_sum/w_prod_norm + precision_bits + is_fp16/is_int8 + fam_*(6) + lat_sched_default。
- 目标: ap70 / latency_ms / energy_j_per_inference。
- AP 回归两版: allpts(含 frontier_01) 与 excl_anom(排除)。**下游搜索用 excl_anom AP 模型**。
- latency/energy 全 180 点(含 frontier_01、含 2 个 default-salvage; salvage 由 lat_sched_default 特征标记)。

## 4. 已知局限(交 P2/审查)
1. **180 点小样本 + LOO**: GBM 在 180 行上 CV, 指标是诚实估计但样本有限。
2. **AP 信号弱**: AP70 跨宽度变化幅度小(设计文档已述), 故即便 excl_anom Spearman 0.86 也主要靠 width 主序; 绝对 MAE 0.002 很小是因 AP 本身方差小。
3. **frontier_01 悬崖**未被模型学习(被排除); 若搜索会探到极窄宽度, 需额外悬崖检测或保留 allpts 模型作保守预测。
4. cost model 是 backbone-scope latency/AP; e2e 需经 Amdahl 桥(见既有 f_lat 相关记忆)。

## 5. P2 计划(搜索闭环 smoke)
目标: 跑通 冷启动→候选生成→cost model 预测→高质量点实测→证据回流→重训 的最小闭环。
- 载体: `framework/`(已有 accuracy_predictor/latency_estimator/feature_encoder + search_three_arm.py 等)。
- 建议先 smoke: 用 P1 三模型预测一批候选宽度×精度 → 选 top-K → H800 实测验证预测排序 → 计算实测 vs 预测 Spearman。
- 待确认: P2 是否复用 framework 既有搜索器, 还是接本阶段 cost model 新建轻量搜索 smoke。
