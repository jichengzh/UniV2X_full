# 7_7_2 进度报告 — P0/P1 完成 + cost_model v2 诚实评估 + SMBO 入口

**时间**: 2026-07-02
**会话时长**: ~106 分钟

## 已完成

### P0 数据收口(全部完成, validator PASS)
- 补 FP32 latency 两点 lhc_17/s2_096: 路径 A(strict-direct tuned)两 label **可复现 CUDA illegal memory access**; 转路径 B **default-only salvage**(lhc_17=39.49ms, s2_096=45.60ms, us÷1000), provenance 标清不冒充 tuned。
- **根因**: 06-29 已写 salvage row 但 generator 的 `fp32_smoke_paths` 不读 strict_direct 文件 → summary 判 no_claim。修法: 新增 gap2 文件 `rows/fp32_latency_gap2_default_salvage_rows_20260702.jsonl` + generator 加 additive 源(只影响这2点)。
- 三精度三指标 **180/180 全 measured**, validator PASS。
- frontier_01 异常策略落定(降精度精度悬崖, 非 bug); frontier_31 复核为正常点。

### P1 cost model(v1 有缺陷 → v2 修正)
- **v1(作废)**: LOO-CV + 绝对值 + L2。缺陷: 每宽度3精度近重复 → LOO 泄漏。v1 报的 AP Spearman 0.86 是泄漏假象, **作废**。
- **v2(采信)**: 按 IEEE 论文方法重构:
  - latency/energy: 预测 **log1p(y)** 值头 + **lambdarank** 排序头, **leave-one-width-out** 诚实 CV。
  - AP: **残差预测**(锚点=同宽度FP32 AP, 预测量化惩罚 ΔAP)。

## 当前状态(v2 诚实指标, leave-one-width-out CV)
| 指标 | 方法 | MAE | Spearman | 基线 |
|---|---|---|---|---|
| latency | log1p 值头 | 1.25ms | **0.987** | 12.3ms |
| energy | log1p 值头 | 0.39J | **0.974** | 2.33J |
| AP ΔAP残差 | 同宽FP32锚 | 0.00194 | — | mean=0.00149(模型未超基线) |

**结论**:
1. **latency/energy cost model 扎实可用**(Spearman 0.97-0.99, MAE ~10× 优于基线, log-target 确认正确)。
2. **lambdarank 在 180 行单分组数据上无增益**(0.89/0.83 < 值头), 数据量太小; 方向对但当前规模价值为零。
3. **AP 量化惩罚在噪声底之下**(ΔAP 均值 −0.0009 vs 标准差 0.0028), 残差模型打不过常数预测 → 高原采样区 AP 无可学信号(非方法失败, 是数据真相)。

## 下一步(SMBO, 用户已拍板)
- **暂不找 AP 敏感信号**, 直接用 latency/energy 模型跑 SMBO 闭环, 看闭环能否提升模型。
- 参考: `method_zh_ieee_v1.md` + `3_design_exploring_tvm_integration_v1.md` + `2_design_cost_model_v1.md`。
- 闭环: 候选生成 → 预测(+不确定度) → 采集函数选点 → H800 实测 → 回灌重训 → 看实测vs预测排序改善。
- **待补前置**: GBM 无原生不确定度, SMBO 采集函数(EI/UCB)需 σ → 用分位数 LGBM / RF 方差。

## 关键文件清单
| 文件 | 类型 | 状态 |
|---|---|---|
| scripts/stage2_train_cost_model_v1.py | 新增 | 作废(泄漏) |
| scripts/stage2_train_cost_model_v2.py | 新增 | 已验证(需 n_jobs=1 避免线程震荡) |
| scripts/stage2_h800_run_measurement_job.py | 复用 | latency 补点入口(须 TVM env: ${V2X_DATA_ROOT}/tvm310/bin/python + LD_LIBRARY_PATH) |
| rows/fp32_latency_gap2_default_salvage_rows_20260702.jsonl | 新增 | 已合入 |
| scripts/stage2_generate_original60_quant_state_coverage.py | 修改 | 加 gap2 源 |
| cost_model/reports/stage2_cost_model_v2_{report,metrics}_latest.* | 新增 | 已生成 |
| cost_model/models/stage2_cost_v2_*_v2.txt | 新增 | latency/energy/AP残差 LGBM |
| progress/7_2/{4,5,6}_7_2_*.md | 新增 | 异常策略/冻结记录/P2入口 |

## 坑
- **lightgbm 必须 n_jobs=1 num_threads=1**: 180 行数据默认多线程 → 191 线程震荡, 10min 跑不完; 单线程秒级。
- H800: <PRIVATE_HOST>:30001, 密码 12345678; TVM 任务须用 tvm310 env + LD_LIBRARY_PATH; latency 一卡一进程, 用完全空闲 GPU。
