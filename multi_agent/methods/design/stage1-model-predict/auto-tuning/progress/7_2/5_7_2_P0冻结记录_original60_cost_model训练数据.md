# 5_7_2_P0 冻结记录 — original60 cost_model 训练数据

## 0. 冻结时间与结论
- 时间: 2026-07-02
- 结论: **original60 三精度三指标表 180/180 全 measured, validator PASS, 冻结为 cost-model 训练/校准数据快照 v1。**

## 1. 覆盖状态(权威 summary,冻结时)
| precision | latency | AP | energy |
|---|---|---|---|
| FP32 | 60/60 | 60/60 | 60/60 |
| FP16 | 60/60 | 60/60 | 60/60 |
| INT8 | 60/60 | 60/60 | 60/60 |

validator: status=PASS。

## 2. 本轮 P0 变更(相对上一冻结候选)
1. **补 FP32 latency 两点(lhc_17/s2_096)**:
   - 路径 A(strict-direct tuned)于 20260702 重跑,**两 label 均可复现 CUDA illegal memory access**(确定性失败,tuning 阶段 meta_schedule)。
   - 转路径 B(default-only salvage):采用 default 调度实测 latency。
     - lhc_17: latency_p50=39488.259 us → **39.488259 ms**
     - s2_096: latency_p50=45600.732 us → **45.600732 ms**
   - 本 session fresh default_us 复现历史值 <0.05%(lhc_17 39474.719us / s2_096 45575.66us),证明 default 测量稳定。
   - provenance 标注(不冒充 tuned):`schedule_policy=default`,`measurement_source=true_measurement_default_salvage`,`quality_gate_status=fp32_latency_strict_direct_h800_default_salvaged_after_tuned_failure`,`failure_reason=tuned_cuda_illegal_memory_access`。
   - 载体: 新增 `rows/fp32_latency_gap2_default_salvage_rows_20260702.jsonl`(仅 2 行),**不改旧 strict_direct 文件、不删历史失败 raw**。
   - generator 变更: `scripts/stage2_generate_original60_quant_state_coverage.py` 的 `fp32_smoke_paths` 增加该 gap2 文件为 additive 源(仅这 2 label 受影响,其余 58 点不动)。
   - 新 raw(H800,不覆盖旧失败): `multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/raw/original60_latency_fp32_gap2_20260702/`
2. **frontier_01 异常策略落定**: 见 `4_7_2_frontier_01异常策略与cost_model数据冻结口径.md`。

## 3. 根因备忘(为何之前 no_claim,非缺测量)
上一 agent 06-29 已写了 lhc_17/s2_096 的 default-salvage row(measurement_status=measured, latency 值齐全),但 generator 的 `fp32_smoke_paths` **不读 strict_direct 文件**,故 summary 仍判 no_claim。本轮通过新增 gap2 源修复,并补齐 provenance 字段。→ 教训: "数据覆盖" ≠ "productization",measured row 存在不等于已并入权威 summary。

## 4. 冻结清单(sha256 前16位)
| file | sha256_16 | lines |
|---|---|---|
| exports/original60_quant_three_metric_summary_latest.json | 92bff886370d6c95 | - |
| exports/original60_quant_three_metric_summary_latest.csv | 6404c8eaad1956bc | - |
| rows/native_int8_original60_ap_rows_v1.jsonl | b50bf8dfecbaa120 | 62 |
| rows/fp16_true_original60_ap_rows_v1.jsonl | 935a314726beb716 | 61 |
| rows/fp32_true_original60_ap_rows_v1.jsonl | 2d953a3de7853849 | 60 |
| rows/fp16_rewritten_tensorcore_full60_latency_rows_v1.jsonl | 22788c14a3899012 | 60 |
| rows/fp32_original60_energy_threaded60_rows_v1.jsonl | 568f63a1e1866bb9 | 60 |
| rows/fp32_latency_gap2_default_salvage_rows_20260702.jsonl | ecaac623057b00e2 | 2 |

注: rows 文件行数 > 60(native_int8 62 / fp16 61)是历史/复核 rerun 保留,canonical 选择在 generator 构建时按 `is_compliant + score=(created_at,run_id)` 每 label 取最新合规行。**不在 rows 层删重复。**

## 5. cost-model 训练用数据口径(交 P1)
- 训练特征来源: summary 的 180 measured cells(width三维 × precision)。
- AP 回归: **排除 frontier_01 的 FP16/INT8 AP=0 悬崖点**(单列为已知精度悬崖证据),FP32 保留。
- latency/energy 回归: 全 180 点保留(含 frontier_01,含 2 个 default-salvage;salvage 点建议带 `schedule_policy=default` 特征标记,避免与 tuned 点混淆调度轴)。
- 详见 `4_7_2_frontier_01异常策略与cost_model数据冻结口径.md` §3/§4。

## 6. P0 勾稽(全部完成)
- [x] P0-A: lhc_17/s2_096 FP32 latency 补 measured(default salvage,provenance 标清)
- [x] P0-B: generator + validator PASS,latency 60/60
- [x] P0-C: frontier_01 异常策略
- [x] P0-D: 冻结 cost-model 训练数据(本文)

## 7. 下一步(P1)
训练 AP/latency/energy 三指标 cost model,按 §5 口径处理 frontier_01 与 salvage 点。
