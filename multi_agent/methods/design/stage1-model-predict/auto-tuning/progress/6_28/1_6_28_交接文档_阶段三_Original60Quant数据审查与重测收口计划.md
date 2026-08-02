# 6/28 交接文档：阶段三 Original60 Quant 数据审查与重测收口计划

更新时间：2026-06-28  
当前状态：总表已补入 FP32 energy 与少量 FP32 AP 参考数据，并新增 latency schedule / TVM strategy 字段；但当前数据集仍不能作为三种精度的同口径最终对比结论。

## 0. 启动初期必须先读

1. 本文档：`${V2X_ROOT}/multi_agent/methods/design/auto-tuning/progress/6_27/57_6_28_交接文档_阶段三_Original60Quant数据审查与重测收口计划.md`
2. 当前最可信的数据审查记录：
   - `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_measurement_source_audit_latest.md`
   - `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_measurement_source_audit_latest.json`
3. FP32 重分类审查记录：
   - `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_original60_remap_audit_latest.md`
   - `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_original60_remap_audit_latest.json`
4. 当前三精度总表：
   - `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md`
   - `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.csv`
   - `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.json`

固定规则：

- latency 指标统一使用 `ms`。
- 当前速度/能耗口径是 TVM backbone/subnet 侧实测，不能写成 full-network/end-to-end 完整系统推理。
- `remap` 不等于“非实测”。当前 FP32 remap 的含义是：历史结果原先被标成 fp16，但 `source_files` 指向 plain `*_backbone.onnx`，不是 `*_backbone_true_fp16.onnx`，因此被重分类为 FP32 证据。
- 不能把历史重分类数据、参考 AP、当前同批次直接实测数据混写成同一级别的最终结论。
- 遇到缺 checkpoint、AP 崩溃、energy 异常、latency 反常时，先记录证据、做最小复现实验和原因审查；除非明确无法推进，不要直接停止。

## 1. 当前最可信的数据审查记录

目前最可信的数据审查记录是：

`original60_quant_measurement_source_audit_latest.md/json`

该审查记录逐格检查了当前总表中 FP32 / FP16 / INT8 三种精度的 latency、energy、AP 来源，并给出是否是直接实测、历史实测重分类、参考数据或 no_claim。

核心结论：

| precision | latency 覆盖 | energy 覆盖 | AP 覆盖 | 当前可信口径 |
|---|---:|---:|---:|---|
| FP32 | 58/60 | 56/60 | 5/60 | latency/energy 多数是历史 H800 实测重分类；AP 是 reference 来源 |
| FP16 | 60/60 | 60/60 | 18/60 | latency/energy 是当前 FP16 路线实测；AP 尚未补齐 |
| INT8 | 60/60 | 60/60 | 1/60 | latency/energy 是 native INT8 full-ONNX topology route 实测；AP 仍是主要空白 |

审查记录中的直接判断：

`all_cells_are_direct_same-level_measurements = false`

因此当前总表可以作为“数据资产盘点表”和“下一阶段补测输入表”，但不能作为三种精度最终同口径性能结论。

## 2. 当前总表内容与空白

当前总表位置：

`${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md`

总表已经包含：

- 60 个 original60 配置候选。
- FP32 / FP16 / INT8 三种 precision 行。
- latency、energy、AP 三类指标列。
- `latency schedule` 与 `latency tvm strategy` 列，用于审查不同路线是否采用相同调度策略。

当前主要空白：

| 空白项 | 当前状态 | 下一步要求 |
|---|---|---|
| FP32 latency | 58/60，有 2 个 no_claim | 若用于最终三精度对比，需要对 60 个配置做同口径重测或至少做 paired audit |
| FP32 energy | 56/60，但存在 0 或极低能耗异常 | 建议全部重测，旧值仅保留为历史证据 |
| FP32 AP | 5/60，且是 reference AP 来源 | 不能当作同口径 FP32 TVM AP；若要三轴完整表，需要补齐或明确标注 reference |
| FP16 AP | 18/60 | 继续补齐剩余 42 个配置；缺 checkpoint 时应进入生成/微调流程 |
| INT8 AP | 1/60 | 先做 5 点 smoke，再扩大到 60 点；需要继续处理精度崩溃/scale-aware 问题 |

## 3. 当前数据存在的问题

### 3.1 FP16 与 FP32 latency 过于接近，且部分 FP32 更快

当前审查显示 FP32 与 FP16 的 measured latency schedule 都是 `metaschedule_tuned`，因此“schedule 策略不同”不是 FP16/FP32 速度接近的主要解释。

已统计的共同配置点：

- FP16/FP32 共同可比 measured label：58 个。
- `median(fp16_ms / fp32_ms) = 0.962920`。
- FP16 与 FP32 相差在 ±5% 内：31 个。
- FP16 快于 FP32 超过 5%：10 个。
- FP16 慢于 FP32 超过 5%：17 个。

异常示例：

| label | FP32 latency ms | FP16 latency ms | fp16/fp32 |
|---|---:|---:|---:|
| frontier_12 | 10.106415 | 14.453522 | 1.430133 |
| lhc_07 | 12.952994 | 18.512662 | 1.429219 |
| lhc_20 | 10.127507 | 14.472070 | 1.428986 |
| lhc_08 | 24.642524 | 34.461323 | 1.398449 |

结论：当前总表不能直接支持“FP16 相比 FP32 有稳定速度收益”的结论。下一阶段必须做 paired latency audit。

### 3.2 FP32 energy 出现 0 或极低值

当前 FP32 energy 多数来自历史 H800 telemetry 重分类。它是实测 telemetry，不是手工编造，但存在明显比较风险。

已确认的问题模式：

- 历史 energy 计算使用 active watt 减 idle baseline。
- 部分运行中 active watt 平均值小于或接近 idle watt 平均值。
- 结果导致 `joule_per_inference` 被裁剪为 0 或接近 0。

异常示例：

| label | FP32 J/inference | active watt avg | idle watt avg | 说明 |
|---|---:|---:|---:|---|
| lhc_07 | 0.000000 | 133.924180 | 135.576429 | active 低于 idle |
| frontier_07 | 0.000000 | 133.090967 | 134.110417 | active 低于 idle |
| frontier_10 | 0.000000 | 128.745773 | 134.057391 | active 低于 idle |
| frontier_11 | 0.000000 | 133.776660 | 141.035714 | active 低于 idle |

结论：FP32 energy 当前只能作为历史测量证据，不能作为最终能耗对比。建议 60 个配置全部重测。

### 3.3 INT8 latency/energy 已覆盖，但 AP 严重不足

INT8 当前已经有 60/60 latency 和 60/60 energy，来源是 native INT8 full-ONNX topology direct route。

但 AP 只有 1/60 进入 canonical 总表。`native_int8_original60_ap_rows` 中存在多个 row，但 unique label 当前只有 `s0_024`，因此总表只能导入 1 个配置点。

结论：INT8 的下一步重点不是继续证明 latency/energy 有无，而是解决 AP 崩溃/非零趋势/scale-aware gate 后，完成 AP 补点。

### 3.4 FP32 数据“映射/重分类”的口径容易误解

当前所谓 FP32 remap 的来源：

- 历史数据最初被标成 fp16。
- 后续审查发现 `source_files` 指向 plain backbone ONNX。
- plain backbone ONNX 对应 FP32 route，而不是 true FP16 route。
- 因此这些历史实测被重分类为 FP32 evidence。

这不是从 FP16 数值推导 FP32，也不是模型估计；但由于它不是当前 FP16/INT8 同一批次、同一协议下重新跑出的 paired FP32 batch，所以不能和当前 FP16/INT8 直接混成最终同口径结论。

## 4. 数据源位置

| 类型 | 路径 |
|---|---|
| 三精度总表 MD | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md` |
| 三精度总表 CSV | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.csv` |
| 三精度总表 JSON | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.json` |
| 最可信 source audit MD | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_measurement_source_audit_latest.md` |
| 最可信 source audit JSON | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_measurement_source_audit_latest.json` |
| FP32 remap audit MD | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_original60_remap_audit_latest.md` |
| FP32 remap audit JSON | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_original60_remap_audit_latest.json` |
| canonical latency rows | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/latency_original60_quant_rows_v1.jsonl` |
| canonical energy rows | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/energy_original60_quant_rows_v1.jsonl` |
| canonical AP rows | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/ap_original60_quant_rows_v1.jsonl` |
| FP32 latency remap rows | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/fp32_latency_original60_remapped_rows_v1.jsonl` |
| FP32 energy remap rows | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/fp32_energy_original60_remapped_rows_v1.jsonl` |
| FP16 true latency rows | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/fp16_true_original60_latency_rows_v1.jsonl` |
| FP16 true energy rows | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/fp16_true_original60_energy_rows_v1.jsonl` |
| FP16 true AP rows | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/fp16_true_original60_ap_rows_v1.jsonl` |
| INT8 latency rows | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/native_int8_full_onnx_original60_latency_rows_v1.jsonl` |
| INT8 energy rows | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/native_int8_full_onnx_original60_energy_rows_v1.jsonl` |
| INT8 AP rows | `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/native_int8_original60_ap_rows_v1.jsonl` |

## 5. 可能原因判断

### 5.1 FP16/FP32 latency 接近或反向

可能原因按优先级排序：

1. FP32 来源是历史重分类数据，不是当前 FP16 同批次 paired rerun，协议差异可能残留。
2. FP16 route 虽然标记为 true FP16，但 TVM lowering 中可能仍存在大量 float32-heavy 计算、cast、layout transform 或非 tensorcore 友好的 kernel。
3. MetaSchedule tuned database 复用策略可能对 FP32/FP16 的收益不同，需要检查 generated TIR、算子 dtype 和 kernel schedule。
4. 输入形状、batch、warmup、repeat、measure_iter、GPU 空闲状态、clock 状态可能存在批次差异。
5. 如果 paired rerun 后仍然 FP16 不快，则说明当前 backbone/subnet 对 FP16 不敏感，或 lowering 没有真正转化为高效 half/tensorcore 路径。

### 5.2 FP32 energy 异常偏小

最可能原因是 idle baseline subtraction 策略导致能耗被抵消。这个问题不是单个配置异常，而是历史 telemetry 协议风险。

下一阶段 energy 重测需要：

- 记录 raw GPU power trace。
- 同时保存 active watt、idle watt、duration、inference count、joule/inference。
- 对 active <= idle 的情况单独标记为 `energy_baseline_invalid`，不能直接裁剪为可信 0。
- 重新生成总表时保留 `measurement_source` 和 `evidence_scope`。

### 5.3 INT8 AP 崩溃

已知当前 INT8 latency/energy 路线打通，但 AP 仍严重不足。可能原因包括：

- QDQ scale/zero-point 与 head/postprocess 分布不匹配。
- backbone/subnet 量化后特征尺度改变，后续检测头阈值或解码分布未适配。
- ScaleAwareGate 尚未在足够配置上证明 AP30/AP50/AP70 呈合理非零趋势。
- INT8 AP rows 当前 unique label 太少，无法判断是单点问题还是系统性问题。

## 6. 下一步解决方案

### 阶段 A：冻结现有表为审查快照

目标：

- 当前总表继续保留，但标记为 audit snapshot。
- 不再把该表当成最终三精度同口径结论。
- 所有新增数据必须进入 row-level jsonl，并带 `measurement_source`、`evidence_scope`、`schedule_policy`、`tvm_strategy`、`run_id`。

完成标准：

- `original60_quant_three_metric_summary_latest.md/csv/json` 中保留 schedule 列。
- `original60_quant_measurement_source_audit_latest.md/json` 可以解释每个 precision/metric 的来源。

### 阶段 B：FP16/FP32 paired latency audit

目标：

- 先选 8-10 个配置做 paired rerun，其中必须包含上文 FP32 更快的异常点和若干正常点。
- 同一 H800 类型 GPU、同一 TVM route、同一 schedule 策略、同一 warmup/repeat/measure_iter、同一输入形状。
- 输出 FP32 与 FP16 的 latency、TIR/operator dtype、主要 kernel schedule、ONNX source path。

建议优先配置：

- `frontier_12`
- `lhc_07`
- `lhc_20`
- `lhc_08`
- `s0_024`
- 另选 3-5 个接近中位数的 original60 配置。

完成标准：

- 能解释 FP16/FP32 latency 接近或反向的主要原因。
- 如果 paired rerun 证明历史 FP32 remap 仍可用，则保留其对比口径。
- 如果 paired rerun 与历史 remap 差异明显，则把历史 FP32 latency 降级为 reference，并开启 60 点 FP32 paired latency 重测。

### 阶段 C：FP32 energy 60 点重测

目标：

- 对 60 个 original60 配置重新测量 FP32 energy。
- 不再接受 active <= idle 后裁剪成 0 的结果作为可信最终值。

完成标准：

- FP32 energy 60/60 direct measurement，或者每个失败点有明确 quarantine/no_claim 原因。
- 总表中 FP32 energy 不再出现无法解释的 0 或极小值。
- 保存 raw telemetry、idle trace、active trace 和 row-level jsonl。

### 阶段 D：FP16 AP 补齐

目标：

- 从当前 18/60 补到 60/60。
- 如果没有可用 checkpoint，不能只反复查找；应进入 checkpoint 生成/微调流程。

完成标准：

- FP16 AP30/AP50/AP70 至少 60 个配置有 direct eval 或明确失败原因。
- 每个 AP row 保存 checkpoint path、eval config、dataset split、run_id。

### 阶段 E：INT8 AP 先 5 点 smoke，再扩到 60 点

目标：

- 先完成 5 个 original60 配置点 INT8 AP smoke。
- 验收口径为：非空，AP30/AP50/AP70 有合理非零趋势，head/postprocess 分布能和 FP16 baseline 解释。
- smoke 通过后扩大到 60 点。

完成标准：

- INT8 AP 从 1/60 扩到至少 5/60 smoke。
- 若 smoke 可恢复，再扩到 60/60。
- 若 smoke 不可恢复，必须输出 scale-aware/debug 证据，而不是只报告失败。

## 7. 最终收口目标

下一阶段最终目标应该收口到非常具体的交付物：

1. 一张可审阅的三精度总表，60 个 original60 配置，FP32 / FP16 / INT8 每个 precision 都有 latency、energy、AP 三轴数据或明确 no_claim/quarantine 原因。
2. FP32 energy 完成 60 点重测，解决历史 0 能耗问题。
3. FP16 AP 从 18 点补到 60 点，缺 checkpoint 时完成 checkpoint 生成/微调。
4. INT8 AP 先完成 5 点 smoke，再推进 60 点补齐。
5. 对 FP16/FP32 latency 接近或反向给出 paired audit 证据，不能只做口头解释。
6. 所有最终结论必须引用 `measurement_source`、`evidence_scope`、`schedule_policy` 和原始 row-level 文件。

## 8. 推荐 /goal 命令

```text
/goal 阅读并严格遵守 ${V2X_ROOT}/multi_agent/methods/design/auto-tuning/progress/6_27/57_6_28_交接文档_阶段三_Original60Quant数据审查与重测收口计划.md。先读取 original60_quant_measurement_source_audit_latest.md/json、fp32_original60_remap_audit_latest.md/json、original60_quant_three_metric_summary_latest.md/csv/json，确认当前最可信审查记录与总表状态。下一阶段目标：把当前 original60 三精度数据从 audit snapshot 推进到可审阅的同口径实测表。必须完成：1) FP16/FP32 paired latency audit，至少覆盖 frontier_12、lhc_07、lhc_20、lhc_08、s0_024 及 3-5 个中位数配置，解释 FP16/FP32 latency 接近或反向；2) FP32 energy 60 点重测或逐点 quarantine/no_claim，解决历史 active-idle baseline 导致的 0/极低能耗问题；3) FP16 AP 从现有 18/60 补齐，缺 checkpoint 时进入 checkpoint 生成/微调流程；4) INT8 AP 先完成 5 点 smoke，要求非空且 AP30/AP50/AP70 有合理非零趋势，再推进 60 点；5) 总表必须保留 latency ms、energy、AP、measurement_source、evidence_scope、schedule_policy、tvm_strategy、run_id。固定规则：当前速度/能耗是 TVM backbone/subnet 口径，不写成 full-network/end-to-end；remap 表示历史实测重分类，不表示非实测；不能把 reference AP、历史重分类、当前直接实测混作同级最终结论；遇到失败必须先做复现、记录证据、分析和修复，除非明确不可解决，否则不要直接停止。
```

