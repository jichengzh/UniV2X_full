# 近期数据索引 README_2

更新时间：2026-07-01

本文只做近期数据目录索引，重点说明每批数据的生成背景、覆盖面积和可信度。可信度分三类：

- 可直接使用：可以进入性能预测器或阶段性分析，但仍需保留 provenance 字段。
- 谨慎使用：有参考价值，但硬件、后端、测量链路或重复策略和当前目标不完全一致。
- 仅调试证据：用于定位链路问题，不应作为 LUT 主训练数据。

## 总规则

- 当前主目标是 H800 + TVM 路线，量化维度已进入搜索空间；旧 4090 + TRT 数据只能作为历史参考。
- latency 和 energy 未来正式测量必须尽量保证目标 GPU 空闲；AP 评估可以多卡并行，但必须记录 ckpt、数据集、评估命令和配置标签。
- `original60_quant_20260627` 是近期主工作根；更新 `exports/*latest*` 主表前必须先备份旧版本，并在 README 或交接文档中写明 provenance。
- 需要补表或修表时，优先先在备份根目录操作，例如 `original60_quant_20260627_backup_work_20260630_int8_ap52`；如果确需提升为主表，必须记录旧表备份文件。
- H800 上的大型 raw/full-val 产物未全部迁回本机；本次迁回的是行表、jobs、exports、manifest 和 raw 引用。

## 按生成时间的数据索引

### 2026-06-23：dataset_v2 历史基线

- 路径：`${V2X_ROOT}/multi_agent/data/dataset_v2.csv`
- 覆盖：66 条 CSV 记录；另有 `dataset_v2_learning.csv`、parquet 副本和 `sources/*` 原始来源表。
- 背景：早期 4090 + TRT/历史实测整理，用于建立初版 cost model 数据格式和字段。
- 可信度：谨慎使用。硬件和后端不是当前 H800 + TVM 主路线，不能直接解释当前 H800 latency/energy。

### 2026-06-25：dataset_v2 已整理子集

- 路径：`stage2_lut_generation_v1/existing/`
- 覆盖：`dataset_v2_full_66.csv` 66 条，`dataset_v2_ap_valid_63.csv/jsonl` 63 条 AP 有效记录，`dataset_v2_front_32.csv` 32 条 front 子集。
- 背景：把 dataset_v2 迁入 Stage2 LUT 目录，方便和新生成 LUT 对齐。
- 可信度：谨慎使用。适合做字段/schema 对齐和历史对照，不适合作为 H800 + TVM 主训练数据。

### 2026-06-25：H800 + TVM 历史证据整理

- 路径：`stage2_lut_generation_v1/existing_h800_tvm/`
- 覆盖：pyramid、codriving、int8 scope-limited 等少量 CSV 和 summary。
- 背景：整理历史 H800 + TVM 实测证据，曾支持“优化后约 10x 加速”的初步判断。
- 可信度：可作为结论证据使用，但覆盖面积小，不足以训练 LUT 预测器。

### 2026-06-26：outlier、registry、readiness 早期门控结果

- 路径：`stage2_lut_generation_v1/exports/`、`stage2_lut_generation_v1/artifacts/`、`stage2_lut_generation_v1/registry/`
- 覆盖：outlier report、artifact registry、quick review、readiness gate 等。
- 背景：早期为三臂 LUT 生产搭建校验和审查链路。
- 可信度：仅调试证据。可参考门控字段和失败原因，不作为训练数据。

### 2026-06-26：overnight_6h 长跑尝试

- 路径：`stage2_lut_generation_v1/generated/overnight_6h_20260626/`
- 覆盖：job plan 和 continuation plan 摘要为主。
- 背景：首次 6 小时后台 LUT 生产尝试。
- 可信度：仅调试证据。重复测量较多，配置覆盖效率不足。

### 2026-06-26 至 2026-06-27：coverage_pipeline_v1

- 路径：`stage2_lut_generation_v1/generated/coverage_pipeline_v1/`
- 覆盖：`exports/original60_measured_lut_summary.csv` 60 条；`rows/latency_lut_rows_original60_v1.jsonl` 116 条；`rows/energy_lut_rows_original60_v1.jsonl` 56 条。
- 背景：original60 候选点早期 latency/energy 覆盖流水线。
- 可信度：谨慎使用。可用于搜索空间覆盖和工程链路验证；energy/latency 需结合 provenance 和 outlier 审查。

### 2026-06-26 至 2026-06-27：AP 稳定性链路

- 路径：`stage2_lut_generation_v1/generated/ap_stability_20260626/`
- 覆盖：s0/s1/s2 等少量 AP 稳定 smoke、ckpt/数据同步/评估日志。
- 背景：定位 AP 不稳定问题，验证 exact-label checkpoint 与 AP 评估链路。
- 可信度：仅调试证据。适合回看 AP 链路如何跑通，不是完整 LUT。

### 2026-06-27：quant_smoke_20260627

- 路径：`stage2_lut_generation_v1/generated/quant_smoke_20260627/`
- 覆盖：量化 smoke 的 jobs/raw/exports，规模较小。
- 背景：把 FP16/INT8/FP32 精度维度纳入搜索空间前的 smoke。
- 可信度：仅调试证据。不能代表最终 INT8 AP 或性能。

### 2026-06-27 至 2026-06-30：original60_quant_20260627 主工作根

- 路径：`stage2_lut_generation_v1/generated/original60_quant_20260627/`
- 覆盖：original60 在 FP16/FP32/INT8 下的 rows、jobs、exports、raw、artifacts。
- 背景：近期量化维度主路线，目标是构建 latency/AP/energy 三指标 LUT。
- 可信度：混合目录，需要按行表区分使用，不要只看目录名。

主要可用行表：

- `rows/fp16_true_original60_latency_rows_v1.jsonl`：60 条，FP16 latency；可直接使用。
- `rows/fp16_true_original60_energy_rows_v1.jsonl`：60 条，FP16 energy；谨慎使用，需检查 GPU 空闲和功耗门控。
- `rows/fp16_true_original60_ap_rows_v1.jsonl`：61 条，约 60 个 label；历史 FP16 AP 行表，有重复 label，训练前需去重；当前主表已优先使用 rewritten full-val 新行表。
- `rows/fp16_rewritten_original60_ap_rows_v1.jsonl`：64 条，其中 full-val 60 条 / 60 个 label，smoke 4 条；当前 FP16 AP 的最新主来源。full-val 行可直接用于 FP16 AP 分析，smoke 行仅作为链路调试证据。AP30/AP50/AP70 完整值在该行表内；总表继续沿用历史 AP70 汇总列。
- `rows/fp32_latency_strict_direct_rows_v1.jsonl`：116 条，约 60 个 label；可直接使用，包含重复测量。
- `rows/fp32_original60_energy_threaded60_rows_v1.jsonl`：60 条；谨慎使用，energy 仍需保留测量可信度标记。
- `rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl`：60 条；可作为 INT8 latency 候选，但需确认 full_onnx/native_int8 路径与当前 AP 路径一致。
- `rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl`：60 条；谨慎使用，energy 需继续审查。
- `rows/native_int8_original60_ap_rows_v1.jsonl`：本地主根原始行表仍只有 4 条；最新 52 条 INT8 AP 已从备份工作目录合入 `exports/original60_quant_three_metric_summary_latest.*` 主表。

主表提示：

- `exports/original60_quant_three_metric_summary_latest.*`：180 行三指标汇总；2026-07-01 已合入 FP16 rewritten full-val AP60，INT8 AP 覆盖仍为 52/60，FP32 AP 覆盖仍为 5/60。
- `${V2X_ROOT}/multi_agent/data/original60_quant_three_metric_summary_latest.md`：根目录下的摘要副本，已同步到 2026-07-01 最新总表内容；权威机器可读文件仍以 `exports/*.json` 为准。

### 2026-06-29：FP16 tensorcore/rewrite 单点探索

- 路径：`stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_tensorcore_rewrite_20260629*` 和相关 `exports/fp16_lhc07_*`
- 覆盖：主要围绕 lhc_07 等少量配置。
- 背景：排查 FP16 rewrite、tensorcore、group conv、full engine 相关问题。
- 可信度：仅调试证据。用于解释为什么某些 tuned/default 速度不匹配，不进入广覆盖 LUT。

### 2026-06-29：original60_quant_20260629 临时目录

- 路径：`stage2_lut_generation_v1/generated/original60_quant_20260629/`
- 覆盖：少量 FP16 tensorcore rewrite raw。
- 背景：临时/误分支式探索目录。
- 可信度：仅调试证据。优先看 `original60_quant_20260627` 主工作根。

### 2026-06-30：FP16 rewritten AP bridge

- 路径：`stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_rewritten_original60_ap_rows_v1.jsonl`
- 覆盖：64 条，其中 60 条 full-val、4 条 smoke；full-val 覆盖 original60 全部 60 个 label。
- 背景：验证并批量运行 rewritten full-engine AP bridge，解决 FP16 AP-shape/group conv rewrite 之后的 full-val AP 评估链路。
- 可信度：full-val 行可直接作为 FP16 AP 主表来源；smoke 行仅调试证据。注意该行表中的 `rewritten_latency_ms` 不是总表 latency，不应混入 latency 汇总。
- 回传状态：2026-07-01 已从 H800 回传到 4090 本地同一路径，并合入 `original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.{json,csv,md}` 以及根目录摘要副本。

### 2026-06-30：INT8 AP52 备份工作目录

- 路径：`stage2_lut_generation_v1/generated/original60_quant_20260627_backup_work_20260630_int8_ap52/`
- 覆盖：已迁回本机；`rows/native_int8_original60_ap_rows_v1.jsonl` 52 条 / 52 个 label；同时复制了 rows/jobs/exports/raw_refs 等轻量产物；这 52 条 AP 已合入主工作根 summary。
- 背景：H800 上补 exact-label checkpoint 并生成 AP 评估前置资源后，得到 native INT8 AP52；为了不动主表，先迁回备份目录。
- 可信度：可直接使用于 INT8 AP 分析，但有边界：不是全网络 INT8 claim，通常仍包含 PyTorch FP32 head/postprocess；大型 raw full-val 证据仍在 H800 原始目录。

关键文件：

- `BACKUP_MANIFEST.json`：说明备份来源、52 条 AP 行数和 raw 引用。
- `README_DO_NOT_TOUCH_SOURCE_MAIN_TABLES.md`：明确不要直接覆盖主表。
- `raw_refs/int8_ap_bulk_run_dir.txt`：记录 H800 原始 raw 目录引用。
- `exports/original60_quant_three_metric_summary_latest.*`：备份目录内仍是 2026-06-29 23:41 旧 summary 副本；已更新的是主工作根 `original60_quant_20260627/exports/*latest*`。

## 快速使用建议

- 做当前 H800 + TVM 性能预测器：优先从 `original60_quant_20260627` 和 `original60_quant_20260627_backup_work_20260630_int8_ap52` 中抽取带 provenance 的行表。
- 做历史对照：使用 `existing_h800_tvm/` 和 `dataset_v2.csv`，但不要和当前 H800 + TVM 新测数据混为同一分布。
- 做 AP 链路问题复盘：看 `ap_stability_20260626/`、`quant_smoke_20260627/`、`fp16_lhc07_*` exports。
- 做主表刷新：先在 backup/work root 生成新 summary，再人工审查，不直接覆盖 `original60_quant_20260627/exports/*latest*`。

## 2026-07-01 当前数据空白和可信度审查

最新总表路径：

- 权威 JSON：`stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.json`
- 权威 CSV/MD：`stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.{csv,md}`
- 审阅副本：`${V2X_ROOT}/multi_agent/data/original60_quant_three_metric_summary_latest.md`

覆盖统计：

- latency：FP32 58/60，FP16 60/60，INT8 60/60。
- energy：FP32 60/60，FP16 60/60，INT8 60/60。
- AP：FP32 5/60，FP16 60/60，INT8 52/60。

尚未测量或 no_claim：

- FP32 latency 缺 2 个 label：`lhc_17`、`s2_096`。
- FP32 AP 缺 55 个 label；当前只有 5 个 measured，且说明为 Stage2 original60 AP source map 的 full-model reference AP，不是新执行的 precision-specific TVM FP32 AP。
- INT8 AP 缺 8 个 label：`frontier_16`、`frontier_18`、`frontier_25`、`frontier_26`、`frontier_27`、`frontier_31`、`s1_112`、`s2_096`。

需要谨慎使用或继续审计：

- FP16 latency/energy 仍来自早期 true-fp16 smoke/energy 链路，虽然覆盖 60/60，但和 2026-07-01 的 rewritten AP full-val 链路不是同一条测量链路。
- INT8 latency/energy 覆盖 60/60，但 AP 仍只有 52/60；训练时不能把 8 个 AP no_claim 行当作真实 0。
- energy 三组虽然均为 60/60，但功耗数据对 GPU 空闲、窗口切分、后台占用敏感；使用时必须保留 `energy_schedule_policy` 和 measurement source。
- 总表的 latency/energy 是 backbone/subnet TVM 编译模块指标，不是完整业务端到端流水线指标；AP 是评估链路输出指标。
