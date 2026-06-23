# multi_agent/data — 有效数据 (2026-06-01; ★2026-06-12 双表 + 维度精简 + 闭环列)

## ★ 两张权威表
| 文件 | 形状 | 用途 |
|---|---|---|
| **`dataset_v2.{csv,parquet}`** | 65 行 × **93 列** | ★**最全面统计源**: 全部开环指标 + 全部闭环 23 列; 下游图/校验/溯源读它 |
| **`dataset_v2_learning.{csv,parquet}`** | 65 行 × **65 列** | 精简学习视图 = 全表 − 28 列(8 开环冗余 + 20 闭环非核心)+ 剔 norsu 行; 给预测器/选择器学习 |

- **开环维度精简**(每维度只留一个规范值, 学习视图): 精度`ap70` / 延迟`lat_p50_ms` / 吞吐`throughput_fps` / 能耗`energy_per_frame_mj` / 体积`engine_size_mb`; 另留约束信号 `mATE/mASE/mAOE` + 分组列 `latency_kind`/`throughput_kind`/`model_class`/`regime`。详见 `schema_v2.md §双表结构`。
- **闭环驾驶 23 列全 wire 进全表**(`cl_driving_score`=DS 等 15 性能 + 8 元数据): 现 0 行(`CL_INGEST=False` gated, pilot DS 全平未达 GO 门), **时延补全后**(tau-ego 大路线库扫延迟)接入。`model_class=codriving_v2xverse`, **严禁与感知行混 Pareto/同曲线**。
- **学习视图闭环只留 `cl_driving_score`(DS) + `cl_route_completion`(RC)**(+`latency_inject_ms` 作 τ 特征), **剔 norsu 行**(单车 ego-only 消融基线只在全表作 V2X 收益对照, 部署永远有 RSU)。
- Schema 定义: `schema_v2.md`;构建脚本: `build_dataset_v2.py`(可复跑,加新数据改它重建; `ROOT=/home/jichengzhi/V2X`)。
- ⚠️ **4 种 latency 口径不可混比**(`body_subnet_collab2`/`engine_board_energy`/`body_subnet_collab2_orin`/`dla_pipeline_e2e_single_frame`)。训练/比较前按 `latency_kind` 分组。
- 训感知预测器筛: `df[(df.model_class=='pyramid_fusion') & df.lat_p50_ms.notna() & df.ap70.notna()]`。

> 历史说明: 早先按 `tier1_complete/`(完整点)+`tier2_single_axis/`(单轴)分层。**latency 补齐后 27 个单轴 AP 点升级为完整点, 分层已失效** → 合并进唯一权威表 `dataset_v2`。原始分层文件全部降级为**构建输入**, 收进 `sources/`。

## 目录结构
```
multi_agent/data/
├── dataset_v2.{csv,parquet}            ★ 全表 (65 行 × 93 列, 含全冗余指标 + cl_闭环列) — 构建输出
├── dataset_v2_learning.{csv,parquet}   ★ 精简学习视图 (65 行 × 65 列, 开环每维度1规范值+闭环DS+RC) — 构建输出
├── schema_v2.md                        列定义 + 双表结构 + 维度精简 + cl_闭环列说明
├── build_dataset_v2.py                 构建脚本 (读 sources/ + results/ → 两张表; LEARN_DROP 控精简; CL_INGEST 控闭环接入)
├── 数据说明_v1.md                      早期 6 点视图的指标词典 (列含义仍适用; 全表以 schema_v2 为准)
└── sources/                            构建输入 (原 tier1/tier2 文件, 合并前快照, 仅供追溯/重建)
```

## sources/ 里有什么(构建输入,非最终数据)
- `complete_points_pyramid_v1.csv`(6, 单 agent body latency+AP)→ dataset_v2 中 6 个 `body_subnet` 点
- `perstage_*` 的 AP/latency 在 `results/`(被 build 脚本读)→ dataset_v2 中 27 个 `body_subnet_collab2` 点
- `stage_a_ap_real`(8 真 AP)、`doe_dataset_v1_real`(7 真 lat)、`perstage_quant_bench`(18 lat)、`4090/orin_dspace`、`cudagraph`、`pyramid_random`(未 finetune)、`pyramid_fusion_4090`、`baseline_4090`(3 模型,禁跨模型拼)—— 这些是各维度的原始 bench,部分已被合并进 dataset_v2,其余作扩展时的素材。

## 引用纪律(摘自档案 §3)
1. 延迟数必标口径/数据源;subnet ≠ e2e;`body_subnet` 与 `body_subnet_collab2` 不混比。
2. "某方案无价值"结论须 **延迟+AP 双轴**都有数据才能下(perstage 是反例)。
3. 未 finetune 子网 AP 不作有效数据;subnet/buggy AP 高估 0.10-0.16。
4. `baseline_4090` 必须按 `model_class` 拆分使用。
5. **[ISS-026 修正 2026-06-04]** `baseline_4090.csv` 的 `amota` 列:
   - **pyramid_fusion (26行)**: 无跟踪头,无 AMOTA。原 amota 列存的是 **AP50(检测指标,已错填)**。已修正: AP 值移到新增 `ap50` 列,`amota` 全部置 NaN。
   - **univ2x_full / uniad_tiny_variant**: 真实 AMOTA(跟踪指标,plan_b 真测),`amota` 列保留不动。
   - ⚠️ 引用 `baseline_4090` 精度时: pyramid 用 `ap50`; univ2x/uniad 用 `amota`。勿混用。
