# 交接:完整 rebuild + 出图(下个会话 team-lead 亲自做)

> 写于 2026-06-03 19:35。背景:4-agent 团队进程已死(~17:03 后无活动,僵尸进程),数据全在盘上但最新 6 类结果未 wire 进 build 脚本。本会话 token 耗尽,改下会话由 main 亲自做。**不重启团队。**

## 环境
- python: `/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python`(默认 python3 无 pandas)
- cwd: `/home/jichengzhi/UniV2X`
- build 脚本: `multi_agent/data/build_dataset_v2.py`(能跑,现产 **45 行**,只整合到 P0-1)
- 输出: `multi_agent/data/dataset_v2.{csv,parquet}`
- figure: `multi_agent/figure/make_dataset_pareto_coverage.py`(主 Pareto/coverage 图)+ `make_data_figures.py`

## 任务:把这 6 类新结果 wire 进 build_dataset_v2.py 再重跑 + 重出图

| # | 新结果文件(已验真实存在) | 贡献 | 口径/纪律(务必标对) |
|---|---|---|---|
| 1 | `results/p0_2_136_ap.json` + `results/P0_2_136_hw.csv` + `results/p0_2_136/p50b2_136_{int8,fp16}.json` | P0-2 [32,64,136] 完整点 | latency_kind=`body_subnet_collab2`;**stage2=136 非对齐但无 cliff(INT8 1.34×)** |
| 2 | `results/E6_orin_p03_lat_energy.csv` + `results/P0_3_orin_output_match.csv` | Orin FP16 3 跨硬件完整点 + 能耗 | hardware=orin_agx;能耗 **module-total(VIN_SYS_5V0)对 4090 整板**,非 GPU-rail;INT8 ap_valid=False(透明标);nvpmodel 限 30W |
| 3 | `results/pathA_forced_int8_ap.json` + `results/P_ab_hw_bench.csv` | A 剪枝×forced-INT8 消融 | **regime=ablation_guardrail_off,非前沿**;cliff系≠backbone系不拼曲线;禁 2dp-vs-4dp 假象 |
| 4 | `results/pathB_head_int8_ablation.json` | B 护栏消融 | regime=ablation;head-INT8 不崩(Δap70 噪声内)+ 不省延迟 |
| 5 | `results/pathE_distance_binned_ap.json` | E 难子集**负结果** | 诚实记 floor(spread 随距离收缩非放大);非前沿 trade-off |
| 6 | `results/P12_collab2_request_batch.csv`(+ `P12_subnet_batch_sweep.csv` 仅脚注) | #12 collab2 多流 throughput | collab2 多流 **1.106×**(SM@1=94% 饱和)→ **throughput 塌回 1/lat → 主前沿诚实 2D**;subnet 标 `throughput_kind=subnet_capability` **不进主 Pareto** |

## 主线结论(已定,出图按此)
- 主前沿 = **(latency, energy) 2D Pareto + 耦合陷阱**;AP/throughput/size 作约束。throughput 不硬撑 3D。
- **耦合陷阱 scope(精确,勿过度泛化)**: kernel-cliff 限 **grouped-conv(3×3 g32)在非 tile 对齐宽度**(p25 stage0=48 / prune90 stage1=13 实证 + per-layer 10.9×);**1×1 conv 非对齐不触发**(stage2=136 1.34× 反证)。
- AP 近常数 = **阶段性结论**(Pyramid/DAIR 过参数化);V2X-ViT/换模型 = 下阶段(Task#13,待讨论)。

## 要出的图
1. **2D 成本 Pareto 主图** (latency, energy),点按 precision/prune 着色,AP 作约束维度标注,耦合陷阱点(p25)高亮。
2. **耦合陷阱图**:对齐档(base/p50/p75)INT8 加速 1.25-1.57× vs p25 1.06× / prune90,grouped-conv scope 注。
3. **跨硬件**:4090 vs Orin FP16 完整点(能耗 module-total 对整板)。
4. coverage/stocktake 更新(完整点数、有效独立锚点≈6-8 声明)。

## 出图后核验(自己当 supervisor)
- 口径隔离:body_subnet_collab2 / engine_board_energy / orin / subnet_capability **不混进同一 Pareto**。
- 真测/估算/复用标注:Orin AP 复用 4090 标 ap_reuse_basis;forced 标 ablation;E 标负结果。
- 读原始 csv/json 抽样核对数值,别只信 build 脚本。

## 事实源(续接读这些)
`team_charter_v1.md` · `issues_log_v1.md`(ISS-001~019)· `session_report_2026-06-03_teamlead.md` · `background/00_*` · `ap_activation_strategy_v1.md`
