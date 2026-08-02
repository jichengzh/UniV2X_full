# 6_7_14 口径审计 — Pyramid vs CoDriving backbone latency 为何差近两个数量级(测量不可比)

> 起因:用户看到 CoDriving `body_lat` 只有 0.2–0.5ms、Pyramid backbone-subnet 实测 10–54ms,差近两个数量级,要求核查两算法测量方法是否一致。
> 日期:2026-07-04。**结论:两套 latency 语料不同后端/不同设备/不同 batch,绝对值完全不可比;~100× 差是测量口径错配的假象,不是模型真实速度差。** 所有数字均主控亲自读 raw 文件核实(附 `文件:行号`)。
> ★[本文纠正]:先前口头把差异归因于 "Pyramid 含 fusion / 双 agent"——**错**。两套语料都是 backbone 主干、都不含 fusion。真因见 §3。

---

## 1. TL;DR

| 问题 | 结论 |
|---|---|
| 两者测量方法一致吗? | **否**。Pyramid=未有效调优的 **TVM**@H800、batch=2、backbone-only;CoDriving=优化过的 **TRT**@4090、batch=1、backbone+heads。 |
| 100× 差来自哪? | **后端(未优化 TVM vs 优化 TRT)为主因(~100×)** + batch 2vs1(~2×)。设备/输入/scope 三项都指向"Pyramid 本应更快",说明真实后端 gap 比 100× 还大。 |
| 跟 fusion / 双 agent 有关吗? | **无关**。两者都是 backbone 主干、都不含 fusion。(Pyramid 是 batch=2 但只贡献 ~2×,不是 100×。) |
| 两个 backbone 真实速度差多少? | **同量级**。同后端(eager PyTorch)下:CoDriving backbone 3.63ms vs Pyramid backbone 4.49ms(FP32)/3.27ms(FP16)。 |

---

## 2. 七维逐项对照(每格带 `文件:行号` 证据)

| 维度 | **Pyramid original60**(10–54ms) | **CoDriving 8-point body**(0.2–0.5ms) |
|---|---|---|
| **后端** | **TVM Relax**。`command.json` `tvm_strategy=relax_default`(default)/ `relax_metaschedule_reuse_existing_ms_db`(tuned),notes "**no fresh tune**";worker `scripts/stage2_h800_run_measurement_job.py:380` `relax.build(target="cuda")`、`:407` MetaScheduleApplyDatabase | **TensorRT**。`output/codriving_pilot/bench_pareto_axes.py:43` `deserialize_cuda_engine`、`:81` `execute_async_v3`;csv 注 `codriving_dair_grid_8point_clean.csv:14` "TRT 0.47/0.27/0.28" |
| **设备** | **H800**。raw `nvidia_smi_preflight.csv` 全 `NVIDIA H800 81559 MiB`,host `zs-nj-tap-gpu18` | **RTX 4090**。`codriving_dair_grid_8point_clean.csv:17` "GPU idle 25W";pilot host 4090 |
| **测量 scope** | **backbone_only**(无 VFE/scatter/fusion/**head**/NMS)。`command.json` `optimized_scope=backbone_only`,`software_point_id=pyramid_lidar:backbone:...:fp16` | **backbone→cls/reg heads**(无 voxelize/scatter/fusion/NMS)。`bench_pareto_axes.py:7` `spatial_features -> cls_pred + reg_pred; excludes voxelize/scatter/fusion/NMS` ⇒ **CoDriving 比 Pyramid 多算 heads,却更快** |
| **输入张量** | **`spatial_features [2,64,128,256]`(batch=2)**。raw `latency_result.json` `input_shape`。`width` 列(如 "64,48,256")= **剪枝 num_filters 标签,非输入形状**;`in_name` 同为标签 | **`(1,64,256,512)`(batch=1)**。`bench_pareto_axes.py:34` `INPUT_SHAPE=(1,64,256,512)` |
| **精度** | fp16(`quant_policy=fp16`),仅此一档 | fp16 **与** int8 两档(`codriving_dair_grid_8point_clean.csv:2-9`) |
| **计时** | **TVM `time_evaluator`**,`stage2_h800_run_measurement_job.py:205` `number=100,warmup=30,repeat=3` | **CUDA Event**,`bench_pareto_axes.py:35` `WARMUP=300,RUNS=300` |
| **调优状态★** | **tuned≈default = 未有效优化**。s1_048 `default_us=44294.9` vs `tuned_us=44452.8`(**tuned 反而略慢,ratio 1.004**);reuse_existing_ms_db 复用残缺 db,对这些算子近乎无效(撞已知假 ratio 现象) | TRT builder 已优化(engine 16.04MB fp16 / 9.06MB int8) |

---

## 3. 反常点(输入更大却快 100×)根因排序

| 排序 | 根因 | 强度 | 依据 |
|---|---|---|---|
| **1** | **后端:未优化 TVM(tuned≈default)vs 优化 TRT** | **实锤** | tuned/default ratio≈1.0(44294 vs 44452 μs);两侧后端字段直读;§4 eager 锚点证明模型本身同量级 |
| 2 | **batch 2 vs 1**(Pyramid 一次算 2 agent) | **实锤** | `input_shape [2,64,128,256]` vs `INPUT_SHAPE (1,64,256,512)` |
| 3 | 设备 H800 vs 4090 —— **排除**(H800 更强,方向相反,不解释 Pyramid 更慢) | 实锤(排除) | preflight H800 vs csv 4090 |
| 4 | 输入分辨率 —— **方向相反**(CoDriving 256×512 反而 4× 更多像素) | 实锤(排除) | 两侧 input shape |
| 5 | scope —— **方向相反**(CoDriving 多算 heads) | 实锤(排除) | scope 两侧证据 |
| 6 | 计时 bug / μs-ms 混淆 —— **排除** | 推断(强) | raw 是 μs,汇总正确 ÷1000;两侧 warmup/repeat 规范 |
| — | fusion / 双 agent 协同 —— **无关**(先前口头误判,已纠) | 实锤(排除) | 两侧 scope 均 excludes fusion |

---

## 4. ★决定性锚点:同后端下两个 backbone 同量级

`codriving_dair_grid_8point_clean.csv:14`:CoDriving 隔离 **eager PyTorch backbone** base=**3.63ms**(p50 3.10 / p75 2.32),TRT 后 0.47ms(1.7×)。
Pyramid backbone **eager PyTorch**(CLAUDE.md §A.4):**FP32 4.49ms / FP16 3.27ms**。

⇒ **同一后端(eager)下:CoDriving 3.63ms ≈ Pyramid 4.49ms(FP32)/3.27ms(FP16),同量级。** 那 ~100× 差 100% 来自"Pyramid 用未优化 TVM-H800-batch2 报数、CoDriving 用优化 TRT-4090-batch1 报数"的口径错配,**与两个模型真实算力无关**。

---

## 5. 对已有结论的影响

1. **P1 "latency vs 剪枝率 ρ=−0.795 / 5.2× 跨度"仍成立(方向),但绝对值是未优化-TVM 口径**:剪枝→更少 FLOPs→更快 的单调关系与后端优化程度无关,故 ρ 结论稳;但 9.8–54.4ms 这些绝对数是 **relax_default/reuse_ms_db 口径**(tuned≈default),不代表框架能达到的优化后延迟。真正的 TVM 调优增益(pad64 7.71× 等)在别的 fresh-tune 语料里,不在 original60 这张表。见 [[feedback-tvm-tune-apply-fresh-workdir]]。
2. **CoDriving int8/fp16 body 加速(base 1.61×/2.21× energy)是 TRT-4090-单核口径**,占全网 wall-clock <5%,兑现不到 e2e(全网 fp16 28.75ms≈int8 29.15ms,CPU/launch/copy-bound;csv:16)。见 `5_7_14` §诚实遗留。
3. **禁止把 original60 的 ms 数直接与 CoDriving body 的 ms 数并列比较**——不同后端/设备/batch。data/README 已有 "body_subnet 与 collab2 不混比" 的类似告诫,本文补一条:**"TVM-original60 与 TRT-codriving-body 不混比"**。

---

## 6. 要做真公平的 Pyramid-vs-CoDriving backbone latency 对比,须统一

1. **同后端**:两模型都 TRT,或都 fresh-tune 的 TVM(Pyramid 必须真调优,不能 reuse_existing_ms_db)。当前 TVM-untuned vs TRT-optimized 对比无意义。
2. **同设备**:同一张空闲卡(util0%/mem≤50MiB)。
3. **同 batch**:都 batch=1(把 Pyramid [2,...] 改 [1,...])或都 N=2。
4. **同 scope 边界**:都到 backbone 输出,或都 backbone+heads(CoDriving 现含 heads,Pyramid 纯 backbone,须对齐)。
5. **同输入分辨率口径**:若比"同规模 backbone",对齐 feature map 尺寸/FLOPs。
6. **同计时协议**:同 timer + 同 warmup/repeat。
7. **最小重测项**:Pyramid backbone 在 **TRT/tuned-TVM + 同卡 + batch1 + fp16** 下重测,与 CoDriving base fp16 0.47ms 直接对齐,预期落亚毫秒~数毫秒,而非当前 21–54ms。

> ★注:框架前进口径是 **TVM**(严禁 TRT 作前进方向,见 [[feedback-framework-on-tvm-not-trt]])。故公平对比的正解 = **两模型都走 fresh-tune 的 TVM**(Pyramid 重跑真 metaschedule tune、CoDriving 也上 TVM),而不是都退回 TRT。CoDriving 的 TVM 迁移已打通(backbone 2.07–2.26×,见 [[project-codriving-tvm-migration]]),具备同口径重测条件。

---

## 7. 关联

- 收尾主文档 [[5_7_14 三线P0P1全闭合]];精度轴 fair 前置 [[project-precision-axis-double-unfairness]]。
- TVM tune 假 ratio 坑 [[feedback-tvm-tune-apply-fresh-workdir]];框架 TVM 口径 [[feedback-framework-on-tvm-not-trt]]。
- CoDriving 瓶颈归因 [[project-codriving-optimization-pilot]]、TVM 迁移 [[project-codriving-tvm-migration]];Pyramid latency ceiling [[project_f_lat_ceiling]]。

---

### 证据文件清单(可复核)
- Pyramid:`multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/raw/original60_latency/*/{command,latency_result,measurement_payload_*}.json`、`scripts/stage2_h800_run_measurement_job.py:205/380/407`、汇总 `exports/original60_three_metric_summary_latest.{csv,md}`
- CoDriving:`output/codriving_pilot/bench_pareto_axes.py:7/34/43/58-59/81`、`results/codriving_dair_grid_8point_clean.csv:13-18`、`output/codriving_pilot/build_trt_engines.py:28-29`
