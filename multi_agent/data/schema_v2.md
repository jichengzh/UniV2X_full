# dataset_v2 统一 Schema (v2, 2026-06-01; 2026-06-03 加能耗+吞吐口径列; 2026-06-03 晚 wire 6 类新结果; 2026-06-05 +V2X-ViT A-1/A-2 3行 + 补录 dair_real 分段计时第4行; **★2026-06-12 wire 闭环 cl_ 列(70→93, 最全面) + 另出精简学习视图**)

> 文件: `multi_agent/data/dataset_v2.{csv,parquet}` (**65 行 × 93 列**; ★2026-06-12 wire 闭环驾驶 23 列(8 元数据 + 15 `cl_`, 最全面统计源), 见文末 §"[ACTIVE] 闭环驾驶指标列")

## ★ 2026-06-12 双表结构 + 评价维度精简 (供后续模型学习)
> **两张表, 分工明确**:
> - **`dataset_v2.{csv,parquet}` (全表, 93 列)** = 唯一权威事实源, **保留全部冗余指标列**(ap30/50/70 三档、lat p50/p99/mean 三口径、能耗三表达、engine_size+params 等)。下游图/校验/溯源仍读它, 零破坏。
> - **`dataset_v2_learning.{csv,parquet}` (精简学习视图, 65 列)** = 全表 **减 28 列**(8 开环冗余指标 + 20 闭环非核心列)+ **剔 norsu 行**, 给预测器/选择器学习用。开环**每个评价维度只留一个规范值**, 闭环**只留 DS + cl_route_completion** (+`latency_inject_ms` 作 τ 特征):
>
> | 评价维度 | 全表候选 | **学习视图保留(规范值)** | 选择依据 |
> |---|---|---|---|
> | 精度 | ap30/ap50/**ap70** | **`ap70`** ↑ | CV 0.085 信号最强; ap30/50 近饱和(CV 0.025/0.035) |
> | 延迟 | **lat_p50_ms**/p99/mean | **`lat_p50_ms`** ↓ (+`latency_kind` 分组) | 满覆盖 62/65 + 抗尾部; 三口径相关≈1.000 冗余 |
> | 吞吐 | throughput_fps | **`throughput_fps`** ↑ (+`throughput_kind` 分组) | 唯一列; kind 标退化(inv_latency)vs 真吞吐(batched/pipelined) |
> | 能耗 | power/**perframe**/perfwatt | **`energy_per_frame_mj`** ↓ | 每帧真成本; perfwatt 混吞吐(重复计数), power 缺时长 |
> | 体积 | **engine_size_mb**/params_total | **`engine_size_mb`** ↓ | 真部署约束+反映量化降字节; params 覆盖仅 9/65 且对量化不敏感 |
>
> **学习视图删除的 8 列**: `ap30, ap50, lat_p99_ms, lat_mean_ms, mean_power_w, perf_per_watt_fps_per_w, params_total, delta_ap50_vs_baseline`(原始值仍在全表+sources, 改 `build_dataset_v2.py` 的 `LEARN_DROP` 可调)。
> **学习视图保留的非主目标信号**: `mATE/mASE/mAOE(+CI+n_tp)` 6 列 = 剪枝退化**约束信号**(标 regime, 非前沿目标轴, 用户拍板保留)。
> **学习视图闭环**: 只 `cl_driving_score`(DS) + `cl_route_completion`(RC) 两目标 + `latency_inject_ms`(τ 特征); 其余 20 闭环列(碰撞分解/违规/ZOH/诊断/元数据)仅全表有; norsu 行已剔(只学 RSU-present 主曲线)。
> **分组列必带**: `latency_kind`(4 种 scope 不可混比)/ `throughput_kind`(退化 vs 真吞吐)/ `model_class` + `regime`(感知 vs 闭环不可混)。学习前必按这几列分组/过滤。

> 历史: 早先文件为 **65 行 × 70 列**, 2026-06-05 加 V2X-ViT A-1/A-2 3 行 + 补录 dair_real 分段计时第4行, 见文末 §"2026-06-05 V2X-ViT 接入"
> 构建: `multi_agent/data/build_dataset_v2.py`(可复跑)
> 合并源: `complete_points_pyramid_v1.csv`(6, latency+AP, 单值精度) + `perstage_quant_AP_real_v2.csv`(27, per-stage 精度, AP-only) + **`E4_energy_4090.csv`(13, NVML board 能耗 + batch 吞吐)** + **`E3_orin_dla_pipeline.csv`(2, 异构 DLA 流水解耦吞吐)** + **`P0_1_p25_int8_trap_4090.csv`(2, p25 INT8 auto 完整点 + forced 消融点; AP 来自 stage_a/forced-json)** + **`v2xvit_baseline_a1.json` + `v2xvit_a2_p50.json` + `v2xvit_a2_p75.json`(3, V2X-ViT AP-only, Task#4)** + **`v2xvit_dair_real.json`(1, V2X-ViT base PyTorch hook 分段计时, latency_kind=forward_hook_pytorch_fp32, Task#4第4行)**
> Pareto 5 指标定义见 `multi_agent/methods/design/pareto_definition_v1.md`。
> 设计原则: **扁平列只放当前在测维度**; 跨模型/更多模块的结构靠 `config_json` 承载, **不预留空列**(避免 v1 那种大面积空列)。每行自带 `latency_kind`/`throughput_kind`/`ap_valid`/`is_real_measured` 标真伪与口径, **不强行 join 不同测量**(E4/E3 作为新行追加, 非并到 collab2-body 行)。

## 相对 v1 的扩展(回应"列维度需扩展")
| 新增能力 | 新列 | 解决的旧缺口 |
|---|---|---|
| **per-stage 混精** | `stage0_prec / stage1_prec / stage2_prec` + `q_mode` | v1 只有单值 `q_bits`, 表达不了"stage0 INT8 / stage1 FP16" |
| **全网络剪枝** | `deblocks_keep` / `shrink_keep`(+ config_json 可扩更多模块) | v1 只有 backbone `stageX_planes` |
| **口径/有效性** | `latency_kind`(body_subnet/e2e/NA)、`ap_valid`、`finetune_epochs`、`ckpt_status` | v1 没标 body≠e2e、没标 AP 是否 finetune 有效 |
| **公平对比** | `ap_baseline_ref` + `delta_ap50_vs_baseline` | v1 delta 对不可比基线算 |
| **跨模型可扩** | `model_class` + `config_json`(结构化 Config 引用) | v1 Pyramid-3stage 写死 |

## 列分组 (按重要性排序, 诊断列在后)
**标识**: `config_id`(模型_planes_精度签名) · `model_class` · `dataset_src`(来源) · `config_label` · `triplet`
**剪枝 B1**: `prune_rate` · `prune_object`(channel/none) · `prune_criterion` · `stage0/1/2_planes` · `deblocks_keep` · `shrink_keep` · `params_total` · `finetune_epochs` · `ckpt_status`(pretrained / pruned_finetuned)
**量化 B2**: `stage0/1/2_prec`(FP16/INT8/FP32) · `q_mode`(uniform / per_stage_mixed / uniform_forced / auto) · `q_granularity` · `q_object` · `calibrator`
**硬件 D**: `hardware` · `d_scheme` · `d_tactic` · `d_workspace_gb`
**标签·latency**: `latency_kind`(★body_subnet_collab2=双agent模型体 / engine_board_energy=E4 / dla_pipeline_e2e_single_frame=E3 / NA_pending_idle_gpu=待空闲GPU测 / **forward_hook_pytorch_fp32=PyTorch_FP32 hook 级分段计时(≠collab2 ≠e2e_trt, 不与任何TRT行比较)**; **不同口径不可混比**) · `lat_p50_ms` · `lat_p99_ms` · `lat_mean_ms`
**标签·throughput** (★非冗余轴): `throughput_fps` · `throughput_kind`(`inv_latency`=单流batch1即1000/lat / `batched`=batch>1真批处理 / `pipelined`=跨帧流水, 后两者脱离1/lat) · `batch`
**标签·accuracy**: `ap30/50/70` · `n_ap_samples` · `ap_pipeline`(DAIR_val_1789_TRT) · `ap_baseline_ref`(delta 对谁算) · `delta_ap50_vs_baseline`
**标签·TP几何误差** (ISS-020/024/029 终核 PASS, 2026-06-04; supervisor step⑦ 复核通过, 留痕 issues_log ISS-020):
> **终核数**:
> - **剪枝轴**: SNR=14.2×(全档均半宽口径; 备用13.1×avg-half/17.1×pooled-SE); base→p75 Δ=+0.0244 rad。**信号在全剪枝程(base↔p75)可分辨; 相邻细档 p25↔p50 CI 重叠, 不可分辨**。
> - **INT8 量化轴**: p25/p50/p75 SNR=0.58/0.05/0.82×(噪声级, CI重叠, 跨档非单调变号) → **无量化轴信号**; base档 Δ+0.0030 在全集 CI 可分辨但非单调+量级仅5%, 按 ISS-020#4 **不构成轴信号**。勿写"全部噪声级" — base档 CI 可分辨但不满足轴信号判据。
> - **n_tp 修正归因(ISS-024)**: 权重错配修复使检测质量恢复 → pruned 6行 n_tp +112~+851; base 2行微变(fp16: -6 / int8: +21)。与幸存者偏差是两个不同概念。
> - **⚠️ pruned50 INT8 幸存者偏差(修正后仍存在)**: n_tp 25577 vs FP16 26154 (少 577), mAOE 不可直接与 FP16 比。
> - **⚠️ ISS-029 pruned50 AP70 口径**: eval_script 两精度同向偏 stage_a gold +0.009 (fp16: 0.5730 vs 0.5641; int8: 0.5636 vs 0.5542); 其余 6 行 ≤0.002。AP 列保留 stage_a 金标准值, 须显式标来源差异 caveat。
`mATE`(m, 越小越好) · `mASE`(0-1 无量纲) · `mAOE`(rad [0,π/2], 越小越好) · `mAOE_n_tp`(TP 样本数, 诊断幸存者偏差) · `metric_pipeline`(口径+脚本)
> **mAOE 定位 (ISS-020/024/029 终核 PASS)**: mAOE 是**预测器约束信号**(标 regime), **不是前沿级 trade-off 目标轴**。**对剪枝轴有信号** (SNR=14.2×, base→p75 Δ=+0.0244; 全程可分辨, **p25↔p50 相邻细档 CI 重叠不可分辨**); **对 INT8 量化轴**: p25/p50/p75 SNR=0.58/0.05/0.82×(噪声级, CI重叠, 非单调) → 无量化轴信号; base档 Δ+0.0030 CI可分辨但非单调+量级5%, 按 ISS-020#4 不构成轴信号。**禁止用 mAOE 画成本 Pareto 图, 禁止引用"×AP70"判据**(两噪声相除假精确, 禁)。
> **口径**: DAIR val 1789, stage_a_cache 引擎, 1618 TRT collab+171 PyTorch fallback; IoU=0.5 TP 匹配; `scripts/phase2/eval_tp_errors_corrected_full.py`(ISS-024修正版)。仅填入 uniform prec + ap_valid=True + q_mode 无 forced 的行(25/61 行)。⚠️ pruned50-INT8 幸存者偏差(n_tp-577: INT8 25577 vs FP16 26154, 差 2.2%; 其余档 int8-fp16 gap≤112, 仅 p50 须 caveat)。⚠️ ISS-029 pruned50 AP70 口径差(eval偏gold+0.009两精度同向; 其余6行≤0.002; AP列保留stage_a)。
**标签·energy** (★Pareto 第5轴, NVML board 实测): `mean_power_w` · `energy_per_frame_mj`(mJ/frame) · `perf_per_watt_fps_per_w`(fps/W)
**标签·其它**: `engine_size_mb` · `fp16_layer_count` · `int8_layer_count` · `build_secs`
**Pareto regime** (★ISS-013): `regime` —— `front`(**可部署候选池**: uniform/auto)/ `ablation_guardrail_off`(forced-all-int8, 护栏 OFF, 被 auto 双轴支配, 仅预测器训练+消融)/ `ablation_perstage_dominated`(per-stage 混精, 被 TRT-auto 支配)。
> ★**语义澄清(supervisor, 防误读)**: `front` = **可部署候选池**(已排除 ablation), **≠ 字面 Pareto 前沿点**。**真前沿 = front 的非支配子集, 由 selector(NSGA-II)计算**。front 内可含被支配候选 —— 例 `p25_int8_automix`(lat 2.733/ap70 0.584)标 front, 但被 base int8(0.811/0.623)双轴支配, selector 会自动排出真前沿。**这恰是框架正确工作(自动避开非对齐陷阱)的证据, 应留在候选池**。⇒ 别把 front 行数当前沿点数。
**AP 溯源强度** (★ISS-017, 机器可读): `ap_reuse_basis` —— `independent_measured`(本 config 逐 engine 独立真测, 22 行)/ `stage_a_gold_exact_reuse`(uniform 配置 AP 取自 4090 stage_a 金标准同 config, 8 行)/ `core_exact_reuse_4090`(E4/E3 跨 scope 借 4090 同 config core AP, 11 行)/ `fp16_crossplatform_exact`(Orin FP16 借 4090 FP16 数学一致, 1 行)/ `orin_int8_gate1_scale_contract[+spotcheck]`(Orin INT8 双闸门, 待 Orin 批)/ `none`(无 AP, 3 行)。**训练/画图前可一眼区分独立测 vs 各级复用**(对齐 ISS-010: 22 独立 + 8 金标准复用 + 11 跨scope复用)。
**有效性/溯源**: `is_real_measured` · `ap_valid` · `source`(逐标签来源) · `exec_path` · `ts` · `onnx_path` · `engine_path` · `config_json`
**末尾**: `notes`

## ★ 2026-06-03 P0-1 更新 (43→45 行)
- **新增 2 行 P0-1 p25(48/96/192)INT8 耦合点**(填平 B1×B2 耦合矩阵唯一无 INT8 的 prune0.25 档):
  - `p25_int8_automix`(regime=front, 完整点): lat 2.733ms / ap70 0.5841(stage_a 金标准同引擎)/ energy 877.6mJ / 11.5MB。INT8 加速仅 **1.06×**(对齐档 1.25-1.57×)= 耦合陷阱(非对齐 48 通道 kernel-selection cliff, profile 实证, 非 padding/reformat)。
  - `p25_c_all_int8_forced`(regime=ablation_guardrail_off, 消融): lat 4.322ms(0.67×)/ ap70 0.5804。
- **加 `regime` 列**(ISS-013): front 23 / ablation_perstage_dominated 18 / ablation_guardrail_off 4。
- **修 5 行 `ap:TODO` 陈旧 source 标注** → `ap:stage_a_ap_real(gold)`(只改标注, 数值本就是金标准真值)。
- **p25 FP16 collab2 lat 覆盖** 2.946→**2.905**(hw GPU7-clean rebaseline; 旧值 noise 内)。
- **完整点 (lat+AP) = 42; 5 指标完整 = 41**。forced-int8 ΔAP70 分段: base -0.011 / **p25 -0.010(≈base, 平)** / p50 -0.018 / p75 -0.034(**非严格单调, 50% 后才放大**)。

## 旧内容快照 (2026-06-03 加能耗+吞吐 + AP/size 补全 + 去重后, 43 行)
- **去重**: 删了 5 个 `complete_points_v1`↔`perstage_AP_v2` 完全重复行(uniform 配置两源录两遍)。48→43。
- **三种 latency_kind(=三种测量 scope, 不可混比 Pareto)**: `body_subnet_collab2`(28, 双 agent 协同体)/ `engine_board_energy`(13, E4 **单 agent** 引擎, ≈collab2 的 0.65×)/ `dla_pipeline_e2e_single_frame`(2, E3 Orin)。
- **5 轴覆盖(非空行数, P0+P2 后)**: AP **40** / latency 43 / throughput 43 / **energy 41** / model_size **41**。唯一 config = 34。
- **★ 5 指标完整行 = 39**:**collab2 双 agent 主口径 28/28 全完整**(E5 双 agent 能耗实测填回, frame=1)+ E4 单 agent 11/13(余 2 个 [32,64,136] AP 在 finetune)。
- **能耗双口径**: `body_subnet_collab2` 用 E5(双 agent, frame=1, 比单 agent 高 1.6–1.85×)/ `engine_board_energy` 用 E4(单 agent)。两口径分别自洽, **不混**。
- **能耗实测性核验留痕**: `results/E5_energy_verification.md` —— bench 记两条独立路径(公式 P̄×t vs NVML 硬件焦耳计数器 `nvmlDeviceGetTotalEnergyConsumption`), 28/28 引擎两法吻合**均值 0.40% / 最大 0.89%**, 证明真硬件实测非估算。
- **未完成**: 2 个 [32,64,136] E4 行 AP finetune 中(~5-6h); 2 个 E3 Orin 行无能耗(Orin tegrastats 未采, P3 可选)。
- **AP 补全规则**: E4/E3 行的 AP 由 core 同 `(planes, 三段精度)` **EXACT 匹配**补入(AP 由架构+权重+val集决定, 与 hw/batch/测量路径无关)。**绝不近似借用**: `[32,64,136]≠core[32,64,128]` 异架构→不补; `T1_base_FP32` core 无 FP32 AP→不补。
- **throughput_kind**: `inv_latency` 40 · `batched` 6(E4 batch=2) · `pipelined` 2(E3 Orin DLA0‖DLA1)。
- ⚠️ **不可补的缺口**: 能耗只有 13 行(E4/4090); Orin 行无能耗采集; core 33 行的能耗**不可**用 E4 能耗补(core=collab2-body 延迟口径, 与 E4 full-engine 能耗不同测量, 强补会与该行延迟不自洽)。
- ⚠️ **latency 口径不可混比**: `body_subnet_collab2`(33) / `engine_board_energy`(13, E4) / `dla_pipeline_e2e_single_frame`(2, E3)。训练/比较前按 `latency_kind` 分组。
- Pareto 结论 (per-stage 混精): AP 有正向点但被 TRT-auto 支配, 详见 `results/perstage_quant_pareto_verdict_v2.md`。

## 用法纪律
1. 训预测器筛完整点: `df[df.lat_p50_ms.notna() & df.ap50.notna()]`。
2. **body latency ≠ e2e**: 用前看 `latency_kind`; 混用前先统一口径。
3. delta 只在 `ap_baseline_ref` 相同口径内比较。
4. 跨模型/多模块的精确配置以 `config_json` 为准, 扁平 stageX 列只是 Pyramid 的去规范化视图。
5. 新维度有真数据了再加列(沿用"不预留空列"原则), 改 `build_dataset_v2.py` 重建。
6. **throughput 别当 1/latency 用**: 先看 `throughput_kind`; 只有 `batched`/`pipelined` 行才是脱离延迟的真吞吐, `inv_latency` 行的吞吐=1000/lat(欠采样退化, 非真冗余)。
7. **能耗只在 13 行 E4 上有**, 且这些行无 AP; 跑 5 轴 Pareto 前需先补"同 ckpt 同测 AP+lat+energy"的完整点。

## ★ 2026-06-03 晚 61 行更新 (45→61, wire 6 类新结果)
> 详见 `multi_agent/methods/progress/HANDOFF_rebuild_figures.md`。每类各带独立口径/regime/ap_reuse_basis, 不混 Pareto。

| dataset_src | 行 | 类型 | 口径/纪律 |
|---|---|---|---|
| `P0_2_136` | 2 | [32,64,136] 完整点(fp16+int8) | `body_subnet_collab2`; regime=front; **stage2=136 非对齐但 INT8 1.34×(对齐档 1.25-1.57× 内, 无 cliff)** → 反证耦合陷阱 scope 限 grouped-conv(3×3 g32), 1×1 conv 非对齐不触发。AP 逐 engine 4dp 真测(independent_measured)。**副效果: 同架构 AP 回填 E4/E3 的 [32,64,136] 行**(原 AP 待 finetune)。 |
| `E6_orin_p03` | 6 | Orin FP16×3 完整点 + INT8×3(仅 lat/energy) | `body_subnet_collab2_orin`(≠4090, 不混比); 能耗=**module-total VIN_SYS_5V0 整模块对 4090 整板, 非 GPU-rail**; nvpmodel 30W。FP16 AP 借 4090(fp16_crossplatform_exact, P0_3 output-match near-identical 验证); **INT8 ap_valid=False**(Orin INT8 输出 NOT match 4090, cross-TRT divergence, 透明标)。无 model_size(引擎在 Orin)。 |
| `pathA_forced_int8` | 3 | A 剪枝×forced-INT8 消融 | regime=**ablation_guardrail_off, 非前沿**; cliff系(深剪 pyramid_backbone)≠backbone系不拼曲线; **禁与 2dp fp16-ref 做 delta**(2dp-vs-4dp 假象)。cliff2_c/prune90/prune95, AP 4dp 真测。 |
| `pathB_head_int8` | 2 | B 护栏消融 head-INT8/rest-FP16 | regime=ablation_guardrail_off; head-INT8 **不崩**(Δap70_vs_fp16 在 ±0.005 噪声内)+ **不省延迟**。stage 保 FP16(`q_mode=head_int8_forced`)。 |
| `pathE`(填 `ap70_by_range`) | 4 行回填 | E 难子集距离分箱 **负结果** | 不新增行, 把 7-bin ap70 写入对应 collab2 front 行的 `ap70_by_range`(JSON)。诚实记 **spread 随距离收缩(非放大), 远距离 AP floor**; 非前沿 trade-off。 |
| `P12_collab2_throughput` | 3 | #12 collab2 多流 throughput | `throughput_kind=batched_request`, regime=**ablation_throughput_saturation**(非主 Pareto); collab2 多 context 并发峰值仅 **1.106×**(n=2, SM@1=94% 饱和)→ throughput 塌回 ~1/lat → **主前沿诚实 2D(lat,energy)**。`P12_subnet_batch_sweep.csv`=subnet_capability batch ablation, **仅脚注, 不入表**。 |

- **5 轴覆盖(非空/61)**: AP 58 / latency 61 / throughput 61 / energy 59 / model_size 50。**5 指标完整 = 50 行**。
- **regime 分布**: front 31 / ablation_perstage_dominated 18 / ablation_guardrail_off 9 / ablation_throughput_saturation 3。
- **front 有效独立锚点 = 5 个 distinct 训练架构**(planes): [16,32,64]/[32,64,128]/[32,64,136]/[48,96,192]/[64,128,256](precision/hw 是同锚点的廉价变体, 不另计)。
- **图**: `multi_agent/figure/dataset_pareto_coverage.png`(4 panel: 2D 成本 Pareto / 耦合陷阱 / 跨硬件 / coverage), 可复跑 `make_dataset_pareto_coverage.py`。

## ★ 2026-06-05 V2X-ViT A-1/A-2 接入 (61→64 行, Task #4)
> 授权: team-lead 授权消息 "team-lead 授权 data-orchestrator 执行 Task #4: V2X-ViT A-1/A-2 四行数据接入 dataset_v2"
> 事实源: `results/v2xvit_baseline_a1.json` / `v2xvit_a2_p50.json` / `v2xvit_a2_p75.json`

| dataset_src | 行 | 类型 | 口径/纪律 |
|---|---|---|---|
| `v2xvit_A1A2_ap` | 3 | V2X-ViT A-1 baseline + A-2 p50/p75 AP 行 | `latency_kind=NA_pending_trt_build`(A-3 TRT HOLD); `q_mode=fp32_pytorch`; `ap_pipeline=DAIR_val_1789_PyTorch_FP32`; `regime=ap_reference_pending_trt`; **model_class=v2x_vit 严禁与 pyramid_fusion 混表** |
| `v2xvit_dair_real_timing` | 1 | V2X-ViT base PyTorch FP32 hook 分段计时行 | `latency_kind=forward_hook_pytorch_fp32`(e2e_walltime p50=56.609ms); `ap_reuse_basis=core_exact_reuse_4090`(借用A-1); `regime=ap_reference_pending_trt`; ★口径差异见 notes(待 supervisor 核源) |

**新增字段/值说明:**
- `regime=ap_reference_pending_trt`: AP 真测但无 TRT latency → 非 front 候选, 等 A-3 TRT build 后 hw 补 lat 列
- `q_mode=fp32_pytorch`: FP32 PyTorch inference, 无 TRT 量化; 现有 `_regime_of()` 逻辑不覆盖 → 行中显式标 regime
- `latency_kind=NA_pending_trt_build`: 与现有 `NA_pending_idle_gpu` 不同含义(已确认空闲 GPU 但未测 vs TRT build 未做)
- **`latency_kind=forward_hook_pytorch_fp32`**: **PyTorch FP32 hook 级分段计时**, 来源=`v2xvit_dair_real.json`(CUDA Event, warmup=20, measure=100, RTX4090). **≠ collab2 ≠ e2e_trt; 不与任何 TRT 行比较**. 包含 e2e_walltime(p50/p99/mean) 和 per-stage 分段(encoder/backbone/shrinker_m1/fusion_net/nms)存于 config_json。
- **★ 口径交叉注 (ISS-035 supervisor 裁决)**: `v2xvit_structure_audit_v1.md(v1.3)` 引 e2e=61.86ms/fusion=27.39ms/NMS=23.75ms — 是同一 run 的 **mean** 口径; 本行入库用 json **p50**: e2e=56.61ms/fusion=24.02ms/NMS=22.21ms. **fusion_net p99=111ms 重尾致 mean>p50**, 两组勿当矛盾测量混比(同 run 不同统计量, 非两次实验).
- `mATE/mASE/mAOE+CI`: 直接从 JSON 读入, 逐行独立真测(`mAOE_basis=measured`), DAIR_val_1789 同人口全集 CI, ISS-024 规范

**★ 纪律 (禁止)**:
1. **禁跨模型混表**: V2X-ViT 与 Pyramid 不同模型, `model_class` 不同 → 预测器/Pareto 前必须 `groupby('model_class')` 拆分
2. **禁拼单调"剪枝率→AP"曲线**: p75 `actual_filters=[64,32,64]` 非单调(L1+round_to=32 致 stage0 留满); config_json 承载 actual_filters, stage0/1/2_planes 已用实测值
3. **禁把 ap_reference_pending_trt 当 front Pareto 候选**: 无 latency = 无法上 Pareto; A-3 TRT build 后 hw 补 lat 才能转 front
4. **禁把 forward_hook_pytorch_fp32 与 collab2/TRT 行混比**: 口径不同(walltime vs engine lat), 不可在同一 latency 轴上绘图

**5 轴覆盖(非空/65)**: AP 62 / latency 62 / throughput 62 / energy 59 / model_size 50。
**5 指标完整 = 50 行** (V2X-ViT 4行: AP-only 3行无lat/energy/size + timing 1行有lat但latency_kind≠collab2, 均不计入完整点)。
**regime 分布 (65行)**: front 31 / ablation_perstage_dominated 18 / ablation_guardrail_off 9 / ablation_throughput_saturation 3 / **ap_reference_pending_trt 4 (新, 含1个timing行)**。
**model_class 分布**: pyramid_fusion 61 / **v2x_vit 4 (新)**。
**mAOE 覆盖**: 29/65 行 (Pyramid 25 + V2X-ViT 4 = 29, `measured`或`reuse` basis)。

---

## ★ [ACTIVE — 列已 wire, gated 待补全] 闭环驾驶指标列 — Sim-D (2026-06-12 转正)

> **状态**: ★2026-06-12 列已正式 wire 进 `build_dataset_v2.py` COLS + `dataset_v2.{csv,parquet}` 表头(全表 70→**93 列**, 新增 8 元数据 + 15 `cl_` 性能列 = 23 列; ★dataset_v2 = **最全面统计源**, 闭环全列保留)。**当前 0 行闭环数据**(`CL_INGEST=False` gated): pilot r0 八档 DS 全平=100(场景太易非有效信号), 未达 GO 门。**时延补全后**(tau-ego 大路线库扫延迟梯度, 聚合均值 DS+碰撞率), 满足 GO 门时把 `build_dataset_v2.py` 的 `CL_INGEST` 置 `True` 即自动接入(列已就位, 无需改 schema)。
> **★精简学习视图 (`dataset_v2_learning`) 闭环口径**: 只留 `cl_driving_score`(DS) + `cl_route_completion`(RC) 两个目标指标 + `latency_inject_ms`(τ 特征); **剔除 norsu(单车 ego-only 消融基线)行** —— 闭环学习只针对 RSU-present 的 τ_ego→驾驶主曲线。norsu 完整保留在全表作 V2X 收益对照(CoDriving +62.49%DS/−53.50%碰撞)。
> **GO 门 (team-lead 把, sim_test_design_v1.md §5)**: ①P0诊断确认路侧融合BEV真接入ego control(✅ §4.4 已达) + ②至少1 route出现DS随时延可分辨区分(🔶 待 tau-ego 探针)。
> **数据源**: `V2Xverse/results/closedloop_sweep_v1.csv` (sweeper 落在 V2Xverse repo results/, 非 V2X/results/)
> **关键隔离纪律**: 闭环指标与现有感知指标(AP/AMOTA/latency) **不同任务口径, 严禁混入同列/同曲线/同 Pareto 图**。
> **学习视图**: `cl_` 闭环列**同时进精简学习视图** `dataset_v2_learning.{csv,parquet}`(闭环也是后续模型学习目标, 经时延补全); 用前必 `groupby('model_class')` 拆 codriving_v2xverse。

### 新 regime + model_class 值

| 字段 | 新值 | 说明 |
|------|------|------|
| `regime` | `sim_closedloop` | 完全隔离感知 Pareto 候选池(front/ablation*) |
| `model_class` | `codriving_v2xverse` | 严禁与 pyramid_fusion/v2x_vit 混表 |
| `latency_kind` | `injected_sim_arm` | 注入值≠真测值; ≠ body_subnet_collab2 ≠ e2e_trt |
| `lat_p50_ms` | NaN | 注入值绝不写入主 latency 轴 |
| `ap_valid` | False | 闭环任务无 AP |
| `is_real_measured` | True | CARLA 真跑(非 dry-run) |

### 全表落表列: 23 列 (8 元数据 + 15 cl_ 性能) — dataset_v2 最全面统计源

> ★dataset_v2 保留全部闭环列。**学习视图只取下表加粗的 3 列** (`cl_driving_score` / `cl_route_completion` / `latency_inject_ms`), 其余仅在全表。

**性能 (15 cl_):**

| 列名 | 类型 | 来源列 | 有效范围 | 说明 |
|------|------|--------|----------|------|
| **`cl_driving_score`** | float | driving_score | [0, 100] | ★**学习视图保留** DS = RC × infraction_penalty × 100 (主纵轴) |
| **`cl_route_completion`** | float | route_completion | [0, 100] | ★**学习视图保留** 路线完成率(区分"撞了跑完"vs"卡死") |
| `cl_infraction_penalty` | float | infraction_penalty | [0, 1] | 违规惩罚因子(=DS/RC 可反推) |
| `cl_collision_ped` | float | collisions_pedestrian | ≥ 0 | 行人碰撞次数(最干净安全信号, 晚刹车机理) |
| `cl_collision_veh` | float | collisions_vehicle | ≥ 0 | 车辆碰撞次数(机理分解, 噪声较大) |
| `cl_collision_layout` | float | collisions_layout | ≥ 0 | 布局碰撞次数(稀有) |
| `cl_red_light` | float | red_light | ≥ 0 | 闯红灯次数 |
| `cl_outside_lanes` | float | outside_route_lanes | [0, 1] | 偏出道路比例 |
| `cl_zoh_age_mean` | float | zoh_age_mean | ≥ 0 / NaN | ZOH 平均帧龄; norsu 行 = NaN |
| `cl_delta_frames` | float | delta_mean | ≥ 0 / NaN | = ceil(inject_ms/50) for d* 档; **norsu = NaN** |
| `cl_zoh_held_frac` | float | zoh_held_frac | [0, 1] / NaN | ZOH 持帧比例; 质检: per-route 随 inject_ms 单调非降 |
| `cl_audit_frames` | int | audit_frames | ≥ 0 | 总帧数; norsu = 0 |
| `cl_duration_system_s` | float | duration_system | > 0 | 墙钟秒数 |
| `cl_route_length_m` | float | route_length | > 0 | 路线长度(米) |
| `cl_status` | str | status | Completed/Failed | 非 Completed 不入库(故落表行恒 Completed) |

**元数据 (8):**

| 列名 | 类型 | 值/来源 | 说明 |
|------|------|---------|------|
| **`latency_inject_ms`** | float | inject_ms | ★**学习视图保留(τ特征)** 注入档 ms; norsu = NaN; **不写入 lat_p50_ms** |
| `latency_ms_source` | str | `injected_from_E7` | Orin E7 真测为注入值依据(≠现场实测) |
| `isolation` | str | `single_card_shared` | CARLA+推理同卡; P2 Orin 真隔离后改 separate_gpu |
| `sim_route_id` | int | route_id | tau-ego 路线库 route 编号(聚合 mean DS/碰撞率用) |
| `sim_arm` | str | arm | d0/d50/d108/d150/d219/d300/d500/norsu |
| `sim_route_set` | str | `tau_ego_route_library_v1` | ★演进自旧 6route: tau-ego 大路线库(~105 条, 聚合消单路噪声) |
| `n_repeat` | int | 1 | 确定性单次(种子固定 CARLA/Traffic=2000) |
| `rsu_enabled` | bool | arm != 'norsu' | ★norsu=单车 ego-only **消融基线(非部署模式)**; d*=True 有路侧。**学习视图剔 norsu 行后不需此列** |

### 入库质检门槛 (7 条, 违反打回 sim-integrator)

> ★这些检查在 loader 接入时对 **raw csv** (`V2Xverse/results/closedloop_sweep_v1.csv`) 执行; 其中 `status`/`delta_mean`/`zoh_held_frac`/`zoh_age_mean`/`collisions_layout` 等是 raw 列(精简后**不落表**), 仅用于 QC, 通过后只写 7 个核心列。

1. raw `driving_score` ∈ [0, 100] (→ 落表 `cl_driving_score`)
2. raw `collisions_pedestrian/vehicle/layout` ≥ 0 且非负 (ped/veh 落表, layout 仅 QC)
3. raw `route_completion` ∈ [0, 100] (→ 落表 `cl_route_completion`)
4. raw `status` == "Completed" (Failed 不入库; 故落表行恒 Completed, 不存 status 列)
5. raw `delta_mean` 精确 = ceil(`inject_ms`/50) for d* 档 (机制正确性检验, raw-only)
6. raw `zoh_held_frac` per-route 随 `inject_ms` 单调非降 (ZOH 机制健康度探针, raw-only)
7. norsu 行: `inject_ms`=NaN / `zoh_age_mean`=NaN / `delta_mean`=NaN / 落表 `rsu_enabled`=False

### ⚠️ 禁止事项

1. **禁跨 model_class 混 Pareto**: codriving_v2xverse 与 pyramid_fusion/v2x_vit 不同任务/模型 → 所有预测器/前沿图必须先 `groupby('model_class')`
2. **禁把 `sim_closedloop` 行当感知 front 候选**: 无 AP/latency_p50 → 无法上感知 Pareto; 仅用于闭环 DS-vs-时延曲线分析
3. **禁把 norsu 当 d0 连续档**: norsu 是"无路侧分支关闭", d0 是"0ms延迟有路侧"; `cl_delta_frames` 语义不同, 不可在同一时延轴连续绘制
4. **禁以 injected_from_E7 等同现场实测**: 论文/图中必须标注 `latency_inject_ms` 是注入值, 不是 CARLA 现场测量的推理延迟

### 待定 (P2 Orin 真隔离阶段扩展)

- P2: CARLA 在 4090 / 推理在 Orin → `isolation=separate_gpu` / `latency_ms_source=measured_on_orin` / 补真实边缘时延锚点
- 4090 档: e2e ~27-35ms → Δ=1帧(亚帧), 与 Orin(Δ=3-5)形成"强力 RSU vs 边缘 RSU"对照
- Orin 能耗: tegrastats 采集(P3 可选)
