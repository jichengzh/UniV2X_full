# Pareto 前沿定义 — 本研究"优化效果"的最终评判标准 (v1, 2026-06-03)

> 本文档定义本研究**如何判定一个软硬件协同优化配置是否更优**:用哪些指标、怎么构成多目标 Pareto 前沿、selector 在不同部署场景(regime)下如何切换目标/约束。
>
> 上游决策来源:`background/00_研究目标与实验档案_v1.md` §8(用户拍板:吞吐+能耗升一级指标)+ §0′(ISS-018 用户拍板:AP 降为约束轴 → **★2026-06-04 用户拍板 v2 部分撤销: AP 恢复常驻目标轴**, 见本文 §三/§七)。
> 数据落地:`multi_agent/data/dataset_v2.{csv,parquet}` + `schema_v2.md`。
> 可视化:`multi_agent/figure/dataset_pareto_coverage.png`(可复跑 `make_dataset_pareto_coverage.py`)。
> **精度轴(AP/mAOE)的定位见 §七** —— 该节整合自原 `methods/design/ap_activation_strategy_v1.md`(2026-06-04 doc-curator 并入;过程版 ROI 分析完整存档于 `multi_agent/archive/`)。

---

## 一、为什么需要多目标 Pareto,而不是单一分数

协同加速的配置空间 = **剪枝 × 量化 × 硬件部署(D)× batch**,每个配置在多个相互冲突的指标上各有取舍(更快往往更耗精度/更吃显存,更省能往往更慢)。**没有单一标量能正确排序**——必须用 Pareto 前沿:一个配置若在所有指标上都不被另一个配置支配,即在前沿上,是"可选的最优解之一"。selector 的任务是在前沿上按部署场景选点。

---

## 二、5 个一级指标(Pareto 轴)

预测器(predictor)**预测全部 5 个指标**;selector 按 regime 条件决定哪些当目标、哪些当约束。

| # | 指标 | 数据集列 | 单位 | 方向 | 口径/测量 |
|---|---|---|---|---|---|
| 1 | **精度 AP** | `ap30 / ap50 / ap70`(+ Phase M 新增 mATE/mASE/mAOE TP 误差列 42–47, 见 §七.2) | — | ↑ 越大越好 | DAIR-V2X val 1789 真测, intermediate fusion。报告主用 **AP70**(信号最强;AP50/30 近饱和)。★[2026-06-04 用户拍板 v2] AP = **常驻纯目标轴, 不作任何约束, 任何情况下不得从 Pareto 目标中删除**(见 §三/§七) |
| 2 | **延迟 latency** | `lat_p50_ms / lat_p99_ms / lat_mean_ms` | ms | ↓ | TRT 引擎实测;**必须标 `latency_kind`**(body_subnet_collab2 / engine_board_energy / dla_pipeline_e2e_single_frame …),**不同口径不可混比** |
| 3 | **吞吐 throughput** | `throughput_fps` + `throughput_kind` + `batch` | fps (QPS) | ↑ | 见 §四。**非冗余维度**,但只有 `batched`/`pipelined` 口径才脱离 1/latency |
| 4 | **能耗 energy** | `energy_per_frame_mj` / `mean_power_w` / `perf_per_watt_fps_per_w` | mJ/frame, W, fps/W | ↓ (J/frame) / ↑ (perf/watt) | NVML board 功率实测 (E4);INT8 比 FP16 省 30–52% J/frame |
| 5 | **模型体积 model_size** | `engine_size_mb`(+`params_total`) | MB | ↓ | TRT 引擎文件大小;约束显存/存储预算 |

> 注:FPGA **仅作 related-work 讨论**,不作为本研究的真实部署轴。

> ★**[2026-06-12 双表 + 评价维度精简]** 上表"数据集列"是**全表 `dataset_v2`(最全面)** 保留的多档候选;**精简学习视图 `dataset_v2_learning` 每维度只留一个规范列**供预测器/选择器学习:精度 **`ap70`**(信号最强,AP30/50 近饱和)/ 延迟 **`lat_p50_ms`**(+`latency_kind` 分组)/ 吞吐 **`throughput_fps`**(+`throughput_kind`)/ 能耗 **`energy_per_frame_mj`**(每帧真成本)/ 体积 **`engine_size_mb`**;另留约束信号 `mATE/mASE/mAOE`。删的 8 冗余列(ap30/ap50、lat_p99/mean、mean_power_w/perf_per_watt、params_total、delta_ap50)原值仍在全表。预测器仍**预测全 5 指标**,只是各用其规范列。详见 `multi_agent/data/schema_v2.md §双表结构`。
>
> ★**[闭环驾驶指标 = 独立任务轴,非 5 感知轴之一]** 闭环 DS/碰撞率/RC(`cl_driving_score` 等,`model_class=codriving_v2xverse`/`regime=sim_closedloop`)与上 5 个感知 Pareto 轴**不同任务口径,严禁混入同 Pareto/同曲线**。全表 wire 全部 23 闭环列(最全面),学习视图闭环只留 **`cl_driving_score`(DS)+`cl_route_completion`(RC)** + `latency_inject_ms`(τ 特征),且剔 norsu(单车 ego-only 消融基线)行。现 0 行(gated),时延补全后填。详见 `schema_v2.md §闭环驾驶指标列`。

---

## 三、前沿结构:默认目标 + 约束 + regime 切换

### 默认主前沿(目标轴)

★[2026-06-04 用户拍板 v2, 部分撤销 ISS-018] **精度轴(AP)是常驻目标轴, 任何情况下不得从 Pareto 目标中删除**。默认主前沿:

> **(AP ↑, latency ↓) 双核心目标 + energy(及 regime 决定的 throughput)**

- **核心 trade-off = 精度 vs 速度/吞吐**。本研究的根本问题就是"剪枝/量化/硬件部署在多大代价下换多少加速", 精度是代价的第一度量。即使精度轴在当前模型/数据上信息量低(Pyramid/DAIR 近常数, 实证见 §七.1), 也**以"前沿沿 AP 轴退化/塌缩"的形式诚实呈现**, 而不是把该轴删掉 —— 换指标(Phase M mAOE)/换模型(V2X-ViT, Task#8)的全部动机正是**把精度轴做实**, 删轴与该路线自相矛盾。
- **历史沿革(留痕防回退)**: ISS-018(2026-06-03)曾把 AP 降为纯约束轴、主前沿转 (latency, energy, throughput)。该决策中 **"AP 移出目标轴"部分已被用户 2026-06-04 撤销**;其仍然有效的部分 = ① Pyramid/DAIR 上 AP 近常数的**实证本身**(§七.1, 仍成立, 这是 finding 不是删轴依据) ② 成本轴(latency/energy/throughput)+ 耦合陷阱(ISS-014 kernel-cliff)是**当前数据上差异化信息的主要来源**(报告仍重点展开)。
- 精度轴的具体度量: **AP70 为主**(信号最强) + mATE/mASE/mAOE TP 误差项作补充/约束信号(§七.2)。

### 约束轴(满足即可,不进目标)
★[2026-06-04 用户拍板 v2] **AP 不在约束轴** —— 精度是纯目标轴(见上), 不参与可行性筛选。约束轴只含部署可行性条件:
- `throughput ≥ FPS_floor`(实时帧率下限,如 10/20 Hz;经 batch/流水线解耦后升为目标轴, 见 §四)
- `model_size ≤ 显存/存储预算`
- (regime 决定的)`latency ≤ 上限`

mAOE 等 TP 误差项作剪枝退化**参考信号**入库(§七.2), 供预测器/分析使用, 不作筛选约束。

### regime-conditional 切换(selector 行为)
★[2026-06-04 用户拍板 v2] AP 在**所有 regime 的目标轴中常驻**, regime 只切换成本侧哪个指标主导:

| regime | 绑定 | 目标轴(AP 常驻) | 约束轴 |
|---|---|---|---|
| **ego / 车载单机** | latency-binding | (**AP**, **latency**, energy) | throughput≥FPS下限, size≤预算 |
| **RSU / 路侧多车** | **throughput-binding** | (**AP**, **throughput**, energy) | latency≤上限, size≤预算 |
| **能耗敏感 / 电池** | energy-binding | (**AP**, **energy**, latency) | latency≤上限, throughput≥下限, size≤预算 |

**关键**:同一组预测的 5 指标,selector 仅切换"哪个当目标、哪个当约束",不重训预测器。

---

## 四、throughput 为什么是独立轴(不是 1/latency 的冗余)

throughput 只有在**单流 batch=1 无流水**时才退化为 `1000/latency`。本研究用 `throughput_kind` 显式区分三种口径:

| `throughput_kind` | 含义 | 是否脱离 1/lat | 数据来源 | 行数 |
|---|---|---|---|---|
| `inv_latency` | 单流 batch1,throughput ≡ 1000/lat_p50 | 否(退化) | collab2-body + E4 batch1 + Orin | **50** |
| `batched` | batch>1 真批处理,throughput = batch×(1/batch_lat) | **是** | E4 batch=2 (cudagraph) | 6 |
| `batched_request` | collab2 并发多请求峰值吞吐 | **近否**(SM饱和1.106×→≈1/lat) | P12 collab2 多流 | 3 |
| `pipelined` | 跨帧流水,interframe < e2e 单帧延迟 | **是** | E3 Orin DLA0‖DLA1 跨进程 | 2 |

**真实解耦证据(E3,已真测)**:split `016_032_064` 单帧端到端延迟 = 23.71 ms,但稳态 interframe = 16.05 ms → **throughput = 62.3 QPS ≠ 1000/23.71 = 42 QPS**。流水把吞吐拉离了延迟。

**经验结论(本项目实测,见 `dims_hardware_v2.md`)**:
- 单 GPU stage 流水 **不成立**(1.08×,撞 L2/DRAM 带宽墙);
- 异构 **GPU‖DLA(Orin)成立**:跨进程 handoff 真测 **1.29×**(小 split),但大 split 因 LPDDR5 带宽争用转 0.91×;
- batch(空间维=多车纯吞吐净赚 / 时间维=拿延迟换吞吐)是吞吐脱离 1/lat 的另一途径。

⇒ throughput 是真维度;当前 51/65 行仍是 inv_latency(退化)。P12 batched_request 3行已测并发峰值仅 1.106×(SM@1=94% 饱和)→ **主前沿诚实 2D(lat, energy)**;Orin pipelined 2行保持 1.29×(跨进程异构)。补采方向 = Orin 多分割 / 真 batch>1 RSU 场景。

---

## 五、当前数据覆盖与缺口(`dataset_v2`, **65 行 × 93 列**; 2026-06-12 wire 闭环 23 列 + 另出精简学习视图)

**演进**: 原 48 行去重→43 → P0-1 +2(p25 INT8 耦合点) → 晚批 wire +16(路径 A/B/E 消融+P12 吞吐+[32,64,136]+Orin E6) → Phase M 八步定稿(mAOE 终核, 列扩 62→70) = 61 行 → +V2X-ViT 4 行(model_class=v2x_vit) = **65 行** → ★2026-06-12 wire 闭环驾驶 23 列(8 元数据 + 15 `cl_`, 列 70→**93**)。完整演进详见 `schema_v2.md`。
> ★**双表**: `dataset_v2`(全表 65×93, 最全面统计源)+ `dataset_v2_learning`(精简学习视图 65×65, 开环每维度1规范列 + 闭环只 DS/RC + 剔 norsu 行)。下文覆盖数按**感知行(65)** 统计;闭环行现 0(gated, 时延补全后填)。

**★ 关键:数据集有 4 种测量 scope(latency_kind),不可混在一条 Pareto:**
| scope (`latency_kind`) | 行数 | 含义 |
|---|---|---|
| `body_subnet_collab2` | **40** | **双 agent 协同体**(V2X 主部署口径, RTX4090)。含 front(31 行)+ ablation(9 行) |
| `engine_board_energy` | 13 | E4 **单 agent** 引擎(≈collab2 体的 0.65×)。有 lat+throughput+energy+size+(补)AP |
| `body_subnet_collab2_orin` | **6** | **Orin 双 agent 体**(Jetson Orin AGX, 30W; ≠4090 不混比 Pareto)。FP16 AP=fp16_crossplatform_exact; INT8 ap_valid=False(跨 TRT 输出发散) |
| `dla_pipeline_e2e_single_frame` | 2 | E3 Orin 异构 GPU‖DLA 流水(跨进程, 1.29×) |

> 上表 4 scope 合计 61 = pyramid_fusion 感知行;另 +4 V2X-ViT 行(`latency_kind=NA_pending_trt_build` 3 + `forward_hook_pytorch_fp32` 1)= 65。闭环行(`injected_sim_arm`)现 0(gated)。

**当前覆盖(2026-06-12, /65 含 +4 V2X-ViT 行; V2X-ViT=v2x_vit model_class, AP-only 3 + hook计时 1, latency_kind=NA_pending_trt_build/forward_hook_pytorch_fp32, 独立口径不混 pyramid Pareto)**:

| 指标 | 非空行数 | 说明 |
|---|---|---|
| AP (ap70) | **62 / 65** | 3缺口: Orin **INT8 3行** ap_valid=False(不计; FP16 3行有 EXACT 复用 AP) |
| latency | 62 / 65 | 缺口=V2X-ViT 3 AP-only 行待 A-3 TRT build; 四种 scope 不可混比 |
| throughput | 62 / 65 | inv_latency 51 / batched 6 / batched_request 3(P12) / pipelined 2 |
| **energy** | **59 / 65** | collab2(E5 双 agent) + E4 + Orin E6; 缺口=dla_pipeline 2行 + V2X-ViT 4行 |
| model_size | **50 / 65** | 缺口: Orin 6行无 engine_size + P12 batched_request 3行 + dla_pipeline 2行 + V2X-ViT 4行 |

**★ 5 指标完整行 = 50**:
- **collab2 双 agent 主口径(4090) 37/40 行 5 指标完整**(P12 吞吐消融 3 行缺 engine_size → 不计)。**37 行是 selector 可直接跑 5 轴 Pareto 的主数据。**
- **E4 单 agent 13/13 行全完整**(lat+AP+throughput+energy+size)。
- collab2(37) + E4(13) = **50**。
- Orin E6 6 行: FP16 3行有能耗(VIN_SYS_5V0 整模块), INT8 3行 ap_valid=False(不计入 AP 覆盖; lat+energy 可用)。

**能耗双口径(分别自洽, 不混)**: collab2/Orin 用 E5(双 agent 输入); 单 agent 用 E4(board-level NVML)。跨口径强比会因测量路径不一致而失真。

### 5.X 缺口与补采计划(Phase M 后状态)

> 原 §六-bis "下一步采集计划" P0–P3 已全部执行，归档于此备查。

| 计划 | 状态 | 结果/说明 |
|---|---|---|
| **P0 — collab2 能耗真测** | ✅ 完成 | E5 双 agent NVML 管线 28/28 引擎; 两路验证均值 0.40%/最大 0.89% 吻合(见 `E5_energy_verification.md`) |
| **P1 — collab2 batch 吞吐** | ✅ 完成(零结论) | P12 collab2 多请求并发峰值仅 **1.106×**(SM@1=94% 饱和) → throughput≈1/lat; **主前沿诚实 2D(lat, energy)** |
| **P2 — 缺口补测** | ✅ 完成 | `T1_base_FP32` AP 已补; `[32,64,136]` AP 4dp 独立真测(P0_2_136); 同架构已回填 E4/E3 行 |
| **P3 — Orin 能耗(可选)** | ⏸ 未采 | E3 dla_pipeline 2行无 Orin tegrastats 能耗; 可选追补, 不阻其他分析 |

**AP 补全规则(写死,防造假)**:仅 `(planes, 三段精度)` **EXACT 匹配** 才补; 跨架构(`[32,64,136]≠[32,64,128]`)不借用; 跨平台 FP16 AP 借用须 output-match 验证(P0_3 已验 Orin E6 FP16 近同值); INT8 跨平台 ap_valid=False(TRT 输出发散, 透明标)。

---

## 六、与 predictor / selector 的接口

1. **predictor** 输入 = config(剪枝×量化×硬件×batch,见 `config_json`),输出 = 预测的 5 指标向量。训练只用真测点,按 `latency_kind` 分组,不跨口径。
2. **selector** = 在预测的 5 维空间上做 regime-conditional Pareto 过滤(§三),返回该 regime 下的前沿配置集。
3. **评判优化效果** = 看新配置是否**推进了对应 regime 的 Pareto 前沿**(在目标轴上不被支配、且满足约束轴),而非看任何单一指标的提升。

---

## 七、精度轴定位(AP / mATE / mASE / mAOE)— 常驻目标轴 + 当前模型上信息量的实证评估 ★[2026-06-04 整合自原 `ap_activation_strategy_v1.md`]

> 原独立文档 `ap_activation_strategy_v1.md`("激活 AP 故事"四路径 ROI 过程分析)已整体存档于 `multi_agent/archive/`;本节只保留其**当前有效定论**。决策链:ISS-013(forced-int8 被支配)→ ISS-018(用户拍板 AP 作约束)→ Phase M ISS-020(换指标 mAOE 终核 PASS)→ **★2026-06-04 用户拍板 v2(撤销 ISS-018 的"AP 移出目标轴"部分, AP 恢复常驻目标轴;本节实证结论不变, 其解读从"删轴依据"改为"当前模型上前沿沿精度轴展开有限的 finding")**。

### 7.1 实证:Pyramid/DAIR 上 AP 无可提取的成本-trade-off(ISS-018, 用户拍板 + supervisor 独立核验)

**AP 轴在 DAIR/Pyramid 上近乎平**(整个剪枝+量化空间 AP70 仅 0.49–0.63, 无悬崖;AP50 ≈0.79 天花板近饱和)。"激活 AP 故事"的 A/B/E 三路免训实测**全部为负结果**:

| 路径 | 全量结果(实测) | 判定 |
|---|---|---|
| **A 剪枝×forced-INT8** | forced-all-int8 ΔAP70 分段放大(base/p25 ~-0.010, p50 -0.018, p75 -0.034, 为噪声 10-34×;非严格单调, base≈p25);cliff 系(pb 0.364/0.270/0.088M)forced ap70 0.62/0.61/0.60 无悬崖 | ✓ 预测器真信号 + 护栏消融证据;✗ **forced 是 Pareto 被支配点**(auto-int8 延迟+AP 双轴碾压, ISS-013, base/p50/p75 全档成立)→ 入主表标 `regime=ablation_guardrail_off`, **不进前沿叙事** |
| **B 拆护栏 head-INT8** | base Δap70 **-0.0046** / p50 +0.0044(噪声内)→ **不崩** | 崩溃-规避叙事**不能在 Pyramid conv head 声称**(崩在 transformer/MSDA);本阶段引 QuantV2X 文献(V2X-ViT INT8 AP30 57.4→40.0/AP50 49.5→11.0; ★ISS-031 勘误, 旧 "75.1→29.9" 系跨模型拼接勿用)作 motivation, **不声称自有模型 demonstrated** |
| **E 难子集重 eval** | 剪枝 spread 随距离 **shrink**(near -0.147 → r80plus -0.074, n_gt 3172);INT8 全箱免费 | 按 **ISS-015 判据**严格判定 = **floor 效应非难样本 trade-off**(smoke 3.6× 是 60 样本假象, 全量推翻), 不硬凑"激活"。诚实负结果(加分科学, 非失败) |

**根因 = Pyramid 对 DAIR 严重过参数化** → INT8 处处近免费、剪枝无悬崖、难样本 spread 被 AP 地板压缩。详见 `dims_pruning_v1.md §8`、`dims_quantization_v1.md`。

**决策(★2026-06-04 用户拍板 v2 修订)**:① ~~AP 作约束轴(不当目标搜)~~ → **AP 恢复常驻纯目标轴, 不再作任何约束**(§三);② 成本轴(latency/energy/throughput)+耦合陷阱仍是当前数据上差异化信息主要来源, 报告重点展开 —— 但**以"(AP, 成本)前沿沿 AP 轴塌缩"呈现, 不删轴**;③ A/B 数据入表作 ablation/predictor(`regime=ablation`);④ V2X-ViT/换模型 = 把精度轴做实的确定工作(Task#8 已授权进行中)。

### 7.2 Phase M:换指标(mATE/mASE/mAOE)实测定论(2026-06-04, ISS-020 终核 PASS)

用户提出"AP 信息量低可能是评价指标问题"→ Phase M 用 NDS TP 误差项在 4 锚点(base/p25/p50/p75)× {FP16,INT8} × DAIR val 1789 真测验证(sw 真测 → data 独立复算 → supervisor 独立复跑 三层核验, 含 ISS-024 权重污染修正 + 共同 GT 子集消幸存者偏差 + 帧级 bootstrap CI):

1. **剪枝轴: mAOE 信号成立 ✅** — 严格单调。**全集 1789(epoch25 修正, B=1000 帧级 bootstrap)**: SNR=**14.2×**(全档均半宽; 备用 13.1×avg-half / 17.1×pooled-SE; 全部 >5× 阈值), base→p75 Δ=+0.0244 rad, base↔p75 CI 严格不重叠; **p25↔p50 相邻细档 CI 重叠, 不可分辨**。**共同 GT 子集(固定 2967 框)独立确认信号存活**(SNR 4.0×pooled / 7.5×avg-half, Δ≈+0.0346), 排除幸存者偏差。mAOE 捕捉到 AP70 低估的朝向退化(剪枝先伤连续回归)。(注: 勿与 ISS-024 污染版数字混用 —— 2849 框/SNR 7.7× 属 epoch31 污染版, 已废。)
2. **量化轴: 无信号 ✗** — INT8 单步 ΔmAOE 非单调变号, p25/p50/p75 SNR=**0.58/0.05/0.82×**(噪声级, 3/4 档 CI 重叠+非单调); base 档 Δ+0.0030 CI 可分辨但非单调+量级仅 5%, **不构成轴信号**(ISS-020 §4)。**"量化在 Pyramid/DAIR 近无损"是真结论, 不是 mAP 指标盲区** —— 换再多精度指标也测不出量化 trade-off。注意: 勿写"全部档噪声级" —— base 档 CI 可分辨但不满足轴信号判据。
2-bis. **mATE/mASE 与 NDS 的取舍(★2026-06-04 补记, 回答"为什么只讲 mAOE")** — 全集 1789(FP16, base→p75)三个 TP 误差项**全部**过 ≥5× 判据且单调: **mATE** Δ+0.0297(+12.9%, SNR≈11.9× avg-half)/ **mASE** Δ+0.0090(+6.9%, ≈7.7×)/ **mAOE** Δ+0.0244(**+40.9%**, 13.1–14.2×)。只把 mAOE 升格为叙事主信号的原因: ① 相对退化幅度最大(+41% vs +13%/+7%), 对剪枝最敏感(剪枝先伤朝向回归); ② **只有 mAOE 跑完完整核验链**(共同 GT 子集消幸存者偏差仅算了 mAOE + ISS-024 修正), mATE/mASE 未做共同 GT 控制 → 按"未核验不升格"纪律**只入库不进结论**(dataset_v2 列 42–44 存真值, 共同 GT 框集已固定、可低成本补核)。**不采用 NDS 综合分**: ① 标准 NDS 需 5 个 TP 项, 其中 mAVE(速度)/mAAE(属性)在 PyramidFusion/DAIR 单帧检测上**物理不可测**(无速度/属性输出头), 只能算非标准截断 NDS, 审稿风险大; ② 加权合成分把近饱和 mAP 与 TP 项混合、经 1−clip 归一化稀释, **掩盖分项信号**, 与诊断目的相反; ③ Pareto 轴方法论用原始量纲指标, 不用预设权重的合成分(§一)。**[★2026-06-04 截断 NDS-3 实算验证(应用户要求)**: NDS3=(5·mAP+Σ₃(1−err))/8, mAP=mean(ap30/50/70), 基于 `tp_errors_corrected_full.csv` 点估计+TP-CI 传播(mAP 管线噪声取 0.002 保守)。**剪枝轴显著**: 单调降 0.7923→0.7765→0.7694→0.7526, base→p75 Δ=−0.0397, SNR=21.3×(sqrt 传播)/10.2×(保守线性), p50 档起保守口径也过 ≥5×。**量化轴不显著**: Δ −0.0024/−0.0023/−0.0086/−0.0032 非单调, SNR 0.6–4.7×, 仅 p50 一档冒头不构成轴信号 —— 与分项结论一致。**稀释实锤**: base→p75 的 ΔNDS3 中 mAP 部分贡献 −0.0318(80%)、TP 项仅 −0.0079(20%); mAOE 相对退化 +41% 被 NDS3 压成整体 −5% —— NDS 基本是 mAP 的重述, 确证"合成分掩盖分项信号", 维持不采用决策。caveat: 非标截断版(3/5 TP 项)、CI 为传播近似非联合 bootstrap。]**
3. **定位(★2026-06-04 用户拍板 v2 修订)**:精度轴 = **常驻目标轴**(§三;ISS-018 的"仅约束"定位已撤销)。实证事实保留: mAOE 相对动态范围仍 ≪ 成本轴 5×, 即**当前模型上前沿沿精度轴的展开有限** —— 这呈现为前沿形状的 finding, 不作删轴依据。**mAOE 升格为预测器"剪枝退化参考信号"入库**(dataset_v2 新列, 带 `mAOE_basis` 溯源 + CI, 详见 `schema_v2.md`;不作筛选约束)。信号仅在全剪枝程(base↔p75)可分辨, 相邻细档(p25↔p50)CI 重叠。
4. **判据纪律(宪章 MUST-NOT-5)**:信号强弱只认 "Δ vs 自身 bootstrap CI(≥5×)",**禁用 "×AP70" 比值判据**(两噪声相除=假精确, 曾被撤回两次)。
5. **路线(用户拍板)**:真正激活精度-成本 trade-off 须换模型/换数据,顺序 = **V2X-ViT 先行(Task#8)**;UniV2X/V2X-Seq+AMOTA 暂缓(Task#7;注:真 AMOTA 只能来自有跟踪头的 UniV2X 家族, PyramidFusion 物理上不产 AMOTA, ISS-026)。

出处:`results/tp_errors_corrected_full.csv`(终核版, 8行) · `tp_errors_ci_v1.csv` · `common_gt_corrected.csv`(2967框) · `scripts/phase2/eval_tp_errors_corrected_full.py`(ISS-024 修正版) · issues_log ISS-020/024/025/026/029。

### 7.3 其它已知事实约束(避免被旧假设误导)

- **INT8 ≈ 免费的速度/体积/能耗轴**(ΔAP70 亚噪声,省 30–52% J/frame),不是质量 trade-off 轴。
- latency/throughput/energy 必须在**完全空闲 GPU**上测;每个数字标 `is_real_measured` + `source` + 口径。

---
*维护:改 5 指标定义或 regime 切换逻辑时同步更新本文件 + `schema_v2.md` + `background/00 §8`;精度轴定位(§七)有新定论时更新本节(勿再另开新文档), 过程分析归 `methods/progress/` 或 `archive/`。*
