# 9_7_3 P2 实施 — 空间落地 + measure 接口 + SMBO 闭环

**时间**: 2026-07-03
**承接**: `8_7_2_交接文档...待审.md`(决策已冻结) → 本文执行 P2.0/P2.1/P2.2
**范围**: 仅 Pyramid + H800(冻结)。量化轴全局单精度;3-stage 粗分区;|Θ_sw|=7³·3=1029(未测 849)。

---

## P2.0 空间落地 ✅

**产物**: `.../original60_quant_20260627/space/space_manifest_pyramid_h800_v1.json`
**脚本**: `scripts/stage2_p2_0_space_manifest_v1.py`(无 GPU, 可复现)

- 枚举冻结 3-stage grid: W0×W1×W2 × {fp32,fp16,int8} = 7³·3 = **1029**;asserts PASS(1029/180 measured/849 unmeasured)。
- **合法性核实(关键)**: bridge 静态 `int8_buildable_align=128`(grouped stage)会判几乎全不可 build;但 **original60 实测 60/60 int8 全 build 且 60/60 非 128 倍数、49/60 非 32 倍数** → 证明 bridge 的 align 门是 **TRT-tiling 产物, 不进 TVM native int8 可建性**(与 codriving Cin=48/16 全 build 一致)。⇒ 粗 3-stage 粒度 **无 int8 可建性悬崖, 1029 全合法**。
- **可分离性**: 粗 stage 粒度 TVM 全 build → P×Q 若有耦合是"测量驱动"非"结构性";联合价值留 P2.3 三臂消融判。

## P2.1 整网 measure 接口 + feasibility ✅

**产物**: `framework/measure_config.py`(整子网真测编排) + `framework/feasibility_gate.py`

### measure_config.py(4 步复现 original60 口径, 跑在 H800)
`(width,precision,gpu)` → ① export backbone ONNX(torch env)→ ② TVM MetaSchedule tune(`max_trials=32,seed=0`)→ ③ latency 真测(apply DB, default+tuned)→ ④ energy 真测。返回 `{lat_default_ms,lat_tuned_ms,energy_j,build_success}`。
- **latency_kind = pyramid_backbone 子网**(与 LUT/cost model 标签同口径, 非 e2e)。
- **口径校验 PASS**: 复测已测 s2_160=[64,128,160] fp32, tuned=**49.74ms ≈ 存储 49.78ms**(差 0.08%)。
- **工程坑**:①TVM 步须 pre-set `LD_LIBRARY_PATH`(bundled libcudart, 否则 `cudaGraphAddDependencies_v2` undefined);②measure job 写完行后 **CUDA teardown 挂起** → `run()` 轮询 ok_check 早杀(不空等)。
- **★口径边界(重要)**: `run_measurement_job` **只实现 fp32**;fp16 需 fp16-rewrite 管线、int8 需 native-int8 bridge。误用 fp16/int8 会静默测出 fp32 级延迟(实证:[32,64,128] "fp16" 测得 26.19ms=fp32 区间, 真 fp16≈8ms)。已加 `NotImplementedError` 守卫。⇒ **本阶段闭环用 fp32(已验证口径, 且有真调优余量)**;fp16/int8 路由为已记录 follow-up。

### feasibility_gate.py
- 唯一实测不可行信号 = frontier_01 width (24,64,128) 降精度 AP 崩(fp32 AP70=0.592→fp16/int8=0)。**仅 1 个正例 → 学不出判别器**, 诚实做成 **保守规则门 + 风险分**(降精度配置按到崩点距离给 risk, 前沿近崩点者才真 finetune 验 AP)。grid 上 2 hard-infeasible / 184 降精度 at-risk。

## P2.2 SMBO 闭环(主戏, fp32)🔄

**脚本**: `scripts/stage2_smbo_loop_v1.py`(`--step select|feedback`)
- 空间可枚举(283 未测宽度/精度)→ **精确 3 目标非支配排序**(非进化近似 NSGA-II)。
- select: inline 重训 log1p LGB(f_lat/f_energy)+ 分位 σ + AP 高原锚 → feasibility 门 → 精确 Pareto 前沿 → 采集(前沿∪高不确定)→ Top-K。
- feedback: pred vs actual 校准 + Pareto 推进(在**已测 (lat,energy) 平面**判, AP 未测不参与)。

**Round1(fp32, K=6)**: 预测 283 未测 → 18 点前沿 → 选 6:[16,32,224]/[16,32,256]/[16,64,256]/[16,128,256]/[24,32,128]/[24,32,160](全前沿, 全 feasible)。预测延迟 10-33ms。
→ 6 点 H800 真测中(GPU 0/2/3/4/5/6 并行)。**结果与闭环结论见下节(待填)**。

### Round1 闭环结果 ✅(停止点达成)
6 点 fp32 真测(GPU 0/2/3/4/5/6 并行, 每点 export+tune(max_trials=32)+lat+energy):

| width | pred_lat | **actual_lat_tuned** | lat_default | energy(真测) |
|---|---|---|---|---|
| 16x32x224 | 12.15 | 23.83 | 23.29 | 0.18(失效) |
| 16x32x256 | 12.16 | 25.16 | 25.20 | 0.25(失效) |
| 16x64x256 | 25.54 | 29.77 | 29.67 | 1.51(疑) |
| 16x128x256 | 32.59 | 39.45 | 39.45 | 4.93(合理) |
| 24x32x128 | 10.14 | **10.49** | 16.52 | 0(失效) |
| 24x32x160 | 10.04 | 24.31 | 20.75 | 0(失效) |

**闭环证据**(`smbo_loop/round1_fp32_feedback_report.json`):
1. **预测→真测→校准**: surrogate 对 6 个**真正未测**配置 latency MAPE=**32.4%**(MAE 8.4ms)。误差集中在 **w0=16 窄区**(原始 60 点该区稀疏 → surrogate 过度外推乐观:预测 ~10-12ms, 实测 ~24ms)。24x32x128 预测 10.14 vs 实测 10.49(3% 极准)佐证非全局崩、是特定区外推。
2. **★模型改善(回灌生效)**: before(60 点模型)MAPE 32.4% → after(+5 邻点重训, LOO)MAPE **18.2%**, **improved=true** —— 窄区预测误差近乎减半。这是"回灌→模型改善"的直接可复现证据。
3. **latency 前沿**: 最优 9.83ms → 新点最低 10.49ms, **未推进**(诚实): 采集函数探索窄区, 实测该区比 surrogate 预想慢 → latency 地板未破。前沿未进本身是有效信息(告诉搜索窄区无 latency 优势), 非失败。
4. **energy 轴本轮不可用**: joule_per_inference 在 <~25ms 快配置上采样失效(0 或偏低, 仅 16x128x256=4.93J 物理合理)。已在 measure_config 加 `energy_measure_status` 标记, 前沿/校准**不采信 energy 本轮**。需修 energy 采样窗口(follow-up)。

**结论**: 一轮真闭环(预测→真测→回灌→**模型改善** 32%→18%)可复现证据已取得 = **停止点达成**。附带产出 4 个新 fp32 真测锚点(16x64x256/16x128x256 energy 有效点 + 全部 latency)。

### 多轮 SMBO 收敛(rounds 1-3, 用户要求"多跑几轮找最优")✅
`scripts/stage2_smbo_convergence.py` → `smbo_loop/convergence_fp32.json`。每轮 K=6, surrogate 累积前轮真测重训。

| round | n_new | round_min_lat | **cum_best_lat** | 前沿推进 | MAPE(before→after) |
|---|---|---|---|---|---|
| base(60) | 60 | — | **9.83** | — | — |
| 1 | 6 | 10.49 | 9.83 | ✗ | 0.324→0.182 |
| 2 | 5 | 10.52 | 9.83 | ✗ | 0.157→0.156 |
| 3 | 5 | 12.72 | 9.83 | ✗ | 0.189→0.169 |

**converged=True, 最终 latency 最优 = 24x32x64 @ 9.83ms(原 60 点内, w0=24)**。三轮结论:
1. **前沿三轮不动 = 收敛**;latency 最优点原始 60 点已覆盖。
2. **★关键真测发现: 更小≠更快**。SMBO 第 3 轮的校正 surrogate 主动去测网格角点 **16x32x64=12.72ms, 反而比最优 24x32x64(9.83)慢**, 也慢于 16x32x128(10.52)。w0=16 窄配置 **GPU 欠利用/launch-overhead-bound** → 缩到最小反变慢。多轮搜索**经验性证伪了"最小即最快"假设**(非枚举, 是 surrogate 引导下定点验证)。印证全项目 "narrow w0=16 慢 / launch-bound" 结论。
3. **surrogate 校准是多轮主要收益**: MAPE 32.4%(R1, 窄区过度外推)→ 15.7%(R2, 回灌减半)→ 18.9%(R3, 因测到反直觉角点略回升但仍≈初值一半)。
4. 稳定性: 16x32x160 **两轮均 build/measure 崩(rc=-6)**, 该宽度真不可测(honest 记录, 非漏测)。
5. 新增真测: 3 轮 16 个新宽度(14 有效 + 2 崩)→ fp32 已测点 60→74。

**对"多轮找最优"的诚实回答**: 已收敛。**但这是 latency 单轴最优**;Pyramid+DAIR 的 AP 是高原(近平)→ AP×latency Pareto **退化**(过参数化, 缩宽度近"免费"直到崖口)→ 单轴 SMBO 必然快收敛。**联合搜索真正增益(Gap1)只在有真 trade-off 的载体上显现 = AP 崖口 / int8 边界**, 正是 P2.3 三臂消融的目标。故多轮 SMBO 收敛 ⇒ 应转 P2.3(而非继续单轴刷)。

**工程坑(本轮新增)**:①energy 采样对快配置失效;②`0.0 or x` Python 陷阱(joule=0.0 被判 falsy)已修为显式 None 检查;③fp32 tuned 偶尔 ≈ 或略慢于 default(max_trials=32 轻预算 + 小 width dlight 已近优), 是真测非 bug。

---

## Problem 1+2 修复(2026-07-03, 用户提出的遗留问题)

### Problem 2 — energy 快配置采样失效 ✅ 已解决
- **根因**(非窗口长度): `joule_per_inference` 减 idle 基线, 小/快配置增量功率(几 W)**低于 H800 idle 漂移(~10W)** → active−idle≈0/负 → 0(窗口已 5.4min, 延长无用)。实证: 24x32x128 active 126.8W < idle 134.8W。
- **修复**: energy 改 **总功率×延迟 = watt_avg × lat**(不减 idle, 物理=专用设备每帧能耗, 恒正)。`measure_config.parse_energy` 已改; 16 个 SMBO 点全部重算有效(1.29→11.18J 单调)。**注: energy 与 latency 近单调(功率仅 120-320W 变化), 独立 Pareto 信息有限**。

### Problem 1 — fp16/int8 接入
- **★共性教训**: 权威口径脚本在 **native_int8_pack** 内(非 repo)。repo 的 int8 route 是**回退变体(632 行差异)**, 建慢核(s2_160 repo 11.44ms vs pack 8.75ms≈存储 8.70)。**必须用 pack 版**。
- **int8 ✅ 已接入+口径验证**: `measure_config_int8` 走 pack `native_int8_full_onnx_route`(真 dp4a 直接拓扑, 无需 tune)。验证 s2_160=8.75ms≈存储8.70(0.6%)。端到端 [32,64,128] int8=**3.80ms**(fp32 26.2→fp16~8→int8 3.8, 真 6.9× 加速), energy 用总功率法。ONNX 需在 `/exdata/.../s2_tvm/models/<label>_backbone.onnx`(route 硬编码约定)。
- **fp16 ❌→✅纠正: 我走错路线了, 正解见 `10_7_3_交接文档_FP16接入正解...md`**。★下面这段"metaschedule 受阻"的结论**作废**: 我复现的 47ms/39.5ms 恰是被废弃的 **fp16_true metaschedule 慢路线**(核验: fp16_true s2_160=47.53ms)。**训练表权威 fp16=11.64ms 来自 TIR 组卷积 TensorCore 重写**(`fp16_rewritten_tensorcore_full60`, 非 metaschedule), 用 `stage2_fp16_tensorcore_convblock_and_engine_probe.py --cast-fp16-source`。以下原文保留仅作"错误路线"记录:
- ~~**fp16 深查受阻(tensorcore WMMA tune 脆弱)**~~: 权威口径=`fp16_true_smoke`(pack, 权重 cast float16 + `MetaScheduleApplyDatabase(work_dir=/exdata/.../workdirs/<label>)` **复用已调 DB**, 不现调)。机理已完全查明并逐步验证:
  1. 空 DB → 建成 **fp32 级 47.2ms**(存储 fp16=11.64)。
  2. 用 artifact-worker **32-trial tune** fp16-cast ONNX 产 DB → apply → **39.5ms**(tune 生效但**未命中 tensorcore/WMMA**)。
  3. **128-trial tune** → 找到 schedule 但 apply 时 **CUDA illegal memory access 崩**(与 P0 lhc_17/s2_096 tuned 崩同类, 已知 tuned-schedule 脆弱性)。
  ⇒ fp16 11.64ms 需原 batch 的**确切 WMMA-tune 配置 + 崩溃重试**(batch002_**retry2** 名字即暗示反复重试筛掉崩溃 schedule), 该 tuner 不在已恢复脚本内。`Target.from_device` 得 H800 target(应含 sm_90 tensorcore), 故非 target 缺失, 是 tune 搜索/稳定性问题。
  **务实路线(建议)**: fp16 近无损、非 trade-off 关键轴 → **用 original60 现有 60 个 fp16 实测点当 surrogate 锚点**(select 已支持); 三臂消融用 **fp32(74 真测)+ int8(已验证可扩展)两条实测精度轴**。fp16 新宽度 latency 待 WMMA-tuner 复现后补。

## 停止点 / 下一步
- **停止点已达成** ✅ = P2.2 一轮真闭环(预测→真测→回灌→模型改善 32→18% MAPE)。
- 增量 follow-up(非停止点必需): ①修 energy 采样窗口后补 energy 轴前沿;②fp16/int8 走各自 rewrite 管线接入闭环;③多轮 SMBO(K=6×≤3 轮)推进 latency 前沿并补前沿 Top-K 的真 finetune AP 锚点。
- P2.3 三臂消融 Gap1(S0/S1/S2, `run_pqs_ablation.py`)= 论文级证据(下一阶段)。
