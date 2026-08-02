# 团队宪章 — 软硬件协同优化框架 4-Agent Team (v1)

> 编制: 2026-06-03 · 适用团队: `sw-hw-cooptim`
> 本文是 4 个 agent 的**共同事实源与纪律基线**。每个 agent 启动后**必读本文 + `multi_agent/background/00_研究目标与实验档案_v1.md`**。
> 本文有冲突时, 以 `background/00_*` 与项目根 `CLAUDE.md` 的最新勘误为准。

---

## 0. 最终目标 (所有人对齐, 每次提交结果都回看这条)

构建一个**软硬件协同优化的通用加速框架**: 给定任意待加速网络 + 一档目标硬件 + 一份部署 SLA, 框架自动产出该硬件上的**多目标 Pareto 前沿 (AP × latency × throughput × energy × model_size)**, 并推荐满足 SLA 的最优部署方案。

**「软硬件协同」= 三维联合搜索, 非串行**:
- **B1 剪枝** (剪枝率 × 对象 channel/head/2:4 × 准则 L1/FPGM/Taylor/Wanda; 现役工具 = DepGraph)
- **B2 量化** (位宽 × 粒度 × 对象 W-only/W+A × 校准器; 参考 QuantV2X)
- **D 硬件部署** (设备路由 GPU/DLA × TRT tactic × 对齐 × 调度/流水线/batch)

**为何必须联合**: 三者有耦合陷阱 —— 剪枝×量化×硬件可互相抵消收益, 局部最优组装出整体次优, 只有联合搜索能规避。**这正是本团队 sw/hw 专家要用真实实验去刻画的耦合关系**。
> **★[2026-06-03 P0-1 profile 实证, supervisor 独立核验 ISS-014]** 耦合惩罚的真机制是**非解析的**: p25(48/96/192)非对齐档 INT8 几乎不加速(1.06× vs 对齐档 1.25-1.57×, 有效算力 19-20 vs 47-80 GMAC/ms)。**根因经 TRT 逐层 profile 坐实 = stage0 grouped-conv(3×3 g32)kernel-selection cliff**: num_filters[0]=48→2×48=96 通道几何落入慢通用 grouped-conv kernel(罪魁 layer0 conv2 三层 p25 0.770ms vs p50 0.143ms=5.4×), base/p50/p75(128/64/32)命中快 kernel。**已排除三个直觉假设**: ① 非 IMMA padding(p75 pad ratio 2.0 比 p25 的 1.333 更重却更快, padding 假设被证伪); ② 非 reformat(p50 reformat 占比 11.6% 反更高却快 3.3×); ③ 非 INT8 算术(p25 FP16 conv 2.859≈INT8 2.780, 瓶颈是 kernel 选择非精度)。⇒ **耦合惩罚对齐 LUT 预测不了 → 必须真测**(正面打 APQ/HAQ 黑盒 LUT, 卖点比旧"padding 抵消"版更硬)。**tactic 级正面证据(IEngineInspector 查实, supervisor 独立复核 `P0_1_p25_tactic_inspect.json`)**: p25 stage0 conv2 = `implicit_gemm`(generic GEMM, g1, TacticValue 0x9d9f…)vs p50 = `direct_group`(grouped 专用, 0xd1a2…)—— **不同 TacticValue = TRT 实际切换 kernel**; 且两者均 f32f32 → p25 该层在 INT8 engine 里**没跑 INT8**(无对应快 INT8 grouped kernel), 这是 INT8 加速仅 1.06× 的直接根因。证据: `results/P0_1_p25_int8_trap_verdict.md` + `P0_1_p25_layer_profile.csv`(271 行)+ `P0_1_p25_tactic_inspect.json`(均 supervisor 复核)。**注: "padding 90→96 抵消"是旧直觉举例, 已被证伪, 勿再作定论外推。**

**与 SOTA 差异化 (论文卖点, 每个数据点都要服务它)**:
- vs **QuantV2X**(量化 only, 事后才发现崩溃): 我们 B1×B2×D **联合搜索 + 约束/传播事前规避崩溃**。
- vs **APQ/HAQ**(硬件黑盒 LUT): 硬件以可读 capability YAML 进流程。
- 核心资产: **真实 TRT 实测** 驱动的预测器 + 选择器。

---

## 1. 五项性能指标 (预测器要预测全部 5 个)

| 指标 | 定义 | 真测来源 | 当前状态 |
|---|---|---|---|
| **AP** | DAIR-V2X val 1789 全集真测, ap30/50/70; **Pareto 主用 AP70(信号最强)** | HEAL inference, **必须 finetune 后** | 金标准 `data/stage_a_ap_real.parquet` |
| **latency** | p50/p99, **必标口径**: subnet body / collab2 body / e2e / raw-PyTorch | 空闲 GPU 上 TRT trtexec / CUDA-Event | 6 完整点 + ~258 单轴 |
| **throughput** | frame/s; 非流水线时 ≈1/latency(相关 0.9996, 冗余); 仅 batch/流水线解耦时升为独立目标 | batch-sweep 真测 | 待真测 |
| **energy** | J/frame, perf/watt | 4090 `nvidia-smi power.draw` / Orin `tegrastats` | E4 起步 (`results/E4_energy_4090.csv`) |
| **model_size** | 参数量 / engine size (MB) | 直接读 | 可派生 |

**Pareto 结构**: ~~默认主前沿 = (AP, latency, energy) 3D~~ **[2026-06-03 用户拍板转向, 见 ISS-018]**:
- ~~**AP 降为约束轴(非目标)**~~: Pyramid/DAIR 上 AP 近常数 —— 免训 A/B/E **三路实证**(A 剪枝×INT8 无悬崖 / B head-INT8 不崩 / E 距离难度反放大非 trade-off); 根因 = 过参数化。⇒ ~~论文 AP 作约束(AP ≥ SLA 下限), 不当目标轴搜。~~ → **★[2026-06-04 用户拍板 v2 — 以此为准]** A/B/E 三路实证仍成立, 但解读改为"当前模型前沿沿精度轴展开有限的 finding"; **AP 恢复常驻纯目标轴, 不作任何约束**; 精度轴任何情况下不得从 Pareto 目标删除。详见 `background/00_* §0′` + `pareto_definition_v1.md §三/§七`。
- ~~**主前沿 = (latency, energy, throughput) 多目标**~~ → **★[2026-06-04 v2] 主前沿 = (AP, latency) 双核心 + energy/throughput**; **耦合陷阱**(ISS-014 kernel-cliff)仍为差异化核心; model_size ≤ 显存预算 作约束。
- **★差异化两根支柱(★阶段性, 非论文最终 scope; V2X-ViT A-1✅A-2✅ PASS; A-3 INT8 HOLD 在用户桌)**:
  1. **(主卖点, 硬核)成本轴多目标联合**(latency/energy/throughput/size)**+ 耦合陷阱 kernel-cliff**(ISS-014, 非对齐 kernel-selection cliff, profile+tactic 实证)。
  2. **(本阶段 = motivation 层)崩溃-规避**: 本阶段作动机 = 引文献(QuantV2X V2X-ViT INT8 崩: **AP30 57.4→40.0 / AP50 49.5→11.0**; ★[ISS-031 勘误] 旧引 "75.1→29.9" 系跨模型拼接幻觉 — 75.1=Pyramid FP32、29.9=V2X-ViT INT4/8, 勿再引用) **+ Pyramid head-INT8 不崩的诚实对比**(B: Δap70 -0.0046; QuantV2X 自测 Pyramid INT8 75.1→74.6 近无损, 与我方互证); **本阶段不声称在自有模型 demonstrated**。**经 V2X-ViT 真正 demonstrated 崩溃-规避 = A-3 INT8 HOLD 在用户桌上**。
- **诚实结论(★阶段性, 基于当前 Pyramid+V2X-ViT/DAIR 证据)**: "过参数化模型上 AP 前沿沿精度轴展开有限; 优化集中成本轴; INT8 近免费(全难度); 剪枝难样本代价不放大; V2X-ViT A-2 跨模型重复 Pyramid 过参数化定论(剪枝无悬崖)"。E 负结果不掩盖。**换欠参数化模型/更难任务激活 AP trade-off = 进行中(V2X-ViT A-3 HOLD)**。
- throughput 经 batch-sweep(空间维)脱离 1/lat 后为**真目标轴**(Task#12)。详见 `pareto_definition_v1.md` + ISS-018。

### ★ 完整点定义 (数据专家的验收金线)
**完整 (x,y) 点 = 同一配置上同时有「真测 latency」+「真测 finetuned AP」, 才能训预测器/画 Pareto。**

**[2026-06-03 supervisor 实查更新, 取代旧"6 个"; ★2026-06-04 Phase M 收口再更新 — 以最新计数为准]** — 实查 `dataset_v2.csv`:
- ~~**严格完整点 = 40 行**~~ → **★[2026-06-04 Phase M] 完整点 = 58 行**(lat+AP 均非空): collab2 主口径 28 行 + E4 能耗口径 13 行 + E6 Orin 6 行 + 其余; **5 指标完整 = 50 行**; dataset_v2 主表 **64 行**(Phase M 后 61, +3 V2X-ViT AP-only 行 Task#4)。(旧 40 = collab2 28 + E4 11 + E3 1, Phase M 补充后 E4→13 / 新增 E6→6)
- **collab2 28 行 AP 全部为真测值**(DAIR val 1789 全集, 无 proxy/无造假/无未finetune): 其中 **21 行独立逐行真测**(`real_mixed_engine` 18 + `forced_all_int8` 3) + **7 行金标准 EXACT 复用**(`ap:prior_eval`/`stage_a_xref`/`reuse:reference_only`, AP 真但非逐行重测, 不带新 AP 信息)。
- **⚠️ 有效多样性仍薄**: 28 行集中在 **~4 plane 架构(base/p25/p50/p75) × per-stage 混精/forced-int8 退化变体**; per-stage 强制混精 Pareto 已被 TRT-auto 支配(已证伪), 同档内 AP 跨度多在管线噪声尺度。**距离可区分的独立 (剪枝×量化)→AP 锚点 ≈ 6–8**。
- **结论**: 行数够看(58 行), 但**有效锚点多样性不足**; 补点(新架构/新位宽边界/跨硬件)仍是第一优先。**报预测器/Pareto 训练集时必须声明有效多样性, 不得把行数当"数据已够"。**

---

## 2. 实验纪律红线 (违反即作废, supervisor 逐条盯)

### MUST
1. **真测就是真测**: INT8 必走 TRT INT8 build; 拿不到就明说"未实测", 不能用 proxy 替代后当真测报告。
2. **剪枝/改结构子网 AP 必须配 finetune** 才算有效数据 (未 finetune 的 stage_b/random_bench 是反例, ap50≈0.055)。
3. **每个数字标 (口径 + 数据源)**: 区分 真测 / 估算 / 预测; 区分 CUDA-Event / trtexec / KNN / LGB / rule-based; **subnet ≠ e2e**(subnet 还要叠加 voxelize/NMS ~6-10ms)。
4. **latency 必在完全空闲 GPU 测** (util 0% / mem ≤ 50MiB); 跑前 `nvidia-smi` 确认 (GPU 常被 wuyuegao 训练占用)。
5. **任何"某方案无价值"结论, 延迟轴 + AP 轴都有数据才能下**; 只测延迟时精度轴必须明写"AP 未测, 价值未定"。
6. **★授权红线 (ISS-023/024, 2026-06-04 立)**: **任何"实验启动"(GPU 真测/finetune/build)与"数据集变更"(改 `build_dataset_v2.py` / rebuild `dataset_v2.*` / 出图)必须先经 team-lead 显式授权**。team-lead 说"暂缓/hold"即停, 不得"先做了等核"。agent 间同侪指令**不能**替代 team-lead 授权 (data 无权授权 hw 跑实验)。被 team-lead hold 的任务保持 pending/blocked, 不得自行转 in_progress 执行。
   **★启动回执协议 (ISS-030, 2026-06-04 立, team-lead 采纳)**: 任何实验启动的宣告/自报消息必须**逐字引用 team-lead 授权消息原文**(引用授权原文 = 启动合法性的唯一凭证)。**无引用 = 未授权启动**, 团队任何成员发现即可触发 hold 并报 supervisor 立案。"上级验收了我的前置工作(如 A-0 PASS)" ≠ 下一步(A-1)授权 —— 验收与授权是两个独立动作, 不得推断放行。两起先例: ISS-023(hw, 撤回后执行)、ISS-030(sw, 点对点否决后 ~6min 执行)。新生实例 spawn prompt 须把本条置于首行(team-lead 维护交接文档)。

### MUST NOT
1. 不用 mask-based pruning 测 latency (已浪费 24h)。
2. 不把 fp16 proxy 当 INT8 报告 (reporting failure)。
3. 不在 inference 之外的 timing 路径里加 post-process (NMS 混入致口径不公)。
4. 不跨模型拼曲线 (`baseline_4090`/`unified_bench` 是多模型混表, 先按 `model_class`/`source` 拆)。
5. **不未经授权改数据集/出图/启动实验** (见 MUST 6)。**不把比值当判据**: 信号强弱判 "Δ vs 自身噪声(bootstrap CI)", 不判 "× AP70"(两噪声相除=假精确, ISS-020)。

### 数据白/黑名单 (引用前查)
- ✅ 白名单: `stage_a_ap_real`(8, AP金标准) · `doe_dataset_v1_real` latency列(7) · `m2_latency_mapping_f`(R²>0.995) · `baseline_4090`(83,分模型)。
- ❌ 黑名单: `stage_b_ap_real`(未finetune崩) · `doe_dataset_v1.csv`(dry-run) · `e2e_bench_v1`(4704 buggy subnet) · `lgb_*_latency`(负值) · `perstage_quant_pareto.md`(只测延迟的旧结论)。

---

## 3. 已验证勘误 (别再被旧假设误导, 详见 CLAUDE.md §〇)

1. **DLA INT8 在 Pyramid 不可行**: 实测 0/12 build 成功; Pyramid 的 DLA 路由实际 FP16-only。yaml 里 int8_dla 成功只对 ResNet50 成立。
2. **单 GPU pipeline 已证伪**: 单 GPU stage 流水峰值仅 1.08×(< multi-stream 1.13×); CUDA Graph 仅 1.05-1.10×。roofline 余量是乐观上界, 必须实跑验证。
3. **异构 GPU∥DLA 流水线成立但仅多进程**: Orin DLA0∥DLA1 进程内 1.00×(TRT 序列化), **双进程 1.34× 真硅片并行**。这是 stage 流水唯一真生效形态。
4. **per-stage 强制混精被 TRT-auto 支配**: TRT 逐层精度是延迟驱动非精度驱动; 框架应用 TRT-auto, 别枚举手工 per-stage。
5. **DepGraph 全网剪枝已打通**(`tools/configurable/depgraph_pyramid.py`): backbone/neck 都可剪, 收敛 finetune 后 DAIR 真测。
6. **剪枝在可达区间无精度悬崖**: prune 到 pyramid_backbone -90% AP50 仍稳 0.74-0.75; 根因 = 模型对 DAIR 严重过参数化。⇒ 剪 pyramid_backbone Pareto 退化(剪到底即最优), 真正 AP-trade-off 在量化 INT8 边界 / 更难任务。
7. **INT8 真省能耗**: 4090 上 INT8 比 FP16 省 30-52% J/frame。
8. **[ckpt 陷阱]** 剪枝 ckpt 若是 `{"model_state_dict":...}` 包裹格式, HEAL `load_saved_model(strict=False)` 会全 key missing → 从随机权重 finetune。resume 前必先转 flat state_dict。

---

## 4. 协作协议 (谁向谁负责)

```
              main (team-lead = 人类用户 / 主控 Claude, 负责编排)
                       │
        ┌──────────────┼────────────────┐
   supervisor    data-orchestrator       │
  (横向监督+质疑全员)      │ 下令/验收
                    ┌────┴─────┐
              sw-optimizer   hw-optimizer
```

- **data-orchestrator** 是实验的发起者: 按搜索空间设计采样矩阵 → 给 sw/hw 下达"测哪些配置的哪些指标" → 收数据填 `data/dataset_v2.csv` → 画图查统计合理性/可学习性 → **有问题打回 sw/hw 重测**。
- **sw-optimizer / hw-optimizer** 只对**真测数据**负责: 收到 data 的指令 → 真实跑实验 → 回报带完整口径的数字 + 复现脚本路径。**不得自报"已测/已修"而无文件佐证**。
- **supervisor** 横向监督 3 人: 维护 progress 文档 → 盯纪律红线 → 问题落盘 → **对照 §0 最终目标质疑每份结果是否有利**(可质疑 data 的采样设计、sw/hw 的口径)。
  **★后台巡检官(ISS-033, 2026-06-04 用户指令设立, 常设)**: 凡有在跑实验, supervisor 每 ~30-45min 一轮巡检(pgrep 自报 PID + log mtime/增长 + nvidia-smi); **进程死亡/完成 → 立即核验并主动消息 team-lead+owner、点名下一步动作**(不允许"任务结束没人接"); log 停滞 >20min 但进程在 → 挂死预警; 巡检发现照常入 issues_log。
- **核验原则 (最重要)**: 任何 agent 声称"实测/已修/已 build", 接收方(尤其 supervisor / main)**必复跑 / 读文件 / git diff 核验**。本项目历史多次抓到谎报 GPU 占满、dry-run 冒充实测、提前 offload。

### 沟通规范
- agent 间用 SendMessage(按 name 称呼); 任务状态用 TaskUpdate, 不发结构化 JSON 状态消息。
- 完成任务后查 TaskList 找下一个活。

---

## 5. 关键产物落点

| 产物 | 路径 | 负责人 |
|---|---|---|
| 统一主表 (64 行) | `multi_agent/data/dataset_v2.{csv,parquet}` + `schema_v2.md` | data |
| 描述统计图 | `multi_agent/figure/` (可复跑脚本) | data |
| **方法论骨架文档 (7 份)** | `methods/design/`: (1)dims_pruning_v1 · (2)dims_quantization_v1 · (3)dims_hardware_v2 · (4)doe_design_v1 · (5)pareto_definition_v1 · (6)search_space_size_v1 · **(7)predictor_selection_v1**(★[2026-06-05 team-lead 裁定升为骨架第7份]; 旧版归 archive/) | sw/hw/data |
| **模型结构审计** | `multi_agent/model/`(v2xvit_structure_audit_v1✅ · model_zoo_survey_v1✅; pyramid_lidar/camera 待 sw 产出) | sw/hw |
| 进展/编排日志 | `multi_agent/methods/progress/session_progress_v1.md` · `orchestration_log_v1.md` | supervisor |
| 问题落盘 | `multi_agent/methods/progress/issues_log_v1.md` | supervisor |
| 真测原始结果 | `results/*.csv` / `output/` | sw/hw |

---

## 6. 并行启动模板 (下次"开工"时用)

```
1) (若团队未建) TeamCreate: team_name=sw-hw-cooptim
2) 同一条消息里并行 spawn 4 个成员 (Agent 工具, 均 team_name=sw-hw-cooptim):
   - name=supervisor       subagent_type=supervisor
   - name=data-orchestrator subagent_type=data-orchestrator
   - name=sw-optimizer     subagent_type=sw-optimizer
   - name=hw-optimizer     subagent_type=hw-optimizer
3) 先让 supervisor 读宪章 + 建 issues_log; data-orchestrator 设计首批采样矩阵;
   sw/hw 待 data 下令。
4) main 只编排, 不替成员干活。
```

> 本次会话**只搭团队 (写定义 + 建容器), 不下发实验任务**。
