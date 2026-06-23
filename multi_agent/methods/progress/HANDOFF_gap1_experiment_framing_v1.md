# HANDOFF — Gap1 / co-design 实验整理 + 待梳理框架问题 (v1, 2026-06-19)

> **定位**: 本会话(AP 轴纠错 → 公平曲线 → Gap1 首轮结果)的**状态固化** + **下一步要和用户一起梳理的框架问题清单**。
> **接手第一步 = 不要急着跑实验**(用户明确要求先梳理)。先和用户过一遍 §2 的开放问题(实验 section 含哪几部分 / 搜索空间怎么构建), 定清楚再动手。
> 衔接: 主文档 [HANDOFF_codesign_unified_v1.md](HANDOFF_codesign_unified_v1.md)(技术现状 Part I + 论文叙事 Part II) · Gap1 设计 [gap1_joint_vs_serial_design_v1.md](../design/gap1_joint_vs_serial_design_v1.md) · 文献 [study_joint_search_methods_v1.md](../../references/study_joint_search_methods_v1.md)。

---

## §0 本会话轨迹 (含一次错误结论与纠正, 诚实留痕)

1. 整合两份旧 handoff → unified; 研读 ALT/CHaNAS/AutoTVM(CHaNAS 原文已精读, LCTES'21); 写 Gap1 设计文档。
2. 确认 V2X-ViT 就绪度 → 测得"剪枝↑AP / sim INT8 近无损" → **一度错误下结论"DAIR AP 轴塌缩, 载体是数据集问题"**。
3. **用户质疑(正确)**: 剪枝量化怎么可能不伤精度。复测纠正:
   - **剪枝真崩 AP**(V2X-ViT p50 raw AP50 0.710→0.025), **finetune 恢复**(= 过参数化使 finetune 能恢复, 非"剪枝免费")。
   - **"剪枝↑AP" = baseline 欠训的训练协议 confound**(不是 harness bug); 修法 = iso-budget(给 base 同 finetune)。
   - **INT8 sim 不可信**(FP32 累加 / 注意力 matmul proxy / 12 层未覆盖, 系统偏向无损); 真实代价待真 TRT INT8。
   - AP harness 本身没坏(原始剪枝 p50 测出 0.025 崩 = harness 能捕捉崩溃)。
4. 用户拍板**用 Pyramid + CoDriving 两条公平曲线推进**。
5. H800 全空 → 跑 Gap1 schedule 轴首轮: **联合(对齐感知选 p50)比串行(剪到 25%→trap25 kernel-cliff)快 7.18× @ −0.026 AP70**; pad 局部救援负结果证对齐是全网属性。
6. 用户喊停: **先梳理框架再继续**。← 当前位置。

---

## §1 已确立 (corrected, 可作为论文素材)

**A. AP 轴(公平, 2 模型)— 不塌缩, 剪枝平滑伤 AP**
- Pyramid(stage_a, base=官方收敛 ckpt): AP70 单调 0.631→0.590→0.564→0.530(span 0.10); 配 TRT fp16 lat 1.27/2.87/1.01/0.78ms。★p25=[48,96,192] 被 base 严格支配(对齐 trap)。
- CoDriving(iso-budget): 公平 base AP50 0.626 > 所有剪枝(0.586-0.618)→ 剪枝真伤 AP(浅)。
- 教训: 看 AP70 不看 AP50(span 太小); 公平对比必 iso-finetune baseline。

**B. 对齐耦合(精炼根因)**
- Pyramid grouped conv(g=32)需 **in_per_g = 2·num_filters/32 ∈ 2^k** 才命中快核; num_filters=48 → in_per_g={3,6,12} 全非 2 的幂 → kernel-cliff(tuned 仍 21615µs, 救不回)。
- **全网属性**: 局部 pad stage0(48→64)无效(stage1/2 仍 6/12)→ 必须整网联合搜。
- 跨平台一致: 4090 TRT 与 H800 TVM 都把 trap25 打成被支配点。

**C. Gap1 首轮(联合>串行机制实证)**: S1 串行 21615µs vs S2a 联合 3011µs = 7.18×@−0.026 AP70。`results/gap1_schedule_lut.json`。**边界**: 对齐陷阱罚分(非泛化), backbone-only(Amdahl), 仅剪枝×schedule 轴。

**D. 文献骨架**: ALT(layout↔loop 联合, cross-exploration)/ CHaNAS(arch×schedule 联合, W/WO 消融 1.68× iso-acc, 分解 R·B·S+R^B, divisible-split=对齐他证)/ AutoTVM(rank loss / XGBoost / 进化退火)。

**E. 工具链**: TVM relax 导入+对齐+MS tune(H800, 无 INT8 pass); fusion neck 不可 trace BLOCKED; searcher_v0(随机采样+传播+约束过滤, **无 schedule 轴/评估器/Pareto**, D 仅 GPU/DLA 路由)。

---

## §2 ★待梳理的框架问题 (下次讨论的 agenda; 每条给"现状 + 我的提议 + 待定")

### 2.1 这个"实验 section"到底包含哪几个部分?
候选组件(待用户确认哪些必需 / 哪些 nice-to-have / 边界划哪):
| 组件 | 内容 | 现状 |
|---|---|---|
| E1 耦合刻画 | 对齐×schedule(Pyramid trap)+ prune×quant(CoDriving INT8 ΔAP 随剪枝放大)+ 跨模型(grouped 显耦合/标准可分离) | 部分(对齐已实测; 跨模型 2 点) |
| E2 单轴 Pareto | 公平剪枝曲线 × latency(+ energy) | 剪枝done; 量化轴待真 INT8 |
| E3 联合 vs 串行(Gap1 payoff) | 在搜索空间上证联合 Pareto 支配串行 | 首轮机制(对齐 trap)done; 需全搜+量化轴 |
| E4 条件化判据 | 何时联合 / 何时可分离(model-dependent) | 2 模型; 是否需第 3 个? |
| E5 瓶颈再定位 | Amdahl, backbone vs fusion, RSU/车端 | done |
| E6(NICE) | 闭环驾驶分 | 未接 |
> **待定**: 论文主线 = E3(联合>串行框架)为主, 还是 E1+E4(刻画+判据)为主? 这决定 E3 要做多重(全搜 vs 机制 demo)。

### 2.2 搜索空间到底怎么构建? (核心)
- **P(剪枝)轴**: per-stage num_filters。★**合法网格由对齐约束定义**: in_per_g=2·num_filters/g∈2^k(Pyramid grouped); searcher_v0 已有 ALIGNED_PRUNE_RATES/NEAR_ALIGNED。**待定**: 离散锚点(base/p50/p75…)还是更密? 对齐是硬约束(只搜合法宽度)还是软(让搜索器"踩坑"以展示耦合)?
- **Q(量化)轴**: 精度 per-module {FP16, INT8}(+粒度)。**待定**: 需真 INT8 测量(TRT/ModelOpt)才有意义; 是否纳入本轮 / 还是先只做 P×schedule?
- **D/schedule 轴**: TVM schedule(tile/loop/fusion), 每个(宽度,精度)用 MetaSchedule 搜; 或 TRT-auto(串行基线)。**待定**: schedule 是"default vs tuned"二元, 还是真在 tile 空间多点搜?
- **目标函数**: **待定** — (AP70, latency, energy) 多目标 hypervolume, 还是 CHaNAS 式 constrained-single(max AP s.t. lat<阈值)产 iso-AP-latency? 提议: **两者都报**(iso-AP-latency 当 legible headline + hypervolume 当严谨)。
- **分解(防爆炸)**: CHaNAS R·B·S+R^B → 按 stage 预调度建 schedule-LUT + 加性 Lat_net。提议采用。
- **载体**: Pyramid(grouped, 耦合)+ CoDriving(标准, 可分离); 平台 H800 TVM(relative 坐实)/ 4090 TRT(边缘绝对); 数据 DAIR(AP 用 finetuned 模型)。

### 2.3 联合 vs 串行协议的精确定义
- 现 gap1 设计的 S0/S1/S2(§1)。**待定/要拍死**: "串行"先优化什么? (a) 纯按 AP/压缩率选 P, 再交给编译器调度(= 现 S1, 落 trap25); (b) 还是别的。"联合"看到什么反馈?(schedulability → 上游 P 决策)。要把这个模型讲清楚, 否则 7.18× 易被质疑"不公平对比"。

### 2.4 模型/平台/数据集口径 (铁律, 别违)
- 不拼跨平台单一 e2e(H800 TVM / 4090 TRT / Orin 分开报); AP 平台无关可跨用; latency 必空闲卡。
- backbone-only ≠ e2e(Amdahl 稀释); 报 e2e 影响时必须折算。

### 2.5 开放决策清单 (要用户拍板的)
1. **量化轴是否本轮纳入?** 若纳入须先解决**真 INT8 测量**(TVM relax 无 INT8 pass; 选 TRT 4090 / TensorRT-ModelOpt / pytorch-quantization)。
2. **V2X-ViT 是否做第 3 载体?** 需 iso-budget 重训 base(GPU 工程); 还是 2 模型够。
3. **E3 做到什么程度?** 手挑三臂机制 demo(已有)够, 还是要真·联合搜(NSGA over 宽度×schedule)证搜索器自己找到对齐前沿。
4. **目标函数**: 多目标 hypervolume vs constrained-single iso-AP-latency(提议都报)。
5. **e2e 落地**: 是否补 Orin 整段 e2e(消跨平台拼接窟窿)/ 闭环驾驶分(NICE)。

---

## §3 接手第一步
1. **和用户过 §2**(尤其 2.1 主线定位 + 2.2 搜索空间构建 + 2.5 决策清单)—— 不跑实验, 先把"实验 section 的骨架 + 搜索空间定义"讨论定。
2. 讨论结论回写 gap1 design(把搜索空间/协议/目标函数定死)+ 必要时改 searcher_v0(加 schedule 轴 + 评估器 + Pareto = 它 docstring 说的 Stage 4)。
3. 再据定好的框架排实验(H800 当前全空, 是 schedule 轴实验的窗口)。

## §4 资源指针 (接手即用)
- 结果: `results/{gap1_schedule_lut, codriving_isobudget_verdict, v2xvit_ap_corrected, v2xvit_int8_ap_sim}.json` + `data/stage_a_ap_real.parquet`。
- 脚本: `scripts/phase2/{gap1_run_v2, gap1_pad_rescue_v3, s2_2e_e2e, eval_v2xvit_int8_ap_sim_b, eval_v2xvit_ap_corrected}.py`。
- H800 TVM env + 资产: `/exdata/jichengzhi/tvm310/bin/python` + `/exdata/jichengzhi/s2_tvm/models/{base,p50,trap25}_backbone.onnx`(+codriving_cache)。SSH `-p 30001 jichengzhi@222.95.84.215`(12345678)。
- 代码基底: `framework/searcher_v0.py`(待加 schedule 轴)。
- memory: `project-dair-ap-axis-collapse`(AP 轴定稿)+ `project-hwsw-codesign-route2` + `project-codriving-optimization-pilot` + `project_heal_resnext_width_formula`(in_per_g/width 公式)。
