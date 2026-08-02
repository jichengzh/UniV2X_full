# 11_7_14 交接文档 — Phase 2.A 多框架对比(★标题过时,见顶部撤回横幅)

> ⚠️⚠️⚠️ **[2026-07-06 撤回横幅 — 文件名"决胜发现TVM追不上TRT"是错的,已撤]**
> 用户纠正 + 我承认: 本文档原写的"**决胜发现 TVM 后端追不上 TRT 7.4×**"是**假结论**,我违反了"别过早断言不可能"——从一次 metaschedule 崩就断言 TVM 不能,却没用上**我们自己 6_29 已做出的 group-conv im2col TensorCore rewrite**(单层 7-14×,full-engine 1.5×)。metaschedule 崩是它直接 WMMA group conv(OOB),我的 6.52ms 基准是 1x1-only rewrite(6_29 已证 1x1 非加速解),**根本没套 group-conv rewrite**。⇒ **"TVM 追不上 TRT" 未证,gap 是"我没优化好"不是后端硬伤**(ALT 2210.12415 e2e 也显示 TVM vs TRT 无大 gap)。§2/§4/§9 相关结论均已就地更正。正确下一步 = 把 group-conv im2col rewrite 套到 base backbone H800 真测再谈 gap。

> 状态: **本轮实验全部收尾暂停**,等用户方向。
> 日期: 2026-07-05(撤回更正 2026-07-06)。
> 上游: `10_7_14_交接文档_Phase2启动计划_多框架对比优先.md`(Phase 2.A 计划) + `9_7_14_..._Phase1完成_SMBO框架执行全记录_v1.md`(Phase 1 收口) + **`6_6_29`/`9_6_29`(group-conv rewrite 已有解,本轮遗漏)**。

---

## §0 接手须知(自包含)

- **背景**: Phase 1(Pyramid-LiDAR co-design 搜索,joint P×Q NSGA-II)已收口,真前沿=对角线(过参数化)。Phase 2.A = **多框架对比**,证明我们方法有效——把我们的 TVM co-design 前沿对比 TRT / AutoTVM / Ansor 等现成优化器。
- **本轮核心事件**: 用户拍板"先全量 tune TVM 决胜"→ 跑完发现 **TVM 后端在 Pyramid grouped-conv 上硬慢 TRT 7.4×(且快 schedule 崩)**→ "our-TVM 打过 TRT" 不成立 → 用户拍板转 **Option A 双 backend(搜索方法跨后端)**。
- **口径铁律(全程遵守)**: backbone-subnet(pyramid_backbone resnet,3 多尺度输出,无 deblocks/shrink);input_hw=[128,256];batch=2;latency=p50(CUDA-event/time_evaluator);energy=watt_avg×lat(NVML)。**同 ONNX / 同 shape / 同 batch / 同 device 才可比。**
- **TRT 纪律澄清**: 本轮 TRT/AutoTVM/Ansor 作【对比 baseline=要打败/对标的对手】,**不是**我们采用的后端;不违"框架已全迁 TVM 非 TRT"铁律(该铁律禁的是把 TRT 当**前进方向的后端**)。

---

## §1 本轮目标(Phase 2.A)

按 10_7_14 计划: L-A(后端质量·iso-config)/ L-B(co-design 价值·主结果)/ 两消融(compression-only 证 S 价值、serial 证联合 vs 串行)。本轮实际推进 = 任务0(TRT harness)+ L-A 决胜探测 + 两消融真测口径升级 + 前沿点 AP 全量微调 block1 启动。

---

## §2 ★★★决胜发现:TVM 后端在 Pyramid grouped-conv 上追不上 TRT

**iso-config base fp16,同一 ONNX(`${V2X_DATA_ROOT}/s2_tvm/smbo/models/smbo_64x128x256_backbone.onnx`),同口径,全真测**:

| 后端 | latency (p50) | 说明 | 数据源 |
|------|------|------|--------|
| **TRT-FP16** | **0.88 ms** / 0.29 J | 能跑,复现稳 | trt_profile_v1.py 真测 |
| TVM 1x1-only rewrite(**未套 group-conv rewrite**) | 6.52 ms | 6_29 已证 1x1 不是加速解 | 框架 fp16 route |
| TVM metaschedule 64-trial | 40.4 ms | 未套 rewrite | tvm_fulltune smoke |
| TVM metaschedule **4000-trial**(tune 65min) | **崩** | metaschedule **直接**对 group conv 做 WMMA → OOB(GPU5+6 两次复现) | tvm_fulltune full |

**★[2026-07-06 用户纠错 + 我承认错误 — 上面这版"决胜结论"是错的,已撤]**:
- 我原写"TVM 后端硬伤/追不上 TRT 7.4×"是**假结论**,违反 memory `feedback-no-premature-impossible`:从一次 metaschedule 崩就断言"TVM 不能",没穷尽我们**自己已有的解法**。
- **真相(6_6_29 / 9_6_29 已证)**: grouped 3×3 conv 在 TVM 里**可以 TensorCore 加速** —— im2col→batched MatMul TensorCore→NCHW restore,单层 `fused_conv2d4` **7.64×** / `fused_conv2d6` **14.24×**;full-engine group-conv rewrite **33.44→22.26ms(1.50×,wmma=72)**。
- metaschedule 崩 = 它**直接**对 group conv 做 WMMA(OOB);我们的 **im2col rewrite 路径恰好绕开这个崩点**,而我基准里那个 6.52ms 是 **1x1-only rewrite**(6_29 已证 1x1 不是加速解),**根本没套 group-conv im2col rewrite**。
- ⇒ **"TVM 追不上 TRT" 未证。** 我拿"没优化好的 TVM"去比 TRT。**正确做法**: 把 group-conv im2col TensorCore rewrite 套到 base [64,128,256] backbone、H800 上真测,再谈 gap。ALT 论文(2210.12415)e2e 也显示 TVM vs TRT 无大 gap——支持"gap 是我没优化好"而非后端硬伤。
- **遗留未闭(6_29)**: rewrite 的 output2 数值漂移(mean_rel ~0.18,callsite-aware 未完全闭);H800/sm90 未重跑;AP smoke 未做。这些是**待完成的工程**,不是"不可能"。

---

## §2.1 明确结论:TVM 与 TRT 不应有大 gap(文献 + 我们自己前期工作双证据)

### 文献证据 — ALT(Xu et al., arXiv:2210.12415, "ALT: Boosting DL Performance by Breaking the Wall between Graph and Operator Level Optimizations")
- **§7 EVALUATION 原文**: *"Besides, **Ansor outperforms** Tensorflow Lite [1] and other hardware-specific compilers [44, 83] such as OpenVINO [34] and **TensorRT** [52]. Thus, we do not include them as baselines here."*
  → **TVM 家族的 Ansor(auto-scheduler)已优于 TensorRT**;ALT 作者因此**连 TRT 都不放进 baseline**(嫌其弱,直接用更强的 Ansor 作参照)。
- **ALT 在 Ansor 之上再提升**: 单算子平均 **1.5×**、端到端 **1.4×**(NVIDIA GPU 上 e2e **1.39×**,Fig 9b/10b;摘要 + §7.2)。
- ⇒ **调好的 TVM 家族编译器 ≥ TRT**。我先前"TVM 追不上 TRT 7.4×"与文献直接矛盾,**再次坐实那是"我没调好"而非后端能力上限**。

### 我们自己前期工作证据(6_29 系列,本轮遗漏,补入 + 加引用)
| 来源文档(progress/6_29/) | 关键结论 / 启发 |
|---|---|
| **`6_6_29`**_FP16Rewritten1x1FullEngine结果与3x3下一步 | grouped 3×3 conv **im2col→batched MatMul TensorCore→NCHW restore**: 单层 `fused_conv2d4` **7.64×** / `fused_conv2d6` **14.24×**;full-engine group-conv rewrite **33.44→22.26ms(1.50×,wmma=72)**。★**1x1-only rewrite 非加速解**(甚至更慢)——我本轮的 6.52ms 基准正是踩了这个坑。 |
| **`7_6_29`** / **`8_6_29`**_FP16INT8加速原因差异与端到端图内加速计划/收口 | FP16 vs INT8 加速**根因区别** + 端到端**图内加速**执行/收口计划(不是逐算子孤立加速,是整图 rewrite)。 |
| **`9_6_29`**_FP16INT8加速根因区别与端到端图内加速执行计划 | 遗留: FP16 output2 数值漂移 **mean_rel~0.18**(callsite-aware 未闭,FP32 累加缓解不消除);INT8 native route 已 build/run(float32/quantize/dequantize=0);H800/sm90 gate check 待重跑;AP smoke 待做。 |

### ★★★[2026-07-06 真测结果 — 我第二次结论"套上 rewrite 就无 gap"也被打脸]

**把 6_29 的 group-conv im2col TensorCore rewrite 套到 base [64,128,256] backbone、H800 GPU4 真测**(probe 泛型 shape-pattern 发现,零改代码;500-call 测量;经主控独立读 json 核验):

| 配置 | lat_p50 | 备注 |
|------|---------|------|
| TVM default(无 rewrite) | **53.26 ms** | vs TRT 60.5× |
| **TVM group-conv TC rewrite**(合法,`tensorcore_gate=true`,180 WMMA,5 个 group-conv primfunc 正确改写,8.18× vs default) | **6.512 ms** | **vs TRT 7.40×,gap 未闭** |
| TRT-FP16 | 0.88 ms | |

- **我先前的"6.52ms 是 1x1-only 假 proxy、套上真 rewrite 就无 gap"——被真测否定**:合法的 group-conv TC rewrite(真 WMMA、8.18× over default)真测 **6.512ms**,与先前 6.52 essentially 同值。**gap(7.4×)是真的,rewrite 没闭掉它。**
- 诊断(device-synced 逐 primfunc,~10% overhead,仅定性): **5 个已 TC 改写的 group-conv primfunc 本身占 ~75% latency**,gap 不是残留未张量化的 1x1 层(~25%)。**疑似主因(未进一步验证)**: groups=32 使 per-group GEMM 的 **K 很小(36/72/144)**,WMMA tile 欠利用;且**固定 schedule 未 autotune/未 shape-specialize**。与 6_29 自己的发现一致(孤立层 7.6-14.2× 但 full engine 仅 1.5×)。
- 正确性: output0/1 max_rel 0.16%/5.2%;**output2 mean_rel 17.4%**(复现 6_29 已知 ~0.18 漂移,pre-existing 非新问题)。

### 诚实的三方图景(★别再单方下结论)
1. **真测事实**: 我们当前最优 TVM(group-conv TC rewrite,8.18× over default)= **6.51ms,7.4× off TRT 0.88ms,gap 真实、未闭**。
2. **文献预期**: ALT 报告 Ansor(TVM 家族)> TensorRT ——**但那是 Ansor 的完整 auto-schedule(layout+loop 联合自动调优)**,不是我们的**固定手写 im2col rewrite**。即"TVM 原则上能 ≥ TRT",但**要靠 auto-tuning,不是我们现在这个固定 rewrite**。
3. **开放问题(未证)**: 把 per-group 小-K GEMM 的 schedule **auto-tune / shape-specialize**(或修 metaschedule 直接 WMMA 的 OOB 崩)能否闭掉 7.4×?**未知,是真研究工作**。metaschedule 直搜崩、手写 im2col 到 6.5ms 触顶——两条路都还没拿到 TRT 级别的数。

**⇒ 结论(诚实版,交用户定方向)**: "TVM 追不上 TRT" 未证(文献反例在),但"套上我们现成 rewrite 就无 gap" **也被真测否定**。真实状态 = **当前工程能力下 TVM 6.5ms vs TRT 0.88ms,7.4× 真 gap;闭合它需要尚未做出的 auto-tuned 小-K group-conv schedule**。是否为此投入(vs 承认 within-TVM gap / 双 backend / 换 vehicle),交用户判断。

---

## §2.2 ★多 agent 探索 + 批判者 adversarial 复评的最终综合(2026-07-06,supersede 前面所有反复)

用户要求:多 agent 探索 + 维护批判者。起了 3 个 agent(公平审计 / auto-tune 可行性 / 批判者),批判者独立从 .npy 重算误差、从 nsys 原始 sqlite 重算 kernel-sum、直接解析 trt_layerinfo.json 核验,产物 `${V2X_DATA_ROOT}/s2_tvm/la_fulltune/{fairness_audit,tvm_autotune_feasibility,critic_review}_result.json`。

### 经批判者 CONFIRMED(站得住)
1. **测量口径公平**(两边都核实):不是测量假象。TVM kernel fraction 99% / TRT 80%,修正到纯 kernel 反而 gap **更大 8.79×**(TVM 6.488ms / TRT 0.738ms,批判者从 nsys 原始数据精确复现)。**"VM overhead 造成假 gap"假设被证伪。**
2. **TRT 真算 grouped conv**:批判者在 `trt_layerinfo.json` 直接找到 `Groups=32` 字段(比 E1"间接确认"更硬);50 conv 全 fp16,3 输出齐,无 int8/dead-layer。
3. **TVM rewrite 有真 bug**:批判者从 .npy 重算 output_2 误差 **18.2%**(rewrite vs default,无 TRT 在环)→ **6.51ms 不是个正确的数**。

### 经批判者 OVER-CLAIMED(纠正)
1. **8.79× gap 描述的是"我们这一个手写 im2col schedule vs 未知的 auto-tuned 天花板",不是"TVM 后端 vs TRT"**。措辞不能滑成后者。
2. **E1"ALT 持平只在 dense conv"是错的挡箭牌**:批判者查证——TRT-parity 实来自 **Ansor(OSDI'20)**,其 eval 含 **MobileNetV2(depthwise/极端 grouped conv 主导)**,Ansor 在那也 **匹配/超过 TRT**。⇒ **"grouped conv 对 TVM 本质更难"被推翻;auto-tuned TVM 很可能能打平 TRT**(depthwise 比 groups=32 更极端都行)。**这实际支持用户的直觉。**
3. **E2"不是性能天花板"需强 caveat**:tuned schedule 跑不起来 = 天花板**根本没测到**;其引的 **1224us 是 per-task 孤立和**,用 **6_29 铁证反驳(孤立层 7-14× → 组合后仅 1.5×)**,孤立测量极可能严重高估,**别当可达值**。

### 真正确立的事实 vs 真正的未知(★诚实底线)
- **确立**:① 测量公平;② 当前"最优 TVM"(手写 rewrite)= **6.51ms 但 output_2 错 18%**,既不快也不对;③ auto-tune 路径**被一个真实、可复现、非 TensorCore-specific 的 infra 崩**挡住(SIMT 也崩,疑 Relax 内存 planner vs grouped-conv auto-tile 组合)。
- **未知**:**今天没有任何一个"正确 + 快于 6.51ms"的 TVM 数**。auto-tuned TVM 的真实天花板**没测到**(崩挡住)。
- **⇒ 问题定位(答用户"问题所在")**:gap **不是**测量不公,**不是**后端能力上限(Ansor 在 depthwise 已证 TVM 能打平 TRT),**而是**我们 TVM 路径上的**两个工程问题**:(a) 手写 rewrite 有 bug 且触顶 6.5ms;(b) auto-tune 被一个**可能可修**的 TVM infra 崩挡住(约束 tile 到 group-divisible / 手动 apply trace 绕开 ApplyDatabase,像 hand-rewrite 那样)。
- **⇒ 决定性下一步**:**修 auto-tune 组合崩 → 拿到"正确 + auto-tuned"的 TVM latency**。文献(Ansor≈TRT on depthwise)预测它应逼近 TRT;但必须真跑出来。**在拿到这个数前,7.4× 不可作为干净数据引用。**

### 批判者也标的证据瑕疵(记录)
- E2 的"SIMT-only 也崩"缺一份 time-matched 崩日志(同类崩存在但没对上那次具体 run)——结论方向对但该点证据略弱。
- 目录里有个 stale `trt_kernsum.csv`(在 profiled window 内 build engine 的污染产物,E1 没用它报数,但是 clutter,建议清)。

---

## §3 已核验的实验结果(独立 stat + 真读数值,非自报)

### 3.1 TRT baseline(任务0,✅ 经核验)
| precision | lat_p50 | lat_mean | lat_p99 | watt_avg | energy_j | build | engine |
|-----------|---------|----------|---------|----------|----------|-------|--------|
| TRT-FP16 | 0.88 ms | — | — | — | 0.29 J | ✓ | — |
| **TRT-INT8-PTQ** | **0.835 ms** | 0.860 | 0.956 | 265.9 W | **0.222 J** | ✓ | 8.12 MB |

- INT8 vs FP16: 仅省 **~5% latency / ~24% energy**——**非**预期的大 INT8 加速。印证项目既有结论(TRT 逐层精度是**延迟驱动非精度驱动**,小 backbone-subnet 上多数 tactic 停在 FP16 附近)。**如实报告,未调整。**
- **DAIR 真实校准集**: 123 个 `.npy`(hook `HeterPyramidCollab.pyramid_backbone.get_multiscale_feature` 抓 resnet 输入,shape (2,64,128,256) fp32,无 NaN,nonzero≈0.66),`${V2X_DATA_ROOT}/s2_tvm/la_fulltune/dair_calib/`;manifest 记录 123 个 val 索引供未来 PTQ-AP no-peek 排除。提取脚本 `trt_int8_dair_calib_extract.py`。
- caveat: ① INT8 加速小,若"TRT baseline 要显得有竞争力"需后续 `--dumpLayerInfo`/polygraphy 查逐层精度;② calib 与未来 PTQ-AP eval 同 DAIR val split(1789),123-index manifest 是缓解非 split 级保证。

### 3.2 两消融真测口径升级(✅ 经核验,harness 齐)
Agent B 把 `run_pqs_ablation.py` 的 **proxy `UNIFORM_INT8_SPEEDUP=1.449`**(stage0-only 微基准常数,均匀施加)升级为**逐 width 真测比**:
- 新文件(proxy 版未动): `framework/q_lookup_real.py`(RealQLookup,逐 width 真测 fp16/int8_tc 比乘到 ablation 自己的 fp16 lat)、`scripts/phase2/build_q_int8tc_real_ratios_v1.py`→`results/q_int8tc_real_ratios_v1.json`、`framework/run_pqs_ablation_real_v1.py`(默认 `enforce_int8_buildable=False`——2026-07-03 已证 int8_tc 每 width 可 build,dp4a 墙被证伪;`uniform_int8_speedup=None` 诚实回退)。
- **真口径三臂(seeds12/budget60/pop8)**: **A-joint 100.0% > A-serial 86.5% > A-noS 49.6%**(vs proxy 85.3%/56.7%——serial 更近 joint、noS 更远,与更弱/非均匀真比一致)。**Wilcoxon joint-vs-serial p=4.88e-4**(与 proxy 同显著)。`lat_source`: 33 exact-fp16 / 6 real-int8_tc / 25 诚实未测中性——**零捏造**。
- **覆盖 4/9 grid**(p75/p50/trap25/base,真比 1.055–1.226×,弱于 1.449 proxy);**缺 5/9**: pad64/mix_b/mix_d/s1_64/s2_128。
- ★**关键坑**: 3 个 W_g/P_g 耦合对的 **P_g 成员(pad64/s1_64/s2_128)全在缺测集**→`detect_q_rank_flip_pairs` 真数据下返回 **0 对**,Q-rank-flip 定量 claim 暂无法验证。**补测 pad64 单点即解锁 1/3 对**(它与已覆盖的 trap25 配对)。
- 补全计划: 跑 5×2 个 `measure_config.py --width ... --precision {int8_tc,fp16}` → 重建 ratio json(9/9)→ 重跑 `run_pqs_ablation_real_v1` → 耦合对自动填充。

### 3.3 前沿点 AP 全量微调 block1(✅ 4/4 全完成,经独立 json+ckpt 核验)
**协议(严格 gold stage_a)**: `structural_prune_pyramid.py`(L1,wpg=4,groups=32)→ flatten init ckpt → patch epoches=31(init@23 → 8 finetune epoch)→ `train_ddp.py --half` 收敛 → PyTorch 原生 `inference.py` 在 DAIR val **n=1789** eval(**无 TRT**),AP 从 `eval_intermediate_*.yaml` full-precision 真读。

| width | gold AP70 | AP50 | conv | vs base 0.6313 | 类型 |
|-------|-----------|------|------|----------------|------|
| **[24,32,64]** (w1=32) | **0.5961** | 0.7496 | @31 | −0.035 | minimal-w1 |
| **[32,32,96]** (w1=32) | **0.5973** | 0.7524 | @29 | −0.034 | minimal-w1 |
| **[64,32,96]** (w1=32) | **0.5927** | 0.7482 | @27 | −0.039 | minimal-w1 |
| **[56,64,64]** (w1=64) | **0.5948** | 0.7560 | @27 | −0.037 | **唯一非 minimal-w1** |

全 4 点收敛,n=1789 核验(samples 0..1788),ckpt flat 格式 `${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_b4expand_mn_*_2026_06_20/`。base 全模型 [64,128,256] gold AP70=0.6313。汇总 `results/phase1_minneck_probe/front_ap_finetune_block1.json`。

#### 数据记录(供框架/surrogate 自学,★不手搓规则)
- 4 点真测 AP70 在 0.5927-0.5973,比 base 0.6313 低 ~0.034-0.039。**这只是真测数据点**,喂给框架让它自己学 width→AP 趋势。
- ★[用户纠正 2026-07-06]: **删除**先前"推翻性发现 / 无地板 / 与 §10.2 张力"的整段编排。原因:① 这跟论文叙事**无关**——不存在"揭穿假前沿"一说(用户已多次强调),框架的价值就是**自己从数据学这种变化**;② 我**不该手搓**"minimal-w1 塌到地板"这类经验规则再拿它当选点/叙事依据,而且我手搓的规则**经常错**(这次预测就被真测打脸)。我的职责=产出正确真测数据,不是替框架编规则。

#### 发现的 bug(已绕过,建议修)
`phase1_minneck_ap_probe.pytorch_ap_eval()` 漏设 `PYTHONPATH=HEAL` → `ModuleNotFoundError: opencood`,probe 自带 AP 返回空 `{}`(且其 regex 只截 2 位小数)。**微调本身不受影响**(train step 自设 PYTHONPATH)。Agent 用修正版 `gold_ap_eval.py` 读 full-precision yaml 重算。**建议下个 block 前修 probe 的 eval。**

---

## §4 [已作废] Option A 是建立在"TVM 追不上 TRT"假前提上的,前提已撤

> ⚠️ **本节作废**: 当时是我拿"TVM 追不上 TRT 7.4×"(假结论)逼出的战略岔口,用户在**错误信息下**选了 Option A(双 backend)。**前提已撤**(§2),故 Option A 是否仍要采纳**需用户在正确信息下重新判断**。
> 正确的前序动作 = **先把 group-conv im2col TensorCore rewrite 套到 base backbone、H800 真测 TVM 的真实速度**,拿到"优化好的 TVM"再决定 backend 立场。若优化好的 TVM 逼近 TRT,则"我们方法(TVM 后端)"叙事根本不用退让,双 backend 也非被迫。
> (Option A 的技术内容——搜索方法 backend-agnostic / TVM 保 S 轴 / TRT 作对标——本身不坏,可作为选项之一保留,但**不再作为"被 TVM 硬伤逼出的唯一出路"**。)

---

## §5 踩坑记录(本轮)

1. **TVM tune/build 与 measure 必须 .so 隔离进程**: `tvm.compile` 同进程 measure 崩 CUDA_ILLEGAL_ADDRESS;正解=`ex.export_library(.so)`→另进程 `tvm.runtime.load_module`。(memory `feedback-tvm-tune-apply-fresh-workdir` 已警,本轮再验。)
2. **LD_LIBRARY_PATH 必须 bash 预设**: glibc 进程启动时读一次;python 内 `os.environ` 设无效 → `undefined symbol: cudaGraphAddDependencies_v2`。须 `export LD_LIBRARY_PATH=$SITE/nvidia/cuda_runtime/lib:$SITE/tvm/lib:$(cat tvm_nvlibs.path):$LD` 再启 python。
3. **本 fork API 差异**: `tvm.nd.array`→`tvm.runtime.tensor(np, device=dev)`;namespace `tvm.s_tir`。
4. **tvm310 缺 pynvml**: H800 无网(代理 7897 拒),从 UniV2X_2.0 site-packages 拷 `pynvml*`+`nvidia_ml_py*`。
5. **NVML index 与 CUDA 不一致**: CUDA_VISIBLE_DEVICES 让 CUDA 见 device0 但 NVML 用绝对号 → `cuda.Device(gpu).make_context()`+`CUDA_DEVICE_ORDER=PCI_BUS_ID`+给 NVML 绝对号。
6. **TRT IHostMemory**: `len(serialized)`→`int(serialized.nbytes)`。
7. **metaschedule 4000-trial 的快 schedule 确定性崩**: 不是 harness bug(64-trial .so 能跑),是 metaschedule **直接对 group conv 做 WMMA** 产 OOB kernel。★**但这不代表 TVM 不能加速 group conv**——我们的 **im2col rewrite 路径绕开此崩点**(6_29 已证 7-14×/层)。教训: metaschedule 直搜崩 ≠ 后端不能,要用已有的 rewrite 解法。
8. **★agent 自报核验的双向教训**: 我 21:30 凭一次"无进程+GPU空"快照**误判 Agent C 谎报**——实为撞任务间隙(前2跑完+eval卡PYTHONPATH+后2未起)。**教训**: 自报必核验,但**别凭单次时序快照下"谎报"重锤**;核验要看产物+多时点,不是一张快照。
9. **train_ddp 必从 HEAL cwd + docker torch27**(老 env sm90 卡死);剪枝 ckpt 必 flat state_dict(非 {"model_state_dict":...} 包裹)。
10. **rtk hook 搅乱 SSH 输出**: 关键核验用 python heredoc + `/usr/bin/*` 二进制,数值真读不推断。

---

## §6 反思

1. **决胜实验是对的**: 用户拍板"先全量 tune TVM 决胜"避免了盲目为全前沿烧几十小时算力——一个 base fp16 全量 tune(65min)就钉死了"TVM 追不上 TRT",省下全前沿重调的浪费。**先做最便宜的决定性实验。**
2. **"公平"概念澄清(用户纠我)**: 180 点数据集 = **只是浅层硬件探针**,给代理模型筛选阶段融入硬件信息;本就浅,不涉公平。**"公平"=双层架构筛出前沿后全量微调所有前沿点**(外层真测验证),与 180 无关。我先前把"180 欠调"与"前沿公平"混为一谈是错的。
3. **后端硬伤 ≠ 方法失败**: TVM 慢不否定 co-design 搜索方法的价值;Option A 把贡献正确定位到 backend-agnostic 搜索,TVM/TRT 各司其职,是诚实且更强的立场。
4. **口径纪律救命**: 全程同 ONNX/shape/batch/device,才让 7.4× 这个数可信;若混口径这结论就废了。
5. **别过早外推 AP**: mn_32_32_96=0.5973 与预期地板矛盾,坚持"未坐实不采信"——交用户判断,不自圆其说。

---

## §7 下一步(Option A 双 backend,待用户确认结果合理后启动)

1. **TRT-backend 搜索前沿(L-B on TRT)**: 用搜索器的 width×precision 候选,每个用 TRT build+auto-precision,产 TRT-backend Pareto;对比现成 TRT-auto 单点(domination + iso-AP speedup + HV)。
2. **TVM-backend 前沿(L-B on TVM)** + 三臂消融(S 轴价值,TVM-only,已有真口径 harness)。
3. **前沿点 AP 全量微调续块**: block1(4点)→ 以 4 点为基础分块补完剩余;AP backend-无关,两后端共用;4090+H800 共卡并行。
4. **补消融耦合对**: 先测 **pad64**(解锁 1/3 W_g/P_g 对),再补 mix_b/mix_d/s1_64/s2_128。
5. **AutoTVM/Ansor baseline**: iso-config 后端质量矩阵补齐(L-A)。

---

## §8 产物清单(全经 stat 核验落共享 fs)

**H800 `${V2X_DATA_ROOT}/s2_tvm/la_fulltune/`**:
- `trt_int8_base_result.json`(TRT-INT8=0.835ms/0.222J)、`dair_calib/`(123 npy)、`dair_calib_manifest.json`、`trt_int8_dair_calib_extract.py`
- `full_base_fp16_4000/tuned_fp16.so`(4000-trial tuned,measure 崩)、`full_base_fp16_4000.log`
- `front_ap_finetune_block1.json`(⏳ Agent C 收尾)

**本地 repo `${V2X_ROOT}/`(未 commit)**:
- `framework/trt_baseline/trt_profile_v1.py`(TRT profiler)、`framework/trt_baseline/tvm_fulltune_fp16.py`(TVM 全量 tune harness)
- `framework/run_pqs_ablation_real_v1.py`、`framework/q_lookup_real.py`、`scripts/phase2/build_q_int8tc_real_ratios_v1.py`
- `results/q_int8tc_real_ratios_v1.json`、`results/pqs_ablation_results_real_v1.json`、`results/pqs_ablation_real_v1_missing_measurements.json`
- `multi_agent/figure/pqs_hv_boxplot_real_v1.png`、`multi_agent/figure/pqs_pareto_int8_real_v1.png`

---

## §9 交给用户核验的存疑点(结果合理性)

1. ~~TVM 7.4× 慢是否后端硬伤~~ **[已撤/已答]**: 用户已判定"不对",我的结论违反"别过早断言不可能"。**真问题** = 我没套 group-conv im2col rewrite(6_29 已有解)。**下一步 = 把 rewrite 套到 base backbone H800 真测**,拿到"优化好的 TVM"再谈 gap。
2. **TRT-INT8 仅省 5% lat**: 是否需 `--dumpLayerInfo` 查逐层精度确认 TRT 真选了 INT8 kernel(还是几乎全 FP16)?这影响"TRT-INT8 baseline"是否有意义。(仍开放)
3. **消融真口径 4/9 覆盖 + 耦合对 0**: 是否接受先补 pad64 解锁 1/3 再全补?A-joint 100/A-serial 86.5/A-noS 49.6 是否合理?(仍开放)
4. ~~block1 AP > 地板~~ **[已撤]**: 不再作为"疑点/发现"。就是 4 个真测数据点(0.5927-0.5973),喂框架自学,不手搓规则,不涉 §10.2 叙事。
5. ~~Option A 工作量~~ **[前提已撤]**: Option A 是假前提逼出的,待正确信息(优化好的 TVM 真测)后重判,见 §4。
