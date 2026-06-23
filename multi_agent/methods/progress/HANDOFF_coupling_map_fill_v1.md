# HANDOFF — 协同搜索空间「耦合地图」填图计划 v1 (2026-06-22)

> **设计依据**: `multi_agent/methods/design/coupling_map_v1.md`(轴定义 + 裁决判据 + 已种 cells)。
> **本文 = 执行交接**: 列全部 cell、每个 cell 用哪个模型/哪个 backend/哪块 GPU、预期裁决、输出文件,并回答"能否 Pyramid ∥ CoDriving ∥ V2X-ViT 三者并行"。
> **新窗口读法**: §1 目标(为什么不再钉死通道对齐)→ §2 三模型分工/就绪度 → §3 完整 cell 清单 → §4 并行计划+GPU → §5 复用基建 → §6 判据/铁律。

---

## §1 真正目的(用户 2026-06-22 核心澄清 —— 第一性前提,设计依据 `coupling_map_v1.md §0`)
绘制耦合地图**最底层的目的有两个**(不是"分类/省成本",而是证明"多维联合耦合"现象本身):

**① 首要 — 让 framework 真正做到 P×Q×S 三维全面联合搜索(不止"一个通道维度的结论")。** 当前"三轴强耦合"实质 = 三轴**经由单一机制(P 通道对齐)**耦合(P 枢纽:P×S、P×Q 各 JOINT,Q×S 是否独立耦合未知)。若空间能因子化成"P-hub + 其余独立",就**不是**真三维联合。**首要目标 = 证明 P×Q×S 联合不可约(多条耦合机制、Q×S 也独立耦合)**,使全维联合搜成为真正必要。→ 每个 cell 都在挖"超出 P-对齐"的新机制。

**② 次要 — 证明 CoDriving 也有耦合陷阱(用更高维度找它)。** "CoDriving 可分离"只是 1–2 维观察。**次要目标 = 在高维组合里搜出 CoDriving 的陷阱**(假设:标准 conv 单维无对齐陷阱,但 Q-granularity×P / bit-width×S / 某三维联合里仍有 serial 漏掉的最优)。找到 → 耦合更普适;高维搜完仍无 → 诚实负结果(但**必须真搜够维度**)。⇒ **CoDriving 的 cell 一律按"高维陷阱猎取"设计,不是"确认可分离"**。

**③ "避开两个极端 / 搜索可分解性"= 下游作用,非核心。** 把搜索拆成"耦合块联合 + 其余因子化"省成本,是**地图日后被搜索器使用时**的事,不是绘图的最底层目的。先有 ① ②。

**填图方法(服务上述目的)**: 在 **TVM(MetaSchedule)** 后端裁决每 cell 的 **JOINT/SERIAL/SINGLE/PAIR**。三条技术原则:
1. **TVM ≠ TRT**:TVM 上 schedule 是被搜的一等轴(非黑箱 auto-tuner)。TRT 旧结论(per-stage 混精被 auto 支配 / 2:4 弱 / D 维塌缩)**不外推**到 TVM,单列。
2. **P 是当前已知的耦合枢纽**:已证 (P×S)JOINT + (P×Q)JOINT。**目标 ① 要把它从"P-hub 可约"推到"三维不可约"**(C1 是第一块试金石,但若 Q×S 不独立耦合,要继续挖别的机制)。
3. **耦合 severity = f(轴基数)**:P-width 连续/大 → 必联合(贵);Q-bit∈3 值 → 即便耦合也可枚举(廉价)。标"联合在哪几轴真贵"——这服务下游 §1③,非核心目的。

---

## §2 三模型分工 + 就绪度
| 模型 | 角色 | 卷积类型 | 能填的 cell | 就绪度 |
|---|---|---|---|---|
| **Pyramid**(HEAL, DAIR) | grouped-conv 载体(对齐耦合住在这) | ResNeXt Bottleneck, groups=32, wpg=4(`resnext:true`, 架构本身非诱导) | C1/C2/C3 + 重定位 C0 | ✅ **就绪**(TVM 导入+延迟+int8 dp4a 全打通) |
| **CoDriving**(V2Xverse, DAIR) | standard-conv 对照(给"可分离"列) | 标准 conv, groups=1(`resnet:true`) | C1/C3 + C0c 复算 | ✅ **就绪**(TVM 导入+延迟+int8 复验全打通) |
| **V2X-ViT** | **Q 精度轴 + device-routing 耦合**载体 | 含 DCN/attention(grid_sample) | C4(Q 精度×剪枝)+ C5(DLA op-blacklist 路由) | ⚠️ **未就绪**:需(a)DAIR 训练 ckpt(b)量化+AP eval 管线(c)Orin/TRT 路由(DCN 大概率**不能导 relax/TVM** → C4 走 PyTorch/TRT 量化、C5 走 Orin/TRT,非 TVM) |

**关键诚实点**:
- **TVM-延迟耦合 cell(C1/C2/C3)= Pyramid + CoDriving 的主战场,现在就能并行开跑。**
- **V2X-ViT 是第三条轨,但**:① 要先 setup(训练/导出/量化 eval);② 它填的是 **Q 的精度轴(C4,PyTorch/TRT 量化+finetune+AP,非 TVM)** 和 **device 路由(C5,Orin/TRT DLA,非 TVM)** —— 与 TVM 延迟地图**不同 backend/不同实验类型**。所以"三者并行"成立,但 V2X-ViT 轨是 **setup-first + 异 backend**,不会像 Pyramid/CoDriving 那样快出 TVM cell。
- 为什么非要 V2X-ViT:**Pyramid 上 Q 的 AP 信号不可提取**(int8 近无损,连 mATE/mASE 都无信号,`dims_quantization_v1.md` 头注)→ 量化的精度 trade-off 只能换有真 int8 精度代价的模型(DCN/attention)才测得出。

---

## §3 完整 Cell 清单(填图主表)

### 已种 cells(前序会话已真测,只需重定位/小补)
| id | 维度对 | 模型 | 裁决 | 证据 | 待办 |
|---|---|---|---|---|---|
| C0a | P-width × S | Pyramid | **JOINT** | rank-flip 3.83×, `b4_ablation_results.json` | 重定位为"图中一 cell" |
| C0a' | P-width × S | CoDriving | **SERIAL/SINGLE** | 无 flip, `codriving_s0probe_fp16.csv` | — |
| C0b | P-width × Q(int8 build) | Pyramid | **JOINT(定性)** | s0=48 建不出 int8, `q_int8_dp4a_pairs.csv` | — |
| C0b' | P-width × Q | CoDriving | **SERIAL** | Cin 都 build, `codriving_int8_verify.json` | — |
| C0c | P×Q×S(全) | Pyramid | **JOINT(经 P)** | A-joint100%/serial85.3% p=4.88e-4 | — |
| **C0c'** | P×Q×S | CoDriving | (低维看 SERIAL) | — | 复算 run_pqs(CoDriving),纯 Python。**注:SERIAL ≠ 终点** —— 按目标② 这只是低维一探,要继续升维猎陷阱(见 C6/C7) |

### 开放 cells(本轮要填)
| id | 维度对 | 模型 | backend | 方法(最小实验) | 预期裁决 | 输出文件 |
|---|---|---|---|---|---|---|
| **C1** | **Q × S**(固定 P) | Pyramid + CoDriving | **TVM** H800 | 同一宽度(s0=64),fp16 与 int8 各真调 MetaSchedule;记 default(dlight)+tuned 延迟,比 (tuned/default) 比是否随 bit-width 变 + argmin schedule 是否不同 | **目标① 第一块试金石**:Q×S 若独立耦合 → 三维朝"不可约"推进一步;若仅 severity-低(枚举 Q 即可)→ 还没到不可约,需继续挖别的机制。**CoDriving 侧 = 目标② 高维陷阱第一探** | `results/coupling_map/C1_QxS_{pyramid,codriving}.json` |
| **C2** | **wpg × S** | Pyramid only | **TVM** H800 | 固定 num_filters,wpg∈{4,8,16}(改 width=int(planes·wpg/64)·groups 与 in_per_g),各做 P×S rank-flip 探针 | wpg 调制 P×S 耦合强度(3-way wpg×P×S) | `results/coupling_map/C2_wpgxS_pyramid.json` |
| **C3** | **round_to × P**(对齐能否被工程消除) | Pyramid(+CoDriving 平凡) | **TVM** H800 | 取失配宽 s0=48(建不出 int8/调优差),加 round_to=16 pad 到 s0=64,看是否变 P_g(能 build int8 + tune 7-8×) | 若 pad 后变 P_g → 耦合**可被 round_to 旋钮消除** → 重大重定位:"搜索须含 round_to,含了对齐就不再是陷阱"(连 [[feedback-no-premature-impossible]]) | `results/coupling_map/C3_roundto_pyramid.json` |
| **C4** | **Q-granularity × P**(per-tensor vs per-channel int8 × 剪枝,**需 AP**) | **V2X-ViT** | **PyTorch/TRT**(非 TVM) | V2X-ViT,per-tensor/per-channel int8 × 剪枝档,真测 AP(V2X-ViT 有 int8 AP 代价)→ 看 Q 精度轴是否存在 + 与 prune 是否耦合 | Q 精度轴存在(对照 Pyramid 无信号)+ 可能与 prune 耦合 | `results/coupling_map/C4_QgranxP_v2xvit.json` |
| **C5** | **device-routing × op + batch** | V2X-ViT(DLA)/ 全模型(batch) | **Orin/TRT**(非 TVM) | Orin DLA op-blacklist(grid_sample/DCN 必落 GPU)= routing×op 定性耦合;batch∈{1,2,4} 吞吐 regime | routing×op **JOINT 定性**(与对齐同构,不同轴);batch 改 regime | `results/coupling_map/C5_routing_v2xvit.json` |

### ★目标驱动的追加 cells(直接服务 §1① / §1②,不是普通分类)
| id | 目标 | 维度组合 | 模型 | 方法 | 成功判据 | 输出 |
|---|---|---|---|---|---|---|
| **C6** | **②CoDriving 高维陷阱猎取** | P × Q × S 三维全联合(含 int8 真路径,非 1.449 proxy)+ 必要时加 Q-granularity/batch | CoDriving | 真 int8 逐宽度 default+tuned(非 proxy)→ 跑 CoDriving 三臂 run_pqs;若无 → **升维**:加 per-channel/per-tensor × 宽度、bit-width(int4)× schedule、batch × 宽度,逐组找 serial 漏点 | 找到一个**干净的高维 rank-flip / serial 漏最优**(W_g 低维占优、高维联合才现的 P_g)→ 证 CoDriving 也有陷阱 | `results/coupling_map/C6_codriving_highdim.json` |
| **C7** | **①Pyramid 三维不可约** | 找"超出 P-对齐"的第二条耦合机制 | Pyramid | 在固定对齐的宽度集内,测 Q×S、Q-granularity×S、wpg×Q 是否仍耦合(即去掉 P 这条路后,Q/S 之间是否还互锁) | 找到**不经 P 的独立耦合机制** → P×Q×S 联合不可约,真三维联合搜成立 | `results/coupling_map/C7_pyramid_irreducible.json` |

> **C6/C7 是本轮的"真正目标 cell"**(C1-C5 是支撑/铺垫)。C7 成 → 目标① 达成(三维不可约);C6 成 → 目标② 达成(CoDriving 陷阱)。两者都要求**真 int8 路径**(非 1.449 uniform proxy —— proxy 假设 int8 加速与 schedule 无关,正好抹掉 Q×S 耦合,**不能用于判 Q×S/三维不可约**),必须逐宽度真测 int8 default+tuned。
>
> 每个 cell 输出 JSON 必含:`verdict ∈ {JOINT,SERIAL,SINGLE,PAIR}`、`severity`、`evidence`(真测数+口径)、`architecture_conditional`、`backend`、`serves_goal ∈ {①,②,downstream}`。汇总到 `results/coupling_map_matrix.json`。

---

## §4 并行计划(回答"能否三者并行")
**能,3 条独立轨,但成熟度不同**:

```
轨 A — Pyramid (TVM)     [就绪]  → C1, C2, C3, ★C7(三维不可约=目标①) (+C0 重定位)
轨 B — CoDriving (TVM)   [就绪]  → C1, C3, C0c', ★C6(高维陷阱猎取=目标②)
轨 C — V2X-ViT          [setup] → 先 setup → C4(PyTorch/TRT Q精度), C5(Orin/TRT 路由)
```
★ = 本轮真正目标 cell;C1-C5 是铺垫。轨 A/B 各自先用 C1 铺路、再冲 C7/C6。
- **轨 A ∥ 轨 B 完全独立**(不同模型、不同 GPU、不同输出文件)→ 真并行,无依赖。
- **轨 C 并行但慢热**:与 A/B 同时起一个 agent 做 V2X-ViT setup(训练/导出/量化 eval),setup 完成前出不了 cell;且 C4/C5 是异 backend(PyTorch/TRT/Orin),不抢 TVM 资源。
- **依赖**:C2/C3 可在 C1 之后或并行(都是 Pyramid TVM,排队同 GPU 或分 GPU);C0c' 无依赖随时算;C4 依赖轨 C setup;C5 依赖 Orin 可用 + V2X-ViT TRT 导出。

**GPU 分配建议**(H800 可用 0/1/2/6;⚠️ 3/4/5=CARLA 闭环他人勿用):
| 轨 | 主 GPU | 备注 |
|---|---|---|
| A Pyramid TVM | H800 **GPU0**(+6 并行多 cell) | C1/C2/C3 逐宽度 fresh workdir |
| B CoDriving TVM | H800 **GPU2** | C1/C3 |
| C V2X-ViT | **4090 GPU0/1**(训练/量化/AP)+ Orin(C5 路由) | setup 重;不占 H800 TVM |
| C0c' | 无 GPU(主控直驱) | 纯 Python run_pqs(CoDriving) |

**起几个 agent**:建议 **单层 3 agent**(轨 A / 轨 B / 轨 C 各一),每个明确"立即 ssh 跑 + 回报 PID + GPU util 非零",复用现成脚本不重写(承袭上轮"多层 orchestrator 卡 dispatch"教训)。C0c' 主控自己跑。

---

## §5 复用基建(别重写)
- **真仓库** `/home/jichengzhi/V2X`(⚠️ UniV2X 是断链空壳)。framework conda:`/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python` + `PYTHONPATH=/home/jichengzhi/V2X`。
- **H800 ssh**:`sshpass -p 12345678 ssh -p 30001 -o ConnectTimeout=45 -o StrictHostKeyChecking=no jichengzhi@222.95.84.215`(连太频限流,批量)。
- **H800 TVM env**:`export PATH=/usr/local/cuda-12.2/bin:$PATH; export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path)`;python `/exdata/jichengzhi/tvm310/bin/python`;工作目录 `/exdata/jichengzhi/s2_tvm/`。
- **现成 TVM 脚本**(复用):`b1_single_width.py`(逐宽度 fp16 default+tuned)、`int8_ms_tiled_gpu6.py`(int8 dp4a/WMMA MetaSchedule)、`int8_verify_correctness.py`(数值)、`remeasure_wg_pair4_default.py`(default-only 计时,注意删 `graph_executor` 导入)。CoDriving:`cod_int8_verify_v2.py`(relax int8)、`cod_s0_probe_export.py`。
- **三臂内核**:`framework/search_three_arm.py`(`detect_wg_pg_pairs`/`CostModelPQS`/`run_*_pqs`/`run_pqs_ablation.py`)。C0c' = 给 CoDriving 数据跑 `run_pqs`。
- **V2X-ViT**:OpenCOOD/V2Xverse 内有实现;DAIR 训练 recipe 参考 CoDriving/Pyramid 的 stage_a;DCN 导出/Orin 见 `DL4AGX/dcnv4-trt`(N6 闸门已过)+ `dims_hardware_v2.md` D3 op-blacklist。

---

## §6 判据 + 铁律
- **裁决判据**(每 cell):argmin 漂移检验 —— 固定其余,在 (A,B) 网格真调 B,看赢家是否随 A 翻转;量化用 `joint(A,B) HV vs serial(A→B) HV`(只在该子集)。gap 大=JOINT;gap≈0 但两轴都动 HV=SERIAL;一轴不动=SINGLE。**再标 severity**(大轴联合=贵)。
- **真 int8 铁律**:必 dump CUDA grep `__dp4a`/`wmma::mma_sync` + 数值 max_rel_err vs fp32。relax VMExecutable 不暴露 CUDA source 时,退而用 MetaSchedule DB trace(含 Tensorize+wmma)+ 数值 0.0 佐证,**并标这一档弱**(见 CoDriving int8 caveat)。绝不 simulated/fp16/QDQ 假 int8。
- **延迟纪律**:只在 idle pinned GPU(util 0%/mem≤50MiB);**fresh workdir 逐宽度 + subprocess 隔离**(复用 workdir 出假 ratio≈1.0 / CUDA illegal-access)。
- **不跨 backend/硬件混**:TVM(H800)/ TRT(4090)/ Orin-DLA 分列;微基准 ≠ 全 backbone,标口径。
- **不轻信自报**:凡"已测/已 build",主控复跑/读文件/grep 核验。
- **诚实裁决**:SERIAL/SINGLE/"可分离"都是**有用结果**(划定不必联合搜的区),不强凑 JOINT。

---

## §7 交付物
1. `results/coupling_map_matrix.json` —— 填满的分类矩阵(每 cell verdict+severity+evidence+architecture/backend 条件)。
2. 每 cell 的 `results/coupling_map/C*.json` + 必要图。
3. 回写 `coupling_map_v1.md`(填实)+ **重构 doc4 主叙事**:从"单条三轴耦合证明"升级为"**搜索空间耦合结构图 + 可从架构预测联合搜必要性的判据**"(通道对齐 = 图中一 JOINT cell)。
4. 更新 memory `project-codesign-nextstage-v3`(或新建 coupling-map 记忆)。

> **优先级(目标驱动)**:
> 1. **C0c'**(立即,纯 Python)+ **C1**(Pyramid∥CoDriving 真 int8 逐宽度 default+tuned,**非 proxy**)—— 铺路,验 Q×S 是否独立耦合。
> 2. **★C7(目标①)**:在 C1 基础上挖"不经 P 的第二条耦合机制",把 Pyramid 从"P-hub 可约"推到"P×Q×S 不可约"。**这是首要目标 cell。**
> 3. **★C6(目标②)**:CoDriving 真 int8 三维全联合 + 升维(granularity/int4/batch)猎陷阱。**这是次要目标 cell。**
> 4. C3(对齐能否被 round_to 消除,可能重定位故事)→ C2 → 轨 C setup → C4/C5。
> **铁律重申**:C1/C6/C7 判 Q×S/三维耦合,**必须用真 int8 default+tuned 延迟,严禁 1.449 uniform proxy**(proxy 假设 int8 加速与 schedule 无关 → 系统性抹掉 Q×S 耦合 → 会假阴)。

---

## §8 执行进度快照(2026-06-22 14:51 CST — ★新窗口接续从这里读起)

> 本节 = 上一窗口跑了一轮填图的真实状态。**新窗口接续前必读 §8.3(待修/待核实,不可直接采信)+ §8.4(方法学铁律,血泪教训)**。已核验通过的见 §8.2 可直接用。

### 8.1 cell 状态总表
| cell | 模型 | verdict | 文件(results/coupling_map/) | 核验状态 |
|---|---|---|---|---|
| C0c' | CoDriving | **SERIAL**(低维) | C0c_codriving_pqs.json | ✅ 主控核验通过 |
| C4 granularity×P | V2X-ViT | **JOINT**(弱档) | C4_QgranxP_v2xvit.json | ✅ confound 排除核验 |
| C5 routing×op | V2X-ViT | **JOINT**(定性) | C5_routing_v2xvit.json | ✅ 结构论证扎实 |
| C1 Q×S | CoDriving | **NO_WIDTH_SPECIFIC_TRAP** | C1_QxS_codriving.json | ✅ 主控改判+核验(p25假阳性已驳,2026-06-22) |
| C1 Q×S | Pyramid | COUPLED(待降级) | C1_QxS_pyramid_final.json | ⚠️ 真 int8 过关,声明待降级(trackA 处理中) |
| C7 三维不可约 | Pyramid | **降级: P-hub 非真三维不可约** | C7_pyramid_irreducible.json | ✅ mech3=BACKEND_ARTIFACT,trackA+main 双核验(2026-06-22) |
| C6 高维陷阱 | CoDriving | **NO_ROBUST_HIGHDIM_TRAP** | C6_codriving_highdim.json | ✅ 主控修序列化复用缓存重出+改判(2026-06-22) |
| C2 wpg×S | Pyramid | **CONFIRMED rank-flip(P-hub)** | C2_C3_pyramid.json | ✅ argmin_S native→padded-WMMA,确认 mech2(trackA+main) |
| C3 round_to×P | Pyramid | **mech1 软约束(可工程消除)** | C2_C3_pyramid.json | ✅ 定性成立;net-speedup 量级 confounded,补 trap25-FP16-tuned 中 |

### 8.2 已核验通过(可直接采信)
- **C0c' CoDriving P×Q×S = SERIAL**:4 宽度 A-joint=A-serial=100%, A-noS=97.1%, 0 rank-flip 对。标准 conv 低维可分离。**主控脚本 `framework/run_pqs_codriving.py`**(纯 Python,可复跑)。注:这是低维基线,**非终点**——目标②要 C6 在高维猎陷阱。
- **C4 V2X-ViT granularity×P = JOINT**:最优量化粒度随剪枝翻转(base 偏 per-tensor,p75 偏 per-channel),static 统一口径 coupling_signal=+0.0085(confound 量化排除占原 31%)。**弱档:simulated fake-quant 非真 TRT**(open_followup)。V2X-ViT 是地图唯一 Q 轴有真 AP trade-off 的模型(ΔAP50 -0.008~-0.013 vs Pyramid ~0)。
- **C5 V2X-ViT routing×op = JOINT(定性)**:grid_sample(STTF)触发 Orin DLA blacklist,~4% backbone 可路由。架构层确定性,不依赖 Orin 实测。

### 8.3 待修/待核实(★不可直接采信,新窗口要先处理)
1. **C1 CoDriving p25"软陷阱"= 假阳性(已驳回)**:轨B 报 p25 INT8 慢 1.76× 是陷阱,但**对齐的 base int8 也慢 1.9×(比 p25 还慢)** → "int8 慢"是合成 backbone 24 层 int8→int32→int8 cast-chain 的普遍 artifact,**非 p25 对齐陷阱**。tuning ratio base(1.038)≈p25(1.048)也非 p25 专属。stage0 micro-bench(无cast)int8 快 1.42× 无陷阱。→ verdict 待改 `NO_WIDTH_SPECIFIC_TRAP`。
2. **C6 CoDriving 崩在序列化(数据没丢)**:`c6_cod_highdim.py` 实验**全跑完**(H800 `/exdata/jichengzhi/s2_tvm/ms_work_c6_*` 有 base/p25/p50×batch{1,2,4}+stage0 已 tune,14:38),但写 JSON 崩 `TypeError: Object of type bool is not JSON serializable`(np.bool_)。**别重跑 tune**,只需修序列化(np.bool_→bool)从 ms_work/log 重写 C6 json。**且必须确认用同图换-schedule 判据**(见 §8.4.1),若用了 int8-vs-fp16 绝对速度则结果作废。log:`/exdata/jichengzhi/s2_tvm/c6_cod_highdim.log`(250KB)。
3. **C7/C1 Pyramid"三维不可约"声明待降级 + mech3 待核实**:真 int8 证据**过关**(C7 frag_line=`nvcuda::wmma...int` 真 WMMA int8,C1 max_rel_err=0.0,ms_gain 1.6/GFLOPS)。但"不可约"的命脉 mechanism_3(唯一"不经 P"的 Q×S)全靠 **"fp16 MetaSchedule 0/100 valid trials"** —— H800(SM90)向后兼容 sm80 fp16 WMMA m16n16k16,fp16 tensor-core 物理可用,**0 valid trials 疑为 TVM 对 conv2d_NCHWc fp16 缺自动 tensorize rule(后端 artifact)非物理耦合**。待办:① 核实 fp16 显式手动能否 tensorize(sm80 wmma_m16n16k16 intrin/dlight tensorcore);② **诚实降级**声明为 **"TVM 0.20+H800 后端上 P×Q×S 联合不可分解"**(标 mech3 根源=后端 tensorize-rule 覆盖不均;这反而强化 framework 论点);③ 澄清 mech1(P×Q buildability)/mech2(wpg×Q×S gain)经 IC_BN=IC/groups=**P-hub**,只 mech3 不经 P。

### 8.4 方法学铁律(★血泪教训,新窗口判耦合前必读)
1. **cast artifact 陷阱(本轮踩过2次)**:**判 Q×S 耦合绝不能用 int8-vs-fp16 绝对速度**。合成 backbone 的 int8→int32→int8 cast-chain 让 int8 普遍慢于 fp16(对齐的 base 也慢 1.9×),这是 cast artifact 非耦合信号,会诱导"发现陷阱"假阳性。**正确判据 = 同一 int8 计算图内换 schedule**:int8 用 int8-own-tuned-schedule(joint) vs int8 用 fp16-tuned-schedule(serial),比延迟。cast 开销两者相同→纯隔离 schedule 耦合。serial 明显慢=Q×S 真耦合(serial 漏最优)。
2. **后端 artifact ≠ 算法不可约**:"fp16 不能 tensorize"若是 TVM rule 覆盖问题,要标"后端限制",不能当"算法内禀不可约"。诚实表述=给定后端不可分解。
3. **mech 经不经 P 要分清**:IC_BN=IC/groups,凡经 groups/wpg/width 的都是 P-hub;"不经 P"= 固定 P 下 Q 与 S 仍互锁。
4. **真 int8 铁律**:grep `__dp4a`/`wmma::mma_sync` + 数值 max_rel_err vs fp32。relax 不暴露 source 时用 MS-DB trace+数值 0.0 并标弱档。
5. **不轻信自标 completed**:本轮抓到 task5(C2/C3)空标无文件、C6 自标完成实际崩溃、archived 字段存错值——凡"已测/已修"必读文件/grep 核验。

### 8.5 各 agent 状态 + H800 可复用资源
- **轨 A `trackA-pyramid`**:★**agent 已退出**(14:5x SendMessage 报 "no teammate")——报完 C1/C7 后即终止,**mech3 核实 / C2 / C3 完全未做**(派出的指令 agent 已退出未执行)。
- **轨 B `trackB-codriving`**:★**agent 已退出**——C6 进程崩(np.bool_ 序列化)后即终止,**C6 修复 / C1cod p25 verdict 改 / 判据确认 均未做**。C6 已 tune 数据在 H800 `ms_work_c6_*`(可复用)。
- **轨 C `trackC-v2xvit`**:✅ C4/C5 完成(产出完整,agent 亦已退出)。
- ⚠️ **结论:本 session 三轨 agent 全部已退出,无活跃 agent;所有"进行中"实为停滞。新窗口(或本窗口续作)必须重新 spawn 才能推进**(C6 修复 / mech3 核实 / C2 / C3)。**H800 tune 数据可复用别重跑**:`ms_work_c6_*`(C6)、各脚本(`c6_cod_highdim.py`/`c1_cod_int8_fullbb.py`/Pyramid C1/C7 脚本在 `/exdata/jichengzhi/s2_tvm/`)。重 spawn 时明确"复用已 tune ms_work"。

### 8.6 新窗口接续待办(优先级)
1. **轨B**:修 C6 序列化(np.bool_→bool,数据在 ms_work_c6,**别重跑**)+ 确认同图换-schedule 判据 + 改 C1cod p25 verdict。
2. **轨A**:mech3 fp16-tensorize 核实(决定不可约真假)→ 降级声明 → C3(round_to×P,失配宽 pad 到 IC_BN=4 看能否消除对齐陷阱)→ C2(wpg×S)。
3. **汇总**:全 cell 落定后写 `results/coupling_map_matrix.json`(每 cell verdict+severity+evidence+architecture/backend/serves_goal)+ 回写 `coupling_map_v1.md` 填实 + 重构 doc4 §9 主叙事(从"单条三轴耦合"升级为"耦合结构图")。

### 8.7 环境(重申)
- 真仓库 `/home/jichengzhi/V2X`;framework conda `/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python` + `PYTHONPATH=/home/jichengzhi/V2X`。
- H800 ssh `sshpass -p 12345678 ssh -p 30001 -o ConnectTimeout=45 -o StrictHostKeyChecking=no jichengzhi@222.95.84.215`;TVM env `export PATH=/usr/local/cuda-12.2/bin:$PATH; export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path)`;python `/exdata/jichengzhi/tvm310/bin/python`;workdir `/exdata/jichengzhi/s2_tvm/`。
- ★**GPU 使用规则(用户 2026-06-22 指定,覆盖此前所有 GPU 规则)**:**只允许使用 H800 GPU 4 / 5 / 6 三张卡**,其余(0/1/2/3/7)一律不可用。新窗口重 spawn agent 时按此分配(如 轨A→GPU6 / 轨B→GPU5 / GPU4 机动或第三轨)。**用前必 `nvidia-smi` 确认目标卡 idle(util 0%/mem≤50MiB)再跑**;若 4/5/6 被占需等待,不得挪用其它卡。fresh workdir 逐宽度 + subprocess 隔离。

---

## §9 新窗口续跑进度(2026-06-22 ~15:15 CST — ★接 §8 续)

新窗口已重启工作。GPU4=C6 / GPU5=trackA(mech3)。进度:

### 9.1 已收尾(主控亲自核验)
- **C6 CoDriving 高维(task#1)✅**: 修 np.bool_ 序列化(`_json_safe` default)+ 跳过重 tune(DB 存在则复用 ms_work_c6 缓存)→ GPU4 复用 12 个缓存 DB 秒级重出 `C6_codriving_highdim.json`。**主控复核把 verdict 从 agent 原报 `HIGHDIM_TRAP_FOUND` 改判为 `NO_ROBUST_HIGHDIM_TRAP`**: 所谓 batch×width rank-flip 仅 b1 处 1.2% 噪声级近平局(base 1565.93 vs p25 1584.5),b2/b4 p25 领先纯因它更小,方向与 K-对齐陷阱预测相反;且 verdict_lines 复活了已否的 cast-artifact 假阳性。脚本 `c6_cod_highdim_fix.py`(H800)。
- **C1cod p25 verdict(task#2)✅**: 改 `NO_WIDTH_SPECIFIC_TRAP`。本表自证: aligned base int8/fp16=0.528× 比 misaligned p25 0.569× 还慢 → cast artifact 非对齐陷阱;int8 tune ratio base(aligned)1.038 < p25 1.048 → ratio 跟 size 不跟 K 对齐。json + analysis md 均加修正块。

### 9.2 CoDriving 三 cell 汇成连贯结论(goal②)
C0c'(SERIAL)+ C1cod(NO_WIDTH_SPECIFIC_TRAP)+ C6(NO_ROBUST_HIGHDIM_TRAP)三者一致 ⇒ **CoDriving 标准 conv(groups=1)在低维与高维均无稳健耦合陷阱;耦合是 grouped-conv(Pyramid)特有现象**。这是 goal②的干净科学结论(找了但没找到=阴性也是结论)。

### 9.3 ★mech3 命脉已闭合(task#3 ✅, trackA+main 双核验)
**mech3 = BACKEND_ARTIFACT,C7 "三维不可约" 最强 claim 已降级**:
- trackA 真测(H800 GPU5): FP16 **也走 WMMA**(g8 IC_BN=16 gain 8.59×,g32 IC_BN=4 gain 1.64×),CUDA 源含 `matrix_a/b 16×16×16 half` 片段。mech3 声称的 "FP16→SCALAR / INT8→WMMA" 是假命题。
- 根因双重错误: (1) C1 的 `count_db_valid()` 解析 TVM0.20 DB 格式 bug,对 50 条有效记录返 0; (2) `write_c7_verdict.py:44` 把 `all_fp16_ms_valid_trials:0` **硬编码 literal**,g8 从未实测就外推到 "ALL groups"。
- **main 独立核验通过**: frag_lines 确含 half matrix_a/b(非仅 float accumulator); gain 8.59×配 50 DB records(若真 0 记录 gain 应≈1); 硬编码 literal 经 grep 坐实(write_c7_verdict.py:44)。
- **降级后可辩护 claim**: mech1(IC_BN≥4 阈值)+ mech2(IC_BN→gain scaling)仍成立,但都**经 P-hub**(IC_BN=IC/groups)。"存在不经P的不可约耦合" **无干净证据**。正确表述 = **P(IC_BN) 是 hub 的 P×(Q+S) 联合依赖,非三维真正独立不可约**。
- 证据: `results/coupling_map/mech3_fp16_verify.json` + C7 json 顶层 `mech3_verdict/claim_downgraded/honest_claim/revised_top_claim/main_review_2026_06_22`。

> ★对 goal① 的影响: "找一个不经 P 的耦合机制证明不可约" 这条最强路线 **目前没有干净证据**(mech3 倒了)。Pyramid 的耦合是 **P 为 hub 的 P×(Q+S)**;CoDriving 标准conv **完全可分(SERIAL)**。这是诚实且自洽的图谱。

### 9.4 C2/C3 完成(task#4 ✅,trackA+main 核验)
- **C2 wpg×S rank-flip = CONFIRMED(P-hub)**: argmin_S 随 IC_BN 翻转 — IC_BN=16 用 native-WMMA,IC_BN=4 因 k=4<WMMA min k=16 只能 padded-WMMA(CUDA 证据 `reindex_shared_` vs `reindex_pad_shared_`);INT8 相对 FP16 的 MS 优势从 1.37×(IC_BN16)压到 1.04×(IC_BN4)。⇒ 最优 (Q,S) 对是 IC_BN(=P)依赖,**确认 mech2,且是 P-hub**(非不经P)。
- **C3 round_to×P: mech1 = FORMAT 障壁,延迟中性(best-vs-best 定稿)**: main 打回原 confounded "1.36× net speedup"(FP16-**default** vs INT8-tuned)后,trackA 补齐全部 tuned 真测做 apples-to-apples。**misaligned IC=96 上三选项(全 MetaSchedule-tuned/或 best-path)**: FP16-tuned=**139.29µs** ≈ INT8-NCHW-fallback=**140.92µs**(打平~1%) **都快过** padded-128-NCHWc-tuned=**149.45µs**(padding +33% 通道代价 > NCHWc dp4a gain)。⇒ mech1(IC_BN≥4 才能走 NCHWc dp4a)是 **format gate 不是延迟代价** —— **padding 修对齐不划算,留在 misaligned 宽度反而更快**;INT8 的优势**仅 vs 未调优 FP16**(1.46×),vs tuned FP16 无优势。文件 `trap25_fp16_tuned.json`(FP16 真 WMMA half,50 records,std 0.02µs) + `C3_int8_same_graph.json`。★main 自纠: 早先"NCHW int8 比 FP16 快 1.45×"用的是 FP16-default,误导,已改 best-vs-best。caveat: FP16-tuned/padded-INT8 是 WMMA tensorized,INT8-NCHW 是非张量核 fallback,结论 best-vs-best robust。

### 9.5 ★耦合图谱合并结论(核验后,可写论文)
| 机制 | 状态(复核后) | 是否经 P |
|------|------|------|
| mech1 P×Q 阈值(IC_BN≥4 NCHWc dp4a) | **format 障壁,延迟中性**(NCHW int8 fallback 吸收,padding 不划算,C3) | 经 P |
| mech2 wpg×Q×S gain scaling | **REAL,核心**,C2 rank-flip 确认 | 经 P(IC_BN) |
| mech3 Q×S 独立(不经P) | **FALSE = BACKEND_ARTIFACT**(FP16 也 WMMA) | — |

**最终诚实图谱**:
- **Pyramid(grouped conv)**: 耦合真实但 **P(IC_BN) 是 hub**,唯一存活的真机制 mech2 是 P-mediated;mech1 软可消除;mech3 是后端 artifact。
- **CoDriving(标准 conv)**: 低维高维**全可分(SERIAL)**,无陷阱 ⇒ 耦合是 grouped-conv 特有。
- **goal① 现状**: "不经 P 的不可约耦合" **无干净证据**(mech3 倒);真耦合(mech2)经 P。论文应主张 "P(IC_BN)-hub 的 P×(Q+S) 联合依赖",而非"三维独立不可约"。
- **goal② 现状**: 闭合(阴性)——CoDriving 高维也无陷阱。

### 9.6 待办
- 补 trap25-FP16-tuned 后 C3 完全 close。
- **task#5 汇总已解阻塞**: `results/coupling_map_matrix.json`(已搭 9/9 cell 骨架)→ 改写 `coupling_map_v1.md` → 重构 doc4 §9 主叙事。

### 9.7 轮询
主控每 30min 轮询 trackA 存活 + GPU4/5/6 + cell 产出。

### 9.4 轮询
主控每 30min 轮询 trackA 存活 + GPU4/5/6 占用 + cell 产出,持续推进。
