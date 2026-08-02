# AnonymousSubmission2027 技术内容修改审查 v1

更新时间：2026-07-24  
审查对象：[AnonymousSubmission2027.tex](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex)  
目的：在不直接改动论文正文的前提下，明确当前 LaTeX 与最新框架实现之间的差异，并给出按章节执行的修改清单。

---

## 1. 审查结论

当前正文的研究动机仍可保留，但方法部分主要描述的是早期方案，尚未反映 2026-07-16 之后冻结的正式实现。最关键的差异有五项：

1. 当前正文仍把完整候选写成 `(compression, quantization, schedule)` 的统一联合空间；正式搜索任务实际固定 `target model + hardware + backend capability profile`，外层 genome 只搜索结构宽度与 `q_mode`，后端 schedule/tactic 是候选被选中后的条件化内层实现过程。
2. 当前正文把三个目标统一写成 LightGBM，并引入独立 ranking loss；正式实现已经冻结为按目标选择的低容量多头模型，独立 ranker 已被消融拒绝。
3. 当前正文把候选生成写成 NSGA-II；正式在线策略是对当前合法池预测后执行 `predicted_frontier_diversity` 批量采集，固定 `B=4、4 rounds、T=16`。
4. 当前正文没有写清候选图特征的双层合同；正式流程对未物化候选使用 cold-start graph surrogate，对已实测候选提取 materialized ONNX actual graph features，并从下一轮开始与真实三指标一起回流。
5. 当前实验表把已撤回的 F-Cooper pilot 当作正式结果，并宣称三个模型均已完成独立复测。F-Cooper 正式 v2 的 4 轮、16 个在线点已经闭合，但五臂对照、动态 winner 独立复测与最终 `paper_ready` 审计尚未闭合，相关行和结论当前必须撤回或标记为占位。

因此，技术部分不宜做局部润色，应按本文件第 5 节重新组织方法章节。

---

## 2. 必须立即处理的 P0 问题

| 编号 | 当前位置 | 问题 | 修改要求 |
|---|---|---|---|
| P0-1 | [第 94–126 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:94) | 标题、作者和单位仍是 AAAI 模板内容 | 换成匿名论文标题和匿名作者格式；方法名统一后再冻结标题 |
| P0-2 | [第 163–165 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:163) | Abstract 仍是 AAAI 模板说明 | 用研究摘要替换；摘要只能采用已收口的结果，不得引用当前 F-Cooper pilot 数值 |
| P0-3 | [第 210 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:210)、[第 240 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:240)、[第 455 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:455) | `SHCoSearch`、`SHCoS`、`GEAR` 三个名称混用 | 正文、公式、图、表统一为 `GEAR`；若 GEAR 只是展示名，需要在首次出现处给出全称 |
| P0-4 | [第 249–258 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:249)、[第 335–355 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:335) | 把 schedule 写成与结构、精度并列的外层显式基因，并把 AutoTVM/XGBoost 写成统一内层实现 | 改成固定 profile 下的条件化内层 realization；TRT tactic 与 TVM schedule 使用各自后端合同，不把 backend 或 schedule ID 放入外层 genome |
| P0-5 | [第 356–411 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:356) | 三目标统一 LightGBM、value+rank 联合损失与实际冻结模型不一致 | 按第 4.4 节重写；删除独立 rank loss 公式及其有效性声明 |
| P0-6 | [第 432–437 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:432) | 把在线 candidate generation 写成 NSGA-II | 改成预测前沿与多样性驱动的 batch acquisition，并写明 `B=4、T=16`、全池预测和无人工精度配额 |
| P0-7 | [第 439–446 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:439) | 声称只有最终前沿候选才做 AP，且未说明 actual graph feedback | 正式预算内每个被选中 genome 都必须执行恢复训练、后端构建、latency、energy、完整 AP、actual graph feature 和证据绑定；四行终态后原子回流 |
| P0-8 | [第 455–516 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:455) | F-Cooper pilot 被写成正式三模型结果 | 在 F-Cooper v2 `paper_ready=true` 前删除 F-Cooper 数值、三模型全胜声明及相应百分比 |
| P0-9 | [第 520 行以后](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:520) | 正文后仍保留大量 AAAI 示例章节 | 最终稿必须删除模板正文，只保留论文内容、参考文献和必要附录 |

---

## 3. 论文必须统一的技术口径

### 3.1 搜索任务边界

一次正式搜索任务定义为固定的：

```text
target model + dataset/evaluation contract + hardware + backend capability profile
```

TVM 和 TensorRT 是两个独立 profile、两次独立搜索，不是同一次搜索中的 backend 基因。硬件与后端信息通过 capability context 影响候选合法性、cost-model 响应和真实 realization，但不允许框架在一次任务中把 backend 名称当作可选策略。

推荐记号：

\[
\tau=(M,D,H,\rho),
\]

其中 \(M\) 是目标模型，\(D\) 是冻结的数据与 AP 合同，\(H\) 是硬件，\(\rho\) 是后端 capability profile。

### 3.2 外层 genome

推荐统一写成：

\[
z=(w_1,\ldots,w_K,q),\qquad q\in\{\mathrm{FP16},\mathrm{INT8}\}.
\]

- \(K\) 由当前模型 scanner/adapter 暴露的结构依赖组决定，不应在公式中固定为 3。
- Pyramid 和 CoDriving 当前为三个多尺度 backbone width 轴。
- F-Cooper 正式 v2 scanner 输出五个结构轴：`backbone.s0`、`backbone.s1`、`backbone.s2` 及两个由 neck 依赖关系要求的接口宽度。
- FP32 当前用于 `original/default` 和 `schedule-only` 基线，不属于正式在线 GEAR 的 `q_mode` 搜索集合。
- 当前正式 genome 不包含 mixed-policy ID、graph-boundary ID、backend ID 或 schedule ID。

需要避免“genome 包含图特征”的表述。图特征是 genome 物化后对应图的描述特征，是 cost-model 输入上下文，不是决策基因。

### 3.3 内层后端实现

后端调度应写成条件化 realization，而不是外层笛卡尔积：

\[
u_\rho(z)=\mathcal{R}_\rho
\bigl(z,s_\rho^\star(z)\bigr).
\]

其中 \(s_\rho^\star(z)\) 表示目标后端针对候选 \(z\) 产生的 tactic、schedule 或编译实现。外层不枚举通用 schedule ID，而是测量该候选在固定 profile 下经过冻结后端策略得到的真实产物。

“软硬件联合”的准确含义是：

> 外层结构与精度选择由经过目标后端实现后的真实 AP、latency、energy、feasibility 和 actual graph evidence 驱动，而不是由 FLOPs 或软件代理独立决定。

### 3.4 优化目标

当前公式把 AP 写成只依赖软件配置的 `AP(x)`，这一点需要修正。量化数值路径、fallback 和后端 realization 均可能影响 AP，因此推荐写成 profile 条件化实测值：

\[
\mathcal{P}_{\tau}^{\star}=
\operatorname{Pareto}_{z\in\Theta_\tau}
\left(
-\operatorname{AP}_{\tau}(u_\rho(z)),
L_{\tau}(u_\rho(z)),
E_{\tau}(u_\rho(z))
\right).
\]

论文可说明 AP 的主要变化来自结构压缩与精度，但不得从定义上排除后端数值路径对 AP 的影响。

---

## 4. 按当前实现重写方法细节

### 4.1 Capability characterization

对应当前正文：[第 284–300 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:284)。

建议保留三层能力信息，但降低过度声明：

1. 静态能力：设备、dtype、算子、shape 和编译入口。
2. S1 中性结构探针：INT8 propagation、Q/DQ folding、fallback、reformat、group/small-channel buildability。
3. 必要时才启用更贵的性能探针；不能把历史 TVM/TRT 实测比率伪装成扫描结果。

必须写明：

- capability probe 用于形成可执行上下文和 feasibility 证据；
- probe 的 AP/latency 不进入正式 T16 cost-model 标签；
- probe 不消耗 T16 在线预算，也不能直接成为最终 winner；
- capability 不预先硬编码“TRT 选 INT8、TVM 选 FP16”。

F-Cooper v2 对 probe 隔离的实现检查见：

- [stage5_initialize_fcooper_actual_v2.py:136](${V2X_ROOT}/scripts/stage5_initialize_fcooper_actual_v2.py:136)
- [32 号交接文档第 12 章](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/32_7_14_交接文档_FCooper_Table1_TVM_CPU与Orin边缘部署计划_v1.md)

### 4.2 Graph scanner 与搜索空间

对应当前正文：[第 302–326 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:302)。

当前描述“输入任意模型即可自动发现所有 dependency blocks 并合并”为过强声明，应改为分层口径：

- 框架根据模型图和 dependency contract 构造合法结构组，保证 residual、concat、deblock、neck 接口等通道约束。
- Pyramid/CoDriving 的三个 stage 语义由模型 adapter 暴露，不能宣称已经实现任意模型的全自动旋钮发现。
- F-Cooper v2 已实现从 fresh scanner manifest 动态构造五轴 genome，禁止套用三宽度模板或人工覆盖。
- 当前优化 scope 是协同 3D 感知模型的 post-scatter dense backbone/neck body，不等价于完整通信、融合、检测和规划系统的端到端加速。

适用范围与限制应同时披露。对应反思见：

- [29 号交接文档第 2、5 节](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/29_7_14_交接文档_阶段5端到端自动性与图特征回流反思_v1.md)
- [31 号交接文档第 3.1 节](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/31*_7_14_交接文档_框架待完善项与阶段6六臂对照实施计划_v1.md)

### 4.3 冷启动与特征合同

当前正文只有笼统的 hardware-probe-enhanced feature，需要补齐以下正式合同。

初始数据：

- `Gold176` 是冻结的跨模型 cold-start 趋势数据；
- 共 176 行、44 个完整四臂组，其中 174 行为完整三指标真测，2 行为确认的 feasibility failure；
- 在线 Round 0 只读取 `initial_coldstart`，不读取旧 Stage5 pilot、Feedback16 或目标模型专项历史在线标签；
- Gold176 用于学习跨配置趋势，不把 Pyramid 和 CoDriving 绑定为一次联合搜索。

输入特征：

\[
\phi_t(z)=
\left[
\phi_{\rm cfg}(z),
\phi_{\rm graph}^{(t)}(z),
\phi_{\rm cap}(\rho),
q\otimes\phi_{\rm cap}(\rho)
\right].
\]

其中：

- `cfg`：最多五个 width 轴、width product、`q:int8` 和 model indicator；
- `graph`：shape、Conv、FLOPs、参数量和图结构统计；
- `cap`：不含标签的 capability features；
- `cap × q`：使模型学习同一精度在不同 profile 下的响应差异；
- latency、energy、AP 和 terminal status 均禁止进入图或 capability 特征。

双层 graph feature 合同：

```text
未选择候选
  -> 使用由结构宽度预测的 cold-start surrogate graph features

被 acquisition 选中并物化
  -> 从实际 ONNX 提取 actual graph features
  -> 与真实 latency、energy、AP、feasibility 一起进入反馈

下一轮
  -> 已测点使用 actual features
  -> 未测候选仍使用 surrogate features
```

不得写成“全部候选先物化后再搜索”，也不得写成“实测标签回流但图特征保持预测值”。

实现证据：

- [cost_model_selection_v1.py:121](${V2X_ROOT}/framework/stage4/cost_model_selection_v1.py:121)
- [stage5_advance_fcooper_round_v2.py:182](${V2X_ROOT}/scripts/stage5_advance_fcooper_round_v2.py:182)
- [29 号交接文档第 4 节](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/29_7_14_交接文档_阶段5端到端自动性与图特征回流反思_v1.md)

### 4.4 Cost model

对应当前正文：[第 356–431 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:356)。

正式冻结配置为：

| 目标/组件 | 正式实现 | 论文口径 |
|---|---|---|
| latency value head | ExtraTrees on `log1p(latency)` | 小样本、非光滑响应下的树模型 |
| energy value head | ExtraTrees on `log1p(energy)` | 与 latency 分开训练 |
| AP70 value head | LightGBM Huber residual | 相对模型自身 AP anchor 建模 |
| independent ranker | 已拒绝 | 不进入正式方法，不保留 pairwise rank loss 公式 |
| uncertainty | LightGBM quantile + grouped conformal correction | 用于区间和采集保护，不作为单独目标 |
| acquisition | predicted-frontier diversity | 见第 4.5 节 |

因此需要：

1. 删除“全部目标统一 LightGBM”的公式。
2. 删除独立 ranking predictor、pairwise ranking loss 和 `value + lambda rank` 的正式方法描述。
3. 保留 AP residual，但把 anchor 明确为目标模型自身实测 AP 参考；F-Cooper Round 0 只注入该 AP anchor，不注入旧 pilot latency/energy。
4. 加入 quantile/conformal uncertainty，但不能宣称得到高精度置信区间；当前作用是 acquisition 保护。
5. 说明不同目标的在线反馈更新具有 target specificity，不保证少量反馈同时改善三目标。

冻结依据：

- [26 号交接文档第 9.1–9.3 节](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/26_7_14_交接文档_Feedback16_AP收口与阶段4问题反思_v1.md)
- [single_target_search_v2.py:443](${V2X_ROOT}/framework/stage5/single_target_search_v2.py:443)

### 4.5 Acquisition 与预算

对应当前正文：[第 432–437 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:432)。

正式方法不是当前正文所写的在线 NSGA-II。每一轮执行：

1. 对固定任务的全部当前合法、未测候选做 cost-model 预测。
2. 根据三目标预测识别 predicted Pareto frontier。
3. 选一个 latency exploitation anchor。
4. 选一个 AP exploitation anchor。
5. 其余两个位置通过 feature/objective maximin diversity 选择，uncertainty 只用于次级判定。
6. 输出四个独立 genome，不设置 FP16/INT8 人工配额，不因选中某个 width 自动带出另一个 q_mode。

固定预算：

```text
B = 4 candidates per round
R = 4 rounds
T = 16 measured genomes
main-search early stopping = disabled
```

一个 batch 的四个候选全部达到可信终态后，才原子释放反馈并重训一次 cost model。基础设施失败不消耗预算；可信 feasibility failure 按合同回流并消耗一个 genome。

实现证据：

- [single_target_search_v2.py:308](${V2X_ROOT}/framework/stage5/single_target_search_v2.py:308)
- [single_target_search_v2.py:112](${V2X_ROOT}/framework/stage5/single_target_search_v2.py:112)
- [fcooper_finalize_formal_round_v2.py:191](${V2X_ROOT}/scripts/fcooper_finalize_formal_round_v2.py:191)

如果后续决定重新引入 NSGA-II，必须先完成与当前 acquisition 的同预算消融；在此之前不能把 NSGA-II 写成已实现正式方法。

### 4.6 真实测量与反馈

每个被选中的正式候选需要写成以下完整流水线：

```text
scanner-derived genome
-> frozen recovery training
-> checkpoint/config SHA
-> ONNX materialization
-> backend build/tune/tactic selection
-> latency and energy
-> full AP30/AP50/AP70
-> actual graph features
-> feasibility/fallback evidence
-> artifact and report SHA
-> atomic feedback
-> task-local cost-model refit
```

当前正文“只对最终 Pareto 候选做 AP”的表述不符合正式搜索。正式 T16 中每个成功候选都执行完整 AP；否则 AP cost model 无法获得逐轮真实反馈。

F-Cooper 压缩候选还必须经过统一恢复训练合同。不能再把零微调的 prefix projection checkpoint 当作正式测量源。恢复训练是正式搜索时间的主要组成部分，也应进入搜索成本表。

---

## 5. 建议的方法章节新结构

建议把当前 Method 重排为以下结构。

### 5.1 Overview: Profile-Conditioned Closed-Loop Search

只讲清一条主线：

```text
固定 SearchTask
-> scanner 构造合法结构与精度空间
-> Gold176 初始化多头 cost model
-> 全池预测与 B=4 acquisition
-> 选中点恢复训练和后端真实实现
-> actual features + AP/latency/energy 原子回流
-> 重训并更新实测 Pareto
```

### 5.2 Profile and Scanner-Constrained Genome

包含：

- SearchTask 定义；
- capability profile；
- scanner/adapter；
- 可变长度 width genome；
- 合法空间 \(\Theta_\tau\)；
- 当前 scope 和适用范围。

### 5.3 Multi-Source Feature Contract and Cold Start

包含：

- Gold176；
- config、graph、capability 和交互特征；
- surrogate/actual 双层图特征；
- 防标签泄漏和跨任务在线证据隔离。

### 5.4 Metric-Specific Cost Models

包含：

- ExtraTrees log latency/energy；
- LightGBM Huber AP residual；
- quantile + grouped conformal；
- target-wise feedback acceptance。

### 5.5 Pareto-Diversity Batch Acquisition

包含：

- predicted frontier；
- latency/AP anchors；
- maximin diversity；
- `B=4、T=16`；
- 无 q 配额、无人工补点、无主搜索提前停止。

### 5.6 Conditional Backend Realization and Atomic Feedback

包含：

- TRT tactic / TVM schedule 的 profile-specific realization；
- 恢复训练、构建、三指标测量和 actual graph extraction；
- failure contract；
- batch atomicity；
- 实测 Pareto 与独立 winner 复测。

---

## 6. Introduction 与贡献点修改

### 6.1 动机图

[Figure 1(b)](${V2X_ROOT}/multi_agent/paper/Latex/Figures/fig_background_joint_evidence.pdf) 的三条曲线混合了不同执行路径：

- FP32：早期 TVM Relax/MetaSchedule 分类口径；
- FP16：手写 rewritten TensorCore 路径；
- INT8：另一条 TVM native INT8 路径。

因此 [第 201–202 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:201) 只能用它说明“结构宽度、精度标签与真实延迟并非严格单调”，不能称为同一后端、同一调优预算下的公平三精度对照。建议二选一：

1. 用统一 profile 和统一测量合同的新数据重绘，作为正式动机图。
2. 保留现图，但图注明确是 heterogeneous deployment paths 下的历史响应，只承担非单调性动机。

详细来源审计见 [31 号交接文档第 10 节](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/31*_7_14_交接文档_框架待完善项与阶段6六臂对照实施计划_v1.md)。

### 6.2 问题定义

[第 210–211 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:210) 建议从“搜索 joint software and schedule configurations”改为：

> 在固定部署 profile 下，如何用有限真实测量预算，自动发现结构压缩与精度配置，并通过候选条件化的后端实现与实测反馈学习非单调、后端相关的 AP–latency–energy 前沿。

这一表述保留软硬件联合，但不把 schedule 误写为外层显式基因。

### 6.3 三个贡献点

建议重写为：

1. **Profile-conditioned scanner space**：从目标模型依赖图和目标 capability profile 构造合法、可变维的结构–精度 genome，避免用 FLOPs、bit-width 或跨后端经验直接排序。
2. **Metric-specific evidence model**：用结构、图和 capability 交互特征分别建模 AP、latency、energy，并维护 surrogate-to-actual graph feature contract 与不确定度校准。
3. **Budgeted hardware-in-the-loop search**：通过 predicted-frontier diversity 选择独立 genome，执行恢复训练和后端真实 realization，将三指标与 actual graph evidence 原子回流，在固定预算内更新实测 Pareto。

不要把“发明了新的 NSGA-II、XGBoost schedule tuner 或新的树模型”写成贡献。创新点应放在问题建模、证据合同和闭环搜索机制。

---

## 7. Framework Figure 修改

当前 [framework.pdf](${V2X_ROOT}/multi_agent/paper/Latex/Figures/framework.pdf) 必须重绘，原因如下：

1. 左上角仍写 `SMBO + NSGA-II`。
2. cost model 仍统一写 `LightGBM`。
3. inner scheduling 仍统一写 `AutoTVM / XGBoost`。
4. 未显示 `fixed SearchTask/profile`，容易被理解成 backend 也是基因。
5. 未显示 `Gold176 -> Round 0`。
6. 未区分 candidate surrogate graph features 和 measured actual graph features。
7. 未显示恢复训练、完整 AP、SHA evidence 与 batch atomic feedback。
8. 图中存在多个未收口布局草案拼接，不适合作为正式论文图。

新图建议从左到右分为：

```text
Target Model + Fixed Profile
        |
Scanner / Capability Context
        |
Legal Genome Pool
        |
Gold176 + Metric-Specific Cost Models
        |
Full-Pool Prediction
        |
Pareto-Diversity Acquisition (B=4)
        |
Recovery + Backend Realization + AP/Latency/Energy
        |
Actual Graph Features + Evidence Bundle
        |
Atomic Feedback / Refit / Measured Pareto
```

TRT 和 TVM 画成两个独立 profile 实例，不画成同一 genome 内的下拉选项。

---

## 8. Experimental Results 修改

### 8.1 Table 1 当前可用边界

Pyramid 和 CoDriving 行可以保留为待最终证据审计的 Stage6 结果；F-Cooper 当前不能保留正式数值。

[第 455–462 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:455) 当前以下声明需要暂时删除：

```text
Every numerical entry ... is independently validated ...
GEAR achieves the lowest ... on all three target models.
```

原因：

- 32 号文档第 11 章的 F-Cooper 结果已撤回；
- 旧 winner 来自搜索预算外 probe，而不是 T16 正式在线轨迹；
- 旧压缩点未执行与 Pyramid/CoDriving 同等级的恢复训练合同；
- F-Cooper v2 正式 T16 已闭合，winner 已严格从 16 个在线终态中产生；但五臂对照和最终独立证据审计仍在进行，尚未得到 `paper_ready` 判定。

### 8.2 F-Cooper 正式准入条件

只有同时满足以下条件才允许回填 Table 1：

1. fresh scanner 与五轴候选空间审计通过；
2. recovery numeric gate 通过；
3. probe 标签、预算和 winner 完全隔离；
4. 4/4 rounds、16/16 正式在线反馈闭合；
5. 每轮使用 actual graph feedback；
6. winner 只从 T16 正式终态选择；
7. winner 完成三次独立 latency/energy 和一次 full AP 复核；
8. 五臂均为可信成功或明确失败终态；
9. `paper_ready=true`。

在此之前，Table 1 的 F-Cooper 三列应使用 `--` 或整列暂时删除，不得保留 pilot 数值。

### 8.3 结果解释

[第 507–516 行](${V2X_ROOT}/multi_agent/paper/Latex/AnonymousSubmission2027.tex:507) 的 F-Cooper 百分比和“unpruned INT8 incumbent”结论应全部撤回。

Pyramid/CoDriving 的解释也应避免把优势简单归因于“jointly searching schedule gene”。推荐解释为：

> GEAR 的优势来自 profile-conditioned measurement feedback 对结构与精度候选的重新排序，以及每个候选经过相同后端 realization 后的真实证据回流；它不是通过在外层显式枚举 schedule ID 获得优势。

---

## 9. 当前允许与禁止的论文声明

### 9.1 当前允许

- 结构宽度、精度与真实 latency 在目标后端上存在非单调关系。
- 固定 profile 下，FP16/INT8 的相对收益依赖模型图、shape、量化边界和后端 realization。
- Gold176 可以作为第一轮跨模型趋势 cold start，但不等于已经证明任意未见模型的普适泛化。
- 按目标选择的低容量树模型和 predicted-frontier diversity 已通过阶段 4 工程消融，可用于启动闭环搜索。
- Pyramid 和 CoDriving 的 actual-feature feedback v3 已验证“已测点 actual features 回流、未测点 surrogate features”的闭环合同。
- H800 上 TensorRT 是强部署后端；TVM 的研究价值主要是可编程、可检查和跨硬件能力，跨硬件性能优势仍需 Lane B/C 结果支持。

### 9.2 当前禁止

- GEAR 已在任意模型上自动发现全部可剪枝 dependency groups。
- backend 是同一次搜索中的可选 genome。
- 外层 genome 包含 graph features。
- 正式方法使用统一 LightGBM 三头和 pairwise rank loss。
- 正式在线候选由 NSGA-II 产生。
- capability scanner 可以不经测量直接判断 TVM 选 FP16、TRT 选 INT8。
- 只对最终 winner 做 AP，其余在线点使用 surrogate AP。
- Gold176 已证明对任意新模型的准确跨模型泛化。
- F-Cooper pilot 已证明 GEAR 获胜或可写入正式 Table 1。
- 当前 H800 TVM 结果已经证明 TVM 优于 cuDNN/TRT。
- Figure 1(b) 是同一后端、同一调优预算下的公平 FP32/FP16/INT8 对照。

---

## 10. 推荐修改顺序

1. 冻结方法名称为 GEAR，并确定标题。
2. 按第 5 节重写 Method，不在旧方法段上逐句修补。
3. 重绘 framework figure，使图与正式数据合同一致。
4. 修正 Introduction 的问题定义、贡献点和 Figure 1(b) 口径。
5. 在 F-Cooper v2 完成前撤下 F-Cooper 正式数值。
6. 根据最终 `paper_ready` 结果更新 Table 1 和结果解释。
7. 再撰写 abstract，避免摘要先于方法和正式主表冻结。
8. 删除 LaTeX 中 AAAI 模板示例正文，重新检查页数、引用和匿名性。

---

## 11. 主要依据

- [论文级最终实验与表格实现路线](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/论文级最终实验与表格实现路线_v1.md)
- [26 号：阶段 4 cost-model 与 acquisition 收口](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/26_7_14_交接文档_Feedback16_AP收口与阶段4问题反思_v1.md)
- [28 号：单目标模型、单 profile 与预算协议](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/28_7_14_交接文档_阶段5单目标模型单Profile自主搜索反思与预算协议_v1.md)
- [29 号：actual graph feature 回流反思](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/29_7_14_交接文档_阶段5端到端自动性与图特征回流反思_v1.md)
- [30 号：阶段 5 v2/v3 结论边界](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/30_7_14_交接文档_阶段5完整搜索与独立复测最终结论_v1.md)
- [31 号：框架缺口、Stage6 与历史 latency 口径](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/31*_7_14_交接文档_框架待完善项与阶段6六臂对照实施计划_v1.md)
- [32 号：F-Cooper pilot 撤回与正式 v2 合同](${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/32_7_14_交接文档_FCooper_Table1_TVM_CPU与Orin边缘部署计划_v1.md)
- [Stage5 SearchTask、genome、acquisition 与在线模型](${V2X_ROOT}/framework/stage5/single_target_search_v2.py)
- [Stage4 feature encoder 与模型选择](${V2X_ROOT}/framework/stage4/cost_model_selection_v1.py)
- [F-Cooper 正式 round advance](${V2X_ROOT}/scripts/stage5_advance_fcooper_round_v2.py)
- [F-Cooper batch 原子反馈](${V2X_ROOT}/scripts/fcooper_finalize_formal_round_v2.py)
