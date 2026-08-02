# 交接文档 24：Gold Cold-start 96 v3 最终收口

**日期**：2026-07-13  
**状态**：阶段 3 收口结论已撤回；TVM-INT8 24 行待补齐  
**目标**：记录 Gold Cold-start 96 的最终行级结果、feasibility 终态、证据产物和 24 组完整性审计。

## 1. 结论更正（2026-07-13）

此前将 24 行 TVM-INT8 记为最终 `numerical_feasibility_failure` 并宣布阶段 3 收口，**该结论不成立，现予撤回**。

直接失败原因是 Stage3 自动产物缺少同源静态输入/输出 `scale/zero_point`，属于 `blocked_missing_quant_contract`，不是已经证明无法解决的后端数值不可行。21 号文档已经记录 scale-aware TVM INT8 在代表宽度上完成 1789 帧检测评估，AP70 恢复到约 `0.3959-0.3979`，说明 TVM INT8 精度路径原则上可闭合。

因此当前表只能称为 **provisional terminal table**，不能作为最终 Gold Cold-start 96，也不能直接进入阶段 4 AP/cost-model 训练。

## 2. 当前 provisional 统计

当前聚合器形式上输出：

```text
70 measured_success_gold
+ 24 numerical_feasibility_failure
+ 2 feasibility_failure
+ 0 pending
= 96 rows
```

这 96 行严格对应：

```text
2 models x 12 widths x 2 q_mode x 2 compiler profiles
= 24 groups x 4 rows
= 96 rows
```

但研究语义上应重标为：

- 70 行完整绑定 H800 真测 latency、energy、AP30/AP50/AP70、性能产物 SHA 和 AP 报告 SHA；
- 24 行为 Pyramid/CoDriving 的 TVM-INT8 `blocked_missing_quant_contract`，必须修复并补测，不能作为最终 feasibility 标签；
- 2 行为 Pyramid TVM-FP16 shape feasibility failure；
- 没有 pending、历史 hand-rewrite 补行、AP 外推或跨 pipeline 指标拼接。

## 3. 模型级 provisional 分区

| 模型 | 完整 Gold AP | numerical feasibility failure | performance/shape feasibility failure | 合计 |
|---|---:|---:|---:|---:|
| Pyramid | 34 | 12 | 2 | 48 |
| CoDriving | 36 | 12 | 0 | 48 |
| 总计 | **70** | **24** | **2** | **96** |

CoDriving 的 36 行完整测量由以下路径组成：

```text
12 TVM FP16
12 TRT FP16
12 TRT INT8
```

其 12 行 TVM INT8 因缺少可验证的静态输入/输出量化参数而阻塞。该缺口来自数据/产物合同，不能训练为“TVM INT8 不可行”的负标签，也不能据此让搜索器学习避开 TVM INT8。

## 4. Provisional 机器可读产物

本机 provisional 目录：

```text
${V2X_ROOT}/results/stage3_gold96_final_v3_20260713/
```

| 文件 | 用途 | SHA256 |
|---|---|---|
| `gold96_final.csv` | cost model/人工审阅表 | `83e5c8915e18140a8c4b65bc1e080ec15cd70e93633c72ff4a57e6edcdb5a1e9` |
| `gold96_final.json` | canonical 96 行结果 | `20899ed8a665a781c1b1048c1fbd390f0381c5ddb363f4ec0e061e741ec4b5f2` |
| `gold96_final.jsonl` | 流式训练/回流入口 | `d113c01ec8fa526931dc45900ad2aacf2556cb3c6b5892660f4c7a37ed4495bb` |
| `gold96_audit.json` | 24 组完整性与终态审计 | `314d15242363d1fa7a4499d08945c830bf7fa4aec1d3dfacd08def2f56caf247` |

注意：以上 SHA 对应 2026-07-13 provisional 聚合产物，禁止作为阶段 4 最终训练输入。24 行补齐后必须重新生成目录和 SHA，不能覆盖性沿用本表。

## 5. 24 组形式完整性审计

最终审计结果：

```text
groups                  = 24
row_count per group     = 4
unique manifest_job_id  = 96
all_terminal groups     = 24/24
pending rows            = 0
```

每个 `(model, width)` 都保留四条配对语义：

```text
TVM FP16 / TVM INT8 / TRT FP16 / TRT INT8
```

当前 `all_terminal=true` 只是聚合器的形式结果。由于 24 行被错误归类为 feasibility，该字段不能支撑阶段 3 研究收口；修复后必须重新审计。

## 6. AP 与数值正确性缺口

1. numerical pass 的 70 行均执行完整 AP，不使用 sanity AP 代替 full AP。
2. FP16 与 INT8 之间没有共享 AP。
3. 本轮最终表未依赖跨后端 AP 外推；每个 measured 行绑定自身后端报告。
4. TVM-INT8 当前 AP 字段为空，且失败原因是可修复的量化合同缺失，因此不能作为 Gold 终态。
5. 两行 Pyramid TVM-FP16 shape failure 不伪造 latency、energy 或 AP。

## 7. 执行中修复的公共入口问题

本轮发现并修复了以下证据链问题，旧失败记录不得作为论文结论：

1. AP 输出路径在 runner 切换工作目录后发生碰撞或解析到错误根目录；修复为 capability/profile 隔离并预先绝对化。
2. Pyramid TVM-INT8 persistent worker 使用相对 request 路径；修复为绝对路径。
3. CoDriving TRT pilot engine 相对路径在切换仓库后解析错误；修复为切换前绝对化输入产物。
4. CoDriving TVM-FP16 未创建评估目录；修复为显式创建。
5. CoDriving TVM-INT8 runtime weights 实际位于 `native_direct_reference/`；修复产物定位后，继续由静态量化参数门禁决定 numerical feasibility。
6. CoDriving 一个场景会产生两次 backbone 调用；最终合同区分 `engine_samples`（场景数）和 `engine_calls`（调用数），不再误判 `16 scenes / 32 calls`。

所有无效 harness/path 失败均已从最终终态中排除；最终表只接受当前报告内容、报告 SHA 与计划绑定一致的证据。

## 8. 阶段 3 是否收口

**阶段 3 未收口。** 剩余工作固定为 24 行 TVM-INT8：

1. 为 Pyramid、CoDriving 各 12 个 width 生成与当前自动 TVM INT8 产物同源的 calibration manifest；
2. 绑定 graph input 与三个 graph output 的静态 scale/zero-point、ONNX/route/checkpoint/dataset SHA；
3. 先完成 16 样本数值 sanity，区分真实 numerical failure 与实现缺口；
4. sanity 通过的行完成 1789 帧 AP；
5. 以同一 pipeline 重新确认 latency/energy，禁止拼接旧 scale-aware AP 与当前无 scale 性能；
6. 最终只允许 measured Gold 或经过正确量化合同后仍失败的真实 feasibility 进入 96 行表。

在以上工作完成前，不启动阶段 4 locked holdout/cost-model 主实验。

## 9. 下一执行入口

下一阶段仍属于阶段 3 修复：

```text
24 行 TVM-INT8 calibration/quant contract
-> 16-sample numerical sanity
-> 1789-frame AP
-> 同源 latency/energy 复核
-> 96 行重新聚合与审计
```

训练数据装配必须遵守：

- 70 行 measured success 可进入对应 value heads；
- 26 行 failure 进入 feasibility head；
- failure 行不得以 `AP=0` 注入 AP 回归；
- split 必须按 `(model, width)` 四行整组切分，禁止后端/精度泄漏。

## 10. TVM-INT8 修复后最终收口（2026-07-13）

阶段 3 已正式收口。24 行 TVM-INT8 已完成同源静态量化合同、自动 Route-B build、16 样本数值 gate、1789 样本完整 AP、latency 与 energy 绑定；不再保留 provisional numerical feasibility failure。

```text
measured_success_gold = 94
confirmed shape failure = 2
pending = 0
complete groups = 24/24
```

- Pyramid TVM-INT8：12/12 measured；AP30/AP50/AP70 范围为 `0.790-0.811 / 0.735-0.761 / 0.532-0.577`。
- CoDriving TVM-INT8：12/12 measured；AP30/AP50/AP70 范围为 `0.586-0.738 / 0.510-0.644 / 0.317-0.420`。
- `64x96x192`、`24x32x96`、`64x64x128` 的 absmax sanity 不足，自动 percentile-99.99 静态标定通过，其中三行均完成 full AP；该选择来自 calibration/gate，不是手写层规则。
- 唯一两行 failure 仍为 Pyramid TVM-FP16 shape/codegen failure，不伪造性能或 AP。

最终机器可读目录：

```text
${V2X_ROOT}/results/stage3_gold96_final_v3_20260713_int8_repaired/
```

| 文件 | SHA256 |
|---|---|
| `gold96_final.csv` | `7ddf24f2f12b2cee4f61bad370186949f2b9924fc2e3fec95e3aa0530137b8a6` |
| `gold96_final.json` | `e42ea377683c76901987cee101693d7279c042e7d39ff24cdd269ae97ca412b2` |
| `gold96_final.jsonl` | `f608f841203fee4d42f7fb0ab1d73412f13de36ed32db829cbdc5adff32168e5` |
| `gold96_audit.json` | `ae69e87fae907c4ac35509837eebb8231d3697ac54b3f706879c95f108e62cb9` |

本轮同时修复三项公共入口：CoDriving 主模型 GPU 绑定、full-AP `eval/` 目录创建、sanity report path-bound SHA 一致性。上述 harness 失败均未计入研究结论。

## 11. 对此前 24 行失败问题的反思

### 11.1 核心误判

此前把 24 行 TVM-INT8 记为 `numerical_feasibility_failure`，实际把“证据链尚未实现完整”错误等同于“自动后端数值不可行”。当时缺少与当前 Route-B 编译产物同源的静态输入、中间张量和输出 scale/zero-point，AP runner 因而不能执行有效的量化、反量化和完整检测评估。这个状态最多只能标记为 `blocked_missing_quant_contract`，不能作为 INT8 负样本。

这一误判尤其不应发生，因为 21 号文档中的代表点已经给出有限但非零 AP，说明问题具有可修复性。正确做法应是先追查代表点与批量入口的合同差异，再决定是否形成 feasibility 结论，而不是直接将缺失 AP 解释为量化失败。

### 11.2 实际问题分层

24 行问题最终由四层原因组成：

1. **量化合同缺失**：旧批量入口只有 uint8 artifact，没有为 graph input、Conv/Add/ReLU/Identity 中间张量和三个 graph output 绑定静态 scale/zero-point 及其 calibration、ONNX SHA。
2. **标定策略不充分**：常规 train-16 absmax 合同使 3 个 CoDriving width 的 `res1/res2` 相关性不足；自动 percentile-99.99 标定排除离群值影响后，`64x96x192`、`24x32x96`、`64x64x128` 均通过 sanity 和 full AP。该修复是数据驱动的 calibration 选择，不是手写层名单。
3. **执行入口缺陷**：sanity SHA 的两端算法不一致；CoDriving 主模型忽略 `--gpu-id` 而落到默认 GPU 0；full AP 未预建 `eval/` 目录；切换工作目录后部分相对路径失效。这些错误会分别表现为 SHA mismatch、CUDA OOM、末尾写报告失败和找不到产物，但都不是模型数值失败。
4. **聚合器合同滞后**：Pyramid 新报告使用 `gates.full_1789`、`ap_row_allowed` 和 `feasibility_blockers`，旧 finalizer 仍要求 `smoke_gate_passed`，导致已有完整证据的 12 行被错误保留为 pending。

### 11.3 为什么旧结论不能进入训练或论文

- 24 行旧 failure 不具备完整静态量化合同，不能进入 feasibility head；否则模型会错误学习“TVM + INT8 必然不可行”。
- AP 缺失不能填成 `AP=0`，因为缺失表示评估未成立，而不是模型检测精度为零。
- harness/path/GPU/OOM 失败只能进入执行审计，不能作为后端能力证据。
- 修复后的 24/24 行均为 measured Gold，因此在本批 width 上没有证据支持“TVM-INT8 数值不可行”；但这也不等于 INT8 必然比 FP16 更快，速度优劣仍由真实 latency/energy 数据和 cost model 学习。

### 11.4 后续必须执行的负标签准入顺序

任何量化臂只有依次满足以下检查后，才能形成论文级 numerical feasibility 结论：

```text
build 成功
-> 自动 TensorCore/route 覆盖与产物绑定成立
-> candidate/native direct exactness 通过
-> 同源静态 quant contract 完整
-> 16-sample numerical sanity
-> 1789-sample full AP
-> 同一合同下 latency/energy 绑定
```

其中任何基础设施或合同步骤缺失，只能标记为 `blocked` 并进入修复队列；只有合同完整且 numerical gate 仍失败，才允许写入 numerical feasibility failure。shape/codegen failure 则必须像现有两行 TVM-FP16 一样单独保留，不能与数值失败混淆。

### 11.5 对自动搜索叙事的修正

本轮不支持删除 TVM-INT8 搜索臂，也不支持人为规定某个后端必须选择 FP16 或 INT8。正确叙事是：能力扫描和自动 calibration 建立候选执行上下文，冷启动 Gold 数据提供 AP、latency、energy 与 feasibility 证据，搜索模型再按 width、量化模式和 capability context 学习选择。三行 percentile fallback 也应被记录为自动标定上下文，而不是人工性能规则。

第 3-9 节保留的是 provisional 阶段的历史快照；其“阶段 3 未收口”和“24 行失败”状态已由第 10-11 节及最终 Gold96 审计取代。

## 12. 阶段 4 前的 Gold96 数据集反思

### 12.1 当前 96 行能证明什么

Gold96 已证明统一数据合同和四臂测量闭环可用，但不能仅凭“96”这个行数宣称 cost model 数据充分。四条 `(TVM FP16, TVM INT8, TRT FP16, TRT INT8)` 是同一个 `(model,width)` 的强相关配对观测，因此防泄漏后的有效独立单元是 24 个组，而不是 96 个独立样本：

| 模型 | 表格行 | 独立 width 组 | measured | failure |
|---|---:|---:|---:|---:|
| Pyramid | 48 | 12 | 46 | 2 |
| CoDriving | 48 | 12 | 48 | 0 |
| 合计 | 96 | 24 | 94 | 2 |

这批数据适合作为统一口径的 **seed Gold/cold-start base pool**，用于学习初始 shape、量化模式和 capability context 的条件趋势；它还不是跨模型泛化已经成立的证据。

### 12.2 已确认的数据风险

1. **跨模型标签尺度不同**：Pyramid 与 CoDriving 的 AP70 measured 均值约为 `0.521` 和 `0.354`，latency 均值约为 `6.08 ms` 和 `1.64 ms`，energy 均值约为 `1.21 J` 和 `0.28 J`。直接混合绝对值会让模型把数据集/模型基线当成剪枝或量化效应。
2. **model 与 shape 混杂**：每个模型只有 12 个 width，两个模型仅共享 `32x32x128`、`48x64x128` 两个 width。模型差异与宽度分布无法充分解耦。
3. **有效样本量较小**：24 个组不足以支持高容量多头模型、复杂交互或可靠的 leave-one-model-out 结论；普通随机行切分会虚高精度。
4. **失败标签极少**：只有 2 个 TVM-FP16 shape failure，不能据此训练稳定的二分类 feasibility head，也不能推断失败边界。
5. **标定上下文存在差异**：3 行 CoDriving INT8 使用 percentile-99.99，其余使用 absmax。若不记录 calibration method、clip ratio 和统计量，cost model 会看到无法解释的同臂差异。
6. **测量噪声尚未量化**：当前每行有 latency repeats 和 energy window，但缺少跨时段重复锚点；AP 还受 checkpoint、训练 seed 和后处理合同影响。
7. **Pareto 不可跨模型直接合并**：两个模型对应不同任务基线，绝对 AP/latency/energy 的共同前沿没有论文含义。Pareto、regret 和 HV 必须先在模型内计算，再汇总相对改善。
8. **迁移时容易泄漏**：若目标模型 calibration 点同时用于超参数选择、acquisition 评估或最终测试，会把 few-shot 校准误写成泛化能力。

### 12.3 推荐的两层预测目标

不建议只训练一个跨模型 absolute-value regressor。推荐拆成：

```text
global trend model
  学习 shape/capability/q_mode 引起的相对变化、排序和可迁移残差

target-model calibrator
  用目标模型少量实测锚点恢复 AP、latency、energy 的绝对基线与尺度
```

| 指标 | global target | 目标模型校准 |
|---|---|---|
| AP | `Delta AP = AP - AP_ref(model)`；同时保留同 width 的量化 AP 差值 | 加性 intercept/scale 或低容量 residual calibrator |
| latency | `log(latency / latency_ref(model,hardware))`、同组 speedup/ranking | 指数还原后的 scale/intercept |
| energy | `log(energy / energy_ref(model,hardware))`、同组 energy ratio | 指数还原后的 scale/intercept |
| feasibility | 独立状态/不确定性，不向 value head 填零 | 目标模型补测后更新状态 |

`AP_ref`、`latency_ref` 和 `energy_ref` 必须来自每个模型固定的 reference checkpoint/profile，并写入 manifest；不能由测试集统计量临时计算。absolute head 可作为对照，但只有相对/残差目标在 grouped holdout 和迁移测试中更好时才进入主方法。

### 12.4 模型输入与搜索变量边界

- 不用裸 `model_id` 代替泛化能力；应输入可扫描的结构描述，如各 stage Conv shape、FLOPs、参数量、activation bytes、group/depthwise 比例、通道对齐和算术强度。
- backend/capability profile、TensorCore 覆盖、Q/DQ propagation、reformat 数等是扫描/测量上下文，不是人工写入 genome 的先验规则。
- `calibration_method` 及其统计量属于执行上下文；搜索器可以比较其真实证据，但不能按模型名硬编码 percentile 或 absmax。
- 主搜索动作仍保持剪枝向量与 `q_mode`；上下文负责解释“同一动作在不同模型/后端上为何结果不同”。

### 12.5 目标模型 few-shot 校准协议

对新模型迁移时采用三份严格隔离的数据：

1. **base pool**：当前 Gold96 及之后增加的其他模型/shape Gold，用于训练 global trend model。
2. **calibration pool**：从目标模型搜索空间选择少量、空间分散的完整四臂组，覆盖 base、alignment trap、off-diagonal/AP-sensitive 区域；只拟合低容量校准器并允许实测回流。
3. **locked transfer test**：目标模型剩余组，不能参与 anchor 选择、模型选择、早停或 calibration。

校准点数量不预先写死。以 `K=2/4/6/8` 个完整组绘制 transfer learning curve，选择达到预注册误差、排序和 Pareto recall 门槛的最小 K。论文至少报告：`global-only`、`local-only K-shot`、`global + K-shot calibration` 三个对照，以检查迁移收益和 negative transfer。

### 12.6 阶段 4 前必须完成的数据充分性审查

1. **严格 grouped split**：四臂同组；所有 scaler、reference 和特征选择只在 train/calibration 内拟合。
2. **group-level learning curve**：按 6/12/18/24 组或可行的嵌套子集训练，报告 bootstrap 置信区间；若误差和 Pareto recall 未趋于稳定，则继续补组。
3. **双向 few-shot transfer**：Pyramid -> CoDriving+K 与 CoDriving -> Pyramid+K。只有两个模型时该实验是迁移 pilot，不能表述为普适跨模型泛化。
4. **共同 width 对照**：优先补充跨模型共享 width，使 model effect 与 shape effect 可辨识；当前只有 2 个共享点明显不足。
5. **重复测量审计**：选取至少 base、小通道、alignment trap 和大模型四类锚点跨时段复测，估计 latency/energy CV 与 AP 可重复性。
6. **失败边界补测**：围绕两行 shape failure 做局部、对称的可行/不可行补点；样本量不足时 feasibility 只作为 gate/uncertainty，不训练高容量分类器。
7. **模型容量约束**：先从低容量树模型和少量 target/head 开始；任何额外预测头必须通过 grouped holdout 的消融证明有增益。

### 12.7 是否需要扩大 cold-start

是否扩充不能由“96 行看起来很多”决定，而由以下停止条件决定：grouped learning curve 是否进入平台期；Spearman/top-k 与 measured Pareto recall 是否稳定；预测区间 coverage 是否合理；`global + K-shot` 是否持续优于 global-only 和 local-only；新增完整组后搜索选择与 measured HV/regret 是否显著改善。

若未达标，新增数据应优先提高信息量，而不是均匀堆行：先补跨模型共同 width、当前特征空间稀疏区、模型分歧点、失败边界和高不确定度候选。每次仍以完整四臂组回流，保持自动搜索叙事和 paired evidence。

### 12.8 阶段 4 的修订入口

阶段 3 的“测量闭环收口”不等于阶段 4 的“数据充分性已验收”。进入正式 cost-model/loss 选择前，先执行：

```text
Gold96 descriptive audit
-> grouped leakage audit
-> normalized target comparison
-> group-level learning curve
-> bidirectional K-shot transfer pilot
-> decide: expand cold-start or lock Stage4 dataset
```

只有上述审查给出可接受的排序、Pareto recall、校准误差和不确定性后，才能锁定阶段 4 数据；否则先按信息增益补点。这样既保留“大范围 cold-start 学趋势 + 目标模型小范围实测校准”的方法叙事，也避免让两个模型的绝对数值基线污染同一个回归目标。
