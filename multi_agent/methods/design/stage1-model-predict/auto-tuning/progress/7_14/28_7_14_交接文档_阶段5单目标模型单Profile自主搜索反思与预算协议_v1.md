# 交接文档：阶段 5 单目标模型、单 Profile 自主搜索反思与预算协议 v1

> 日期：2026-07-18  
> 状态：阶段 5 搜索架构已重新定义；27 号 P0/P1 的测量证据保留，但其双模型同步与四臂组 acquisition 设计不再作为正式主搜索协议  
> 下一入口：单次任务固定 `target_model + hardware_id + capability_profile_id`，搜索器只优化结构压缩与精度 genome  
> 核心要求：完成任务配置后，候选生成、批量选择、测量、回流、再训练、Pareto 更新和停止均由框架执行，不由人工逐点指挥

---

## 1. 本次反思要纠正什么

用户最初要求的是：

> 冷启动数据可以来自多个模型、多个硬件或多个 compiler profile；但部署搜索一旦启动，就只面向输入的一个目标模型和一个确定部署环境，由框架自动寻找该环境下的结构压缩与精度 Pareto 前沿。

27 号文档和已实现的 Stage5 P0/P1 偏离了该目标，主要表现在：

1. 把 Pyramid 和 CoDriving 同时放入一个 round，每轮人工规定“每个模型各选一组”。
2. 把 `(model,width)` 四臂完整组当成 acquisition 最小单位，选中一个 width 就强制测量 TVM/TRT × FP16/INT8 四行。
3. 虽然 width 由 acquisition 给出，但模型配额、profile 配额和四臂展开方式由 orchestration 预先写死，搜索器没有真正决定完整的结构与精度 genome。
4. round-1 继续按两个模型各生成一个 width，证明的是批处理测量入口，而不是“输入一个目标模型后自主搜索其前沿”。

因此必须区分：

- **冷启动证据组织单位**：可以使用配对完整组，便于学习趋势、控制数据泄漏和比较 capability context；
- **在线搜索决策单位**：必须是固定部署上下文下的单个完整 genome，不应自动展开成四臂组。

---

## 2. 根因分析

### 2.1 混淆了 global training 与 target search

Gold176 的作用是训练阶段 5 的初始 global cost model，并为新任务提供趋势先验。Feedback16 只用于阶段 4 增量回流机制验证，不进入阶段 5 的初始训练视图。Gold176 中同时出现 Pyramid 和 CoDriving，不意味着一次搜索任务也必须同时调度两个模型。

正确关系是：

```text
multi-model cold-start
        |
        v
global cost model
        |
        +--> Pyramid / H800 / TVM 独立搜索
        +--> Pyramid / H800 / TRT 独立搜索
        +--> CoDriving / H800 / TVM 独立搜索
        +--> CoDriving / H800 / TRT 独立搜索
```

四个任务可以复用相同算法和冻结初始数据，但状态、预算、反馈、Pareto 和停止条件彼此独立。

### 2.2 混淆了审计组与 acquisition candidate

冷启动阶段按同一 width 收集四臂数据有合理性：它提供反事实配对，可以比较相同结构在不同 profile 和精度下的结果，并保证 grouped split 不泄漏。

但在线搜索阶段 capability profile 已经固定。此时一个合法候选应为：

```text
(w0, w1, w2, q_mode)
```

而不是：

```text
(w0, w1, w2) -> 强制展开 TVM-FP16、TVM-INT8、TRT-FP16、TRT-INT8
```

若 acquisition 独立选择了同一 width 的 FP16 和 INT8，它们可以同时进入一批；但这必须是预测和采集函数的结果，不能由固定配对规则强制产生。

### 2.3 Smoke 入口被误写成了搜索方法

“每个模型各一组、每组四臂”适合作为 runner wiring smoke，因为可以一次覆盖所有执行路径；它不适合作为正式搜索协议。之前把 smoke 的覆盖目标延伸为主搜索的 acquisition 规则，是本轮最主要的设计错误。

### 2.4 阶段 4 acquisition 证据与新决策单位不完全一致

阶段 4 中 `predicted_frontier_diversity` 优于 random 的结果来自 `(model,width)` 完整组 replay。它证明该策略在组级候选上有价值，但不能直接证明它在 686 个 `(width,q_mode)` 单 genome 候选上仍然保持相同优势。

因此 v2 真实测量前必须增加单 genome compatibility replay：

- 固定一个 target model 和一个 capability profile；
- 把已有 gold 行拆成单 genome 候选；
- 对比 `predicted_frontier_diversity` 与 random 的 HV/recall/精度选择；
- 检查 acquisition 是否会根据证据选择 q_mode，而不是在内部仍按 width 聚合；
- 该 replay 只验证接口与采样单位兼容性，不重新无限期优化阶段 4。

如果单 genome replay 明显弱于 random，必须先修正 acquisition scoring，再启动 H800 16 点预算；不能依靠旧组级结果直接放行。

---

## 3. 修正后的阶段 5 基本命题

一次 Stage5 搜索任务固定：

```text
SearchTask = (
    target_model,
    hardware_id,
    capability_profile_id,
    coldstart_version,
    search_space_version,
    objective_contract,
    budget_contract,
    seed
)
```

其中：

- `target_model`：本轮唯一目标模型，例如 Pyramid；
- `hardware_id`：实际部署设备，例如 H800；
- `capability_profile_id`：能力扫描后冻结的 compiler/backend 上下文；
- compiler/backend 不是 genome，也不能在搜索过程中切换；
- capability scan 只生成和验证部署上下文，不负责根据历史性能替用户选择 TVM 或 TRT；
- 搜索 genome 只包含结构压缩和精度决策；
- 一次任务只产生一个目标模型、一个硬件和一个 profile 下的 Pareto 前沿。

正式入口应类似：

```text
stage5_single_target_search_v2.py \
  --target-model pyramid \
  --hardware-id h800 \
  --capability-profile h800-tvm-probe-conditioned-v3 \
  --search-budget 16 \
  --batch-size 4 \
  --seed 20260718
```

任务启动后，不再由人工指定下一 width、下一精度或各模型配额。

---

## 4. 为什么 TRT 很强仍然必须研究 TVM

论文不能把 TVM 描述成“为了凑一个较慢后端”。TRT 与 TVM 在研究中承担不同角色。

### 4.1 原因一：硬件覆盖范围不同

TensorRT 的官方定位是针对 NVIDIA GPU 的高性能推理 SDK；其支持矩阵也以 NVIDIA GPU compute capability、CUDA 和对应平台为边界。TVM 则通过显式 `Target`/target kind 描述编译目标，定位是可编程的统一机器学习编译栈。

这意味着：

- TRT 可以代表 NVIDIA 平台上的强工业后端和性能参照；
- TVM 承担框架向 CPU、其他 GPU 或专用加速器迁移的开放编译入口；
- 协同感知部署可能覆盖车端、路侧和边缘设备，不能把全部论文结论限制为“只要使用 NVIDIA + TRT”。

官方依据：

- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html)
- [NVIDIA TensorRT Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- [Apache TVM Target API](https://tvm.apache.org/docs/reference/api/python/target.html)

### 4.2 原因二：TVM 提供可观测、可扩展的编译研究接口

本论文研究的不只是“调用最快的现成 SDK”，还包括 capability scan、图边界、TensorIR、tensorization、MetaSchedule 和自动后端适配。TVM 官方架构明确暴露 Relax/TensorIR transformation、schedule、MetaSchedule、operator fusion 和 target lowering；BYOC 还允许把匹配子图交给外部 codegen。

因此 TVM 能回答 TRT 黑盒性能数字本身无法回答的问题：

- 为什么某种剪枝 shape 无法有效 tensorize；
- 为什么 INT8 的 Q/DQ、requant 或 materialization 边界吞掉收益；
- 哪些 capability 特征会改变最优精度选择；
- 如何把同一个自动搜索框架迁移到新的编译目标。

官方依据：

- [Apache TVM Design and Architecture](https://tvm.apache.org/docs/arch/index.html)
- [Apache TVM TensorIR](https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html)
- [Apache TVM External Library Dispatch/BYOC](https://tvm.apache.org/docs/arch/external_library_dispatch.html)

### 4.3 原因三：验证“最优策略依赖部署上下文”是论文命题的一部分

如果只在 TRT 上实验，框架容易退化为“所有任务都使用 TRT 的默认强优化”。同时研究 TVM/TRT 的意义不是在同一次搜索中让 backend 竞争，而是分别运行同一搜索算法，观察：

- 同一目标模型在不同 capability profile 下是否形成不同 Pareto；
- INT8 是否在一个 profile 中占优、在另一个 profile 中被 FP16 支配；
- 固定“软件侧统一采用 INT8”的解耦方法为什么可能失败；
- capability-conditioned cost model 是否比忽略部署上下文的模型更可靠。

这是对论文核心论点的正面验证：**结构压缩与精度决策不能脱离实际硬件和编译能力独立确定。**

### 4.4 原因四：TRT 是强参照，不是对 TVM 的替代

TRT 的图优化、融合和 kernel auto-tuning 能提供 NVIDIA H800 上的重要性能上界参照。TVM 若慢于 TRT，结果仍然有价值：它量化了开放自动编译路径与成熟商业后端之间的差距，也能检验本文搜索器是否会在不同 profile 下自动规避无收益精度配置。

论文应采用以下叙事：

> 本文不声称 TVM 在 NVIDIA GPU 上普遍超过 TensorRT。TensorRT 用于代表 NVIDIA 专用高性能部署路径，TVM 用于验证开放、可观测、可迁移的自动编译路径。本文的贡献是让相同的结构压缩与精度搜索方法在固定 capability context 下自主形成不同决策，而不是预先写死某个 backend 或精度必然最优。

### 4.5 原因五：TRT engine 本身也存在部署边界

NVIDIA 官方文档指出，TensorRT serialized engine 默认与构建设备类型、平台和版本兼容条件相关；扩大 hardware compatibility 可能牺牲部分性能。这进一步说明“在 H800 上获得一个 TRT 最快结果”不能替代对不同部署上下文进行能力扫描和独立搜索。

官方依据：[TensorRT Engine Compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)。

---

## 5. 修正后的 genome、候选与测量单位

### 5.1 Genome

当前主空间保持：

```text
g = (w0, w1, w2, q_mode)
q_mode in {fp16, int8}
```

其中：

- `target_model` 不在 genome 中；
- `hardware_id` 不在 genome 中；
- `capability_profile_id` 不在 genome 中；
- backend 不在 genome 中；
- schedule route 不作为人工策略 gene；它由已冻结 profile 对应的自动 backend runner 实现。

### 5.2 候选池

按当前离散 width 设计，每个任务有：

```text
7 x 7 x 7 x 2 = 686 个候选 genome
```

这 686 个候选都属于同一个目标模型、硬件和 capability profile。候选特征由：

- width/shape；
- 目标模型 graph features；
- `q_mode`；
- 已冻结 capability features；
- source confidence/feasibility 状态

组成。后端名称只用于找到已经冻结的 runner，不作为模型可以选择的动作。

### 5.3 在线测量单位

一个 acquisition candidate 对应一行真实结果：

```text
(target_model, hardware_id, capability_profile_id, w0, w1, w2, q_mode)
-> latency, energy, AP30/AP50/AP70 or feasibility terminal state
```

不再因为选中一个 width 自动生成另一种 q_mode，也不再自动生成另一个 compiler profile 的结果。

---

## 6. Cold-start 数据如何进入单目标搜索

四个 v2 SearchTask 的初始训练视图统一冻结为：

```text
D_init_v2 = Gold176 (initial_coldstart)
总计 176 行、44 个历史完整组
```

`Feedback16` 是阶段 4 的增量学习与回流机制验证证据，不属于阶段 5 冻结 cold-start。它与 27 号 8 行 `pre_stage5_smoke`、INT8 专项补点及其他历史实测结果一并排除于 `D_init_v2`，不得向 round-0 暴露标签、改变候选排序或缩小搜索空间。它们可以保留为诊断证据，但不能冒充正式搜索输入或轨迹。

Gold176 的角色如下：

1. 多模型、多 profile 数据用于训练 global trend model。
2. 任务启动时，将固定 `target_model + capability_profile` 作为条件输入，不把其他模型混入当前 Pareto。
3. 如果 cold-start 已含目标模型数据，可用于初始 anchor/residual；如果没有，round-0 直接由 global model 和 uncertainty/diversity 选择首批点。
4. 首批真实反馈自动完成目标模型数值尺度校准，不要求人工指定 calibration width。
5. 当前任务产生的数据只回流当前任务；论文四个独立搜索完成前，不允许前一个任务的新反馈改变后一个任务的初始 cold-start，以保证公平。
6. 四个任务全部结束后，新增数据才可作为下一版本 global dataset 的候选补充。

每个 SearchTask 的完整搜索宇宙固定为 `7×7×7×2=686` 个 genome。Gold176 中属于当前模型、profile 且落在规范 343-width 网格内的已知行只进入初始 observed set，不重复消耗在线预算；因此 Pyramid 两个任务各有 42 行已知、round-0 未观测候选为 644，CoDriving 两个任务各有 38 行已知、round-0 未观测候选为 648。Gold176 中网格外的历史点仍可训练 global trend model，但不能从规范候选全集中扣除。论文统一报告 `M=686`，并另外报告初始 observed/unobserved 数，不能把历史 source 数量称为候选池规模。

因此，“全局趋势学习 + 目标任务在线校准”仍保留，但校准发生在单目标任务内部。

---

## 7. 完整搜索预算

### 7.1 论文主实验的四个独立任务

| Search ID | target model | hardware | capability profile | 独立 Pareto |
|---|---|---|---|---|
| S5-PYR-TVM | Pyramid | H800 | H800-TVM automatic | 是 |
| S5-PYR-TRT | Pyramid | H800 | H800-TRT | 是 |
| S5-COD-TVM | CoDriving | H800 | H800-TVM automatic | 是 |
| S5-COD-TRT | CoDriving | H800 | H800-TRT | 是 |

四个任务使用相同搜索空间、预算、batch size、cost-model 配置和停止口径，但不能共享在线状态。

### 7.2 固定搜索预算

阶段 5 v2 首版论文预算冻结为：

```text
N_search = 16 个新 genome / SearchTask
batch_size = 4
round_count = 4
四个任务总在线搜索预算 = 64 个新 genome
```

每个 genome 必须获得 latency、energy、full AP 或可信 feasibility terminal state。预算按 genome 计数，不再使用“width 组数”或“四臂行数”。

除候选数外还必须单独审计：

- 新 materialize 的 unique width 数；
- checkpoint/ONNX/calibration 构建时间；
- backend build/tune GPU 时间；
- full AP GPU 时间；
- 搜索总 wall-clock。

同一任务内若 FP16/INT8 分别选中相同 width，可以复用该 width 的模型 source，但两次 backend build、测量和 AP 仍是两个 genome 预算。论文同时报告 sample budget 和 wall-clock，避免复用 source 导致表面测量数公平但实际成本不可比。

选择 16 的理由：

- 足以形成 4 次真实“预测 -> 批采样 -> 测量 -> 回流 -> 再预测”；
- 比旧计划每模型 4 至 8 个 width 四臂组的 32 至 64 行总量保持同一量级；
- 四个任务合计 64 行 full AP，能够在当前时间和 H800 资源约束下完成；
- 阶段 4 的目的不是第一次就精确预测全部 686 个候选，而是用有限预算提高 Pareto 发现效率。

### 7.3 独立最终验证预算

搜索预算之外，每个任务允许：

```text
N_verify <= 4 个最终可信 Pareto 配置
```

验证配置只在搜索结束后确定：

- latency/energy 进行 3 次独立重复测量并报告均值、标准差和 CV；
- AP 使用冻结 artifact 完成一次 full evaluation 和 SHA 审计；
- 验证结果不再回流 cost model，也不改变搜索轨迹；
- `N_verify` 单独报告，不得混入 16 个搜索预算。

四个任务最多增加 16 个验证配置。论文必须分别报告 search cost 与 verification cost。

### 7.4 旧 smoke 数据的预算处理

27 号 P0/P1 的 8 行数据是真实可信测量，保留为：

```text
training_source = pre_stage5_smoke
```

但它们：

- 不计入任何一个 v2 SearchTask 的 16 点预算；
- 不作为 v2 的 round-0/round-1 轨迹；
- 不进入 `D_init_v2` 或任何 SearchTask 的 initial training view；
- 只用于 runner、数值合同和 evidence binding 诊断；
- 不得用于声称 v2 acquisition 已完成两模型搜索。

---

## 8. 批量采样协议

### 8.1 每轮固定流程

每个 SearchTask 的第 `r` 轮执行：

```text
1. 冻结 D_r、模型 SHA、候选池和 RNG state
2. 对全部未测 genome 预测 latency/energy/AP/uncertainty/feasibility
3. predicted_frontier_diversity 一次选择 B=4 个单 genome
4. 生成 4 条 measurement jobs
5. 并行完成性能、能耗、full AP 或 terminal failure
6. 等待 4 条全部达到可信终态
7. 原子追加 batch feedback
8. target-wise accept/reject 更新预测头
9. 重算当前 profile 下的 measured Pareto/HV
10. 保存 checkpoint 后进入下一轮
```

若同一批含相同 width 的多个 q_mode，source materializer 只构建一次 width checkpoint/ONNX，再分派独立精度 job；这种复用不得改变候选身份或把两行合并为一个预算单位。

### 8.2 Batch 原子性

- 同一 batch 未全部终态前，不使用其中部分结果重训；
- GPU 可以并行执行，但 acquisition 不能因某一行先完成而临时改变本批剩余点；
- 每轮只更新一次 cost model；
- 这样可避免异步完成顺序和 GPU 调度对搜索轨迹造成不可复现实验偏差。

### 8.3 不设置人工精度配额

每批 4 个候选：

- 不要求 `2 FP16 + 2 INT8`；
- 不要求覆盖 4 个不同 width；
- 不要求 FP16/INT8 成对；
- 不要求与另一个 backend 选择同一结构；
- 只执行 acquisition 在固定预算下给出的 4 个最高价值、满足多样性约束的单 genome。

如果一批全部为 FP16 或全部为 INT8，只要 acquisition 计算和输入证据可复核，就属于允许结果。

### 8.4 失败与预算计费

| 状态 | 是否消耗搜索预算 | 处理 |
|---|---:|---|
| build/shape feasibility failure | 是 | 记录真实失败并回流 feasibility evidence |
| numerical feasibility failure | 是 | 保留性能和数值失败证据，不伪造 AP |
| 正常成功 | 是 | 回流完整三目标 |
| GPU 抢占、SSH 中断、磁盘等基础设施失败 | 否 | 最多重试 2 次；仍失败则暂停任务修复 harness |
| runner 代码公共错误 | 否 | 暂停整批，修复后以相同 request SHA 恢复 |

真实候选失败本身是搜索信息，必须消耗预算；基础设施失败不是候选属性，不能污染 cost model 或预算。

批内账本进一步固定为：

1. 单个 job 遭遇可隔离的 GPU 抢占、SSH 中断或磁盘瞬时错误时，其他已成功 sibling 结果冻结但暂不回流；失败 job 以原 request SHA 重试，成功 sibling 不重测、不替换。
2. 待失败 job 也达到可信终态后，4 个 sibling 一次性提交；成功/真实 feasibility terminal 的 4 点各消耗一个预算。
3. 若确认是公共 runner、量化合同或共享 source bug，可能污染整批，则 4 点全部进入 `quarantined_infrastructure_batch`，不回流、不计预算；修复后必须以原 4 个 genome 和原 batch identity 整批复测，禁止选择性保留看起来较好的结果。
4. 所有重试、隔离、quarantine 和恢复 wall-clock 均进入资源审计。

### 8.5 主实验不提前停止

为保证四个任务及阶段 6 方法对比公平，论文主实验固定执行 16 个 genome，不采用基于当前结果的提前停止。HV 增益、frontier stability 和 uncertainty 只作为收敛曲线报告。

部署版以后可以增加提前停止，但不能回写改变本轮论文主表。

---

## 9. 完全自主的边界

人工只允许在任务启动前提供：

- 目标模型及其 source adapter；
- 硬件设备；
- capability scan 生成的 profile；
- 冻结搜索空间版本；
- 预算、batch size 和 seed。

任务启动后禁止人工：

- 指定下一 width 或 q_mode；
- 规定每批 FP16/INT8 数量；
- 规定 TVM/TRT 采用相同点；
- 因看到结果而临时加入邻域点；
- 删除失败候选；
- 修改 cost-model family、acquisition 或预算；
- 在四个独立任务间传播未预先冻结的在线反馈。

人工可以处理基础设施故障，但恢复后必须继续执行原 request 和原 batch，不能借修复机会改点。

---

## 10. Pareto、比较与论文表口径

每个 SearchTask 单独计算：

```text
maximize AP
minimize latency
minimize energy
```

最终得到四份前沿，不把 TVM/TRT 合并为一个搜索空间：

- Pyramid/TVM Pareto；
- Pyramid/TRT Pareto；
- CoDriving/TVM Pareto；
- CoDriving/TRT Pareto。

跨 profile 比较在搜索结束后进行，比较内容包括：

- 相同预算下 measured HV；
- FP16/INT8 在最终前沿中的占比；
- 最低延迟、最低能耗、最高 AP 三极值；
- 搜索过程中被自动支配的精度配置；
- capability profile 改变后 genome 选择如何变化。

论文不能把“TRT 比 TVM 快”本身写成方法贡献；应把它用于证明部署上下文会改变搜索结果，以及固定量化策略无法跨后端成立。

---

## 11. 对 27 号结果的重新定性

27 号文档中的以下证据继续有效：

- Stage5 runner 可以在 H800 上执行 TVM/TRT × FP16/INT8；
- Pyramid `32x64x96` 和 CoDriving `16x48x64` 的 8 行 latency、energy、AP 与 SHA；
- TVM-INT8 自动数值合同修复路径；
- feedback/request/source/checkpoint 的证据绑定和恢复机制。

以下结论撤回或降级：

- “双模型每轮各选一个 width”不是正确主搜索协议；
- “完整四臂组是在线 acquisition 最小单位”不再成立；
- 旧 round-1 的 Pyramid `32x48x128`、CoDriving `16x32x96` 不能直接作为 v2 下一批；
- 旧 P0/P1 只能证明基础设施闭环，不能证明单目标模型自主搜索已经启动。

27 号文档保留为历史错误和修复前证据，不覆盖删除。

---

## 12. 必要的框架改造

### 12.1 入口拆分

将 `stage5_two_model_search_v1.py` 降级为历史 smoke/orchestration，新增单任务入口：

```text
stage5_single_target_search_v2.py
```

入口一次只接受一个 `target_model/hardware/profile`。

### 12.2 Candidate 与 acquisition

- 删除 `group_budget_by_model`；
- 删除“按 model 各取一个组”的固定分配；
- candidate ID 必须包含 `q_mode`；
- acquisition 输入和输出均以单 genome 为单位；
- batch selector 一次返回 4 个 genome；
- source materialization 只针对被选 genome 对应的目标模型宽度执行，可复用同 width 已存在的 checkpoint。

### 12.3 Measurement request

每个选中 genome 只生成一条固定 profile 的测量请求。request 必须绑定：

- SearchTask SHA；
- genome SHA；
- capability profile SHA；
- model/checkpoint/source SHA；
- cost-model/acquisition SHA；
- round、batch 和 budget counter。

### 12.4 状态机

状态至少包括：

```text
initialized
-> batch_selected
-> measuring
-> batch_terminal
-> feedback_committed
-> model_updated
-> round_closed
-> budget_exhausted
-> final_validation
-> complete
```

恢复必须回到未完成状态，不能重新选择候选。

---

## 13. 下一阶段执行顺序

### A. 合同与测试修正

1. 实现单目标、单 profile SearchTask schema。
2. 将单 genome 设为 candidate、measurement 和 budget 单位。
3. 增加“不自动四臂展开”和“不跨模型调度”的负测试。
4. 增加 batch 原子回流、预算计数和 checkpoint/resume 测试。
5. 在冻结 Gold176 cold-start 上完成单目标、单 profile、单 genome compatibility replay，并与 random 比较。

### B. Dry-run

分别对四个任务生成 round-0，但不测量，检查：

- 每个任务只含一个 model/profile；
- 每批严格 4 个 genome/4 条请求；
- q_mode 由 acquisition 决定；
- 不读取其他任务的在线状态；
- 重复运行产生相同 selection/request SHA。
- 单 genome replay 不弱于 random，且 q_mode 确实参与 acquisition 排序。

### C. Pyramid/TVM 首轮真实闭环

先启动 `S5-PYR-TVM` 的第一个 4 点 batch。这里“先启动 Pyramid/TVM”只是实验执行顺序，不是人工指定候选。四个点必须由 v2 acquisition 自动产生。

首批成功后完成回流和 round-1，再确认预算、AP 和恢复合同没有问题。

### D. 四任务固定预算搜索

按冻结顺序运行：

```text
S5-PYR-TVM
S5-PYR-TRT
S5-COD-TVM
S5-COD-TRT
```

顺序只用于资源调度；各任务的初始数据快照必须在第一项启动前统一冻结，防止前序任务反馈泄漏到后序任务。

### E. 最终验证与阶段 6

四个任务各完成 16 点后，执行独立 Pareto 验证预算，冻结 T2；随后阶段 6 的 random/旧策略/六臂对照必须使用相同 16 点搜索预算和相同 4 点 batch 协议。

---

## 14. 收口判据

本次阶段 5 架构修正只有满足以下条件才算收口：

- 单次任务只含一个 target model、hardware 和 capability profile；
- 四个任务的完整候选空间均为 686，source 是否已存在不得改变搜索空间；
- round-0 训练输入严格为 Gold176，Feedback16、smoke 和其他历史实测标签均不可见；
- backend/profile 不出现在 genome 中；
- candidate、measurement、feedback 和 budget 均以单 genome 为单位；
- 四个独立任务各完成 16 个新 genome；
- 每批 4 点，批内不发生中途反馈；
- 没有人工精度配额、width 配对或跨 backend 配对；
- 每个成功 genome 有 latency、energy、full AP 和证据 SHA；
- 真实 feasibility failure 正确计入预算；
- 四份 Pareto、四条 HV 曲线和完整预算审计均已生成；
- 最终验证预算与搜索预算分开报告；
- 旧 pre-stage5 smoke 不冒充 v2 搜索轨迹。

---

## 15. 2026-07-18 候选池与历史证据泄漏反思

### 15.1 发现的问题

首版 v2 dry-run 报告 Pyramid/TVM、Pyramid/TRT 的有效候选各为 100，CoDriving/TVM、CoDriving/TRT 各为 640。复核后确认这些数字不是论文定义的搜索空间，而是历史 source registry 与错误过滤共同产生的实现结果：

1. registry 只登记了 60 个 Pyramid width，却登记了完整 343 个 CoDriving width，导致模型过去准备过多少 checkpoint/ONNX 决定当前能搜索什么；
2. round-0 cost model 使用了 `Gold176+Feedback16`，使阶段 4 后续 feedback 进入了阶段 5 初始视图；
3. `already_measured` 同时依据 cold-start 和非 cold-start 历史行过滤，`locked holdout` 也被直接移出候选空间；
4. 因此旧 `M0=100/640` 不可作为论文实验口径，已完成的 Pyramid/TVM round-0 也不能计入正式 16 点预算。

根因是把“source 是否已经物化”“数据是否曾经测过”和“论文允许搜索器在启动时知道什么”混成了一个过滤条件。这样会让历史实验路径影响正式搜索结果，破坏固定搜索空间、统一 cold-start 和独立任务公平性。

### 15.2 修正后的唯一口径

```text
U_task = {(w0,w1,w2,q_mode)} = 7×7×7×2 = 686
D0 = Gold176 initial_coldstart only
B = 4 genomes/round
T = 16 genomes/SearchTask = 4 rounds
```

四个任务都从同一个 Gold176 SHA 启动。Feedback16、pre-stage5 smoke、INT8 专项诊断、hand-rewrite、Route B/TRT 补点及本次错误 round-0 均不得进入 `D0`，不得作为排除候选的依据，也不得跨任务传播。

Gold176 中当前任务已经合法观测的行可以进入 measured Pareto 与模型训练，并从未观测集合中扣除，避免重复付费；这属于 cold-start 的正常使用，不是历史泄漏。对应口径为：

| SearchTask | 完整 `M` | Gold176 已知 | round-0 未观测 |
|---|---:|---:|---:|
| Pyramid/TVM | 686 | 42 | 644 |
| Pyramid/TRT | 686 | 42 | 644 |
| CoDriving/TVM | 686 | 38 | 648 |
| CoDriving/TRT | 686 | 38 | 648 |

source registry 必须覆盖四个任务的完整 343 个 width。checkpoint/ONNX/calibration 改为 acquisition 选中后按需物化；已有 source 只能减少构建时间，不能赋予候选资格。真实物化、编译或数值失败只有在对被选 genome 实际执行后，才能记录为 feasibility terminal 并计入预算。基础设施错误仍不计预算，修复后恢复原 request。

### 15.3 已完成错误 round-0 的处理

Pyramid/TVM 已完成的 4 行 latency、energy、full AP 和 SHA 仍是真实测量证据，但统一改标为 `invalidated_stage5_pilot`：

- 不进入新的 Gold176 `D0`；
- 不计入任一 SearchTask 的 `T=16`；
- 不用于生成正式 round-1；
- 仅用于证明量化合同、AP runner、原子反馈和恢复机制可执行。

正式阶段 5 必须在修正候选生成器、训练输入合同和 lazy materializer 后，从新的 round-0 重新启动。

### 15.4 下一阶段与阶段 6 准入

下一阶段首先完成合同修复和四任务 dry-run，确认 `M=686`、`D0=Gold176`、历史标签不可见及确定性 request SHA；随后完成四个独立 SearchTask 的完整预算搜索，即每任务 4 轮、每轮 4 个单 genome，共 64 个正式在线测量终态。

阶段 5 收口要求同时满足：

1. 四个任务均耗尽各自 16 点固定预算，批内原子回流且任务间不共享在线反馈；
2. 每个成功点具有 latency、energy、full AP 和证据 SHA，真实 feasibility failure 与基础设施失败正确区分；
3. 生成四份 measured Pareto、四条逐轮 HV 曲线、完整预算/wall-clock 审计和独立最终验证；
4. 复查训练输入、候选全集、request、反馈和模型 SHA，确认没有 Feedback16、pilot 或其他历史标签泄漏；
5. 不因阶段 5 结果临时改变 `B`、`T`、acquisition、精度比例或候选空间。

满足以上合同与证据门禁后，阶段 5 可以工程收口并进入阶段 6 方法有效性实验。阶段 5 不以“已经证明优于所有基线”为准入条件，因为 random、旧策略和六臂方法的同预算显著性比较正是阶段 6 的任务；但如果阶段 5 轨迹存在数据泄漏、预算不完整或证据缺失，则不得进入阶段 6。

### 15.5 Gold176-only 启动前回放风险

修正训练入口后，以独立 Gold176 做 24-genome 隐藏标签回放，四个任务的 `q_mode` 预测均产生非零差异；Pyramid/TRT、CoDriving/TVM、CoDriving/TRT 通过当前 acquisition 对照门，但 Pyramid/TVM 的 Pareto recall 高于随机中位数时，HV 仍约为随机中位数的 `0.93×`。这说明 cost model/acquisition 在该 profile 上仍有不足，不能提前宣称方法有效性已经成立。

该风险不允许通过加入 Feedback16、旧 pilot 或其他历史标签消除。下一步仍按冻结的 `M/B/T` 完成四个独立真实搜索，以在线反馈、逐轮 HV 和最终 Pareto 判断实际行为；阶段 5 负责证明搜索闭环和证据合同成立，是否优于随机及旧策略由阶段 6 同预算实验判断。

---

## 16. 下一轮 `/goal` 建议

```text
/goal 修正并完成阶段5单目标模型、单硬件、单capability-profile自主搜索。四个SearchTask的搜索宇宙统一为7x7x7x2=686，round-0训练输入严格冻结为Gold176 initial_coldstart，禁止Feedback16、pre-stage5 smoke、专项补点、旧pilot及其他历史实测标签进入训练、候选过滤或排序；source registry覆盖两个模型全部343个width，checkpoint/ONNX/calibration仅在acquisition选中后按需物化。一次SearchTask固定target_model、hardware_id和capability_profile_id，genome仅为(w0,w1,w2,q_mode)，每批B=4单genome原子回流，每任务固定T=16、4轮，四任务在线状态隔离。旧M0=100/640和已完成Pyramid/TVM round-0统一隔离为invalidated_stage5_pilot，不计正式预算。停止条件：合同与泄漏测试通过，四任务确定性dry-run确认M=686和统一Gold176 SHA；四任务各完成16个可信终态，共64个正式在线点；生成四份measured Pareto、四条HV曲线、预算/wall-clock/证据SHA审计及独立最终验证，并给出阶段5是否收口和是否准入阶段6的明确结论。
```
