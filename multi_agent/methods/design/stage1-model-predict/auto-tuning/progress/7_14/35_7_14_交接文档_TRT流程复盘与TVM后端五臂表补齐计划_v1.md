# 35_7_14 交接文档：TRT 流程复盘与 F-Cooper/TVM 表格补齐计划 v1

**日期**：2026-07-24  
**任务范围**：只补充 `F-Cooper + H800 + TVM automatic`；Pyramid/TVM 和
CoDriving/TVM 直接复用 31 号文档已经闭合的正式结果。  
**上游文档**：

- [31 号：Pyramid/CoDriving Stage6 TVM/TRT 正式结果](./31*_7_14_交接文档_框架待完善项与阶段6六臂对照实施计划_v1.md)
- [32 号：F-Cooper/TRT v2 正式端到端搜索](./32_7_14_交接文档_FCooper_Table1_TVM_CPU与Orin边缘部署计划_v1.md)

---

## 1. 范围更正与当前完成度

此前把下一阶段误写成“三个模型重新执行 TVM 搜索”，范围过大。核对 31 号文档后，
正式状态如下：

| 模型/profile | GEAR 搜索 | Stage6 对照 | 独立复测 | 最终状态 |
|---|---:|---:|---:|---|
| Pyramid + TVM/H800 | `T=16` 已闭合 | 六臂已闭合 | 已完成 | `paper_ready=true` |
| CoDriving + TVM/H800 | `T=16` 已闭合 | 六臂已闭合 | 已完成 | `paper_ready=true` |
| F-Cooper + TVM/H800 | 尚未执行 | 尚未执行 | 尚未执行 | 当前唯一缺口 |

31 号文档当时使用六臂：

```text
Original/default
Compression only
Schedule only
Compress -> Tune
Tune -> Compress
joint SHCoSearch
```

当前论文已经删除 `Tune -> Compress`，并将 `joint SHCoSearch` 统一命名为
`GEAR`。因此 Pyramid 和 CoDriving 不需要重跑，只需从各自已经通过审计的六臂表中
抽取以下五行：

```text
Original/default
Compression only
Schedule only
Compress -> Tune
GEAR
```

本阶段只运行 F-Cooper 的 TVM profile，最终把三个模型合并成 TVM 五臂表。

### 1.1 明确排除的工作

- 不重新运行 Pyramid + TVM；
- 不重新运行 CoDriving + TVM；
- 不执行 CPU/LLVM/TVM 实验；
- 不读取或续跑此前的 CPU Lane B；
- 不执行 Orin；
- 不将 TVM 与 TRT 作为同一次搜索中的可选 genome；
- 不恢复 `Tune -> Compress`。

CPU Lane B 已正式停止，其结果不属于本任务输入、对照或论文表格。

---

## 2. 已完成的 TVM 正式证据

### 2.1 Pyramid + TVM/H800

31 号文档第 11.4～11.6 节已记录完整 TVM/H800 结果。`DeltaAP_max=0.10` 下，
抽取当前五臂后为：

| 方法 | 终态 | 配置 | AP70 | Latency (ms) | Energy (J) |
|---|---|---|---:|---:|---:|
| Original/default | selected | `(64,128,256,fp32)` | 0.6311 | 3.2187 | 1.1636 |
| Compression only | selected | `(24,48,96,fp16)` | 0.5501 | 16.7404 | 2.7766 |
| Schedule only | numerical feasibility failure | `N/A` | `N/A` | `N/A` | `N/A` |
| Compress -> Tune | selected | `(32,48,128,fp16)` | 0.5336 | 23.6673 | 4.1850 |
| GEAR | selected | `(16,32,64,int8)` | 0.6159 | 3.3407 | 0.6490 |

边界解释：

- GEAR latency 略慢于 native FP32 基线，但 energy 明显更低；
- 旧 Schedule-only 已执行，当前是可信数值 feasibility failure，不是“没有填表”；
- 该结果用于说明 TVM/H800 下联合搜索仍能找到后端内最优决策，但不宣称 TVM
  必然超过 cuDNN/TRT。

正式证据：

- [Pyramid TVM 主表 CSV](${V2X_ROOT}/results/stage6_pyramid_formal_20260720/paper_tables/pyramid_stage6_tvm_delta_ap_0.10.csv)
- [Pyramid Stage6 主表审计](${V2X_ROOT}/results/stage6_pyramid_formal_20260720/paper_tables/stage6_paper_main_table_audit_v1.json)
- [Pyramid Stage6 证据 bundle](${V2X_ROOT}/results/stage6_pyramid_formal_20260720/paper_evidence/stage6_paper_evidence_bundle_v1.json)

### 2.2 CoDriving + TVM/H800

31 号文档第 13 章确认：

- `S5-COD-TVM` 为 `budget_exhausted`；
- 4/4 轮、16/16 在线反馈完成；
- 16/16 行包含真实 latency、energy、AP 和 actual graph features；
- 0 silent surrogate fallback。

第 14.4～14.7 节确认 TVM 六臂、独立复测和最终审计均已收口。
`DeltaAP_max=0.10` 下抽取当前五臂为：

| 方法 | 终态 | 配置 | AP70 | Latency (ms) | Energy (J) |
|---|---|---|---:|---:|---:|
| Original/default | selected | `(64,128,256,fp32)` | 0.3975 | 1.8500 | 0.6999 |
| Compression only | selected | `(24,80,128,fp16)` | 0.3846 | 1.9067 | 0.2938 |
| Schedule only | selected | `(64,128,256,fp32)` | 0.3971 | 10.8852 | 1.4592 |
| Compress -> Tune | selected | `(24,80,128,fp16)` | 0.3852 | 1.8339 | 0.2600 |
| GEAR | selected | `(16,32,64,fp16)` | 0.3571 | 1.5027 | 0.2076 |

该结果已经证明：

- TVM profile 下 GEAR 自动选择 FP16，而 TRT profile 下同模型选择 INT8；
- q_mode 不是人工规则，而是 capability context 与实测回流共同影响的结果；
- GEAR 相对 native FP32 latency 约提升 `1.2311x`；
- GEAR 同时优于该 profile 下的 Compression-only、Schedule-only 和
  Compress -> Tune。

正式证据：

- [CoDriving TVM/TRT 主表目录](${V2X_ROOT}/results/stage6_codriving_formal_20260722/paper_tables)
- [CoDriving 主表审计](${V2X_ROOT}/results/stage6_codriving_formal_20260722/paper_tables/stage6_codriving_paper_main_table_audit_v1.json)
- [CoDriving completion audit](${V2X_ROOT}/results/stage6_codriving_formal_20260722/paper_evidence/stage6_codriving_completion_audit_v1.json)
- [CoDriving evidence bundle](${V2X_ROOT}/results/stage6_codriving_formal_20260722/paper_evidence/stage6_codriving_paper_evidence_bundle_v1.json)

### 2.3 当前 TVM 表格完成度

Pyramid 与 CoDriving 已经填满 `2 models x 5 methods = 10` 行模型方法结果。
F-Cooper 的 Original/default 与后端无关，若 32 号文档中的 scope、checkpoint、
数据 split 和三次独立复测 SHA 审计一致，可以直接复用：

```text
F-Cooper Original/default
AP70 = 0.633161
latency = 11.801734 ms
energy = 6.757156 J
```

因此真正需要新增的 F-Cooper/TVM 工作是四条方法：

1. Compression only；
2. Schedule only；
3. Compress -> Tune；
4. GEAR。

---

## 3. TRT 端到端流程总结

F-Cooper/TRT v2 已经形成当前最完整的单目标模型、单 profile 自动搜索实现。

### 3.1 正式流程

| 顺序 | 阶段 | 输入 | 自动行为 | 输出 |
|---:|---|---|---|---|
| 0 | 合同冻结 | model/config/checkpoint、test manifest、profile | 固定 scope、训练、AP、能耗、失败和 SHA 合同 | formal contract |
| 1 | Fresh scanner | 模型图和 capability | 扫描依赖组、合法宽度和精度空间 | partition、candidate manifest |
| 2 | Capability probes | 原始/边界 shape | build/run、数值、coverage、precision propagation/fallback | capability context |
| 3 | Probe 隔离 | probe evidence | 禁止 probe 指标进入 cost model、T16 和 winner 集 | isolation audit |
| 4 | 恢复训练门禁 | scanner 生成的剪枝点 | 统一恢复训练、ONNX、完整 AP | non-collapse gate |
| 5 | 初始 cost model | Gold176、capability、目标模型 AP anchor | 拟合三目标响应和 uncertainty | round-0 model |
| 6 | Acquisition | 合法候选池 | Pareto/uncertainty/约束感知选择 `B=4` | measurement request |
| 7 | 候选真测 | 每轮 4 个 genome | 恢复训练、物化、后端 build、latency、energy、full AP | 三指标 |
| 8 | Actual feedback | 真实 ONNX 和后端结果 | actual graph features 与三指标、SHA 四行原子回流 | feedback batch |
| 9 | 在线更新 | 已闭合历史轮 | 重拟合 cost model，生成下一轮 | 4 轮、`T=16` |
| 10 | Winner | 16 个在线终态 | AP floor 下选最低 latency，1% 内以 energy 破平局 | selected row |
| 11 | 独立复测 | winner row id | 3 次 latency/energy + 1 次 full AP | independent evidence |
| 12 | 五臂 | 同 scanner 和测量合同 | Original、Compression、Schedule、12+4、GEAR | 五臂结果 |
| 13 | Finalizer | request、feedback、artifact、AP、repeat、timing | 回算 SHA 和预算，原子生成正式输出 | CSV/audit/bundle |

### 3.2 Actual-feature 回流的正确语义

```text
未选择候选
  -> 只用 surrogate graph features 进行廉价预测
  -> acquisition 选择 B=4
  -> 只物化选中的 4 点
  -> 提取 actual graph features
  -> 测量 latency、energy、AP
  -> 四行终态原子回流
  -> 下一轮重训
```

未选中的候选不需要物化。当前批次物化后的 actual features 不反改已经冻结的 request，
而是在下一轮训练时生效。

---

## 4. TRT 实现中的主要错误与反思

### 4.1 旧 partition 伪装成 fresh scanner

旧 F-Cooper pilot 复制历史 partition 后替换硬件字段，不能证明当前搜索空间来自本轮
模型扫描。正式 v2 改为从模型入口运行 scanner，保存命令、manifest、日志和 SHA。

**F-Cooper/TVM 要求**：F-Cooper 的结构依赖扫描结果可以在模型、checkpoint 和扫描器
SHA 一致时复用，但必须重新绑定 TVM capability；不能只修改 profile 字符串。

### 4.2 Probe 泄漏进入 cost model 和 winner

probe 只应验证 capability。旧 pilot 把 probe INT8 点升级成最终 winner，使其绕过
T16 预算。

正式规则：

```text
probe labels in cost model = false
probe rows in T16 = false
probe rows as winner = false
```

F-Cooper/TVM 必须沿用。

### 4.3 Winner 来源被程序限制

旧 finalizer 要求 winner 必须属于预设 incumbent，导致真正的 T16 最优点反而无法
收口。正式选择器允许 T16 中任意终态胜出，再动态生成复测任务。

F-Cooper/TVM 不得预设 winner 为 FP16、INT8、Gold 点或 probe。

### 4.4 剪枝候选没有恢复训练

旧 pilot 只裁切 checkpoint 通道，没有统一恢复训练，AP 崩溃和异常短耗时都不具有
论文解释力。正式 v2 对所有剪枝候选使用冻结训练合同。

F-Cooper/TVM 可以复用 TRT v2 已经完成且 SHA 一致的 backend-neutral checkpoint 和
ONNX；不能复用 TRT latency、energy、engine 或在线标签。

### 4.5 实测结果没有带 actual features 回流

早期轨迹虽然测了三指标，却继续用 surrogate features 更新 cost model，因此只能算
pilot。正式流程只允许 actual features + 三指标进入下一轮。

F-Cooper/TVM 必须保证 16/16 成功在线行绑定 actual ONNX graph features；feasibility
failure 则绑定失败上下文。

### 4.6 AP bridge 自动猜错 checkpoint

Pyramid schedule-only 曾因 AP bridge 回退到错误 epoch 被误判为精度失败。正式入口
必须显式绑定 checkpoint/config/test-manifest SHA。

F-Cooper/TVM 的 AP runner 不允许扫描目录后自行选择“看起来可用”的 checkpoint。

### 4.7 INT8 quant contract 回退

CoDriving TVM 独立复测曾回退到默认 quant contract，导致 sanity/AP 异常。INT8
证据必须绑定 checkpoint、ONNX、calibration、quant contract、compiled module 和
prediction hash。

F-Cooper/TVM 的 INT8 runner 必须在 T16 前完成该门禁，禁止 silent fallback。

### 4.8 并行 source 覆盖

多个 GPU 同时构建相同 width 时曾出现 source 覆盖风险。正式流程使用同 width lock、
不可变 ready marker 和 SHA 复用。

F-Cooper/TVM 可以复用已有恢复训练 source，但 TVM database、IR、module 和量化产物
必须写入独立 TVM 结果目录。

### 4.9 缺失 timing 被错误补成 0

正式 v2 最终要求：

- 缺失恢复训练 timing 必须从 SHA 绑定报告恢复；
- 只有 width、checkpoint、ONNX、recovery-report SHA 全部匹配，才允许把 source
  阶段记为零工作复用；
- TVM 还需增加 tuning trials、database、build 和 measurement timing。

### 4.10 Finalizer 覆盖破坏 bundle

正式输出必须由 staging 原子提交。不能把临时 finalizer 目录中的三个文件单独复制到
正式目录，否则 bundle 内绝对路径和 SHA 会漂移。

F-Cooper/TVM 使用全新版本化结果根目录，不覆盖 TRT v2 或旧 TVM pilot。

### 4.11 GPU 空闲不等于任务未并行

单轮只有 `B=4` 点；恢复训练、CPU 编译、AP 后处理和 round barrier 会限制 GPU
并行上限。不能通过跨轮提前执行或重复测量制造 GPU 占用。

F-Cooper/TVM 可将 GEAR、Compression-only 和 Compress -> Tune 放在互斥 lane
并行，但 GEAR 的每轮 4 点必须完整原子回流后才能生成下一轮。

### 4.12 Original/default 不能使用 TVM 50 ms 弱基线

Original/default 是用户真实原始部署，即 PyTorch eager FP32 + cuDNN。TVM default
只能属于 Schedule-only 或 TVM 内部消融，不能替代 Original 来制造加速比。

---

## 5. F-Cooper/TVM 的固定任务定义

### 5.1 SearchTask

```text
task_id: S5-FCO-TVM-V1
target_model: fcooper
hardware_id: h800
capability_profile: h800-tvm-auto-fcooper-v1
dispatch_key: tvm_auto
scope: post_scatter_backbone_shrinker
test samples: 2170
```

Backend 固定为 TVM，不进入 genome。

### 5.2 Genome

复用 F-Cooper fresh scanner 已确认的动态结构轴：

```text
(
  backbone.s0,
  backbone.s1,
  backbone.s2,
  neck.deblock,
  neck.output,
  q_mode
)

q_mode in {fp16, int8}
```

不能退回 Pyramid/CoDriving 的三宽度模板，也不能人工列举少量配置代替 scanner
候选池。

### 5.3 冷启动边界

初始 cost model 使用：

- Gold176 全局 cold-start；
- F-Cooper 原始 AP anchor；
- 新生成的 TVM capability context；
- backend-neutral 图特征。

禁止读取：

- F-Cooper TRT v2 的 latency、energy、online feedback；
- Pyramid/CoDriving TVM 的在线标签；
- F-Cooper 旧 pilot；
- capability probe 的 AP/latency；
- hand-rewrite prior。

允许复用的是训练好的 source artifact，不是其他 profile 的性能标签。

---

## 6. F-Cooper TVM 自动后端合同

### 6.1 自动路径

正式后端只允许：

- FP16：Route B 或同等级自动分解、tensorization 和 tuning；
- INT8：自动 per-PrimFunc/per-block 分解、自动暴露 INT8 matmul，再
  tensorize/tune；
- 自动 fusion、layout 和 constant folding；
- 机器可审计的 TVM database、IR/module 和 target。

禁止：

- hand-written `im2col+MMA`；
- Phase1 hand-rewrite family；
- 旧 hand-rewrite latency prior 作为正式结果；
- build 失败后静默回退 PyTorch/cuDNN/TRT。

### 6.2 每个被选点的测量流程

```text
selected genome
  -> recovery source lookup
  -> SHA 一致则复用，否则执行冻结恢复训练
  -> checkpoint / ONNX
  -> actual graph features
  -> FP16/INT8 numerical preparation
  -> TVM import / automatic decomposition / tensorization
  -> fixed-budget tuning
  -> database + compiled module
  -> numerical sanity
  -> latency + energy
  -> 2170-sample full AP
  -> atomic actual feedback
```

### 6.3 TVM 内层预算

为与 31 号 Pyramid/CoDriving TVM 协议保持一致，启动前冻结：

| 方法 | 外层测量 | TVM tuning |
|---|---:|---:|
| Original/default | 复用固定基线 | 0 |
| Compression only | 16 | 0，固定 default measurement |
| Schedule only | 1 | 64 trials |
| Compress -> Tune | 12 screen + 4 tuned | 4 个 finalist 各 64 trials |
| GEAR | `B=4,T=16` | 16 个在线点各 64 trials |

如果当前 TVM 版本无法复现 31 号的 trials 语义，必须先修复 runner 或冻结等价 workload
预算，不能临时给某条臂更多调优时间。

### 6.4 失败合同

- build feasibility failure：保存 target、IR、compiler error、database 和日志；
- numerical feasibility failure：保存输入输出 hash、误差和失败层；
- INT8 quant failure：保存 calibration/scale/quant contract；
- infrastructure failure：不消耗预算；
- feasibility failure：作为真实终态回流；
- AP 不可用时不得填 `0`；
- silent fallback 直接使该行不具备正式资格。

---

## 7. F-Cooper TVM 五臂执行

### 7.1 Original/default

复用 F-Cooper/TRT v2 的固定基线，前提是以下合同完全一致：

- scope；
- checkpoint/config；
- 2170 样本 test manifest；
- input/batch；
- 三次独立性能复测；
- full AP 和 SHA。

该行不重新运行 TVM，因为 Original/default 的定义就是未使用 TVM/TRT 编译的原始
PyTorch/cuDNN 路径。

### 7.2 Compression only

F-Cooper/TRT v2 的 Compression-only 候选生成是 backend-blind 的，因此其冻结
16 点 candidate plan 可在 plan-contract SHA 一致时复用。

但是必须重新执行：

- TVM FP16/INT8 build；
- latency/energy；
- full AP；
- actual backend terminal；
- TVM 独立 winner 选择。

不能复用 TRT 选择出的最终 winner 数值。

### 7.3 Schedule only

- 固定原始 F-Cooper 结构；
- q_mode 为 FP32；
- 执行 TVM 自动 schedule；
- 使用与 31 号一致的 64-trial 合同；
- 先通过数值 sanity，再执行 latency/energy/full AP。

如果发生 failure，必须先排除 runner、checkpoint、AP bridge 和 unsupported op 的工程
错误。只有自动 TVM 路径本身无法闭合时，才能作为可信 feasibility failure。

### 7.4 Compress -> Tune

F-Cooper/TRT v2 的 backend-blind 12 点 screen candidate plan 可在 SHA 一致时复用，
但不能复用 TRT screen 指标或 tuned four。

TVM 必须重新：

1. 对 12 点执行 TVM default screen；
2. 按冻结规则自动选择 4 个 finalist；
3. 对 4 点分别执行 64 trials；
4. 真实测量三指标；
5. 选择 TVM profile 下的最终点。

### 7.5 GEAR

GEAR 是唯一需要完整新建在线搜索轨迹的部分：

```text
B = 4
rounds = 4
T = 16
```

每轮：

1. cost model 对合法候选池预测；
2. acquisition 自动选择 4 个单 genome；
3. 4 点执行 source、TVM tune、latency、energy、full AP、actual features；
4. 4 行终态原子回流；
5. 重拟合后生成下一轮。

T16 结束后只从正式在线终态选择 winner，不允许 probe、TRT winner 或人工补点进入
winner 集。

---

## 8. 执行顺序和并行方案

### P0：TVM runner 与 capability 门禁

1. 生成 F-Cooper TVM capability profile；
2. 原始/边界 FP16、INT8 和 FP32 schedule-only smoke；
3. 检查 unsupported op，特别是 neck/deblock、transpose-conv 和输出合同；
4. 冻结 quant contract、target、TVM commit、trials 和结果目录；
5. 验证 probe 隔离。

停止点：TVM 自动路径能真实 build/run，或形成明确的 profile-level failure。

### P1：GEAR T16

- 新建 `S5-FCO-TVM-V1`；
- 4 轮、16 点；
- 只使用 Gold176 和当前 capability 初始化；
- 每轮 actual feedback 原子闭合。

### P2：三个 TVM 控制臂

并行执行：

- Compression-only 16；
- Schedule-only 1；
- Compress -> Tune 12+4。

source artifact 可以按 SHA 复用，TVM build/tune/measure 必须独立。

### P3：独立复测和总表

对 F-Cooper 五臂中的每个成功最终点完成：

- 3 次独立 latency/energy；
- 1 次 2170 样本 full AP；
- checkpoint、ONNX、quant contract、TVM database/module、AP report SHA；
- CV、failure rate、trials、GPU-hours 和 wall-clock。

随后把 F-Cooper 列与现有 Pyramid/CoDriving TVM 正式行合并。

### 8.1 GPU 并行

- GEAR 单轮最多 4 个候选并行；
- Compression-only、Compress -> Tune、Schedule-only 和 capability probe 使用其余
  GPU 建立独立 lane，与 GEAR 同时推进；
- 同 width source 使用文件锁，不重复训练；
- GPU0～7 均可在 P0～P2 使用，不从实验开始永久保留 GPU7；
- 只有进入 P3 独立复测前，才排空 GPU7 并建立 exclusivity guard；
- 不跨轮提前执行 GEAR 候选。

调度器必须动态发现并租用 H800 GPU `0--7` 的全部真实空闲卡，采用“一卡一个正式
GPU job”的互斥策略。推荐初始分配为：

```text
GPU0--3: 当前 GEAR round 的 4 个候选
GPU4--5: Compression-only 队列
GPU6:    Compress -> Tune 队列
GPU7:    probe / Schedule-only / 其他已冻结控制臂
```

任一 lane 进入 CPU 编译、AP 后处理、文件等待或结束后，空闲 GPU 应从其他已冻结队列
自动领取任务；只要存在可运行 job，调度器应在 60 秒内补位。若某轮可运行任务不足，
允许 GPU 暂时空闲，但禁止通过重复测量、跨轮提前选择或修改预算制造利用率。

最终审计必须报告：

- GPU0～7 各自执行的 row id、阶段、start/end 和退出状态；
- 每个 5 分钟窗口的有效并行 GPU 数；
- GPU 空闲来自无可运行任务、CPU build、round barrier、重试还是调度器异常；
- source lock 命中率和避免的重复恢复训练次数；
- 排队时间、GPU-hours 与总 wall-clock。

---

## 9. 时间和工作规模

真正新增的 F-Cooper/TVM 测量规模：

| 部分 | 新增测量 |
|---|---:|
| GEAR | 16 个在线 genome |
| Compression only | 16 个 TVM 终态 |
| Schedule only | 1 个 TVM 终态 |
| Compress -> Tune | 12 screen + 4 tuned |
| Original/default | 0，审计复用 |

同一 width 的恢复训练 source 可以跨臂复用，因此新增终态不等于新增同数量的恢复训练。

根据 F-Cooper/TRT v2 的恢复训练成本和历史 TVM build/tuning 成本，建议预留：

| 阶段 | 预计墙钟 |
|---|---:|
| P0 runner/capability | 2--6 小时 |
| P1 GEAR T16 | 14--22 小时 |
| P2 控制臂 | 6--12 小时，与 P1 大部分并行 |
| P3 独立复测/审计 | 4--8 小时 |
| 正常总墙钟 | 约 24--36 小时 |
| 保守预留 | 48 小时 |

具体时间取决于 TVM 对 F-Cooper neck/deblock 的 build 成功率和恢复训练 source 命中率。
如果 TVM GEAR 选中的 16 个 width 大量命中 TRT v2 已有 source，正常耗时接近区间
下限；若 16 点大部分是新 width，需要重新恢复训练，则接近上限。超过 48 小时仍未
闭合时必须输出阶段级阻塞审计，而不是无界继续运行。

---

## 10. TVM 三模型五臂总表

主阈值固定为模型自身 `DeltaAP70 <= 0.10`。Pyramid 和 CoDriving 直接使用 31 号
正式数值；F-Cooper 五臂已于 2026-07-25 完成统一严格审计。

| Method | Pyramid AP70 | Pyramid Lat. (ms) | Pyramid Energy (J) | CoDriving AP70 | CoDriving Lat. (ms) | CoDriving Energy (J) | F-Cooper AP70 | F-Cooper Lat. (ms) | F-Cooper Energy (J) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original/default | 0.6311 | 3.2187 | 1.1636 | 0.3975 | 1.8500 | 0.6999 | 0.6332 | 11.8017 | 6.7572 |
| Compression only | 0.5501 | 16.7404 | 2.7766 | 0.3846 | 1.9067 | 0.2938 | 0.5946 | 9.1297 | 2.2997 |
| Schedule only | `N/A` | `N/A` | `N/A` | 0.3971 | 10.8852 | 1.4592 | 0.6328 | 102.8649 | 24.7240 |
| Compress -> Tune | 0.5336 | 23.6673 | 4.1850 | 0.3852 | 1.8339 | 0.2600 | 0.5948 | **8.6385** | **2.0459** |
| **GEAR (Ours)** | 0.6159 | 3.3407 | 0.6490 | 0.3571 | 1.5027 | 0.2076 | 0.6022 | 10.2033 | 2.1676 |

Pyramid Schedule-only 的 `N/A` 是已完成实验后得到的可信 numerical feasibility
failure，不能补成估算值，也不能写成尚未执行。

### 10.1 F-Cooper 内部审计表

| Method | terminal | config | AP70 | latency | energy | outer budget | TVM trials | failure rate | evidence SHA |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Original/default | measured success | `(64,128,256,128,256,fp32)` | 0.633161 | 11.801734 | 6.757156 | 1 | 0 | 0 | `9cad3ee79f3d` |
| Compression only | measured success | `(32,96,64,32,64,fp16)` | 0.594635 | 9.129696 | 2.299660 | 16 | 0 | 0 | `059b40f2321e` |
| Schedule only | measured success | `(64,128,256,128,256,fp32)` | 0.632819 | 102.864932 | 24.723991 | 1 | 64 | 0 | `1ea1487c180a` |
| Compress -> Tune | measured success | `(32,128,64,32,64,fp16)` | 0.594829 | 8.638469 | 2.045894 | 12+4 | 256 | 0 | `65f0713d2089` |
| GEAR | measured success | `(32,32,32,128,96,fp16)` | 0.602156 | 10.203284 | 2.167600 | 16 | 1024 | 0 | `7c9f231c81ab` |

### 10.2 正式收口

- GEAR 正式轨迹完成 `4/4` 轮、`16/16` 在线反馈；四份原子批审计均为每轮 4 行。
- 五臂 winner 均完成 3 次独立 latency/energy 复测和 2170 样本 full AP；GEAR
  和 Schedule-only 因 GPU7 被其他任务长期占用，在独占检查通过后幂等迁移到
  同机 GPU0。
- 49 个正式 `(row_id, TVM trials)` 身份全部通过审计，共执行 1344 个 TVM trials；
  41/49 行复用 SHA 与恢复训练合同一致的 backend-neutral TRT v2 source。
- 强化收口同时绑定 capability probe 隔离、恢复训练数值门禁、三条对照臂来源，
  以及 round 1--3 仅消费此前原子释放反馈的状态链；总审计仍为
  `paper_ready=true`，三模型各 5 臂共 15 行。
- round 0 的 SHA 合同进一步绑定 Gold176、`initial_coldstart_only`、单一
  F-Cooper/TVM profile 和扫描得到的五个结构/接口轴加 `q_mode`；没有读取
  pilot 或 Pyramid/CoDriving 在线标签。
- 最终发布门禁从 original/default 实测 AP70 锁定 `DeltaAP70` 基线，禁止把
  成功但违反 AP floor 的点写成 paper-ready；49/49 个正式测量身份均由逐卡任务
  日志覆盖，且每行性能 source 与 TVM module/database/quant 合同一致。
- Schedule-only 的旧 AP bridge 虽按 FP32 执行，却把报告 schema 固定写成 FP16。
  严格审计发现后已修复公共 runner，并在 GPU0 重新完成三次性能复测和 2170
  样本 AP；新证据同时声明 `q_mode=fp32`、FP32 performance precision 与 FP32
  AP schema，0 failure、0 fallback。
- F-Cooper/TVM 的 GEAR 相比 Original/default 将延迟和能耗分别降低约 13.5% 和
  67.9%，但没有击败 Compress -> Tune；GEAR 相对后者延迟高约 18.1%、能耗高约
  5.9%。该负结果保留在主表中，不人工改点或回退到 probe。
- [F-Cooper 五臂 raw CSV](${V2X_ROOT}/results/S5-FCO-TVM-V1/closure/paper_ready_v5/fcooper_stage6_tvm_delta_ap_0.10_v1.csv)
- [F-Cooper 完整 audit](${V2X_ROOT}/results/S5-FCO-TVM-V1/closure/paper_ready_v5/fcooper_stage6_tvm_audit_v1.json)
- [F-Cooper evidence bundle](${V2X_ROOT}/results/S5-FCO-TVM-V1/closure/paper_ready_v5/fcooper_stage6_tvm_evidence_bundle_v1.json)
- [三模型 TVM raw CSV](${V2X_ROOT}/results/S5-FCO-TVM-V1/closure/paper_ready_v5/tvm_three_model_stage6_delta_ap_0.10_v1.csv)
- [三模型 TVM 表](${V2X_ROOT}/results/S5-FCO-TVM-V1/closure/paper_ready_v5/tvm_three_model_stage6_table_v1.json)
- [paper-ready 标记](${V2X_ROOT}/results/S5-FCO-TVM-V1/closure/paper_ready_v5/paper_ready.json)

---

## 11. 论文结论边界

F-Cooper/TVM 完成后，TVM 总表可以回答：

1. 相同 GEAR 流程在三个模型的 TVM profile 下能否形成不同结构和精度决策；
2. GEAR 是否优于 TVM profile 下的 backend-blind 和串行对照；
3. TVM 的自动后端失败边界是否随模型结构变化；
4. 为什么不能把 TRT 上的 INT8 或 winner 直接迁移到 TVM；
5. 后端 capability 和真实反馈为何是联合搜索的必要输入。

不能回答：

1. TVM 在 NVIDIA GPU 上普遍快于 TRT/cuDNN；
2. TVM 的跨 CPU/非 NVIDIA 优势已经得到本表验证；
3. INT8 在 TVM 上恒快或恒慢；
4. hand rewrite 可以代表当前自动后端；
5. T16 等同于穷举全局最优。

若 F-Cooper/TVM 最终仍慢于 native FP32，必须如实报告。方法有效性主要比较 TVM
profile 内的 GEAR 与其他 TVM 控制臂，而不是预设 TVM 必须击败 cuDNN。

---

## 12. 停止条件

本阶段只在以下条件全部满足后收口：

1. F-Cooper TVM capability 与数值门禁完成；
2. GEAR `4/4` 轮、`16/16` 在线预算闭合；
3. 16 个成功行全部绑定 actual features、TVM 产物、三指标和 SHA；
4. Compression-only 16、Schedule-only 1、Compress -> Tune 12+4 获得可信终态；
5. Original/default 复用通过 scope 与 SHA 审计；
6. 每个成功最终点完成 3 次性能复测和 full AP；
7. probe、TRT online labels、Pyramid/CoDriving online labels 均未泄漏；
8. 生成 F-Cooper TVM CSV、audit、bundle 和完成标记；
9. 抽取 31 号 Pyramid/CoDriving 五臂行，生成 TVM 三模型合并表；
10. 更新论文 TeX、编译 PDF 并视觉检查；
11. 总审计给出 `paper_ready=true`，或明确指出真实受阻的 F-Cooper TVM 臂。

---

## 13. 下一窗口 `/goal`

```text
/goal 只补齐 F-Cooper + H800 + TVM automatic 的正式搜索和五臂结果，并与 31 号交接文档已经 paper_ready 的 Pyramid/TVM、CoDriving/TVM 结果合并成 TVM 三模型主表。以 ${V2X_ROOT}/multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/35_7_14_交接文档_TRT流程复盘与TVM后端五臂表补齐计划_v1.md 为唯一执行合同；不得重新运行 Pyramid/TVM 或 CoDriving/TVM，不执行 CPU Lane B 或 Orin，不把 TVM/TRT 作为同一次搜索中的 genome。新建独立 S5-FCO-TVM-V1 结果目录；最大程度复用 32 号 F-Cooper/TRT v2 已有且 SHA、scope、width、恢复训练合同完全一致的 backend-neutral checkpoint、config、recovery report、ONNX 和 actual ONNX graph features，复用时必须记录 source provenance 和节省的恢复训练时间；禁止复用任何 TRT latency、energy、engine/tactic、compiled artifact、T16 feedback、winner 标签、预测输出或 AP 结果，TVM quant/calibration contract、build/tune、latency、energy 和 full AP 必须重新生成。重新生成 F-Cooper TVM capability profile，完成原始/边界 FP16、INT8 与 FP32 schedule-only build/run、数值、quant contract、fallback、fusion/materialization probes；probe 只形成 capability context，不进入 cost model、T16 或 winner 集。正式后端只允许 Route B 或同等级自动分解、tensorization 和固定预算 tuning，禁止 hand-written im2col+MMA、hand rewrite prior 和 silent fallback。GEAR 从 Gold176、当前 TVM capability context 和 F-Cooper 原始 AP anchor 初始化，执行 B=4、4轮、T=16；每轮4个 genome 必须由 cost model/acquisition 自动产生，每点完成恢复训练 source 审计、物化、actual ONNX features、TVM build/tune、数值 sanity、真实 latency/energy、2170样本 full AP 和全部 SHA，四行终态原子回流后才能生成下一轮；infrastructure failure 不消耗预算，feasibility failure 按真实终态回流，不伪造 AP=0。按当前五臂协议补齐 F-Cooper：Original/default 复用同 scope 的 TRT v2 native FP32 基线并重新做 SHA 准入，Compression-only 复用 backend-blind 16点 candidate plan和已有 source但重新执行 TVM 测量，Schedule-only 固定原始结构执行64 trials，Compress->Tune 复用 backend-blind 12点 plan和已有 source但重新完成 TVM screen并自动选择4点各64 trials，GEAR 使用本轮正式T16。调度器必须动态检查并租用 H800 GPU0、1、2、3、4、5、6、7 的全部真实空闲卡，每卡最多运行一个正式GPU job；GEAR当前轮4点优先分配GPU0--3，其余空闲卡并行执行Compression-only、Compress->Tune、Schedule-only和probe，任务结束或进入CPU阶段后60秒内从其他已冻结队列自动补位，不得因为固定lane让空闲GPU长期闲置；不得跨GEAR轮提前选择候选或破坏四行原子反馈。GPU7在P0--P2正常参与任务，只在最终独立复测前排空并由exclusivity guard独占；若实际空闲卡集合变化，允许幂等迁移未启动job但不得修改measurement request、row id或预算。每个成功最终点在GPU7执行三次独立latency/energy和一次完整AP，绑定checkpoint、ONNX、quant/calibration、TVM database/module、prediction和AP report SHA，报告GPU0--7逐卡任务、有效并行度、排队/重试、source复用率、outer budget、TVM trials、GPU-hours和wall-clock。正常目标墙钟为24--36小时，保守停止审计点为48小时；超过48小时未闭合必须输出具体阻塞阶段和剩余行，不得无界运行。停止条件：F-Cooper五臂均有可信成功或明确失败终态，生成raw CSV、完整audit、evidence bundle和完成标记；从31号正式结果中删除Tune->Compress、将joint SHCoSearch统一命名为GEAR，直接抽取Pyramid/CoDriving各五行，不重测；填满35号第10章TVM三模型表。只有总audit为paper_ready=true时才更新论文TeX、编译PDF并视觉检查。不得预设TVM、INT8或GEAR必须胜出，不得用pilot、surrogate、hand rewrite或跨后端性能结果补表。
```
