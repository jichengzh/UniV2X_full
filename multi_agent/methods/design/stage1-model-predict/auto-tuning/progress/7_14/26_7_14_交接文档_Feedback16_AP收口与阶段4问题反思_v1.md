# 交接文档 26：Feedback16 AP 收口与阶段 4 问题反思

日期：2026-07-16  
状态：Feedback16 已收口，阶段 4 未收口  
范围：完成首批 16 行在线反馈的 AP/TVM-INT8 证据闭环，并明确阶段 3.5 与阶段 4 的边界。

## 1. 相关文档与结果

- [论文级最终实验与表格实现路线](../../论文级最终实验与表格实现路线_v1.md)
- [24 号交接文档：Gold96 最终收口](24_7_14_交接文档_Gold_Coldstart96_v3最终收口_v1.md)
- [25 号交接文档：Gold176 与阶段 3.5](25_7_14_交接文档_Gold176有效性验证与阶段3.5收口_v1.md)
- [Feedback16 候选计划](../../../../../../../results/stage4_feedback16_v1_20260716/candidate_plan.json)
- [Feedback16 最终 16 行表](../../../../../../../results/stage4_feedback16_v1_20260716/final_feedback16_v1/feedback16_final.csv)
- [Feedback16 四组完整性审计](../../../../../../../results/stage4_feedback16_v1_20260716/final_feedback16_v1/feedback16_audit.json)
- [阶段 4 首轮完整评估](../../../../../../../results/stage4_selection_completion_v1_20260716/)

## 2. 本轮问题与目标

本轮首先处理 Feedback16 的 AP 缺口。候选为两个模型共享的两个 width：

| model | width | 四臂 |
|---|---|---|
| CoDriving | `48x32x64` | TVM-FP16、TVM-INT8、TRT-FP16、TRT-INT8 |
| CoDriving | `16x64x256` | TVM-FP16、TVM-INT8、TRT-FP16、TRT-INT8 |
| Pyramid | `48x32x64` | TVM-FP16、TVM-INT8、TRT-FP16、TRT-INT8 |
| Pyramid | `16x64x256` | TVM-FP16、TVM-INT8、TRT-FP16、TRT-INT8 |

合计 `4` 个完整组、`16` 行。停止条件不是只有 latency/energy，而是每行均形成：

```text
(model, width, q_mode, capability context)
-> latency + energy + AP30/AP50/AP70
-> performance result SHA + AP report SHA
```

所有行必须保持 `split=online_feedback`，不得改写为 Gold176 的 `train` 数据。

## 3. TVM-INT8 问题诊断与修复

### 3.1 暴露的问题

首轮 16 行 performance executor 均返回 success，但 4 条 TVM-INT8 产物没有绑定 `tensor_quant_params.json`。普通 AP runner 因缺少输入/输出量化参数而失败。

该失败属于 **artifact contract 缺失**，不是以下结论：

1. 不是模型 AP 必然为 `0`；
2. 不是 TVM-INT8 numerical feasibility failure；
3. 不能使用无量化合同的 provisional latency/energy 作为最终反馈。

### 3.2 正确修复路径

4 条 TVM-INT8 已切换到阶段 3 验证过的自动 scale-aware 路径：

```text
calibration NPZ
-> 自动生成 tensor quant contract
-> RouteB scale-aware rebuild
-> candidate/native exact gate
-> latency/energy
-> sanity AP
-> full-1789 AP
```

该路径没有使用手写 `im2col+MMA` 算子，也没有向搜索器注入“TVM 应选 FP16”之类人工规则，符合自动搜索叙事。

截至本轮记录时，4 条 TVM-INT8 均已完成 quant contract、重建、性能/能耗、sanity 和 full AP。导入器已严格验证 `4/4` 行，并生成：

- `results/stage4_feedback16_v1_20260716/tvm_int8_repair_performance_import.jsonl`
- `results/stage4_feedback16_v1_20260716/tvm_int8_repair_ap_import.jsonl`

原 4 条无 quant-contract performance 结果只保留为失败反思，不进入最终 Feedback16 表。

## 4. Feedback16 AP 执行状态

| 子任务 | 状态 |
|---|---|
| 16 行 source / checkpoint / calibration source | 已完成 |
| 16 行首轮 performance/energy | `16/16` 完成，其中 4 条 TVM-INT8 为 provisional |
| 4 条 TVM-INT8 scale-aware repair | `4/4` 完成 |
| 12 条普通臂 sanity | `12/12` 通过 |
| 12 条普通臂 full-1789 AP | `12/12` 通过，均为首次尝试成功 |
| 最终 16 行表与 4 组完整性审计 | `16 measured / 0 failure / 0 pending`，`4/4` 组完整 |

普通 full AP 按模型拆分到 GPU6/GPU7，使用独立状态文件，最终只合并 terminal success；TVM-INT8 使用 repair import 覆盖 provisional evidence。最终表中所有 latency、energy、AP 均为有限值，性能/AP 文件 SHA 全部复核通过。

最终化入口同时 fail-closed 绑定 `manifest schema -> split -> output schema`，并逐行核对 performance/AP 路径与计划；TVM-INT8 repair 例外必须匹配同一 `model + width`。相关回归测试与 Stage4 测试合计 `44/44` 通过。

### 4.1 最终 16 行核心结果

| model | width | backend | q | latency ms | energy J | AP70 |
|---|---|---|---|---:|---:|---:|
| CoDriving | `16x64x256` | TRT | FP16 | `0.372448` | `0.133360` | `0.369175` |
| CoDriving | `16x64x256` | TRT | INT8 | `0.332768` | `0.109286` | `0.243576` |
| CoDriving | `16x64x256` | TVM | FP16 | `2.568896` | `0.417682` | `0.368919` |
| CoDriving | `16x64x256` | TVM | INT8 | `3.351461` | `0.395071` | `0.314457` |
| CoDriving | `48x32x64` | TRT | FP16 | `0.358080` | `0.127957` | `0.355659` |
| CoDriving | `48x32x64` | TRT | INT8 | `0.306880` | `0.101120` | `0.246778` |
| CoDriving | `48x32x64` | TVM | FP16 | `2.264400` | `0.375672` | `0.354988` |
| CoDriving | `48x32x64` | TVM | INT8 | `2.346259` | `0.371897` | `0.346189` |
| Pyramid | `16x64x256` | TRT | FP16 | `0.529504` | `0.169607` | `0.562404` |
| Pyramid | `16x64x256` | TRT | INT8 | `0.440128` | `0.135724` | `0.497757` |
| Pyramid | `16x64x256` | TVM | FP16 | `37.968975` | `2.834196` | `0.562566` |
| Pyramid | `16x64x256` | TVM | INT8 | `4.872094` | `1.058093` | `0.559432` |
| Pyramid | `48x32x64` | TRT | FP16 | `1.491520` | `0.467433` | `0.557869` |
| Pyramid | `48x32x64` | TRT | INT8 | `1.472960` | `0.469738` | `0.452119` |
| Pyramid | `48x32x64` | TVM | FP16 | `24.325888` | `2.295607` | `0.557265` |
| Pyramid | `48x32x64` | TVM | INT8 | `4.045826` | `0.907614` | `0.549343` |

最终表 SHA：

```text
feedback16_final.jsonl
d71a5f0ac30d58ba4187814bc8312623949e17bc1080e0bdc5e0546a5de88f47

feedback16_audit.json
41bb48e8e67ed71a1e78ba83db452675adb8c29c2298dde61275ad3051f3061b
```

### 4.2 TVM-INT8 本轮结论边界

1. AP pipeline 问题已解决：4 条 TVM-INT8 均完成 scale-aware 编译、full AP 和 SHA 绑定，AP70 为 `0.3145/0.3462/0.5594/0.5493`，不存在 `AP=0` 或数值崩溃。
2. 速度结论仍是条件性的：CoDriving 两点的 TVM-INT8 相对 TVM-FP16 分别慢约 `30.5%` 和 `3.6%`；Pyramid 两点则分别快约 `7.79x` 和 `6.01x`。
3. 不能由这 4 点写成“TVM-INT8 恒快”或“恒慢”。它们支持把模型结构、图特征和 capability probe 作为 cost-model 条件，让搜索器依据冷启动与回流数据选择量化臂。
4. Pyramid 的 TVM-FP16 延迟显著高于同组 TRT 和 TVM-INT8，需要在阶段 4 回流前作为高分歧样本复核 runner/graph-feature 解释；当前真实结果可以入 feedback，但不能单独上升为后端普遍规律。

## 5. 阶段 3.5 数据现状回顾

### 5.1 已完成的部分

Gold176 已冻结为 `initial cold-start pool v1`：

1. `176` 行、`44` 个完整四臂组；
2. `174` 行具有 latency、energy、AP 真测证据；
3. `2` 行为确认的 TVM-FP16 shape feasibility failure；
4. graph features 为 `44/44` 组；
5. 证据文件 SHA 审计已完成。

因此，**阶段 3 数据生产已经收口，Gold176 足以启动阶段 4**。

### 5.2 尚未完成的部分

阶段 3.5 的严格泛化充分性仍未收口，原因是：

1. 原 6 个 holdout 的标签参与过补点决策，只能称为 `adaptive_validation_v1`；
2. Gold176 新增 8 组尚无独立跨时段 repeat audit；
3. latency top-k、Pareto recall、联合 coverage 和双向 few-shot 收益未全部达到预设门槛；
4. 当前只有 Pyramid 与 CoDriving 两个模型，不能证明普适跨模型泛化。

### 5.3 是否影响阶段 4

不阻塞阶段 4 的 cost-model/loss 选择，但限制结论范围：

1. 可以用 nested grouped CV 比较低容量模型、target encoding、ranker 和 uncertainty；
2. 不能把当前结果写成独立模型 holdout 或普适跨模型泛化；
3. Feedback16 必须作为 `online_feedback` 增量批次，不能回溯修改 Gold176 baseline；
4. 最终方法锁定前仍需冻结一个不读取标签的新 holdout，并完成新增边界 repeat audit。

结论应分别表述为：

```text
Stage 3 data production: closed
Stage 3.5 strict generalization sufficiency: open
Stage 4 admission: allowed
```

## 6. 阶段 4 当前主要问题

### 6.1 当前 acquisition 不优于 random

同一 model-internal HV replay 下：

| policy | 达到 95% HV 的中位完整组数 | 中位实测行数 |
|---|---:|---:|
| Pareto + uncertainty | `12` | `48` |
| random group | `10` | `40` |

当前 acquisition 多用 `2` 组、`8` 行，必须 reject，不能进入最终方法。

### 6.2 Pareto recall 偏低

value OOF 的平均 Pareto recall 为 `0.417`，precision 为 `0.527`，平均 HV regret 为 `0.0247`。这说明模型保住了主要支配体积，但漏掉较多次级前沿点。

### 6.3 uncertainty 可用但区间过宽

LGBM quantile + group conformal 的三个 target-wise group coverage 均超过 `0.80`，但平均相对区间宽度为 `0.620`；三个目标与四臂同时覆盖仅为 `0.595`。当前只能作为保守的首轮 uncertainty，不能宣称校准已经完成。

### 6.4 独立 ranker 没有增益

LGBMRanker 相对 value-head 的 top-10% recall 平均下降 `0.118`。独立 rank head 已 reject，下一轮不再扩大 ranker 候选矩阵。

### 6.5 泛化声明受限

当前 OOF 是两个模型各自内部的 grouped interpolation，不是 leave-one-model-out，也不是新的独立 holdout。阶段 4 可以做工程选择，但不能形成最终跨模型泛化结论。

## 7. 阶段 4 解决方案与停止条件

### P0：先回流 Feedback16

1. 已完成 16 行证据闭环和 4 组审计；
2. 为 4 个新组生成同版本 graph features；
3. 保持 Gold176 baseline 不变，建立 `Gold176 + Feedback16` 增量训练视图；
4. 重跑 ranking、coverage、Pareto/HV 与相同 seed 的 replay，报告绝对值和相对 Gold176 的变化。

### P1：只改 acquisition，不扩大 value heads

在同一初始组、seed、预算和候选可见性合同下比较：

1. uncertainty-only；
2. expected hypervolume improvement；
3. predicted-frontier diversity；
4. EHVI 与 uncertainty/diversity 的低容量组合。

任何策略都不得读取未测候选标签。选择标准首先是 samples-to-95%-HV 中位数不弱于 random，其次才比较 Pareto recall 和方差。

### P2：校准与前沿恢复

1. 使用 Feedback16 作为真正的 post-cold-start calibration batch；
2. 比较更新前后 target-wise coverage、simultaneous coverage 和 interval width；
3. 若 coverage 提升但区间不收窄，不继续增加 quantile 模型复杂度，而在高不确定完整组上补测；
4. 评价必须按模型计算 Pareto/HV，再汇总，禁止跨模型直接混算绝对 AP/latency。

### P3：阶段 4 收口门槛

阶段 4 只有同时满足以下条件才收口：

1. canonical value heads 和 uncertainty calibrator 已冻结；
2. 独立 ranker 的启用/拒绝决定有消融证据；
3. acquisition 在固定 replay 上至少不弱于 random；
4. Feedback16 回流后 ranking、coverage、Pareto/HV 结果可复现；
5. 训练表明确区分 `initial_coldstart` 与 `online_feedback`；
6. 新独立 holdout 的候选在读取标签前冻结。

## 8. 当前决策

```text
Gold176:
  frozen initial cold-start baseline

Feedback16:
  16/16 measured, 4/4 complete groups, independent online-feedback batch

TVM-INT8 AP contract:
  repaired with automatic scale-aware RouteB pipeline

Stage 3.5 strict closure:
  false; does not block Stage 4, but limits claims

Stage 4:
  canonical value heads frozen
  independent ranker rejected
  LGBM quantile + group conformal frozen
  predicted-frontier diversity selected
  stage4_closed=true
  stage5_search_ready=true
```

## 9. P1-P3 实施结果与阶段 4 收口

### 9.1 P1：固定 replay 的 acquisition 对照

所有策略使用相同初始完整组、seed、预算和候选特征可见性；未测候选的 latency、energy、AP 标签不可见。标签扰动回归测试覆盖全部非随机策略。

| policy | Gold176 中位组数 | random | Feedback 视图中位组数 | random | 决策 |
|---|---:|---:|---:|---:|---|
| uncertainty-only | `22` | `10` | `17` | `13` | reject |
| EHVI | `11` | `10` | `10` | `13` | reject，基线弱于 random |
| 旧 Pareto + uncertainty | `12` | `10` | `15` | `13` | reject |
| predicted-frontier diversity | `8` | `10` | `10` | `13` | **select** |
| EHVI + uncertainty + diversity | `8` | `10` | `13` | `13` | eligible，作为备选 |

最终 acquisition 冻结为 `predicted_frontier_diversity`。该策略在两套 replay 上均优于 random；纯 uncertainty 的结果说明不能把“不确定度最大”直接写成采集规则。

### 9.2 P2：Feedback16 真增量对照

合并视图包含 `192` 行、`48` 组，显式保留：

- `initial_coldstart=176` 行；
- `online_feedback=16` 行；
- Gold176 原 `train/locked_holdout` split 不变；
- 两批 group overlap 为 `0`。

严格增量协议为：原 6 个 `locked_holdout` 组只做 conformal 校准；4 个 Feedback 组逐组留出，比较 Gold176-only 与 Gold176 + 其余 3 个反馈组。价值头在回流前冻结。

| target | 指标 | 回流前 | 回流后 | 变化 |
|---|---|---:|---:|---:|
| latency | MAE ms | `1.3531` | `1.4698` | `+8.62%` |
| latency | Spearman | `0.9934` | `0.9941` | 基本不变 |
| latency | interval width ms | `35.4574` | `36.8250` | `+3.86%` |
| energy | MAE J | `0.1991` | `0.2077` | `+4.29%` |
| energy | Spearman | `0.8706` | `0.9235` | 改善 |
| AP70 | MAE | `0.0390` | `0.0291` | `-25.44%` |
| AP70 | Spearman | `0.8437` | `0.8938` | 改善 |
| AP70 | interval width | `0.4871` | `0.4682` | `-3.89%` |

三目标同时 group coverage 回流前后均为 `0.75`。结论是：Feedback 回流机制可复现且对 AP 有明确收益，但少量反馈不会保证所有目标同步改善。阶段 5 必须按 target 独立执行更新验收，不能因 AP 改善而自动接受 latency/energy 新模型。

合并 grouped OOF 的 Pareto recall 从 `0.417` 升至 `0.472`，但平均 HV regret 从 `0.0247` 升至 `0.0537`；该结果同样禁止表述为“反馈后前沿全面改善”。

### 9.3 P3：冻结项与门禁

冻结配置：

| 项目 | 冻结结果 |
|---|---|
| latency value head | `extra_trees_log` |
| energy value head | `extra_trees_log` |
| AP70 value head | `lgbm_huber_residual` |
| ranker | reject，使用 value heads |
| uncertainty | `lgbm_quantile + group conformal` |
| acquisition | `predicted_frontier_diversity` |

新的独立 holdout 已在读取标签前冻结，共 4 组、Pyramid/CoDriving 各 2 组；与 Gold176/Feedback16 无重叠，manifest 中不含 latency、energy、AP 或 terminal status。注意这些组的 source/checkpoint 尚待阶段 5 物化，当前冻结的是评估身份，不是测量完成状态。

Feedback cost-model 与增量校准均以相同 seed 重跑，输出逐字节一致：

```text
feedback cost-model SHA256:
00785aa2fe5112aaab130ffebce492ed15843ed59aee0ab9ae5c8944b715cbf4

feedback update eval SHA256:
7539864717ee51156b45df319c2a19740892cc06b6fa7886b252c488de5cc75d
```

P3 八项 fail-closed 门禁均通过；除原停止条件外，收口器还要求至少一个目标同时满足 MAE 改善、区间不变宽且 coverage 不下降。本轮通过该附加门禁的目标是 `AP70`。正式结论：

```text
Stage 4 P1-P3: closed
Stage 5 two-model search: ready to start
Stage 3.5 universal cross-model generalization claim: still open
```

### 9.4 证据入口

- `results/stage4_p1_p3_closure_v1_20260716/stage4_p1_p3_closure_audit.json`
- `results/stage4_p1_p3_closure_v1_20260716/feedback_update_eval.json`
- `results/stage4_p1_p3_closure_v1_20260716/stage5_independent_holdout_manifest.json`
- `results/stage4_p1_p3_closure_v1_20260716/reproducibility_audit.json`
- `results/stage4_p1_p3_closure_v1_20260716/gold176_baseline/`
- `results/stage4_p1_p3_closure_v1_20260716/feedback_completion/`
