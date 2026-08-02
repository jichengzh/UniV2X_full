# SHCoS Abstract Draft

> 状态：AAAI 初始摘要版本。方法描述基于当前设计；结论中的模型数、硬件数和性能提升为初始预估值，最终必须替换为统一实验协议下的实测结果。

## English Draft

As intelligent perception expands to edge environments, including vehicles and roadside infrastructure, efficient deployment has become increasingly important. Modern perception models use complex architectures to improve accuracy, but their computational demands conflict with constrained edge resources. Addressing this tension requires unified optimization of model compression and hardware execution scheduling. Yet existing approaches often optimize them separately, overlooking how compression reshapes operators and execution paths and how hardware backends determine realized deployment gains. This separation can lead to stage-wise local optima and yield a suboptimal Pareto frontier. To address this gap, we present SHCoS, a software–hardware co-optimization framework that models compression–execution coupling at both feasibility and performance levels. For feasibility, computation-graph dependency partitioning and backend capability profiling jointly define a compact space of structurally valid and deployable compression choices. For performance, a bilevel exploration architecture couples hardware-informed outer-level compression search with inner-level schedule refinement for selected candidates. Multi-metric cost models predict both the values and relative rankings of AP, latency, and energy for each candidate configuration, enabling rapid and accurate identification of Pareto-promising candidates for actual evaluation. The resulting observations are fed back to incrementally update the cost models and guide subsequent Pareto-front exploration. Across five representative V2X perception models and three heterogeneous hardware platforms, SHCoS improves Pareto hypervolume by 15.7\% on average over the strongest sequential compression-then-scheduling baseline.

## 中文逻辑对照

1. 随着智能感知向车端、路侧基础设施等边缘环境拓展，高效部署日益重要。
2. 现代感知模型依赖复杂架构提升精度，但其计算需求与受限的边缘资源之间存在矛盾。
3. 解决这一矛盾，需要统一优化模型压缩与硬件执行调度。
4. 现有方法往往将二者分开优化，忽视了压缩对算子和执行路径的重塑，以及硬件后端对实际部署收益的决定作用。
5. 这种割裂可能使分阶段优化陷入局部最优，并产生次优的 Pareto 前沿。
6. 为弥补上述不足，本文提出 SHCoS，一种在可行性和性能两个层面建模压缩--执行耦合的软件--硬件协同优化框架。
7. 在可行性层面，计算图依赖分区与后端能力刻画共同定义结构有效且可部署的紧凑压缩空间。
8. 在性能层面，双层探索架构将硬件信息驱动的外层压缩搜索与面向所选候选的内层调度细化相结合。
9. 多指标代价模型同时预测各候选配置的 AP、延迟和能耗数值及相对排序，从而快速、准确地识别值得实测的 Pareto 潜力候选。
10. 实测结果随后回流，用于增量更新代价模型并指导后续 Pareto 前沿探索。
11. 在五个代表性 V2X 感知模型和三个异构硬件平台上，SHCoS 相比最强的串行压缩后调度基线，平均将 Pareto 超体积提升 15.7\%。

## 结论句证据边界

| 结论表述 | 必须绑定的证据 |
|---|---|
| `hardware-adaptive` | 不同目标硬件上的候选排序、调度实现或 Pareto 配置存在可验证差异 |
| `diverse perception workloads` | 多个感知模型或结构在统一协议下完成评测 |
| `heterogeneous hardware platforms` | 多类目标硬件平台完成一致口径的真实测量 |
| `trade-offs among accuracy, latency, and energy` | 最终前沿坐标来自统一协议下的真实 AP、延迟和能耗评测 |

## 当前不能写入摘要的主张

- automatic RouteB INT8 比 FP16 更快；
- mixed INT8 已恢复有效 AP；
- per-node calibration 或 automatic region formation 已带来端到端性能提升；
- 所有设备或所有 V2X 模型均能获得相同收益；
- pairwise rank loss、AP residual predictor 已作为 Phase1 主搜索器运行；
- energy、AP、latency 来自不同 pipeline 的各自最佳数字。

## 结论范围调整规则

如果最终实验尚未覆盖多个感知模型或异构平台，应将 `five representative V2X perception models and three heterogeneous hardware platforms` 收紧为实际完成统一评测的模型和硬件范围。automatic INT8 只有在速度、数值正确性和 AP 门禁均闭合后，才能作为 Pareto 候选的支撑证据，但不在摘要中单独点名具体后端或精度实现。
