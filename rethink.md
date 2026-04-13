# UniV2X TRT 优化经验与反思

**最后更新**：2026-04-08

---

## 一、ONNX 导出相关

### 1.1 多代理 checkpoint 的权重前缀检测

**问题**：`export_onnx_univ2x.py` 使用 `next(iter(sd))` 自动检测 checkpoint 前缀，在 cooperative checkpoint（含 `model_ego_agent.*` 和 `model_other_agent_inf.*` 两组 key）中始终取到 ego 前缀，导出 infra 模型时加载了 ego 权重。

**教训**：多代理 checkpoint 必须显式指定前缀（使用 config key name），不能依赖 key 顺序自动推导。字典迭代顺序在不同 Python 版本和序列化方式下可能不同，是隐式假设。

**修复**：`prefix = model_key + '.'`，仅在匹配不到时 fallback 到旧逻辑。

---

### 1.2 数据集摄像头配置不匹配

**问题**：早期 BEV 引擎用 NuScenes 默认 6-cam 配置导出，但 V2X-Seq-SPD 数据集只有 1 个摄像头，输入形状完全不匹配。

**教训**：导出 TRT 引擎前必须验证数据集的真实配置（cam 数、图像分辨率）。引擎文件名应编码关键配置参数（如 `_1cam`、`_200`），避免错误复用旧引擎。

---

### 1.3 `nn.Parameter` vs `torch.Tensor` 在 ONNX 中的表示差异

**问题**：量化框架中 `QuantModule.org_weight` 存储为普通 `torch.Tensor`（非 `nn.Parameter`），ONNX tracer 将其内嵌为 `Constant` 节点而非 `Initializer`。导致 Q/DQ 节点插入脚本搜索 `Initializer → Transpose → MatMul` 模式时只匹配到 24 个（未量化的 Linear），而非目标 36 个 QuantModule。

**教训**：分析 ONNX 图结构前，必须先确认权重的存储类型（Parameter → Initializer，Tensor → Constant），两者在图中的表示有本质差异。

---

### 1.4 ONNX 图编辑的拓扑排序

**问题**：将 Q/DQ 节点统一前置（`graph.node.insert(0, node)`）导致 Q/DQ 出现在其依赖的 Constant 节点之前，违反 ONNX 拓扑顺序。

**教训**：ONNX 图编辑时，新节点必须插入到其所有输入节点之后。推荐在图遍历中就地追加（`new_node_list.append`），而非统一前置或后置。

---

## 二、TRT 构建与量化相关

### 2.1 PLUGIN_V2 层不应设置显式精度

**问题**：`layer.precision = trt.DataType.HALF` 显式设置 PLUGIN_V2 精度时，TRT 在该层输入侧插入额外 Dequantize 节点，引入级联误差（AMOTA 从 0.355 跌至 0.278）。

**教训**：TRT 混合精度应交给 TRT 自动决策。对于无 INT8 实现的自定义 plugin，TRT 会自动分配 FP16，无需手动指定。

---

### 2.2 INT8 校准数据必须覆盖真实推理分布

**问题**：BEV 编码器的 Temporal Self-Attention 依赖前帧 `prev_bev` 状态。用零初始化 `prev_bev` 校准时，TSA 激活范围偏低，导致 INT8 scale 不准确。

**教训**：校准数据必须模拟真实推理分布。对于有时序依赖的模型，校准数据需包含真实的历史状态（非零初始化）。50 帧 temporal 校准显著优于 10 帧零初始化（AMOTA 0.364 vs 0.334）。

---

### 2.3 `sampling_offsets` 等坐标层必须保持高精度

**问题**：对 MSDeformableAttention 的 `sampling_offsets` 和 `attention_weights` 层进行 INT8 量化后，BEV 特征完全失效。

**教训**：空间坐标相关的层（offset、attention weight）对量化误差极度敏感，且误差无法被下游层吸收。量化敏感层需通过逐层分析人工标记 skip。

---

### 2.4 AdaRound 的双重量化陷阱

**问题**：AdaRound 在 PyTorch 侧优化舍入（产出 W_fq），内嵌 ONNX 后 TRT 重新推导 scale 并再次量化，AdaRound 优化完全被覆盖（AMOTA 0.190）。

**教训**：TRT INT8 必须通过 Q/DQ 节点或 Calibrator 显式指定 scale，否则 TRT 会从权重重新推导。不能假设 TRT 会"尊重"已有的 fake-quantized 权重。

---

### 2.5 TRT 显式 Q/DQ 模式的全局性

**问题**：只插入权重 Q/DQ（无激活 Q/DQ）时，TRT 检测到任何 Q/DQ 节点即进入 explicit 模式，Calibrator 完全失效，激活层退化为 FP16。结果 AMOTA 从 0.353 跌至 0.137。

**教训**：TRT 的量化模式是二元对立的（implicit vs explicit），不存在混合模式。一旦引入 Q/DQ 节点，即承诺为所有量化层（包括激活）提供显式 scale。权重-only Q/DQ 是常见误区。

---

### 2.6 非对称量化与 TRT GPU 不兼容

**问题**：AdaRound 使用非对称 UINT8 量化（zero_point ∈ [0, 255]），TRT GPU 路径要求对称 INT8（zero_point = 0）。

**教训**：若目标部署平台是 TRT GPU，量化框架应从一开始使用对称量化（`sym=True`），避免后期适配成本。

---

## 三、端到端测试与排查相关

### 3.1 AMOTA 下降排查应先隔离变量

**问题**：Section 18 中 AMOTA 下降被误归因为"Hook E 干扰跟踪状态"，实际根因是 infra BEV 引擎配置错误（错误权重 + 错误摄像头数）。

**教训**：AMOTA 下降排查应逐个开关各 Hook 并与 baseline 对比，而非直接怀疑最后添加的组件。正确的排查方法：
1. 先确认无 Hook 的 PyTorch baseline
2. 逐个启用 Hook，观察每步增量变化
3. 对比 ego-only vs ego+infra，隔离 infra 引擎的影响

---

### 3.2 TRT FP16 数值波动是正常现象

**问题**：ego-only TRT（AMOTA 0.378）显著优于 PyTorch（0.338），但 ego+infra TRT（0.341）回落到 PyTorch 水平。

**解释**：TRT FP16 数值特性与 PyTorch FP32 存在微小差异，某些情况下恰好对特定评估集有利。这不是"TRT 比 PyTorch 更好"，而是数值波动。Ego BEV 恰好获益，infra BEV 没有，整体回归到均值附近。

---

### 3.3 Hook 包装不应改变调用链语义

**问题**：Hook E 的 `_hook_get_det` 只是捕获中间结果并透传返回值，但最初怀疑它干扰了跟踪状态。

**结论**：经验证 Hook E 对 AMOTA 零影响（三组配置完全一致 0.341），hook 包装的纯旁路设计是正确的。但作为最佳实践，monkey-patch hook 应在设计时明确声明：(1) 是否修改输入/输出；(2) 是否有副作用。

---

### 3.4 INT8 BEV + Hook D 的超线性误差叠加

**问题**：INT8 BEV 编码器 + Hook D（FP16 检测头 TRT）组合后 AMOTA 从 0.341 暴跌至 0.241（-0.100），但两者单独使用时精度都正常。

**关键对照实验**：

| BEV 精度 | 检测头 | AMOTA | 结论 |
|:--------:|:------:|:-----:|------|
| FP16 | PyTorch | 0.341 | Baseline |
| FP16 | FP16 TRT (Hook D) | 0.345 | Hook D 零损 ✅ |
| FP16 | INT8 TRT (Hook D) | 0.332 | INT8 量化损失 -0.013 ✅ |
| INT8 | PyTorch | 0.341 | INT8 BEV 在 Hook E 下零损 ✅ |
| INT8 | FP16 TRT (Hook D) | **0.241** | ❌ 超线性退化 |

**根因分析**：
1. **INT8 BEV 改变了特征分布**：虽然 CosSim > 0.999，但 INT8 BEV 的输出与 FP16 存在微小的系统性偏移（非随机噪声）。
2. **Hook D 的 Cross-Attention 放大了偏移**：检测头 Decoder 的 6 层 Cross-Attention 以 BEV 作为 key/value。BEV 特征的系统性偏移导致注意力权重分布改变，采样位置偏移，误差在每层 Decoder 中累积。
3. **零填充 query 的 Self-Attention 进一步放大**：1101-query 中约 100-200 个零填充 query 在 Self-Attention 中与真实 query 交互。BEV 偏移导致真实 query 特征略微偏移 → 零填充 query 接收到不同的 cross-attention 结果 → Self-Attention 中的干扰模式改变 → 级联放大。
4. **跟踪状态帧间传播**：检测头输出的 ref_pts、track_score 直接影响下一帧的查询状态，误差在 168 帧序列中累积。

**为什么 Hook E（下游头）不受影响？** Hook E 使用 INT8 BEV 作为输入但不做 Cross-Attention — 下游头（Motion/Occ/Planning）通过全连接层处理 BEV 特征，对系统性偏移不敏感。而 Hook D 的 Decoder 通过 Deformable Cross-Attention 对 BEV 空间位置做精细采样，对偏移极度敏感。

**教训**：
- 模块间的量化误差不是简单相加的。两个单独可接受的 INT8 模块串联后可能产生超线性退化。
- BEV→Decoder 的 Cross-Attention 是误差放大的关键路径。如果上游（BEV）使用 INT8，下游（Decoder）最好保持 FP16，反之亦然。
- **正确的 INT8 全链路方案**：如果同时需要 INT8 BEV 和 INT8 检测头，应使用 INT8 BEV 的输出作为检测头的校准数据（当前校准数据使用 FP16 BEV 输出，与实际 INT8 部署时不匹配）。

---

### 3.5 检测头 INT8 量化本身是有效的

**验证**（FP16 BEV 条件下，隔离 INT8 检测头的纯影响）：
- FP16 Hook D：AMOTA = 0.345
- INT8 Hook D：AMOTA = 0.332（差 0.013）
- 引擎大小：33.4 MB → 18.2 MB（-45.5%）

**教训**：检测头 INT8 量化在 FP16 BEV 条件下精度可接受（-0.013）。量化策略（MSDAPlugin 保持 FP16、其余层 INT8）与 BEV 编码器一致，验证了该策略对 Deformable Attention 模型的通用性。

---

## 四、QuantV2X 参考实现的启示

### 4.1 AdaRound 和 TRT INT8 是两条独立管线

QuantV2X 的量化体系由两条**完全独立**的管线构成：

| 管线 | 核心方法 | 产物 |
|------|---------|------|
| **PyTorch W8A8 路径** | AdaRound（W8）+ LSQ（A8） | 精度指标（论文主体贡献） |
| **TRT INT8 部署路径** | IInt8EntropyCalibrator2 | `.plan` TRT 引擎（实测延迟） |

AdaRound 优化出的 `alpha` 参数完全不传入 TRT 路径。TRT INT8 使用标准 Calibrator，与 AdaRound 结果无关。

### 4.2 论文中提及的 custom CUDA kernels 未开源

QuantV2X 公开代码中不存在任何 TRT plugin 源码。UniV2X 的 MSDAPlugin 已完整实现并开源，TRT plugin 工程化程度优于 QuantV2X 的公开代码。
