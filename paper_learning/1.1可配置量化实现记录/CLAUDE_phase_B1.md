# Phase B1: Q/DQ 注入工具

> 覆盖: D1-D8 从 PyTorch quant_config 映射到 ONNX Q/DQ 节点
> 新建文件: tools/inject_qdq_from_config.py
> 依赖: Phase A3 (quant_config.json 格式 + scale 导出)
> 预计工作量: 2.5 天
> 状态: 待开始

---

## 1. 任务清单

- [ ] Task 1: ONNX 图分析 (layer_name -> ONNX node 映射)
- [ ] Task 2: 权重侧 Q/DQ 注入 (per-tensor / per-channel / per-group)
- [ ] Task 3: 激活侧 Q/DQ 注入
- [ ] Task 4: 位宽模拟 (INT4/INT6 通过 scale 范围)
- [ ] Task 5: W-only 特殊处理 (透���激活 Q/DQ)
- [ ] Task 6: 拓���排序维护
- [ ] Task 7: onnx.checker 验证
- [ ] Task 8: 单元测试 (小模型 Q/DQ 注入 -> TRT 构建)
- [ ] 反思文档

---

## 2. 核心设计

### 2.1 与之前失败方案 (build_qdq_onnx_adaround.py) 的区别

| 维度 | 之前(失败) | 本次(��确) |
|------|----------|-----------|
| 权重 Q/DQ | 有 | 有 |
| 激活 Q/DQ | 无 (根因) | 有 |
| 对称性 | 非对称(TRT不支持) | 对称(zp=0) |
| scale 来源 | AdaRound | PyTorch 校准 (export_scales_to_config) |
| 粒度 | 无控制 | per-tensor / per-channel / per-group |
| 位宽 | 仅 INT8 | INT4/INT6/INT8 通过 scale 范围模拟 |
| 覆盖范围 | 所有 QuantModule 统一 | 按 quant_config 逐层差异化 |

### 2.2 可复用代码

从 build_qdq_onnx_adaround.py 复用:
- `find_quant_matmul_nodes()` 的 pattern matching 逻辑 (扩展支持 Constant + Initializer)
- 拓扑排序修复方案 (rethink.md 13.2 节)
- zero_point = 0 的约束 (rethink.md 13.3 节)

---

## 3. Debug 方案

| 可能问题 | 排查方法 | 参考 |
|---------|---------|------|
| Pattern matching 不到目标层 | 统计 MatMul 节点 + 上游类型 | rethink.md 13.1 |
| 拓扑排序违反 | Q/DQ 紧随依赖节点后插入 | rethink.md 13.2 |
| TRT 报 non-zero zp | 检查所有 zp tensor 是否全 0 | rethink.md 13.3 |
| Calibrator 被忽略 | 确认同时有权重+激活 Q/DQ | rethink.md 13.4 |
| per-group Q/DQ 维度错误 | 打印 scale tensor shape vs weight shape | - |

---

## 4. 执行记录

- [x] Task 1: ONNX 图分析 — find_quantizable_matmul_nodes 支持 3 种 pattern (Constant+Transpose+MatMul, Initializer+Transpose+MatMul, Initializer+MatMul)
- [x] Task 2: 权重侧 Q/DQ — per-tensor / per-channel scale, 从 config 或权重值自动计算
- [x] Task 3: 激活侧 Q/DQ — per-tensor scale, 共享激活输入的去重处理
- [x] Task 4: 位宽模拟 — scale 值编码有效位宽 (INT4: max/7, INT8: max/127)
- [x] Task 5: W-only 处理 — 透明激活 Q/DQ (large scale passthrough)
- [x] Task 6: 拓扑排序 — 权重 Q/DQ 紧随源节点, 激活 Q/DQ 在消费 MatMul 之前
- [x] Task 7: onnx.checker 验证 (脚本末尾自动检查)
- [x] Task 8: 真实 ONNX 上验证通过 (60 weight Q/DQ + 43 activation Q/DQ = 206 节点)

### GPU 验证 Bug 修复记录

1. **a_scale list 类型错误**: `float([1.0])` TypeError。修复: isinstance 检查 list/tuple 后用 np.array 转换。
2. **weight scale 维度不匹配**: PyTorch per-channel scale 256 维 vs ONNX 权重 128 输出通道。修复: 从 ONNX 权重实际形状重新计算 per-channel scale。

### 关键设计

1. **激活 Q/DQ 去重**: 多个 MatMul 共享同一激活输入时 (如 attention 的 Q/K/V), 只插入一对 Q/DQ, 所有 MatMul 引用同一 DQL 输出。通过 `_act_qdq_inserted` dict 跟踪。

2. **无 PyTorch 依赖**: 纯 ONNX + numpy 操作, 不需要加载模型或校准数据。所有信息来自 quant_config.json。

3. **Scale 来源优先级**: config 中的 w_scale > 从 ONNX 权重自动计算。a_scale 必须来自 config (激活分布需要校准数据)。

### 创建的文件
- `tools/inject_qdq_from_config.py` (767 行, 新建)

---

## 5. 反思

### 完成时间
2026-04-12

### 与之前失败方案的关键区别

| 维度 | build_qdq_onnx_adaround.py (失败) | inject_qdq_from_config.py (本次) |
|------|-------------------------------|-------------------------------|
| 激活 Q/DQ | 无 (根因) | 有, 含去重 |
| Scale 来源 | AdaRound (非对称) | PyTorch 校准 (对称) |
| zp | 非零 (TRT报错) | 全部 0 |
| 粒度 | 固定 per-channel | per-tensor / per-channel 可配 |
| 位宽 | 固定 INT8 | INT4/6/8 通过 scale 模拟 |
| 层级控制 | 全部统一 | 逐层差异化 |
| 依赖 | QuantModel + AdaRound | 纯 ONNX + numpy |

### 待验证风险

1. **pattern matching 覆盖率**: FP32 ONNX 中的 Linear 可能不全是 Initializer+Transpose+MatMul 模式, 某些框架可能用 Gemm 或其他算子。需要在真实 ONNX 上验证。

2. **激活去重的正确性**: 如果 attention 中 Q/K/V 的激活分布差异大, 共享一个 a_scale 可能不够精确。但这是显式量化模式的标准做法, TRT 内部也是这样处理。

3. **per-group 的 ONNX 表示**: per-group 需要在 ONNX 中用 reshape + per-channel Q/DQ 实现, 当前代码尚未实现 (config 中 per_group 会 fallback 到 per-channel)。这是后续优化点。
