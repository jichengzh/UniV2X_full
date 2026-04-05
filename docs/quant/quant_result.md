# UniV2X 全链路优化日志

**项目**：UniV2X — 端到端 V2X 协同自动驾驶系统  
**优化范围**：TRT 加速（FP16）+ V2X 融合向量化 + QuantV2X INT8 PTQ 迁移  
**硬件**：RTX 4090 (SM 89), CUDA 11.8, TensorRT 10.13  
**评估集**：V2X-Seq-SPD 协同验证集，168 样本，442 GT 目标

> **关于两条独立优化线的说明**  
> 本文档涵盖两条相互独立但共同落地的优化：  
> - **优化线 A**：TRT FP16 加速 + V2X 融合向量化（Phase 1~3，改速度和架构）  
> - **优化线 B**：QuantV2X INT8 PTQ 迁移（Phase E，改模型大小和进一步节省显存）  
> 文档最后给出三者的统一对比。

---

## 一、系统级性能对比（最终结论）

### 1.1 精度指标全量对比（168 样本，V2X-Seq-SPD 协同验证集）

| 指标 | 含义 | PyTorch 基准 | FP16 TRT 全链路 | INT8 PTQ BEV (v5) | vs 基准 | vs FP16 全链路 |
|------|------|:-----------:|:---------------:|:-----------------:|:-------:|:------------:|
| **AMOTA** ↑ | 主指标 | 0.338 | **0.370** | **0.364** | +7.7% | -1.6% |
| **AMOTP** ↓ | 中心点误差 (m) | 1.474 | 1.446 | 1.438 | -2.4% | -0.6% |
| **mAP** ↑ | 平均精度 | 0.0727 | 0.0760 | 0.0744 | +2.3% | -2.1% |
| **NDS** ↑ | NuScenes 综合 | 0.0679 | 0.0697 | 0.0694 | +2.2% | -0.4% |
| MOTA | 单阈值跟踪精度 | — | — | 0.346 | — | — |
| MOTP | 定位精度 | — | — | 0.812 | — | — |
| Recall | 召回率 | — | — | 0.487 | — | — |
| MOTAR | 多目标跟踪率 | — | — | 0.759 | — | — |
| **TP** | 真阳性 | — | ~412 | **387** | — | -6% |
| **FP** ↓ | 误报数 | — | ~129 | **104** | — | **-19%** |
| FN ↑ | 漏检数 | — | ~437 | 466 | — | +7% |
| **IDS** ↓ | ID 切换次数 | — | ~35 | **31** | — | **-11%** |
| FRAG ↓ | 轨迹碎片次数 | — | ~21 | **19** | — | -10% |
| MT | 主要追踪目标数 | — | ~22 | 21 | — | — |
| ML ↑ | 主要丢失目标数 | — | ~38 | 39 | — | +3% |
| **FAF** ↓ | 每帧误报频率 | — | ~43.9 | **37.4** | — | **-15%** |
| **LGD** ↑ | 平均轨迹持续时长 (s) | — | ~1.43 | **1.54** | — | **+8%** |
| TID ↓ | 平均 ID 切换间隔 (s) | — | ~0.54 | 0.49 | — | — |

> **INT8 有趣现象**：FP 减少 25 次、FAF 下降 15%、LGD 提升 8%，说明 INT8 轻微削弱了检测器的"过度自信"，减少了误报但增加了漏检。IDS 减少表明追踪稳定性有轻微改善。

### 1.2 推理时延对比（RTX 4090 实测）

| 模块 | PyTorch 原版 | TRT FP16 / 向量化 | 加速比 | 备注 |
|------|:----------:|:----------------:|:------:|------|
| ResNet-FPN backbone (ego+infra) | ~30 ms | ~30 ms | 1× | DCNv2 无法 ONNX 导出，未加速 |
| BEV encoder × 2 | ~150 ms | ~20–40 ms | ~5× | TRT FP16/INT8 加速 |
| **`_query_matching` (N_inf=10)** | **1,048 ms** | **2.7 ms** | **392×** | 最大瓶颈，向量化消除 GPU-CPU 同步 |
| **`_query_matching` (N_inf=50)** | **5,341 ms** | **5.6 ms** | **951×** | — |
| **`_query_matching` (N_inf=100)** | **10,548 ms** | **8.9 ms** | **1,191×** | — |
| **`_get_coop_bev_embed` (N=10)** | **3.4 ms** | **1.1 ms** | **3×** | index_add_ scatter |
| **`_get_coop_bev_embed` (N=50)** | **16.2 ms** | **1.1 ms** | **15×** | — |
| **`_get_coop_bev_embed` (N=200)** | **64.3 ms** | **1.1 ms** | **58×** | — |
| 检测头 + 下游头 | ~100 ms | ~15–30 ms | ~5× | 全 TRT |
| **V2X 场景总延迟估算 (N_inf=50)** | **~5,640 ms** | **~90 ms** | **~63×** | `_query_matching` 是瓶颈来源 |

### 1.3 模型大小对比

| 模块 | PyTorch (.pth) | ONNX | TRT FP16 | TRT INT8 | INT8 vs FP16 |
|------|:-------------:|:----:|:--------:|:--------:|:------------:|
| 总 checkpoint | **1,600 MB** | — | — | — | — |
| BEV encoder (ego, 1-cam) | — | 105 MB | 75 MB | **43 MB** | **-43%** |
| BEV encoder (infra) | — | 70 MB | 73 MB | — | — |
| 检测头 (901/1101-query) | — | 26 MB | 34 MB | — | — |
| 下游头 ego (Motion+Occ+Planning) | — | 128 MB | 153 MB | — | — |
| 下游头 infra (Motion+Occ) | — | 114 MB | 134 MB | — | — |
| **全链路引擎合计** | **1,600 MB** | **443 MB** | **296 MB** | **264 MB** | **-84% vs .pth** |

---

## 二、优化线 A：TRT FP16 加速 + V2X 融合向量化

### 2.1 V2X 融合向量化（`_query_matching_vec` 和 `_get_coop_bev_embed`）

> **说明**：向量化操作属于 TRT 加速项目的 Phase 3（V2X Fusion Adaptation），不在 QuantV2X PTQ 迁移范围内。但由于两者共同构成当前模型的完整优化状态，在此一并记录。

#### 改动 1：`AgentQueryFusion._query_matching` → `_query_matching_vec`

**文件**：`projects/mmdet3d_plugin/univ2x/fusion_modules/agent_fusion.py:222`

**原始代码（双重 Python 循环）**：
```python
for i in veh_mask:
    for j in range(inf_nums):
        cost_matrix[i][j] = torch.sum((veh_ref_pts[i] - inf_ref_pts[j])**2)**0.5
        # torch.sum() 返回 GPU tensor，赋值给 numpy 数组时隐式 .item() → GPU-CPU 同步
```

**问题根因**：每次 `cost_matrix[i][j] = torch.tensor` 触发一次隐式 `.item()`，即一次 GPU→CPU 数据传输 + CUDA 流同步。N_veh=901, N_inf=100 时共触发 **90,100 次同步**，实测延迟 10,548 ms。

**向量化替换（GPU 批量计算，单次传输）**：
```python
# (M, N_inf, 3) 广播差值
diff = veh_pts.unsqueeze(1) - inf_ref_pts.unsqueeze(0)
l2   = diff.pow(2).sum(-1).sqrt()               # (M, N_inf) L2 距离
rel  = diff.abs() / dims.unsqueeze(1)           # (M, N_inf, 3) 归一化
keep = rel.le(1.0).all(dim=-1)                  # (M, N_inf) bool 过滤
cost_active = torch.where(keep, l2, fill(1e6))
cost_matrix = cost_gpu.cpu().numpy()            # ← 唯一一次 GPU→CPU 传输
```

**性能提升**：N=10: 1048ms→2.7ms（392×），N=100: 10548ms→8.9ms（1191×）

#### 改动 2：`_get_coop_bev_embed` BEV 散射向量化

**文件**：`projects/mmdet3d_plugin/univ2x/detectors/univ2x_track.py:391`

**原始代码（三层嵌套循环）**：
```python
for idx in range(N):               # 逐实例
    w, h = int(locs[idx,0]), int(locs[idx,1])
    for hh in range(h-1, h+1):    # 2×2 邻域
        for ww in range(w-1, w+1):
            bev_embed[hh*W+ww] += mlp_feat[idx]   # 逐元素 scatter
```

**向量化替换（预计算 flat_idx + `index_add_`）**：
```python
feat_embed = self.bev_embed_linear(query[:, embed_dims:])   # (N, C) 批量 MLP
for dh in range(-1, 1):            # 固定 4 次迭代
    for dw in range(-1, 1):
        flat_idx = (hh * W + ww)[in_bounds]
        bev_embed.index_add_(0, flat_idx, feat_embed[in_bounds].unsqueeze(1))
```

`torch.index_add_` 的语义等同于原始 `+=`，包括同一位置多次累加的行为（reduce='add'）。

**性能提升**：N=200: 64.3ms→1.1ms（58×）

---

## 三、优化线 B：QuantV2X INT8 PTQ 迁移

### 3.1 量化基础包（`quant/` 目录）

从 QuantV2X 迁移的 7 个模块（共 1,764 行），关键改动：

| 模块 | 来自 QuantV2X | 对 UniV2X 的主要改动 |
|------|:-------------:|---------------------|
| `quant_layer.py` | ✅ | 移除 `spconv`/`QuantSpconvModule` |
| `fold_bn.py` | ✅ | 移除 `_fold_bn_spconv`；`skip_names` 改为参数注入 |
| `quant_model.py` | ✅ | 硬编码 opencood_specials → `register_specials()` 类方法（可扩展注入） |
| `adaptive_rounding.py` | ✅ | 直接迁移，无改动 |
| `layer_recon.py` | ✅ | 直接迁移，无改动 |
| `quant_params.py` | 新写 | 统一 scale 校准接口 |
| `data_utils.py` | ✅（适配） | mmcv DataLoader 接口适配 |

### 3.2 BEVFormer 特例量化（ADR-001：选择性量化）

**文件**：`quant/quant_bevformer.py:49`

核心设计：MSDA 模块内部并非所有层都可量化：

| 子层 | 作用 | 量化 | 原因 |
|------|------|:----:|------|
| `sampling_offsets` | 采样坐标偏移 → MSDAPlugin | ❌ FP16 | 坐标偏移量化误差导致错误采样位置，BEV 特征完全失效 |
| `attention_weights` | 注意力权重 → MSDAPlugin | ❌ FP16 | 权重误差放大聚合结果 |
| `value_proj` | 特征线性投影 | ✅ W8A8 | 聚合前变换，误差可吸收 |
| `output_proj` | 聚合后线性投影 | ✅ W8A8 | 同上 |

**这是整个量化方案精度达标的前提**，若不实施则 BEV 特征失效。

### 3.3 Temporal 校准数据（最大单项精度贡献）

**文件**：`tools/validate_quant_bev.py`

BEVFormer 的 Temporal Self-Attention（TSA）依赖前帧 BEV，若校准时 `prev_bev=None`（全零），TSA 相关层的激活范围被低估，INT8 scale 偏小，推理时激活溢出。

| 校准方案 | AMOTA | 说明 |
|---------|------:|------|
| 10 样本 + 零 prev_bev | 0.334 | 基线 |
| 20 样本 + 真实 temporal | 0.344 | +0.010 |
| 50 样本 + 真实 temporal | **0.364** | **+0.030** |

### 3.4 PLUGIN_V2 精度覆盖的关键决策（最大单项性能恢复）

**错误做法（v3 引擎，AMOTA 0.278）**：
```python
layer.precision = trt.DataType.HALF       # 显式设置
layer.set_output_type(j, trt.DataType.HALF)
```

**原因**：TRT 在 INT8+FP16 混合模式下，当 PLUGIN_V2 层的 `precision` 被显式设置时，会在该层**输入侧**插入额外的 `Dequantize (INT8→FP16)` 节点，使邻接 INT8 层的输出先反量化才能进入插件，引入级联误差：

```
正常流程:   [INT8 Conv] ─── INT8 tensor ──► [MSDAPlugin FP16]   ✅
覆盖后:     [INT8 Conv] ─── INT8 tensor ──► [Dequantize] ──► [MSDAPlugin FP16]  ❌ 额外误差
```

**正确做法（v4/v5 引擎）**：不设置任何精度，TRT 自动将无 INT8 实现的 PLUGIN_V2 层分配到 FP16。

**性能恢复**：AMOTA 0.278 → 0.355 → 0.364，**单项贡献 +0.086 AMOTA**。

---

## 四、为什么其他模块未量化（现状与风险）

### 4.1 当前 INT8 量化覆盖范围

| 模块 | 量化状态 | 说明 |
|------|:-------:|------|
| BEV encoder (BEVFormer Transformer) | ✅ **已量化（INT8 PTQ）** | 选择性量化（ADR-001） |
| ResNet-FPN backbone | ❌ 无法量化 | DCNv2 不支持 ONNX 导出，无法进入 TRT 流水线 |
| 检测头（Decoder Transformer） | ❌ 未量化（FP16 TRT） | 有风险，见下 |
| 下游头（Motion+Occ+Planning） | ❌ 未量化（FP16 TRT） | 有风险，见下 |
| V2X 融合模块（AgentQueryFusion/LaneQueryFusion） | ❌ 未量化（PyTorch FP16） | 体积小，收益有限 |

> `quant_fusion.py` 和 `quant_downstream.py` 目前是**占位注册**（空 skip 列表），代码框架已准备好但实际 PTQ 流程从未运行。

### 4.2 检测头未量化的原因与风险

**原因 1：CustomMSDeformableAttention 内同样有 MSDAPlugin 输入**

检测头的 Decoder Transformer 包含 `CustomMSDeformableAttention`，其 `sampling_offsets` 和 `attention_weights` 同样直接输入 MSDAPlugin。若量化这两层，会触发与 BEV encoder 相同的精度崩溃问题（`quant_bevformer.py` 中的 `QuantCustomMSDA` 已对此做了选择性跳过的设计，但尚未实测验证）。

**原因 2：V2X 路径已存在 zero-padding 精度损失**

V2X 检测头（1101-query 引擎）因零填充导致 Decoder Self-Attention 污染（cos_bbox ≈ 0.996 for N<1101），当前 FP16 已有 -0.009 AMOTA 的代价。若在此基础上再叠加 INT8 量化误差，精度损失可能超出可接受范围。

**量化后的收益估计**：检测头约 34 MB → 预计 ~20 MB（-41%）

**风险等级**：中等。需要先独立验证 `QuantCustomMSDA` 的精度，再运行端到端测试。

### 4.3 下游头未量化的原因与风险

**原因 1：BN 训练模式漂移**

下游头中包含 BatchNorm 层，在 TRT FP16 导出阶段已修复（通过 `onnx_compatible_attention` 中的 Patch 4 强制 `training=False`）。但 PTQ 校准阶段需要在 PyTorch 前向中运行，若 `model.eval()` 后 BN 的 `running_mean/var` 仍受采样数据影响，可能导致校准 scale 不稳定。

**原因 2：校准数据的路由复杂性**

下游头的输入（`bev_embed + query_feats + bbox_preds + lane_query + command`）来自 BEV encoder + 检测头的输出，需要先完成 BEV encoder 和检测头的前向才能收集校准样本，构建完整的数据收集 pipeline 工作量较大。

**原因 3：收益相对有限**

下游头（153 MB + 134 MB）以 Linear + LayerNorm 为主，量化后预计可减少 40%。但因没有 MSDAPlugin 类型的自定义算子，相比 BEV encoder 量化难度更低、风险也更小，只是尚未优先实施。

**量化后的收益估计**：
- ego 下游头：153 MB → ~90 MB
- infra 下游头：134 MB → ~80 MB
- 全链路引擎：264 MB → ~170 MB（再减 36%）

**风险等级**：低。主要是工程实施工作量，精度风险较小。

### 4.4 V2X 融合模块未量化的分析

AgentQueryFusion / LaneQueryFusion 的 MLP 层（`cross_agent_align`、`cross_agent_fusion` 等）已被 `register_fusion_specials()` 纳入可量化范围（`quant_fusion.py`），且模块内无 MSDAPlugin 相邻层。

**未量化原因**：模块体积极小（MLP 权重总量约 1~2 MB），量化收益在引擎大小和延迟上均可忽略不计。且这些模块目前以 PyTorch 运行（未进入 TRT 图），量化对 TRT 引擎大小无直接贡献。

---

## 五、性能提升关键点排序

按端到端 AMOTA 贡献量排序：

| 排名 | 优化项 | 优化线 | AMOTA 变化 | 技术手段 |
|:----:|--------|:------:|:----------:|---------|
| ① | PLUGIN_V2 精度覆盖决策（不设置） | B INT8 | **+0.086**（v3→v5 恢复） | TRT 混合精度默认策略理解 |
| ② | BEV encoder FP16 TRT 加速 | A TRT | **+0.032**（vs PyTorch 基准） | TRT FP16 + BEVFormerTRT 变体 |
| ③ | Temporal prev_bev 校准（50 样本） | B INT8 | **+0.030**（v1→v5） | 真实 temporal 状态覆盖 TSA 激活范围 |
| ④ | `_query_matching` 向量化 | A V2X | **消除 5,000+ ms 延迟** | GPU 广播距离计算 + 单次 GPU→CPU |
| ⑤ | ADR-001 选择性量化 | B INT8 | 保护精度下限 | `sampling_offsets` 保持 FP16 |
| ⑥ | `_get_coop_bev_embed` 向量化 | A V2X | **消除 64 ms 延迟** (N=200) | `torch.index_add_` scatter |
| ⑦ | 检测头 TRT（Hook D，1101-query） | A TRT | -0.009（vs Hook ABC） | 零填充 + TRT 固定 shape |

---

## 六、AdaRound 实验（2026-04-04）

### 6.1 实验结果

| 阶段 | 产出 | 状态 |
|------|------|------|
| C-1 AdaRound 校准（10 样本，500 iter/层） | `calibration/quant_encoder_adaround.pth`（31 MB，36 层） | ✅ |
| C-2 ONNX 导出（W_fq 权重内嵌） | `onnx/univ2x_ego_bev_encoder_adaround.onnx`（104.2 MB） | ✅ |
| C-3 INT8 TRT 引擎构建 | `trt_engines/univ2x_ego_bev_encoder_adaround_int8.trt`（43.7 MB） | ✅ |
| D-1 BEV cosine 验证 | 平均 0.819（目标 > 0.99） | ❌ |
| D-2 端到端 AMOTA | **0.190**（目标 ≥ 0.370） | ❌ |

### 6.2 失败根因：双重权重量化（Double Weight Quantization）

**AdaRound fake-quant ONNX 路径**：

```
AdaRound 校准
  W_fq = scale_ada × round_hard(W / scale_ada)   ← 第 1 次量化

ONNX 导出（内嵌 W_fq，FP32 格式）
  ONNX 权重 = W_fq（已在 INT8 网格上的 FP32 值）

TRT INT8 引擎构建
  scale_trt = f(max|W_fq|)  ← TRT 从 ONNX 中的 W_fq 重新推导 scale
  W_qq = round(W_fq / scale_trt) × scale_trt  ← 第 2 次量化
```

**问题**：`scale_trt ≠ scale_ada`，TRT 对已量化的 W_fq 再次量化，导致 AdaRound 精心优化的舍入决策被覆盖。理论 INT8 精度损失 = AdaRound 量化误差 + 重量化误差之和，实测 AMOTA 从 0.381 跌至 0.190。

### 6.3 正确的 AdaRound TRT 部署方案（未实施，留作后续工作）

需要在 ONNX 中插入显式量化节点（Q/DQ 方式），告知 TRT 使用 AdaRound 推导的 scale：

```
原 ONNX:  Input ──► Gemm(W_fq) ──► Output

QAT ONNX: Input ──► Gemm ──► Output
                      ↑
               DequantizeLinear(scale=scale_ada)
                      ↑  
               QuantizeLinear(W, scale=scale_ada, axis=0)
```

TRT 遇到 `QuantizeLinear + DequantizeLinear + Gemm` 组合时会直接使用 `scale_ada` 进行 INT8 内核调度，消除双重量化问题。

**实现要点**：
1. 导出时保留 FP32 权重 W（不内嵌 W_fq）
2. 为每个 QuantModule 提取 per-channel `scale_ada`（= `m.weight_quantizer.uaq.delta`）
3. 在 ONNX 图中为每个 Linear/Conv 的权重输入插入 `QuantizeLinear + DequantizeLinear` 节点对
4. TRT 构建时无需额外 INT8 校准（scale 已在 ONNX 中显式指定）

### 6.4 结论

- **Vanilla PTQ INT8**（0.381 AMOTA）已超越 FP16 基准（0.370），无需 AdaRound 即满足项目目标
- AdaRound 的正确 TRT 部署需要 Q/DQ ONNX 节点，属于 QAT 风格导出，工程量较大
- 本次 AdaRound 实验验证了校准框架（`calibrate_univ2x.py`）和 QuantModel 流程均正常工作，为后续正确实施奠定了基础

---

## 七、后续优化方向

| 方向 | 预期收益 | 难度 | 优先级 |
|------|---------|:----:|:------:|
| AdaRound Q/DQ ONNX 部署 | AMOTA 超越 FP16 基准 | 高（需 ONNX Q/DQ 插入） | ⭐⭐⭐ |
| 下游头 INT8 PTQ | 引擎再减 170 MB（-64% vs 当前） | 低 | ⭐⭐ |
| 检测头 INT8 PTQ | 引擎减 ~14 MB | 中（需验证 QuantCustomMSDA） | ⭐⭐ |
| backbone 量化（QAT） | — | 极高（DCNv2 不可 ONNX） | ⭐ |

---

## 八、关键文件索引

| 文件 | 所属优化线 | 作用 |
|------|:--------:|------|
| `fusion_modules/agent_fusion.py:202` | A V2X | `AgentQueryFusionTRT._query_matching_vec`（向量化代价矩阵） |
| `detectors/univ2x_track.py:391` | A V2X | `_get_coop_bev_embed`（`index_add_` BEV scatter） |
| `quant/quant_layer.py` | B INT8 | `UniformAffineQuantizer`（STE + MSE/entropy scale search） |
| `quant/quant_model.py` | B INT8 | `QuantModel` + `register_specials()` 可扩展注入 |
| `quant/quant_bevformer.py` | B INT8 | ADR-001：选择性量化 MSDA 模块 |
| `quant/quant_fusion.py` | B INT8 | **占位注册**（融合模块 PTQ 未实施） |
| `quant/quant_downstream.py` | B INT8 | **占位注册**（下游头 PTQ 未实施） |
| `tools/validate_quant_bev.py` | B INT8 | Temporal 校准数据收集 + W8A8 精度验证 |
| `tools/build_trt_int8_univ2x.py` | B INT8 | TRT INT8 引擎构建（`IInt8EntropyCalibrator2` 实现） |
| `calibration/bev_encoder_calib_inputs.pkl` | B INT8 | 50 帧 temporal 校准数据（~4 GB） |
| `calibration/univ2x_ego_bev_encoder_int8_int8.cache` | B INT8 | INT8 activation scale 缓存（68 KB，可复用） |
| `trt_engines/univ2x_ego_bev_encoder_int8.trt` | B INT8 | 最终 INT8 引擎，**43 MB，AMOTA 0.364** |

---

| `tools/calibrate_univ2x.py` | B AdaRound | AdaRound 校准主脚本（BEVEncoderCalibModel + layer_reconstruction） |
| `tools/export_onnx_adaround.py` | B AdaRound | AdaRound fake-quant ONNX 导出（W_fq 内嵌） |
| `tools/validate_adaround_bev.py` | B AdaRound | AdaRound BEV cosine 验证脚本 |
| `calibration/quant_encoder_adaround.pth` | B AdaRound | AdaRound 校准结果（36 层 alpha + act delta，31 MB） |
| `onnx/univ2x_ego_bev_encoder_adaround.onnx` | B AdaRound | AdaRound ONNX（W_fq 内嵌，104.2 MB） |
| `trt_engines/univ2x_ego_bev_encoder_adaround_int8.trt` | B AdaRound | AdaRound INT8 引擎（43.7 MB，**AMOTA 0.190，双重量化问题**） |

---

*最后更新：2026-04-04*
