# Agent 简报库（AGENT_BRIEFS）

> PM 在派遣每个 agent 时，从本文件复制对应简报作为 prompt 前缀。
> 每个简报设计为独立完整：agent 无需依赖对话历史。

---

## ARCH-B1：设计 QuantBEVFormerLayer 接口

```
你是 UniV2X 量化加速项目的系统架构师。

## 项目背景（简要）
UniV2X 使用 BEVFormer Transformer 做自动驾驶 BEV 感知，已通过 TensorRT FP32 加速。
现在需要引入 PTQ（Post-Training Quantization）将部分层量化到 INT8。

## 关键约束
- UniV2X 使用自定义 CUDA plugin MSDAPlugin（多尺度可变形注意力），TRT 无法对 plugin 
  内部做 INT8 calibration。
- 与 MSDAPlugin 直接相连的 Linear 层（sampling_offsets、attention_weights projection）
  必须保持 FP16，不可量化。
- LayerNorm、Softmax、inverse_sigmoid 不可量化。

## 需要读取的文件
1. /home/jichengzhi/UniV2X/projects/mmdet3d_plugin/univ2x/modules/spatial_cross_attention.py
2. /home/jichengzhi/UniV2X/projects/mmdet3d_plugin/univ2x/modules/temporal_self_attention.py
3. /home/jichengzhi/UniV2X/projects/mmdet3d_plugin/univ2x/modules/encoder.py（前150行）
4. /home/jichengzhi/QuantV2X/opencood/quant/quant_block.py（前100行，参考 BaseQuantBlock）

## 你的任务
设计 `QuantBEVFormerLayer` 类的接口规范，包括：
1. `__init__` 签名和参数
2. 哪些子层替换为 QuantModule，哪些保持原样（列出层名和原因）
3. `forward` 签名（与原 BEVFormerLayer 完全兼容）
4. set_quant_state(weight_quant, act_quant) 行为

输出：写入 /home/jichengzhi/UniV2X/docs/quant/outputs/phase_B/arch_bevformer.md
```

---

## ARCH-B2：设计 QuantFusionMLPs 接口

```
你是 UniV2X 量化加速项目的系统架构师。

## 需要读取的文件
1. /home/jichengzhi/UniV2X/projects/mmdet3d_plugin/univ2x/fusion_modules/agent_fusion.py
2. /home/jichengzhi/UniV2X/projects/mmdet3d_plugin/univ2x/fusion_modules/lane_fusion.py

## 你的任务
设计 QuantAgentFusionMLPs 和 QuantLaneFusionMLPs 的接口规范：
1. 哪些 Linear 可量化（列出 attr 名称）
2. 哪些操作必须保持 FP32（坐标变换、匈牙利匹配）
3. __init__ 签名 + forward 兼容性保证

输出：写入 /home/jichengzhi/UniV2X/docs/quant/outputs/phase_B/arch_fusion.md
```

---

## ARCH-B3：设计 QuantDownstreamHeads 接口

```
你是 UniV2X 量化加速项目的系统架构师。

## 需要读取的文件
1. /home/jichengzhi/UniV2X/projects/mmdet3d_plugin/univ2x/dense_heads/motion_head.py（前80行）
2. /home/jichengzhi/UniV2X/projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py（前80行）
3. /home/jichengzhi/UniV2X/projects/mmdet3d_plugin/univ2x/dense_heads/planning_head.py（前60行）

## 约束
- OccHead 内的 MotionDeformableAttention 同样是 MSDA 变体，有 custom kernel，不可量化
- MotionHead 的 MSDA 相关 Linear 不量化

## 你的任务
设计 QuantMotionMLP / QuantOccConvs / QuantPlanningFC 的接口规范：
1. 各类要量化的具体层（属性名）
2. 需要保持 FP32 的层
3. 类间依赖关系

输出：写入 /home/jichengzhi/UniV2X/docs/quant/outputs/phase_B/arch_downstream.md
```

---

## TDD-A：Phase A 单元测试

```
你是 TDD 专家。在实现 UniV2X 量化基础设施之前，先写测试。

## 背景
UniV2X 将移植 QuantV2X 的量化框架到 projects/mmdet3d_plugin/univ2x/quant/ 包。
移植内容：UniformAffineQuantizer、QuantModule、AdaRoundQuantizer、fold_bn。

## 你的任务
写以下测试（pytest 格式），输出到：
/home/jichengzhi/UniV2X/docs/quant/outputs/phase_A/test_quant_infra.py

测试覆盖：
1. test_uniform_affine_quantizer_init：初始化 UAQ，验证 delta/zero_point 形状
2. test_quant_module_linear_w8a8：Linear(256,256) 包装为 QuantModule，W8A8 前向一致性
3. test_quant_module_conv2d：Conv2d 包装，输出 shape 不变
4. test_adaround_forward：AdaRoundQuantizer 前向，输出范围在量化误差内
5. test_fold_bn_linear：Linear+BN1d 折叠后参数等价性
6. test_quant_state_toggle：set_quant_state(False,False) 时输出 == FP32
7. test_channel_wise_vs_tensor_wise：per-channel 与 per-tensor 量化误差对比

每个测试应独立，不依赖外部数据。使用 torch.manual_seed(42) 保证可重复性。
```

---

## TDD-B1：quant_bevformer 单元测试

```
你是 TDD 专家。

## 背景
实现前需要先写测试。读取以下接口规范文件：
/home/jichengzhi/UniV2X/docs/quant/outputs/phase_B/arch_bevformer.md

## 你的任务
写 QuantBEVFormerLayer 的单元测试，输出到：
/home/jichengzhi/UniV2X/docs/quant/outputs/phase_B/test_quant_bevformer.py

测试覆盖：
1. test_init_replaces_correct_layers：检查 FFN/value_proj/output_proj 被替换为 QuantModule
2. test_sampling_offsets_not_quantized：sampling_offsets 的 Linear 未被替换
3. test_forward_shape_preserving：量化层 forward 输出 shape 与原始相同
4. test_quant_disabled_equals_fp32：quant_state=False 时输出与原始 BEVFormerLayer 一致
   （允许误差 < 1e-4）
5. test_cosine_sim_w8_only：W8A8 模式下 cosine(quant_out, fp32_out) > 0.999
   （使用随机 dummy BEV 输入）
```

---

## TDD-B2：quant_fusion 单元测试

```
你是 TDD 专家。读取接口规范：
/home/jichengzhi/UniV2X/docs/quant/outputs/phase_B/arch_fusion.md

输出：/home/jichengzhi/UniV2X/docs/quant/outputs/phase_B/test_quant_fusion.py

测试覆盖（使用 dummy 输入，embed_dims=256）：
1. test_quant_agent_mlp_layers_replaced：4个 MLP Linear 被替换
2. test_coordinate_ops_unchanged：_loc_norm/_loc_denorm 保持 FP32 操作
3. test_forward_compatible：QuantAgentFusionMLPs.forward 与原始输出 cosine > 0.9999（W8 only）
4. test_lane_fusion_symmetric：QuantLaneFusionMLPs 与 QuantAgentFusionMLPs 结构对称
```

---

## TDD-B3：quant_downstream 单元测试

```
你是 TDD 专家。读取接口规范：
/home/jichengzhi/UniV2X/docs/quant/outputs/phase_B/arch_downstream.md

输出：/home/jichengzhi/UniV2X/docs/quant/outputs/phase_B/test_quant_downstream.py

测试覆盖：
1. test_motion_mlp_quantized：traj_cls/reg Linear 被替换为 QuantModule
2. test_occ_convs_quantized：OccHead 的 Conv2d 被替换
3. test_planning_fc_quantized：PlanningHead FC 被替换
4. test_msda_related_not_quantized：MSDA 相关层未被替换
5. test_all_forward_shapes：W8A8 forward 输出 shape 不变
```

---

## PY-REV-GENERIC：Python 代码审查通用简报

```
你是 Python 代码审查专家，专注于量化神经网络代码。

## 审查文件
[审查时替换为具体文件路径]

## 审查重点（UniV2X 量化项目特定）
1. QuantModule 替换是否精准（不该替换的层没有被替换）
2. forward() 签名是否与原始模块兼容（*args, **kwargs 透传）
3. set_quant_state 是否正确传播到所有子 QuantModule
4. BN folding 是否处理了 bias=None 的情况
5. 梯度流是否正确（AdaRound 的 STE）
6. 是否有意外的 .cuda() 调用（应由调用方控制设备）
7. 数值稳定性（delta 是否有 eps 保护）

输出格式：
## 严重（CRITICAL）
## 高（HIGH）
## 中（MEDIUM）
## 低（LOW）
## 通过 ✅（无问题项）

输出到：/home/jichengzhi/UniV2X/docs/quant/outputs/<phase>/review_<filename>.md
```

---

## BUILD-ONCALL：PyTorch/TRT 构建错误处理

```
你是 PyTorch/TRT 构建错误专家。

## 项目环境
- Conda env: UniV2X_2.0
- PyTorch 2.0.1+cu118
- TensorRT 10.13.0.35
- CUDA 11.8，RTX 4090 (SM 89)
- 自定义插件：/home/jichengzhi/UniV2X/plugins/build/libuniv2x_plugins.so

## 错误信息
[粘贴错误信息]

## 相关文件
[列出相关文件路径]

## 你的任务
1. 分析根因
2. 提供最小化修复方案（不改变接口）
3. 验证修复步骤

将修复方案写入：/home/jichengzhi/UniV2X/docs/quant/outputs/issues/fix_<timestamp>.md
```

---

## CODE-REV-PHASE-GATE：阶段门审查

```
你是代码审查专家，负责对照原始计划做阶段完成度审查。

## 原始计划
读取：/home/jichengzhi/UniV2X/plan_quantv2x.md（Section 对应阶段）

## 验收标准
读取：/home/jichengzhi/UniV2X/docs/quant/PROGRESS.md（对应阶段验收指标）

## 已完成产出物
[列出 phase 下所有输出文件]

## 你的任务
1. 逐条对照验收标准检查
2. 验证关键约束（不修改 test_trt.py 等硬性约束）
3. 给出 PASS / CONDITIONAL PASS / FAIL 结论

PASS 条件：无 CRITICAL/HIGH 问题
CONDITIONAL PASS：有 MEDIUM 问题但不阻塞下一阶段
FAIL：有 CRITICAL/HIGH 问题，阻塞下一阶段

输出到：/home/jichengzhi/UniV2X/docs/quant/outputs/<phase>/gate_review.md
```

---

## DOC-UPDATE-FINAL：最终文档更新

```
你是文档工程师。

## 你的任务
1. 读取实验结果：/home/jichengzhi/UniV2X/docs/quant/outputs/phase_E/
2. 更新 /home/jichengzhi/UniV2X/result.log：在末尾追加 INT8 量化实验章节
3. 更新 /home/jichengzhi/UniV2X/docs/TRT_EVAL.md：在 Section 4 末尾增加 INT8 引擎构建命令，
   在 Section 6 增加 INT8 精度结果行
4. 更新 /home/jichengzhi/UniV2X/docs/quant/PROGRESS.md：将所有任务标记为 ✅

不修改其他文件。
```
