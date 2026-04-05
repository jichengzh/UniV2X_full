# AdaRound Agent 简报库

> PM 派遣 agent 前先从此文件复制对应 Brief，
> 作为 agent prompt 的上下文前缀。

---

## BRIEF-A1：EXPLORE — 读取 AdaRound 基础设施 API

**角色**：代码探索员  
**任务**：读取以下4个文件，提取关键API，输出简洁摘要  
**上下文预算**：< 6k tokens  

**需读取的文件**（按顺序，read全文）：
1. `projects/mmdet3d_plugin/univ2x/quant/adaptive_rounding.py`
2. `projects/mmdet3d_plugin/univ2x/quant/layer_recon.py`
3. `projects/mmdet3d_plugin/univ2x/quant/quant_bevformer.py`（只读前200行）
4. `tools/calibrate_univ2x.py`（只读前150行，了解现有接口）

**输出要求**（写入 `docs/quant/adaround/outputs/phase_A/api_summary.md`）：
```
# API 摘要

## adaptive_rounding.py
- AdaRoundQuantizer 的构造参数
- init_delta_from_weights() 签名
- 重要的 forward 模式（soft/hard）

## layer_recon.py
- layer_reconstruction() 函数签名和参数含义
- 支持的 block 类型（单层 or block）
- 是否需要 dataloader or tensor list 作为校准数据输入

## quant_bevformer.py（前200行）
- 已注册的 QuantBlock 类名列表
- 哪些层被标记为 skip（即 FP16）

## calibrate_univ2x.py（前150行）
- 主流程：scale calibration 如何触发
- 保存权重的接口
```

**严格不做**：不修改任何文件，不执行代码，只读取并总结。

---

## BRIEF-A2：ARCH — AdaRound 集成方案设计

**角色**：系统架构师  
**任务**：基于 EXPLORE 输出，设计 AdaRound 集成到 UniV2X BEV encoder 的具体方案  
**上下文预算**：< 5k tokens  

**必读文件**：
- `docs/quant/adaround/outputs/phase_A/api_summary.md`（EXPLORE 输出）
- `docs/quant/adaround/DECISIONS.md`（已有决策）

**输出要求**（写入 `docs/quant/adaround/outputs/phase_A/arch_decision.md`）：
```
# AdaRound 集成方案

## 1. calibrate_univ2x_adaround.py 伪代码（50行以内）
## 2. layer_reconstruction() 调用方式
   - 传入 block 类型（QuantBEVFormerLayer or QuantMSDA3D）
   - 校准数据格式（取自 calib_inputs pkl 的 feat tensors）
   - 是否需要 temporal prev_bev tracking
## 3. 待更新的 ADR-004（填写具体答案）
## 4. 已知风险（最多3条）
```

**必须回答**：
1. `layer_recon.py` 的 `layer_reconstruction()` 是否可以直接用，还是需要适配 BEVFormerLayerTRT 的 forward 签名？
2. 50帧校准数据 `bev_encoder_calib_inputs.pkl` 的格式是否与 layer_recon 的 dataloader 参数兼容？
3. AdaRound 每层 iter=5000 × 约40层 ≈ 200K iters，GPU内存是否可接受？

---

## BRIEF-B2：PY-REV — calibrate_univ2x_adaround.py 代码审查

**角色**：Python代码审查员  
**任务**：审查新实现的 AdaRound 校准脚本  
**上下文预算**：< 6k tokens  

**必读文件**：
- `tools/calibrate_univ2x_adaround.py`（全文）
- `docs/quant/adaround/outputs/phase_A/arch_decision.md`（设计规范）

**检查重点**（按优先级）：
1. **CRITICAL**：AdaRoundQuantizer 初始化是否在 scale calibration 之后？（顺序错误会导致收敛失败）
2. **CRITICAL**：layer_reconstruction() 调用时 model 是否处于正确的 quant_state（weight_quant=True, act_quant=False）？
3. **HIGH**：temporal prev_bev 在 reconstruction forward 中是否正确初始化（不能为 None/zeros）？
4. **HIGH**：保存权重时是否只保存 BEV encoder 的参数（不保存整个 model）？
5. **MEDIUM**：GPU 内存是否在每层 recon 后释放（torch.cuda.empty_cache）？

**输出要求**（写入 `docs/quant/adaround/outputs/phase_B/review.md`）：
严重问题列表 + 每个问题的建议修复方式（< 500字）

---

## BRIEF-D1：BUILD — PyTorch/TRT 错误修复（on-call）

**角色**：构建工程师（按需触发）  
**任务**：修复运行 calibrate_univ2x_adaround.py 或 build_trt_int8_univ2x.py 时遇到的错误  
**触发条件**：PM 遇到 RuntimeError / CUDA OOM / TRT 构建失败时派遣  

**PM 派遣时必须提供**：
- 完整错误 traceback（不截断）
- 当前运行的命令
- 最近修改的代码片段（< 30行）

**处理原则**：
- 最小化改动（不重构，只修错误）
- 优先考虑 CUDA OOM → 减少 batch_size 或 per-layer sequential 处理
- TRT plugin 错误 → 检查 INT64→INT32 patch 是否仍有效
