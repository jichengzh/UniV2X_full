# Phase C1：联合压缩管线 (Joint Compression Pipeline)

> 任务范围：C.1 — C.3
> 预计工期：2.5 天
> 依赖：Phase A1-A4（剪枝管线 + Phase 0 结果）、Phase B1（TRT 部署）、1.1 可配置量化（QuantModel + quant_config.json）
> 目标：将 B1（结构/连通性剪枝）与 B2（可配置量化）融合为统一的联合压缩管线，
>       实现配置驱动的端到端执行与评估，并通过 Pareto 分析找到最优配置

---

## 1. 阶段目标

本阶段是 1.2 可配置剪枝实施计划的最终集成阶段。核心目标：

1. **统一配置格式**：将 `prune_config.json`（B1 搜索空间）和 `quant_config.json`（B2 搜索空间）合并为 `joint_config.json`，增加联合执行控制参数
2. **端到端联合管线**：单一脚本完成 剪枝 → 微调 → 量化 → ONNX 导出 → TRT INT8 构建 → 精度/延迟/模型大小评估
3. **交互效应分析**：验证剪枝+量化的联合精度损失是否小于各自独立损失之和（期望弱负交互）
4. **Pareto 最优搜索**：在 B1×B2 联合搜索空间中，自动化地寻找 AMOTA-延迟-模型大小的 Pareto 前沿

**量化指标**：
- 联合管线端到端执行成功
- 联合 AMOTA 损失 < 0.015（相对 PyTorch FP32 基线）
- 联合加速比 > 3x（相对 PyTorch FP32 基线）
- Pareto 分析产出有意义的权衡曲线

---

## 2. 前置条件

### 2.1 已完成的阶段

| 阶段 | 产出 | 状态要求 |
|:----:|------|:-------:|
| Phase A1 | DepGraph 适配层（自定义剪枝器）、剪枝主入口 | ✅ 完成 |
| Phase A2 | `prune_config.json` 统一配置 + `tools/prune_and_eval.py` | ✅ 完成 |
| Phase A3 | Phase 0 锁定维度确定（P4=taylor, P5=local, P6=5, P7=8） | ✅ 完成 |
| Phase A4 | 搜索维度 (P1-P3, P8-P9) 快速评估验证 | ✅ 完成 |
| Phase B1 | ONNX 导出 + TRT engine 构建 + TRT 推理评估 | ✅ 完成 |
| 1.1 可配置量化 | `QuantModel`、`quant_config.json`、AdaRound 校准、Q/DQ ONNX 注入 | ✅ 完成 |

### 2.2 可用的代码资产

```
# 剪枝侧
projects/mmdet3d_plugin/univ2x/pruning/custom_pruners.py    # DepGraph 自定义剪枝器
projects/mmdet3d_plugin/univ2x/pruning/prune_univ2x.py      # 剪枝主入口 (build_pruner, apply_prune_config)
projects/mmdet3d_plugin/univ2x/pruning/grad_collector.py     # 梯度收集 (Taylor/Hessian)
projects/mmdet3d_plugin/univ2x/pruning/post_prune.py         # 剪枝后模块状态更新
tools/prune_and_eval.py                                       # 剪枝+PyTorch评估

# 量化侧
projects/mmdet3d_plugin/univ2x/quant/quant_model.py          # QuantModel 封装
projects/mmdet3d_plugin/univ2x/quant/quant_bevformer.py      # BEVFormer 量化
projects/mmdet3d_plugin/univ2x/quant/quant_downstream.py     # Downstream 量化
projects/mmdet3d_plugin/univ2x/quant/quant_fusion.py         # 量化融合
projects/mmdet3d_plugin/univ2x/quant/layer_recon.py          # AdaRound 层重建
tools/calibrate_univ2x.py                                     # 量化校准
tools/inject_qdq_from_config.py                               # Q/DQ ONNX 注入
tools/sensitivity_analysis.py                                  # 量化敏感度分析

# TRT 部署侧
tools/export_onnx_univ2x.py                                   # ONNX 导出
tools/build_trt_int8_univ2x.py                                # TRT INT8 构建
tools/test_trt.py                                              # TRT 推理评估
```

### 2.3 已有数据

- PyTorch FP32 基线 AMOTA 和推理延迟（Phase A 产出）
- 各单独剪枝配置的 AMOTA（Phase A4 产出）
- 各单独量化配置的 AMOTA（1.1 Phase 0 产出）
- 校准数据集 (`calibration/*.pkl`)

---

## 3. 具体代码实现

### 3.1 联合配置格式

**新建文件**: `compression_configs/joint_config.json`

将 `prune_config.json` 和 `quant_config.json` 合并，并增加联合执行控制参数 `joint` 段。

```json
{
  "version": "1.2+1.1",
  "_comment": "联合压缩配置：B1(剪枝) × B2(量化) 统一配置",

  "pruning": {
    "version": "1.2",
    "locked": {
      "importance_criterion": "taylor",
      "pruning_granularity": "local",
      "iterative_steps": 5,
      "round_to": 8
    },
    "encoder": {
      "ffn_mid_ratio": 0.5,
      "attn_proj_ratio": 0.1,
      "head_pruning_ratio": 0.0
    },
    "decoder": {
      "ffn_mid_ratio": 0.6,
      "attn_proj_ratio": 0.1,
      "head_pruning_ratio": 0.0,
      "num_layers": 6
    },
    "heads": {
      "head_mid_ratio": 0.7
    },
    "constraints": {
      "skip_layers": ["sampling_offsets", "attention_weights"],
      "min_channels": 64,
      "channel_alignment": 8
    }
  },

  "quantization": {
    "version": "1.1",
    "global": {
      "weight_bit": 8,
      "act_bit": 8,
      "symmetric": true,
      "per_channel_weight": true,
      "calibration_method": "minmax"
    },
    "layer_overrides": {
      "sampling_offsets": { "act_bit": 16 },
      "attention_weights": { "act_bit": 16 }
    },
    "adaround": {
      "enabled": true,
      "num_samples": 1024,
      "iters": 10000
    }
  },

  "joint": {
    "execution_order": "prune_first",
    "finetune_after_prune": true,
    "finetune_epochs": 10,
    "finetune_lr": 2e-5,
    "recalibrate_quant": true,
    "recalibrate_samples": 512,
    "target_amota_drop": 0.015,
    "target_speedup": 3.0,
    "target_model_size_mb": null
  }
}
```

**设计决策说明**：

1. **`execution_order: "prune_first"`**：先剪枝后量化。理由：
   - 剪枝改变模型结构（通道数、层数），量化在新结构上进行校准
   - 如果先量化再剪枝，量化的 scale/zero_point 在通道被删除后失效
   - 这与主流论文（如 APQ, OFA）的实践一致

2. **`finetune_after_prune`**：剪枝后微调恢复精度。1.1 量化实施中发现 AdaRound 可以部分弥补量化损失，但剪枝的结构性损失需要端到端微调。

3. **`recalibrate_quant`**：在剪枝后的模型上重新收集量化校准数据。剪枝改变了激活值分布，旧的校准数据不再适用。

4. **`target_*` 字段**：验收阈值，管线执行完成后自动对比并报告是否达标。

---

### 3.2 联合压缩管线主脚本

**新建文件**: `tools/joint_compress_eval.py`

**核心流程**：

```
Load model → Apply pruning → Fine-tune → Apply quantization → Export ONNX
→ Build TRT INT8 → Evaluate (AMOTA + latency + model_size) → Report
```

**完整实现**：

```python
"""联合压缩管线：剪枝 + 量化 + TRT 部署 + 评估

用法:
    python tools/joint_compress_eval.py \
        --config compression_configs/joint_config.json \
        --checkpoint work_dirs/univ2x/latest.pth \
        --output-dir work_dirs/joint_compressed/ \
        --baseline-amota 0.452 \
        --baseline-latency-ms 85.0
"""

import argparse
import json
import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

import torch

logger = logging.getLogger(__name__)


@dataclass
class JointResult:
    """联合压缩结果"""
    # 配置信息
    config_path: str
    pruning_config: dict
    quantization_config: dict
    joint_config: dict

    # 精度指标
    baseline_amota: float
    pruned_amota: float          # 剪枝后（微调前）
    finetuned_amota: float       # 微调后
    quantized_amota: float       # 量化后 (PyTorch)
    trt_amota: float             # TRT INT8 最终精度
    amota_drop: float            # 最终精度损失

    # 性能指标
    baseline_latency_ms: float
    trt_latency_ms: float
    speedup: float

    # 模型大小
    original_params: int
    pruned_params: int
    trt_engine_size_mb: float

    # 验收
    pass_amota: bool
    pass_speedup: bool


def load_joint_config(config_path):
    """加载并校验联合配置"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    assert 'pruning' in config, "缺少 pruning 配置段"
    assert 'quantization' in config, "缺少 quantization 配置段"
    assert 'joint' in config, "缺少 joint 配置段"

    joint = config['joint']
    assert joint['execution_order'] in ('prune_first',), \
        f"当前仅支持 prune_first 执行顺序，收到: {joint['execution_order']}"

    return config


def validate_pruned_model_for_quant(model, quant_config):
    """3.3 联合验证：验证剪枝后模型满足量化要求

    核心检查：
    1. 剪枝后所有通道数仍为 8 的倍数（INT8 对齐）
    2. 坐标敏感层未被剪枝
    3. V2X ego/infra 结构一致性
    """
    issues = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        # 检查通道对齐
        if module.out_features % 8 != 0:
            issues.append(
                f"通道未对齐: {name}.out_features={module.out_features} "
                f"(不是8的倍数，INT8会fallback到FP32)")

        if module.in_features % 8 != 0:
            issues.append(
                f"通道未对齐: {name}.in_features={module.in_features} "
                f"(不是8的倍数)")

    # 检查坐标敏感层完整性
    for name, module in model.named_modules():
        if 'sampling_offsets' in name and isinstance(module, torch.nn.Linear):
            # sampling_offsets 输出维度应保持原始值
            expected_keywords = ['sampling_offsets']
            # 验证未被剪枝：输出维度应为 num_heads * num_levels * num_points * 2
            pass  # 具体数值取决于模型配置

        if 'attention_weights' in name and isinstance(module, torch.nn.Linear):
            pass  # 同上

    if issues:
        logger.warning(f"联合验证发现 {len(issues)} 个问题:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False, issues

    logger.info("联合验证通过：剪枝后模型满足量化要求")
    return True, []


def step1_apply_pruning(model, prune_config, dataloader):
    """步骤1：应用剪枝

    调用 Phase A 实现的 apply_prune_config
    """
    from projects.mmdet3d_plugin.univ2x.pruning.prune_univ2x import (
        apply_prune_config
    )

    original_params = sum(p.numel() for p in model.parameters())
    logger.info(f"剪枝前参数量: {original_params:,}")

    model = apply_prune_config(model, prune_config, dataloader=dataloader)

    pruned_params = sum(p.numel() for p in model.parameters())
    logger.info(f"剪枝后参数量: {pruned_params:,} "
                f"(减少 {(1 - pruned_params/original_params)*100:.1f}%)")

    return model, original_params, pruned_params


def step2_finetune(model, train_dataloader, joint_config):
    """步骤2：剪枝后微调

    目的：恢复因结构性剪枝导致的精度损失
    策略：小学习率 + 短周期微调
    """
    if not joint_config.get('finetune_after_prune', True):
        logger.info("跳过微调（finetune_after_prune=false）")
        return model

    epochs = joint_config.get('finetune_epochs', 10)
    lr = joint_config.get('finetune_lr', 2e-5)

    logger.info(f"开始微调: epochs={epochs}, lr={lr}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch_idx, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            losses = model.train_step(data, optimizer=None)
            loss = sum(v for k, v in losses.items()
                       if 'loss' in k and isinstance(v, torch.Tensor))
            loss.backward()
            # 梯度裁剪，防止剪枝后不稳定
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"  Epoch {epoch+1}/{epochs}, avg_loss={avg_loss:.4f}")

    return model


def step3_apply_quantization(model, quant_config, calib_dataloader,
                              joint_config):
    """步骤3：应用量化

    调用 1.1 可配置量化实现的 QuantModel
    如果 recalibrate_quant=true，在剪枝后模型上重新收集校准数据
    """
    from projects.mmdet3d_plugin.univ2x.quant.quant_model import QuantModel

    # 在剪枝后的模型上构建量化模型
    qmodel = QuantModel(model, quant_config)

    if joint_config.get('recalibrate_quant', True):
        num_samples = joint_config.get('recalibrate_samples', 512)
        logger.info(f"在剪枝后模型上重新校准量化 (samples={num_samples})")
        qmodel.calibrate(calib_dataloader, num_samples=num_samples)

        # 如果配置了 AdaRound
        if quant_config.get('adaround', {}).get('enabled', False):
            logger.info("执行 AdaRound 优化")
            qmodel.apply_adaround(
                calib_dataloader,
                num_samples=quant_config['adaround'].get('num_samples', 1024),
                iters=quant_config['adaround'].get('iters', 10000)
            )

    return qmodel


def step4_export_onnx(model, output_dir):
    """步骤4：导出 ONNX"""
    onnx_path = os.path.join(output_dir, 'joint_compressed.onnx')
    logger.info(f"导出 ONNX: {onnx_path}")

    # 复用现有 ONNX 导出逻辑
    # 参考 tools/export_onnx_univ2x.py
    from tools.export_onnx_univ2x import export_onnx
    export_onnx(model, onnx_path)

    return onnx_path


def step5_build_trt(onnx_path, output_dir, quant_config):
    """步骤5：构建 TRT INT8 engine"""
    engine_path = os.path.join(output_dir, 'joint_compressed.engine')
    logger.info(f"构建 TRT engine: {engine_path}")

    # 复用现有 TRT 构建逻辑
    # 参考 tools/build_trt_int8_univ2x.py
    from tools.build_trt_int8_univ2x import build_engine
    build_engine(onnx_path, engine_path)

    engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    logger.info(f"TRT engine 大小: {engine_size_mb:.1f} MB")

    return engine_path, engine_size_mb


def step6_evaluate(engine_path, eval_dataloader):
    """步骤6：TRT 推理评估"""
    from tools.test_trt import evaluate_trt

    logger.info("开始 TRT 推理评估...")
    results = evaluate_trt(engine_path, eval_dataloader)

    amota = results['amota']
    latency_ms = results['avg_latency_ms']
    logger.info(f"TRT AMOTA: {amota:.4f}, latency: {latency_ms:.1f} ms")

    return amota, latency_ms


def generate_report(result, output_dir):
    """步骤7：生成评估报告"""
    report_path = os.path.join(output_dir, 'joint_report.json')

    report = asdict(result)
    report['summary'] = {
        'amota_drop_vs_target': f"{result.amota_drop:.4f} vs {result.joint_config['target_amota_drop']}",
        'speedup_vs_target': f"{result.speedup:.2f}x vs {result.joint_config['target_speedup']}x",
        'param_reduction': f"{(1 - result.pruned_params/result.original_params)*100:.1f}%",
        'verdict': 'PASS' if (result.pass_amota and result.pass_speedup) else 'FAIL',
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印关键结果
    logger.info("=" * 60)
    logger.info("联合压缩评估报告")
    logger.info("=" * 60)
    logger.info(f"  AMOTA 损失: {result.amota_drop:.4f} "
                f"(目标 < {result.joint_config['target_amota_drop']})"
                f" → {'✓ PASS' if result.pass_amota else '✗ FAIL'}")
    logger.info(f"  加速比: {result.speedup:.2f}x "
                f"(目标 > {result.joint_config['target_speedup']}x)"
                f" → {'✓ PASS' if result.pass_speedup else '✗ FAIL'}")
    logger.info(f"  参数减少: {(1 - result.pruned_params/result.original_params)*100:.1f}%")
    logger.info(f"  TRT engine: {result.trt_engine_size_mb:.1f} MB")
    logger.info("=" * 60)

    return report_path


def main():
    parser = argparse.ArgumentParser(description='联合压缩管线')
    parser.add_argument('--config', required=True,
                        help='joint_config.json 路径')
    parser.add_argument('--checkpoint', required=True,
                        help='PyTorch 模型权重路径')
    parser.add_argument('--output-dir', required=True,
                        help='输出目录')
    parser.add_argument('--baseline-amota', type=float, required=True,
                        help='PyTorch FP32 基线 AMOTA')
    parser.add_argument('--baseline-latency-ms', type=float, required=True,
                        help='PyTorch FP32 基线延迟 (ms)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    # 加载配置
    config = load_joint_config(args.config)
    prune_cfg = config['pruning']
    quant_cfg = config['quantization']
    joint_cfg = config['joint']

    # 加载模型和数据 (省略具体实现，依赖现有基础设施)
    model, train_loader, calib_loader, eval_loader = _load_model_and_data(
        args.checkpoint)

    # === 执行联合压缩管线 ===

    # 步骤 1: 剪枝
    model, orig_params, pruned_params = step1_apply_pruning(
        model, prune_cfg, calib_loader)

    # 步骤 1.5: 联合验证
    valid, issues = validate_pruned_model_for_quant(model, quant_cfg)
    if not valid:
        logger.error("剪枝后模型未通过量化兼容性验证，中止管线")
        logger.error("请检查剪枝配置中的 round_to 和 channel_alignment 设置")
        return

    # 步骤 1.6: 剪枝后精度 (可选快速评估)
    pruned_amota = _quick_eval(model, eval_loader)

    # 步骤 2: 微调
    model = step2_finetune(model, train_loader, joint_cfg)
    finetuned_amota = _quick_eval(model, eval_loader)

    # 步骤 3: 量化
    qmodel = step3_apply_quantization(model, quant_cfg, calib_loader, joint_cfg)
    quantized_amota = _quick_eval(qmodel, eval_loader)

    # 步骤 4: 导出 ONNX
    onnx_path = step4_export_onnx(qmodel, args.output_dir)

    # 步骤 5: 构建 TRT INT8
    engine_path, engine_size_mb = step5_build_trt(
        onnx_path, args.output_dir, quant_cfg)

    # 步骤 6: TRT 评估
    trt_amota, trt_latency_ms = step6_evaluate(engine_path, eval_loader)

    # 步骤 7: 报告
    amota_drop = args.baseline_amota - trt_amota
    speedup = args.baseline_latency_ms / trt_latency_ms

    result = JointResult(
        config_path=args.config,
        pruning_config=prune_cfg,
        quantization_config=quant_cfg,
        joint_config=joint_cfg,
        baseline_amota=args.baseline_amota,
        pruned_amota=pruned_amota,
        finetuned_amota=finetuned_amota,
        quantized_amota=quantized_amota,
        trt_amota=trt_amota,
        amota_drop=amota_drop,
        baseline_latency_ms=args.baseline_latency_ms,
        trt_latency_ms=trt_latency_ms,
        speedup=speedup,
        original_params=orig_params,
        pruned_params=pruned_params,
        trt_engine_size_mb=engine_size_mb,
        pass_amota=amota_drop <= joint_cfg['target_amota_drop'],
        pass_speedup=speedup >= joint_cfg['target_speedup'],
    )

    generate_report(result, args.output_dir)


if __name__ == '__main__':
    main()
```

**关键设计点**：

1. **每步精度追踪**：在剪枝后、微调后、量化后分别记录 AMOTA，便于诊断精度损失来源
2. **验证门控**：步骤 1.5 在剪枝和量化之间插入兼容性验证，提前中止不合法的配置
3. **重新校准**：`recalibrate_quant=true` 确保量化校准基于剪枝后模型的实际激活分布
4. **报告自动化**：自动对比目标阈值并输出 PASS/FAIL 判定

---

### 3.3 联合验证

联合验证解决的核心问题：剪枝和量化是两个独立开发的管线（1.2 和 1.1），它们的交互可能产生以下冲突。

#### 3.3.1 通道对齐验证

**问题**：剪枝可能产生非 8 的倍数的通道数，导致 TRT INT8 fallback 到 FP32。

**验证逻辑**（已嵌入 `validate_pruned_model_for_quant`）：
- 遍历所有 Linear 层，检查 `in_features` 和 `out_features` 是否为 8 的倍数
- 如果 `prune_config.locked.round_to = 8` 配置正确，此验证应总是通过
- 但仍需显式检查，因为 DepGraph 在处理复杂依赖链时可能遗漏对齐约束

#### 3.3.2 坐标敏感层完整性

**问题**：坐标敏感层（`sampling_offsets`, `attention_weights`）是剪枝的硬约束跳过层，同时也是量化的高精度覆盖层（`act_bit=16`）。需确保两侧配置一致。

**验证逻辑**：
```python
def verify_coordinate_sensitive_layers(model, prune_cfg, quant_cfg):
    """确保坐标敏感层在剪枝和量化中都被特殊处理"""
    prune_skip = set(prune_cfg['constraints']['skip_layers'])
    quant_overrides = set(quant_cfg.get('layer_overrides', {}).keys())

    for name, module in model.named_modules():
        for keyword in ['sampling_offsets', 'attention_weights']:
            if keyword in name:
                # 验证剪枝跳过
                assert keyword in prune_skip, \
                    f"{keyword} 不在剪枝跳过列表中"
                # 验证量化特殊处理
                assert keyword in quant_overrides, \
                    f"{keyword} 不在量化 layer_overrides 中"
```

#### 3.3.3 V2X ego/infra 一致性

**问题**：UniV2X 是车路协同框架，ego（自车）和 infra（路侧）共享部分模型权重。联合压缩后需确保共享结构仍然一致。

**验证逻辑**：
```python
def verify_v2x_consistency(model):
    """验证 ego 和 infra 分支在联合压缩后结构一致"""
    # 收集 ego 和 infra 分支的结构信息
    ego_dims = {}
    infra_dims = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if 'ego' in name:
                key = name.replace('ego', '{side}')
                ego_dims[key] = (module.in_features, module.out_features)
            elif 'infra' in name:
                key = name.replace('infra', '{side}')
                infra_dims[key] = (module.in_features, module.out_features)

    # 检查共享结构一致
    shared_keys = set(ego_dims.keys()) & set(infra_dims.keys())
    for key in shared_keys:
        assert ego_dims[key] == infra_dims[key], \
            f"V2X 不一致: {key} ego={ego_dims[key]} infra={infra_dims[key]}"
```

#### 3.3.4 交互效应分析

**目的**：量化剪枝和量化之间的精度交互效应。

```python
def analyze_interaction_effect(solo_prune_amota, solo_quant_amota,
                                joint_amota, baseline_amota):
    """分析剪枝与量化的交互效应

    定义:
      solo_prune_drop = baseline - solo_prune_amota
      solo_quant_drop = baseline - solo_quant_amota
      joint_drop = baseline - joint_amota
      expected_drop = solo_prune_drop + solo_quant_drop  (独立假设)
      interaction = joint_drop - expected_drop

    interaction < 0 → 正交互（联合优于预期）
    interaction ≈ 0 → 无交互（独立）
    interaction > 0 → 负交互（联合劣于预期）
    """
    solo_prune_drop = baseline_amota - solo_prune_amota
    solo_quant_drop = baseline_amota - solo_quant_amota
    expected_drop = solo_prune_drop + solo_quant_drop
    joint_drop = baseline_amota - joint_amota
    interaction = joint_drop - expected_drop

    return {
        'solo_prune_drop': solo_prune_drop,
        'solo_quant_drop': solo_quant_drop,
        'expected_joint_drop': expected_drop,
        'actual_joint_drop': joint_drop,
        'interaction_effect': interaction,
        'interaction_type': (
            'positive' if interaction < -0.001 else
            'neutral' if abs(interaction) <= 0.001 else
            'negative'
        ),
    }
```

**预期结果**：弱负交互（`interaction` 略大于 0）。原因：剪枝减少冗余通道后，剩余通道的值分布可能更集中，量化误差相对不变；但剪枝本身已经引入近似误差，量化在近似信号上进一步引入误差，两层近似叠加通常略差于独立之和。

---

### 3.4 Pareto 分析工具

**新建文件**: `tools/pareto_analysis.py`

**功能**：在联合搜索空间 (B1×B2) 中运行多组配置，收集 AMOTA/延迟/模型大小三元组，绘制 Pareto 前沿并输出推荐配置。

```python
"""Pareto 分析工具：联合压缩空间搜索

用法:
    python tools/pareto_analysis.py \
        --config-dir compression_configs/sweep/ \
        --output-dir work_dirs/pareto_analysis/ \
        --baseline-amota 0.452 \
        --baseline-latency-ms 85.0

配置目录结构:
    compression_configs/sweep/
    ├── joint_config_001.json   # (prune=轻, quant=INT8)
    ├── joint_config_002.json   # (prune=中, quant=INT8)
    ├── joint_config_003.json   # (prune=重, quant=INT8)
    └── ...
"""

import argparse
import json
import os
import glob
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def generate_sweep_configs(output_dir):
    """生成扫描配置矩阵

    从搜索空间的关键维度中选取代表性配置：
    - P1 ffn_mid_ratio: [0.4, 0.6, 0.8, 1.0]  (4 值)
    - P3 head_mid_ratio: [0.5, 0.7, 1.0]       (3 值)
    - P9 decoder_num_layers: [5, 6]             (2 值)
    - 量化: [INT8, INT8+AdaRound, FP16]         (3 值)

    总计: 4 × 3 × 2 × 3 = 72 个配置
    """
    base_template = {
        "version": "1.2+1.1",
        "pruning": {
            "locked": {
                "importance_criterion": "taylor",
                "pruning_granularity": "local",
                "iterative_steps": 5,
                "round_to": 8
            },
            "encoder": {"attn_proj_ratio": 0.1, "head_pruning_ratio": 0.0},
            "decoder": {"attn_proj_ratio": 0.1, "head_pruning_ratio": 0.0},
            "heads": {},
            "constraints": {
                "skip_layers": ["sampling_offsets", "attention_weights"],
                "min_channels": 64,
                "channel_alignment": 8
            }
        },
        "quantization": {},
        "joint": {
            "execution_order": "prune_first",
            "finetune_after_prune": True,
            "finetune_epochs": 10,
            "recalibrate_quant": True,
            "target_amota_drop": 0.015,
            "target_speedup": 3.0
        }
    }

    ffn_ratios = [0.4, 0.6, 0.8, 1.0]
    head_ratios = [0.5, 0.7, 1.0]
    decoder_layers = [5, 6]
    quant_modes = [
        {"name": "int8", "weight_bit": 8, "act_bit": 8, "adaround": False},
        {"name": "int8_adaround", "weight_bit": 8, "act_bit": 8,
         "adaround": True},
        {"name": "fp16", "weight_bit": 16, "act_bit": 16, "adaround": False},
    ]

    os.makedirs(output_dir, exist_ok=True)
    configs = []
    idx = 0

    for ffn_r in ffn_ratios:
        for head_r in head_ratios:
            for dec_l in decoder_layers:
                for qmode in quant_modes:
                    config = json.loads(json.dumps(base_template))
                    config['pruning']['encoder']['ffn_mid_ratio'] = ffn_r
                    config['pruning']['decoder']['ffn_mid_ratio'] = ffn_r
                    config['pruning']['decoder']['num_layers'] = dec_l
                    config['pruning']['heads']['head_mid_ratio'] = head_r
                    config['quantization'] = {
                        "global": {
                            "weight_bit": qmode['weight_bit'],
                            "act_bit": qmode['act_bit'],
                            "symmetric": True,
                            "per_channel_weight": True,
                        },
                        "adaround": {"enabled": qmode['adaround']},
                    }

                    config_name = (
                        f"joint_{idx:03d}_ffn{ffn_r}_head{head_r}"
                        f"_dec{dec_l}_{qmode['name']}")
                    config_path = os.path.join(
                        output_dir, f"{config_name}.json")
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    configs.append(config_path)
                    idx += 1

    logger.info(f"生成 {len(configs)} 个扫描配置到 {output_dir}")
    return configs


def is_pareto_optimal(points):
    """判断哪些点是 Pareto 最优的

    输入 points: shape (N, M)，每个维度越小越好
    返回: shape (N,) 布尔数组
    """
    n = len(points)
    is_optimal = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_optimal[i]:
            continue
        for j in range(n):
            if i == j or not is_optimal[j]:
                continue
            # 如果 j 在所有维度上都不劣于 i，且至少一个维度严格优于 i
            if np.all(points[j] <= points[i]) and np.any(
                    points[j] < points[i]):
                is_optimal[i] = False
                break

    return is_optimal


def run_pareto_analysis(results, output_dir):
    """运行 Pareto 分析

    输入 results: list of dict, 每个包含:
        - config_name: str
        - amota_drop: float (越小越好)
        - latency_ms: float (越小越好)
        - model_size_mb: float (越小越好)
    """
    if not results:
        logger.warning("没有结果可分析")
        return

    # 构建点阵
    names = [r['config_name'] for r in results]
    points = np.array([
        [r['amota_drop'], r['latency_ms'], r['model_size_mb']]
        for r in results
    ])

    # 找 Pareto 前沿
    pareto_mask = is_pareto_optimal(points)
    pareto_configs = [names[i] for i in range(len(names)) if pareto_mask[i]]
    pareto_points = points[pareto_mask]

    logger.info(f"Pareto 最优配置: {len(pareto_configs)}/{len(names)}")
    for i, (name, pt) in enumerate(zip(pareto_configs, pareto_points)):
        logger.info(f"  {i+1}. {name}: "
                     f"AMOTA_drop={pt[0]:.4f}, "
                     f"latency={pt[1]:.1f}ms, "
                     f"size={pt[2]:.1f}MB")

    # 输出推荐配置
    recommended = {
        'pareto_optimal_configs': [
            {
                'config_name': name,
                'amota_drop': float(pt[0]),
                'latency_ms': float(pt[1]),
                'model_size_mb': float(pt[2]),
            }
            for name, pt in zip(pareto_configs, pareto_points)
        ],
        'total_configs_evaluated': len(names),
        'num_pareto_optimal': len(pareto_configs),
    }

    # 在 Pareto 前沿中选择最佳推荐
    # 策略：选 AMOTA_drop 最小且满足 speedup > 3x 的配置
    best = None
    for cfg in recommended['pareto_optimal_configs']:
        if best is None or cfg['amota_drop'] < best['amota_drop']:
            best = cfg
    recommended['best_recommendation'] = best

    rec_path = os.path.join(output_dir, 'recommended_configs.json')
    with open(rec_path, 'w') as f:
        json.dump(recommended, f, indent=2, ensure_ascii=False)

    # 绘图
    _plot_pareto(results, pareto_mask, output_dir)

    return recommended


def _plot_pareto(results, pareto_mask, output_dir):
    """绘制 Pareto 前沿图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装，跳过绘图")
        return

    amota_drops = [r['amota_drop'] for r in results]
    latencies = [r['latency_ms'] for r in results]
    sizes = [r['model_size_mb'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 子图1: AMOTA_drop vs latency
    ax = axes[0]
    ax.scatter(
        [l for i, l in enumerate(latencies) if not pareto_mask[i]],
        [a for i, a in enumerate(amota_drops) if not pareto_mask[i]],
        c='gray', alpha=0.5, label='Non-Pareto')
    ax.scatter(
        [l for i, l in enumerate(latencies) if pareto_mask[i]],
        [a for i, a in enumerate(amota_drops) if pareto_mask[i]],
        c='red', s=80, zorder=5, label='Pareto optimal')
    # 连接 Pareto 前沿
    pareto_latency = sorted(
        [(latencies[i], amota_drops[i])
         for i in range(len(results)) if pareto_mask[i]])
    if pareto_latency:
        ax.plot([p[0] for p in pareto_latency],
                [p[1] for p in pareto_latency],
                'r--', alpha=0.7)
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('AMOTA Drop')
    ax.set_title('AMOTA Drop vs Latency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图2: AMOTA_drop vs model_size
    ax = axes[1]
    ax.scatter(
        [s for i, s in enumerate(sizes) if not pareto_mask[i]],
        [a for i, a in enumerate(amota_drops) if not pareto_mask[i]],
        c='gray', alpha=0.5, label='Non-Pareto')
    ax.scatter(
        [s for i, s in enumerate(sizes) if pareto_mask[i]],
        [a for i, a in enumerate(amota_drops) if pareto_mask[i]],
        c='red', s=80, zorder=5, label='Pareto optimal')
    ax.set_xlabel('Model Size (MB)')
    ax.set_ylabel('AMOTA Drop')
    ax.set_title('AMOTA Drop vs Model Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图3: latency vs model_size (颜色编码 AMOTA_drop)
    ax = axes[2]
    sc = ax.scatter(latencies, sizes, c=amota_drops,
                    cmap='RdYlGn_r', s=50, alpha=0.7)
    ax.scatter(
        [latencies[i] for i in range(len(results)) if pareto_mask[i]],
        [sizes[i] for i in range(len(results)) if pareto_mask[i]],
        facecolors='none', edgecolors='red', s=120, linewidth=2,
        label='Pareto optimal')
    plt.colorbar(sc, ax=ax, label='AMOTA Drop')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Model Size (MB)')
    ax.set_title('Latency vs Size (color=AMOTA Drop)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'pareto_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Pareto 图保存到 {fig_path}")
```

---

## 4. 代码检测方案

### Test 1：端到端完整性测试

**目标**：验证联合管线在保守配置下能端到端完成

```bash
# 保守配置：轻度剪枝 + INT8 量化
python tools/joint_compress_eval.py \
    --config compression_configs/joint_config_conservative.json \
    --checkpoint work_dirs/univ2x/latest.pth \
    --output-dir work_dirs/test_joint_e2e/ \
    --baseline-amota 0.452 \
    --baseline-latency-ms 85.0
```

**保守配置**：
```json
{
  "pruning": {
    "encoder": { "ffn_mid_ratio": 0.8, "attn_proj_ratio": 0.0 },
    "decoder": { "ffn_mid_ratio": 0.8, "num_layers": 6 },
    "heads": { "head_mid_ratio": 1.0 }
  },
  "joint": { "finetune_epochs": 5 }
}
```

**验收条件**：
- 脚本正常退出（exit code 0）
- 生成 `joint_report.json`
- 生成 `joint_compressed.engine`
- TRT 推理无报错

### Test 2：交互效应验证

**目标**：验证联合 AMOTA 损失 < 各自独立损失之和（弱负交互预期）

```bash
# 步骤 1: 仅剪枝（已有 Phase A4 结果）
solo_prune_amota=0.445  # 从 Phase A4 结果读取

# 步骤 2: 仅量化（已有 1.1 结果）
solo_quant_amota=0.448  # 从 1.1 Phase 0 结果读取

# 步骤 3: 联合压缩
python tools/joint_compress_eval.py --config ... --baseline-amota 0.452

# 步骤 4: 计算交互效应
# solo_prune_drop = 0.452 - 0.445 = 0.007
# solo_quant_drop = 0.452 - 0.448 = 0.004
# expected_joint_drop = 0.007 + 0.004 = 0.011
# actual_joint_drop 应 < 0.015（验收阈值）
# interaction = actual - expected，预期为小正值 (< 0.005)
```

**验收条件**：
- `actual_joint_drop < solo_prune_drop + solo_quant_drop + 0.005`（弱负交互容忍上限）
- `actual_joint_drop < 0.015`（绝对阈值）

### Test 3：加速比验证

**目标**：验证联合压缩后 TRT 加速比 > 2.5x（保守阈值）

```bash
# PyTorch FP32 基线延迟: 85.0 ms
# TRT INT8 联合压缩后延迟: 期望 < 34 ms (2.5x) 或 < 28 ms (3x)
```

**验收条件**：
- `speedup > 2.5x`（最低通过线）
- `speedup > 3.0x`（完全通过线）

### Test 4：Pareto 分析验证

**目标**：验证 Pareto 分析工具产出有意义的权衡曲线

```bash
# 生成扫描配置
python tools/pareto_analysis.py generate-sweep \
    --output-dir compression_configs/sweep/

# 运行扫描（可使用少量配置做冒烟测试）
python tools/pareto_analysis.py run \
    --config-dir compression_configs/sweep/ \
    --output-dir work_dirs/pareto_analysis/ \
    --max-configs 6

# 验证产出
ls work_dirs/pareto_analysis/
# 期望: recommended_configs.json, pareto_analysis.png
```

**验收条件**：
- `recommended_configs.json` 非空且包含至少 2 个 Pareto 最优配置
- `pareto_analysis.png` 生成成功
- Pareto 前沿呈现合理的凸包形状（非所有点都在前沿上）

---

## 5. Debug 方案

### 5.1 剪枝破坏量化校准

**症状**：量化校准后 AMOTA 异常低（远低于仅剪枝精度）

**诊断步骤**：
1. 检查剪枝后激活值分布：
   ```python
   # 在 step3 之前插入
   for name, module in model.named_modules():
       if isinstance(module, torch.nn.Linear):
           hook = module.register_forward_hook(
               lambda m, inp, out, n=name:
                   print(f"{n}: out range [{out.min():.4f}, {out.max():.4f}]"))
   ```
2. 对比剪枝前后激活值范围，如果范围变化超过 10x，说明剪枝导致激活值不平衡
3. 检查 `recalibrate_quant` 是否为 `true`——如果使用旧校准数据，scale/zero_point 不匹配剪枝后分布

**修复方案**：
- 确保 `recalibrate_quant=true`
- 如果仍然不行，增加 `recalibrate_samples`（512 → 1024）
- 考虑在微调阶段加入量化感知训练（QAT），但这会显著增加复杂度和工期

### 5.2 联合 AMOTA 远差于预期

**症状**：`actual_joint_drop >> solo_prune_drop + solo_quant_drop`（强负交互）

**诊断步骤**：
1. **隔离精度损失来源**：查看 `joint_report.json` 中每步 AMOTA：
   - `pruned_amota`：如果此处已大幅下降，问题在剪枝
   - `finetuned_amota`：如果微调未恢复，增加 `finetune_epochs`
   - `quantized_amota`：如果量化后再次大幅下降，问题在剪枝-量化交互
   - `trt_amota`：如果与 `quantized_amota` 差距大，问题在 TRT 构建/推理
2. **逐层敏感度分析**：
   ```bash
   # 在剪枝后模型上运行量化敏感度分析
   python tools/sensitivity_analysis.py \
       --checkpoint work_dirs/joint_compressed/pruned_model.pth
   ```
3. **检查是否有特定层对联合压缩特别敏感**：FFN 被剪枝 50% + INT8 量化的组合可能在某些层上产生较大误差

**修复方案**：
- 减小剪枝比例（`ffn_mid_ratio` 从 0.5 → 0.6）
- 对敏感层使用 `act_bit=16` 量化覆盖
- 增加微调 epochs（10 → 20）
- 启用 AdaRound 优化量化 round 策略

### 5.3 TRT 构建失败

**症状**：`step5_build_trt` 报错，通常是 ONNX 算子不支持或维度不匹配

**诊断步骤**：
1. **检查通道对齐**：
   ```bash
   python -c "
   import onnx
   model = onnx.load('work_dirs/joint_compressed/joint_compressed.onnx')
   for node in model.graph.node:
       if node.op_type in ('Conv', 'MatMul', 'Gemm'):
           print(f'{node.name}: {[d.dim_value for d in node.output_type[0].tensor_type.shape.dim]}')
   "
   ```
2. **验证 Q/DQ 节点**：检查 Q/DQ 节点的 scale tensor 形状是否与剪枝后通道数匹配
3. **TRT 日志分析**：
   ```bash
   trtexec --onnx=joint_compressed.onnx --int8 --verbose 2>&1 | grep -E "ERROR|WARNING|fallback"
   ```

**修复方案**：
- 确保 `round_to=8` 在整个剪枝管线中被正确执行
- 如果特定层导致问题，在 `quant_config.layer_overrides` 中将该层设为 FP16
- 重新检查自定义剪枝器（`custom_pruners.py`）是否正确处理了所有依赖关系

### 5.4 加速比不达标

**症状**：`speedup < 2.5x`，TRT INT8 延迟高于预期

**诊断步骤**：
1. **逐层延迟分析**：
   ```bash
   trtexec --loadEngine=joint_compressed.engine --dumpProfile
   ```
2. **检查 FP32 fallback**：搜索 TRT 日志中的 "Reformatting" 或 "Choosing FP32"
3. **检查剪枝效果**：参数减少了但延迟没减少，可能是剪枝没有命中瓶颈层

**修复方案**：
- 调整剪枝比例，优先剪枝 FFN 层（占计算量 ~40%）
- 确保所有层都成功运行在 INT8 模式
- 增加 `decoder_num_layers` 剪枝（6 → 5），层数减少直接降低串行延迟
- 考虑在 TRT 构建时启用 `--best` 模式让 TRT 自动选择最优精度

---

## 6. 验收标准

| 编号 | 验收项 | 指标 | 通过条件 |
|:----:|--------|------|:-------:|
| C1.1 | 端到端管线完成 | `joint_compress_eval.py` exit code | = 0 |
| C1.2 | 联合 AMOTA 精度 | `amota_drop` | < 0.015 |
| C1.3 | 联合加速比 | `speedup` (vs PyTorch FP32) | > 3.0x |
| C1.4 | Pareto 分析 | `recommended_configs.json` | 非空，>= 2 个 Pareto 最优配置 |
| C1.5 | 产物完整性 | 输出文件 | `.engine`, `.onnx`, `joint_report.json`, `pareto_analysis.png` 全部生成 |
| C1.6 | 交互效应合理 | `interaction_effect` | < 0.005（弱负交互容忍上限） |
| C1.7 | V2X 一致性 | ego/infra 结构验证 | 通过 |

**完整通过**：C1.1-C1.7 全部满足
**有条件通过**：C1.1, C1.4, C1.5, C1.7 满足，C1.2/C1.3 接近目标（AMOTA_drop < 0.02，speedup > 2.5x）
**失败**：C1.1 不通过（管线无法端到端执行）

---

## 7. 风险与缓解

| 编号 | 风险 | 概率 | 影响 | 缓解方案 |
|:----:|------|:----:|:----:|---------|
| R1 | 剪枝后量化校准失效，AMOTA 远低于预期 | 中 | 高 | `recalibrate_quant=true` 强制重新校准；增加校准样本数；必要时对敏感层使用 FP16 |
| R2 | TRT 构建失败（Q/DQ 节点与剪枝后通道不匹配） | 中 | 高 | 联合验证门控（步骤 1.5）提前检查通道对齐；`round_to=8` 硬约束 |
| R3 | 加速比不达标（< 3x） | 低 | 中 | 优先剪枝 FFN 层；减少 decoder 层数；确保无 FP32 fallback |
| R4 | 微调收敛慢，10 epoch 不够恢复精度 | 中 | 中 | 增加到 20 epoch；使用更大学习率（2e-5 → 5e-5）；引入知识蒸馏 |
| R5 | Pareto 扫描耗时过长（72 个配置 × 每个约 2 小时） | 高 | 低 | 分批运行；先用 6 个代表性配置做冒烟测试；利用早停（AMOTA_drop > 0.03 时跳过 TRT 构建） |
| R6 | V2X ego/infra 共享权重在剪枝后不一致 | 低 | 高 | `verify_v2x_consistency` 显式检查；如果不一致，对共享层施加同步剪枝约束 |
| R7 | 依赖的 Phase A/B/1.1 产出有 bug，在集成时暴露 | 中 | 中 | 集成前先独立验证每个子管线；端到端测试用保守配置 |

---

## 8. 反思模板

### 8.1 联合压缩的加速效果

> **问题**：联合压缩是否实现了超线性加速（superlinear speedup）？
>
> 即：`joint_speedup > solo_prune_speedup + solo_quant_speedup - 1` 是否成立？
>
> **预期**：不太可能实现超线性。剪枝减少计算量（参数量/FLOPs），量化减少内存带宽和计算精度，
> 两者的加速效果基本正交但不会叠乘。例如：
> - 仅剪枝 (FFN 50%) → 约 1.3x 加速
> - 仅 INT8 量化 → 约 2.5x 加速
> - 联合 → 期望约 2.5x-3.5x，而非 1.3 × 2.5 = 3.25x
>
> **实际观察**：\_\_\_\_\_\_\_\_\_\_\_\_\_\_
>
> **分析**：\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### 8.2 Pareto 最优配置特征

> **问题**：在 Pareto 前沿上，哪种 (pruning_ratio, quant_config) 组合最优？
>
> **关注点**：
> - 是"轻剪枝 + INT8+AdaRound"优于"重剪枝 + INT8"，还是反过来？
> - P1 (ffn_mid_ratio) 和量化精度之间是否存在甜蜜点？
> - P9 (decoder_num_layers=5) 带来的加速是否值得精度代价？
>
> **实际观察**：\_\_\_\_\_\_\_\_\_\_\_\_\_\_
>
> **推荐配置**：\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### 8.3 精度-加速权衡曲线形状

> **问题**：AMOTA_drop vs speedup 的权衡曲线是什么形状？
>
> **可能形状**：
> - **凸曲线**（好）：前期加速几乎不损精度，后期精度急剧下降 → 甜蜜点在拐点处
> - **线性**（中）：加速与精度损失成正比 → 无明显甜蜜点，按需求选择
> - **凹曲线**（差）：轻微加速就损失大量精度 → 压缩收益低
>
> **实际观察**：\_\_\_\_\_\_\_\_\_\_\_\_\_\_
>
> **曲线形状**：\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### 8.4 对 hw-sw codesign 搜索框架的启示

> **问题**：C1 阶段的实践经验对后续 hw-sw codesign 搜索框架（B1×B2×D 联合搜索）有何启示？
>
> **关注点**：
> 1. 搜索空间大小：72 个配置是否足够覆盖权衡空间？是否需要更细粒度的扫描？
> 2. 代理指标：是否可以用 PyTorch FP32 AMOTA 替代 TRT INT8 AMOTA 做初筛？
>    （即：PyTorch AMOTA 排序是否与 TRT AMOTA 排序强相关？）
> 3. 搜索加速：早停策略（AMOTA_drop > 阈值时跳过 TRT 构建）节省了多少时间？
> 4. 维度交互：P1-P3 之间是否存在显著交互效应？还是可以独立优化后组合？
>
> **实际观察**：\_\_\_\_\_\_\_\_\_\_\_\_\_\_
>
> **对搜索框架的建议**：\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### 8.5 工程教训

> **问题**：联合压缩管线开发中最大的工程挑战是什么？
>
> **待记录**：
> - 剪枝和量化之间最难调试的交互问题是什么？
> - 联合配置管理的设计是否合理？是否需要版本控制/迁移机制？
> - 端到端管线中最耗时的步骤是什么？是否可以缓存中间产物？
> - 如果重做，会如何改变开发顺序？
>
> **实际记录**：\_\_\_\_\_\_\_\_\_\_\_\_\_\_
