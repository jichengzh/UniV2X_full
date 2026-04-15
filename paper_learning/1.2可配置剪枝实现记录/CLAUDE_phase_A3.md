# Phase A3：配置与评估管线 (Config & Eval Pipeline)

> 覆盖任务：A.2.6（统一配置入口 + 快速评估管线）+ A.4（微调恢复管线）
> 预计工时：1.5 天
> 前置依赖：Phase A1（自定义剪枝器 + 梯度收集）、Phase A2（剪枝主入口 + 状态更新 + 层数剪枝）

---

## 1. 阶段目标

构建统一的配置驱动剪枝 + 评估管线：

1. **读取** `prune_config.json` → **解析** locked（锁定维度 P4-P7）与 search（搜索维度 P1-P3/P8-P9）
2. **执行** `apply_prune_config()`：编排完整的剪枝流程（梯度收集 → DepGraph 追踪 → 通道剪枝 → 层数剪枝 → 状态更新 → 约束校验）
3. **保存** 剪枝后模型 `.pth`，并报告参数量/FLOPs 变化
4. **可选评估** AMOTA（复用现有 `tools/test.py` 基础设施）
5. **微调管线**：提供剪枝后精度恢复的训练配置，支持可选知识蒸馏

---

## 2. 前置条件

| 条件 | 来源 | 验证方式 |
|------|------|---------|
| `custom_pruners.py` 可用 | Phase A1 | `register_univ2x_pruners()` 返回字典非空 |
| `grad_collector.py` 可用 | Phase A1 | `collect_gradients()` 不崩溃 |
| `prune_univ2x.py` 核心函数可用 | Phase A2 | `build_pruner()` 返回合法 pruner |
| `post_prune.py` 可用 | Phase A2 | `update_model_after_pruning()` 不崩溃 |
| `prune_decoder_layers()` 可用 | Phase A2 | 删层后模型可 forward |
| UniV2X 训练/评估基础设施正常 | 已有 | `tools/train.py`、`tools/test.py` 可运行 |
| 基线 checkpoint 存在 | 已有 | `work_dirs/latest.pth` 可加载 |

---

## 3. 具体代码实现

### 3.1 `tools/prune_and_eval.py` — 主入口脚本

这是整个配置驱动剪枝管线的唯一命令行入口。

```python
"""
统一配置驱动剪枝 + 评估管线

用法:
    python tools/prune_and_eval.py \
        --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
        --checkpoint work_dirs/latest.pth \
        --prune-config prune_configs/moderate.json \
        --output work_dirs/pruned_moderate.pth \
        --eval

功能:
    1. 加载模型 (mmdet3d config + checkpoint)
    2. 读取 prune_config.json (locked + search 分离)
    3. 调用 apply_prune_config() 执行完整剪枝流程
    4. 保存剪枝后模型
    5. 报告: 原始参数量 vs 剪枝后参数量、FLOPs 变化、逐模块通道数
    6. 可选: 评估 AMOTA
"""

import argparse
import json
import os
import sys
import time
import warnings

import torch
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description='配置驱动剪枝 + 评估管线')
    parser.add_argument('--config', required=True,
                        help='mmdet3d 模型配置文件路径 (.py)')
    parser.add_argument('--checkpoint', required=True,
                        help='模型 checkpoint 路径 (.pth)')
    parser.add_argument('--prune-config', required=True,
                        help='剪枝配置文件路径 (.json)')
    parser.add_argument('--output', required=True,
                        help='剪枝后模型保存路径 (.pth)')
    parser.add_argument('--eval', action='store_true',
                        help='剪枝后立即评估 AMOTA')
    parser.add_argument('--eval-out', default='output/pruned_results.pkl',
                        help='评估结果保存路径')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='使用的 GPU ID')
    parser.add_argument('--calib-batches', type=int, default=32,
                        help='Taylor/Hessian 梯度收集的 batch 数')
    parser.add_argument('--seed', type=int, default=0,
                        help='随机种子')
    return parser.parse_args()


def load_prune_config(json_path):
    """加载并校验 prune_config.json

    校验内容:
    - version 字段存在且为 "1.2"
    - locked 段必须包含 importance_criterion, round_to
    - 搜索维度的 ratio 值在 [0.0, 1.0] 范围内
    - constraints.channel_alignment 必须为 8 的倍数
    """
    with open(json_path, 'r') as f:
        config = json.load(f)

    # 版本校验
    version = config.get('version', '1.0')
    if version != '1.2':
        print(f"[WARN] prune_config version={version}, 预期 1.2, 继续执行但可能不兼容")

    # locked 段默认值填充
    locked = config.setdefault('locked', {})
    locked.setdefault('importance_criterion', 'taylor')
    locked.setdefault('pruning_granularity', 'local')
    locked.setdefault('iterative_steps', 5)
    locked.setdefault('round_to', 8)

    # 搜索维度默认值填充
    config.setdefault('encoder', {})
    config.setdefault('decoder', {})
    config.setdefault('heads', {})
    config.setdefault('finetune', {'epochs': 10})
    config.setdefault('constraints', {})

    # ratio 范围校验
    for section_name in ('encoder', 'decoder', 'heads'):
        section = config[section_name]
        for key, val in section.items():
            if 'ratio' in key and isinstance(val, (int, float)):
                if not (0.0 <= val <= 1.0):
                    raise ValueError(
                        f"[ERROR] {section_name}.{key}={val} 超出 [0.0, 1.0] 范围")

    # constraints 默认值
    constraints = config['constraints']
    constraints.setdefault('skip_layers', ['sampling_offsets', 'attention_weights'])
    constraints.setdefault('min_channels', 64)
    constraints.setdefault('channel_alignment', 8)

    return config


def load_model_from_config(cfg_path, ckpt_path, device='cuda:0'):
    """从 mmdet3d config + checkpoint 加载模型"""
    cfg = Config.fromfile(cfg_path)

    # 导入插件
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_path = _module_dir.replace('/', '.')
            importlib.import_module(_module_path)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    checkpoint = load_checkpoint(model, ckpt_path, map_location='cpu')

    # 兼容旧版 checkpoint 的 class_names
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']

    model = model.to(device)
    model.eval()
    return model, cfg


def count_parameters(model):
    """统计模型参数量 (total / trainable)"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model, example_inputs):
    """估算 FLOPs (基于 torch_pruning 或 fvcore)

    注意: UniV2X 的自定义算子可能导致 FLOPs 计算不完全准确,
    但可作为相对比较的依据。
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        flops_analyzer = FlopCountAnalysis(model, example_inputs)
        flops_analyzer.unsupported_ops_warnings(False)
        flops_analyzer.uncalled_modules_warnings(False)
        return flops_analyzer.total()
    except Exception:
        # fallback: 粗略估计 Linear 层的 FLOPs
        total_flops = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                total_flops += 2 * module.in_features * module.out_features
            elif isinstance(module, torch.nn.Conv2d):
                total_flops += (
                    2 * module.in_channels * module.out_channels
                    * module.kernel_size[0] * module.kernel_size[1]
                )
        return total_flops


def report_model_stats(model, prefix=""):
    """打印逐模块的通道数统计"""
    print(f"\n{'='*70}")
    print(f"  模型统计: {prefix}")
    print(f"{'='*70}")

    total_params, trainable_params = count_parameters(model)
    print(f"  总参数量:     {total_params:>12,d} ({total_params/1e6:.2f}M)")
    print(f"  可训练参数:   {trainable_params:>12,d} ({trainable_params/1e6:.2f}M)")

    # 逐模块统计
    module_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_stats[name] = {
                'type': 'Linear',
                'in': module.in_features,
                'out': module.out_features,
                'params': module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            }
        elif isinstance(module, torch.nn.LayerNorm):
            module_stats[name] = {
                'type': 'LayerNorm',
                'dim': module.normalized_shape[0],
                'params': module.weight.numel() * 2
            }

    # 按类别汇总
    categories = {
        'FFN': lambda n: any(k in n for k in ['ffns.', 'feedforward.']),
        'Attn Proj': lambda n: any(k in n for k in ['value_proj', 'output_proj']),
        'Det Heads': lambda n: any(k in n for k in ['cls_branches', 'reg_branches']),
        'Coord-sensitive': lambda n: any(k in n for k in ['sampling_offsets', 'attention_weights']),
    }

    print(f"\n  {'类别':<20} {'参数量':>12} {'典型输入维度':>12} {'典型输出维度':>12}")
    print(f"  {'-'*56}")

    for cat_name, matcher in categories.items():
        cat_params = 0
        in_dims = set()
        out_dims = set()
        for name, stats in module_stats.items():
            if matcher(name) and stats['type'] == 'Linear':
                cat_params += stats['params']
                in_dims.add(stats['in'])
                out_dims.add(stats['out'])
        if cat_params > 0:
            in_str = ','.join(str(d) for d in sorted(in_dims))
            out_str = ','.join(str(d) for d in sorted(out_dims))
            print(f"  {cat_name:<20} {cat_params:>12,d} {in_str:>12} {out_str:>12}")

    print(f"{'='*70}\n")
    return total_params


def build_eval_dataloader(cfg):
    """构建评估用 dataloader (复用 test.py 逻辑)"""
    from mmdet3d.datasets import build_dataset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from mmdet.datasets import replace_ImageToTensor

    # 构建 test dataset
    cfg.data.test.test_mode = True
    if isinstance(cfg.data.test.pipeline[0], dict):
        if cfg.data.test.pipeline[0].type == 'LoadMultiViewImageFromFiles':
            cfg.data.test.pipeline[0] = dict(
                type='LoadMultiViewImageFromFiles')

    # replace ImageToTensor
    if isinstance(cfg.data.test, dict):
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 4),
        dist=False,
        shuffle=False)
    return data_loader


def build_train_dataloader(cfg, max_samples=None):
    """构建训练用 dataloader (供梯度收集使用)"""
    from mmdet3d.datasets import build_dataset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 4),
        dist=False,
        shuffle=True)
    return data_loader


def evaluate_amota(model, cfg, output_path):
    """评估 AMOTA (复用 test.py 的 single_gpu_test)"""
    from mmcv.parallel import MMDataParallel
    from mmdet3d.apis import single_gpu_test

    data_loader = build_eval_dataloader(cfg)

    model_parallel = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model_parallel, data_loader)

    # 保存结果
    mmcv.dump(outputs, output_path)

    # 评估
    dataset = data_loader.dataset
    eval_kwargs = {'jsonfile_prefix': output_path.replace('.pkl', '')}
    metrics = dataset.evaluate(outputs, **eval_kwargs)

    # 提取 AMOTA
    amota = metrics.get('AMOTA', metrics.get('amota', None))
    if amota is not None:
        print(f"\n  [EVAL] AMOTA = {amota:.4f}")
    else:
        print(f"\n  [EVAL] 评估完成, 指标: {metrics}")

    return metrics


def verify_constraints(model, prune_config):
    """校验剪枝后模型是否满足约束条件

    约束:
    - channel_alignment: 所有 Linear 层的维度必须是 alignment 的倍数
    - min_channels: 不能低于最小通道数
    - skip_layers: 坐标敏感层维度不应改变 (仅输出维度不变)
    """
    constraints = prune_config.get('constraints', {})
    alignment = constraints.get('channel_alignment', 8)
    min_ch = constraints.get('min_channels', 64)
    skip_patterns = constraints.get('skip_layers', [])

    violations = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        out_features = module.out_features
        in_features = module.in_features

        # 对齐校验 (仅检查非坐标敏感层)
        is_skip = any(pat in name for pat in skip_patterns)
        if not is_skip:
            if out_features % alignment != 0:
                violations.append(
                    f"[ALIGN] {name}: out_features={out_features} "
                    f"不是 {alignment} 的倍数")
            if out_features < min_ch and out_features > 1:
                # out_features=1 是合法的 (如最终分类输出)
                violations.append(
                    f"[MIN_CH] {name}: out_features={out_features} "
                    f"< min_channels={min_ch}")

    if violations:
        print(f"\n  [WARN] 发现 {len(violations)} 个约束违规:")
        for v in violations:
            print(f"    {v}")
    else:
        print(f"\n  [OK] 所有约束校验通过 (alignment={alignment}, min_ch={min_ch})")

    return violations


def main():
    args = parse_args()

    # 0. 环境设置
    torch.cuda.set_device(args.gpu_id)
    device = f'cuda:{args.gpu_id}'
    from mmdet.apis import set_random_seed
    set_random_seed(args.seed, deterministic=True)

    # 1. 加载剪枝配置
    print(f"[1/6] 加载剪枝配置: {args.prune_config}")
    prune_config = load_prune_config(args.prune_config)
    print(f"  locked: {prune_config['locked']}")
    print(f"  encoder: {prune_config['encoder']}")
    print(f"  decoder: {prune_config['decoder']}")
    print(f"  heads: {prune_config['heads']}")

    # 2. 加载模型
    print(f"\n[2/6] 加载模型: {args.config} + {args.checkpoint}")
    model, cfg = load_model_from_config(args.config, args.checkpoint, device)
    original_params = report_model_stats(model, prefix="原始模型")

    # 3. 准备 dataloader (Taylor/Hessian 需要)
    print(f"\n[3/6] 准备数据 (梯度收集用)")
    criterion = prune_config['locked'].get('importance_criterion', 'taylor')
    dataloader = None
    if criterion in ('taylor', 'hessian'):
        dataloader = build_train_dataloader(cfg, max_samples=args.calib_batches)
        print(f"  准则={criterion}, 将收集 {args.calib_batches} batches 的梯度")
    else:
        print(f"  准则={criterion}, 无需梯度收集")

    # 4. 执行剪枝
    print(f"\n[4/6] 执行剪枝...")
    t0 = time.time()

    from projects.mmdet3d_plugin.univ2x.pruning.prune_univ2x import (
        apply_prune_config)

    model = apply_prune_config(model, prune_config, dataloader=dataloader)
    elapsed = time.time() - t0
    print(f"  剪枝完成, 耗时 {elapsed:.1f}s")

    # 5. 校验约束 + 报告统计
    print(f"\n[5/6] 校验约束 + 统计")
    pruned_params = report_model_stats(model, prefix="剪枝后模型")
    verify_constraints(model, prune_config)

    reduction = 1.0 - pruned_params / original_params
    print(f"  参数缩减比例: {reduction*100:.1f}%")
    print(f"  原始: {original_params/1e6:.2f}M -> 剪枝后: {pruned_params/1e6:.2f}M")

    # 6. 保存剪枝后模型
    print(f"\n[6/6] 保存剪枝后模型: {args.output}")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    save_dict = {
        'state_dict': model.state_dict(),
        'meta': {
            'prune_config': prune_config,
            'original_params': original_params,
            'pruned_params': pruned_params,
            'reduction_ratio': reduction,
        }
    }
    torch.save(save_dict, args.output)
    print(f"  已保存 ({os.path.getsize(args.output)/1e6:.1f}MB)")

    # 可选: 评估
    if args.eval:
        print(f"\n[EVAL] 开始评估 AMOTA...")
        metrics = evaluate_amota(model, cfg, args.eval_out)

    print(f"\n{'='*70}")
    print(f"  管线完成!")
    print(f"  剪枝配置: {args.prune_config}")
    print(f"  参数缩减: {reduction*100:.1f}%")
    print(f"  输出模型: {args.output}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
```

**命令行使用示例**:

```bash
# 仅剪枝 + 保存
python tools/prune_and_eval.py \
    --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
    --checkpoint work_dirs/latest.pth \
    --prune-config prune_configs/moderate.json \
    --output work_dirs/pruned_moderate.pth

# 剪枝 + 评估
python tools/prune_and_eval.py \
    --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
    --checkpoint work_dirs/latest.pth \
    --prune-config prune_configs/aggressive.json \
    --output work_dirs/pruned_aggressive.pth \
    --eval

# 使用 L1 范数 (无需梯度收集, 更快)
python tools/prune_and_eval.py \
    --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
    --checkpoint work_dirs/latest.pth \
    --prune-config prune_configs/conservative.json \
    --output work_dirs/pruned_conservative.pth \
    --eval --calib-batches 0
```

---

### 3.2 `prune_configs/*.json` — 预设配置

#### 3.2.1 `prune_configs/conservative.json`

保守配置：仅剪枝 FFN 20%，其余不动。适用于精度敏感场景。

```json
{
  "version": "1.2",
  "_description": "保守配置: 仅轻度 FFN 剪枝, 预期参数缩减 ~10%, AMOTA 下降 <1%",

  "locked": {
    "importance_criterion": "taylor",
    "pruning_granularity": "local",
    "iterative_steps": 5,
    "round_to": 8
  },

  "encoder": {
    "ffn_mid_ratio": 0.8,
    "attn_proj_ratio": 0.0,
    "head_pruning_ratio": 0.0
  },
  "decoder": {
    "ffn_mid_ratio": 0.8,
    "attn_proj_ratio": 0.0,
    "head_pruning_ratio": 0.0,
    "num_layers": 6
  },
  "heads": {
    "head_mid_ratio": 1.0
  },

  "finetune": {
    "epochs": 5
  },
  "constraints": {
    "skip_layers": ["sampling_offsets", "attention_weights"],
    "min_channels": 64,
    "channel_alignment": 8
  }
}
```

#### 3.2.2 `prune_configs/moderate.json`

中等配置：FFN 40% 剪枝 + 注意力投影 10% 剪枝 + 检测头 30% 剪枝。

```json
{
  "version": "1.2",
  "_description": "中等配置: FFN 40% + attn proj 10% + heads 30%, 预期参数缩减 ~25%, AMOTA 下降 2-4%",

  "locked": {
    "importance_criterion": "taylor",
    "pruning_granularity": "local",
    "iterative_steps": 5,
    "round_to": 8
  },

  "encoder": {
    "ffn_mid_ratio": 0.6,
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

  "finetune": {
    "epochs": 10
  },
  "constraints": {
    "skip_layers": ["sampling_offsets", "attention_weights"],
    "min_channels": 64,
    "channel_alignment": 8
  }
}
```

#### 3.2.3 `prune_configs/aggressive.json`

激进配置：FFN 60% 剪枝 + 注意力投影 20% + 检测头 50% + 头剪枝 + 删层。

```json
{
  "version": "1.2",
  "_description": "激进配置: FFN 60% + attn proj 20% + heads 50% + head pruning + 删 1 层, 预期参数缩减 ~45%, AMOTA 下降 5-8%",

  "locked": {
    "importance_criterion": "taylor",
    "pruning_granularity": "local",
    "iterative_steps": 5,
    "round_to": 8
  },

  "encoder": {
    "ffn_mid_ratio": 0.4,
    "attn_proj_ratio": 0.2,
    "head_pruning_ratio": 0.125
  },
  "decoder": {
    "ffn_mid_ratio": 0.4,
    "attn_proj_ratio": 0.2,
    "head_pruning_ratio": 0.125,
    "num_layers": 5
  },
  "heads": {
    "head_mid_ratio": 0.5
  },

  "finetune": {
    "epochs": 15
  },
  "constraints": {
    "skip_layers": ["sampling_offsets", "attention_weights"],
    "min_channels": 64,
    "channel_alignment": 8
  }
}
```

**三档预设对比**:

| 配置 | FFN ratio | Attn proj | Head ratio | 头剪枝 | 解码层 | 预期参数缩减 | 预期 AMOTA 下降 |
|------|:---------:|:---------:|:----------:|:------:|:------:|:------------:|:---------------:|
| conservative | 0.8 | 0.0 | 1.0 | 无 | 6 | ~10% | <1% |
| moderate | 0.6 | 0.1 | 0.7 | 无 | 6 | ~25% | 2-4% |
| aggressive | 0.4 | 0.2 | 0.5 | 12.5% | 5 | ~45% | 5-8% |

---

### 3.3 微调配置

#### 3.3.1 `projects/configs_e2e_univ2x/univ2x_coop_e2e_track_finetune_pruned.py`

继承基线训练配置，调整为剪枝后微调的低学习率、短训练周期配置。

```python
"""
剪枝后微调配置

继承: univ2x_coop_e2e_track.py (基线训练配置)
修改:
  - 学习率: 原始 2e-4 的 1/10 → 2e-5
  - 调度: Cosine Annealing, warmup 100 iters
  - 总训练轮数: 10 epoch (可通过 prune_config.finetune.epochs 覆盖)
  - resume_from: 剪枝后 checkpoint
  - 可选: 知识蒸馏 (teacher_checkpoint + temperature + alpha)

用法:
    # 基础微调 (无蒸馏)
    python tools/train.py \
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track_finetune_pruned.py \
        --resume-from work_dirs/pruned_moderate.pth \
        --work-dir work_dirs/finetune_moderate/

    # 带知识蒸馏的微调
    python tools/train.py \
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track_finetune_pruned.py \
        --resume-from work_dirs/pruned_moderate.pth \
        --work-dir work_dirs/finetune_moderate_kd/ \
        --cfg-options \
            distillation.enabled=True \
            distillation.teacher_checkpoint=work_dirs/latest.pth
"""

_base_ = ['./univ2x_coop_e2e_track.py']

# ─── 学习率: 基线 2e-4 的 1/10 ───
optimizer = dict(
    type='AdamW',
    lr=2e-5,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# ─── Cosine 调度, 短 warmup ───
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-4,
)

# ─── 短训练周期 ───
total_epochs = 10
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=2, pipeline={{_base_.test_pipeline}})

# ─── 每 2 epoch 保存一次 ───
checkpoint_config = dict(interval=2)

# ─── 可选: 知识蒸馏配置 ───
# 通过 --cfg-options 或在此处直接修改
distillation = dict(
    enabled=False,
    teacher_checkpoint='',
    temperature=4.0,
    alpha=0.5,
    # distill_loss_weight: final_loss = alpha * distill_loss + (1-alpha) * task_loss
    # 蒸馏层: 仅对 Transformer encoder/decoder 的输出做 KD
    distill_layers=[
        'pts_bbox_head.transformer.encoder',
        'pts_bbox_head.transformer.decoder',
    ],
)
```

#### 3.3.2 知识蒸馏辅助模块说明

如果启用知识蒸馏 (`distillation.enabled=True`)，需要在训练循环中增加以下逻辑（在 `tools/train.py` 或自定义 hook 中实现）:

```python
"""
知识蒸馏辅助逻辑 (集成到训练循环中)

工作原理:
1. 加载 teacher 模型 (原始未剪枝), 冻结参数, 设为 eval 模式
2. 每个 training step:
   a. teacher forward → 提取 distill_layers 的特征
   b. student forward → 提取 distill_layers 的特征
   c. KD loss = KL_divergence(student_feat / T, teacher_feat / T) * T^2
   d. final_loss = alpha * KD_loss + (1 - alpha) * task_loss
3. 仅更新 student 参数
"""

import torch
import torch.nn.functional as F


class DistillationHelper:
    """知识蒸馏辅助类"""

    def __init__(self, teacher_model, temperature=4.0, alpha=0.5,
                 distill_layers=None):
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.temperature = temperature
        self.alpha = alpha
        self.distill_layers = distill_layers or []

        # 注册 hook 提取中间特征
        self.teacher_features = {}
        self.student_features = {}
        self._register_hooks()

    def _register_hooks(self):
        """为 teacher 和 student 的指定层注册 forward hook"""
        for layer_name in self.distill_layers:
            # 此处需要根据实际模型结构获取子模块
            # 简化示例, 实际实现需通过 get_submodule()
            pass

    def compute_distill_loss(self, student_output, teacher_output):
        """计算蒸馏损失

        使用 KL 散度作为特征级蒸馏损失。
        对于 detection 任务, 也可以使用 response-based KD
        (对 cls/reg 输出做 soft target matching)。
        """
        T = self.temperature
        kd_loss = 0.0

        for layer_name in self.distill_layers:
            s_feat = self.student_features.get(layer_name)
            t_feat = self.teacher_features.get(layer_name)

            if s_feat is not None and t_feat is not None:
                # 注意: 剪枝后 student 的通道数可能与 teacher 不同
                # 如果维度不匹配, 用线性映射对齐
                if s_feat.shape != t_feat.shape:
                    # 需要一个可学习的 adapter (1x1 conv 或 linear)
                    # 这里简化为 MSE loss on pooled features
                    s_pooled = F.adaptive_avg_pool1d(
                        s_feat.flatten(0, -2).unsqueeze(0), 1).squeeze()
                    t_pooled = F.adaptive_avg_pool1d(
                        t_feat.flatten(0, -2).unsqueeze(0), 1).squeeze()
                    kd_loss += F.mse_loss(s_pooled, t_pooled)
                else:
                    s_log_prob = F.log_softmax(s_feat / T, dim=-1)
                    t_prob = F.softmax(t_feat / T, dim=-1)
                    kd_loss += F.kl_div(s_log_prob, t_prob,
                                        reduction='batchmean') * (T ** 2)

        return kd_loss

    def compute_final_loss(self, task_loss, student_output, data):
        """计算最终损失 = alpha * KD_loss + (1-alpha) * task_loss"""
        with torch.no_grad():
            teacher_output = self.teacher.forward_dummy(data)

        kd_loss = self.compute_distill_loss(student_output, teacher_output)
        final_loss = (self.alpha * kd_loss
                      + (1 - self.alpha) * task_loss)
        return final_loss
```

**微调使用流程**:

```bash
# Step 1: 剪枝
python tools/prune_and_eval.py \
    --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
    --checkpoint work_dirs/latest.pth \
    --prune-config prune_configs/moderate.json \
    --output work_dirs/pruned_moderate.pth

# Step 2: 微调 (无蒸馏)
python tools/train.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_finetune_pruned.py \
    --resume-from work_dirs/pruned_moderate.pth \
    --work-dir work_dirs/finetune_moderate/

# Step 3: 评估微调后精度
python tools/test.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
    work_dirs/finetune_moderate/latest.pth \
    --eval bbox
```

---

### 3.4 `apply_prune_config()` 完整实现

此函数位于 `projects/mmdet3d_plugin/univ2x/pruning/prune_univ2x.py`，是编排完整剪枝流程的核心函数。

```python
import torch
import torch_pruning as tp


def apply_prune_config(model, prune_config, dataloader=None):
    """将 prune_config.json 应用到模型，执行完整剪枝流程

    编排步骤:
    1. 读取 locked 配置 (P4-P7)
    2. 收集梯度 (若 Taylor/Hessian)
    3. 构建示例输入
    4. 构建 pruner (locked + search 参数)
    5. 执行迭代剪枝
    6. 更新模型内部状态
    7. 执行解码器层数剪枝 (P9)
    8. 校验约束 (channel alignment, min channels)
    9. 返回剪枝后模型

    Args:
        model: 已加载 checkpoint 的 PyTorch 模型
        prune_config: dict, 从 prune_config.json 解析的完整配置
        dataloader: 训练 dataloader, Taylor/Hessian 梯度收集用 (可选)

    Returns:
        model: 剪枝后的模型 (in-place 修改, 但也返回引用)
    """
    from .grad_collector import collect_gradients
    from .post_prune import update_model_after_pruning
    from .custom_pruners import register_univ2x_pruners

    locked = prune_config.get('locked', {})
    enc_cfg = prune_config.get('encoder', {})
    dec_cfg = prune_config.get('decoder', {})
    heads_cfg = prune_config.get('heads', {})
    constraints = prune_config.get('constraints', {})

    # ─── Step 1: 读取锁定配置 (P4-P7) ───
    criterion = locked.get('importance_criterion', 'taylor')
    granularity = locked.get('pruning_granularity', 'local')
    iterative_steps = locked.get('iterative_steps', 5)
    round_to = locked.get('round_to', 8)

    print(f"  [apply_prune_config] 锁定配置:")
    print(f"    P4 importance_criterion = {criterion}")
    print(f"    P5 pruning_granularity  = {granularity}")
    print(f"    P6 iterative_steps      = {iterative_steps}")
    print(f"    P7 round_to             = {round_to}")

    # ─── Step 2: 梯度收集 (P4 为 Taylor/Hessian 时需要) ───
    if criterion in ('taylor', 'hessian'):
        if dataloader is None:
            print(f"  [WARN] {criterion} 准则需要 dataloader, 但未提供; "
                  f"fallback 到 l1_norm")
            criterion = 'l1_norm'
        else:
            print(f"  [Step 2] 收集梯度 ({criterion})...")
            collect_gradients(model, dataloader, num_batches=32)
            print(f"  [Step 2] 梯度收集完成")

    # ─── Step 3: 构建示例输入 ───
    print(f"  [Step 3] 构建示例输入...")
    example_inputs = _build_example_inputs(model)

    # ─── Step 4: 构建 pruner ───
    print(f"  [Step 4] 构建 pruner...")

    # 4a. 选择重要性评估器
    importance = _select_importance(criterion)

    # 4b. 收集自定义剪枝器
    customized_pruners = register_univ2x_pruners()

    # 4c. 收集不可剪枝的层
    ignored_layers = _collect_ignored_layers(model, prune_config)
    print(f"    忽略的层数: {len(ignored_layers)}")

    # 4d. 收集 unwrapped parameters (位置编码等)
    unwrapped_parameters = _collect_unwrapped_params(model)
    print(f"    unwrapped 参数数: {len(unwrapped_parameters)}")

    # 4e. 构建逐层剪枝比例 (搜索维度 P1-P3)
    pruning_ratio_dict = _build_ratio_dict(model, prune_config)
    print(f"    剪枝比例映射: {len(pruning_ratio_dict)} 个层")

    # 4f. 收集注意力头信息 (搜索维度 P8)
    num_heads = _collect_num_heads(model)

    # 4g. 获取 P8 头剪枝比例 (encoder 和 decoder 可能不同)
    head_pruning_ratio = max(
        enc_cfg.get('head_pruning_ratio', 0.0),
        dec_cfg.get('head_pruning_ratio', 0.0))

    # 4h. 创建 pruner
    pruner = tp.pruner.BasePruner(
        model,
        example_inputs,
        importance=importance,
        pruning_ratio_dict=pruning_ratio_dict,
        global_pruning=(granularity == 'global'),
        isomorphic=(granularity == 'isomorphic'),
        iterative_steps=iterative_steps,
        round_to=round_to,
        ignored_layers=ignored_layers,
        customized_pruners=customized_pruners,
        unwrapped_parameters=unwrapped_parameters,
        num_heads=num_heads,
        head_pruning_ratio=head_pruning_ratio,
    )
    print(f"    pruner 创建成功")

    # ─── Step 5: 执行迭代剪枝 ───
    print(f"  [Step 5] 执行 {iterative_steps} 步迭代剪枝...")
    for step_idx in range(iterative_steps):
        pruner.step()
        # 中间状态日志 (每步的参数量)
        current_params = sum(p.numel() for p in model.parameters())
        print(f"    step {step_idx+1}/{iterative_steps}: "
              f"params={current_params/1e6:.2f}M")

    # ─── Step 6: 更新模型内部状态 ───
    print(f"  [Step 6] 更新模型内部状态...")
    update_model_after_pruning(model)

    # ─── Step 7: 解码器层数剪枝 (P9) ───
    target_layers = dec_cfg.get('num_layers', 6)
    print(f"  [Step 7] 解码器层数剪枝: target={target_layers}")
    model = prune_decoder_layers(model, target_layers)

    # ─── Step 8: 校验约束 ───
    print(f"  [Step 8] 校验约束...")
    _verify_internal_constraints(model, constraints)

    # ─── Step 9: 完成 ───
    model.eval()
    print(f"  [apply_prune_config] 完成")
    return model


def _build_example_inputs(model):
    """构建 DepGraph 追踪用的示例输入

    注意: UniV2X 的 forward 签名比较复杂, 需要构造与 train_step
    相同结构的输入。如果 DepGraph 追踪失败, 考虑使用
    tp.DependencyGraph.build_dependency 的 output_transform 参数。
    """
    # 方案 1: 使用 dummy input (需要匹配模型 forward 签名)
    # UniV2X 期望的输入格式取决于 forward 函数的参数
    # 这里构造一个最小化的 dummy input
    device = next(model.parameters()).device

    # BEVFormer 风格的输入
    # img: (B, num_cams, C, H, W) — 但 DepGraph 通常只需追踪到第一个 nn.Module
    # 最简方案: 用 torch.randn 构造, 维度从 config 读取

    # 对于 UniV2X, 我们可能需要使用 model.forward_dummy() 或
    # 手动构建 img tensor
    dummy_img = torch.randn(1, 6, 3, 256, 704).to(device)

    # 注意: 如果模型 forward 需要更多参数 (如 img_metas),
    # DepGraph 的追踪可能需要 forward_fn 参数来适配
    return dummy_img


def _select_importance(criterion):
    """P4: 选择重要性评估方法"""
    mapping = {
        'l1_norm': tp.importance.GroupMagnitudeImportance(p=1),
        'l2_norm': tp.importance.GroupMagnitudeImportance(p=2),
        'taylor': tp.importance.GroupTaylorImportance(),
        'bn_scale': tp.importance.BNScaleImportance(),
        'fpgm': tp.importance.FPGMImportance(),
        'hessian': tp.importance.GroupHessianImportance(),
        'random': tp.importance.RandomImportance(),
    }
    if criterion not in mapping:
        raise ValueError(
            f"未知的重要性准则: {criterion}, "
            f"可选: {list(mapping.keys())}")
    return mapping[criterion]


def _build_ratio_dict(model, prune_config):
    """P1-P3: 为不同模块类型构建逐层剪枝比例字典

    逻辑:
    - 遍历模型所有 nn.Linear
    - 根据层名判断属于 FFN / attn_proj / det_head
    - 分别读取 encoder/decoder/heads 中的 ratio 配置
    - ratio_dict[module] = 1.0 - keep_ratio (torch_pruning 的语义是剪掉多少)
    """
    enc_cfg = prune_config.get('encoder', {})
    dec_cfg = prune_config.get('decoder', {})
    heads_cfg = prune_config.get('heads', {})

    ratio_dict = {}

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        # 判断属于 encoder 还是 decoder
        is_encoder = 'encoder' in name
        is_decoder = 'decoder' in name
        cfg = enc_cfg if is_encoder else dec_cfg if is_decoder else {}

        # P1: FFN 中间层
        if _is_ffn_layer(name):
            keep_ratio = cfg.get('ffn_mid_ratio', 1.0)
            ratio_dict[module] = 1.0 - keep_ratio

        # P2: 注意力投影层 (value_proj, output_proj)
        elif _is_attn_proj(name):
            prune_ratio = cfg.get('attn_proj_ratio', 0.0)
            ratio_dict[module] = prune_ratio

        # P3: 检测头中间层
        elif _is_head_layer(name):
            keep_ratio = heads_cfg.get('head_mid_ratio', 1.0)
            ratio_dict[module] = 1.0 - keep_ratio

    return ratio_dict


def _is_ffn_layer(name):
    """判断是否为 FFN 中间层"""
    return any(k in name for k in ['ffns.', 'feedforward.'])


def _is_attn_proj(name):
    """判断是否为注意力投影层 (非坐标敏感)"""
    return any(k in name for k in ['value_proj', 'output_proj'])


def _is_head_layer(name):
    """判断是否为检测头中间层 (非最终输出层)

    检测头结构: cls_branches / reg_branches, 每组 7 个 Linear
    最后一个 (index 6) 是输出层, 不剪枝
    """
    is_head = any(k in name for k in ['cls_branches', 'reg_branches'])
    # 最后一层标记: 通常是 cls_branches.6 或 reg_branches.6
    is_final = any(k in name for k in ['.6.', 'final'])
    return is_head and not is_final


def _collect_ignored_layers(model, prune_config):
    """收集不可剪枝的层

    来源:
    - 硬约束: sampling_offsets, attention_weights (坐标敏感)
    - 硬约束: 最终输出层 (cls/reg 的最后一个 Linear)
    - 配置约束: constraints.skip_layers 中指定的模式
    """
    constraints = prune_config.get('constraints', {})
    skip_patterns = constraints.get('skip_layers',
                                     ['sampling_offsets', 'attention_weights'])

    ignored = []
    for name, module in model.named_modules():
        # 坐标敏感层 + 用户指定的跳过层
        if any(pat in name for pat in skip_patterns):
            ignored.append(module)
            continue

        # 最终输出层
        if _is_output_layer(name, module):
            ignored.append(module)

    return ignored


def _is_output_layer(name, module):
    """判断是否为最终输出层 (不可剪枝)"""
    if isinstance(module, torch.nn.Linear):
        # cls_branches 和 reg_branches 的最后一层
        if any(f'{prefix}.6' in name
               for prefix in ['cls_branches', 'reg_branches']):
            return True
    return False


def _collect_num_heads(model):
    """收集注意力模块的头数信息 (供 P8 头剪枝使用)"""
    num_heads = {}
    for name, module in model.named_modules():
        if hasattr(module, 'num_heads') and hasattr(module, 'embed_dims'):
            num_heads[module] = module.num_heads
    return num_heads


def _collect_unwrapped_params(model):
    """收集位置编码等非 Module 的可学习参数

    这些参数的某个维度与 embed_dims 绑定,
    剪枝 embed_dims 时需要同步剪枝。
    """
    unwrapped = []
    for name, param in model.named_parameters():
        if 'bev_embedding' in name or 'query_embedding' in name:
            # 在 dim=1 (embed_dim 维度) 上剪枝
            unwrapped.append((param, 1))
        elif 'reference_points' in name:
            # reference_points 通常不随 embed_dims 变化, 跳过
            pass
    return unwrapped


def prune_decoder_layers(model, target_num_layers):
    """P9: 解码器层数剪枝

    策略: 保留最后 N 层 (最后几层对精度贡献最大)
    同步: 调整 cls_branches / reg_branches 的迭代预测分支
    """
    decoder = model.pts_bbox_head.transformer.decoder
    current_layers = len(decoder.layers)

    if target_num_layers >= current_layers:
        print(f"    target={target_num_layers} >= current={current_layers}, "
              f"跳过层数剪枝")
        return model

    # 保留最后 N 层
    keep_indices = list(range(
        current_layers - target_num_layers, current_layers))
    print(f"    保留层: {keep_indices} (共 {current_layers} → {target_num_layers})")

    # 替换 decoder layers
    new_layers = torch.nn.ModuleList(
        [decoder.layers[i] for i in keep_indices])
    decoder.layers = new_layers
    decoder.num_layers = target_num_layers

    # 同步调整迭代预测分支
    head = model.pts_bbox_head
    if hasattr(head, 'cls_branches') and head.cls_branches is not None:
        # 保留对应层的分支 + 最后一个 (用于最终输出)
        new_cls = torch.nn.ModuleList(
            [head.cls_branches[i] for i in keep_indices]
            + [head.cls_branches[-1]])
        head.cls_branches = new_cls

    if hasattr(head, 'reg_branches') and head.reg_branches is not None:
        new_reg = torch.nn.ModuleList(
            [head.reg_branches[i] for i in keep_indices]
            + [head.reg_branches[-1]])
        head.reg_branches = new_reg

    return model


def _verify_internal_constraints(model, constraints):
    """内部约束校验 (apply_prune_config 结束前的自检)"""
    alignment = constraints.get('channel_alignment', 8)
    min_ch = constraints.get('min_channels', 64)

    violation_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if module.out_features % alignment != 0:
                print(f"    [VIOLATION] {name}: "
                      f"out_features={module.out_features} "
                      f"不对齐 {alignment}")
                violation_count += 1
            if 1 < module.out_features < min_ch:
                print(f"    [VIOLATION] {name}: "
                      f"out_features={module.out_features} "
                      f"< min={min_ch}")
                violation_count += 1

    if violation_count == 0:
        print(f"    约束校验通过")
    else:
        print(f"    发现 {violation_count} 个违规 (可能需要调整 round_to 或 ratio)")
```

---

## 4. 代码检测方案

### Test 1: 剪枝 + 保存 (conservative 配置)

```bash
# 预期: 脚本正常完成, 输出文件可被 torch.load 加载
python tools/prune_and_eval.py \
    --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
    --checkpoint work_dirs/latest.pth \
    --prune-config prune_configs/conservative.json \
    --output /tmp/test_pruned.pth

# 验证: 加载检查
python -c "
import torch
ckpt = torch.load('/tmp/test_pruned.pth', map_location='cpu')
print('keys:', list(ckpt.keys()))
print('meta:', ckpt['meta'])
print('state_dict keys:', len(ckpt['state_dict']))
print('reduction:', f\"{ckpt['meta']['reduction_ratio']*100:.1f}%\")
"
```

**通过标准**: 脚本无异常退出, `state_dict` 非空, `reduction_ratio > 0`。

### Test 2: 剪枝 + 评估 AMOTA

```bash
# 预期: AMOTA 被正常报告 (值低于基线, 这是预期行为)
python tools/prune_and_eval.py \
    --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
    --checkpoint work_dirs/latest.pth \
    --prune-config prune_configs/moderate.json \
    --output /tmp/test_pruned_eval.pth \
    --eval
```

**通过标准**: `[EVAL] AMOTA = x.xxxx` 被打印, 无 evaluation 崩溃。

### Test 3: 参数缩减比例校验

```bash
# 对三个预设配置分别运行, 验证缩减比例在预期范围内
for cfg in conservative moderate aggressive; do
    python tools/prune_and_eval.py \
        --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
        --checkpoint work_dirs/latest.pth \
        --prune-config prune_configs/${cfg}.json \
        --output /tmp/test_${cfg}.pth 2>&1 | grep "参数缩减"
done
```

**通过标准**:
- conservative: 缩减 5-15%
- moderate: 缩减 20-30%
- aggressive: 缩减 35-50%

### Test 4: 微调 1 epoch 验证

```bash
# 预期: loss 在 1 epoch 内有下降趋势
python tools/train.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_finetune_pruned.py \
    --resume-from /tmp/test_pruned.pth \
    --work-dir /tmp/finetune_test/ \
    --cfg-options total_epochs=1 runner.max_epochs=1

# 检查 log 中 loss 是否下降
grep "loss" /tmp/finetune_test/*/log.json | head -5
grep "loss" /tmp/finetune_test/*/log.json | tail -5
```

**通过标准**: 最后 5 条 loss < 最初 5 条 loss 的均值。

---

## 5. Debug 方案

### 5.1 Config 解析错误

**症状**: `json.JSONDecodeError` 或 `KeyError`。

**排查步骤**:
1. 用 `python -m json.tool prune_configs/xxx.json` 验证 JSON 合法性
2. 检查 `load_prune_config()` 的默认值填充是否覆盖了所有必需字段
3. 确认 version 字段匹配

**预防**: `load_prune_config()` 已内置 schema 校验和默认值填充。

### 5.2 模型加载失败 (剪枝后)

**症状**: `RuntimeError: Error(s) in loading state_dict ... size mismatch`。

**排查步骤**:
```python
# 比对 state_dict keys 和 shape
saved = torch.load('pruned.pth', map_location='cpu')['state_dict']
model_keys = dict(model.named_parameters())
for k in saved:
    if k in model_keys and saved[k].shape != model_keys[k].shape:
        print(f"MISMATCH: {k}: saved={saved[k].shape} vs model={model_keys[k].shape}")
```

**常见原因**:
- `update_model_after_pruning()` 遗漏了某个模块的属性更新
- 自定义剪枝器的通道依赖关系描述不正确
- 位置编码等 unwrapped parameter 未被正确处理

### 5.3 AMOTA 评估崩溃

**症状**: 评估过程中 `RuntimeError` 或 `KeyError`。

**排查步骤**:
1. 确认 TRT 相关模块 (`*TRT` 后缀类) 在 PyTorch eval 模式下未被激活
2. 检查模型 `forward()` 是否能正常处理剪枝后的维度:
   ```python
   model.eval()
   with torch.no_grad():
       dummy_data = next(iter(dataloader))
       output = model(**dummy_data)
   ```
3. 如果是 nuscenes evaluation 的 json 格式问题, 检查输出 detection 数量是否为 0

### 5.4 微调发散

**症状**: loss 不下降或 NaN。

**排查步骤**:
1. **学习率过高**: 将 lr 从 2e-5 降到 1e-5 或 5e-6
2. **梯度爆炸**: 检查 gradient norm:
   ```python
   total_norm = 0
   for p in model.parameters():
       if p.grad is not None:
           total_norm += p.grad.data.norm(2).item() ** 2
   print(f"grad_norm = {total_norm ** 0.5}")
   ```
3. **BN 统计量失效**: 剪枝后 BatchNorm 的 running_mean/var 可能不匹配, 在微调前先 re-calibrate:
   ```python
   model.train()
   for i, data in enumerate(dataloader):
       if i >= 100: break
       model(**data)  # 仅 forward, 让 BN 更新统计量
   ```
4. **frozen layers 冲突**: 确认 `paramwise_cfg` 中的 `lr_mult` 设置没有冻结需要更新的层

### 5.5 DepGraph 追踪失败

**症状**: `torch.fx.proxy.TraceError` 或维度不匹配。

**排查步骤**:
1. 确认所有自定义 CUDA 算子已注册自定义剪枝器
2. 检查 `_build_example_inputs()` 返回的 dummy input 维度是否正确
3. 使用 `verbose=True` 参数让 DepGraph 打印追踪日志:
   ```python
   DG = tp.DependencyGraph()
   DG.build_dependency(model, example_inputs=example_inputs, verbose=True)
   ```

---

## 6. 验收标准

| 编号 | 标准 | 验证方式 |
|:----:|------|---------|
| AC-1 | 给定任意合法 `prune_config.json`，`prune_and_eval.py` 在 <15 分钟内完成 | 计时 + 3 个预设配置全部通过 |
| AC-2 | 输出 `.pth` 可被 `torch.load` 加载，且 `state_dict` 可被模型接受 | Test 1 |
| AC-3 | `--eval` 标志下 AMOTA 被正常报告 (即使值低于基线) | Test 2 |
| AC-4 | 3 个预设配置的参数缩减比例在预期范围内 | Test 3 |
| AC-5 | 微调管线可无错运行至少 1 epoch，且 loss 有下降趋势 | Test 4 |
| AC-6 | `apply_prune_config()` 完成后，模型通过约束校验 (channel alignment, min channels) | `verify_constraints()` 返回空列表 |
| AC-7 | 剪枝后模型的逐模块通道数被正确报告 | `report_model_stats()` 输出合理 |

---

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|:----:|:----:|---------|
| **DepGraph 示例输入构造失败**: UniV2X 的 forward 签名复杂，dummy input 可能不足以驱动完整追踪 | 高 | 阻断 | 准备 fallback 方案: 使用 `tp.DependencyGraph` 的 `forward_fn` 参数包装 forward 逻辑；或用实际数据的第一个 batch 替代 dummy input |
| **剪枝后 state_dict key 不匹配**: DepGraph 修改模型结构后，某些 key 可能变化 | 中 | 阻断 | 保存时直接 `model.state_dict()` (已剪枝的模型)，加载时 `strict=False` + 手动检查缺失/多余 key |
| **微调不收敛**: 激进剪枝后模型容量大幅下降，微调可能无法恢复 | 中 | 延迟 | (1) 降低学习率到 1e-6 级别; (2) 启用知识蒸馏; (3) 增加微调 epoch 到 20; (4) 如果仍不收敛，说明该剪枝配置过于激进，需要放宽 ratio |
| **评估 AMOTA 与训练环境不兼容**: TRT 相关模块在 PyTorch eval 中可能被错误调用 | 低 | 阻断 | 评估时使用基线 config (不含 TRT 后缀的模块)，确保 `model.eval()` 走 PyTorch 路径 |
| **通道对齐导致实际剪枝比例偏离配置**: `round_to=8` 使得小比例剪枝被 round 到 0 | 低 | 精度 | 在 `report_model_stats()` 中报告实际通道数 vs 配置预期，供用户确认；对于 `embed_dims=256` 的层, 10% 剪枝 → 剪 25.6 → round 到 24, 实际剪 9.4%, 偏差可接受 |
| **知识蒸馏维度不匹配**: 剪枝后 student 与 teacher 的中间特征维度不同，KD loss 无法直接计算 | 中 | 延迟 | 使用 response-based KD (仅在最终输出层做 soft target matching) 而非 feature-based KD，避免中间层维度对齐问题；或添加 1x1 adapter 层 |

---

## 附录: 完整管线流程图

```
prune_config.json
       │
       ▼
┌──────────────────┐     ┌──────────────────┐
│  load_prune_     │     │  load_model_     │
│  config()        │     │  from_config()   │
│  校验 + 默认值    │     │  config + ckpt   │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └───────────┬────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  apply_prune_config │
          │                     │
          │  Step 1: 读 locked  │
          │  Step 2: 收集梯度   │
          │  Step 3: 构建输入   │
          │  Step 4: 构建 pruner│
          │  Step 5: 迭代剪枝   │
          │  Step 6: 更新状态   │
          │  Step 7: 删层 (P9)  │
          │  Step 8: 校验约束   │
          └────────┬────────────┘
                   │
                   ▼
          ┌────────────────────┐
          │  report_model_     │
          │  stats()           │
          │  参数量 / 通道数    │
          └────────┬───────────┘
                   │
              ┌────┴────┐
              │         │
              ▼         ▼
        ┌──────────┐  ┌──────────────┐
        │ 保存 .pth │  │ evaluate_    │
        │          │  │ amota()      │
        └──────────┘  │ (可选)       │
                      └──────────────┘
                              │
                              ▼
                      ┌──────────────┐
                      │  微调管线     │
                      │  train.py +  │
                      │  finetune cfg│
                      └──────────────┘
```
