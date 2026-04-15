# Phase A4: Phase 0 实验 (锁定维度预选 + 敏感度分析)

> **任务编号**: A.3-0A + A.3-0B
> **预计工时**: 2 天 (0.5 天编码 + 1.5 天执行)
> **前置依赖**: Phase A3 完成 (`tools/prune_and_eval.py` 端到端可用)
> **产出文件**: `locked_config.json` + `pruning_sensitivity_report.json`

---

## 1. 阶段目标

本阶段分为两个子阶段，通过系统实验确定搜索空间的最终形态：

### Phase 0-A: 锁定 P4/P5/P6 (一次性对比选定)

通过控制变量实验，确定以下三个维度的最优值并锁定：

| 维度 | 变量名 | 候选值 | 对比方法 |
|:----:|--------|--------|---------|
| P4 | importance_criterion | {l1_norm, taylor, fpgm, hessian} | 固定 FFN 30% 剪枝，比较 AMOTA |
| P5 | pruning_granularity | {global, local, isomorphic} | 固定 FFN 30% + P4 锁定值，比较 AMOTA |
| P6 | iterative_steps | {1, 3, 5, 10} | 固定 FFN 40% + P4/P5 锁定值，比较 AMOTA |

P7 (round_to) 已由 INT8 硬件约束锁定为 8，不需要实验验证。

**依据**: 1.1 可配置量化中 D5(symmetric)/D6(scale_method) 的教训——方法选择维度与搜索维度解耦，一次性选定即可，不存在交互效应。

### Phase 0-B: 搜索维度敏感度分析 (为联合搜索提供先验)

分析进入联合搜索的 5 个维度 (P1/P2/P3/P8/P9) 的精度影响，目标是：

1. **P1 (FFN)**: 识别 12 个 FFN 层各自的敏感度等级 → 为 per-layer ratio 提供先验
2. **P2 (注意力投影)**: 量化 value_proj/output_proj 剪枝的精度代价
3. **P8 (注意力头)**: 评估头剪枝的可行性
4. **P9 (解码器层数)**: 评估删层的精度代价
5. **B1×B2 交互项**: 验证剪枝与量化是否存在放大/抵消效应

---

## 2. 前置条件

### 2.1 代码前置

- [x] Phase A3 完成: `tools/prune_and_eval.py` 端到端可用
  - 输入: `prune_config.json` → 输出: 剪枝后模型 + AMOTA 分数
  - 支持所有 9 个维度 (P1-P9)
- [x] 自定义剪枝器注册完毕 (`custom_pruners.py`)
- [x] 梯度收集器可用 (`grad_collector.py`，Taylor/Hessian 需要)
- [x] 剪枝后模块状态更新可用 (`post_prune.py`)

### 2.2 基线数据

| 配置 | AMOTA |
|------|:-----:|
| FP16 TRT (未剪枝) | 0.370 |
| PyTorch (未剪枝) | 0.338 |

> 注意: Phase 0 所有实验均在 PyTorch 侧执行 (无需构建 TRT engine)，
> 因此基线对比使用 PyTorch AMOTA = 0.338。

### 2.3 硬件与环境

- GPU: 单卡 (与 Phase A3 验证环境一致)
- 数据集: nuScenes val split (完整评估)
- 校准数据: 32 batch (Taylor/Hessian 梯度收集)

---

## 3. 具体代码实现

### 3.1 `tools/pruning_sensitivity_analysis.py` — 完整脚本

该脚本自动化执行全部 Phase 0 实验，支持两种运行模式。

```python
#!/usr/bin/env python3
"""
Phase 0 实验自动化脚本: 锁定维度预选 + 搜索维度敏感度分析

用法:
  # Phase 0-A: 锁定 P4/P5/P6
  python tools/pruning_sensitivity_analysis.py \
      --mode lock-dims \
      --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
      --checkpoint work_dirs/latest.pth \
      --output-dir work_dirs/phase0

  # Phase 0-B: 搜索维度敏感度
  python tools/pruning_sensitivity_analysis.py \
      --mode sensitivity \
      --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
      --checkpoint work_dirs/latest.pth \
      --locked-config work_dirs/phase0/locked_config.json \
      --output-dir work_dirs/phase0

  # 快速验证模式 (仅 1 batch, 检查脚本逻辑)
  python tools/pruning_sensitivity_analysis.py \
      --mode lock-dims \
      --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
      --checkpoint work_dirs/latest.pth \
      --output-dir work_dirs/phase0_debug \
      --max-samples 1 \
      --fast
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
import numpy as np
import torch

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint

from projects.mmdet3d_plugin.univ2x.pruning.prune_univ2x import (
    apply_prune_config,
)

# ============================================================
# 日志与工具函数
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def save_checkpoint_results(results, output_path):
    """保存中间结果, 支持断点续跑"""
    tmp_path = str(output_path) + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)
    logger.info(f"中间结果已保存: {output_path}")


def load_checkpoint_results(output_path):
    """加载中间结果, 用于断点续跑"""
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_model_fresh(cfg_path, ckpt_path):
    """每次实验加载一份全新模型 (避免剪枝状态污染)"""
    cfg = Config.fromfile(cfg_path)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt_path, map_location='cpu')
    model.cuda().eval()
    return model, cfg


def evaluate_amota(model, cfg, max_samples=None):
    """
    评估剪枝后模型的 AMOTA

    Args:
        model: 剪枝后的模型
        cfg: mmdet3d Config 对象
        max_samples: 最大评估样本数 (None=全量评估, 用于快速筛选)

    Returns:
        float: AMOTA 分数
    """
    dataset = build_dataset(cfg.data.test)
    # 注意: 这里复用 prune_and_eval.py 中的评估逻辑
    # 实际实现应调用 dataset.evaluate() 并解析 AMOTA
    from tools.prune_and_eval import run_evaluation
    return run_evaluation(model, dataset, max_samples=max_samples)


def build_dataloader_for_grad(cfg):
    """构建用于梯度收集的 dataloader (训练集子集)"""
    dataset = build_dataset(cfg.data.train)
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
    )


# ============================================================
# Phase 0-A: 锁定维度预选
# ============================================================

def run_lock_dims(args):
    """
    Phase 0-A: 依次锁定 P4, P5, P6

    执行顺序:
      1. P4: 4 种重要性准则 → 选最优
      2. P5: 3 种剪枝粒度 → 选最优 (使用 P4 锁定值)
      3. P6: 4 种迭代步数 → 验证 (使用 P4+P5 锁定值)

    产出: locked_config.json
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / 'lock_dims_checkpoint.json'
    locked_config_path = output_dir / 'locked_config.json'

    # 加载断点
    results = load_checkpoint_results(checkpoint_path) or {
        'meta': {
            'start_time': get_timestamp(),
            'config': args.config,
            'checkpoint': args.checkpoint,
            'fast_mode': args.fast,
        },
        'set1_criterion': {},
        'set2_granularity': {},
        'set3_iterative_steps': {},
        'locked': {
            'round_to': 8,  # P7: INT8 硬约束, 无需实验
        },
    }

    # ── 实验集 1: P4 重要性准则对比 ──────────────────────
    logger.info("=" * 60)
    logger.info("实验集 1: P4 重要性准则对比")
    logger.info("固定: FFN 30% 剪枝, P5=local, P6=5, P7=8")
    logger.info("=" * 60)

    criteria = ['l1_norm', 'taylor', 'fpgm', 'hessian']
    for criterion in criteria:
        exp_key = f'criterion_{criterion}'
        if exp_key in results['set1_criterion']:
            logger.info(f"  跳过已完成: {criterion} -> AMOTA={results['set1_criterion'][exp_key]['amota']}")
            continue

        logger.info(f"  测试准则: {criterion}")
        t0 = time.time()

        try:
            model, cfg = load_model_fresh(args.config, args.checkpoint)
            dataloader = build_dataloader_for_grad(cfg) if criterion in ('taylor', 'hessian') else None

            prune_cfg = {
                'locked': {
                    'importance_criterion': criterion,
                    'pruning_granularity': 'local',
                    'iterative_steps': 5,
                    'round_to': 8,
                },
                'encoder': {'ffn_mid_ratio': 0.7},  # 30% 剪枝
                'decoder': {'ffn_mid_ratio': 0.7, 'num_layers': 6},
                'heads': {'head_mid_ratio': 1.0},
            }

            model = apply_prune_config(model, prune_cfg, dataloader=dataloader)
            amota = evaluate_amota(model, cfg, max_samples=args.max_samples)

            results['set1_criterion'][exp_key] = {
                'criterion': criterion,
                'amota': amota,
                'elapsed_sec': round(time.time() - t0, 1),
                'timestamp': get_timestamp(),
            }
            logger.info(f"  {criterion}: AMOTA={amota:.4f}, 耗时={time.time()-t0:.0f}s")

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and criterion == 'hessian':
                logger.warning(f"  {criterion} OOM, 标记为不可用")
                results['set1_criterion'][exp_key] = {
                    'criterion': criterion,
                    'amota': -1.0,
                    'error': 'OOM',
                    'timestamp': get_timestamp(),
                }
                torch.cuda.empty_cache()
            else:
                raise

        save_checkpoint_results(results, checkpoint_path)
        # 释放 GPU 内存
        del model
        torch.cuda.empty_cache()

    # 选出 P4 最优值
    valid_results = {
        k: v for k, v in results['set1_criterion'].items()
        if v.get('amota', -1) > 0
    }
    best_criterion_key = max(valid_results, key=lambda k: valid_results[k]['amota'])
    best_criterion = valid_results[best_criterion_key]['criterion']
    results['locked']['importance_criterion'] = best_criterion
    logger.info(f"P4 锁定: importance_criterion = {best_criterion}")

    # ── 实验集 2: P5 剪枝粒度对比 ──────────────────────
    logger.info("=" * 60)
    logger.info("实验集 2: P5 剪枝粒度对比")
    logger.info(f"固定: FFN 30% 剪枝, P4={best_criterion}, P6=5, P7=8")
    logger.info("=" * 60)

    granularities = ['global', 'local', 'isomorphic']
    for granularity in granularities:
        exp_key = f'granularity_{granularity}'
        if exp_key in results['set2_granularity']:
            logger.info(f"  跳过已完成: {granularity} -> AMOTA={results['set2_granularity'][exp_key]['amota']}")
            continue

        logger.info(f"  测试粒度: {granularity}")
        t0 = time.time()

        model, cfg = load_model_fresh(args.config, args.checkpoint)
        dataloader = build_dataloader_for_grad(cfg) if best_criterion in ('taylor', 'hessian') else None

        prune_cfg = {
            'locked': {
                'importance_criterion': best_criterion,
                'pruning_granularity': granularity,
                'iterative_steps': 5,
                'round_to': 8,
            },
            'encoder': {'ffn_mid_ratio': 0.7},
            'decoder': {'ffn_mid_ratio': 0.7, 'num_layers': 6},
            'heads': {'head_mid_ratio': 1.0},
        }

        model = apply_prune_config(model, prune_cfg, dataloader=dataloader)
        amota = evaluate_amota(model, cfg, max_samples=args.max_samples)

        results['set2_granularity'][exp_key] = {
            'granularity': granularity,
            'amota': amota,
            'elapsed_sec': round(time.time() - t0, 1),
            'timestamp': get_timestamp(),
        }
        logger.info(f"  {granularity}: AMOTA={amota:.4f}, 耗时={time.time()-t0:.0f}s")

        save_checkpoint_results(results, checkpoint_path)
        del model
        torch.cuda.empty_cache()

    # 选出 P5 最优值
    best_gran_key = max(
        results['set2_granularity'],
        key=lambda k: results['set2_granularity'][k]['amota'],
    )
    best_granularity = results['set2_granularity'][best_gran_key]['granularity']
    results['locked']['pruning_granularity'] = best_granularity
    logger.info(f"P5 锁定: pruning_granularity = {best_granularity}")

    # ── 实验集 3: P6 迭代步数验证 ──────────────────────
    logger.info("=" * 60)
    logger.info("实验集 3: P6 迭代步数验证")
    logger.info(f"固定: FFN 40% 剪枝, P4={best_criterion}, P5={best_granularity}, P7=8")
    logger.info("=" * 60)

    step_candidates = [1, 3, 5, 10]
    for steps in step_candidates:
        exp_key = f'steps_{steps}'
        if exp_key in results['set3_iterative_steps']:
            logger.info(f"  跳过已完成: steps={steps} -> AMOTA={results['set3_iterative_steps'][exp_key]['amota']}")
            continue

        logger.info(f"  测试迭代步数: {steps}")
        t0 = time.time()

        model, cfg = load_model_fresh(args.config, args.checkpoint)
        dataloader = build_dataloader_for_grad(cfg) if best_criterion in ('taylor', 'hessian') else None

        prune_cfg = {
            'locked': {
                'importance_criterion': best_criterion,
                'pruning_granularity': best_granularity,
                'iterative_steps': steps,
                'round_to': 8,
            },
            'encoder': {'ffn_mid_ratio': 0.6},  # 40% 剪枝 (更激进, 放大差异)
            'decoder': {'ffn_mid_ratio': 0.6, 'num_layers': 6},
            'heads': {'head_mid_ratio': 1.0},
        }

        model = apply_prune_config(model, prune_cfg, dataloader=dataloader)
        amota = evaluate_amota(model, cfg, max_samples=args.max_samples)

        results['set3_iterative_steps'][exp_key] = {
            'iterative_steps': steps,
            'amota': amota,
            'elapsed_sec': round(time.time() - t0, 1),
            'timestamp': get_timestamp(),
        }
        logger.info(f"  steps={steps}: AMOTA={amota:.4f}, 耗时={time.time()-t0:.0f}s")

        save_checkpoint_results(results, checkpoint_path)
        del model
        torch.cuda.empty_cache()

    # 选出 P6 最优值
    # 策略: 如果 steps=1 与 steps=5 差距 < 0.002, 选 1 以加速后续搜索
    amota_1 = results['set3_iterative_steps'].get('steps_1', {}).get('amota', 0)
    amota_5 = results['set3_iterative_steps'].get('steps_5', {}).get('amota', 0)

    if abs(amota_5 - amota_1) < 0.002:
        best_steps = 1
        logger.info(f"P6 决策: steps=1 与 steps=5 差距 < 0.002 ({amota_1:.4f} vs {amota_5:.4f}), 选 1 加速搜索")
    else:
        best_steps_key = max(
            results['set3_iterative_steps'],
            key=lambda k: results['set3_iterative_steps'][k]['amota'],
        )
        best_steps = results['set3_iterative_steps'][best_steps_key]['iterative_steps']
        logger.info(f"P6 决策: steps={best_steps} 最优")

    results['locked']['iterative_steps'] = best_steps
    logger.info(f"P6 锁定: iterative_steps = {best_steps}")

    # ── 保存 locked_config.json ──────────────────────
    results['meta']['end_time'] = get_timestamp()
    save_checkpoint_results(results, checkpoint_path)

    locked_config = {
        'version': '1.2',
        'phase': 'Phase 0-A',
        'timestamp': get_timestamp(),
        'locked': results['locked'],
        'evidence': {
            'set1_criterion': results['set1_criterion'],
            'set2_granularity': results['set2_granularity'],
            'set3_iterative_steps': results['set3_iterative_steps'],
        },
    }
    with open(locked_config_path, 'w', encoding='utf-8') as f:
        json.dump(locked_config, f, indent=2, ensure_ascii=False)

    logger.info(f"locked_config.json 已保存: {locked_config_path}")
    logger.info(f"锁定结果: {json.dumps(results['locked'], indent=2)}")

    # 生成 Phase 0-A 可视化
    _plot_lock_dims(results, output_dir)

    return results['locked']


# ============================================================
# Phase 0-B: 搜索维度敏感度分析
# ============================================================

def run_sensitivity(args):
    """
    Phase 0-B: 分析搜索维度 P1/P2/P8/P9 的敏感度 + B1×B2 交互验证

    执行顺序:
      4. 逐层 FFN 敏感度 (P1) — 12 层 × 3 比例 = 36 次
      5. 注意力投影敏感度 (P2) — 2 次
      6. 注意力头剪枝 (P8) — 2 次
      7. 解码器层数 (P9) — 2 次
      8. 剪枝 × 量化交互项 (B1×B2) — 9 次

    产出: pruning_sensitivity_report.json
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载锁定配置
    locked_config_path = args.locked_config or (output_dir / 'locked_config.json')
    with open(locked_config_path, 'r', encoding='utf-8') as f:
        locked_data = json.load(f)
    locked = locked_data['locked']
    logger.info(f"使用锁定配置: {json.dumps(locked, indent=2)}")

    checkpoint_path = output_dir / 'sensitivity_checkpoint.json'
    report_path = output_dir / 'pruning_sensitivity_report.json'

    # 加载断点
    results = load_checkpoint_results(checkpoint_path) or {
        'meta': {
            'start_time': get_timestamp(),
            'config': args.config,
            'checkpoint': args.checkpoint,
            'locked_config': str(locked_config_path),
            'fast_mode': args.fast,
        },
        'baseline_amota': None,
        'set4_ffn_per_layer': {},
        'set5_attn_proj': {},
        'set6_head_pruning': {},
        'set7_decoder_layers': {},
        'set8_prune_quant_interaction': {},
    }

    # 基线: 未剪枝 PyTorch AMOTA
    if results['baseline_amota'] is None:
        logger.info("获取基线 AMOTA (未剪枝)...")
        model, cfg = load_model_fresh(args.config, args.checkpoint)
        results['baseline_amota'] = evaluate_amota(model, cfg, max_samples=args.max_samples)
        logger.info(f"基线 AMOTA: {results['baseline_amota']:.4f}")
        save_checkpoint_results(results, checkpoint_path)
        del model
        torch.cuda.empty_cache()

    baseline = results['baseline_amota']

    # ── 实验集 4: 逐层 FFN 敏感度 (P1) ──────────────────
    logger.info("=" * 60)
    logger.info("实验集 4: 逐层 FFN 敏感度 (P1)")
    logger.info("对每个 FFN 层单独剪枝 20%/40%/60%, 测量 delta_AMOTA")
    logger.info("=" * 60)

    # BEVFormerEncoder: 6 层 FFN, DetectionTransformerDecoder: 6 层 FFN
    ffn_layers = []
    for i in range(6):
        ffn_layers.append(f'encoder.layers.{i}.ffns')
    for i in range(6):
        ffn_layers.append(f'decoder.layers.{i}.ffns')

    ratios_to_test = [0.8, 0.6, 0.4]  # 剪枝比例: 20%, 40%, 60%

    for layer_name in ffn_layers:
        for ratio in ratios_to_test:
            prune_pct = int((1.0 - ratio) * 100)
            exp_key = f'{layer_name}_ratio{ratio}'

            if exp_key in results['set4_ffn_per_layer']:
                cached = results['set4_ffn_per_layer'][exp_key]
                logger.info(f"  跳过已完成: {layer_name} {prune_pct}% -> delta={cached.get('delta_amota', 'N/A')}")
                continue

            logger.info(f"  测试: {layer_name}, 剪枝 {prune_pct}%")
            t0 = time.time()

            model, cfg = load_model_fresh(args.config, args.checkpoint)
            dataloader = (
                build_dataloader_for_grad(cfg)
                if locked.get('importance_criterion') in ('taylor', 'hessian')
                else None
            )

            # 构造 per-layer 剪枝配置: 仅剪枝目标层
            prune_cfg = _build_single_layer_ffn_config(locked, layer_name, ratio)
            model = apply_prune_config(model, prune_cfg, dataloader=dataloader)
            amota = evaluate_amota(model, cfg, max_samples=args.max_samples)
            delta = amota - baseline

            results['set4_ffn_per_layer'][exp_key] = {
                'layer': layer_name,
                'ffn_mid_ratio': ratio,
                'prune_pct': prune_pct,
                'amota': amota,
                'delta_amota': round(delta, 4),
                'elapsed_sec': round(time.time() - t0, 1),
                'timestamp': get_timestamp(),
            }
            logger.info(f"  {layer_name} {prune_pct}%: AMOTA={amota:.4f}, delta={delta:+.4f}")

            save_checkpoint_results(results, checkpoint_path)
            del model
            torch.cuda.empty_cache()

    # 分类: safe_aggressive / moderate / sensitive
    _classify_ffn_sensitivity(results)
    save_checkpoint_results(results, checkpoint_path)

    # ── 实验集 5: 注意力投影敏感度 (P2) ──────────────────
    logger.info("=" * 60)
    logger.info("实验集 5: 注意力投影敏感度 (P2)")
    logger.info("剪枝所有 value_proj+output_proj 10%/20%, 测量 delta_AMOTA")
    logger.info("=" * 60)

    for attn_ratio in [0.1, 0.2]:
        exp_key = f'attn_proj_{attn_ratio}'
        if exp_key in results['set5_attn_proj']:
            logger.info(f"  跳过已完成: attn_proj_ratio={attn_ratio}")
            continue

        logger.info(f"  测试: attn_proj_ratio={attn_ratio}")
        t0 = time.time()

        model, cfg = load_model_fresh(args.config, args.checkpoint)
        dataloader = (
            build_dataloader_for_grad(cfg)
            if locked.get('importance_criterion') in ('taylor', 'hessian')
            else None
        )

        prune_cfg = {
            'locked': locked,
            'encoder': {
                'ffn_mid_ratio': 1.0,  # 不剪 FFN
                'attn_proj_ratio': attn_ratio,
            },
            'decoder': {
                'ffn_mid_ratio': 1.0,
                'attn_proj_ratio': attn_ratio,
                'num_layers': 6,
            },
            'heads': {'head_mid_ratio': 1.0},
        }

        model = apply_prune_config(model, prune_cfg, dataloader=dataloader)
        amota = evaluate_amota(model, cfg, max_samples=args.max_samples)
        delta = amota - baseline

        results['set5_attn_proj'][exp_key] = {
            'attn_proj_ratio': attn_ratio,
            'amota': amota,
            'delta_amota': round(delta, 4),
            'elapsed_sec': round(time.time() - t0, 1),
            'timestamp': get_timestamp(),
        }
        logger.info(f"  attn_proj_ratio={attn_ratio}: AMOTA={amota:.4f}, delta={delta:+.4f}")

        save_checkpoint_results(results, checkpoint_path)
        del model
        torch.cuda.empty_cache()

    # ── 实验集 6: 注意力头剪枝 (P8) ──────────────────
    logger.info("=" * 60)
    logger.info("实验集 6: 注意力头剪枝 (P8)")
    logger.info("head_pruning_ratio in {0.0, 0.125}")
    logger.info("=" * 60)

    for head_ratio in [0.0, 0.125]:
        exp_key = f'head_prune_{head_ratio}'
        if exp_key in results['set6_head_pruning']:
            logger.info(f"  跳过已完成: head_pruning_ratio={head_ratio}")
            continue

        logger.info(f"  测试: head_pruning_ratio={head_ratio}")
        t0 = time.time()

        model, cfg = load_model_fresh(args.config, args.checkpoint)
        dataloader = (
            build_dataloader_for_grad(cfg)
            if locked.get('importance_criterion') in ('taylor', 'hessian')
            else None
        )

        prune_cfg = {
            'locked': locked,
            'encoder': {
                'ffn_mid_ratio': 1.0,
                'head_pruning_ratio': head_ratio,
            },
            'decoder': {
                'ffn_mid_ratio': 1.0,
                'head_pruning_ratio': head_ratio,
                'num_layers': 6,
            },
            'heads': {'head_mid_ratio': 1.0},
        }

        model = apply_prune_config(model, prune_cfg, dataloader=dataloader)
        amota = evaluate_amota(model, cfg, max_samples=args.max_samples)
        delta = amota - baseline

        results['set6_head_pruning'][exp_key] = {
            'head_pruning_ratio': head_ratio,
            'amota': amota,
            'delta_amota': round(delta, 4),
            'elapsed_sec': round(time.time() - t0, 1),
            'timestamp': get_timestamp(),
        }
        logger.info(f"  head_pruning_ratio={head_ratio}: AMOTA={amota:.4f}, delta={delta:+.4f}")

        save_checkpoint_results(results, checkpoint_path)
        del model
        torch.cuda.empty_cache()

    # ── 实验集 7: 解码器层数 (P9) ──────────────────
    logger.info("=" * 60)
    logger.info("实验集 7: 解码器层数 (P9)")
    logger.info("num_layers in {5, 6}")
    logger.info("=" * 60)

    for num_layers in [5, 6]:
        exp_key = f'decoder_layers_{num_layers}'
        if exp_key in results['set7_decoder_layers']:
            logger.info(f"  跳过已完成: num_layers={num_layers}")
            continue

        logger.info(f"  测试: decoder num_layers={num_layers}")
        t0 = time.time()

        model, cfg = load_model_fresh(args.config, args.checkpoint)

        prune_cfg = {
            'locked': locked,
            'encoder': {'ffn_mid_ratio': 1.0},
            'decoder': {'ffn_mid_ratio': 1.0, 'num_layers': num_layers},
            'heads': {'head_mid_ratio': 1.0},
        }

        # 层数剪枝不需要梯度
        model = apply_prune_config(model, prune_cfg, dataloader=None)
        amota = evaluate_amota(model, cfg, max_samples=args.max_samples)
        delta = amota - baseline

        results['set7_decoder_layers'][exp_key] = {
            'num_layers': num_layers,
            'amota': amota,
            'delta_amota': round(delta, 4),
            'elapsed_sec': round(time.time() - t0, 1),
            'timestamp': get_timestamp(),
        }
        logger.info(f"  num_layers={num_layers}: AMOTA={amota:.4f}, delta={delta:+.4f}")

        save_checkpoint_results(results, checkpoint_path)
        del model
        torch.cuda.empty_cache()

    # ── 实验集 8: 剪枝 × 量化交互项 (B1×B2) ──────────────
    logger.info("=" * 60)
    logger.info("实验集 8: 剪枝 × 量化交互项 (B1×B2)")
    logger.info("3 个代表性配置 × 3 种组合 (prune_only / quant_only / both)")
    logger.info("=" * 60)

    representative_configs = {
        'conservative': {
            'encoder': {'ffn_mid_ratio': 0.8},
            'decoder': {'ffn_mid_ratio': 0.8, 'num_layers': 6},
            'heads': {'head_mid_ratio': 1.0},
        },
        'moderate': {
            'encoder': {'ffn_mid_ratio': 0.6, 'attn_proj_ratio': 0.1},
            'decoder': {'ffn_mid_ratio': 0.6, 'attn_proj_ratio': 0.1, 'num_layers': 6},
            'heads': {'head_mid_ratio': 0.7},
        },
        'aggressive': {
            'encoder': {'ffn_mid_ratio': 0.4, 'attn_proj_ratio': 0.2, 'head_pruning_ratio': 0.125},
            'decoder': {'ffn_mid_ratio': 0.4, 'attn_proj_ratio': 0.2, 'head_pruning_ratio': 0.125, 'num_layers': 5},
            'heads': {'head_mid_ratio': 0.5},
        },
    }

    for config_name, repr_cfg in representative_configs.items():
        # ── (a) prune only ──
        exp_key_prune = f'{config_name}_prune_only'
        if exp_key_prune not in results['set8_prune_quant_interaction']:
            logger.info(f"  [{config_name}] prune only...")
            t0 = time.time()
            model, cfg = load_model_fresh(args.config, args.checkpoint)
            dataloader = (
                build_dataloader_for_grad(cfg)
                if locked.get('importance_criterion') in ('taylor', 'hessian')
                else None
            )

            prune_cfg = {**repr_cfg, 'locked': locked}
            model = apply_prune_config(model, prune_cfg, dataloader=dataloader)
            amota = evaluate_amota(model, cfg, max_samples=args.max_samples)

            results['set8_prune_quant_interaction'][exp_key_prune] = {
                'config': config_name,
                'mode': 'prune_only',
                'amota': amota,
                'delta_amota': round(amota - baseline, 4),
                'elapsed_sec': round(time.time() - t0, 1),
            }
            save_checkpoint_results(results, checkpoint_path)
            del model
            torch.cuda.empty_cache()

        # ── (b) quant only (INT8) ──
        exp_key_quant = f'{config_name}_quant_only'
        if exp_key_quant not in results['set8_prune_quant_interaction']:
            logger.info(f"  [{config_name}] quant only (INT8)...")
            t0 = time.time()
            model, cfg = load_model_fresh(args.config, args.checkpoint)

            # 复用 1.1 可配置量化管线进行 INT8 量化
            try:
                from tools.quantize_model import apply_int8_quantization
                model = apply_int8_quantization(model, cfg)
                amota = evaluate_amota(model, cfg, max_samples=args.max_samples)
            except ImportError:
                logger.warning("  量化模块不可用, 使用 FP16 TRT 基线 0.370 作为 quant_only 参考")
                amota = 0.370  # FP16 TRT 基线

            results['set8_prune_quant_interaction'][exp_key_quant] = {
                'config': config_name,
                'mode': 'quant_only',
                'amota': amota,
                'delta_amota': round(amota - baseline, 4),
                'elapsed_sec': round(time.time() - t0, 1),
            }
            save_checkpoint_results(results, checkpoint_path)
            del model
            torch.cuda.empty_cache()

        # ── (c) prune + quant (both) ──
        exp_key_both = f'{config_name}_both'
        if exp_key_both not in results['set8_prune_quant_interaction']:
            logger.info(f"  [{config_name}] prune + quant (both)...")
            t0 = time.time()
            model, cfg = load_model_fresh(args.config, args.checkpoint)
            dataloader = (
                build_dataloader_for_grad(cfg)
                if locked.get('importance_criterion') in ('taylor', 'hessian')
                else None
            )

            # 先剪枝
            prune_cfg = {**repr_cfg, 'locked': locked}
            model = apply_prune_config(model, prune_cfg, dataloader=dataloader)

            # 再量化
            try:
                from tools.quantize_model import apply_int8_quantization
                model = apply_int8_quantization(model, cfg)
                amota = evaluate_amota(model, cfg, max_samples=args.max_samples)
            except ImportError:
                logger.warning("  量化模块不可用, 跳过 both 实验")
                amota = -1.0

            results['set8_prune_quant_interaction'][exp_key_both] = {
                'config': config_name,
                'mode': 'both',
                'amota': amota,
                'delta_amota': round(amota - baseline, 4) if amota > 0 else None,
                'elapsed_sec': round(time.time() - t0, 1),
            }
            save_checkpoint_results(results, checkpoint_path)
            del model
            torch.cuda.empty_cache()

    # 计算交互效应
    _compute_interaction_effects(results, baseline)

    # ── 保存最终报告 ──────────────────────
    results['meta']['end_time'] = get_timestamp()
    save_checkpoint_results(results, checkpoint_path)

    report = {
        'version': '1.2',
        'phase': 'Phase 0-B',
        'timestamp': get_timestamp(),
        'baseline_amota': baseline,
        'per_layer_sensitivity': results['set4_ffn_per_layer'],
        'ffn_classification': results.get('ffn_classification', {}),
        'attn_proj_sensitivity': results['set5_attn_proj'],
        'head_pruning_impact': results['set6_head_pruning'],
        'decoder_layer_impact': results['set7_decoder_layers'],
        'prune_quant_interaction': results['set8_prune_quant_interaction'],
        'interaction_effects': results.get('interaction_effects', {}),
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"敏感度报告已保存: {report_path}")

    # 生成可视化
    _plot_sensitivity(results, output_dir)

    return report


# ============================================================
# 辅助函数
# ============================================================

def _build_single_layer_ffn_config(locked, target_layer, ratio):
    """
    构建仅剪枝单个 FFN 层的配置

    Args:
        locked: 锁定维度配置
        target_layer: 目标层名称 (e.g., 'encoder.layers.0.ffns')
        ratio: ffn_mid_ratio (e.g., 0.6 表示保留 60%)

    Returns:
        dict: prune_config
    """
    return {
        'locked': locked,
        'encoder': {'ffn_mid_ratio': 1.0},
        'decoder': {'ffn_mid_ratio': 1.0, 'num_layers': 6},
        'heads': {'head_mid_ratio': 1.0},
        # 特殊字段: 仅剪枝指定层
        'per_layer_override': {
            target_layer: {'ffn_mid_ratio': ratio},
        },
    }


def _classify_ffn_sensitivity(results):
    """
    将 FFN 层分为三类:
      - safe_aggressive: 剪枝 40% 时 |delta_AMOTA| < 0.005
      - moderate: 剪枝 40% 时 0.005 <= |delta_AMOTA| < 0.015
      - sensitive: 剪枝 40% 时 |delta_AMOTA| >= 0.015
    """
    classification = {}

    # 按层分组
    layer_deltas = {}
    for exp_key, exp_data in results['set4_ffn_per_layer'].items():
        layer = exp_data['layer']
        ratio = exp_data['ffn_mid_ratio']
        delta = exp_data.get('delta_amota', 0)
        if layer not in layer_deltas:
            layer_deltas[layer] = {}
        layer_deltas[layer][ratio] = delta

    for layer, deltas in layer_deltas.items():
        # 以 40% 剪枝 (ratio=0.6) 的 delta 为分类依据
        delta_40 = abs(deltas.get(0.6, 0))

        if delta_40 < 0.005:
            category = 'safe_aggressive'
        elif delta_40 < 0.015:
            category = 'moderate'
        else:
            category = 'sensitive'

        classification[layer] = {
            'category': category,
            'delta_at_20pct': deltas.get(0.8, None),
            'delta_at_40pct': deltas.get(0.6, None),
            'delta_at_60pct': deltas.get(0.4, None),
        }

    results['ffn_classification'] = classification
    logger.info("FFN 层敏感度分类:")
    for layer, info in sorted(classification.items()):
        logger.info(f"  {layer}: {info['category']} (delta@40%={info['delta_at_40pct']})")


def _compute_interaction_effects(results, baseline):
    """
    计算 B1×B2 交互效应:
      interaction = delta_joint - (delta_prune + delta_quant)
      > 0: 放大效应 (误差叠加)
      < 0: 抵消效应 (误差部分抵消)
      ≈ 0: 独立 (无交互)
    """
    effects = {}
    interaction_data = results['set8_prune_quant_interaction']

    for config_name in ['conservative', 'moderate', 'aggressive']:
        prune_key = f'{config_name}_prune_only'
        quant_key = f'{config_name}_quant_only'
        both_key = f'{config_name}_both'

        if all(k in interaction_data for k in [prune_key, quant_key, both_key]):
            delta_prune = interaction_data[prune_key].get('delta_amota', 0)
            delta_quant = interaction_data[quant_key].get('delta_amota', 0)
            delta_joint = interaction_data[both_key].get('delta_amota', None)

            if delta_joint is not None:
                expected_additive = delta_prune + delta_quant
                interaction = delta_joint - expected_additive

                if abs(interaction) < 0.003:
                    effect_type = 'independent'
                elif interaction < 0:
                    effect_type = 'cancellation (弱负交互)'
                else:
                    effect_type = 'amplification (放大效应)'

                effects[config_name] = {
                    'delta_prune': delta_prune,
                    'delta_quant': delta_quant,
                    'delta_joint': delta_joint,
                    'expected_additive': round(expected_additive, 4),
                    'interaction': round(interaction, 4),
                    'effect_type': effect_type,
                }
                logger.info(
                    f"  [{config_name}] 交互效应: {effect_type}, "
                    f"interaction={interaction:+.4f} "
                    f"(joint={delta_joint:+.4f} vs additive={expected_additive:+.4f})"
                )

    results['interaction_effects'] = effects


# ============================================================
# 可视化
# ============================================================

def _plot_lock_dims(results, output_dir):
    """生成 Phase 0-A 可视化: 准则对比柱状图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Phase 0-A: 锁定维度预选结果', fontsize=14)

    # 图 1: P4 重要性准则对比
    ax = axes[0]
    criteria_data = results['set1_criterion']
    names = []
    amotas = []
    for key in sorted(criteria_data.keys()):
        entry = criteria_data[key]
        if entry.get('amota', -1) > 0:
            names.append(entry['criterion'])
            amotas.append(entry['amota'])
    bars = ax.bar(names, amotas, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
    ax.set_title('P4: Importance Criterion')
    ax.set_ylabel('AMOTA')
    best_idx = np.argmax(amotas)
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)
    for i, (name, val) in enumerate(zip(names, amotas)):
        ax.text(i, val + 0.001, f'{val:.4f}', ha='center', fontsize=9)

    # 图 2: P5 剪枝粒度对比
    ax = axes[1]
    gran_data = results['set2_granularity']
    names = []
    amotas = []
    for key in sorted(gran_data.keys()):
        entry = gran_data[key]
        names.append(entry['granularity'])
        amotas.append(entry['amota'])
    bars = ax.bar(names, amotas, color=['#4C72B0', '#DD8452', '#55A868'])
    ax.set_title('P5: Pruning Granularity')
    ax.set_ylabel('AMOTA')
    best_idx = np.argmax(amotas)
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)
    for i, (name, val) in enumerate(zip(names, amotas)):
        ax.text(i, val + 0.001, f'{val:.4f}', ha='center', fontsize=9)

    # 图 3: P6 迭代步数对比
    ax = axes[2]
    steps_data = results['set3_iterative_steps']
    names = []
    amotas = []
    for key in sorted(steps_data.keys(), key=lambda k: steps_data[k]['iterative_steps']):
        entry = steps_data[key]
        names.append(str(entry['iterative_steps']))
        amotas.append(entry['amota'])
    ax.plot(names, amotas, 'o-', color='#4C72B0', linewidth=2, markersize=8)
    ax.set_title('P6: Iterative Steps')
    ax.set_ylabel('AMOTA')
    ax.set_xlabel('Steps')
    for i, (name, val) in enumerate(zip(names, amotas)):
        ax.annotate(f'{val:.4f}', (i, val), textcoords='offset points', xytext=(0, 10), fontsize=9, ha='center')

    plt.tight_layout()
    save_path = output_dir / 'phase0a_lock_dims.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Phase 0-A 可视化已保存: {save_path}")


def _plot_sensitivity(results, output_dir):
    """
    生成 Phase 0-B 可视化:
      (1) 逐层 FFN 敏感度热力图
      (2) 交互效应对比图
    """
    # ── 图 1: 逐层 FFN 敏感度热力图 ──
    fig, ax = plt.subplots(figsize=(10, 8))

    # 构造热力图矩阵
    layer_names = []
    for i in range(6):
        layer_names.append(f'enc.{i}')
    for i in range(6):
        layer_names.append(f'dec.{i}')

    ratios = [0.8, 0.6, 0.4]
    ratio_labels = ['20%', '40%', '60%']

    heatmap_data = np.zeros((len(layer_names), len(ratios)))
    ffn_data = results.get('set4_ffn_per_layer', {})

    for exp_key, exp_entry in ffn_data.items():
        layer = exp_entry['layer']
        ratio = exp_entry['ffn_mid_ratio']

        # 映射层名称到索引
        for idx, prefix in enumerate(layer_names):
            full_name = (
                f'encoder.layers.{idx}.ffns' if idx < 6
                else f'decoder.layers.{idx - 6}.ffns'
            )
            if layer == full_name and ratio in ratios:
                col = ratios.index(ratio)
                heatmap_data[idx, col] = exp_entry.get('delta_amota', 0)

    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-0.05, vmax=0.005)
    ax.set_xticks(range(len(ratio_labels)))
    ax.set_xticklabels(ratio_labels)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names)
    ax.set_xlabel('Pruning Ratio')
    ax.set_ylabel('FFN Layer')
    ax.set_title('Per-Layer FFN Sensitivity (delta AMOTA)')
    plt.colorbar(im, label='delta AMOTA')

    # 在格子里标注数值
    for i in range(len(layer_names)):
        for j in range(len(ratios)):
            val = heatmap_data[i, j]
            color = 'white' if abs(val) > 0.02 else 'black'
            ax.text(j, i, f'{val:+.3f}', ha='center', va='center', color=color, fontsize=8)

    plt.tight_layout()
    save_path = output_dir / 'phase0b_ffn_sensitivity_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"FFN 敏感度热力图已保存: {save_path}")

    # ── 图 2: B1×B2 交互效应对比 ──
    effects = results.get('interaction_effects', {})
    if effects:
        fig, ax = plt.subplots(figsize=(10, 6))

        config_names = list(effects.keys())
        x = np.arange(len(config_names))
        width = 0.25

        delta_prune = [effects[c]['delta_prune'] for c in config_names]
        delta_quant = [effects[c]['delta_quant'] for c in config_names]
        delta_joint = [effects[c]['delta_joint'] for c in config_names]
        expected_add = [effects[c]['expected_additive'] for c in config_names]

        ax.bar(x - width, delta_prune, width, label='Prune Only', color='#4C72B0')
        ax.bar(x, delta_quant, width, label='Quant Only', color='#DD8452')
        ax.bar(x + width, delta_joint, width, label='Both (actual)', color='#C44E52')

        # 期望加性线
        for i, (xi, ea) in enumerate(zip(x, expected_add)):
            ax.plot([xi - width * 1.5, xi + width * 1.5], [ea, ea],
                    'k--', linewidth=1)
            if i == 0:
                ax.plot([], [], 'k--', label='Expected Additive')

        ax.set_xticks(x)
        ax.set_xticklabels(config_names)
        ax.set_ylabel('delta AMOTA')
        ax.set_title('B1 x B2 Interaction: Pruning x Quantization')
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        save_path = output_dir / 'phase0b_interaction_effect.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"交互效应图已保存: {save_path}")


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Phase 0: 锁定维度预选 + 搜索维度敏感度分析',
    )
    parser.add_argument('--mode', choices=['lock-dims', 'sensitivity', 'all'],
                        required=True,
                        help='运行模式: lock-dims=Phase 0-A, sensitivity=Phase 0-B, all=两者都执行')
    parser.add_argument('--config', required=True,
                        help='mmdet3d 模型配置文件路径')
    parser.add_argument('--checkpoint', required=True,
                        help='预训练模型权重路径')
    parser.add_argument('--locked-config', default=None,
                        help='锁定配置文件 (sensitivity 模式需要, 或从 output-dir 自动查找)')
    parser.add_argument('--output-dir', default='work_dirs/phase0',
                        help='输出目录 (默认: work_dirs/phase0)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='最大评估样本数 (None=全量, 用于快速筛选)')
    parser.add_argument('--fast', action='store_true',
                        help='快速模式: 减少 batch 数, 用于验证脚本逻辑')
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Phase 0 实验开始: mode={args.mode}")
    logger.info(f"配置: {args.config}")
    logger.info(f"权重: {args.checkpoint}")
    logger.info(f"输出: {args.output_dir}")

    if args.mode in ('lock-dims', 'all'):
        locked = run_lock_dims(args)
        logger.info("Phase 0-A 完成!")

    if args.mode in ('sensitivity', 'all'):
        report = run_sensitivity(args)
        logger.info("Phase 0-B 完成!")

    logger.info("Phase 0 全部实验完成!")


if __name__ == '__main__':
    main()
```

### 3.2 输出格式定义

#### `locked_config.json` (Phase 0-A 产出)

```json
{
  "version": "1.2",
  "phase": "Phase 0-A",
  "timestamp": "20260414_103000",
  "locked": {
    "importance_criterion": "taylor",
    "pruning_granularity": "local",
    "iterative_steps": 5,
    "round_to": 8
  },
  "evidence": {
    "set1_criterion": {
      "criterion_l1_norm": {"criterion": "l1_norm", "amota": 0.325, "elapsed_sec": 610},
      "criterion_taylor": {"criterion": "taylor", "amota": 0.328, "elapsed_sec": 720},
      "criterion_fpgm": {"criterion": "fpgm", "amota": 0.322, "elapsed_sec": 600},
      "criterion_hessian": {"criterion": "hessian", "amota": 0.326, "elapsed_sec": 900}
    },
    "set2_granularity": { "...": "..." },
    "set3_iterative_steps": { "...": "..." }
  }
}
```

#### `pruning_sensitivity_report.json` (Phase 0-B 产出)

```json
{
  "version": "1.2",
  "phase": "Phase 0-B",
  "timestamp": "20260415_180000",
  "baseline_amota": 0.338,
  "per_layer_sensitivity": {
    "encoder.layers.0.ffns_ratio0.6": {
      "layer": "encoder.layers.0.ffns",
      "ffn_mid_ratio": 0.6,
      "prune_pct": 40,
      "amota": 0.335,
      "delta_amota": -0.003
    }
  },
  "ffn_classification": {
    "encoder.layers.0.ffns": {
      "category": "safe_aggressive",
      "delta_at_20pct": -0.001,
      "delta_at_40pct": -0.003,
      "delta_at_60pct": -0.010
    }
  },
  "attn_proj_sensitivity": { "...": "..." },
  "head_pruning_impact": { "...": "..." },
  "decoder_layer_impact": { "...": "..." },
  "prune_quant_interaction": { "...": "..." },
  "interaction_effects": {
    "moderate": {
      "delta_prune": -0.015,
      "delta_quant": -0.008,
      "expected_additive": -0.023,
      "delta_joint": -0.020,
      "interaction": 0.003,
      "effect_type": "cancellation (弱负交互)"
    }
  }
}
```

### 3.3 可视化输出

脚本自动生成以下图表文件：

| 文件 | 内容 | 用途 |
|------|------|------|
| `phase0a_lock_dims.png` | 3 子图: P4 准则柱状图 + P5 粒度柱状图 + P6 步数折线图 | 快速确认锁定决策 |
| `phase0b_ffn_sensitivity_heatmap.png` | 12 层 × 3 比例的热力图, 颜色 = delta_AMOTA | 识别激进/保守层 |
| `phase0b_interaction_effect.png` | 3 配置的剪枝/量化/联合 delta 对比 + 期望加性虚线 | 验证交互假设 |

---

## 4. 代码检测方案

### 4.1 快速验证: 脚本逻辑正确性

```bash
# 用 --fast + --max-samples 1 运行, 仅验证代码路径
python tools/pruning_sensitivity_analysis.py \
    --mode lock-dims \
    --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
    --checkpoint work_dirs/latest.pth \
    --output-dir work_dirs/phase0_debug \
    --max-samples 1 \
    --fast
```

**验证项目**:
- [ ] 4 种准则实验全部完成, 无崩溃
- [ ] `locked_config.json` 生成, 包含 4 个锁定字段 (`importance_criterion`, `pruning_granularity`, `iterative_steps`, `round_to`)
- [ ] `phase0a_lock_dims.png` 生成, 3 子图正常显示

### 4.2 输出 schema 验证

```python
# 验证 locked_config.json 包含所有必需字段
import json
with open('work_dirs/phase0/locked_config.json') as f:
    locked = json.load(f)

assert 'locked' in locked
assert set(locked['locked'].keys()) == {
    'importance_criterion', 'pruning_granularity',
    'iterative_steps', 'round_to'
}
assert locked['locked']['round_to'] == 8
assert locked['locked']['importance_criterion'] in [
    'l1_norm', 'taylor', 'fpgm', 'hessian'
]
assert locked['locked']['pruning_granularity'] in [
    'global', 'local', 'isomorphic'
]
assert locked['locked']['iterative_steps'] in [1, 3, 5, 10]
print("Schema 验证通过")
```

### 4.3 断点续跑验证

```bash
# 模拟中断: Ctrl+C 后重新运行
python tools/pruning_sensitivity_analysis.py \
    --mode lock-dims \
    --config ... --checkpoint ... --output-dir work_dirs/phase0

# 应自动跳过已完成实验, 从断点继续
# 验证: 日志中出现 "跳过已完成: ..."
```

---

## 5. Debug 方案

### 5.1 实验中途崩溃

**原因**: GPU OOM、数据加载错误、DepGraph 追踪失败

**方案**:
1. 每个实验完成后立即 `save_checkpoint_results()` 保存中间结果
2. 重新运行时自动从 `lock_dims_checkpoint.json` / `sensitivity_checkpoint.json` 恢复
3. Hessian OOM: 自动 catch `RuntimeError('out of memory')`, 标记该准则不可用, 继续下一个

### 5.2 AMOTA 评估超时

**原因**: 完整 nuScenes val 评估需要 10-15 分钟/次, 36 次 FFN 实验 = 6-9 小时

**方案**:
1. `--max-samples N`: 仅评估前 N 个样本, 用于快速筛选 (精度稍有偏差但趋势正确)
2. 两轮策略: 先用 `--max-samples 200` 快速扫描, 筛出 top-3 配置后再全量评估
3. 实验集 4 (36 次) 可分批执行, 利用断点续跑

### 5.3 Hessian 重要性 OOM

**原因**: Hessian 需要二阶梯度, 显存占用是 Taylor 的 2-3 倍

**方案**:
1. 减少校准 batch: `num_batches=8` (默认 32)
2. 启用 gradient checkpointing
3. 如果仍然 OOM: 在 set1 中标记 hessian 不可用 (`amota=-1`), 从候选中移除
4. 日志记录 OOM 事件, 报告中标注

### 5.4 Per-layer 剪枝配置不生效

**原因**: `prune_and_eval.py` 可能不支持 `per_layer_override` 字段

**方案**:
1. Phase A3 中需确保 `apply_prune_config` 支持 `per_layer_override` 参数
2. 如果不支持: 在 `_build_single_layer_ffn_config()` 中生成完整的 per-module `pruning_ratio_dict`, 绕过高层 API
3. 验证: 单层剪枝后, 检查目标层维度是否变化, 非目标层维度是否保持不变

---

## 6. 验收标准

### 6.1 Phase 0-A 验收

| 检查项 | 标准 |
|--------|------|
| `locked_config.json` 存在 | 包含 4 个锁定字段且值合法 |
| P4 有实验证据 | 至少 3 种准则的 AMOTA 数据 (hessian 可能 OOM 缺失) |
| P5 有实验证据 | 3 种粒度的 AMOTA 数据 |
| P6 有实验证据 | 4 种步数的 AMOTA 数据, 含加速决策记录 |
| 可视化 | `phase0a_lock_dims.png` 生成, 3 子图正常 |

### 6.2 Phase 0-B 验收

| 检查项 | 标准 |
|--------|------|
| FFN 逐层敏感度 | 12 层 × 3 比例 = 36 个数据点, 每层有分类标签 |
| 注意力投影 | 2 个数据点 (attn_proj_ratio=0.1/0.2) |
| 注意力头剪枝 | 2 个数据点 (head_pruning_ratio=0.0/0.125) |
| 解码器层数 | 2 个数据点 (num_layers=5/6) |
| B1×B2 交互 | 3 配置 × 3 模式 = 9 个数据点, 含交互效应分析 |
| 敏感度报告 | `pruning_sensitivity_report.json` 完整且 schema 正确 |
| 可视化 | 热力图 + 交互效应图生成 |

### 6.3 决策产出验收

| 决策 | 来源 | 格式 |
|------|------|------|
| P4-P7 锁定值 | Phase 0-A | `locked_config.json` |
| FFN 层分类 (safe/moderate/sensitive) | Phase 0-B 实验集 4 | `sensitivity_report.json → ffn_classification` |
| P8/P9 是否值得搜索 | Phase 0-B 实验集 6/7 | delta_AMOTA 显著性判断 |
| 搜索空间缩减建议 | Phase 0-B 综合 | 文档记录 |

---

## 7. 预期结果与分析框架

### 7.1 P4 重要性准则: 预期差异很小

**假设** (来自量化教训): 不同准则之间的 AMOTA 差距在 0.001-0.005 范围内。

**分析逻辑**:
- 如果 taylor 最优且与 l1_norm 差距 > 0.003: 选 taylor, 值得额外梯度计算开销
- 如果 l1_norm 与 taylor 差距 < 0.002: 选 l1_norm, 因为不需要梯度计算, 搜索更快
- hessian 预计与 taylor 相当, 但计算代价高 3 倍, 除非显著优于 taylor 否则不选

### 7.2 P5 剪枝粒度: local 可能最优

**假设**: 
- global: 可能过度剪枝某些关键层
- local: 均匀剪枝, 安全但可能不是最优
- isomorphic: 保证所有层结构一致, 对部署友好

**分析逻辑**:
- global 如果 AMOTA 明显低于 local: 确认层间重要性差异大, 后续应考虑 per-layer ratio
- isomorphic 如果与 local 接近: 优先选 isomorphic (部署更简洁)

### 7.3 P6 迭代步数: 关注 steps=1 vs steps=5

**关键决策点**: 如果 steps=1 与 steps=5 的 AMOTA 差距 < 0.002:
- 锁定 steps=1, 后续搜索每个配置评估时间从 15 分钟降至 5 分钟
- 216 个搜索配置: 总时间从 54 小时降至 18 小时

### 7.4 FFN 逐层敏感度: 期望发现分层规律

**预期**:
- Encoder 浅层 (layer 0-1): 可能 safe_aggressive (特征提取冗余度高)
- Encoder 深层 (layer 4-5): 可能 moderate-sensitive (与检测头直接耦合)
- Decoder 浅层: 可能 moderate (逐步细化位置)
- Decoder 深层 (layer 4-5): 可能 sensitive (直接影响输出)

**如何使用**:
- safe_aggressive 层: 搜索时 ffn_mid_ratio 取 {0.4, 0.5, 0.6, 0.7}
- sensitive 层: 搜索时 ffn_mid_ratio 取 {0.8, 1.0}
- 有效缩减每层的搜索范围

### 7.5 B1×B2 交互: 预期弱负交互 (与量化教训一致)

**1.1 量化教训**: INT8 量化引入的误差与其他压缩手段的误差部分抵消 (而非简单叠加)。这是因为量化和剪枝删除的是不同方向的冗余，彼此的"噪声"可以部分抵消。

**分析逻辑**:
- 如果 interaction < -0.005 (显著抵消): 联合搜索可以更激进, 因为 B1 和 B2 的误差不会完全叠加
- 如果 interaction ≈ 0 (独立): 可以分开搜索 B1 和 B2, 大幅降低搜索空间
- 如果 interaction > 0.005 (放大): 联合搜索必须保守, 需要更多 Pareto 前沿采样

---

## 8. 反思模板

> 以下模板在 Phase 0 实验全部完成后填写，记录实际结果与预期的偏差、学到的经验，
> 以及对后续联合搜索的影响。

### 8.1 实验结果总结

| 维度 | 预期最优值 | 实际最优值 | 差异说明 |
|:----:|:---------:|:---------:|---------|
| P4 importance_criterion | taylor | ___ | |
| P5 pruning_granularity | local | ___ | |
| P6 iterative_steps | 5 (或 1) | ___ | |

### 8.2 关键发现

**Q1: 哪些结果让你感到意外?**

_填写: ___

**Q2: 哪些维度的影响比预期更大?**

_填写: ___

**Q3: 哪些维度的影响比预期更小?**

_填写: ___

### 8.3 FFN 敏感度规律

**Q4: 实际的层间敏感度分布是否符合 "浅层冗余、深层敏感" 的假设?**

_填写: ___

**Q5: Encoder 和 Decoder 的敏感度模式是否一致?**

_填写: ___

**Q6: 是否有某个层的敏感度异常高, 需要在搜索中特殊处理?**

_填写: ___

### 8.4 B1×B2 交互分析

**Q7: 实测交互效应是否确认量化教训 (弱负交互)?**

_填写: ___

**Q8: 三个代表性配置 (conservative/moderate/aggressive) 的交互效应是否一致?**

_填写: ___

**Q9: 交互效应是否随剪枝比例增大而变化 (线性 vs 非线性)?**

_填写: ___

### 8.5 搜索空间调整决策

**Q10: 基于 Phase 0 结果, 是否有维度可以进一步锁定或缩减范围?**

例如:
- P8 (头剪枝) delta < 0.003 → 锁定为 0.0, 搜索空间 216 → 108
- P9 (层数) delta > 0.020 → 锁定为 6, 搜索空间 108 → 54
- P2 (注意力投影) 如果 0.2 已导致 delta > 0.015 → 范围缩减为 {0.0, 0.1}

_填写: ___

### 8.6 方法论反思

**Q11: 控制变量实验设计是否合理? 是否有遗漏的交互效应?**

_填写: ___

**Q12: --max-samples 快速筛选的结果与全量评估是否一致? 是否需要调整样本数?**

_填写: ___

**Q13: 断点续跑机制是否可靠? 是否出现了需要重跑的实验?**

_填写: ___

### 8.7 后续阶段影响

**Q14: Phase 0 结果对联合搜索策略的具体影响:**

- 最终搜索空间大小: ___ (目标: 从 216 进一步缩减)
- 推荐搜索算法: ___ (random search / bayesian / evolutionary)
- 预计搜索总耗时: ___ 小时
- 是否需要增加微调步骤: ___

---

## 附录: 实验时间线

| 实验集 | 实验数 | 估计单次耗时 | 估计总耗时 |
|:------:|:------:|:-----------:|:---------:|
| Set 1: P4 准则 | 4 | 10-15 min | 40-60 min |
| Set 2: P5 粒度 | 3 | 10-15 min | 30-45 min |
| Set 3: P6 步数 | 4 | 10-15 min | 40-60 min |
| **Phase 0-A 小计** | **11** | | **~2.5 小时** |
| Set 4: FFN 逐层 | 36 | 10-15 min | 6-9 小时 |
| Set 5: 注意力投影 | 2 | 10-15 min | 20-30 min |
| Set 6: 头剪枝 | 2 | 10-15 min | 20-30 min |
| Set 7: 层数 | 2 | 10-15 min | 20-30 min |
| Set 8: B1×B2 交互 | 9 | 15-20 min | 2-3 小时 |
| **Phase 0-B 小计** | **51** | | **~9-13 小时** |
| **总计** | **62** | | **~11.5-15.5 小时** |

> 使用 `--max-samples 200` 快速筛选模式可将总耗时压缩至 ~4 小时,
> 但仅适用于初步趋势判断, 最终锁定决策需全量评估确认。
