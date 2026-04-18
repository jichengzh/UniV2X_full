"""
Quick quantization config evaluation tool.

Given a quant_config.json, apply fake-quantization to the model and evaluate AMOTA
on a subset of the validation set. No TRT engine needed -- pure PyTorch evaluation.

Usage:
    python tools/quick_eval_quant.py \
        --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
        --checkpoint ckpts/univ2x_coop_e2e_stg2.pth \
        --quant-config quant_configs/default_int8.json \
        --eval-samples 17 \
        --cali-data calibration/bev_encoder_calib_inputs.pkl \
        [--export-scales]
"""

import argparse
import json
import os
import pickle
import sys
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Precision string helpers
# ---------------------------------------------------------------------------

_PRECISION_TO_BITS = {
    'int4': 4,
    'int8': 8,
    'fp16': 16,
    'fp32': 32,
    'none': 32,
}


def _parse_precision(precision_str: str) -> int:
    """Convert a precision string (e.g. 'int8', 'fp16') to a bit-width integer."""
    key = precision_str.strip().lower()
    if key not in _PRECISION_TO_BITS:
        raise ValueError(
            f"Unknown precision '{precision_str}'. "
            f"Valid options: {list(_PRECISION_TO_BITS.keys())}")
    return _PRECISION_TO_BITS[key]


def _granularity_to_channel_wise(granularity: str) -> bool:
    """Convert granularity string to the channel_wise bool used by UniformAffineQuantizer."""
    g = granularity.strip().lower()
    if g == 'per_channel':
        return True
    if g == 'per_tensor':
        return False
    raise ValueError(f"Unknown granularity '{granularity}'. Use 'per_channel' or 'per_tensor'.")


# ---------------------------------------------------------------------------
# Core: apply_quant_config
# ---------------------------------------------------------------------------

def apply_quant_config(
    model: nn.Module,
    quant_config: Dict[str, Any],
    *,
    is_fusing: bool = True,
) -> "QuantModel":
    """Apply a JSON quantization config to *model* and return a QuantModel.

    This is the single entry-point used by quick_eval_quant, sensitivity_analysis,
    and the search framework.  It is intentionally import-safe: the heavy
    ``projects.mmdet3d_plugin`` imports happen inside this function so callers can
    import ``apply_quant_config`` without triggering mmdet3d at module load time.

    Parameters
    ----------
    model : nn.Module
        The raw (FP32) model or sub-model to quantize.  Typically this is the
        BEV encoder extracted from the MultiAgent wrapper, but it can be any
        ``nn.Module``.
    quant_config : dict
        Parsed JSON config with ``global``, ``layers``, and optionally
        ``v2x_comm`` sections.  See ``quant_configs/default_int8.json`` for the
        schema.
    is_fusing : bool
        Whether to fold BatchNorm into Conv/Linear before quantizing.  Default
        ``True`` (recommended for inference).

    Returns
    -------
    QuantModel
        The wrapped model with per-layer quantization state applied.
    """
    from projects.mmdet3d_plugin.univ2x.quant import (
        QuantModel,
        QuantModule,
        BaseQuantBlock,
        CommQuantizer,
    )

    # ── 1. Read global defaults ──────────────────────────────────────────
    g = quant_config.get('global', {})
    symmetric = g.get('symmetric', True)
    scale_method = g.get('scale_method', 'minmax')
    default_w_bits = g.get('default_w_bits', 8)
    default_a_bits = g.get('default_a_bits', 8)
    default_w_granularity = g.get('default_w_granularity', 'per_tensor')
    default_a_granularity = g.get('default_a_granularity', 'per_tensor')
    default_quant_target = g.get('default_quant_target', 'W+A')

    weight_quant_params = dict(
        n_bits=default_w_bits,
        symmetric=symmetric,
        channel_wise=_granularity_to_channel_wise(default_w_granularity),
        scale_method=scale_method,
    )
    act_quant_params = dict(
        n_bits=default_a_bits,
        symmetric=symmetric,
        channel_wise=_granularity_to_channel_wise(default_a_granularity),
        scale_method=scale_method,
        leaf_param=True,
    )

    # ── 2. Wrap in QuantModel ────────────────────────────────────────────
    model.eval()
    qmodel = QuantModel(
        model, weight_quant_params, act_quant_params, is_fusing=is_fusing)

    # Enable quantization globally using the default quant_target
    w_quant_default, a_quant_default = _parse_quant_target(default_quant_target)
    qmodel.set_quant_state(weight_quant=w_quant_default, act_quant=a_quant_default)

    # ── 3. Apply per-layer overrides ─────────────────────────────────────
    layer_configs = quant_config.get('layers', {})
    if layer_configs:
        named_modules = dict(qmodel.model.named_modules())
        for layer_name, lcfg in layer_configs.items():
            module = named_modules.get(layer_name)
            if module is None:
                print(f'[apply_quant_config] WARNING: layer "{layer_name}" '
                      f'not found in model, skipping.')
                continue

            # Determine quant target for this layer
            layer_target = lcfg.get('quant_target', default_quant_target)
            w_quant, a_quant = _parse_quant_target(layer_target)

            if isinstance(module, (QuantModule, BaseQuantBlock)):
                module.set_quant_state(weight_quant=w_quant, act_quant=a_quant)

                # Apply weight quantizer overrides
                if isinstance(module, QuantModule):
                    _apply_quantizer_overrides(
                        module.weight_quantizer,
                        lcfg,
                        prefix='w',
                        defaults=dict(
                            bits=default_w_bits,
                            granularity=default_w_granularity,
                            scale_method=scale_method,
                        ),
                    )
                    # Apply activation quantizer overrides
                    _apply_quantizer_overrides(
                        module.act_quantizer,
                        lcfg,
                        prefix='a',
                        defaults=dict(
                            bits=default_a_bits,
                            granularity=default_a_granularity,
                            scale_method=scale_method,
                        ),
                    )
            else:
                # For non-quant modules that contain QuantModules, recurse
                for sub in module.modules():
                    if isinstance(sub, (QuantModule, BaseQuantBlock)):
                        sub.set_quant_state(weight_quant=w_quant, act_quant=a_quant)

    # ── 4. V2X communication quantization ────────────────────────────────
    v2x_cfg = quant_config.get('v2x_comm', {})
    comm_quantizers = {}
    if v2x_cfg:
        for key in ('agent_query_precision', 'lane_query_precision',
                     'bev_scatter_precision'):
            precision_str = v2x_cfg.get(key, 'fp16')
            bits = _parse_precision(precision_str)
            comm_quantizers[key] = CommQuantizer(
                n_bits=bits, symmetric=symmetric, scale_method='minmax')
        # Store on qmodel so callers can retrieve them
        qmodel._comm_quantizers = nn.ModuleDict(comm_quantizers)

    return qmodel


def _parse_quant_target(target: str) -> tuple:
    """Parse quant_target string into (weight_quant, act_quant) bools.

    Supported values:
        'W+A'  -> (True, True)
        'W'    -> (True, False)
        'A'    -> (False, True)
        'none' -> (False, False)
    """
    t = target.strip().upper()
    if t == 'W+A':
        return True, True
    if t == 'W':
        return True, False
    if t == 'A':
        return False, True
    if t == 'NONE':
        return False, False
    raise ValueError(
        f"Unknown quant_target '{target}'. Use 'W+A', 'W', 'A', or 'none'.")


def _apply_quantizer_overrides(
    quantizer,
    layer_cfg: Dict[str, Any],
    prefix: str,
    defaults: Dict[str, Any],
) -> None:
    """Apply per-layer overrides to a single UniformAffineQuantizer.

    Parameters
    ----------
    quantizer : UniformAffineQuantizer
        The weight or activation quantizer to configure.
    layer_cfg : dict
        The layer-specific config dict from the JSON.
    prefix : str
        'w' for weight quantizer, 'a' for activation quantizer.
    defaults : dict
        Global defaults for bits, granularity, scale_method.
    """
    # Bit-width
    bits_key = f'{prefix}_bits'
    bits = layer_cfg.get(bits_key, defaults['bits'])
    if bits != quantizer.n_bits:
        quantizer.bitwidth_refactor(bits)

    # Granularity -> channel_wise
    gran_key = f'{prefix}_granularity'
    granularity = layer_cfg.get(gran_key, defaults['granularity'])
    quantizer.channel_wise = _granularity_to_channel_wise(granularity)

    # Scale method
    sm = layer_cfg.get('scale_method', defaults['scale_method'])
    quantizer.scale_method = sm

    # Reset calibration state so scales are recomputed
    quantizer.inited = False


# ---------------------------------------------------------------------------
# Scale export
# ---------------------------------------------------------------------------

def export_scales_to_config(
    qmodel,
    quant_config: Dict[str, Any],
    output_path: str,
) -> None:
    """Write calibrated delta/zero_point back into a quant_config JSON file.

    After calibration (set_weight_quantize_params + activation calibration),
    each UniformAffineQuantizer holds computed ``delta`` and ``zero_point``.
    This function serialises those values into the layer config so that a
    subsequent run can skip calibration and load scales directly.
    """
    from projects.mmdet3d_plugin.univ2x.quant import QuantModule

    updated = deepcopy(quant_config)
    layers_section = updated.setdefault('layers', {})

    for name, module in qmodel.model.named_modules():
        if not isinstance(module, QuantModule):
            continue
        entry = layers_section.setdefault(name, {})

        # Weight scales
        wq = module.weight_quantizer
        if hasattr(wq, 'delta') and isinstance(wq.delta, torch.Tensor):
            entry['w_delta'] = wq.delta.cpu().tolist()
            entry['w_zero_point'] = wq.zero_point.cpu().tolist() \
                if isinstance(wq.zero_point, torch.Tensor) else float(wq.zero_point)

        # Activation scales
        aq = module.act_quantizer
        if hasattr(aq, 'delta') and isinstance(aq.delta, torch.Tensor):
            entry['a_delta'] = aq.delta.cpu().tolist()
            entry['a_zero_point'] = aq.zero_point.cpu().tolist() \
                if isinstance(aq.zero_point, torch.Tensor) else float(aq.zero_point)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(updated, f, indent=2)
    print(f'Exported calibrated scales -> {output_path}')


# ---------------------------------------------------------------------------
# Model loading (mirrors calibrate_univ2x.py)
# ---------------------------------------------------------------------------

def _load_model(cfg, checkpoint_path: str):
    """Load the full MultiAgent model from config + checkpoint.

    Returns (model_multi, agent, encoder) where encoder is the BEV encoder
    sub-module typically targeted for quantization.
    """
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model

    cfg.model_ego_agent.pretrained = None
    cfg.model_ego_agent.train_cfg = None

    other_agent_names = [k for k in cfg.keys() if 'model_other_agent' in k]
    model_other_agents = {}
    for name in other_agent_names:
        cfg.get(name).train_cfg = None
        m = build_model(cfg.get(name), test_cfg=cfg.get('test_cfg'))
        load_from = cfg.get(name).load_from
        if load_from:
            load_checkpoint(m, load_from, map_location='cpu',
                            revise_keys=[(r'^model_ego_agent\.', '')])
        model_other_agents[name] = m

    model_ego = build_model(cfg.model_ego_agent, test_cfg=cfg.get('test_cfg'))
    load_from = cfg.model_ego_agent.load_from
    if load_from:
        load_checkpoint(model_ego, load_from, map_location='cpu',
                        revise_keys=[(r'^model_ego_agent\.', '')])

    from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent
    model_multi = MultiAgent(model_ego, model_other_agents)
    load_checkpoint(model_multi, checkpoint_path, map_location='cpu',
                    revise_keys=[(r'^model_ego_agent\.', '')])
    model_multi.eval()
    model_multi.cuda()

    # Extract agent and encoder
    inner = model_multi.module if hasattr(model_multi, 'module') else model_multi
    agent = inner.model_ego_agent
    encoder = agent.pts_bbox_head.transformer.encoder
    return model_multi, agent, encoder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Quick quantization config evaluation (fake-quant, no TRT)')
    p.add_argument('--config', required=True,
                   help='Model config file (e.g. univ2x_coop_e2e_track_trt_p2.py)')
    p.add_argument('--checkpoint', required=True,
                   help='Checkpoint .pth file')
    p.add_argument('--quant-config', required=True,
                   help='Quantization config JSON file')
    p.add_argument('--eval-samples', type=int, default=17,
                   help='Number of validation samples to evaluate')
    p.add_argument('--cali-data', default='calibration/bev_encoder_calib_inputs.pkl',
                   help='Calibration data pickle for weight/activation scale calibration')
    p.add_argument('--export-scales', action='store_true',
                   help='Export calibrated scales back to a quant_config JSON')
    p.add_argument('--export-scales-path', default=None,
                   help='Output path for exported scales (default: <quant-config>.scales.json)')
    p.add_argument('--model', choices=['ego', 'infra'], default='ego',
                   help='Which agent model to quantize')
    p.add_argument('--prune-config', default=None,
                   help='Optional: apply pruning before quantization (剪枝×量化联合)')
    p.add_argument('--finetuned-ckpt', default=None,
                   help='Optional: load finetuned checkpoint after pruning')
    p.add_argument('--cfg-options', nargs='+', action='append',
                   help='Override config options')
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, '.')

    # ── Config ───────────────────────────────────────────────────────────
    from mmcv import Config
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        for kv in args.cfg_options:
            cfg.merge_from_dict(dict(kv))

    if hasattr(cfg, 'plugin') and cfg.plugin:
        if hasattr(cfg, 'plugin_dir'):
            from importlib import import_module
            import_module(cfg.plugin_dir.replace('/', '.').rstrip('.py'))

    # ── Register quantization specials ───────────────────────────────────
    from projects.mmdet3d_plugin.univ2x.quant import (
        register_bevformer_specials,
        register_fusion_specials,
        register_downstream_specials,
        set_weight_quantize_params,
    )
    register_bevformer_specials()
    register_fusion_specials()
    register_downstream_specials()
    print('Registered BEVFormer/fusion/downstream quantization specials.')

    # ── Load quant config ────────────────────────────────────────────────
    print(f'Loading quant config from {args.quant_config} ...')
    with open(args.quant_config, 'r') as f:
        quant_config = json.load(f)
    print(f'  Config version: {quant_config.get("version", "unknown")}')

    # ── Load model ───────────────────────────────────────────────────────
    print(f'Loading model from {args.checkpoint} ...')
    model_multi, agent, encoder = _load_model(cfg, args.checkpoint)
    print(f'  BEV encoder: {encoder.__class__.__name__}')

    # ── 可选: 剪枝 (在量化之前) ─────────────────────────────────────────
    if args.prune_config:
        print(f'\n[prune] Applying pruning: {args.prune_config}')
        from projects.mmdet3d_plugin.univ2x.pruning.prune_univ2x import apply_prune_config
        import json as _json
        with open(args.prune_config) as _f:
            _pcfg = _json.load(_f)
        _locked = _pcfg.setdefault('locked', {})
        _locked.setdefault('importance_criterion', 'l1_norm')
        _locked.setdefault('pruning_granularity', 'local')
        _locked.setdefault('iterative_steps', 5)
        _locked.setdefault('round_to', 8)
        _pcfg.setdefault('encoder', {})
        _pcfg.setdefault('decoder', {})
        _pcfg.setdefault('heads', {})
        _pcfg.setdefault('constraints', {
            'skip_layers': ['sampling_offsets', 'attention_weights'],
            'min_channels': 64, 'channel_alignment': 8,
        })
        n_before = sum(p.numel() for p in agent.parameters())
        apply_prune_config(agent, _pcfg, dataloader=None)
        n_after = sum(p.numel() for p in agent.parameters())
        print(f'[prune] {n_before:,} -> {n_after:,} (-{(1-n_after/n_before)*100:.2f}%)')

        if args.finetuned_ckpt:
            print(f'[prune] Loading finetuned ckpt: {args.finetuned_ckpt}')
            from mmcv.runner import load_checkpoint as _lc
            _lc(model_multi, args.finetuned_ckpt, map_location='cpu')
            print('[prune] Finetuned weights loaded.')

        # 重新获取 encoder 引用（剪枝可能改变了内部结构）
        encoder = agent.pts_bbox_head.transformer.encoder

    # ── Apply quant config ───────────────────────────────────────────────
    print('Applying quantization config ...')
    qmodel = apply_quant_config(encoder, quant_config, is_fusing=True)
    qmodel.cuda()
    qmodel.eval()
    print(f'  {qmodel.get_memory_footprint()}')

    # ── Calibrate weights ────────────────────────────────────────────────
    print('\nCalibrating weight quantization scales ...')
    # Temporarily disable act quant for weight calibration
    qmodel.set_quant_state(weight_quant=False, act_quant=False)
    set_weight_quantize_params(qmodel)
    print('  Weight scales calibrated.')

    # Re-enable quant state according to config
    g = quant_config.get('global', {})
    default_target = g.get('default_quant_target', 'W+A')
    w_q, a_q = _parse_quant_target(default_target)
    qmodel.set_quant_state(weight_quant=w_q, act_quant=a_q)

    # ── Activation calibration (if cali_data available) ──────────────────
    if os.path.exists(args.cali_data):
        print(f'\nLoading calibration data from {args.cali_data} ...')
        with open(args.cali_data, 'rb') as f:
            cali_data = pickle.load(f)
        print(f'  {len(cali_data)} calibration samples loaded.')
        print('  (Activation calibration requires BEVEncoderCalibModel '
              '-- see calibrate_univ2x.py for full pipeline.)')
    else:
        print(f'\nCalibration data not found at {args.cali_data}, '
              f'skipping activation calibration.')

    # ── Evaluate AMOTA ────────────────────────────────────────────────────
    print(f'\n--- Evaluation ({args.eval_samples} samples) ---')

    # 把量化后的 encoder 放回 agent (QuantModel wraps encoder)
    agent.pts_bbox_head.transformer.encoder = qmodel

    from mmcv.parallel import MMDataParallel
    from mmdet3d.apis import single_gpu_test
    from mmdet3d.datasets import build_dataset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from mmdet.datasets import replace_ImageToTensor

    eval_cfg = Config.fromfile(args.config)
    if hasattr(eval_cfg, 'plugin') and eval_cfg.plugin:
        pass  # already imported
    eval_cfg.data.test.test_mode = True
    if isinstance(eval_cfg.data.test, dict):
        eval_cfg.data.test.pipeline = replace_ImageToTensor(eval_cfg.data.test.pipeline)

    dataset = build_dataset(eval_cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0,
        dist=False, shuffle=False,
    )

    model_parallel = MMDataParallel(model_multi.cuda(), device_ids=[0])
    model_parallel.eval()

    with torch.no_grad():
        outputs = single_gpu_test(model_parallel, data_loader)

    # 如果 eval_samples < 全量，跳过 AMOTA（需要完整序列），只输出 mAP
    n_eval = len(outputs)

    # Evaluate
    eval_prefix = f'output/quant_eval_{os.path.basename(args.quant_config).replace(".json","")}'
    if args.prune_config:
        eval_prefix += f'_pruned_{os.path.basename(args.prune_config).replace(".json","")}'
    metrics = dataset.evaluate(outputs, jsonfile_prefix=eval_prefix)

    print(f'\n{"="*60}')
    print(f'  Quantization Evaluation Results')
    print(f'{"="*60}')
    key_metrics = [
        'pts_bbox_NuScenes/amota', 'pts_bbox_NuScenes/amotp',
        'pts_bbox_NuScenes/mAP', 'pts_bbox_NuScenes/NDS',
        'pts_bbox_NuScenes/recall', 'pts_bbox_NuScenes/motar',
        'pts_bbox_NuScenes/mota',
        'pts_bbox_NuScenes/tp', 'pts_bbox_NuScenes/fp',
        'pts_bbox_NuScenes/fn', 'pts_bbox_NuScenes/ids',
        'drivable_iou', 'lanes_iou', 'crossing_iou', 'contour_iou',
    ]
    for k in key_metrics:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                print(f'  {k:<40} {v:.4f}')
            else:
                print(f'  {k:<40} {v}')

    # Save metrics JSON
    metrics_path = f'{eval_prefix}_metrics.json'
    clean = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v == v}
    clean['_meta'] = {
        'quant_config': args.quant_config,
        'prune_config': args.prune_config,
        'finetuned_ckpt': args.finetuned_ckpt,
        'eval_samples': args.eval_samples,
    }
    with open(metrics_path, 'w') as f:
        json.dump(clean, f, indent=2, default=str)
    print(f'\n  Metrics saved: {metrics_path}')

    # ── Export scales ────────────────────────────────────────────────────
    if args.export_scales:
        out_path = args.export_scales_path
        if out_path is None:
            base, ext = os.path.splitext(args.quant_config)
            out_path = f'{base}.scales{ext}'
        export_scales_to_config(qmodel, quant_config, out_path)

    print('\nDone.')


if __name__ == '__main__':
    main()
