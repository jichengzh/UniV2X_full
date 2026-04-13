"""
Quantization sensitivity analysis for UniV2X BEV encoder.

Analyzes per-layer sensitivity to quantization by measuring BEV output cosine
similarity between quantized and FP32 models. Outputs sensitivity_report.json
with layer classification (safe_int8 / search / skip_fp16).

Usage:
    python tools/sensitivity_analysis.py \
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
        ckpts/univ2x_coop_e2e_stg2.pth \
        --cali-data calibration/bev_encoder_calib_inputs.pkl \
        --output calibration/sensitivity_report.json \
        --n-eval 10 \
        [--analysis all]  # or: layer_sensitivity, calibration, granularity, symmetry, bitwidth, interaction
"""

import argparse
import json
import os
import pickle
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Thresholds for layer classification
# ---------------------------------------------------------------------------
THRESHOLD_SAFE = 0.9999
THRESHOLD_SEARCH = 0.999

ALL_ANALYSES = [
    'layer_sensitivity',
    'calibration',
    'granularity',
    'symmetry',
    'bitwidth',
    'interaction',
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Quantization sensitivity analysis for UniV2X BEV encoder')
    p.add_argument('config', help='Model config file path')
    p.add_argument('checkpoint', help='Checkpoint file (.pth)')
    p.add_argument('--cali-data', default='calibration/bev_encoder_calib_inputs.pkl',
                   help='Calibration data pickle')
    p.add_argument('--output', default='calibration/sensitivity_report.json',
                   help='Output JSON report path')
    p.add_argument('--model', choices=['ego', 'infra'], default='ego',
                   help='Which agent model to analyze')
    p.add_argument('--n-eval', type=int, default=10,
                   help='Number of calibration samples for evaluation')
    p.add_argument('--analysis', nargs='+', default=['all'],
                   choices=['all'] + ALL_ANALYSES,
                   help='Which analyses to run (default: all)')
    p.add_argument('--cfg-options', nargs='+', action='append')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading (mirrors calibrate_univ2x.py)
# ---------------------------------------------------------------------------

def load_model(cfg, checkpoint_path: str):
    """Load the full MultiAgent model from config + checkpoint."""
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
    return model_multi


def get_agent_and_encoder(model_multi, model_name: str):
    """Extract agent model and its BEV encoder sub-module."""
    inner = model_multi.module if hasattr(model_multi, 'module') else model_multi
    if model_name == 'ego':
        agent = inner.model_ego_agent
    else:
        for n in inner.other_agent_names:
            agent = getattr(inner, n)
            break
    encoder = agent.pts_bbox_head.transformer.encoder
    return agent, encoder


# ---------------------------------------------------------------------------
# BEVEncoderCalibModel (simplified from calibrate_univ2x.py)
# ---------------------------------------------------------------------------

class BEVEncoderCalibModel(nn.Module):
    """Bridge that routes cali_data dicts through the full BEV head pipeline.

    Accepts cali_data[i] dict in forward(), runs it through
    agent.pts_bbox_head.get_bev_features_trt(), and returns the BEV embedding.
    Proxies set_quant_state / modules / named_modules to qmodel.
    """

    def __init__(self, agent: nn.Module, qmodel: nn.Module):
        super().__init__()
        self._agent = agent
        self._qmodel = qmodel
        self._head = agent.pts_bbox_head

    def forward(self, cali_item):
        def _t(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).cuda()
            if isinstance(x, torch.Tensor):
                return x.cuda()
            return x

        feat0 = _t(cali_item['feat0'])
        feat1 = _t(cali_item['feat1'])
        feat2 = _t(cali_item['feat2'])
        feat3 = _t(cali_item['feat3'])
        can_bus = _t(cali_item['can_bus'])
        lidar2img = _t(cali_item['lidar2img'])
        image_shape = _t(cali_item['image_shape'])

        prev_bev_raw = _t(cali_item['prev_bev'])
        if prev_bev_raw.dim() == 3 and prev_bev_raw.shape[0] == 1:
            prev_bev = prev_bev_raw.permute(1, 0, 2)
        else:
            prev_bev = prev_bev_raw

        upb = cali_item.get('use_prev_bev', True)
        if isinstance(upb, np.ndarray):
            use_prev_bev = torch.tensor([float(upb.item())], device='cuda')
        elif isinstance(upb, torch.Tensor):
            use_prev_bev = upb.float().cuda().reshape(1)
        else:
            use_prev_bev = torch.tensor([float(upb)], device='cuda')

        bev_embed, _ = self._head.get_bev_features_trt(
            mlvl_feats=(feat0, feat1, feat2, feat3),
            can_bus=can_bus,
            lidar2img=lidar2img,
            image_shape=image_shape,
            prev_bev=prev_bev,
            use_prev_bev=use_prev_bev,
        )
        return bev_embed

    def set_quant_state(self, weight_quant: bool, act_quant: bool):
        self._qmodel.set_quant_state(weight_quant, act_quant)

    def parameters(self, recurse=True):
        return self._qmodel.parameters(recurse)

    def modules(self):
        return self._qmodel.modules()

    def named_modules(self, *args, **kwargs):
        return self._qmodel.named_modules(*args, **kwargs)


# ---------------------------------------------------------------------------
# Core evaluation helper
# ---------------------------------------------------------------------------

def collect_fp_outputs(
    calib_model: BEVEncoderCalibModel,
    cali_data: list,
    n_eval: int,
) -> List[torch.Tensor]:
    """Run FP32 baseline and collect outputs."""
    calib_model.set_quant_state(False, False)
    fp_outputs = []
    n = min(n_eval, len(cali_data))
    for i in range(n):
        with torch.no_grad():
            out = calib_model(cali_data[i])
        fp_outputs.append(out.detach().cpu().clone())
    return fp_outputs


def compute_cosine_similarity(
    calib_model: BEVEncoderCalibModel,
    cali_data: list,
    fp_outputs: List[torch.Tensor],
    n_eval: int,
) -> float:
    """Compute mean BEV output cosine similarity against FP32 baseline.

    Assumes the model is already in the desired quant state (weights calibrated,
    activations calibrated).
    """
    n = min(n_eval, len(cali_data), len(fp_outputs))
    cos_sims = []
    for i in range(n):
        with torch.no_grad():
            out_q = calib_model(cali_data[i])
        cos = torch.nn.functional.cosine_similarity(
            fp_outputs[i].flatten().cuda(),
            out_q.flatten(),
            dim=0,
        ).item()
        cos_sims.append(cos)
    return sum(cos_sims) / len(cos_sims)


def evaluate_quant_config_cosine(
    qmodel,
    calib_model: BEVEncoderCalibModel,
    cali_data: list,
    fp_outputs: List[torch.Tensor],
    n_eval: int,
    set_weight_quantize_params_fn,
    set_act_quantize_params_fn,
) -> float:
    """Full evaluation: calibrate weights + activations, then measure cosine sim.

    Steps:
      1. Calibrate weight scales (set_weight_quantize_params)
      2. Calibrate activation scales by running n_eval forward passes
      3. Compute mean cosine similarity vs FP32 baseline
    """
    # Calibrate weights
    set_weight_quantize_params_fn(qmodel)

    # Calibrate activations by running forward passes with quant enabled
    from projects.mmdet3d_plugin.univ2x.quant import QuantModule, BaseQuantBlock
    for m in qmodel.modules():
        if isinstance(m, (QuantModule, BaseQuantBlock)) and hasattr(m, 'act_quantizer'):
            m.act_quantizer.set_inited(False)

    n = min(n_eval, len(cali_data))
    for i in range(n):
        with torch.no_grad():
            calib_model(cali_data[i])

    for m in qmodel.modules():
        if isinstance(m, (QuantModule, BaseQuantBlock)) and hasattr(m, 'act_quantizer'):
            m.act_quantizer.set_inited(True)

    # Compute cosine similarity
    return compute_cosine_similarity(calib_model, cali_data, fp_outputs, n_eval)


def classify_layer(cosine: float) -> str:
    """Classify a layer based on its cosine similarity to FP32."""
    if cosine >= THRESHOLD_SAFE:
        return 'safe_int8'
    if cosine >= THRESHOLD_SEARCH:
        return 'search'
    return 'skip_fp16'


# ---------------------------------------------------------------------------
# Analysis: layer_sensitivity
# ---------------------------------------------------------------------------

def run_layer_sensitivity(
    qmodel,
    calib_model: BEVEncoderCalibModel,
    cali_data: list,
    fp_outputs: List[torch.Tensor],
    n_eval: int,
    set_weight_quantize_params_fn,
) -> Dict[str, Any]:
    """Per-layer sensitivity scan.

    For each quantizable layer:
      1. Disable all quantization
      2. Enable only that layer (W+A)
      3. Calibrate weights for that layer
      4. Run n_eval forward passes to calibrate activations
      5. Compute cosine similarity vs FP32
    """
    from projects.mmdet3d_plugin.univ2x.quant import QuantModule, BaseQuantBlock

    # Collect all quantizable modules with their names
    quant_modules = []
    for name, module in qmodel.model.named_modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            quant_modules.append((name, module))

    print(f'  Found {len(quant_modules)} quantizable layers.')
    results = {}
    t0 = time.time()

    for idx, (name, module) in enumerate(quant_modules):
        layer_t0 = time.time()

        # Step 1: Disable all quantization
        qmodel.set_quant_state(False, False)

        # Step 2: Enable only this layer
        module.set_quant_state(True, True)

        # Step 3: Calibrate weights for this module only
        if isinstance(module, QuantModule):
            module.weight_quantizer.set_inited(False)
            module.weight_quantizer(module.weight)
            module.weight_quantizer.set_inited(True)
        elif isinstance(module, BaseQuantBlock):
            # Calibrate all QuantModules inside this block
            for sub in module.modules():
                if isinstance(sub, QuantModule):
                    sub.weight_quantizer.set_inited(False)
                    sub.weight_quantizer(sub.weight)
                    sub.weight_quantizer.set_inited(True)

        # Step 4: Calibrate activations for this module
        if isinstance(module, QuantModule) and hasattr(module, 'act_quantizer'):
            module.act_quantizer.set_inited(False)
        elif isinstance(module, BaseQuantBlock):
            for sub in module.modules():
                if isinstance(sub, (QuantModule, BaseQuantBlock)) and hasattr(sub, 'act_quantizer'):
                    sub.act_quantizer.set_inited(False)

        n = min(n_eval, len(cali_data))
        for i in range(n):
            with torch.no_grad():
                calib_model(cali_data[i])

        if isinstance(module, QuantModule) and hasattr(module, 'act_quantizer'):
            module.act_quantizer.set_inited(True)
        elif isinstance(module, BaseQuantBlock):
            for sub in module.modules():
                if isinstance(sub, (QuantModule, BaseQuantBlock)) and hasattr(sub, 'act_quantizer'):
                    sub.act_quantizer.set_inited(True)

        # Step 5: Compute cosine similarity
        cosine = compute_cosine_similarity(
            calib_model, cali_data, fp_outputs, n_eval)

        classification = classify_layer(cosine)
        results[name] = {
            'cosine': cosine,
            'classification': classification,
        }

        elapsed = time.time() - layer_t0
        total_elapsed = time.time() - t0
        eta = (total_elapsed / (idx + 1)) * (len(quant_modules) - idx - 1)
        print(f'  [{idx+1}/{len(quant_modules)}] {name}: '
              f'cos={cosine:.6f} ({classification}) '
              f'[{elapsed:.1f}s, ETA {eta:.0f}s]')

    # Reset to all-off state
    qmodel.set_quant_state(False, False)
    return results


# ---------------------------------------------------------------------------
# Analysis: calibration method comparison
# ---------------------------------------------------------------------------

def run_calibration_comparison(
    encoder_original: nn.Module,
    agent: nn.Module,
    cali_data: list,
    fp_outputs: List[torch.Tensor],
    n_eval: int,
    layer_results: Dict[str, Any],
    n_bits_w: int = 8,
    n_bits_a: int = 8,
) -> Dict[str, float]:
    """Compare calibration methods (mse, minmax, entropy, percentile).

    Quantizes all safe+search layers together and measures cosine sim.
    """
    from projects.mmdet3d_plugin.univ2x.quant import (
        QuantModel, QuantModule, BaseQuantBlock,
        set_weight_quantize_params,
    )

    safe_and_search = {
        name for name, info in layer_results.items()
        if info['classification'] in ('safe_int8', 'search')
    }

    methods = ['minmax', 'mse', 'entropy', 'percentile']
    results = {}

    for method in methods:
        print(f'  Calibration method: {method} ...')
        t0 = time.time()

        weight_quant_params = dict(
            n_bits=n_bits_w, symmetric=True,
            channel_wise=False, scale_method=method,
        )
        act_quant_params = dict(
            n_bits=n_bits_a, symmetric=True,
            channel_wise=False, scale_method=method, leaf_param=True,
        )

        # Fresh QuantModel wrap
        encoder_copy = deepcopy(encoder_original)
        encoder_copy.eval()
        qm = QuantModel(encoder_copy, weight_quant_params, act_quant_params, is_fusing=True)
        qm.cuda()
        qm.eval()

        # Enable only safe+search layers
        qm.set_quant_state(False, False)
        named_mods = dict(qm.model.named_modules())
        for layer_name in safe_and_search:
            mod = named_mods.get(layer_name)
            if mod is not None and isinstance(mod, (QuantModule, BaseQuantBlock)):
                mod.set_quant_state(True, True)

        cm = BEVEncoderCalibModel(agent, qm)

        cosine = evaluate_quant_config_cosine(
            qm, cm, cali_data, fp_outputs, n_eval,
            set_weight_quantize_params, set_weight_quantize_params,  # second arg unused
        )
        results[method] = cosine

        elapsed = time.time() - t0
        print(f'    cos={cosine:.6f} [{elapsed:.1f}s]')

        # Free GPU memory
        del qm, cm, encoder_copy
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Analysis: granularity comparison
# ---------------------------------------------------------------------------

def run_granularity_comparison(
    encoder_original: nn.Module,
    agent: nn.Module,
    cali_data: list,
    fp_outputs: List[torch.Tensor],
    n_eval: int,
    layer_results: Dict[str, Any],
    n_bits_w: int = 8,
    n_bits_a: int = 8,
) -> Dict[str, float]:
    """Compare per-tensor vs per-channel granularity."""
    from projects.mmdet3d_plugin.univ2x.quant import (
        QuantModel, QuantModule, BaseQuantBlock,
        set_weight_quantize_params,
    )

    safe_and_search = {
        name for name, info in layer_results.items()
        if info['classification'] in ('safe_int8', 'search')
    }

    granularities = {'per_tensor': False, 'per_channel': True}
    results = {}

    for gran_name, channel_wise in granularities.items():
        print(f'  Granularity: {gran_name} ...')
        t0 = time.time()

        weight_quant_params = dict(
            n_bits=n_bits_w, symmetric=True,
            channel_wise=channel_wise, scale_method='minmax',
        )
        act_quant_params = dict(
            n_bits=n_bits_a, symmetric=True,
            channel_wise=False, scale_method='minmax', leaf_param=True,
        )

        encoder_copy = deepcopy(encoder_original)
        encoder_copy.eval()
        qm = QuantModel(encoder_copy, weight_quant_params, act_quant_params, is_fusing=True)
        qm.cuda()
        qm.eval()

        qm.set_quant_state(False, False)
        named_mods = dict(qm.model.named_modules())
        for layer_name in safe_and_search:
            mod = named_mods.get(layer_name)
            if mod is not None and isinstance(mod, (QuantModule, BaseQuantBlock)):
                mod.set_quant_state(True, True)

        cm = BEVEncoderCalibModel(agent, qm)
        cosine = evaluate_quant_config_cosine(
            qm, cm, cali_data, fp_outputs, n_eval,
            set_weight_quantize_params, set_weight_quantize_params,
        )
        results[gran_name] = cosine

        elapsed = time.time() - t0
        print(f'    cos={cosine:.6f} [{elapsed:.1f}s]')

        del qm, cm, encoder_copy
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Analysis: symmetry comparison
# ---------------------------------------------------------------------------

def run_symmetry_comparison(
    encoder_original: nn.Module,
    agent: nn.Module,
    cali_data: list,
    fp_outputs: List[torch.Tensor],
    n_eval: int,
    n_bits_w: int = 8,
    n_bits_a: int = 8,
) -> Dict[str, float]:
    """Compare symmetric vs asymmetric quantization (full model)."""
    from projects.mmdet3d_plugin.univ2x.quant import (
        QuantModel, set_weight_quantize_params,
    )

    sym_options = {'symmetric': True, 'asymmetric': False}
    results = {}

    for sym_name, symmetric in sym_options.items():
        print(f'  Symmetry: {sym_name} ...')
        t0 = time.time()

        weight_quant_params = dict(
            n_bits=n_bits_w, symmetric=symmetric,
            channel_wise=False, scale_method='minmax',
        )
        act_quant_params = dict(
            n_bits=n_bits_a, symmetric=symmetric,
            channel_wise=False, scale_method='minmax', leaf_param=True,
        )

        encoder_copy = deepcopy(encoder_original)
        encoder_copy.eval()
        qm = QuantModel(encoder_copy, weight_quant_params, act_quant_params, is_fusing=True)
        qm.cuda()
        qm.eval()

        qm.set_quant_state(True, True)

        cm = BEVEncoderCalibModel(agent, qm)
        cosine = evaluate_quant_config_cosine(
            qm, cm, cali_data, fp_outputs, n_eval,
            set_weight_quantize_params, set_weight_quantize_params,
        )
        results[sym_name] = cosine

        elapsed = time.time() - t0
        print(f'    cos={cosine:.6f} [{elapsed:.1f}s]')

        del qm, cm, encoder_copy
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Analysis: bitwidth comparison
# ---------------------------------------------------------------------------

def run_bitwidth_comparison(
    encoder_original: nn.Module,
    agent: nn.Module,
    cali_data: list,
    fp_outputs: List[torch.Tensor],
    n_eval: int,
    layer_results: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Per-layer bitwidth comparison (INT4/INT6/INT8) for search layers only."""
    from projects.mmdet3d_plugin.univ2x.quant import (
        QuantModel, QuantModule, BaseQuantBlock,
    )

    search_layers = [
        name for name, info in layer_results.items()
        if info['classification'] == 'search'
    ]
    if not search_layers:
        print('  No search layers found, skipping bitwidth analysis.')
        return {}

    bitwidths = [4, 6, 8]
    results = {}

    for layer_name in search_layers:
        print(f'  Layer: {layer_name}')
        layer_results_bw = {}

        for bw in bitwidths:
            print(f'    INT{bw} ...', end=' ', flush=True)
            t0 = time.time()

            weight_quant_params = dict(
                n_bits=8, symmetric=True,
                channel_wise=False, scale_method='minmax',
            )
            act_quant_params = dict(
                n_bits=8, symmetric=True,
                channel_wise=False, scale_method='minmax', leaf_param=True,
            )

            encoder_copy = deepcopy(encoder_original)
            encoder_copy.eval()
            qm = QuantModel(encoder_copy, weight_quant_params, act_quant_params, is_fusing=True)
            qm.cuda()
            qm.eval()

            # Disable all quantization
            qm.set_quant_state(False, False)

            # Enable and set bitwidth for target layer only
            named_mods = dict(qm.model.named_modules())
            mod = named_mods.get(layer_name)
            if mod is None:
                print(f'not found, skipping.')
                del qm, encoder_copy
                torch.cuda.empty_cache()
                continue

            if isinstance(mod, QuantModule):
                mod.set_quant_state(True, True)
                mod.weight_quantizer.bitwidth_refactor(bw)
                mod.act_quantizer.bitwidth_refactor(bw)
                # Calibrate weight
                mod.weight_quantizer.set_inited(False)
                mod.weight_quantizer(mod.weight)
                mod.weight_quantizer.set_inited(True)
                # Calibrate activation
                mod.act_quantizer.set_inited(False)
            elif isinstance(mod, BaseQuantBlock):
                mod.set_quant_state(True, True)
                for sub in mod.modules():
                    if isinstance(sub, QuantModule):
                        sub.weight_quantizer.bitwidth_refactor(bw)
                        sub.act_quantizer.bitwidth_refactor(bw)
                        sub.weight_quantizer.set_inited(False)
                        sub.weight_quantizer(sub.weight)
                        sub.weight_quantizer.set_inited(True)
                        sub.act_quantizer.set_inited(False)

            cm = BEVEncoderCalibModel(agent, qm)

            # Calibrate activations
            n = min(n_eval, len(cali_data))
            for i in range(n):
                with torch.no_grad():
                    cm(cali_data[i])

            # Set activation inited
            if isinstance(mod, QuantModule) and hasattr(mod, 'act_quantizer'):
                mod.act_quantizer.set_inited(True)
            elif isinstance(mod, BaseQuantBlock):
                for sub in mod.modules():
                    if isinstance(sub, (QuantModule, BaseQuantBlock)) and hasattr(sub, 'act_quantizer'):
                        sub.act_quantizer.set_inited(True)

            cosine = compute_cosine_similarity(cm, cali_data, fp_outputs, n_eval)
            layer_results_bw[f'int{bw}'] = cosine

            elapsed = time.time() - t0
            print(f'cos={cosine:.6f} [{elapsed:.1f}s]')

            del qm, cm, encoder_copy
            torch.cuda.empty_cache()

        results[layer_name] = layer_results_bw

    return results


# ---------------------------------------------------------------------------
# Analysis: interaction analysis
# ---------------------------------------------------------------------------

def run_interaction_analysis(
    encoder_original: nn.Module,
    agent: nn.Module,
    cali_data: list,
    fp_outputs: List[torch.Tensor],
    n_eval: int,
    layer_results: Dict[str, Any],
    top_k: int = 3,
) -> Dict[str, Any]:
    """Analyze layer-pair interactions for the most sensitive search layers.

    For the top-k most sensitive search layers (lowest cosine), test all pairs:
    quantize both layers together and compare the joint delta against the sum
    of individual deltas.
    """
    from projects.mmdet3d_plugin.univ2x.quant import (
        QuantModel, QuantModule, BaseQuantBlock,
    )

    search_layers = [
        (name, info['cosine'])
        for name, info in layer_results.items()
        if info['classification'] == 'search'
    ]
    # Sort by cosine ascending (most sensitive first)
    search_layers.sort(key=lambda x: x[1])
    top_layers = search_layers[:top_k]

    if len(top_layers) < 2:
        print('  Fewer than 2 search layers, skipping interaction analysis.')
        return {}

    print(f'  Top-{top_k} most sensitive search layers:')
    for name, cos in top_layers:
        print(f'    {name}: cos={cos:.6f}')

    # Individual deltas (1.0 - cosine)
    individual_deltas = {name: 1.0 - cos for name, cos in top_layers}

    results = {}
    pairs = list(combinations([name for name, _ in top_layers], 2))

    for layer_a, layer_b in pairs:
        print(f'  Pair: ({layer_a}, {layer_b}) ...', end=' ', flush=True)
        t0 = time.time()

        weight_quant_params = dict(
            n_bits=8, symmetric=True,
            channel_wise=False, scale_method='minmax',
        )
        act_quant_params = dict(
            n_bits=8, symmetric=True,
            channel_wise=False, scale_method='minmax', leaf_param=True,
        )

        encoder_copy = deepcopy(encoder_original)
        encoder_copy.eval()
        qm = QuantModel(encoder_copy, weight_quant_params, act_quant_params, is_fusing=True)
        qm.cuda()
        qm.eval()

        qm.set_quant_state(False, False)

        named_mods = dict(qm.model.named_modules())
        for layer_name in (layer_a, layer_b):
            mod = named_mods.get(layer_name)
            if mod is not None and isinstance(mod, (QuantModule, BaseQuantBlock)):
                mod.set_quant_state(True, True)
                # Calibrate weights
                if isinstance(mod, QuantModule):
                    mod.weight_quantizer.set_inited(False)
                    mod.weight_quantizer(mod.weight)
                    mod.weight_quantizer.set_inited(True)
                    if hasattr(mod, 'act_quantizer'):
                        mod.act_quantizer.set_inited(False)
                elif isinstance(mod, BaseQuantBlock):
                    for sub in mod.modules():
                        if isinstance(sub, QuantModule):
                            sub.weight_quantizer.set_inited(False)
                            sub.weight_quantizer(sub.weight)
                            sub.weight_quantizer.set_inited(True)
                        if isinstance(sub, (QuantModule, BaseQuantBlock)) and hasattr(sub, 'act_quantizer'):
                            sub.act_quantizer.set_inited(False)

        cm = BEVEncoderCalibModel(agent, qm)

        # Calibrate activations
        n = min(n_eval, len(cali_data))
        for i in range(n):
            with torch.no_grad():
                cm(cali_data[i])

        for layer_name in (layer_a, layer_b):
            mod = named_mods.get(layer_name)
            if mod is not None:
                if isinstance(mod, QuantModule) and hasattr(mod, 'act_quantizer'):
                    mod.act_quantizer.set_inited(True)
                elif isinstance(mod, BaseQuantBlock):
                    for sub in mod.modules():
                        if isinstance(sub, (QuantModule, BaseQuantBlock)) and hasattr(sub, 'act_quantizer'):
                            sub.act_quantizer.set_inited(True)

        joint_cosine = compute_cosine_similarity(cm, cali_data, fp_outputs, n_eval)
        delta_joint = 1.0 - joint_cosine
        delta_sum = individual_deltas[layer_a] + individual_deltas[layer_b]
        interaction_term = delta_joint - delta_sum

        pair_key = f'({layer_a}, {layer_b})'
        results[pair_key] = {
            'delta_joint': round(delta_joint, 8),
            'delta_sum': round(delta_sum, 8),
            'interaction_term': round(interaction_term, 8),
        }

        elapsed = time.time() - t0
        print(f'joint={joint_cosine:.6f}, interaction={interaction_term:.6f} [{elapsed:.1f}s]')

        del qm, cm, encoder_copy
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Search space reduction summary
# ---------------------------------------------------------------------------

def compute_search_space_reduction(
    layer_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Summarize which layers are locked INT8, locked FP16, or need search."""
    locked_int8 = sorted([
        name for name, info in layer_results.items()
        if info['classification'] == 'safe_int8'
    ])
    locked_fp16 = sorted([
        name for name, info in layer_results.items()
        if info['classification'] == 'skip_fp16'
    ])
    search_layers = sorted([
        name for name, info in layer_results.items()
        if info['classification'] == 'search'
    ])
    original_dims = len(layer_results)
    reduced_dims = len(search_layers)

    return {
        'locked_int8': locked_int8,
        'locked_fp16': locked_fp16,
        'search_layers': search_layers,
        'original_dims': original_dims,
        'reduced_dims': reduced_dims,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    sys.path.insert(0, '.')

    # Determine which analyses to run
    if 'all' in args.analysis:
        analyses = set(ALL_ANALYSES)
    else:
        analyses = set(args.analysis)

    # ---- Config ----
    from mmcv import Config
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        for kv in args.cfg_options:
            cfg.merge_from_dict(dict(kv))

    if hasattr(cfg, 'plugin') and cfg.plugin:
        if hasattr(cfg, 'plugin_dir'):
            from importlib import import_module
            import_module(cfg.plugin_dir.replace('/', '.').rstrip('.py'))

    # ---- Quant package ----
    from projects.mmdet3d_plugin.univ2x.quant import (
        QuantModel,
        QuantModule,
        BaseQuantBlock,
        register_bevformer_specials,
        register_fusion_specials,
        register_downstream_specials,
        set_weight_quantize_params,
        set_act_quantize_params,
    )

    register_bevformer_specials()
    register_fusion_specials()
    register_downstream_specials()
    print('Registered BEVFormer/fusion/downstream quantization specials.')

    # ---- Load calibration data ----
    print(f'Loading calibration data from {args.cali_data} ...')
    with open(args.cali_data, 'rb') as f:
        cali_data = pickle.load(f)
    print(f'  {len(cali_data)} calibration samples loaded.')

    # ---- Load model ----
    print(f'Loading {args.model} model from {args.checkpoint} ...')
    model_multi = load_model(cfg, args.checkpoint)
    agent, encoder = get_agent_and_encoder(model_multi, args.model)
    print(f'  BEV encoder: {encoder.__class__.__name__}')

    # Keep a CPU copy of the original encoder for creating fresh QuantModels
    encoder_original = deepcopy(encoder).cpu()

    # ---- Wrap encoder in QuantModel (for layer_sensitivity) ----
    weight_quant_params = dict(
        n_bits=8, symmetric=True,
        channel_wise=False, scale_method='minmax',
    )
    act_quant_params = dict(
        n_bits=8, symmetric=True,
        channel_wise=False, scale_method='minmax', leaf_param=True,
    )

    print('Wrapping BEV encoder in QuantModel ...')
    encoder.eval()
    qmodel = QuantModel(encoder, weight_quant_params, act_quant_params, is_fusing=True)
    qmodel.cuda()
    qmodel.eval()

    calib_model = BEVEncoderCalibModel(agent, qmodel)

    # ---- Collect FP32 baseline outputs ----
    print(f'\nCollecting FP32 baseline outputs ({args.n_eval} samples) ...')
    fp_outputs = collect_fp_outputs(calib_model, cali_data, args.n_eval)
    print('  Done.')

    # ---- Report ----
    report: Dict[str, Any] = {
        'metadata': {
            'model': args.model,
            'n_eval': args.n_eval,
            'cali_data': args.cali_data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'analyses': sorted(analyses),
        },
        'baseline_cosine': 1.0,
    }

    # ---- 1. Layer sensitivity ----
    layer_results = {}
    if 'layer_sensitivity' in analyses:
        print('\n' + '='*70)
        print('Analysis: layer_sensitivity')
        print('='*70)
        layer_results = run_layer_sensitivity(
            qmodel, calib_model, cali_data, fp_outputs,
            args.n_eval, set_weight_quantize_params,
        )
        report['layer_sensitivity'] = layer_results

        # Print summary
        n_safe = sum(1 for v in layer_results.values() if v['classification'] == 'safe_int8')
        n_search = sum(1 for v in layer_results.values() if v['classification'] == 'search')
        n_skip = sum(1 for v in layer_results.values() if v['classification'] == 'skip_fp16')
        print(f'\n  Summary: {n_safe} safe_int8, {n_search} search, {n_skip} skip_fp16')
    else:
        # Try to load layer_results from existing report file
        if os.path.exists(args.output):
            import json as _json
            with open(args.output) as _f:
                _prev = _json.load(_f)
            if 'layer_sensitivity' in _prev:
                layer_results = _prev['layer_sensitivity']
                report['layer_sensitivity'] = layer_results
                n_safe = sum(1 for v in layer_results.values() if v['classification'] == 'safe_int8')
                n_search = sum(1 for v in layer_results.values() if v['classification'] == 'search')
                n_skip = sum(1 for v in layer_results.values() if v['classification'] == 'skip_fp16')
                print(f'\nLoaded previous layer_sensitivity from {args.output}:'
                      f' {n_safe} safe, {n_search} search, {n_skip} skip')
            else:
                print(f'\nNo layer_sensitivity in {args.output}, analyses that depend on it will be skipped.')
        else:
            print('\nNo previous report found. Analyses that depend on layer_sensitivity will be skipped.')

    # ---- 2. Calibration comparison ----
    if 'calibration' in analyses and layer_results:
        print('\n' + '='*70)
        print('Analysis: calibration method comparison')
        print('='*70)
        report['calibration_comparison'] = run_calibration_comparison(
            encoder_original, agent, cali_data, fp_outputs,
            args.n_eval, layer_results,
        )

    # ---- 3. Granularity comparison ----
    if 'granularity' in analyses and layer_results:
        print('\n' + '='*70)
        print('Analysis: granularity comparison')
        print('='*70)
        report['granularity_comparison'] = run_granularity_comparison(
            encoder_original, agent, cali_data, fp_outputs,
            args.n_eval, layer_results,
        )

    # ---- 4. Symmetry comparison ----
    if 'symmetry' in analyses:
        print('\n' + '='*70)
        print('Analysis: symmetry comparison')
        print('='*70)
        report['symmetry_comparison'] = run_symmetry_comparison(
            encoder_original, agent, cali_data, fp_outputs,
            args.n_eval,
        )

    # ---- 5. Bitwidth comparison ----
    if 'bitwidth' in analyses and layer_results:
        print('\n' + '='*70)
        print('Analysis: bitwidth comparison (search layers)')
        print('='*70)
        report['bitwidth_comparison'] = run_bitwidth_comparison(
            encoder_original, agent, cali_data, fp_outputs,
            args.n_eval, layer_results,
        )

    # ---- 6. Interaction analysis ----
    if 'interaction' in analyses and layer_results:
        print('\n' + '='*70)
        print('Analysis: layer-pair interaction')
        print('='*70)
        report['interaction_analysis'] = run_interaction_analysis(
            encoder_original, agent, cali_data, fp_outputs,
            args.n_eval, layer_results, top_k=3,
        )

    # ---- Search space reduction ----
    if layer_results:
        report['search_space_reduction'] = compute_search_space_reduction(layer_results)

    # ---- Save report (merge with existing if present) ----
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        # Merge: new results overwrite existing keys, but keep old keys not in this run
        existing.update(report)
        report = existing
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nSaved sensitivity report -> {args.output}')
    print('Done.')


if __name__ == '__main__':
    main()
