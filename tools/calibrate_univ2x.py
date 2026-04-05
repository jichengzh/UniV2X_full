"""
calibrate_univ2x.py — Post-Training Quantization for UniV2X BEV encoder.

Pipeline:
  1. Load model (same as test_trt.py) + calibration data (from dump_univ2x_calibration.py)
  2. Register BEVFormer MSDA specials (ADR-001: skip sampling_offsets/attention_weights)
  3. Wrap BEV encoder in QuantModel
  4. Run set_weight_quantize_params() — calibrate weight scales (MSE/minmax/entropy)
  5. Optionally run layer_reconstruction() (AdaRound) for each QuantModule
  6. Save quantized state dict → calibration/quant_encoder_weights.pth

Usage:
    python tools/calibrate_univ2x.py \\
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \\
        ckpts/univ2x_coop_e2e_stg2.pth \\
        --cali-data calibration/cali_data.pkl \\
        --out calibration/quant_encoder_weights.pth \\
        --scale-method entropy \\
        [--adaround]  # optional AdaRound fine-tuning

Note: Wrap the infra model by passing --model infra.
"""

import argparse
import os
import pickle
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.models import build_model


def parse_args():
    p = argparse.ArgumentParser(description='Calibrate UniV2X BEV encoder (PTQ)')
    p.add_argument('config',        help='Test config file path')
    p.add_argument('checkpoint',    help='Checkpoint file (stg2 .pth)')
    p.add_argument('--cali-data',   default='calibration/cali_data.pkl',
                   help='Calibration data pickle from dump_univ2x_calibration.py')
    p.add_argument('--out',         default='calibration/quant_encoder_weights.pth',
                   help='Output quantized weight path')
    p.add_argument('--model',       choices=['ego', 'infra'], default='ego',
                   help='Which agent model to calibrate')
    p.add_argument('--scale-method', choices=['mse', 'minmax', 'entropy'],
                   default='entropy', help='Weight quantizer scale calibration method')
    p.add_argument('--n-bits-w',    type=int, default=8, help='Weight bit-width')
    p.add_argument('--n-bits-a',    type=int, default=8, help='Activation bit-width')
    p.add_argument('--adaround',    action='store_true',
                   help='Run AdaRound layer reconstruction (slow but more accurate)')
    p.add_argument('--adaround-iters', type=int, default=10000,
                   help='AdaRound iterations per layer (default 10000)')
    p.add_argument('--batch-size',  type=int, default=1,
                   help='Mini-batch size for calibration data capture')
    p.add_argument('--adaround-n-samples', type=int, default=0,
                   help='Number of cali_data samples for AdaRound per-layer (0=use all)')
    p.add_argument('--cfg-options', nargs='+', action='append')
    return p.parse_args()


def load_model(cfg, args):
    """Load UniV2X model (ego or infra agent)."""
    cfg.model_ego_agent.pretrained = None
    cfg.model_ego_agent.train_cfg = None

    # Build other agents if present (needed for MultiAgent init)
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
    load_checkpoint(model_multi, args.checkpoint, map_location='cpu',
                    revise_keys=[(r'^model_ego_agent\.', '')])

    model_multi.eval()
    model_multi.cuda()
    return model_multi


def get_agent_and_encoder(model_multi, model_name):
    """Extract agent model and its BEV encoder sub-module."""
    inner = model_multi.module if hasattr(model_multi, 'module') else model_multi
    if model_name == 'ego':
        agent = inner.model_ego_agent
    else:
        # infra is stored under other_agent_names
        for n in inner.other_agent_names:
            agent = getattr(inner, n)
            break
    # BEV encoder: pts_bbox_head.transformer.encoder
    encoder = agent.pts_bbox_head.transformer.encoder
    return agent, encoder


class BEVEncoderCalibModel(nn.Module):
    """Bridge: makes cali_data dict callable for GetLayerInpOut.

    The calibration data format (feat0-feat3 + can_bus + lidar2img + ...) is designed
    for the TRT ONNX pipeline.  But GetLayerInpOut.__call__ does self.model(cali_data[i])
    which would call encoder.forward(dict) directly — a TypeError.

    This wrapper:
      - Accepts cali_data[i] dict in forward()
      - Routes it through agent.pts_bbox_head.get_bev_features_trt() so the full
        preprocessing (bev_query construction, key/value projection) happens before
        reaching the encoder, causing QuantModule hooks to fire correctly.
      - For fp_model usage: pass use_encoder=fp_model.model to temporarily swap the
        head's encoder with a full-precision copy during the forward pass.
      - Proxies set_quant_state / parameters / modules / state_dict to qmodel.
    """

    def __init__(self, agent: nn.Module, qmodel: nn.Module, use_encoder=None):
        super().__init__()
        self._agent = agent
        self._qmodel = qmodel
        self._head = agent.pts_bbox_head
        # If set, temporarily replace head.transformer.encoder during forward.
        # Used to run the FP (deepcopy) encoder through the same BEV head path.
        self._use_encoder = use_encoder

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

        # prev_bev in calibration data: (bs, bev_h*bev_w, embed_dims)
        # get_bev_features_trt / transformer expects: (bev_h*bev_w, bs, embed_dims)
        prev_bev_raw = _t(cali_item['prev_bev'])
        if prev_bev_raw.dim() == 3 and prev_bev_raw.shape[0] == 1:
            # (1, bev_h*bev_w, embed_dims) → (bev_h*bev_w, 1, embed_dims)
            prev_bev = prev_bev_raw.permute(1, 0, 2)
        else:
            prev_bev = prev_bev_raw

        # use_prev_bev: scalar bool in calibration data → float tensor for the head
        upb = cali_item.get('use_prev_bev', True)
        if isinstance(upb, np.ndarray):
            use_prev_bev = torch.tensor([float(upb.item())], device='cuda')
        elif isinstance(upb, torch.Tensor):
            use_prev_bev = upb.float().cuda().reshape(1)
        else:
            use_prev_bev = torch.tensor([float(upb)], device='cuda')

        # Temporarily swap encoder for fp_model path
        orig_encoder = None
        if self._use_encoder is not None:
            orig_encoder = self._head.transformer.encoder
            self._head.transformer.encoder = self._use_encoder

        try:
            bev_embed, _ = self._head.get_bev_features_trt(
                mlvl_feats=(feat0, feat1, feat2, feat3),
                can_bus=can_bus,
                lidar2img=lidar2img,
                image_shape=image_shape,
                prev_bev=prev_bev,
                use_prev_bev=use_prev_bev,
            )
        finally:
            if orig_encoder is not None:
                self._head.transformer.encoder = orig_encoder

        return bev_embed

    # ---------- Proxy quant management to qmodel ----------
    def set_quant_state(self, weight_quant: bool, act_quant: bool):
        self._qmodel.set_quant_state(weight_quant, act_quant)

    def parameters(self, recurse=True):
        return self._qmodel.parameters(recurse)

    def modules(self):
        return self._qmodel.modules()

    def named_modules(self, *args, **kwargs):
        return self._qmodel.named_modules(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self._qmodel.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self._qmodel.load_state_dict(*args, **kwargs)

    def get_memory_footprint(self):
        return self._qmodel.get_memory_footprint()

    def eval(self):
        self._qmodel.eval()
        return self


def main():
    args = parse_args()
    sys.path.insert(0, '.')

    # ── Config ───────────────────────────────────────────────────────────
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        for kv in args.cfg_options:
            cfg.merge_from_dict(dict(kv))

    if hasattr(cfg, 'plugin') and cfg.plugin:
        if hasattr(cfg, 'plugin_dir'):
            from importlib import import_module
            import_module(cfg.plugin_dir.replace('/', '.').rstrip('.py'))

    # ── Quant package ────────────────────────────────────────────────────
    from projects.mmdet3d_plugin.univ2x.quant import (
        QuantModel,
        register_bevformer_specials,
        register_fusion_specials,
        register_downstream_specials,
        set_weight_quantize_params,
        set_act_quantize_params,
        save_quantized_weight,
        layer_reconstruction,
    )

    register_bevformer_specials()
    register_fusion_specials()
    register_downstream_specials()
    print('Registered BEVFormer/fusion/downstream quantization specials.')

    # ── Load calibration data ─────────────────────────────────────────────
    print(f'Loading calibration data from {args.cali_data} ...')
    with open(args.cali_data, 'rb') as f:
        cali_data = pickle.load(f)
    print(f'  {len(cali_data)} calibration samples loaded.')

    # ── Load model ────────────────────────────────────────────────────────
    print(f'Loading {args.model} model from {args.checkpoint} ...')
    model_multi = load_model(cfg, args)
    agent, encoder = get_agent_and_encoder(model_multi, args.model)
    print(f'  BEV encoder: {encoder.__class__.__name__}')

    # ── Quantization params ──────────────────────────────────────────────
    weight_quant_params = dict(
        n_bits=args.n_bits_w,
        channel_wise=True,
        scale_method=args.scale_method,
    )
    act_quant_params = dict(
        n_bits=args.n_bits_a,
        channel_wise=False,
        scale_method='entropy',
        leaf_param=True,
    )

    # ── Wrap encoder in QuantModel ────────────────────────────────────────
    print('Wrapping BEV encoder in QuantModel (with BN folding) ...')
    encoder.eval()
    qmodel = QuantModel(encoder, weight_quant_params, act_quant_params, is_fusing=True)
    qmodel.cuda()
    qmodel.eval()
    print(f'  {qmodel.get_memory_footprint()}')

    # Calibration wrapper: routes cali_data dict through the full BEV head pipeline
    # so that GetLayerInpOut hooks on QuantModule leaf layers fire correctly.
    calib_model = BEVEncoderCalibModel(agent, qmodel)

    # ── Step 1: calibrate weight scales ──────────────────────────────────
    print('\nStep 1: Calibrating weight quantization scales ...')
    qmodel.set_quant_state(weight_quant=False, act_quant=False)
    set_weight_quantize_params(qmodel)
    print('  Weight scales calibrated.')

    # ── Step 2: AdaRound (optional) ────────────────────────────────────
    if args.adaround:
        print(f'\nStep 2: Running AdaRound ({args.adaround_iters} iters/layer) ...')
        fp_model = deepcopy(qmodel)
        fp_model.set_quant_state(weight_quant=False, act_quant=False)
        # fp_calib_model uses the same BEV head pipeline but temporarily swaps
        # head.transformer.encoder with the FP deepcopy so no quantization runs.
        fp_calib_model = BEVEncoderCalibModel(agent, fp_model, use_encoder=fp_model.model)

        # Optionally limit cali_data to fewer samples (reduces per-layer wall time
        # at the cost of fewer reconstruction examples; 10-20 is typically sufficient).
        n_ada = args.adaround_n_samples if args.adaround_n_samples > 0 else len(cali_data)
        ada_cali_data = cali_data[:n_ada]
        print(f'  Using {len(ada_cali_data)}/{len(cali_data)} cali samples for AdaRound.')

        from projects.mmdet3d_plugin.univ2x.quant import QuantModule
        modules = [(n, m) for n, m in qmodel.model.named_modules()
                   if isinstance(m, QuantModule) and not m.ignore_reconstruction]
        print(f'  Found {len(modules)} QuantModules to reconstruct.')

        for idx, (name, layer) in enumerate(modules):
            # find corresponding layer in fp_model
            fp_layer = dict(fp_model.model.named_modules()).get(name)
            if fp_layer is None:
                print(f'  [{idx+1}/{len(modules)}] SKIP {name} (not in fp_model)')
                continue
            print(f'  [{idx+1}/{len(modules)}] AdaRound: {name}')
            layer_reconstruction(
                model=calib_model,
                fp_model=fp_calib_model,
                layer=layer,
                fp_layer=fp_layer,
                cali_data=ada_cali_data,
                batch_size=args.batch_size,
                iters=args.adaround_iters,
                keep_gpu=False,  # avoid GPU OOM from caching N×BEV embeddings
            )
        print('  AdaRound complete.')
    else:
        print('\nStep 2: Skipped (use --adaround to enable).')

    # ── Save quantized weights ────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # Save the full qmodel state_dict (includes delta/zero_point params)
    torch.save({
        'state_dict': qmodel.state_dict(),
        'weight_quant_params': weight_quant_params,
        'act_quant_params':    act_quant_params,
        'adaround':            args.adaround,
        'n_bits_w':            args.n_bits_w,
        'n_bits_a':            args.n_bits_a,
        'model':               args.model,
    }, args.out)
    print(f'\nSaved quantized weights → {args.out}')
    print(f'File size: {os.path.getsize(args.out) / 1024 / 1024:.1f} MB')


if __name__ == '__main__':
    main()
