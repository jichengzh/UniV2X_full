"""
validate_adaround_bev.py — Cosine similarity between FP32 and AdaRound W8A8 BEV output.

Loads the AdaRound checkpoint (calibration/quant_encoder_adaround.pth),
applies fake-quantized weights via apply_adaround_weights() logic,
then runs N calibration samples through both FP32 and AdaRound fake-quant paths
and reports cosine similarity.

Usage:
    python tools/validate_adaround_bev.py \
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
        ckpts/univ2x_coop_e2e_stg2.pth \
        --adaround-ckpt calibration/quant_encoder_adaround.pth \
        --cali-data calibration/bev_encoder_calib_inputs.pkl \
        --n-samples 10
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from mmcv import Config
from mmcv.runner import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--adaround-ckpt', required=True)
    p.add_argument('--cali-data', default='calibration/bev_encoder_calib_inputs.pkl')
    p.add_argument('--n-samples', type=int, default=10)
    return p.parse_args()


def cosine_sim(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()


def main():
    args = parse_args()
    sys.path.insert(0, '.')

    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin') and cfg.plugin:
        if hasattr(cfg, 'plugin_dir'):
            from importlib import import_module
            import_module(cfg.plugin_dir.replace('/', '.').rstrip('.py'))

    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.univ2x.quant import (
        QuantModel, QuantModule,
        register_bevformer_specials,
        set_weight_quantize_params,
    )
    from projects.mmdet3d_plugin.univ2x.quant.adaptive_rounding import AdaRoundQuantizer
    from projects.mmdet3d_plugin.univ2x.quant.quant_params import save_quantized_weight

    # Add tools/ dir so we can import build_model_from_cfg from export_onnx_univ2x
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from export_onnx_univ2x import build_model_from_cfg

    register_bevformer_specials()

    # ── Load calibration data ─────────────────────────────────────────────
    print(f'Loading calibration data from {args.cali_data} ...')
    with open(args.cali_data, 'rb') as f:
        cali_data = pickle.load(f)
    n = min(args.n_samples, len(cali_data))
    print(f'  Using {n}/{len(cali_data)} samples.')

    # ── Load model — use build_model_from_cfg (same as ONNX export) ───────
    # This correctly strips the 'model_ego_agent.' prefix from cooperative ckpt keys.
    print(f'Loading model from {args.checkpoint} ...')
    cfg.model_ego_agent.pretrained = None
    cfg.model_ego_agent.train_cfg = None
    agent = build_model_from_cfg(cfg, 'model_ego_agent', ckpt_path=args.checkpoint)
    agent.cuda().eval()
    head = agent.pts_bbox_head

    # ── Wrap encoder in QuantModel (same params as calibration) ──────────
    print('Wrapping BEV encoder in QuantModel ...')
    ckpt_data = torch.load(args.adaround_ckpt, map_location='cpu')
    wqp = ckpt_data.get('weight_quant_params',
                        dict(n_bits=8, channel_wise=True, scale_method='mse'))
    aqp = ckpt_data.get('act_quant_params',
                        dict(n_bits=8, channel_wise=False, scale_method='entropy',
                             leaf_param=True))
    print(f'  weight_quant_params: {wqp}')

    encoder = head.transformer.encoder
    encoder.eval()
    qmodel = QuantModel(encoder, wqp, aqp, is_fusing=True)
    qmodel.cuda()

    # ── Apply AdaRound weights ────────────────────────────────────────────
    print('Applying AdaRound weights ...')
    # Step 1: Re-calibrate delta from weights (delta not in state_dict!)
    qmodel.set_quant_state(weight_quant=False, act_quant=False)
    set_weight_quantize_params(qmodel)
    print('  Weight scales re-calibrated.')

    # Wrap weight_quantizer → AdaRoundQuantizer so state_dict keys align
    for m in qmodel.modules():
        if isinstance(m, QuantModule) and not isinstance(m.weight_quantizer, AdaRoundQuantizer):
            m.weight_quantizer = AdaRoundQuantizer(
                uaq=m.weight_quantizer,
                round_mode='learned_hard_sigmoid',
                weight_tensor=m.org_weight.data)

    missing, unexpected = qmodel.load_state_dict(ckpt_data['state_dict'], strict=False)
    if unexpected:
        print(f'  ignored {len(unexpected)} unexpected keys')

    qmodel.set_quant_state(weight_quant=True, act_quant=False)
    for m in qmodel.modules():
        if hasattr(m, 'weight_quantizer') and hasattr(m.weight_quantizer, 'soft_targets'):
            m.weight_quantizer.soft_targets = False

    save_quantized_weight(qmodel)  # write W_fq into weight.data
    n_updated = 0
    for m in qmodel.modules():
        if isinstance(m, QuantModule):
            m.org_weight.copy_(m.weight.data)  # forward reads org_weight
            n_updated += 1
    qmodel.set_quant_state(weight_quant=False, act_quant=False)
    print(f'  W_fq baked into org_weight for {n_updated} QuantModules.')

    # ── Make FP32 reference model (fresh load, no QuantModel applied) ─────
    agent_fp = build_model_from_cfg(cfg, 'model_ego_agent', ckpt_path=args.checkpoint)
    agent_fp.cuda().eval()
    head_fp = agent_fp.pts_bbox_head

    # ── Validation loop ───────────────────────────────────────────────────
    def _t(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).cuda()
        return x.cuda()

    cos_sims, max_diffs, mean_diffs = [], [], []

    for i in range(n):
        sample = cali_data[i]
        feat0 = _t(sample['feat0'])
        feat1 = _t(sample['feat1'])
        feat2 = _t(sample['feat2'])
        feat3 = _t(sample['feat3'])
        can_bus = _t(sample['can_bus'])
        lidar2img = _t(sample['lidar2img'])
        image_shape = _t(sample['image_shape'])

        prev_bev_raw = _t(sample['prev_bev'])
        # (1, 40000, 256) → (40000, 1, 256)
        prev_bev = (prev_bev_raw.permute(1, 0, 2)
                    if prev_bev_raw.dim() == 3 and prev_bev_raw.shape[0] == 1
                    else prev_bev_raw)
        upb = sample.get('use_prev_bev', True)
        if isinstance(upb, np.ndarray):
            use_prev_bev = torch.tensor([float(upb.item())], device='cuda')
        else:
            use_prev_bev = torch.tensor([float(upb)], device='cuda')

        with torch.no_grad():
            # FP32 reference
            bev_fp, _ = head_fp.get_bev_features_trt(
                mlvl_feats=(feat0, feat1, feat2, feat3),
                can_bus=can_bus, lidar2img=lidar2img,
                image_shape=image_shape, prev_bev=prev_bev,
                use_prev_bev=use_prev_bev)

            # AdaRound fake-quant (W_fq, no act quant)
            bev_ada, _ = head.get_bev_features_trt(
                mlvl_feats=(feat0, feat1, feat2, feat3),
                can_bus=can_bus, lidar2img=lidar2img,
                image_shape=image_shape, prev_bev=prev_bev,
                use_prev_bev=use_prev_bev)

        cos = cosine_sim(bev_fp, bev_ada)
        mx = (bev_fp.float() - bev_ada.float()).abs().max().item()
        mn = (bev_fp.float() - bev_ada.float()).abs().mean().item()
        cos_sims.append(cos)
        max_diffs.append(mx)
        mean_diffs.append(mn)
        print(f'  [{i+1:3d}/{n}] cos={cos:.7f}  max_abs={mx:.4e}  mean_abs={mn:.4e}')

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print(f'AdaRound W8A8-weight  BEV Accuracy  ({n} samples)')
    print('=' * 60)
    print(f'  Cosine Similarity : mean={np.mean(cos_sims):.7f}  '
          f'min={np.min(cos_sims):.7f}')
    print(f'  Max Abs Diff      : mean={np.mean(max_diffs):.4e}  '
          f'max={np.max(max_diffs):.4e}')
    print(f'  Mean Abs Diff     : {np.mean(mean_diffs):.4e}')
    min_cos = np.min(cos_sims)
    if min_cos > 0.997:
        verdict = 'PASS ✓  Cosine > 0.997 (target met)'
    elif min_cos > 0.99:
        verdict = 'MARGINAL  Cosine < 0.997 but > 0.99'
    else:
        verdict = 'FAIL  Cosine < 0.99 — check AdaRound calibration'
    print(f'\n  VERDICT: {verdict}')
    print('=' * 60)


if __name__ == '__main__':
    main()
