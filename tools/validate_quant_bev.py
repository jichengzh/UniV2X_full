"""
validate_quant_bev.py — PyTorch W8A8 PTQ accuracy validation for the BEV encoder.

Wraps the BEV encoder in QuantModel, runs N val samples through
pts_bbox_head.get_bev_features(), and reports cosine similarity / abs diff
between FP32 and W8A8 BEV embed outputs.

Also dumps backbone feature tensors (feat0..feat3 + metadata) to a pkl file
that can be passed directly to build_trt_int8_univ2x.py for TRT INT8 calibration.

Usage:
    python tools/validate_quant_bev.py \\
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \\
        ckpts/univ2x_coop_e2e_stg2.pth \\
        --n-samples 10
"""

import argparse
import os
import pickle
import sys
from copy import deepcopy

import numpy as np
import torch
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--n-samples',    type=int, default=10)
    p.add_argument('--scale-method', default='entropy',
                   choices=['mse', 'minmax', 'entropy'])
    p.add_argument('--n-bits-w',     type=int, default=8)
    p.add_argument('--n-bits-a',     type=int, default=8)
    p.add_argument('--dump-calib',
                   default='calibration/bev_encoder_calib_inputs.pkl',
                   help='Dump backbone feature tensors for TRT INT8 calibration')
    p.add_argument('--cfg-options',  nargs='+', action='append')
    return p.parse_args()


def cosine_sim(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()


def main():
    args = parse_args()
    sys.path.insert(0, '.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        for kv in args.cfg_options:
            cfg.merge_from_dict(dict(kv))

    if hasattr(cfg, 'plugin') and cfg.plugin:
        if hasattr(cfg, 'plugin_dir'):
            from importlib import import_module
            import_module(cfg.plugin_dir.replace('/', '.').rstrip('.py'))

    cfg.model_ego_agent.pretrained = None
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test.get('pipeline'):
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    # ── Quant package ────────────────────────────────────────────────────
    from projects.mmdet3d_plugin.univ2x.quant import (
        QuantModel, QuantModule,
        register_bevformer_specials,
        set_weight_quantize_params,
    )
    register_bevformer_specials()

    # ── Dataset ──────────────────────────────────────────────────────────
    print('Building dataset...')
    dataset = build_dataset(cfg.data.test)
    n = min(args.n_samples, len(dataset))
    print(f'Validating on {n} samples.')

    # ── Build model ───────────────────────────────────────────────────────
    cfg.model_ego_agent.train_cfg = None
    other_names = [k for k in cfg.keys() if 'model_other_agent' in k]
    other_agents = {}
    for name in other_names:
        cfg.get(name).train_cfg = None
        m = build_model(cfg.get(name), test_cfg=cfg.get('test_cfg'))
        lf = cfg.get(name).load_from
        if lf:
            load_checkpoint(m, lf, map_location='cpu',
                            revise_keys=[(r'^model_ego_agent\.', '')])
        other_agents[name] = m

    model_ego = build_model(cfg.model_ego_agent, test_cfg=cfg.get('test_cfg'))
    lf = cfg.model_ego_agent.load_from
    if lf:
        load_checkpoint(model_ego, lf, map_location='cpu',
                        revise_keys=[(r'^model_ego_agent\.', '')])

    from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent
    model_multi = MultiAgent(model_ego, other_agents)
    load_checkpoint(model_multi, args.checkpoint, map_location='cpu',
                    revise_keys=[(r'^model_ego_agent\.', '')])
    model_multi.cuda().eval()

    ego = model_multi.model_ego_agent
    head = ego.pts_bbox_head

    # ── Deep-copy head for FP32 reference ────────────────────────────────
    print('Copying FP32 reference head...')
    head_fp = deepcopy(head)
    head_fp.cuda().eval()

    # ── Wrap encoder in QuantModel ────────────────────────────────────────
    # channel_wise weight quant: entropy not supported channel-wise → use mse
    w_scale = args.scale_method if args.scale_method != 'entropy' else 'mse'
    wqp = dict(n_bits=args.n_bits_w, channel_wise=True,
               scale_method=w_scale)
    aqp = dict(n_bits=args.n_bits_a, channel_wise=False,
               scale_method='entropy', leaf_param=True)

    print('Wrapping BEV encoder in QuantModel (no BN folding, BEVFormer has LN)...')
    encoder = head.transformer.encoder
    encoder.eval()
    qmodel = QuantModel(encoder, wqp, aqp, is_fusing=False)
    qmodel.cuda().eval()

    print('Calibrating weight scales...')
    qmodel.set_quant_state(weight_quant=False, act_quant=False)
    set_weight_quantize_params(qmodel)
    print('  Done.\n')

    # ── DataLoader ────────────────────────────────────────────────────────
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0,
        dist=False, shuffle=False,
    )

    # ── Validation loop ───────────────────────────────────────────────────
    cos_sims, max_diffs, mean_diffs = [], [], []
    calib_inputs = []
    act_inited = False
    # Track temporal prev_bev state (matching actual test_trt.py inference)
    prev_bev_fp = None
    prev_bev_q  = None

    for step, data in enumerate(data_loader):
        if step >= n:
            break

        # collate → single GPU
        from mmcv.parallel import scatter
        data = scatter(data, [0])[0]

        # V2X-Seq dataset nests ego inputs under 'ego_agent_data'
        ego_data  = data.get('ego_agent_data', data)
        img       = ego_data['img'][0]          # (1, num_cam, 3, H, W)
        img_metas = ego_data['img_metas'][0]    # list of dict, len=bs

        with torch.no_grad():
            mlvl_feats = ego.extract_img_feat(img=img)

            # ---- FP32 BEV (with temporal prev_bev state) ----
            bev_fp, _ = head_fp.get_bev_features(
                mlvl_feats, img_metas, prev_bev=prev_bev_fp)
            prev_bev_fp = bev_fp.detach()

            # ---- W8A8 quantized BEV ----
            # Init activation scales on first forward
            if not act_inited:
                for m in qmodel.modules():
                    if isinstance(m, QuantModule):
                        m.act_quantizer.set_inited(False)
            qmodel.set_quant_state(weight_quant=True, act_quant=True)
            bev_q, _ = head.get_bev_features(
                mlvl_feats, img_metas, prev_bev=prev_bev_q)
            prev_bev_q = bev_q.detach()
            act_inited = True

        bev_fp = bev_fp.float()
        bev_q  = bev_q.float()
        cos   = cosine_sim(bev_fp, bev_q)
        mx    = (bev_fp - bev_q).abs().max().item()
        mn    = (bev_fp - bev_q).abs().mean().item()
        cos_sims.append(cos)
        max_diffs.append(mx)
        mean_diffs.append(mn)
        print(f'  [{step+1:3d}/{n}] cos={cos:.7f}  max_abs={mx:.4e}  mean_abs={mn:.4e}')

        # Collect TRT calibration inputs (with temporal prev_bev for realistic ranges)
        raw_shape = img_metas[0]['img_shape']
        if isinstance(raw_shape, (list, tuple)) and isinstance(raw_shape[0], (list, tuple)):
            img_h, img_w = raw_shape[0][0], raw_shape[0][1]
        else:
            img_h, img_w = raw_shape[0], raw_shape[1]

        import numpy as np
        lidar2img_np = np.stack([np.stack(m['lidar2img']) for m in img_metas])
        # Save prev_bev_fp (FP32 temporal state) for TRT INT8 calibration
        # Use zeros for first frame (no prev), then actual prev_bev for subsequent frames
        prev_bev_np = (prev_bev_fp.cpu().float().numpy()
                       if prev_bev_fp is not None else
                       np.zeros((40000, 1, 256), dtype=np.float32))
        calib_inputs.append({
            'feat0':       mlvl_feats[0].cpu().float().numpy(),
            'feat1':       mlvl_feats[1].cpu().float().numpy(),
            'feat2':       mlvl_feats[2].cpu().float().numpy(),
            'feat3':       mlvl_feats[3].cpu().float().numpy(),
            'can_bus':     np.array(img_metas[0]['can_bus'], dtype=np.float32),
            'lidar2img':   lidar2img_np.astype(np.float32),
            'image_shape': np.array([img_h, img_w], dtype=np.float32),
            'prev_bev':    prev_bev_np,
            'use_prev_bev': np.array(step > 0),  # True after first frame
        })

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print('='*56)
    print(f'PTQ Accuracy  W{args.n_bits_w}A{args.n_bits_a}  ({len(cos_sims)} samples)')
    print('='*56)
    print(f'  Cosine Similarity : mean={np.mean(cos_sims):.7f}  '
          f'min={np.min(cos_sims):.7f}')
    print(f'  Max Abs Diff      : mean={np.mean(max_diffs):.4e}  '
          f'max={np.max(max_diffs):.4e}')
    print(f'  Mean Abs Diff     : {np.mean(mean_diffs):.4e}')
    min_cos = np.min(cos_sims)
    if min_cos > 0.999:
        verdict = 'PASS ✓  INT8 TRT build recommended.'
    elif min_cos > 0.99:
        verdict = 'MARGINAL  Consider --adaround or larger calib set.'
    else:
        verdict = 'FAIL  Need AdaRound (calibrate_univ2x.py --adaround) or W8A16 fallback.'
    print(f'\n  VERDICT: {verdict}')
    print('='*56)

    # ── Dump TRT calibration inputs ───────────────────────────────────────
    if calib_inputs and args.dump_calib:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_calib)), exist_ok=True)
        with open(args.dump_calib, 'wb') as f:
            pickle.dump(calib_inputs, f)
        print(f'\nSaved {len(calib_inputs)} TRT calib input tensors → {args.dump_calib}')
        print('(Pass this to build_trt_int8_univ2x.py --cali-data)')


if __name__ == '__main__':
    main()
