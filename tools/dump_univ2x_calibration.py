"""dump_univ2x_calibration.py — Capture BEV encoder inputs during real inference.

The BEV encoder TRT engine (BEVEncoderWrapper) expects:
    feat0       (1, 1, 256, 136, 240)  — FPN level 0
    feat1       (1, 1, 256,  68, 120)  — FPN level 1
    feat2       (1, 1, 256,  34,  60)  — FPN level 2
    feat3       (1, 1, 256,  17,  30)  — FPN level 3
    can_bus     (18,)
    lidar2img   (1, 1, 4, 4)
    image_shape (2,)
    prev_bev    (bev_h*bev_w, 1, embed_dims)
    use_prev_bev scalar float32

These are captured by hooking ``pts_bbox_head.get_bev_features`` on both the
ego and infra sub-models inside the cooperative MultiAgent forward pass.
The DataLoader (and backbone) run in PyTorch; the hook fires just before the
BEV transformer would be called, giving us the exact inputs in the right shape.

Usage
-----
    cd /home/jichengzhi/UniV2X
    conda run -n UniV2X_2.0 python tools/dump_univ2x_calibration.py \\
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \\
        ckpts/univ2x_coop_e2e_stg2.pth \\
        --n-frames 50 \\
        --out-ego  calibration/bev_encoder_ego_calib_inputs.pkl \\
        --out-infra calibration/bev_encoder_infra_calib_inputs.pkl
"""

import argparse
import os
import pickle
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, 'projects')

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent


# ---------------------------------------------------------------------------
# BEV input extraction (mirrors test_trt.py _extract_bev_inputs)
# ---------------------------------------------------------------------------

def _extract_bev_inputs(mlvl_feats, img_metas, prev_bev, head):
    """Build the numpy dict for BEVEncoderWrapper TRT calibration."""
    device = mlvl_feats[0].device
    bs     = mlvl_feats[0].shape[0]

    can_bus = torch.tensor(
        img_metas[0]['can_bus'], dtype=torch.float32, device=device)

    lidar2img_np = np.stack([np.stack(m['lidar2img']) for m in img_metas])
    lidar2img = torch.tensor(lidar2img_np, dtype=torch.float32, device=device)

    raw_shape = img_metas[0]['img_shape']
    if isinstance(raw_shape, (list, tuple)) and \
            isinstance(raw_shape[0], (list, tuple)):
        img_h, img_w = raw_shape[0][0], raw_shape[0][1]
    else:
        img_h, img_w = raw_shape[0], raw_shape[1]
    image_shape = torch.tensor([img_h, img_w], dtype=torch.float32, device=device)

    bev_h      = head.bev_h
    bev_w      = head.bev_w
    embed_dims = head.embed_dims
    num_query  = bev_h * bev_w

    if prev_bev is None or prev_bev.shape[0] != num_query:
        prev_bev_t   = torch.zeros(num_query, bs, embed_dims,
                                   dtype=torch.float32, device=device)
        use_prev_bev = torch.tensor(0.0, dtype=torch.float32, device=device)
    else:
        prev_bev_t   = prev_bev.float().contiguous()
        use_prev_bev = torch.tensor(1.0, dtype=torch.float32, device=device)

    assert len(mlvl_feats) == 4, \
        f'BEV encoder expects 4 FPN levels, got {len(mlvl_feats)}'

    def _np(t):
        return t.cpu().float().numpy()

    return {
        'feat0':        _np(mlvl_feats[0].float().contiguous()),
        'feat1':        _np(mlvl_feats[1].float().contiguous()),
        'feat2':        _np(mlvl_feats[2].float().contiguous()),
        'feat3':        _np(mlvl_feats[3].float().contiguous()),
        'can_bus':      _np(can_bus.contiguous()),
        'lidar2img':    _np(lidar2img.contiguous()),
        'image_shape':  _np(image_shape.contiguous()),
        'prev_bev':     _np(prev_bev_t.contiguous()),
        'use_prev_bev': _np(use_prev_bev.contiguous()),
    }


# ---------------------------------------------------------------------------
# Hook attachment
# ---------------------------------------------------------------------------

def attach_bev_capture_hook(model_agent, calib_list: list, label: str,
                             n_target: int):
    """Patch pts_bbox_head.get_bev_features to capture BEV encoder inputs.

    The original function is called normally so inference continues unchanged.
    Captured samples are appended to ``calib_list`` until ``n_target`` reached.
    """
    head = model_agent.pts_bbox_head
    orig = head.get_bev_features

    def _hook(mlvl_feats, img_metas, prev_bev=None):
        result = orig(mlvl_feats, img_metas, prev_bev=prev_bev)
        if len(calib_list) < n_target:
            sample = _extract_bev_inputs(mlvl_feats, img_metas, prev_bev, head)
            calib_list.append(sample)
        return result

    head.get_bev_features = _hook
    print(f'[{label}] BEV capture hook attached')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Dump BEV encoder calibration inputs for INT8 TRT')
    p.add_argument('config',      help='Test config file path')
    p.add_argument('checkpoint',  help='Checkpoint file (.pth)')
    p.add_argument('--n-frames',  type=int, default=50,
                   help='Number of frames to capture per agent (default: 50)')
    p.add_argument('--out-ego',   default='calibration/bev_encoder_ego_calib_inputs.pkl',
                   help='Output PKL path for ego BEV encoder inputs')
    p.add_argument('--out-infra', default='calibration/bev_encoder_infra_calib_inputs.pkl',
                   help='Output PKL path for infra BEV encoder inputs')
    p.add_argument('--cfg-options', nargs='+', action='append')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        for kv in args.cfg_options:
            cfg.merge_from_dict(dict(kv))

    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        plugin_dir = getattr(cfg, 'plugin_dir', '')
        if plugin_dir:
            parts = os.path.dirname(plugin_dir).split('/')
            importlib.import_module('.'.join(p for p in parts if p))

    cfg.model_ego_agent.pretrained = None

    # ── Dataset / DataLoader ─────────────────────────────────────────────────
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    print(f'Dataset: {len(dataset)} samples  |  capturing up to {args.n_frames} frames')

    # ── Build model (mirrors dump_downstream_calibration.py) ─────────────────
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

    cfg.model_ego_agent.train_cfg = None
    model_ego = build_model(cfg.model_ego_agent, test_cfg=cfg.get('test_cfg'))
    load_from = cfg.model_ego_agent.load_from
    if load_from:
        load_checkpoint(model_ego, load_from, map_location='cpu',
                        revise_keys=[(r'^model_ego_agent\.', '')])

    model_multi = MultiAgent(model_ego, model_other_agents)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model_multi)

    load_checkpoint(model_multi, args.checkpoint, map_location='cpu')

    model_multi = MMDataParallel(model_multi.cuda(), device_ids=[0])
    model_multi.eval()

    inner = model_multi.module   # MultiAgent
    ego   = inner.model_ego_agent

    # ── Attach capture hooks ─────────────────────────────────────────────────
    n_target = min(args.n_frames, len(dataset))

    calib_ego   = []
    calib_infra = []

    attach_bev_capture_hook(ego, calib_ego, 'ego', n_target)

    infra_model = None
    for inf_name in inner.other_agent_names:
        infra_model = getattr(inner, inf_name)
        attach_bev_capture_hook(infra_model, calib_infra, 'infra', n_target)
        break   # single infra agent

    if infra_model is None:
        print('[WARN] No infra agent found — infra calibration will be empty')

    # ── Capture loop ─────────────────────────────────────────────────────────
    prog = mmcv.ProgressBar(n_target)

    for i, data in enumerate(data_loader):
        if len(calib_ego) >= n_target and (
                infra_model is None or len(calib_infra) >= n_target):
            break

        with torch.no_grad():
            model_multi(return_loss=False, rescale=True, **data)

        prog.update()

    print(f'\nCapture done: {len(calib_ego)} ego frames, {len(calib_infra)} infra frames')

    # Print first-sample shapes for verification
    if calib_ego:
        print('\n── Ego sample shapes ──────────────────────────────────────')
        for k, v in calib_ego[0].items():
            print(f'  {k}: {v.shape}  dtype={v.dtype}')
    if calib_infra:
        print('\n── Infra sample shapes ────────────────────────────────────')
        for k, v in calib_infra[0].items():
            print(f'  {k}: {v.shape}  dtype={v.dtype}')

    # ── Save ─────────────────────────────────────────────────────────────────
    for out_path, data_list, label in [
        (args.out_ego,   calib_ego,   'ego'),
        (args.out_infra, calib_infra, 'infra'),
    ]:
        if not data_list:
            print(f'[WARN] No {label} frames captured — output not written')
            continue
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, 'wb') as f:
            pickle.dump(data_list, f)
        size_mb = os.path.getsize(out_path) / 1024 ** 2
        print(f'Saved {len(data_list)} {label} frames → {out_path}  ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
