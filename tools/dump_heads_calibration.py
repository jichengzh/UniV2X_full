"""dump_heads_calibration.py — Capture detection-head inputs during real inference.

The V2X detection head TRT engine (HeadsDecoderOnlyWrapper, N_PAD=1101) expects:
    bev_embed        (40000, 1, 256)   — BEV features (after V2X fusion)
    track_query      (1101, 512)       — zero-padded pos+feat concat
    track_ref_pts    (1101, 3)         — zero-padded inv-sigmoid ref points

Hook point: pts_bbox_head.get_detections() on the ego model.
In V2X mode, after cross_agent_query_interaction, the query count N can be
anywhere from 901 to 1101.  We zero-pad to N_PAD=1101 to match the ONNX shape.

Usage
-----
    cd /home/jichengzhi/UniV2X
    conda run -n UniV2X_2.0 python tools/dump_heads_calibration.py \\
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \\
        ckpts/univ2x_coop_e2e_stg2.pth \\
        --n-frames 50 \\
        --out calibration/heads_ego_calib_inputs.pkl
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


N_PAD = 1101   # 901 ego queries + 200 max infra complementation slots


# ---------------------------------------------------------------------------
# Hook attachment
# ---------------------------------------------------------------------------

def attach_heads_capture_hook(model_agent, calib_list: list, label: str,
                               n_target: int):
    """Patch pts_bbox_head.get_detections to capture detection head inputs.

    Captures bev_embed, track_query (object_query_embeds), and track_ref_pts
    (ref_points).  Queries are zero-padded to N_PAD=1101 to match the ONNX
    export shape of HeadsDecoderOnlyWrapper.
    """
    orig = model_agent.pts_bbox_head.get_detections

    def _hook(bev_embed, object_query_embeds=None, ref_points=None,
              img_metas=None):
        result = orig(bev_embed,
                      object_query_embeds=object_query_embeds,
                      ref_points=ref_points,
                      img_metas=img_metas)

        if len(calib_list) < n_target:
            N = object_query_embeds.shape[0]
            C2 = object_query_embeds.shape[1]
            device = bev_embed.device
            dtype = bev_embed.dtype

            # Zero-pad queries to N_PAD
            if N < N_PAD:
                pad_q = torch.zeros(N_PAD - N, C2, device=device, dtype=dtype)
                track_query = torch.cat([object_query_embeds, pad_q], dim=0)
                pad_r = torch.zeros(N_PAD - N, 3, device=device, dtype=dtype)
                track_ref_pts = torch.cat([ref_points, pad_r], dim=0)
            elif N > N_PAD:
                track_query = object_query_embeds[:N_PAD]
                track_ref_pts = ref_points[:N_PAD]
            else:
                track_query = object_query_embeds
                track_ref_pts = ref_points

            def _np(t):
                return t.cpu().float().numpy()

            sample = {
                'bev_embed':     _np(bev_embed.float().contiguous()),
                'track_query':   _np(track_query.float().contiguous()),
                'track_ref_pts': _np(track_ref_pts.float().contiguous()),
            }
            calib_list.append(sample)

        return result

    model_agent.pts_bbox_head.get_detections = _hook
    print(f'[{label}] Detection head capture hook attached (N_PAD={N_PAD})')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Dump detection-head calibration inputs for INT8 TRT')
    p.add_argument('config',      help='Test config file path')
    p.add_argument('checkpoint',  help='Checkpoint file (.pth)')
    p.add_argument('--n-frames',  type=int, default=50,
                   help='Number of frames to capture (default: 50)')
    p.add_argument('--out',       default='calibration/heads_ego_calib_inputs.pkl',
                   help='Output PKL path for heads calibration inputs')
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
    ego = inner.model_ego_agent

    # ── Attach capture hook on ego ───────────────────────────────────────────
    n_target = min(args.n_frames, len(dataset))
    calib_list = []
    attach_heads_capture_hook(ego, calib_list, 'ego', n_target)

    # ── Capture loop ─────────────────────────────────────────────────────────
    prog = mmcv.ProgressBar(n_target)

    for i, data in enumerate(data_loader):
        if len(calib_list) >= n_target:
            break

        with torch.no_grad():
            model_multi(return_loss=False, rescale=True, **data)

        prog.update()

    print(f'\nCapture done: {len(calib_list)} frames')

    # Print first-sample shapes for verification
    if calib_list:
        print('\n── Sample shapes ──────────────────────────────────────────')
        for k, v in calib_list[0].items():
            print(f'  {k}: {v.shape}  dtype={v.dtype}')

    # ── Save ─────────────────────────────────────────────────────────────────
    if not calib_list:
        print('[WARN] No frames captured — output not written')
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(calib_list, f)
    size_mb = os.path.getsize(args.out) / 1024 ** 2
    print(f'Saved {len(calib_list)} frames → {args.out}  ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
