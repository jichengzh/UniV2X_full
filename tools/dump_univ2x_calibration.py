"""
dump_univ2x_calibration.py — Dump N calibration batches from the UniV2X DataLoader.

Usage:
    python tools/dump_univ2x_calibration.py \\
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \\
        ckpts/univ2x_coop_e2e_stg2.pth \\
        --n-cal 32 \\
        --out calibration/cali_data.pkl

Output: a pickle file containing a list of N dicts, each being one DataLoader batch
(the exact format consumed by the UniV2X model's forward method).
"""

import argparse
import os
import pickle
import sys

import mmcv
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset


def parse_args():
    p = argparse.ArgumentParser(description='Dump UniV2X calibration data')
    p.add_argument('config',        help='Test config file path')
    p.add_argument('checkpoint',    help='Checkpoint file')
    p.add_argument('--n-cal',       type=int, default=32,
                   help='Number of calibration samples to dump (default: 32)')
    p.add_argument('--out',         default='calibration/cali_data.pkl',
                   help='Output pickle path')
    p.add_argument('--plugin',      default='plugins/build/libuniv2x_plugins.so',
                   help='Path to custom TRT plugin SO (not used here, kept for parity)')
    p.add_argument('--cfg-options', nargs='+', action='append')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Env setup ────────────────────────────────────────────────────────
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
        if cfg.data.test.get('pipeline') is not None:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # ── Dataset ──────────────────────────────────────────────────────────
    print(f'Building dataset...')
    dataset = build_dataset(cfg.data.test)
    print(f'Dataset size: {len(dataset)} samples')

    n_cal = min(args.n_cal, len(dataset))
    print(f'Collecting {n_cal} calibration samples...')

    cali_data = []
    for i in range(n_cal):
        sample = dataset[i]
        # Convert to batch format (add batch dim via collate)
        # DataLoader items are dicts; we keep them as-is for cali_data
        # (GetLayerInpOut calls _to_device internally)
        cali_data.append(sample)
        if (i + 1) % 8 == 0:
            print(f'  {i+1}/{n_cal}')

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(cali_data, f)
    print(f'\nSaved {n_cal} calibration samples → {args.out}')
    print(f'File size: {os.path.getsize(args.out) / 1024 / 1024:.1f} MB')


if __name__ == '__main__':
    main()
