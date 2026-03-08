"""Validate downstream-heads TRT engine against PyTorch forward_trt.

Step A (random weights) — structure check:
  Verifies the TRT engine parses, runs, and produces correct-shaped outputs
  with no NaN/Inf.  Accuracy comparison is skipped because TRT weights come
  from a separate random export while this script re-initialises randomly.

Step B (real checkpoint) — accuracy check:
  Both PyTorch wrapper and TRT engine use the same checkpoint, so outputs
  should be numerically close.

Usage
-----
# Step A — no checkpoint (shape/sanity check)
python tools/validate_downstream_trt.py \
    --engine trt_engines/univ2x_ego_downstream_50_rand.trt \
    --model ego

# Step B — real checkpoint (accuracy check)
python tools/validate_downstream_trt.py \
    --engine trt_engines/univ2x_ego_downstream.trt \
    --model ego \
    --checkpoint ckpts/univ2x_coop_e2e_stg1.pth
"""

import argparse
import ctypes
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, 'projects')
sys.path.insert(0, 'tools')

import mmcv
from mmcv.utils.registry import Registry

_orig = Registry._register_module
def _force(self, module=None, module_name=None, force=False, **kwargs):
    try:
        _orig(self, module=module, module_name=module_name, force=False)
    except (KeyError, TypeError):
        pass
Registry._register_module = _force
import mmdet3d_plugin  # noqa
Registry._register_module = _orig

import torch
import torch.nn as nn
import numpy as np

from export_onnx_univ2x import (
    DownstreamHeadsWrapper,
    _register_onnx_symbolics,
    build_model_from_cfg,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--engine', required=True, help='TRT engine file')
    p.add_argument('--config',
                   default='projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py')
    p.add_argument('--model', choices=['ego', 'infra'], default='ego')
    p.add_argument('--checkpoint', default=None,
                   help='Checkpoint (Step B accuracy). If absent, Step A shape-only.')
    p.add_argument('--plugin', default='plugins/build/libuniv2x_plugins.so')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def build_pytorch_wrapper(cfg, model_key, checkpoint):
    model = build_model_from_cfg(cfg, model_key, ckpt_path=checkpoint,
                                 random_weights=(checkpoint is None))
    pc_range = getattr(cfg, model_key).motion_head.get(
        'pc_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    wrapper = DownstreamHeadsWrapper(model, pc_range).cuda().eval()

    try:
        bev_h, bev_w = model.occ_head.bev_size
    except Exception:
        bev_h = bev_w = 200
    det_head = model.pts_bbox_head
    det_head.bev_h = bev_h
    det_head.bev_w = bev_w
    if hasattr(det_head, 'positional_encoding'):
        from mmdet.models.utils.positional_encoding import LearnedPositionalEncoding
        nf = det_head.positional_encoding.num_feats
        det_head.positional_encoding = LearnedPositionalEncoding(nf, bev_h, bev_w).cuda()

    return wrapper, model, bev_h, bev_w


def make_dummy(model, bev_h, bev_w, seed):
    torch.manual_seed(seed)
    embed_dims = model.pts_bbox_head.embed_dims
    num_bev    = bev_h * bev_w
    num_query  = model.num_query + 1
    num_dec    = 6
    num_cls    = 10
    M          = 300
    bev_embed      = torch.randn(num_bev, 1, embed_dims, device='cuda')
    query_feats    = torch.randn(num_dec, 1, num_query, embed_dims, device='cuda')
    all_bbox_preds = torch.randn(num_dec, 1, num_query, 10, device='cuda')
    all_cls_scores = torch.randn(num_dec, 1, num_query, num_cls, device='cuda')
    lane_query     = torch.zeros(1, M, embed_dims, device='cuda')
    lane_query_pos = torch.zeros(1, M, embed_dims, device='cuda')
    command        = torch.tensor(0, dtype=torch.long, device='cuda')
    return (bev_embed, query_feats, all_bbox_preds, all_cls_scores,
            lane_query, lane_query_pos, command)


def run_pytorch(wrapper, dummy):
    _register_onnx_symbolics()
    with torch.no_grad():
        return [o.cpu().numpy() for o in wrapper(*dummy)]


def run_trt(engine_path, plugin_path, dummy):
    import tensorrt as trt

    ctypes.CDLL(plugin_path)
    trt.init_libnvinfer_plugins(None, '')

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    out_tensors = {}
    input_names = ['bev_embed', 'query_feats', 'all_bbox_preds', 'all_cls_scores',
                   'lane_query', 'lane_query_pos', 'command']
    output_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = ctx.get_tensor_shape(name)
            out_tensors[name] = torch.zeros(*shape, dtype=torch.float32, device='cuda')
            output_names.append(name)

    for name, tensor in zip(input_names, dummy):
        t = tensor.contiguous().cuda()
        if name == 'command':
            t = t.to(torch.int64)
        ctx.set_tensor_address(name, t.data_ptr())
    for name, t in out_tensors.items():
        ctx.set_tensor_address(name, t.data_ptr())

    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    return output_names, {n: t.cpu().numpy() for n, t in out_tensors.items()}


def cosine_sim(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    args = parse_args()
    do_accuracy = (args.checkpoint is not None)

    cfg = mmcv.Config.fromfile(args.config)
    model_key = 'model_ego_agent' if args.model == 'ego' else 'model_other_agent_inf'
    if not hasattr(cfg, model_key):
        model_key = 'model'

    print(f'Building PyTorch wrapper ({args.model}, ckpt={args.checkpoint}) ...')
    wrapper, model, bev_h, bev_w = build_pytorch_wrapper(cfg, model_key, args.checkpoint)

    print(f'Generating dummy inputs (seed={args.seed}, bev={bev_h}×{bev_w}) ...')
    dummy = make_dummy(model, bev_h, bev_w, args.seed)

    print('Running PyTorch forward ...')
    pt_outs = run_pytorch(wrapper, dummy)
    output_names_pt = (['traj_scores', 'traj_preds', 'occ_logits', 'sdc_traj']
                       if len(pt_outs) == 4 else
                       ['traj_scores', 'traj_preds', 'occ_logits'])
    print(f'  {len(pt_outs)} outputs: {[o.shape for o in pt_outs]}')
    for name, o in zip(output_names_pt, pt_outs):
        has_nan = np.any(np.isnan(o)) or np.any(np.isinf(o))
        print(f'  {name}: {o.shape}  {"NaN/Inf ❌" if has_nan else "finite ✅"}')

    print(f'\nRunning TRT engine ({args.engine}) ...')
    trt_names, trt_outs = run_trt(args.engine, args.plugin, dummy)
    print(f'  {len(trt_outs)} outputs: {list(trt_outs.keys())}')
    for name, o in trt_outs.items():
        has_nan = np.any(np.isnan(o)) or np.any(np.isinf(o))
        print(f'  {name}: {o.shape}  {"NaN/Inf ❌" if has_nan else "finite ✅"}')

    # Shape check
    print('\n─── Shape check ────────────────────────────────────────────────')
    shapes_ok = True
    for name, pt in zip(output_names_pt, pt_outs):
        if name not in trt_outs:
            print(f'  ❌ {name}: MISSING in TRT')
            shapes_ok = False
        elif pt.shape != trt_outs[name].shape:
            print(f'  ❌ {name}: PT={pt.shape} vs TRT={trt_outs[name].shape}')
            shapes_ok = False
        else:
            print(f'  ✅ {name}: {pt.shape}')
    print(f'Shape check: {"PASS ✅" if shapes_ok else "FAIL ❌"}')

    if not do_accuracy:
        print('\n[Step A] Random-weights mode — accuracy comparison skipped.')
        print('(Both TRT and PyTorch use DIFFERENT random initializations.)')
        print('Re-run with --checkpoint for Step B accuracy validation.')
        return

    # Accuracy check (Step B: checkpoint provided, weights match)
    print('\n─── Accuracy comparison (Step B) ──────────────────────────────')
    print(f'{"Output":<16} {"Shape":<30} {"MaxAbs":>10} {"MeanAbs":>10} {"CosSim":>10}')
    print('─' * 80)
    all_pass = True
    for name, pt_out in zip(output_names_pt, pt_outs):
        if name not in trt_outs:
            all_pass = False
            continue
        trt_out = trt_outs[name]
        diff = np.abs(pt_out.astype(np.float32) - trt_out.astype(np.float32))
        max_d  = float(diff.max())
        mean_d = float(diff.mean())
        cos    = cosine_sim(pt_out, trt_out)
        ok = mean_d < 1e-2 and cos > 0.999
        flag = '✅' if ok else '❌'
        print(f'{flag} {name:<14} {str(pt_out.shape):<30} {max_d:>10.4f} {mean_d:>10.2e} {cos:>10.7f}')
        if not ok:
            all_pass = False
    print('─' * 80)
    print(f'\nStep B Accuracy: {"PASS ✅" if all_pass else "FAIL ❌"}')


if __name__ == '__main__':
    main()
