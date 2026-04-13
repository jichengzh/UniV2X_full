"""dump_downstream_calibration.py — Capture downstream-head inputs during real inference.

The downstream TRT engine (DownstreamHeadsWrapper) expects:
    bev_embed      (H*W, 1, C)              — BEV features after V2X fusion
    query_feats    (num_dec, 1, N, C)       — detection decoder hidden states (all N queries)
    all_bbox_preds (num_dec, 1, N, 10)      — normalised bbox predictions
    all_cls_scores (num_dec, 1, N, num_cls) — class logits
    lane_query     (1, 300, C)              — map/lane queries from seg head
    lane_query_pos (1, 300, C)              — lane query positions
    command        int64                    — navigation command (ego only)

Hook points
-----------
  pts_bbox_head.get_detections()   → bev_embed (1st arg), query_feats/bbox/cls (outputs)
  seg_head.forward_test()          → lane_query/lane_query_pos from result_seg[0]['args_tuple']
  univ2x_e2e.forward_test()        → command kwarg

Notes
-----
* In V2X cooperative mode, get_detections is called with N > num_query+1 (up to 1101).
  We slice the first num_query+1 (= 901) to match the exported ONNX shape.
* If the model has no seg_head (infra in some configs), lane queries default to zeros.
* ego and infra calibration data are saved to separate PKL files.

Usage
-----
    cd /home/jichengzhi/UniV2X
    conda run -n UniV2X_2.0 python tools/dump_downstream_calibration.py \\
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \\
        ckpts/univ2x_coop_e2e_stg2.pth \\
        --n-frames 50 \\
        --out-ego  calibration/downstream_ego_calib_inputs.pkl \\
        --out-infra calibration/downstream_infra_calib_inputs.pkl
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
# Per-frame capture buffer
# ---------------------------------------------------------------------------

class FrameCapture:
    """Single-frame capture buffer for downstream head inputs."""

    def __init__(self, label: str = ''):
        self.label = label
        self.reset()

    def reset(self):
        self.bev_embed      = None   # (H*W, 1, C)
        self.query_feats    = None   # (num_dec, 1, N, C)
        self.all_bbox_preds = None   # (num_dec, 1, N, 10)
        self.all_cls_scores = None   # (num_dec, 1, N, num_cls)
        self.lane_query     = None   # (1, 300, C)   or zeros
        self.lane_query_pos = None   # (1, 300, C)   or zeros

    def is_det_ready(self):
        return all(getattr(self, f) is not None
                   for f in ('bev_embed', 'query_feats',
                              'all_bbox_preds', 'all_cls_scores'))

    def to_dict(self, command=None, num_query: int = 901):
        """Assemble numpy dict in DownstreamHeadsWrapper input format.

        Slices query tensors to first `num_query` rows to guarantee the
        shape matches the exported ONNX (independent of V2X augmentation).
        """
        def _t(tensor, max_q=None):
            t = tensor.cpu().float()
            if max_q is not None and t.shape[2] > max_q:
                t = t[:, :, :max_q, :]
            return t.numpy()

        def _lane(tensor, max_lane: int = 300):
            if tensor is None:
                return None
            t = tensor.cpu().float()
            if t.shape[1] > max_lane:
                t = t[:, :max_lane, :]
            return t.numpy()

        d = {
            'bev_embed':      self.bev_embed.cpu().float().numpy(),
            'query_feats':    _t(self.query_feats, num_query),
            'all_bbox_preds': _t(self.all_bbox_preds, num_query),
            'all_cls_scores': _t(self.all_cls_scores, num_query),
            'lane_query':     _lane(self.lane_query),
            'lane_query_pos': _lane(self.lane_query_pos),
        }
        if command is not None:
            d['command'] = np.array(int(command), dtype=np.int64)
        return d


# ---------------------------------------------------------------------------
# Hook attachment
# ---------------------------------------------------------------------------

def attach_det_hook(model_agent, capture: FrameCapture):
    """Patch pts_bbox_head.get_detections to capture BEV + decoder outputs."""
    orig = model_agent.pts_bbox_head.get_detections

    def _hook(bev_embed, object_query_embeds=None, ref_points=None,
              img_metas=None):
        result = orig(bev_embed,
                      object_query_embeds=object_query_embeds,
                      ref_points=ref_points,
                      img_metas=img_metas)
        capture.bev_embed      = bev_embed.detach()
        capture.query_feats    = result['query_feats'].detach()
        capture.all_bbox_preds = result['all_bbox_preds'].detach()
        capture.all_cls_scores = result['all_cls_scores'].detach()
        return result

    model_agent.pts_bbox_head.get_detections = _hook


def attach_seg_hook(model_agent, capture: FrameCapture):
    """Patch seg_head.forward_test to capture lane_query/lane_query_pos.

    The seg head (panseg_head) returns (result_seg, drivable_gt, drivable_pred).
    result_seg[0]['args_tuple'] = [memory, mask, pos, lane_query, None, lane_query_pos, hw_lvl]
    """
    if not hasattr(model_agent, 'seg_head'):
        return False

    orig = model_agent.seg_head.forward_test

    def _hook(*args, **kwargs):
        ret = orig(*args, **kwargs)
        # ret can be (result_seg, drivable_gt, drivable_pred) or just result_seg
        result_seg = ret[0] if isinstance(ret, (list, tuple)) else ret
        seg_dict = result_seg[0] if isinstance(result_seg, (list, tuple)) else result_seg
        args_tuple = seg_dict.get('args_tuple', None) if isinstance(seg_dict, dict) else None
        if args_tuple is not None:
            # [memory, memory_mask, memory_pos, lane_query, None, lane_query_pos, hw_lvl]
            capture.lane_query     = args_tuple[3].detach()
            capture.lane_query_pos = args_tuple[5].detach()
        return ret

    model_agent.seg_head.forward_test = _hook
    return True


def attach_command_hook(model_agent, captured_command: list):
    """Patch univ2x_e2e.forward_test to capture the command kwarg."""
    orig = model_agent.forward_test

    def _hook(*args, **kwargs):
        cmd = kwargs.get('command', None)
        if cmd is not None:
            # DataLoader wraps scalars in lists of tensors
            if isinstance(cmd, (list, tuple)) and len(cmd) > 0:
                cmd = cmd[0]
            if isinstance(cmd, torch.Tensor):
                captured_command[0] = int(cmd.item())
            else:
                captured_command[0] = int(cmd)
        return orig(*args, **kwargs)

    model_agent.forward_test = _hook


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Dump downstream-head calibration inputs for INT8 TRT')
    p.add_argument('config',      help='Test config file path')
    p.add_argument('checkpoint',  help='Checkpoint file (.pth)')
    p.add_argument('--n-frames',  type=int, default=50,
                   help='Number of frames to capture (default: 50)')
    p.add_argument('--out-ego',   default='calibration/downstream_ego_calib_inputs.pkl',
                   help='Output PKL path for ego downstream inputs')
    p.add_argument('--out-infra', default='calibration/downstream_infra_calib_inputs.pkl',
                   help='Output PKL path for infra downstream inputs')
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

    # ── Build model (mirrors test_trt.py exactly) ────────────────────────────
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

    # num_query+1 = 901 for the standard ego model  (matches ONNX export shape)
    num_query_ego   = ego.num_query + 1 if hasattr(ego, 'num_query') else 901
    print(f'ego num_query = {num_query_ego}')

    # ── Attach capture hooks ─────────────────────────────────────────────────
    cap_ego   = FrameCapture('ego')
    cap_infra = FrameCapture('infra')

    attach_det_hook(ego, cap_ego)
    has_seg_ego = attach_seg_hook(ego, cap_ego)
    print(f'ego   seg_head hook: {"attached" if has_seg_ego else "no seg_head (lane=zeros)"}')

    infra_model = None
    num_query_infra = num_query_ego
    for inf_name in inner.other_agent_names:
        infra_model = getattr(inner, inf_name)
        attach_det_hook(infra_model, cap_infra)
        has_seg_inf = attach_seg_hook(infra_model, cap_infra)
        num_query_infra = (infra_model.num_query + 1
                           if hasattr(infra_model, 'num_query') else 901)
        print(f'infra seg_head hook: {"attached" if has_seg_inf else "no seg_head (lane=zeros)"}')
        print(f'infra num_query = {num_query_infra}')
        break   # single infra agent

    # Command capture on ego model
    captured_command = [None]
    attach_command_hook(ego, captured_command)

    # ── Capture loop ─────────────────────────────────────────────────────────
    calib_ego   = []
    calib_infra = []
    n_target = min(args.n_frames, len(dataset))

    prog = mmcv.ProgressBar(n_target)

    for i, data in enumerate(data_loader):
        if len(calib_ego) >= n_target and len(calib_infra) >= n_target:
            break

        cap_ego.reset()
        cap_infra.reset()
        captured_command[0] = None

        with torch.no_grad():
            model_multi(return_loss=False, rescale=True, **data)

        # ── Ego sample ────────────────────────────────────────────────────────
        if len(calib_ego) < n_target and cap_ego.is_det_ready():
            # lane_query may still be None if seg_head hook wasn't triggered
            # (e.g., the first frame in a scene has no seg output) — use zeros
            if cap_ego.lane_query is None:
                C = cap_ego.query_feats.shape[-1]
                cap_ego.lane_query     = torch.zeros(1, 300, C)
                cap_ego.lane_query_pos = torch.zeros(1, 300, C)
            sample_ego = cap_ego.to_dict(command=captured_command[0],
                                         num_query=num_query_ego)
            calib_ego.append(sample_ego)

        # ── Infra sample ──────────────────────────────────────────────────────
        if len(calib_infra) < n_target and cap_infra.is_det_ready():
            if cap_infra.lane_query is None:
                C = cap_infra.query_feats.shape[-1]
                cap_infra.lane_query     = torch.zeros(1, 300, C)
                cap_infra.lane_query_pos = torch.zeros(1, 300, C)
            sample_infra = cap_infra.to_dict(command=None,
                                              num_query=num_query_infra)
            calib_infra.append(sample_infra)

        prog.update()

    print(f'\nCapture done: {len(calib_ego)} ego frames, {len(calib_infra)} infra frames')

    # Print first-sample shapes for verification
    if calib_ego:
        print('\n── Ego sample shapes ─────────────────────────────────────')
        for k, v in calib_ego[0].items():
            print(f'  {k}: {v.shape if hasattr(v, "shape") else v}')
    if calib_infra:
        print('\n── Infra sample shapes ────────────────────────────────────')
        for k, v in calib_infra[0].items():
            print(f'  {k}: {v.shape if hasattr(v, "shape") else v}')

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
