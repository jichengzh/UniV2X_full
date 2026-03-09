"""VAD-TRT style end-to-end TRT evaluation for UniV2X.

Design (follows vad-trt/export_eval/test_tensorrt.py pattern)
--------------------------------------------------------------
* Single GPU, **non-distributed** — ``MMDataParallel`` + simple for-loop.
* Exact same data path as ``custom_multi_gpu_test``:
    - planning / occ metrics computed *inside* the loop (same as original)
    - returns the same ``ret_results`` dict that ``dataset.evaluate()`` expects
* TRT engines are injected via method monkey-patching *before* the loop,
  so the rest of the call chain (MultiAgent → UniV2X → heads) is unchanged.

Hook points
-----------
Hook A — BEV encoder (Phase 1 TRT)
    Replaces ``pts_bbox_head.get_bev_features`` on the ego sub-model (and
    optionally on the infra sub-model).  The backbone (DCNv2/ResNet-FPN)
    continues to run in PyTorch; only the BEV transformer is TRT.
    ``prev_bev`` state is already managed by the outer frame loop — the hook
    just receives the correct value as an argument.

Usage
-----
# Ego BEV encoder TRT only
python tools/test_trt.py \\
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \\
    ckpts/univ2x_coop_e2e_stg2.pth \\
    --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_200.trt \\
    --eval bbox \\
    --out output/trt_results.pkl

# Both ego + infra BEV TRT
python tools/test_trt.py \\
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \\
    ckpts/univ2x_coop_e2e_stg2.pth \\
    --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_200.trt \\
    --bev-engine-inf trt_engines/univ2x_infra_bev_encoder_200.trt \\
    --eval bbox \\
    --out output/trt_coop_results.pkl
"""

import argparse
import ctypes
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, 'projects')

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, load_checkpoint, wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent
from projects.mmdet3d_plugin.univ2x.dense_heads.occ_head_plugin import (
    IntersectionOverUnion, PanopticMetric)
from projects.mmdet3d_plugin.univ2x.dense_heads.planning_head_plugin import (
    PlanningMetric)

torch.multiprocessing.set_sharing_strategy('file_system')


# ---------------------------------------------------------------------------
# TRT engine wrapper
# ---------------------------------------------------------------------------

class TrtEngine:
    """Thin TRT engine wrapper — named-tensor address binding, no pycuda.

    Loads a serialised ``.trt`` / ``.engine`` file, binds custom plugins,
    and exposes a single ``infer(inputs_dict) -> outputs_dict`` interface.
    """

    def __init__(self, engine_path: str, plugin_path: str):
        import tensorrt as trt

        ctypes.CDLL(plugin_path)
        trt.init_libnvinfer_plugins(None, '')

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        self.input_names  = []
        self.output_names = []
        self.output_shapes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                self.output_shapes[name] = tuple(
                    self.ctx.get_tensor_shape(name))

        print(f'[TrtEngine] {engine_path}')
        print(f'  inputs : {self.input_names}')
        print(f'  outputs: {self.output_names}')

    def infer(self, inputs: dict) -> dict:
        """Run synchronous inference.

        Args:
            inputs: ``{name: contiguous CUDA tensor}``
        Returns:
            ``{name: CUDA float32 tensor}``
        """
        for name, t in inputs.items():
            self.ctx.set_tensor_address(name, t.contiguous().data_ptr())

        out = {}
        for name, shape in self.output_shapes.items():
            out[name] = torch.zeros(*shape, dtype=torch.float32, device='cuda')
            self.ctx.set_tensor_address(name, out[name].data_ptr())

        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return out


# ---------------------------------------------------------------------------
# Hook A — BEV encoder
# ---------------------------------------------------------------------------

def _extract_bev_inputs(mlvl_feats, img_metas, prev_bev, head):
    """Build the input dict for ``BEVEncoderWrapper`` TRT engine.

    Converts Python-dict ``img_metas`` fields to CUDA tensors.
    Handles both ``img_shape = [(H,W,C), ...]`` and ``img_shape = (H,W,C)``.
    """
    device = mlvl_feats[0].device
    bs     = mlvl_feats[0].shape[0]

    # can_bus: (18,)
    can_bus = torch.tensor(
        img_metas[0]['can_bus'], dtype=torch.float32, device=device)

    # lidar2img: (bs, num_cam, 4, 4)
    lidar2img_np = np.stack([np.stack(m['lidar2img']) for m in img_metas])
    lidar2img = torch.tensor(lidar2img_np, dtype=torch.float32, device=device)

    # image_shape: (2,) = [img_h, img_w]
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

    return {
        'feat0':        mlvl_feats[0].float().contiguous(),
        'feat1':        mlvl_feats[1].float().contiguous(),
        'feat2':        mlvl_feats[2].float().contiguous(),
        'feat3':        mlvl_feats[3].float().contiguous(),
        'can_bus':      can_bus.contiguous(),
        'lidar2img':    lidar2img.contiguous(),
        'image_shape':  image_shape.contiguous(),
        'prev_bev':     prev_bev_t.contiguous(),
        'use_prev_bev': use_prev_bev.contiguous(),
    }


def attach_bev_hook(model_agent, bev_engine: TrtEngine, label: str):
    """Monkey-patch ``pts_bbox_head.get_bev_features`` with TRT inference.

    The replacement function has the same signature as the original and can
    be called transparently by ``get_bevs()`` in ``univ2x_track.py``.
    """
    head = model_agent.pts_bbox_head

    def _trt_get_bev_features(mlvl_feats, img_metas, prev_bev=None):
        device = mlvl_feats[0].device
        dtype  = mlvl_feats[0].dtype
        bs     = mlvl_feats[0].shape[0]

        trt_inputs = _extract_bev_inputs(mlvl_feats, img_metas, prev_bev, head)
        trt_out    = bev_engine.infer(trt_inputs)
        bev_embed  = trt_out['bev_embed'].to(dtype=dtype, device=device)

        # Positional encoding: unchanged
        bev_mask = torch.zeros((bs, head.bev_h, head.bev_w),
                               device=device).to(dtype)
        bev_pos = head.positional_encoding(bev_mask).to(dtype)
        return bev_embed, bev_pos

    head.get_bev_features = _trt_get_bev_features
    print(f'[Hook-A/{label}] pts_bbox_head.get_bev_features → TRT')


# ---------------------------------------------------------------------------
# Single-GPU evaluation loop
# (mirrors custom_multi_gpu_test without distributed communication)
# ---------------------------------------------------------------------------

def single_gpu_test_trt(model, data_loader):
    """Single-GPU evaluation loop matching ``custom_multi_gpu_test`` output format.

    Computes planning and occ metrics inline (exactly as the original does)
    so the returned ``ret_results`` dict can be passed directly to
    ``dataset.evaluate()``.

    Args:
        model:       ``MMDataParallel``-wrapped ``MultiAgent`` (already on GPU).
        data_loader: Standard mmdet3d test dataloader.

    Returns:
        dict with keys:
            ``bbox_results``             — list, per-sample detection results
            ``occ_results_computed``     — dict (if occ head present)
            ``planning_results_computed``— dict (if planning head present)
    """
    model.eval()
    inner = model.module   # MultiAgent
    ego   = inner.model_ego_agent

    # --- init occ metrics ---
    eval_occ = getattr(ego, 'with_occ_head', False)
    if eval_occ:
        EVAL_RANGES = {'30x30': (70, 130), '100x100': (0, 200)}
        n_cls = 2
        iou_metrics = {k: IntersectionOverUnion(n_cls).cuda()
                       for k in EVAL_RANGES}
        pan_metrics = {k: PanopticMetric(n_classes=n_cls,
                                         temporally_consistent=True).cuda()
                       for k in EVAL_RANGES}
        num_occ = 0

    # --- init planning metrics ---
    eval_planning = getattr(ego, 'with_planning_head', False)
    planning_steps = data_loader.dataset.planning_steps
    if eval_planning:
        plan_metrics = PlanningMetric(n_future=planning_steps).cuda()

    bbox_results = []
    dataset      = data_loader.dataset
    prog_bar     = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        # ── planning metrics (inline, same as custom_multi_gpu_test) ────
        if eval_planning:
            pg  = result[0]['planning']['planning_gt']
            rp  = result[0]['planning']['result_planning']
            result[0]['planning_traj']    = rp['sdc_traj']
            result[0]['planning_traj_gt'] = pg['sdc_planning']
            result[0]['command']          = pg['command']
            plan_metrics(
                rp['sdc_traj'][:, :planning_steps, :2],
                pg['sdc_planning'][0][0, :, :planning_steps, :2],
                pg['sdc_planning_mask'][0][0, :, :planning_steps, :2],
                pg['segmentation'][0][:, 1:planning_steps + 1],
                pg['drivable_gt'],
            )

        # ── occ metrics (inline) ─────────────────────────────────────────
        if eval_occ:
            invalid = data['ego_agent_data']['gt_occ_has_invalid_frame'][0]
            if not invalid.item() and 'occ' in result[0]:
                num_occ += 1
                for key, grid in EVAL_RANGES.items():
                    s = slice(grid[0], grid[1])
                    iou_metrics[key](
                        result[0]['occ']['seg_out'][..., s, s].contiguous(),
                        result[0]['occ']['seg_gt'][..., s, s].contiguous())
                    pan_metrics[key](
                        result[0]['occ']['ins_seg_out'][..., s, s].contiguous().detach(),
                        result[0]['occ']['ins_seg_gt'][..., s, s].contiguous())

        # ── drop heavy tensors before CPU collection ──────────────────────
        if os.environ.get('ENABLE_PLOT_MODE') is None:
            result[0].pop('occ', None)
            result[0].pop('planning', None)

        bbox_results.extend(result)
        for _ in range(len(result)):
            prog_bar.update()

    # ── assemble return dict ─────────────────────────────────────────────
    ret = {'bbox_results': bbox_results}

    if eval_occ:
        occ_out = {}
        for key, grid in EVAL_RANGES.items():
            pan_scores = pan_metrics[key].compute()
            for pk, pv in pan_scores.items():
                occ_out[pk] = occ_out.get(pk, []) + [100 * pv[1].item()]
            pan_metrics[key].reset()
            iou_scores = iou_metrics[key].compute()
            occ_out['iou'] = occ_out.get('iou', []) + [
                100 * iou_scores[1].item()]
            iou_metrics[key].reset()
        occ_out['num_occ']   = num_occ
        occ_out['ratio_occ'] = num_occ / len(dataset)
        ret['occ_results_computed'] = occ_out

    if eval_planning:
        ret['planning_results_computed'] = plan_metrics.compute()
        plan_metrics.reset()

    return ret


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='UniV2X TRT end-to-end evaluation (VAD-TRT style, single GPU)')
    p.add_argument('config',     help='Test config file path')
    p.add_argument('checkpoint', help='Checkpoint file')

    # TRT engine paths (Hook A)
    p.add_argument('--bev-engine-ego', default=None,
                   help='BEV encoder TRT engine for ego model')
    p.add_argument('--bev-engine-inf', default=None,
                   help='BEV encoder TRT engine for infra model')
    p.add_argument('--plugin', default='plugins/build/libuniv2x_plugins.so',
                   help='Custom TRT plugin shared library (.so)')

    # Output / evaluation
    p.add_argument('--out', default='output/trt_results.pkl',
                   help='Output result pkl file')
    p.add_argument('--eval', type=str, nargs='+',
                   help='Evaluation metrics, e.g., bbox')
    p.add_argument('--eval-options', nargs='+', action=DictAction)
    p.add_argument('--format-only', action='store_true')

    # Standard options
    p.add_argument('--fuse-conv-bn', action='store_true')
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--local_rank', type=int, default=0)

    args = p.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only, \
        'Specify at least one of --out / --eval / --format-only'
    if args.eval and args.format_only:
        raise ValueError('--eval and --format-only are mutually exclusive')
    if args.out and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('--out must end with .pkl')

    # ── Config ───────────────────────────────────────────────────────────
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # Register mmdet3d plugin modules (same as test.py)
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        plugin_dir = getattr(cfg, 'plugin_dir', '')
        if plugin_dir:
            parts = os.path.dirname(plugin_dir).split('/')
            importlib.import_module('.'.join(p for p in parts if p))

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model_ego_agent.pretrained = None

    # ── Dataset / DataLoader ─────────────────────────────────────────────
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # ── Build model (mirrors test.py exactly) ────────────────────────────
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
    model_ego = build_model(cfg.model_ego_agent,
                             test_cfg=cfg.get('test_cfg'))
    load_from = cfg.model_ego_agent.load_from
    if load_from:
        load_checkpoint(model_ego, load_from, map_location='cpu',
                        revise_keys=[(r'^model_ego_agent\.', '')])

    model_multi = MultiAgent(model_ego, model_other_agents)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model_multi)

    checkpoint = load_checkpoint(model_multi, args.checkpoint,
                                  map_location='cpu')
    if args.fuse_conv_bn:
        model_multi = fuse_conv_bn(model_multi)

    if 'CLASSES' in checkpoint.get('meta', {}):
        model_ego.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model_ego.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model_ego.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        model_ego.PALETTE = dataset.PALETTE

    # ── Wrap with MMDataParallel (single GPU) ────────────────────────────
    model_multi = MMDataParallel(model_multi.cuda(), device_ids=[0])

    # ── Inject TRT hooks (before eval loop, after model is on GPU) ───────
    inner = model_multi.module   # MultiAgent

    if args.bev_engine_ego:
        print(f'\nLoading ego BEV TRT engine: {args.bev_engine_ego}')
        ego_trt = TrtEngine(args.bev_engine_ego, args.plugin)
        attach_bev_hook(inner.model_ego_agent, ego_trt, 'ego')
    else:
        print('[Hook-A/ego] No BEV engine specified — running full PyTorch')

    if args.bev_engine_inf:
        for inf_name, inf_model in inner.model_other_agents.items():
            print(f'\nLoading infra BEV TRT engine: {args.bev_engine_inf}')
            inf_trt = TrtEngine(args.bev_engine_inf, args.plugin)
            attach_bev_hook(inf_model, inf_trt, inf_name)

    # ── Run evaluation loop ───────────────────────────────────────────────
    print('\nRunning TRT evaluation loop...')
    outputs = single_gpu_test_trt(model_multi, data_loader)

    # ── Save & evaluate ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f'\nWriting results → {args.out}')
    mmcv.dump(outputs, args.out)

    import os.path as osp
    ts = time.ctime().replace(' ', '_').replace(':', '_')
    json_prefix = osp.join('test',
                            args.config.split('/')[-1].split('.')[0], ts)
    kwargs = {} if args.eval_options is None else args.eval_options
    kwargs['jsonfile_prefix'] = json_prefix

    if args.format_only:
        dataset.format_results(outputs['bbox_results'], **kwargs)

    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect',
                    'save_best', 'rule']:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print('\n' + '=' * 60)
        print('TRT evaluation results')
        print('=' * 60)
        print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
