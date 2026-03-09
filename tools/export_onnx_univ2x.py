"""ONNX export script for UniV2X BEVFormer backbone.

Phase 1 only exports the BEV encoder (backbone + neck + BEVFormerEncoder).
Supports both ego and infra models.

Usage
-----
Step A — random weights, 50×50 BEV (graph-structure check):
    python tools/export_onnx_univ2x.py \
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p.py \
        --model ego \
        --random-weights \
        --backbone-only \
        --bev-size 50 \
        --out onnx/univ2x_ego_bev_backbone_50_rand.onnx

Step B — real checkpoint, 200×200 BEV (accuracy check):
    python tools/export_onnx_univ2x.py \
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p.py \
        ckpts/univ2x_coop_e2e_stg1.pth \
        --model ego \
        --backbone-only \
        --bev-size 200 \
        --out onnx/univ2x_ego_bev_backbone_200.onnx
"""

import argparse
import contextlib
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# ONNX compatibility helpers
# ---------------------------------------------------------------------------

def _register_onnx_symbolics():
    """Register custom ONNX symbolics needed for downstream heads.

    PyTorch 2.0 + TrainingMode.TRAINING emits ``aten::feature_dropout`` for
    every nn.Dropout module (even with p=0).  We register a no-op passthrough
    symbolic here.

    Notes
    -----
    * ``aten::unflatten`` (from MHA's QKV reshape) is handled at the Tensor
      level inside :func:`onnx_compatible_attention` by monkey-patching
      ``torch.Tensor.unflatten`` → ``reshape``.  Registering an ONNX symbolic
      for it causes heap corruption when sizes are dynamic (computed via
      SymInt / Concat), so the Tensor-level patch is the only safe approach.
    * ``_transformer_encoder_layer_fwd`` / ``_native_multi_head_attention`` /
      ``scaled_dot_product_attention`` are eliminated by TrainingMode.TRAINING
      and the SDPA patch inside :func:`onnx_compatible_attention`.
    """
    # aten::feature_dropout(Tensor input, float p, bool train) -> Tensor
    # Appears when nn.Dropout is in training mode.  Since all dropout probs are
    # zeroed before export (inside onnx_compatible_attention), this is always
    # a passthrough no-op.
    def _feature_dropout_symbolic(g, input, p, train):
        return input

    # Register at opset 9+ (opset 1 is NOT picked up by the exporter for this op).
    torch.onnx.register_custom_op_symbolic('aten::feature_dropout',
                                            _feature_dropout_symbolic, 9)


@contextlib.contextmanager
def onnx_compatible_attention(wrapper_model):
    """Context manager: patches PyTorch ops for ONNX-compatible tracing.

    Three patches are applied together:

    1. ``F.scaled_dot_product_attention`` → pure matmul+softmax so that
       cross-attention traces as standard ONNX Matmul/Softmax nodes.

    2. ``torch.Tensor.unflatten`` → ``Tensor.reshape`` with concrete Python
       int sizes.  PyTorch 2.0's MHA uses ``proj.unflatten(-1, (3, E))``
       where E is a traced SymInt; any ONNX symbolic for this op causes C++
       heap corruption, so we handle it at the Tensor level instead.

    3. All ``nn.Dropout.p`` are zeroed so that ``feature_dropout`` ops emitted
       by ``TrainingMode.TRAINING`` are trivially passthrough (handled by the
       registered symbolic in ``_register_onnx_symbolics``).

    4. All ``nn.BatchNorm*`` instances have their ``forward`` overridden to
       call ``F.batch_norm(..., training=False)``.  ``TrainingMode.TRAINING``
       calls ``model.train()`` which makes BN update running stats in-place,
       so the ONNX would embed post-trace running stats that differ from the
       checkpoint.  Forcing eval-mode computation prevents this drift and makes
       the TRT engine exactly match the PyTorch-eval results.

    The C++ fast paths (``_transformer_encoder_layer_fwd``,
    ``_native_multi_head_attention``) are eliminated by passing
    ``training=torch.onnx.TrainingMode.TRAINING`` to ``torch.onnx.export``
    and do not require additional patching here.
    """
    # ── Patch 1: SDPA ────────────────────────────────────────────────────────
    orig_sdpa = F.scaled_dot_product_attention

    def _onnx_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        scale_f = q.size(-1) ** -0.5 if scale is None else scale
        weights = torch.softmax(q @ k.transpose(-2, -1) * scale_f, dim=-1)
        return weights @ v

    F.scaled_dot_product_attention = _onnx_sdpa

    # ── Patch 2: unflatten → reshape ─────────────────────────────────────────
    orig_unflatten = torch.Tensor.unflatten

    def _reshape_unflatten(self, dim, sizes):
        shape = list(self.shape)
        ndim = len(shape)
        if dim < 0:
            dim = ndim + dim
        new_sizes = [int(s) for s in sizes]
        return self.reshape(shape[:dim] + new_sizes + shape[dim + 1:])

    torch.Tensor.unflatten = _reshape_unflatten

    # ── Patch 3: zero all dropout probs ──────────────────────────────────────
    _DROPOUT_TYPES = (nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                      nn.AlphaDropout, nn.FeatureAlphaDropout)
    saved_dp = {}       # id(module) -> (module, original_p)
    for m in wrapper_model.modules():
        if isinstance(m, _DROPOUT_TYPES) and id(m) not in saved_dp:
            saved_dp[id(m)] = (m, m.p)
            m.p = 0.0

    # ── Patch 4: freeze BN to eval-mode computation ───────────────────────────
    # With TrainingMode.TRAINING, PyTorch calls model.train() which makes BN
    # compute batch statistics (instead of running stats) AND update running_mean/
    # running_var in-place.  The ONNX then embeds post-trace running stats that
    # differ from checkpoint values, so TRT and PyTorch-eval diverge.
    # Fix: patch each BN instance's forward to always call F.batch_norm with
    # training=False, so the trace embeds eval-mode BN (training_mode=0) and
    # running stats are never modified.
    _BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                 nn.SyncBatchNorm)
    patched_bn_ids = set()
    for m in wrapper_model.modules():
        if isinstance(m, _BN_TYPES) and id(m) not in patched_bn_ids:
            patched_bn_ids.add(id(m))
            bn_ref = m

            def _make_eval_bn(bn):
                def _eval_bn_fwd(input):
                    return F.batch_norm(
                        input, bn.running_mean, bn.running_var,
                        bn.weight, bn.bias,
                        False,   # training=False → eval-mode, no stat update
                        0,       # momentum (ignored when training=False)
                        bn.eps,
                    )
                return _eval_bn_fwd

            m.forward = _make_eval_bn(m)

    try:
        yield
    finally:
        F.scaled_dot_product_attention = orig_sdpa
        torch.Tensor.unflatten = orig_unflatten
        for _, (m, orig_p) in saved_dp.items():
            m.p = orig_p
        # Remove instance-level forward overrides to restore class defaults
        for m in wrapper_model.modules():
            if isinstance(m, _BN_TYPES) and 'forward' in m.__dict__:
                del m.forward

# Ensure determinism
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['PYTHONHASHSEED'] = '42'


# ---------------------------------------------------------------------------
# Wrapper module
# ---------------------------------------------------------------------------

class HeadsWrapper(nn.Module):
    """Export wrapper: bev_embed + track state tensors → detection outputs.

    Stage C of the three-stage TRT inference pipeline.  Takes the BEV feature
    map produced by the Stage-B engine and the per-frame track state tensors,
    runs the detection decoder (BEVFormerTrackHeadTRT.get_detections_trt),
    and returns all decoder outputs as flat tensors.

    All inputs/outputs are plain tensors so torch.onnx.export can trace them.

    Inputs:
        bev_embed        (Tensor): (bev_h*bev_w, bs, C)
        track_query      (Tensor): (num_query, C*2)  pos+feat concat
        track_ref_pts    (Tensor): (num_query, 3)    inv-sigmoid ref points
        l2g_r1           (Tensor): (3, 3)  prev-frame local->global rotation
        l2g_t1           (Tensor): (3,)   prev-frame local->global translation
        l2g_r2           (Tensor): (3, 3)  cur-frame  local->global rotation
        l2g_t2           (Tensor): (3,)   cur-frame  local->global translation
        time_delta       (Tensor): scalar  frame time gap (seconds)

    Outputs:
        all_cls_scores   (Tensor): (num_dec_layers, bs, num_query, num_classes)
        all_bbox_preds   (Tensor): (num_dec_layers, bs, num_query, code_size)
        all_past_trajs   (Tensor): (num_dec_layers, bs, num_query, steps, 2)
        last_ref_pts     (Tensor): (bs, num_query, 3)  next-frame ref points
        query_feats      (Tensor): (num_dec_layers, bs, num_query, C)
        new_ref_pts      (Tensor): (num_query, 3)  velocity-updated ref pts
    """

    def __init__(self, detector):
        super().__init__()
        self.detector = detector  # UniV2XTrack (or UniV2X) instance

    def forward(self, bev_embed, track_query, track_ref_pts,
                l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta):
        head = self.detector.pts_bbox_head

        # Detection decoder (Stage C core)
        all_cls, all_bbox, all_traj, last_ref_pts, query_feats = \
            head.get_detections_trt(bev_embed, track_query, track_ref_pts)

        # Velocity-based reference-point update for the next frame
        velo = all_bbox[-1, 0, :, -2:]   # (num_query, 2)
        new_ref_pts = self.detector.velo_update(
            last_ref_pts[0],
            velo,
            l2g_r1, l2g_t1,
            l2g_r2, l2g_t2,
            time_delta=time_delta,
        )

        return all_cls, all_bbox, all_traj, last_ref_pts, query_feats, new_ref_pts


class DownstreamHeadsWrapper(nn.Module):
    """Export wrapper: Stages D+E+F — Motion → Occ → Planning.

    Takes outputs of Stage C (detection decoder) together with BEV features
    and runs the downstream perception heads in sequence:
      D — MotionHeadTRT.forward_trt  (trajectory prediction)
      E — OccHeadTRT.forward_trt     (occupancy forecast)
      F — PlanningHeadSingleModeTRT.forward_trt  (ego planning; ego-only)

    Inputs:
        bev_embed      (H*W, 1, C)            from Stage B
        query_feats    (num_dec, 1, A, C)      detection decoder hidden states
        all_bbox_preds (num_dec, 1, A, 10)     normalized bbox predictions
        all_cls_scores (num_dec, 1, A, num_cls) class scores
        lane_query     (1, M, C)               map/lane queries (zeros if absent)
        lane_query_pos (1, M, C)               map/lane query positions
        command        scalar long (0/1/2)      navigation command (ego-only)

    Outputs (ego):
        traj_scores    (num_layers, 1, A, num_modes)
        traj_preds     (num_layers, 1, A, num_modes, predict_steps, 5)
        occ_logits     (1, A, T, H_occ, W_occ)
        sdc_traj       (1, planning_steps, 2)

    Outputs (infra, no planning head):
        traj_scores, traj_preds, occ_logits    (same shapes as above)
    """

    def __init__(self, model, pc_range):
        super().__init__()
        self.det_head    = model.pts_bbox_head
        self.motion_head = model.motion_head
        self.occ_head    = model.occ_head
        self.planning_head = getattr(model, 'planning_head', None)
        self.pc_range = pc_range

    def forward(self, bev_embed, query_feats, all_bbox_preds, all_cls_scores,
                lane_query, lane_query_pos, command):
        from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox_trt

        # ── Decode last-decoder bbox → gravity_center + yaw ─────────────────
        last_bbox = all_bbox_preds[-1, 0]    # (A, 10) normalised
        last_cls  = all_cls_scores[-1, 0]    # (A, num_cls)

        denorm = denormalize_bbox_trt(last_bbox, self.pc_range)  # (A, 9)
        gravity_center = denorm[:, :3]        # (A, 3)
        yaw            = denorm[:, 6:7]       # (A, 1)

        track_scores = last_cls.softmax(-1).max(-1)[0]     # (A,)
        track_labels = last_cls.argmax(-1).long()          # (A,)

        # query_feats: (num_dec, 1, A, C) → (1, num_dec, A, C)
        track_query_in = query_feats.permute(1, 0, 2, 3)

        # ── Stage D: Motion head ─────────────────────────────────────────────
        (traj_scores, traj_preds, _valid,
         inter_states, track_query, track_query_pos) = self.motion_head.forward_trt(
            bev_embed, track_query_in, lane_query, lane_query_pos,
            track_scores, track_labels, gravity_center, yaw,
        )
        # inter_states: (num_layers, 1, A, P, C)
        # track_query : (1, A, C)

        # ── Stage E: Occ head ────────────────────────────────────────────────
        ins_query  = self.occ_head.merge_queries_trt(
            track_query, track_query_pos, inter_states)
        occ_logits = self.occ_head.forward_trt(bev_embed, ins_query)

        if self.planning_head is None:
            return traj_scores, traj_preds, occ_logits

        # ── Stage F: Planning head (ego only) ────────────────────────────────
        # bev_pos: compute from detection head's positional encoding
        device = bev_embed.device
        dtype  = bev_embed.dtype
        bev_mask = torch.zeros(
            (1, self.det_head.bev_h, self.det_head.bev_w), device=device).to(dtype)
        bev_pos = self.det_head.positional_encoding(bev_mask).to(dtype)  # (1, C, H, W)

        # SDC is last agent (index -1)
        sdc_traj_query  = inter_states[:, :, -1]  # (num_layers, 1, P, C)
        sdc_track_query = track_query[:, -1]       # (1, C)

        # occ_mask unused in forward_trt (use_col_optim=False)
        occ_mask_dummy = torch.zeros(1, 1, self.det_head.bev_h, self.det_head.bev_w,
                                     device=device, dtype=dtype)
        sdc_traj = self.planning_head.forward_trt(
            bev_embed, occ_mask_dummy, bev_pos,
            sdc_traj_query, sdc_track_query, command,
        )

        return traj_scores, traj_preds, occ_logits, sdc_traj


class BEVEncoderWrapper(nn.Module):
    """Export wrapper: pre-extracted multi-scale features → BEV embed.

    The image backbone (ResNet+FPN) uses DCNv2 which has no ONNX symbolic,
    so it is exported separately.  This wrapper takes the backbone outputs
    (4 FPN-level tensors) and runs only the BEV encoder (transformer).

    Inputs (all tensors, no Python dicts):
        feat0       (Tensor): (bs, num_cam, C, H0, W0)  — FPN level 0 (largest)
        feat1       (Tensor): (bs, num_cam, C, H1, W1)
        feat2       (Tensor): (bs, num_cam, C, H2, W2)
        feat3       (Tensor): (bs, num_cam, C, H3, W3)
        can_bus     (Tensor): (18,)
        lidar2img   (Tensor): (bs, num_cam, 4, 4)
        image_shape (Tensor): (2,) = [img_h, img_w]
        prev_bev    (Tensor): (num_query, bs, embed_dims)
        use_prev_bev(Tensor): scalar float 0.0 or 1.0

    Outputs:
        bev_embed (Tensor): (num_query, bs, embed_dims)
    """

    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, feat0, feat1, feat2, feat3,
                can_bus, lidar2img, image_shape, prev_bev, use_prev_bev):
        img_feats = (feat0, feat1, feat2, feat3)

        # Get BEV features via TRT-compatible path
        bev_embed, _ = self.detector.pts_bbox_head.get_bev_features_trt(
            mlvl_feats=img_feats,
            can_bus=can_bus,
            lidar2img=lidar2img,
            image_shape=image_shape,
            prev_bev=prev_bev,
            use_prev_bev=use_prev_bev,
        )
        return bev_embed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_plugin(cfg):
    """Import the mmdet3d plugin defined in the config."""
    if not getattr(cfg, 'plugin', False):
        return
    import importlib
    # Ensure cwd is on sys.path so 'projects.*' imports work
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    plugin_dir = getattr(cfg, 'plugin_dir', '')
    if plugin_dir:
        parts = os.path.dirname(plugin_dir).split('/')
        module_path = '.'.join(p for p in parts if p)
        importlib.import_module(module_path)


def build_model_from_cfg(cfg, model_key, ckpt_path=None, random_weights=False):
    """Build a single sub-model (ego or infra) from cfg and optionally load ckpt."""
    model_cfg = getattr(cfg, model_key)
    model_cfg.train_cfg = None
    model = build_model(model_cfg, test_cfg=cfg.get('test_cfg'))

    if not random_weights and ckpt_path is not None:
        # The cooperative checkpoint stores sub-model weights under prefixed keys
        # (e.g. "model_ego_agent.*" or "model_other_agent_inf.*").
        # Strip the prefix so they match the standalone sub-model's key names.
        import torch as _torch
        raw = _torch.load(ckpt_path, map_location='cpu')
        sd = raw.get('state_dict', raw)
        # Detect prefix: find the common leading component(s) before the first "."
        sample_key = next(iter(sd))
        prefix = sample_key.split('.')[0] + '.'  # e.g. "model_ego_agent."
        stripped = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if stripped:
            missing, unexpected = model.load_state_dict(stripped, strict=False)
            if missing:
                print(f'  Missing keys ({len(missing)}): {missing[:3]} ...')
            print(f'Loaded checkpoint (prefix="{prefix}" stripped): {ckpt_path}')
        else:
            load_checkpoint(model, ckpt_path, map_location='cpu')
            print(f'Loaded checkpoint: {ckpt_path}')
    else:
        print('Using random weights (no checkpoint loaded)')

    model.cuda().eval()
    return model


def make_dummy_inputs(model, bev_size, num_cam=6, img_h=256, img_w=416):
    """Create dummy tensors that match the BEV encoder's expected inputs.

    The backbone (DCNv2-based) is not exported here, so we produce dummy
    pre-extracted feature tensors at the 4 FPN scale levels.
    """
    bev_h = bev_w = bev_size
    num_query = bev_h * bev_w
    embed_dims = model.pts_bbox_head.embed_dims

    # num_cam argument takes precedence; use model's num_cams only as fallback
    if num_cam and num_cam > 0:
        nc = num_cam
    else:
        try:
            nc = model.pts_bbox_head.transformer.encoder.num_cams
        except AttributeError:
            nc = 6

    # Run backbone with a real image to get feature shapes
    with torch.no_grad():
        dummy_img = torch.randn(1, nc, 3, img_h, img_w, device='cuda')
        img_feats = model.extract_img_feat(img=dummy_img)
    feat_shapes = [f.shape for f in img_feats]
    print(f'Backbone feature shapes: {feat_shapes}')

    # Dummy pre-extracted features (detached, no grad)
    feats = tuple(torch.randn(*s, device='cuda') for s in feat_shapes)

    can_bus = torch.randn(18, device='cuda')
    lidar2img = torch.eye(4, device='cuda').unsqueeze(0).unsqueeze(0).repeat(1, nc, 1, 1)
    image_shape = torch.tensor([img_h, img_w], dtype=torch.float32, device='cuda')
    prev_bev = torch.zeros(num_query, 1, embed_dims, device='cuda')
    use_prev_bev = torch.tensor(0.0, device='cuda')

    return feats + (can_bus, lidar2img, image_shape, prev_bev, use_prev_bev)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Export UniV2X BEV backbone to ONNX')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', nargs='?', default=None,
                        help='Checkpoint file (optional if --random-weights)')
    parser.add_argument('--model', choices=['ego', 'infra'], default='ego',
                        help='Which model to export (default: ego)')
    parser.add_argument('--random-weights', action='store_true',
                        help='Use random initialization (skip checkpoint)')
    parser.add_argument('--backbone-only', action='store_true',
                        help='Phase 1: export only BEV backbone + encoder')
    parser.add_argument('--heads-only', action='store_true',
                        help='Phase 2: export detection heads (Stage C) only')
    parser.add_argument('--downstream', action='store_true',
                        help='Phase 2: export downstream heads (Stages D+E+F: motion+occ+planning)')
    parser.add_argument('--bev-size', type=int, default=50,
                        help='BEV grid size H=W (default: 50 for Step A)')
    parser.add_argument('--img-h', type=int, default=256, help='Input image height')
    parser.add_argument('--img-w', type=int, default=416, help='Input image width')
    parser.add_argument('--num-cam', type=int, default=0,
                        help='Number of cameras (0 = auto-detect from model config)')
    parser.add_argument('--out', default='onnx/univ2x_bev.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=16,
                        help='ONNX opset version (default: 16)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='Override config keys (key=value)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # Load plugin (registers mmdet3d modules)
    load_plugin(cfg)

    # Determine model config key
    model_key = 'model_ego_agent' if args.model == 'ego' else 'model_other_agent_inf'
    if not hasattr(cfg, model_key):
        # Fall back to single-model config
        model_key = 'model'

    # Override bev_h / bev_w if different from config default
    bev_size = args.bev_size

    # Build model
    model = build_model_from_cfg(
        cfg, model_key,
        ckpt_path=args.checkpoint,
        random_weights=args.random_weights,
    )

    # Patch bev_h / bev_w in the head so the model uses the requested size
    head = model.pts_bbox_head
    orig_bev_h, orig_bev_w = head.bev_h, head.bev_w
    head.bev_h = bev_size
    head.bev_w = bev_size
    model.bev_h = bev_size
    model.bev_w = bev_size

    # Rebuild bev_embedding if size changed
    if orig_bev_h != bev_size or orig_bev_w != bev_size:
        embed_dims = head.embed_dims
        head.bev_embedding = nn.Embedding(bev_size * bev_size, embed_dims).cuda()
        print(f'Rebuilt bev_embedding for {bev_size}×{bev_size}')

    # Register inverse symbolic for torch.linalg.inv / torch.inverse
    from projects.mmdet3d_plugin.univ2x.functions import register_inverse_symbolic
    register_inverse_symbolic()

    # Register additional ONNX symbolics needed for downstream heads
    _register_onnx_symbolics()

    flags = [args.backbone_only, args.heads_only, args.downstream]
    assert sum(flags) == 1, \
        'Specify exactly one of --backbone-only / --heads-only / --downstream'

    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.backbone_only:
        _export_bev_encoder(model, bev_size, args)
    elif args.heads_only:
        _export_heads(model, bev_size, args)
    else:
        _export_downstream_heads(model, cfg, model_key, bev_size, args)


def _export_bev_encoder(model, bev_size, args):
    """Phase 1: export BEV encoder (Stage B) — backbone+encoder only."""
    wrapper = BEVEncoderWrapper(model).cuda().eval()

    # Dummy inputs: (feat0..featN, can_bus, lidar2img, image_shape, prev_bev, use_prev_bev)
    dummy = make_dummy_inputs(model, bev_size, num_cam=args.num_cam,
                              img_h=args.img_h, img_w=args.img_w)
    num_feat_levels = len(dummy) - 5
    feats = dummy[:num_feat_levels]
    can_bus, lidar2img, image_shape, prev_bev, use_prev_bev = dummy[num_feat_levels:]

    with torch.no_grad():
        out = wrapper(*feats, can_bus, lidar2img, image_shape, prev_bev, use_prev_bev)
    print(f'Forward OK, output shape: {out.shape}')

    feat_names = [f'feat{i}' for i in range(num_feat_levels)]
    input_names = feat_names + ['can_bus', 'lidar2img', 'image_shape', 'prev_bev', 'use_prev_bev']
    output_names = ['bev_embed']

    print(f'Exporting to {args.out} (opset={args.opset}) ...')
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (*feats, can_bus, lidar2img, image_shape, prev_bev, use_prev_bev),
            args.out,
            opset_version=args.opset,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
            verbose=False,
        )
    print(f'ONNX saved to {args.out}')
    _patch_and_verify_onnx(args.out)


def _export_heads(model, bev_size, args):
    """Phase 2: export detection heads (Stage C) — decoder + velo_update."""
    wrapper = HeadsWrapper(model).cuda().eval()

    embed_dims = model.pts_bbox_head.embed_dims
    num_query = model.num_query + 1  # +1 for ego query
    bev_h = bev_w = bev_size
    num_bev = bev_h * bev_w

    # Dummy inputs
    bev_embed     = torch.randn(num_bev, 1, embed_dims, device='cuda')
    track_query   = torch.randn(num_query, embed_dims * 2, device='cuda')
    track_ref_pts = torch.randn(num_query, 3, device='cuda')
    l2g_r1 = torch.eye(3, device='cuda')
    l2g_t1 = torch.zeros(3, device='cuda')
    l2g_r2 = torch.eye(3, device='cuda')
    l2g_t2 = torch.zeros(3, device='cuda')
    time_delta = torch.tensor(0.5, device='cuda')

    dummy = (bev_embed, track_query, track_ref_pts,
             l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta)

    with torch.no_grad():
        outs = wrapper(*dummy)
    print(f'Forward OK — output shapes: {[o.shape for o in outs]}')

    input_names = [
        'bev_embed', 'track_query', 'track_ref_pts',
        'l2g_r1', 'l2g_t1', 'l2g_r2', 'l2g_t2', 'time_delta',
    ]
    output_names = [
        'all_cls_scores', 'all_bbox_preds', 'all_past_trajs',
        'last_ref_pts', 'query_feats', 'new_ref_pts',
    ]

    print(f'Exporting to {args.out} (opset={args.opset}) ...')
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            args.out,
            opset_version=args.opset,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
            verbose=False,
        )
    print(f'ONNX saved to {args.out}')
    _patch_and_verify_onnx(args.out)


def _export_downstream_heads(model, cfg, model_key, bev_size, args):
    """Phase 2: export downstream heads (Stages D+E+F) — motion+occ+planning."""
    # Retrieve pc_range from the motion head config so the wrapper uses the
    # correct coordinate frame (ego vs infra).
    model_cfg = getattr(cfg, model_key)
    pc_range = model_cfg.motion_head.get('pc_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])

    embed_dims = model.pts_bbox_head.embed_dims
    # Downstream heads use the OCC head's configured BEV size (always 200×200
    # in the standard config), NOT the --bev-size arg (which controls only the
    # BEV encoder Stage-B export).
    try:
        occ_bev_h, occ_bev_w = model.occ_head.bev_size
    except Exception:
        occ_bev_h = occ_bev_w = 200
    bev_h, bev_w = occ_bev_h, occ_bev_w
    num_bev    = bev_h * bev_w

    # Restore pts_bbox_head bev_h/bev_w and positional_encoding to match the
    # actual BEV size consumed by downstream heads (may differ from --bev-size).
    det_head = model.pts_bbox_head
    det_head.bev_h = bev_h
    det_head.bev_w = bev_w
    if hasattr(det_head, 'positional_encoding'):
        from mmdet.models.utils.positional_encoding import LearnedPositionalEncoding
        num_feats = det_head.positional_encoding.num_feats
        det_head.positional_encoding = LearnedPositionalEncoding(
            num_feats, bev_h, bev_w).cuda()

    wrapper = DownstreamHeadsWrapper(model, pc_range).cuda().eval()
    num_query  = model.num_query + 1   # +1 for SDC/ego query
    num_dec    = 6                     # detection decoder layers
    num_cls    = 10
    M          = 300                   # lane query count (dummy)

    # Dummy inputs
    bev_embed      = torch.randn(num_bev, 1, embed_dims, device='cuda')
    query_feats    = torch.randn(num_dec, 1, num_query, embed_dims, device='cuda')
    all_bbox_preds = torch.randn(num_dec, 1, num_query, 10, device='cuda')
    all_cls_scores = torch.randn(num_dec, 1, num_query, num_cls, device='cuda')
    lane_query     = torch.zeros(1, M, embed_dims, device='cuda')
    lane_query_pos = torch.zeros(1, M, embed_dims, device='cuda')
    command        = torch.tensor(0, dtype=torch.long, device='cuda')

    dummy = (bev_embed, query_feats, all_bbox_preds, all_cls_scores,
             lane_query, lane_query_pos, command)

    with torch.no_grad():
        outs = wrapper(*dummy)
    has_planning = len(outs) == 4
    print(f'Forward OK — {len(outs)} outputs, shapes: {[o.shape for o in outs]}')

    input_names = [
        'bev_embed', 'query_feats', 'all_bbox_preds', 'all_cls_scores',
        'lane_query', 'lane_query_pos', 'command',
    ]
    output_names = ['traj_scores', 'traj_preds', 'occ_logits']
    if has_planning:
        output_names.append('sdc_traj')

    print(f'Exporting to {args.out} (opset={args.opset}) ...')
    # TrainingMode.TRAINING is required to prevent PyTorch 2.0 C++ fast paths:
    #   _transformer_encoder_layer_fwd and _native_multi_head_attention
    # Both are triggered by TrainingMode.EVAL (the default) when batch_first=True.
    # onnx_compatible_attention patches SDPA and zeros dropout (needed in train mode).
    with onnx_compatible_attention(wrapper), torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            args.out,
            opset_version=args.opset,
            training=torch.onnx.TrainingMode.TRAINING,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
            verbose=False,
        )
    print(f'ONNX saved to {args.out}')
    _patch_and_verify_onnx(args.out)


def _patch_and_verify_onnx(onnx_path):
    """Post-process the exported ONNX for TRT compatibility.

    Two patches are applied:

    1. MSDAPlugin INT64→INT32: ``spatial_shapes`` and ``level_start_index``
       constants must be INT32 for the C++ TRT plugin.

    2. Dropout training_mode=1→0: exporting with ``TrainingMode.TRAINING``
       emits ONNX ``Dropout`` nodes with ``training_mode=1``, which TRT
       refuses to parse.  Since all dropout probs are zeroed before export
       the behaviour is identical in eval mode (ratio=0 passthrough).
    """
    try:
        import onnx
        import numpy as np
        from onnx import numpy_helper

        m = onnx.load(onnx_path)

        # ── Patch 1: MSDAPlugin INT64 → INT32 ────────────────────────────────
        producer_map = {}
        for node in m.graph.node:
            for out in node.output:
                producer_map[out] = node

        patched_msda = 0
        for node in m.graph.node:
            if node.op_type != 'MSDAPlugin':
                continue
            for slot in [1, 2]:  # spatial_shapes, level_start_index
                inp_name = node.input[slot]
                prod = producer_map.get(inp_name)
                if prod is None or prod.op_type != 'Constant':
                    continue
                for attr in prod.attribute:
                    if attr.name != 'value':
                        continue
                    arr = numpy_helper.to_array(attr.t)
                    if arr.dtype != np.int64:
                        continue
                    arr32 = arr.astype(np.int32)
                    new_tensor = numpy_helper.from_array(arr32, name=attr.t.name)
                    attr.t.CopyFrom(new_tensor)
                    patched_msda += 1

        if patched_msda:
            print(f'Patched {patched_msda} INT64→INT32 MSDAPlugin constant(s), re-saved ONNX.')

        # ── Patch 2: Dropout training_mode → eval mode ───────────────────────
        # ONNX Dropout has optional inputs: [data, ratio, training_mode]
        # When training_mode is a constant 1 (True), TRT refuses to parse.
        # Since p=0 for all dropout layers, flip training_mode to 0.
        patched_dp = 0
        for node in m.graph.node:
            if node.op_type != 'Dropout' or len(node.input) < 3:
                continue
            tm_name = node.input[2]   # training_mode tensor
            prod = producer_map.get(tm_name)
            if prod is None or prod.op_type != 'Constant':
                continue
            for attr in prod.attribute:
                if attr.name != 'value':
                    continue
                arr = numpy_helper.to_array(attr.t)
                if arr.dtype == np.bool_ and arr.item():
                    new_arr = np.array(False)
                    new_tensor = numpy_helper.from_array(new_arr, name=attr.t.name)
                    attr.t.CopyFrom(new_tensor)
                    patched_dp += 1

        if patched_dp:
            print(f'Patched {patched_dp} Dropout training_mode True→False.')

        # ── Patch 3: BatchNormalization training_mode → eval mode ────────────
        # TrainingMode.TRAINING causes BN nodes to have training_mode=1 attribute.
        # TRT does not support training_mode=1 in BatchNormalization.
        # For inference (all models in eval), this is always safe to flip to 0.
        # Note: in training_mode=1 ONNX BN has 3 outputs; in eval mode only 1.
        # We remove extra outputs and set training_mode=0.
        patched_bn = 0
        for node in m.graph.node:
            if node.op_type != 'BatchNormalization':
                continue
            for attr in node.attribute:
                if attr.name == 'training_mode' and attr.i == 1:
                    attr.i = 0
                    # Remove extra outputs (mean/var) that appear in training mode
                    while len(node.output) > 1:
                        node.output.pop()
                    patched_bn += 1
                    break

        if patched_bn:
            print(f'Patched {patched_bn} BatchNormalization training_mode 1→0.')

        if patched_msda or patched_dp:
            onnx.save(m, onnx_path)

        # ── Verify ────────────────────────────────────────────────────────────
        onnx.checker.check_model(m)
        op_types = {n.op_type for n in m.graph.node}
        plugin_nodes = {t for t in op_types if 'Plugin' in t}
        aten_ops = {n.op_type for n in m.graph.node if n.domain == 'org.pytorch.aten'}
        print(f'ONNX check passed. Plugin nodes: {plugin_nodes}')
        if aten_ops:
            print(f'  WARNING: remaining ATen ops (TRT-incompatible): {aten_ops}')
        else:
            print(f'  No ATen ops remain — fully TRT-compatible.')
    except ImportError:
        print('onnx not installed, skipping verification')
    except Exception as e:
        print(f'ONNX check warning: {e}')


if __name__ == '__main__':
    main()
