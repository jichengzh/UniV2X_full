"""
export_onnx_adaround.py — Export BEV encoder ONNX with AdaRound fake-quantized weights.

Pipeline position:
  calibrate_univ2x.py --adaround  →  calibration/quant_encoder_adaround.pth
  THIS SCRIPT                      →  onnx/univ2x_ego_bev_encoder_adaround.onnx
  build_trt_int8_univ2x.py         →  trt_engines/univ2x_ego_bev_encoder_adaround_int8.trt

What "fake-quantized weights" means:
  For each quantizable Linear/Conv layer (value_proj, output_proj, FFN linears):
    W_fq = dequantize( quantize_adaround(W_original) )
  W_fq is FP32 but its values lie exactly on the INT8 grid (AdaRound rounding decisions
  are baked in). When TRT INT8 calibrates this ONNX, it will find W_fq already on the
  INT8 grid and preserve the AdaRound rounding decisions.

Usage:
    python tools/export_onnx_adaround.py \\
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \\
        ckpts/univ2x_coop_e2e_stg2.pth \\
        --model ego \\
        --bev-size 200 --num-cam 1 --img-h 1088 --img-w 1920 \\
        --adaround-ckpt calibration/quant_encoder_adaround.pth \\
        --out onnx/univ2x_ego_bev_encoder_adaround.onnx
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from mmcv import Config, DictAction


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Export BEV encoder ONNX with AdaRound fake-quantized weights')
    p.add_argument('config',       help='Test config file path')
    p.add_argument('checkpoint',   help='Model checkpoint (.pth, full cooperative ckpt)')
    p.add_argument('--model',      choices=['ego', 'infra'], default='ego',
                   help='Which agent model to export (default: ego)')
    p.add_argument('--adaround-ckpt', required=True,
                   help='AdaRound weights from calibrate_univ2x.py --adaround')
    p.add_argument('--bev-size',   type=int, default=200,
                   help='BEV grid size H=W (default: 200)')
    p.add_argument('--num-cam',    type=int, default=1,
                   help='Number of cameras (default: 1 for V2X-Seq-SPD)')
    p.add_argument('--img-h',      type=int, default=1088,
                   help='Input image height (default: 1088)')
    p.add_argument('--img-w',      type=int, default=1920,
                   help='Input image width (default: 1920)')
    p.add_argument('--opset',      type=int, default=16,
                   help='ONNX opset version (default: 16)')
    p.add_argument('--out',        default='onnx/univ2x_ego_bev_encoder_adaround.onnx',
                   help='Output ONNX file path')
    p.add_argument('--cfg-options', nargs='+', action=DictAction,
                   help='Override config keys (key=value)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# AdaRound weight application
# ---------------------------------------------------------------------------

def apply_adaround_weights(qmodel, ckpt_data: dict) -> int:
    """Bake AdaRound fake-quantized weights (W_fq) into QuantModule.org_weight.

    Steps:
      1. Re-calibrate weight scales (delta) via set_weight_quantize_params so delta
         has the same values as during AdaRound calibration (not saved in state_dict).
      2. Wrap each QuantModule.weight_quantizer with AdaRoundQuantizer (copies delta).
      3. Load AdaRound state dict: loads .alpha and other params (strict=False).
      4. Enable weight quantization with hard rounding (soft_targets=False).
      5. save_quantized_weight: module.weight.data = dequant(quant_adaround(W)).
      6. Copy weight.data (= W_fq) into org_weight (which forward() reads).
      7. Disable all quantizers → forward() = plain F.linear(input, W_fq).

    NOTE on soft_targets: AdaRoundQuantizer.soft_targets lives on weight_quantizer
    (not on QuantModule itself). arch_decision.md has a typo where it checks
    hasattr(m, 'soft_targets') — that would never match. We correctly check
    hasattr(m.weight_quantizer, 'soft_targets').

    Returns number of QuantModules updated.
    """
    from projects.mmdet3d_plugin.univ2x.quant.quant_layer import QuantModule
    from projects.mmdet3d_plugin.univ2x.quant.quant_params import (
        save_quantized_weight, set_weight_quantize_params)
    from projects.mmdet3d_plugin.univ2x.quant.adaptive_rounding import AdaRoundQuantizer

    # Step 1: Calibrate weight scales (delta/zero_point) from module weights.
    # weight_quantizer.delta is a plain float attr NOT saved in state_dict.
    # We must re-derive delta using the same scale_method as calibration (read from
    # ckpt_data) so that AdaRoundQuantizer copies the correct delta and W_fq values
    # land on the correct INT8 grid.
    qmodel.set_quant_state(weight_quant=False, act_quant=False)
    set_weight_quantize_params(qmodel)
    print('  Weight scales re-calibrated (to initialize delta before AdaRound wrap).')

    # Step 2: Wrap weight_quantizer → AdaRoundQuantizer for each QuantModule
    # so that state_dict keys (.alpha) match the saved checkpoint.
    for m in qmodel.modules():
        if isinstance(m, QuantModule) and not isinstance(m.weight_quantizer, AdaRoundQuantizer):
            m.weight_quantizer = AdaRoundQuantizer(
                uaq=m.weight_quantizer,      # copies calibrated delta/zero_point
                round_mode='learned_hard_sigmoid',
                weight_tensor=m.org_weight.data)

    # Step 3: Load the saved state dict (now alpha keys align).
    # Use strict=False to tolerate act_quantizer.delta being a nn.Parameter in the
    # checkpoint (upgraded by layer_reconstruction) but a plain attribute in this
    # fresh model — we disable all quantizers after baking W_fq anyway.
    missing, unexpected = qmodel.load_state_dict(ckpt_data['state_dict'], strict=False)
    if unexpected:
        print(f'  [load_state_dict] ignored {len(unexpected)} unexpected keys '
              f'(act_quantizer.delta etc.): {unexpected[:3]}...')
    if missing:
        print(f'  [load_state_dict] WARNING: {len(missing)} missing keys: {missing[:3]}...')
    adaround_flag = ckpt_data.get('adaround', 'unknown')
    print(f'  Loaded state dict  (adaround={adaround_flag},'
          f' n_bits_w={ckpt_data.get("n_bits_w", "?")}, n_bits_a={ckpt_data.get("n_bits_a", "?")})')

    # Step 2: enable hard-rounding weight quantization
    # soft_targets lives on AdaRoundQuantizer (= weight_quantizer), not on QuantModule
    qmodel.set_quant_state(weight_quant=True, act_quant=False)
    for m in qmodel.modules():
        if hasattr(m, 'weight_quantizer') and hasattr(m.weight_quantizer, 'soft_targets'):
            m.weight_quantizer.soft_targets = False  # hard rounding (AdaRound final state)

    # Step 3: bake W_fq = dequant(quant_adaround(W)) into module.weight.data
    save_quantized_weight(qmodel)

    # Step 4: copy W_fq → org_weight (QuantModule.forward reads org_weight, NOT weight.data)
    n_updated = 0
    for m in qmodel.modules():
        if isinstance(m, QuantModule):
            m.org_weight.copy_(m.weight.data)
            n_updated += 1

    # Step 5: disable all quantizers → forward = plain FP32 with W_fq
    qmodel.set_quant_state(weight_quant=False, act_quant=False)

    print(f'  Baked W_fq into org_weight for {n_updated} QuantModules.')
    return n_updated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    # Use __file__-based path so the script works from any working directory
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, tools_dir)
    sys.path.insert(0, os.path.dirname(tools_dir))  # project root

    # ── Imports (deferred to allow sys.path setup) ────────────────────────
    from export_onnx_univ2x import (
        load_plugin,
        build_model_from_cfg,
        _register_onnx_symbolics,
        _export_bev_encoder,
    )
    from projects.mmdet3d_plugin.univ2x.quant import (
        QuantModel,
        register_bevformer_specials,
    )
    from projects.mmdet3d_plugin.univ2x.functions import register_inverse_symbolic

    # ── Config ────────────────────────────────────────────────────────────
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    load_plugin(cfg)

    # ── Load AdaRound ckpt once (reused for metadata + state_dict) ────────
    print(f'Reading AdaRound checkpoint: {args.adaround_ckpt}')
    ckpt_data = torch.load(args.adaround_ckpt, map_location='cpu')
    weight_quant_params = ckpt_data.get(
        'weight_quant_params',
        dict(n_bits=8, channel_wise=True, scale_method='mse'))
    act_quant_params = ckpt_data.get(
        'act_quant_params',
        dict(n_bits=8, channel_wise=False, scale_method='entropy', leaf_param=True))
    print(f'  weight_quant_params: {weight_quant_params}')
    print(f'  act_quant_params:    {act_quant_params}')

    # ── Build agent model (same as export_onnx_univ2x.py) ─────────────────
    model_key = 'model_ego_agent' if args.model == 'ego' else 'model_other_agent_inf'
    if not hasattr(cfg, model_key):
        model_key = 'model'
    print(f'Building model (key={model_key}) from {args.checkpoint} ...')
    agent_model = build_model_from_cfg(cfg, model_key, ckpt_path=args.checkpoint)

    # Patch bev_h / bev_w to requested size
    head = agent_model.pts_bbox_head
    orig_bev_h, orig_bev_w = head.bev_h, head.bev_w
    head.bev_h = args.bev_size
    head.bev_w = args.bev_size
    agent_model.bev_h = args.bev_size
    agent_model.bev_w = args.bev_size
    if orig_bev_h != args.bev_size or orig_bev_w != args.bev_size:
        from torch.nn import Embedding
        head.bev_embedding = Embedding(
            args.bev_size * args.bev_size, head.embed_dims).cuda()
        print(f'  Rebuilt bev_embedding for {args.bev_size}×{args.bev_size}')

    # ── Wrap encoder in QuantModel (same params as calibration) ───────────
    encoder = agent_model.pts_bbox_head.transformer.encoder
    register_bevformer_specials()
    qmodel = QuantModel(encoder, weight_quant_params, act_quant_params, is_fusing=True)
    qmodel.cuda()
    print(f'  QuantModel built. {qmodel.get_memory_footprint()}')

    # ── Apply AdaRound: bake W_fq into QuantModule.org_weight ─────────────
    # Pass ckpt_data directly to avoid loading the file a second time
    print('Applying AdaRound fake-quantized weights ...')
    n_updated = apply_adaround_weights(qmodel, ckpt_data)
    if n_updated == 0:
        raise RuntimeError(
            'No QuantModules updated — check that the adaround_ckpt matches the model.')

    # (Cosine similarity validation is done post-build via validate_quant_bev.py)

    # ── Export ONNX ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    register_inverse_symbolic()
    _register_onnx_symbolics()

    agent_model.eval()
    print(f'\nExporting AdaRound ONNX → {args.out}  (bev_size={args.bev_size},'
          f' num_cam={args.num_cam}, img={args.img_h}×{args.img_w})')
    _export_bev_encoder(agent_model, args.bev_size, args)

    size_mb = os.path.getsize(args.out) / 1024**2
    print(f'\n✓ AdaRound ONNX saved: {args.out}  ({size_mb:.1f} MB)')
    print(f'\nNext step — build TRT INT8 engine:')
    print(f'  python tools/build_trt_int8_univ2x.py \\')
    print(f'      --onnx {args.out} \\')
    print(f'      --out trt_engines/univ2x_ego_bev_encoder_adaround_int8.trt \\')
    print(f'      --target bev_encoder \\')
    print(f'      --plugin plugins/build/libuniv2x_plugins.so \\')
    print(f'      --cali-data calibration/bev_encoder_calib_inputs.pkl')


if __name__ == '__main__':
    main()
