"""
build_qdq_onnx_adaround.py — Insert Q/DQ nodes into AdaRound ONNX for correct TRT INT8.

Root cause of double-quantization failure:
  existing ONNX: W_fq (INT8-grid FP32) → Transpose → MatMul
  TRT INT8 build: re-derives weight scale from W_fq values (scale_trt ≠ scale_ada)
  TRT then re-quantizes W_fq with scale_trt → second rounding destroys AdaRound decisions

Fix (Q/DQ insertion on weights):
  W_fq → QuantizeLinear(scale=delta_w, zp=zp_w) → DequantizeLinear(scale=delta_w, zp=zp_w)
       → Transpose → MatMul
  Since W_fq / delta_w = exact integers (AdaRound construction), QL produces exact INT8 values.
  TRT fuses Q/DQ + MatMul into INT8 kernel using delta_w as the weight scale.
  No re-derivation → no double quantization → AdaRound rounding decisions preserved.

For activations: leave them without Q/DQ nodes so TRT uses INT8 calibration (calibration data).
TRT hybrid mode: explicit weight scales (from Q/DQ) + implicit activation scales (calibrator).

Usage:
    python tools/build_qdq_onnx_adaround.py \\
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \\
        ckpts/univ2x_coop_e2e_stg2.pth \\
        --adaround-ckpt calibration/quant_encoder_adaround.pth \\
        --input-onnx onnx/univ2x_ego_bev_encoder_adaround.onnx \\
        --out onnx/univ2x_ego_bev_encoder_adaround_qdq.onnx \\
        --bev-size 200 --num-cam 1 --img-h 1088 --img-w 1920

    # Then build TRT engine (same as vanilla PTQ, needs calibration data for activation scales):
    python tools/build_trt_int8_univ2x.py \\
        --onnx onnx/univ2x_ego_bev_encoder_adaround_qdq.onnx \\
        --out trt_engines/univ2x_ego_bev_encoder_adaround_qdq_int8.trt \\
        --target bev_encoder \\
        --plugin plugins/build/libuniv2x_plugins.so \\
        --cali-data calibration/bev_encoder_calib_inputs.pkl
"""

import argparse
import os
import sys

import numpy as np
import torch
import onnx
from onnx import numpy_helper, TensorProto, helper as onnx_helper
from mmcv import Config, DictAction


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Insert Q/DQ nodes into AdaRound ONNX for correct TRT INT8 deployment')
    p.add_argument('config',            help='Test config file path')
    p.add_argument('checkpoint',        help='Model checkpoint (.pth, full cooperative ckpt)')
    p.add_argument('--model',           choices=['ego', 'infra'], default='ego')
    p.add_argument('--adaround-ckpt',   required=True,
                   help='AdaRound checkpoint from calibrate_univ2x.py --adaround')
    p.add_argument('--input-onnx',      default='onnx/univ2x_ego_bev_encoder_adaround.onnx',
                   help='AdaRound fake-quant ONNX (W_fq baked in)')
    p.add_argument('--out',             default='onnx/univ2x_ego_bev_encoder_adaround_qdq.onnx',
                   help='Output Q/DQ ONNX file path')
    p.add_argument('--bev-size',        type=int, default=200)
    p.add_argument('--num-cam',         type=int, default=1)
    p.add_argument('--img-h',           type=int, default=1088)
    p.add_argument('--img-w',           type=int, default=1920)
    p.add_argument('--match-tol',       type=float, default=1e-3,
                   help='Tolerance for weight value matching (default: 1e-3)')
    p.add_argument('--no-check',        action='store_true',
                   help='Skip onnx.checker.check_model (faster)')
    p.add_argument('--cfg-options',     nargs='+', action=DictAction)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Build QuantModel and collect per-layer AdaRound scales
# ---------------------------------------------------------------------------

def collect_adaround_scales(cfg, args, ckpt_data):
    """
    Build QuantModel with AdaRound weights applied, then collect per-QuantModule scales.

    Returns:
        quant_mods: list of dicts in named_modules() traversal order:
            {
              'name': str,              # e.g. 'model.layers.0.attentions.0...'
              'w_fq': np.ndarray,       # W_fq values [N, K] — used for ONNX matching
              'w_delta': np.ndarray,    # per-channel scale [N] float32
              'w_zp': np.ndarray,       # per-channel zero_point [N] uint8
            }
    """
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from export_onnx_univ2x import load_plugin, build_model_from_cfg
    from projects.mmdet3d_plugin.univ2x.quant import QuantModel, register_bevformer_specials
    from projects.mmdet3d_plugin.univ2x.quant.quant_layer import QuantModule
    from projects.mmdet3d_plugin.univ2x.quant.quant_params import set_weight_quantize_params
    from projects.mmdet3d_plugin.univ2x.quant.adaptive_rounding import AdaRoundQuantizer

    # ── Build agent model ────────────────────────────────────────────────
    load_plugin(cfg)
    model_key = 'model_ego_agent' if args.model == 'ego' else 'model_other_agent_inf'
    if not hasattr(cfg, model_key):
        model_key = 'model'
    print(f'[scale_extraction] Building model (key={model_key}) ...')
    agent_model = build_model_from_cfg(cfg, model_key, ckpt_path=args.checkpoint)

    # Patch bev sizes
    head = agent_model.pts_bbox_head
    orig_bev_h, orig_bev_w = head.bev_h, head.bev_w
    head.bev_h = args.bev_size
    head.bev_w = args.bev_size
    agent_model.bev_h = args.bev_size
    agent_model.bev_w = args.bev_size
    if orig_bev_h != args.bev_size or orig_bev_w != args.bev_size:
        from torch.nn import Embedding
        head.bev_embedding = Embedding(args.bev_size * args.bev_size, head.embed_dims).cuda()

    # ── Build QuantModel ─────────────────────────────────────────────────
    weight_quant_params = ckpt_data.get(
        'weight_quant_params', dict(n_bits=8, channel_wise=True, scale_method='mse'))
    act_quant_params = ckpt_data.get(
        'act_quant_params',
        dict(n_bits=8, channel_wise=False, scale_method='entropy', leaf_param=True))

    encoder = agent_model.pts_bbox_head.transformer.encoder
    register_bevformer_specials()
    qmodel = QuantModel(encoder, weight_quant_params, act_quant_params, is_fusing=True)
    qmodel.cuda()
    print(f'[scale_extraction] QuantModel built. {qmodel.get_memory_footprint()}')

    # ── Apply AdaRound (step 1: re-calibrate deltas; step 2: wrap; step 3: load; step 4: bake) ──
    # Step 1: calibrate weight scales from FP32 weights
    qmodel.set_quant_state(weight_quant=False, act_quant=False)
    set_weight_quantize_params(qmodel)
    print('[scale_extraction] Weight scales calibrated.')

    # Step 2: wrap weight_quantizers with AdaRoundQuantizer
    for m in qmodel.modules():
        if isinstance(m, QuantModule) and not isinstance(m.weight_quantizer, AdaRoundQuantizer):
            m.weight_quantizer = AdaRoundQuantizer(
                uaq=m.weight_quantizer,
                round_mode='learned_hard_sigmoid',
                weight_tensor=m.org_weight.data)

    # Step 3: load AdaRound state dict (loads .alpha for each layer)
    missing, unexpected = qmodel.load_state_dict(ckpt_data['state_dict'], strict=False)
    n_alpha_keys = sum(1 for k in ckpt_data['state_dict'] if 'weight_quantizer.alpha' in k)
    print(f'[scale_extraction] Loaded state dict: {n_alpha_keys} alpha keys, '
          f'{len(unexpected)} unexpected (act_quantizer.delta etc.)')

    # Step 4: enable hard rounding + bake W_fq
    qmodel.set_quant_state(weight_quant=True, act_quant=False)
    for m in qmodel.modules():
        if hasattr(m, 'weight_quantizer') and hasattr(m.weight_quantizer, 'soft_targets'):
            m.weight_quantizer.soft_targets = False

    from projects.mmdet3d_plugin.univ2x.quant.quant_params import save_quantized_weight
    save_quantized_weight(qmodel)

    n_updated = 0
    for m in qmodel.modules():
        if isinstance(m, QuantModule):
            m.org_weight.copy_(m.weight.data)
            n_updated += 1

    qmodel.set_quant_state(weight_quant=False, act_quant=False)
    print(f'[scale_extraction] W_fq baked into org_weight for {n_updated} QuantModules.')

    # ── Collect scales in named_modules() order ──────────────────────────
    quant_mods = []
    for name, m in qmodel.named_modules():
        if not isinstance(m, QuantModule):
            continue

        delta = m.weight_quantizer.delta
        zp    = m.weight_quantizer.zero_point

        # delta / zp may be tensors of shape [N, 1] (channel-wise with reshape) or [N]
        if isinstance(delta, torch.Tensor):
            delta_np = delta.reshape(-1).detach().cpu().numpy().astype(np.float32)
        else:
            delta_np = np.array([float(delta)], dtype=np.float32)

        if isinstance(zp, torch.Tensor):
            zp_np = zp.reshape(-1).detach().cpu().numpy()
            # zero_point values from calculate_qparams are in [0, 255]
            zp_np = np.clip(zp_np, 0, 255).astype(np.uint8)
        else:
            zp_np = np.array([int(zp)], dtype=np.uint8)

        # W_fq values for value-based matching with ONNX initializers
        w_fq = m.org_weight.data.detach().cpu().numpy()

        quant_mods.append({
            'name':    name,
            'w_fq':    w_fq,
            'w_delta': delta_np,
            'w_zp':    zp_np,
        })

    print(f'[scale_extraction] Collected scales for {len(quant_mods)} QuantModules.')
    return quant_mods


# ---------------------------------------------------------------------------
# ONNX graph analysis: find quantized MatMul nodes
# ---------------------------------------------------------------------------

def find_quant_matmul_nodes(graph):
    """
    Find MatMul nodes whose weight input comes through a Transpose from a Constant node.

    Pattern:   Constant(W_fq) → Transpose → MatMul(activation, transposed_weight)

    This is the pattern produced when QuantModule.forward calls F.linear(input, org_weight, bias)
    with all quantizers disabled.  org_weight is a plain tensor (not nn.Parameter), so the ONNX
    tracer embeds it as a Constant node (not an Initializer).

    Non-quantized layers (plain nn.Linear with nn.Parameter weight) use:
      Initializer → Transpose → MatMul

    Returns:
        list of dicts, in ONNX graph traversal order:
            {
              'matmul_node':         NodeProto,
              'transpose_node':      NodeProto,
              'constant_node':       NodeProto,    # Constant node holding W_fq
              'weight_tensor_name':  str,           # output name of the Constant node
              'act_input':           str,           # name of activation input tensor
            }
    """
    # Map: output_name → node that produces it
    output_to_node = {}
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node

    quant_matmuls = []
    for node in graph.node:
        if node.op_type != 'MatMul':
            continue

        if len(node.input) < 2:
            continue

        weight_inp = node.input[1]

        # Weight input must come from a Transpose node
        if weight_inp not in output_to_node:
            continue
        transpose_node = output_to_node[weight_inp]
        if transpose_node.op_type != 'Transpose':
            continue

        # Transpose's input must come from a Constant node (= org_weight in QuantModule)
        transpose_inp = transpose_node.input[0]
        if transpose_inp not in output_to_node:
            continue
        constant_node = output_to_node[transpose_inp]
        if constant_node.op_type != 'Constant':
            continue

        quant_matmuls.append({
            'matmul_node':        node,
            'transpose_node':     transpose_node,
            'constant_node':      constant_node,
            'weight_tensor_name': transpose_inp,   # = constant_node.output[0]
            'act_input':          node.input[0],
        })

    return quant_matmuls


# ---------------------------------------------------------------------------
# Match ONNX quantized MatMul nodes to QuantModules by weight value
# ---------------------------------------------------------------------------

def build_weight_value_map(graph):
    """Build dict: tensor_name → numpy array for Constant-node weights."""
    output_to_node = {}
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node

    val_map = {}
    for node in graph.node:
        if node.op_type == 'Constant' and node.output:
            name = node.output[0]
            try:
                arr = numpy_helper.to_array(node.attribute[0].t)
                val_map[name] = arr
            except Exception:
                pass
    return val_map


def match_onnx_to_quant_mods(quant_matmuls, quant_mods, init_val_map, tol=1e-3):
    """
    Match each ONNX quantized MatMul to a QuantModule using weight value comparison.

    First attempts order-based matching (should be 1:1 if model is deterministic).
    Falls back to value-based matching if order mismatch is detected.

    Returns:
        List of (quant_matmul_dict, quant_mod_dict) pairs, same length as quant_matmuls.

    Raises:
        RuntimeError if matching fails.
    """
    n_onnx = len(quant_matmuls)
    n_mods = len(quant_mods)
    print(f'[matching] ONNX quantized MatMul nodes: {n_onnx}')
    print(f'[matching] QuantModules from model:     {n_mods}')

    if n_onnx != n_mods:
        raise RuntimeError(
            f'Count mismatch: {n_onnx} ONNX quant MatMuls vs {n_mods} QuantModules. '
            f'Ensure --bev-size / --num-cam match the ONNX export args.')

    # Try order-based matching first with value verification
    pairs = []
    for i, (onnx_node, qmod) in enumerate(zip(quant_matmuls, quant_mods)):
        onnx_w = init_val_map[onnx_node['weight_tensor_name']]  # shape [N, K] or transposed
        py_w   = qmod['w_fq']  # shape [N, K]

        # Compare as-is or transposed (ONNX may store weight in either orientation)
        match_direct = onnx_w.shape == py_w.shape and np.allclose(onnx_w, py_w, atol=tol)
        match_T      = onnx_w.shape == py_w.T.shape and np.allclose(onnx_w, py_w.T, atol=tol)

        if not (match_direct or match_T):
            print(f'  [WARNING] Order-based match failed at index {i}.')
            print(f'    ONNX weight: {onnx_w.shape}, range [{onnx_w.min():.4f}, {onnx_w.max():.4f}]')
            print(f'    PyTorch w_fq: {py_w.shape}, range [{py_w.min():.4f}, {py_w.max():.4f}]')
            print(f'    Falling back to value-based matching for remaining nodes...')
            return _fallback_value_match(quant_matmuls, quant_mods, init_val_map, tol)
        else:
            orient = 'direct' if match_direct else 'transposed'
            if i < 5 or i == n_onnx - 1:  # print first 5 and last
                print(f'  [{i:02d}] {qmod["name"]!r:60s} → matched ({orient})')
            elif i == 5:
                print(f'  ... (skipping {n_onnx - 6} middle entries) ...')
        pairs.append((onnx_node, qmod))

    print(f'[matching] All {n_onnx} nodes matched via order-based matching ✓')
    return pairs


def _fallback_value_match(quant_matmuls, quant_mods, init_val_map, tol):
    """Value-based fallback: match each ONNX node to the best-fitting QuantModule."""
    pairs = []
    used = set()
    for onnx_node in quant_matmuls:
        onnx_w = init_val_map[onnx_node['weight_tensor_name']]
        best_idx = None
        for i, qmod in enumerate(quant_mods):
            if i in used:
                continue
            py_w = qmod['w_fq']
            if (onnx_w.shape == py_w.shape and np.allclose(onnx_w, py_w, atol=tol)) or \
               (onnx_w.shape == py_w.T.shape and np.allclose(onnx_w, py_w.T, atol=tol)):
                best_idx = i
                break
        if best_idx is None:
            raise RuntimeError(
                f'Could not match ONNX weight init '
                f'{onnx_node["weight_init_name"]!r} to any QuantModule.')
        used.add(best_idx)
        pairs.append((onnx_node, quant_mods[best_idx]))
    print(f'[matching] Fallback value-based matching succeeded for all {len(pairs)} nodes ✓')
    return pairs


# ---------------------------------------------------------------------------
# Q/DQ node insertion
# ---------------------------------------------------------------------------

def _make_scale_init(name: str, values: np.ndarray) -> onnx.TensorProto:
    """Create an ONNX float32 initializer for a scale tensor."""
    arr = np.array(values, dtype=np.float32).reshape(-1)
    if arr.shape == (1,):
        arr = arr.reshape(())  # scalar
    return numpy_helper.from_array(arr, name=name)


def _make_zp_init(name: str, values: np.ndarray) -> onnx.TensorProto:
    """Create an ONNX uint8 initializer for a zero_point tensor."""
    arr = np.array(values, dtype=np.uint8).reshape(-1)
    if arr.shape == (1,):
        arr = arr.reshape(())  # scalar
    return numpy_helper.from_array(arr, name=name)


def insert_weight_qdq(graph, pairs):
    """
    For each matched (ONNX node, QuantModule) pair, insert weight Q/DQ:

        Constant(W_fq) → QuantizeLinear(scale_sym, zp=0)
                       → DequantizeLinear(scale_sym, zp=0)
                       → Transpose → MatMul

    Uses SYMMETRIC per-channel INT8 quantization because TRT requires zero_point = 0 on GPU.

    Per-channel scale is computed from W_fq directly:
        scale_sym[i] = max(|W_fq[i, :]|) / 127.0

    This preserves AdaRound rounding decisions as closely as possible within the
    symmetric INT8 grid constraint imposed by TRT.

    zero_point: INT8 scalar = 0 (TRT requires symmetric quantization on non-DLA targets)
    axis:       0  (output channel = axis 0 of weight tensor [N, K])
    """
    # Map: constant_output_name → list of (ql_node, dql_node) to insert after it
    # We will do a single pass over graph.node and insert Q/DQ right after each Constant.
    qdq_after_const = {}  # constant_output_name → (ql_node, dql_node)
    new_inits = []
    n_inserted = 0

    for onnx_node, qmod in pairs:
        w_tensor_name = onnx_node['weight_tensor_name']   # output of Constant node
        transpose_node = onnx_node['transpose_node']
        mod_name = qmod['name']

        # Unique base name for new tensors (sanitise '.' → '_')
        safe_name = mod_name.replace('.', '_')

        # Scale and zero_point initializer names
        scale_init_name = f'_qdq_{safe_name}_w_scale'
        zp_init_name    = f'_qdq_{safe_name}_w_zp'

        # Intermediate tensor names
        ql_out_name  = f'_qdq_{safe_name}_w_quant'
        dql_out_name = f'_qdq_{safe_name}_w_dequant'

        # ── Compute symmetric per-channel scale from W_fq ─────────────────
        # TRT requires zero_point = 0 (symmetric INT8) on non-DLA targets.
        # We derive scale_sym from W_fq directly: scale[i] = max|W_fq[i,:]| / 127
        # This encodes AdaRound's optimized values as faithfully as possible
        # within the symmetric INT8 grid.
        w_fq = qmod['w_fq']  # shape [N, K], FP32 on AdaRound grid
        n_out = w_fq.shape[0]
        scale_sym = (np.abs(w_fq).max(axis=1) / 127.0).astype(np.float32)
        # Avoid division by zero for all-zero channels
        scale_sym = np.maximum(scale_sym, 1e-8)
        # Zero point = 0, dtype = int8 (TRT symmetric INT8)
        zp_sym = np.zeros(n_out, dtype=np.int8)

        axis = 0  # per output channel

        # ------------------------------------------------------------------
        # New initializers: scale and zero_point
        # ------------------------------------------------------------------
        new_inits.append(_make_scale_init(scale_init_name, scale_sym))
        # Use int8 zero_point for symmetric quantization
        zp_tensor = numpy_helper.from_array(zp_sym, name=zp_init_name)
        new_inits.append(zp_tensor)

        # ------------------------------------------------------------------
        # QuantizeLinear: Constant(W_fq) → QL output (UINT8)
        # ------------------------------------------------------------------
        ql_node = onnx_helper.make_node(
            'QuantizeLinear',
            inputs=[w_tensor_name, scale_init_name, zp_init_name],
            outputs=[ql_out_name],
            name=f'_qdq_{safe_name}_QL',
            axis=axis,
        )

        # ------------------------------------------------------------------
        # DequantizeLinear: QL output → DQL output (FP32 = W_fq reconstructed)
        # ------------------------------------------------------------------
        dql_node = onnx_helper.make_node(
            'DequantizeLinear',
            inputs=[ql_out_name, scale_init_name, zp_init_name],
            outputs=[dql_out_name],
            name=f'_qdq_{safe_name}_DQL',
            axis=axis,
        )

        # Register for topologically-correct insertion after the Constant node
        qdq_after_const[w_tensor_name] = (ql_node, dql_node)

        # ------------------------------------------------------------------
        # Re-wire: Transpose takes dql_out instead of Constant output
        # ------------------------------------------------------------------
        transpose_node.input[0] = dql_out_name

        n_inserted += 1

    # ── Rebuild graph.node with Q/DQ inserted after each Constant ─────────
    new_node_list = []
    for node in graph.node:
        new_node_list.append(node)
        if node.op_type == 'Constant' and node.output:
            const_out = node.output[0]
            if const_out in qdq_after_const:
                ql_node, dql_node = qdq_after_const[const_out]
                new_node_list.append(ql_node)
                new_node_list.append(dql_node)

    # Replace graph nodes in-place
    del graph.node[:]
    graph.node.extend(new_node_list)

    # Append new initializers (scale/zp tensors)
    graph.initializer.extend(new_inits)

    print(f'[qdq_insert] Inserted weight Q/DQ for {n_inserted} QuantModule layers.')
    print(f'             Added {len(new_inits)} new initializers '
          f'(scale + zp pairs: {len(new_inits) // 2}).')
    return n_inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, tools_dir)
    sys.path.insert(0, os.path.dirname(tools_dir))  # project root

    # ── Load AdaRound checkpoint ──────────────────────────────────────────
    print(f'Loading AdaRound checkpoint: {args.adaround_ckpt}')
    ckpt_data = torch.load(args.adaround_ckpt, map_location='cpu')
    n_adaround_keys = sum(1 for k in ckpt_data.get('state_dict', {})
                         if 'weight_quantizer.alpha' in k)
    print(f'  Found {n_adaround_keys} AdaRound alpha keys in checkpoint.')

    # ── Load config ───────────────────────────────────────────────────────
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # ── Step 1: collect AdaRound scales from QuantModel ───────────────────
    print('\n=== Step 1: Extract AdaRound weight scales ===')
    quant_mods = collect_adaround_scales(cfg, args, ckpt_data)

    # ── Step 2: load ONNX and find quantized MatMul patterns ─────────────
    print(f'\n=== Step 2: Analyse ONNX graph: {args.input_onnx} ===')
    model_onnx = onnx.load(args.input_onnx)
    graph = model_onnx.graph

    print(f'  ONNX opset: {[op.version for op in model_onnx.opset_import]}')
    print(f'  Graph nodes:      {len(graph.node)}')
    print(f'  Initializers:     {len(graph.initializer)}')

    quant_matmuls = find_quant_matmul_nodes(graph)
    print(f'  Quantized MatMul nodes (Constant→Transpose→MatMul): {len(quant_matmuls)}')

    # Sanity check
    n_gemm = sum(1 for n in graph.node if n.op_type == 'Gemm')
    n_matmul = sum(1 for n in graph.node if n.op_type == 'MatMul')
    n_init_transpose_matmul = sum(
        1 for n in graph.node
        if n.op_type == 'MatMul' and len(n.input) >= 2
    )
    print(f'  Total Gemm nodes:   {n_gemm}  (non-quantized nn.Parameter weights)')
    print(f'  Total MatMul nodes: {n_matmul}  ({len(quant_matmuls)} QuantModule + rest are attention ops)')

    # ── Step 3: match ONNX nodes to QuantModules ─────────────────────────
    print('\n=== Step 3: Match ONNX nodes to QuantModules ===')
    init_val_map = build_weight_value_map(graph)
    pairs = match_onnx_to_quant_mods(quant_matmuls, quant_mods, init_val_map, tol=args.match_tol)

    # ── Step 4: insert weight Q/DQ nodes ─────────────────────────────────
    print('\n=== Step 4: Insert weight Q/DQ nodes ===')
    n_inserted = insert_weight_qdq(graph, pairs)

    if n_inserted != len(quant_mods):
        print(f'  [WARNING] Expected {len(quant_mods)} insertions, got {n_inserted}.')

    # ── Step 5: verify and save ───────────────────────────────────────────
    print(f'\n=== Step 5: Save Q/DQ ONNX → {args.out} ===')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if not args.no_check:
        print('  Running onnx.checker.check_model ...')
        try:
            onnx.checker.check_model(model_onnx)
            print('  onnx.checker: PASS ✓')
        except Exception as e:
            print(f'  [WARNING] onnx.checker raised: {e}')
            print('  Saving anyway — TRT may still parse correctly.')

    onnx.save(model_onnx, args.out)
    size_mb = os.path.getsize(args.out) / 1024**2
    print(f'  Saved: {args.out}  ({size_mb:.1f} MB)')
    print(f'  Nodes now: {len(graph.node)} (+{2 * n_inserted} Q/DQ nodes added)')
    print(f'  Initializers now: {len(graph.initializer)} (+{2 * n_inserted} scale/zp tensors)')

    print('\n=== Next steps ===')
    print('Build TRT INT8 engine (same command as vanilla PTQ):')
    print(f'  python tools/build_trt_int8_univ2x.py \\')
    print(f'      --onnx {args.out} \\')
    print(f'      --out trt_engines/univ2x_ego_bev_encoder_adaround_qdq_int8.trt \\')
    print(f'      --target bev_encoder \\')
    print(f'      --plugin plugins/build/libuniv2x_plugins.so \\')
    print(f'      --cali-data calibration/bev_encoder_calib_inputs.pkl')
    print()
    print('Validate cosine similarity (D-1):')
    print(f'  python tools/validate_adaround_bev.py \\')
    print(f'      projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \\')
    print(f'      ckpts/univ2x_coop_e2e_stg2.pth \\')
    print(f'      --adaround-ckpt calibration/quant_encoder_adaround.pth \\')
    print(f'      --trt-engine trt_engines/univ2x_ego_bev_encoder_adaround_qdq_int8.trt')
    print()
    print('End-to-end AMOTA validation (D-2):')
    print(f'  python tools/test_trt.py ... \\')
    print(f'      --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_adaround_qdq_int8.trt \\')
    print(f'      --eval bbox')


if __name__ == '__main__':
    main()
