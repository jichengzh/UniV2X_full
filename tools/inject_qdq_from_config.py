"""
inject_qdq_from_config.py — Inject weight + activation Q/DQ nodes into ONNX
based on a quant_config.json produced by PyTorch-side calibration.

This is the bridge between PyTorch-side search (Part A) and TRT deployment (Part B).

Key difference from build_qdq_onnx_adaround.py:
  - Inserts BOTH weight AND activation Q/DQ (previous tool only did weights)
  - Reads scales from quant_config.json (not from AdaRound checkpoint)
  - Supports per-layer bit-width, granularity, and skip configuration

Why both weight AND activation Q/DQ are needed:
  Weight-only Q/DQ causes TRT to enter "explicit quantization mode" where the
  Calibrator is completely ignored, activations fall back to FP16, and accuracy
  collapses (AMOTA 0.137 vs 0.353 with implicit INT8).
  When both are present, TRT correctly fuses:
    DQ(activation) + DQ(weight) -> MatMul -> Q(output)
  into a single INT8 kernel.

Usage:
    python tools/inject_qdq_from_config.py \\
        --input-onnx onnx/univ2x_ego_bev_encoder_200_1cam.onnx \\
        --quant-config quant_configs/searched_config.json \\
        --output onnx/univ2x_ego_bev_encoder_qdq.onnx
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper as onnx_helper, numpy_helper


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Inject weight + activation Q/DQ nodes into ONNX from quant_config.json')
    p.add_argument('--input-onnx', required=True,
                   help='FP32 ONNX model (no existing Q/DQ nodes)')
    p.add_argument('--quant-config', required=True,
                   help='quant_config.json with per-layer w_scale, a_scale')
    p.add_argument('--output', required=True,
                   help='Output ONNX path with Q/DQ nodes inserted')
    p.add_argument('--no-check', action='store_true',
                   help='Skip onnx.checker.check_model (faster)')
    p.add_argument('--w-only-act-scale', type=float, default=1e4,
                   help='Transparent activation scale for W-only layers (default: 1e4)')
    p.add_argument('--verbose', action='store_true',
                   help='Print detailed per-layer matching info')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loading and normalization
# ---------------------------------------------------------------------------

def load_quant_config(config_path: str) -> dict:
    """
    Load quant_config.json and normalize per-layer entries.

    Expected format:
    {
      "version": "1.1",
      "global": {
          "symmetric": true,
          "default_w_bits": 8,
          "default_a_bits": 8,
          "default_quant_target": "W+A",
          ...
      },
      "layers": {
          "layers.0.attentions.0.attn.in_proj_weight": {
              "w_bits": 8, "a_bits": 8,
              "quant_target": "W+A",
              "w_scale": 0.0123,
              "a_scale": 0.0456
          },
          ...
      }
    }

    Returns the full config dict with defaults merged into each layer entry.
    """
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    global_cfg = cfg.get('global', {})
    layers = cfg.get('layers', {})

    # Merge global defaults into each layer entry
    defaults = {
        'w_bits': global_cfg.get('default_w_bits', 8),
        'a_bits': global_cfg.get('default_a_bits', 8),
        'quant_target': global_cfg.get('default_quant_target', 'W+A'),
        'w_granularity': global_cfg.get('default_w_granularity', 'per_tensor'),
        'a_granularity': global_cfg.get('default_a_granularity', 'per_tensor'),
        'symmetric': global_cfg.get('symmetric', True),
    }

    normalized_layers = OrderedDict()
    for layer_name, layer_cfg in layers.items():
        entry = dict(defaults)
        entry.update(layer_cfg)
        normalized_layers[layer_name] = entry

    cfg['layers'] = normalized_layers
    cfg['_defaults'] = defaults
    return cfg


# ---------------------------------------------------------------------------
# ONNX graph analysis: find quantizable MatMul nodes
# ---------------------------------------------------------------------------

def _build_output_to_node_map(graph: onnx.GraphProto) -> Dict[str, onnx.NodeProto]:
    """Map: output tensor name -> node that produces it."""
    result = {}
    for node in graph.node:
        for out in node.output:
            result[out] = node
    return result


def _build_initializer_map(graph: onnx.GraphProto) -> Dict[str, onnx.TensorProto]:
    """Map: initializer name -> TensorProto."""
    return {init.name: init for init in graph.initializer}


def _get_initializer_names(graph: onnx.GraphProto) -> set:
    """Set of all initializer tensor names."""
    return {init.name for init in graph.initializer}


def find_quantizable_matmul_nodes(graph: onnx.GraphProto) -> List[dict]:
    """
    Find MatMul nodes whose weight input is a static tensor (quantizable linear layer).

    Handles three patterns:
      1. Constant -> Transpose -> MatMul  (QuantModule org_weight from AdaRound export)
      2. Initializer -> Transpose -> MatMul  (standard nn.Linear)
      3. Initializer -> MatMul  (Linear without transpose, some frameworks)

    Returns list of dicts in ONNX graph traversal order:
        {
          'matmul_node':        NodeProto,
          'transpose_node':     NodeProto or None,
          'weight_source':      'constant' | 'initializer',
          'weight_tensor_name': str,   # name of the raw weight tensor (before transpose)
          'act_input':          str,   # name of the activation input tensor
        }
    """
    output_to_node = _build_output_to_node_map(graph)
    init_names = _get_initializer_names(graph)

    quantizable = []
    for node in graph.node:
        if node.op_type != 'MatMul':
            continue
        if len(node.input) < 2:
            continue

        weight_inp = node.input[1]
        act_input = node.input[0]

        # --- Pattern 1 & 2: ... -> Transpose -> MatMul ---
        if weight_inp in output_to_node:
            transpose_node = output_to_node[weight_inp]
            if transpose_node.op_type == 'Transpose':
                raw_weight_name = transpose_node.input[0]

                # Pattern 1: Constant -> Transpose -> MatMul
                if raw_weight_name in output_to_node:
                    source_node = output_to_node[raw_weight_name]
                    if source_node.op_type == 'Constant':
                        quantizable.append({
                            'matmul_node': node,
                            'transpose_node': transpose_node,
                            'weight_source': 'constant',
                            'weight_tensor_name': raw_weight_name,
                            'act_input': act_input,
                        })
                        continue

                # Pattern 2: Initializer -> Transpose -> MatMul
                if raw_weight_name in init_names:
                    quantizable.append({
                        'matmul_node': node,
                        'transpose_node': transpose_node,
                        'weight_source': 'initializer',
                        'weight_tensor_name': raw_weight_name,
                        'act_input': act_input,
                    })
                    continue

        # --- Pattern 3: Initializer -> MatMul (no transpose) ---
        if weight_inp in init_names:
            quantizable.append({
                'matmul_node': node,
                'transpose_node': None,
                'weight_source': 'initializer',
                'weight_tensor_name': weight_inp,
                'act_input': act_input,
            })

    return quantizable


# ---------------------------------------------------------------------------
# Weight value extraction for matching
# ---------------------------------------------------------------------------

def get_weight_array(
    graph: onnx.GraphProto,
    entry: dict,
    output_to_node: Dict[str, onnx.NodeProto],
    init_map: Dict[str, onnx.TensorProto],
) -> Optional[np.ndarray]:
    """Extract the numpy weight array for a quantizable MatMul entry."""
    name = entry['weight_tensor_name']
    source = entry['weight_source']

    if source == 'initializer':
        if name in init_map:
            return numpy_helper.to_array(init_map[name])
        return None

    if source == 'constant':
        if name in output_to_node:
            const_node = output_to_node[name]
            if const_node.op_type == 'Constant' and const_node.attribute:
                try:
                    return numpy_helper.to_array(const_node.attribute[0].t)
                except Exception:
                    return None
    return None


# ---------------------------------------------------------------------------
# Match ONNX nodes to quant_config layers
# ---------------------------------------------------------------------------

def match_onnx_to_config(
    quantizable: List[dict],
    config: dict,
    graph: onnx.GraphProto,
    verbose: bool = False,
) -> List[Tuple[dict, Optional[dict]]]:
    """
    Match each quantizable ONNX MatMul node to a quant_config layer entry.

    Strategy:
      - If config has per-layer entries with w_scale/a_scale, use order-based matching:
        quantizable MatMul nodes in ONNX graph order correspond to config layers
        in their dict iteration order (Python 3.7+ preserves insertion order).
      - If config has no per-layer entries (only global defaults), apply global
        defaults to all quantizable nodes.

    Returns:
        List of (onnx_entry, config_entry_or_None) pairs.
        config_entry is None for layers that should be skipped (quant_target="none").
    """
    layers = config.get('layers', {})
    defaults = config.get('_defaults', {})

    if not layers:
        # No per-layer config: apply global defaults to all quantizable nodes.
        # Global defaults won't have w_scale/a_scale, so we compute them from weights.
        print(f'[matching] No per-layer config found. '
              f'Applying global defaults to all {len(quantizable)} quantizable nodes.')
        result = []
        for entry in quantizable:
            cfg_entry = dict(defaults)
            cfg_entry['_needs_scale_computation'] = True
            result.append((entry, cfg_entry))
        return result

    layer_list = list(layers.items())
    n_onnx = len(quantizable)
    n_cfg = len(layer_list)

    print(f'[matching] Quantizable ONNX MatMul nodes: {n_onnx}')
    print(f'[matching] Config layer entries:           {n_cfg}')

    if n_onnx != n_cfg:
        print(f'[WARNING] Count mismatch: {n_onnx} ONNX nodes vs {n_cfg} config layers.')
        print(f'          Will match min({n_onnx}, {n_cfg}) by order; extras will use defaults.')

    result = []
    for i, onnx_entry in enumerate(quantizable):
        if i < n_cfg:
            layer_name, layer_cfg = layer_list[i]
            quant_target = layer_cfg.get('quant_target', 'W+A')

            if quant_target == 'none':
                if verbose:
                    print(f'  [{i:03d}] SKIP  {layer_name} (quant_target=none)')
                result.append((onnx_entry, None))
            else:
                if verbose or i < 3 or i == n_onnx - 1:
                    print(f'  [{i:03d}] {layer_name:60s} -> '
                          f'target={quant_target}, '
                          f'w_scale={layer_cfg.get("w_scale", "auto")}, '
                          f'a_scale={layer_cfg.get("a_scale", "auto")}')
                elif i == 3:
                    print(f'  ... ({n_cfg - 4} more entries) ...')
                result.append((onnx_entry, layer_cfg))
        else:
            # Extra ONNX nodes beyond config: apply global defaults
            cfg_entry = dict(defaults)
            cfg_entry['_needs_scale_computation'] = True
            result.append((onnx_entry, cfg_entry))

    return result


# ---------------------------------------------------------------------------
# Scale computation helpers
# ---------------------------------------------------------------------------

def compute_weight_scale(
    weight: np.ndarray,
    n_bits: int = 8,
    granularity: str = 'per_tensor',
) -> np.ndarray:
    """
    Compute symmetric weight quantization scale.

    Args:
        weight: weight tensor, shape [out_channels, ...] or [out, in]
        n_bits: quantization bit-width (8 -> /127, 4 -> /7)
        granularity: 'per_tensor' or 'per_channel' (axis=0)

    Returns:
        scale as float32 numpy array (scalar or [out_channels])
    """
    qmax = (1 << (n_bits - 1)) - 1  # 127 for int8, 7 for int4

    if granularity == 'per_channel':
        # Flatten all dims except axis 0
        w_flat = weight.reshape(weight.shape[0], -1)
        max_abs = np.abs(w_flat).max(axis=1).astype(np.float32)
    else:
        max_abs = np.array([np.abs(weight).max()], dtype=np.float32)

    scale = max_abs / float(qmax)
    scale = np.maximum(scale, 1e-8)  # avoid division by zero
    return scale.astype(np.float32)


# ---------------------------------------------------------------------------
# ONNX initializer / node creation helpers
# ---------------------------------------------------------------------------

def _make_scale_init(name: str, values: np.ndarray) -> onnx.TensorProto:
    """Create a float32 initializer for a scale tensor."""
    arr = np.array(values, dtype=np.float32).reshape(-1)
    if arr.shape == (1,):
        arr = arr.reshape(())  # scalar for per-tensor
    return numpy_helper.from_array(arr, name=name)


def _make_zp_int8_init(name: str, size: int = 1) -> onnx.TensorProto:
    """Create an int8 zero_point initializer (all zeros, symmetric)."""
    arr = np.zeros(size, dtype=np.int8)
    if size == 1:
        arr = arr.reshape(())  # scalar
    return numpy_helper.from_array(arr, name=name)


# ---------------------------------------------------------------------------
# Q/DQ node insertion
# ---------------------------------------------------------------------------

def inject_qdq_nodes(
    graph: onnx.GraphProto,
    matched_pairs: List[Tuple[dict, Optional[dict]]],
    w_only_act_scale: float = 1e4,
    verbose: bool = False,
) -> Tuple[int, int, int]:
    """
    Insert Q/DQ nodes for both weights and activations.

    For each matched pair where config is not None:
      - Weight Q/DQ: inserted between weight source and Transpose/MatMul
      - Activation Q/DQ: inserted on the activation input to MatMul

    For W-only layers:
      - Weight Q/DQ: normal
      - Activation Q/DQ: transparent (very large scale) to prevent TRT
        from ignoring the layer in explicit quantization mode

    Activation deduplication:
      Multiple MatMul nodes may share the same activation input (e.g., Q/K/V
      projections in attention). We insert ONE Q/DQ pair per unique activation
      input and have all downstream MatMuls reference the same DQL output.

    Returns:
        (n_weight_qdq, n_activation_qdq, n_skipped)
    """
    output_to_node = _build_output_to_node_map(graph)
    init_map = _build_initializer_map(graph)

    new_inits = []
    # Track weight Q/DQ insertion points: weight_tensor_name -> (ql_node, dql_node)
    weight_qdq_after = {}
    # Track activation Q/DQ: act_input_name -> dql_output_name
    act_qdq_map: Dict[str, str] = {}
    # Activation Q/DQ nodes to insert before specific MatMul nodes
    act_qdq_nodes: List[Tuple[str, onnx.NodeProto, onnx.NodeProto]] = []

    n_weight_qdq = 0
    n_activation_qdq = 0
    n_skipped = 0

    for idx, (onnx_entry, cfg_entry) in enumerate(matched_pairs):
        if cfg_entry is None:
            n_skipped += 1
            continue

        quant_target = cfg_entry.get('quant_target', 'W+A')
        w_bits = cfg_entry.get('w_bits', 8)
        a_bits = cfg_entry.get('a_bits', 8)
        w_granularity = cfg_entry.get('w_granularity', 'per_tensor')

        safe_name = f'layer{idx:03d}'
        matmul_node = onnx_entry['matmul_node']
        transpose_node = onnx_entry['transpose_node']
        weight_tensor_name = onnx_entry['weight_tensor_name']
        act_input_name = onnx_entry['act_input']

        # ── Weight Q/DQ ──────────────────────────────────────────────────
        if quant_target in ('W+A', 'W-only'):
            # Determine weight scale
            w_scale_val = cfg_entry.get('w_scale')
            if w_scale_val is None or cfg_entry.get('_needs_scale_computation'):
                # Compute scale from weight values
                w_arr = get_weight_array(graph, onnx_entry, output_to_node, init_map)
                if w_arr is not None:
                    w_scale_np = compute_weight_scale(w_arr, w_bits, w_granularity)
                else:
                    print(f'  [WARNING] Cannot extract weight for node {idx}, '
                          f'skipping weight Q/DQ.')
                    w_scale_np = None
            else:
                # Scale from config (could be scalar or list for per-channel)
                w_scale_np = np.array(w_scale_val, dtype=np.float32).reshape(-1)
                w_scale_np = np.maximum(w_scale_np, 1e-8)

                # Validate scale size against ONNX weight shape.
                # PyTorch per-channel quantizes along dim 0 of weight (out_features),
                # but the ONNX weight may be stored un-transposed (out, in) with
                # a Transpose node before MatMul. The Q/DQ axis must match the
                # weight tensor as stored in ONNX (before any Transpose).
                w_arr = get_weight_array(graph, onnx_entry, output_to_node, init_map)
                if w_arr is not None and w_scale_np.size > 1:
                    onnx_out_ch = w_arr.shape[0]  # first dim of stored weight
                    if w_scale_np.size != onnx_out_ch:
                        # Scale was computed for a different axis; try the other dim
                        if w_scale_np.size == w_arr.shape[-1]:
                            # Recompute per-channel scale along axis 0 of ONNX weight
                            w_scale_np = (np.abs(w_arr).max(axis=tuple(range(1, w_arr.ndim)),
                                                            keepdims=False) / (2**(w_bits-1)-1)
                                          ).astype(np.float32)
                            w_scale_np = np.maximum(w_scale_np, 1e-8)
                        else:
                            # Fallback: use per-tensor
                            w_scale_np = np.array([np.abs(w_arr).max() / (2**(w_bits-1)-1)],
                                                  dtype=np.float32)
                            w_scale_np = np.maximum(w_scale_np, 1e-8)

            if w_scale_np is not None:
                # Determine axis for per-channel vs per-tensor
                is_per_channel = (w_granularity == 'per_channel' and w_scale_np.size > 1)
                axis = 0 if is_per_channel else None

                # Scale and zero_point initializer names
                w_scale_name = f'_qdq_{safe_name}_w_scale'
                w_zp_name = f'_qdq_{safe_name}_w_zp'
                ql_out_name = f'_qdq_{safe_name}_w_quant'
                dql_out_name = f'_qdq_{safe_name}_w_dequant'

                new_inits.append(_make_scale_init(w_scale_name, w_scale_np))
                zp_size = w_scale_np.size if is_per_channel else 1
                new_inits.append(_make_zp_int8_init(w_zp_name, zp_size))

                # QuantizeLinear node
                ql_kwargs = dict(
                    inputs=[weight_tensor_name, w_scale_name, w_zp_name],
                    outputs=[ql_out_name],
                    name=f'_qdq_{safe_name}_w_QL',
                )
                if axis is not None:
                    ql_kwargs['axis'] = axis
                ql_node = onnx_helper.make_node('QuantizeLinear', **ql_kwargs)

                # DequantizeLinear node
                dql_kwargs = dict(
                    inputs=[ql_out_name, w_scale_name, w_zp_name],
                    outputs=[dql_out_name],
                    name=f'_qdq_{safe_name}_w_DQL',
                )
                if axis is not None:
                    dql_kwargs['axis'] = axis
                dql_node = onnx_helper.make_node('DequantizeLinear', **dql_kwargs)

                # Re-wire: the consumer of the weight tensor now reads from DQL output
                if transpose_node is not None:
                    transpose_node.input[0] = dql_out_name
                else:
                    # No transpose: MatMul reads weight directly
                    matmul_node.input[1] = dql_out_name

                # Register for topological insertion
                weight_qdq_after[weight_tensor_name] = (ql_node, dql_node)
                n_weight_qdq += 1

        # ── Activation Q/DQ ──────────────────────────────────────────────
        if quant_target in ('W+A', 'W-only'):
            # For W-only: use transparent scale (very large) so activation
            # effectively passes through but TRT still sees explicit Q/DQ
            # and does not fall back to ignoring the layer.
            if quant_target == 'W-only':
                a_scale_val = w_only_act_scale
            else:
                a_scale_val = cfg_entry.get('a_scale')
                if a_scale_val is None or cfg_entry.get('_needs_scale_computation'):
                    # No activation scale available: use transparent pass-through
                    # This should not happen in a properly calibrated config, but
                    # we handle it gracefully.
                    print(f'  [WARNING] No a_scale for node {idx}, '
                          f'using transparent scale {w_only_act_scale}')
                    a_scale_val = w_only_act_scale

            # a_scale_val may be a scalar, a list, or a numpy array
            if isinstance(a_scale_val, (list, tuple)):
                a_scale_np = np.array(a_scale_val, dtype=np.float32).reshape(-1)
            else:
                a_scale_np = np.array([float(a_scale_val)], dtype=np.float32)
            a_scale_np = np.maximum(a_scale_np, 1e-8)

            # Deduplicate: if this activation input already has Q/DQ, reuse it
            if act_input_name in act_qdq_map:
                # Re-wire MatMul to use the existing DQL output
                matmul_node.input[0] = act_qdq_map[act_input_name]
                if verbose:
                    print(f'  [{idx:03d}] Activation Q/DQ reused for input '
                          f'{act_input_name!r}')
            else:
                # Create new activation Q/DQ pair
                a_scale_name = f'_qdq_{safe_name}_a_scale'
                a_zp_name = f'_qdq_{safe_name}_a_zp'
                a_ql_out = f'_qdq_{safe_name}_a_quant'
                a_dql_out = f'_qdq_{safe_name}_a_dequant'

                new_inits.append(_make_scale_init(a_scale_name, a_scale_np))
                new_inits.append(_make_zp_int8_init(a_zp_name, 1))

                a_ql_node = onnx_helper.make_node(
                    'QuantizeLinear',
                    inputs=[act_input_name, a_scale_name, a_zp_name],
                    outputs=[a_ql_out],
                    name=f'_qdq_{safe_name}_a_QL',
                )
                a_dql_node = onnx_helper.make_node(
                    'DequantizeLinear',
                    inputs=[a_ql_out, a_scale_name, a_zp_name],
                    outputs=[a_dql_out],
                    name=f'_qdq_{safe_name}_a_DQL',
                )

                # Re-wire MatMul activation input
                matmul_node.input[0] = a_dql_out

                # Register for insertion and deduplication
                act_qdq_map[act_input_name] = a_dql_out
                act_qdq_nodes.append((act_input_name, a_ql_node, a_dql_node))
                n_activation_qdq += 1

    # ── Rebuild graph.node in topological order ──────────────────────────
    # Strategy: single pass over existing nodes.
    # - After a Constant node whose output has weight Q/DQ, insert QL + DQL.
    # - Before a MatMul node that uses an activation Q/DQ, insert the
    #   activation QL + DQL (if not already inserted).

    # Build lookup: which act Q/DQ nodes should be inserted before which MatMul
    # We collect all MatMul nodes that need activation Q/DQ, and insert the
    # Q/DQ nodes right before the FIRST MatMul that uses each unique activation.
    act_qdq_by_input: Dict[str, Tuple[onnx.NodeProto, onnx.NodeProto]] = {}
    for act_inp, ql_node, dql_node in act_qdq_nodes:
        act_qdq_by_input[act_inp] = (ql_node, dql_node)

    # Track which activation Q/DQ pairs have been inserted
    act_qdq_inserted: set = set()

    # Also track MatMul nodes that were re-wired to activation DQL outputs
    matmul_act_sources: Dict[str, str] = {}
    for onnx_entry, cfg_entry in matched_pairs:
        if cfg_entry is None:
            continue
        matmul_node = onnx_entry['matmul_node']
        act_input_name = onnx_entry['act_input']
        if act_input_name in act_qdq_map:
            matmul_act_sources[id(matmul_node)] = act_input_name

    new_node_list = []
    for node in graph.node:
        # Insert weight Q/DQ after Constant nodes
        if node.op_type == 'Constant' and node.output:
            new_node_list.append(node)
            const_out = node.output[0]
            if const_out in weight_qdq_after:
                ql_node, dql_node = weight_qdq_after[const_out]
                new_node_list.append(ql_node)
                new_node_list.append(dql_node)
            continue

        # Insert activation Q/DQ before the first MatMul that uses each unique
        # activation input. We check if this MatMul was re-wired.
        if node.op_type == 'MatMul' and id(node) in matmul_act_sources:
            orig_act_input = matmul_act_sources[id(node)]
            if orig_act_input in act_qdq_by_input and orig_act_input not in act_qdq_inserted:
                ql_node, dql_node = act_qdq_by_input[orig_act_input]
                new_node_list.append(ql_node)
                new_node_list.append(dql_node)
                act_qdq_inserted.add(orig_act_input)

        new_node_list.append(node)

    # Handle weight Q/DQ for Initializer-sourced weights (not Constant nodes).
    # These won't be caught by the "after Constant" insertion above.
    # We need to insert them. Find which weight Q/DQ pairs were NOT yet inserted.
    init_names = _get_initializer_names(graph)
    pending_weight_qdq = []
    for w_name, (ql_node, dql_node) in weight_qdq_after.items():
        if w_name in init_names:
            # This weight comes from an Initializer, not a Constant node.
            # The Q/DQ was not inserted in the loop above.
            pending_weight_qdq.append((w_name, ql_node, dql_node))

    if pending_weight_qdq:
        # Insert Initializer-sourced weight Q/DQ before their consuming node.
        # Build a map from DQL output name to the pending pair.
        pending_by_dql_out = {}
        for w_name, ql_node, dql_node in pending_weight_qdq:
            pending_by_dql_out[dql_node.output[0]] = (ql_node, dql_node)

        # Re-scan and insert before the first consumer (Transpose or MatMul)
        final_node_list = []
        inserted_dql_outs = set()
        for node in new_node_list:
            # Check if any of this node's inputs is a pending DQL output
            for inp in node.input:
                if inp in pending_by_dql_out and inp not in inserted_dql_outs:
                    ql_node, dql_node = pending_by_dql_out[inp]
                    final_node_list.append(ql_node)
                    final_node_list.append(dql_node)
                    inserted_dql_outs.add(inp)
            final_node_list.append(node)
        new_node_list = final_node_list

    # Also check if any activation Q/DQ pairs were not inserted yet
    # (e.g., if the activation source was an Initializer or graph input)
    for act_inp in act_qdq_by_input:
        if act_inp not in act_qdq_inserted:
            # Insert at the beginning of the node list (safe since act inputs
            # are either graph inputs or initializers)
            ql_node, dql_node = act_qdq_by_input[act_inp]
            new_node_list.insert(0, dql_node)
            new_node_list.insert(0, ql_node)
            act_qdq_inserted.add(act_inp)

    # Replace graph nodes in-place
    del graph.node[:]
    graph.node.extend(new_node_list)

    # Append new initializers
    graph.initializer.extend(new_inits)

    return n_weight_qdq, n_activation_qdq, n_skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Load quant config ────────────────────────────────────────────────
    print(f'Loading quant config: {args.quant_config}')
    config = load_quant_config(args.quant_config)

    n_layers = len(config.get('layers', {}))
    global_cfg = config.get('global', {})
    print(f'  Version:       {config.get("version", "unknown")}')
    print(f'  Global target: {global_cfg.get("default_quant_target", "W+A")}')
    print(f'  Layer entries: {n_layers}')

    # ── Load ONNX model ──────────────────────────────────────────────────
    print(f'\nLoading ONNX model: {args.input_onnx}')
    model_onnx = onnx.load(args.input_onnx)
    graph = model_onnx.graph

    print(f'  ONNX opset:   {[op.version for op in model_onnx.opset_import]}')
    print(f'  Graph nodes:  {len(graph.node)}')
    print(f'  Initializers: {len(graph.initializer)}')

    # ── Find quantizable MatMul nodes ────────────────────────────────────
    print('\n=== Step 1: Find quantizable MatMul nodes ===')
    quantizable = find_quantizable_matmul_nodes(graph)

    n_matmul = sum(1 for n in graph.node if n.op_type == 'MatMul')
    n_gemm = sum(1 for n in graph.node if n.op_type == 'Gemm')
    print(f'  Total MatMul nodes:       {n_matmul}')
    print(f'  Total Gemm nodes:         {n_gemm}')
    print(f'  Quantizable MatMul nodes: {len(quantizable)}')

    # Report pattern distribution
    n_const = sum(1 for q in quantizable if q['weight_source'] == 'constant')
    n_init = sum(1 for q in quantizable if q['weight_source'] == 'initializer')
    n_with_transpose = sum(1 for q in quantizable if q['transpose_node'] is not None)
    n_without_transpose = sum(1 for q in quantizable if q['transpose_node'] is None)
    print(f'    From Constant nodes:    {n_const}')
    print(f'    From Initializers:      {n_init}')
    print(f'    With Transpose:         {n_with_transpose}')
    print(f'    Without Transpose:      {n_without_transpose}')

    if not quantizable:
        print('[ERROR] No quantizable MatMul nodes found. Check ONNX model structure.')
        sys.exit(1)

    # ── Match ONNX nodes to config ───────────────────────────────────────
    print('\n=== Step 2: Match ONNX nodes to config ===')
    matched_pairs = match_onnx_to_config(
        quantizable, config, graph, verbose=args.verbose)

    # ── Inject Q/DQ nodes ────────────────────────────────────────────────
    print('\n=== Step 3: Inject Q/DQ nodes ===')
    n_w, n_a, n_skip = inject_qdq_nodes(
        graph, matched_pairs,
        w_only_act_scale=args.w_only_act_scale,
        verbose=args.verbose)

    print(f'\n  Summary:')
    print(f'    Weight Q/DQ pairs:     {n_w}')
    print(f'    Activation Q/DQ pairs: {n_a}  (unique activation inputs)')
    print(f'    Layers skipped (FP16): {n_skip}')
    print(f'    Total Q/DQ nodes:      {2 * n_w + 2 * n_a}')

    # ── Verify and save ──────────────────────────────────────────────────
    print(f'\n=== Step 4: Save Q/DQ ONNX -> {args.output} ===')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if not args.no_check:
        print('  Running onnx.checker.check_model ...')
        try:
            onnx.checker.check_model(model_onnx)
            print('  onnx.checker: PASS')
        except Exception as e:
            print(f'  [WARNING] onnx.checker raised: {e}')
            print('  Saving anyway -- TRT may still parse correctly.')

    onnx.save(model_onnx, args.output)
    size_mb = os.path.getsize(args.output) / 1024**2
    print(f'  Saved: {args.output}  ({size_mb:.1f} MB)')
    print(f'  Nodes now:        {len(graph.node)}')
    print(f'  Initializers now: {len(graph.initializer)}')

    # ── Next steps ───────────────────────────────────────────────────────
    print('\n=== Next steps ===')
    print('Build TRT INT8 engine (explicit quantization, no calibrator needed):')
    print(f'  python tools/build_trt_int8_univ2x.py \\')
    print(f'      --onnx {args.output} \\')
    print(f'      --out trt_engines/univ2x_ego_bev_encoder_qdq_int8.trt \\')
    print(f'      --target bev_encoder \\')
    print(f'      --plugin plugins/build/libuniv2x_plugins.so')
    print()
    print('End-to-end AMOTA validation:')
    print(f'  python tools/test_trt.py ... \\')
    print(f'      --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_qdq_int8.trt \\')
    print(f'      --eval bbox')


if __name__ == '__main__':
    main()
