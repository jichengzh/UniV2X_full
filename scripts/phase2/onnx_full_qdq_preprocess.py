"""完整 W+A Q/DQ ONNX 预处理 — 解决 path B 调查的 23pp 损失.

根因 (经 phase2 调查确认):
  TRT 10 走 INT8 IMMA 要求 weight 和 activation **都** 有 Q/DQ. 单 weight Q/DQ →
  TRT 当作 "noisy weight in FP16" 退化 FP16 compute → 等价 W-only → -23pp AP.

修复: 给 ONNX 每个 Conv input activation 也加 Q/DQ. Activation per-tensor scale 复用
TRT calibrator cache 文件的 scale (= 与 TRT default Q_int8_mm 完全一致的 activation 量化).

每个 Conv 收到 4 个 Q/DQ:
  weight  → Q (per-tensor or per-channel scale) → DQ → Conv input[1]
  activation → Q (per-tensor scale from cache) → DQ → Conv input[0]

支持 granularity (per-tensor/per-channel) × object (W+A/W-only) = 4 变体:
  Q_int8_pc_wa: per-channel weight + per-tensor activation Q/DQ → 期望 ≈ 0.78 (sanity)
  Q_int8_pt_wa: per-tensor weight + per-tensor activation Q/DQ
  Q_int8_pc_wo: per-channel weight Q/DQ, NO activation Q/DQ → 等价 --w-only flag
  Q_int8_pt_wo: per-tensor weight Q/DQ, NO activation Q/DQ
"""
from __future__ import annotations
import argparse
import struct
from pathlib import Path
import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as nh


def parse_calib_cache(cache_path: Path) -> dict[str, float]:
    """Parse TRT calib cache: 每行 'name: hex_float32_big_endian'."""
    scales = {}
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("TRT-"):
                continue
            if ":" not in line:
                continue
            name, hex_val = line.split(":", 1)
            hex_val = hex_val.strip()
            try:
                # TRT cache 用 big-endian hex 存 IEEE 754 float32
                scale = struct.unpack(">f", bytes.fromhex(hex_val))[0]
                scales[name.strip()] = scale
            except Exception:
                continue
    return scales


def preprocess(in_path: Path, out_path: Path,
               w_granularity: str = "per-tensor",
               add_activation_qdq: bool = True,
               calib_cache: Path = None,
               skip_conv_substrs: list = None) -> dict:
    """granularity ∈ {per-tensor, per-channel}; add_activation_qdq False ⇒ W-only mode."""
    assert w_granularity in ("per-tensor", "per-channel")
    m = onnx.load(str(in_path))
    g = m.graph
    init_dict = {i.name: i for i in g.initializer}

    # Parse calib cache for activation scales (only needed if add_activation_qdq)
    act_scales = {}
    if add_activation_qdq:
        if calib_cache is None or not calib_cache.exists():
            raise ValueError(f"add_activation_qdq=True 但 calib_cache 缺/不存在: {calib_cache}")
        act_scales = parse_calib_cache(calib_cache)
        print(f"[preproc] loaded {len(act_scales)} activation scales from {calib_cache.name}")

    new_nodes = []
    new_inits = []
    stats = {"conv_processed": 0, "conv_skipped_no_init": 0,
             "conv_skipped_excluded": 0,
             "act_qdq_added": 0, "act_qdq_missing_scale": 0,
             "w_granularity": w_granularity,
             "add_activation_qdq": add_activation_qdq,
             "skip_conv_substrs": skip_conv_substrs or []}
    skip_substrs = skip_conv_substrs or []

    MIN_SCALE = 1e-7

    for node in g.node:
        if node.op_type != "Conv":
            continue
        if len(node.input) < 2 or node.input[1] not in init_dict:
            stats["conv_skipped_no_init"] += 1
            continue
        if any(s in node.name for s in skip_substrs):
            stats["conv_skipped_excluded"] += 1
            continue
        w_name = node.input[1]
        w_init = init_dict[w_name]
        w_arr = nh.to_array(w_init)

        # === Weight Q/DQ ===
        if w_granularity == "per-tensor":
            scale_val = max(float(np.abs(w_arr).max()) / 127.0, MIN_SCALE)
            w_scale_arr = np.array(scale_val, dtype=np.float32)
            w_zp_arr = np.array(0, dtype=np.int8)
            w_axis = None
        else:
            per_ch_max = np.abs(w_arr).reshape(w_arr.shape[0], -1).max(axis=1)
            w_scale_arr = np.maximum(per_ch_max / 127.0, MIN_SCALE).astype(np.float32)
            w_zp_arr = np.zeros(w_arr.shape[0], dtype=np.int8)
            w_axis = 0

        w_suffix = "pt" if w_granularity == "per-tensor" else "pc"
        w_scale_init = nh.from_array(w_scale_arr, name=f"{w_name}_{w_suffix}_scale")
        w_zp_init = nh.from_array(w_zp_arr, name=f"{w_name}_{w_suffix}_zp")
        new_inits.extend([w_scale_init, w_zp_init])

        w_q_out = f"{w_name}_q_{w_suffix}"
        w_dq_out = f"{w_name}_dq_{w_suffix}"
        w_q_kw = {"inputs": [w_name, w_scale_init.name, w_zp_init.name],
                  "outputs": [w_q_out], "name": f"{w_name}_Q_{w_suffix}"}
        w_dq_kw = {"inputs": [w_q_out, w_scale_init.name, w_zp_init.name],
                   "outputs": [w_dq_out], "name": f"{w_name}_DQ_{w_suffix}"}
        if w_axis is not None:
            w_q_kw["axis"] = w_axis
            w_dq_kw["axis"] = w_axis
        new_nodes.extend([helper.make_node("QuantizeLinear", **w_q_kw),
                          helper.make_node("DequantizeLinear", **w_dq_kw)])
        node.input[1] = w_dq_out

        # === Activation Q/DQ (optional) ===
        if add_activation_qdq:
            act_name = node.input[0]
            if act_name in act_scales:
                act_scale = max(act_scales[act_name], MIN_SCALE)
                a_scale_arr = np.array(act_scale, dtype=np.float32)
                a_zp_arr = np.array(0, dtype=np.int8)
                a_scale_init = nh.from_array(a_scale_arr, name=f"{act_name}_a_scale_{stats['conv_processed']}")
                a_zp_init = nh.from_array(a_zp_arr, name=f"{act_name}_a_zp_{stats['conv_processed']}")
                new_inits.extend([a_scale_init, a_zp_init])
                a_q_out = f"{act_name}_aq_{stats['conv_processed']}"
                a_dq_out = f"{act_name}_adq_{stats['conv_processed']}"
                new_nodes.append(helper.make_node(
                    "QuantizeLinear",
                    inputs=[act_name, a_scale_init.name, a_zp_init.name],
                    outputs=[a_q_out],
                    name=f"{act_name}_AQ_{stats['conv_processed']}"))
                new_nodes.append(helper.make_node(
                    "DequantizeLinear",
                    inputs=[a_q_out, a_scale_init.name, a_zp_init.name],
                    outputs=[a_dq_out],
                    name=f"{act_name}_ADQ_{stats['conv_processed']}"))
                node.input[0] = a_dq_out
                stats["act_qdq_added"] += 1
            else:
                stats["act_qdq_missing_scale"] += 1

        stats["conv_processed"] += 1

    g.initializer.extend(new_inits)
    new_node_list = list(new_nodes) + list(g.node)
    del g.node[:]
    g.node.extend(new_node_list)
    onnx.save(m, str(out_path))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-granularity", choices=["per-tensor", "per-channel"],
                    default="per-channel")
    ap.add_argument("--w-only", action="store_true",
                    help="不加 activation Q/DQ (W-only mode)")
    ap.add_argument("--calib-cache",
                    help="TRT calibrator cache file (含 activation scales). "
                         "W-only mode 时可省")
    ap.add_argument("--skip-conv-substr", action="append", default=[],
                    help="跳过名字含此 substring 的 Conv (例如 'cls_head' 保留 head 不量化)")
    args = ap.parse_args()
    stats = preprocess(Path(args.in_path), Path(args.out),
                       w_granularity=args.w_granularity,
                       add_activation_qdq=(not args.w_only),
                       calib_cache=Path(args.calib_cache) if args.calib_cache else None,
                       skip_conv_substrs=args.skip_conv_substr)
    print(f"in : {args.in_path}")
    print(f"out: {args.out}")
    print(f"  config: w_gran={args.w_granularity}, add_act_qdq={not args.w_only}")
    print(f"  Conv processed: {stats['conv_processed']}, skipped: {stats['conv_skipped_no_init']}")
    if not args.w_only:
        print(f"  activation Q/DQ added: {stats['act_qdq_added']}, "
              f"missing scale: {stats['act_qdq_missing_scale']}")


if __name__ == "__main__":
    main()
