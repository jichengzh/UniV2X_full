"""ONNX preprocessor: 给所有 Conv 权重加 per-tensor Q/DQ nodes.

TRT 10 Python API 不能直接 force Conv weight per-tensor quantization (默认 per-channel).
通过 ONNX QDQ explicit precision 路径: 在 ONNX 里给每个 Conv 的 weight init 接
QuantizeLinear → DequantizeLinear, 用 per-tensor scale (max(|W|)/127). Activations
仍走 TRT calibrator (per-tensor by default).

用法:
    python onnx_pt_weight_quant.py --in T1_base.onnx --out T1_base_pt_weight.onnx
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as nh


def preprocess(in_path: Path, out_path: Path, granularity: str = "per-tensor") -> dict:
    """granularity ∈ {per-tensor, per-channel}.
    per-channel: 每个 output channel 一个 scale (axis=0). 对照 TRT default 行为.
    per-tensor: 整个 weight tensor 一个 scale.
    """
    assert granularity in ("per-tensor", "per-channel")
    m = onnx.load(str(in_path))
    g = m.graph
    init_dict = {i.name: i for i in g.initializer}

    new_nodes = []
    new_inits = []
    stats = {"conv_processed": 0, "conv_skipped_no_init": 0, "scale_range": [],
             "granularity": granularity}

    for node in g.node:
        if node.op_type != "Conv":
            continue
        if len(node.input) < 2:
            stats["conv_skipped_no_init"] += 1
            continue
        w_name = node.input[1]
        if w_name not in init_dict:
            stats["conv_skipped_no_init"] += 1
            continue
        w_init = init_dict[w_name]
        w_arr = nh.to_array(w_init)

        MIN_SCALE = 1e-7  # TRT 要求 scale > 0, 小到 ~0 的 channel/tensor 用 floor 防 INVALID_NODE
        if granularity == "per-tensor":
            max_abs = float(np.abs(w_arr).max())
            scale_val = max(max_abs / 127.0, MIN_SCALE)
            scale_arr = np.array(scale_val, dtype=np.float32)
            zp_arr = np.array(0, dtype=np.int8)
            stats["scale_range"].append(scale_val)
        else:  # per-channel along axis=0 (output channels)
            per_ch_max = np.abs(w_arr).reshape(w_arr.shape[0], -1).max(axis=1)
            scale_arr = np.maximum(per_ch_max / 127.0, MIN_SCALE).astype(np.float32)
            zp_arr = np.zeros(w_arr.shape[0], dtype=np.int8)
            stats["scale_range"].append(float(scale_arr.max()))

        suffix = "pt" if granularity == "per-tensor" else "pc"
        scale_init = nh.from_array(scale_arr, name=f"{w_name}_{suffix}_scale")
        zp_init = nh.from_array(zp_arr, name=f"{w_name}_{suffix}_zp")
        new_inits.extend([scale_init, zp_init])

        q_out = f"{w_name}_q_{suffix}"
        dq_out = f"{w_name}_dq_{suffix}"
        q_kwargs = {"inputs": [w_name, scale_init.name, zp_init.name],
                    "outputs": [q_out], "name": f"{w_name}_Q_{suffix}"}
        dq_kwargs = {"inputs": [q_out, scale_init.name, zp_init.name],
                     "outputs": [dq_out], "name": f"{w_name}_DQ_{suffix}"}
        if granularity == "per-channel":
            # ONNX QuantizeLinear/DequantizeLinear axis attr (default 1 for opset >= 13)
            # For Conv weight shape (out, in, kH, kW), per-output-channel = axis 0
            q_kwargs["axis"] = 0
            dq_kwargs["axis"] = 0
        q_node = helper.make_node("QuantizeLinear", **q_kwargs)
        dq_node = helper.make_node("DequantizeLinear", **dq_kwargs)
        new_nodes.extend([q_node, dq_node])
        node.input[1] = dq_out
        stats["conv_processed"] += 1

    # 加 inits + nodes
    g.initializer.extend(new_inits)
    # 在 g.node 开头插入 Q/DQ (保证它们在 Conv 之前; ONNX 允许任意顺序但保守起见)
    # Q/DQ 仅依赖 weight init (永远先于 Conv), 所以加在前面安全
    new_node_list = list(new_nodes) + list(g.node)
    del g.node[:]
    g.node.extend(new_node_list)

    onnx.save(m, str(out_path))

    if stats["scale_range"]:
        stats["scale_min"] = min(stats["scale_range"])
        stats["scale_max"] = max(stats["scale_range"])
        stats["scale_mean"] = sum(stats["scale_range"]) / len(stats["scale_range"])
    stats.pop("scale_range")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--granularity", choices=["per-tensor", "per-channel"],
                    default="per-tensor")
    args = ap.parse_args()
    stats = preprocess(Path(args.in_path), Path(args.out),
                       granularity=args.granularity)
    print(f"in : {args.in_path}")
    print(f"out: {args.out}")
    print(f"  Conv processed: {stats['conv_processed']}")
    print(f"  Conv skipped:   {stats['conv_skipped_no_init']}")
    print(f"  weight pt-scale min/mean/max: {stats.get('scale_min', 0):.6f} / "
          f"{stats.get('scale_mean', 0):.6f} / {stats.get('scale_max', 0):.6f}")


if __name__ == "__main__":
    main()
