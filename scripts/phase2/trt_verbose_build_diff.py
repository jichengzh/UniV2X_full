"""TRT verbose build for kernel selection diff.

Build 2 engines (Q_int8_mm vs pc_wa_full) with trt.Logger.VERBOSE, dump per-layer
kernel + precision info. Find layers where two paths picked different kernels/precision.

Usage:
    python trt_verbose_build_diff.py \
        --onnx-a base.onnx --onnx-b pc_wa.onnx \
        --log-a /tmp/build_a.log --log-b /tmp/build_b.log
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt


def build(onnx_path: str, engine_path: str, log_path: str,
          calib_data: dict, calib_cache: str,
          workspace_mb: int = 4096) -> int:
    """Build INT8 engine with VERBOSE log captured."""
    # Redirect TRT log to file by writing logger callbacks
    log_buf = []
    class FileLogger(trt.ILogger):
        def __init__(self):
            super().__init__()
            self.lines = []
        def log(self, severity, msg):
            self.lines.append(f"[{severity}] {msg}")
    log = FileLogger()
    log_inner = trt.Logger(trt.Logger.VERBOSE)
    # Use the trt.Logger directly — TRT will print to stderr; we redirect via subprocess

    # Build by invoking m4_8_trt_build_bench.py via subprocess so we can capture stderr
    # Simpler: just import build_engine, but TRT logger goes to stderr from C++ side
    import subprocess
    REPO = Path("/home/jichengzhi/UniV2X")
    cmd = ["/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python",
           "-c",
           f"""
import sys, os
sys.path.insert(0, '{REPO}/scripts/phase1')
# Monkey-patch logger to VERBOSE
import m4_8_trt_build_bench as mod
import tensorrt as trt
mod.TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
sys.argv = ['build',
    '--onnx', {onnx_path!r},
    '--precision', 'int8',
    '--engine', {engine_path!r},
    '--report', {engine_path + '.json'!r},
    '--workspace-mb', '{workspace_mb}',
    '--skip-bench',
    '--calibrator', 'minmax',
    '--calib-cache', {calib_cache!r},
""" + "".join(f"    '--calib-multi', '{k}:{v}',\n" for k, v in calib_data.items()) +
f"""]
mod.main()
"""]
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, stderr=subprocess.STDOUT, stdout=f, timeout=600)
    return r.returncode


def parse_layer_precision(log_path: str) -> dict:
    """Extract per-layer kernel + precision info from TRT VERBOSE log."""
    import re
    layers = {}
    pat_layer = re.compile(r"(--*?Layer:|Layer\((\w+)\):|\[VERBOSE\].*?Reformatting.*)")
    with open(log_path) as f:
        cur_layer = None
        for line in f:
            ll = line.strip()
            # Look for "Best tactic for layer: ..." style or "Layer ... precision: INT8"
            if "Best tactic" in ll or "Selected" in ll or "precision" in ll.lower():
                pass  # collect
    # Simpler: just grep for key phrases
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-a", required=True)
    ap.add_argument("--onnx-b", required=True)
    ap.add_argument("--log-a", required=True)
    ap.add_argument("--log-b", required=True)
    ap.add_argument("--engine-a", required=True)
    ap.add_argument("--engine-b", required=True)
    ap.add_argument("--calib-cache-a", required=True)
    ap.add_argument("--calib-cache-b", required=True)
    ap.add_argument("--calib-dir", required=True)
    args = ap.parse_args()

    calib_dir = Path(args.calib_dir)
    calib_data = {
        "voxel_features": f"{calib_dir}/voxel_features.npy",
        "voxel_num_points": f"{calib_dir}/voxel_num_points.npy",
        "voxel_coords": f"{calib_dir}/voxel_coords.npy",
        "voxel_mask": f"{calib_dir}/voxel_mask.npy",
        "t_ego": f"{calib_dir}/t_ego.npy",
    }
    print(f"Building A: {args.onnx_a}")
    rc_a = build(args.onnx_a, args.engine_a, args.log_a, calib_data, args.calib_cache_a)
    print(f"Building B: {args.onnx_b}")
    rc_b = build(args.onnx_b, args.engine_b, args.log_b, calib_data, args.calib_cache_b)
    print(f"\nrc_a={rc_a} log: {args.log_a}")
    print(f"rc_b={rc_b} log: {args.log_b}")


if __name__ == "__main__":
    main()
