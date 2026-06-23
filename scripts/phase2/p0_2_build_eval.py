"""P0.2 — build multi-output TRT engines (FP16 / INT8-real-calib) and eval AP.

Closes the plan v5 Phase C blocker. The TRT engine replaces ONLY
get_multiscale_feature (3 stage features); the V2X collab fusion + decode stay in
PyTorch, so AP is preserved. INT8 uses a REAL MinMax calibrator fed with the
anchor's own harvested spatial_features (Q1), not a stale reused cache.

Pipeline:
  ONNX (p0_2_multiscale_trt.py)  +  calib .npy (p0_2_calibrate.py)
    -> build FP16 / INT8 engines (3 outputs)
    -> patch get_multiscale_feature with TRTBackbone3 (per-agent batch loop)
    -> AP30/50/70 on DAIR val subset, compare fp32 / fp16 / int8

Run (idle GPU):
  cd /home/jichengzhi/heal_research/HEAL && CUDA_VISIBLE_DEVICES=7 python \
    /home/jichengzhi/UniV2X/scripts/phase2/p0_2_build_eval.py --n-samples 300
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tensorrt as trt

REPO = Path("/home/jichengzhi/UniV2X")
DATA_DIR = REPO / "paper_learning" / "2. AAAI最终故事" / "data"
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
sys.path.insert(0, str(REPO / "scripts" / "phase2"))

from p0_2_multiscale_trt import (  # noqa: E402
    INPUT_NAME, INPUT_SHAPE, ONNX_DIR, ENGINE_DIR, P64_BASELINE,
    boot_pyramid, export_multiscale_onnx,
)

CALIB_DIR = Path("/tmp/plan6_p0_2_calib")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ─────────────────────────── INT8 real calibrator (Q1) ───────────────────────────


class RealMinMaxCalibrator(trt.IInt8MinMaxCalibrator):
    """Feeds real harvested spatial_features. Caches scales to disk."""

    def __init__(self, calib_npy: Path, cache_path: Path, batch_size: int = 1):
        super().__init__()
        self.data = np.load(calib_npy)  # (M, C, H, W) float32
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.idx = 0
        self._dev = None  # keep a ref so the GPU buffer isn't freed mid-calibration

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.idx + self.batch_size > self.data.shape[0]:
            return None
        batch = np.ascontiguousarray(self.data[self.idx:self.idx + self.batch_size])
        self.idx += self.batch_size
        self._dev = torch.from_numpy(batch).cuda()
        return [int(self._dev.data_ptr())]

    def read_calibration_cache(self):
        if self.cache_path.exists() and self.cache_path.stat().st_size > 0:
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_path.write_bytes(cache)


def build_engine(onnx_path: Path, engine_path: Path, q: str,
                 calib_npy: Optional[Path] = None,
                 calib_cache: Optional[Path] = None) -> dict:
    builder = trt.Builder(TRT_LOGGER)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            return {"ok": False, "error": "parse failed",
                    "details": [str(parser.get_error(i)) for i in range(parser.num_errors)]}

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 * (1 << 20))
    calib_ref = None
    if q == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif q == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # allow FP16 for non-quantized layers
        calib_ref = RealMinMaxCalibrator(calib_npy, calib_cache)
        config.int8_calibrator = calib_ref

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    secs = time.time() - t0
    if serialized is None:
        return {"ok": False, "error": "build failed", "build_secs": round(secs, 1)}
    engine_path.write_bytes(bytes(serialized))
    return {"ok": True, "engine_mb": round(engine_path.stat().st_size / 1e6, 2),
            "build_secs": round(secs, 1)}


# ─────────────────────────── 3-output TRT backbone ───────────────────────────


class TRTBackbone3:
    """Wraps a 3-output TRT engine as a replacement for get_multiscale_feature.

    Handles variable agent count by looping the fixed batch=1 engine per agent.
    Returns a tuple of 3 stage features (batched over agents), matching
    base_bev_backbone_resnet.get_multiscale_feature.
    """

    def __init__(self, engine_path: Path):
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.in_name = None
        self.out_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_name = name
            else:
                self.out_names.append(name)
        # output names from ONNX may be unordered; sort by spatial size desc (feat0 largest)
        shapes = {n: tuple(self.context.get_tensor_shape(n)) for n in self.out_names}
        self.out_names.sort(key=lambda n: -(shapes[n][2] * shapes[n][3]))
        self.out_shapes = [tuple(self.context.get_tensor_shape(n)) for n in self.out_names]
        self.d_outs = [torch.empty(s, dtype=torch.float32, device="cuda") for s in self.out_shapes]

    def __call__(self, x: torch.Tensor):
        n_agents = x.shape[0]
        collected = [[] for _ in self.out_names]
        for a in range(n_agents):
            xi = x[a:a + 1].contiguous()
            self.context.set_tensor_address(self.in_name, xi.data_ptr())
            for j, name in enumerate(self.out_names):
                self.context.set_tensor_address(name, self.d_outs[j].data_ptr())
            self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            for j in range(len(self.out_names)):
                collected[j].append(self.d_outs[j].clone())
        return tuple(torch.cat(collected[j], dim=0) for j in range(len(self.out_names)))


# ─────────────────────────── AP eval ───────────────────────────


def eval_ap(tag: str, model_dir: Path, num_filters: List[int], ckpt_name: str,
            q: str, engine_path: Optional[Path], n_samples: int) -> dict:
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils, inference_utils
    from opencood.utils import eval_utils
    from opencood.data_utils.datasets import build_dataset
    from torch.utils.data import DataLoader

    cfg = yaml_utils.load_yaml(str(model_dir / "config.yaml"))
    cfg["model"]["args"]["fusion_backbone"]["num_filters"] = num_filters
    cfg["validate_dir"] = cfg.get("validate_dir") or cfg.get("data_dir")

    val_ds = build_dataset(cfg, visualize=False, train=False)
    n_use = min(n_samples, len(val_ds))
    model = train_utils.create_model(cfg)
    sd = torch.load(model_dir / ckpt_name, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    device = torch.device("cuda")
    model.to(device).eval()

    if q != "fp32":
        trt_bb = TRTBackbone3(engine_path)
        model.pyramid_backbone.get_multiscale_feature = trt_bb
        print(f"  [{tag}|{q}] patched get_multiscale_feature with 3-output TRT engine")

    result_stat = {iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for iou in (0.3, 0.5, 0.7)}
    loader = DataLoader([val_ds[i] for i in range(n_use)], batch_size=1, num_workers=0,
                        collate_fn=val_ds.collate_batch_test, shuffle=False)
    n_ok = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        if batch is None:
            continue
        try:
            with torch.no_grad():
                batch = train_utils.to_device(batch, device)
                r = inference_utils.inference_intermediate_fusion(batch, model, val_ds)
                for iou in (0.3, 0.5, 0.7):
                    eval_utils.caluclate_tp_fp(r['pred_box_tensor'], r['pred_score'],
                                               r['gt_box_tensor'], result_stat, iou)
            n_ok += 1
        except Exception as e:
            if i < 3:
                print(f"    sample {i} failed: {type(e).__name__}: {str(e)[:120]}")
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_use} ({time.time()-t0:.0f}s)", flush=True)

    try:
        ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(model_dir))
    except Exception as e:
        print(f"  eval_final_results failed: {e}")
        ap30 = ap50 = ap70 = float("nan")
    return {"tag": tag, "q": q, "num_filters": str(num_filters),
            "n_ok": n_ok, "n_use": n_use,
            "ap30": round(float(ap30), 4), "ap50": round(float(ap50), 4),
            "ap70": round(float(ap70), 4), "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--q", nargs="*", default=["fp32", "fp16", "int8"])
    args = ap.parse_args()

    tag, model_dir, num_filters, ckpt_name = P64_BASELINE
    onnx_path = ONNX_DIR / f"{tag}_multiscale.onnx"
    calib_npy = CALIB_DIR / f"{tag}_calib.npy"
    assert onnx_path.exists(), f"missing onnx {onnx_path} (run p0_2_multiscale_trt.py)"

    engines = {}
    for q in args.q:
        if q == "fp32":
            continue
        eng = ENGINE_DIR / f"{tag}_{q}.engine"
        if not eng.exists():
            print(f"[build] {q} engine ...", flush=True)
            info = build_engine(onnx_path, eng, q, calib_npy,
                                CALIB_DIR / f"{tag}_{q}.cache")
            print(f"  -> {info}", flush=True)
            if not info["ok"]:
                continue
        engines[q] = eng

    rows = []
    for q in args.q:
        print(f"\n=== eval {tag} | {q} ===", flush=True)
        row = eval_ap(tag, model_dir, num_filters, ckpt_name, q,
                      engines.get(q), args.n_samples)
        print(f"  -> {row}", flush=True)
        rows.append(row)

    out_csv = DATA_DIR / "p0_2_multiscale_ap.csv"
    fields = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    print("\n==================== P0.2 multi-output AP ====================")
    print(f"{'Q':8s} {'AP30':>8s} {'AP50':>8s} {'AP70':>8s}  n_ok")
    fp32 = next((r for r in rows if r["q"] == "fp32"), None)
    for r in rows:
        gap = ""
        if fp32 and r["q"] != "fp32":
            gap = f"  (ΔAP50 vs fp32 = {r['ap50']-fp32['ap50']:+.4f})"
        print(f"{r['q']:8s} {r['ap30']:8.4f} {r['ap50']:8.4f} {r['ap70']:8.4f}  "
              f"{r['n_ok']}/{r['n_use']}{gap}")
    print(f"[done] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
