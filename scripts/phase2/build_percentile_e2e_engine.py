"""方案 2 — Percentile-calibrated INT8 e2e engine (TRT manual dynamic_range API).

实现思路 (替代 IInt8MinMaxCalibrator / IInt8EntropyCalibrator2):
  1. 加载 HEAL Pyramid 完整模型 (FP32)
  2. 在所有 nn.Conv2d 和 nn.ReLU 模块挂 forward hook
  3. 用 N 个真实 DAIR sample 前向, 记录每个模块输出的 absolute activation
  4. 每个模块算 p_percentile 分位数 (默认 99.99) → 作为该层的 clip
  5. 解析 e2e ONNX, 对每个 TRT 层 (Conv / Activation / Elementwise) 找匹配的
     PyTorch hook, 把输出 tensor 的 dynamic_range 设为 (-clip, +clip)
  6. 用 BuilderFlag.INT8 + BuilderFlag.FP16 build engine — 没匹配的 tensor 自动
     fallback 到 FP16 (不需要 IInt8Calibrator)

vs Entropy / MinMax:
  Entropy: KL min → 对宽分布过度激进裁尾
  MinMax:  全保留 → 步长粗 (对窄分布无影响, 对宽分布损失精度)
  Percentile (本方案): 介于两者 — 自定义裁尾比例, 不靠 KL
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import tensorrt as trt

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/phase2"))
os.chdir(str(HEAL_ROOT))

# Patch missing label files
import opencood.utils.common_utils as _cu
_orig_read_json = _cu.read_json


def _safe_read_json(p):
    if not os.path.exists(p) and ("backup" in str(p) or "label" in str(p).split("/")[-2:]):
        return []
    return _orig_read_json(p)


_cu.read_json = _safe_read_json
import opencood.data_utils.datasets.basedataset.dairv2x_basedataset as _dair_mod
_dair_mod.read_json = _safe_read_json

from opencood.hypes_yaml import yaml_utils
from opencood.utils.common_utils import update_dict
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from tools.export_onnx_pyramid import build_pyramid_from_ckpt  # noqa: E402

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------


def collect_percentile_scales(model, loader, percentile=99.99, n_samples=50):
    """Forward N samples, hook Conv2d/ReLU outputs, compute per-module percentile.

    Returns:
      scales: {pytorch_path: float}
    """
    device = next(model.parameters()).device
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)

    # Hook BatchNorm2d (post-Conv, equivalent to TRT's fused Conv+BN output) and
    # ReLU (post Conv+BN, equivalent to TRT's Conv+BN+ReLU fused output).
    # Plus Conv2d for terminal heads without BN/ReLU after.
    handles = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.ConvTranspose2d)):
            def make_hook(n):
                def hook(mod, inp, out):
                    if isinstance(out, torch.Tensor):
                        buckets[n].append(out.detach().float().abs().flatten().cpu().numpy())
                return hook
            handles.append(m.register_forward_hook(make_hook(name)))

    n_done = 0
    with torch.no_grad():
        for batch in loader:
            if n_done >= n_samples:
                break
            if batch is None:
                continue
            batch = train_utils.to_device(batch, device)
            try:
                _ = model(batch["ego"])
            except Exception as e:
                print(f"  [skip] forward fail: {e}")
                continue
            n_done += 1
            if n_done % 10 == 0:
                print(f"  collected {n_done}/{n_samples}", flush=True)

    for h in handles:
        h.remove()

    scales = {}
    for name, bucket in buckets.items():
        # Subsample if too large
        arrs = []
        for a in bucket:
            if a.size > 50000:
                idx = np.random.choice(a.size, 50000, replace=False)
                arrs.append(a[idx])
            else:
                arrs.append(a)
        allv = np.concatenate(arrs) if arrs else np.array([0.0])
        scale = float(np.percentile(allv, percentile))
        scales[name] = max(scale, 1e-6)  # floor to avoid zero scale

    print(f"  collected {len(scales)} layer scales (p{percentile})")
    return scales, n_done


# ---------------------------------------------------------------------------
# PyTorch ↔ ONNX/TRT name matching
# ---------------------------------------------------------------------------


def normalize_path(name: str) -> str:
    """Normalize ONNX/TRT layer name or PyTorch path for matching.

    "/resnet/layer0/layer0.0/conv1/Conv_output_0" → "resnet.layer0.layer0.0.conv1"
    "/backbone_m1/resnet/layer0/layer0.0/conv1/Conv" → "backbone_m1.resnet.layer0.layer0.0.conv1"
    "/cls_head/Conv" → "cls_head"
    "/single_head_0/Conv" → "single_head_0"
    "pyramid_backbone.resnet.layer0.0.conv1" → "resnet.layer0.0.conv1" (last 4 segments)
    """
    s = name.strip("/").replace("/", ".")
    # Strip op suffix
    for suf in ["_output_0", ".Conv", ".Relu", ".Conv_output_0", ".Relu_output_0"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    # PyTorch path: pyramid_backbone might be exported as no prefix or as `pb`
    # We use last 4 segments for matching to be tolerant
    return s


def match_pytorch_to_trt(pytorch_paths: list, trt_tensor_name: str,
                         prefer_relu: bool = False) -> str | None:
    """Find PyTorch path whose tail matches TRT tensor name's tail.

    For TRT Conv tensors (e.g. /conv1/Conv_output_0), prefer matching to BN/ReLU
    output at the same parent path (since TRT fuses Conv+BN+ReLU).

    Strategy: extract parent path from TRT name. Look for PyTorch hooks under
    that parent path. Prefer: BN > ReLU > Conv (TRT fuses; final fused output
    matches whichever activation comes last in PyTorch).
    """
    trt_norm = normalize_path(trt_tensor_name)
    trt_segs = trt_norm.split(".")

    # Try matching by varied prefix lengths (3,4,5 segments) of TRT name
    # against varied PyTorch normalizations (with/without pyramid_backbone prefix)
    best, best_score, best_priority = None, 0, -1
    for ppath in pytorch_paths:
        for candidate in [ppath, ppath.replace("pyramid_backbone.", "")]:
            cand_segs = candidate.split(".")
            # Score = how many segments suffix-match between cand and trt
            score = 0
            for a, b in zip(reversed(cand_segs), reversed(trt_segs)):
                if a == b:
                    score += 1
                else:
                    break

            # Sibling match: parent path of cand matches parent path of trt
            # e.g., cand="resnet.layer0.0.relu" trt_norm="resnet.layer0.0.conv1"
            # both share parent "resnet.layer0.0" → score parent match
            if len(cand_segs) > 1 and len(trt_segs) > 1:
                cand_parent = cand_segs[:-1]
                trt_parent = trt_segs[:-1]
                parent_score = 0
                for a, b in zip(reversed(cand_parent), reversed(trt_parent)):
                    if a == b:
                        parent_score += 1
                    else:
                        break
                # Bonus: parent matches and last segment is BN/ReLU (post-fusion eq)
                if parent_score >= 2:
                    last = cand_segs[-1]
                    # Priority: BN-like (norm/bn) > ReLU > Conv
                    if last in ("relu", "relu_1", "relu_2", "Relu") and not prefer_relu:
                        # Use as sibling match if better priority
                        prio = 2
                        if parent_score > best_score or \
                           (parent_score == best_score and prio > best_priority):
                            best, best_score, best_priority = ppath, parent_score, prio
                    elif "norm" in last or "bn" in last:
                        prio = 3
                        if parent_score > best_score or \
                           (parent_score == best_score and prio > best_priority):
                            best, best_score, best_priority = ppath, parent_score, prio

            # Direct match has high priority
            if score >= 2 and (score > best_score or
                              (score == best_score and best_priority < 1)):
                best, best_score, best_priority = ppath, score, 1

    return best if best_score >= 2 else None


# ---------------------------------------------------------------------------
# TRT engine build with manual dynamic_range
# ---------------------------------------------------------------------------


def build_engine(onnx_path: str, scales: dict, engine_path: str,
                 max_voxels: int = 32000, workspace_mb: int = 4096):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"ONNX parse failed: {onnx_path}")

    print(f"  network: {network.num_layers} layers, {network.num_inputs} inputs, "
          f"{network.num_outputs} outputs")

    # Set INT8 + FP16 flags
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    # Set workspace pool size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    # Set DEFAULT dynamic_range for ALL tensors (fallback for unmatched ones)
    # to avoid TRT complaining about missing scales. Use a safe default of 1.0
    # for tensors we can't find; TRT will pick FP16 fallback for these.
    pytorch_paths = list(scales.keys())

    # Track matching stats
    n_matched = 0
    n_unmatched = 0
    matched_layers = []
    # Use median of matched scales as a safer fallback than 1.0
    sorted_scales = sorted(scales.values())
    fallback_scale = sorted_scales[len(sorted_scales) // 2] if sorted_scales else 1.0
    print(f"  fallback scale for unmatched (median of matched): {fallback_scale:.3f}")

    # Walk network layers and set dynamic_range on outputs of Conv/Activation/Elementwise
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type not in (trt.LayerType.CONVOLUTION,
                              trt.LayerType.ACTIVATION,
                              trt.LayerType.ELEMENTWISE):
            continue
        for j in range(layer.num_outputs):
            out_t = layer.get_output(j)
            ppath = match_pytorch_to_trt(pytorch_paths, out_t.name)
            if ppath is not None:
                scale = scales[ppath]
                out_t.dynamic_range = (-scale, scale)
                n_matched += 1
                if len(matched_layers) < 5:
                    matched_layers.append((layer.name[:60], ppath, scale))
            else:
                # No match → use median of matched scales (better than 1.0 default)
                out_t.dynamic_range = (-fallback_scale, fallback_scale)
                n_unmatched += 1

    print(f"  matched {n_matched} tensors, {n_unmatched} fallback to default")
    for nm, pp, sc in matched_layers:
        print(f"    e.g. {nm:60s} ← {pp[:40]:40s} scale={sc:.3f}")

    # Set network input/output dtypes
    for i in range(network.num_inputs):
        t = network.get_input(i)
        # Keep input dtype as exported (FP32 / INT32)
        # Set default dynamic_range for FP inputs
        if t.dtype == trt.float32 and not hasattr(t, "_dr_set"):
            # Heuristic: input range is small (voxel_features ~ 100, voxel_mask 0-1, t_ego ~ 1)
            t.dynamic_range = (-200.0, 200.0)

    # Build
    print(f"  building (INT8 + FP16 fallback, workspace={workspace_mb}MB)...", flush=True)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT build failed")
    build_secs = time.time() - t0
    print(f"  built in {build_secs:.1f}s", flush=True)

    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(bytes(serialized))
    size_mb = Path(engine_path).stat().st_size / 1e6
    print(f"  saved {engine_path}  ({size_mb:.2f} MB)")
    return build_secs, size_mb, n_matched, n_unmatched


# ---------------------------------------------------------------------------
# Dataset loader (same as e2e_bench)
# ---------------------------------------------------------------------------


def load_dair_loader(ckpt_dir, dair_root, batch_size=1):
    class _Opt:
        model_dir = ckpt_dir

    hypes = yaml_utils.load_yaml(None, _Opt())
    new_range = [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5]
    hypes = update_dict(hypes, {
        "cav_lidar_range": new_range, "lidar_range": new_range, "gt_range": new_range
    })
    val_split = f"{dair_root}/val.json"
    hypes["data_dir"] = dair_root
    hypes["test_dir"] = val_split
    hypes["validate_dir"] = val_split
    hypes["root_dir"] = val_split

    import importlib
    parser_func = getattr(importlib.import_module("opencood.hypes_yaml.yaml_utils"),
                          hypes["yaml_parser"])
    hypes = parser_func(hypes)
    hypes["data_dir"] = dair_root
    hypes["test_dir"] = val_split
    hypes["validate_dir"] = val_split
    hypes["root_dir"] = val_split

    ds = build_dataset(hypes, visualize=False, train=False)
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch_size, num_workers=2,
                      collate_fn=ds.collate_batch_test, shuffle=False, pin_memory=False), hypes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hypes", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--dair-root",
                    default="/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure")
    ap.add_argument("--percentile", type=float, default=99.99,
                    help="Percentile for activation clipping (default 99.99)")
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--max-voxels", type=int, default=32000)
    ap.add_argument("--workspace-mb", type=int, default=4096)
    ap.add_argument("--scales-out", default=None,
                    help="optional JSON to save layer scales")
    args = ap.parse_args()

    # Phase 1: Build PyTorch model + collect activations
    print(f"[1/3] Load HEAL Pyramid: {args.ckpt}")
    model = build_pyramid_from_ckpt(args.hypes, args.ckpt, device="cuda")
    model.eval()
    ckpt_dir = str(Path(args.ckpt).parent)
    loader, hypes = load_dair_loader(ckpt_dir, args.dair_root, batch_size=1)

    print(f"[2/3] Collect activation percentiles (p{args.percentile}, "
          f"n={args.n_samples} samples)")
    scales, n_done = collect_percentile_scales(model, loader,
                                               percentile=args.percentile,
                                               n_samples=args.n_samples)
    if args.scales_out:
        Path(args.scales_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.scales_out).write_text(json.dumps(scales, indent=2))
        print(f"  scales saved → {args.scales_out}")

    # Print key layer scales
    print("\n  key layer scales:")
    for key in ("shrink_conv", "cls_head", "reg_head", "dir_head"):
        for k, v in scales.items():
            if k.endswith(key):
                print(f"    {k:50s} p99.99 = {v:.3f}")
                break

    # Free PyTorch model memory before TRT build
    del model
    torch.cuda.empty_cache()

    # Phase 2: Build engine
    print(f"[3/3] Build percentile INT8 engine → {args.engine}")
    build_secs, size_mb, n_matched, n_unmatched = build_engine(
        args.onnx, scales, args.engine, args.max_voxels, args.workspace_mb
    )

    # Save report
    report_path = Path(args.engine).with_suffix(".build.json")
    report_path.write_text(json.dumps({
        "onnx": args.onnx,
        "engine": args.engine,
        "ckpt": args.ckpt,
        "percentile": args.percentile,
        "n_calib_samples": n_done,
        "n_layer_scales": len(scales),
        "n_matched_tensors": n_matched,
        "n_unmatched_tensors": n_unmatched,
        "build_secs": build_secs,
        "engine_size_mb": size_mb,
        "key_scales": {k: scales[k] for k in scales if any(
            k.endswith(x) for x in ("shrink_conv", "cls_head", "reg_head", "dir_head"))},
    }, indent=2))
    print(f"\n  report → {report_path}")


if __name__ == "__main__":
    main()
