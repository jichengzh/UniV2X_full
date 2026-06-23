"""P0.1 — validate CUDA nms_rotated_gpu against HEAL Shapely nms_rotated.

Three passes over OPV2V test (real inference path, intermediate fusion):
  1. compare : run BOTH NMS on identical pre-NMS boxes; record keep-set agreement
               + per-call latency of each. Pipeline continues with Shapely result
               so this pass is authentic.
  2. shapely : full e2e wall-clock with stock Shapely NMS.
  3. cuda    : full e2e wall-clock with CUDA NMS patched in.

Outputs: correctness (exact keep-set match rate + Jaccard), NMS speedup, and the
real e2e speedup (the number that matters for the paper claim ~3.6x).

Run from HEAL_ROOT cwd, pin an idle GPU:
  cd /home/jichengzhi/heal_research/HEAL && CUDA_VISIBLE_DEVICES=7 python \
    /home/jichengzhi/UniV2X/scripts/phase2/p0_1_nms_validate.py --measure 150
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"
sys.path.insert(0, HEAL_ROOT)
sys.path.insert(0, "/home/jichengzhi/UniV2X/scripts/phase2")

import opencood.hypes_yaml.yaml_utils as yaml_utils  # noqa: E402
from opencood.tools import train_utils, inference_utils  # noqa: E402
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.utils.common_utils import update_dict  # noqa: E402
from opencood.utils import box_utils  # noqa: E402

from p0_1_nms_cuda import nms_rotated_gpu  # noqa: E402

torch.multiprocessing.set_sharing_strategy("file_system")

# capture buffers for the compare pass
_CAP = {k: [] for k in
        ["n_in", "shapely_ms", "cuda_ms", "n_keep_shapely", "n_keep_cuda",
         "exact_match", "jaccard"]}


def make_compare_nms(orig):
    """Wrapper that runs both NMS variants, records agreement, returns Shapely."""
    def _nms(boxes, scores, threshold):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        keep_s = orig(boxes, scores, threshold)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        keep_c = nms_rotated_gpu(boxes, scores, threshold)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        ss = set(int(x) for x in np.asarray(keep_s).tolist())
        cc = set(int(x) for x in np.asarray(keep_c).tolist())
        union = len(ss | cc)
        _CAP["n_in"].append(int(boxes.shape[0]))
        _CAP["shapely_ms"].append((t1 - t0) * 1000)
        _CAP["cuda_ms"].append((t3 - t2) * 1000)
        _CAP["n_keep_shapely"].append(len(ss))
        _CAP["n_keep_cuda"].append(len(cc))
        _CAP["exact_match"].append(1.0 if ss == cc else 0.0)
        _CAP["jaccard"].append(len(ss & cc) / union if union else 1.0)
        return keep_s
    return _nms


def boot(ckpt):
    class _Opt:
        model_dir = ckpt

    hypes = yaml_utils.load_yaml(None, _Opt())
    x_min, x_max = -102.4, 102.4
    rng = hypes["postprocess"]["anchor_args"]["cav_lidar_range"]
    new_range = [x_min, x_min, rng[2], x_max, x_max, rng[5]]
    hypes = update_dict(hypes, {"cav_lidar_range": new_range,
                                "lidar_range": new_range, "gt_range": new_range})
    import importlib
    lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
    hypes = getattr(lib, hypes["yaml_parser"])(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    print("[boot] building model", flush=True)
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(ckpt, model)
    model.cuda().eval()

    print("[boot] building dataset", flush=True)
    ds = build_dataset(hypes, visualize=False, train=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, num_workers=2,
                        collate_fn=ds.collate_batch_test, shuffle=False,
                        pin_memory=False)
    return model, ds, loader


def run_e2e_pass(model, ds, loader, device, warmup, measure, label):
    """Time full inference_intermediate_fusion wall-clock per sample."""
    e2e_ms = []
    print(f"[{label}] e2e pass: warmup={warmup} measure={measure}", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            if i >= warmup + measure:
                break
            batch = train_utils.to_device(batch, device)
            torch.cuda.synchronize()
            t0 = time.time()
            _ = inference_utils.inference_intermediate_fusion(batch, model, ds)
            torch.cuda.synchronize()
            if i >= warmup:
                e2e_ms.append((time.time() - t0) * 1000)
    return e2e_ms


def stats(arr):
    if not arr:
        return None
    a = np.asarray(arr, dtype=float)
    return {"n": len(a), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p99": float(np.percentile(a, 99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/jichengzhi/heal_research/checkpoints/"
                    "stage1/Pyramid_m1_base_2023_08_14_04_28_12")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=150)
    ap.add_argument("--out", default="/home/jichengzhi/UniV2X/paper_learning/"
                    "2. AAAI最终故事/data/p0_1_nms_validation.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    model, ds, loader = boot(args.ckpt)

    # ── pass 1: correctness + NMS-level latency (both variants on same input) ──
    original_nms = box_utils.nms_rotated
    box_utils.nms_rotated = make_compare_nms(original_nms)
    print("[pass1] compare Shapely vs CUDA on identical pre-NMS boxes", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            if i >= args.warmup + args.measure:
                break
            batch = train_utils.to_device(batch, device)
            _ = inference_utils.inference_intermediate_fusion(batch, model, ds)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}]", flush=True)
    box_utils.nms_rotated = original_nms  # restore

    # ── pass 2: e2e with stock Shapely ──
    e2e_shapely = run_e2e_pass(model, ds, loader, device, args.warmup,
                               args.measure, "shapely")

    # ── pass 3: e2e with CUDA NMS ──
    box_utils.nms_rotated = nms_rotated_gpu
    e2e_cuda = run_e2e_pass(model, ds, loader, device, args.warmup,
                            args.measure, "cuda")
    box_utils.nms_rotated = original_nms

    nms_shapely = stats(_CAP["shapely_ms"])
    nms_cuda = stats(_CAP["cuda_ms"])
    e2e_s = stats(e2e_shapely)
    e2e_c = stats(e2e_cuda)
    exact_rate = float(np.mean(_CAP["exact_match"])) if _CAP["exact_match"] else None
    jaccard = float(np.mean(_CAP["jaccard"])) if _CAP["jaccard"] else None

    report = {
        "ckpt": args.ckpt,
        "device": torch.cuda.get_device_name(0),
        "n_nms_calls": len(_CAP["shapely_ms"]),
        "correctness": {
            "exact_keepset_match_rate": exact_rate,
            "mean_jaccard": jaccard,
            "mean_n_in": float(np.mean(_CAP["n_in"])) if _CAP["n_in"] else None,
            "mean_keep_shapely": float(np.mean(_CAP["n_keep_shapely"])) if _CAP["n_keep_shapely"] else None,
            "mean_keep_cuda": float(np.mean(_CAP["n_keep_cuda"])) if _CAP["n_keep_cuda"] else None,
        },
        "nms_latency_ms": {"shapely": nms_shapely, "cuda": nms_cuda,
                           "speedup_mean": (nms_shapely["mean"] / nms_cuda["mean"])
                           if nms_cuda and nms_cuda["mean"] else None},
        "e2e_latency_ms": {"shapely": e2e_s, "cuda": e2e_c,
                           "speedup_mean": (e2e_s["mean"] / e2e_c["mean"])
                           if e2e_c and e2e_c["mean"] else None},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n==================== P0.1 NMS validation ====================")
    print(f"correctness: exact keep-set match = {exact_rate*100:.1f}%  "
          f"mean Jaccard = {jaccard:.4f}  (n_nms_calls={len(_CAP['shapely_ms'])})")
    print(f"  mean #in={report['correctness']['mean_n_in']:.1f}  "
          f"keep shapely={report['correctness']['mean_keep_shapely']:.1f}  "
          f"cuda={report['correctness']['mean_keep_cuda']:.1f}")
    print(f"NMS  : shapely {nms_shapely['mean']:.2f}ms  cuda {nms_cuda['mean']:.3f}ms"
          f"  -> {report['nms_latency_ms']['speedup_mean']:.1f}x")
    print(f"e2e  : shapely {e2e_s['mean']:.2f}ms  cuda {e2e_c['mean']:.2f}ms"
          f"  -> {report['e2e_latency_ms']['speedup_mean']:.2f}x")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
