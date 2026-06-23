"""Plan v5 Phase C — Hybrid TRT real INT8 AP eval.

For each Phase A finetuned ckpt (5 plane × 3 Q variant = 15 anchor):
  1. Load HEAL model (PyTorch)
  2. Replace model.pyramid_backbone forward with TRT engine forward
  3. Run inference on DAIR-V2X val subset (default n=200 for speed)
  4. Compute AP30/50/70 via HEAL eval_utils
  5. Compare TRT real AP with PyTorch FP32 baseline AP per anchor

Run:
    python scripts/phase2/plan5_phaseC_hybrid_eval.py --bench-gpu 2 [--n-samples 200]
Output:
    paper_learning/2. AAAI最终故事/data/plan5_phaseC_real_ap.csv
    paper_learning/2. AAAI最终故事/data/plan5_phaseC_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tensorrt as trt

warnings.filterwarnings("ignore")

REPO = Path("/home/jichengzhi/UniV2X")
DATA_DIR = REPO / "paper_learning" / "2. AAAI最终故事" / "data"
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
PHASE_A_ROOT = Path("/tmp/plan5_phaseA_finetune")
ENGINE_DIR = Path("/tmp/plan5_phaseA_engines")

sys.path.insert(0, str(HEAL_ROOT))

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

ANCHORS = [
    ("p64_baseline",
     "/home/jichengzhi/heal_research/HEAL/opencood/logs/Pyramid_DAIR_m1_base_g8_2026_05_21_22_07_45",
     "net_epoch_bestval_at19.pth", [64, 128, 256]),
    ("p48", "/tmp/plan5_phaseA_finetune/p48", "net_epoch_bestval_at27.pth", [48, 96, 192]),
    ("p32", "/tmp/plan5_phaseA_finetune/p32", "net_epoch_bestval_at23.pth", [32, 64, 128]),
    ("p16", "/tmp/plan5_phaseA_finetune/p16", "net_epoch_bestval_at27.pth", [16, 32, 64]),
    ("p8",  "/tmp/plan5_phaseA_finetune/p8",  "net_epoch_bestval_at27.pth", [8, 16, 32]),
]

Q_PRECISIONS = ["fp32", "fp16", "int8_mm"]


def assert_gpu_isolated(gpu_id: int):
    for i in range(3):
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            text=True,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        util, mem = int(parts[1]), int(parts[2])
        if util > 1 or mem > 500:
            raise RuntimeError(f"GPU {gpu_id} NOT idle (util={util}% mem={mem}MB)")
        if i < 2:
            time.sleep(2.0)


class TRTBackbone:
    """Wraps a TRT engine as a callable replacing pyramid_backbone.get_multiscale_feature
    + decode_multiscale_feature. Engine output is a single (bev_out) tensor."""
    def __init__(self, engine_path: Path):
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        self.d_out = torch.empty(self.output_shape, dtype=torch.float32, device="cuda")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        self.context.set_tensor_address(self.output_name, self.d_out.data_ptr())
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        return self.d_out.clone()


def patch_pyramid_backbone(model, trt_backbone: TRTBackbone):
    """Monkey-patch model.pyramid_backbone to use TRT engine.

    Original flow:
      feats = pyramid.get_multiscale_feature(x)   # list[3 stage features]
      bev = pyramid.decode_multiscale_feature(feats)  # 384 ch bev

    TRT engine wraps both steps into single forward -> bev tensor directly.
    We replace pyramid_backbone.forward_single's internal call (or whichever fusion path)
    by setting a 'forward_replaced' attribute and modifying the relevant method.
    """
    pyramid = model.pyramid_backbone
    orig_get_multiscale = pyramid.get_multiscale_feature
    orig_decode_multiscale = pyramid.decode_multiscale_feature

    class FeatProxy:
        """Marker object: holds the TRT output bev tensor so decode is a no-op."""
        def __init__(self, bev): self.bev = bev

    def trt_get_multiscale(x):
        bev = trt_backbone(x)
        return FeatProxy(bev)

    def trt_decode_multiscale(proxy):
        if isinstance(proxy, FeatProxy):
            return proxy.bev
        return orig_decode_multiscale(proxy)

    pyramid.get_multiscale_feature = trt_get_multiscale
    pyramid.decode_multiscale_feature = trt_decode_multiscale


def eval_one_anchor(tag: str, model_dir: Path, ckpt_name: str,
                    num_filters: List[int], q: str, n_samples: int) -> dict:
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils, inference_utils
    from opencood.utils import eval_utils
    from torch.utils.data import DataLoader

    cfg_path = model_dir / "config.yaml"
    cfg = yaml_utils.load_yaml(str(cfg_path))
    cfg["model"]["args"]["fusion_backbone"]["num_filters"] = num_filters
    cfg["validate_dir"] = cfg.get("validate_dir") or cfg.get("data_dir")

    from opencood.data_utils.datasets import build_dataset
    print(f"  [{tag}|{q}] building dataset (val) ...")
    val_ds = build_dataset(cfg, visualize=False, train=False)
    if hasattr(val_ds, "_unset_visualize"):
        try: val_ds._unset_visualize()
        except Exception: pass

    n_total = len(val_ds)
    n_use = min(n_samples, n_total)
    idx_subset = list(range(n_use))
    print(f"  [{tag}|{q}] dataset n_total={n_total}, using first {n_use}")

    model = train_utils.create_model(cfg)
    sd = torch.load(model_dir / ckpt_name, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    device = torch.device("cuda")
    model.to(device).eval()

    if q != "fp32":
        engine_path = ENGINE_DIR / f"g8_{tag}_{q}_D1_default.engine"
        if not engine_path.exists():
            return {"tag": tag, "q": q, "status": f"engine_missing:{engine_path.name}"}
        trt_backbone = TRTBackbone(engine_path)
        patch_pyramid_backbone(model, trt_backbone)
        print(f"  [{tag}|{q}] patched pyramid_backbone with TRT engine {engine_path.name}")

    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    loader = DataLoader([val_ds[i] for i in idx_subset], batch_size=1,
                        num_workers=0,
                        collate_fn=val_ds.collate_batch_test,
                        shuffle=False, drop_last=False)

    n_ok = 0
    t0 = time.time()
    for i, batch_data in enumerate(loader):
        if batch_data is None:
            continue
        try:
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)
                infer_result = inference_utils.inference_intermediate_fusion(
                    batch_data, model, val_ds)
                pred_box_tensor = infer_result['pred_box_tensor']
                gt_box_tensor = infer_result['gt_box_tensor']
                pred_score = infer_result['pred_score']
                for iou in (0.3, 0.5, 0.7):
                    eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                                gt_box_tensor, result_stat, iou)
            n_ok += 1
        except Exception as e:
            print(f"    sample {i} eval failed: {type(e).__name__}: {str(e)[:100]}")
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_use} processed, elapsed {time.time()-t0:.1f}s")

    elapsed = time.time() - t0

    try:
        ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                                          str(model_dir))
    except Exception as e:
        print(f"  [{tag}|{q}] eval_final_results failed: {e}")
        ap30 = ap50 = ap70 = float("nan")

    return {
        "tag": tag, "q": q,
        "num_filters": str(num_filters),
        "n_samples_used": n_use,
        "n_samples_ok": n_ok,
        "ap30": round(float(ap30), 4) if not np.isnan(ap30) else None,
        "ap50": round(float(ap50), 4) if not np.isnan(ap50) else None,
        "ap70": round(float(ap70), 4) if not np.isnan(ap70) else None,
        "eval_secs": round(elapsed, 1),
        "status": "ok" if n_ok == n_use else f"partial_{n_ok}/{n_use}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-gpu", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=200,
                        help="DAIR-V2X val subset size (default 200 for speed)")
    parser.add_argument("--anchors", nargs="*", default=None)
    parser.add_argument("--q-variants", nargs="*", default=None)
    args = parser.parse_args()

    assert_gpu_isolated(args.bench_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.bench_gpu)
    os.chdir(str(HEAL_ROOT))
    print(f"[Plan v5 Phase C] GPU {args.bench_gpu} STRICT idle PASS, pinned; cwd={os.getcwd()}")

    anchors = [a for a in ANCHORS if args.anchors is None or a[0] in args.anchors]
    qs = args.q_variants or Q_PRECISIONS
    print(f"[Plan v5 Phase C] eval {len(anchors)} anchor × {len(qs)} Q = {len(anchors)*len(qs)} cells, "
          f"n_samples={args.n_samples}")

    rows = []
    for tag, model_dir_str, ckpt_name, nf in anchors:
        model_dir = Path(model_dir_str)
        for q in qs:
            print(f"\n=== {tag} | {q} ===")
            try:
                row = eval_one_anchor(tag, model_dir, ckpt_name, nf, q, args.n_samples)
            except Exception as e:
                row = {"tag": tag, "q": q, "status": f"eval_fail: {type(e).__name__}: {str(e)[:120]}"}
                import traceback; traceback.print_exc()
            print(f"  -> {row}")
            rows.append(row)

    out_csv = DATA_DIR / "plan5_phaseC_real_ap.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"\n[Plan v5 Phase C] wrote {len(rows)} rows -> {out_csv}")

    summary = ["# Plan v5 Phase C — TRT engine real INT8 AP", "",
               f"Subset size: {args.n_samples} DAIR-V2X val samples (full val: 1789)", "",
               "## Per anchor AP",
               "",
               "| tag | Q | n_ok/n | AP30 | AP50 | AP70 | eval_secs | status |",
               "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        summary.append(f"| {r.get('tag','?')} | {r.get('q','?')} | "
                       f"{r.get('n_samples_ok','-')}/{r.get('n_samples_used','-')} | "
                       f"{r.get('ap30','-')} | {r.get('ap50','-')} | {r.get('ap70','-')} | "
                       f"{r.get('eval_secs','-')} | {r.get('status','?')} |")
    summary.extend(["", "## Gate G_C analysis",
                    "Compare INT8 AP vs FP32 AP per anchor; |gap| <= 0.02 → H3 PASS"])
    fp32_aps = {r["tag"]: r.get("ap50") for r in rows if r.get("q") == "fp32" and r.get("ap50") is not None}
    int8_aps = {r["tag"]: r.get("ap50") for r in rows if r.get("q") == "int8_mm" and r.get("ap50") is not None}
    gaps = []
    for tag in fp32_aps:
        if tag in int8_aps:
            g = int8_aps[tag] - fp32_aps[tag]
            gaps.append((tag, g))
            summary.append(f"- {tag}: gap_INT8-FP32 = {g:+.4f}")
    if gaps:
        median_gap = float(np.median([abs(g) for _, g in gaps]))
        summary.append(f"\n**median |gap| = {median_gap:.4f}** (threshold 0.02)")
        summary.append(f"Gate G_C: {'PASS' if median_gap <= 0.02 else 'FAIL'}")
    (DATA_DIR / "plan5_phaseC_summary.md").write_text("\n".join(summary))
    print(f"[Plan v5 Phase C] wrote {DATA_DIR / 'plan5_phaseC_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
