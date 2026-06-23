"""Phase 1.2 (plan v3) — D-tactic dispatcher: 36 anchor (B × Q × D, FT=8 fixed).

For each (triplet ∈ 3) × (Q ∈ 4) × (D ∈ 3):
  - build TRT engine with the (Q, D) combo
  - bench latency (mean/p50/p99) via 200 warmup + 200 measure
  - reuse AP50 from plan v2 combo_3b.csv (AP unchanged by D, only Q affects accuracy
    materially in this regime — D modifies builder/runtime, not numerics)

D variants:
  - D_default: --builder-opt-level=3 (plan v2 baseline)
  - D_sparse:  --precision={fp16,int8}_sparse + opt-level=3 (SPARSE_WEIGHTS flag)
  - D_optlow:  --builder-opt-level=0 (skip kernel autotuning, faster build, often slower lat)

Total: 3 × 4 × 3 = 36 anchors. Reuses plan v2 ONNX exports (FT=8 ckpts).

Output: /tmp/a11_d_tactic/{tag}.build.json (lat) + dtactic.csv (36 rows).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from multiprocessing import Pool, current_process
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
CALIB_DIR = REPO / "calibration/pyramid_dair_e2e_32k"
OUT = Path("/tmp/a11_d_tactic")
OUT.mkdir(parents=True, exist_ok=True)

# ONNX from plan v2 phase 3b (FT=8 ckpts)
ONNX_SRC = Path("/tmp/a10_phase3b")
# AP from plan v2 combo_3b.csv
AP_SRC = ONNX_SRC / "combo_3b.csv"

TRIPLETS = ["T_g8_p87", "T_g8_p93", "T_g8_p97"]
FT_FIXED = 8


def make_calib_args() -> list[str]:
    return ["--calib-multi", f"voxel_features:{CALIB_DIR}/voxel_features.npy",
            "--calib-multi", f"voxel_num_points:{CALIB_DIR}/voxel_num_points.npy",
            "--calib-multi", f"voxel_coords:{CALIB_DIR}/voxel_coords.npy",
            "--calib-multi", f"voxel_mask:{CALIB_DIR}/voxel_mask.npy",
            "--calib-multi", f"t_ego:{CALIB_DIR}/t_ego.npy"]


# (Q_tag, base build args without precision-specific D adjustments)
def q_base_args(qvar: str, use_sparse: bool) -> list[str]:
    """Return (precision flag list, w_only flag, requires calib) per Q variant."""
    calib = make_calib_args()
    if qvar == "fp16":
        prec = "fp16_sparse" if use_sparse else "fp16"
        return ["--precision", prec]
    if qvar == "int8_mm":
        prec = "int8_sparse" if use_sparse else "int8"
        return ["--precision", prec, "--calibrator", "minmax"] + calib
    if qvar == "int8_pc_wo":
        prec = "int8_sparse" if use_sparse else "int8"
        return ["--precision", prec, "--calibrator", "minmax", "--w-only"] + calib
    if qvar == "int8_ent":
        prec = "int8_sparse" if use_sparse else "int8"
        return ["--precision", prec, "--calibrator", "entropy"] + calib
    raise ValueError(qvar)


# (D_tag, builder_opt_level, use_sparse)
D_VARIANTS = [
    ("default", 3, False),
    ("sparse",  3, True),
    ("optlow",  0, False),
]
Q_VARIANTS = ["fp16", "int8_mm", "int8_pc_wo", "int8_ent"]

_GPU = None


def _init(gpus):
    global _GPU
    wid = current_process()._identity[0]
    _GPU = gpus[(wid - 1) % len(gpus)]
    print(f"[worker {wid}] GPU {_GPU}", flush=True)


def run_one(spec):
    triplet, qvar, d_tag, opt_lvl, use_sparse = spec
    tag = f"{triplet}_ft{FT_FIXED:02d}_{qvar}_D{d_tag}"
    onnx = ONNX_SRC / f"{triplet}_ft{FT_FIXED:02d}.onnx"
    engine = OUT / f"{tag}.engine"
    cache = OUT / f"{tag}.cache"
    build_rep = OUT / f"{tag}.build.json"

    if build_rep.exists():
        d = json.loads(build_rep.read_text())
        lat = d.get("lat_p50_ms") or d.get("mean_ms")
        return (tag, triplet, qvar, d_tag, lat, 0, "cached")
    if not onnx.exists():
        return (tag, triplet, qvar, d_tag, None, 0,
                f"onnx missing: {onnx.name}")

    args = q_base_args(qvar, use_sparse)
    # multi-input model: voxel_features (primary), 4 extras
    extra_shapes = [
        "--extra-input-shape", "voxel_num_points:32000",
        "--extra-input-shape", "voxel_coords:32000,4",
        "--extra-input-shape", "voxel_mask:32000",
        "--extra-input-shape", "t_ego:2,2,3",
    ]
    skip_build = ["--skip-build"] if engine.exists() else []
    cmd = [PYTHON, str(REPO / "scripts/phase1/m4_8_trt_build_bench.py"),
           "--onnx", str(onnx),
           "--engine", str(engine), "--report", str(build_rep),
           "--workspace-mb", "4096", "--tactic", "default",
           "--builder-opt-level", str(opt_lvl),
           "--n-warmup", "100", "--n-measure", "200",
           "--input-shape", "32000,32,4",
           "--calib-cache", str(cache)] + extra_shapes + skip_build + args

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(_GPU),
           "PYTHONPATH": str(HEAL)}
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       timeout=1200)
    if r.returncode != 0:
        (OUT / f"{tag}.engine.err").write_text(r.stdout + r.stderr)
        return (tag, triplet, qvar, d_tag, None,
                time.time()-t0, "build/bench fail")

    if not build_rep.exists():
        return (tag, triplet, qvar, d_tag, None,
                time.time()-t0, "no build report")
    d = json.loads(build_rep.read_text())
    lat = d.get("lat_p50_ms") or d.get("mean_ms")
    return (tag, triplet, qvar, d_tag, lat, time.time()-t0, "OK")


def load_ap_from_v2() -> dict:
    """Parse plan v2 combo_3b.csv → {(triplet, qvar): ap50}."""
    if not AP_SRC.exists():
        print(f"WARN: {AP_SRC} missing — AP column will be NA")
        return {}
    ap = {}
    lines = AP_SRC.read_text().strip().splitlines()
    header = lines[0].split(",")  # triplet, fp16, int8_mm, int8_pc_wo, int8_ent
    for ln in lines[1:]:
        parts = ln.split(",")
        triplet = parts[0]
        for i, qvar in enumerate(header[1:], start=1):
            try:
                ap[(triplet, qvar)] = float(parts[i])
            except (ValueError, IndexError):
                ap[(triplet, qvar)] = None
    return ap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="2,3,4,5,6,7")
    args = p.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]

    # Build spec list
    specs = []
    for triplet in TRIPLETS:
        for qvar in Q_VARIANTS:
            for d_tag, opt_lvl, use_sparse in D_VARIANTS:
                specs.append((triplet, qvar, d_tag, opt_lvl, use_sparse))

    print(f"[1.2] {len(specs)} (triplet × Q × D) build+bench on "
          f"{len(gpus)} GPU")

    with Pool(processes=len(gpus), initializer=_init, initargs=(gpus,)) as pool:
        results = list(pool.imap_unordered(run_one, specs))

    ap_map = load_ap_from_v2()

    # Report
    print(f"\n{'tag':<48}{'lat_p50':>10}{'AP50':>9}{'secs':>7}  status")
    rows = []
    for tag, triplet, qvar, d_tag, lat, secs, status in sorted(
            results, key=lambda r: r[0]):
        ap50 = ap_map.get((triplet, qvar))
        lat_s = f'{lat:.2f}' if lat is not None else 'FAIL'
        ap_s = f'{ap50:.3f}' if ap50 is not None else 'NA'
        print(f'{tag:<48}{lat_s:>10}{ap_s:>9}{secs:>7.0f}  {status}')
        rows.append((triplet, qvar, d_tag, lat, ap50))

    # CSV
    csv_path = OUT / "dtactic.csv"
    with open(csv_path, "w") as f:
        f.write("triplet,qvar,d_tag,lat_p50_ms,ap50\n")
        for triplet, qvar, d_tag, lat, ap in rows:
            lat_s = f"{lat:.4f}" if lat is not None else "NA"
            ap_s = f"{ap:.4f}" if ap is not None else "NA"
            f.write(f"{triplet},{qvar},{d_tag},{lat_s},{ap_s}\n")
    print(f"\ndtactic → {csv_path}")

    # Success criteria: lat span ≥ 1.3× across D for at least one (triplet, qvar)
    print(f"\n[1.2 SUCCESS CRITERIA]")
    span_ok_count = 0
    for triplet in TRIPLETS:
        for qvar in Q_VARIANTS:
            lats = [lat for t, q, _, lat, _ in rows
                    if t == triplet and q == qvar and lat is not None]
            if len(lats) >= 2:
                span = max(lats) / min(lats)
                if span >= 1.3:
                    span_ok_count += 1
    n_total_cells = len(TRIPLETS) * len(Q_VARIANTS)
    print(f"  D-span ≥ 1.3× cells: {span_ok_count}/{n_total_cells}")
    print(f"  (≥1 cell expected to pass — D effect confirmed)")


if __name__ == "__main__":
    main()
