"""Plan 4 Phase E.1 — 多 build sanity (LAT predictor kernel 选择噪声测量).

按 plan4.md §9.1.1 + §5.1.1:
  - 选 100 (B, Q, D) cell 从 bench v1 (4584 anchor)
  - 每个 cell 跑 3 次独立 TRT build + bench
  - 测每 anchor 3-build lat_p50 的 std/mean
  - Gate L2: ≥95% anchor std/mean ≤5%

Strategy: 100 sample 跨 21 triplet × 7 Q × 32 D 的多样子集, 用 stratified random:
  - 21 triplet × 5 cell each = 105 cell. 简化到 100: 21 × 5 = 105, drop 5
  - Q 分布: 每 triplet 在 5 cell 上跨多 Q
  - D 分布: 每 cell 选不同 D

Output:
  /tmp/plan4_phaseE1/{anchor}_trial{0,1,2}.engine + bench.json
  /tmp/plan4_phaseE1/multi_build.csv (300 行)
  /tmp/plan4_phaseE1/sanity_summary.json (per-anchor std/mean + L2 gate)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from multiprocessing import Pool, current_process
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
CALIB_DIR = REPO / "calibration/pyramid_dair_e2e_32k"

OUT = Path("/tmp/plan4_phaseE1")
OUT.mkdir(parents=True, exist_ok=True)


def sample_cells(n: int = 100, seed: int = 42) -> list:
    """Sample N (triplet, q_tag, d_tag) cells from bench v1.csv."""
    import pandas as pd
    df = pd.read_csv(
        REPO / "paper_learning/2. AAAI最终故事/data/e2e_bench_v1.csv")
    # filter to build_success
    df = df[df["build_success"] == True].reset_index(drop=True)
    # group by (triplet, q_tag, d_tag) cells
    cells = df[["triplet", "q_tag", "d_tag"]].drop_duplicates()
    print(f"[sample] {len(cells)} unique (B, Q, D) cells in bench v1")

    # Stratified: per triplet take ~5 cells with diverse (Q, D)
    rng = random.Random(seed)
    sampled = []
    for trip, grp in cells.groupby("triplet"):
        per_trip = grp.sample(n=min(5, len(grp)), random_state=rng.randint(0, 9999))
        sampled.extend([(r["triplet"], r["q_tag"], r["d_tag"])
                        for _, r in per_trip.iterrows()])
    rng.shuffle(sampled)
    sampled = sampled[:n]
    print(f"[sample] {len(sampled)} cells sampled, triplet="
          f"{len(set(s[0] for s in sampled))}, "
          f"q={len(set(s[1] for s in sampled))}, "
          f"d={len(set(s[2] for s in sampled))}")
    return sampled


def find_onnx_for_triplet(triplet: str) -> Path:
    """find ONNX file for a triplet from bench v1 caches."""
    candidates = [
        REPO / "models/e2e_cache" / f"{triplet}.onnx",
        REPO / "models/e2e_cache_a2" / f"{triplet}.onnx",
        Path("/tmp/plan4_phaseA") / f"{triplet}.onnx",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"no ONNX for {triplet}")


# D_tag → (tactic, ws_mb, builder_opt_level)
def parse_d_tag(d_tag: str) -> tuple[str, int, int]:
    # Format examples: D1_default_4gb, D11_all_enabled_4gb, D21_default_4gb_BL0, D29_default_4gb_BL5
    parts = d_tag.split("_")
    # parts: ['D{n}', tactic, '{ws}gb', maybe 'BL{0,3,5}']
    if "BL" in d_tag:
        # find last part starting with BL
        bl_part = [p for p in parts if p.startswith("BL")][0]
        bl = int(bl_part[2:])
    else:
        bl = 3
    # tactic = parts[1] (or 1-2 if multi-word like all_enabled, edge_only, cublas_lt, with_cudnn)
    # Easier: anchor by ws part (ends with 'gb'), tactic is everything between Dn and ws
    ws_idx = next(i for i, p in enumerate(parts) if p.endswith("gb"))
    tactic = "_".join(parts[1:ws_idx])
    ws_mb = int(parts[ws_idx].rstrip("gb")) * 1024
    return tactic, ws_mb, bl


def q_args_for_q_tag(q_tag: str, calib_cache: Path) -> list:
    """Map Q_tag → m4_8_trt_build_bench.py args."""
    base_calib = ["--calib-multi", f"voxel_features:{CALIB_DIR}/voxel_features.npy",
                  "--calib-multi", f"voxel_num_points:{CALIB_DIR}/voxel_num_points.npy",
                  "--calib-multi", f"voxel_coords:{CALIB_DIR}/voxel_coords.npy",
                  "--calib-multi", f"voxel_mask:{CALIB_DIR}/voxel_mask.npy",
                  "--calib-multi", f"t_ego:{CALIB_DIR}/t_ego.npy",
                  "--calib-cache", str(calib_cache)]
    if q_tag == "Q_fp32":
        return ["--precision", "fp32"]
    if q_tag == "Q_fp16":
        return ["--precision", "fp16"]
    if q_tag == "Q_int8_mm":
        return ["--precision", "int8", "--calibrator", "minmax"] + base_calib
    if q_tag == "Q_int8_ent":
        return ["--precision", "int8", "--calibrator", "entropy"] + base_calib
    if q_tag == "Q_int8_pc_wo":
        return ["--precision", "int8", "--calibrator", "minmax", "--w-only"] + base_calib
    if q_tag == "Q_mix_s0":
        return (["--precision", "mixed", "--mixed-int8-substr", "layer0",
                 "--calibrator", "minmax"] + base_calib)
    if q_tag == "Q_mix_s2":
        return (["--precision", "mixed", "--mixed-int8-substr", "layer2",
                 "--calibrator", "minmax"] + base_calib)
    raise ValueError(f"unknown q_tag: {q_tag}")


_GPU = None


def _init(gpus):
    global _GPU
    wid = current_process()._identity[0]
    _GPU = gpus[(wid - 1) % len(gpus)]


def run_one(spec):
    """spec = (triplet, q_tag, d_tag, trial)"""
    triplet, q_tag, d_tag, trial = spec
    gpu = _GPU
    tag = f"{triplet}_{q_tag}_{d_tag}_t{trial}"
    engine = OUT / f"{tag}.engine"
    report = OUT / f"{tag}.report.json"
    calib = OUT / f"{tag}.cache"

    if report.exists():
        d = json.loads(report.read_text())
        return (tag, triplet, q_tag, d_tag, trial,
                d.get("p50_ms"), 0, "cached")

    onnx = find_onnx_for_triplet(triplet)
    tactic, ws_mb, bl = parse_d_tag(d_tag)

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL)}

    t0 = time.time()
    cmd = [PYTHON, str(REPO / "scripts/phase1/m4_8_trt_build_bench.py"),
           "--onnx", str(onnx), "--engine", str(engine),
           "--report", str(report),
           "--workspace-mb", str(ws_mb),
           "--tactic", tactic,
           "--builder-opt-level", str(bl),
           "--input-shape", "32000,32,4",
           "--extra-input-shape", "voxel_num_points:32000",
           "--extra-input-shape", "voxel_coords:32000,4",
           "--extra-input-shape", "voxel_mask:32000",
           "--extra-input-shape", "t_ego:2,2,3"]
    cmd += q_args_for_q_tag(q_tag, calib)

    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    if r.returncode != 0:
        (OUT / f"{tag}.err").write_text(r.stdout + r.stderr)
        return (tag, triplet, q_tag, d_tag, trial, None,
                time.time()-t0, "build/bench fail")
    if not report.exists():
        return (tag, triplet, q_tag, d_tag, trial, None,
                time.time()-t0, "no report")
    d = json.loads(report.read_text())
    lat = d.get("p50_ms")
    return (tag, triplet, q_tag, d_tag, trial, lat, time.time()-t0, "OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-cells", type=int, default=100)
    p.add_argument("--gpus", default="0,1,2,3,4,6,7")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]

    # Step 1: sample 100 cells
    cells = sample_cells(args.n_cells, args.seed)

    # Step 2: build specs (cells × 3 trials)
    specs = []
    for trip, q, d in cells:
        for trial in range(3):
            specs.append((trip, q, d, trial))
    print(f"[E.1] {len(cells)} cells × 3 trials = {len(specs)} builds "
          f"on {len(gpus)} GPU")

    t0 = time.time()
    with Pool(processes=len(gpus), initializer=_init,
              initargs=(gpus,)) as pool:
        results = list(pool.imap_unordered(run_one, specs))
    print(f"[E.1] dispatch wall: {(time.time()-t0)/60:.1f} min")

    # Step 3: write csv + aggregate per-cell stats
    by_cell = {}
    n_ok = n_fail = 0
    for tag, trip, q, d, trial, lat, secs, status in results:
        key = (trip, q, d)
        if key not in by_cell:
            by_cell[key] = []
        if lat is not None:
            by_cell[key].append(lat)
            n_ok += 1
        else:
            n_fail += 1

    csv_path = OUT / "multi_build.csv"
    with open(csv_path, "w") as f:
        f.write("triplet,q_tag,d_tag,trial0_lat,trial1_lat,trial2_lat,mean,std,cv_pct,n_ok\n")
        for (trip, q, d), lats in sorted(by_cell.items()):
            lats_padded = lats + [None] * (3 - len(lats))
            import numpy as np
            arr = np.array([l for l in lats if l is not None])
            if len(arr) >= 2:
                mean = arr.mean(); std = arr.std(ddof=1)
                cv = std / mean if mean > 0 else 0
            elif len(arr) == 1:
                mean = arr[0]; std = 0; cv = 0
            else:
                mean = std = cv = 0
            f.write(f"{trip},{q},{d},"
                    f"{lats_padded[0] if lats_padded[0] else 'NA'},"
                    f"{lats_padded[1] if lats_padded[1] else 'NA'},"
                    f"{lats_padded[2] if lats_padded[2] else 'NA'},"
                    f"{mean:.4f},{std:.4f},{cv*100:.2f},{len(arr)}\n")

    # Step 4: L2 gate check
    import numpy as np
    cvs = []
    for lats in by_cell.values():
        if len(lats) >= 2:
            arr = np.array(lats)
            cvs.append(arr.std(ddof=1) / arr.mean() if arr.mean() > 0 else 0)
    cvs = np.array(cvs)
    pct_under_5 = (cvs <= 0.05).mean() * 100 if len(cvs) > 0 else 0
    median_cv = float(np.median(cvs)) if len(cvs) > 0 else 0
    p99_cv = float(np.percentile(cvs, 99)) if len(cvs) > 0 else 0
    l2_pass = pct_under_5 >= 95.0

    summary = {
        "n_cells_sampled": len(cells),
        "n_builds_total": len(specs),
        "n_builds_ok": n_ok,
        "n_builds_fail": n_fail,
        "median_cv_pct": median_cv * 100,
        "p99_cv_pct": p99_cv * 100,
        "pct_cells_cv_under_5pct": pct_under_5,
        "l2_gate_pass": l2_pass,
        "l2_threshold": 95.0,
    }
    (OUT / "sanity_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[E.1 SUMMARY]")
    print(f"  N cells: {len(cells)} × 3 trial = {n_ok}/{len(specs)} builds OK")
    print(f"  median cell CV: {median_cv*100:.2f}%")
    print(f"  p99 cell CV: {p99_cv*100:.2f}%")
    print(f"  % cells CV ≤ 5%: {pct_under_5:.1f}%")
    print(f"  L2 gate (≥95% cells CV≤5%): {'PASS' if l2_pass else 'FAIL'}")
    print(f"  → {OUT/'sanity_summary.json'}")


if __name__ == "__main__":
    main()
