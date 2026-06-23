"""S0 — HW↔SW coupling probe (route-2 feasibility smoke, TRT-native, NO TVM).

Question this answers (the make-or-break for route 2):
  Does the OPTIMAL hardware build-config depend on the SOFTWARE (prune/precision) decision?
  If argmax_{opt_level, workspace} latency differs across {dense, p50, p75}×{fp16,int8},
  then a HW↔SW coupling exists and is measurable INSIDE TensorRT (no TVM needed).
  If TRT-auto's default config is invariant-optimal across all software points,
  that is the evidence that TRT gives no co-design axis → escalate to S2 (TVM).

Also records int8/fp16/fp32 layer counts per point: the STRONG coupling signal is whether
the INT8-eligible layer fraction depends on the pruned channel alignment (IMMA tile rule).

口径: body_subnet_collab2 (2×64×128×256), CUDA-Event, warmup=200/measure=200, idle-GPU gated.
Reuses PROVEN functions from h2_workspace_scan (build/bench/layer-count/idle-gate).
Assets reused: models/stage_a_cache/{base,pruned50,pruned75}.onnx + int8 calib caches
  (model assets only; S0 results are FRESH under route-2 authorization, NOT the frozen H2 CSV).

Output: results/S0_hwsw_coupling_4090.csv  + stdout argmax coupling summary.

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/phase2/s0_hwsw_coupling_probe.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "phase2"))

# Reuse proven harness (build/bench/layer-count/idle-gate) — do not rewrite.
from h2_workspace_scan import (  # noqa: E402
    assert_gpu_idle,
    benchmark_engine,
    build_engine,
    count_layer_precisions,
    PHYS_GPU,
    N_WARMUP,
    N_MEASURE,
)

STAGE_A = REPO / "models" / "stage_a_cache"
S0_ENGINE_DIR = REPO / "models" / "s0_coupling_cache"
S0_ENGINE_DIR.mkdir(parents=True, exist_ok=True)

# ── Software points (the SW axis): 3 prune levels ────────────────────────────
PRUNE_POINTS = [
    {"prune": "dense", "planes": "64_128_256", "onnx": str(STAGE_A / "base.onnx"),
     "calib": str(STAGE_A / "base_int8_calib.cache")},
    {"prune": "p50", "planes": "32_64_128", "onnx": str(STAGE_A / "pruned50.onnx"),
     "calib": str(STAGE_A / "pruned50_int8_calib.cache")},
    {"prune": "p75", "planes": "16_32_64", "onnx": str(STAGE_A / "pruned75.onnx"),
     "calib": str(STAGE_A / "pruned75_int8_calib.cache")},
]

PRECISIONS = ["fp16", "int8"]

# ── Hardware axis (TRT-controllable build knobs) ─────────────────────────────
OPT_LEVELS = [0, 3, 5]
WORKSPACES_MB = [1024, 4096]


def main():
    import torch
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES handles physical mapping

    n_pts = len(PRUNE_POINTS) * len(PRECISIONS) * len(OPT_LEVELS) * len(WORKSPACES_MB)
    print(f"\n{'='*64}")
    print(f"S0 HW↔SW coupling probe | GPU={PHYS_GPU} | {n_pts} engines")
    print(f"  SW axis: {len(PRUNE_POINTS)} prune × {len(PRECISIONS)} prec")
    print(f"  HW axis: opt{OPT_LEVELS} × ws{WORKSPACES_MB}MB")
    print(f"{'='*64}\n")

    assert_gpu_idle(PHYS_GPU)

    rows = []
    for pt in PRUNE_POINTS:
        for prec in PRECISIONS:
            for opt in OPT_LEVELS:
                for ws in WORKSPACES_MB:
                    tag = f"{pt['prune']}_{prec}_opt{opt}_ws{ws}"
                    engine_path = str(S0_ENGINE_DIR / f"{tag}.engine")
                    print(f"\n── {tag} ──")
                    try:
                        assert_gpu_idle(PHYS_GPU)
                        build_engine(
                            onnx_path=pt["onnx"],
                            precision=prec,
                            engine_path=engine_path,
                            workspace_mb=ws,
                            calib_cache=(pt["calib"] if prec == "int8" else None),
                            opt_level=opt,
                        )
                        lc = count_layer_precisions(engine_path)
                        lat = benchmark_engine(engine_path, N_WARMUP, N_MEASURE)
                        print(f"  [result] p50={lat['lat_p50_ms']:.4f}ms "
                              f"int8={lc['int8']} fp16={lc['fp16']} fp32={lc['fp32']}")
                        rows.append({
                            "prune": pt["prune"], "planes": pt["planes"],
                            "precision": prec, "opt_level": opt, "workspace_mb": ws,
                            "lat_p50_ms": round(lat["lat_p50_ms"], 4),
                            "lat_p99_ms": round(lat["lat_p99_ms"], 4),
                            "lat_mean_ms": round(lat["lat_mean_ms"], 4),
                            "lat_std_ms": round(lat["lat_std_ms"], 4),
                            "int8_layer_count": lc["int8"],
                            "fp16_layer_count": lc["fp16"],
                            "fp32_layer_count": lc["fp32"],
                            "engine_size_mb": lat["engine_size_mb"],
                            "gpu": PHYS_GPU,
                            "latency_kind": "body_subnet_collab2",
                            "gpu_idle_verified": True,
                            "source": "S0_hwsw_coupling;lat:CUDA-Event;build:TRT-python-api;route2-auth",
                            "status": "OK",
                        })
                    except Exception as e:  # noqa: BLE001
                        print(f"  [ERROR] {e}")
                        rows.append({
                            "prune": pt["prune"], "planes": pt["planes"],
                            "precision": prec, "opt_level": opt, "workspace_mb": ws,
                            "status": f"ERROR:{e}", "gpu": PHYS_GPU,
                        })

    # ── Write CSV ────────────────────────────────────────────────────────────
    out_path = REPO / "results" / "S0_hwsw_coupling_4090.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                all_keys.append(k); seen.add(k)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ── Coupling summary: argmax HW config per software point ────────────────
    print(f"\n{'='*64}\nCOUPLING SUMMARY (argmax HW config per SW point)\n{'='*64}")
    ok = [r for r in rows if r.get("status") == "OK"]
    print(f"{'SW point':<16}{'best (opt,ws)':<16}{'best p50(ms)':<14}{'int8 layers':<12}")
    argmax_by_point = {}
    for pt in PRUNE_POINTS:
        for prec in PRECISIONS:
            sub = [r for r in ok if r["prune"] == pt["prune"] and r["precision"] == prec]
            if not sub:
                continue
            best = min(sub, key=lambda r: r["lat_p50_ms"])
            key = f"{pt['prune']}/{prec}"
            b_opt = best["opt_level"]
            b_ws = best["workspace_mb"]
            argmax_by_point[key] = (b_opt, b_ws)
            cfg_str = f"opt{b_opt},ws{b_ws}"
            print(f"{key:<16}{cfg_str:<16}"
                  f"{best['lat_p50_ms']:<14}{best['int8_layer_count']:<12}")
    distinct = set(argmax_by_point.values())
    print(f"\nDistinct argmax HW configs across SW points: {len(distinct)} → {distinct}")
    print("VERDICT: HW↔SW coupling present" if len(distinct) > 1
          else "VERDICT: argmax HW config INVARIANT across SW points (weak coupling on this axis)")
    print(f"\n[done] {len(ok)}/{len(rows)} OK → {out_path}")
    return rows


if __name__ == "__main__":
    main()
