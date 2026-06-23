"""B2 — Eval finetuned calibration points + fit monotone AP70(s0,s1,s2) model.

Run AFTER b2_finetune_launcher.py finetunes have completed.

Pipeline per calibration point:
  1. Find highest bestval ckpt in the finetune dir
  2. Export ONNX (same as stage_a: tools/export_onnx_pyramid_collab.py)
  3. Build TRT FP16 engine (same as stage_a: m4_8_trt_build_bench.py)
  4. AP eval (same as stage_a: m4_8_hybrid_infer_ap.py)

Then:
  5. Combine calibration points with stage_a anchors
  6. Fit log-linear monotone AP70 model: AP70 = a + b0*log(s0/64) + b1*log(s1/128) + b2*log(s2/256)
  7. Validate on held-out point (mixed [32,96,192])
  8. Write results/ap70_model_pyramid.json + scripts/phase2/ap70_model_query.py

Usage:
  cd /home/jichengzhi/V2X
  PYTHONPATH=/home/jichengzhi/heal_research/HEAL \\
  /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \\
    scripts/phase2/b2_eval_and_fit.py
"""
from __future__ import annotations
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/jichengzhi/V2X")
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
CACHE = REPO / "models/b2_eval_cache"
CACHE.mkdir(parents=True, exist_ok=True)
OUT_DIR = REPO / "results/b2_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKROOT = Path("/home/jichengzhi/heal_research/checkpoints/stage1")

CALIB_SPATIAL = REPO / "calibration/pyramid_dair_collab_spatial.npy"
CALIB_TEGO = REPO / "calibration/pyramid_dair_collab_tego.npy"

# Stage_a anchors (known ground truth, TRT FP16 pipeline, DAIR val 1789)
STAGE_A_ANCHORS = [
    {"tag": "base",     "s0": 64, "s1": 128, "s2": 256, "ap70": 0.6309,
     "source": "stage_a_ap_real.parquet fp16", "role": "anchor"},
    {"tag": "pruned25", "s0": 48, "s1":  96, "s2": 192, "ap70": 0.5905,
     "source": "stage_a_ap_real.parquet fp16", "role": "anchor"},
    {"tag": "pruned50", "s0": 32, "s1":  64, "s2": 128, "ap70": 0.5641,
     "source": "stage_a_ap_real.parquet fp16", "role": "anchor"},
    {"tag": "pruned75", "s0": 16, "s1":  32, "s2":  64, "ap70": 0.5300,
     "source": "stage_a_ap_real.parquet fp16", "role": "anchor"},
]

# NOTE: pad64 [64,96,192] AP70=0.5905 same as pruned25 by zero-fill property
# but NOT included separately since it's the same as pruned25 just with s0 zero-padded

# Calibration points from b2 finetune (to be evaluated)
CALIBRATION_CONFIGS = [
    ("iso_s0",  48, 128, 256, "calibration"),  # for fitting
    ("iso_s1",  64,  96, 256, "calibration"),  # for fitting
    ("iso_s2",  64, 128, 192, "calibration"),  # for fitting
    ("mixed",   32,  96, 192, "validation"),   # held-out validation (NOT in fit)
]

EVAL_GPU = "0"  # Run evals sequentially on GPU 0


def find_best_ckpt(ft_dir: Path) -> Path | None:
    """Find highest bestval ckpt in ft_dir."""
    bests = list(ft_dir.glob("net_epoch_bestval_at*.pth"))
    if not bests:
        # Fallback: highest epoch number
        all_pths = [p for p in ft_dir.glob("net_epoch*.pth") if "bestval" not in p.name]
        if not all_pths:
            return None
        return max(all_pths, key=lambda p: int(re.search(r"net_epoch(\d+)", p.name).group(1)))
    return max(bests, key=lambda p: int(re.search(r"at(\d+)\.pth", p.name).group(1)))


def export_onnx(tag: str, ckpt_dir: Path) -> Path | None:
    out_onnx = CACHE / f"b2_{tag}.onnx"
    if out_onnx.exists():
        print(f"[{tag}] ONNX already cached: {out_onnx.name}")
        return out_onnx
    best = find_best_ckpt(ckpt_dir)
    if best is None:
        print(f"[{tag}] ERROR: no checkpoint found in {ckpt_dir}")
        return None
    print(f"[{tag}] exporting ONNX from {best.name}")
    cmd = [PY, str(REPO / "tools/export_onnx_pyramid_collab.py"),
           "--ckpt", str(best),
           "--hypes", str(ckpt_dir / "config.yaml"),
           "--out", str(out_onnx),
           "--feat-h", "128"]
    env = {"CUDA_VISIBLE_DEVICES": EVAL_GPU, "PATH": os.environ.get("PATH", ""),
           "PYTHONPATH": str(HEAL)}
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env, cwd=REPO)
    if r.returncode != 0 or not out_onnx.exists():
        print(f"[{tag}] ONNX export FAILED: {r.stderr[-500:]}")
        return None
    print(f"[{tag}] ONNX export OK: {out_onnx.name} ({out_onnx.stat().st_size//1024}KB)")
    return out_onnx


def build_trt_fp16(tag: str, onnx_path: Path) -> Path | None:
    eng = CACHE / f"b2_{tag}_fp16.engine"
    if eng.exists():
        print(f"[{tag}] TRT FP16 engine already cached: {eng.name}")
        return eng
    print(f"[{tag}] building TRT FP16 engine ...")
    cmd = [PY, str(REPO / "scripts/phase1/m4_8_trt_build_bench.py"),
           "--onnx", str(onnx_path),
           "--precision", "fp16",
           "--engine", str(eng),
           "--report", str(CACHE / f"b2_{tag}_fp16_build.json"),
           "--input-shape", "2,64,128,256",
           "--extra-input-shape", "t_ego:2,2,3",
           "--n-warmup", "100", "--n-measure", "200"]
    env = {"CUDA_VISIBLE_DEVICES": EVAL_GPU, "PATH": os.environ.get("PATH", ""),
           "PYTHONPATH": str(HEAL)}
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       env=env, cwd=REPO)
    if r.returncode != 0 or not eng.exists():
        print(f"[{tag}] TRT build FAILED: {r.stderr[-500:]}")
        return None
    print(f"[{tag}] TRT FP16 engine OK: {eng.name}")
    return eng


def ap_eval_trt(tag: str, engine: Path, ckpt_dir: Path) -> dict | None:
    report = OUT_DIR / f"b2_{tag}_fp16.json"
    if report.exists():
        print(f"[{tag}] AP eval result already cached: {report.name}")
        return json.loads(report.read_text())
    print(f"[{tag}] running AP eval (1789 DAIR samples) ...")
    cmd = [PY, str(REPO / "scripts/phase1/m4_8_hybrid_infer_ap.py"),
           "--engine-collab", str(engine),
           "--tag", f"b2_{tag}_fp16",
           "--model-dir", str(ckpt_dir),
           "--n-samples", "1789",
           "--dataset", "dair", "--range", "102.4,51.2",
           "--collab-spatial-shape", "2,64,128,256",
           "--collab-tego-shape", "2,2,3",
           "--report", str(report)]
    env = {"CUDA_VISIBLE_DEVICES": EVAL_GPU, "PATH": os.environ.get("PATH", ""),
           "PYTHONPATH": str(HEAL)}
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HEAL, env=env,
                       capture_output=True, text=True, timeout=1500)
    elapsed = time.time() - t0
    if r.returncode != 0 or not report.exists():
        print(f"[{tag}] AP eval FAILED ({elapsed:.0f}s): {r.stderr[-500:]}")
        return None
    rep = json.loads(report.read_text())
    print(f"[{tag}] AP30={rep['ap30']:.4f} AP50={rep['ap50']:.4f} AP70={rep['ap70']:.4f} ({elapsed:.0f}s)")
    return rep


def eval_all_calibration() -> list[dict]:
    """Evaluate all 4 calibration point finetunes. Returns list of result dicts."""
    rows = []
    for tag, s0, s1, s2, role in CALIBRATION_CONFIGS:
        ft_dir = CKROOT / f"Pyramid_DAIR_m1_b2_{tag}_2026_06_20"
        if not ft_dir.exists():
            print(f"[{tag}] ERROR: finetune dir not found: {ft_dir}")
            rows.append({"tag": tag, "s0": s0, "s1": s1, "s2": s2, "role": role,
                         "ap70": None, "source": "MISSING"})
            continue

        onnx = export_onnx(tag, ft_dir)
        if onnx is None:
            rows.append({"tag": tag, "s0": s0, "s1": s1, "s2": s2, "role": role,
                         "ap70": None, "source": "ONNX_FAILED"})
            continue

        eng = build_trt_fp16(tag, onnx)
        if eng is None:
            rows.append({"tag": tag, "s0": s0, "s1": s1, "s2": s2, "role": role,
                         "ap70": None, "source": "TRT_FAILED"})
            continue

        rep = ap_eval_trt(tag, eng, ft_dir)
        if rep is None:
            rows.append({"tag": tag, "s0": s0, "s1": s1, "s2": s2, "role": role,
                         "ap70": None, "source": "EVAL_FAILED"})
            continue

        best_ckpt = find_best_ckpt(ft_dir)
        rows.append({
            "tag": tag, "s0": s0, "s1": s1, "s2": s2, "role": role,
            "ap30": rep["ap30"], "ap50": rep["ap50"], "ap70": rep["ap70"],
            "source": "b2_finetune_TRT_FP16_DAIR_1789",
            "ckpt": str(best_ckpt),
            "n_samples": rep.get("n_samples", 1789),
        })
    return rows


def fit_ap70_model(all_points: list[dict]) -> dict:
    """Fit log-linear monotone AP70 model.

    Model: AP70 = a + b0*log(s0/64) + b1*log(s1/128) + b2*log(s2/256)

    This is monotone non-decreasing in each stage because b0, b1, b2 >= 0
    (we enforce this via non-negative least squares: scipy.optimize.nnls).

    Parameters: a = AP70 at base [64,128,256]; b0,b1,b2 = per-stage log coefficients.
    """
    from scipy.optimize import nnls

    # Separate fit points (anchors + calibration) from held-out (validation)
    fit_pts = [p for p in all_points if p["role"] in ("anchor", "calibration") and p.get("ap70") is not None]
    val_pts = [p for p in all_points if p["role"] == "validation" and p.get("ap70") is not None]

    print(f"\n=== Model Fitting: {len(fit_pts)} fit points, {len(val_pts)} validation points ===")

    # The baseline AP70 = a (when all stages at base widths)
    a = next((p["ap70"] for p in fit_pts if p["tag"] == "base"), 0.631)

    # For log-linear model: AP70 - a = b0*log(s0/64) + b1*log(s1/128) + b2*log(s2/256)
    # Note: log(1) = 0 so base point gives residual = 0 naturally.
    # Since b0,b1,b2 must be >= 0 for monotonicity, use NNLS.
    # But also: AP70 is LOWER for smaller s → residuals are NEGATIVE for pruned models.
    # So actually: a - AP70 = (-b0)*log(s0/64) + (-b1)*log(s1/128) + (-b2)*log(s2/256)
    # With log(si/si_base) <= 0 for si <= si_base, this means coefficients should be >=0.
    # Reformulate: AP70 = a + sum_i b_i * log(si/si_base)
    # Since log(si/si_base) is negative for si < si_base, and AP70 decreases, b_i must be POSITIVE.

    X = []
    y = []
    for p in fit_pts:
        if p["tag"] == "base":
            continue  # Skip base; its residual is 0 by definition
        x_row = [
            np.log(p["s0"] / 64.0),
            np.log(p["s1"] / 128.0),
            np.log(p["s2"] / 256.0),
        ]
        X.append(x_row)
        y.append(p["ap70"] - a)

    X = np.array(X)
    y = np.array(y)

    # NNLS on negated: we want b >= 0 but y < 0 for pruned, x_row < 0 for pruned
    # So X @ b = y where all x_row < 0 and y < 0; nnls gives b >= 0.
    # Check: nnls solves min ||A b - c||^2 s.t. b >= 0
    # A = X (each row has log(si/si_base) <= 0), c = y (AP70 residuals <= 0)
    # Since both A and c have same sign structure, nnls can find b >= 0 solution.
    try:
        b, residual_sq = nnls(X, y)
        print(f"  NNLS fit: b0={b[0]:.4f}, b1={b[1]:.4f}, b2={b[2]:.4f}")
    except Exception as e:
        print(f"  NNLS failed: {e}, falling back to unconstrained least squares")
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        b = np.abs(b)  # Force non-negative for monotonicity
        print(f"  lstsq fit (abs): b0={b[0]:.4f}, b1={b[1]:.4f}, b2={b[2]:.4f}")

    # Compute fit residuals on all fit points
    print(f"\n  Model: AP70 = {a:.4f} + {b[0]:.4f}*log(s0/64) + {b[1]:.4f}*log(s1/128) + {b[2]:.4f}*log(s2/256)")
    print(f"\n  {'Tag':<12} {'s0,s1,s2':<18} {'Real AP70':>10} {'Pred AP70':>10} {'Residual':>10} {'Role':<15}")
    print("  " + "-"*75)

    residuals = []
    for p in fit_pts:
        pred = (a + b[0] * np.log(p["s0"] / 64.0)
                  + b[1] * np.log(p["s1"] / 128.0)
                  + b[2] * np.log(p["s2"] / 256.0))
        resid = p["ap70"] - pred
        residuals.append(abs(resid))
        print(f"  {p['tag']:<12} [{p['s0']:3d},{p['s1']:3d},{p['s2']:3d}]       "
              f"{p['ap70']:>10.4f} {pred:>10.4f} {resid:>+10.4f} {p['role']:<15}")

    mae_fit = np.mean(residuals)
    max_abs_fit = np.max(residuals) if residuals else 0
    print(f"\n  Fit MAE: {mae_fit:.4f}, Max |residual|: {max_abs_fit:.4f}")

    # Validation residuals (held-out)
    print(f"\n  Held-out validation:")
    val_residuals = []
    for p in val_pts:
        pred = (a + b[0] * np.log(p["s0"] / 64.0)
                  + b[1] * np.log(p["s1"] / 128.0)
                  + b[2] * np.log(p["s2"] / 256.0))
        resid = p["ap70"] - pred
        val_residuals.append(abs(resid))
        print(f"  {p['tag']:<12} [{p['s0']:3d},{p['s1']:3d},{p['s2']:3d}]       "
              f"{p['ap70']:>10.4f} {pred:>10.4f} {resid:>+10.4f} {'validation':<15}")
    mae_val = np.mean(val_residuals) if val_residuals else None
    print(f"  Validation MAE: {mae_val:.4f}" if mae_val is not None else "  No validation points with AP70")

    # Check monotonicity
    all_b_nonneg = all(bi >= 0 for bi in b)
    print(f"\n  Monotonicity check: b0={b[0]:.4f}>=0? {b[0]>=0}, b1={b[1]:.4f}>=0? {b[1]>=0}, b2={b[2]:.4f}>=0? {b[2]>=0}")
    print(f"  Model is {'✅ MONOTONE' if all_b_nonneg else '⚠️ NOT MONOTONE (some b_i < 0)'}")

    return {
        "model_form": "AP70 = a + b0*log(s0/64) + b1*log(s1/128) + b2*log(s2/256)",
        "params": {"a": float(a), "b0": float(b[0]), "b1": float(b[1]), "b2": float(b[2])},
        "is_monotone": bool(all_b_nonneg),
        "base_widths": [64, 128, 256],
        "fit_points": fit_pts,
        "validation_points": val_pts,
        "fit_mae": float(mae_fit),
        "fit_max_abs_residual": float(max_abs_fit),
        "validation_mae": float(mae_val) if mae_val is not None else None,
        "n_fit_points": len(fit_pts),
        "n_val_points": len(val_pts),
    }


def write_query_script(model: dict):
    """Write ap70_model_query.py helper."""
    a = model["params"]["a"]
    b0 = model["params"]["b0"]
    b1 = model["params"]["b1"]
    b2 = model["params"]["b2"]

    query_path = REPO / "scripts/phase2/ap70_model_query.py"
    query_path.write_text(f'''"""AP70 model query helper — auto-generated by b2_eval_and_fit.py.

Model: AP70 = {a:.6f} + {b0:.6f}*log(s0/64) + {b1:.6f}*log(s1/128) + {b2:.6f}*log(s2/256)
Fitted on DAIR val 1789, FP16 TRT eval, stage_a pipeline.
Monotone non-decreasing in each stage width (b0, b1, b2 >= 0).
Fit MAE: {model["fit_mae"]:.4f}, Val MAE: {model["validation_mae"]}
"""
from __future__ import annotations
import math

def ap70(s0: int, s1: int, s2: int) -> float:
    """Predict AP70 for backbone widths [s0, s1, s2].

    Monotone: larger widths → higher AP70.
    Valid range: s0 in [16,64], s1 in [32,128], s2 in [64,256].
    Base widths: [64, 128, 256] → AP70 ≈ {a:.4f}.

    Args:
        s0: stage 0 num_filters (output channels of pyramid_backbone stage 0)
        s1: stage 1 num_filters
        s2: stage 2 num_filters

    Returns:
        Predicted AP70 (float)
    """
    a   = {a:.6f}
    b0  = {b0:.6f}
    b1  = {b1:.6f}
    b2  = {b2:.6f}
    return a + b0 * math.log(s0 / 64.0) + b1 * math.log(s1 / 128.0) + b2 * math.log(s2 / 256.0)


if __name__ == "__main__":
    import sys
    configs = [
        ("base",     64, 128, 256),
        ("pruned25", 48,  96, 192),
        ("pruned50", 32,  64, 128),
        ("pruned75", 16,  32,  64),
        ("iso_s0",   48, 128, 256),
        ("iso_s1",   64,  96, 256),
        ("iso_s2",   64, 128, 192),
        ("mixed",    32,  96, 192),
        ("pad64",    64,  96, 192),
    ]
    print(f"{{\'tag\':<12}} {{\'s0,s1,s2\':<20}} {{\'AP70_pred\':>12}}")
    print("-" * 48)
    for tag, s0, s1, s2 in configs:
        pred = ap70(s0, s1, s2)
        print(f"{{tag:<12}} [{{s0:3d}},{{s1:3d}},{{s2:3d}}]             {{pred:>12.4f}}")
''')
    print(f"\n[query script] written: {query_path}")


def main():
    print("=== B2: Eval Calibration Points + Fit AP70 Model ===\n")

    # Step 1: Eval all calibration finetune outputs
    print("--- Step 1: Evaluating calibration points (TRT FP16, same as stage_a) ---")
    calib_rows = eval_all_calibration()

    # Step 2: Combine with stage_a anchors
    all_points = STAGE_A_ANCHORS.copy()
    all_points.extend(calib_rows)

    # Step 3: Fit model
    print("\n--- Step 2: Fitting monotone AP70 model ---")
    model = fit_ap70_model(all_points)

    # Step 4: Write output files
    print("\n--- Step 3: Writing output files ---")
    out_json = REPO / "results/ap70_model_pyramid.json"
    model["all_data_points"] = all_points
    model["provenance"] = (
        "Anchors from stage_a_ap_real.parquet (fp16 TRT, DAIR val 1789); "
        "calibration from b2 finetune (TRT FP16, same pipeline). "
        "Model fit via NNLS (non-negative least squares) to enforce monotonicity."
    )
    out_json.write_text(json.dumps(model, indent=2))
    print(f"  AP70 model: {out_json}")

    write_query_script(model)

    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"  Model: AP70 = {model['params']['a']:.4f} + "
          f"{model['params']['b0']:.4f}*log(s0/64) + "
          f"{model['params']['b1']:.4f}*log(s1/128) + "
          f"{model['params']['b2']:.4f}*log(s2/256)")
    print(f"  Is monotone: {model['is_monotone']}")
    print(f"  Fit MAE: {model['fit_mae']:.4f}")
    print(f"  Validation MAE: {model['validation_mae']}")
    print(f"  Fit points: {model['n_fit_points']}, Validation points: {model['n_val_points']}")
    print(f"\nOutput: {REPO}/results/ap70_model_pyramid.json")
    print(f"Query:  python scripts/phase2/ap70_model_query.py")


if __name__ == "__main__":
    main()
