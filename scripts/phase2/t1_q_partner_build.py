"""T1 Q-axis: build TRT FP16+INT8 backbone subnet engines for W_g/P_g partner widths.

Commissioned by data-orchestrator 2026-06-21.

MISSION: Detect Q-axis rank-flips between W_g (trap25=[48,96,192]) and P_g (pad64=[64,96,192]).
  Hypothesis: aligned s0=64 gets 1.4-1.6x INT8 speedup; non-aligned s0=48 only 1.07x.
  If pad64 INT8 < trap25 INT8 in latency despite pad64 FP16 > trap25 FP16 → Q-axis rank-flip CONFIRMED.

Known baselines (stage_a_cache, backbone-only, 4090, n=200):
  trap25 [48,96,192]: FP16=2.919ms  INT8=2.728ms  speedup=1.070x  ← alignment trap
  base   [64,128,256]: FP16=1.266ms  INT8=0.803ms  speedup=1.577x

Missing (all need FP16+INT8 unless noted):
  pad64  [64,96,192]:  FP16=? INT8=?  (PRIORITY 1 — no ONNX yet, generate from random init)
  s1_64  [64,64,256]:  FP16=? INT8=?  (ONNX: stage_a_cache/s1_64_backbone.onnx)
  s2_128 [64,128,128]: FP16=? INT8=?  (ONNX: stage_a_cache/s2_128_backbone.onnx)
  mix_b  [48,64,256]:  FP16=1.757ms (done), INT8=? (ONNX: stage_a_cache/mix_b_backbone.onnx)
  mix_d  [48,128,128]: FP16=1.780ms (done), INT8=? (ONNX: stage_a_cache/mix_d_backbone.onnx)

GPU: CUDA_VISIBLE_DEVICES=2 (confirmed idle 5MiB 0% util at commissioning time)
     Fallback: GPU 3, 6, 7

Usage:
    cd /home/jichengzhi/V2X
    CUDA_VISIBLE_DEVICES=2 /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \\
        scripts/phase2/t1_q_partner_build.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
sys.path.insert(0, str(REPO))

PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
STAGE_A = REPO / "models/stage_a_cache"
CALIB_SPATIAL = REPO / "calibration/pyramid_dair_collab_spatial.npy"
CALIB_TEGO = REPO / "calibration/pyramid_dair_collab_tego.npy"
BUILD_SCRIPT = REPO / "scripts/phase1/m4_8_trt_build_bench.py"
OUT_JSON = REPO / "results/t1_partner_widths_q.json"

GPU_VISIBLE = os.environ.get("CUDA_VISIBLE_DEVICES", "2")


# ── 1. Generate pad64 backbone ONNX (random weights, latency only) ────────────

def generate_pad64_onnx() -> Path:
    """Export PyramidCollabSubnetN2 [64,96,192] backbone ONNX from random init.

    Random weights are fine for latency — TRT latency depends on graph structure
    (channel counts, op types, shapes), not weight values.
    """
    out_path = STAGE_A / "pad64_backbone.onnx"
    if out_path.exists():
        print(f"[pad64] ONNX already exists: {out_path}")
        return out_path

    print("[pad64] Generating backbone ONNX from random init [64,96,192] ...")
    os.chdir(HEAL_ROOT)

    # Load hypes from pruned25 (closest config), override num_filters
    from opencood.hypes_yaml import yaml_utils
    hypes_path = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10/config.yaml"
    hypes = yaml_utils.load_yaml(hypes_path)
    # Override num_filters in pyramid_backbone args
    hypes["model"]["args"]["fusion_backbone"]["num_filters"] = [64, 96, 192]
    # Also update deblock upsample channels if needed (deblocks always use 128 output)
    # shrink_conv input is 3*128=384 always, output=384 — unchanged

    from opencood.models.heter_pyramid_collab import HeterPyramidCollab  # noqa: E402
    model = HeterPyramidCollab(hypes["model"]["args"])
    model.cuda().eval()
    print(f"[pad64] Model created (random init): {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    # Wrap in collab subnet  — use the same approach as export_onnx_pyramid_collab.py
    sys.path.insert(0, str(REPO))
    from tools.export_onnx_pyramid_collab import PyramidCollabSubnetN2  # noqa: E402
    subnet = PyramidCollabSubnetN2(model)
    subnet.eval()

    # Dummy inputs
    spatial = torch.randn(2, 64, 128, 256, device="cuda", dtype=torch.float32)
    t_ego = torch.eye(2, 3, device="cuda", dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)

    torch.onnx.export(
        subnet,
        (spatial, t_ego),
        str(out_path),
        input_names=["spatial_features", "t_ego"],
        output_names=["cls_preds", "reg_preds", "dir_preds"],
        opset_version=16,
        do_constant_folding=True,
        dynamic_axes=None,  # fixed shapes
    )
    print(f"[pad64] ONNX exported: {out_path} ({out_path.stat().st_size//1024}KB)")
    return out_path


# ── 2. TRT build + bench per target ──────────────────────────────────────────

def run_build(label: str, onnx_path: Path, precision: str, gpu: str) -> dict | None:
    """Run m4_8_trt_build_bench.py and return the JSON report."""
    engine = STAGE_A / f"{label}_{precision}_build.engine"
    report = STAGE_A / f"{label}_{precision}_build.json"

    if report.exists():
        print(f"[{label}/{precision}] Cached: {report.name}")
        return json.loads(report.read_text())

    cmd = [
        PYTHON, str(BUILD_SCRIPT),
        "--onnx", str(onnx_path),
        "--precision", precision,
        "--engine", str(engine),
        "--report", str(report),
        "--workspace-mb", "4096",
        "--input-shape", "2,64,128,256",
        "--extra-input-shape", "t_ego:2,2,3",
        "--n-warmup", "100",
        "--n-measure", "200",
        "--calibrator", "minmax",  # ★ NEVER entropy (catastrophic AP drop)
    ]
    if precision == "int8":
        calib_cache = STAGE_A / f"{label}_int8_calib.cache"
        cmd += [
            "--calib-multi", f"spatial_features:{CALIB_SPATIAL}",
            "--calib-multi", f"t_ego:{CALIB_TEGO}",
            "--calib-cache", str(calib_cache),
        ]

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    print(f"[{label}/{precision}] Building TRT engine (GPU={gpu}) ...")
    r = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1200,
        env=env, cwd=str(REPO)
    )
    if r.returncode != 0 or not report.exists():
        print(f"[{label}/{precision}] FAILED:\n  stdout: {r.stdout[-300:]}\n  stderr: {r.stderr[-500:]}")
        return None
    result = json.loads(report.read_text())
    print(f"[{label}/{precision}] OK: p50={result.get('p50_ms', result.get('lat_p50_ms', '?')):.3f}ms "
          f"engine={result.get('engine_size_mb','?'):.1f}MB")
    return result


# ── 3. Main ───────────────────────────────────────────────────────────────────

def main():
    import pynvml
    pynvml.nvmlInit()
    phys = int(GPU_VISIBLE)
    h = pynvml.nvmlDeviceGetHandleByIndex(phys)
    util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(h)
    used_mib = mem_info.used / 1024**2
    print(f"[gate] GPU{phys}: util={util}% mem_used={used_mib:.0f}MiB")
    # NOTE: pynvml reports ~490MiB for "idle" 4090s on this host due to system
    # driver contexts (confirmed via nvidia-smi --query-compute-apps = no apps).
    # Using 600MiB threshold to pass idle GPUs; we double-check util < 5%.
    if util > 5 or used_mib > 600:
        print("ABORT: GPU not idle. Switch to GPU 3/6/7 via CUDA_VISIBLE_DEVICES.")
        pynvml.nvmlShutdown()
        return 1
    pynvml.nvmlShutdown()

    # Step 1: Generate pad64 ONNX
    pad64_onnx = generate_pad64_onnx()

    # Step 2: Define all build targets
    # (label, onnx_path, [precisions to build])
    TARGETS = [
        # P_g partners — need FP16 + INT8 (both missing)
        ("pad64",  pad64_onnx,                                    ["fp16", "int8"]),
        ("s1_64",  STAGE_A / "s1_64_backbone.onnx",              ["fp16", "int8"]),
        ("s2_128", STAGE_A / "s2_128_backbone.onnx",             ["fp16", "int8"]),
        # W_g partners — FP16 done in b4expand_cache; only INT8 missing
        ("mix_b",  STAGE_A / "mix_b_backbone.onnx",              ["int8"]),
        ("mix_d",  STAGE_A / "mix_d_backbone.onnx",              ["int8"]),
    ]

    # Step 3: Run builds
    results = {}
    for label, onnx_path, precisions in TARGETS:
        if not onnx_path.exists():
            print(f"[{label}] ONNX missing: {onnx_path} — SKIP")
            results[label] = {"status": "MISSING_ONNX", "onnx_path": str(onnx_path)}
            continue
        entry = {"onnx_path": str(onnx_path)}
        for precision in precisions:
            rep = run_build(label, onnx_path, precision, GPU_VISIBLE)
            if rep is not None:
                p50_key = "p50_ms" if "p50_ms" in rep else "lat_p50_ms"
                entry[f"{precision}_p50_ms"] = rep.get(p50_key)
                entry[f"{precision}_engine_mb"] = rep.get("engine_size_mb")
                entry[f"{precision}_source"] = f"t1_q_partner_build/{label}_{precision}_build.json"
            else:
                entry[f"{precision}_p50_ms"] = None
                entry[f"{precision}_status"] = "FAILED"
        results[label] = entry

    # Step 4: Compute speedups and detect rank-flips
    print("\n=== Q-axis rank-flip analysis ===")
    TRAP25_FP16 = 2.919
    TRAP25_INT8 = 2.728

    for label, entry in results.items():
        fp16 = entry.get("fp16_p50_ms")
        int8 = entry.get("int8_p50_ms")
        if fp16 and int8:
            speedup = fp16 / int8
            entry["int8_speedup"] = round(speedup, 4)
            print(f"  {label}: FP16={fp16:.3f}ms  INT8={int8:.3f}ms  speedup={speedup:.3f}x")
            # Check if this is a P_g and we have a W_g comparison
            if label in ("pad64", "s1_64", "s2_128"):
                if int8 < TRAP25_INT8 and speedup > 1.10:
                    print(f"    ★ RANK-FLIP CANDIDATE: {label} INT8 ({int8:.3f}ms) < trap25 INT8 ({TRAP25_INT8:.3f}ms)")
        elif fp16:
            # FP16 only for W_g partners
            print(f"  {label}: FP16={fp16:.3f}ms  INT8=? (from b4expand_cache: mix_b=1.757ms / mix_d=1.780ms)")

    # Also report vs b4expand FP16 for mix_b/mix_d
    results["mix_b"]["fp16_p50_ms_from_b4expand"] = 1.7572
    results["mix_d"]["fp16_p50_ms_from_b4expand"] = 1.7797

    # Write summary
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "_created": "2026-06-21 T1 Q-axis expansion hw-optimizer build",
        "_scope": "backbone_subnet 4090, n_warmup=100, n_measure=200, calibrator=minmax",
        "_WARNING": "FP16 for mix_b/mix_d is from b4expand_cache (b4expand_mix_b_fp16_build.json), not stage_a_cache",
        "trap25_reference": {
            "num_filters": [48, 96, 192], "fp16_p50_ms": TRAP25_FP16, "int8_p50_ms": TRAP25_INT8,
            "int8_speedup": TRAP25_INT8 / TRAP25_FP16
        },
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] Results written to {OUT_JSON}")

    # Return non-zero if any target failed
    failed = [k for k, v in results.items() if v.get("status") == "FAILED" or v.get("int8_p50_ms") is None and "int8" in str(v.get("int8_status",""))]
    return len(failed)


if __name__ == "__main__":
    raise SystemExit(main())
