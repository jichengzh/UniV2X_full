"""M4.9 framework closed-loop measurement op for Pyramid_DAIR_m1.

Maps a v1.5 ``Config`` (B/D/M space) to a real (latency, AP) data point
by automatically:
  1. Choosing baseline vs pruned ckpt based on ``prune_rate.decoder``
  2. Building an ONNX model (sub-module or collab N=2)
  3. Building a TRT engine at the precision implied by ``q_bits`` / ``d_runtime``
  4. Running hybrid PyTorch+TRT AP eval on DAIR val
  5. Returning a structured measurement record

Used by ``scripts/phase2/m4_9_closed_loop.py`` to populate the framework
Pareto front with **real** measurements, replacing rule-based / LGB-predicted
estimates with on-hardware data.

Design:
  * Idempotent — same Config twice gives same measurements.
  * Cache engines/calib by config_id under ``models/m4_9_cache/``.
  * Honest: any divergence from CLAUDE.md "MUST do" rules raises.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from framework.config_schema import Config  # noqa: E402

CACHE_DIR = REPO_ROOT / "models/m4_9_cache"
RESULTS_DIR = REPO_ROOT / "results/m4_9"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Available pre-pruned ckpts for Pyramid_DAIR_m1. Each maps prune_rate.decoder
# range to a ckpt + config dir. Add finetuned variants once Phase B.2 done.
PYRAMID_CKPTS = {
    "baseline": "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29",
    "pruned50_zero_ft": "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10",
    # populated post-finetune:
    "pruned50_finetuned": "",  # set by setup_finetuned_ckpt() once B.2 done
}


@dataclass
class Measurement:
    config_id: str
    ckpt_tag: str
    precision: str
    engine_path: str
    engine_size_mb: float
    lat_p50_ms: float
    lat_p99_ms: float
    ap30: float
    ap50: float
    ap70: float
    n_trt_path: int
    n_pytorch_fallback: int
    notes: str = ""


def config_signature(cfg: Config) -> str:
    """Stable hash of config dict — used as cache key."""
    s = json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode()
    return hashlib.sha1(s).hexdigest()[:12]


def select_ckpt(cfg: Config) -> tuple[str, str]:
    """Map Config to (ckpt_tag, ckpt_dir) — the closest pre-prepared ckpt.

    Pyramid Phase B selection rules:
      * prune_rate.decoder == 0  → baseline
      * prune_rate.decoder ~ 0.5 → pruned50
        - if PYRAMID_CKPTS['pruned50_finetuned'] set, use it
        - else pruned50_zero_ft (unfinetuned, AP poor — explicit caveat)
      * other prune rates: not yet supported (raises)
    """
    pr = float(cfg.prune_rate.get("decoder", 0.0))
    if pr < 0.05:
        return "baseline", PYRAMID_CKPTS["baseline"]
    if 0.4 <= pr <= 0.6:
        if PYRAMID_CKPTS.get("pruned50_finetuned"):
            return "pruned50_finetuned", PYRAMID_CKPTS["pruned50_finetuned"]
        return "pruned50_zero_ft", PYRAMID_CKPTS["pruned50_zero_ft"]
    raise NotImplementedError(
        f"prune_rate.decoder={pr} not in supported set {{0, 0.5}}; "
        f"Phase B currently only has 50% pruned ckpt."
    )


def select_precision(cfg: Config) -> str:
    """Map Config q_bits to TRT engine precision (INT8 > FP16 > FP32)."""
    bits = set(cfg.q_bits.values())
    if "INT8" in bits:
        return "int8"
    if "FP16" in bits:
        return "fp16"
    return "fp32"


def measure(cfg: Config,
            engine_kind: str = "collab",
            n_samples: int = 1789,
            cuda_device: int = 4,
            allow_zero_finetune: bool = True) -> Measurement:
    """Build engine + run hybrid AP eval for the given Config.

    Args:
        engine_kind: "collab" (N=2 e2e, recommended) or "subnet" (single-agent).
        cuda_device: which GPU to pin (parallel across devices for speed).
        allow_zero_finetune: if False, raises when no finetuned ckpt available.

    Returns:
        Measurement dataclass.
    """
    sig = config_signature(cfg)
    ckpt_tag, ckpt_dir = select_ckpt(cfg)
    precision = select_precision(cfg)
    if ckpt_tag == "pruned50_zero_ft" and not allow_zero_finetune:
        raise RuntimeError("pruned50 finetuned ckpt not yet available; "
                           "set allow_zero_finetune=True to proceed with PTQ-only baseline")

    cache_subdir = CACHE_DIR / f"{ckpt_tag}_{precision}_{engine_kind}_{sig}"
    cache_subdir.mkdir(parents=True, exist_ok=True)
    onnx_path = cache_subdir / f"model.onnx"
    engine_path = cache_subdir / f"model.engine"
    report_path = cache_subdir / "report.json"

    # If already measured, return cached
    if report_path.exists():
        with open(report_path) as f:
            d = json.load(f)
        return Measurement(**d)

    # 1. Export ONNX
    if not onnx_path.exists():
        export_script = ("export_onnx_pyramid_collab.py" if engine_kind == "collab"
                         else "export_onnx_pyramid.py")
        ck_path = str(Path(ckpt_dir) / "net_epoch_bestval_at23.pth")
        hyp_path = str(Path(ckpt_dir) / "config.yaml")
        cmd = [
            "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python",
            f"tools/{export_script}",
            "--ckpt", ck_path,
            "--hypes", hyp_path,
            "--out", str(onnx_path),
        ]
        if engine_kind == "subnet":
            cmd += ["--input-shape", "1,64,128,256"]
        else:
            cmd += ["--feat-h", "128", "--feat-w", "256"]
        print(f"[m4_9.measure] ONNX export: {' '.join(cmd[-4:])}")
        r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"ONNX export failed: {r.stderr}")

    # 2. TRT build (only if not cached)
    if not engine_path.exists():
        if engine_kind == "collab":
            input_shape = "2,64,128,256"
            extra = ["--extra-input-shape", "t_ego:2,2,3"]
        else:
            input_shape = "1,64,128,256"
            extra = []

        bench_report = str(cache_subdir / "bench.json")
        cmd = [
            "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python",
            "scripts/phase1/m4_8_trt_build_bench.py",
            "--onnx", str(onnx_path),
            "--precision", precision,
            "--engine", str(engine_path),
            "--report", bench_report,
            "--input-shape", input_shape,
            "--n-warmup", "100", "--n-measure", "200",
        ] + extra
        if precision == "int8":
            if engine_kind == "collab":
                cmd += [
                    "--calib-multi", "spatial_features:calibration/pyramid_dair_collab_spatial.npy",
                    "--calib-multi", "t_ego:calibration/pyramid_dair_collab_tego.npy",
                ]
            else:
                cmd += ["--calib-data", "calibration/pyramid_dair_calib.npy"]
        print(f"[m4_9.measure] TRT build {precision}/{engine_kind}: {engine_path.name}")
        r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"TRT build failed: {r.stderr}")

    with open(cache_subdir / "bench.json") as f:
        bench = json.load(f)

    # 3. AP eval
    if engine_kind == "collab":
        ap_args = ["--engine-collab", str(engine_path),
                   "--collab-spatial-shape", "2,64,128,256"]
    else:
        ap_args = ["--engine", str(engine_path), "--input-shape", "1,64,128,256"]
    cmd = [
        "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python",
        "scripts/phase1/m4_8_hybrid_infer_ap.py",
        "--tag", f"m4_9_{sig}",
        "--model-dir", ckpt_dir,
        "--range", "102.4,51.2",
        "--n-samples", str(n_samples),
        "--report", str(cache_subdir / "ap.json"),
    ] + ap_args
    env = {"CUDA_VISIBLE_DEVICES": str(cuda_device), "PATH": "/usr/bin:/bin"}
    import os
    env["PATH"] = os.environ.get("PATH", "")
    print(f"[m4_9.measure] AP eval: {ckpt_tag} {precision} {engine_kind}")
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                       timeout=2400, env={**os.environ, **env})
    if r.returncode != 0:
        raise RuntimeError(f"AP eval failed: {r.stderr[-2000:]}")

    with open(cache_subdir / "ap.json") as f:
        ap = json.load(f)

    m = Measurement(
        config_id=cfg.config_id or sig,
        ckpt_tag=ckpt_tag,
        precision=precision,
        engine_path=str(engine_path),
        engine_size_mb=engine_path.stat().st_size / 1e6,
        lat_p50_ms=bench["p50_ms"],
        lat_p99_ms=bench["p99_ms"],
        ap30=ap["ap30"], ap50=ap["ap50"], ap70=ap["ap70"],
        n_trt_path=ap.get("n_trt_path", ap.get("n_trt_collab_path", 0) + ap.get("n_trt_single_path", 0)),
        n_pytorch_fallback=ap["n_pytorch_fallback"],
        notes=f"{engine_kind}_{ckpt_tag}",
    )
    with open(report_path, "w") as f:
        json.dump(asdict(m), f, indent=2)
    return m


def setup_finetuned_ckpt(model_dir: str):
    """Call this once Phase B.2 finetune completes to register the finetuned
    Pyramid pruned50 ckpt for measurement."""
    PYRAMID_CKPTS["pruned50_finetuned"] = model_dir
