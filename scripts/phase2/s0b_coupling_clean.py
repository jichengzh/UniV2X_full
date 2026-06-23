"""S0b — decisive HW↔SW coupling test at a FIXED build config.

Corrects S0 v1: layer-count is confounded by opt-level (same p75 = 5..54 int8 layers).
The honest co-design question is LATENCY: does INT8's real speedup shrink as you prune?
Hold the HW build config FIXED (opt3, ws4096) and vary only prune × precision.

Phase 1 (build, ANY gpu, no timing): build/cache 4 prune × 2 prec engines @ (opt3,ws4096)
  + a determinism rebuild of dense_int8 → _v2. Tabulate fixed-parser INT8 compute-layer count.
Phase 2 (bench, CLEAN gpu only): CUDA-Event latency → per prune: lat_fp16, lat_int8,
  INT8 speedup = fp16/int8. Verdict: does INT8 speedup decay with prune (real coupling)?

Parser FIXED for TRT 10.13: key 'Format/Datatype' ∈ {Float, Half, Int8};
  count compute layers (CaskConvolution/GemmConvolution/DeconvolutionV2) by input[0] dtype.

Usage:
  build phase:  CUDA_VISIBLE_DEVICES=<any> python scripts/phase2/s0b_coupling_clean.py build
  bench phase:  CUDA_VISIBLE_DEVICES=<clean> python scripts/phase2/s0b_coupling_clean.py bench
"""
from __future__ import annotations
import csv, glob, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "phase2"))
from h2_workspace_scan import (  # noqa: E402
    assert_gpu_idle, benchmark_engine, build_engine, PHYS_GPU, N_WARMUP, N_MEASURE,
)
import tensorrt as trt  # noqa: E402

STAGE = REPO / "models" / "stage_a_cache"
OUT = REPO / "models" / "s0b_cache"
OUT.mkdir(parents=True, exist_ok=True)
OPT, WS = 3, 4096  # FIXED hardware build config

POINTS = [
    {"prune": "dense",  "planes": "64_128_256", "onnx": "base.onnx",     "calib": "base_int8_calib.cache"},
    {"prune": "p50",    "planes": "32_64_128",  "onnx": "pruned50.onnx", "calib": "pruned50_int8_calib.cache"},
    {"prune": "trap25", "planes": "48_96_192",  "onnx": "pruned25.onnx", "calib": "pruned25_int8_calib.cache"},
    {"prune": "p75",    "planes": "16_32_64",   "onnx": "pruned75.onnx", "calib": "pruned75_int8_calib.cache"},
]

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
_RT = trt.Runtime(TRT_LOGGER)
COMPUTE = {"CaskConvolution", "CaskGemmConvolution", "CaskDeconvolutionV2"}


def int8_compute_count(engine_path: str) -> dict:
    """FIXED parser: count compute layers by input[0] Format/Datatype."""
    eng = _RT.deserialize_cuda_engine(open(engine_path, "rb").read())
    data = json.loads(eng.create_engine_inspector().get_engine_information(
        trt.LayerInformationFormat.JSON))
    c = {"Int8": 0, "Half": 0, "Float": 0, "other": 0}
    for L in data.get("Layers", []):
        if not isinstance(L, dict) or L.get("LayerType") not in COMPUTE:
            continue
        ins = L.get("Inputs", [])
        dt = (ins[0].get("Format/Datatype", "").split()[0] if ins else "")
        c[dt if dt in c else "other"] += 1
    c["compute_total"] = c["Int8"] + c["Half"] + c["Float"] + c["other"]
    return c


def epath(prune, prec, suffix=""):
    return str(OUT / f"{prune}_{prec}_opt{OPT}_ws{WS}{suffix}.engine")


def do_build():
    import torch; torch.cuda.set_device(0)
    print(f"[build] fixed HW config opt={OPT} ws={WS}MB | {len(POINTS)} prune × 2 prec")
    rows = []
    for pt in POINTS:
        for prec in ["fp16", "int8"]:
            ep = epath(pt["prune"], prec)
            build_engine(str(STAGE / pt["onnx"]), prec, ep, workspace_mb=WS,
                         calib_cache=(str(STAGE / pt["calib"]) if prec == "int8" else None),
                         opt_level=OPT)
            lc = int8_compute_count(ep)
            rows.append({"prune": pt["prune"], "planes": pt["planes"], "precision": prec,
                         "engine": os.path.basename(ep), **lc})
            print(f"  {pt['prune']}/{prec}: compute={lc['compute_total']} "
                  f"Int8={lc['Int8']} Half={lc['Half']} Float={lc['Float']}")
    # determinism rebuild
    ep2 = epath("dense", "int8", "_v2")
    build_engine(str(STAGE / "base.onnx"), "int8", ep2, workspace_mb=WS,
                 calib_cache=str(STAGE / "base_int8_calib.cache"), opt_level=OPT)
    lc2 = int8_compute_count(ep2)
    print(f"  [determinism] dense_int8 rebuild Int8={lc2['Int8']} "
          f"(orig {int8_compute_count(epath('dense','int8'))['Int8']})")
    # write layer-count CSV
    p = REPO / "results" / "S0b_layercount_4090.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"[build] layer-count → {p}")


def wait_until_idle(phys, timeout=240, poll=10):
    """Poll until GPU idle (util<=2, foreign_mem<=50); raise only on timeout.
    Robust to transient contention — does NOT hard-abort the whole bench."""
    import time
    from h2_workspace_scan import gpu_idle_check, foreign_mem
    t0 = time.time()
    while True:
        util, _ = gpu_idle_check(phys)
        other = foreign_mem(phys)
        if util <= 2 and other <= 50:
            return
        if time.time() - t0 > timeout:
            raise RuntimeError(f"GPU{phys} not idle within {timeout}s (util={util} other={other})")
        time.sleep(poll)


def do_bench():
    import torch; torch.cuda.set_device(0)
    wait_until_idle(PHYS_GPU)
    print(f"[bench] CLEAN gpu={PHYS_GPU} | fixed opt={OPT} ws={WS}")
    res = {}
    rows = []
    for pt in POINTS:
        for prec in ["fp16", "int8"]:
            ep = epath(pt["prune"], prec)
            if not os.path.exists(ep):
                print(f"  [skip] {ep} not built"); continue
            wait_until_idle(PHYS_GPU)  # per-cell wait, not abort
            lat = benchmark_engine(ep, N_WARMUP, N_MEASURE)
            lc = int8_compute_count(ep)
            res[(pt["prune"], prec)] = lat["lat_p50_ms"]
            rows.append({"prune": pt["prune"], "planes": pt["planes"], "precision": prec,
                         "lat_p50_ms": round(lat["lat_p50_ms"], 4),
                         "lat_std_ms": round(lat["lat_std_ms"], 4),
                         "int8_compute": lc["Int8"], "compute_total": lc["compute_total"],
                         "gpu": PHYS_GPU, "latency_kind": "body_subnet_collab2",
                         "gpu_idle_verified": True,
                         "source": "S0b_clean;lat:CUDA-Event;fixed-opt3-ws4096;route2-auth"})
            print(f"  {pt['prune']}/{prec}: p50={lat['lat_p50_ms']:.4f}±{lat['lat_std_ms']:.4f}ms int8L={lc['Int8']}")
    # verdict: does INT8 speedup decay with prune?
    print(f"\n{'='*60}\nVERDICT: INT8 speedup (fp16/int8) vs prune\n{'='*60}")
    print(f"{'prune':<10}{'fp16(ms)':<11}{'int8(ms)':<11}{'INT8 speedup':<14}{'int8 layers':<12}")
    for pt in POINTS:
        f16 = res.get((pt["prune"], "fp16")); i8 = res.get((pt["prune"], "int8"))
        if f16 and i8:
            sp = f16 / i8
            lc = int8_compute_count(epath(pt["prune"], "int8"))["Int8"]
            print(f"{pt['prune']:<10}{f16:<11.4f}{i8:<11.4f}{sp:<14.3f}{lc:<12}")
    p = REPO / "results" / "S0b_coupling_clean_4090.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"\n[bench] → {p}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    (do_build if mode == "build" else do_bench)()
