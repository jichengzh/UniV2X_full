"""P0.3.b — Orin INT8 補充 sweep (288 anchor).

仅跑 INT8 维度 (Q_int8_mm / Q_int8_ent / Q_best / Q_int8_nofallback) × 9 D × 8 T,
用 TRT-8502-patched cache files. 跑在 P0.3 完成之后.

Pre-deployed:
  /home/jichengzhi/orin_class_a/calibration/calib_*_patched.cache  ← TRT-8502 magic
"""
from __future__ import annotations
import json, os, re, subprocess, time
from pathlib import Path

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
ORIN_ROOT = Path("/home/jichengzhi/orin_class_a")
ONNX_DIR = ORIN_ROOT / "onnx"
ENGINE_DIR = ORIN_ROOT / "engines"
RESULT_DIR = ORIN_ROOT / "results"
CALIB_DIR = ORIN_ROOT / "calibration"
ENGINE_DIR.mkdir(parents=True, exist_ok=True)


TRIPLETS = [
    ("T1_base",  (64, 128, 256)),
    ("T2_p25",   (48,  96, 192)),
    ("T3_p37",   (40,  80, 160)),
    ("T4_p50",   (32,  64, 128)),
    ("T5_p62",   (24,  56, 128)),
    ("T6_p75",   (16,  32,  64)),
    ("T7_wide",  (48,  64, 128)),
    ("T8_deep",  (24,  48, 192)),
]

# INT8-only Q cells (use _patched cache files)
Q_CELLS = [
    ("Q_int8_mm",         ["--int8", "--fp16"],            "int8_mm_patched"),
    ("Q_int8_ent",        ["--int8", "--fp16"],            "int8_ent_patched"),
    ("Q_best",            ["--best"],                      "int8_mm_patched"),
    ("Q_int8_nofallback", ["--int8"],                      "int8_mm_patched"),
]

D_CELLS = [
    ("D1_gpu_def_2gb",     [],                                                                  2048),
    ("D2_gpu_def_8gb",     [],                                                                  8192),
    ("D3_gpu_nocudnn_4gb", ["--tacticSources=-CUDNN"],                                          4096),
    ("D4_dla0_def_1gb",    ["--useDLACore=0", "--allowGPUFallback"],                            1024),
    ("D5_dla1_def_1gb",    ["--useDLACore=1", "--allowGPUFallback"],                            1024),
    ("D6_dla0_nocudnn_2gb",["--useDLACore=0", "--allowGPUFallback", "--tacticSources=-CUDNN"],  2048),
    ("D10_gpu_def_1gb",    [],                                                                  1024),
    ("D11_gpu_def_16gb",   [],                                                                  16384),
    ("D12_dla0_strict_2gb",["--useDLACore=0"],                                                  2048),
]


def parse_lat(stdout):
    m = re.search(r"GPU Compute Time:\s*min\s*=\s*([\d.]+)\s*ms,\s*max\s*=\s*([\d.]+)\s*ms,\s*"
                  r"mean\s*=\s*([\d.]+)\s*ms,\s*median\s*=\s*([\d.]+)\s*ms.*?"
                  r"percentile\(99%\)\s*=\s*([\d.]+)\s*ms", stdout)
    if not m: return None
    return {"lat_min_ms": float(m.group(1)), "lat_max_ms": float(m.group(2)),
            "lat_mean_ms": float(m.group(3)), "lat_p50_ms": float(m.group(4)),
            "lat_p99_ms": float(m.group(5))}


def run_anchor(tag, planes, q, d):
    q_label, q_flags, calib_suffix = q
    d_label, d_extras, ws_mb = d
    anchor_id = f"{tag}__{q_label}__{d_label}"
    report = RESULT_DIR / f"{anchor_id}.json"
    # Skip if already OK (from previous P0.3 successful runs — preserves data)
    if report.exists():
        try:
            existing = json.loads(report.read_text())
            if existing.get("status") == "OK":
                return existing
        except: pass

    onnx = ONNX_DIR / f"{tag}.onnx"
    engine = ENGINE_DIR / f"{anchor_id}.engine"
    calib_cache = CALIB_DIR / f"calib_{tag}_{calib_suffix}.cache"

    cmd = [TRTEXEC,
           f"--onnx={onnx}",
           f"--saveEngine={engine}",
           f"--workspace={ws_mb}",
           "--warmUp=200", "--duration=3", "--avgRuns=200",
           "--noDataTransfers"]
    cmd += q_flags
    cmd += d_extras
    if calib_cache.exists():
        cmd.append(f"--calib={calib_cache}")

    rep = {
        "anchor_id": anchor_id, "triplet_tag": tag,
        "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
        "q_label": q_label, "calibrator": calib_suffix.replace("_patched", ""),
        "d_label": d_label, "d_workspace_gb": ws_mb / 1024,
        "use_dla": "--useDLACore" in " ".join(d_extras),
        "tactic_no_cudnn": "-CUDNN" in " ".join(d_extras),
        "gpu_fallback": "--allowGPUFallback" in " ".join(d_extras),
        "hardware": "orin_agx_64gb",
        "cmd": " ".join(cmd),
        "supplement_batch": "P0_3_b_int8_patched",
    }

    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
    except subprocess.TimeoutExpired:
        rep["status"] = "timeout"; rep["fail_reason"] = "trtexec_timeout_1500s"
        rep["elapsed_secs"] = time.time() - t0
        report.write_text(json.dumps(rep, indent=2)); return rep

    rep["elapsed_secs"] = time.time() - t0
    rep["build_secs_approx"] = rep["elapsed_secs"]
    out = r.stdout + r.stderr

    if r.returncode != 0:
        rep["build_success"] = False; rep["status"] = "build_failed"
        err_lines = [l for l in out.split("\n") if "error" in l.lower() or "failed" in l.lower()]
        rep["fail_reason"] = "; ".join(err_lines[-3:])[:300]
        report.write_text(json.dumps(rep, indent=2)); return rep

    rep["build_success"] = True
    rep["engine_size_mb"] = engine.stat().st_size / 1024 / 1024 if engine.exists() else 0
    lat = parse_lat(out)
    if lat is None:
        rep["status"] = "lat_parse_failed"
    else:
        rep.update(lat)
        rep["throughput_fps"] = 1000.0 / lat["lat_mean_ms"] if lat["lat_mean_ms"] > 0 else None
        rep["status"] = "OK"

    try: engine.unlink()
    except: pass
    report.write_text(json.dumps(rep, indent=2))
    return rep


def main():
    total = len(TRIPLETS) * len(Q_CELLS) * len(D_CELLS)
    print(f"P0.3.b INT8 sweep: {total} anchor (patched cache)")
    t_start = time.time()
    rows = []; idx = 0
    for tag, planes in TRIPLETS:
        for q in Q_CELLS:
            for d in D_CELLS:
                idx += 1
                t0 = time.time()
                rep = run_anchor(tag, planes, q, d)
                rows.append(rep)
                status = rep.get("status", "?")
                lat = rep.get("lat_p50_ms")
                ls = f"lat_p50={lat:.3f}ms" if lat else f"({(rep.get('fail_reason') or '')[:50]})"
                ok = sum(1 for r in rows if r.get("status") == "OK")
                print(f"[{idx}/{total}] {rep['anchor_id']:42s} {status:10s} {ls}  ({time.time()-t0:.0f}s) OK={ok}", flush=True)
    elapsed = time.time() - t_start
    ok_cnt = sum(1 for r in rows if r.get("status") == "OK")
    print(f"\n[done] {idx}/{total} in {elapsed/3600:.2f}h, OK={ok_cnt}", flush=True)
    (RESULT_DIR / "_supplement_summary.json").write_text(
        json.dumps({"completed": idx, "total": total, "wall_secs": elapsed,
                    "ok_count": ok_cnt, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
