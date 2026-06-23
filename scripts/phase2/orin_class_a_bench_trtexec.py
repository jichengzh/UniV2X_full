"""P0.3 — Orin AGX Class A lat-only bench, using trtexec subprocess (no pycuda dep).

Runs ON ORIN. Builds TRT engines via trtexec, parses GPU Compute Time stats from
trtexec output. lat-only (no AP eval per plan §4.1.3).

8 triplet × 6 Q × 9 single-IP D = 432 anchor. Multi-IP D7/D8/D9 deferred to P1.b.

Pre-deployed assets:
  /home/jichengzhi/orin_class_a/onnx/{T1..T8}_*.onnx       (97MB total)
  /home/jichengzhi/orin_class_a/calibration/calib_*.cache  (16 files, ~150KB)
  /home/jichengzhi/orin_class_a/orin_class_a_bench_trtexec.py
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
RESULT_DIR.mkdir(parents=True, exist_ok=True)


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

# 6 Q cells - all supported by trtexec on TRT 8.5
Q_CELLS = [
    # (label, trtexec_flags, calib_suffix)
    ("Q_fp32",            [],                              None),
    ("Q_fp16",            ["--fp16"],                      None),
    ("Q_int8_mm",         ["--int8", "--fp16"],            "int8_mm"),
    ("Q_int8_ent",        ["--int8", "--fp16"],            "int8_ent"),
    ("Q_best",            ["--best"],                      "int8_mm"),
    ("Q_int8_nofallback", ["--int8"],                      "int8_mm"),
]

# 9 single-IP D cells
D_CELLS = [
    # (label, trtexec extras, workspace_mb)
    ("D1_gpu_def_2gb",     [],                                   2048),
    ("D2_gpu_def_8gb",     [],                                   8192),
    ("D3_gpu_nocudnn_4gb", ["--tacticSources=-CUDNN"],           4096),
    ("D4_dla0_def_1gb",    ["--useDLACore=0", "--allowGPUFallback"], 1024),
    ("D5_dla1_def_1gb",    ["--useDLACore=1", "--allowGPUFallback"], 1024),
    ("D6_dla0_nocudnn_2gb",["--useDLACore=0", "--allowGPUFallback", "--tacticSources=-CUDNN"], 2048),
    ("D10_gpu_def_1gb",    [],                                   1024),
    ("D11_gpu_def_16gb",   [],                                   16384),
    ("D12_dla0_strict_2gb",["--useDLACore=0"],                   2048),  # no --allowGPUFallback = strict
]


def parse_trtexec_lat(stdout: str) -> dict | None:
    """Parse 'GPU Compute Time' block from trtexec stdout."""
    m = re.search(r"GPU Compute Time:\s*min\s*=\s*([\d.]+)\s*ms,\s*max\s*=\s*([\d.]+)\s*ms,\s*"
                  r"mean\s*=\s*([\d.]+)\s*ms,\s*median\s*=\s*([\d.]+)\s*ms.*?"
                  r"percentile\(99%\)\s*=\s*([\d.]+)\s*ms", stdout)
    if not m:
        return None
    return {
        "lat_min_ms":  float(m.group(1)),
        "lat_max_ms":  float(m.group(2)),
        "lat_mean_ms": float(m.group(3)),
        "lat_p50_ms":  float(m.group(4)),
        "lat_p99_ms":  float(m.group(5)),
    }


def run_anchor(tag, planes, q, d):
    q_label, q_flags, calib_suffix = q
    d_label, d_extras, ws_mb = d
    anchor_id = f"{tag}__{q_label}__{d_label}"
    report = RESULT_DIR / f"{anchor_id}.json"
    if report.exists():
        return json.loads(report.read_text())

    onnx = ONNX_DIR / f"{tag}.onnx"
    if not onnx.exists():
        return {"anchor_id": anchor_id, "status": "onnx_missing"}

    engine = ENGINE_DIR / f"{anchor_id}.engine"

    cmd = [TRTEXEC,
           f"--onnx={onnx}",
           f"--saveEngine={engine}",
           f"--workspace={ws_mb}",
           "--warmUp=200", "--duration=3", "--avgRuns=200",
           "--noDataTransfers"]
    cmd += q_flags
    cmd += d_extras
    if calib_suffix:
        calib_cache = CALIB_DIR / f"calib_{tag}_{calib_suffix}.cache"
        if calib_cache.exists():
            cmd.append(f"--calib={calib_cache}")

    rep = {
        "anchor_id": anchor_id,
        "triplet_tag": tag,
        "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
        "q_label": q_label,
        "calibrator": calib_suffix or "none",
        "d_label": d_label,
        "d_workspace_gb": ws_mb / 1024,
        "use_dla": "--useDLACore" in " ".join(d_extras),
        "tactic_no_cudnn": "-CUDNN" in " ".join(d_extras),
        "gpu_fallback": "--allowGPUFallback" in " ".join(d_extras),
        "hardware": "orin_agx_64gb",
        "cmd": " ".join(cmd),
    }

    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
    except subprocess.TimeoutExpired:
        rep["status"] = "timeout"
        rep["fail_reason"] = "trtexec_timeout_1500s"
        rep["elapsed_secs"] = time.time() - t0
        report.write_text(json.dumps(rep, indent=2))
        return rep

    rep["elapsed_secs"] = time.time() - t0
    rep["build_secs_approx"] = rep["elapsed_secs"]  # build+bench combined
    out = r.stdout + r.stderr

    if r.returncode != 0:
        rep["build_success"] = False
        rep["status"] = "build_failed"
        # Try to find a useful error message
        err_lines = [l for l in out.split("\n") if "error" in l.lower() or "failed" in l.lower()]
        rep["fail_reason"] = "; ".join(err_lines[-3:])[:300] if err_lines else f"trtexec_rc={r.returncode}"
        report.write_text(json.dumps(rep, indent=2))
        return rep

    rep["build_success"] = True
    rep["engine_size_mb"] = engine.stat().st_size / 1024 / 1024 if engine.exists() else 0

    lat = parse_trtexec_lat(out)
    if lat is None:
        rep["status"] = "lat_parse_failed"
        rep["fail_reason"] = "no GPU Compute Time line in trtexec output"
    else:
        rep.update(lat)
        rep["throughput_fps"] = 1000.0 / lat["lat_mean_ms"] if lat["lat_mean_ms"] > 0 else None
        rep["status"] = "OK"

    # Cleanup engine to save disk
    try: engine.unlink()
    except: pass

    report.write_text(json.dumps(rep, indent=2))
    return rep


def main():
    total = len(TRIPLETS) * len(Q_CELLS) * len(D_CELLS)
    print(f"=" * 72)
    print(f"P0.3 Orin Class A lat-only (trtexec): {len(TRIPLETS)} T × {len(Q_CELLS)} Q × {len(D_CELLS)} D = {total} anchor")
    print(f"=" * 72, flush=True)

    t_start = time.time()
    results = []
    idx = 0
    for tag, planes in TRIPLETS:
        for q in Q_CELLS:
            for d in D_CELLS:
                idx += 1
                t0 = time.time()
                rep = run_anchor(tag, planes, q, d)
                results.append(rep)
                status = rep.get("status", "?")
                lat = rep.get("lat_p50_ms")
                lat_str = f"lat_p50={lat:.3f}ms" if lat else f"({(rep.get('fail_reason') or '')[:50]})"
                ok_so_far = sum(1 for r in results if r.get("status") == "OK")
                print(f"[{idx}/{total}] {rep['anchor_id']:42s} {status:10s} {lat_str}  ({time.time()-t0:.0f}s) OK={ok_so_far}",
                      flush=True)
                if idx % 20 == 0:
                    summary = RESULT_DIR / "_summary.json"
                    summary.write_text(json.dumps({
                        "completed": idx, "total": total,
                        "wall_secs": time.time() - t_start,
                        "ok_count": ok_so_far,
                        "fail_count": idx - ok_so_far,
                    }, indent=2))

    elapsed = time.time() - t_start
    ok_cnt = sum(1 for r in results if r.get("status") == "OK")
    print(f"\n[done] {idx}/{total} anchors in {elapsed/3600:.2f}h, OK={ok_cnt}, fail={idx - ok_cnt}",
          flush=True)

    summary = RESULT_DIR / "_summary.json"
    summary.write_text(json.dumps({
        "completed": idx, "total": total,
        "wall_secs": elapsed,
        "ok_count": ok_cnt, "fail_count": idx - ok_cnt,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
