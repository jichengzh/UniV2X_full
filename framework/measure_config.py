#!/usr/bin/env python3
"""P2.1 — reproducible whole-(sub)network TVM measure interface for Pyramid+H800.

Given a joint software config (width=[w0,w1,w2], precision in {fp32,fp16}),
drive the SAME 4-step pipeline that produced original60, on H800:

  1. EXPORT   pyramid_backbone subnet -> {label}_backbone.onnx   (torch env)
  2. TUNE     TVM MetaSchedule, max_trials=32, seed=0 -> schedule DB   (tvm env)
  3. MEASURE  latency: apply DB, report default_us + tuned_us         (tvm env)
  4. MEASURE  energy : apply DB, report joule_per_inference           (tvm env)

Returns {lat_default_ms, lat_tuned_ms, energy_j, build_success, ...}. This is the
INNER-LOOP measurement oracle the SMBO closed loop (P2.2) calls on NEW widths
(outside original60). latency_kind == original60 == pyramid_backbone subnet
(NOT e2e) -> stays口径-consistent with the frozen LUT / cost model labels.

Run ON H800. Steps are skip-if-exists (idempotent / resumable). fp32+fp16 go
through run_measurement_job; int8 routes to the native-int8 bridge (--precision
int8) which needs real calibration activations (heavier; used by P2.2 only when
an int8 candidate reaches the Pareto front).
"""
from __future__ import annotations
import argparse, json, subprocess, sys, os
from pathlib import Path
from datetime import datetime, timezone

REPO = Path(os.environ.get("V2X_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
EXPORT_PY = os.environ.get("V2X_EXPORT_PYTHON", sys.executable)  # torch/HEAL
TVM_PY = os.environ.get("V2X_TVM_PYTHON", sys.executable)         # TVM runtime
RUN_ROOT = Path(os.environ.get("V2X_TVM_RUN_ROOT", REPO / "results/tvm_measurements"))
MODELS_DIR = RUN_ROOT / "models"
WORKDIRS = RUN_ROOT / "workdirs"
RESULTS = RUN_ROOT / "results"
MAX_TRIALS = int(os.environ.get("V2X_TVM_MAX_TRIALS", "32"))
SEED = int(os.environ.get("V2X_TVM_TUNE_SEED", "0"))

QUANT = {  # precision -> (quant_policy, quant_method)
    "fp32": ("fp32", "h800_tvm_relax_fp32"),
    "fp16": ("fp16", "h800_tvm_relax_fp16"),
    "int8": ("int8", "h800_tvm_relax_native_int8"),
}

TVM_SITE = os.environ.get("V2X_TVM_SITE", "").strip()
NVLIBS_FILE = os.environ.get("V2X_TVM_NVLIBS_FILE", "").strip()
CUDA_BIN = os.environ.get("V2X_CUDA_BIN", "").strip()


def tvm_env(gpu):
    """Child env for TVM steps: mirror the working lane runner (LD_LIBRARY_PATH must
    be pre-set before python start so dlopen resolves the bundled libcudart)."""
    env = os.environ.copy()
    nvlibs = []
    nvroot = Path(TVM_SITE) / "nvidia" if TVM_SITE else None
    if nvroot and nvroot.is_dir():  # all bundled nvidia/*/lib (cublas, cudnn, cuda_runtime, ...)
        nvlibs = [str(p) for p in sorted(nvroot.glob("*/lib")) if p.is_dir()]
    try:
        extra = Path(NVLIBS_FILE).read_text().strip() if NVLIBS_FILE else ""
    except OSError:
        extra = ""
    ld_parts = [f"{TVM_SITE}/nvidia/cuda_runtime/lib" if TVM_SITE else "", f"{TVM_SITE}/tvm/lib" if TVM_SITE else "",
                *nvlibs, extra, env.get("LD_LIBRARY_PATH", "")]
    env["LD_LIBRARY_PATH"] = ":".join(p for p in ld_parts if p)
    env["PYTHONPATH"] = ":".join(p for p in [str(REPO), TVM_SITE, env.get("PYTHONPATH", "")] if p)
    if CUDA_BIN:
        env["PATH"] = CUDA_BIN + ":" + env.get("PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def label_of(width) -> str:
    return "smbo_" + "x".join(str(int(x)) for x in width)


def run(cmd, log, env=None, cwd=REPO, timeout=None, ok_check=None):
    """Run cmd. If it hangs past `timeout` but `ok_check()` says outputs are
    already written (known post-measurement CUDA-teardown hang), kill it and
    return 0. Otherwise return the real rc."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        fh.write("$ " + " ".join(map(str, cmd)) + "\n\n")
        fh.flush()
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), env=env,
                             stdout=fh, stderr=subprocess.STDOUT)
        if ok_check is None:
            return p.wait(timeout=timeout)
        # poll for outputs; the measure job hangs on CUDA teardown after writing
        # rows -> kill early once ok_check() passes (+grace) instead of blocking.
        import time as _t
        deadline = _t.time() + (timeout or 1e9)
        while _t.time() < deadline:
            rc = p.poll()
            if rc is not None:                     # exited on its own
                return rc
            if ok_check():
                _t.sleep(8)                        # grace: let the 2nd row flush
                p.kill()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                fh.write("\n[run] outputs present; killed post-teardown hang\n")
                return 0
            _t.sleep(5)
        p.kill()
        fh.write(f"\n[run] timeout after {timeout}s; ok_check={ok_check()}\n")
        return 0 if ok_check() else 124


# ---- step 1: export ONNX ---------------------------------------------------
def export_onnx(width, label, log_dir, input_hw=(128, 256)) -> Path:
    """Export pyramid_backbone subnet ONNX. input_hw=(128,256) is the fp32/int8 LUT
    口径 (default filename); (256,256) is the AP-shape used by the frozen fp16 rewrite
    LUT -> suffixed filename so it never clobbers the default-shape ONNX."""
    ih, iw = int(input_hw[0]), int(input_hw[1])
    suffix = "" if (ih, iw) == (128, 256) else f"_ap{ih}x{iw}"
    onnx = MODELS_DIR / f"{label}_backbone{suffix}.onnx"
    if onnx.is_file() and onnx.stat().st_size > 0:
        return onnx
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    q = log_dir / f"{label}_export_queue.jsonl"
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text(json.dumps({
        "candidate_id": f"smbo:pyramid_lidar:w{'x'.join(map(str,width))}",
        "label": label, "width": list(int(x) for x in width)}) + "\n")
    rc = run([EXPORT_PY, "scripts/stage2_original60_export_onnx.py",
              "--candidate-queue", q, "--out-dir", MODELS_DIR,
              "--input-hw", f"{ih},{iw}",
              "--manifest-out", log_dir / f"{label}{suffix}_export_manifest.json"],
             log_dir / f"{label}{suffix}_export.log")
    if rc != 0 or not onnx.is_file():
        raise RuntimeError(f"export failed rc={rc} (see {log_dir}/{label}{suffix}_export.log)")
    return onnx


# ---- step 2: tune (MetaSchedule) ------------------------------------------
def tune(width, label, precision, onnx, gpu, log_dir) -> tuple[Path, Path]:
    work_dir = WORKDIRS / f"{label}_{precision}"
    wl = work_dir / "database_workload.json"
    tr = work_dir / "database_tuning_record.json"
    if wl.is_file() and tr.is_file():
        return wl, tr
    work_dir.mkdir(parents=True, exist_ok=True)
    plan = log_dir / f"{label}_{precision}_tune_plan.jsonl"
    plan.write_text(json.dumps({
        "gpu": int(gpu), "onnx_path": str(onnx), "tvm_work_dir": str(work_dir),
        "database_workload_path": str(wl), "database_tuning_record_path": str(tr),
        "label": label, "candidate_id": f"smbo:{label}:{precision}",
        "job_id": f"smbo_tune_{label}_{precision}", "width": list(int(x) for x in width),
    }) + "\n")
    rc = run([TVM_PY, "scripts/stage2_original60_tvm_artifact_worker.py",
              "--job-plan", plan, "--job-state", log_dir / f"{label}_{precision}_tune_state.json",
              "--manifest-out", log_dir / f"{label}_{precision}_tune_manifest.json",
              "--log-dir", log_dir, "--max-trials", MAX_TRIALS, "--seed", SEED],
             log_dir / f"{label}_{precision}_tune.log", env=tvm_env(gpu))
    if rc != 0 or not (wl.is_file() and tr.is_file()):
        raise RuntimeError(f"tune failed rc={rc} (see {log_dir}/{label}_{precision}_tune.log)")
    return wl, tr


# ---- steps 3/4: measure latency + energy ----------------------------------
def measure(kind, width, label, precision, onnx, gpu, log_dir):
    qpol, qmethod = QUANT[precision]
    work_dir = WORKDIRS / f"{label}_{precision}"
    run_id = f"smbo_{kind}_{label}_{precision}_gpu{gpu}"
    raw_root = RESULTS / f"{label}_{precision}"
    out_jsonl = raw_root / f"{kind}_row.jsonl"
    raw_root.mkdir(parents=True, exist_ok=True)
    cmd = [TVM_PY, "scripts/stage2_h800_run_measurement_job.py",
           "--kind", kind, "--label", label, "--gpu", int(gpu),
           "--onnx", onnx, "--work-dir", work_dir,
           "--width", ",".join(str(int(x)) for x in width),
           "--candidate-id", f"smbo:{label}:{precision}",
           "--software-point-id", f"smbo:pyramid_lidar:w{'x'.join(map(str,width))}",
           "--config-id-tuned", f"coverage_h800_tvm_pyramid_w{'x'.join(map(str,width))}_{precision}_tuned",
           "--config-id-default", f"coverage_h800_tvm_pyramid_w{'x'.join(map(str,width))}_{precision}_default",
           "--run-id", run_id, "--precision", precision,
           "--quant-policy", qpol, "--quant-method", qmethod,
           "--measurement-source", f"smbo_{kind}_h800_tvm",
           "--full-network-claim", "false",
           "--tune-budget", "remeasure_existing_ms_db",
           "--raw-root", raw_root, "--out-jsonl", out_jsonl]
    # both schedules' rows are written to out_jsonl; process may hang on teardown
    n_expected = 2 if kind == "latency" else 1
    ok = lambda: out_jsonl.is_file() and sum(1 for _ in open(out_jsonl)) >= n_expected
    rc = run(cmd, log_dir / f"{label}_{precision}_{kind}.log", env=tvm_env(gpu),
             timeout=1200, ok_check=ok)
    return rc, out_jsonl


def _rows(out_jsonl: Path):
    if not out_jsonl.is_file():
        return []
    return [json.loads(l) for l in open(out_jsonl) if l.strip()]


def parse_latency(out_jsonl: Path):
    rows = _rows(out_jsonl)
    if not rows:
        return None
    def us(policy):
        for r in rows:
            if r.get("schedule_policy") == policy and r.get("latency_p50_us"):
                return r["latency_p50_us"] / 1000.0
        return None
    return {"lat_default_ms": us("default"), "lat_tuned_ms": us("metaschedule_tuned")}


def parse_energy(out_jsonl: Path, lat_tuned_ms=None):
    """Energy per frame. The stock 'joule_per_inference' subtracts idle power, which
    for small/fast configs falls BELOW the H800 idle-power drift (~10W) -> ~0/negative
    (NOT a window-length problem: the window is already ~5min). The robust,
    always-defined metric = TOTAL power x latency = watt_avg * lat/1000 (no idle
    subtraction; physically the energy the dedicated device spends per frame)."""
    rows = _rows(out_jsonl)
    if not rows:
        return None
    r = rows[-1]  # tuned-schedule energy row
    marginal = next((r[k] for k in ("energy_joule_per_inference", "joule_per_inference")
                     if r.get(k) is not None), None)
    watt_avg = r.get("watt_avg") or r.get("watt_p50")
    lat = lat_tuned_ms if lat_tuned_ms is not None else (r.get("latency_p50_us") or 0) / 1000.0
    total = (watt_avg * lat / 1000.0) if (watt_avg and lat) else None
    return {"energy_j": total,                       # total-power metric (robust)
            "energy_marginal_j": marginal if (marginal and marginal > 0) else None,
            "energy_metric": "total_power_x_latency",
            "energy_measure_status": "ok" if total else "no_watt_avg",
            "watt_avg": watt_avg, "watt_p50": r.get("watt_p50")}


# int8 native route (real dp4a, direct topology, no MetaSchedule tune). Reads a
# 1-config completion queue; expects ONNX at H800_MODELS/<label>_backbone.onnx.
# ★口径: MUST use the PACK route (the exact version that built the stored original60
# int8 LUT). The repo copy is a regressed 632-line-different variant that builds a
# slower int8 kernel (s2_160: repo 11.44ms vs pack 8.75ms ≈ stored 8.70ms). Verified
# 2026-07-03.
_int8_route = os.environ.get("V2X_NATIVE_INT8_ROUTE", "").strip()
INT8_ROUTE = Path(_int8_route).expanduser() if _int8_route else None
H800_MODELS = Path(os.environ.get("V2X_NATIVE_INT8_MODEL_ROOT", MODELS_DIR)).expanduser()


# fp16 native route: TIR group-conv TensorCore rewrite (im2col/matmul/restore +
# selective MatmulTensorization -> WMMA), NOT MetaSchedule. Default fp16 lowering
# leaves the hot 3x3 group conv un-tensorized -> fp16≈fp32 (~47ms metaschedule was
# the DEPRECATED fp16_true path). This route reproduces the stored full60 LUT.
# ★口径: --cast-fp16-source is REQUIRED (wmma=0 without it); --route MUST stay
# apshape_tensorcore_rewritten_full60 to match the frozen fp16 LUT rows. Verified
# 2026-07-03: s2_160=[64,128,160] reproduces 11.66ms ≈ stored 11.642ms.
FP16_PROBE = REPO / "scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py"
FP16_MEASURE = REPO / "scripts/stage2_measure_fp16_rewritten_artifact.py"
FP16_ROUTE = "apshape_tensorcore_rewritten_full60"
FP16_FULL_REPS = 30
# ★口径 [2026-07-03 方案A 已裁决]: fp16 harmonized to 128x256 to MATCH fp32/int8.
# The frozen full60 fp16 LUT was built at AP-shape 256x256 (=11.64ms for s2_160) while
# fp32/int8 are at 128x256; mixing shapes inverts the fp16/int8 latency order and
# corrupts the joint Pareto. Now all three precisions export at 128x256. The old
# 256x256 fp16 LUT is archived under backup_fp16_apshape256_preharmonize_20260703/.
# (set to (256,256) only to reproduce the archived frozen number.)
FP16_INPUT_HW = (128, 256)


def _int8_rows(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).is_file() else []


def measure_config_int8(width, gpu, log_dir, do_export=True):
    """int8 latency+energy via native_int8_full_onnx_route (real dp4a)."""
    label = label_of(width)
    out = {"schema": "measure_config_int8_v1", "generated_at": utc(),
           "width": list(int(x) for x in width), "precision": "int8", "label": label,
           "gpu": int(gpu), "latency_kind": "pyramid_backbone_subnet",
           "quant_method": "h800_tvm_native_int8_backbone_subnet", "build_success": False}
    try:
        if INT8_ROUTE is None or not INT8_ROUTE.is_file():
            raise RuntimeError(
                "set V2X_NATIVE_INT8_ROUTE to the validated native-INT8 route script"
            )
        onnx = export_onnx(width, label, log_dir) if do_export else MODELS_DIR / f"{label}_backbone.onnx"
        H800_MODELS.mkdir(parents=True, exist_ok=True)
        dst = H800_MODELS / f"{label}_backbone.onnx"        # route convention path
        if not dst.exists() or dst.stat().st_size != onnx.stat().st_size:
            import shutil; shutil.copy(onnx, dst)
        d = RESULTS / f"{label}_int8"; d.mkdir(parents=True, exist_ok=True)
        q = d / "queue.jsonl"
        q.write_text(json.dumps({
            "job_id": f"smbo:int8:{label}", "label": label, "width": list(int(x) for x in width),
            "precision": "int8", "onnx_backbone_path": str(dst), "workdir": str(d / "work"),
            "schema": "original60_fp16_int8_completion_job_v1",
            "latency_status": "pending_measurement", "energy_status": "pending_measurement",
            "required_actions": ["run_native_int8_full_onnx_latency",
                                 "run_native_int8_full_onnx_energy"]}) + "\n")
        lat_out, en_out = d / "lat.jsonl", d / "energy.jsonl"
        ok = lambda: lat_out.is_file() and en_out.is_file() and _int8_rows(lat_out) and _int8_rows(en_out)
        rc = run([TVM_PY, INT8_ROUTE, "--gpu", int(gpu), "--completion-queue", q,
                  "--latency-rows-output", lat_out, "--energy-rows-output", en_out,
                  "--labels", label, "--number", 100, "--repeat", 5, "--energy-iters", 300],
                 log_dir / f"{label}_int8_route.log", env=tvm_env(gpu), timeout=1800, ok_check=ok)
        out["route_rc"] = rc
        lr, er = _int8_rows(lat_out), _int8_rows(en_out)
        if lr:
            out["lat_tuned_ms"] = lr[-1].get("latency_p50_us", 0) / 1000.0
        if er:
            wa = er[-1].get("watt_avg") or er[-1].get("watt_p50")
            out["watt_avg"] = wa
            out["energy_marginal_j"] = er[-1].get("joule_per_inference")
            if wa and out.get("lat_tuned_ms"):
                out["energy_j"] = wa * out["lat_tuned_ms"] / 1000.0   # total-power metric
                out["energy_metric"] = "total_power_x_latency"
        out["build_success"] = bool(out.get("lat_tuned_ms"))
    except Exception as e:
        out["error"] = repr(e)
    json.dump(out, open(RESULTS / f"{label}_int8" / "measure_config_result.json", "w"),
              ensure_ascii=False, indent=1)
    return out


def _fp16_rows(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).is_file() else []


def measure_config_fp16(width, gpu, log_dir, do_export=True, rewrite_dtype="float16"):
    """fp16 / int8 latency+energy via full-engine group-conv TensorCore rewrite.

    rewrite_dtype="float16" -> WMMA (fp16 tensor cores). rewrite_dtype="int8" ->
    int8xint8->int32 MMA (MatmulInt8Tensorization); fixes the precision-axis
    optimization-level unfairness where the old int8 pack route ran naive SIMT.
    2-step recipe (NOT metaschedule): (1) probe --mode full-engine-group-conv-rewrite
    [--rewrite-dtype int8] -> rewritten_full_engine.so + report; (2) measure the
    exported artifact's latency+energy. Same 128x256 export ONNX as fp32 -> 口径-aligned.
    """
    label = label_of(width)
    is_int8 = rewrite_dtype == "int8"
    tag = "int8tc" if is_int8 else "fp16"
    prec = "int8_tc" if is_int8 else "fp16"
    qm = ("h800_tvm_int8_rewritten_tensorcore" if is_int8
          else "h800_tvm_fp16_rewritten_tensorcore")
    route = "int8_tensorcore_rewritten_full60" if is_int8 else FP16_ROUTE
    out = {"schema": f"measure_config_{tag}_v1", "generated_at": utc(),
           "width": list(int(x) for x in width), "precision": prec, "label": label,
           "gpu": int(gpu), "latency_kind": "pyramid_backbone_subnet",
           "quant_method": qm, "route": route,
           "input_hw": list(FP16_INPUT_HW), "build_success": False}
    try:
        ih, iw = FP16_INPUT_HW
        # mirror export_onnx's filename rule: default (128,256) has no suffix.
        _sfx = "" if (ih, iw) == (128, 256) else f"_ap{ih}x{iw}"
        ap_onnx = MODELS_DIR / f"{label}_backbone{_sfx}.onnx"
        onnx = export_onnx(width, label, log_dir, input_hw=FP16_INPUT_HW) if do_export else ap_onnx
        d = RESULTS / f"{label}_{tag}"; d.mkdir(parents=True, exist_ok=True)
        raw_dir = d / "rewrite_raw"; export_dir = d / "rewrite_exports"
        # probe writes the report here (filename is "fp16_"-prefixed regardless of dtype);
        # the .so lands in a probe-chosen subdir -> derive from report export_library.path.
        report = export_dir / f"fp16_{label}_full_engine_group_conv_rewrite_latest.json"
        rw_cmd = [TVM_PY, FP16_PROBE, "--mode", "full-engine-group-conv-rewrite",
                  "--onnx", onnx, "--cast-fp16-source", "--label", label,
                  "--gpu", int(gpu), "--raw-dir", raw_dir, "--export-dir", export_dir,
                  "--full-reps", FP16_FULL_REPS]
        if is_int8:
            rw_cmd += ["--rewrite-dtype", "int8"]
        # step 1: full-engine TensorCore rewrite (idempotent). --cast-fp16-source mandatory.
        if not report.is_file():
            ok_rw = lambda: report.is_file()
            rc_rw = run(rw_cmd, log_dir / f"{label}_{tag}_rewrite.log", env=tvm_env(gpu),
                        timeout=1800, ok_check=ok_rw)
            out["rewrite_rc"] = rc_rw
        if not report.is_file():
            raise RuntimeError(f"{tag} rewrite produced no report (see {log_dir}/{label}_{tag}_rewrite.log)")
        rep = json.load(open(report))
        rw = rep.get("rewritten_full_engine", {}) or {}
        exp_lib = rw.get("export_library") or {}
        artifact = Path(exp_lib.get("path", "")) if exp_lib.get("status") == "success" else Path("")
        if not artifact.is_file():
            raise RuntimeError(f"{tag} rewrite exported no artifact (export_library={exp_lib})")
        counts = rw.get("scheduled_counts") or {}
        out["tensorcore_gate"] = bool(rw.get("tensorcore_gate"))
        out["wmma_count"] = counts.get("wmma")
        out["mma_sync_count"] = counts.get("tvm_mma_sync")
        out["rewrite_latency_ms"] = rw.get("latency_mean_ms")
        cmp0 = (rw.get("output_compare") or [{}])[0]
        out["rewrite_max_abs_err"] = cmp0.get("max_abs_err")
        if not out["tensorcore_gate"]:  # honest feasibility signal: rewrite didn't tensorize
            out["warning"] = f"tensorcore_gate=False (wmma=0): {tag} fell back to non-TensorCore path"
        # step 2: measure exported artifact (latency then energy)
        lat_out, en_out = d / "lat.jsonl", d / "energy.jsonl"
        wstr = ",".join(str(int(x)) for x in width)
        common = [TVM_PY, FP16_MEASURE, "--label", label, "--gpu", int(gpu),
                  "--artifact", artifact, "--rewrite-report", report,
                  "--route", route, "--width", wstr, "--config-id", f"{tag}_{label}"]
        ok_l = lambda: bool(_fp16_rows(lat_out))
        rc_l = run(common + ["--kind", "latency", "--run-id", f"smbo_{tag}_lat_{label}",
                             "--raw-root", d, "--out-jsonl", lat_out,
                             "--warmup-iters", 20, "--measure-iters", 300, "--repeat", 5],
                   log_dir / f"{label}_{tag}_latency.log", env=tvm_env(gpu),
                   timeout=1200, ok_check=ok_l)
        out["latency_rc"] = rc_l
        lr = _fp16_rows(lat_out)
        if lr:
            out["lat_tuned_ms"] = lr[-1].get("latency_p50_us", 0) / 1000.0
        ok_e = lambda: bool(_fp16_rows(en_out))
        rc_e = run(common + ["--kind", "energy", "--latency-run-id", f"smbo_{tag}_lat_{label}",
                             "--run-id", f"smbo_{tag}_en_{label}", "--raw-root", d, "--out-jsonl", en_out,
                             "--energy-warmup-iters", 20, "--energy-measure-iters", 300,
                             "--energy-min-active-s", 5, "--energy-sync-interval-iters", 50],
                   log_dir / f"{label}_{tag}_energy.log", env=tvm_env(gpu),
                   timeout=1200, ok_check=ok_e)
        out["energy_rc"] = rc_e
        er = _fp16_rows(en_out)
        if er:
            wa = er[-1].get("watt_avg") or er[-1].get("watt_p50")
            out["watt_avg"] = wa
            out["watt_p50"] = er[-1].get("watt_p50")
            out["energy_marginal_j"] = er[-1].get("joule_per_inference")
            if wa and out.get("lat_tuned_ms"):
                out["energy_j"] = wa * out["lat_tuned_ms"] / 1000.0   # total-power metric (口径同 fp32/int8)
                out["energy_metric"] = "total_power_x_latency"
        out["build_success"] = bool(out.get("lat_tuned_ms"))
    except Exception as e:
        out["error"] = repr(e)
    rdir = RESULTS / f"{label}_{tag}"
    rdir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(rdir / "measure_config_result.json", "w"),
              ensure_ascii=False, indent=1)
    return out


def measure_config(width, precision, gpu, log_dir=None, do_export=True, do_tune=True):
    label = label_of(width)
    log_dir = Path(log_dir) if log_dir else (RUN_ROOT / "logs" / f"{label}_{precision}")
    log_dir.mkdir(parents=True, exist_ok=True)
    if precision == "int8":
        raise ValueError(
            "formal int8 is not available through legacy measure_config; "
            "dispatch via stage5_route_b_int8_auto_decomp"
        )
    if precision == "int8_legacy_dp4a":
        return measure_config_int8(width, gpu, log_dir, do_export=do_export)
    if precision == "fp16":
        return measure_config_fp16(width, gpu, log_dir, do_export=do_export)
    if precision in ("int8_tc", "int8tc"):
        return measure_config_fp16(width, gpu, log_dir, do_export=do_export, rewrite_dtype="int8")
    out = {"schema": "measure_config_v1", "generated_at": utc(),
           "width": list(int(x) for x in width), "precision": precision,
           "label": label, "gpu": int(gpu), "max_trials": MAX_TRIALS,
           "latency_kind": "pyramid_backbone_subnet", "build_success": False}
    try:
        onnx = MODELS_DIR / f"{label}_backbone.onnx"
        if do_export:
            onnx = export_onnx(width, label, log_dir)
        out["onnx"] = str(onnx)
        if do_tune:
            tune(width, label, precision, onnx, gpu, log_dir)
        rc_l, jl = measure("latency", width, label, precision, onnx, gpu, log_dir)
        lat = parse_latency(jl)
        out["latency_rc"] = rc_l
        if lat:
            out.update(lat)
        rc_e, je = measure("energy", width, label, precision, onnx, gpu, log_dir)
        eng = parse_energy(je, lat_tuned_ms=(lat or {}).get("lat_tuned_ms"))
        out["energy_rc"] = rc_e
        if eng:
            out.update(eng)
        out["build_success"] = bool(lat and lat.get("lat_tuned_ms"))
    except Exception as e:  # feasibility signal: build/tune failure captured, not swallowed
        out["error"] = repr(e)
        out["build_success"] = False
    (RESULTS / f"{label}_{precision}").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RESULTS / f"{label}_{precision}" / "measure_config_result.json", "w"),
              ensure_ascii=False, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", required=True, help="w0,w1,w2")
    ap.add_argument(
        "--precision",
        default="fp16",
        choices=["fp32", "fp16", "int8", "int8_tc", "int8_legacy_dp4a"],
    )
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--no-export", action="store_true", help="reuse existing ONNX")
    ap.add_argument("--no-tune", action="store_true", help="reuse existing DB")
    args = ap.parse_args()
    width = [int(x) for x in args.width.replace("x", ",").split(",")]
    out = measure_config(width, args.precision, args.gpu,
                         do_export=not args.no_export, do_tune=not args.no_tune)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if out["build_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
