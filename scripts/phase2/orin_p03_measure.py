"""P0-3 Orin measure (runs ON Orin, TRT 8.5, uniad_venv) — POST-BUILD only.

For each built engine, in ONE pass on the idle GPU:
  1. layer precision dump (IEngineInspector tactic token -> int8/fp16/fp32 count
     + per-layer precision set for the cross-TRT ap_valid gate);
  2. latency = trtexec "GPU Compute Time" median (body_subnet_collab2_orin口径);
  3. energy = sudo tegrastats VIN_SYS_5V0(module) + VDD_GPU_SOC, steady-state,
     J/frame = P_module * (1/throughput).

★ DISCIPLINE: only run when the build is DONE and GPU is idle (GR3D 0%).
   trtexec drives the GPU; we never run this during a build.

Output: /home/jichengzhi/m4_8_orin/p03_build/orin_p03_measure.csv
        + per-engine layer json under same dir.
Run:
  cd /home/jichengzhi/m4_8_orin/p03_build
  ORIN_SUDO_PW=<pw> /home/jichengzhi/uniad_venv/bin/python orin_p03_measure.py
"""
import os, re, json, csv, subprocess, time
from pathlib import Path
import tensorrt as trt

HERE = Path("/home/jichengzhi/m4_8_orin/p03_build")
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
PW = os.environ.get("ORIN_SUDO_PW", "")
LOG = trt.Logger(trt.Logger.ERROR)

# rails verified on-box 2026-06-03
R_GPU = re.compile(r"VDD_GPU_SOC (\d+)mW")
R_MOD = re.compile(r"VIN_SYS_5V0 (\d+)mW")
R_CPU = re.compile(r"VDD_CPU_CV (\d+)mW")

ENGINES = [  # (config, precision, engine_file)
    ("base", "fp16", "base_fp16_orin.engine"),
    ("base", "int8", "base_int8_orin.engine"),
    ("p50", "fp16", "p50_fp16_orin.engine"),
    ("p50", "int8", "p50_int8_orin.engine"),
    ("p75", "fp16", "p75_fp16_orin.engine"),
    ("p75", "int8", "p75_int8_orin.engine"),
]


def prec_token(tactic_name):
    t = (tactic_name or "").lower()
    if "i8i8" in t or "imma" in t or "_i8_" in t:
        return "int8"
    if "h884" in t or "h1688" in t or "hmma" in t or "f16f16" in t or "_h_" in t:
        return "fp16"
    if "f32f32" in t or "f32" in t:
        return "fp32"
    return "other"


def dump_layers(engine_path):
    with open(engine_path, "rb") as f:
        eng = trt.Runtime(LOG).deserialize_cuda_engine(f.read())
    insp = eng.create_engine_inspector()
    d = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = d.get("Layers", []) if isinstance(d, dict) else []
    counts = {"int8": 0, "fp16": 0, "fp32": 0, "other": 0}
    per_layer = {}
    has_scale_field = False
    dict_layers = [l for l in layers if isinstance(l, dict)]
    if not dict_layers:
        # TRT 8.5 non-DETAILED engine: Layers are bare name strings, no tactic.
        # Layer-precision gate needs a DETAILED rebuild; mark unavailable here.
        return counts, {"_note": "no_detailed_verbosity_layers_are_strings"}, False, []
    for l in dict_layers:
        nm = str(l.get("Name", ""))
        p = prec_token(str(l.get("TacticName", "")))
        counts[p] += 1
        if "conv" in nm.lower():
            per_layer[nm[:70]] = {"prec": p, "tactic": str(l.get("TacticName", ""))[:60]}
        for k in l:
            if "scale" in k.lower() or "dynamicrange" in k.lower() or "range" in k.lower():
                has_scale_field = True
    return counts, per_layer, has_scale_field, list(dict_layers[0].keys())


def parse_trtexec(out):
    lat = None; tput = None
    m = re.search(r"GPU Compute Time:.*?median = ([\d.]+) ms", out, re.S)
    if m: lat = float(m.group(1))
    m = re.search(r"Throughput: ([\d.]+) qps", out)
    if m: tput = float(m.group(1))
    return lat, tput


def measure_one(engine_path, hold_s=12):
    log = HERE / "tegra_meas.log"
    if log.exists(): log.unlink()
    # start sudo tegrastats -> logfile
    teg = subprocess.Popen(
        ["sudo", "-S", "-p", "", "tegrastats", "--interval", "50",
         "--logfile", str(log)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, text=True)
    if PW:
        try: teg.stdin.write(PW + "\n"); teg.stdin.flush()
        except Exception: pass
    time.sleep(0.5)
    r = subprocess.run([TRTEXEC, f"--loadEngine={engine_path}",
                        f"--duration={hold_s}", "--warmUp=2000", "--avgRuns=200"],
                       capture_output=True, text=True)
    subprocess.run(["sudo", "-S", "-p", "", "pkill", "-f", "tegrastats"],
                   input=(PW + "\n"), text=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    lat, tput = parse_trtexec(r.stdout + r.stderr)
    # parse power log (steady = drop first 20%)
    g = []; v = []; c = []
    try:
        txt = subprocess.run(["sudo", "-S", "-p", "", "cat", str(log)],
                             input=(PW + "\n"), text=True,
                             capture_output=True).stdout
    except Exception:
        txt = ""
    for ln in txt.splitlines():
        mg = R_GPU.search(ln); mv = R_MOD.search(ln); mc = R_CPU.search(ln)
        if mg: g.append(int(mg.group(1)))
        if mv: v.append(int(mv.group(1)))
        if mc: c.append(int(mc.group(1)))
    def steady(x):
        x = x[len(x)//5:]; return sum(x)/len(x)/1000.0 if x else float("nan")
    return lat, tput, steady(g), steady(v), steady(c), len(v)


def main():
    rows = []
    for cfg, prec, ef in ENGINES:
        p = HERE / ef
        if not p.exists():
            print(f"[skip] {ef} missing"); continue
        counts, per_layer, has_scale, keys = dump_layers(str(p))
        (HERE / f"layers_{cfg}_{prec}.json").write_text(
            json.dumps({"counts": counts, "conv_layers": per_layer,
                        "has_scale_field": has_scale, "info_keys": keys}, indent=2))
        lat, tput, pg, pv, pc, ns = measure_one(str(p))
        jpf = pv * (1.0/tput) * 1000 if (tput and pv == pv) else None  # mJ
        print(f"[{cfg}_{prec}] lat={lat}ms tput={tput}qps  P_mod={pv:.2f}W "
              f"J/frame={jpf:.1f}mJ  int8L={counts['int8']} fp16L={counts['fp16']} "
              f"fp32L={counts['fp32']}  scale_field={has_scale}")
        rows.append({
            "config": cfg, "precision": prec, "engine": ef,
            "power_mode": "MODE_30W", "latency_kind": "body_subnet_collab2_orin",
            "lat_gpu_compute_median_ms": lat, "throughput_qps": tput,
            "power_gpu_soc_w": round(pg, 3), "power_module_vin_w": round(pv, 3),
            "power_cpu_cv_w": round(pc, 3),
            "energy_per_frame_module_mj": round(jpf, 3) if jpf else None,
            "perf_per_watt_module_fps_per_w": round(tput/pv, 4) if (tput and pv==pv) else None,
            "int8_layer_count": counts["int8"], "fp16_layer_count": counts["fp16"],
            "fp32_layer_count": counts["fp32"], "other_layer_count": counts["other"],
            "scale_field_available": has_scale,
            "n_power_samples": ns,
            "source": "orin_p03_measure;trtexec-GPU;sudo-tegrastats-VIN_SYS;IEngineInspector",
        })
    out = HERE / "orin_p03_measure.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["config"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\n[done] {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
