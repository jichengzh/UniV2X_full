"""E6 — Orin AGX energy bench via tegrastats power rails (DRAFT, UNTESTED).

Mirrors the E4/E5 (4090 NVML) methodology on Jetson Orin AGX, where NVML's
nvmlDeviceGetPowerUsage / nvmlDeviceGetTotalEnergyConsumption are NOT available
(Tegra has no NVML power domain). Instead we sample on-board INA3221 rails
exposed by `tegrastats`.

============================ STATUS: DRAFT / UNTESTED ==========================
Written on the 4090 host (no Orin SSH yet). MUST be validated on the box before
any number is reported as "real":
  1. Run `tegrastats --interval 50` on Orin, capture ~10 lines, CONFIRM the
     exact rail token names. On Orin AGX (jetson_clocks/MAXN) the GPU rail is
     usually `VDD_GPU_SOC`, CPU rail `VDD_CPU_CPU`, module total `VDD_IN`
     (older BSP) or `VIN_SYS_5V0`. The regex CONFIG below assumes AGX rail
     names — adjust RAILS to match the real output.
  2. trtexec path = /usr/src/tensorrt/bin/trtexec (TRT 8.5.2.2), venv =
     /home/jichengzhi/uniad_venv/.
  3. Until run + cross-checked on Orin, every output row is `is_real=False`.
===============================================================================

CROSS-PLATFORM FAIRNESS CAVEAT (must document on every comparison vs 4090):
  4090 `nvmlDeviceGetPowerUsage` = WHOLE-BOARD power (GPU die + GDDR6X + VRM +
  fans). The closest Orin analogue is the MODULE-TOTAL rail (VDD_IN / VIN_SYS),
  NOT the GPU-only rail. So we report BOTH:
    - energy_per_frame_gpu_mj   (VDD_GPU_SOC only)      -> GPU-die comparison
    - energy_per_frame_module_mj(module-total rail)     -> board-vs-board, the
      fair counterpart to 4090 NVML for "Orin perf/watt dominates 4090" claims.
  Never compare Orin GPU-rail J/frame against 4090 board J/frame.

METHODOLOGY (verbatim parallel to E4):
  - idle subtraction: sample rails 5 s with no workload -> idle_power per rail.
  - sustained load: drive `trtexec --loadEngine --duration=hold_s` (trtexec
    keeps the engine resident & GPU saturated; robust on TRT 8.5, avoids
    Python-API version pitfalls). Sample tegrastats concurrently.
  - steady-state mean: drop first 10 % of load-phase samples.
  - latency p50 + throughput come from trtexec's own report (GPU Compute Time
    median ms, Throughput qps) — same instrument used for Orin D-space bench.
  - energy_per_frame = P_mean_rail * lat_p50 (per ONE forward).

Latency口径: this measures whatever the engine is. For collab2 alignment the
caller must pass a collab2 (2x64x128x256) engine; the口径 is then
`body_subnet_collab2` (NOT e2e — no voxelize/NMS).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ---- Rail name CONFIG (★ VERIFIED on-box 2026-06-03, Orin AGX 172.16.62.222) --
# CRITICAL: tegrastats shows power rails ONLY when run with sudo (root). Non-sudo
# tegrastats omits all mW tokens. INA3221 sysfs (1-0040: VDD_GPU_SOC/VDD_CPU_CV/
# VIN_SYS_5V0) also requires root. Verified line fragment (sudo):
#   ... VDD_GPU_SOC 3212mW/3212mW VDD_CPU_CV 401mW/401mW VIN_SYS_5V0 4543mW/4543mW
#       VDDQ_VDD2_1V8AO 504mW/504mW
# token format: <RAIL> <inst>mW/<avg>mW   (we read the INSTANTANEOUS value)
RAILS = {
    "gpu": ["VDD_GPU_SOC"],            # GPU+SOC compute rail
    "cpu": ["VDD_CPU_CV"],             # CPU rail (NOT VDD_CPU_CPU on this BSP)
    "module_total": ["VIN_SYS_5V0"],   # board-input rail = fair counterpart to 4090 NVML board power
    "mem": ["VDDQ_VDD2_1V8AO"],        # LPDDR5 rail (on 1-0041)
}
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
# tegrastats power needs root -> prefix with sudo (-S reads pw from stdin).
TEGRASTATS_CMD = ["sudo", "-S", "-p", "", "tegrastats", "--interval"]
SUDO_PW = os.environ.get("ORIN_SUDO_PW", "")  # pass via env, never hardcode


def _rail_regex(names):
    # match  NAME 1234mW/1190mW  -> capture instantaneous 1234
    alt = "|".join(re.escape(n) for n in names)
    return re.compile(rf"(?:{alt})\s+(\d+)mW/\d+mW")


class TegrastatsSampler(threading.Thread):
    """Background tegrastats parser. Collects per-rail instantaneous mW."""

    def __init__(self, interval_ms=50):
        super().__init__(daemon=True)
        self.interval_ms = interval_ms
        self._stop = threading.Event()
        self.samples = {k: [] for k in RAILS}      # rail -> list[watts]
        self._res = {k: _rail_regex(v) for k, v in RAILS.items()}
        self._proc = None

    def run(self):
        self._proc = subprocess.Popen(
            TEGRASTATS_CMD + [str(self.interval_ms)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
        )
        if SUDO_PW:  # feed sudo password once on stdin
            try:
                self._proc.stdin.write(SUDO_PW + "\n")
                self._proc.stdin.flush()
            except Exception:
                pass
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            for rail, rgx in self._res.items():
                m = rgx.search(line)
                if m:
                    self.samples[rail].append(int(m.group(1)) / 1000.0)  # W

    def stop_sampling(self):
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()


def _steady_mean(arr):
    a = np.array(arr, dtype=float)
    if a.size == 0:
        return float("nan"), 0
    a = a[a.size // 10:] if a.size > 20 else a
    return float(a.mean()), int(a.size)


def measure_idle(secs=5.0):
    s = TegrastatsSampler()
    s.start()
    time.sleep(secs)
    s.stop_sampling()
    s.join(timeout=2)
    return {k: _steady_mean(v)[0] for k, v in s.samples.items()}


def _parse_trtexec(out: str):
    """Pull median GPU compute time (ms) + throughput (qps) from trtexec log."""
    lat_p50 = None
    tput = None
    # "GPU Compute Time: ... median = 6.18 ms ..."
    m = re.search(r"GPU Compute Time:.*?median\s*=\s*([\d.]+)\s*ms", out, re.S)
    if m:
        lat_p50 = float(m.group(1))
    m = re.search(r"Throughput:\s*([\d.]+)\s*qps", out)
    if m:
        tput = float(m.group(1))
    return lat_p50, tput


def measure_engine_energy(engine_path: str, hold_s=12.0, dla_core=None,
                          use_int8=False, use_fp16=True):
    """Drive trtexec on the engine while sampling tegrastats."""
    sampler = TegrastatsSampler()
    sampler.start()
    time.sleep(0.3)  # let first lines arrive

    cmd = [TRTEXEC, f"--loadEngine={engine_path}",
           f"--duration={hold_s:.0f}", "--warmUp=2000", "--avgRuns=200",
           "--useSpawnedThread"]
    # NOTE: precision/DLA are baked into the prebuilt engine; flags here are
    # only for documentation of the route. Re-building is the build step's job.
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0

    sampler.stop_sampling()
    sampler.join(timeout=2)

    lat_p50, tput = _parse_trtexec(proc.stdout + proc.stderr)
    rail_mean = {k: _steady_mean(v)[0] for k, v in sampler.samples.items()}
    rail_n = {k: _steady_mean(v)[1] for k, v in sampler.samples.items()}

    lat_s = (lat_p50 or float("nan")) / 1000.0
    return {
        "lat_p50_ms": lat_p50,
        "throughput_qps": tput,
        "wall_s": wall,
        "power_gpu_w": rail_mean["gpu"],
        "power_cpu_w": rail_mean["cpu"],
        "power_module_w": rail_mean["module_total"],
        "n_samples_gpu": rail_n["gpu"],
        "energy_per_frame_gpu_mj": (rail_mean["gpu"] * lat_s * 1000.0
                                    if lat_p50 else None),
        "energy_per_frame_module_mj": (rail_mean["module_total"] * lat_s * 1000.0
                                       if lat_p50 else None),
        "perf_per_watt_module_fps_per_w": (tput / rail_mean["module_total"]
                                           if tput and rail_mean["module_total"]
                                           else None),
        "trtexec_ok": lat_p50 is not None,
        "stderr_tail": proc.stderr[-300:] if proc.returncode else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="path to prebuilt .engine")
    ap.add_argument("--hold_s", type=float, default=12.0)
    ap.add_argument("--latency_kind", default="body_subnet_collab2",
                    help="document口径 of this engine (collab2 to align主表)")
    ap.add_argument("--route", default="GPU", help="GPU / DLA0 / DLA1 (doc only)")
    ap.add_argument("--output", default="results/E6_orin_energy.csv")
    args = ap.parse_args()

    if not Path(TRTEXEC).exists():
        print(f"ABORT: trtexec not at {TRTEXEC} (run this ON Orin)")
        return 2

    print("[E6] measuring idle rails (5s, no workload) ...")
    idle = measure_idle(5.0)
    print(f"  idle  gpu={idle['gpu']:.2f}W  cpu={idle['cpu']:.2f}W  "
          f"module={idle['module_total']:.2f}W")
    if np.isnan(idle["gpu"]):
        print("ABORT: no GPU-rail samples parsed — RAILS names wrong for this "
              "BSP. Capture `tegrastats` output and fix RAILS at top of script.")
        return 3

    print(f"[E6] load phase on {Path(args.engine).name} (route={args.route}) ...")
    r = measure_engine_energy(args.engine, hold_s=args.hold_s)
    if not r["trtexec_ok"]:
        print(f"ABORT: trtexec failed / no latency parsed. tail:\n{r['stderr_tail']}")
        return 4

    print(f"  lat_p50={r['lat_p50_ms']:.3f}ms  tput={r['throughput_qps']:.0f}qps "
          f"P_gpu={r['power_gpu_w']:.2f}W  P_module={r['power_module_w']:.2f}W")
    print(f"  J/frame  gpu-rail={r['energy_per_frame_gpu_mj']:.3f}mJ  "
          f"module-total={r['energy_per_frame_module_mj']:.3f}mJ")

    import csv
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "engine": Path(args.engine).name, "route": args.route,
        "latency_kind": args.latency_kind,
        "lat_p50_ms": r["lat_p50_ms"], "throughput_qps": r["throughput_qps"],
        "power_gpu_w": round(r["power_gpu_w"], 2),
        "power_cpu_w": round(r["power_cpu_w"], 2),
        "power_module_w": round(r["power_module_w"], 2),
        "idle_gpu_w": round(idle["gpu"], 2),
        "idle_module_w": round(idle["module_total"], 2),
        "energy_per_frame_gpu_mj": round(r["energy_per_frame_gpu_mj"], 4),
        "energy_per_frame_module_mj": round(r["energy_per_frame_module_mj"], 4),
        "perf_per_watt_module_fps_per_w": (
            round(r["perf_per_watt_module_fps_per_w"], 3)
            if r["perf_per_watt_module_fps_per_w"] else None),
        "n_samples_gpu": r["n_samples_gpu"],
        "source": "E6_orin_tegrastats;trtexec-driven;idle-baseline",
        "note": ("Orin AGX. GPU rail = VDD_GPU_SOC; module rail compares to "
                 "4090 NVML board power. energy=P_mean*lat_p50. VERIFY rail "
                 "names on-box before trusting."),
    }
    write_header = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"appended 1 row -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
