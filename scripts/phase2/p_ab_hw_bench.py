"""Task #8 (A) + #10 (B) hw — forced-INT8 / head-INT8 collab2 lat+energy+size+GMAC.

#8 A: {cliff2_c,prune90,prune95}_c_all_int8_forced (+ cliff2_c fp16 if present)
#10 B: {base,pruned50}_headINT8_restFP16

口径: body_subnet_collab2 (2x64x128x256 + t_ego). GPU 7/0 clean. CUDA-Event lat
(m4_9 benchmark_engine) + e5 collab2 energy + nominal GMAC (ONNX conv FLOP).
These are ablation/predictor points (regime=ablation_guardrail_off, NOT frontier).
★ watch: prune90[6,13,26]/prune95[4,6,13] tiny non-32 channels may hit the same
  kernel-selection cliff as p25 -> if GMAC/ms far below scale expectation, flag.
Output: results/P_ab_hw_bench.csv
"""
import os, sys, csv
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("HW_GPU", "7"))
sys.path.insert(0, str(REPO / "scripts" / "phase1"))
sys.path.insert(0, str(REPO / "scripts" / "phase2"))
import pynvml, torch
import onnx
from onnx import shape_inference, numpy_helper
from m4_9_bench_perstage_v2 import benchmark_engine
from e4_energy_bench import load_engine, gpu_status, measure_idle_power
from e5_collab2_energy_bench import measure_engine_energy_collab2

# (task, config, engine_rel, onnx_rel, precision, regime)
PRE = "models/perstage_quant_ap_cache_v2"
SA = "models/stage_a_cache"
TARGETS = [
    ("A", "cliff2_c_forced_int8", f"{PRE}/cliff2_c_c_all_int8_forced.engine", f"{SA}/cliff2_c.onnx", "INT8", "ablation_guardrail_off"),
    ("A", "prune90_forced_int8", f"{PRE}/prune90_c_all_int8_forced.engine", f"{SA}/prune90.onnx", "INT8", "ablation_guardrail_off"),
    ("A", "prune95_forced_int8", f"{PRE}/prune95_c_all_int8_forced.engine", f"{SA}/prune95.onnx", "INT8", "ablation_guardrail_off"),
    ("A", "cliff2_c_fp16", "models/pyramid_dair_m1_cliff2_c_collab_n2_fp16.engine", f"{SA}/cliff2_c.onnx", "FP16", "ablation_fp16_ref"),
    ("B", "base_headINT8_restFP16", f"{PRE}/base_headINT8_restFP16.engine", f"{SA}/base.onnx", "MIXED", "ablation_guardrail_off"),
    ("B", "p50_headINT8_restFP16", f"{PRE}/pruned50_headINT8_restFP16.engine", f"{SA}/pruned50.onnx", "MIXED", "ablation_guardrail_off"),
]


def gmac(onnx_path):
    m = shape_inference.infer_shapes(onnx.load(onnx_path))
    vi = {v.name: v for v in list(m.graph.value_info)+list(m.graph.output)+list(m.graph.input)}
    init = {i.name: i for i in m.graph.initializer}
    def shp(n): return [d.dim_value for d in vi[n].type.tensor_type.shape.dim] if n in vi else None
    tot = 0.0
    for nd in m.graph.node:
        if nd.op_type == "Conv":
            w = init.get(nd.input[1])
            if w is None: continue
            ws = list(numpy_helper.to_array(w).shape); o = shp(nd.output[0])
            if o is None or len(o) < 4: continue
            tot += ws[0]*ws[1]*ws[2]*ws[3]*o[2]*o[3]
    return tot/1e9


def main():
    phys = int(os.environ.get("HW_GPU", "7"))
    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(phys)
    u, m, o = gpu_status(h)
    if u > 2 or o > 50:
        print(f"ABORT GPU{phys} not idle util={u}% other={o:.0f}MiB"); return 2
    print(f"[gate] GPU{phys} idle util={u}% other={o:.0f}MiB")
    torch.cuda.set_device(0)
    idle_w, _ = measure_idle_power(h, 5.0)
    print(f"[idle] {idle_w:.1f}W")
    rows = []
    for task, cfg, erel, orel, prec, regime in TARGETS:
        ep = REPO / erel
        if not ep.exists():
            print(f"[skip] {cfg} missing {erel}");
            rows.append({"task": task, "config": cfg, "status": "MISSING_ENGINE", "engine_path": erel})
            continue
        _, _, oo = gpu_status(h)
        if oo > 50: print(f"[ABORT] co-tenant {oo:.0f}MiB"); break
        g = gmac(str(REPO / orel))
        lat = benchmark_engine(str(ep), (2, 64, 128, 256), {"t_ego": (2, 2, 3)}, 200, 200)
        en = measure_engine_energy_collab2(load_engine(ep), h, hold_s=12)
        eff = g / lat["mean_ms"]  # GMAC/ms
        print(f"[{task}:{cfg}] lat_p50={lat['p50_ms']:.4f} p99={lat['p99_ms']:.4f}  "
              f"GMAC={g:.1f} eff={eff:.1f}GMAC/ms  J/frame={en['energy_per_frame_j']*1000:.1f}mJ "
              f"size={lat['engine_size_mb']:.2f}MB")
        rows.append({
            "task": task, "config": cfg, "precision": prec, "regime": regime,
            "latency_kind": "body_subnet_collab2",
            "lat_p50_ms": round(lat["p50_ms"], 4), "lat_p99_ms": round(lat["p99_ms"], 4),
            "lat_mean_ms": round(lat["mean_ms"], 4),
            "throughput_fps": round(1000.0/lat["mean_ms"], 1),
            "nominal_gmac": round(g, 3), "eff_gmac_per_ms": round(eff, 2),
            "mean_power_w": round(en["mean_power_w"], 2), "idle_power_w": round(idle_w, 2),
            "energy_per_frame_mj": round(en["energy_per_frame_j"]*1000, 4),
            "perf_per_watt_fps_per_w": round(en["perf_per_watt_fps_per_w"], 2),
            "engine_size_mb": round(lat["engine_size_mb"], 3),
            "engine_path": erel, "gpu": phys, "status": "OK",
            "source": "p_ab_hw_bench;lat:m4_9;energy:e5;CUDA-Event;NVML-idle-excl;collab2",
        })
    pynvml.nvmlShutdown()
    out = REPO / "results/P_ab_hw_bench.csv"
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\n[done] {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
