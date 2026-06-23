"""Unified cost bench — 9 collab2 engines, ONE script/GPU/session.
Purpose: strictly comparable lat/throughput/energy/size across
pruning x quantization anchors (user comparability requirement 2026-06-04).
Method = E5 measurement core (CUDA-Event p50, NVML board power, GPU-exclusive)."""
import sys, json, csv
from pathlib import Path
import pynvml, torch

REPO_ROOT = Path("/home/jichengzhi/UniV2X")
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase2"))
from e5_collab2_energy_bench import measure_engine_energy_collab2, reverify_idle
from e4_energy_bench import load_engine, measure_idle_power

GPU = int(sys.argv[1]) if len(sys.argv) > 1 else 1
CACHE = REPO_ROOT / "models/stage_a_cache"
CONFIGS = [
    ("base",     "fp32"), ("base",     "fp16"), ("base",     "int8"),
    ("pruned25", "fp16"), ("pruned25", "int8"),
    ("pruned50", "fp16"), ("pruned50", "int8"),
    ("pruned75", "fp16"), ("pruned75", "int8"),
]

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(GPU)
dirty, util, others = reverify_idle(handle, GPU, 50.0)
print(f"[GPU{GPU}] pre-check: util={util}% other_mem={others:.0f}MiB")
assert not dirty, "GPU not exclusively idle — abort"
torch.cuda.set_device(GPU)
idle_power, _ = measure_idle_power(handle, secs=5.0)
print(f"idle board power = {idle_power:.1f} W")

rows = []
for tag, prec in CONFIGS:
    path = CACHE / f"{tag}_{prec}.engine"
    if not path.exists():
        print(f"  [MISS] {path.name}"); continue
    d2, _, o2 = reverify_idle(handle, GPU, 50.0, check_util=False)
    assert not d2, f"co-tenant appeared mid-run ({o2:.0f}MiB) — abort, energy invalid"
    engine = load_engine(path)
    r = measure_engine_energy_collab2(engine, handle, hold_s=12.0)
    row = {
        "anchor": tag, "precision": prec,
        "lat_p50_ms": round(r["lat_p50_ms"], 4),
        "throughput_fps_inv_lat": round(1000.0 / r["lat_p50_ms"], 1),
        "mean_power_w": round(r["mean_power_w"], 2),
        "energy_per_frame_mj": round(r["energy_per_frame_j"] * 1000, 2),
        "energy_mj_counter_xcheck": round(r["counter_j_per_frame"] * 1000, 2) if r["counter_j_per_frame"] else None,
        "engine_size_mb": round(path.stat().st_size / 1e6, 3),
        "input_shape": "x".join(str(x) for x in r["input_shape"]) if r["input_shape"] else "NA",
        "idle_power_w": round(idle_power, 2), "gpu": GPU,
    }
    rows.append(row)
    print(f"  {tag}-{prec}: lat={row['lat_p50_ms']}ms fps={row['throughput_fps_inv_lat']} "
          f"E={row['energy_per_frame_mj']}mJ (xchk {row['energy_mj_counter_xcheck']}) size={row['engine_size_mb']}MB")

out = REPO_ROOT / "results/unified_cost_bench_9cfg.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\nsaved {out} ({len(rows)} rows)")
pynvml.nvmlShutdown()
