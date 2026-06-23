"""E5-methodology cost measurement for the new base_fp32 collab2 engine.
Same primitives/conventions as e5_collab2_energy_bench.py (one-off, 1 engine)."""
import sys, json
from pathlib import Path
import pynvml, torch

REPO_ROOT = Path("/home/jichengzhi/UniV2X")
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase2"))
from e5_collab2_energy_bench import measure_engine_energy_collab2, reverify_idle
from e4_energy_bench import load_engine, measure_idle_power

GPU = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ENGINE = REPO_ROOT / "models/stage_a_cache/base_fp32.engine"

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(GPU)
dirty, util, others = reverify_idle(handle, GPU, 50.0)
print(f"[GPU{GPU}] pre-check: util={util}% other_mem={others:.0f}MiB")
assert not dirty, "GPU not exclusively idle — abort (energy invalid)"

torch.cuda.set_device(GPU)
idle_power, idle_n = measure_idle_power(handle, secs=5.0)
print(f"idle board power = {idle_power:.1f} W")

engine = load_engine(ENGINE)
r = measure_engine_energy_collab2(engine, handle, hold_s=12.0)
out = {
    "config_label": "rtx4090_base_fp32",
    "engine_path": "models/stage_a_cache/base_fp32.engine",
    "precision": "FP32", "planes": "64_128_256",
    "latency_p50_ms": round(r["lat_p50_ms"], 4),
    "mean_power_w": round(r["mean_power_w"], 2),
    "idle_power_w": round(idle_power, 2),
    "energy_per_frame_mj": round(r["energy_per_frame_j"] * 1000, 4),
    "energy_per_frame_mj_counter": round(r["counter_j_per_frame"] * 1000, 4) if r["counter_j_per_frame"] else None,
    "throughput_fps_sustained": round(r["throughput_fps"], 1),
    "throughput_fps_inv_latency": round(1000.0 / r["lat_p50_ms"], 1),
    "engine_size_mb": round(ENGINE.stat().st_size / 1e6, 3),
    "input_shape": "x".join(str(x) for x in r["input_shape"]) if r["input_shape"] else "NA",
    "gpu": GPU,
    "source": "E5-methodology one-off;NVML;idle-exclusive",
}
Path(REPO_ROOT / "results/E5_fp32_base_oneoff.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
pynvml.nvmlShutdown()
