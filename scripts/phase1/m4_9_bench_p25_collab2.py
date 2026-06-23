"""Bench the p25 (48,96,192) FP16 collab2 engine — the only uniform config not
covered by results/perstage_quant_latency_global_v2.csv (which has base/p50/p75).

Same口径 as m4_9_bench_perstage_v2: body_subnet_collab2, CUDA-Event,
warmup200/measure200, input spatial_features(2,64,128,256)+t_ego(2,2,3),
idle-GPU gate (util<=2% & mem<=50MiB). Run with CUDA_VISIBLE_DEVICES=6.

Output: appends one row (config_label=global_fp16_automix, triplet=pruned25)
to results/perstage_quant_latency_global_v2.csv so build_dataset_v2 picks it up.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/jichengzhi/UniV2X")
sys.path.insert(0, str(ROOT / "scripts/phase1"))
from m4_9_bench_perstage_v2 import (  # noqa: E402
    benchmark_engine, foreign_procs, gpu_idle_evidence, gpu_util_mem, _phys_id,
)

ENGINE = ROOT / "models/stage_a_cache/pruned25_fp16.engine"
OUT = ROOT / "results/perstage_quant_latency_global_v2.csv"


def main():
    base_util, base_mem = gpu_util_mem()
    if base_util > 2 or base_mem > 50:
        raise RuntimeError(
            f"target GPU (phys {_phys_id()}) NOT idle: util={base_util}% "
            f"mem={base_mem}MiB — abort (只在空闲卡测)")
    print(f"[gate] phys {_phys_id()} util={base_util}% mem={base_mem}MiB — idle, proceed")

    fp = foreign_procs()
    if fp:
        raise RuntimeError(f"foreign procs {fp} on phys {_phys_id()} — abort")

    util, mem = gpu_idle_evidence()
    print(f"[bench] pruned25_fp16 (util={util}% mem={mem}MiB, foreign=none)")
    stats = benchmark_engine(
        str(ENGINE), input_shape=(2, 64, 128, 256),
        extra_input_shapes={"t_ego": (2, 2, 3)},
        n_warmup=200, n_measure=200)
    util2, mem2 = gpu_idle_evidence()
    shape_str = "x".join(str(s) for s in stats["input_shape"])
    print(f"        p50={stats['p50_ms']:.4f} p99={stats['p99_ms']:.4f} "
          f"mean={stats['mean_ms']:.4f} fps={1000.0/stats['mean_ms']:.1f}")

    row = {
        "config_label": "global_fp16_automix", "triplet": "pruned25",
        "lat_p50_ms": round(stats["p50_ms"], 4),
        "lat_p99_ms": round(stats["p99_ms"], 4),
        "lat_mean_ms": round(stats["mean_ms"], 4),
        "lat_std_ms": round(stats["std_ms"], 4),
        "throughput_fps": round(1000.0 / stats["mean_ms"], 1),
        "input_shape": shape_str,
        "n_inputs": len(stats["input_names"]),
        "latency_kind": "body_subnet_collab2",
        "gpu_idle_verified": (f"phys{_phys_id()};baseline util{base_util}%/mem{base_mem}MiB;"
                              f"foreign=none(200/200)"),
        "n_warmup": stats["n_warmup"], "n_measure": stats["n_measure"],
        "engine_size_mb": round(stats["engine_size_mb"], 3),
        "engine_path": str(ENGINE.relative_to(ROOT)),
        "source": "m4_9_bench_p25_collab2;CUDA-Event;collab2_shape;new-measure",
        "notes": ("p25 (48,96,192) fp16 collab2 NEW measure; engine from stage_a_cache "
                  "(same build as other 5 reused global engines); fills the only "
                  "uniform config missing from perstage triplet"),
    }

    df = pd.read_csv(OUT)
    df = df[~((df["config_label"] == "global_fp16_automix") &
              (df["triplet"] == "pruned25"))]  # idempotent re-run
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(OUT, index=False)
    print(f"[done] appended pruned25 fp16 collab2 row -> {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
