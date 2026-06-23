"""Build a single inspection CSV of all collected anchors (target 720).

Run AFTER P0.3 Orin bench completes. Workflow:
  1. rsync Orin /home/jichengzhi/orin_class_a/results/*.json to 4090 cache
  2. parse all Orin JSON → DataFrame matching §3 schema
  3. append to unified_bench.parquet (existing 660 + new ~432 Orin = ~1092)
  4. dedup + write data/unified_bench_full.csv for human inspection
  5. write a per-Class summary CSV (Class A/B/C/D + amota/legacy)

Output:
  data/unified_bench_full.parquet     ← all rows (Class A + Orin P0.3 + legacy)
  data/unified_bench_full.csv         ← same, CSV for user inspection
  data/inspection_summary.csv         ← per (Class, hardware, metric_type) row counts
"""
from __future__ import annotations
import json, os, subprocess
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
ORIN_CACHE = REPO_ROOT / "data/orin_p0_3_results_cache"
ORIN_CACHE.mkdir(parents=True, exist_ok=True)

SCHEMA_COLS = [
    "model_class", "stage0_planes", "stage1_planes", "stage2_planes",
    "prune_object", "sparse_mask", "q_bits", "q_granularity",
    "q_object", "d_scheme", "d_tactic", "d_workspace_gb", "hardware",
    "lat_p50_ms", "lat_p99_ms", "lat_mean_ms", "throughput_fps",
    "ap30", "ap50", "ap70",
    "engine_size_mb", "peak_gpu_mem_mb", "params_kb", "build_secs",
    "build_success", "fail_reason", "n_trt_path", "n_pytorch_fallback",
    "calibration_method", "finetune_epochs", "source", "timestamp",
    "metric_type", "anchor_id",
]


def rsync_orin_results():
    """Pull Orin P0.3 JSONs back to 4090 cache."""
    cmd = ["sshpass", "-p", "shuai123", "rsync", "-av",
           "-e", "ssh -o StrictHostKeyChecking=no",
           "jichengzhi@172.16.62.222:/home/jichengzhi/orin_class_a/results/",
           str(ORIN_CACHE) + "/"]
    print(f"[rsync] pulling Orin results → {ORIN_CACHE}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"[rsync] WARN: {r.stderr[-200:]}")
    n_jsons = len(list(ORIN_CACHE.glob("*.json")))
    print(f"[rsync] pulled {n_jsons} JSONs (incl _summary)")
    return n_jsons


def _empty_row():
    return {c: np.nan for c in SCHEMA_COLS}


def parse_orin_jsons():
    """Parse Orin P0.3 anchor JSONs → list of schema dicts."""
    rows = []
    for j in ORIN_CACHE.glob("*.json"):
        if j.name == "_summary.json":
            continue
        try:
            rep = json.loads(j.read_text())
        except Exception as e:
            print(f"[parse] skip {j.name}: {e}")
            continue
        row = _empty_row()
        row["model_class"] = "pyramid_fusion"
        row["stage0_planes"] = rep.get("stage0_planes")
        row["stage1_planes"] = rep.get("stage1_planes")
        row["stage2_planes"] = rep.get("stage2_planes")
        row["prune_object"] = "channel" if (rep.get("stage0_planes") or 64) < 64 else "none"
        row["sparse_mask"] = "dense"
        q_label = rep.get("q_label", "")
        if "fp32" in q_label.lower(): row["q_bits"] = "FP32"
        elif "fp16" in q_label.lower(): row["q_bits"] = "FP16"
        elif "int8" in q_label.lower() or "best" in q_label.lower(): row["q_bits"] = "INT8"
        else: row["q_bits"] = "mixed"
        row["q_granularity"] = "per-tensor"
        row["q_object"] = "W+A" if row["q_bits"] == "INT8" else "W-only"
        # D scheme decode
        d_label = rep.get("d_label", "")
        if "dla0" in d_label.lower(): row["d_scheme"] = "DLA0"
        elif "dla1" in d_label.lower(): row["d_scheme"] = "DLA1"
        else: row["d_scheme"] = "GPU"
        row["d_tactic"] = "no_cudnn" if rep.get("tactic_no_cudnn") else "default"
        row["d_workspace_gb"] = rep.get("d_workspace_gb")
        row["hardware"] = "orin_agx_64gb"
        row["lat_p50_ms"] = rep.get("lat_p50_ms")
        row["lat_p99_ms"] = rep.get("lat_p99_ms")
        row["lat_mean_ms"] = rep.get("lat_mean_ms")
        row["throughput_fps"] = rep.get("throughput_fps")
        row["engine_size_mb"] = rep.get("engine_size_mb")
        row["build_secs"] = rep.get("build_secs_approx")
        row["build_success"] = bool(rep.get("build_success", False))
        row["fail_reason"] = rep.get("fail_reason")
        row["calibration_method"] = rep.get("calibrator", "none")
        row["finetune_epochs"] = 23
        row["source"] = "orin_class_a_bench_trtexec (P0.3)"
        row["timestamp"] = "2026-05-13"
        row["metric_type"] = "lat_only"  # Orin lat-only per plan §4.1.3
        row["anchor_id"] = rep.get("anchor_id", j.stem)
        rows.append(row)
    return rows


def main():
    print("=" * 70)
    print("Build inspection CSV — aggregate unified_bench + Orin P0.3 results")
    print("=" * 70)

    # 1. rsync Orin results
    rsync_orin_results()

    # 2. Load unified_bench (P0.4 output)
    unified = DATA / "unified_bench.parquet"
    if not unified.exists():
        print(f"FAIL: {unified} missing — run P0.4 first")
        return
    df_existing = pd.read_parquet(unified)
    print(f"[existing] {len(df_existing)} rows from unified_bench.parquet")

    # 3. Parse Orin rows
    orin_rows = parse_orin_jsons()
    df_orin = pd.DataFrame(orin_rows, columns=SCHEMA_COLS)
    print(f"[orin] {len(df_orin)} Orin P0.3 rows parsed")
    if len(df_orin) > 0:
        ok = df_orin[df_orin["build_success"] == True]
        fail = df_orin[df_orin["build_success"] == False]
        print(f"        OK={len(ok)}, fail={len(fail)}")

    # 4. Concat + dedup by anchor_id
    df_all = pd.concat([df_existing, df_orin], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["anchor_id"], keep="last")
    print(f"[merged] {len(df_all)} unique rows after dedup")

    # 5. Add Class tag for inspection clarity
    def _tag_class(r):
        src = r.get("source", "") or ""
        hw = r.get("hardware", "")
        mt = r.get("metric_type", "")
        if "class_a_pyramid_full" in src: return "A_4090"
        if "stage_a" in src: return "A_paper"
        if "stage_b" in src: return "A_negative"
        if "orin_class_a_bench" in src: return "A_orin"
        if "4090_dspace" in src or "perstage_quant" in src: return "B_4090_legacy"
        if "orin_dspace" in src or "orin_multi_engine" in src: return "C_orin_legacy"
        if "pyramid_random_bench" in src: return "B_random"
        if "amota" in mt: return "X_cross_model_amota"
        return "unknown"
    df_all["dataset_class"] = df_all.apply(_tag_class, axis=1)

    # 6. Save full CSV
    full_pq = DATA / "unified_bench_full.parquet"
    full_csv = DATA / "unified_bench_full.csv"
    df_all.to_parquet(full_pq)
    df_all.to_csv(full_csv, index=False)
    print(f"\n✅ saved {len(df_all)} rows →")
    print(f"   {full_pq}")
    print(f"   {full_csv}")

    # 7. Per-Class summary
    print("\n=== Per (dataset_class × hardware) row counts ===")
    pivot = df_all.groupby(["dataset_class", "hardware"]).size().unstack(fill_value=0)
    print(pivot.to_string())

    summary_rows = []
    for cls in sorted(df_all["dataset_class"].unique()):
        sub = df_all[df_all["dataset_class"] == cls]
        for hw in sorted(sub["hardware"].dropna().unique()):
            ss = sub[sub["hardware"] == hw]
            summary_rows.append({
                "dataset_class": cls, "hardware": hw, "n_rows": len(ss),
                "build_success_n": (ss["build_success"] == True).sum(),
                "lat_real_n": ss["lat_mean_ms"].notna().sum(),
                "ap_real_n": ss["ap50"].notna().sum(),
                "min_lat_ms": ss["lat_mean_ms"].min(),
                "max_lat_ms": ss["lat_mean_ms"].max(),
                "min_ap50": ss["ap50"].min() if ss["ap50"].notna().any() else None,
                "max_ap50": ss["ap50"].max() if ss["ap50"].notna().any() else None,
            })
    df_sum = pd.DataFrame(summary_rows)
    out_sum = DATA / "inspection_summary.csv"
    df_sum.to_csv(out_sum, index=False)
    print(f"\n✅ summary → {out_sum}")
    print(f"\nTotal toward 720 target: {len(df_all)} rows")

    # 8. Goal progress
    target = 720
    print(f"\nProgress: {len(df_all)}/{target} ({100*len(df_all)/target:.1f}%)")
    if len(df_all) >= target:
        print(f"  🎉 surpassed 720 target")
    else:
        gap = target - len(df_all)
        print(f"  缺 {gap} 行: Class B {(120-((df_all['dataset_class']=='B_4090_legacy').sum()))} + "
              f"Class C {(170-((df_all['dataset_class']=='C_orin_legacy').sum()))} + "
              f"Class D 100 (未启动)")


if __name__ == "__main__":
    main()
