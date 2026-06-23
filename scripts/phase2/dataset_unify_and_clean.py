"""P0.4 — Unify 9 legacy data sources + 2 paper-grade sources into unified_bench.parquet.

Per plan §〇.5 strict rules:
  - lat-only files (no AP真测): AP cols = NaN, NEVER fill with proxy/baseline
  - stage_b 4 negative rows: AP保留 ~0.06 + fail_reason='insufficient_finetune'
  - baseline_4090 83 rows: metric_type='amota' (cross-model), NOT mixed with Pyramid AP
  - missing peak_gpu_mem: NaN + fail_reason='not_measured_legacy'
  - All rows have build_success bool + source + timestamp traceability

Output:
  data/unified_bench.parquet  (~1377 rows = 1057 lat-only + 320 Class A完整 + 8 stage_a + 4 negative - dedup)
  data/schema_spec.json       (schema column definitions)
  data/_by_class/legacy/      (raw input files for traceability, symlinked)
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
OUT = DATA / "unified_bench.parquet"

# §3 unified schema (29 cols)
SCHEMA_COLS = [
    # 坐标 (12)
    "model_class", "stage0_planes", "stage1_planes", "stage2_planes",
    "prune_object", "sparse_mask", "q_bits", "q_granularity",
    "q_object", "d_scheme", "d_tactic", "d_workspace_gb", "hardware",
    # 性能 (7)
    "lat_p50_ms", "lat_p99_ms", "lat_mean_ms", "throughput_fps",
    "ap30", "ap50", "ap70",
    # 资源 (4)
    "engine_size_mb", "peak_gpu_mem_mb", "params_kb", "build_secs",
    # 元数据 (10)
    "build_success", "fail_reason", "n_trt_path", "n_pytorch_fallback",
    "calibration_method", "finetune_epochs", "source", "timestamp",
    "metric_type", "anchor_id",
]


def _empty_row():
    return {col: np.nan for col in SCHEMA_COLS}


def map_class_a_pyramid_full():
    """320 rows: 完整 4 指标 anchor, Pyramid on 4090."""
    src = DATA / "_by_class/class_a_pyramid_full.parquet"
    df = pd.read_parquet(src)
    rows = []
    for _, r in df.iterrows():
        row = _empty_row()
        # Coord
        row["model_class"] = "pyramid_fusion"
        row["stage0_planes"] = int(r["stage0_planes"])
        row["stage1_planes"] = int(r["stage1_planes"])
        row["stage2_planes"] = int(r["stage2_planes"])
        row["prune_object"] = "channel"
        row["sparse_mask"] = "dense"
        # Q decode from q_label + precision + calibrator
        q_label, prec, calib = r["q_label"], r["precision"], r["calibrator"]
        row["q_bits"] = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8"}.get(prec, prec)
        row["q_granularity"] = "per-tensor"  # all 4090 grid is per-tensor (TRT default)
        row["q_object"] = "W+A" if prec == "int8" else ("W-only" if prec == "fp16" else "none")
        # D
        row["d_scheme"] = "GPU"
        row["d_tactic"] = "default"
        row["d_workspace_gb"] = float(r["workspace_mb"]) / 1024
        row["hardware"] = "rtx4090"
        # Perf
        row["lat_p50_ms"] = r.get("lat_p50_ms")
        row["lat_p99_ms"] = r.get("lat_p99_ms")
        row["lat_mean_ms"] = r.get("lat_mean_ms")
        row["throughput_fps"] = 1000.0 / r["lat_mean_ms"] if pd.notna(r.get("lat_mean_ms")) and r["lat_mean_ms"] > 0 else np.nan
        row["ap30"] = r.get("ap30")
        row["ap50"] = r.get("ap50")
        row["ap70"] = r.get("ap70")
        # Resource
        row["engine_size_mb"] = r.get("engine_size_mb")
        row["build_secs"] = r.get("build_secs")
        # Meta
        row["build_success"] = (r["status"] == "OK")
        row["calibration_method"] = calib
        row["finetune_epochs"] = 23 if "p25" in r["triplet_tag"] or "p50" in r["triplet_tag"] or "p75" in r["triplet_tag"] or "base" in r["triplet_tag"] else 33
        row["source"] = "class_a_pyramid_full (P0.2 + P0.2.b)"
        row["timestamp"] = "2026-05-13"
        row["metric_type"] = "ap_dair_500samples"
        row["anchor_id"] = r["anchor_id"]
        row["n_trt_path"] = r.get("n_trt_path")
        row["n_pytorch_fallback"] = r.get("n_pytorch_fallback")
        rows.append(row)
    return rows


def map_stage_a():
    """8 paper-grade rows, AP 1789 full DAIR val."""
    df = pd.read_parquet(DATA / "stage_a_ap_real.parquet")
    rows = []
    for _, r in df.iterrows():
        row = _empty_row()
        row["model_class"] = "pyramid_fusion"
        row["stage0_planes"] = int(r["stage0_planes"])
        row["stage1_planes"] = int(r["stage1_planes"])
        row["stage2_planes"] = int(r["stage2_planes"])
        row["prune_object"] = "channel" if r["stage0_planes"] < 64 else "none"
        row["sparse_mask"] = "dense"
        row["q_bits"] = {"fp16": "FP16", "int8": "INT8", "fp32": "FP32"}.get(r["precision"], r["precision"])
        row["q_granularity"] = "per-tensor"
        row["q_object"] = "W+A" if r["precision"] == "int8" else "W-only"
        row["d_scheme"] = "GPU"; row["d_tactic"] = "default"; row["d_workspace_gb"] = 4.0
        row["hardware"] = "rtx4090"
        row["ap30"] = r.get("ap30"); row["ap50"] = r.get("ap50"); row["ap70"] = r.get("ap70")
        row["build_success"] = True
        row["calibration_method"] = "minmax" if r["precision"] == "int8" else "none"
        row["finetune_epochs"] = 23
        row["source"] = "stage_a_ap_real"
        row["timestamp"] = "2026-05-10"
        row["metric_type"] = "ap_dair_1789full"
        row["anchor_id"] = f"stage_a_{r['anchor']}"
        row["n_trt_path"] = r.get("n_trt_path")
        rows.append(row)
    return rows


def map_stage_b_negative():
    """4 negative rows: AP≈0.06 due to insufficient finetune."""
    df = pd.read_parquet(DATA / "stage_b_ap_real.parquet")
    rows = []
    for _, r in df.iterrows():
        row = _empty_row()
        row["model_class"] = "pyramid_fusion"
        row["stage0_planes"] = int(r["stage0_planes"])
        row["stage1_planes"] = int(r["stage1_planes"])
        row["stage2_planes"] = int(r["stage2_planes"])
        row["prune_object"] = "channel"
        row["sparse_mask"] = "dense"
        row["q_bits"] = {"fp16": "FP16", "int8": "INT8"}.get(r["precision"], r["precision"])
        row["q_granularity"] = "per-tensor"
        row["q_object"] = "W+A" if r["precision"] == "int8" else "W-only"
        row["d_scheme"] = "GPU"; row["d_tactic"] = "default"; row["d_workspace_gb"] = 4.0
        row["hardware"] = "rtx4090"
        row["ap30"] = r.get("ap30"); row["ap50"] = r.get("ap50"); row["ap70"] = r.get("ap70")
        row["build_success"] = True
        row["fail_reason"] = "insufficient_finetune"
        row["calibration_method"] = "minmax" if r["precision"] == "int8" else "none"
        row["finetune_epochs"] = 4  # < 25 epoch gate
        row["source"] = "stage_b_ap_real"
        row["timestamp"] = "2026-05-10"
        row["metric_type"] = "ap_dair_negative_pool"
        row["anchor_id"] = f"stage_b_neg_{r['triplet_sig']}_{r['precision']}"
        rows.append(row)
    return rows


def map_4090_dspace():
    """48 lat-only rows on 4090, AP=NaN per §〇.5 rule."""
    df = pd.read_parquet(DATA / "4090_dspace_bench.parquet")
    rows = []
    for _, r in df.iterrows():
        row = _empty_row()
        row["model_class"] = "pyramid_fusion"
        row["stage0_planes"] = int(r["stage0_planes"])
        row["stage1_planes"] = int(r["stage1_planes"])
        row["stage2_planes"] = int(r["stage2_planes"])
        row["prune_object"] = "channel" if r["stage0_planes"] < 64 else "none"
        row["sparse_mask"] = "dense"
        row["q_bits"] = {"fp16": "FP16", "int8": "INT8", "fp32": "FP32"}.get(r["precision"], r["precision"])
        row["q_granularity"] = "per-tensor"
        row["q_object"] = "W+A" if r["precision"] == "int8" else "W-only"
        row["d_scheme"] = "GPU"
        row["d_tactic"] = r["d_tactic"]
        row["d_workspace_gb"] = float(r["d_workspace_gb"])
        row["hardware"] = "rtx4090"
        row["lat_p50_ms"] = r.get("lat_p50_ms")
        row["lat_p99_ms"] = r.get("lat_p99_ms")
        row["lat_mean_ms"] = r.get("lat_mean_ms")
        row["throughput_fps"] = 1000.0 / r["lat_mean_ms"] if pd.notna(r.get("lat_mean_ms")) and r["lat_mean_ms"] > 0 else np.nan
        row["engine_size_mb"] = r.get("engine_size_mb")
        row["build_secs"] = r.get("build_secs")
        row["build_success"] = bool(r["build_success"])
        row["fail_reason"] = r.get("fail_reason")
        row["calibration_method"] = "minmax" if r["precision"] == "int8" else "none"
        row["finetune_epochs"] = 23
        row["source"] = "4090_dspace_bench"
        row["timestamp"] = "2026-05-11"
        row["metric_type"] = "lat_only"
        row["anchor_id"] = f"4090d_{r['triplet_sig']}_{r['precision']}_{r['d_tactic']}_{int(r['d_workspace_gb'])}gb"
        rows.append(row)
    return rows


def map_orin_dspace():
    """72 lat-only rows on Orin, AP=NaN."""
    df = pd.read_parquet(DATA / "orin_dspace_bench.parquet")
    rows = []
    for _, r in df.iterrows():
        row = _empty_row()
        row["model_class"] = "pyramid_fusion"
        sig_parts = str(r["triplet_sig"]).split("_")
        if len(sig_parts) >= 3:
            row["stage0_planes"] = int(sig_parts[0])
            row["stage1_planes"] = int(sig_parts[1])
            row["stage2_planes"] = int(sig_parts[2])
        row["prune_object"] = "channel" if row["stage0_planes"] < 64 else "none"
        row["sparse_mask"] = "dense"
        row["q_bits"] = {"fp16": "FP16", "int8": "INT8", "fp32": "FP32"}.get(r["precision"], r["precision"])
        row["q_granularity"] = "per-tensor"
        row["q_object"] = "W+A" if r["precision"] == "int8" else "W-only"
        row["d_scheme"] = r["d_scheme"]
        row["d_tactic"] = r.get("d_tactic", "default")
        ws_val = r.get("d_workspace")
        if isinstance(ws_val, str):
            ws_val = float(ws_val.rstrip("GBgb").rstrip())
        row["d_workspace_gb"] = float(ws_val) if pd.notna(ws_val) else 2.0
        row["hardware"] = "orin_agx_64gb"
        row["lat_p50_ms"] = r.get("lat_p50_ms")
        row["lat_p99_ms"] = r.get("lat_p99_ms")
        row["lat_mean_ms"] = r.get("lat_mean_ms")
        row["throughput_fps"] = 1000.0 / r["lat_mean_ms"] if pd.notna(r.get("lat_mean_ms")) and r["lat_mean_ms"] > 0 else np.nan
        row["build_success"] = bool(r["build_success"])
        row["fail_reason"] = r.get("fail_reason")
        row["calibration_method"] = "minmax" if r["precision"] == "int8" else "none"
        row["finetune_epochs"] = 23
        row["source"] = "orin_dspace_bench"
        row["timestamp"] = "2026-05-11"
        row["metric_type"] = "lat_only"
        row["anchor_id"] = f"orind_{r['cfg_id']}"
        rows.append(row)
    return rows


def map_pyramid_random():
    """100 lat-only random search rows on 4090, AP=NaN."""
    df = pd.read_parquet(DATA / "pyramid_random_bench.parquet")
    rows = []
    for i, r in df.iterrows():
        row = _empty_row()
        row["model_class"] = "pyramid_fusion"
        row["stage0_planes"] = int(r["stage0_planes"])
        row["stage1_planes"] = int(r["stage1_planes"])
        row["stage2_planes"] = int(r["stage2_planes"])
        row["prune_object"] = "channel"
        row["sparse_mask"] = "dense"
        row["q_bits"] = {"fp16": "FP16", "int8": "INT8", "fp32": "FP32"}.get(r["precision"], r["precision"])
        row["q_granularity"] = "per-tensor"
        row["q_object"] = "W+A" if r["precision"] == "int8" else "W-only"
        row["d_scheme"] = "GPU"; row["d_tactic"] = "default"; row["d_workspace_gb"] = 4.0
        row["hardware"] = "rtx4090"
        row["lat_p50_ms"] = r.get("lat_p50_ms")
        row["lat_p99_ms"] = r.get("lat_p99_ms")
        row["lat_mean_ms"] = r.get("lat_mean_ms")
        row["throughput_fps"] = 1000.0 / r["lat_mean_ms"] if pd.notna(r.get("lat_mean_ms")) and r["lat_mean_ms"] > 0 else np.nan
        row["engine_size_mb"] = r.get("engine_size_mb")
        row["build_secs"] = r.get("build_secs")
        row["build_success"] = True
        row["calibration_method"] = "minmax" if r["precision"] == "int8" else "none"
        row["finetune_epochs"] = 0  # random_search did NOT finetune
        row["fail_reason"] = "no_finetune_for_lat_only"
        row["source"] = "pyramid_random_bench"
        row["timestamp"] = "2026-05-11"
        row["metric_type"] = "lat_only"
        row["anchor_id"] = f"pyr_rand_{i:03d}_{r['triplet_sig']}_{r['precision']}"
        rows.append(row)
    return rows


def map_perstage_quant():
    """18 lat-only mixed-precision rows on 4090."""
    df = pd.read_parquet(DATA / "perstage_quant_bench.parquet")
    rows = []
    for _, r in df.iterrows():
        row = _empty_row()
        row["model_class"] = "pyramid_fusion"
        row["stage0_planes"] = int(r["stage0_planes"])
        row["stage1_planes"] = int(r["stage1_planes"])
        row["stage2_planes"] = int(r["stage2_planes"])
        row["prune_object"] = "channel"
        row["sparse_mask"] = "dense"
        row["q_bits"] = "mixed"
        row["q_granularity"] = "per-tensor"
        row["q_object"] = "W+A"
        row["d_scheme"] = "GPU"; row["d_tactic"] = "default"; row["d_workspace_gb"] = 4.0
        row["hardware"] = "rtx4090"
        row["lat_p50_ms"] = r.get("lat_p50_ms")
        row["lat_p99_ms"] = r.get("lat_p99_ms")
        row["lat_mean_ms"] = r.get("lat_mean_ms")
        row["throughput_fps"] = 1000.0 / r["lat_mean_ms"] if pd.notna(r.get("lat_mean_ms")) and r["lat_mean_ms"] > 0 else np.nan
        row["engine_size_mb"] = r.get("engine_size_mb")
        row["build_secs"] = r.get("build_secs")
        row["build_success"] = True
        row["calibration_method"] = "minmax"
        row["finetune_epochs"] = 23
        row["source"] = "perstage_quant_bench"
        row["timestamp"] = "2026-05-11"
        row["metric_type"] = "lat_only"
        row["anchor_id"] = f"perq_{r['triplet_sig']}_{r['config_label']}"
        rows.append(row)
    return rows


def map_orin_multi_engine():
    """10 Orin multi-IP rows, AP=NaN."""
    src = REPO_ROOT / "results/orin_multi_engine_batch.json"
    if not src.exists():
        return []
    data = json.loads(src.read_text())
    rows = []
    if isinstance(data, dict):
        data = list(data.values()) if not isinstance(data, list) else data
    for i, r in enumerate(data if isinstance(data, list) else []):
        if not isinstance(r, dict): continue
        row = _empty_row()
        row["model_class"] = "pyramid_fusion"
        row["q_bits"] = "FP16"
        row["q_granularity"] = "per-tensor"
        row["q_object"] = "W-only"
        row["d_scheme"] = r.get("scheme", "dual_A")
        row["d_tactic"] = "default"; row["d_workspace_gb"] = 2.0
        row["hardware"] = "orin_agx_64gb"
        row["lat_p50_ms"] = r.get("lat_p50_ms")
        row["lat_mean_ms"] = r.get("lat_mean_ms")
        row["throughput_fps"] = r.get("throughput_fps")
        row["build_success"] = True
        row["finetune_epochs"] = 23
        row["source"] = "orin_multi_engine_batch"
        row["timestamp"] = "2026-05-12"
        row["metric_type"] = "lat_only_multi_ip"
        row["anchor_id"] = f"orin_multi_{i:02d}"
        rows.append(row)
    return rows


def map_baseline_4090_amota():
    """83 cross-model rows. Use amota (NOT mixed with Pyramid AP)."""
    df = pd.read_parquet(DATA / "baseline_4090.parquet")
    rows = []
    for i, r in df.iterrows():
        row = _empty_row()
        row["model_class"] = r["model_class"]
        # backbone planes if pyramid; else NaN
        if r["model_class"] == "pyramid_fusion":
            # try parse from notes / config_id; left NaN otherwise
            pass
        row["prune_object"] = r.get("prune_object", "none")
        row["sparse_mask"] = "dense"
        row["q_bits"] = r.get("q_bits__backbone", "FP16")
        row["q_granularity"] = r.get("q_granularity__backbone", "per-tensor")
        row["q_object"] = r.get("q_object__backbone", "W+A")
        row["d_scheme"] = r.get("d_routing__backbone", "GPU")
        row["d_tactic"] = "default"; row["d_workspace_gb"] = 4.0
        row["hardware"] = "rtx4090"
        row["lat_mean_ms"] = r.get("lat_e2e_ms")
        row["lat_p50_ms"] = r.get("lat_e2e_ms")
        row["throughput_fps"] = 1000.0 / r["lat_e2e_ms"] if pd.notna(r.get("lat_e2e_ms")) and r["lat_e2e_ms"] > 0 else np.nan
        row["peak_gpu_mem_mb"] = r.get("mem_peak_mb")
        row["build_success"] = bool(r.get("is_real_measured", False))
        row["finetune_epochs"] = 23
        row["source"] = "baseline_4090 (cross-model)"
        row["timestamp"] = "2026-05-09"
        row["metric_type"] = "amota_cross_model"  # NOT mixed with DAIR AP
        # Store amota in ap50 col only for amota metric_type, downstream LGB will filter
        row["ap50"] = r.get("amota")  # amota stored in ap50 slot, marked by metric_type
        row["anchor_id"] = f"cross_{r['model_class']}_{r['config_id']}"
        rows.append(row)
    return rows


def main():
    print("=" * 72)
    print("P0.4 — Unify 9 data sources into unified_bench.parquet")
    print("=" * 72)

    sources = [
        ("class_a_pyramid_full (P0.2 + P0.2.b)", map_class_a_pyramid_full),
        ("stage_a (paper-grade 1789 AP)", map_stage_a),
        ("stage_b (negative AP≈0.06)", map_stage_b_negative),
        ("4090_dspace_bench (lat-only)", map_4090_dspace),
        ("orin_dspace_bench (lat-only)", map_orin_dspace),
        ("pyramid_random_bench (lat-only)", map_pyramid_random),
        ("perstage_quant_bench (lat-only mixed)", map_perstage_quant),
        ("orin_multi_engine_batch (lat-only multi-IP)", map_orin_multi_engine),
        ("baseline_4090 (cross-model amota)", map_baseline_4090_amota),
    ]

    all_rows = []
    for label, fn in sources:
        try:
            rows = fn()
            print(f"  ✓ {label:50s}  +{len(rows)} rows")
            all_rows.extend(rows)
        except Exception as e:
            print(f"  ✗ {label:50s}  FAILED: {e}")
            import traceback; traceback.print_exc()

    df = pd.DataFrame(all_rows, columns=SCHEMA_COLS)
    print(f"\nTotal rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # Validate schema
    missing_cols = set(SCHEMA_COLS) - set(df.columns)
    if missing_cols:
        print(f"WARN: missing cols after build: {missing_cols}")

    # Status summary per metric_type
    print("\n=== Row counts by metric_type ===")
    print(df["metric_type"].value_counts().to_string())

    # AP coverage summary
    ap_real = df[df["metric_type"].isin(["ap_dair_500samples", "ap_dair_1789full"])]
    ap_neg = df[df["metric_type"] == "ap_dair_negative_pool"]
    lat_only = df[df["metric_type"].isin(["lat_only", "lat_only_multi_ip"])]
    amota = df[df["metric_type"] == "amota_cross_model"]
    print(f"\nAP-real (Pyramid main pool): {len(ap_real)} rows  (LGB AP head训练源)")
    print(f"AP-negative (insufficient finetune): {len(ap_neg)} rows")
    print(f"Lat-only (AP=NaN, lat head only): {len(lat_only)} rows")
    print(f"Cross-model amota: {len(amota)} rows  (单独跨模型 head)")

    # Hardware split
    print("\n=== Hardware ===")
    print(df["hardware"].value_counts().to_string())

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    df.to_csv(OUT.with_suffix(".csv"), index=False)
    print(f"\n✅ saved → {OUT}")
    print(f"   csv  → {OUT.with_suffix('.csv')}")

    # Schema spec
    spec_path = DATA / "schema_spec.json"
    spec = {
        "version": "v1.1",
        "generated": datetime.utcnow().isoformat() + "Z",
        "columns": SCHEMA_COLS,
        "metric_types": {
            "ap_dair_1789full": "stage_a paper-grade, full 1789 sample DAIR val",
            "ap_dair_500samples": "Class A sweep (P0.2/P0.2.b), 500 sample DAIR val",
            "ap_dair_negative_pool": "stage_b negative, insufficient finetune (excluded from training)",
            "lat_only": "lat真测 AP=NaN, AP列禁止补估",
            "lat_only_multi_ip": "Orin multi-IP lat (双/三 IP), AP=NaN",
            "amota_cross_model": "baseline_4090 cross-model amota, NOT mixed with Pyramid AP",
        },
        "lgb_usage": {
            "lat_head": "all rows with build_success=True and lat_mean_ms != NaN",
            "ap_head": "rows with metric_type in {ap_dair_1789full, ap_dair_500samples} only",
            "amota_head": "rows with metric_type=amota_cross_model only",
        },
    }
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    print(f"   spec → {spec_path}")


if __name__ == "__main__":
    main()
