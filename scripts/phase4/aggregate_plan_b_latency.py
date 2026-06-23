"""Merge latency results into stage5_baseline_v3.csv.

Reads:
  - data/phase4/stage5_baseline_v3.csv  (config + AMOTA)
  - output/plan_b/latency/<id>.json     (per-config latency)

Writes:
  - data/phase4/stage5_baseline_v3.csv  (in-place, adds latency columns)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data/phase4/stage5_baseline_v3.csv"
LATENCY_DIR = ROOT / "output/plan_b/latency"


def load_lat(cfg_id: str) -> dict:
    p = LATENCY_DIR / f"{cfg_id}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    out = {
        "lat_e2e_ms": d.get("e2e_ms_mean"),
        "lat_e2e_std": d.get("e2e_ms_std"),
        "params_after_M": d.get("params_after", 0) / 1e6,
        "params_reduction": d.get("reduction"),
    }
    mods = d.get("modules", {})
    for name in ("backbone", "neck", "bev_encoder", "seg_head", "track_head_decoder"):
        m = mods.get(name, {})
        out[f"lat_{name}_ms"] = m.get("mean")
    return out


def main() -> None:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    # Build new latency columns
    lat_records = []
    for _, row in df.iterrows():
        cfg_id = row["config_id"]
        lat = load_lat(cfg_id)
        if not lat:
            print(f"  warn: no latency for {cfg_id}", file=sys.stderr)
        lat_records.append(lat)

    lat_df = pd.DataFrame(lat_records)
    # Drop existing lat_* cols if re-running
    df = df.drop(columns=[c for c in df.columns if c.startswith("lat_") or c in ("params_after_M", "params_reduction")], errors="ignore")
    out = pd.concat([df, lat_df], axis=1)
    out.to_csv(CSV_PATH, index=False)
    print(f"\nWrote {len(out)} rows × {len(out.columns)} cols to {CSV_PATH}")
    print("\nLatency summary:")
    valid = out.dropna(subset=["lat_e2e_ms"])
    if len(valid) > 0:
        print(valid[["config_id", "amota", "lat_e2e_ms", "params_after_M", "params_reduction"]].to_string(index=False))
        print(f"\nlat_e2e_ms range: [{valid['lat_e2e_ms'].min():.1f}, {valid['lat_e2e_ms'].max():.1f}]")


if __name__ == "__main__":
    main()
