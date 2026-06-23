"""Plan v5 Phase A.4 — Smooth poly fit + gate G_A judgment.

Reads plan5_phaseA_anchors.csv (Phase A.3 output, 5 plane × 3 Q × 3 D = 45 cells)
+ adds reference points (g8_p3 extreme from plan v3 if available).

For each (Q, D) cell (9 cells), fit lat vs plane with 3rd-order poly,
compute R². Weighted-average R² across 9 cells is R²_smooth.

Gate G_A:
  R²_smooth >= 0.85 → H1 confirmed (lat is smooth) → path 1 → Phase B
  0.50 <= R² < 0.85 → ambiguous → path 2b
  R² < 0.50 → step jump confirmed → path 2a

Run:
    python scripts/phase2/plan5_phaseA_attribution.py
Output:
    paper_learning/2. AAAI最终故事/data/plan5_phaseA_summary.md
    plan5_state.json updated with G_A verdict + selected path
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO = Path("/home/jichengzhi/UniV2X")
DATA_DIR = REPO / "paper_learning" / "2. AAAI最终故事" / "data"
ANCHORS_CSV = DATA_DIR / "plan5_phaseA_anchors.csv"
STATE_FILE = DATA_DIR / "plan5_state.json"
SUMMARY_FILE = DATA_DIR / "plan5_phaseA_summary.md"

PLANE_INT = {
    "p64_baseline": 64,
    "p48": 48,
    "p32": 32,
    "p16": 16,
    "p8":  8,
    "p3_extreme": 3,
}


def fit_poly3_r2(planes: np.ndarray, lats: np.ndarray) -> float:
    """Fit 3rd-order poly, return R²."""
    if len(planes) < 4:
        deg = max(1, len(planes) - 1)
    else:
        deg = 3
    if len(planes) < 2:
        return 0.0
    coefs = np.polyfit(planes, lats, deg=deg)
    pred = np.polyval(coefs, planes)
    ss_res = np.sum((lats - pred) ** 2)
    ss_tot = np.sum((lats - lats.mean()) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(max(0.0, 1.0 - ss_res / ss_tot))


def load_anchors_with_extreme(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["plane_int"] = df["tag"].map(PLANE_INT)
    df = df.dropna(subset=["plane_int", "lat_p50"])
    df["plane_int"] = df["plane_int"].astype(int)

    plan_v3_summary = DATA_DIR.parent / "data" / "plan_v3_g8_extreme.json"
    if plan_v3_summary.exists():
        with open(plan_v3_summary) as f:
            v3 = json.load(f)
        for q, d_dict in v3.get("p3_extreme", {}).items():
            for d, lat in d_dict.items():
                df = pd.concat([df, pd.DataFrame([{
                    "tag": "p3_extreme", "q": q, "d": d,
                    "plane_int": 3, "lat_p50": lat,
                    "num_filters": "[3,6,12]", "status": "from_v3",
                }])], ignore_index=True)

    return df


def compute_gate_g_a(df: pd.DataFrame) -> Dict:
    cells = []
    for (q, d), grp in df.groupby(["q", "d"]):
        if len(grp) < 3:
            continue
        planes = grp["plane_int"].to_numpy(dtype=float)
        lats = grp["lat_p50"].to_numpy(dtype=float)
        order = np.argsort(planes)
        planes, lats = planes[order], lats[order]
        r2 = fit_poly3_r2(planes, lats)
        cells.append({
            "q": q, "d": d,
            "n_points": int(len(planes)),
            "plane_range": [int(planes.min()), int(planes.max())],
            "lat_range":   [float(lats.min()), float(lats.max())],
            "lat_ratio":   float(lats.max() / max(lats.min(), 1e-6)),
            "poly3_r2":    round(r2, 3),
        })

    if not cells:
        return {"r2_smooth_mean": 0.0, "cells": [], "n_cells_passed": 0, "verdict": "no_data"}

    r2_smooth = float(np.mean([c["poly3_r2"] for c in cells]))
    return {
        "r2_smooth_mean": round(r2_smooth, 3),
        "cells": cells,
        "n_cells": len(cells),
        "n_cells_r2_pass": sum(1 for c in cells if c["poly3_r2"] >= 0.85),
    }


def decide_path(r2_smooth: float) -> Tuple[str, str]:
    if r2_smooth >= 0.85:
        return "path_1", "G_A PASS — H1 confirmed (smooth curve). Proceed to Phase B."
    if r2_smooth >= 0.50:
        return "path_2b", "G_A AMBIGUOUS — add plane=24/40 mid-points (+1 day), retry."
    return "path_2a", "G_A FAIL — step jump confirmed. Zoom in plane=4/6/10/12/14 (+2 day)."


def update_state(verdict: dict, path: str, reason: str):
    if not STATE_FILE.exists():
        return
    state = json.loads(STATE_FILE.read_text())
    state["phase_A_gate_result"] = {
        "G_A_smooth_r2_min_required": 0.85,
        "G_A_r2_smooth_actual": verdict["r2_smooth_mean"],
        "n_cells_evaluated": verdict.get("n_cells", 0),
        "verdict_path": path,
        "verdict_reason": reason,
    }
    state["phase_status"]["phase_A"] = "completed"
    if path == "path_1":
        state["current_phase"] = "phase_B"
        state["last_completed_phase"] = "phase_A"
    elif path == "path_2b":
        state["current_phase"] = "phase_A_path_2b"
    else:
        state["current_phase"] = "phase_A_path_2a"
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def write_summary(df: pd.DataFrame, verdict: dict, path: str, reason: str):
    lines = [
        "# Plan v5 Phase A.4 — Attribution + Gate G_A",
        "",
        f"**Run date**: 2026-05-29",
        f"**Anchors evaluated**: {len(df)} rows ({df['tag'].nunique()} plane, "
        f"{df['q'].nunique()} Q, {df['d'].nunique()} D)",
        "",
        "## Gate G_A — smooth poly3 fit (lat vs plane)",
        "",
        f"- Threshold: R²_smooth ≥ 0.85 (pre-registered, Phase 0)",
        f"- Measured: **R²_smooth = {verdict['r2_smooth_mean']}** "
        f"({verdict.get('n_cells_r2_pass', 0)}/{verdict.get('n_cells', 0)} cells PASS individually)",
        f"- Verdict path: **{path}**",
        f"- Reason: {reason}",
        "",
        "## Per (Q, D) cell breakdown",
        "",
        "| Q | D | n | plane range | lat range | lat ratio | poly3 R² |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in verdict.get("cells", []):
        lines.append(
            f"| {c['q']} | {c['d']} | {c['n_points']} | "
            f"{c['plane_range']} | {c['lat_range'][0]:.2f}-{c['lat_range'][1]:.2f} ms | "
            f"{c['lat_ratio']:.2f}× | {c['poly3_r2']} |"
        )
    lines.extend([
        "",
        "## Next phase",
        f"- {reason}",
        "",
        "## Raw data",
        f"- Anchor CSV: `{ANCHORS_CSV.relative_to(REPO) if ANCHORS_CSV.is_relative_to(REPO) else ANCHORS_CSV}`",
        f"- State file updated: `{STATE_FILE.relative_to(REPO) if STATE_FILE.is_relative_to(REPO) else STATE_FILE}`",
    ])
    SUMMARY_FILE.write_text("\n".join(lines))


def main():
    if not ANCHORS_CSV.exists():
        print(f"ERROR: {ANCHORS_CSV} not found. Run Phase A.3 (bench) first.")
        return 1

    df = load_anchors_with_extreme(ANCHORS_CSV)
    print(f"[Plan v5 Phase A.4] loaded {len(df)} anchors")
    verdict = compute_gate_g_a(df)
    print(f"  R²_smooth = {verdict['r2_smooth_mean']} across {verdict.get('n_cells', 0)} cells")

    path, reason = decide_path(verdict["r2_smooth_mean"])
    print(f"  Verdict: {path} — {reason}")

    update_state(verdict, path, reason)
    write_summary(df, verdict, path, reason)
    print(f"[Plan v5 Phase A.4] wrote {SUMMARY_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
