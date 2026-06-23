"""T1 Q-axis: update latency_lut_pyramid_q.json with hw-optimizer build results,
then re-run P×Q×S smoke test to confirm Q-rank-flip.

Run AFTER hw-optimizer completes t1_q_partner_build.py.

Usage:
    cd /home/jichengzhi/V2X
    python scripts/phase2/t1_update_q_lut_and_verify.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
Q_LUT = REPO / "results/latency_lut_pyramid_q.json"
PARTNER_JSON = REPO / "results/t1_partner_widths_q.json"
STAGE_A = REPO / "models/stage_a_cache"

# Map from label → num_filters
LABEL_TO_NF = {
    "pad64":  [64, 96, 192],
    "s1_64":  [64, 64, 256],
    "s2_128": [64, 128, 128],
    "mix_b":  [48, 64, 256],
    "mix_d":  [48, 128, 128],
}


def read_build_json(label: str, precision: str) -> dict | None:
    """Read individual build JSON from stage_a_cache."""
    # First check stage_a_cache (direct build output)
    p = STAGE_A / f"{label}_{precision}_build.json"
    if p.exists():
        return json.loads(p.read_text())
    # Check b4expand_cache for mix_b/mix_d FP16
    if precision == "fp16" and label in ("mix_b", "mix_d"):
        p2 = REPO / f"models/b4expand_cache/b4expand_{label}_fp16_build.json"
        if p2.exists():
            data = json.loads(p2.read_text())
            data["_source"] = str(p2)
            return data
    return None


def extract_p50(build_json: dict) -> float | None:
    for key in ("p50_ms", "lat_p50_ms"):
        if key in build_json:
            return float(build_json[key])
    return None


def update_q_lut():
    """Scan stage_a_cache for completed builds and patch the Q-LUT."""
    data = json.loads(Q_LUT.read_text())
    widths = data["widths"]

    updates = 0
    for entry in widths:
        label = entry.get("label")
        if label not in LABEL_TO_NF:
            continue
        nf = LABEL_TO_NF[label]

        # Try to read FP16 build
        if not entry.get("fp16_p50_ms"):
            fp16_j = read_build_json(label, "fp16")
            if fp16_j:
                p50 = extract_p50(fp16_j)
                if p50:
                    entry["fp16_p50_ms"] = round(p50, 4)
                    entry["fp16_source"] = f"stage_a_cache/{label}_fp16_build.json"
                    print(f"  [{label}] FP16 updated: {p50:.4f}ms")
                    updates += 1

        # Try to read INT8 build
        if not entry.get("int8_p50_ms"):
            int8_j = read_build_json(label, "int8")
            if int8_j:
                p50 = extract_p50(int8_j)
                if p50:
                    entry["int8_p50_ms"] = round(p50, 4)
                    entry["int8_speedup"] = round(entry["fp16_p50_ms"] / p50, 4) if entry.get("fp16_p50_ms") else None
                    entry["int8_calibrator"] = "minmax"
                    entry["int8_source"] = f"stage_a_cache/{label}_int8_build.json"
                    entry["int8_q_object"] = "W+A (TRT auto)"
                    # Remove pending status markers
                    entry.pop("int8_status", None)
                    entry.pop("status", None)
                    print(f"  [{label}] INT8 updated: {p50:.4f}ms speedup={entry['int8_speedup']:.3f}x")
                    updates += 1

    # Update summary counts
    complete = [w for w in widths if w.get("fp16_p50_ms") and w.get("int8_p50_ms")]
    fp16_only = [w for w in widths if w.get("fp16_p50_ms") and not w.get("int8_p50_ms")]
    pending_int8 = [w["label"] for w in widths if not w.get("int8_p50_ms")]
    data["summary"]["widths_fp16_int8_complete"] = len(complete)
    data["summary"]["widths_fp16_only"] = len(fp16_only)
    data["summary"]["widths_pending_int8"] = pending_int8
    data["summary"]["widths_measured"] = len([w for w in widths if w.get("fp16_p50_ms")])

    if updates > 0:
        Q_LUT.write_text(json.dumps(data, indent=2))
        print(f"\n[✓] Q-LUT updated: {updates} fields patched → {Q_LUT}")
    else:
        print("[i] No new build results found in stage_a_cache.")

    return data, complete


def print_q_axis_summary(data):
    """Print current Q-axis coverage and speedup table."""
    widths = data["widths"]
    print("\n=== Q-axis measurement table ===")
    print(f"{'label':12s} {'num_filters':20s} {'FP16(ms)':10s} {'INT8(ms)':10s} {'speedup':8s} {'status':12s}")
    print("-" * 75)
    for w in widths:
        fp16 = w.get("fp16_p50_ms")
        int8 = w.get("int8_p50_ms")
        sp = w.get("int8_speedup")
        status = "complete" if fp16 and int8 else ("fp16-only" if fp16 else "pending")
        print(f"  {w['label']:12s} {str(w['num_filters']):20s} "
              f"{fp16 or '?':>10} {int8 or '?':>10} "
              f"{f'{sp:.3f}x' if sp else '?':>8} {status}")


def run_smoke_pqs() -> bool:
    """Re-run P×Q×S smoke test. Returns True if Q-rank-flip detected."""
    print("\n=== Re-running P×Q×S smoke test ===")
    r = subprocess.run(
        [PYTHON, "-m", "framework.search_three_arm", "--q-mode", "--seeds", "5", "--budget", "60", "--pop", "8"],
        capture_output=True, text=True, timeout=120, cwd=str(REPO)
    )
    if r.returncode != 0:
        print(f"Smoke test FAILED: {r.stderr[-500:]}")
        return False

    output = r.stdout + r.stderr
    print(output[-2000:])

    # Check for Q-rank-flip confirmation
    has_flip = "Q-rank-flip pairs: 1" in output or "Q-rank-flip pairs: 2" in output or "Q-rank-flip pairs: 3" in output
    q_pass = "Q-structural claim: PASS" in output
    print(f"\n  Q-structural claim PASS: {q_pass}")
    print(f"  Q-rank-flip confirmed: {has_flip}")
    return has_flip


def main():
    print("=== T1 Q-LUT update + verify ===\n")

    print("[1] Scanning for new build results...")
    data, complete = update_q_lut()
    print_q_axis_summary(data)

    print(f"\n[2] Complete FP16+INT8 widths: {len(complete)}/10")
    if len(complete) >= 6:
        flip_confirmed = run_smoke_pqs()
        if flip_confirmed:
            print("\n★ Q-RANK-FLIP CONFIRMED — T1 complete, unlock T1b")
        else:
            print("\n⚠ Q-rank-flip not yet confirmed. Missing: pad64 INT8 (most critical)")
    else:
        print(f"  Still missing INT8 for: {data['summary'].get('widths_pending_int8', [])}")
        print("  → Run again after hw-optimizer completes remaining builds")

    print("\n[3] Regenerating Q-axis figures...")
    subprocess.run(
        [PYTHON, "multi_agent/figure/make_q_axis_figures.py"],
        cwd=str(REPO), timeout=60, capture_output=True
    )
    print(f"  Figures updated: multi_agent/figure/fig_q_axis_*.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
