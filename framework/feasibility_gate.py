#!/usr/bin/env python3
"""P2.1 — feasibility gate for the Pyramid+H800 coarse search space.

Two failure modes were considered:
  (A) int8 build-failure  -> P2.0 established there is NONE at coarse-stage
      granularity on TVM native int8 (60/60 int8 widths built; the bridge's
      align=128 gate is a TRT artifact). So build-feasibility is trivially True.
  (B) reduced-precision AP collapse -> the ONLY observed signal: width
      (24,64,128) has fp32 AP70=0.592 but fp16 AND int8 AP70 -> 0.0. This is the
      single frontier_01 anomaly (降精度精度悬崖, not a bug). Every other
      width/precision in original60 is feasible.

With exactly ONE positive example, a *learned* classifier is not identifiable.
The honest artifact is therefore a CONSERVATIVE RULE-BASED GATE:
  - hard-infeasible: the known collapse (width, reduced-precision) pair.
  - risk score: proximity (in stage-width space) to the collapse anchor, used
    only to flag reduced-precision candidates for AP re-verification when they
    reach the Pareto front (the SMBO loop verifies risky front points by real
    finetune). fp32 is always feasible (no reduced-precision collapse).
This gate is a GATE + RISK FLAG, not a discriminative model, and says so.
"""
from __future__ import annotations
import json
from pathlib import Path

# The single empirically-observed reduced-precision AP collapse (frontier_01).
COLLAPSE_ANCHORS = [((24, 64, 128), "fp16"), ((24, 64, 128), "int8")]
COLLAPSE_WIDTH = (24, 64, 128)


def _dist(w, ref) -> float:
    """Relative L1 distance in stage-width space (scale-normalized per stage)."""
    return sum(abs(a - b) / max(b, 1) for a, b in zip(w, ref)) / len(w)


def feasibility(width, precision: str) -> dict:
    w = tuple(int(x) for x in width)
    if (w, precision) in COLLAPSE_ANCHORS:
        return {"feasible": False, "hard_infeasible": True, "risk": 1.0,
                "reason": "known reduced-precision AP collapse (frontier_01)"}
    if precision == "fp32":
        return {"feasible": True, "hard_infeasible": False, "risk": 0.0,
                "reason": "fp32 never shows reduced-precision collapse"}
    # reduced precision: feasible but carry a risk flag by proximity to anchor
    d = _dist(w, COLLAPSE_WIDTH)
    risk = max(0.0, 1.0 - d)                      # 1 at the anchor, decays with distance
    verify = risk >= 0.6                          # flag near-anchor reduced-prec configs
    return {"feasible": True, "hard_infeasible": False, "risk": round(risk, 3),
            "verify_ap_if_on_front": verify,
            "reason": "reduced precision; verify AP by finetune only if near collapse anchor and on Pareto front"}


def gate_report(out_path: Path | None = None) -> dict:
    """Self-report: apply the gate to the full 1029 grid, count infeasible/at-risk."""
    import itertools
    W0 = [16, 24, 32, 40, 48, 56, 64]
    W1 = [32, 48, 64, 80, 96, 112, 128]
    W2 = [64, 96, 128, 160, 192, 224, 256]
    n_infeasible = n_risky = 0
    for w in itertools.product(W0, W1, W2):
        for p in ("fp32", "fp16", "int8"):
            f = feasibility(w, p)
            n_infeasible += not f["feasible"]
            n_risky += bool(f.get("verify_ap_if_on_front"))
    rep = {"schema": "feasibility_gate_v1", "kind": "rule_based_gate_plus_risk_flag",
           "positives_in_data": len(COLLAPSE_ANCHORS), "learned_classifier": False,
           "learned_classifier_reason": "only 1 collapse width observed; not identifiable",
           "collapse_anchors": [[list(w), p] for w, p in COLLAPSE_ANCHORS],
           "grid_configs": 1029, "hard_infeasible": n_infeasible,
           "reduced_prec_at_risk_flagged": n_risky}
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(out_path, "w"), ensure_ascii=False, indent=1)
    return rep


if __name__ == "__main__":
    r = gate_report(Path("/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/"
                         "generated/original60_quant_20260627/space/feasibility_gate_v1.json"))
    print(json.dumps(r, ensure_ascii=False, indent=1))
    # spot checks
    for w, p in [((24, 64, 128), "fp16"), ((24, 64, 128), "fp32"),
                 ((32, 64, 128), "fp16"), ((64, 128, 256), "int8")]:
        print(w, p, "->", feasibility(w, p))
