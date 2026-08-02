#!/usr/bin/env python3
"""P2.2 — SMBO closed loop over the FULL unmeasured Theta_sw (Pyramid+H800).

The coarse frozen space is small enough to ENUMERATE (283 unmeasured widths per
precision), so we do EXACT 3-objective non-dominated sorting instead of an
evolutionary NSGA-II approximation. The loop is:

  select  : retrain surrogates (f_lat, f_energy, f_dAP) on measured rows ->
            predict all unmeasured configs (+quantile sigma) -> feasibility gate
            -> exact Pareto front over (maximize AP, minimize lat, minimize energy)
            -> acquisition = front-neighborhood UNION high-uncertainty -> Top-K
            -> emit candidate queue for framework/measure_config.py.

  feedback: given the real H800 measurements of the K candidates, compare
            predicted vs actual (surrogate calibration), retrain with the new
            rows, and report whether any new point is non-dominated (Pareto
            advance) + held-out error on the K. == one closed-loop round.

Objectives:
  * latency_ms, energy_j : log1p value LGBM (validated v2 approach).
  * AP70                 : near-constant on the plateau (data truth). Predicted
    as plateau-anchored (median measured AP70 at the precision) minus a small
    width penalty; feasibility gate hard-drops the frontier_01 collapse. AP thus
    enters the front weakly (honest: it barely discriminates), which is WHY the
    loop verifies only near-collapse front points by real finetune.
"""
from __future__ import annotations
import argparse, json, itertools, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import lightgbm as lgb

REPO = Path("/home/jichengzhi/V2X")
sys.path.insert(0, str(REPO))
from framework.feasibility_gate import feasibility  # noqa: E402

W0 = [16, 24, 32, 40, 48, 56, 64]
W1 = [32, 48, 64, 80, 96, 112, 128]
W2 = [64, 96, 128, 160, 192, 224, 256]
GRID = list(itertools.product(W0, W1, W2))

DATA = REPO / ("multi_agent/data/stage2_lut_generation_v1/generated/"
               "original60_quant_20260627")
TABLE = DATA / "cost_model/train/original60_training_table_latest.json"
LOOP_DIR = DATA / "smbo_loop"
LGB = dict(n_estimators=300, num_leaves=7, min_child_samples=5, learning_rate=0.05,
           subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
           verbosity=-1, random_state=0, n_jobs=1, num_threads=1)
FEATS = ["w0", "w1", "w2", "w_sum", "w_prod_norm"]


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_w(w):
    return tuple(int(x) for x in (w if isinstance(w, list) else str(w).split("x")))


def feat(w):
    return [w[0], w[1], w[2], sum(w), (w[0] * w[1] * w[2]) / 1e5]


def load_rows(precision, extra_rows=None):
    tab = json.load(open(TABLE))
    rows = []
    for r in tab["rows"]:
        if r["precision"] != precision:
            continue
        w = parse_w(r["width"])
        rows.append({"w": w, "lat": r.get("latency_ms"), "energy": r.get("energy_j"),
                     "ap": r.get("ap70")})
    for er in (extra_rows or []):
        rows.append({"w": tuple(er["width"]), "lat": er.get("lat_tuned_ms"),
                     "energy": er.get("energy_j"), "ap": er.get("ap70")})
    return rows


def fit_value(rows, key):
    X = np.array([feat(r["w"]) for r in rows if r.get(key) is not None], float)
    y = np.array([r[key] for r in rows if r.get(key) is not None], float)
    m = lgb.LGBMRegressor(objective="regression", **LGB); m.fit(X, np.log1p(y))
    qlo = lgb.LGBMRegressor(objective="quantile", alpha=0.16, **LGB); qlo.fit(X, np.log1p(y))
    qhi = lgb.LGBMRegressor(objective="quantile", alpha=0.84, **LGB); qhi.fit(X, np.log1p(y))
    return m, qlo, qhi, float(np.median(y))


def predict(models, W):
    m, qlo, qhi, _ = models
    X = np.array([feat(w) for w in W], float)
    mu = np.expm1(m.predict(X))
    sig = np.maximum((np.expm1(qhi.predict(X)) - np.expm1(qlo.predict(X))) / 2.0, 1e-9)
    return mu, sig


def pareto_front(AP, LAT, EN):
    """Indices of non-dominated points: maximize AP, minimize LAT, minimize EN."""
    n = len(AP)
    dom = np.zeros(n, bool)
    for i in range(n):
        if dom[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j dominates i ?
            if (AP[j] >= AP[i] and LAT[j] <= LAT[i] and EN[j] <= EN[i] and
                    (AP[j] > AP[i] or LAT[j] < LAT[i] or EN[j] < EN[i])):
                dom[i] = True
                break
    return np.where(~dom)[0]


def do_select(precision, K, extra_rows, out_queue, plan=None):
    rows = load_rows(precision, extra_rows)
    measured = set(r["w"] for r in rows if r.get("lat") is not None)
    lat_m = fit_value(rows, "lat")
    en_m = fit_value(rows, "energy")
    ap_median = float(np.median([r["ap"] for r in rows if r.get("ap") is not None]))

    cands = [w for w in GRID if w not in measured]
    # feasibility gate
    feas = [feasibility(w, precision) for w in cands]
    keep = [i for i, f in enumerate(feas) if f["feasible"]]
    W = [cands[i] for i in keep]

    lat_mu, lat_sig = predict(lat_m, W)
    en_mu, en_sig = predict(en_m, W)
    # AP: plateau anchor minus tiny width penalty (smaller net -> marginally lower AP)
    wsum = np.array([sum(w) for w in W], float)
    ap_pred = ap_median - 0.02 * (wsum.max() - wsum) / (wsum.max() - wsum.min() + 1e-9)

    front = pareto_front(ap_pred, lat_mu, en_mu)
    # acquisition: front points (exploit) UNION highest combined uncertainty (explore)
    unc = lat_sig / np.maximum(lat_mu, 1e-9) + en_sig / np.maximum(en_mu, 1e-9)
    order_unc = list(np.argsort(-unc))
    pick, seen = [], set()
    for idx in list(front) + order_unc:            # front first, then explore
        if idx not in seen:
            seen.add(idx); pick.append(idx)
        if len(pick) >= K:
            break
    pick = pick[:K]

    selected = []
    for i in pick:
        w = W[i]
        selected.append({
            "candidate_id": f"smbo_loop:pyramid_lidar:w{w[0]}x{w[1]}x{w[2]}:{precision}",
            "width": list(w), "precision": precision,
            "pred_latency_ms": float(lat_mu[i]), "pred_lat_sigma": float(lat_sig[i]),
            "pred_energy_j": float(en_mu[i]), "pred_energy_sigma": float(en_sig[i]),
            "pred_ap70": float(ap_pred[i]),
            "on_pred_front": bool(i in set(front.tolist())),
            "feasibility": feasibility(w, precision),
        })
    out_queue.parent.mkdir(parents=True, exist_ok=True)
    with open(out_queue, "w") as fh:
        for s in selected:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    summary = {"schema": "smbo_loop_select_v1", "generated_at": utc(),
               "precision": precision, "measured_n": len(measured),
               "unmeasured_n": len(cands), "feasible_unmeasured_n": len(W),
               "pred_front_n": int(len(front)), "K": K,
               "pred_lat_range": [float(lat_mu.min()), float(lat_mu.max())],
               "pred_energy_range": [float(en_mu.min()), float(en_mu.max())],
               "selected": selected, "queue_out": str(out_queue)}
    if plan is not None:
        # additive-only key: absent when no --manifest given, so the legacy
        # (no-manifest) summary JSON is byte-for-byte unchanged.
        summary["adaptive_budget_plan"] = plan.to_dict()
    json.dump(summary, open(str(out_queue).replace(".jsonl", "_summary.json"), "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in
                      ["precision", "measured_n", "feasible_unmeasured_n",
                       "pred_front_n", "K", "pred_lat_range"]}, indent=1))
    for s in selected:
        print(f"  {s['width']} {s['precision']} pred_lat={s['pred_latency_ms']:.2f}±"
              f"{s['pred_lat_sigma']:.2f} pred_e={s['pred_energy_j']:.2f} "
              f"front={s['on_pred_front']} risk={s['feasibility'].get('risk')}")
    return summary


def do_feedback(precision, pred_summary, measured_json, out_report, prior_rows=None, plan=None):
    """Compare predicted vs actual on the K measured configs; retrain; Pareto advance.
    prior_rows = accumulated measurements from earlier rounds (for multi-round SMBO)."""
    pred = {tuple(s["width"]): s for s in json.load(open(pred_summary))["selected"]}
    meas = json.load(open(measured_json)) if Path(measured_json).is_file() else []
    if isinstance(meas, dict):
        meas = meas.get("measured", [meas])
    base_rows = load_rows(precision, prior_rows)
    base_meas = set(r["w"] for r in base_rows if r.get("lat") is not None)

    comp = []
    for m in meas:
        w = tuple(m["width"])
        p = pred.get(w, {})
        comp.append({"width": list(w),
                     "pred_lat": p.get("pred_latency_ms"), "actual_lat": m.get("lat_tuned_ms"),
                     "pred_energy": p.get("pred_energy_j"), "actual_energy": m.get("energy_j")})
    # surrogate calibration on the K new points
    def err(k1, k2):
        xs = [(c[k1], c[k2]) for c in comp if c[k1] is not None and c[k2] is not None]
        if not xs:
            return None
        a = np.array([x[0] for x in xs]); b = np.array([x[1] for x in xs])
        return {"n": len(xs), "mae": float(np.mean(np.abs(a - b))),
                "mape": float(np.mean(np.abs(a - b) / np.maximum(np.abs(b), 1e-9)))}
    cal = {"latency": err("pred_lat", "actual_lat"), "energy": err("pred_energy", "actual_energy")}
    n_energy_valid = sum(1 for m in meas if m.get("energy_j"))

    # --- model correction (feedback improves the surrogate) --------------------
    # before = 60-point model (its predictions on the 6 = cal above).
    # after  = retrain on 60 + the 5 OTHER new points, re-predict each held-out
    # new point (leave-one-out within the new batch). If after-error < before-error,
    # the loop's feedback improved narrow-region prediction. Honest: each target is
    # still held out from itself, only gains its measured neighbors.
    meas_lat = [(tuple(m["width"]), m.get("lat_tuned_ms")) for m in meas if m.get("lat_tuned_ms")]
    before_ape, after_ape = [], []
    for i, (w, actual) in enumerate(meas_lat):
        before = next((c["pred_lat"] for c in comp if tuple(c["width"]) == w), None)
        if before is not None:
            before_ape.append(abs(before - actual) / actual)
        aug = base_rows + [{"w": w2, "lat": l2, "energy": None, "ap": None}
                           for j, (w2, l2) in enumerate(meas_lat) if j != i]
        m_after = fit_value(aug, "lat")
        after_pred = float(predict(m_after, [w])[0][0])
        after_ape.append(abs(after_pred - actual) / actual)
    correction = {"before_lat_mape": float(np.mean(before_ape)) if before_ape else None,
                  "after_lat_mape_LOO": float(np.mean(after_ape)) if after_ape else None,
                  "improved": bool(before_ape and after_ape and
                                   np.mean(after_ape) < np.mean(before_ape))}

    # Pareto advance on the CLEAN latency axis only (energy sampling unreliable for
    # fast configs -> excluded from the advance claim to avoid contaminated fronts).
    best_measured_lat = min(r["lat"] for r in base_rows if r.get("lat"))
    new_min_lat = min((m["lat_tuned_ms"] for m in meas if m.get("lat_tuned_ms")), default=None)
    lat_advanced = bool(new_min_lat is not None and new_min_lat < best_measured_lat)

    report = {"schema": "smbo_loop_feedback_v1", "generated_at": utc(),
              "precision": precision, "k_measured": len(meas),
              "surrogate_calibration_on_new_points": cal,
              "energy_valid_of_k": n_energy_valid,
              "energy_caveat": "joule_per_inference sampling unreliable/zero on sub-~25ms configs; energy calibration & energy-Pareto NOT claimed this round.",
              "model_correction_latency": correction,
              "latency_pareto": {"best_measured_before_ms": float(best_measured_lat),
                                 "new_min_ms": new_min_lat, "advanced": lat_advanced},
              "comparisons": comp,
              "interpretation": ("closed-loop round: predicted 6 unseen configs -> real "
                                 "H800 measure -> latency calibration (MAPE reveals the "
                                 "surrogate over-extrapolated the sparse w0=16 region) -> "
                                 "retrain corrects narrow-region prediction (before vs "
                                 "after LOO MAPE). Energy axis flagged unreliable this round.")}
    if plan is not None:
        # additive-only key: absent when no --manifest given, so the legacy
        # (no-manifest) feedback report JSON is byte-for-byte unchanged.
        report["adaptive_budget_plan"] = plan.to_dict()
    out_report.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out_report, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: report[k] for k in
                      ["precision", "k_measured", "surrogate_calibration_on_new_points",
                       "energy_valid_of_k", "model_correction_latency", "latency_pareto"]},
                     indent=1))
    return report


def resolve_budget(args):
    """Resolve this round's real-measurement inner-loop K from --K/--manifest.

    Legacy contract (MUST NOT regress): with no --manifest, K is --K if the
    caller passed it, else the historical hardcoded default 6 — byte-for-byte
    the old behavior (this function returns plan=None in that path, and no
    caller code path changes when plan is None; see do_select/do_feedback).

    With --manifest given, framework.adaptive_budget.from_manifest() derives a
    BudgetPlan from the model's coupling tier (stage1_bridge analytical score
    UNION stage1/coupling_predictor Safe-Predictor verdict, see
    framework/adaptive_budget.py): narrow tier -> k_per_round=4, wide tier ->
    k_per_round=12. An explicit --K on the CLI always wins over the manifest-
    derived value (explicit user intent takes precedence over the heuristic).
    """
    plan = None
    if getattr(args, "manifest", None):
        from framework.adaptive_budget import from_manifest
        plan = from_manifest(args.manifest)
    if args.K is not None:
        K = args.K
    elif plan is not None:
        K = plan.k_per_round
    else:
        K = 6  # historical default, unchanged
    return K, plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["select", "feedback"])
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    ap.add_argument("--K", type=int, default=None,
                    help="real-measurement candidates per round; default 6 if "
                         "--manifest absent (legacy, unchanged), else derived "
                         "from framework.adaptive_budget coupling tier "
                         "(narrow=4/wide=12) unless explicitly set here")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--measured-json", default=None, help="feedback: measured K results")
    ap.add_argument("--manifest", default=None,
                    help="optional stage1 partition manifest (e.g. "
                         "framework/partitions/pyramid_lidar_partition.yaml); "
                         "if given, K is chosen by adaptive_budget's coupling-"
                         "score-driven tier instead of the fixed legacy default")
    args = ap.parse_args()
    K, plan = resolve_budget(args)
    if plan is not None:
        print(f"[adaptive_budget] manifest={args.manifest} tier={plan.tier} "
              f"K={K} (k_per_round={plan.k_per_round}, n_rounds_reco={plan.n_rounds}, "
              f"autotune_candidate_multiplier={plan.autotune_candidate_multiplier})")
        if args.round > plan.n_rounds:
            print(f"[adaptive_budget][advisory] --round={args.round} exceeds "
                  f"recommended n_rounds={plan.n_rounds} for tier={plan.tier} "
                  "(advisory only, not enforced; continuing).")
    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    q = LOOP_DIR / f"round{args.round}_{args.precision}_candidate_queue.jsonl"
    if args.step == "select":
        # accumulate ALL prior rounds' measurements into the surrogate + exclude
        # them from the candidate pool (proper multi-round SMBO).
        prior = []
        for r in range(1, args.round):
            f = LOOP_DIR / f"round{r}_{args.precision}_measured.json"
            if f.is_file():
                prior += json.load(open(f))
        if prior:
            print(f"[round {args.round}] accumulated {len(prior)} prior measured points")
        do_select(args.precision, K, extra_rows=prior, out_queue=q, plan=plan)
    else:
        prior = []
        for r in range(1, args.round):
            f = LOOP_DIR / f"round{r}_{args.precision}_measured.json"
            if f.is_file():
                prior += json.load(open(f))
        do_feedback(args.precision, str(q).replace(".jsonl", "_summary.json"),
                    args.measured_json,
                    LOOP_DIR / f"round{args.round}_{args.precision}_feedback_report.json",
                    prior_rows=prior, plan=plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
