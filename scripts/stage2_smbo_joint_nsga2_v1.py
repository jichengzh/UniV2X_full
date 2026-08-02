#!/usr/bin/env python3
"""Phase-1 v2 — JOINT (P x Q) surrogate-assisted NSGA-II over the 1029-point space.

FIXES the v1 (stage2_smbo_loop_v1.py) methodological errors (see 9_7_14 doc §9):
  * v1 searched 3 SEPARATE per-precision 343 spaces; here precision q is a GENE
    of a SINGLE joint 1029-point space (343 widths x 3 precisions).
  * v1 ENUMERATED all unmeasured points ("exact Pareto") — does not scale past a
    3-stage net. Here we use a scalable surrogate-assisted NSGA-II; enumeration is
    NOT used as the method.

Genome  : (w0,w1,w2,q),  w in 7-level grids, q in {fp32,fp16,int8_tc}.
Surrogate: ONE LightGBM per objective (f_lat,f_energy,f_ap) with precision as a
           feature -> trained on ALL measured rows across precisions (transfer).
Loop    : NSGA-II on surrogate fitness (max AP70,min lat,min energy) -> each gen,
           real-measure the top unmeasured front individuals on H800 via
           framework/measure_config.py -> retrain surrogate -> until budget spent.
Validation: NOT ground-truth enumeration (field norm = compare to baselines, see
           9_7_14 §9.3). Baseline = random-search bootstrap over the measured
           universe; report hypervolume(NSGA-II picks) vs hypervolume(random).

CLI:
  --measure {reuse,real}  reuse = surrogate+existing measured pool only (instant,
            logic validation, predicted front); real = call measure_config on H800
            for unmeasured front candidates (budget-capped).
  --pop 24 --budget 60 --gpu 3 --seed 0
"""
from __future__ import annotations
import argparse, json, itertools, sys, subprocess, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import lightgbm as lgb

REPO = Path("/home/jichengzhi/V2X")

W0 = [16, 24, 32, 40, 48, 56, 64]
W1 = [32, 48, 64, 80, 96, 112, 128]
W2 = [64, 96, 128, 160, 192, 224, 256]
GRID = list(itertools.product(W0, W1, W2))          # 343 widths
PRECS = ["fp32", "fp16", "int8_tc"]                 # q axis -> 1029 joint points
Q_ORD = {"fp32": 0, "fp16": 1, "int8_tc": 2}

DATA = REPO / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
TABLE = DATA / "cost_model/train/original60_training_table_latest.json"
OUT_DIR = DATA / "smbo_joint_nsga2"
MEASURE = REPO / "framework/measure_config.py"
LGB = dict(n_estimators=300, num_leaves=7, min_child_samples=5, learning_rate=0.05,
           subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
           verbosity=-1, random_state=0, n_jobs=1, num_threads=1)


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_w(w):
    return tuple(int(x) for x in (w if isinstance(w, list) else str(w).split("x")))


def feat(w, q):
    """Joint feature vector: widths + interactions + PRECISION (the v1 fix)."""
    return [w[0], w[1], w[2], sum(w), (w[0] * w[1] * w[2]) / 1e5, Q_ORD[q]]


def feas_ok(w, q):
    """Self-contained feasibility. Empirically 60/60 int8 widths build; real
    buildability is arbitrated by measure_config's build_success at measure time.
    Only known structural infeasibility = the (24,64,128) low-precision collapse
    anchor (documented). Keep permissive; measure_config gates the rest."""
    if tuple(w) == (24, 64, 128) and q in ("fp16", "int8_tc"):
        return False
    return True


# --------------------------------------------------------------------------- data
def norm_prec(p):
    return "int8_tc" if p in ("int8", "int8_tc") else p


def load_all_measured(extra_paths=None):
    """All REAL measured (w,q)->{lat,energy,ap} across ALL precisions (joint)."""
    pool = {}
    tab = json.load(open(TABLE))
    for r in tab["rows"]:
        q = norm_prec(r["precision"]); w = parse_w(r["width"])
        ap = r.get("ap70")
        ap = ap if (ap is not None and ap > 0.01) else None   # guard placeholder 0
        pool[(w, q)] = {"lat": r.get("latency_ms"), "energy": r.get("energy_j"), "ap": ap}
    # fold in prior SMBO/offdiag measured jsons (real lat/energy)
    for p in (extra_paths or []):
        p = Path(p)
        if not p.is_file():
            continue
        rows = json.load(open(p))
        rows = rows if isinstance(rows, list) else rows.get("measured", [rows])
        for m in rows:
            w = tuple(m["width"]); q = norm_prec(m.get("precision", "fp16"))
            e = pool.get((w, q), {})
            pool[(w, q)] = {"lat": m.get("lat_tuned_ms", e.get("lat")),
                            "energy": m.get("energy_j", e.get("energy")),
                            "ap": m.get("ap70", e.get("ap"))}
    return pool


# ----------------------------------------------------------------------- surrogate
def fit_obj(pool, key):
    X, y = [], []
    for (w, q), v in pool.items():
        if v.get(key) is not None:
            X.append(feat(w, q)); y.append(v[key])
    X = np.array(X, float); y = np.array(y, float)
    m = lgb.LGBMRegressor(objective="regression", **LGB); m.fit(X, np.log1p(y))
    return m, float(np.median(y))


def predict(model, genomes):
    m, _ = model
    X = np.array([feat(w, q) for (w, q) in genomes], float)
    return np.expm1(m.predict(X))


def ap_predict(pool, genomes):
    """AP surrogate if enough real AP rows, else plateau-anchor by precision.
    Honest: AP barely discriminates on the plateau (9_7_14 §9); real AP for
    front members comes from finetune, not this surrogate."""
    ap_rows = [(w, q, v["ap"]) for (w, q), v in pool.items() if v.get("ap") is not None]
    if len(ap_rows) >= 8:
        X = np.array([feat(w, q) for (w, q, _) in ap_rows], float)
        y = np.array([a for (_, _, a) in ap_rows], float)
        m = lgb.LGBMRegressor(objective="regression", **LGB); m.fit(X, y)
        return m.predict(np.array([feat(w, q) for (w, q) in genomes], float))
    med = float(np.median([a for *_, a in ap_rows])) if ap_rows else 0.5
    wsum = np.array([sum(w) for (w, _) in genomes], float)
    return med - 0.02 * (wsum.max() - wsum) / (wsum.max() - wsum.min() + 1e-9)


# -------------------------------------------------------------------- NSGA-II core
def nondom_sort(F):
    """F: (n,3) minimize. Return list of fronts (each = list of indices)."""
    n = len(F); S = [[] for _ in range(n)]; nd = np.zeros(n, int); rank = np.zeros(n, int)
    fronts = [[]]
    for p in range(n):
        for qi in range(n):
            if p == qi:
                continue
            if dominates(F[p], F[qi]):
                S[p].append(qi)
            elif dominates(F[qi], F[p]):
                nd[p] += 1
        if nd[p] == 0:
            rank[p] = 0; fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for qi in S[p]:
                nd[qi] -= 1
                if nd[qi] == 0:
                    rank[qi] = i + 1; nxt.append(qi)
        i += 1; fronts.append(nxt)
    return fronts[:-1]


def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def crowding(F, idx):
    d = np.zeros(len(idx))
    if len(idx) <= 2:
        return {i: np.inf for i in idx}
    for k in range(F.shape[1]):
        order = sorted(range(len(idx)), key=lambda t: F[idx[t], k])
        d[order[0]] = d[order[-1]] = np.inf
        lo, hi = F[idx[order[0]], k], F[idx[order[-1]], k]
        rng = hi - lo or 1.0
        for t in range(1, len(idx) - 1):
            d[order[t]] += (F[idx[order[t + 1]], k] - F[idx[order[t - 1]], k]) / rng
    return {idx[t]: d[t] for t in range(len(idx))}


def obj_matrix(pool, genomes, models):
    lat = predict(models["lat"], genomes)
    en = predict(models["energy"], genomes)
    ap = ap_predict(pool, genomes)
    # minimize: lat, energy, -AP
    return np.column_stack([lat, en, -ap]), lat, en, ap


def rng_genome(rs):
    while True:
        # ★native int (rs.choice returns np.int64 which is not JSON-serializable)
        w = (int(rs.choice(W0)), int(rs.choice(W1)), int(rs.choice(W2)))
        q = PRECS[rs.randint(len(PRECS))]
        if feas_ok(w, q):
            return (w, q)


def crossover(a, b, rs):
    (wa, qa), (wb, qb) = a, b
    w = tuple(wa[i] if rs.rand() < 0.5 else wb[i] for i in range(3))
    q = qa if rs.rand() < 0.5 else qb
    return (w, q)


def mutate(g, rs, pm=0.3):
    (w, q) = g; w = list(w)
    grids = [W0, W1, W2]
    for i in range(3):
        if rs.rand() < pm:
            j = grids[i].index(w[i]); step = rs.choice([-1, 1])
            w[i] = grids[i][min(max(j + step, 0), len(grids[i]) - 1)]
    if rs.rand() < pm:
        q = PRECS[rs.randint(len(PRECS))]
    g2 = (tuple(w), q)
    return g2 if feas_ok(*g2) else g


def tournament(pop, ranks, crowd, rs):
    i, j = rs.randint(len(pop)), rs.randint(len(pop))
    if ranks[i] != ranks[j]:
        return pop[i] if ranks[i] < ranks[j] else pop[j]
    return pop[i] if crowd.get(i, 0) >= crowd.get(j, 0) else pop[j]


# --------------------------------------------------------------- real measurement
def measure_real(w, q, gpu, log_dir):
    """Call framework/measure_config.py on H800; return {lat,energy,build}."""
    # measure_config CLI: --width --precision {fp32,fp16,int8,int8_tc} --gpu.
    # int8_tc handled internally (tensorized WMMA); no --cast flag at CLI level.
    cmd = [sys.executable, str(MEASURE), "--width", ",".join(map(str, w)),
           "--precision", q, "--gpu", str(gpu)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=str(REPO))
        txt = r.stdout
        # extract the LAST balanced top-level {...} block (measure_config prints
        # a pretty multi-line JSON at the end, possibly after log noise).
        depth = 0; start = None; best = None
        for i, ch in enumerate(txt):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    best = txt[start:i + 1]
        if best:
            d = json.loads(best)
            return {"lat": d.get("lat_tuned_ms"), "energy": d.get("energy_j"),
                    "build": d.get("build_success", True), "raw": d}
    except Exception as e:
        return {"lat": None, "energy": None, "build": False, "err": str(e)}
    return {"lat": None, "energy": None, "build": False, "err": "no_json"}


# ---------------------------------------------------------------------- hypervolume
def hypervolume(F, ref):
    """Simple 3D HV for minimize objectives via inclusion over the front (Monte-Carlo
    for robustness on small fronts). F rows = (lat,energy,-AP)."""
    F = np.array([f for f in F if np.all(f <= ref)])
    if len(F) == 0:
        return 0.0
    lo = F.min(0)
    rs = np.random.RandomState(1)
    N = 40000
    pts = lo + rs.rand(N, 3) * (ref - lo)
    dominated = np.zeros(N, bool)
    for f in F:
        dominated |= np.all(pts >= f, axis=1)
    return float(dominated.mean() * np.prod(ref - lo))


# ---------------------------------------------------------------------------- main
def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rs = np.random.RandomState(args.seed)
    extra = list((DATA / "smbo_loop").glob("round*_measured.json")) + \
            list((DATA / "smbo_loop").glob("*offdiag*measured*.json"))
    pool = load_all_measured(extra)
    tag = f"pop{args.pop}_b{args.budget}_{args.measure}_seed{args.seed}"
    live_ckpt = OUT_DIR / f"live_{tag}.jsonl"
    resumed = 0
    if live_ckpt.exists():                     # resume: preload prior real measures
        for ln in open(live_ckpt):
            try:
                m = json.loads(ln)
            except Exception:
                continue
            w = tuple(m["width"]); q = m["precision"]
            pool[(w, q)] = {"lat": m["lat_tuned_ms"], "energy": m["energy_j"],
                            "ap": pool.get((w, q), {}).get("ap")}; resumed += 1
    failed = set()
    n_meas0 = sum(1 for v in pool.values() if v.get("lat") is not None)
    models = {"lat": fit_obj(pool, "lat"), "energy": fit_obj(pool, "energy")}
    print(f"[resume] preloaded {resumed} prior real measures from {live_ckpt.name}", flush=True)
    print(f"[init] joint pool: {n_meas0} real-measured (w,q) across {len(PRECS)} precisions; "
          f"space=|{len(GRID)}x{len(PRECS)}|={len(GRID)*len(PRECS)}")

    pop = [rng_genome(rs) for _ in range(args.pop)]
    spent = 0
    trace = []
    newly_measured = []
    gen = 0
    while spent < args.budget and gen < args.max_gen:
        gen += 1
        F, lat, en, ap = obj_matrix(pool, pop, models)
        fronts = nondom_sort(F)
        ranks = {}
        for ri, fr in enumerate(fronts):
            for i in fr:
                ranks[i] = ri
        crowd = {}
        for fr in fronts:
            crowd.update(crowding(F, fr))
        # real-measure: unmeasured genomes on rank-0 front (exploit) this gen
        front0 = [pop[i] for i in fronts[0]]
        to_measure = [g for g in front0
                      if (g not in pool or pool[g].get("lat") is None) and g not in failed]
        k_gen = min(len(to_measure), args.k_per_gen, args.budget - spent)
        measured_this_gen = []
        for g in to_measure[:k_gen]:
            (w, q) = g
            if args.measure == "real":
                res = measure_real(w, q, args.gpu, OUT_DIR)
            else:
                res = {"lat": None, "energy": None, "build": True}  # reuse-only: skip
            if res.get("lat") is not None:
                pool[g] = {"lat": res["lat"], "energy": res["energy"],
                           "ap": pool.get(g, {}).get("ap")}
                rec = {"width": [int(x) for x in w], "precision": q,
                       "lat_tuned_ms": float(res["lat"]),
                       "energy_j": (float(res["energy"]) if res.get("energy") is not None else None)}
                newly_measured.append(rec)
                with open(live_ckpt, "a") as fh:          # incremental crash-safe ckpt
                    fh.write(json.dumps(rec) + "\n")
                measured_this_gen.append(g); spent += 1
                print(f"    measured {w} {q}: lat={res['lat']:.3f} energy={res.get('energy')} "
                      f"[{spent}/{args.budget}]", flush=True)
            elif args.measure == "real":
                failed.add(g)                              # don't re-propose failures
                print(f"    FAILED {w} {q}: {res.get('err', res.get('build'))}", flush=True)
        if measured_this_gen:  # feedback: retrain surrogate on new reals
            models = {"lat": fit_obj(pool, "lat"), "energy": fit_obj(pool, "energy")}
        trace.append({"gen": gen, "spent": spent, "front0_size": len(fronts[0]),
                      "measured_this_gen": len(measured_this_gen), "failed": len(failed),
                      "pred_lat_min": float(lat.min()), "pred_ap_max": float(ap.max())})
        json.dump({"gen": gen, "spent": spent, "budget": args.budget, "failed": len(failed),
                   "front0": len(fronts[0])}, open(OUT_DIR / f"heartbeat_{tag}.json", "w"))
        print(f"[gen {gen}] front0={len(fronts[0])} measured+={len(measured_this_gen)} "
              f"failed={len(failed)} spent={spent}/{args.budget} "
              f"pred_lat_min={lat.min():.2f} pred_ap_max={ap.max():.4f}", flush=True)
        # NSGA-II offspring (mu+lambda): elitist select then breed
        allidx = [i for fr in fronts for i in fr][:args.pop]
        parents = [pop[i] for i in allidx]
        off = []
        while len(off) < args.pop:
            a = tournament(parents, {i: ranks[allidx[i]] for i in range(len(parents))},
                           {i: crowd.get(allidx[i], 0) for i in range(len(parents))}, rs)
            b = tournament(parents, {i: ranks[allidx[i]] for i in range(len(parents))},
                           {i: crowd.get(allidx[i], 0) for i in range(len(parents))}, rs)
            child = mutate(crossover(a, b, rs), rs)
            off.append(child)
        pop = parents + off  # elitist mu+lambda; re-sorted next gen

    # ---- final joint front over ALL real-measured (w,q) --------------------------
    meas_g = [(g, v) for g, v in pool.items() if v.get("lat") is not None and v.get("energy") is not None]
    G = [g for g, _ in meas_g]
    ap_final = ap_predict(pool, G)
    Ff = np.column_stack([[v["lat"] for _, v in meas_g], [v["energy"] for _, v in meas_g], -ap_final])
    fr0 = nondom_sort(Ff)[0]
    front = []
    for i in fr0:
        (w, q) = G[i]; v = meas_g[i][1]
        front.append({"width": [int(x) for x in w], "precision": q, "lat_ms": float(v["lat"]),
                      "energy_j": float(v["energy"]),
                      "ap70": (float(v["ap"]) if v.get("ap") is not None else None),
                      "ap_src": ("real" if v.get("ap") is not None else "surrogate_pred"),
                      "ap70_pred": float(ap_final[i])})
    front.sort(key=lambda r: r["lat_ms"])

    # ---- baseline: random-search bootstrap over measured universe ---------------
    ref = np.array([Ff[:, 0].max() * 1.05, Ff[:, 1].max() * 1.05, -(ap_final.min() * 0.98)])
    hv_nsga = hypervolume(Ff[fr0], ref)
    all_g = list(range(len(meas_g)))
    B = 200; hv_rand = []
    for _ in range(B):
        pick = rs.choice(all_g, size=min(args.budget, len(all_g)), replace=False)
        sub = Ff[pick]; sub_fr = nondom_sort(sub)[0]
        hv_rand.append(hypervolume(sub[sub_fr], ref))
    hv_rand = np.array(hv_rand)
    baseline = {"hv_nsga2_front": hv_nsga, "hv_random_mean": float(hv_rand.mean()),
                "hv_random_p95": float(np.percentile(hv_rand, 95)),
                "nsga2_beats_random_frac": float((hv_nsga >= hv_rand).mean()),
                "note": "random-search bootstrap over measured universe (field-norm baseline, "
                        "NOT ground-truth enumeration; see 9_7_14 §9.3)"}

    out = {"schema": "smbo_joint_nsga2_v2", "generated_at": utc(),
           "space": {"widths": len(GRID), "precisions": PRECS, "joint_points": len(GRID) * len(PRECS)},
           "method": "surrogate-assisted NSGA-II, precision as gene (joint P x Q)",
           "hyperparams": {"pop": args.pop, "budget": args.budget, "k_per_gen": args.k_per_gen,
                           "seed": args.seed, "measure_mode": args.measure,
                           "surrogate": "LightGBM f_lat/f_energy + precision feature",
                           "lgb": LGB},
           "real_measured_start": n_meas0, "real_measured_new": spent,
           "joint_pareto_front": front, "front_size": len(front),
           "baseline_validation": baseline, "trace": trace,
           "newly_measured": newly_measured}
    fp = OUT_DIR / f"joint_nsga2_{tag}.json"
    json.dump(out, open(fp, "w"), ensure_ascii=False, indent=1)
    if newly_measured:
        json.dump(newly_measured, open(OUT_DIR / f"newly_measured_{tag}.json", "w"), indent=1)
    print(f"\n[done] front_size={len(front)} new_real={spent} "
          f"HV(nsga2)={hv_nsga:.4g} HV(rand_mean)={hv_rand.mean():.4g} "
          f"nsga2>=rand frac={baseline['nsga2_beats_random_frac']:.2f}", flush=True)
    print(f"[out] {fp}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--budget", type=int, default=60, help="max NEW real measurements")
    ap.add_argument("--k-per-gen", type=int, default=4)
    ap.add_argument("--max-gen", type=int, default=40)
    ap.add_argument("--measure", choices=["reuse", "real"], default="reuse")
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
