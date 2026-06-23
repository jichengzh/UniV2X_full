"""联合折线图: ego-only(τ_RSU=0, 原18点) vs 耦合(τ_RSU=τ_ego+50ms comm, 10点)。
诚实口径(超时记 DS=0), 同一 tp0-clean matched 路集。
渲染: /data/jichengzhi_v2x/envs/v2xverse/bin/python
"""
import json, glob, re, math, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = "/data/jichengzhi_v2x/V2Xverse/results"
OUT = "/data/jichengzhi_v2x/percinj_out"; os.makedirs(OUT, exist_ok=True)
# ego-only(RSU=0): tp{τ_ego}; 耦合(RSU=τ_ego+50): c50e{τ_ego}
EGOS_EO = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 800, 900, 1000]
EGOS_CP = [0, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

def load(tag):
    out = {}
    for d in glob.glob("%s/results_driving_percinj_%s_r*_n*" % (BASE, tag)):
        m = re.search(r"%s_r(\d+)_n(\d+)$" % tag, d)
        if not m:
            continue
        r = int(m.group(1))
        fs = glob.glob(d + "/v2x_final/town05_short_collab/*/ego_vehicle_0/results.json")
        if not fs:
            continue
        try:
            rec = json.load(open(fs[0]))["_checkpoint"]["global_record"]
            st = rec.get("status")
            if not st:
                continue
            sc = rec.get("scores", {})
            if st == "TIMEOUT_SKIP" or "score_composed" not in sc:
                out.setdefault(r, []).append((None, True))
            else:
                out.setdefault(r, []).append((sc["score_composed"], False))
        except Exception:
            pass
    return out

EO = {e: load("tp%d" % e) for e in EGOS_EO}
CP = {e: load("c50e%d" % e) for e in EGOS_CP}

tp0 = EO[0]
clean = set(r for r, reps in tp0.items()
            if sum(1 for ds, to in reps if (not to) and ds is not None and ds >= 99.9) >= 2)
common = set(clean)
for e in EGOS_EO:
    common &= set(EO[e].keys())
for e in EGOS_CP:
    common &= set(CP[e].keys())
common = sorted(common)

def ci95(xs):
    n = len(xs)
    if n < 2: return 0.0
    m = sum(xs) / n; sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)); return 1.96 * sd / math.sqrt(n)

def honest(table, e):
    reps = [(ds, to) for r in common for (ds, to) in table[e].get(r, [])]
    if not reps: return float("nan"), float("nan"), 0.0
    n = len(reps); ncat = sum(1 for _, to in reps if to)
    dsh = [ds for ds, to in reps if not to] + [0.0] * ncat
    return sum(dsh) / n, ci95(dsh), 100.0 * ncat / n

eo_ds = [honest(EO, e)[0] for e in EGOS_EO]; eo_ci = [honest(EO, e)[1] for e in EGOS_EO]
cp_ds = [honest(CP, e)[0] for e in EGOS_CP]; cp_ci = [honest(CP, e)[1] for e in EGOS_CP]

print("matched=%d 路" % len(common))
print("--- ego-only (τ_RSU=0) ---")
for k, e in enumerate(EGOS_EO):
    print("  τ_ego=%4d  DS=%.1f±%.1f  cat=%.1f%%" % (e, eo_ds[k], eo_ci[k], honest(EO, e)[2]))
print("--- coupled (τ_RSU=τ_ego+50) ---")
for k, e in enumerate(EGOS_CP):
    print("  τ_ego=%4d (rsu%d)  DS=%.1f±%.1f  cat=%.1f%%" % (e, e + 50, cp_ds[k], cp_ci[k], honest(CP, e)[2]))

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(EGOS_EO, eo_ds, yerr=eo_ci, fmt="o-", color="tab:blue", lw=2, ms=6, capsize=3,
            label=r"ego-only  ($\tau_{RSU}=0$)")
ax.errorbar(EGOS_CP, cp_ds, yerr=cp_ci, fmt="s--", color="tab:red", lw=2, ms=7, capsize=4,
            label=r"coupled  ($\tau_{RSU}=\tau_{ego}+50$ms)")
ax.set_xlabel(r"ego compute (inference) latency  $\tau_{ego}$ (ms)", fontsize=11)
ax.set_ylabel("Honest Driving Score (timeout=0)", fontsize=11)
ax.set_ylim(40, 102); ax.grid(alpha=0.3)
ax.set_title("Latency -> V2X driving degradation: ego-only vs ego+RSU coupled\n"
             "(beta=1, RSU shares accelerated net + 50ms comm; H800 _1 full-traffic, %d routes x N=3)" % len(common),
             fontsize=11)
ax.legend(fontsize=10, loc="lower left")
fig.tight_layout(); fig.savefig(OUT + "/percinj_c50_compare.png", dpi=145); plt.close()

with open(OUT + "/percinj_c50_compare.csv", "w") as f:
    f.write("arm,tau_ego,tau_rsu,honest_DS,ci95,catastrophe_pct\n")
    for k, e in enumerate(EGOS_EO):
        f.write("ego_only,%d,0,%.1f,%.1f,%.1f\n" % (e, eo_ds[k], eo_ci[k], honest(EO, e)[2]))
    for k, e in enumerate(EGOS_CP):
        f.write("coupled,%d,%d,%.1f,%.1f,%.1f\n" % (e, e + 50, cp_ds[k], cp_ci[k], honest(CP, e)[2]))
print("-> percinj_c50_compare.{png,csv}  (ego-only %d pts + coupled %d pts)" % (len(EGOS_EO), len(EGOS_CP)))
