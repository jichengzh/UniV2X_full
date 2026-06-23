"""ego×RSU 联合二维网格热图: 诚实口径 DS(τ_ego, τ_comm)。
耦合: τ_RSU = τ_ego + τ_comm (β=1, ego/RSU 同款加速网络)。
code = τ_ego*1000 + τ_comm; tag = grid_g{code}; 超时记 DS=0。
渲染: /data/jichengzhi_v2x/envs/v2xverse/bin/python
"""
import json, glob, re, math, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np

BASE = "/data/jichengzhi_v2x/V2Xverse/results"
OUT = "/data/jichengzhi_v2x/percinj_out"; os.makedirs(OUT, exist_ok=True)
EGOS = [0, 200, 400, 600]
COMMS = [0, 100, 200, 400]
PREFIX, CFG = "grid", "g"

def load(code):
    out = {}
    for d in glob.glob("%s/results_driving_%s_%s%d_r*_n*" % (BASE, PREFIX, CFG, code)):
        m = re.search(r"%s%d_r(\d+)_n(\d+)$" % (CFG, code), d)
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
                out.setdefault(r, []).append((None, True))     # 灾难
            else:
                out.setdefault(r, []).append((sc["score_composed"], False))
        except Exception:
            pass
    return out

codes = [e * 1000 + c for e in EGOS for c in COMMS]
T = {code: load(code) for code in codes}
# tp0-clean = baseline g0 干净路线; matched = 与所有点交集
g0 = T[0]
clean = set(r for r, reps in g0.items()
            if sum(1 for ds, to in reps if (not to) and ds is not None and ds >= 99.9) >= 2)
common = set(clean)
for code in codes:
    common &= set(T[code].keys())
common = sorted(common)

def honest(code):
    reps = [(ds, to) for r in common for (ds, to) in T[code].get(r, [])]
    if not reps:
        return float("nan"), float("nan"), 0
    n = len(reps); ncat = sum(1 for _, to in reps if to)
    dsh = [ds for ds, to in reps if not to] + [0.0] * ncat
    return sum(dsh) / n, 100.0 * ncat / n, n

DS = np.full((len(EGOS), len(COMMS)), np.nan)
CAT = np.full((len(EGOS), len(COMMS)), np.nan)
print("matched=%d 路" % len(common))
print("τ_ego | τ_comm | τ_RSU | 诚实DS | 灾难% | n")
for i, e in enumerate(EGOS):
    for j, c in enumerate(COMMS):
        ds, cat, n = honest(e * 1000 + c)
        DS[i, j] = ds; CAT[i, j] = cat
        print("  %4d | %5d | %5d | %5.1f | %5.1f | %d" % (e, c, e + c, ds, cat, n))

# ---- 热图 ----
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(DS, origin="lower", cmap="RdYlGn", vmin=40, vmax=100, aspect="auto")
ax.set_xticks(range(len(COMMS))); ax.set_xticklabels([str(c) for c in COMMS])
ax.set_yticks(range(len(EGOS))); ax.set_yticklabels([str(e) for e in EGOS])
ax.set_xlabel(r"RSU communication latency  $\tau_{comm}$ (ms)   [$\tau_{RSU}=\tau_{ego}+\tau_{comm}$]", fontsize=11)
ax.set_ylabel(r"ego compute latency  $\tau_{ego}$ (ms)", fontsize=11)
ax.set_title("ego x RSU joint latency grid -> honest Driving Score\n"
             "(beta=1: ego & RSU share accelerated net; H800 _1 full-traffic, %d routes x N=3)" % len(common),
             fontsize=11)
for i in range(len(EGOS)):
    for j in range(len(COMMS)):
        if not np.isnan(DS[i, j]):
            ax.text(j, i, "DS %.0f\ncat %.0f%%\n(rsu %d)" % (DS[i, j], CAT[i, j], EGOS[i] + COMMS[j]),
                    ha="center", va="center", fontsize=8,
                    color="black" if DS[i, j] > 60 else "white")
fig.colorbar(im, ax=ax, label="honest Driving Score (timeout=0)")
fig.tight_layout(); fig.savefig(OUT + "/percinj_grid_heatmap.png", dpi=145); plt.close()

# CSV
with open(OUT + "/percinj_grid.csv", "w") as f:
    f.write("tau_ego,tau_comm,tau_rsu,honest_DS,catastrophe_pct,n\n")
    for i, e in enumerate(EGOS):
        for j, c in enumerate(COMMS):
            f.write("%d,%d,%d,%.1f,%.1f,%d\n" % (e, c, e + c, DS[i, j], CAT[i, j], honest(e * 1000 + c)[2]))
print("-> percinj_grid_heatmap.png + percinj_grid.csv")
