"""AP x tau 闭环网格分析: honest DS(tau, drop) 热图 + iso-DS, 同 (tau=0,drop=0)-clean 交集路集。
   code = tau_ms*100 + drop_pct ; tag = grid_g{code} ; 超时记 DS=0。
   knob->veh_AP50 标定(离线200帧, 线性插值)。
   渲染: /data/jichengzhi_v2x/envs/v2xverse/bin/python (或任意有 matplotlib 的)
"""
import json, glob, re, math, os, sys
BASE = "/exdata/jichengzhi/V2Xverse_apknob/results"
OUT = "/exdata/jichengzhi/grid_out"; os.makedirs(OUT, exist_ok=True)
TAUS = [0, 200, 400, 600, 800]
DROPS = [0.0, 0.25, 0.5, 0.7, 0.85]
ROUTES = [3, 17, 18, 104, 136, 312, 317, 324]
# 离线标定点 drop->veh_AP50
_CAL = [(0.0, 0.841), (0.25, 0.783), (0.5, 0.559), (0.75, 0.310), (0.9, 0.266)]
def drop2ap(d):
    if d <= _CAL[0][0]: return _CAL[0][1]
    if d >= _CAL[-1][0]: return _CAL[-1][1]
    for i in range(len(_CAL) - 1):
        x0, y0 = _CAL[i]; x1, y1 = _CAL[i + 1]
        if x0 <= d <= x1:
            return y0 + (d - x0) / (x1 - x0) * (y1 - y0)
    return float("nan")

ONLY_TAUS = [int(x) for x in os.environ.get("ONLY_TAUS", ",".join(map(str, TAUS))).split(",")]

def code_of(t, d): return t * 100 + int(round(d * 100))
def load(code):
    out = {}
    for r in ROUTES:
        for rep in (1, 2, 3):
            d = "%s/results_driving_grid_g%d_r%d_n%d" % (BASE, code, r, rep)
            fs = glob.glob(d + "/v2x_final/town05_short_collab/*/ego_vehicle_0/results.json")
            if not fs: continue
            try:
                rec = json.load(open(fs[0]))["_checkpoint"]["global_record"]
                st = rec.get("status")
                if not st: continue
                sc = rec.get("scores", {})
                if st == "TIMEOUT_SKIP" or "score_composed" not in sc:
                    out.setdefault(r, []).append((None, True))
                else:
                    out.setdefault(r, []).append((sc["score_composed"], False))
            except Exception: pass
    return out

T = {}
for t in ONLY_TAUS:
    for d in DROPS:
        T[code_of(t, d)] = load(code_of(t, d))
# clean = (tau=0,drop=0) DS>=99.9 >=2/3
g0 = T.get(0, {})
clean = set(r for r, reps in g0.items()
            if sum(1 for ds, to in reps if (not to) and ds is not None and ds >= 99.9) >= 2)
common = set(clean)
for c in T:
    common &= set(T[c].keys())
common = sorted(common)
def honest(code):
    reps = [(ds, to) for r in common for (ds, to) in T[code].get(r, [])]
    if not reps: return float("nan"), 0.0, 0
    n = len(reps); ncat = sum(1 for _, to in reps if to)
    dsh = [ds for ds, to in reps if not to] + [0.0] * ncat
    return sum(dsh) / n, 100.0 * ncat / n, n

print("== AP x tau 网格 (clean 交集 %d 路: %s) ==" % (len(common), common))
print("tau\\drop | " + " | ".join("d%.2f(AP%.2f)" % (d, drop2ap(d)) for d in DROPS))
DS = {}
for t in ONLY_TAUS:
    row = []
    for d in DROPS:
        ds, cat, n = honest(code_of(t, d)); DS[(t, d)] = (ds, cat, n)
        row.append("%5.1f%s" % (ds, "*" if cat > 0 else " "))
    print("%4d     | %s" % (t, " | ".join(row)))
print("(* = 含灾难超时; DS 已记 0)")

# CSV
with open(OUT + "/grid_ds.csv", "w") as f:
    f.write("tau_ms,drop,veh_ap50,honest_DS,catastrophe_pct,n\n")
    for t in ONLY_TAUS:
        for d in DROPS:
            ds, cat, n = DS[(t, d)]
            f.write("%d,%.2f,%.3f,%.1f,%.1f,%d\n" % (t, d, drop2ap(d), ds, cat, n))
print("-> grid_ds.csv")

# 热图 (仅完整网格时)
if len(ONLY_TAUS) >= 3 and len(DROPS) >= 3:
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
        Z = np.full((len(ONLY_TAUS), len(DROPS)), np.nan)
        for i, t in enumerate(ONLY_TAUS):
            for j, d in enumerate(DROPS):
                Z[i, j] = DS[(t, d)][0]
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        im = ax.imshow(Z, origin="lower", cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(DROPS))); ax.set_xticklabels(["%.2f\n(AP%.2f)" % (d, drop2ap(d)) for d in DROPS])
        ax.set_yticks(range(len(ONLY_TAUS))); ax.set_yticklabels([str(t) for t in ONLY_TAUS])
        ax.set_xlabel("perception degradation  ap_drop  (-> veh AP50)", fontsize=11)
        ax.set_ylabel(r"ego inference latency  $\tau_{perc}$ (ms)", fontsize=11)
        ax.set_title("AP x latency joint -> honest Driving Score\n(CoDriving, featmask=1, H800 _1 full-traffic, %d routes x N=3)" % len(common), fontsize=11)
        for i in range(len(ONLY_TAUS)):
            for j in range(len(DROPS)):
                if not np.isnan(Z[i, j]):
                    cat = DS[(ONLY_TAUS[i], DROPS[j])][1]
                    ax.text(j, i, "%.0f%s" % (Z[i, j], "\ncat%.0f%%" % cat if cat > 0 else ""),
                            ha="center", va="center", fontsize=8, color="black" if Z[i, j] > 50 else "white")
        try:
            CS = ax.contour(Z, levels=[40, 60, 80], colors="k", linewidths=0.8, alpha=0.5)
            ax.clabel(CS, inline=True, fontsize=7, fmt="%d")
        except Exception: pass
        fig.colorbar(im, ax=ax, label="honest Driving Score (timeout=0)")
        fig.tight_layout(); fig.savefig(OUT + "/grid_ap_tau_heatmap.png", dpi=145); plt.close()
        print("-> grid_ap_tau_heatmap.png")
    except Exception as e:
        print("heatmap skipped:", e)
