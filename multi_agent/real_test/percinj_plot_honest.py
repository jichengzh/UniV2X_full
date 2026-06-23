"""最终诚实口径绘图: 感知延迟 τ_perc → V2X驾驶退化 14点曲线
诚实口径: tp0-clean 路线在某档 TIMEOUT_SKIP/无score → 记 DS=0 (灾难性失败),而非排除。
对比旧 percinj_plot.py (排除超时=乐观偏差)。双轴: 诚实平均DS±CI + 灾难(超时)率%。
渲染用 v2xverse env: /data/jichengzhi_v2x/envs/v2xverse/bin/python
"""
import json, glob, re, math, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = "/data/jichengzhi_v2x/V2Xverse/results"
OUT  = "/data/jichengzhi_v2x/percinj_out"; os.makedirs(OUT, exist_ok=True)
TAUS = [0,50,100,150,200,250,300,350,400,450,500,650,800,1000]

def load(tau):
    """返回 {route: [(ds, col, is_catastrophe), ...]}; 超时/无score => ds=None,catastrophe=True"""
    out = {}
    for d in glob.glob("%s/results_driving_percinj_tp%d_r*_n*" % (BASE, tau)):
        m = re.search(r"tp%d_r(\d+)_n(\d+)$" % tau, d)
        if not m: continue
        r = int(m.group(1))
        fs = glob.glob(d + "/v2x_final/town05_short_collab/*/ego_vehicle_0/results.json")
        if not fs: continue
        try:
            rec = json.load(open(fs[0]))["_checkpoint"]["global_record"]
            st = rec.get("status")
            if not st: continue
            sc = rec.get("scores", {})
            if st == "TIMEOUT_SKIP" or "score_composed" not in sc:
                out.setdefault(r, []).append((None, None, True))   # 灾难
            else:
                inf = rec["infractions"]
                col = (inf["collisions_pedestrian"] > 0 or inf["collisions_vehicle"] > 0 or inf["collisions_layout"] > 0)
                out.setdefault(r, []).append((sc["score_composed"], col, False))
        except Exception:
            pass
    return out

T = {t: load(t) for t in TAUS}
# tp0-clean: tp0 时 >=2/3 rep 干净 (DS>=99.9 无碰撞 非灾难)
tp0 = T[0]
clean = set(r for r, reps in tp0.items()
            if sum(1 for ds, c, to in reps if (not to) and ds is not None and ds >= 99.9 and not c) >= 2)
# matched: tp0-clean ∩ 各档都有数据
common = set(clean)
for t in TAUS:
    common &= set(T[t].keys())
common = sorted(common)

def ci95(xs):
    n = len(xs)
    if n < 2: return 0.0
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return 1.96 * sd / math.sqrt(n)

rows = []
for t in TAUS:
    reps = [(ds, c, to) for r in common for (ds, c, to) in T[t].get(r, [])]
    if not reps: continue
    n = len(reps)
    n_cat = sum(1 for _, _, to in reps if to)
    done = [(ds, c) for ds, c, to in reps if not to]
    # 诚实DS: 灾难按0计入分母
    ds_list_honest = [ds for ds, c in done] + [0.0] * n_cat
    ds_honest = sum(ds_list_honest) / n
    col = sum(1 for ds, c in done if c) / len(done) if done else 0.0
    rows.append((t, 100 * n_cat / n, ds_honest, ci95(ds_list_honest), col, n))

with open(OUT + "/percinj_curve_full.csv", "w") as f:
    f.write("tau_perc_ms,catastrophe_rate_pct,honest_meanDS,ci95,completed_collision_rate,n_run\n")
    for r in rows:
        f.write("%d,%.1f,%.1f,%.1f,%.3f,%d\n" % r)

taus = [r[0] for r in rows]; cats = [r[1] for r in rows]; dss = [r[2] for r in rows]; cis = [r[3] for r in rows]
fig, ax1 = plt.subplots(figsize=(9, 5.5))
ax1.set_xlabel(r"perception latency $\tau_{perc}$ (ms)")
ax1.set_ylabel("Honest Driving Score (timeout=0, mean +/- 95%CI)", color="tab:blue")
ax1.errorbar(taus, dss, yerr=cis, fmt="s-", color="tab:blue", lw=2, ms=7, capsize=4, label="Honest DS")
ax1.set_ylim(0, 102); ax1.tick_params(axis="y", labelcolor="tab:blue")
for x, y in zip(taus, dss): ax1.annotate("%.0f" % y, (x, y), textcoords="offset points", xytext=(4, 6), color="tab:blue", fontsize=8)
ax2 = ax1.twinx()
ax2.set_ylabel("Catastrophe (timeout) rate %", color="tab:red")
ax2.plot(taus, cats, "o--", color="tab:red", lw=1.5, ms=7, label="Catastrophe rate")
ax2.set_ylim(0, 45); ax2.tick_params(axis="y", labelcolor="tab:red")
for x, y in zip(taus, cats): ax2.annotate("%.0f%%" % y, (x, y), textcoords="offset points", xytext=(4, -12), color="tab:red", fontsize=8)
plt.title("Perception-latency -> V2X driving degradation (HONEST: timeout=DS0)\n(H800 _1 full-traffic, %d tp0-clean routes x N=3)" % len(common))
fig.tight_layout(); plt.savefig(OUT + "/percinj_curve_full.png", dpi=140); plt.close()
print("matched %d routes; 14点诚实曲线 -> percinj_curve_full.{png,csv}" % len(common))
print("tau | 灾难率% | 诚实DS+/-CI | 完成路碰撞率 | n")
for t, cat, ds, ci, col, n in rows:
    print("  %4d | %5.1f | %5.1f +/-%4.1f | %.2f | %d" % (t, cat, ds, ci, col, n))
