#!/usr/bin/env python3
"""
tau_curve_aggregate.py — 汇总 τ_ego 扫描结果 + 出曲线

读 results/results_driving_{PREFIX}_te{tau}_r{route}_n{rep}_notraffic/.../results.json
输出:
  - {OUT}/tau_curve_runs.csv      逐 run 原始
  - {OUT}/tau_curve_agg.csv       逐 tau 聚合 (碰撞率 / DS mean±95%CI / RC)
  - {OUT}/tau_curve_main.png      主图: 碰撞率↑ / DS↓ / 效率(RC)↓
  - {OUT}/tau_curve_routes.png    每 route 小多图

口径: 平台 H800 / 配置 _1notraffic / DS=score_composed / 失败=collisions_pedestrian>0
用法: PREFIX=taucurve TAUS=0,100,200,300,400,500 python3 tau_curve_aggregate.py
"""
import os, json, glob, csv, math, sys

VXDIR  = "/data/jichengzhi_v2x/V2Xverse"
OUT    = os.environ.get("OUT", "/data/jichengzhi_v2x/taucurve_out")
PREFIX = os.environ.get("PREFIX", "taucurve")
TAUS   = [int(x) for x in os.environ.get("TAUS", "0,100,200,300,400,500").split(",")]
os.makedirs(OUT, exist_ok=True)

def parse_run(rj):
    try:
        d = json.load(open(rj))
        rec = d['_checkpoint']['global_record']
        sc = rec.get('scores', {})
        inf = rec.get('infractions', {})
        return dict(
            status=rec.get('status', ''),
            ds=sc.get('score_composed'),
            rc=sc.get('score_route'),
            pen=sc.get('score_penalty'),
            colped=inf.get('collisions_pedestrian', 0.0),
            colveh=inf.get('collisions_vehicle', 0.0),
            collay=inf.get('collisions_layout', 0.0),
            vblock=inf.get('vehicle_blocked', 0.0),
        )
    except Exception as e:
        return None

# ── gather ────────────────────────────────────────────────────────────────────
rows = []
import re
pat = re.compile(rf"results_driving_{PREFIX}_te(\d+)_r(\d+)_n(\d+)_notraffic")
for d in sorted(glob.glob(f"{VXDIR}/results/results_driving_{PREFIX}_te*_r*_n*_notraffic")):
    m = pat.search(d)
    if not m: continue
    tau, route, rep = int(m.group(1)), int(m.group(2)), int(m.group(3))
    rjs = glob.glob(f"{d}/v2x_final/town05_short_collab/r{route}_repeat0/ego_vehicle_0/results.json")
    if not rjs: continue
    r = parse_run(rjs[0])
    if r is None or r['ds'] is None: continue
    r.update(tau=tau, route=route, rep=rep)
    rows.append(r)

if not rows:
    print("NO RESULTS FOUND for prefix", PREFIX); sys.exit(1)

with open(f"{OUT}/tau_curve_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["tau","route","rep","status","ds","rc","pen","colped","colveh","collay","vblock"])
    w.writeheader()
    for r in sorted(rows, key=lambda x:(x['tau'],x['route'],x['rep'])):
        w.writerow(r)
print(f"wrote {len(rows)} runs -> {OUT}/tau_curve_runs.csv")

# ── aggregate per tau ─────────────────────────────────────────────────────────
def mean(xs): return sum(xs)/len(xs) if xs else float('nan')
def ci95(xs):
    n=len(xs)
    if n<2: return float('nan')
    m=mean(xs); sd=math.sqrt(sum((x-m)**2 for x in xs)/(n-1))
    return 1.96*sd/math.sqrt(n)

agg=[]
for tau in TAUS:
    sub=[r for r in rows if r['tau']==tau]
    if not sub: continue
    ds=[r['ds'] for r in sub]
    rc=[r['rc'] for r in sub]
    ncol=sum(1 for r in sub if r['colped']>0)
    nvcol=sum(1 for r in sub if r['colveh']>0)
    agg.append(dict(tau=tau, n=len(sub), nroute=len(set(r['route'] for r in sub)),
                    col_rate=ncol/len(sub), colveh_rate=nvcol/len(sub),
                    ds_mean=mean(ds), ds_ci=ci95(ds),
                    rc_mean=mean(rc), rc_ci=ci95(rc)))
with open(f"{OUT}/tau_curve_agg.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["tau","n","nroute","col_rate","colveh_rate","ds_mean","ds_ci","rc_mean","rc_ci"])
    w.writeheader()
    for a in agg: w.writerow({k:(round(v,4) if isinstance(v,float) else v) for k,v in a.items()})
print(f"wrote agg -> {OUT}/tau_curve_agg.csv")
for a in agg:
    print(f"  te{a['tau']:>3}: n={a['n']} routes={a['nroute']} "
          f"col_rate={a['col_rate']:.2f} DS={a['ds_mean']:.1f}±{a['ds_ci']:.1f} RC={a['rc_mean']:.1f}")

# ── plot ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs=[a['tau'] for a in agg]
    # main 3-line
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.set_xlabel(r"$\tau_{ego}$ (ms)")
    ax1.set_ylabel("Collision rate (ped) / RouteCompletion", color='tab:red')
    l1,=ax1.plot(xs,[a['col_rate'] for a in agg],'o-',color='tab:red',label='ped-collision rate')
    l3,=ax1.plot(xs,[a['rc_mean']/100 for a in agg],'^--',color='tab:green',label='RouteCompletion (eff proxy, /100)')
    ax1.set_ylim(-0.05,1.05); ax1.tick_params(axis='y',labelcolor='tab:red')
    ax2=ax1.twinx()
    ax2.set_ylabel("Driving Score (mean±95%CI)", color='tab:blue')
    ds=[a['ds_mean'] for a in agg]; dsci=[a['ds_ci'] if not math.isnan(a['ds_ci']) else 0 for a in agg]
    l2=ax2.errorbar(xs,ds,yerr=dsci,fmt='s-',color='tab:blue',capsize=4,label='Driving Score')
    ax2.set_ylim(0,105); ax2.tick_params(axis='y',labelcolor='tab:blue')
    lines=[l1,l2,l3]; labels=[l.get_label() for l in lines]
    ax1.legend(lines,labels,loc='center left',fontsize=8)
    plt.title(f"τ_ego → V2X driving degradation (H800/_1notraffic, {agg[0]['nroute']}routes×N)")
    plt.tight_layout(); plt.savefig(f"{OUT}/tau_curve_main.png",dpi=130); plt.close()
    print(f"wrote {OUT}/tau_curve_main.png")

    # per-route small multiples (DS vs tau)
    routes=sorted(set(r['route'] for r in rows))
    ncol=4; nrow=math.ceil(len(routes)/ncol)
    fig,axes=plt.subplots(nrow,ncol,figsize=(3*ncol,2.2*nrow),squeeze=False)
    for i,rt in enumerate(routes):
        ax=axes[i//ncol][i%ncol]
        for tau in TAUS:
            pts=[r['ds'] for r in rows if r['route']==rt and r['tau']==tau]
            if pts: ax.scatter([tau]*len(pts),pts,s=12,color='tab:blue')
        mxs=[tau for tau in TAUS if any(r['route']==rt and r['tau']==tau for r in rows)]
        mys=[mean([r['ds'] for r in rows if r['route']==rt and r['tau']==tau]) for tau in mxs]
        ax.plot(mxs,mys,'-',color='tab:red',lw=1)
        ax.set_title(f"r{rt}",fontsize=8); ax.set_ylim(0,105)
        ax.tick_params(labelsize=6)
    for j in range(len(routes),nrow*ncol): axes[j//ncol][j%ncol].axis('off')
    fig.suptitle("Per-route DS vs τ_ego (route-dependent thresholds)",fontsize=11)
    plt.tight_layout(); plt.savefig(f"{OUT}/tau_curve_routes.png",dpi=120); plt.close()
    print(f"wrote {OUT}/tau_curve_routes.png")
except Exception as e:
    print("plot skipped:", e)
