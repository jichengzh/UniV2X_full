import json, glob, re, math, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BASE="/data/jichengzhi_v2x/V2Xverse/results"
OUT="/data/jichengzhi_v2x/percinj_out"; os.makedirs(OUT,exist_ok=True)
TAUS=[0,100,200,300,400,500,600,800]
def load(prefix,cfg):
    out={}
    for d in glob.glob("%s/results_driving_%s_%s_r*_n*"%(BASE,prefix,cfg)):
        m=re.search(r"_r(\d+)_n(\d+)$",d)
        if not m: continue
        r=int(m.group(1)); fs=glob.glob(d+"/**/ego_vehicle_0/results.json",recursive=True)
        if not fs: continue
        try:
            rec=json.load(open(fs[0]))["_checkpoint"]["global_record"]; st=rec.get("status"); sc=rec.get("scores",{})
            if not st: continue
            if st=="TIMEOUT_SKIP" or "score_composed" not in sc: out.setdefault(r,[]).append((None,True))
            else: out.setdefault(r,[]).append((sc["score_composed"],False))
        except: pass
    return out
def ci95(xs):
    n=len(xs)
    if n<2: return 0.0
    m=sum(xs)/n; sd=math.sqrt(sum((x-m)**2 for x in xs)/(n-1)); return 1.96*sd/math.sqrt(n)
def honest(tab,common):
    reps=[(ds,to) for r in common for (ds,to) in tab.get(r,[])]
    if not reps: return float("nan"),float("nan"),0.0
    n=len(reps); nc=sum(1 for _,to in reps if to)
    dsh=[ds for ds,to in reps if not to]+[0.0]*nc
    return sum(dsh)/n, ci95(dsh), 100.0*nc/n
def clean(t0): return set(r for r,reps in t0.items() if sum(1 for ds,to in reps if (not to) and ds is not None and ds>=99.9)>=2)
F={t:load("percinjfast","tpfast%d"%t) for t in TAUS}
L={t:load("percinj","tp%d"%t) for t in TAUS}
isect=set(clean(F[0]))&set(clean(L[0]))
for t in TAUS: isect&=set(F[t])&set(L[t])
isect=sorted(isect)
print("=== INTERSECTION-matched = %d 路 ==="%len(isect))
fd=[];fc=[];lc=[];ld=[];fcat=[];lcat=[]
print(" tau | FAST DS(cat%%) | LOW DS(cat%%) | gap(L-F)")
for t in TAUS:
    a,aci,acat=honest(F[t],isect); b,bci,bcat=honest(L[t],isect)
    fd.append(a);fc.append(aci);ld.append(b);lc.append(bci);fcat.append(acat);lcat.append(bcat)
    print(" %4d | %5.1f (%4.1f) | %5.1f (%4.1f) | %+.1f"%(t,a,acat,b,bcat,b-a))
print("ΔDS(0->800): FAST=%.1f  LOW=%.1f"%(fd[-1]-fd[0],ld[-1]-ld[0]))
print("ΔDS(0->600): FAST=%.1f  LOW=%.1f"%(fd[TAUS.index(600)]-fd[0],ld[TAUS.index(600)]-ld[0]))
fig,ax=plt.subplots(figsize=(10,6))
ax.errorbar(TAUS,fd,yerr=fc,fmt="o-",color="tab:blue",lw=2.2,ms=7,capsize=3,label=r"high-speed ego (max_speed=8 m/s, Eff~150%)")
ax.errorbar(TAUS,ld,yerr=lc,fmt="s--",color="tab:red",lw=2,ms=6,capsize=3,label=r"low-speed ego (max_speed=5 m/s, Eff~90%)")
ax.set_xlabel(r"ego perception(inference) latency  $\tau_{perc}$ (ms)",fontsize=11)
ax.set_ylabel("Honest Driving Score (timeout=0)",fontsize=11)
ax.set_ylim(30,103); ax.grid(alpha=0.3); ax.set_xticks(TAUS)
ax.set_title("Latency harm AMPLIFIES with ego speed: same perception delay, two speed regimes\n"
             "(H800 _1 full-traffic, intersection-matched %d routes x N=3, honest DS)"%len(isect),fontsize=11)
ax.legend(fontsize=10,loc="lower left")
fig.tight_layout(); fig.savefig(OUT+"/percinj_fastslow_compare.png",dpi=145); plt.close()
with open(OUT+"/percinj_fastslow_compare.csv","w") as f:
    f.write("tau_perc,fast_DS,fast_ci95,fast_cat,low_DS,low_ci95,low_cat,gap_low_minus_fast\n")
    for k,t in enumerate(TAUS):
        f.write("%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n"%(t,fd[k],fc[k],fcat[k],ld[k],lc[k],lcat[k],ld[k]-fd[k]))
print("-> percinj_fastslow_compare.{png,csv} (8 pts, matched %d)"%len(isect))
