#!/usr/bin/env python3
"""Build §1.3 Table 1 (experiment-1) from REAL measured backbone-scope data.

Two backends (TVM tuned + TRT), 6 arms each. All latency/energy numbers are REAL
H800 backbone-subnet measurements (input [2,64,128,256], input_hw=128x256):
  - TVM tuned : original60 SMBO corpus (metaschedule-tuned, int8_tc=WMMA).
  - TVM untuned (default schedule): framework measurement-job default-schedule row
    (fresh empty work-dir => no tuned DB => default lowering timed).
  - TRT       : framework/trt_baseline/trt_profile_v1.py (FP16 / INT8-PTQ, DAIR calib),
                CUDA-event p50 warmup20/iters300/repeat5, NVML energy.
AP70 = calibrated log model (results/ap70_model_pyramid.json), anchored on gold
stage_a_ap_real.parquet (DAIR val 1789). int8 AP delta = -0.008 (historical median).
model_size = real ONNX initializer param count.

Arms (from 13_7_14 §1.1):
  1 original        : base, default precision (fp16), DEFAULT schedule (untuned)
  2 compression-only: prune+quant selected by size proxy, DEFAULT schedule (no S feedback)
  3 schedule-only   : base fp16, TUNED (S only, no compression)
  4 serial c->t     : compression locked by size proxy first, THEN tuned
  5 serial t->c     : tune base first, then compress (pruned width on base's schedule => mismatch)
  6 ours (joint)    : joint P x Q x S with REAL backend latency feedback (min real latency at AP floor)

TRT has no untuned mode (always tactic-optimized) => on TRT arm1==arm3 and arm2's
"default schedule" == optimized; this asymmetry is itself a finding (closed vs open backend).
"""
import json, math, itertools, os

AP = json.load(open("results/ap70_model_pyramid.json"))["params"]
def ap70(w):
    s0, s1, s2 = w
    return AP["a"] + AP["b0"]*math.log(s0/64) + AP["b1"]*math.log(s1/128) + AP["b2"]*math.log(s2/256)
INT8_AP_DELTA = -0.008

# width -> real params (M), from ONNX initializers
PARAMS_M = {(64,128,256):3.130, (48,96,192):1.765, (32,64,128):0.787, (16,32,64):0.198}
WIDTHS = [(64,128,256),(48,96,192),(32,64,128),(16,32,64)]
LABEL = {(64,128,256):"base",(48,96,192):"trap25",(32,64,128):"p50",(16,32,64):"p75"}

# ---- REAL measured latency(ms) + energy(J) ---------------------------------
# TVM tuned (metaschedule): fp16 / int8_tc
TVM_TUNED = {
 (64,128,256):{"fp16":(6.523,1.924), "int8":(6.104,1.771)},
 (48,96,192): {"fp16":(5.329,1.571), "int8":(5.050,1.430)},
 (32,64,128): {"fp16":(4.400,1.246), "int8":(3.589,1.020)},
 (16,32,64):  {"fp16":(1.625,0.483), "int8":(1.343,0.383)},
}
# TVM untuned (default schedule), fp16, REAL (default-schedule row, fresh empty work-dir)
TVM_UNTUNED_FP16 = {  # ms
 (64,128,256):56.235, (48,96,192):42.285, (32,64,128):26.186, (16,32,64):12.659,
}
# trap-pair REAL TVM tuned (W_g vs P_g): the ablation LUT claimed iso-AP 3.827x gap;
# real measurement REFUTES it -> W_g[48,128,128] is FASTER than P_g[64,128,128] at both
# precisions => NO width-trap on real tuned TVM (latency tracks size monotonically).
TVM_TRAPPAIR = {  # (lat_fp16, lat_int8_tc), ms, tuned
 (64,128,128):(6.031,5.635),  # P_g  AP70=0.6163
 (48,128,128):(5.837,5.016),  # W_g  AP70=0.6063
}
# TRT (always optimized): fp16 / int8-PTQ
TRT = {
 (64,128,256):{"fp16":(0.870,0.2865), "int8":(0.851,0.2284)},
 (48,96,192): {"fp16":(3.629,0.9429), "int8":(3.585,0.9336)},
 (32,64,128): {"fp16":(0.872,0.2483), "int8":(0.864,0.2494)},
 (16,32,64):  {"fp16":(0.585,0.1333), "int8":(0.588,0.1369)},
}

def ap_of(w, prec):
    return ap70(w) + (INT8_AP_DELTA if prec=="int8" else 0.0)

def all_configs(lat_tab):
    """(w, prec) -> (lat, energy, ap70) over the measured space for one backend/schedule."""
    out = {}
    for w in WIDTHS:
        for prec in ("fp16","int8"):
            if w in lat_tab and prec in lat_tab[w] and lat_tab[w][prec] is not None:
                lat,e = lat_tab[w][prec]
                out[(w,prec)] = (lat, e, ap_of(w,prec))
    return out

def dominated(p, front):
    """p=(lat,ap,energy) dominated (>=) by any q in front (min lat, min energy, max ap)."""
    for q in front:
        if q is p: continue
        if q[0] <= p[0] and q[2] <= p[2] and q[1] >= p[1] and (q[0]<p[0] or q[2]<p[2] or q[1]>p[1]):
            return True
    return False

def pareto_front(cfgs):
    pts = [(lat,ap,e,k) for k,(lat,e,ap) in cfgs.items()]
    front = []
    for p in pts:
        if not dominated((p[0],p[1],p[2]), [(q[0],q[1],q[2]) for q in pts]):
            front.append(p)
    return sorted(front)

def min_lat_at_floor(cfgs, floor):
    """min-latency config with ap70 >= floor."""
    cand = [(lat,ap,e,k) for k,(lat,e,ap) in cfgs.items() if ap >= floor]
    return min(cand) if cand else None

def proxy_pick(floor):
    """compression proxy: min real params among widths meeting AP floor (fp16 unless quant chosen).
    Returns width. Quant is 'allowed' by the compression axis => picks int8 (smaller footprint)."""
    cand = [(PARAMS_M[w], w) for w in WIDTHS if ap_of(w,"int8") >= floor]
    return min(cand)[1] if cand else None

AP_BASE = ap70((64,128,256))
BUDGETS = [0.0, 0.01, 0.02, 0.05]  # absolute AP70 drop

def build_backend(name, tuned_tab, untuned_fp16, is_trt):
    tuned = all_configs(tuned_tab)
    # untuned config table (schedule=default). TRT: no untuned => reuse optimized.
    if is_trt:
        untuned = tuned
    else:
        untuned = {}
        for w in WIDTHS:
            if untuned_fp16.get(w) is not None:
                untuned[(w,"fp16")] = (untuned_fp16[w], None, ap_of(w,"fp16"))
    rows = {}
    floor5 = AP_BASE - 0.05  # headline budget: <=5% absolute AP70 drop
    # arm3: base fp16 tuned
    l,e,a = tuned[((64,128,256),"fp16")]; rows[3]=("base","fp16","tuned",l,a,e)
    # arm1: base fp16 untuned
    if ((64,128,256),"fp16") in untuned:
        l,e,a = untuned[((64,128,256),"fp16")]; rows[1]=("base","fp16","default",l,a,e)
    else:
        rows[1]=None
    # arm6: ours = min real latency at floor (joint over width x prec, tuned schedule)
    m = min_lat_at_floor(tuned, floor5)
    if m: rows[6]=(LABEL[m[3][0]],m[3][1],"tuned",m[0],m[1],m[2])
    # arm4: serial compress->tune = proxy-pick width, tuned, fp16 (compression axis fixes width; precision from proxy=int8)
    wp = proxy_pick(floor5)
    if wp:
        l,e,a = tuned[(wp,"int8")]; rows[4]=(LABEL[wp],"int8","tuned(locked-first)",l,a,e)
    # arm5: serial tune->compress = tune base, then compress => pruned width on base's schedule (mismatch ~ untuned pruned)
    if wp and ((wp,"fp16") in untuned):
        l,e,a = untuned[(wp,"fp16")]; rows[5]=(LABEL[wp],"fp16","base-sched(mismatch)",l,a,e)
    else:
        rows[5]=None
    # arm2: compression-only = proxy-pick width + quant, DEFAULT schedule (untuned).
    # int8 on an UNTUNED schedule ~= fp16 untuned (WMMA/dp4a tensorization is inactive
    # without tuning), so we price it with the measured untuned-fp16 latency of the
    # proxy width (honest, clearly labeled).
    if wp and not is_trt and ((wp,"fp16") in untuned):
        l,e,a = untuned[(wp,"fp16")]; a = ap_of(wp,"int8")
        rows[2]=(LABEL[wp],"int8","default(untuned)",l,a,e)
    elif is_trt and wp:
        l,e,a = tuned[(wp,"int8")]; rows[2]=(LABEL[wp],"int8","optimized(no-untuned-on-TRT)",l,a,e)
    # judging: ours front + iso-AP speedups
    front = pareto_front(tuned)
    return {"rows":rows, "front":[(round(p[0],3),round(p[1],4),LABEL[p[3][0]],p[3][1]) for p in front],
            "floor5":round(floor5,4)}

def iso_ap_speedup(cfgs, floor, ref_lat):
    """min-latency at floor vs a reference arm latency -> speedup (ref/ours)."""
    m = min_lat_at_floor(cfgs, floor)
    if not m or not ref_lat: return None
    return round(ref_lat / m[0], 2), (LABEL[m[3][0]], m[3][1], round(m[0],3), round(m[1],4))

def hv_2d(front, lat_ref, ap_ref):
    """simple 2D hypervolume (min lat, max ap) vs a nadir ref; area dominated."""
    pts = sorted([(p[0],p[1]) for p in front])  # (lat, ap)
    # keep upper-left staircase
    hv=0.0; prev_lat=0.0; best_ap=0.0
    for lat,ap in pts:
        if ap>best_ap:
            hv += (lat-prev_lat)*(ap_ref-best_ap) if False else 0
    # standard: integrate (ap - ap0) over lat axis up to lat_ref, taking best ap reachable at <= lat
    xs=sorted(set([p[0] for p in front]+[lat_ref]))
    area=0.0
    for i in range(len(xs)-1):
        l0,l1=xs[i],xs[i+1]
        if l0>=lat_ref: break
        best=max([p[1] for p in front if p[0]<=l0]+[0])
        area+=(min(l1,lat_ref)-l0)*max(best-ap_ref,0)
    return round(area,4)

if __name__=="__main__":
    import sys
    for w in WIDTHS:
        print(f"AP70[{LABEL[w]:7s}{list(w)}] = {ap70(w):.4f}  params={PARAMS_M[w]}M")
    print(f"AP_BASE={AP_BASE:.4f}")
    RESULT={}
    for name,tab,is_trt in [("TVM",TVM_TUNED,False),("TRT",TRT,True)]:
        r = build_backend(name, tab, TVM_UNTUNED_FP16, is_trt)
        tuned=all_configs(tab); front=pareto_front(tuned)
        print(f"\n===== {name} =====  ours-front(lat,ap,label,prec)={r['front']}")
        for i in range(1,7):
            v=r["rows"].get(i)
            if v: print(f"  arm{i}: {v[0]:8s} {v[1]:5s} {v[2]:26s} lat={v[3]:.3f}ms AP70={v[4]:.4f} E={v[5]}")
            else: print(f"  arm{i}: (pending untuned)")
        # judging: iso-AP speedup ours vs serial(arm4) at budgets; dominance of arms by ours-front
        print("  --- iso-AP min-latency at budgets (abs AP70 drop) ---")
        for b in BUDGETS:
            floor=AP_BASE-b
            m=min_lat_at_floor(tuned,floor)
            print(f"    drop<= {b:.2f} (floor {floor:.3f}): ours={LABEL[m[3][0]] if m else None} {m[3][1] if m else ''} lat={m[0] if m else None}")
        # dominance: is each arm's rep point (lat,ap,energy) dominated by the ours-front?
        ours = r["rows"].get(6)
        ndom=0; doms=[]
        for i in range(1,6):
            v=r["rows"].get(i)
            if not v: continue
            lat,ap,e = v[3],v[4],v[5]
            # dominated if ours has a point with <=lat, >=ap (energy ignored if None)
            dom = any(p[0]<=lat and p[1]>=ap and (p[0]<lat or p[1]>ap) for p in front)
            # also compare to ours rep point directly
            if ours and ours[3] <= lat and ours[4] >= ap-1e-9 and (ours[3]<lat or ours[4]>ap):
                dom = True
            if dom: ndom+=1; doms.append(i)
        print(f"  --- JUDGING --- ours-rep = {ours[0]} {ours[1]} lat={ours[3]:.3f} AP={ours[4]:.4f}")
        print(f"    arms dominated by ours: {doms}  ({ndom}/5)")
        if ours:
            for i in [2,4,5]:
                v=r["rows"].get(i)
                if v and v[3]: print(f"    ours vs arm{i}: {v[3]/ours[3]:.2f}x latency (arm{i}={v[3]:.3f} @AP{v[4]:.4f})")
        RESULT[name]=r
    print("\n=== TVM trap-pair (real, refutes LUT 3.827x) ===")
    for w,(f16,i8) in TVM_TRAPPAIR.items():
        print(f"  {list(w)} AP70={ap70(w):.4f}  fp16={f16} int8_tc={i8}")
    json.dump({k:{"rows":{i:v for i,v in R["rows"].items()},"front":R["front"]} for k,R in RESULT.items()},
              open("results/exp1_table_arms_v1.json","w"), indent=1, default=str)
    print("\n[saved] results/exp1_table_arms_v1.json")
