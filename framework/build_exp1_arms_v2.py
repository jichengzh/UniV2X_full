#!/usr/bin/env python3
"""实验一 TVM 4 臂对比(框架一致,轻调优 regime)。

臂定义(§1.6):
  joint (ours)   : Pareto over (w,q) @ tuned latency  = 框架 SMBO 真前沿
  noS (压缩-only): Pareto over (w,q) @ UNTUNED latency (int8 untuned~=fp16 untuned)
  serial c->t    : stage1 按 (AP70, untuned-fp16) proxy 锁宽(无 tuned/int8 反馈)
                   -> stage2 锁定宽上取 best (q@tuned, int8 若可建否则 fp16)
  serial t->c    : base 调度锁定 -> 剪枝宽跑 base 调度(失配 ~= untuned fp16)

数据(全 H800 backbone [2,64,128,256] input_hw=128x256 轻调优真测):
  tuned lat/ap/energy : results/phase1_pyramid_pareto.json (joint front) + 训练池
  untuned fp16 lat    : untuned/raw_<w>_fp16/latency_row.jsonl (default row)
  int8 可建性         : 该宽有 int8_tc 真测点即可建
"""
import json, math, os, glob

AP = json.load(open("results/ap70_model_pyramid.json"))["params"]
def ap70(w, int8=False):
    s0,s1,s2 = w
    a = AP["a"]+AP["b0"]*math.log(s0/64)+AP["b1"]*math.log(s1/128)+AP["b2"]*math.log(s2/256)
    return a + (-0.008 if int8 else 0.0)

# ★gold AP70(真测)权威源:4 对角锚(stage_a_ap_real)+ block1 minimal-w1 finetune。
# 见 results/AP_SOURCE_ORDER_README.md。key_finding: minimal-w1(w1=32)s0>=24 区 AP FLAT ~0.59-0.60(过参数化),
# 无 floor;s0=16 掉到 0.524(对角锚)。勿用 log 模型/surrogate/过时的 0.53-floor verdict。
GOLD_AP = {  # (w0,w1,w2) fp16-scale gold AP70(int8 再 -0.008)
 (16,32,64):0.5236,  (32,64,128):0.5641, (48,96,192):0.5905, (64,128,256):0.6313,  # 4 对角锚
 (24,32,64):0.5961,  (32,32,96):0.5973,  (64,32,96):0.5927,  (56,64,64):0.5948,     # block1 gold 真测
}
FLAT_MINW1_AP = 0.595  # minimal-w1(w1==32)且 s0>=24 的 flat AP(block1 range 0.593-0.597)
def gold_ap(w, int8=False):
    s0,s1,s2 = w
    if w in GOLD_AP: a = GOLD_AP[w]
    elif s1 == 32 and s0 >= 24: a = FLAT_MINW1_AP          # flat 区(block1 实证)
    else: a = ap70(w)                                       # 兜底 log 模型(仅无 gold 时)
    return round(a + (-0.008 if int8 else 0.0), 4)

# ---- 1) tuned 池:(w,prec)->(lat, ap, energy) --------------------------------
def load_tuned():
    tab = {}
    d = json.load(open("results/phase1_pyramid_pareto.json"))
    for p in d.get("joint_pareto_front", []):
        if not isinstance(p, dict): continue
        w = tuple(p.get("width") or p.get("num_filters") or [])
        pr = "int8" if "int8" in str(p.get("precision")) else ("fp16" if "16" in str(p.get("precision")) else p.get("precision"))
        lat = p.get("lat_ms"); e = p.get("energy_j"); ap = p.get("ap70")
        if w and lat: tab[(w,pr)] = (lat, ap if ap else ap70(w,pr=="int8"), e)
    # 训练池补 tuned
    try:
        t = json.load(open("multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/cost_model/train/original60_training_table_latest.json"))
        for r in t["rows"]:
            w = tuple(r["width"]); pr = "int8" if "int8" in r["precision"] else ("fp16" if "16" in r["precision"] else r["precision"])
            if pr not in ("fp16","int8"): continue
            if (w,pr) not in tab and r.get("latency_ms"):
                tab[(w,pr)] = (r["latency_ms"], r.get("ap70") or ap70(w,pr=="int8"), r.get("energy_j"))
    except Exception as ex: print("[warn] pool:",ex)
    return tab

# ---- 2) untuned fp16 lat per width ------------------------------------------
def load_untuned():
    u = {}
    for j in glob.glob("/exdata/.../untuned/raw_*_fp16/latency_row.jsonl"):
        pass  # placeholder; real read below via local mirror
    return u

def read_untuned_local(mirror_dir):
    u = {}
    for j in glob.glob(os.path.join(mirror_dir, "raw_*_fp16", "latency_row.jsonl")):
        tag = j.split("raw_")[1].split("_fp16")[0]
        w = tuple(int(x) for x in tag.split("x"))
        try:
            rows = [json.loads(l) for l in open(j)]
            dv = [r.get("latency_p50_us") for r in rows if r.get("schedule_policy")=="default" and r.get("latency_p50_us")]
            if dv: u[w] = dv[0]/1000.0
        except Exception: pass
    return u

def apval(w, int8):
    return gold_ap(w, int8)

def pareto(points):  # points: list of (lat, ap, energy, tag); minimize lat, maximize ap
    out = []
    for p in points:
        if not any((q[0]<=p[0] and q[1]>=p[1] and (q[0]<p[0] or q[1]>p[1])) for q in points if q is not p):
            out.append(p)
    return sorted(out)

def min_lat_at(front, floor):
    c = [p for p in front if p[1] >= floor]
    return min(c) if c else None

if __name__ == "__main__":
    import sys
    tuned = load_tuned()
    untuned = {tuple(int(x) for x in k.split("x")):v for k,v in json.load(open("results/exp1_untuned.json")).items()}
    widths = sorted({w for (w,_) in tuned} | set(untuned))
    int8_buildable = {w for (w,pr) in tuned if pr=="int8"}
    print(f"tuned (w,q) 点: {len(tuned)} | 有 untuned 的宽: {len(untuned)} | int8-可建宽: {len(int8_buildable)}")

    # joint (ours)
    joint = pareto([(lat, apval(w,pr=='int8'), e, (w,pr)) for (w,pr),(lat,ap,e) in tuned.items()])
    # noS: untuned latency (int8 untuned ~= fp16 untuned)
    nos_pts = []
    for w in widths:
        if w in untuned:
            for q in ("fp16","int8"):
                nos_pts.append((untuned[w], apval(w,q=='int8'), None, (w,q,"untuned")))
    nos = pareto(nos_pts)
    # params proxy(可选): results/exp1_params.json {"w0xw1xw2": params_M}
    PARAMS = {}
    if os.path.exists("results/exp1_params.json"):
        PARAMS = {tuple(int(x) for x in k.split("x")):v for k,v in json.load(open("results/exp1_params.json")).items()}
    # serial c->t: stage1 proxy-lock, stage2 tuned best-q。两种 proxy:
    #   (a) latency: (AP, untuned-fp16) Pareto   (b) params: (AP, model_size) Pareto = "最大剪枝率"
    def stage1_lock(proxy):
        if proxy=="latency":
            pts=[(untuned[w], apval(w,False), None, w) for w in widths if w in untuned]
        else:  # params
            pts=[(PARAMS[w], apval(w,False), None, w) for w in widths if w in PARAMS]
        return [p[3] for p in pareto(pts)]
    def serial_front(locked):
        f=[]
        for w in locked:
            q = "int8" if w in int8_buildable else "fp16"
            if (w,q) in tuned:
                lat,ap,e = tuned[(w,q)]; f.append((lat, apval(w,q=='int8'), e, (w,q,"locked")))
        return pareto(f)
    locked = stage1_lock("latency")
    serial_ct = serial_front(locked)
    locked_p = stage1_lock("params") if PARAMS else []
    serial_ct_p = serial_front(locked_p) if PARAMS else []

    print("\n=== joint (ours) front ===")
    for lat,ap,e,tag in joint: print(f"  {str(list(tag[0])):15s} {tag[1]:5s} lat={lat:.3f} AP70={ap:.4f}")
    for name, lk, sf in [("latency-proxy", locked, serial_ct), ("params-proxy(最大剪枝率)", locked_p, serial_ct_p)]:
        if not lk: continue
        print(f"\n=== serial c->t [{name}]: stage1 锁定宽 ===")
        print("  locked widths:", [list(w) for w in lk])
        print("  ★[24,32,96] 被锁?", (24,32,96) in lk, "| [24,32,*] 系:", [list(w) for w in lk if w[0]==24 and w[1]==32])
        print("  serial c->t front:")
        for lat,ap,e,tag in sf: print(f"    {str(list(tag[0])):15s} {tag[1]:5s} lat={lat:.3f} AP70={ap:.4f}")

    def fmt(p): return f"{p[0]:.3f}({list(p[3][0])})" if p else "—"
    print("\n=== iso-AP min-lat 对比(AP70 用模型值,口径一致)===")
    print(f"  {'AP70>=':8s} {'ours(joint)':22s} {'serial-lat':22s} {'serial-params':22s} {'noS':12s}")
    for floor in (0.52, 0.53, 0.54, 0.55):
        j=min_lat_at(joint,floor); sl=min_lat_at(serial_ct,floor); sp=min_lat_at(serial_ct_p,floor); n=min_lat_at(nos,floor)
        print(f"  {floor:<8.2f} {fmt(j):22s} {fmt(sl):22s} {fmt(sp):22s} {fmt(n):12s}")
        if j and sp: print(f"           -> ours vs serial-params(最大剪枝率): {sp[0]/j[0]:.2f}x @ iso-AP{floor}")
    json.dump({"joint":[(p[0],p[1],list(p[3][0]),p[3][1]) for p in joint],
               "serial_ct":[(p[0],p[1],list(p[3][0])) for p in serial_ct],
               "locked":[list(w) for w in locked],
               "nos":[(p[0],p[1],list(p[3][0])) for p in nos]},
              open("results/exp1_arms_v2.json","w"), indent=1)
    print("\n[saved] results/exp1_arms_v2.json")
