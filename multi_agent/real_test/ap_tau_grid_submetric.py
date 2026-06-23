import json,glob
B="/exdata/jichengzhi/V2Xverse_apknob/results"
TAUS=[0,200,400,600,800]; DROPS=[0,25,50,70,85]
ROUTES=[3,17,18,104,136,317]
def code(t,d): return t*100+d
def load(c):
    rc=[]; col=[]; to=0; n=0
    for r in ROUTES:
        for rep in (1,2,3):
            fs=glob.glob("%s/results_driving_grid_g%d_r%d_n%d/v2x_final/town05_short_collab/*/ego_vehicle_0/results.json"%(B,c,r,rep))
            if not fs: continue
            try:
                rec=json.load(open(fs[0]))["_checkpoint"]["global_record"]
                st=rec.get("status")
                if not st: continue
                n+=1
                sc=rec.get("scores",{}); inf=rec.get("infractions",{})
                if st=="TIMEOUT_SKIP" or "score_route" not in sc:
                    to+=1; rc.append(0.0); continue
                rc.append(sc.get("score_route",0.0))
                cv=inf.get("collisions_vehicle",[])
                col.append(len(cv) if isinstance(cv,list) else float(cv or 0))
            except: pass
    import statistics as s
    return (s.mean(rc) if rc else float("nan"),
            s.mean(col) if col else float("nan"),
            100.0*to/n if n else float("nan"), n)
print("=== mean Route Completion (RC%) ===")
print("tau\\drop |"+"|".join("d%.2f"%(d/100) for d in DROPS))
for t in TAUS:
    print("%4d    |"%t+"|".join("%6.1f"%load(code(t,d))[0] for d in DROPS))
print("=== mean vehicle-collision count/episode ===")
print("tau\\drop |"+"|".join("d%.2f"%(d/100) for d in DROPS))
for t in TAUS:
    print("%4d    |"%t+"|".join("%6.2f"%load(code(t,d))[1] for d in DROPS))
