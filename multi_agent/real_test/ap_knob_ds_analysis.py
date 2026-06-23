"""AP-knob 闭环有效性分析: 4 臂 honest DS / RC / 碰撞, 同 ap0-clean 交集路集。
   arm: ap0(drop0) ap50(drop.5 box+feat) ap55(drop.5 box-only) ap100(drop1 blind)
   knob->AP 标定(离线200帧): drop0.5 -> veh_ap50 0.559 / ap70 0.516; drop1.0 -> ~0.27
"""
import json, glob, re, math, os

BASE = "/data/jichengzhi_v2x/V2Xverse/results"
PREFIX = "apval"
ARMS = [("ap0", 0.0, "baseline"), ("ap50", 0.5, "box+feat"),
        ("ap55", 0.5, "box-only"), ("ap100", 1.0, "blind")]
ROUTES = [3, 17, 104, 136, 18]
AP50_OF_DROP = {0.0: 0.841, 0.5: 0.559, 1.0: 0.266}  # 离线标定 veh AP50

def load(tag):
    out = {}
    for r in ROUTES:
        for rep in (1, 2):
            d = "%s/results_driving_%s_%s_r%d_n%d" % (BASE, PREFIX, tag, r, rep)
            fs = glob.glob(d + "/v2x_final/town05_short_collab/*/ego_vehicle_0/results.json")
            if not fs:
                continue
            try:
                rec = json.load(open(fs[0]))["_checkpoint"]["global_record"]
                st = rec.get("status")
                if not st:
                    continue
                sc = rec.get("scores", {})
                inf = rec.get("infractions", {})
                if st == "TIMEOUT_SKIP" or "score_composed" not in sc:
                    out.setdefault(r, []).append({"ds": None, "to": True, "rc": 0.0, "col": None})
                else:
                    ncol = sum(len(inf.get(k, [])) if isinstance(inf.get(k), list) else (inf.get(k) or 0)
                               for k in ("collisions_vehicle", "collisions_pedestrian", "collisions_layout"))
                    out.setdefault(r, []).append({"ds": sc["score_composed"], "to": False,
                                                  "rc": sc.get("score_route", 0.0), "col": ncol})
            except Exception:
                pass
    return out

T = {a[0]: load(a[0]) for a in ARMS}

# ap0-clean: τ=0(drop0) DS>=99.9 在 >=半数 rep
ap0 = T["ap0"]
clean = set(r for r, reps in ap0.items()
            if sum(1 for x in reps if (not x["to"]) and x["ds"] is not None and x["ds"] >= 99.9) >= 1)
common = sorted(clean)
for a in ARMS:
    common = [r for r in common if r in T[a[0]]]

def honest(tag):
    reps = [x for r in common for x in T[tag].get(r, [])]
    if not reps:
        return float("nan"), float("nan"), 0.0, 0
    n = len(reps)
    ncat = sum(1 for x in reps if x["to"])
    dsh = [x["ds"] for x in reps if not x["to"]] + [0.0] * ncat
    rc = sum(x["rc"] for x in reps) / n
    cols = [x["col"] for x in reps if x["col"] is not None]
    meancol = sum(cols) / len(cols) if cols else float("nan")
    return sum(dsh) / n, rc, 100.0 * ncat / n, meancol

print("== AP-knob 闭环有效性 (ap0-clean 交集 %d 路: %s) ==" % (len(common), common))
print("arm        drop  注入        ~vehAP50  honestDS   RC    灾难%%  平均碰撞")
base_ds = None
for tag, drop, note in ARMS:
    ds, rc, cat, col = honest(tag)
    ap = AP50_OF_DROP.get(drop, float("nan"))
    if base_ds is None:
        base_ds = ds
    dd = ds - base_ds
    print("%-9s  %.2f  %-10s  %.3f     %6.1f  %5.1f  %5.1f   %.2f   (ΔDS=%+.1f)"
          % (tag, drop, note, ap, ds, rc, cat, col, dd))

print("\n判读:")
print("  · ap0 高 / ap100 塌 = 注入接线正确, AP→DS 因果通")
print("  · ap50 vs ap55(box-only) 的 DS 差 = 特征旁路强度 (差越大=越需 featmask)")
