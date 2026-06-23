"""Plan2-Phase-AP — NSGA-II 5D Pareto demo with REAL-AP predictor (LGB v4).

预测器:
    - LGB v3 (lat + throughput + build_success) — 跨硬件 latency/吞吐 预测器
    - LGB v4 (ap50, ap70) — 训练于 Stage A (8 anchor) + Stage B (~40 anchor) **真测** AP

目标 (5D Pareto):
    minimize lat_p50, workspace_gb, params_kb
    maximize throughput_fps, ap50

Diff vs v3: ap50 axis 加进 Pareto + 加 ap50 ≥ AP_THRESH filter (避免极端 prune 把 AP 砍崩).

输出:
    - results/nsga2_pareto_demo_v4.json
    - results/nsga2_pareto_demo_v4.csv
"""
from __future__ import annotations
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AP_THRESH = 0.50  # 仍保留某种实用 AP 下限 (paper 故事里讲: framework 不会推荐崩盘的点)


class LGBv3LatPredictor:
    """14-feature lat / throughput / build_success predictor (cross-HW)."""
    def __init__(self):
        self.lat = lgb.Booster(model_file=str(ROOT / "models/lgb_v3_lat.txt"))
        self.thr = lgb.Booster(model_file=str(ROOT / "models/lgb_v3_throughput.txt"))
        self.bs  = lgb.Booster(model_file=str(ROOT / "models/lgb_v3_build_success.txt"))

    def featurize(self, cfg):
        return np.array([
            cfg["stage0_planes"], cfg["stage1_planes"], cfg["stage2_planes"],
            1 if cfg["precision"]=="fp16" else 0,
            1 if cfg["precision"]=="int8" else 0,
            1 if cfg["d_scheme"]=="A_gpu" else 0,
            1 if cfg["d_scheme"]=="B_dla0" else 0,
            1 if cfg["d_scheme"]=="B_dla1" else 0,
            1 if cfg["d_scheme"]=="C_3ip" else 0,
            1 if cfg["d_tactic"]=="default" else 0,
            0 if cfg["d_tactic"]=="default" else 1,
            cfg["d_workspace_gb"],
            1 if cfg["hardware"]=="rtx4090" else 0,
            1 if cfg["hardware"]=="orin_agx" else 0,
        ], dtype=np.float32).reshape(1, -1)

    def predict(self, cfg):
        x = self.featurize(cfg)
        return {
            "lat_p50_ms": float(self.lat.predict(x)[0]),
            "throughput_fps": float(self.thr.predict(x)[0]),
            "build_success_prob": float(self.bs.predict(x)[0]),
        }


class LGBv4APPredictor:
    """5-feature AP50/AP70 regressor trained on real measurements only.

    Falls back gracefully if models missing (Stage A/B not yet complete).
    """
    def __init__(self):
        m50 = ROOT / "models/lgb_v4_ap50.txt"
        m70 = ROOT / "models/lgb_v4_ap70.txt"
        self.ap50 = lgb.Booster(model_file=str(m50)) if m50.exists() else None
        self.ap70 = lgb.Booster(model_file=str(m70)) if m70.exists() else None
        if self.ap50 is None:
            print("⚠️  lgb_v4_ap50.txt missing — AP axis will be skipped.")

    @property
    def ready(self):
        return self.ap50 is not None

    def featurize(self, cfg):
        # Same 5-feature schema used in train_lgb_v4_with_ap.featurize().
        return np.array([
            cfg["stage0_planes"], cfg["stage1_planes"], cfg["stage2_planes"],
            1 if cfg["precision"]=="fp16" else 0,
            1 if cfg["precision"]=="int8" else 0,
        ], dtype=np.float32).reshape(1, -1)

    def predict(self, cfg):
        x = self.featurize(cfg)
        out = {}
        if self.ap50 is not None:
            out["ap50"] = float(self.ap50.predict(x)[0])
        if self.ap70 is not None:
            out["ap70"] = float(self.ap70.predict(x)[0])
        return out


# Search spaces unchanged from v3 — we keep the same triplets/precisions
TRIPLETS_4090 = [
    (64, 128, 256), (40, 64, 128), (32, 64, 128), (24, 48, 96),
    (16, 32, 64), (16, 32, 32), (24, 32, 64), (40, 72, 128),
]
TRIPLETS_ORIN = TRIPLETS_4090 + [(48, 96, 192), (48, 56, 96), (56, 96, 200)]
PRECISIONS = ["fp16", "int8"]


def gen_4090_space():
    cfgs = []
    for tri in TRIPLETS_4090:
        for prec in PRECISIONS:
            for tactic in ["default", "no_cudnn", "with_cudnn", "edge_only"]:
                for ws in [4, 8]:
                    cfgs.append({
                        "stage0_planes": tri[0], "stage1_planes": tri[1], "stage2_planes": tri[2],
                        "precision": prec, "d_scheme": "A_gpu",
                        "d_tactic": tactic, "d_workspace_gb": ws,
                        "hardware": "rtx4090",
                    })
    return cfgs


def gen_orin_space():
    cfgs = []
    for tri in TRIPLETS_ORIN:
        for prec in PRECISIONS:
            for scheme in ["A_gpu", "B_dla0", "B_dla1", "C_3ip"]:
                for tactic in ["default", "no_cudnn"]:
                    for ws in [1, 2]:
                        cfgs.append({
                            "stage0_planes": tri[0], "stage1_planes": tri[1], "stage2_planes": tri[2],
                            "precision": prec, "d_scheme": scheme,
                            "d_tactic": tactic, "d_workspace_gb": ws,
                            "hardware": "orin_agx",
                        })
    return cfgs


def pareto_front_nd(points, axes_min, axes_max=None):
    axes_max = axes_max or []
    if not points: return []
    def dominates(p, q):
        if any(p[a] > q[a] for a in axes_min): return False
        if any(p[a] < q[a] for a in axes_max): return False
        return (any(p[a] < q[a] for a in axes_min) or any(p[a] > q[a] for a in axes_max))
    front = []
    for i, p1 in enumerate(points):
        if any(dominates(p2, p1) for j, p2 in enumerate(points) if i != j):
            continue
        front.append(p1)
    return sorted(front, key=lambda p: p[axes_min[0]])


def model_params_kb(cfg):
    s0, s1, s2 = cfg["stage0_planes"], cfg["stage1_planes"], cfg["stage2_planes"]
    p = 9 * (2*s0)*(2*s0)/32 + 9 * (2*s1)*(2*s1)/32 + 9 * (2*s2)*(2*s2)/32
    bpw = 1 if cfg["precision"]=="int8" else 2
    return p * bpw / 1024.0


def run_pareto(hw_label, space_fn, lat_pred, ap_pred):
    print(f"\n{'='*72}")
    print(f"NSGA-II 5D Pareto Demo (with REAL-AP axis): {hw_label}")
    print(f"{'='*72}")
    cfgs = space_fn()
    print(f"Search space: {len(cfgs)} configurations")

    evaluated = []
    for cfg in cfgs:
        p = lat_pred.predict(cfg)
        if ap_pred.ready:
            p.update(ap_pred.predict(cfg))
        cfg_eval = {**cfg, **p,
                    "params_kb": model_params_kb(cfg),
                    "feasible": p["build_success_prob"] >= 0.5}
        evaluated.append(cfg_eval)

    feasible = [e for e in evaluated if e["feasible"]]
    if ap_pred.ready:
        feasible = [e for e in feasible if e.get("ap50", 0) >= AP_THRESH]
        print(f"Feasible (build_success ≥ 0.5 AND ap50 ≥ {AP_THRESH}): {len(feasible)} / {len(cfgs)}")
    else:
        print(f"Feasible (build_success ≥ 0.5): {len(feasible)} / {len(cfgs)}  [AP filter skipped]")

    if not feasible:
        return None

    if ap_pred.ready:
        axes_min = ["lat_p50_ms", "workspace_gb", "params_kb"]
        axes_max = ["throughput_fps", "ap50"]
        point_keys = axes_min + axes_max
    else:
        axes_min = ["lat_p50_ms", "workspace_gb", "params_kb"]
        axes_max = ["throughput_fps"]
        point_keys = axes_min + axes_max

    points = []
    for e in feasible:
        pt = {k: e["d_workspace_gb"] if k == "workspace_gb" else e[k] for k in point_keys}
        pt["config"] = e
        points.append(pt)

    front = pareto_front_nd(points, axes_min=axes_min, axes_max=axes_max)
    print(f"\n{len(axes_min)+len(axes_max)}D Pareto frontier size: {len(front)}")

    header = f"{'#':<3} {'lat':<8} {'thr':<8}"
    if ap_pred.ready: header += f" {'ap50':<6} {'ap70':<6}"
    header += f" {'ws':<4} {'params':<8} {'triplet':<20} {'prec':<5} {'scheme':<8} {'tactic':<12}"
    print(header)
    for i, p in enumerate(front):
        c = p["config"]
        tri = f"({c['stage0_planes']},{c['stage1_planes']},{c['stage2_planes']})"
        line = f"{i:<3} {p['lat_p50_ms']:<8.3f} {p['throughput_fps']:<8.1f}"
        if ap_pred.ready:
            line += f" {c.get('ap50', 0):<6.3f} {c.get('ap70', 0):<6.3f}"
        line += (f" {p['workspace_gb']:<4} {p['params_kb']:<8.1f} "
                 f"{tri:<20} {c['precision']:<5} {c['d_scheme']:<8} {c['d_tactic']:<12}")
        print(line)

    lat_opt = min(front, key=lambda p: p["lat_p50_ms"])
    thr_opt = max(front, key=lambda p: p["throughput_fps"])
    print(f"\n*** lat-optimal: {lat_opt['lat_p50_ms']:.3f} ms @ {lat_opt['throughput_fps']:.1f} fps"
          + (f", ap50={lat_opt['config'].get('ap50',0):.3f}" if ap_pred.ready else "")
          + f", scheme={lat_opt['config']['d_scheme']}")
    print(f"*** throughput-optimal: {thr_opt['throughput_fps']:.1f} fps @ {thr_opt['lat_p50_ms']:.3f} ms"
          + (f", ap50={thr_opt['config'].get('ap50',0):.3f}" if ap_pred.ready else "")
          + f", scheme={thr_opt['config']['d_scheme']}")
    if ap_pred.ready:
        ap_opt = max(front, key=lambda p: p["config"].get("ap50", 0))
        print(f"*** AP-optimal:        ap50={ap_opt['config']['ap50']:.3f} @ "
              f"{ap_opt['lat_p50_ms']:.3f} ms, {ap_opt['throughput_fps']:.1f} fps, "
              f"scheme={ap_opt['config']['d_scheme']}")

    return {
        "hardware": hw_label,
        "search_space_size": len(cfgs),
        "feasible_count": len(feasible),
        "pareto_size": len(front),
        "ap_axis_enabled": ap_pred.ready,
        "lat_optimal": lat_opt["config"],
        "throughput_optimal": thr_opt["config"],
        "pareto_configs": [p["config"] for p in front],
    }


def main():
    print("┌" + "─"*78 + "┐")
    print("│ Plan2-Phase-AP: NSGA-II 5D Pareto demo (LGB v3 lat × LGB v4 real-AP)" + " "*9 + "│")
    print("│ 目标: min(lat, workspace, params), max(throughput, ap50)" + " "*22 + "│")
    print("└" + "─"*78 + "┘")

    lat_pred = LGBv3LatPredictor()
    ap_pred = LGBv4APPredictor()

    results = {}
    results["rtx4090"] = run_pareto("RTX 4090", gen_4090_space, lat_pred, ap_pred)
    results["orin_agx"] = run_pareto("Orin AGX", gen_orin_space, lat_pred, ap_pred)
    results["ap_axis_enabled"] = ap_pred.ready

    out = ROOT / "results/nsga2_pareto_demo_v4.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ saved -> {out}")

    rows = []
    for hw, res in results.items():
        if hw == "ap_axis_enabled" or not res: continue
        for c in res["pareto_configs"]:
            rows.append({
                "hardware": hw,
                "triplet": f"({c['stage0_planes']},{c['stage1_planes']},{c['stage2_planes']})",
                "precision": c["precision"],
                "d_scheme": c["d_scheme"],
                "d_tactic": c["d_tactic"],
                "workspace_gb": c["d_workspace_gb"],
                "lat_p50_ms": c["lat_p50_ms"],
                "throughput_fps": c["throughput_fps"],
                "ap50": c.get("ap50"),
                "ap70": c.get("ap70"),
                "params_kb": c.get("params_kb"),
                "build_success_prob": c["build_success_prob"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/nsga2_pareto_demo_v4.csv", index=False)
    print(f"✅ saved -> results/nsga2_pareto_demo_v4.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
