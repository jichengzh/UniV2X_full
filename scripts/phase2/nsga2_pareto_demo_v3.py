"""Plan2-Phase4 — NSGA-II 4D Pareto demo with throughput.

预测器: LGB v3 (lat + throughput + build_success)
目标(4D Pareto): minimize lat_p50, **maximize throughput_fps**, minimize workspace_gb, minimize params_kb

输出:
    - 每硬件 Pareto frontier(展示 lat-optimal vs throughput-optimal 两支)
    - 二维投影散点图数据(lat × throughput)
"""
from __future__ import annotations
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


class LGBv3Predictor:
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
    """N-dim Pareto. axes_min: keys to minimize. axes_max: keys to maximize (negated)."""
    axes_max = axes_max or []
    if not points: return []
    def dominates(p, q):
        # p dominates q if p ≤ q on all min axes AND p ≥ q on all max axes AND strict in one
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


def run_pareto(hw_label, space_fn):
    print(f"\n{'='*72}")
    print(f"NSGA-II 4D Pareto Demo: {hw_label}")
    print(f"{'='*72}")
    pred = LGBv3Predictor()
    cfgs = space_fn()
    print(f"Search space: {len(cfgs)} configurations")

    evaluated = []
    for cfg in cfgs:
        p = pred.predict(cfg)
        cfg_eval = {**cfg, **p,
                    "params_kb": model_params_kb(cfg),
                    "feasible": p["build_success_prob"] >= 0.5}
        evaluated.append(cfg_eval)

    feasible = [e for e in evaluated if e["feasible"]]
    print(f"Feasible (build_success_prob ≥ 0.5): {len(feasible)} / {len(cfgs)}")

    if not feasible:
        return None

    # 4D Pareto: minimize (lat, workspace, params), maximize (throughput)
    points = [{
        "lat_p50_ms": e["lat_p50_ms"],
        "throughput_fps": e["throughput_fps"],
        "workspace_gb": e["d_workspace_gb"],
        "params_kb": e["params_kb"],
        "config": e,
    } for e in feasible]
    front = pareto_front_nd(points,
                             axes_min=["lat_p50_ms","workspace_gb","params_kb"],
                             axes_max=["throughput_fps"])
    print(f"\n4D Pareto frontier size: {len(front)}")
    print(f"\n{'#':<3} {'lat (ms)':<10} {'thr (fps)':<10} {'ws':<4} {'params':<9} "
          f"{'triplet':<22} {'prec':<6} {'scheme':<8} {'tactic':<13}")
    for i, p in enumerate(front):
        c = p["config"]
        tri = f"({c['stage0_planes']},{c['stage1_planes']},{c['stage2_planes']})"
        print(f"{i:<3} {p['lat_p50_ms']:<10.3f} {p['throughput_fps']:<10.1f} "
              f"{p['workspace_gb']:<4} {p['params_kb']:<9.1f} "
              f"{tri:<22} {c['precision']:<6} {c['d_scheme']:<8} {c['d_tactic']:<13}")

    # Find lat-optimal and throughput-optimal
    lat_opt = min(front, key=lambda p: p["lat_p50_ms"])
    thr_opt = max(front, key=lambda p: p["throughput_fps"])
    print(f"\n*** lat-optimal: {lat_opt['lat_p50_ms']:.3f} ms @ {lat_opt['throughput_fps']:.1f} fps,"
          f" scheme={lat_opt['config']['d_scheme']}")
    print(f"*** throughput-optimal: {thr_opt['throughput_fps']:.1f} fps @ {thr_opt['lat_p50_ms']:.3f} ms,"
          f" scheme={thr_opt['config']['d_scheme']}")
    print(f"*** lat vs throughput trade-off: {thr_opt['throughput_fps']/lat_opt['throughput_fps']:.2f}× thr gain"
          f" at {thr_opt['lat_p50_ms']/lat_opt['lat_p50_ms']:.2f}× lat cost")

    return {
        "hardware": hw_label,
        "search_space_size": len(cfgs),
        "feasible_count": len(feasible),
        "pareto_size": len(front),
        "lat_optimal": lat_opt["config"],
        "throughput_optimal": thr_opt["config"],
        "pareto_configs": [p["config"] for p in front],
    }


def main():
    print("┌" + "─"*78 + "┐")
    print("│ Plan2-Phase4: NSGA-II 4D Pareto demo (LGB v3 跨硬件预测器)" + " "*22 + "│")
    print("│ 目标: min(lat,workspace,params), max(throughput)" + " "*30 + "│")
    print("└" + "─"*78 + "┘")

    results = {}
    results["rtx4090"] = run_pareto("RTX 4090", gen_4090_space)
    results["orin_agx"] = run_pareto("Orin AGX", gen_orin_space)

    out = ROOT / "results/nsga2_pareto_demo_v3.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ saved -> {out}")

    rows = []
    for hw, res in results.items():
        if not res: continue
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
                "params_kb": c.get("params_kb"),
                "build_success_prob": c["build_success_prob"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/nsga2_pareto_demo_v3.csv", index=False)
    print(f"✅ saved -> results/nsga2_pareto_demo_v3.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
