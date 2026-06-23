"""Plan2-Step5 — NSGA-II 自动 Pareto 发现 demo.

输入: LGB v2 预测器 (lat + build_success)
搜索空间: B × Q × D_scheme × D_tactic × D_workspace (4090 和 Orin 各跑一次)
目标:
    minimize lat_p50_ms
    minimize workspace_gb (资源 Pareto 轴, 越省越好)
    maximize build_success_prob (≥ 0.5 阈值)

输出: Pareto frontier 配置列表 + Pareto 散点图数据
"""
from __future__ import annotations
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Predictor wrapper
# ============================================================
class LGBv2Predictor:
    def __init__(self):
        self.lat = lgb.Booster(model_file=str(ROOT / "models/lgb_v2_lat.txt"))
        self.bs = lgb.Booster(model_file=str(ROOT / "models/lgb_v2_build_success.txt"))
        self.names = ["stage0_planes","stage1_planes","stage2_planes",
                      "prec_fp16","prec_int8",
                      "scheme_A","scheme_B0","scheme_B1",
                      "tactic_default","tactic_other",
                      "workspace_gb","hw_4090","hw_orin"]

    def featurize_one(self, cfg):
        return np.array([
            cfg["stage0_planes"], cfg["stage1_planes"], cfg["stage2_planes"],
            1 if cfg["precision"]=="fp16" else 0,
            1 if cfg["precision"]=="int8" else 0,
            1 if cfg["d_scheme"]=="A_gpu" else 0,
            1 if cfg["d_scheme"]=="B_dla0" else 0,
            1 if cfg["d_scheme"]=="B_dla1" else 0,
            1 if cfg["d_tactic"]=="default" else 0,
            0 if cfg["d_tactic"]=="default" else 1,
            cfg["d_workspace_gb"],
            1 if cfg["hardware"]=="rtx4090" else 0,
            1 if cfg["hardware"]=="orin_agx" else 0,
        ], dtype=np.float32).reshape(1, -1)

    def predict(self, cfg):
        x = self.featurize_one(cfg)
        return {
            "lat_p50_ms": float(self.lat.predict(x)[0]),
            "build_success_prob": float(self.bs.predict(x)[0]),
        }


# ============================================================
# Search space enumeration (per hardware)
# ============================================================
TRIPLETS_4090 = [
    (64, 128, 256), (48, 96, 192), (40, 64, 128), (32, 64, 136),
    (32, 64, 128), (24, 48, 96), (16, 32, 64), (16, 32, 32),
    (24, 32, 64), (40, 72, 128),
]
TRIPLETS_ORIN = TRIPLETS_4090 + [
    (48, 96, 192), (48, 56, 96), (56, 96, 200),  # Orin soft-alignment specific
]

PRECISIONS = ["fp16", "int8"]
TACTICS = ["default", "no_cudnn", "with_cudnn", "edge_only", "all_enabled"]


def gen_4090_space():
    """4090 search space, restricted to ws ∈ {4, 8} (LGB training range)."""
    cfgs = []
    for tri in TRIPLETS_4090:
        for prec in PRECISIONS:
            for tactic in TACTICS:
                for ws in [4, 8]:
                    cfgs.append({
                        "stage0_planes": tri[0], "stage1_planes": tri[1], "stage2_planes": tri[2],
                        "precision": prec, "d_scheme": "A_gpu",
                        "d_tactic": tactic, "d_workspace_gb": ws,
                        "hardware": "rtx4090",
                    })
    return cfgs


def gen_orin_space():
    """Orin search space, ws ∈ {1, 2} (Orin memory limit + LGB range)."""
    cfgs = []
    for tri in TRIPLETS_ORIN:
        for prec in PRECISIONS:
            for scheme in ["A_gpu", "B_dla0", "B_dla1"]:
                for tactic in ["default", "no_cudnn"]:
                    for ws in [1, 2]:
                        cfgs.append({
                            "stage0_planes": tri[0], "stage1_planes": tri[1], "stage2_planes": tri[2],
                            "precision": prec, "d_scheme": scheme,
                            "d_tactic": tactic, "d_workspace_gb": ws,
                            "hardware": "orin_agx",
                        })
    return cfgs


# ============================================================
# Pareto frontier (2D: lat × workspace, with build_success filter)
# ============================================================
def pareto_front_nd(points, axes):
    """Generic N-dim Pareto (minimize all axes).
    points: list of dicts; axes: list of dict-keys to minimize.
    """
    if not points:
        return []
    front = []
    for i, p1 in enumerate(points):
        dominated = False
        for j, p2 in enumerate(points):
            if i == j: continue
            if all(p2[a] <= p1[a] for a in axes) and any(p2[a] < p1[a] for a in axes):
                dominated = True; break
        if not dominated:
            front.append(p1)
    return sorted(front, key=lambda p: p[axes[0]])


def model_params_proxy(cfg):
    """Approximate model size: dominated by ResNeXt 3x3 grouped conv (groups=32).
    Each stage conv2 params ≈ 9 * 2*planes * 2*planes / 32. Sum over 3 stages.
    Plus shrink_conv + heads (constant ~5K params)."""
    s0, s1, s2 = cfg["stage0_planes"], cfg["stage1_planes"], cfg["stage2_planes"]
    p = 9 * (2*s0)*(2*s0)/32 + 9 * (2*s1)*(2*s1)/32 + 9 * (2*s2)*(2*s2)/32
    # account for INT8 vs FP16 weight size (INT8 = 1 byte, FP16 = 2 bytes)
    bytes_per_w = 1 if cfg["precision"] == "int8" else 2
    return p * bytes_per_w / 1024.0  # KB


# ============================================================
# Main
# ============================================================
def run(hw_label, space_fn):
    print("="*70)
    print(f"NSGA-II Pareto Demo: {hw_label}")
    print("="*70)
    predictor = LGBv2Predictor()
    cfgs = space_fn()
    print(f"\nSearch space size: {len(cfgs)} configurations")

    # Evaluate every config
    evaluated = []
    for cfg in cfgs:
        pred = predictor.predict(cfg)
        params_kb = model_params_proxy(cfg)
        cfg_eval = {**cfg, **pred,
                    "params_kb": params_kb,
                    "feasible": pred["build_success_prob"] >= 0.5}
        evaluated.append(cfg_eval)

    feasible = [e for e in evaluated if e["feasible"]]
    print(f"Feasible (build_success_prob ≥ 0.5): {len(feasible)} / {len(cfgs)}")
    if not feasible:
        return None

    # 3D Pareto: minimize (lat_p50, workspace_gb, params_kb)
    points = [{
        "lat_p50_ms": e["lat_p50_ms"],
        "workspace_gb": e["d_workspace_gb"],
        "params_kb": e["params_kb"],
        "config": e,
    } for e in feasible]
    front = pareto_front_nd(points, ["lat_p50_ms", "workspace_gb", "params_kb"])
    print(f"\nPareto frontier size: {len(front)} (3D: lat × workspace × params)")
    print(f"\n{'idx':<4} {'lat (ms)':<10} {'ws (GB)':<8} {'params (KB)':<13} "
          f"{'triplet':<22} {'prec':<6} {'scheme':<8} {'tactic':<14}")
    for i, p in enumerate(front):
        c = p["config"]
        tri = f"({c['stage0_planes']},{c['stage1_planes']},{c['stage2_planes']})"
        print(f"{i:<4} {p['lat_p50_ms']:<10.3f} {p['workspace_gb']:<8} {p['params_kb']:<13.1f} "
              f"{tri:<22} {c['precision']:<6} {c['d_scheme']:<8} {c['d_tactic']:<14}")

    return {
        "hardware": hw_label,
        "search_space_size": len(cfgs),
        "feasible_count": len(feasible),
        "pareto_front_size": len(front),
        "pareto_configs": [p["config"] for p in front],
    }


def main():
    print("\n┌─" + "─"*68 + "┐")
    print("│ Plan2-Step5 NSGA-II Pareto demo (使用 LGB v2 跨硬件预测器)" + " "*8 + "│")
    print("│ 目标维度: lat × workspace  (二维 Pareto, 待 Step 3 throughput 加入第三维)│")
    print("└─" + "─"*68 + "┘\n")

    results = {}
    results["rtx4090"] = run("RTX 4090", gen_4090_space)
    print()
    results["orin_agx"] = run("Orin AGX", gen_orin_space)

    out = ROOT / "results/nsga2_pareto_demo.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ saved -> {out}")

    # CSV for plotting
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
                "params_kb": c.get("params_kb"),
                "build_success_prob": c["build_success_prob"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/nsga2_pareto_demo.csv", index=False)
    print(f"✅ saved -> results/nsga2_pareto_demo.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
