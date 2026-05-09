"""M4.7 v2: 修复版 — Pyramid 50 候选 Pareto, 用 LGB v5.1 替代 KNN.

vs v1 (m4_7_pyramid_pareto.py) 的关键修复:
  v1 → 用 KNN over 8 行 PyramidFusion 子模块 latency anchors (4.5/3.3 ms 量级)
       Pareto 双轴 = (latency × params)
       结果: m4_7_pyramid_0040 标 Pareto, 实测最差 (AP70=0.140)

  v2 → 用 LGB v5.1 (univ2x baseline 训) 预测 latency + amota, 跨模型 transfer
       Pareto 双轴 = (predicted_lat × predicted_amota), 修复反思 #6 + #7
       sanity check: 用 M4.6.0 实测 Pyramid baseline (FP32 35ms / AP50 0.9635) 验证 LGB 预测对应程度

输出:
  results/phase2_pareto_pyramid_v2.csv (50 候选 + LGB-predicted lat/amota + Pareto)
  results/m4_7_v2_lgb_calibration.txt (LGB 在 M4.6.0 锚点上的预测误差报告)
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from framework.adapters import pyramid_fusion as pf_adapter
from framework.capability_schema import HardwareCapability
from framework.config_schema import Config, UNIV2X_MODULES
from framework.searcher_v0 import random_search
from scripts.phase2.train_lgb_v5_1 import featurize  # 复用 featurize

BASELINE = REPO_ROOT / "data/baseline_4090.parquet"
RESULTS_DIR = REPO_ROOT / "results"
LGB_AMOTA = REPO_ROOT / "models/lgb_v5_1_amota.txt"
LGB_LATENCY = REPO_ROOT / "models/lgb_v5_1_latency.txt"


def cfg_to_predictor_row(cfg: Config) -> pd.Series:
    """把 Config 转成 LGB v5.1 featurize 期望的 Series schema."""
    cd = cfg.to_dict()
    row = {
        "prune_object": cd["prune_object"],
        "d_pipelined": int(cd.get("d_pipelined", 0)),
        "d_temporal_cache_int8": int(cd.get("d_temporal_cache_int8", 0)),
        "d_runtime": cd.get("d_runtime", "pytorch_fp32"),
        # v5.1 source one-hot: 用 'synth_dspace' (跨模型 LGB 训练时认到的 synth 标记)
        "source": "synth_dspace",
    }
    for m in UNIV2X_MODULES:
        row[f"prune_rate__{m}"] = float(cd["prune_rate"].get(m, 0.0))
        row[f"prune_criterion__{m}"] = cd["prune_criterion"].get(m, "none")
        row[f"q_bits__{m}"] = cd["q_bits"].get(m, "FP32")
        row[f"q_granularity__{m}"] = cd["q_granularity"].get(m, "none")
        row[f"q_object__{m}"] = cd["q_object"].get(m, "none")
        row[f"d_routing__{m}"] = cd["d_routing"].get(m, "GPU")
    return pd.Series(row)


def lgb_predict_batch(configs: list[Config], booster_a: lgb.Booster, booster_l: lgb.Booster) -> tuple[np.ndarray, np.ndarray]:
    """对 N 个 Config 用 LGB v5.1 同时预测 (amota, latency)."""
    df = pd.DataFrame([cfg_to_predictor_row(c) for c in configs])
    X, _ = featurize(df)
    return booster_a.predict(X), booster_l.predict(X)


def calibrate_on_pyramid_baseline(booster_a: lgb.Booster, booster_l: lgb.Booster) -> dict:
    """用 M4.6.0 Pyramid baseline 实测 (FP32 35ms / AP50 0.9635) 验证 LGB 预测准度."""
    df = pd.read_parquet(BASELINE)
    pyramid_full = df[df["source"] == "m4_6_0_pyramid_full_e2e"].copy()
    if len(pyramid_full) == 0:
        return {"calibration_unavailable": True}

    X, _ = featurize(pyramid_full)
    pred_a = booster_a.predict(X)
    pred_l = booster_l.predict(X)
    actual_a = pyramid_full["amota"].to_numpy()
    actual_l = pyramid_full["lat_e2e_ms"].to_numpy()

    return {
        "n_anchors": len(pyramid_full),
        "amota_actual": actual_a.tolist(),
        "amota_predicted": pred_a.tolist(),
        "amota_mae": float(np.mean(np.abs(pred_a - actual_a))),
        "latency_actual_ms": actual_l.tolist(),
        "latency_predicted_ms": pred_l.tolist(),
        "latency_mae_ms": float(np.mean(np.abs(pred_l - actual_l))),
    }


def pareto_mask_min_max(lat: np.ndarray, amota: np.ndarray) -> np.ndarray:
    """Pareto: minimize latency + maximize amota. 返回布尔."""
    n = len(lat)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if lat[j] <= lat[i] and amota[j] >= amota[i] and \
               (lat[j] < lat[i] or amota[j] > amota[i]):
                is_pareto[i] = False
                break
    return is_pareto


def cfg_to_row(cfg: Config) -> dict:
    row = {
        "config_id": cfg.config_id,
        "source": "m4_7_v2_pyramid_pareto",
        "model_class": "pyramid_fusion",
        "prune_object": cfg.prune_object,
    }
    for m in UNIV2X_MODULES:
        row[f"prune_rate__{m}"] = float(cfg.prune_rate.get(m, 0.0))
        row[f"prune_criterion__{m}"] = cfg.prune_criterion.get(m, "none")
        row[f"q_bits__{m}"] = cfg.q_bits.get(m, "FP32")
        row[f"q_granularity__{m}"] = cfg.q_granularity.get(m, "none")
        row[f"q_object__{m}"] = cfg.q_object.get(m, "none")
        row[f"d_routing__{m}"] = cfg.d_routing.get(m, "GPU")
    return row


def main(n_candidates: int = 50, seed: int = 42) -> None:
    print("=" * 60)
    print("M4.7 v2 — Pyramid 50 候选 Pareto (LGB v5.1 双预测器)")
    print("=" * 60)

    # 1. Load LGB v5.1
    booster_a = lgb.Booster(model_file=str(LGB_AMOTA))
    booster_l = lgb.Booster(model_file=str(LGB_LATENCY))
    print(f"✅ Loaded LGB v5.1 amota + latency boosters")

    # 1.5 Sanity check on M4.6.0 实测 anchor
    cal = calibrate_on_pyramid_baseline(booster_a, booster_l)
    print(f"\n=== LGB v5.1 在 Pyramid baseline 锚点上的预测准度 ===")
    if cal.get("calibration_unavailable"):
        print(f"  ⚠️ baseline_4090.parquet 没有 m4_6_0_pyramid_full_e2e 行, 跳过校准")
    else:
        for i in range(cal["n_anchors"]):
            print(f"  anchor {i}: actual amota={cal['amota_actual'][i]:.4f}, "
                  f"predicted={cal['amota_predicted'][i]:.4f} "
                  f"(diff {cal['amota_predicted'][i]-cal['amota_actual'][i]:+.4f})")
            print(f"           actual lat={cal['latency_actual_ms'][i]:.2f} ms, "
                  f"predicted={cal['latency_predicted_ms'][i]:.2f} ms "
                  f"(diff {cal['latency_predicted_ms'][i]-cal['latency_actual_ms'][i]:+.2f})")
        print(f"  amota MAE on anchor: {cal['amota_mae']:.4f}")
        print(f"  latency MAE on anchor: {cal['latency_mae_ms']:.2f} ms")

    # 2. random_search + adapter filter
    orin_agx = HardwareCapability.from_yaml(REPO_ROOT / "configs/hardware/orin_agx.yaml")
    candidates_raw, stats = random_search(
        orin_agx, n_candidates=n_candidates * 2, seed=seed,
        lock_d_to_gpu_only=True, verbose=False,
    )
    print(f"\n✅ random_search: {len(candidates_raw)} raw, pass_rate={stats['pass_rate']:.1%}")

    candidates = []
    n_remapped = 0
    pyramid_filter_reasons = {}
    for cfg in candidates_raw:
        new_crit = dict(cfg.prune_criterion)
        for m in UNIV2X_MODULES:
            if cfg.prune_rate.get(m, 0.0) > 0:
                pool = pf_adapter.get_recommended_criterion_pool(m)
                if new_crit.get(m) not in pool and new_crit.get(m) != "none":
                    new_crit[m] = pool[0]
                    n_remapped += 1
        cfg_remapped = cfg.with_field(prune_criterion=new_crit)
        ok, reason = pf_adapter.is_valid_for_pyramid(cfg_remapped)
        if not ok:
            tag = reason.split("]")[0].lstrip("[") if "]" in reason else reason[:30]
            pyramid_filter_reasons[tag] = pyramid_filter_reasons.get(tag, 0) + 1
            continue
        candidates.append(cfg_remapped.with_field(config_id=f"m4_7_v2_pyramid_{len(candidates):04d}"))
        if len(candidates) >= n_candidates:
            break
    print(f"✅ pyramid_fusion adapter: {len(candidates)}/{len(candidates_raw)} 通过 "
          f"({n_remapped} criterion remap)")

    # 3. LGB v5.1 双预测
    pred_amota, pred_lat = lgb_predict_batch(candidates, booster_a, booster_l)
    print(f"✅ LGB 预测 done: amota range [{pred_amota.min():.3f}, {pred_amota.max():.3f}], "
          f"lat range [{pred_lat.min():.1f}, {pred_lat.max():.1f}] ms")

    # 4. Pareto: minimize lat + maximize amota
    is_pareto = pareto_mask_min_max(pred_lat, pred_amota)
    print(f"✅ Pareto 前沿: {is_pareto.sum()} 点")

    # 5. 写表
    rows = []
    for i, cfg in enumerate(candidates):
        row = cfg_to_row(cfg)
        row["predicted_amota"] = float(pred_amota[i])
        row["predicted_lat_ms"] = float(pred_lat[i])
        row["is_pareto"] = bool(is_pareto[i])
        rows.append(row)
    df = pd.DataFrame(rows)

    # 列顺序
    front = ["config_id", "model_class", "source", "is_pareto",
             "predicted_lat_ms", "predicted_amota",
             "prune_object"]
    cfg_cols = []
    for m in UNIV2X_MODULES:
        for p in ("prune_rate", "prune_criterion", "q_bits", "q_granularity", "q_object", "d_routing"):
            cfg_cols.append(f"{p}__{m}")
    df = df[[c for c in front + cfg_cols if c in df.columns]]

    out = RESULTS_DIR / "phase2_pareto_pyramid_v2.csv"
    df.to_csv(out, index=False)
    print(f"\n✅ Wrote {out} ({len(df)} rows)")

    # 校准报告
    cal_out = RESULTS_DIR / "m4_7_v2_lgb_calibration.txt"
    with cal_out.open("w") as f:
        f.write(f"# M4.7 v2 LGB v5.1 calibration on Pyramid baseline anchors\n")
        if not cal.get("calibration_unavailable"):
            for i in range(cal["n_anchors"]):
                f.write(f"anchor {i}: actual_amota={cal['amota_actual'][i]:.4f}, "
                        f"predicted={cal['amota_predicted'][i]:.4f}, "
                        f"actual_lat={cal['latency_actual_ms'][i]:.2f}, "
                        f"predicted_lat={cal['latency_predicted_ms'][i]:.2f}\n")
            f.write(f"amota_mae={cal['amota_mae']:.4f}, lat_mae_ms={cal['latency_mae_ms']:.2f}\n")
    print(f"✅ Wrote {cal_out}")

    # Pareto 摘要
    pareto = df[df["is_pareto"]].sort_values("predicted_lat_ms")
    print(f"\n=== Pareto frontier (sorted by lat) ===")
    show_cols = ["config_id", "predicted_lat_ms", "predicted_amota", "prune_object",
                 "q_bits__encoder", "prune_rate__encoder", "prune_rate__decoder"]
    print(pareto[show_cols].to_string(index=False))

    print(f"\n=== Stats ===")
    print(f"  predicted_lat: min={df['predicted_lat_ms'].min():.2f} max={df['predicted_lat_ms'].max():.2f} mean={df['predicted_lat_ms'].mean():.2f}")
    print(f"  predicted_amota: min={df['predicted_amota'].min():.3f} max={df['predicted_amota'].max():.3f} mean={df['predicted_amota'].mean():.3f}")
    print(f"  prune_object 分布: {df['prune_object'].value_counts().to_dict()}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n_candidates", type=int, default=50)
    ap.add_argument("-s", "--seed", type=int, default=42)
    args = ap.parse_args()
    main(n_candidates=args.n_candidates, seed=args.seed)
