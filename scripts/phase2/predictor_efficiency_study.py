"""Stage 1 (1a + 1b + 1c) — sampling efficiency study on 224 anchor oracle.

实现 创新点.md §2 方法论 + 数据集制作_plan.md §5 Stage 1.

数据: paper_learning/2. AAAI最终故事/data/e2e_bench_v1.csv (224 anchor, e2e n=1789 AP)

Stage 1a — sampling efficiency study (4 sampling × 6 K size × 5 repeat)
  策略: Random / Latin Hypercube / Sobol / Sensitivity-stratified
  K sizes: 16, 32, 48, 64, 96, 128
  Target: throughput_fps (lat predictor), ap50 (AP predictor)
  Eval: R² / MAE on (224 - K) hold-out

Stage 1b — Active learning loop
  Seed K=32 sensitivity-stratified, +8 anchor/round × 8 round → K=32..96
  Selection: LGB leaf-variance uncertainty top-8

Stage 1c — Sub-predictor decomposition
  f_lat (B, Q, D), f_AP_baseline (B, ckpt_source), f_AP_crash (Q, ckpt_source)
  per-dim Shapley importance (LGB feature importance) on each sub-predictor

产出: results/predictor_efficiency/
  - summary.json
  - k_r2_curves.png   (主图 a + b)
  - active_vs_static.png   (主图 c)
  - per_dim_shapley.png    (主图 d)
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.metrics import r2_score, mean_absolute_error
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path("/home/jichengzhi/UniV2X")
CSV = REPO / "paper_learning/2. AAAI最终故事/data/e2e_bench_v1.csv"
OUT_DIR = REPO / "results/predictor_efficiency"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Triplet → prune ratio (B 维度 numeric encoding) — 含 Track A v2 13 新 triplet
TRIPLET_PRUNE = {
    "T1_base": 0, "T2_p25": 25, "T3_p37": 37, "T4_p50": 50,
    "T5_p62": 62, "T6_p75": 75,
    "T7_wide_shallow": 40, "T8_narrow_deep": 55,
    "T10_p11": 11, "T11_p14": 14, "T12_p21": 21, "T13_p29": 29,
    "T14_p36": 36, "T15_p39": 39, "T16_p54": 54, "T17_p57": 57,
    "T18_p64": 64, "T19_p71": 71, "T20_p79": 79, "T21_p82": 82,
    "T22_p89": 89,
}
# 官方 HEAL ckpt vs 我们自训 ckpt — 影响 entropy crash (问题 1).
# T10-T22 全部 self_trained (Track A v2 dataset_a_prepare_ckpts_v2.py).
TRIPLET_CKPT_SOURCE = {
    "T1_base": "official", "T2_p25": "official",
    "T4_p50": "official", "T6_p75": "official",
    "T3_p37": "self_trained", "T5_p62": "self_trained",
    "T7_wide_shallow": "self_trained", "T8_narrow_deep": "self_trained",
    "T10_p11": "self_trained", "T11_p14": "self_trained",
    "T12_p21": "self_trained", "T13_p29": "self_trained",
    "T14_p36": "self_trained", "T15_p39": "self_trained",
    "T16_p54": "self_trained", "T17_p57": "self_trained",
    "T18_p64": "self_trained", "T19_p71": "self_trained",
    "T20_p79": "self_trained", "T21_p82": "self_trained",
    "T22_p89": "self_trained",
}
Q_LIST = ["Q_fp32", "Q_fp16", "Q_int8_mm", "Q_int8_ent",
          "Q_mix_s0", "Q_mix_s2", "Q_int8_pc_wo"]
D_TACTIC_LIST = ["default", "with_cudnn", "cublas_lt", "all_enabled", "edge_only"]


def load_oracle(exclude_entropy=True):
    """Load anchor as oracle dataset + featurize.

    Q_int8_ent 整体不可信: AP 部分崩 (4/8 自训 ckpt + 部分官方 ckpt),
    lat 在某些 D-config 上触发 outlier kernel (T6_p75/Q_int8_ent/default/16GB → 333 fps,
    6× 异常). 排除整个 Q_int8_ent 维度, 让 Stage 1a/1b 在 healthy 子集上验证 sampling 创新.
    Q_int8_ent crashed signal 保留在 Stage 1c f_AP_crash binary classifier.
    """
    df = pd.read_csv(CSV)
    df = df[df["build_success"]].copy().reset_index(drop=True)
    if exclude_entropy:
        before = len(df)
        df = df[df["q_tag"] != "Q_int8_ent"].copy().reset_index(drop=True)
        print(f"[oracle] excluded {before - len(df)} Q_int8_ent anchor "
              "(AP + lat 均不可信)")
    df["prune_pct"] = df["triplet"].map(TRIPLET_PRUNE)
    df["ckpt_source"] = df["triplet"].map(TRIPLET_CKPT_SOURCE)
    df["ckpt_official"] = (df["ckpt_source"] == "official").astype(int)
    df["ap_crashed"] = (df["ap50"] < 0.4).astype(int)  # binary crash label
    # T idx / Q idx / D idx for discrete grid sampling
    df["T_idx"] = df["triplet"].map({t: i for i, t in enumerate(TRIPLET_PRUNE)})
    df["Q_idx"] = df["q_tag"].map({q: i for i, q in enumerate(Q_LIST)})
    df["D_idx"] = df["d_tactic"].map({d: i for i, d in enumerate(D_TACTIC_LIST)})
    return df


# === Feature builders ===

_D_TAG_IDX_CACHE = {}

def _get_d_tag_idx(d_tag_series):
    """Stable d_tag → int mapping (lazy, sorted)."""
    global _D_TAG_IDX_CACHE
    uniq = sorted(d_tag_series.dropna().unique().tolist())
    if list(_D_TAG_IDX_CACHE.keys()) != uniq:
        _D_TAG_IDX_CACHE = {d: i for i, d in enumerate(uniq)}
    return d_tag_series.map(_D_TAG_IDX_CACHE).fillna(-1).astype(int)


def features_lat(df):
    """f_lat: full features (B + Q + D_tactic + D_workspace + D_BL + D_tag) for throughput prediction.

    v1.4: 加 d_builder_opt_level (BL) + d_tag.
    d_tag 是关键: 同 (tactic, ws, BL=3) 重测的 D1 vs D25 在所有数值 feature 上等价,
    但 fps 差 ~28% (TRT autotuner build-to-build 噪声). d_tag 让 LGB 能区分 build 实例.
    """
    bl = df.get("d_builder_opt_level")
    if bl is None:
        bl = pd.Series([3] * len(df))  # 老 csv 兜底
    d_tag = df.get("d_tag")
    if d_tag is None:
        d_tag = pd.Series(["D_unknown"] * len(df))
    feat = pd.DataFrame({
        "prune_pct":     df["prune_pct"],
        "stage0_planes": df["stage0_planes"],
        "stage1_planes": df["stage1_planes"],
        "stage2_planes": df["stage2_planes"],
        "Q_idx":         df["Q_idx"],
        "D_idx":         df["D_idx"],
        "d_workspace_gb": df["d_workspace_gb"],
        "d_builder_opt_level": bl.fillna(3).astype(int),
        "d_tag_idx":     _get_d_tag_idx(d_tag),
        "ckpt_official": df["ckpt_official"],
    })
    return feat.values, ["prune_pct", "s0", "s1", "s2", "Q_idx", "D_idx",
                          "ws_gb", "BL", "d_tag_idx", "ckpt_official"]


def features_ap_full(df):
    """f_AP_full: full features for AP regression."""
    return features_lat(df)


def features_ap_baseline(df):
    """f_AP_baseline: only B + ckpt_source (no Q, no D) — for Stage 1c."""
    feat = pd.DataFrame({
        "prune_pct":     df["prune_pct"],
        "stage0_planes": df["stage0_planes"],
        "stage1_planes": df["stage1_planes"],
        "stage2_planes": df["stage2_planes"],
        "ckpt_official": df["ckpt_official"],
    })
    return feat.values, ["prune_pct", "s0", "s1", "s2", "ckpt_official"]


def features_ap_crash(df):
    """f_AP_crash: only Q + ckpt_source → binary crash."""
    feat = pd.DataFrame({
        "Q_idx":         df["Q_idx"],
        "ckpt_official": df["ckpt_official"],
    })
    return feat.values, ["Q_idx", "ckpt_official"]


# === LGB training + eval ===

def train_eval_lgb(X_train, y_train, X_test, y_test, task="regression"):
    """Train LGB, return R², MAE on hold-out."""
    if task == "regression":
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, min_child_samples=2,
                              verbose=-1, random_state=42)
    else:
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               num_leaves=31, min_child_samples=2,
                               verbose=-1, random_state=42)
    m.fit(X_train, y_train)
    if task == "regression":
        y_pred = m.predict(X_test)
        return r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), m
    else:
        y_pred_proba = m.predict_proba(X_test)[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 1.0
        return auc, None, m


# === Sampling strategies ===

def sample_random(N, K, seed):
    """Uniform random."""
    rng = np.random.default_rng(seed)
    return rng.choice(N, size=K, replace=False)


def sample_lhs(df, K, seed):
    """Latin Hypercube on (T_idx, Q_idx, D_idx) → find nearest anchor."""
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    u = sampler.random(K)
    # Scale to integer grid
    target_T = np.floor(u[:, 0] * 8).astype(int).clip(0, 7)
    target_Q = np.floor(u[:, 1] * 7).astype(int).clip(0, 6)
    target_D = np.floor(u[:, 2] * 4).astype(int).clip(0, 3)
    # Map to row index in df (full cartesian, just find match)
    selected = []
    used = set()
    for t, q, d in zip(target_T, target_Q, target_D):
        mask = (df["T_idx"] == t) & (df["Q_idx"] == q) & (df["D_idx"] == d)
        candidates = df.index[mask].tolist()
        for c in candidates:
            if c not in used:
                selected.append(c); used.add(c); break
        else:
            # Fallback: random unused
            remaining = [i for i in range(len(df)) if i not in used]
            if remaining:
                c = np.random.default_rng(seed + len(used)).choice(remaining)
                selected.append(int(c)); used.add(int(c))
    return np.array(selected)


def sample_sobol(df, K, seed):
    """Sobol low-discrepancy on (T, Q, D) → nearest anchor."""
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    u = sampler.random(K)
    target_T = np.floor(u[:, 0] * 8).astype(int).clip(0, 7)
    target_Q = np.floor(u[:, 1] * 7).astype(int).clip(0, 6)
    target_D = np.floor(u[:, 2] * 4).astype(int).clip(0, 3)
    selected = []
    used = set()
    for t, q, d in zip(target_T, target_Q, target_D):
        mask = (df["T_idx"] == t) & (df["Q_idx"] == q) & (df["D_idx"] == d)
        candidates = df.index[mask].tolist()
        for c in candidates:
            if c not in used:
                selected.append(c); used.add(c); break
        else:
            remaining = [i for i in range(len(df)) if i not in used]
            if remaining:
                c = np.random.default_rng(seed + len(used)).choice(remaining)
                selected.append(int(c)); used.add(int(c))
    return np.array(selected)


def sample_sensitivity_stratified(df, K, seed, target_col="throughput_fps"):
    """Pilot K_pilot=8 random → per-dim variance → weighted stratified."""
    K_pilot = min(8, K)
    rng = np.random.default_rng(seed)
    pilot_idx = rng.choice(len(df), size=K_pilot, replace=False)
    pilot = df.iloc[pilot_idx]

    # Per-dim variance contribution (group means' variance)
    dim_vars = {}
    for dim_col in ["T_idx", "Q_idx", "D_idx"]:
        try:
            group_means = pilot.groupby(dim_col)[target_col].mean()
            dim_vars[dim_col] = group_means.var() if len(group_means) > 1 else 1.0
        except Exception:
            dim_vars[dim_col] = 1.0
    # Normalize → budget allocation
    tot_var = sum(dim_vars.values())
    if tot_var <= 0:
        return sample_random(len(df), K, seed)
    weights = {k: v / tot_var for k, v in dim_vars.items()}

    # 高方差 dim: 用全 stratification (每 bin 至少 1 个)
    # 低方差 dim: 集中 sample 几个代表 bin
    remaining_K = K - K_pilot
    used = set(pilot_idx.tolist())

    # Sample by 加权 stratified within high-importance dim
    # Sort dims by weight descending
    dim_order = sorted(weights.keys(), key=lambda k: -weights[k])
    dim_sizes = {"T_idx": 8, "Q_idx": 7, "D_idx": 4}

    selected = list(pilot_idx)
    # Round-robin per dim, stratifying by that dim's bins
    while len(selected) < K:
        for dim_col in dim_order:
            if len(selected) >= K:
                break
            # 在该 dim 上分层, 每个 bin 抽 1 个 unused
            for bin_v in range(dim_sizes[dim_col]):
                if len(selected) >= K:
                    break
                mask = (df[dim_col] == bin_v)
                candidates = [i for i in df.index[mask] if i not in used]
                if candidates:
                    # 加权 sampling: 权重 = 1/(已 sample 该 bin 的数量+1)
                    c = candidates[rng.integers(0, len(candidates))]
                    selected.append(c); used.add(c)
    return np.array(selected[:K])


SAMPLERS = {
    "random": lambda df, K, seed: sample_random(len(df), K, seed),
    "lhs": sample_lhs,
    "sobol": sample_sobol,
    "sensitivity_stratified": sample_sensitivity_stratified,
}


# === Stage 1a: K-R² curves ===

def stage_1a(df):
    print("\n=== Stage 1a: sampling efficiency study ===")
    X_lat, _ = features_lat(df)
    X_ap, _ = features_ap_full(df)
    y_lat = df["throughput_fps"].values
    y_ap = df["ap50"].values

    K_sizes = [16, 32, 48, 64, 96, 128]
    n_repeat = 5
    results = []
    for sampler_name, sampler_fn in SAMPLERS.items():
        for K in K_sizes:
            for rep in range(n_repeat):
                seed = rep * 1000 + K
                train_idx = sampler_fn(df, K, seed)
                hold_idx = np.array([i for i in range(len(df)) if i not in set(train_idx)])
                r2_lat, mae_lat, _ = train_eval_lgb(X_lat[train_idx], y_lat[train_idx],
                                                    X_lat[hold_idx], y_lat[hold_idx])
                r2_ap, mae_ap, _ = train_eval_lgb(X_ap[train_idx], y_ap[train_idx],
                                                  X_ap[hold_idx], y_ap[hold_idx])
                results.append({
                    "sampler": sampler_name, "K": K, "repeat": rep,
                    "r2_lat": r2_lat, "mae_lat": mae_lat,
                    "r2_ap": r2_ap, "mae_ap": mae_ap,
                })
        # Per-sampler summary
        sub = pd.DataFrame([r for r in results if r["sampler"] == sampler_name])
        agg = sub.groupby("K")[["r2_lat", "r2_ap"]].agg(["mean", "std"])
        print(f"  {sampler_name}:")
        for K in K_sizes:
            mlat, slat = agg.loc[K, ("r2_lat", "mean")], agg.loc[K, ("r2_lat", "std")]
            map_, sap = agg.loc[K, ("r2_ap", "mean")], agg.loc[K, ("r2_ap", "std")]
            print(f"    K={K:3d}: R²_lat={mlat:.3f}±{slat:.3f}, R²_ap={map_:.3f}±{sap:.3f}")
    return pd.DataFrame(results)


def plot_stage_1a(results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for target_idx, (target, ylabel) in enumerate([("r2_lat", "R²(throughput_fps)"),
                                                    ("r2_ap", "R²(ap50)")]):
        ax = axes[target_idx]
        for sampler in SAMPLERS:
            sub = results[results["sampler"] == sampler]
            agg = sub.groupby("K")[target].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["K"], agg["mean"], yerr=agg["std"],
                        marker="o", capsize=3, linewidth=1.5, label=sampler)
        ax.set_xlabel("K (training anchor count)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Stage 1a: K-R² curve ({target.split('_')[1]})", fontsize=12)
        ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5, label="R²=0.9")
        ax.axhline(0.85, color="gray", linestyle=":", alpha=0.5, label="R²=0.85")
        ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
        ax.set_ylim(-0.1, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


# === Stage 1b: Active learning ===

def stage_1b(df):
    print("\n=== Stage 1b: active learning loop ===")
    X_lat, _ = features_lat(df)
    X_ap, _ = features_ap_full(df)
    y_lat = df["throughput_fps"].values
    y_ap = df["ap50"].values

    K_seed = 32
    n_rounds = 8
    step = 8

    # Active learning trace
    active_trace_lat = []
    active_trace_ap = []
    # Static baseline (sensitivity-stratified at same K)
    static_trace_lat = []
    static_trace_ap = []

    rng_seed = 42
    train_idx = list(sample_sensitivity_stratified(df, K_seed, rng_seed))

    for round_i in range(n_rounds + 1):
        train_set = set(train_idx)
        hold_idx = np.array([i for i in range(len(df)) if i not in train_set])
        # Train predictors
        r2_lat, mae_lat, m_lat = train_eval_lgb(
            X_lat[train_idx], y_lat[train_idx], X_lat[hold_idx], y_lat[hold_idx])
        r2_ap, mae_ap, m_ap = train_eval_lgb(
            X_ap[train_idx], y_ap[train_idx], X_ap[hold_idx], y_ap[hold_idx])
        K_cur = len(train_idx)
        active_trace_lat.append((K_cur, r2_lat))
        active_trace_ap.append((K_cur, r2_ap))
        print(f"  active round {round_i}: K={K_cur:3d}  R²_lat={r2_lat:.3f}  R²_ap={r2_ap:.3f}")

        # Static baseline at same K (fresh sensitivity-stratified)
        static_idx = sample_sensitivity_stratified(df, K_cur, rng_seed + round_i)
        static_set = set(static_idx.tolist())
        static_hold = np.array([i for i in range(len(df)) if i not in static_set])
        s_r2_lat, _, _ = train_eval_lgb(X_lat[static_idx], y_lat[static_idx],
                                         X_lat[static_hold], y_lat[static_hold])
        s_r2_ap, _, _ = train_eval_lgb(X_ap[static_idx], y_ap[static_idx],
                                        X_ap[static_hold], y_ap[static_hold])
        static_trace_lat.append((K_cur, s_r2_lat))
        static_trace_ap.append((K_cur, s_r2_ap))

        if round_i == n_rounds:
            break

        # Uncertainty: tree-prediction variance proxy = abs residual on full data
        # (more rigorous: per-tree predictions variance, but LGB doesn't expose easily)
        # Use prediction interval via boost-quantile: train upper/lower quantile models
        m_low = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                  num_leaves=31, min_child_samples=2,
                                  objective="quantile", alpha=0.1,
                                  verbose=-1, random_state=42)
        m_high = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                   num_leaves=31, min_child_samples=2,
                                   objective="quantile", alpha=0.9,
                                   verbose=-1, random_state=42)
        m_low.fit(X_lat[train_idx], y_lat[train_idx])
        m_high.fit(X_lat[train_idx], y_lat[train_idx])
        uncertainty = m_high.predict(X_lat) - m_low.predict(X_lat)
        # Mask out already-selected
        uncertainty_masked = np.where(
            np.isin(np.arange(len(df)), train_idx), -1, uncertainty)
        # Select top-step
        new_idx = np.argsort(-uncertainty_masked)[:step].tolist()
        train_idx.extend(new_idx)

    return active_trace_lat, active_trace_ap, static_trace_lat, static_trace_ap


def plot_stage_1b(active_lat, active_ap, static_lat, static_ap, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for target_idx, (active_trace, static_trace, ylabel) in enumerate([
        (active_lat, static_lat, "R²(throughput_fps)"),
        (active_ap, static_ap, "R²(ap50)"),
    ]):
        ax = axes[target_idx]
        K_a, R_a = zip(*active_trace)
        K_s, R_s = zip(*static_trace)
        ax.plot(K_a, R_a, "o-", linewidth=2, label="Active learning (sens-seeded)")
        ax.plot(K_s, R_s, "s--", linewidth=2, label="Static sensitivity-stratified")
        ax.set_xlabel("K (training anchor count)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Stage 1b: active vs static ({ylabel.split('(')[1][:-1]})", fontsize=12)
        ax.axhline(0.9, color="gray", linestyle="--", alpha=0.4)
        ax.axhline(0.85, color="gray", linestyle=":", alpha=0.4)
        ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="lower right")
        ax.set_ylim(-0.1, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


# === Stage 1c: Sub-predictor decomposition ===

def stage_1c_split(df, df_full):
    """Healthy oracle for regression, full oracle for crash binary."""
    print("\n=== Stage 1c: sub-predictor decomposition ===")
    X_lat, names_lat = features_lat(df)
    X_ap_full, names_ap_full = features_ap_full(df)
    X_ap_base, names_ap_base = features_ap_baseline(df)
    X_ap_crash, names_ap_crash = features_ap_crash(df_full)  # full for crash
    y_lat = df["throughput_fps"].values
    y_ap = df["ap50"].values
    y_crash = df_full["ap_crashed"].values

    # 5-fold cv-like: train/test split 80/20
    rng = np.random.default_rng(42)
    n = len(df)
    perm = rng.permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = perm[:split], perm[split:]

    # 单独 split for crash (full oracle, different size)
    n_full = len(df_full)
    perm_full = rng.permutation(n_full)
    split_full = int(n_full * 0.8)
    train_idx_full, test_idx_full = perm_full[:split_full], perm_full[split_full:]

    results = {}

    # f_lat (full features)
    r2_lat, mae_lat, m_lat = train_eval_lgb(
        X_lat[train_idx], y_lat[train_idx], X_lat[test_idx], y_lat[test_idx])
    results["f_lat"] = {
        "R²": float(r2_lat), "MAE": float(mae_lat),
        "features": names_lat,
        "importance": dict(zip(names_lat, m_lat.feature_importances_.tolist())),
    }
    print(f"  f_lat       R²={r2_lat:.3f}  MAE={mae_lat:.2f}")

    # f_AP_full (full features)
    r2_ap_full, mae_ap_full, m_ap_full = train_eval_lgb(
        X_ap_full[train_idx], y_ap[train_idx], X_ap_full[test_idx], y_ap[test_idx])
    results["f_AP_full"] = {
        "R²": float(r2_ap_full), "MAE": float(mae_ap_full),
        "features": names_ap_full,
        "importance": dict(zip(names_ap_full, m_ap_full.feature_importances_.tolist())),
    }
    print(f"  f_AP_full   R²={r2_ap_full:.3f}  MAE={mae_ap_full:.3f}")

    # f_AP_baseline (no Q, no D)
    r2_ap_base, mae_ap_base, m_ap_base = train_eval_lgb(
        X_ap_base[train_idx], y_ap[train_idx], X_ap_base[test_idx], y_ap[test_idx])
    results["f_AP_baseline"] = {
        "R²": float(r2_ap_base), "MAE": float(mae_ap_base),
        "features": names_ap_base,
        "importance": dict(zip(names_ap_base, m_ap_base.feature_importances_.tolist())),
    }
    print(f"  f_AP_base   R²={r2_ap_base:.3f}  MAE={mae_ap_base:.3f}  (no Q, no D)")

    # f_AP_crash (Q + ckpt_source, binary) — uses full oracle
    auc_crash, _, m_crash = train_eval_lgb(
        X_ap_crash[train_idx_full], y_crash[train_idx_full],
        X_ap_crash[test_idx_full], y_crash[test_idx_full], task="binary")
    results["f_AP_crash"] = {
        "AUC": float(auc_crash),
        "features": names_ap_crash,
        "importance": dict(zip(names_ap_crash, m_crash.feature_importances_.tolist())),
    }
    print(f"  f_AP_crash  AUC={auc_crash:.3f}")

    # Drop-D ablation: f_AP_full minus D features
    # Manually drop D_idx + ws_gb columns
    drop_d_cols = [i for i, n in enumerate(names_ap_full) if n not in ("D_idx", "ws_gb")]
    X_no_D = X_ap_full[:, drop_d_cols]
    r2_no_D, mae_no_D, _ = train_eval_lgb(
        X_no_D[train_idx], y_ap[train_idx], X_no_D[test_idx], y_ap[test_idx])
    results["f_AP_no_D_ablation"] = {
        "R²": float(r2_no_D), "MAE": float(mae_no_D),
        "delta_R²_vs_full": float(r2_ap_full - r2_no_D),
        "claim": "if delta ≈ 0, AP truly invariant to D",
    }
    print(f"  f_AP_no_D   R²={r2_no_D:.3f}  ΔR²={r2_ap_full-r2_no_D:+.3f}")

    return results


def plot_stage_1c(results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_idx, (key, title) in enumerate([
        ("f_lat", "f_lat — throughput_fps (full features)"),
        ("f_AP_full", "f_AP_full — ap50 (full features)"),
        ("f_AP_crash", "f_AP_crash — AP<0.4 binary"),
    ]):
        ax = axes[ax_idx]
        d = results[key]
        items = sorted(d["importance"].items(), key=lambda x: -x[1])
        names_, vals = zip(*items)
        ax.barh(range(len(names_)), vals)
        ax.set_yticks(range(len(names_)))
        ax.set_yticklabels(names_, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("LGB feature importance (gain)", fontsize=10)
        metric = "R²" if "R²" in d else "AUC"
        score = d.get("R²", d.get("AUC"))
        ax.set_title(f"{title}\n{metric}={score:.3f}", fontsize=10)
        ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


# === Main ===

def main():
    # Healthy oracle (排除 Q_int8_ent) for lat / AP regression
    df = load_oracle(exclude_entropy=True)
    # Full oracle (含 entropy) for f_AP_crash binary classifier
    df_full = load_oracle(exclude_entropy=False)
    print(f"Healthy oracle: {len(df)} anchor, {df['triplet'].nunique()} triplets x "
          f"{df['q_tag'].nunique()} Q x {df['d_tactic'].nunique()} D")
    print(f"  ap50 range: [{df['ap50'].min():.3f}, {df['ap50'].max():.3f}]")
    print(f"  fps range: [{df['throughput_fps'].min():.1f}, {df['throughput_fps'].max():.1f}]")
    print(f"Full oracle (含 crash): {len(df_full)} anchor, crashed: {df_full['ap_crashed'].sum()}")

    # Stage 1a + 1b: healthy oracle
    res_1a = stage_1a(df)
    res_1a.to_csv(OUT_DIR / "stage_1a_results.csv", index=False)
    plot_stage_1a(res_1a, OUT_DIR / "k_r2_curves.png")

    active_lat, active_ap, static_lat, static_ap = stage_1b(df)
    plot_stage_1b(active_lat, active_ap, static_lat, static_ap,
                  OUT_DIR / "active_vs_static.png")

    # Stage 1c: f_AP_crash 需 full oracle; f_lat / f_AP_baseline / f_AP_full 用 healthy
    res_1c = stage_1c_split(df, df_full)
    plot_stage_1c(res_1c, OUT_DIR / "per_dim_shapley.png")

    # summary.json
    summary = {
        "oracle_size": len(df),
        "stage_1a_pivot": (res_1a.groupby(["sampler", "K"])[["r2_lat", "r2_ap"]]
                          .mean().round(4).reset_index().to_dict(orient="records")),
        "stage_1b": {
            "active_lat": [{"K": k, "r2": r} for k, r in active_lat],
            "active_ap": [{"K": k, "r2": r} for k, r in active_ap],
            "static_lat": [{"K": k, "r2": r} for k, r in static_lat],
            "static_ap": [{"K": k, "r2": r} for k, r in static_ap],
        },
        "stage_1c": res_1c,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nAll done. Outputs at {OUT_DIR}")


if __name__ == "__main__":
    main()
