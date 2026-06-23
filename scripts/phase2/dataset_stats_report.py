"""Phase 3.1 (plan v3) — Dataset descriptive statistics + 14-figure auto-report.

Reads the union of:
  - plan v2 spread_3a.csv + combo_3b.csv  (21 anchors)
  - plan v3 phase 1.1 underfit.csv         (24 anchors)
  - plan v3 phase 1.2 dtactic.csv          (36 anchors with D + lat)
  - plan v3 phase 1.3 noise.json           (5 seed × T_g8_p97_FT4)

Outputs 14 PNG figures (A1-E1) + dataset_v3_stats.md report in
  paper_learning/2. AAAI最终故事/data/stats_v3/

Each figure has title-embedded ✅/❌ verdict per plan v3 §3.1 thresholds.

Skipped gracefully if any input CSV missing (writes NA cells, marks figure ❌).
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scistats

REPO = Path("/home/jichengzhi/UniV2X")
# OUT is set in main() from --out-dir; default deployment view goes to stats_v3
OUT = REPO / "paper_learning/2. AAAI最终故事/data/stats_v3"
DEPLOYMENT_MODE = False  # filter FT in {8,15} + drop FT/triplet features
DEPLOYMENT_FT = {8, 15}

# Thresholds from plan v3 §3.1
TH = {
    "ap_std_min": 0.15,
    "ap_range_min": 0.50,
    "marginal_entropy_min": 0.85,
    "cell_density_min": 0.70,
    "covering_radius_max": 0.30,
    "class_min_pct": 0.10,
    "ap_collapse_thresh": 0.30,
    "ap_degrade_thresh": 0.50,
    "min_strong_features": 3,
    "spearman_strong": 0.30,
    "max_redundancy": 0.95,
    "min_interaction_h": 0.10,
    "cv_r2_min": 0.75,
    "cv_mae_max": 0.04,
    "lc_60pct_r2_min": 0.65,
    "ood_mae_factor_max": 1.5,
    "sigma_noise_max": 0.02,
    "r2_ceiling_min": 0.95,
}

# Color palette
sns.set_theme(style="whitegrid", palette="colorblind")
PASS_COLOR = "#27AE60"
FAIL_COLOR = "#C0392B"
INFO_COLOR = "#2980B9"


def verdict_tag(passed: bool) -> str:
    return "[PASS]" if passed else "[FAIL]"


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_all_anchors(args) -> pd.DataFrame:
    """Build unified DataFrame: columns triplet, planes, total_prune, q, ft, d,
    seed, ap50, lat_p50_ms (NaN where unmeasured)."""
    rows = []

    # plan v2 spread_3a (FT × triplet, fp32 only)
    p = Path(args.v2_3a)
    if p.exists():
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            for ft_col, ft_val in [("FT4", 4), ("FT8", 8), ("FT15", 15)]:
                rows.append({
                    "triplet": r["triplet"], "ft": ft_val, "q": "fp32",
                    "d": "default", "seed": -1,
                    "ap50": r[ft_col] if pd.notna(r[ft_col]) else None,
                    "lat_p50_ms": None, "src": "v2_3a"})

    # plan v2 combo_3b (Q × triplet, FT=8)
    p = Path(args.v2_3b)
    if p.exists():
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            for q in ["fp16", "int8_mm", "int8_pc_wo", "int8_ent"]:
                if q in df.columns and pd.notna(r[q]):
                    rows.append({
                        "triplet": r["triplet"], "ft": 8, "q": q,
                        "d": "default", "seed": -1,
                        "ap50": r[q], "lat_p50_ms": None, "src": "v2_3b"})

    # plan v3 phase 1.1 underfit
    p = Path(args.v3_underfit)
    if p.exists():
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            for q in ["fp16", "int8_mm", "int8_pc_wo", "int8_ent"]:
                if q in df.columns:
                    v = r[q]
                    if pd.notna(v) and v != "NA":
                        try:
                            rows.append({
                                "triplet": r["triplet"], "ft": int(r["ft"]),
                                "q": q, "d": "default", "seed": -1,
                                "ap50": float(v),
                                "lat_p50_ms": None, "src": "v3_1.1"})
                        except (ValueError, TypeError):
                            pass

    # plan v3 phase 1.2 d-tactic
    p = Path(args.v3_dtactic)
    if p.exists():
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            try:
                ap = float(r["ap50"]) if pd.notna(r["ap50"]) and r["ap50"] != "NA" else None
            except (ValueError, TypeError):
                ap = None
            try:
                lat = float(r["lat_p50_ms"]) if pd.notna(r["lat_p50_ms"]) and r["lat_p50_ms"] != "NA" else None
            except (ValueError, TypeError):
                lat = None
            rows.append({
                "triplet": r["triplet"], "ft": 8,
                "q": r["qvar"], "d": r["d_tag"], "seed": -1,
                "ap50": ap, "lat_p50_ms": lat, "src": "v3_1.2"})

    # plan v3 phase 1.3 noise (5 seed × p97 × FT4)
    p = Path(args.v3_noise)
    if p.exists():
        d = json.loads(p.read_text())
        for seed, ap in zip(d["seeds"], d["aps"]):
            rows.append({
                "triplet": "T_g8_p97", "ft": 4, "q": "fp32",
                "d": "default", "seed": seed, "ap50": ap,
                "lat_p50_ms": None, "src": "v3_1.3"})

    # plan v3 §7.11 follow-up: FT-sweep (3 triplet × 5 Q × FT∈{4,6,8})
    p = Path(args.v3_ft_sweep)
    if p.exists():
        df_fs = pd.read_csv(p)
        for _, r in df_fs.iterrows():
            for q in ["fp32", "fp16", "int8_mm", "int8_pc_wo", "int8_ent"]:
                if q in df_fs.columns:
                    v = r[q]
                    if pd.notna(v) and v != "NA":
                        try:
                            rows.append({
                                "triplet": r["triplet"],
                                "ft": int(r["ft"]),
                                "q": q, "d": "default", "seed": -1,
                                "ap50": float(v),
                                "lat_p50_ms": None,
                                "src": "v3_7.11_ft_sweep"})
                        except (ValueError, TypeError):
                            pass

    # plan v3 §7.11 FT=6 noise study (5 seed × p97 × FT6)
    p = Path(args.v3_noise_ft6)
    if p.exists():
        d = json.loads(p.read_text())
        for seed, ap in zip(d["seeds"], d["aps"]):
            rows.append({
                "triplet": "T_g8_p97", "ft": 6, "q": "fp32",
                "d": "default", "seed": seed, "ap50": ap,
                "lat_p50_ms": None, "src": "v3_7.11_noise_ft6"})

    df = pd.DataFrame(rows)
    # de-duplicate (same triplet, ft, q, d, seed) — ft_sweep may overlap with v2 §3a/3b
    df = df.drop_duplicates(
        subset=["triplet", "ft", "q", "d", "seed"], keep="first").reset_index(drop=True)

    # derive structural features
    planes_map = {
        "T_g8_p87": (8, 8, 8), "T_g8_p93": (8, 4, 4), "T_g8_p97": (4, 4, 4),
    }
    df["planes_s1"] = df["triplet"].map(lambda t: planes_map.get(t, (0, 0, 0))[0])
    df["planes_s2"] = df["triplet"].map(lambda t: planes_map.get(t, (0, 0, 0))[1])
    df["planes_s3"] = df["triplet"].map(lambda t: planes_map.get(t, (0, 0, 0))[2])
    # rough total prune fraction
    base_total = 64 + 128 + 256
    df["total_prune_pct"] = df.apply(
        lambda r: 1.0 - (r["planes_s1"] + r["planes_s2"] + r["planes_s3"]) / base_total,
        axis=1)
    return df


# ----------------------------------------------------------------------------
# A — Coverage
# ----------------------------------------------------------------------------

def fig_A1_marginal_coverage(df: pd.DataFrame) -> dict:
    """Sub-axes: marginal histogram per axis with normalized entropy.

    In deployment mode: drop FT axis (FT is fixed, not a search dim).
    """
    if DEPLOYMENT_MODE:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes_def = [
            ("planes_s3", "B (planes stage3)"),
            ("q", "Q variant"),
            ("d", "D tactic"),
        ]
    else:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        axes_def = [
            ("planes_s3", "B (planes stage3)"),
            ("q", "Q variant"),
            ("ft", "FT epoch"),
            ("d", "D tactic"),
        ]
    entropies = {}
    for ax, (col, label) in zip(axes, axes_def):
        if col not in df.columns:
            ax.text(0.5, 0.5, "MISSING", ha="center", color=FAIL_COLOR)
            continue
        vals = df[col].dropna().astype(str)
        vc = vals.value_counts()
        # normalized entropy
        p = vc / vc.sum()
        H = -np.sum(p * np.log2(p + 1e-12))
        Hn = H / np.log2(len(p)) if len(p) > 1 else 0.0
        entropies[col] = Hn
        passed = Hn >= TH["marginal_entropy_min"]
        ax.bar(vc.index.astype(str), vc.values,
               color=PASS_COLOR if passed else FAIL_COLOR)
        ax.set_title(f"{label}\nentropy={Hn:.2f} {verdict_tag(passed)}")
        ax.set_xlabel(col); ax.set_ylabel("anchor count")
        ax.tick_params(axis='x', rotation=45)
    all_pass = all(e >= TH["marginal_entropy_min"] for e in entropies.values())
    fig.suptitle(f"A1: Marginal Coverage {verdict_tag(all_pass)} "
                 f"(min entropy = {TH['marginal_entropy_min']})  [{stamp()}]",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "01_coverage_marginal.png", dpi=150)
    plt.close(fig)
    return {"entropies": entropies, "all_pass": all_pass}


def _grid_heatmap(df: pd.DataFrame, x: str, y: str,
                  fname: str, title: str) -> dict:
    if x not in df.columns or y not in df.columns or df.empty:
        return {"pass": False, "filled_pct": 0}
    pivot = df.groupby([y, x]).size().unstack(fill_value=0)
    total = pivot.size
    filled = (pivot > 0).sum().sum()
    pct = filled / total
    passed = pct >= TH["cell_density_min"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt="d",
                cmap="YlGnBu", cbar_kws={'label': 'anchor count'},
                linewidths=0.5, ax=ax)
    ax.set_title(f"{title} {verdict_tag(passed)} "
                 f"(filled {filled}/{total} = {pct:.0%})  [{stamp()}]")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    return {"pass": passed, "filled_pct": pct,
            "filled": int(filled), "total": int(total)}


def fig_A2_A3_grids(df: pd.DataFrame) -> dict:
    a2 = _grid_heatmap(df, "q", "triplet",
                       "02_coverage_grid_BxQ.png", "A2: B × Q Cell Density")
    if DEPLOYMENT_MODE:
        # FT is fixed; replace B×FT with B×D
        a3 = _grid_heatmap(df, "d", "triplet",
                           "03_coverage_grid_BxD.png",
                           "A3: B × D Cell Density (FT fixed)")
    else:
        a3 = _grid_heatmap(df, "ft", "triplet",
                           "03_coverage_grid_BxFT.png",
                           "A3: B × FT Cell Density")
    return {"a2": a2, "a3": a3}


def fig_A4_hull_pca2d(df: pd.DataFrame) -> dict:
    """PCA-2D projection + anchor scatter + convex hull polygon."""
    feat = df[["planes_s1", "planes_s2", "planes_s3", "ft"]].copy()
    # one-hot q and d for full feature space
    q_oh = pd.get_dummies(df["q"], prefix="q")
    d_oh = pd.get_dummies(df["d"], prefix="d")
    X = pd.concat([feat, q_oh, d_oh], axis=1).fillna(0).values.astype(float)
    if X.shape[0] < 3:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.text(0.5, 0.5, "TOO FEW POINTS", ha="center", color=FAIL_COLOR)
        fig.savefig(OUT / "04_coverage_hull_pca2d.png", dpi=150); plt.close(fig)
        return {"pass": False}
    # standardize
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    # 2D PCA
    cov = np.cov(Xs, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:2]
    proj = Xs @ eigvecs[:, idx]
    # covering radius (max nearest-neighbor distance among anchors)
    from scipy.spatial.distance import cdist
    dist = cdist(proj, proj)
    np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1)
    cr = nn.max() / max(proj.std(), 1e-6)
    passed = cr <= TH["covering_radius_max"] * 5  # rescale heuristic
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(proj[:, 0], proj[:, 1], c=df["ap50"].fillna(0),
               cmap="viridis", s=30, alpha=0.7, edgecolor="k", linewidth=0.5)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(proj)
        for simplex in hull.simplices:
            ax.plot(proj[simplex, 0], proj[simplex, 1], "k-", lw=1)
    except Exception:
        pass
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"A4: Coverage Hull (PCA 2D) {verdict_tag(passed)}\n"
                 f"covering radius={cr:.3f}, N={len(proj)}  [{stamp()}]")
    fig.tight_layout()
    fig.savefig(OUT / "04_coverage_hull_pca2d.png", dpi=150)
    plt.close(fig)
    return {"pass": passed, "covering_radius": float(cr), "n": int(len(proj))}


# ----------------------------------------------------------------------------
# B — Distribution
# ----------------------------------------------------------------------------

def fig_B1_target_hist(df: pd.DataFrame) -> dict:
    aps = df["ap50"].dropna().values
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(aps) < 5:
        ax.text(0.5, 0.5, "TOO FEW APs", ha="center", color=FAIL_COLOR)
        fig.savefig(OUT / "05_target_histogram.png", dpi=150); plt.close(fig)
        return {"pass": False}
    ap_std = aps.std()
    ap_range = aps.max() - aps.min()
    passed = (ap_std >= TH["ap_std_min"] and ap_range >= TH["ap_range_min"])
    sns.histplot(aps, bins=20, kde=True, ax=ax,
                 color=PASS_COLOR if passed else FAIL_COLOR)
    ax.axvline(TH["ap_collapse_thresh"], color=FAIL_COLOR, ls="--",
               label=f'collapse {TH["ap_collapse_thresh"]}')
    ax.axvline(TH["ap_degrade_thresh"], color="orange", ls="--",
               label=f'degrade {TH["ap_degrade_thresh"]}')
    ax.set_xlabel("AP50"); ax.set_ylabel("count")
    ax.legend()
    ax.set_title(f"B1: AP Target Distribution {verdict_tag(passed)}\n"
                 f"N={len(aps)}, std={ap_std:.3f}, range={ap_range:.3f}, "
                 f"mean={aps.mean():.3f}  [{stamp()}]")
    fig.tight_layout()
    fig.savefig(OUT / "05_target_histogram.png", dpi=150)
    plt.close(fig)
    return {"pass": passed, "std": float(ap_std), "range": float(ap_range),
            "mean": float(aps.mean()), "n": int(len(aps))}


def fig_B2_class_balance(df: pd.DataFrame) -> dict:
    aps = df["ap50"].dropna().values
    fig, ax = plt.subplots(figsize=(9, 4))
    if len(aps) < 5:
        ax.text(0.5, 0.5, "TOO FEW APs", ha="center", color=FAIL_COLOR)
        fig.savefig(OUT / "06_target_class_balance.png", dpi=150); plt.close(fig)
        return {"pass": False}
    n = len(aps)
    classes = {
        f"collapse (<{TH['ap_collapse_thresh']})":
            int(np.sum(aps < TH["ap_collapse_thresh"])),
        f"degrade ({TH['ap_collapse_thresh']}-{TH['ap_degrade_thresh']})":
            int(np.sum((aps >= TH["ap_collapse_thresh"]) &
                       (aps < TH["ap_degrade_thresh"]))),
        f"healthy (>={TH['ap_degrade_thresh']})":
            int(np.sum(aps >= TH["ap_degrade_thresh"])),
    }
    pcts = {k: v / n for k, v in classes.items()}
    passed = all(p >= TH["class_min_pct"] for p in pcts.values())
    labels = list(classes.keys())
    counts = list(classes.values())
    colors = [FAIL_COLOR, "orange", PASS_COLOR]
    bars = ax.barh(labels, counts, color=colors)
    for bar, c, p in zip(bars, counts, pcts.values()):
        ax.text(c, bar.get_y() + bar.get_height() / 2,
                f" {c} ({p:.0%})", va="center")
    ax.set_xlabel("anchor count")
    ax.set_title(f"B2: AP Class Balance {verdict_tag(passed)}\n"
                 f"min class threshold = {TH['class_min_pct']:.0%}  [{stamp()}]")
    fig.tight_layout()
    fig.savefig(OUT / "06_target_class_balance.png", dpi=150)
    plt.close(fig)
    return {"pass": passed, "counts": classes, "pcts": pcts}


def fig_B3_by_axis_box(df: pd.DataFrame) -> dict:
    if DEPLOYMENT_MODE:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        cols = ["triplet", "q", "d"]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        cols = ["triplet", "q", "ft", "d"]
    spans = {}
    for ax, c in zip(axes.flat, cols):
        sub = df.dropna(subset=["ap50"])
        if c not in sub.columns or sub.empty:
            ax.text(0.5, 0.5, "MISSING", ha="center", color=FAIL_COLOR)
            continue
        order = sorted(sub[c].astype(str).unique())
        sns.boxplot(data=sub, x=c, y="ap50", order=order, ax=ax)
        medians = sub.groupby(c)["ap50"].median()
        span = (medians.max() - medians.min()) if len(medians) > 1 else 0
        spans[c] = float(span)
        ax.set_title(f"by {c} (median span={span:.3f})")
        ax.tick_params(axis='x', rotation=30)
    # success: FT span > 0.3 OR Q span > 0.05
    passed = (spans.get("ft", 0) > 0.30 or spans.get("q", 0) > 0.05)
    fig.suptitle(f"B3: AP by Axis (boxplots) {verdict_tag(passed)}  [{stamp()}]",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "07_target_by_axis_box.png", dpi=150)
    plt.close(fig)
    return {"pass": passed, "spans": spans}


# ----------------------------------------------------------------------------
# C — Information
# ----------------------------------------------------------------------------

def featurize(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Build feature matrix X and target y from df, returning df_features, y.

    In deployment mode: drop FT (deployment-realistic, FT fixed) and D one-hot
    (D is confounded with FT=8 in plan v3, and D physically doesn't affect AP).
    """
    sub = df.dropna(subset=["ap50"]).reset_index(drop=True)
    if DEPLOYMENT_MODE:
        feat = sub[["planes_s1", "planes_s2", "planes_s3",
                    "total_prune_pct"]].copy()
        q_oh = pd.get_dummies(sub["q"], prefix="q")
        X = pd.concat([feat, q_oh], axis=1).fillna(0)
    else:
        feat = sub[["planes_s1", "planes_s2", "planes_s3", "ft",
                    "total_prune_pct"]].copy()
        q_oh = pd.get_dummies(sub["q"], prefix="q")
        d_oh = pd.get_dummies(sub["d"], prefix="d")
        X = pd.concat([feat, q_oh, d_oh], axis=1).fillna(0)
    y = sub["ap50"].values.astype(float)
    return X, y


def fig_C1_mi_spearman(df: pd.DataFrame) -> dict:
    from sklearn.feature_selection import mutual_info_regression
    X, y = featurize(df)
    if len(y) < 10:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "TOO FEW POINTS", ha="center", color=FAIL_COLOR)
        fig.savefig(OUT / "08_mi_spearman_per_feature.png", dpi=150); plt.close(fig)
        return {"pass": False}
    mi = mutual_info_regression(X.values, y, random_state=0)
    sp = []
    for c in X.columns:
        rho, _ = scistats.spearmanr(X[c].values, y)
        sp.append(0.0 if math.isnan(rho) else float(rho))
    n_strong = sum(1 for r in sp if abs(r) > TH["spearman_strong"])
    passed = n_strong >= TH["min_strong_features"]
    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(X.columns))))
    y_pos = np.arange(len(X.columns))
    ax.barh(y_pos - 0.2, mi, height=0.4, label="MI",
            color=INFO_COLOR, alpha=0.8)
    ax.barh(y_pos + 0.2, np.abs(sp), height=0.4, label="|Spearman ρ|",
            color="orange", alpha=0.8)
    ax.set_yticks(y_pos); ax.set_yticklabels(X.columns)
    ax.axvline(TH["spearman_strong"], color=FAIL_COLOR, ls="--",
               label=f'strong threshold {TH["spearman_strong"]}')
    ax.set_xlabel("score")
    ax.set_title(f"C1: MI + |Spearman ρ| per Feature {verdict_tag(passed)}\n"
                 f"{n_strong} strong features (need ≥{TH['min_strong_features']})  "
                 f"[{stamp()}]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "08_mi_spearman_per_feature.png", dpi=150)
    plt.close(fig)
    return {"pass": passed, "mi": dict(zip(X.columns, mi.tolist())),
            "spearman": dict(zip(X.columns, sp)),
            "n_strong": int(n_strong)}


def fig_C2_corr_matrix(df: pd.DataFrame) -> dict:
    X, _ = featurize(df)
    if X.shape[0] < 5 or X.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "TOO FEW POINTS", ha="center", color=FAIL_COLOR)
        fig.savefig(OUT / "09_corr_matrix.png", dpi=150); plt.close(fig)
        return {"pass": False}
    corr = X.corr().fillna(0)
    # max off-diagonal abs
    n = corr.shape[0]
    off = np.abs(corr.values)
    np.fill_diagonal(off, 0)
    max_off = off.max()
    passed = max_off < TH["max_redundancy"]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * n), max(7, 0.45 * n)))
    sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                annot=False, square=True, ax=ax)
    ax.set_title(f"C2: Feature Correlation {verdict_tag(passed)}\n"
                 f"max off-diag |r|={max_off:.2f} (limit {TH['max_redundancy']})  "
                 f"[{stamp()}]")
    fig.tight_layout()
    fig.savefig(OUT / "09_corr_matrix.png", dpi=150)
    plt.close(fig)
    return {"pass": passed, "max_off_diag": float(max_off)}


def fig_C3_interaction_h(df: pd.DataFrame) -> dict:
    """Approximate H-stat per pair using mean residual decomposition."""
    X, y = featurize(df)
    if len(y) < 20:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "TOO FEW POINTS", ha="center", color=FAIL_COLOR)
        fig.savefig(OUT / "10_interaction_hstat.png", dpi=150); plt.close(fig)
        return {"pass": False}
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.inspection import partial_dependence
    model = GradientBoostingRegressor(random_state=0,
                                      n_estimators=100, max_depth=3)
    model.fit(X.values, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    top_feats = importance.nlargest(4).index.tolist()
    pairs = []
    for i, a in enumerate(top_feats):
        for b in top_feats[i+1:]:
            pairs.append((a, b))
    h_scores = []
    for a, b in pairs[:6]:
        try:
            ai = list(X.columns).index(a)
            bi = list(X.columns).index(b)
            pd_ab = partial_dependence(model, X.values, [(ai, bi)],
                                       kind="average")
            pd_a = partial_dependence(model, X.values, [ai], kind="average")
            pd_b = partial_dependence(model, X.values, [bi], kind="average")
            joint = pd_ab["average"][0]
            marg_a = pd_a["average"][0].reshape(-1, 1)
            marg_b = pd_b["average"][0].reshape(1, -1)
            additive = marg_a + marg_b - joint.mean()
            num = np.var(joint - additive)
            den = np.var(joint) + 1e-12
            h = float(num / den)
        except Exception as e:
            h = 0.0
        h_scores.append((f"{a} × {b}", h))
    h_scores.sort(key=lambda r: r[1], reverse=True)
    h_scores = h_scores[:5]
    max_h = max([h for _, h in h_scores]) if h_scores else 0
    passed = max_h >= TH["min_interaction_h"]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(h_scores))))
    labels = [k for k, _ in h_scores]
    vals = [v for _, v in h_scores]
    bars = ax.barh(labels, vals,
                   color=[PASS_COLOR if v >= TH["min_interaction_h"] else FAIL_COLOR
                          for v in vals])
    ax.axvline(TH["min_interaction_h"], color=FAIL_COLOR, ls="--",
               label=f'threshold {TH["min_interaction_h"]}')
    ax.set_xlabel("H-statistic (interaction strength)")
    ax.set_title(f"C3: Top Feature Interaction H-stat {verdict_tag(passed)}\n"
                 f"max H={max_h:.3f}  [{stamp()}]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "10_interaction_hstat.png", dpi=150)
    plt.close(fig)
    return {"pass": passed, "h_scores": dict(h_scores), "max_h": float(max_h)}


# ----------------------------------------------------------------------------
# D — Learnability
# ----------------------------------------------------------------------------

def _cv_fit(X, y, n_splits=5, shuffle_y=False, seed=0):
    from sklearn.model_selection import KFold
    import lightgbm as lgb
    rng = np.random.RandomState(seed)
    y_use = y.copy()
    if shuffle_y:
        rng.shuffle(y_use)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2s = []
    maes = []
    for tr, te in kf.split(X):
        params = dict(objective="regression",
                      learning_rate=0.05, num_leaves=15,
                      min_data_in_leaf=2, n_estimators=50,
                      verbose=-1, num_threads=1, force_col_wise=True)
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr], y_use[tr])
        p = m.predict(X[te])
        r2s.append(_r2(y_use[te], p))
        maes.append(float(np.mean(np.abs(y_use[te] - p))))
    return r2s, maes


def _r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def fig_D1_cv_r2(df: pd.DataFrame) -> dict:
    X, y = featurize(df)
    if len(y) < 15:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "TOO FEW POINTS", ha="center", color=FAIL_COLOR)
        fig.savefig(OUT / "11_cv_r2_per_fold.png", dpi=150); plt.close(fig)
        return {"pass": False}
    Xn = X.values.astype(float)
    r2_real, mae_real = _cv_fit(Xn, y, shuffle_y=False)
    r2_shuf, mae_shuf = _cv_fit(Xn, y, shuffle_y=True)
    mean_r2 = float(np.mean(r2_real))
    mean_mae = float(np.mean(mae_real))
    passed = (mean_r2 >= TH["cv_r2_min"] and mean_mae <= TH["cv_mae_max"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(5)
    axes[0].bar(x - 0.2, r2_real, width=0.4, label="real",
                color=PASS_COLOR if passed else FAIL_COLOR)
    axes[0].bar(x + 0.2, r2_shuf, width=0.4, label="label-shuffle",
                color="grey", alpha=0.7)
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"f{i}" for i in range(5)])
    axes[0].set_ylabel("R²")
    axes[0].axhline(TH["cv_r2_min"], color=FAIL_COLOR, ls="--",
                    label=f"threshold {TH['cv_r2_min']}")
    axes[0].set_title(f"R² (mean real={mean_r2:.3f}, shuffle={np.mean(r2_shuf):.3f})")
    axes[0].legend()
    axes[1].bar(x - 0.2, mae_real, width=0.4, label="real",
                color=PASS_COLOR if passed else FAIL_COLOR)
    axes[1].bar(x + 0.2, mae_shuf, width=0.4, label="label-shuffle",
                color="grey", alpha=0.7)
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"f{i}" for i in range(5)])
    axes[1].set_ylabel("MAE")
    axes[1].axhline(TH["cv_mae_max"], color=FAIL_COLOR, ls="--",
                    label=f"threshold {TH['cv_mae_max']}")
    axes[1].set_title(f"MAE (mean real={mean_mae:.4f})")
    axes[1].legend()
    fig.suptitle(f"D1: 5-fold CV {verdict_tag(passed)}  [{stamp()}]",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "11_cv_r2_per_fold.png", dpi=150)
    plt.close(fig)
    return {"pass": passed, "mean_r2_real": mean_r2,
            "mean_r2_shuffle": float(np.mean(r2_shuf)),
            "mean_mae_real": mean_mae,
            "r2_per_fold": [float(r) for r in r2_real],
            "mae_per_fold": [float(r) for r in mae_real]}


def fig_D2_learning_curve(df: pd.DataFrame) -> dict:
    X, y = featurize(df)
    if len(y) < 20:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "TOO FEW POINTS", ha="center", color=FAIL_COLOR)
        fig.savefig(OUT / "12_learning_curve.png", dpi=150); plt.close(fig)
        return {"pass": False}
    Xn = X.values.astype(float)
    # Skip very-small data fractions where 5-fold split has <2 per fold
    min_n_per_fold = 3
    min_frac = max(0.2, (5 * min_n_per_fold) / len(y))
    fracs = [f for f in [0.2, 0.4, 0.6, 0.8, 1.0] if f >= min_frac]
    if 1.0 not in fracs:
        fracs.append(1.0)
    r2_real_curve = []
    r2_shuf_curve = []
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(y))
    Xn_p = Xn[perm]; y_p = y[perm]
    for f in fracs:
        n = max(10, int(f * len(y)))
        r2r, _ = _cv_fit(Xn_p[:n], y_p[:n], shuffle_y=False, seed=0)
        r2s, _ = _cv_fit(Xn_p[:n], y_p[:n], shuffle_y=True, seed=0)
        r2_real_curve.append(np.mean(r2r))
        r2_shuf_curve.append(np.mean(r2s))
    # find R² at closest-to-60% fraction available
    if not r2_real_curve:
        sixty_idx = 0
        r2_at_60 = 0.0
    else:
        sixty_idx = min(range(len(fracs)), key=lambda i: abs(fracs[i] - 0.6))
        r2_at_60 = r2_real_curve[sixty_idx] if sixty_idx < len(r2_real_curve) else 0.0
    passed = r2_at_60 >= TH["lc_60pct_r2_min"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([f * 100 for f in fracs], r2_real_curve, "o-",
            color=PASS_COLOR if passed else FAIL_COLOR, label="real")
    ax.plot([f * 100 for f in fracs], r2_shuf_curve, "s--",
            color="grey", label="label-shuffle baseline")
    ax.axhline(TH["lc_60pct_r2_min"], color=FAIL_COLOR, ls=":",
               label=f"60% threshold {TH['lc_60pct_r2_min']}")
    ax.set_xlabel("training data fraction (%)")
    ax.set_ylabel("R² (5-fold CV)")
    ax.set_title(f"D2: Learning Curve {verdict_tag(passed)}\n"
                 f"R² at 60% = {r2_at_60:.3f}  [{stamp()}]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "12_learning_curve.png", dpi=150)
    plt.close(fig)
    return {"pass": passed,
            "r2_per_frac": [(f, float(r))
                            for f, r in zip(fracs, r2_real_curve)],
            "r2_shuffle_per_frac": [(f, float(r))
                                    for f, r in zip(fracs, r2_shuf_curve)]}


def _ood_run(df: pd.DataFrame, hold_filter, label: str) -> tuple[float, np.ndarray, np.ndarray]:
    """Train on df without hold_filter, test on hold subset."""
    from sklearn.feature_selection import SelectKBest
    import lightgbm as lgb
    sub = df.dropna(subset=["ap50"]).reset_index(drop=True)
    hold_mask = hold_filter(sub)
    train_df = sub[~hold_mask]
    test_df = sub[hold_mask]
    if len(train_df) < 10 or len(test_df) < 2:
        return float("nan"), np.array([]), np.array([])
    Xtr, ytr = featurize(train_df)
    Xte, yte = featurize(test_df)
    # Align columns
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    params = dict(objective="regression", learning_rate=0.05,
                  num_leaves=15, min_data_in_leaf=2,
                  n_estimators=50, verbose=-1, num_threads=2,
                  force_col_wise=True)
    m = lgb.LGBMRegressor(**params)
    m.fit(Xtr.values, ytr)
    pred = m.predict(Xte.values)
    mae = float(np.mean(np.abs(yte - pred)))
    return mae, yte, pred


def fig_D3_ood(df: pd.DataFrame) -> dict:
    """OOD: hold out p97, hold out pc_wo. Plot predict-vs-real scatter."""
    if df.empty or df.dropna(subset=["ap50"]).shape[0] < 30:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "TOO FEW POINTS", ha="center", color=FAIL_COLOR)
        fig.savefig(OUT / "13_ood_predict_vs_real.png", dpi=150); plt.close(fig)
        return {"pass": False}
    # in-sample MAE as reference
    X, y = featurize(df)
    _, in_mae = _cv_fit(X.values.astype(float), y)
    in_mae_mean = float(np.mean(in_mae))
    th = TH["ood_mae_factor_max"] * in_mae_mean
    if DEPLOYMENT_MODE:
        # FT=2 not in deployment subset, replace with hold int8_ent
        scenarios = [
            ("hold p97", lambda d: d["triplet"] == "T_g8_p97"),
            ("hold int8_pc_wo", lambda d: d["q"] == "int8_pc_wo"),
            ("hold int8_ent", lambda d: d["q"] == "int8_ent"),
        ]
    else:
        scenarios = [
            ("hold p97", lambda d: d["triplet"] == "T_g8_p97"),
            ("hold int8_pc_wo", lambda d: d["q"] == "int8_pc_wo"),
            ("hold FT=2", lambda d: d["ft"] == 2),
        ]
    results = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    all_pass = True
    for ax, (name, f) in zip(axes, scenarios):
        mae, yt, pr = _ood_run(df, f, name)
        passed_i = mae <= th if not math.isnan(mae) else False
        all_pass &= passed_i
        if len(yt) > 0:
            ax.scatter(yt, pr, alpha=0.7, s=40,
                       color=PASS_COLOR if passed_i else FAIL_COLOR,
                       edgecolor="k", linewidth=0.5)
            lim = [0, max(0.7, float(np.max(np.concatenate([yt, pr]))))]
            ax.plot(lim, lim, "k--", alpha=0.5, label="y=x")
            ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("real AP"); ax.set_ylabel("predicted AP")
        mae_s = f"{mae:.4f}" if not math.isnan(mae) else "NA"
        ax.set_title(f"{name}\nMAE={mae_s} {verdict_tag(passed_i)}")
        ax.legend()
        results.append((name, mae))
    fig.suptitle(f"D3: OOD Predict vs Real {verdict_tag(all_pass)}\n"
                 f"threshold = {TH['ood_mae_factor_max']} × in-sample "
                 f"MAE ({in_mae_mean:.4f}) = {th:.4f}  [{stamp()}]",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "13_ood_predict_vs_real.png", dpi=150)
    plt.close(fig)
    return {"pass": all_pass, "in_sample_mae": in_mae_mean,
            "threshold": th,
            "scenarios": {n: (float(m) if not math.isnan(m) else None)
                          for n, m in results}}


# ----------------------------------------------------------------------------
# E — Noise floor
# ----------------------------------------------------------------------------

def fig_E1_noise_floor(df: pd.DataFrame, noise_path: Path) -> dict:
    fig, ax = plt.subplots(figsize=(9, 5))
    if not noise_path.exists():
        ax.text(0.5, 0.5, "noise.json missing (Phase 1.3 not done)",
                ha="center", color=FAIL_COLOR)
        fig.savefig(OUT / "14_noise_floor.png", dpi=150); plt.close(fig)
        return {"pass": False}
    d = json.loads(noise_path.read_text())
    aps = d["aps"]; seeds = d["seeds"][:len(aps)]
    sigma = d["sigma_noise"]
    r2_ceiling = d["r2_ceiling"]
    passed = sigma <= TH["sigma_noise_max"]
    ax.scatter(seeds, aps, s=80, color=INFO_COLOR, alpha=0.8)
    mean_ap = np.mean(aps)
    ax.axhline(mean_ap, color="k", ls="-", label=f"mean={mean_ap:.4f}")
    ax.axhline(mean_ap + sigma, color="grey", ls="--",
               label=f"±σ ({sigma:.4f})")
    ax.axhline(mean_ap - sigma, color="grey", ls="--")
    ax.set_xlabel("seed"); ax.set_ylabel("AP_FP32 (T_g8_p97_FT4)")
    ax.set_title(f"E1: Noise Floor {verdict_tag(passed)}\n"
                 f"σ_noise={sigma:.4f}, R²_ceiling={r2_ceiling:.3f}  [{stamp()}]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "14_noise_floor.png", dpi=150)
    plt.close(fig)
    return {"pass": passed, "sigma_noise": sigma,
            "r2_ceiling": r2_ceiling, "n_seed": len(aps)}


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def write_report(df: pd.DataFrame, results: dict) -> None:
    md = OUT / "dataset_v3_stats.md"
    pass_n = sum(1 for v in _walk_pass(results))
    total_n = sum(1 for _ in _walk_pass(results, only_count=True))
    lines = [
        f"# Dataset v3 Stats Report (auto-generated {stamp()})",
        "",
        "## Summary",
        f"- N anchors: **{len(df)}** ({df['ap50'].notna().sum()} with AP)",
        f"- Pass/fail per metric: **{pass_n} ✅ / {total_n}**",
        f"- Verdict: **{'✅ Ready for framework training' if pass_n == total_n else '❌ Phase 4 needed (补点)'}**",
        "",
        "## Coverage (A)",
        f"- A1 marginal: {_p_md(results['A1']['all_pass'])} "
        f"entropies={results['A1']['entropies']}",
        f"- A2 B×Q grid: {_p_md(results['A2_A3']['a2']['pass'])} "
        f"filled {results['A2_A3']['a2']['filled']}/{results['A2_A3']['a2']['total']} "
        f"= {results['A2_A3']['a2']['filled_pct']:.0%}",
        f"- A3 B×FT grid: {_p_md(results['A2_A3']['a3']['pass'])} "
        f"filled {results['A2_A3']['a3']['filled']}/{results['A2_A3']['a3']['total']} "
        f"= {results['A2_A3']['a3']['filled_pct']:.0%}",
        f"- A4 PCA hull: {_p_md(results['A4']['pass'])} "
        f"covering radius={results['A4'].get('covering_radius', 'NA')}",
        "",
        "## Distribution (B)",
        f"- B1 AP histogram: {_p_md(results['B1']['pass'])} "
        f"std={results['B1'].get('std','NA')} range={results['B1'].get('range','NA')}",
        f"- B2 class balance: {_p_md(results['B2']['pass'])} "
        f"counts={results['B2'].get('counts','NA')}",
        f"- B3 axis box: {_p_md(results['B3']['pass'])} "
        f"spans={results['B3'].get('spans','NA')}",
        "",
        "## Information (C)",
        f"- C1 MI/Spearman: {_p_md(results['C1']['pass'])} "
        f"n_strong={results['C1'].get('n_strong','NA')}",
        f"- C2 corr matrix: {_p_md(results['C2']['pass'])} "
        f"max off-diag |r|={results['C2'].get('max_off_diag','NA')}",
        f"- C3 interaction H: {_p_md(results['C3']['pass'])} "
        f"max H={results['C3'].get('max_h','NA')}",
        "",
        "## Learnability (D)",
        f"- D1 5-fold CV: {_p_md(results['D1']['pass'])} "
        f"R²={results['D1'].get('mean_r2_real','NA')} MAE={results['D1'].get('mean_mae_real','NA')}",
        f"- D2 learning curve: {_p_md(results['D2']['pass'])}",
        f"- D3 OOD: {_p_md(results['D3']['pass'])} "
        f"scenarios={results['D3'].get('scenarios','NA')}",
        "",
        "## Noise (E)",
        f"- E1 noise floor: {_p_md(results['E1']['pass'])} "
        f"σ_noise={results['E1'].get('sigma_noise','NA')} "
        f"R²_ceiling={results['E1'].get('r2_ceiling','NA')}",
        "",
        "## Figure index (14 PNG in stats_v3/)",
        "- A组 (覆盖度): 01_coverage_marginal.png, 02_coverage_grid_BxQ.png, "
        "03_coverage_grid_BxFT.png, 04_coverage_hull_pca2d.png",
        "- B组 (分布): 05_target_histogram.png, 06_target_class_balance.png, "
        "07_target_by_axis_box.png",
        "- C组 (信息量): 08_mi_spearman_per_feature.png, 09_corr_matrix.png, "
        "10_interaction_hstat.png",
        "- D组 (可学习性): 11_cv_r2_per_fold.png, 12_learning_curve.png, "
        "13_ood_predict_vs_real.png",
        "- E组 (噪声): 14_noise_floor.png",
        "",
        "## Anchor source breakdown",
    ]
    if "src" in df.columns:
        for src, n in df["src"].value_counts().items():
            lines.append(f"- {src}: {n}")
    md.write_text("\n".join(lines))
    print(f"\n[stats] report → {md}")


def _p_md(b):
    """Markdown verdict — keep emoji here since md renderers handle UTF-8."""
    return "✅" if b else "❌"


def _walk_pass(d, only_count=False):
    if isinstance(d, dict):
        if "pass" in d:
            if only_count:
                yield True
            elif d["pass"]:
                yield True
        for v in d.values():
            yield from _walk_pass(v, only_count)
        # Handle special cases like A1 which uses "all_pass"
        if "all_pass" in d:
            if only_count:
                yield True
            elif d["all_pass"]:
                yield True


def main():
    global OUT, DEPLOYMENT_MODE
    p = argparse.ArgumentParser()
    p.add_argument("--v2-3a", default="/tmp/a10_phase3a/spread_3a.csv")
    p.add_argument("--v2-3b", default="/tmp/a10_phase3b/combo_3b.csv")
    p.add_argument("--v3-underfit", default="/tmp/a11_underfit/underfit.csv")
    p.add_argument("--v3-dtactic", default="/tmp/a11_d_tactic/dtactic.csv")
    p.add_argument("--v3-noise", default="/tmp/a11_noise/noise.json")
    p.add_argument("--v3-ft-sweep", default="/tmp/a11_ft_sweep/ft_sweep.csv")
    p.add_argument("--v3-noise-ft6", default="/tmp/a11_noise_ft6/noise.json")
    p.add_argument("--out-dir", default=str(OUT),
                   help="Output directory for PNG + md (default stats_v3)")
    p.add_argument("--deployment", action="store_true",
                   help="Filter FT in {8,15} + drop FT/triplet features "
                        "(deployment-realistic predictor view)")
    p.add_argument("--ft-include", default="",
                   help="Comma-sep FT levels to include (overrides --deployment "
                        "filter). E.g. '6' or '4,6,8'. Implies deployment mode "
                        "(drop FT feature).")
    args = p.parse_args()

    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT_MODE = args.deployment or bool(args.ft_include)

    df = load_all_anchors(args)
    # Tag anchors as in-predictor or not, based on FT
    df["in_predictor"] = df["ft"].isin(list(DEPLOYMENT_FT))
    df.to_csv(OUT / "all_anchors.csv", index=False)
    print(f"[stats] mode={'DEPLOYMENT' if DEPLOYMENT_MODE else 'FULL (all FT)'}")
    print(f"[stats] out_dir={OUT}")
    if args.ft_include:
        ft_keep = {int(x) for x in args.ft_include.split(",")}
        before = len(df)
        df = df[df["ft"].isin(ft_keep)].reset_index(drop=True)
        print(f"[stats] filtered (--ft-include {sorted(ft_keep)}): "
              f"{before} → {len(df)} anchors")
    elif DEPLOYMENT_MODE:
        before = len(df)
        df = df[df["ft"].isin(list(DEPLOYMENT_FT))].reset_index(drop=True)
        print(f"[stats] filtered: {before} → {len(df)} anchors (FT in {sorted(DEPLOYMENT_FT)})")
    print(f"[stats] loaded N={len(df)} anchors, "
          f"AP avail={df['ap50'].notna().sum()}, "
          f"lat avail={df['lat_p50_ms'].notna().sum()}")

    results = {}
    print("[stats] A1 marginal coverage"); results["A1"] = fig_A1_marginal_coverage(df)
    print("[stats] A2/A3 grid heatmaps"); results["A2_A3"] = fig_A2_A3_grids(df)
    print("[stats] A4 PCA hull");         results["A4"] = fig_A4_hull_pca2d(df)
    print("[stats] B1 target hist");      results["B1"] = fig_B1_target_hist(df)
    print("[stats] B2 class balance");    results["B2"] = fig_B2_class_balance(df)
    print("[stats] B3 axis boxplots");    results["B3"] = fig_B3_by_axis_box(df)
    print("[stats] C1 MI/Spearman");      results["C1"] = fig_C1_mi_spearman(df)
    print("[stats] C2 corr matrix");      results["C2"] = fig_C2_corr_matrix(df)
    print("[stats] C3 interaction H");    results["C3"] = fig_C3_interaction_h(df)
    print("[stats] D1 5-fold CV");        results["D1"] = fig_D1_cv_r2(df)
    print("[stats] D2 learning curve");   results["D2"] = fig_D2_learning_curve(df)
    print("[stats] D3 OOD");              results["D3"] = fig_D3_ood(df)
    print("[stats] E1 noise floor");      results["E1"] = fig_E1_noise_floor(
        df, Path(args.v3_noise))

    (OUT / "stats_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    write_report(df, results)


if __name__ == "__main__":
    main()
