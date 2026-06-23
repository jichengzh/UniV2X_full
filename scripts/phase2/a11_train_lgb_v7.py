"""Phase 3.2 (plan v3) — Train LGB v7 predictors (ap + latency).

Trains two LightGBM predictors:
  v7_ap:  predicts AP50 from (B, Q, FT, D) features
  v7_lat: predicts lat_p50_ms from same features

5-fold CV + OOD holdout (hold p97 / hold int8_pc_wo / hold FT=2).
Strict thresholds per plan v3 §3.2:
  - 5-fold CV R² ≥ 0.75
  - 5-fold CV MAE ≤ 0.04
  - OOD MAE ≤ 1.5 × in-sample

Output:
  models/lgb_v7_ap.txt
  models/lgb_v7_latency.txt
  models/lgb_v7_feature_importance.csv
  results/lgb_v7_cv_metrics.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

REPO = Path("/home/jichengzhi/UniV2X")
ANCHORS = REPO / "paper_learning/2. AAAI最终故事/data/stats_v3/all_anchors.csv"
MODELS_DIR = REPO / "models"
RESULTS_DIR = REPO / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


DEPLOYMENT_MODE = False  # set by --deployment


def featurize(df: pd.DataFrame, target_col: str):
    sub = df.dropna(subset=[target_col]).reset_index(drop=True)
    if DEPLOYMENT_MODE:
        # Deployment view: FT fixed by config, D doesn't affect AP.
        # Use only (B, Q) features.
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
    y = sub[target_col].values.astype(float)
    return X, y, sub


def _r2(yt, yp):
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def cv_evaluate(X, y, n_splits=5, seed=0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2s = []; maes = []
    for tr, te in kf.split(X):
        m = lgb.LGBMRegressor(objective="regression", learning_rate=0.05,
                              num_leaves=15, min_data_in_leaf=2,
                              n_estimators=300, verbose=-1, random_state=seed,
                              num_threads=1, force_col_wise=True)
        m.fit(X.values[tr], y[tr])
        p = m.predict(X.values[te])
        r2s.append(_r2(y[te], p))
        maes.append(float(np.mean(np.abs(y[te] - p))))
    return r2s, maes


def ood_evaluate(df: pd.DataFrame, target_col: str, hold_filter):
    sub = df.dropna(subset=[target_col]).reset_index(drop=True)
    mask = hold_filter(sub)
    train_df = sub[~mask]
    test_df = sub[mask]
    if len(train_df) < 10 or len(test_df) < 2:
        return float("nan")
    Xtr, ytr, _ = featurize(train_df, target_col)
    Xte, yte, _ = featurize(test_df, target_col)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    m = lgb.LGBMRegressor(objective="regression", learning_rate=0.05,
                          num_leaves=15, min_data_in_leaf=2,
                          n_estimators=300, verbose=-1, random_state=0,
                          num_threads=1, force_col_wise=True)
    m.fit(Xtr.values, ytr)
    pred = m.predict(Xte.values)
    return float(np.mean(np.abs(yte - pred)))


def train_final(X, y, name: str) -> tuple[lgb.LGBMRegressor, str]:
    m = lgb.LGBMRegressor(objective="regression", learning_rate=0.05,
                          num_leaves=15, min_data_in_leaf=2,
                          n_estimators=300, verbose=-1, random_state=0,
                          num_threads=1, force_col_wise=True)
    m.fit(X.values, y)
    out = MODELS_DIR / name
    m.booster_.save_model(str(out))
    return m, str(out)


def main():
    global DEPLOYMENT_MODE
    p = argparse.ArgumentParser()
    p.add_argument("--anchors", default=str(ANCHORS))
    p.add_argument("--r2-min", type=float, default=0.75)
    p.add_argument("--mae-max", type=float, default=0.04)
    p.add_argument("--deployment", action="store_true",
                   help="Filter FT in {8,15} + unique (B,Q), drop FT/D features. "
                        "Suffix '_deployment' added to all output filenames.")
    p.add_argument("--ft-include", default="",
                   help="Comma-sep FT levels to include (overrides --deployment "
                        "filter). E.g. '6' or '4,6,8'. Implies deployment mode.")
    p.add_argument("--suffix", default="",
                   help="Output file suffix (e.g. '_ft6'). If empty + "
                        "--deployment → '_deployment'.")
    args = p.parse_args()
    DEPLOYMENT_MODE = args.deployment or bool(args.ft_include)

    df = pd.read_csv(args.anchors)
    print(f"[lgb v7] mode={'DEPLOYMENT' if DEPLOYMENT_MODE else 'FULL'}")
    print(f"[lgb v7] loaded {len(df)} anchors (raw)")

    if DEPLOYMENT_MODE:
        before = len(df)
        if args.ft_include:
            ft_keep = {int(x) for x in args.ft_include.split(",")}
        else:
            ft_keep = {8, 15}
        df = df[df["ft"].isin(ft_keep)]
        df = df.drop_duplicates(subset=["triplet", "q"]).reset_index(drop=True)
        print(f"[lgb v7] deployment filter (FT={sorted(ft_keep)}): "
              f"{before} → {len(df)} unique (B,Q) anchors")

    print(f"  AP avail: {df['ap50'].notna().sum()}")
    print(f"  lat avail: {df['lat_p50_ms'].notna().sum()}")
    if args.suffix:
        suffix = args.suffix
    elif DEPLOYMENT_MODE:
        suffix = "_deployment"
    else:
        suffix = ""

    metrics = {}

    # ============ v7_ap ============
    print("\n=== Training v7_ap ===")
    Xa, ya, _ = featurize(df, "ap50")
    print(f"  X shape: {Xa.shape}, y range: [{ya.min():.3f}, {ya.max():.3f}]")
    r2s, maes = cv_evaluate(Xa, ya)
    mean_r2 = float(np.mean(r2s)); mean_mae = float(np.mean(maes))
    print(f"  5-fold CV: R²={mean_r2:.4f} (per-fold {[f'{r:.3f}' for r in r2s]})")
    print(f"             MAE={mean_mae:.4f}")
    ap_pass = (mean_r2 >= args.r2_min and mean_mae <= args.mae_max)
    print(f"  in-sample threshold pass: {'✅' if ap_pass else '❌'}")

    print("  OOD tests:")
    if DEPLOYMENT_MODE:
        ood_ap = {
            "hold p97": ood_evaluate(df, "ap50",
                                      lambda d: d["triplet"] == "T_g8_p97"),
            "hold int8_pc_wo": ood_evaluate(df, "ap50",
                                             lambda d: d["q"] == "int8_pc_wo"),
            "hold int8_ent": ood_evaluate(df, "ap50",
                                           lambda d: d["q"] == "int8_ent"),
        }
    else:
        ood_ap = {
            "hold p97": ood_evaluate(df, "ap50",
                                      lambda d: d["triplet"] == "T_g8_p97"),
            "hold int8_pc_wo": ood_evaluate(df, "ap50",
                                             lambda d: d["q"] == "int8_pc_wo"),
            "hold FT=2": ood_evaluate(df, "ap50", lambda d: d["ft"] == 2),
        }
    ood_thresh = 1.5 * mean_mae
    for k, v in ood_ap.items():
        ok = (not math.isnan(v)) and v <= ood_thresh
        print(f"    {k}: MAE={v:.4f} {'✅' if ok else '❌'} (threshold {ood_thresh:.4f})")
    m_ap, path_ap = train_final(Xa, ya, f"lgb_v7_ap{suffix}.txt")
    print(f"  saved: {path_ap}")
    metrics["v7_ap"] = {
        "n": len(ya), "mean_r2": mean_r2, "mean_mae": mean_mae,
        "r2_per_fold": r2s, "mae_per_fold": maes,
        "ood": ood_ap, "ood_threshold": ood_thresh,
        "pass": ap_pass,
    }

    # feature importance
    imp = pd.DataFrame({
        "feature": Xa.columns,
        "importance": m_ap.booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    imp.to_csv(MODELS_DIR / f"lgb_v7_feature_importance{suffix}.csv", index=False)
    print(f"  top features: {imp.head(5)['feature'].tolist()}")

    # ============ v7_lat ============
    if df["lat_p50_ms"].notna().sum() >= 15:
        print("\n=== Training v7_lat ===")
        # log-space target to handle wide latency range
        df_lat = df.copy()
        df_lat["log_lat"] = np.log10(df_lat["lat_p50_ms"].astype(float))
        Xl, yl, _ = featurize(df_lat, "log_lat")
        print(f"  X shape: {Xl.shape}, log10(lat) range: "
              f"[{yl.min():.3f}, {yl.max():.3f}]")
        r2s_l, maes_l = cv_evaluate(Xl, yl)
        mean_r2_l = float(np.mean(r2s_l))
        mean_mae_l = float(np.mean(maes_l))
        # MAE in log space → multiplicative factor
        mult_factor = 10 ** mean_mae_l
        print(f"  5-fold CV (log space): R²={mean_r2_l:.4f}")
        print(f"  log10 MAE={mean_mae_l:.4f} (geometric mean error factor "
              f"{mult_factor:.3f}×)")
        lat_pass = mean_r2_l >= args.r2_min
        print(f"  threshold pass: {'✅' if lat_pass else '❌'}")
        m_lat, path_lat = train_final(Xl, yl, f"lgb_v7_latency{suffix}.txt")
        print(f"  saved: {path_lat}")
        metrics["v7_lat"] = {
            "n": len(yl), "mean_r2": mean_r2_l, "mean_mae_log": mean_mae_l,
            "multiplicative_factor": mult_factor,
            "pass": lat_pass,
        }
    else:
        print("\n=== v7_lat SKIPPED (insufficient lat data) ===")
        metrics["v7_lat"] = {"pass": False, "skipped": True,
                             "n": int(df['lat_p50_ms'].notna().sum())}

    # ============ Write metrics ============
    out = RESULTS_DIR / f"lgb_v7_cv_metrics{suffix}.json"
    out.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\n[lgb v7] metrics → {out}")
    print(f"\n=== Final verdict ===")
    print(f"  v7_ap:  {'✅' if metrics['v7_ap']['pass'] else '❌'}")
    print(f"  v7_lat: {'✅' if metrics['v7_lat']['pass'] else '❌'}")


if __name__ == "__main__":
    main()
