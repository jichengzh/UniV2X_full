"""Plot 3-panel histogram comparing AP distribution at FT=4/6/8.

Each panel: 15 anchor (3 triplet × 5 Q), uniform x-axis, std/range/mean annotated.
Layout mirrors B1: collapse=0.30 + degrade=0.50 threshold lines.

Output: stats_v3/16_ft_sweep_comparison.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

REPO = Path("/home/jichengzhi/UniV2X")
OUT = REPO / "paper_learning/2. AAAI最终故事/data/stats_v3"

df = pd.read_csv("/tmp/a11_ft_sweep/ft_sweep.csv")

# Reshape wide → long
rows = []
for _, r in df.iterrows():
    for q in ["fp32", "fp16", "int8_mm", "int8_pc_wo", "int8_ent"]:
        v = r[q]
        if pd.notna(v) and v != "NA":
            try:
                rows.append({"triplet": r["triplet"], "ft": int(r["ft"]),
                             "q": q, "ap50": float(v)})
            except (ValueError, TypeError):
                pass

long = pd.DataFrame(rows)
print(f"long table: {len(long)} rows, FT levels: {sorted(long['ft'].unique())}")

ft_levels = [4, 6, 8]
colors = ["#C0392B", "#F39C12", "#27AE60"]  # red / orange / green
# noise estimates: FT=4 and FT=6 实测 5-seed, FT=8 估算 (推断 <FT=6)
sigma_noise = {4: 0.096, 6: 0.0234, 8: 0.02}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
xmin, xmax = -0.02, 0.65

for ax, ft, color in zip(axes, ft_levels, colors):
    sub = long[long["ft"] == ft]["ap50"].values
    n = len(sub)
    mean = sub.mean()
    std = sub.std()
    rng = sub.max() - sub.min()
    snoise = sigma_noise[ft]
    r2_ceil = 1.0 - (snoise ** 2) / (std ** 2) if std > snoise else max(
        0.0, 1.0 - (snoise ** 2) / (std ** 2))
    if std < snoise:
        r2_ceil_str = f"~0 or NEG (noise > signal)"
    else:
        r2_ceil_str = f"{r2_ceil:.2f}"

    ax.hist(sub, bins=15, color=color, alpha=0.75, edgecolor="k", linewidth=0.5)
    # KDE-ish: estimate by simple gaussian smoothing
    if n > 5:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(sub, bw_method=0.3)
        xs = np.linspace(xmin, xmax, 200)
        kde_y = kde(xs) * n * (xmax - xmin) / 15
        ax.plot(xs, kde_y, color=color, lw=2, alpha=0.9)

    ax.axvline(0.30, color="red", ls="--", lw=1.5, alpha=0.6, label="collapse 0.30")
    ax.axvline(0.50, color="orange", ls="--", lw=1.5, alpha=0.6, label="degrade 0.50")
    ax.axvline(mean, color="black", ls=":", lw=1.5, alpha=0.7, label=f"mean={mean:.3f}")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 7)
    ax.set_xlabel("AP50", fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)

    learn_status = "❌ NOT LEARNABLE" if std < snoise else "⚠️ marginal" if r2_ceil < 0.5 else "✅ learnable"
    learn_ascii = learn_status.replace("❌", "[X]").replace("✅", "[V]").replace("⚠️", "[!]")

    ax.set_title(
        f"FT={ft}  (epoch={19+ft})\n"
        f"N={n}, std={std:.3f}, range={rng:.3f}, mean={mean:.3f}\n"
        f"σ_noise≈{snoise:.3f} → R²_ceiling = {r2_ceil_str}  {learn_ascii}",
        fontsize=11
    )

fig.suptitle(
    f"FT Sweep Comparison (3 triplet x 5 Q x 3 FT = 45 anchor)  "
    f"[gen {datetime.now().strftime('%Y-%m-%d %H:%M')}]\n"
    f"FT=4: large signal but noise>signal -> NOT learnable; "
    f"FT=6 mid; FT=8 small signal but stable -> learnable",
    fontsize=12, fontweight="bold"
)
fig.tight_layout()
out = OUT / "16_ft_sweep_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")

# Print per-FT summary
print("\n=== FT sweep summary ===")
print(f"{'FT':>4}{'N':>5}{'mean':>9}{'std':>9}{'range':>9}{'σ_noise':>10}{'R²_ceiling':>14}")
for ft in ft_levels:
    sub = long[long["ft"] == ft]["ap50"].values
    if len(sub) == 0:
        continue
    std = sub.std()
    snoise = sigma_noise[ft]
    r2_ceil = 1.0 - (snoise ** 2) / (std ** 2)
    print(f"{ft:>4}{len(sub):>5}{sub.mean():>9.4f}{std:>9.4f}"
          f"{sub.max()-sub.min():>9.4f}{snoise:>10.4f}{r2_ceil:>14.4f}")
