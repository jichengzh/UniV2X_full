"""Build CoDriving AP x latency measured maps.

Inputs:
  ds_ap_latency_all_measured.csv

Outputs:
  ds_ap_latency_map_v2.png
  ap_tau_collision_map_v2.png

The CSV is consolidated from H800 raw results. DS uses the honest protocol:
score_composed, with TIMEOUT_SKIP or missing score counted as DS=0. Collision
and RC columns use the previous interaction-analysis protocol: non-timeout
episodes only, with timeout_pct reported separately.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "ds_ap_latency_all_measured.csv"


def _load_rows(csv_path: Path = CSV_PATH) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                if key in {"source"}:
                    parsed[key] = value
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


ROWS = _load_rows()
LATENCIES = np.array(sorted({float(r["latency_ms"]) for r in ROWS}))
APS_ASC = np.array(sorted({float(r["ap50"]) for r in ROWS}))
APS_DESC = APS_ASC[::-1]


def _matrix(metric: str) -> np.ndarray:
    lookup = {(float(r["ap50"]), float(r["latency_ms"])): float(r[metric]) for r in ROWS}
    return np.array([[lookup[(ap, lat)] for lat in LATENCIES] for ap in APS_ASC])


DS = _matrix("honest_DS")
COLL_EP = _matrix("collision_episode_pct")
COLL_PER_EP = _matrix("collision_per_ep")

_ds_interp = RegularGridInterpolator(
    (APS_ASC, LATENCIES),
    DS,
    bounds_error=False,
    fill_value=None,
)


def predict_ds(ap50: float, latency_ms: float) -> float:
    """Return interpolated honest DS for a measured AP50/latency pair."""
    ap = float(np.clip(ap50, APS_ASC.min(), APS_ASC.max()))
    lat = float(np.clip(latency_ms, LATENCIES.min(), LATENCIES.max()))
    return round(float(_ds_interp([[ap, lat]])[0]), 1)


def _plot_map(
    metric: str,
    z: np.ndarray,
    out_name: str,
    title: str,
    cbar_label: str,
    cmap: str,
    levels: np.ndarray,
    contour_levels: list[float],
    value_fmt: str,
) -> None:
    x_dense = np.linspace(float(LATENCIES.min()), float(LATENCIES.max()), 321)
    y_dense = np.linspace(float(APS_ASC.min()), float(APS_ASC.max()), 241)
    interp = RegularGridInterpolator((APS_ASC, LATENCIES), z, bounds_error=False, fill_value=None)
    yy, xx = np.meshgrid(y_dense, x_dense, indexing="ij")
    zz = interp(np.column_stack([yy.ravel(), xx.ravel()])).reshape(yy.shape)

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    filled = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, extend="both")
    contours = ax.contour(xx, yy, zz, levels=contour_levels, colors="black", linewidths=0.75, alpha=0.55)
    ax.clabel(contours, fmt="%g", fontsize=8)
    cb = fig.colorbar(filled, ax=ax)
    cb.set_label(cbar_label)

    for row in ROWS:
        x = float(row["latency_ms"])
        y = float(row["ap50"])
        value = float(row[metric])
        ax.scatter(x, y, c="black", s=14, zorder=4)
        ax.annotate(
            format(value, value_fmt),
            (x, y),
            fontsize=6.5,
            ha="center",
            va="bottom",
            xytext=(0, 2),
            textcoords="offset points",
        )

    ax.axvspan(600, 650, color="#6a3d9a", alpha=0.10, lw=0)
    ax.text(
        625,
        float(APS_ASC.min()) + 0.018,
        "600-650ms cliff band",
        color="#5b2b83",
        fontsize=8.5,
        ha="center",
        va="bottom",
        rotation=90,
    )
    ax.set_xlabel("perception latency tau_perc (ms)")
    ax.set_ylabel("vehicle AP50")
    ax.set_xticks(LATENCIES)
    ax.set_yticks(APS_DESC)
    ax.set_ylim(float(APS_ASC.min()) - 0.025, float(APS_ASC.max()) + 0.025)
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.set_title(title)
    fig.tight_layout()
    out = HERE / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def build_maps() -> None:
    _plot_map(
        metric="honest_DS",
        z=DS,
        out_name="ds_ap_latency_map_v2.png",
        title="CoDriving measured DS map: DS = f(AP50, perception latency)",
        cbar_label="honest composed DS (TIMEOUT_SKIP=0)",
        cmap="RdYlGn",
        levels=np.linspace(15, 95, 17),
        contour_levels=[25, 40, 55, 70, 85],
        value_fmt=".0f",
    )
    _plot_map(
        metric="collision_episode_pct",
        z=COLL_EP,
        out_name="ap_tau_collision_map_v2.png",
        title="CoDriving measured safety map: vehicle collision episode rate",
        cbar_label="episodes with >=1 vehicle collision (%)",
        cmap="YlOrRd",
        levels=np.linspace(0, 60, 13),
        contour_levels=[10, 25, 40, 50],
        value_fmt=".0f",
    )


if __name__ == "__main__":
    build_maps()
    print("\npredict_ds samples:")
    for ap in [0.841, 0.559, 0.281]:
        for lat in [500, 600, 650, 700, 750]:
            print(f"  AP50={ap:.3f} latency={lat:3d}ms -> DS {predict_ds(ap, lat):4.1f}")
