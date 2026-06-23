"""DS(AP, latency) predictive map for CoDriving closed-loop.
Merge: main 5x5 grid (25pts, tau{0,200,400,600,800} x AP{.841,.783,.559,.360,.281})
     + interaction fine-latency (tau{650,700,750} x AP{.841,.360}, 6pts).
honest DS: score_composed, timeout->DS=0, clean6 routes.
Usage: predict_ds(ap50, latency_ms) -> predicted closed-loop DS.
Sources: ap_tau_grid_ds.csv (main) + ix_ds.py output (interaction). 2026-06-23.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, LinearNDInterpolator

# (latency_ms, ap50, honest_DS)
PTS = [
    # --- main 5x5 grid ---
    (0,0.841,85.3),(0,0.783,68.5),(0,0.559,56.4),(0,0.360,63.8),(0,0.281,66.7),
    (200,0.841,84.7),(200,0.783,83.3),(200,0.559,75.0),(200,0.360,69.4),(200,0.281,60.0),
    (400,0.841,67.2),(400,0.783,66.9),(400,0.559,68.5),(400,0.360,54.6),(400,0.281,62.5),
    (600,0.841,77.4),(600,0.783,65.9),(600,0.559,76.8),(600,0.360,59.2),(600,0.281,81.3),
    (800,0.841,26.2),(800,0.783,37.2),(800,0.559,23.1),(800,0.360,27.7),(800,0.281,25.6),
    # --- interaction fine-latency (cliff localization) ---
    (650,0.841,28.0),(650,0.360,26.9),
    (700,0.841,28.2),(700,0.360,31.3),
    (750,0.841,30.2),(750,0.360,29.9),
]
P = np.array([(x[0],x[1]) for x in PTS]); Z = np.array([x[2] for x in PTS])

# query function (linear interp; nearest fallback outside hull)
_lin = LinearNDInterpolator(P, Z)
_near_x, _near_y, _near_z = P[:,0], P[:,1], Z
def predict_ds(ap50, latency_ms):
    v = float(_lin(latency_ms, ap50))
    if np.isnan(v):  # outside convex hull -> nearest measured
        d = (_near_x-latency_ms)**2 + ((_near_y-ap50)*1000)**2
        v = float(_near_z[int(np.argmin(d))])
    return round(v,1)

if __name__ == "__main__":
    # dense grid for heatmap
    gx = np.linspace(0,800,161); gy = np.linspace(0.281,0.841,113)
    GX,GY = np.meshgrid(gx,gy)
    GZ = griddata(P, Z, (GX,GY), method="linear")
    GZ_n = griddata(P, Z, (GX,GY), method="nearest")
    GZ = np.where(np.isnan(GZ), GZ_n, GZ)  # fill hull gaps with nearest

    fig, ax = plt.subplots(figsize=(9.5,6))
    cf = ax.contourf(GX,GY,GZ, levels=np.linspace(20,90,15), cmap="RdYlGn", extend="both")
    cs = ax.contour(GX,GY,GZ, levels=[40,55,70], colors="k", linewidths=0.8, alpha=0.5)
    ax.clabel(cs, fmt="%d", fontsize=8)
    cb = fig.colorbar(cf, ax=ax); cb.set_label("predicted closed-loop DS (honest, timeout=0)")
    # measured points
    ax.scatter(P[:,0],P[:,1], c="k", s=18, zorder=5)
    for (x,y),z in zip(P,Z):
        ax.annotate(f"{z:.0f}", (x,y), fontsize=6.5, ha="center", va="bottom", xytext=(0,2), textcoords="offset points")
    # cliff band
    ax.axvspan(600,650, color="purple", alpha=0.10)
    ax.text(625, 0.30, "DS cliff\n~625ms", color="purple", fontsize=8.5, ha="center", rotation=90, va="bottom")
    ax.set_xlabel("perception latency τ_perc (ms)")
    ax.set_ylabel("vehicle AP50")
    ax.set_title("CoDriving DS map: DS = f(AP, latency)  —  latency cliff at ~625ms dominates; AP modulates the safe plateau")
    ax.set_xticks([0,200,400,600,650,700,750,800])
    fig.tight_layout()
    out="/home/jichengzhi/V2X/multi_agent/real_test/ds_ap_latency_map.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)

    # sanity: print a query table
    print("\npredict_ds samples:")
    for ap in [0.84,0.56,0.36]:
        for lat in [100,400,600,640,700]:
            print(f"  AP50={ap:.2f} lat={lat:4d}ms -> DS {predict_ds(ap,lat)}")
