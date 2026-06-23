"""AP×τ interaction (coupling): dual-arm collision-vs-latency.
Data: clean6 routes x N~29-36/cell, H800 GPU0-3, 2026-06-22/23.
collEp = fraction of episodes with >=1 vehicle collision (robust to outliers).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tau = [600, 650, 700, 750, 800]
# high AP (drop=0.0, veh AP50~0.84)
hi_collep = [0.0, 15.6, 13.3, 25.8, 25.0]
hi_collep_ep = [0.000, 0.061, 0.024, 0.093, 0.076]
# low AP (drop=0.7, veh AP50~0.36)
lo_collep = [5.6, 37.9, 25.8, 36.4, 38.7]
lo_collep_ep = [0.008, 0.163, 0.111, 0.177, 0.197]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

ax = axes[0]
ax.plot(tau, hi_collep, "o-", color="#1f77b4", lw=2.2, ms=7, label="high AP (drop=0.0, AP50≈0.84)")
ax.plot(tau, lo_collep, "s-", color="#d62728", lw=2.2, ms=7, label="low AP (drop=0.7, AP50≈0.36)")
ax.axvspan(640, 660, color="orange", alpha=0.12)
ax.annotate("low-AP cliff onset\n(τ≈650, 38%)", xy=(650, 37.9), xytext=(660, 20),
            fontsize=9, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"))
ax.annotate("high-AP reaches ~25%\nonly at τ≈750-800", xy=(750, 25.8), xytext=(620, 30),
            fontsize=9, color="#1f77b4", arrowprops=dict(arrowstyle="->", color="#1f77b4"))
ax.set_xlabel("perception latency τ_perc (ms)")
ax.set_ylabel("episodes with ≥1 collision (%)")
ax.set_title("Collision-episode rate vs latency (dual AP arm)")
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(tau, hi_collep_ep, "o-", color="#1f77b4", lw=2.2, ms=7, label="high AP")
ax.plot(tau, lo_collep_ep, "s-", color="#d62728", lw=2.2, ms=7, label="low AP")
ax.set_xlabel("perception latency τ_perc (ms)")
ax.set_ylabel("mean collisions per episode")
ax.set_title("Mean collisions/episode vs latency")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle("CoDriving latency×accuracy COUPLING: low AP shifts collision cliff earlier (750→650ms) and deeper",
             fontsize=11, y=1.02)
fig.tight_layout()
out = "/home/jichengzhi/V2X/multi_agent/real_test/ap_tau_interaction_collision.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
