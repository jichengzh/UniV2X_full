import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

taus=[0,200,400,600,650,700,750,800]
aps=[0.841,0.783,0.559,0.360,0.281]
# measured DS + N ; None = not measured (interpolated in map)
M={
 (0,0.841):(85.3,18),(0,0.783):(68.5,18),(0,0.559):(56.4,18),(0,0.360):(63.8,18),(0,0.281):(66.7,18),
 (200,0.841):(84.7,18),(200,0.783):(83.3,18),(200,0.559):(75.0,18),(200,0.360):(69.4,18),(200,0.281):(60.0,18),
 (400,0.841):(67.2,18),(400,0.783):(66.9,18),(400,0.559):(68.5,18),(400,0.360):(54.6,18),(400,0.281):(62.5,18),
 (600,0.841):(77.4,18),(600,0.783):(65.9,18),(600,0.559):(76.8,18),(600,0.360):(59.2,18),(600,0.281):(81.3,18),
 (650,0.841):(28.0,36),(650,0.360):(26.9,36),
 (700,0.841):(28.2,36),(700,0.360):(31.3,36),
 (750,0.841):(30.2,36),(750,0.360):(29.9,36),
 (800,0.841):(26.2,18),(800,0.783):(37.2,18),(800,0.559):(23.1,18),(800,0.360):(27.7,18),(800,0.281):(25.6,18),
}
fig,ax=plt.subplots(figsize=(10,5))
for j,t in enumerate(taus):
    for i,a in enumerate(aps):
        if (t,a) in M:
            ds,n=M[(t,a)]
            col="#2ca02c" if n>=30 else "#7fbf7f"
            ax.add_patch(Rectangle((j-0.5,i-0.5),1,1,facecolor=col,edgecolor="k",lw=0.5))
            ax.text(j,i,f"{ds:.0f}\nN{n}",ha="center",va="center",fontsize=8,color="white",weight="bold")
        else:
            ax.add_patch(Rectangle((j-0.5,i-0.5),1,1,facecolor="#d62728",alpha=0.25,edgecolor="k",lw=0.5,hatch="xxx"))
            ax.text(j,i,"interp\n(gap)",ha="center",va="center",fontsize=7.5,color="#d62728")
ax.set_xticks(range(len(taus))); ax.set_xticklabels(taus)
ax.set_yticks(range(len(aps))); ax.set_yticklabels([f"{a:.2f}" for a in aps])
ax.set_xlim(-0.5,len(taus)-0.5); ax.set_ylim(-0.5,len(aps)-0.5)
ax.set_xlabel("perception latency τ_perc (ms)"); ax.set_ylabel("vehicle AP50")
ax.axvspan(3.5,6.5,color="purple",alpha=0.08)
ax.set_title("DS map COVERAGE: measured (green, dark=N36) vs interpolated gaps (red hatched)\n"
             "9 gaps all in the cliff band 650-750 at mid/extreme AP — the decision-critical region",fontsize=10)
fig.tight_layout()
out="/home/jichengzhi/V2X/multi_agent/real_test/ds_coverage_matrix.png"
fig.savefig(out,dpi=140,bbox_inches="tight"); print("saved",out)
nmeas=len(M); ngap=len(taus)*len(aps)-nmeas
print(f"measured cells={nmeas}, interpolated gaps={ngap}, total grid={len(taus)*len(aps)}")
