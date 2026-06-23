"""W_g 探针 Phase 1 — 最终分析 (用 s2_2e_e2e.py 可信逐宽度重测的真值)。
计算 default-Pareto vs tuned-Pareto (AP70, latency), 标 W_g/P_g;
并输出 stage0-localized 对齐余量消融表。
所有 latency = H800 TVM, backbone-only, REPS=200/500 mean, idle GPU。
"""
import json, os

# (label, num_filters, in_per_g, default_us, tuned_us, ap70_or_None, source)
GRID = [
    ("base",   [64,128,256], [4,8,16],  56321.21, 6319.98, 0.631, "s2_2e base_retest (≈旧9.06×)"),
    ("p50",    [32,64,128],  [2,4,8],   26241.67, 3010.51, 0.564, "s2_2e (ms_work_2e_p50, 旧)"),
    ("p75",    [16,32,64],   [1,2,4],   12689.07, 483.63,  0.530, "s2_2e p75_retest"),
    ("trap25", [48,96,192],  [3,6,12],  42355.27, 21614.80,0.590, "s2_2e (ms_work_2e_trap25, 旧, 已验证)"),
    ("pad64",  [64,96,192],  [4,6,12],  47407.26, 6152.04, 0.590, "s2_2e pad64_retest (trap25权重 s0零填充48->64, AP不变=0.590)"),
    ("iso_s0", [48,128,256], [3,8,16],  51249.03, 21752.09,None,  "s2_2e iso_s0_retest (仅s0失配)"),
    ("iso_s1", [64,96,256],  [4,6,16],  51622.49, 6910.63, None,  "s2_2e iso_s1_retest (仅s1失配)"),
    ("iso_s2", [64,128,192], [4,8,12],  51992.38, 7243.61, None,  "s2_2e iso_s2_retest (仅s2失配)"),
]

def pareto(points):  # points: list of (label, ap, lat); higher ap + lower lat dominates
    front = []
    for lab, ap, lat in points:
        dom = any((a >= ap and l <= lat and (a > ap or l < lat)) for (lb, a, l) in points if lb != lab)
        if not dom:
            front.append(lab)
    return front

rows = []
for lab, nf, ipg, d, t, ap, src in GRID:
    rows.append({"label": lab, "num_filters": nf, "in_per_g": ipg,
                 "aligned": all(x in (1,2,4,8,16,32) for x in ipg),
                 "default_us": d, "tuned_us": t, "ratio": round(d/t, 3),
                 "ap70": ap, "source": src})

ap_known = [(r["label"], r["ap70"], r["default_us"]) for r in rows if r["ap70"] is not None]
ap_known_t = [(r["label"], r["ap70"], r["tuned_us"]) for r in rows if r["ap70"] is not None]
def_front = pareto(ap_known)
tun_front = pareto(ap_known_t)
wg = sorted(set(def_front) - set(tun_front))
pg = sorted(set(tun_front) - set(def_front))

out = {
    "experiment": "W_g probe Phase 1 — prune x schedule (AP,lat) Pareto reshaping (CORRECTED, s2_2e trusted)",
    "hardware": "H800 Hopper TVM 0.20.dev1070; backbone-only; idle GPU; min/mean over REPS",
    "method_note": "用 s2_2e_e2e.py 逐宽度独立进程重测(fresh work dir); 旧 gap1_grid_v2 合并循环因 work-dir 撞名复用残缺db给出假 ratio=1.0, 已废弃",
    "grid": rows,
    "pareto_AP_known": {
        "widths": [r["label"] for r in rows if r["ap70"] is not None],
        "default_front": def_front,
        "tuned_front": tun_front,
        "W_g_on_default_not_tuned": wg,
        "P_g_on_tuned_not_default": pg,
    },
    "verdict": {
        "W_g_exists": len(wg) > 0,
        "W_g": wg,
        "P_g": pg,
        "interpretation": (
            "★W_g/P_g 配对(同 AP70=0.590, 仅 stage0 对齐不同): "
            "W_g=trap25[48,96,192] 在 default-Pareto 上(0.59 niche 内 default 42355 < pad64 47407, 贪心选它), "
            "tuned 后 1.96× 救不动 → 21615µs, 被 P_g=pad64 严格支配。"
            "P_g=pad64[64,96,192](trap25 的 s0 零填充48->64) default 下被 trap25 支配(贪心丢弃), "
            "但 tuned 7.71× → 6152µs, 成为 0.59-AP niche 的全局最优。"
            "= 单维(default)最优/多维(tuned)非最优的贪心吸引子 W_g + 单维次优/多维最优的被错过点 P_g, 完整配对。"
        ),
        "mechanism_stage0_localized": (
            "对齐余量塌缩只由 stage0 失配驱动: iso_s0(仅s0,in_per_g=3) 2.36× ≈ trap25 1.96×; "
            "iso_s1(s1,ipg=6) 7.47× / iso_s2(s2,ipg=12) 7.18× 接近对齐 base 8.91×; "
            "pad64(s0 修复对齐, s1/s2 仍失配) 7.71× ⇒ 修 s0 即恢复余量, 坐实 stage0 主导。"
            "stage0 在最高分辨率(128x256), grouped conv 主导延迟, in_per_g=3 罚最重。"
        ),
        "overturns": (
            "旧结论'pad64 ratio=1.0/pad救援负结果/对齐是全网属性'是 buggy harness 假数(单进程循环复用撞名残缺db); "
            "pad64_retest 实测 7.71× ⇒ pad救援有效, 且 pad64 就是 P_g。"
        ),
    },
}
os.makedirs("results", exist_ok=True)
with open("results/gap1_grid_corrected.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("=== tuning headroom (ratio) vs alignment ===")
for r in rows:
    flag = "ALIGNED" if r["aligned"] else f"MISALIGN(s0..s2 ipg={r['in_per_g']})"
    print(f"  {r['label']:8s} nf={str(r['num_filters']):16s} ratio={r['ratio']:6.2f}x  def={r['default_us']:9.0f}  tuned={r['tuned_us']:9.0f}  ap70={r['ap70']}  [{flag}]")
print("\n=== Pareto (AP70-known widths: base/p50/p75/trap25) ===")
print("default-schedule front:", def_front)
print("tuned-schedule  front:", tun_front)
print("W_g (default-Pareto, tuned-dropped):", wg)
print("P_g (tuned-Pareto, default-new):    ", pg)
print("\nWritten results/gap1_grid_corrected.json")
