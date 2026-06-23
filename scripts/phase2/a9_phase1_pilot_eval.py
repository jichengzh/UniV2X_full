"""Phase 1.3 — AP eval on T11/T17/T22 finetune curve.

For each (triplet, FT_epoch), tag = T{X}_ft{N}:
  1. Find ckpt at HEAL epoch (23+FT) — net_epoch{23+FT}.pth
  2. Export ONNX from ckpt
  3. Build TRT FP32 engine
  4. Run e2e_eval_ap n=1789

FT 档位: 4, 6, 8, 10, 15  (→ HEAL epoch 27, 29, 31, 33, 38)
Triplets: T11_p14 (raw=64,64,256), T17_p57 (32,32,128), T22_p89 (16,16,16)

Total: 3 × 5 = 15 evals. 6 GPU parallel, ~10 min wall.
Output: /tmp/a9_pilot/eval/{tag}_ap.json + spread_table.csv + FT_decision.json
"""
from __future__ import annotations
import json, os, subprocess, time
from multiprocessing import Pool, current_process
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
OUT = Path("/tmp/a9_pilot/eval")
OUT.mkdir(parents=True, exist_ok=True)
DAIR = "/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure"

# (triplet_tag, ft_dir_name, prune_pct)
TRIPLETS = [
    ("T11", "ft_064_064_256_raw", 14),
    ("T17", "ft_032_032_128_raw", 57),
    ("T22", "ft_016_016_016_raw", 89),
]
FT_LEVELS = [4, 6, 8, 10, 15]  # HEAL epochs: 27, 29, 31, 33, 38

_GPU = None

def _init(gpus):
    global _GPU
    wid = current_process()._identity[0]
    _GPU = gpus[(wid - 1) % len(gpus)]
    print(f"[worker {wid}] GPU {_GPU}", flush=True)


def run_eval(spec):
    triplet, ft_dir_name, prune, ft = spec
    heal_epoch = 23 + ft
    tag = f"{triplet}_ft{ft:02d}"
    ft_dir = REPO / "models/dataset_a_cache" / ft_dir_name
    ckpt = ft_dir / f"net_epoch{heal_epoch}.pth"
    cfg = ft_dir / "config.yaml"
    onnx = OUT / f"{tag}.onnx"
    engine = OUT / f"{tag}_fp32.engine"
    build_rep = OUT / f"{tag}_fp32.build.json"
    ap_rep = OUT / f"{tag}_ap.json"

    if ap_rep.exists():
        d = json.loads(ap_rep.read_text())
        return (tag, ft, prune, d.get("ap50") or d.get("ap_50"), 0, "cached")

    if not ckpt.exists():
        return (tag, ft, prune, None, 0, f"ckpt missing: {ckpt.name}")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(_GPU)}
    t0 = time.time()

    # 1. ONNX export
    if not onnx.exists():
        cmd = [PYTHON, str(REPO / "tools/export_onnx_pyramid_e2e.py"),
               "--ckpt", str(ckpt), "--hypes", str(cfg),
               "--out", str(onnx), "--max-voxels", "32000"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
        if r.returncode != 0:
            return (tag, ft, prune, None, time.time()-t0, f"ONNX fail")

    # 2. TRT FP32 engine
    if not engine.exists():
        cmd = [PYTHON, str(REPO / "scripts/phase1/m4_8_trt_build_bench.py"),
               "--onnx", str(onnx), "--precision", "fp32",
               "--engine", str(engine), "--report", str(build_rep),
               "--workspace-mb", "4096", "--tactic", "default",
               "--builder-opt-level", "3", "--skip-bench"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        if r.returncode != 0:
            return (tag, ft, prune, None, time.time()-t0, f"engine fail")

    # 3. AP eval n=1789
    cmd = [PYTHON, str(REPO / "scripts/phase2/e2e_eval_ap.py"),
           "--engine", str(engine), "--ckpt-dir", str(ft_dir),
           "--dair-root", DAIR, "--max-voxels", "32000",
           "--n-samples", "1789", "--tag", tag,
           "--report", str(ap_rep)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    if r.returncode != 0:
        return (tag, ft, prune, None, time.time()-t0, f"AP fail")

    d = json.loads(ap_rep.read_text())
    ap50 = d.get("ap50") or d.get("ap_50")
    return (tag, ft, prune, ap50, time.time()-t0, "OK")


def decide_ft_star(results):
    """Decision logic per AP不敏感解决方案_plan.md Phase 1.2.4:
    FT* = argmax std(AP_triplet) s.t. AP[T11]>0.50 AND AP[T11]-AP[T22]>=0.05
    """
    import numpy as np
    # Pivot: row=FT, col=triplet, val=AP
    by_ft = {}
    for tag, ft, prune, ap50, secs, status in results:
        if ap50 is None: continue
        triplet = tag.split('_')[0]
        by_ft.setdefault(ft, {})[triplet] = ap50

    table = []
    for ft in sorted(by_ft.keys()):
        row = by_ft[ft]
        if not all(t in row for t in ['T11', 'T17', 'T22']):
            continue
        aps = [row['T11'], row['T17'], row['T22']]
        std = np.std(aps)
        rng = max(aps) - min(aps)
        ap_t11 = row['T11']
        ap_t22 = row['T22']
        gap = ap_t11 - ap_t22
        cons_a = ap_t11 > 0.50
        cons_c = gap >= 0.05
        feasible = cons_a and cons_c
        table.append({
            "FT": ft, "AP_T11": ap_t11, "AP_T17": row['T17'], "AP_T22": ap_t22,
            "std": float(std), "range": rng, "gap_T11_T22": gap,
            "cons_a_T11_gt_0.5": cons_a, "cons_c_gap_gt_0.05": cons_c,
            "feasible": feasible,
        })

    # Save table
    spread_path = OUT / "spread_table.csv"
    with open(spread_path, "w") as f:
        f.write("FT,AP_T11,AP_T17,AP_T22,std,range,gap,cons_a,cons_c,feasible\n")
        for r in table:
            f.write(f"{r['FT']},{r['AP_T11']:.4f},{r['AP_T17']:.4f},{r['AP_T22']:.4f},"
                    f"{r['std']:.4f},{r['range']:.4f},{r['gap_T11_T22']:.4f},"
                    f"{r['cons_a_T11_gt_0.5']},{r['cons_c_gap_gt_0.05']},{r['feasible']}\n")
    print(f"\nspread → {spread_path}")
    for r in table:
        marker = "★" if r["feasible"] else " "
        print(f"  {marker} FT={r['FT']:2d}  std={r['std']:.3f}  gap={r['gap_T11_T22']:+.3f}  "
              f"AP_T11={r['AP_T11']:.3f}  AP_T17={r['AP_T17']:.3f}  AP_T22={r['AP_T22']:.3f}")

    # Pick FT*
    feasible = [r for r in table if r["feasible"]]
    decision_path = OUT / "FT_decision.json"
    if not feasible:
        # No sweet spot — Phase 1 早停 (实测发现, 不算 fail)
        decision = {
            "FT_chosen": None,
            "status": "no_sweet_spot",
            "reasoning": "No FT in [4,15] satisfies AP[T11]>0.5 AND AP[T11]-AP[T22]>=0.05. "
                         "Either Pyramid+DAIR is too robust (need re-frame paper §C) or "
                         "the spread emerges at FT<4 (single-seed noisy region).",
            "table": table,
        }
    else:
        # FT* = max std among feasible
        best = max(feasible, key=lambda r: r["std"])
        decision = {
            "FT_chosen": best["FT"],
            "status": "found",
            "spread_std": best["std"],
            "AP_T11": best["AP_T11"],
            "AP_T17": best["AP_T17"],
            "AP_T22": best["AP_T22"],
            "gap_T11_T22": best["gap_T11_T22"],
            "constraint_a_satisfied": True,
            "constraint_c_satisfied": True,
            "reasoning": f"FT={best['FT']} maximizes spread std={best['std']:.3f} "
                         f"while AP[T11]={best['AP_T11']:.3f}>0.5 and "
                         f"AP[T11]-AP[T22]={best['gap_T11_T22']:.3f}>=0.05.",
            "table": table,
        }
    decision_path.write_text(json.dumps(decision, indent=2))
    print(f"\nFT_decision → {decision_path}")
    print(f"FT* = {decision.get('FT_chosen')}, status = {decision['status']}")


def main():
    specs = []
    for triplet, ft_dir, prune in TRIPLETS:
        for ft in FT_LEVELS:
            specs.append((triplet, ft_dir, prune, ft))
    print(f"[a9 eval] {len(specs)} (triplet × FT) evals")

    gpus = [5, 7]  # other GPUs used by other users (2026-05-21 17:30 onwards)
    with Pool(processes=len(gpus), initializer=_init, initargs=(gpus,)) as pool:
        results = list(pool.imap_unordered(run_eval, specs))

    results.sort(key=lambda r: (r[1], r[2]))
    print(f"\n{'tag':<15}{'FT':>4}{'prune':>7}{'AP50':>9}{'secs':>7}  status")
    for tag, ft, prune, ap50, secs, status in results:
        ap_str = f'{ap50:.3f}' if ap50 is not None else 'FAIL'
        print(f'{tag:<15}{ft:>4}{prune:>7}{ap_str:>9}{secs:>7.0f}  {status}')

    decide_ft_star(results)


if __name__ == "__main__":
    main()
