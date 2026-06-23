"""Finetune curve experiment — measure T22 AP50 vs finetune epochs.

Take 6 intermediate T22 ckpts (epoch 27/29/31/33/37/41 = +4/+6/+8/+10/+14/+18 FT epochs),
for each: export ONNX → build TRT FP32 engine → run e2e_eval_ap (n=1789).

Each anchor ~7 min wall on 1 GPU. 6 anchors / 6 GPUs ≈ 7 min total wall.

Output: /tmp/finetune_curve/{tag}_ap.json + summary.
"""
from __future__ import annotations
import json, os, subprocess, time
from multiprocessing import Pool, current_process
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
OUT = Path("/tmp/finetune_curve")
OUT.mkdir(exist_ok=True)
CKPT_DIR = REPO / "models/dataset_a_cache/ft_016_016_016"
CFG = CKPT_DIR / "config.yaml"
DAIR = "/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure"

# (epoch_in_HEAL, +finetune_epochs)
TARGETS = [
    (27, 4),
    (29, 6),
    (31, 8),
    (33, 10),
    (37, 14),
    (41, 18),
]

_GPU = None

def _init(gpus):
    global _GPU
    wid = current_process()._identity[0]
    _GPU = gpus[(wid - 1) % len(gpus)]
    print(f"[worker {wid}] GPU {_GPU}", flush=True)


def run_one(spec):
    epoch_heal, ft_eps = spec
    gpu = _GPU
    tag = f"T22_e{epoch_heal}_ft{ft_eps:02d}"
    ckpt = CKPT_DIR / f"net_epoch{epoch_heal}.pth"
    onnx = OUT / f"{tag}.onnx"
    engine = OUT / f"{tag}.engine"
    build_rep = OUT / f"{tag}.build.json"
    ap_rep = OUT / f"{tag}_ap.json"

    if ap_rep.exists():
        d = json.loads(ap_rep.read_text())
        return (tag, ft_eps, d.get("ap50") or d.get("ap_50"), 0.0, "cached")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    t0 = time.time()

    # 1. ONNX export
    if not onnx.exists():
        cmd = [PYTHON, str(REPO / "tools/export_onnx_pyramid_e2e.py"),
               "--ckpt", str(ckpt), "--hypes", str(CFG),
               "--out", str(onnx), "--max-voxels", "32000"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
        if r.returncode != 0:
            return (tag, ft_eps, None, time.time()-t0, f"ONNX fail: {r.stderr[-200:]}")

    # 2. TRT FP32 engine
    if not engine.exists():
        cmd = [PYTHON, str(REPO / "scripts/phase1/m4_8_trt_build_bench.py"),
               "--onnx", str(onnx), "--precision", "fp32",
               "--engine", str(engine), "--report", str(build_rep),
               "--workspace-mb", "4096", "--tactic", "default",
               "--builder-opt-level", "3", "--skip-bench"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        if r.returncode != 0:
            return (tag, ft_eps, None, time.time()-t0, f"engine fail: {r.stderr[-200:]}")

    # 3. AP eval
    cmd = [PYTHON, str(REPO / "scripts/phase2/e2e_eval_ap.py"),
           "--engine", str(engine), "--ckpt-dir", str(CKPT_DIR),
           "--dair-root", DAIR, "--max-voxels", "32000",
           "--n-samples", "1789", "--tag", tag,
           "--report", str(ap_rep)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    if r.returncode != 0:
        return (tag, ft_eps, None, time.time()-t0, f"AP fail: {r.stderr[-200:]}")

    d = json.loads(ap_rep.read_text())
    ap50 = d.get("ap50") or d.get("ap_50")
    return (tag, ft_eps, ap50, time.time()-t0, "OK")


def main():
    gpus = [0, 1, 2, 3, 4, 5]
    print(f"[a8] {len(TARGETS)} ckpts × n=1789 AP eval, {len(gpus)} GPUs")
    t_start = time.time()
    with Pool(processes=len(gpus), initializer=_init, initargs=(gpus,)) as pool:
        results = list(pool.imap_unordered(run_one, TARGETS))

    results.sort(key=lambda r: r[1])
    print()
    print(f"{'tag':<25}{'+FT ep':>8}{'AP50':>10}{'secs':>8}{'status':<15}")
    for tag, ft, ap50, secs, status in results:
        ap_str = f'{ap50:.3f}' if ap50 is not None else 'FAIL'
        print(f'{tag:<25}{ft:>8}{ap_str:>10}{secs:>8.0f}  {status[:30]}')
    print(f"\nwall: {(time.time()-t_start)/60:.1f} min")

    # Save summary including known datapoints
    summary = {
        "datapoints": [
            {"ft_epochs": 0, "ap50": 5e-8, "source": "raw_pruned_test"},
            {"ft_epochs": 2, "ap50": 0.402, "source": "T22_e25_eval"},
        ],
        "intermediate": [
            {"tag": tag, "ft_epochs": ft, "ap50": ap50, "secs": secs, "status": status}
            for tag, ft, ap50, secs, status in results
        ],
        "endpoint": [{"ft_epochs": 24, "ap50": 0.541, "source": "csv_T22_p89_Q_fp32"}],
        "baseline": [{"ap50": 0.549, "source": "T1_base_Q_fp32"}],
    }
    (OUT / "finetune_curve_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary → {OUT}/finetune_curve_summary.json")


if __name__ == "__main__":
    main()
