"""B2 — AP70 calibration finetune launcher.

Goal: finetune 4 NON-UNIFORM width configs to calibrate per-stage AP70 contributions.
These are the iso + mixed calibration points needed to fit AP70(s0,s1,s2).

Configs (non-uniform, contrast with uniform anchors in stage_a_ap_real.parquet):
  iso_s0: [48,128,256] — only s0 reduced (trap25 level), s1/s2 at base
  iso_s1: [64, 96,256] — s0/s2 at base, s1 at trap25 level
  iso_s2: [64,128,192] — s0/s1 at base, s2 at trap25 level
  mixed:  [32, 96,192] — s0 at p50, s1/s2 at trap25 level (held-out validation)

Also runs pipeline verification: HEAL inference.py eval on existing pruned25 ckpt.

Pipeline per config:
  1. structural_prune_pyramid.py --num-filters-new s0,s1,s2 --width-per-group 16 --groups 32
  2. Flatten init ckpt (unwrap model_state_dict wrapper — CLAUDE.md §〇.5 trap)
  3. Patch config epoches=31 (match stage_a training budget: init@23 → epoch 31)
  4. Launch HEAL train_ddp.py detached (setsid) per GPU

IMPORTANT:
  - GPUs 0,4,6,7 confirmed free (util=0%, mem<10MiB as of 2026-06-20)
  - GPUs 1,2,3,5 are occupied by wuyuegao — DO NOT USE
  - All finetune processes are detached (start_new_session=True)

Usage:
  cd /home/jichengzhi/V2X
  PYTHONPATH=/home/jichengzhi/heal_research/HEAL \\
  /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \\
    scripts/phase2/b2_finetune_launcher.py

Output:
  results/b2_finetune_pids.json  — launched PIDs + metadata
  results/b2_verify_pruned25.log — pipeline verification eval log
  logs for each finetune in results/b2_{tag}_finetune.log
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path("/home/jichengzhi/V2X")
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
BASELINE = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
CKROOT = Path("/home/jichengzhi/heal_research/checkpoints/stage1")
EPOCHES = 31          # Same training budget as stage_a (init@23 → epoch 31 = 8 finetune epochs)
WPG = 4               # Match base model's actual wpg (base/pruned25/50/75 all use wpg=4 confirmed)
                      # Base model: conv1 width=128=int(64*4/64)*32 => wpg=4.
                      # Using wpg=16 FAILS for iso configs where some stages keep base width:
                      # n_keep_w(stage1,wpg16)=1024 > actual old_width(stage1,wpg4)=256 → topk crash
GROUPS = 32
MASTER_PORT_BASE = 29700  # Offset to avoid clash with existing finetune sessions

# (tag, num_filters, gpu)
# Calibration points on free GPUs 0, 4, 6, 7
CALIBRATION = [
    ("iso_s0",  [48, 128, 256], 4),   # Only s0 pruned (=trap25 level); GPU 4
    ("iso_s1",  [64,  96, 256], 6),   # Only s1 pruned (=trap25 level); GPU 6
    ("iso_s2",  [64, 128, 192], 7),   # Only s2 pruned (=trap25 level); GPU 7
    ("mixed",   [32,  96, 192], 0),   # Mixed (s0=p50, s1/s2=trap25); GPU 0 (after verify)
]

# Existing pruned25 ckpt for pipeline verification
VERIFY_DIR = CKROOT / "Pyramid_DAIR_m1_pruned25_2026_05_10"
VERIFY_TAG = "pruned25"


def ckpt_dir_for(tag: str) -> Path:
    return CKROOT / f"Pyramid_DAIR_m1_b2_{tag}_2026_06_20"


def run_prune(tag: str, nf: list, gpu: int) -> Path:
    """Prune backbone to nf widths via structural_prune_pyramid.py."""
    out_dir = ckpt_dir_for(tag)
    init_ckpt = out_dir / "net_epoch_bestval_at23.pth"

    # Skip if post-init ckpt already exists (resume)
    if any(int(re.search(r"epoch(?:_bestval_at)?(\d+)", p.name).group(1)) > 23
           for p in out_dir.glob("net_epoch*.pth")
           if re.search(r"epoch(?:_bestval_at)?(\d+)", p.name)):
        print(f"[{tag}] post-init ckpt exists, skip prune (resume mode)")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "tools/structural_prune_pyramid.py",
           "--orig-dir", BASELINE,
           "--out-dir", str(out_dir),
           "--num-filters-new", ",".join(str(x) for x in nf),
           "--width-per-group", str(WPG),
           "--groups", str(GROUPS)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL)}
    print(f"[{tag}] pruning {nf} wpg={WPG} groups={GROUPS} ...")
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not init_ckpt.exists():
        raise RuntimeError(
            f"[{tag}] pruning FAILED (rc={r.returncode}):\n"
            f"STDOUT: {r.stdout[-1500:]}\nSTDERR: {r.stderr[-1500:]}"
        )
    print(f"[{tag}] prune OK → {init_ckpt}")
    return out_dir


def flatten_init_ckpt(out_dir: Path, tag: str):
    """CLAUDE.md §〇.5 trap: unwrap {'model_state_dict':...} → flat state_dict."""
    init_ckpt = out_dir / "net_epoch_bestval_at23.pth"
    if not init_ckpt.exists():
        print(f"[{tag}] no init ckpt to flatten (already replaced by training)")
        return
    sd = torch.load(init_ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        torch.save(sd["model_state_dict"], init_ckpt)
        print(f"[{tag}] ✅ flattened init ckpt (unwrapped model_state_dict key)")
    else:
        print(f"[{tag}] init ckpt already flat (no unwrap needed)")


def patch_epoches(out_dir: Path, tag: str):
    """Set epoches in config.yaml to EPOCHES to match stage_a training budget."""
    cfg = out_dir / "config.yaml"
    s = cfg.read_text()
    s2 = re.sub(r"epoches:\s*\d+", f"epoches: {EPOCHES}", s)
    if s2 != s:
        cfg.write_text(s2)
        print(f"[{tag}] patched config epoches → {EPOCHES}")
    else:
        print(f"[{tag}] config epoches already {EPOCHES}")


def launch_finetune(out_dir: Path, tag: str, gpu: int) -> int:
    """Launch HEAL train_ddp.py detached. Returns PID."""
    cfg = out_dir / "config.yaml"
    log = REPO / f"results/b2_{tag}_finetune.log"
    port = MASTER_PORT_BASE + gpu
    cmd = [PY, "-m", "torch.distributed.launch",
           "--nproc_per_node=1", "--use_env",
           f"--master_port={port}",
           str(HEAL / "opencood/tools/train_ddp.py"),
           "--hypes_yaml", str(cfg),
           "--model_dir", str(out_dir),
           "--half"]
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL)}
    lf = open(log, "w")
    p = subprocess.Popen(cmd, cwd=HEAL, env=env,
                         stdout=lf, stderr=subprocess.STDOUT,
                         start_new_session=True)   # detached: survives terminal close
    print(f"[{tag}] finetune launched GPU{gpu} pid={p.pid} port={port} log={log.name}")
    return p.pid


def run_pipeline_verify():
    """Pipeline trust gate: eval existing pruned25 via HEAL inference.py on DAIR val."""
    print("\n=== Pipeline Verification: eval pruned25 via HEAL inference.py ===")
    log = REPO / "results/b2_verify_pruned25.log"

    # Remove stale eval yaml to force re-run
    eval_yaml = VERIFY_DIR / "eval_intermediate.yaml"
    if eval_yaml.exists():
        eval_yaml.unlink()
        print(f"  removed stale {eval_yaml.name}")

    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": "0",
           "PYTHONPATH": str(HEAL)}
    cmd = [PY, "opencood/tools/inference.py",
           "--model_dir", str(VERIFY_DIR),
           "--fusion_method", "intermediate"]
    print(f"  launching HEAL inference.py on GPU 0, log={log.name}")
    lf = open(log, "w")
    p = subprocess.Popen(cmd, cwd=HEAL, env=env,
                         stdout=lf, stderr=subprocess.STDOUT)
    print(f"  verify pid={p.pid} (blocking, ~10-20 min)...")
    rc = p.wait()
    lf.close()

    # Parse result
    result = {"rc": rc, "ckpt_dir": str(VERIFY_DIR)}
    if eval_yaml.exists():
        import yaml
        y = yaml.safe_load(eval_yaml.read_text()) or {}
        ap = {
            "ap30": y.get("ap30") or y.get("ap_30"),
            "ap50": y.get("ap_50") or y.get("ap50"),
            "ap70": y.get("ap_70") or y.get("ap70"),
        }
        result["ap"] = ap
        print(f"  pruned25 HEAL inference.py: AP30={ap.get('ap30')} AP50={ap.get('ap50')} AP70={ap.get('ap70')}")
        print(f"  Expected (stage_a TRT FP16): AP70=0.5905")
        ap70 = ap.get("ap70")
        if ap70 is not None and abs(float(ap70) - 0.5905) <= 0.02:
            print("  ✅ Pipeline OK: AP70 within ±0.02 of stage_a anchor")
        elif ap70 is not None:
            print(f"  ⚠️  AP70 offset = {float(ap70)-0.5905:+.4f} (may be eval pipeline diff: HEAL infer.py vs TRT FP16)")
    else:
        print(f"  ⚠️  eval_intermediate.yaml not written (rc={rc}) — check {log.name}")
        # Try parsing stdout
        log_txt = log.read_text()
        for line in log_txt.split("\n"):
            if "AP" in line.upper() or "ap" in line:
                print(f"  stdout: {line.strip()}")

    return result


def main():
    print(f"=== B2 AP70 Calibration Finetune Launcher ===")
    print(f"REPO: {REPO}")
    print(f"Training budget: epoches={EPOCHES} (init@23 → epoch {EPOCHES}, {EPOCHES-23} epochs)")

    # Step 1: Pipeline verification (blocking, ~10-20 min)
    verify_result = run_pipeline_verify()

    # Step 2: Prune all calibration configs (synchronous, fast)
    print("\n=== Step 2: Pruning calibration configs ===")
    out_dirs = {}
    for tag, nf, gpu in CALIBRATION:
        try:
            out_dir = run_prune(tag, nf, gpu)
            flatten_init_ckpt(out_dir, tag)
            patch_epoches(out_dir, tag)
            out_dirs[tag] = out_dir
        except Exception as e:
            print(f"[{tag}] ERROR in prune: {e}")

    # Step 3: Launch all finetune jobs (detached, parallel on 4 GPUs)
    print("\n=== Step 3: Launching finetune jobs ===")
    pids = {}
    for tag, nf, gpu in CALIBRATION:
        if tag not in out_dirs:
            print(f"[{tag}] SKIP: prune failed")
            continue
        try:
            pid = launch_finetune(out_dirs[tag], tag, gpu)
            pids[tag] = {
                "pid": pid,
                "gpu": gpu,
                "num_filters": nf,
                "out_dir": str(out_dirs[tag]),
                "epoches": EPOCHES,
                "wpg": WPG,
                "groups": GROUPS,
            }
        except Exception as e:
            print(f"[{tag}] ERROR launching finetune: {e}")

    # Save PID manifest
    manifest = {
        "pipeline_verify": verify_result,
        "finetune_jobs": pids,
        "stage_a_anchors": {
            "base":     {"s": [64,128,256], "ap70": 0.6309, "source": "stage_a_ap_real.parquet fp16"},
            "pruned25": {"s": [48, 96,192], "ap70": 0.5905, "source": "stage_a_ap_real.parquet fp16"},
            "pruned50": {"s": [32, 64,128], "ap70": 0.5641, "source": "stage_a_ap_real.parquet fp16"},
            "pruned75": {"s": [16, 32, 64], "ap70": 0.5300, "source": "stage_a_ap_real.parquet fp16"},
            "pad64":    {"s": [64, 96,192], "ap70": 0.5905, "source": "W_g/P_g experiment; s0 zero-fill trap25"},
        },
        "calibration_targets": [
            {"tag": t, "s": nf, "gpu": g} for t, nf, g in CALIBRATION
        ],
    }
    out = REPO / "results/b2_finetune_pids.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"\n=== ALL LAUNCHED ===")
    print(f"PID manifest: {out}")
    for tag, info in pids.items():
        print(f"  {tag}: GPU{info['gpu']} pid={info['pid']} nf={info['num_filters']} dir={Path(info['out_dir']).name}")
    print(f"\nMonitor finetune progress:")
    for tag, info in pids.items():
        print(f"  tail -f results/b2_{tag}_finetune.log")
    print(f"\nAfter finetune (~{EPOCHES-23} epochs = several hours), run:")
    print(f"  python scripts/phase2/b2_eval_and_fit.py")


if __name__ == "__main__":
    main()
