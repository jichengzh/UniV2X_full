"""B2 completion check + auto-trigger eval+fit.

Run this to check if all 4 finetunes are done.
If done, automatically runs b2_eval_and_fit.py.
If not done, prints current progress.

Usage:
  cd /home/jichengzhi/V2X
  PYTHONPATH=/home/jichengzhi/heal_research/HEAL \\
  /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \\
    scripts/phase2/b2_check_and_eval.py [--eval-now]

  --eval-now: run eval+fit even if some finetunes aren't done (skip missing)
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/jichengzhi/V2X")
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
CKROOT = Path("/home/jichengzhi/heal_research/checkpoints/stage1")
TARGET_EPOCH = 31

CONFIGS = [
    ("iso_s0",  [48, 128, 256], "calibration"),
    ("iso_s1",  [64,  96, 256], "calibration"),
    ("iso_s2",  [64, 128, 192], "calibration"),
    ("mixed",   [32,  96, 192], "validation"),
]


def get_max_epoch(ft_dir: Path) -> int | None:
    """Return highest epoch number from any ckpt in ft_dir."""
    best_epoch = None
    for p in ft_dir.glob("net_epoch*.pth"):
        m = re.search(r"epoch(?:_bestval_at)?(\d+)", p.name)
        if m:
            e = int(m.group(1))
            if best_epoch is None or e > best_epoch:
                best_epoch = e
    return best_epoch


def main():
    eval_now = "--eval-now" in sys.argv

    print("=== B2 Finetune Completion Check ===")
    print(f"Target epoch: {TARGET_EPOCH}\n")

    all_done = True
    for tag, nf, role in CONFIGS:
        ft_dir = CKROOT / f"Pyramid_DAIR_m1_b2_{tag}_2026_06_20"
        if not ft_dir.exists():
            print(f"  [{tag}] ❌ DIR NOT FOUND: {ft_dir}")
            all_done = False
            continue

        max_ep = get_max_epoch(ft_dir)
        if max_ep is None:
            print(f"  [{tag}] ⏳ No ckpt yet (dir exists, training starting)")
            all_done = False
        elif max_ep < TARGET_EPOCH:
            print(f"  [{tag}] ⏳ epoch {max_ep}/{TARGET_EPOCH} (training in progress)")
            all_done = False
        else:
            best_ckpts = sorted(ft_dir.glob("net_epoch_bestval_at*.pth"),
                                key=lambda p: int(re.search(r"at(\d+)", p.name).group(1)))
            best = best_ckpts[-1].name if best_ckpts else "?"
            print(f"  [{tag}] ✅ epoch {max_ep} DONE, best={best}")

    print()
    if all_done:
        print("✅ All 4 finetunes complete! Launching b2_eval_and_fit.py...")
        cmd = [PY, "scripts/phase2/b2_eval_and_fit.py"]
        env = {**os.environ, "PYTHONPATH": str(HEAL)}
        subprocess.run(cmd, cwd=REPO, env=env)
    elif eval_now:
        print("⚠️  --eval-now: running eval+fit on completed configs only")
        cmd = [PY, "scripts/phase2/b2_eval_and_fit.py"]
        env = {**os.environ, "PYTHONPATH": str(HEAL)}
        subprocess.run(cmd, cwd=REPO, env=env)
    else:
        print(f"⏳ Finetune still running. Re-run this script to check again.")
        print(f"   Or run with --eval-now to eval completed configs immediately.")
        print(f"\nMonitor logs:")
        for tag, _, _ in CONFIGS:
            print(f"  tail -f results/b2_{tag}_finetune.log | grep -E 'epoch|bestval|Loss'")


if __name__ == "__main__":
    main()
