"""Gap 2 — [32,64,136] (prune50_b2 / dla_xproc_032_064_136) AP finetune launcher.

There is NO finetuned ckpt for backbone planes [32,64,136] (D-space only built
random-weight / latency engines). This launches a detached finetune from the
FLAT pruned init (built via tools/wholenet_prune_pyramid.py, backbone-only:
num_filters=[32,64,136], num_upsample=[128,128,128], shrink=256 — exactly the
D-space architecture) resuming from base epoch 23 -> epoch 48 (~25 epochs, like
cliff3) on DAIR. After convergence, eval gives the REAL AP for this arch.

GPU: GPU 2 ONLY (verified idle). Completion detection: marker file with child
rc (NO pgrep — avoids watcher cmdline self-match bug). Same pattern as
scripts/phase2/ap_cliff3_finetune.py.
"""
from __future__ import annotations
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
CKDIR = Path("/home/jichengzhi/heal_research/checkpoints/stage1/"
             "Pyramid_DAIR_m1_prune50b2_032_064_136_2026_06_03")
GPU = "2"
TAG = "prune50b2_032_064_136"


def main():
    cfg = CKDIR / "config.yaml"
    log = REPO / f"results/p2_{TAG}_finetune.log"
    marker = REPO / f"results/p2_{TAG}.done"
    if marker.exists():
        marker.unlink()
    train = (
        f"{PY} -m torch.distributed.launch --nproc_per_node=1 --use_env "
        f"--master_port=29850 "
        f"{HEAL / 'opencood/tools/train_ddp.py'} "
        f"--hypes_yaml {cfg} --model_dir {CKDIR} --half "
        f"> {log} 2>&1"
    )
    chain_sh = (
        f'echo "[ft] start {TAG} $(date)"; '
        f'{train}; rc=$?; '
        f'echo "tag={TAG} rc=$rc finished=$(date +%s)" > {marker}; '
        f'echo "[ft] done {TAG} rc=$rc $(date)"'
    )
    chain_log = REPO / f"results/p2_{TAG}_chain.log"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "PYTHONPATH": str(HEAL)}
    lf = open(chain_log, "w")
    p = subprocess.Popen(["bash", "-c", chain_sh], cwd=HEAL, env=env,
                         stdout=lf, stderr=subprocess.STDOUT,
                         start_new_session=True)
    info = {
        "pid": p.pid, "gpu": GPU, "tag": TAG,
        "ckpt_dir": str(CKDIR), "marker": str(marker),
        "finetune_log": str(log), "chain_log": str(chain_log),
        "launched_at": int(time.time()),
    }
    (REPO / f"results/p2_{TAG}_pids.json").write_text(json.dumps(info, indent=2))
    print(f"[ft] launched GPU{GPU} pid={p.pid} log={log} marker={marker}")


if __name__ == "__main__":
    main()
