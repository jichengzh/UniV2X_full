"""Plan v5 Phase A.2 — Finetune dispatcher.

For each pruned ckpt (g8_p48, p32, p16, p8):
  1. Patch g8 baseline config.yaml -> set num_filters to scaled values
     + train_params.epoches = 27 (init=19 + FT=8)
  2. Create model_dir under /tmp/plan5_phaseA_finetune/<tag>/
  3. Copy pruned ckpt -> <model_dir>/net_epoch19.pth (so HEAL load_saved_model
     scans dir and resumes from epoch 19)
  4. Launch HEAL opencood/tools/train.py with --hypes_yaml --model_dir on a
     dedicated GPU (CUDA_VISIBLE_DEVICES=N)
  5. Stream bg log to <model_dir>/train.log

Run:
    python scripts/phase2/plan5_phaseA_finetune_dispatcher.py [--launch]
    # Without --launch: dry-run (only generates configs, dirs, ckpts).
    # With --launch: also kicks off 4 bg training processes.
Output:
    /tmp/plan5_phaseA_finetune/<tag>/{config.yaml, net_epoch19.pth, train.log, pid}
    paper_learning/2. AAAI最终故事/data/plan5_phaseA_finetune_status.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

sys.path.insert(0, "/home/jichengzhi/heal_research/HEAL")
from opencood.hypes_yaml import yaml_utils as heal_yaml_utils  # noqa: E402

REPO = Path("/home/jichengzhi/UniV2X")
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
DATA_DIR = REPO / "paper_learning" / "2. AAAI最终故事" / "data"
PRUNED_DIR = Path("/tmp/plan5_phaseA_ckpts")
FINETUNE_ROOT = Path("/tmp/plan5_phaseA_finetune")
FINETUNE_ROOT.mkdir(parents=True, exist_ok=True)

BASELINE_CONFIG = HEAL_ROOT / "opencood/logs/Pyramid_DAIR_m1_base_g8_2026_05_21_22_07_45/config.yaml"

PLANES: List[Tuple[str, float, List[int], int]] = [
    ("p48", 0.75,    [48, 96, 192], 0),
    ("p32", 0.50,    [32, 64, 128], 1),
    ("p16", 0.25,    [16, 32, 64],  2),
    ("p8",  0.125,   [8,  16, 32],  3),
]

INIT_EPOCH = 19   # g8 baseline best epoch
FT_EPOCHS = 8     # locked by C1
TARGET_EPOCHES = INIT_EPOCH + FT_EPOCHS   # 27

PYTHON_BIN = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
TRAIN_SCRIPT = HEAL_ROOT / "opencood/tools/train.py"


def load_baseline_config() -> dict:
    return heal_yaml_utils.load_yaml(str(BASELINE_CONFIG))


def patch_config_for_plane(base_cfg: dict, num_filters: List[int], tag: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    fb = cfg["model"]["args"]["fusion_backbone"]
    fb["num_filters"] = list(num_filters)

    cfg.setdefault("train_params", {})
    cfg["train_params"]["epoches"] = TARGET_EPOCHES
    if "save_freq" not in cfg["train_params"]:
        cfg["train_params"]["save_freq"] = 2
    if "eval_freq" not in cfg["train_params"]:
        cfg["train_params"]["eval_freq"] = 1

    cfg.setdefault("name", f"plan5_phaseA_g8_{tag}_ft8")
    return cfg


def setup_one_plane(tag: str, factor: float, num_filters: List[int], gpu: int) -> dict:
    out_dir = FINETUNE_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_baseline_config()
    cfg = patch_config_for_plane(base_cfg, num_filters, tag)
    cfg_path = out_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    pruned_ckpt = PRUNED_DIR / f"g8_{tag}_pruned_unfinetuned.pth"
    seed_ckpt = out_dir / f"net_epoch{INIT_EPOCH}.pth"
    if not seed_ckpt.exists():
        shutil.copy(pruned_ckpt, seed_ckpt)

    log_path = out_dir / "train.log"
    pid_path = out_dir / "pid"

    info = {
        "tag": tag,
        "factor": factor,
        "num_filters": num_filters,
        "gpu": gpu,
        "model_dir": str(out_dir),
        "config_path": str(cfg_path),
        "seed_ckpt": str(seed_ckpt),
        "log_path": str(log_path),
        "pid_path": str(pid_path),
        "init_epoch": INIT_EPOCH,
        "target_epoches": TARGET_EPOCHES,
        "ft_epochs": FT_EPOCHS,
        "status": "configured",
    }
    return info


def launch_one_plane(info: dict) -> dict:
    cmd = [
        PYTHON_BIN, str(TRAIN_SCRIPT),
        "--hypes_yaml", info["config_path"],
        "--model_dir", info["model_dir"],
        "--fusion_method", "intermediate",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(info["gpu"])
    env.setdefault("PYTHONPATH", str(HEAL_ROOT))
    if str(HEAL_ROOT) not in env["PYTHONPATH"]:
        env["PYTHONPATH"] = f"{HEAL_ROOT}:{env['PYTHONPATH']}"

    log_f = open(info["log_path"], "w")
    log_f.write(f"# Plan v5 Phase A.2 finetune dispatcher\n")
    log_f.write(f"# tag={info['tag']} gpu={info['gpu']} init_epoch={info['init_epoch']} target={info['target_epoches']}\n")
    log_f.write(f"# cmd: {' '.join(cmd)}\n")
    log_f.write(f"# env CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n\n")
    log_f.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(HEAL_ROOT),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    Path(info["pid_path"]).write_text(str(proc.pid))
    info["pid"] = proc.pid
    info["status"] = "running"
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true",
                    help="Actually launch 4 bg training procs (otherwise dry-run setup only)")
    args = ap.parse_args()

    if not BASELINE_CONFIG.exists():
        print(f"ERROR: baseline config not found: {BASELINE_CONFIG}")
        return 1
    if not TRAIN_SCRIPT.exists():
        print(f"ERROR: HEAL train.py not found: {TRAIN_SCRIPT}")
        return 1
    for tag, _, _, _ in PLANES:
        ckpt = PRUNED_DIR / f"g8_{tag}_pruned_unfinetuned.pth"
        if not ckpt.exists():
            print(f"ERROR: pruned ckpt not found: {ckpt}")
            return 1

    print(f"[Plan v5 Phase A.2] Setting up 4 plane finetune dirs ...")
    infos = []
    for tag, factor, nf, gpu in PLANES:
        info = setup_one_plane(tag, factor, nf, gpu)
        infos.append(info)
        print(f"  {tag} (factor={factor:.4f}, nf={nf}, gpu={gpu}): {info['model_dir']}")

    if args.launch:
        print(f"\n[Plan v5 Phase A.2] Launching 4 bg train procs (GPU 0-3) ...")
        for info in infos:
            info = launch_one_plane(info)
            print(f"  {info['tag']} launched: pid={info['pid']}, log={info['log_path']}")
    else:
        print(f"\n[Plan v5 Phase A.2] Dry-run only. Re-run with --launch to start training.")

    status_path = DATA_DIR / "plan5_phaseA_finetune_status.json"
    status_payload = {
        "phase": "phase_A.2",
        "init_epoch": INIT_EPOCH,
        "target_epoches": TARGET_EPOCHES,
        "ft_epochs_locked": FT_EPOCHS,
        "launched": args.launch,
        "anchors": infos,
    }
    status_path.write_text(json.dumps(status_payload, ensure_ascii=False, indent=2))
    print(f"\n[Plan v5 Phase A.2] wrote {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
