"""Plan v5 Phase B — 2:4 structured sparsity (Ampere Tensor Core compatible).

Per plan §7:
  B.1: apply 2:4 mask on each Phase A ckpt (g8_p64/48/32/16/8)
        + finetune 1 epoch with sparsity-aware loss to recover AP
  B.2: ONNX export + TRT --sparsity=enable build
  B.3: bench sparse engines (strict GPU isolation)
  B.4: ANOVA decomp + gate G_B

Fallback: apex.contrib.sparsity not installed → use torch.nn.utils.prune
with N:M structured prune (n=2, m=4 along weight input dim).

Run:
    python scripts/phase2/plan5_phaseB_sparsity.py --mode=mask          # B.1 mask
    python scripts/phase2/plan5_phaseB_sparsity.py --mode=finetune-launch
                                                                          # B.1 finetune launch
    python scripts/phase2/plan5_phaseB_sparsity.py --mode=bench           # B.2-3 build+bench
    python scripts/phase2/plan5_phaseB_sparsity.py --mode=attribution     # B.4 gate G_B
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

REPO = Path("/home/jichengzhi/UniV2X")
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
DATA_DIR = REPO / "paper_learning" / "2. AAAI最终故事" / "data"
PHASE_A_FINETUNE_ROOT = Path("/tmp/plan5_phaseA_finetune")
PHASE_B_ROOT = Path("/tmp/plan5_phaseB_sparse")
PHASE_B_ROOT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HEAL_ROOT))

PYTHON_BIN = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
TRAIN_SCRIPT = HEAL_ROOT / "opencood/tools/train.py"

# 5 ckpts: g8 baseline (epoch 19) + 4 finetuned (epoch 27)
SPARSE_TARGETS = [
    ("p64", "/home/jichengzhi/heal_research/HEAL/opencood/logs/Pyramid_DAIR_m1_base_g8_2026_05_21_22_07_45",
     "net_epoch_bestval_at19.pth", [64, 128, 256], 19, 20, 0),
    ("p48", "/tmp/plan5_phaseA_finetune/p48", "net_epoch_bestval_at27.pth", [48, 96, 192], 27, 28, 1),
    ("p32", "/tmp/plan5_phaseA_finetune/p32", "net_epoch_bestval_at23.pth", [32, 64, 128], 23, 24, 2),
    ("p16", "/tmp/plan5_phaseA_finetune/p16", "net_epoch_bestval_at27.pth", [16, 32, 64],  27, 28, 3),
    ("p8",  "/tmp/plan5_phaseA_finetune/p8",  "net_epoch_bestval_at27.pth", [8, 16, 32],   27, 28, 4),
]

FT_EPOCHS_SPARSE = 1  # sparsity-recovery 1 epoch per plan §7.3


def make_2_4_mask(weight: torch.Tensor) -> torch.Tensor:
    """For each consecutive group of 4 along input dim, keep top-2 by abs value.
    Returns binary mask same shape as weight.

    Conv2d weight: (out, in, kH, kW). We group input dim by 4 and mask within
    each group. For grouped conv (1x1 input dim < 4 OR not divisible), apply
    on output dim instead, or skip.
    """
    if weight.ndim != 4:
        return torch.ones_like(weight)
    out_c, in_c, kH, kW = weight.shape
    flat = weight.permute(0, 2, 3, 1).reshape(-1, in_c)
    if in_c % 4 != 0 or in_c < 4:
        return torch.ones_like(weight)
    mask = torch.zeros_like(flat)
    abs_w = flat.abs()
    n_groups = in_c // 4
    for g in range(n_groups):
        s, e = g * 4, (g + 1) * 4
        topk = abs_w[:, s:e].topk(2, dim=1).indices
        for col in range(2):
            mask[torch.arange(mask.size(0)), s + topk[:, col]] = 1.0
    return mask.reshape(out_c, kH, kW, in_c).permute(0, 3, 1, 2).contiguous()


def apply_2_4_to_ckpt(in_ckpt: Path, out_ckpt: Path) -> dict:
    sd = torch.load(in_ckpt, map_location="cpu", weights_only=False)
    n_conv_masked = 0
    n_conv_skipped = 0
    masked_l1_loss = 0.0
    for k, v in sd.items():
        if k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[1] >= 4:
            if v.shape[1] % 4 != 0:
                n_conv_skipped += 1
                continue
            mask = make_2_4_mask(v)
            masked_l1_loss += float((v * (1 - mask)).abs().sum())
            sd[k] = v * mask
            n_conv_masked += 1
        elif k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 4:
            n_conv_skipped += 1
    torch.save(sd, out_ckpt)
    return {
        "n_conv_masked": n_conv_masked,
        "n_conv_skipped": n_conv_skipped,
        "masked_l1_loss": round(masked_l1_loss, 3),
        "out_size_mb": round(out_ckpt.stat().st_size / 1e6, 2),
    }


def setup_sparsity_finetune(tag: str, src_dir: Path, src_ckpt_name: str,
                            num_filters: List[int], init_epoch: int, target_epoches: int) -> dict:
    out_dir = PHASE_B_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    from opencood.hypes_yaml import yaml_utils as heal_yaml
    src_cfg = heal_yaml.load_yaml(str(src_dir / "config.yaml"))
    cfg = copy.deepcopy(src_cfg)
    cfg["model"]["args"]["fusion_backbone"]["num_filters"] = list(num_filters)
    cfg.setdefault("train_params", {})["epoches"] = target_epoches
    cfg["train_params"]["save_freq"] = 1
    cfg["train_params"]["eval_freq"] = 1
    cfg["name"] = f"plan5_phaseB_g8_{tag}_sparse24"
    cfg_path = out_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    src_ckpt = src_dir / src_ckpt_name
    seed_ckpt = out_dir / f"net_epoch{init_epoch}.pth"
    mask_info = apply_2_4_to_ckpt(src_ckpt, seed_ckpt)

    return {
        "tag": tag,
        "src_ckpt": str(src_ckpt),
        "seed_ckpt": str(seed_ckpt),
        "config_path": str(cfg_path),
        "out_dir": str(out_dir),
        "init_epoch": init_epoch,
        "target_epoches": target_epoches,
        "ft_epochs_sparse": FT_EPOCHS_SPARSE,
        "mask_info": mask_info,
    }


def launch_finetune(info: dict, gpu: int) -> dict:
    cmd = [PYTHON_BIN, str(TRAIN_SCRIPT),
           "--hypes_yaml", info["config_path"],
           "--model_dir", info["out_dir"],
           "--fusion_method", "intermediate"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{HEAL_ROOT}:{env.get('PYTHONPATH', '')}"
    log_path = Path(info["out_dir"]) / "train.log"
    pid_path = Path(info["out_dir"]) / "pid"
    log_f = open(log_path, "w")
    log_f.write(f"# Plan v5 Phase B finetune\n# tag={info['tag']} gpu={gpu}\n# cmd: {' '.join(cmd)}\n\n")
    log_f.flush()
    proc = subprocess.Popen(cmd, cwd=str(HEAL_ROOT), env=env,
                            stdout=log_f, stderr=subprocess.STDOUT,
                            start_new_session=True)
    pid_path.write_text(str(proc.pid))
    info.update({"gpu": gpu, "pid": proc.pid, "log_path": str(log_path), "status": "running"})
    return info


def mode_mask():
    print(f"[Plan v5 Phase B.1] Applying 2:4 mask to 5 ckpts ...")
    results = []
    for tag, src_dir_str, src_ckpt, nf, init_e, tgt_e, _ in SPARSE_TARGETS:
        src_dir = Path(src_dir_str)
        info = setup_sparsity_finetune(tag, src_dir, src_ckpt, nf, init_e, tgt_e)
        print(f"  {tag}: masked {info['mask_info']['n_conv_masked']} conv layers, "
              f"skipped {info['mask_info']['n_conv_skipped']}, "
              f"out_size={info['mask_info']['out_size_mb']} MB")
        results.append(info)
    status_path = DATA_DIR / "plan5_phaseB_mask_status.json"
    status_path.write_text(json.dumps({"phase": "B.1_mask", "anchors": results},
                                       ensure_ascii=False, indent=2))
    print(f"[Plan v5 Phase B.1] wrote {status_path}")


def mode_finetune_launch():
    status_path = DATA_DIR / "plan5_phaseB_mask_status.json"
    if not status_path.exists():
        print("ERROR: Phase B.1 mask not done. Run --mode=mask first.")
        return 1
    status = json.loads(status_path.read_text())
    print(f"[Plan v5 Phase B.1.b] Launching 5 sparsity finetune procs ...")
    launched = []
    for info, (tag, _, _, _, _, _, gpu) in zip(status["anchors"], SPARSE_TARGETS):
        info = launch_finetune(info, gpu)
        print(f"  {info['tag']} launched: pid={info['pid']} gpu={info['gpu']}")
        launched.append(info)
    status_path2 = DATA_DIR / "plan5_phaseB_finetune_launched.json"
    status_path2.write_text(json.dumps({"phase": "B.1_finetune", "anchors": launched},
                                        ensure_ascii=False, indent=2))
    print(f"[Plan v5 Phase B.1.b] wrote {status_path2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mask", "finetune-launch", "bench", "attribution"],
                    required=True)
    args = ap.parse_args()
    if args.mode == "mask":
        return mode_mask() or 0
    if args.mode == "finetune-launch":
        return mode_finetune_launch() or 0
    print(f"--mode={args.mode} not implemented yet")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
