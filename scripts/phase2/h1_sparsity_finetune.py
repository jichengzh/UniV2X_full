"""H1 — 2:4 Structured Sparsity: mask + DAIR finetune launcher.

Steps:
1. Apply 2:4 mask to Pyramid_DAIR_m1_base ckpt weights
2. Setup DAIR finetune dir with modified config (epoches += 2)
3. Launch HEAL training for 2 sparse-recovery epochs

After finetune completes, run h1_sparsity_build_bench.py to:
4. Export ONNX from sparse ckpt
5. Build TRT engines (FP16 + INT8) with --sparsity=enable
6. AP eval on DAIR val 1789
7. Latency + energy benchmark

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/phase2/h1_sparsity_finetune.py [--launch]

纪律:
 - 2:4 mask 是真实 weight 修改,非 SPARSE_WEIGHTS flag noop
 - finetune 用 DAIR 数据集 (not OPV2V)
 - AP 必须真测 (不能复用 stage_a)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
CKPT_ROOT = Path("/home/jichengzhi/heal_research/checkpoints/stage1")
PYTHON_BIN = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
TRAIN_SCRIPT = HEAL_ROOT / "opencood/tools/train.py"

# Gold standard DAIR base ckpt
BASE_CKPT_DIR = CKPT_ROOT / "Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
BASE_CKPT_NAME = "net_epoch_bestval_at23.pth"
BASE_EPOCH = 23
SPARSE_FT_EPOCHS = 2   # sparse recovery: train 2 epochs from epoch 23 → 24,25
TARGET_EPOCH = BASE_EPOCH + SPARSE_FT_EPOCHS   # = 25

# Output directory for sparse finetune
SPARSE_CKPT_DIR = CKPT_ROOT / "Pyramid_DAIR_m1_base_sparse24_2026_06_03"

sys.path.insert(0, str(HEAL_ROOT))


# ── 2:4 mask ───────────────────────────────────────────────────────────────

def make_2_4_mask(weight: torch.Tensor) -> torch.Tensor:
    """For each consecutive group of 4 along input dim, keep top-2 by abs value."""
    if weight.ndim != 4:
        return torch.ones_like(weight)
    out_c, in_c, kH, kW = weight.shape
    if in_c < 4 or in_c % 4 != 0:
        return torch.ones_like(weight)
    flat = weight.permute(0, 2, 3, 1).reshape(-1, in_c)
    mask = torch.zeros_like(flat)
    abs_w = flat.abs()
    for g in range(in_c // 4):
        s, e = g * 4, (g + 1) * 4
        topk = abs_w[:, s:e].topk(2, dim=1).indices
        for col in range(2):
            mask[torch.arange(mask.size(0)), s + topk[:, col]] = 1.0
    return mask.reshape(out_c, kH, kW, in_c).permute(0, 3, 1, 2).contiguous()


def apply_2_4_to_ckpt(src_ckpt: Path, dst_ckpt: Path) -> dict:
    """Apply 2:4 mask in-place and save to dst_ckpt."""
    print(f"[2:4 mask] loading {src_ckpt.name}")
    raw = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    # Handle wrapped format
    sd = raw.get("model_state_dict", raw)

    n_masked = 0
    n_skipped = 0
    total_masked_l1 = 0.0
    for k, v in sd.items():
        if k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 4:
            out_c, in_c, kH, kW = v.shape
            if in_c >= 4 and in_c % 4 == 0:
                mask = make_2_4_mask(v)
                total_masked_l1 += float((v * (1 - mask)).abs().sum())
                sd[k] = v * mask
                n_masked += 1
            else:
                n_skipped += 1

    torch.save(sd, dst_ckpt)
    info = {
        "n_masked": n_masked,
        "n_skipped": n_skipped,
        "masked_l1_loss": round(total_masked_l1, 3),
        "dst_size_mb": round(dst_ckpt.stat().st_size / 1e6, 2),
    }
    print(f"[2:4 mask] masked={n_masked} skipped={n_skipped} "
          f"l1_loss={info['masked_l1_loss']:.3f} → {dst_ckpt.name} ({info['dst_size_mb']:.1f}MB)")
    return info


# ── Setup finetune dir ──────────────────────────────────────────────────────

def setup_finetune(gpu: int) -> dict:
    """Create sparse finetune dir, apply mask, set config, return launch info."""
    SPARSE_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and modify config
    from opencood.hypes_yaml import yaml_utils as heal_yaml
    src_cfg = heal_yaml.load_yaml(str(BASE_CKPT_DIR / "config.yaml"))
    cfg = copy.deepcopy(src_cfg)

    # Extend to target epoch
    cfg.setdefault("train_params", {})["epoches"] = TARGET_EPOCH
    cfg["train_params"]["save_freq"] = 1
    cfg["train_params"]["eval_freq"] = 1
    cfg["name"] = "Pyramid_DAIR_m1_base_sparse24"

    cfg_path = SPARSE_CKPT_DIR / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
    print(f"[setup] config written → {cfg_path}")

    # Step 2: Apply 2:4 mask and save as net_epoch23.pth (HEAL resumes from this)
    src_ckpt = BASE_CKPT_DIR / BASE_CKPT_NAME
    seed_ckpt = SPARSE_CKPT_DIR / f"net_epoch{BASE_EPOCH}.pth"

    if seed_ckpt.exists():
        print(f"[setup] seed ckpt already exists: {seed_ckpt}")
        mask_info = {"note": "pre-existing, not re-masked"}
    else:
        mask_info = apply_2_4_to_ckpt(src_ckpt, seed_ckpt)

    info = {
        "model_dir": str(SPARSE_CKPT_DIR),
        "config_path": str(cfg_path),
        "seed_ckpt": str(seed_ckpt),
        "base_epoch": BASE_EPOCH,
        "target_epoches": TARGET_EPOCH,
        "ft_epochs": SPARSE_FT_EPOCHS,
        "gpu": gpu,
        "mask_info": mask_info,
    }

    # Save setup info
    setup_json = REPO / "results" / "H1_sparsity_setup.json"
    with open(setup_json, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[setup] info → {setup_json}")

    return info


def launch_finetune(info: dict) -> int:
    """Launch HEAL training, return PID."""
    cmd = [
        PYTHON_BIN, str(TRAIN_SCRIPT),
        "--hypes_yaml", info["config_path"],
        "--model_dir", info["model_dir"],
        "--fusion_method", "intermediate",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(info["gpu"])
    env["PYTHONPATH"] = f"{HEAL_ROOT}:{env.get('PYTHONPATH', '')}"

    log_path = SPARSE_CKPT_DIR / "train.log"
    log_f = open(log_path, "w")
    log_f.write(f"# H1 2:4 sparsity finetune\n# gpu={info['gpu']}\n# cmd: {' '.join(cmd)}\n\n")
    log_f.flush()

    proc = subprocess.Popen(
        cmd, cwd=str(HEAL_ROOT), env=env,
        stdout=log_f, stderr=subprocess.STDOUT,
    )
    pid = proc.pid
    print(f"[launch] finetune started PID={pid} log={log_path}")

    # Save PID
    with open(SPARSE_CKPT_DIR / "pid", "w") as f:
        f.write(str(pid))

    return pid


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true",
                        help="Actually launch finetune (default: dry-run only)")
    parser.add_argument("--gpu", type=int,
                        default=int(os.environ.get("CUDA_VISIBLE_DEVICES", "4").split(",")[0]),
                        help="GPU index to use (physical)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"H1 — 2:4 Structured Sparsity + DAIR Finetune")
    print(f"GPU: {args.gpu} | ft_epochs: {SPARSE_FT_EPOCHS} | target: epoch {TARGET_EPOCH}")
    print(f"{'='*60}\n")

    # Check GPU idle before launching
    if args.launch:
        out = subprocess.check_output([
            "nvidia-smi", f"--id={args.gpu}",
            "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"
        ]).decode().strip()
        if out:
            print(f"[warn] GPU {args.gpu} has existing apps: {out}")
            print("  → Finetune may compete with existing processes")
        else:
            print(f"[gate] GPU {args.gpu} idle ✓")

    # Setup (always runs)
    info = setup_finetune(args.gpu)

    print(f"\n[summary]")
    print(f"  model_dir: {info['model_dir']}")
    print(f"  seed_ckpt: {info['seed_ckpt']}")
    print(f"  target_epoches: {info['target_epoches']}")
    print(f"  mask_info: {info['mask_info']}")

    if not args.launch:
        print("\n[dry-run] Pass --launch to start finetune.")
        print(f"  Manual command:")
        print(f"  CUDA_VISIBLE_DEVICES={args.gpu} {PYTHON_BIN} {TRAIN_SCRIPT} \\")
        print(f"    --hypes_yaml {info['config_path']} \\")
        print(f"    --model_dir {info['model_dir']} \\")
        print(f"    --fusion_method intermediate")
        return

    pid = launch_finetune(info)
    print(f"\n[H1] Finetune launched PID={pid}")
    print(f"  Log: {SPARSE_CKPT_DIR}/train.log")
    print(f"  Expected ~10-20h. Check: tail -f {SPARSE_CKPT_DIR}/train.log")
    print(f"\n[H1] Next step after finetune:")
    print(f"  python scripts/phase2/h1_sparsity_build_bench.py  # build TRT + bench + AP eval")


if __name__ == "__main__":
    main()
