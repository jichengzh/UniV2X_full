"""方案 3 — 延长 4 个自训 ckpt fine-tune (epoch 47 → 78, 即 +30 epoch).

策略:
  - 复用现有 ft_*/config.yaml (weight_decay=1e-4 已配置)
  - 把 epoches 48 → 78, HEAL train.py 自动从最新 ckpt (epoch 47) 恢复
  - 当前 LR schedule (multistep at [15, 30], gamma=0.1) 已让 epoch 47 时 lr=2e-5,
    后续 30 epoch 走极低 LR, 让 BN running stats 充分收敛
  - 4 GPU 并行, ~6-10h wall

每个 triplet 跑完后, 后续脚本可:
  1. 用 dump_activation_histograms.py 检查 shrink_conv p99 是否回到 ~1.2
  2. 重 export e2e ONNX + INT8 engine, 跑 AP 验证 entropy 是否恢复

启动: 后台 launch 4 job → log 到 logs/extend_ft_v2/train_{sig}.log
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"

A_CACHE = REPO_ROOT / "models/dataset_a_cache"
LOG_DIR = REPO_ROOT / "logs/extend_ft_v2"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_EPOCHES = 78  # 47 (latest existing) + 31 epoch 延长

# (sig, gpu)
TRIPLETS = [
    ("040_080_160", 0),  # T3_p37
    ("024_056_128", 1),  # T5_p62
    ("048_064_128", 2),  # T7_wide_shallow
    ("024_048_192", 3),  # T8_narrow_deep
]


def patch_config_epoches(cfg_path: Path, new_epoches: int):
    txt = cfg_path.read_text()
    patched = re.sub(r"^(\s*epoches:\s*)\d+", rf"\g<1>{new_epoches}", txt, flags=re.M)
    if patched != txt:
        cfg_path.write_text(patched)
        print(f"  [patch] {cfg_path.name}: epoches → {new_epoches}")
    else:
        print(f"  [patch] {cfg_path.name}: no change (already {new_epoches}?)")


def launch_one(sig: str, gpu: int):
    ft_dir = A_CACHE / f"ft_{sig}"
    cfg_path = ft_dir / "config.yaml"
    if not cfg_path.exists():
        print(f"  [skip] {sig}: no config.yaml")
        return None

    patch_config_epoches(cfg_path, TARGET_EPOCHES)

    log_path = LOG_DIR / f"train_{sig}.log"
    cmd = [PYTHON, "-m", "torch.distributed.launch",
           "--nproc_per_node=1", "--use_env",
           f"--master_port={29600 + gpu}",
           str(HEAL_ROOT / "opencood/tools/train_ddp.py"),
           "--hypes_yaml", str(cfg_path),
           "--model_dir", str(ft_dir),
           "--half"]
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL_ROOT)}
    print(f"  [launch] GPU{gpu} {sig} → log {log_path}")
    lf = open(log_path, "w")
    p = subprocess.Popen(cmd, cwd=HEAL_ROOT, env=env,
                        stdout=lf, stderr=subprocess.STDOUT,
                        start_new_session=True)
    return {"sig": sig, "gpu": gpu, "pid": p.pid, "log": str(log_path)}


def main():
    print("=" * 78)
    print(f"方案 3 — 延长 ft training: epoches → {TARGET_EPOCHES} on 4 GPUs")
    print("=" * 78)
    for sig, gpu in TRIPLETS:
        print(f"  T_{sig} → GPU{gpu}")
    print()

    handles = []
    for sig, gpu in TRIPLETS:
        h = launch_one(sig, gpu)
        if h is not None:
            handles.append(h)
        time.sleep(2)  # stagger startup

    print()
    print("=" * 78)
    print(f"launched {len(handles)} jobs:")
    for h in handles:
        print(f"  GPU{h['gpu']}  sig={h['sig']}  pid={h['pid']}  log={h['log']}")
    print()
    print("monitor:  tail -f logs/extend_ft_v2/train_*.log")
    print("kill:     pkill -f train_ddp.py  (或 kill <pid>)")
    print()
    print("ETA: ~6-10h wall (4 ckpt × 30 epoch each, parallel)")


if __name__ == "__main__":
    main()
