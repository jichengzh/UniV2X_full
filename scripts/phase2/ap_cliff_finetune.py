"""P0 — "找 AP 悬崖": 把 backbone L1 结构化剪枝延伸到 75% 以上, 找 AP 骤崩点.

同口径 stage_a 金标准 (backbone L1 结构化剪枝 + 收敛 finetune + DAIR val 1789 eval):
    base     [64,128,256]  AP50 0.791
    pruned25 [48, 96,192]  AP50 0.777
    pruned50 [32, 64,128]  AP50 0.764
    pruned75 [16, 32, 64]  AP50 0.757   <- 高原边缘
    --- 本脚本延伸 ---
    prune85  [10, 20, 40]  ?
    prune90  [ 6, 13, 26]  ?
    prune95  [ 4,  6, 13]  ?   (stage0=3 会令 width=int(3*16/64)*32=0 崩, 抬到 4)

★ width=0 陷阱 (项目记忆): HEAL Bottleneck width = int(planes*wpg/64)*groups.
  默认 wpg=4 + groups=32 时 planes<16 → width=0 崩. 本脚本统一 wpg=16 防崩.
★ ckpt flat 陷阱 (CLAUDE.md §〇#5): structural_prune_pyramid.py 存的是
  {"model_state_dict":...} 包裹格式; HEAL load_saved_model 直接 load_state_dict
  不解包 → 全 key missing → 从随机权重 finetune (AP 崩).
  本脚本在 prune 后立即把 init ckpt 转成 flat state_dict.

流程 (每档):
  1. structural_prune_pyramid.py --num-filters-new s0,s1,s2 --width-per-group 16
     → out_dir/net_epoch_bestval_at23.pth (wrapped) + config.yaml
  2. flatten init ckpt (解 model_state_dict 包裹) [防陷阱1]
  3. patch config epoches=48 (init@23 → finetune 25 epoch, 收敛)
  4. HEAL train_ddp.py --half (AMP) detached 后台启动, GPU 0/1/2 各一档

本脚本只负责 prune+flatten (同步, 快) 再 detached 启动 finetune, 不等训练完.
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path("/home/jichengzhi/UniV2X")
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
BASELINE = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
CKROOT = Path("/home/jichengzhi/heal_research/checkpoints/stage1")
EPOCHES = 48          # init@23 → 25 epoch finetune (golden dataset_a budget)
WPG = 16              # 防 width=0 崩
GROUPS = 32

# (tag, num_filters, gpu)
CANDIDATES = [
    ("prune85", [10, 20, 40], 0),
    ("prune90", [6, 13, 26], 1),
    ("prune95", [4, 6, 13], 2),
]


def out_dir_for(tag: str) -> Path:
    return CKROOT / f"Pyramid_DAIR_m1_{tag}_2026_06_02"


def run_prune(tag: str, nf: list[int], gpu: int) -> Path:
    out_dir = out_dir_for(tag)
    init_ckpt = out_dir / "net_epoch_bestval_at23.pth"
    if list(out_dir.glob("net_epoch*.pth")) and any(
        int(re.search(r"epoch(?:_bestval_at)?(\d+)", p.name).group(1)) > 23
        for p in out_dir.glob("net_epoch*.pth")
        if re.search(r"epoch(?:_bestval_at)?(\d+)", p.name)
    ):
        print(f"[{tag}] post-init ckpt exists, skip prune (resume)")
        return out_dir
    cmd = [PY, "tools/structural_prune_pyramid.py",
           "--orig-dir", BASELINE,
           "--out-dir", str(out_dir),
           "--num-filters-new", ",".join(str(x) for x in nf),
           "--width-per-group", str(WPG),
           "--groups", str(GROUPS)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    print(f"[{tag}] prune {nf} wpg={WPG} ...")
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not init_ckpt.exists():
        raise RuntimeError(f"[{tag}] prune FAILED:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    print(f"[{tag}] prune OK -> {init_ckpt.name}")
    return out_dir


def flatten_init_ckpt(out_dir: Path, tag: str):
    """防陷阱1: 把 {"model_state_dict":...} 包裹格式转成 flat state_dict。"""
    init_ckpt = out_dir / "net_epoch_bestval_at23.pth"
    if not init_ckpt.exists():
        # already finetuning (init renamed/overwritten); nothing to flatten
        return
    sd = torch.load(init_ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        torch.save(sd["model_state_dict"], init_ckpt)
        print(f"[{tag}] flattened init ckpt (unwrapped model_state_dict)")
    else:
        print(f"[{tag}] init ckpt already flat")


def patch_epoches(out_dir: Path, tag: str):
    cfg = out_dir / "config.yaml"
    s = cfg.read_text()
    s2 = re.sub(r"epoches:\s*\d+", f"epoches: {EPOCHES}", s)
    if s2 != s:
        cfg.write_text(s2)
        print(f"[{tag}] config epoches -> {EPOCHES}")


def launch_finetune(out_dir: Path, tag: str, gpu: int) -> int:
    cfg = out_dir / "config.yaml"
    log = REPO / f"results/ap_cliff_{tag}_finetune.log"
    cmd = [PY, "-m", "torch.distributed.launch",
           "--nproc_per_node=1", "--use_env",
           f"--master_port={29600 + gpu}",
           str(HEAL / "opencood/tools/train_ddp.py"),
           "--hypes_yaml", str(cfg),
           "--model_dir", str(out_dir),
           "--half"]
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL)}
    lf = open(log, "w")
    p = subprocess.Popen(cmd, cwd=HEAL, env=env, stdout=lf,
                         stderr=subprocess.STDOUT, start_new_session=True)
    print(f"[{tag}] finetune launched GPU{gpu} pid={p.pid} log={log}")
    return p.pid


def main():
    pids = {}
    for tag, nf, gpu in CANDIDATES:
        out_dir = run_prune(tag, nf, gpu)
        flatten_init_ckpt(out_dir, tag)
        patch_epoches(out_dir, tag)
    # launch all after all prunes ready (so GPUs start ~together)
    for tag, nf, gpu in CANDIDATES:
        out_dir = out_dir_for(tag)
        pid = launch_finetune(out_dir, tag, gpu)
        pids[tag] = {"pid": pid, "gpu": gpu, "num_filters": nf,
                     "out_dir": str(out_dir)}
        time.sleep(3)
    print("\n=== LAUNCHED ===")
    for tag, info in pids.items():
        print(f"  {tag}: GPU{info['gpu']} pid={info['pid']} nf={info['num_filters']} dir={info['out_dir']}")
    import json
    (REPO / "results/ap_cliff_finetune_pids.json").write_text(json.dumps(pids, indent=2))


if __name__ == "__main__":
    main()
