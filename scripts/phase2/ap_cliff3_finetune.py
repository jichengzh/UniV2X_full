"""P0 round-3 — 前两轮"只剪 pyramid_backbone"未触及的两维度真测.

背景 (必读, 已验证):
  - cliff (wpg16) + cliff2 (wpg4) 两轮: 只剪 pyramid_backbone (68.9%), 压到 0.36M
    仍 AP50 0.74-0.75 / AP70 0.57-0.60, 无悬崖. base 全模型 AP50 仅 0.791.
  - 既往 wholenet light/p50/aggr 虽剪了 deblocks+shrink, 但 AP 从未真测
    (wholenet_converged_ap.json 的 ap 全空), 且都固定 backbone nf=[32,64,128].
  - 参数占比实测: pyramid_backbone 68.76% / shrink_conv 26.99% / backbone_m1 4.14%
    / encoder_m1 0.01%. ⇒ shrink_conv 是前两轮基本没动的 27% 大块.

本轮 (cliff3) 要回答的、之前只"断言未实测"的消融:
  1. sh_only_xhard: backbone 满血 [64,128,256], 把 shrink_conv 砍到 92K (-94%),
     deblocks→[32,32,32]. **隔离 shrink_conv 是否携带 AP** —— 若 AP 不掉, 证明
     AP 不在 shrink; 若掉, 证明 shrink 才是 AP 主信号. (前两轮都没单独动 shrink.)
  2. sh_only_hard: backbone 满血, shrink→128, deblocks→[64,64,64]. 中档隔离.
  3. all3_hard: pyramid_backbone + deblocks + shrink 三处全狠剪 (TOTAL -75.3%,
     pb 0.91M, shrink 0.21M). 之前从未把"全部三处"一起剪到收敛测 AP.
  4. underfit: 用 all3_hard 同一剪枝 init, 只 finetune 3 epoch (欠收敛).
     **分离 "无悬崖" 与 "finetune 总能救回"** —— 看欠训下 AP 掉多深、是否现悬崖.

口径 = stage_a / cliff2 金标准: flat ckpt resume finetune + DAIR val 1789
intermediate subnet eval, AP 从 inference.py stdout 解析. 主信号轴 = AP70.

GPU: **只用 GPU 1** (已确认 idle). 四档 GPU1 串行 (一卡一进程, 避免抢显存).
  顺序: all3_hard → sh_only_xhard → sh_only_hard → underfit(3ep, 快).
  (underfit 放最后, 它最快; 前三档各 ~25ep × ~13min ≈ 5h, 全链 ~15h.)

完成检测: 每个 finetune 起 detached nohup, 退出时 touch 一个**唯一 marker 文件**
  results/ap_cliff3_<tag>.done  (内含 child PID + exit code).
  watcher 只 stat marker 文件, 不 pgrep (避免 watcher cmdline 自匹配 bug).
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import time
from pathlib import Path

import torch  # noqa: F401  (kept for parity / availability check)

REPO = Path("/home/jichengzhi/UniV2X")
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
CKROOT = Path("/home/jichengzhi/heal_research/checkpoints/stage1")
GPU = 1

# tag -> ckpt dir (pruned init ckpts already built + flat-verified + epoches patched)
CANDIDATES = [
    ("all3_hard",     CKROOT / "Pyramid_DAIR_m1_cliff3_all3_hard_2026_06_03"),
    ("sh_only_xhard", CKROOT / "Pyramid_DAIR_m1_cliff3_sh_only_xhard_2026_06_03"),
    ("sh_only_hard",  CKROOT / "Pyramid_DAIR_m1_cliff3_sh_only_hard_2026_06_03"),
    ("underfit",      CKROOT / "Pyramid_DAIR_m1_cliff3_underfit_2026_06_03"),
]


def marker_path(tag: str) -> Path:
    return REPO / f"results/ap_cliff3_{tag}.done"


def launch_chain():
    """单 GPU 串行链: 用 bash 把四档串起来, 每档退出 touch marker.

    串行靠 bash 顺序执行 (前一档 train_ddp 阻塞退出后才起下一档),
    天然一卡一进程. 整链一个 nohup 进程; 每档单独 marker + log.
    """
    parts = []
    for tag, d in CANDIDATES:
        cfg = d / "config.yaml"
        log = REPO / f"results/ap_cliff3_{tag}_finetune.log"
        marker = marker_path(tag)
        # 起训前删旧 marker (resume 场景)
        if marker.exists():
            marker.unlink()
        port = 29800 + len(parts)
        train = (
            f"{PY} -m torch.distributed.launch --nproc_per_node=1 --use_env "
            f"--master_port={port} "
            f"{HEAL / 'opencood/tools/train_ddp.py'} "
            f"--hypes_yaml {cfg} --model_dir {d} --half "
            f"> {log} 2>&1"
        )
        # 每档: 跑训练, 无论成败都 touch marker 记录 rc + 完成时刻
        parts.append(
            f'echo "[chain] start {tag} $(date)"; '
            f'{train}; rc=$?; '
            f'echo "tag={tag} rc=$rc finished=$(date +%s)" > {marker}; '
            f'echo "[chain] done {tag} rc=$rc $(date)"'
        )
    chain_sh = "; ".join(parts)
    chain_log = REPO / "results/ap_cliff3_chain.log"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU), "PYTHONPATH": str(HEAL)}
    lf = open(chain_log, "w")
    p = subprocess.Popen(["bash", "-c", chain_sh], cwd=HEAL, env=env,
                         stdout=lf, stderr=subprocess.STDOUT,
                         start_new_session=True)
    return p.pid, chain_log


def main():
    # resume guard: 若某档已有 >23 ckpt, 它会被 train_ddp 自动 resume (HEAL 行为),
    # 这里不跳过, 让链整体重跑剩余 epoch (train_ddp resume 是幂等的).
    pid, chain_log = launch_chain()
    info = {
        "chain_pid": pid,
        "gpu": GPU,
        "chain_log": str(chain_log),
        "candidates": [
            {"tag": t, "dir": str(d), "marker": str(marker_path(t)),
             "finetune_log": str(REPO / f"results/ap_cliff3_{t}_finetune.log")}
            for t, d in CANDIDATES
        ],
        "launched_at": int(time.time()),
    }
    (REPO / "results/ap_cliff3_finetune_pids.json").write_text(
        json.dumps(info, indent=2))
    print(f"[chain] launched GPU{GPU} pid={pid} log={chain_log}")
    for t, d in CANDIDATES:
        print(f"  {t}: dir={d} marker={marker_path(t)}")


if __name__ == "__main__":
    main()
