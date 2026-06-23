"""过夜编排: 每个 wholenet finetune 跑到 epoch 31 后, 自动测 AP (inference.py) + latency。

对每个 variant:
  1. 等 net_epoch31.pth 出现 (finetune 完成)
  2. AP: HEAL inference.py --fusion_method intermediate (DAIR val 1789), 从输出 parse ap30/50/70
     —— 注: inference 用 model_dir 里 epoch 最高的 bestval; 我们把 net_epoch31 复制成
        net_epoch_bestval_at31.pth 让 inference 用最终 finetuned 权重 (HEAL load 规则)。
  3. latency: wholenet_subnet_latency.py (在指定空闲 GPU), epoch 31
  4. 写 results/lat_<tag>.json + results/ap_<tag>.json, 末尾汇总进 CSV (单独脚本合并)。

GPU: AP eval 用 --ap-gpu (默认 7, 空闲), latency 用各自 finetune GPU (此时已释放) 或 --lat-gpu。
不动 GPU 6。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"

VARIANTS = [
    # tag, model_dir, finetune_gpu (freed after ft, used for latency)
    ("wholenet_light",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_wholenet_light_2026_06_01", 1),
    ("wholenet_p50",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_wholenet_p50_2026_06_01", 0),
    ("wholenet_aggr",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_wholenet_aggr_2026_06_01", 2),
]
FINAL_EPOCH = 31
AP_GPU = "7"


def wait_epoch(model_dir, epoch, timeout_s=4 * 3600):
    t0 = time.time()
    target = Path(model_dir) / f"net_epoch{epoch}.pth"
    while time.time() - t0 < timeout_s:
        if target.exists():
            time.sleep(20)  # let the file finish writing
            return True
        time.sleep(60)
    return False


def parse_ap(log_text):
    # HEAL eval prints lines like "The Average Precision at IOU 0.3 is ...; 0.5 ...; 0.7 ..."
    aps = {}
    for iou, key in ((0.3, "ap30"), (0.5, "ap50"), (0.7, "ap70")):
        m = re.search(rf"IOU\s*{iou}\D+([0-9.]+)", log_text)
        if m:
            aps[key] = float(m.group(1))
    # fallback patterns
    for key, pat in (("ap30", r"ap_?30[^0-9]*([0-9.]+)"),
                     ("ap50", r"ap_?50[^0-9]*([0-9.]+)"),
                     ("ap70", r"ap_?70[^0-9]*([0-9.]+)")):
        if key not in aps:
            m = re.search(pat, log_text, re.I)
            if m:
                aps[key] = float(m.group(1))
    return aps


def run_ap(tag, model_dir, epoch):
    # Make finetuned epoch the bestval HEAL will load (single bestval rule).
    src = Path(model_dir) / f"net_epoch{epoch}.pth"
    bestval = Path(model_dir) / f"net_epoch_bestval_at{epoch}.pth"
    # remove other bestvals to satisfy HEAL's assert len==1
    for old in Path(model_dir).glob("net_epoch_bestval_at*.pth"):
        if old != bestval:
            old.rename(old.with_suffix(".pth.bak"))
    if not bestval.exists():
        shutil.copy(src, bestval)
    log = REPO / f"results/ap_{tag}.log"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": AP_GPU,
           "PYTHONPATH": str(HEAL)}
    with open(log, "w") as f:
        subprocess.run(
            [PY, "opencood/tools/inference.py", "--model_dir", model_dir,
             "--fusion_method", "intermediate"],
            cwd=str(HEAL), env=env, stdout=f, stderr=subprocess.STDOUT)
    txt = log.read_text()
    aps = parse_ap(txt)
    # 更稳: 读 inference.py dump 的 eval_intermediate.yaml
    yml = Path(model_dir) / "eval_intermediate.yaml"
    if yml.exists():
        import yaml as _y
        d = _y.safe_load(yml.read_text())
        if d:
            if "ap30" in d:
                aps["ap30"] = float(d["ap30"])
            if "ap_50" in d:
                aps["ap50"] = float(d["ap_50"])
            if "ap_70" in d:
                aps["ap70"] = float(d["ap_70"])
    rep = {"tag": tag, "epoch": epoch, **aps, "log": str(log)}
    (REPO / f"results/ap_{tag}.json").write_text(json.dumps(rep, indent=2))
    print(f"[AP {tag}] {aps}", flush=True)
    return rep


def run_lat(tag, model_dir, epoch, gpu):
    rep_path = REPO / f"results/lat_{tag}.json"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": f"{HEAL}:{REPO}"}
    subprocess.run(
        [PY, str(REPO / "scripts/phase2/wholenet_subnet_latency.py"),
         "--model-dir", model_dir, "--epoch", str(epoch),
         "--tag", tag, "--report", str(rep_path), "--no-gpu-check"],
        env=env)
    print(f"[LAT {tag}] -> {rep_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of tags to run")
    args = ap.parse_args()
    todo = VARIANTS
    if args.only:
        sel = set(args.only.split(","))
        todo = [v for v in VARIANTS if v[0] in sel]
    for tag, mdir, gpu in todo:
        print(f"=== {tag}: wait epoch {FINAL_EPOCH} ===", flush=True)
        if not wait_epoch(mdir, FINAL_EPOCH):
            print(f"[{tag}] TIMEOUT waiting epoch {FINAL_EPOCH}", flush=True)
            continue
        try:
            run_ap(tag, mdir, FINAL_EPOCH)
        except Exception as e:
            print(f"[{tag}] AP FAIL {e}", flush=True)
        try:
            run_lat(tag, mdir, FINAL_EPOCH, gpu)
        except Exception as e:
            print(f"[{tag}] LAT FAIL {e}", flush=True)
    print("=== orchestrator done ===", flush=True)


if __name__ == "__main__":
    main()
