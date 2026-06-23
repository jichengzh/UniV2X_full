"""P0 round-2 — "找 AP 悬崖 (换路线: 降 wpg)".

上一轮结论 (必读): prune85/90/95 用 wpg=16 撑住 bottleneck → 即便 num_filters
压到 [4,6,13] (pb=0.08M), AP50 仍 plateau 0.74-0.75, 没找到悬崖。根因: AP 由
bottleneck 宽度 (width = int(planes*wpg/64)*groups) 主导, 不是 num_filters。
⇒ 本轮换路线: 把 wpg 降到 4 (默认值), 直接缩小 bottleneck width 找悬崖。

口径同 stage_a / wholenet 金标准 (backbone L1 结构化剪 + 收敛 finetune 48ep
+ DAIR val 1789 subnet eval)。subnet_params = total (含固定 encoder_m1 ~1.7M)。

★ 解析 params (build_smaller_model 实测, NOT 估算):
    base       [64,128,256] wpg4  total 5.465M  pb 3.758M  widths [128,256,512]
  --- 本轮 3 档 (全部 wpg=4, width>0 / min_ipg>=1) ---
    cliff2_a   [32, 64,128] wpg4  total 2.808M  pb 1.101M  widths [ 64,128,256]
    cliff2_b   [24, 48, 96] wpg4  total 2.383M  pb 0.676M  widths [ 32, 96,192]
    cliff2_c   [16, 32, 64] wpg4  total 2.063M  pb 0.356M  widths [ 32, 64,128]  <- 最窄, 预期踩悬崖
  (pb 即可剪部分; total 含固定 1.7M encoder/backbone_m1。目标 "pb < 1.3M" 全达成。)

★ width=0 陷阱: wpg=4 → width=int(planes/16)*32, planes<16 崩。三档 stage0
  分别 32/24/16, int(*4/64)*32 = 64/32/32 > 0, min_ipg = width//32 >= 1, 安全。
★ ckpt flat 陷阱 (CLAUDE.md §〇#5): structural_prune_pyramid.py 存 wrapped
  {"model_state_dict":...}; HEAL load 不解包 → 全 key missing → 随机权重训。
  本脚本 prune 后立即 flatten。

GPU: 用 GPU 1 + GPU 7 (跑前 nvidia-smi 确认 idle)。不碰 GPU 0/2/3-6。
本脚本只 prune+flatten (同步) 再 detached nohup 启动 finetune, 不等训完。
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import time
from pathlib import Path

import torch

REPO = Path("/home/jichengzhi/UniV2X")
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
BASELINE = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
CKROOT = Path("/home/jichengzhi/heal_research/checkpoints/stage1")
EPOCHES = 48          # init@23 → 25 epoch finetune (同上轮口径)
WPG = 4               # 本轮路线: 降 wpg 缩 bottleneck width
GROUPS = 32

# (tag, num_filters, gpu)  —— 只用 GPU 1 + GPU 7。三档轮流 (cliff2_c 单独占一卡)。
CANDIDATES = [
    ("cliff2_a", [32, 64, 128], 1),
    ("cliff2_b", [24, 48, 96], 7),
    ("cliff2_c", [16, 32, 64], 1),   # 跟 a 同 GPU1 串行 (a 训完再起); 见下方调度
]
# GPU 调度: GPU1 跑 a→c 串行, GPU7 跑 b。避免一卡两进程抢显存。
GPU_SERIAL = {1: ["cliff2_a", "cliff2_c"], 7: ["cliff2_b"]}


def out_dir_for(tag: str) -> Path:
    return CKROOT / f"Pyramid_DAIR_m1_{tag}_2026_06_02"


def run_prune(tag: str, nf: list, gpu: int) -> Path:
    out_dir = out_dir_for(tag)
    init_ckpt = out_dir / "net_epoch_bestval_at23.pth"
    # resume guard: 若已有 >23 的 ckpt 说明已在 finetune, 跳过 prune
    existing = list(out_dir.glob("net_epoch*.pth"))
    if existing and any(
        (m := re.search(r"epoch(?:_bestval_at)?(\d+)", p.name)) and int(m.group(1)) > 23
        for p in existing
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
    """防 flat 陷阱: {"model_state_dict":...} → flat state_dict。"""
    init_ckpt = out_dir / "net_epoch_bestval_at23.pth"
    if not init_ckpt.exists():
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


def launch_finetune(out_dir: Path, tag: str, gpu: int, after_pid: int | None = None) -> int:
    """detached nohup finetune。after_pid: 若非 None, 包一层 wait 串行 (同 GPU)。"""
    cfg = out_dir / "config.yaml"
    log = REPO / f"results/ap_cliff2_{tag}_finetune.log"
    train_cmd = (
        f"{PY} -m torch.distributed.launch --nproc_per_node=1 --use_env "
        f"--master_port={29700 + gpu} "
        f"{HEAL / 'opencood/tools/train_ddp.py'} "
        f"--hypes_yaml {cfg} --model_dir {out_dir} --half"
    )
    if after_pid:
        # 等同 GPU 前一档进程退出再起 (串行), 避免抢显存
        sh = f"while kill -0 {after_pid} 2>/dev/null; do sleep 30; done; {train_cmd}"
    else:
        sh = train_cmd
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(HEAL)}
    lf = open(log, "w")
    p = subprocess.Popen(["bash", "-c", sh], cwd=HEAL, env=env, stdout=lf,
                         stderr=subprocess.STDOUT, start_new_session=True)
    tail = f" (waits for pid {after_pid})" if after_pid else ""
    print(f"[{tag}] finetune launched GPU{gpu} pid={p.pid} log={log}{tail}")
    return p.pid


def main():
    # 1. prune + flatten + patch (同步)
    for tag, nf, gpu in CANDIDATES:
        out_dir = run_prune(tag, nf, gpu)
        flatten_init_ckpt(out_dir, tag)
        patch_epoches(out_dir, tag)

    # 2. detached 启动 finetune, 按 GPU_SERIAL 调度 (同卡串行)
    pids = {}
    nf_by_tag = {t: nf for t, nf, _ in CANDIDATES}
    for gpu, tags in GPU_SERIAL.items():
        prev_pid = None
        for tag in tags:
            out_dir = out_dir_for(tag)
            pid = launch_finetune(out_dir, tag, gpu, after_pid=prev_pid)
            pids[tag] = {"pid": pid, "gpu": gpu, "num_filters": nf_by_tag[tag],
                         "out_dir": str(out_dir),
                         "serial_after": prev_pid}
            prev_pid = pid
            time.sleep(3)

    print("\n=== LAUNCHED ===")
    for tag, info in pids.items():
        print(f"  {tag}: GPU{info['gpu']} pid={info['pid']} nf={info['num_filters']} dir={info['out_dir']}")
    (REPO / "results/ap_cliff2_finetune_pids.json").write_text(json.dumps(pids, indent=2))


if __name__ == "__main__":
    main()
