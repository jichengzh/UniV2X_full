"""收敛后跑 AP-cliff 3 档 (prune85/90/95) 的 DAIR val 1789 AP eval。

同口径 stage_a / wholenet 金标准: 选最高 bestval → HEAL inference.py
(intermediate fusion, DAIR val 1789) → 解析 eval 输出 AP30/50/70。
3 档并行 GPU 0/1/2 (训练完才跑, 跑前自行确认 GPU 空闲)。

用法:
    PYTHONPATH=/home/jichengzhi/heal_research/HEAL \\
    /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \\
        scripts/phase2/eval_cliff_converged.py

输出: results/ap_cliff_converged.json (含每档 AP) + results/ap_cliff_<tag>_eval.log
"""
import glob
import json
import os
import re
import subprocess
from pathlib import Path

HEAL = "/home/jichengzhi/heal_research/HEAL"
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
CKROOT = "/home/jichengzhi/heal_research/checkpoints/stage1"
REPO = Path("/home/jichengzhi/UniV2X")
# tag -> (ckpt suffix dir, eval GPU, num_filters)
CFG = {
    "prune85": ("Pyramid_DAIR_m1_prune85_2026_06_02", "0", [10, 20, 40]),
    "prune90": ("Pyramid_DAIR_m1_prune90_2026_06_02", "1", [6, 13, 26]),
    "prune95": ("Pyramid_DAIR_m1_prune95_2026_06_02", "2", [4, 6, 13]),
}


def highest_bestval(d):
    fs = glob.glob(f"{d}/net_epoch_bestval_at*.pth")
    if not fs:
        return None
    return max(fs, key=lambda f: int(re.search(r"at(\d+)\.pth", f).group(1)))


def main():
    procs = {}
    for tag, (subdir, gpu, nf) in CFG.items():
        d = f"{CKROOT}/{subdir}"
        best = highest_bestval(d)
        if best is None:
            print(f"[{tag}] no bestval yet, skip", flush=True)
            continue
        # HEAL 要求单一 bestval: 保留最高, 其余 .bak
        for f in glob.glob(f"{d}/net_epoch_bestval_at*.pth"):
            if f != best:
                os.rename(f, f + ".bak")
        yml = f"{d}/eval_intermediate.yaml"
        if os.path.exists(yml):
            os.remove(yml)
        log = open(REPO / f"results/ap_cliff_{tag}_eval.log", "w")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": HEAL}
        p = subprocess.Popen(
            [PY, "opencood/tools/inference.py", "--model_dir", d,
             "--fusion_method", "intermediate"],
            cwd=HEAL, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs[tag] = (p, d, os.path.basename(best), nf)
        print(f"[{tag}] launched GPU{gpu} bestval={os.path.basename(best)} pid={p.pid}", flush=True)

    print("=== waiting for evals ===", flush=True)
    results = {}
    for tag, (p, d, best, nf) in procs.items():
        rc = p.wait()
        yml = f"{d}/eval_intermediate.yaml"
        ap = {}
        if os.path.exists(yml):
            import yaml
            y = yaml.safe_load(open(yml)) or {}
            ap = {"ap30": y.get("ap30") or y.get("ap_30"),
                  "ap50": y.get("ap_50") or y.get("ap50"),
                  "ap70": y.get("ap_70") or y.get("ap70")}
        results[tag] = {"rc": rc, "bestval": best, "num_filters": nf, "ap": ap}
        print(f"[{tag}] done rc={rc} nf={nf} AP={ap}", flush=True)

    (REPO / "results/ap_cliff_converged.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print("=== wrote results/ap_cliff_converged.json ===", flush=True)


if __name__ == "__main__":
    main()
