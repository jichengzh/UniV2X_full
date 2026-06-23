"""收敛后跑 AP-cliff2 3 档 (cliff2_a/b/c) 的 DAIR val 1789 subnet AP eval。

★ 上轮 eval driver 教训: 只读 eval_intermediate.yaml, 漏了 AP (有时 dump 缺键)。
  本 driver 改为 **从 inference.py stdout 直接解析 AP** (主信号):
    "The Average Precision at IOU 0.3 is X, ... 0.5 is Y, ... 0.7 is Z"
  (HEAL eval_utils.py:160 用 %.2f 打印, 故 stdout 是 2 位小数)
  再 fall back 到 eval_intermediate.yaml 的 ap30/ap_50/ap_70 (全精度) 做交叉校验。

口径: intermediate fusion, DAIR val 1789, subnet (≠ e2e)。
GPU: cliff2_a/c 收敛在 GPU1, cliff2_b 在 GPU7 (与 finetune 同卡)。
跑前自行 nvidia-smi 确认对应卡空闲 (finetune 已退)。

用法:
    PYTHONPATH=/home/jichengzhi/heal_research/HEAL \\
    /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \\
        scripts/phase2/eval_cliff2.py

输出: results/ap_cliff2_converged.json + results/ap_cliff2_<tag>_eval.log
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

# tag -> (ckpt dir suffix, eval GPU, num_filters, wpg, parsed_total_M, parsed_pb_M)
CFG = {
    "cliff2_a": ("Pyramid_DAIR_m1_cliff2_a_2026_06_02", "1", [32, 64, 128], 4, 2.808, 1.101),
    "cliff2_b": ("Pyramid_DAIR_m1_cliff2_b_2026_06_02", "7", [24, 48, 96], 4, 2.383, 0.676),
    "cliff2_c": ("Pyramid_DAIR_m1_cliff2_c_2026_06_02", "1", [16, 32, 64], 4, 2.063, 0.356),
}

AP_RE = re.compile(
    r"Average Precision at IOU 0\.3 is ([\d.]+).*?"
    r"0\.5 is ([\d.]+).*?"
    r"0\.7 is ([\d.]+)",
    re.DOTALL,
)


def highest_bestval(d):
    fs = glob.glob(f"{d}/net_epoch_bestval_at*.pth")
    if not fs:
        return None
    return max(fs, key=lambda f: int(re.search(r"at(\d+)\.pth", f).group(1)))


def parse_ap_from_stdout(text: str):
    m = AP_RE.search(text)
    if not m:
        return None
    return {"ap30": float(m.group(1)), "ap50": float(m.group(2)),
            "ap70": float(m.group(3))}


def parse_ap_from_yaml(yml: str):
    if not os.path.exists(yml):
        return None
    import yaml
    y = yaml.safe_load(open(yml)) or {}
    out = {"ap30": y.get("ap30") or y.get("ap_30"),
           "ap50": y.get("ap_50") or y.get("ap50"),
           "ap70": y.get("ap_70") or y.get("ap70")}
    return out if any(v is not None for v in out.values()) else None


def main():
    procs = {}
    for tag, (subdir, gpu, nf, wpg, totM, pbM) in CFG.items():
        d = f"{CKROOT}/{subdir}"
        best = highest_bestval(d)
        if best is None:
            print(f"[{tag}] no bestval yet, skip (not converged?)", flush=True)
            continue
        # HEAL 要求单一 bestval: 保留最高, 其余 .bak
        for f in glob.glob(f"{d}/net_epoch_bestval_at*.pth"):
            if f != best:
                os.rename(f, f + ".bak")
        yml = f"{d}/eval_intermediate.yaml"
        if os.path.exists(yml):
            os.remove(yml)
        logpath = REPO / f"results/ap_cliff2_{tag}_eval.log"
        log = open(logpath, "w")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": HEAL}
        p = subprocess.Popen(
            [PY, "opencood/tools/inference.py", "--model_dir", d,
             "--fusion_method", "intermediate"],
            cwd=HEAL, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs[tag] = (p, d, os.path.basename(best), nf, wpg, totM, pbM, logpath, yml)
        print(f"[{tag}] launched GPU{gpu} bestval={os.path.basename(best)} pid={p.pid}", flush=True)

    print("=== waiting for evals ===", flush=True)
    results = {}
    for tag, (p, d, best, nf, wpg, totM, pbM, logpath, yml) in procs.items():
        rc = p.wait()
        text = Path(logpath).read_text(errors="ignore")
        ap_stdout = parse_ap_from_stdout(text)
        ap_yaml = parse_ap_from_yaml(yml)
        ap = ap_stdout or ap_yaml or {}
        results[tag] = {
            "rc": rc, "bestval": best, "num_filters": nf, "wpg": wpg,
            "params_total_M": totM, "params_pb_M": pbM,
            "ap": ap, "ap_stdout": ap_stdout, "ap_yaml": ap_yaml,
        }
        print(f"[{tag}] done rc={rc} nf={nf} wpg{wpg} pb={pbM}M "
              f"AP(stdout)={ap_stdout} AP(yaml)={ap_yaml}", flush=True)

    (REPO / "results/ap_cliff2_converged.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print("=== wrote results/ap_cliff2_converged.json ===", flush=True)


if __name__ == "__main__":
    main()
