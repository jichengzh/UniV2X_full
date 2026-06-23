"""cliff3 收敛后 AP eval — 等 marker 文件确认 finetune 退出再 eval.

口径同 eval_cliff2: DAIR val 1789, intermediate fusion, subnet (≠ e2e),
AP 从 inference.py stdout 解析 ("Average Precision at IOU 0.x is X", 2dp),
fall back eval_intermediate.yaml. 主信号轴 = AP70.

完成检测: 只 stat results/ap_cliff3_<tag>.done (内含 rc), 不 pgrep
(避免 watcher 自匹配). marker 存在 == 该档 finetune 进程已退出.

用法:
    PYTHONPATH=/home/jichengzhi/heal_research/HEAL \\
    /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \\
        scripts/phase2/eval_cliff3.py
输出: results/ap_cliff3_converged.json + results/ap_cliff3_<tag>_eval.log
"""
import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

HEAL = "/home/jichengzhi/heal_research/HEAL"
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
CKROOT = "/home/jichengzhi/heal_research/checkpoints/stage1"
REPO = Path("/home/jichengzhi/UniV2X")
GPU = "1"

# tag -> (ckpt subdir, num_filters, num_upsample, shrink_dim, total_M, pb_M, shrink_K, note)
CFG = {
    "all3_hard":     ("Pyramid_DAIR_m1_cliff3_all3_hard_2026_06_03",
                      [32, 64, 128], [48, 48, 48], 96, 1.350, 0.914, 207.6,
                      "backbone+deblocks+shrink 全狠剪 (-75.3%)"),
    "sh_only_xhard": ("Pyramid_DAIR_m1_cliff3_sh_only_xhard_2026_06_03",
                      [64, 128, 256], [32, 32, 32], 64, 3.629, 3.309, 92.3,
                      "backbone 满血, shrink -94% (隔离 shrink AP 信号)"),
    "sh_only_hard":  ("Pyramid_DAIR_m1_cliff3_sh_only_hard_2026_06_03",
                      [64, 128, 256], [64, 64, 64], 128, 4.057, 3.458, 368.9,
                      "backbone 满血, shrink 中档剪"),
    "underfit":      ("Pyramid_DAIR_m1_cliff3_underfit_2026_06_03",
                      [32, 64, 128], [48, 48, 48], 96, 1.350, 0.914, 207.6,
                      "= all3_hard 剪枝, 仅 3ep 欠收敛 (force-a-cliff 压力测试)"),
}

AP_RE = re.compile(
    r"Average Precision at IOU 0\.3 is ([\d.]+).*?0\.5 is ([\d.]+).*?0\.7 is ([\d.]+)",
    re.DOTALL,
)


def highest_bestval(d):
    fs = glob.glob(f"{d}/net_epoch_bestval_at*.pth")
    if not fs:
        return None
    return max(fs, key=lambda f: int(re.search(r"at(\d+)\.pth", f).group(1)))


def parse_ap_stdout(text):
    m = AP_RE.search(text)
    if not m:
        return None
    return {"ap30": float(m.group(1)), "ap50": float(m.group(2)),
            "ap70": float(m.group(3))}


def parse_ap_yaml(yml):
    if not os.path.exists(yml):
        return None
    import yaml
    y = yaml.safe_load(open(yml)) or {}
    out = {"ap30": y.get("ap30") or y.get("ap_30"),
           "ap50": y.get("ap_50") or y.get("ap50"),
           "ap70": y.get("ap_70") or y.get("ap70")}
    return out if any(v is not None for v in out.values()) else None


def marker_done(tag):
    """marker 存在即 finetune 退出; 返回 rc (int) 或 None."""
    mp = REPO / f"results/ap_cliff3_{tag}.done"
    if not mp.exists():
        return None
    txt = mp.read_text()
    m = re.search(r"rc=(\d+)", txt)
    return int(m.group(1)) if m else 0


def main():
    results = {}
    pending = dict(CFG)
    poll = 0
    # 逐档等 marker; eval 完一档接着等下一档 (链是串行的, marker 也按序出现)
    while pending:
        ready = [t for t in list(pending) if marker_done(t) is not None]
        if not ready:
            if poll % 20 == 0:
                print(f"[wait] no marker yet; pending={list(pending)}", flush=True)
            poll += 1
            time.sleep(60)
            continue
        for tag in ready:
            subdir, nf, nuf, sh, totM, pbM, shrinkK, note = pending.pop(tag)
            rc_ft = marker_done(tag)
            if rc_ft != 0:
                # finetune failed -> only the raw pruned init ckpt exists; eval'ing it
                # would inject garbage AP. Skip and record the failure instead.
                print(f"[{tag}] finetune rc={rc_ft} != 0, SKIP eval (no valid "
                      f"finetuned ckpt)", flush=True)
                results[tag] = {"ft_rc": rc_ft, "status": "ft_failed",
                                "num_filters": nf, "num_upsample": nuf,
                                "shrink_dim": sh, "note": note}
                continue
            d = f"{CKROOT}/{subdir}"
            best = highest_bestval(d)
            if best is None:
                print(f"[{tag}] marker present (ft rc={rc_ft}) but no bestval ckpt!",
                      flush=True)
                results[tag] = {"ft_rc": rc_ft, "status": "no_bestval",
                                "num_filters": nf, "num_upsample": nuf,
                                "shrink_dim": sh, "note": note}
                continue
            # HEAL 要求单一 bestval: 保留最高其余 .bak
            for f in glob.glob(f"{d}/net_epoch_bestval_at*.pth"):
                if f != best:
                    os.rename(f, f + ".bak")
            yml = f"{d}/eval_intermediate.yaml"
            if os.path.exists(yml):
                os.remove(yml)
            logpath = REPO / f"results/ap_cliff3_{tag}_eval.log"
            log = open(logpath, "w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "PYTHONPATH": HEAL}
            print(f"[{tag}] eval start bestval={os.path.basename(best)} "
                  f"(ft rc={rc_ft})", flush=True)
            rc = subprocess.call(
                [PY, "opencood/tools/inference.py", "--model_dir", d,
                 "--fusion_method", "intermediate"],
                cwd=HEAL, env=env, stdout=log, stderr=subprocess.STDOUT)
            text = Path(logpath).read_text(errors="ignore")
            ap_stdout = parse_ap_stdout(text)
            ap_yaml = parse_ap_yaml(yml)
            ap = ap_stdout or ap_yaml or {}
            results[tag] = {
                "ft_rc": rc_ft, "eval_rc": rc, "status": "ok" if ap else "no_ap",
                "bestval": os.path.basename(best),
                "num_filters": nf, "num_upsample": nuf, "shrink_dim": sh,
                "params_total_M": totM, "params_pb_M": pbM,
                "params_shrink_K": shrinkK, "note": note,
                "ap": ap, "ap_stdout": ap_stdout, "ap_yaml": ap_yaml,
            }
            print(f"[{tag}] done eval_rc={rc} AP={ap} | {note}", flush=True)
            # 增量写, 便于中途看
            (REPO / "results/ap_cliff3_converged.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False))

    (REPO / "results/ap_cliff3_converged.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print("=== wrote results/ap_cliff3_converged.json ===", flush=True)


if __name__ == "__main__":
    main()
