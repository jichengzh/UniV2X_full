"""Gap 2 eval watcher — wait for [32,64,136] finetune marker, then real AP eval.

Waits for results/p2_prune50b2_032_064_136.done (written with child rc by the
finetune launcher; NO pgrep). On rc==0, runs HEAL inference.py
--fusion_method intermediate on the converged bestval ckpt (DAIR val 1789),
parses AP from stdout, and merges into results/P2_ap_fill.json under
planes_032_064_136. Same gold-standard eval口径 as eval_cliff3.

GPU 2 ONLY.
"""
import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL = "/home/jichengzhi/heal_research/HEAL"
CKDIR = ("/home/jichengzhi/heal_research/checkpoints/stage1/"
         "Pyramid_DAIR_m1_prune50b2_032_064_136_2026_06_03")
GPU = "2"
TAG = "prune50b2_032_064_136"
MARKER = REPO / f"results/p2_{TAG}.done"
OUT = REPO / "results/P2_ap_fill.json"

AP_RE = re.compile(
    r"Average Precision at IOU 0\.3 is ([\d.]+).*?0\.5 is ([\d.]+).*?0\.7 is ([\d.]+)",
    re.DOTALL)


def highest_bestval(d):
    fs = glob.glob(f"{d}/net_epoch_bestval_at*.pth")
    fs = [f for f in fs if not f.endswith(".bak")]
    if not fs:
        return None
    return max(fs, key=lambda f: int(re.search(r"at(\d+)\.pth", f).group(1)))


def marker_rc():
    if not MARKER.exists():
        return None
    m = re.search(r"rc=(\d+)", MARKER.read_text())
    return int(m.group(1)) if m else 0


def update_json(payload):
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data["planes_032_064_136"] = {**data.get("planes_032_064_136", {}), **payload}
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    poll = 0
    while marker_rc() is None:
        if poll % 30 == 0:
            print(f"[wait] no marker yet ({TAG}); polling...", flush=True)
        poll += 1
        time.sleep(60)

    rc_ft = marker_rc()
    print(f"[{TAG}] finetune done rc={rc_ft}", flush=True)
    if rc_ft != 0:
        update_json({"status": "ft_failed", "note": f"finetune rc={rc_ft}; no valid ckpt, eval skipped"})
        print(f"[{TAG}] finetune failed rc={rc_ft}", flush=True)
        return

    best = highest_bestval(CKDIR)
    if best is None:
        update_json({"status": "no_bestval", "note": "marker present but no bestval ckpt"})
        return
    # HEAL requires single bestval
    for f in glob.glob(f"{CKDIR}/net_epoch_bestval_at*.pth"):
        if f != best and not f.endswith(".bak"):
            os.rename(f, f + ".bak")
    for yml in glob.glob(f"{CKDIR}/eval_intermediate*.yaml"):
        os.remove(yml)

    logpath = REPO / f"results/p2_{TAG}_eval.log"
    log = open(logpath, "w")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "PYTHONPATH": HEAL}
    print(f"[{TAG}] eval start bestval={os.path.basename(best)}", flush=True)
    rc = subprocess.call(
        [PY, "opencood/tools/inference.py", "--model_dir", CKDIR,
         "--fusion_method", "intermediate"],
        cwd=HEAL, env=env, stdout=log, stderr=subprocess.STDOUT)
    text = Path(logpath).read_text(errors="ignore")
    m = AP_RE.search(text)
    ap = ({"ap30": float(m.group(1)), "ap50": float(m.group(2)),
           "ap70": float(m.group(3))} if m else {})
    payload = {
        "ap30": ap.get("ap30"), "ap50": ap.get("ap50"), "ap70": ap.get("ap70"),
        "status": "ok" if ap else "no_ap",
        "eval_rc": rc, "ft_rc": rc_ft,
        "bestval": os.path.basename(best),
        "source": ("real finetuned eval: HEAL inference.py --fusion_method "
                   "intermediate on converged [32,64,136] ckpt, DAIR val 1789 "
                   "(range 102.4x102.4, inference.py default)"),
    }
    update_json(payload)
    print(f"[{TAG}] done eval_rc={rc} AP={ap} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
