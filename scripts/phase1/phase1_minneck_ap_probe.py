#!/usr/bin/env python3
"""Phase 1 — decisive probe: does shrinking the neck (w1,w2) preserve AP?

The 1029 surrogate (assuming w0-only-AP) predicts minimal-neck [w0,16,16]
dominates the measured diagonal [w0,2w0,4w0]. But [w0,w1,w2] are the 3 BACKBONE
stage widths (not the deblocks/shrink neck), so w1,w2 plausibly DO carry AP.
This probe finetunes minimal-neck / mid-neck widths at fixed w0 and evaluates
real AP70 (DAIR val) to test whether w0-only-AP holds -> determines the TRUE
Pareto front (minimal-neck vs diagonal).

Protocol = EXACT gold stage_a: structural_prune_pyramid.py (L1, wpg=4, groups=32)
-> flatten init ckpt -> patch epoches=31 (init@23 -> 8 finetune epochs) -> finetune
(train_ddp, half) -> PyTorch AP eval (opencood inference.py, DAIR val) — NO TRT.

Usage: phase1_minneck_ap_probe.py <gpu> <master_port> <w0xw1xw2>[,<...>]
Writes results/phase1_minneck_probe/<tag>_ap.json per width.
"""
import sys, os, json, subprocess, re
from pathlib import Path

sys.path.insert(0, "/home/jichengzhi/V2X/scripts/phase2")
import b4expand_finetune_eval as B  # reuse gold-consistent prune/flatten/finetune

REPO = Path("/home/jichengzhi/V2X")
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
OUT = REPO / "results/phase1_minneck_probe"
OUT.mkdir(parents=True, exist_ok=True)


def pytorch_ap_eval(tag, model_dir, gpu):
    """PyTorch AP eval via HEAL opencood inference.py on DAIR val (no TRT)."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logf = OUT / f"{tag}_apeval.log"
    cmd = [PY, "opencood/tools/inference.py",
           "--model_dir", str(model_dir),
           "--fusion_method", "intermediate"]
    with open(logf, "w") as lf:
        r = subprocess.run(cmd, cwd=HEAL, env=env, stdout=lf,
                           stderr=subprocess.STDOUT, timeout=3600)
    txt = logf.read_text(errors="ignore")
    # opencood prints "The Average Precision at IOU 0.7 is ..."
    ap = {}
    for iou, key in ((0.3, "ap30"), (0.5, "ap50"), (0.7, "ap70")):
        m = re.search(rf"IOU\s*{iou}\s*is\s*([0-9.]+)", txt)
        if m:
            ap[key] = float(m.group(1))
    return ap, r.returncode


def main():
    gpu = sys.argv[1]
    port = int(sys.argv[2])
    widths = [tuple(int(x) for x in w.split("x")) for w in sys.argv[3].split(",")]
    B.GPU = str(gpu)
    B.MASTER_PORT = port
    for nf in widths:
        tag = "mn_%d_%d_%d" % nf
        res = {"tag": tag, "num_filters": list(nf), "gpu": gpu,
               "protocol": "gold stage_a: structural_prune L1 wpg4 g32 + 8ep finetune + PyTorch AP eval"}
        try:
            print(f"[{tag}] prune {nf} ...", flush=True)
            out_dir = B.run_prune(tag, nf)
            B.flatten_init_ckpt(out_dir, tag)
            B.patch_epoches(out_dir, tag)
            print(f"[{tag}] finetune (gpu {gpu}) ...", flush=True)
            ok = B.run_finetune(out_dir, tag)
            res["finetune_ok"] = bool(ok)
            if ok:
                print(f"[{tag}] AP eval ...", flush=True)
                ap, rc = pytorch_ap_eval(tag, out_dir, gpu)
                res["ap"] = ap
                res["ap_eval_rc"] = rc
            res["model_dir"] = str(out_dir)
        except Exception as e:
            res["error"] = f"{type(e).__name__}: {e}"
        json.dump(res, open(OUT / f"{tag}_ap.json", "w"), indent=2)
        print(f"[{tag}] DONE -> {res.get('ap')}", flush=True)


if __name__ == "__main__":
    main()
