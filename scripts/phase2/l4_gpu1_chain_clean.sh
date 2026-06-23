#!/bin/bash
# L4 GPU 1 chain: wg_pair4 -> wg_pair5 -> wg_pair6
# Run as: bash l4_gpu1_chain_clean.sh > /home/jichengzhi/V2X/results/l4_gpu1_chain_clean.log 2>&1
set -e

HEAL=/home/jichengzhi/heal_research/HEAL
PY=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python
CKROOT=/home/jichengzhi/heal_research/checkpoints/stage1
REPO=/home/jichengzhi/V2X
RESULT_JSON=$REPO/results/l4_b2_ap_finetune.json

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$HEAL

cd $HEAL

echo "[GPU1] START $(date)"

### wg_pair4 ###
PAIR4=$CKROOT/Pyramid_DAIR_m1_l4_wg_pair4_2026_06_21
echo "[GPU1] Finetune wg_pair4 START $(date)"
$PY opencood/tools/train.py \
    --hypes_yaml $PAIR4/config.yaml \
    --model_dir $PAIR4 \
    > $REPO/results/l4_wg_pair4_finetune.log 2>&1
echo "[GPU1] pair4 finetune done rc=$? $(date)"

echo "[GPU1] Eval wg_pair4 START $(date)"
rm -f $PAIR4/eval*.yaml
$PY opencood/tools/inference.py \
    --model_dir $PAIR4 \
    --fusion_method intermediate \
    > $REPO/results/l4_wg_pair4_eval.log 2>&1
echo "[GPU1] pair4 eval done rc=$? $(date)"

### wg_pair5 ###
PAIR5=$CKROOT/Pyramid_DAIR_m1_l4_wg_pair5_2026_06_21
echo "[GPU1] Finetune wg_pair5 START $(date)"
$PY opencood/tools/train.py \
    --hypes_yaml $PAIR5/config.yaml \
    --model_dir $PAIR5 \
    > $REPO/results/l4_wg_pair5_finetune.log 2>&1
echo "[GPU1] pair5 finetune done rc=$? $(date)"

echo "[GPU1] Eval wg_pair5 START $(date)"
rm -f $PAIR5/eval*.yaml
$PY opencood/tools/inference.py \
    --model_dir $PAIR5 \
    --fusion_method intermediate \
    > $REPO/results/l4_wg_pair5_eval.log 2>&1
echo "[GPU1] pair5 eval done rc=$? $(date)"

### wg_pair6 ###
PAIR6=$CKROOT/Pyramid_DAIR_m1_l4_wg_pair6_2026_06_21
echo "[GPU1] Finetune wg_pair6 START $(date)"
$PY opencood/tools/train.py \
    --hypes_yaml $PAIR6/config.yaml \
    --model_dir $PAIR6 \
    > $REPO/results/l4_wg_pair6_finetune.log 2>&1
echo "[GPU1] pair6 finetune done rc=$? $(date)"

echo "[GPU1] Eval wg_pair6 START $(date)"
rm -f $PAIR6/eval*.yaml
$PY opencood/tools/inference.py \
    --model_dir $PAIR6 \
    --fusion_method intermediate \
    > $REPO/results/l4_wg_pair6_eval.log 2>&1
echo "[GPU1] pair6 eval done rc=$? $(date)"

### parse results ###
$PY - << 'PYEOF'
import json, yaml, re
from pathlib import Path

REPO = Path("/home/jichengzhi/V2X")
CKROOT = Path("/home/jichengzhi/heal_research/checkpoints/stage1")
RESULT_JSON = REPO / "results/l4_b2_ap_finetune.json"

results = json.loads(RESULT_JSON.read_text()) if RESULT_JSON.exists() else {}

configs = [
    ("wg_pair4", CKROOT/"Pyramid_DAIR_m1_l4_wg_pair4_2026_06_21", [48,96,256], 4),
    ("wg_pair5", CKROOT/"Pyramid_DAIR_m1_l4_wg_pair5_2026_06_21", [48,32,128], 4),
    ("wg_pair6", CKROOT/"Pyramid_DAIR_m1_l4_wg_pair6_2026_06_21", [48,64,192], 4),
]

for tag, d, nf, wpg in configs:
    ap = None
    for ev in sorted(d.glob("eval*.yaml")):
        y = yaml.safe_load(ev.read_text()) or {}
        ap30 = float(y.get("ap30") or y.get("ap_30") or 0)
        ap50 = float(y.get("ap_50") or y.get("ap50") or 0)
        ap70 = float(y.get("ap_70") or y.get("ap70") or 0)
        ap = {"ap30": round(ap30,4), "ap50": round(ap50,4), "ap70": round(ap70,4)}
        break
    if ap is None:
        log = REPO / f"results/l4_{tag}_eval.log"
        if log.exists():
            m = re.search(r"0\.3 is ([0-9.]+).*?0\.5 is ([0-9.]+).*?0\.7 is ([0-9.]+)", log.read_text(), re.DOTALL)
            if m:
                ap = {"ap30": round(float(m.group(1)),4), "ap50": round(float(m.group(2)),4), "ap70": round(float(m.group(3)),4)}
    bests = list(d.glob("net_epoch_bestval_at*.pth"))
    best = max(bests, key=lambda p: int(re.search(r"at(\d+)\.pth", p.name).group(1))).name if bests else None
    results[tag] = {
        "num_filters": nf, "wpg": wpg, "groups": 32,
        "ap70": ap["ap70"] if ap else None,
        "ap50": ap["ap50"] if ap else None,
        "ap30": ap["ap30"] if ap else None,
        "bestval_epoch": best,
        "ckpt": str(d),
        "status": "ok" if (ap and (ap["ap70"] or 0) > 0.01) else "eval_failed",
    }
    print(f"{tag}: AP70={results[tag]['ap70']} AP50={results[tag]['ap50']}")

RESULT_JSON.write_text(json.dumps(results, indent=2))
print(f"Written to {RESULT_JSON}")
PYEOF

echo "[GPU1] DONE $(date)"
