#!/bin/bash
# L4 GPU 0 chain: cliff_a_wpg8 -> cliff_b_wpg8
# Run as: bash l4_gpu0_chain_clean.sh > /home/jichengzhi/V2X/results/l4_gpu0_chain_clean.log 2>&1
set -e

HEAL=/home/jichengzhi/heal_research/HEAL
PY=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python
CKROOT=/home/jichengzhi/heal_research/checkpoints/stage1
REPO=/home/jichengzhi/V2X
RESULT_JSON=$REPO/results/l4_b2_ap_finetune.json

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$HEAL

cd $HEAL

echo "[GPU0] START $(date)"

### cliff_a_wpg8 ###
CLIFF_A=$CKROOT/Pyramid_DAIR_m1_l4_cliff_a_wpg8_2026_06_21
echo "[GPU0] Finetune cliff_a_wpg8 START $(date)"
$PY opencood/tools/train.py \
    --hypes_yaml $CLIFF_A/config.yaml \
    --model_dir $CLIFF_A \
    > $REPO/results/l4_cliff_a_wpg8_finetune.log 2>&1
echo "[GPU0] cliff_a finetune done rc=$? $(date)"

echo "[GPU0] Eval cliff_a_wpg8 START $(date)"
rm -f $CLIFF_A/eval*.yaml
$PY opencood/tools/inference.py \
    --model_dir $CLIFF_A \
    --fusion_method intermediate \
    > $REPO/results/l4_cliff_a_wpg8_eval.log 2>&1
echo "[GPU0] cliff_a eval done rc=$? $(date)"

### cliff_b_wpg8 ###
CLIFF_B=$CKROOT/Pyramid_DAIR_m1_l4_cliff_b_wpg8_2026_06_21
echo "[GPU0] Finetune cliff_b_wpg8 START $(date)"
$PY opencood/tools/train.py \
    --hypes_yaml $CLIFF_B/config.yaml \
    --model_dir $CLIFF_B \
    > $REPO/results/l4_cliff_b_wpg8_finetune.log 2>&1
echo "[GPU0] cliff_b finetune done rc=$? $(date)"

echo "[GPU0] Eval cliff_b_wpg8 START $(date)"
rm -f $CLIFF_B/eval*.yaml
$PY opencood/tools/inference.py \
    --model_dir $CLIFF_B \
    --fusion_method intermediate \
    > $REPO/results/l4_cliff_b_wpg8_eval.log 2>&1
echo "[GPU0] cliff_b eval done rc=$? $(date)"

### parse results ###
$PY - << 'PYEOF'
import json, yaml, re
from pathlib import Path

REPO = Path("/home/jichengzhi/V2X")
CKROOT = Path("/home/jichengzhi/heal_research/checkpoints/stage1")
RESULT_JSON = REPO / "results/l4_b2_ap_finetune.json"

results = json.loads(RESULT_JSON.read_text()) if RESULT_JSON.exists() else {}

configs = [
    ("cliff_a_wpg8", CKROOT/"Pyramid_DAIR_m1_l4_cliff_a_wpg8_2026_06_21", [8,16,32], 8),
    ("cliff_b_wpg8", CKROOT/"Pyramid_DAIR_m1_l4_cliff_b_wpg8_2026_06_21", [8,8,16], 8),
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

echo "[GPU0] DONE $(date)"
