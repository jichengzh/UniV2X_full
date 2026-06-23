#!/bin/bash
# GPU 0 chain: cliff_a_wpg8 finetune already running
# After it completes: eval cliff_a, then finetune cliff_b, eval cliff_b
# Usage: bash l4_chain_gpu0.sh >> /home/jichengzhi/V2X/results/l4_gpu0_chain.log 2>&1 &

set -e
REPO=/home/jichengzhi/V2X
PY=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python
HEAL=/home/jichengzhi/heal_research/HEAL
CKROOT=/home/jichengzhi/heal_research/checkpoints/stage1
GPU=0
PORT=29812
RESULT_JSON=$REPO/results/l4_b2_ap_finetune.json

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$HEAL

echo "[GPU0 chain] START $(date)"

### cliff_a_wpg8: wait for finetune to finish ###
echo "[GPU0 chain] Waiting for cliff_a_wpg8 finetune (PID=2672516)..."
wait 2672516 2>/dev/null || true
# Poll until no train_ddp running on GPU0 or bestval at epoch > 23 appears
CLIFF_A_DIR=$CKROOT/Pyramid_DAIR_m1_l4_cliff_a_wpg8_2026_06_21
while true; do
    # Check if training still going (any epoch > 23 pth with recent mtime)
    LATEST=$(ls -t $CLIFF_A_DIR/net_epoch*.pth 2>/dev/null | head -1)
    if [ -n "$LATEST" ] && echo "$LATEST" | grep -qv "bestval_at23"; then
        RUNNING=$(pgrep -f "model_dir.*cliff_a_wpg8" 2>/dev/null | head -1)
        if [ -z "$RUNNING" ]; then
            echo "[GPU0 chain] cliff_a_wpg8 finetune done (no training process)"
            break
        fi
    fi
    sleep 60
    echo "[GPU0 chain] cliff_a_wpg8 still running... $(date)"
done

### eval cliff_a_wpg8 ###
echo "[GPU0 chain] Evaluating cliff_a_wpg8..."
rm -f $CLIFF_A_DIR/eval*.yaml
$PY opencood/tools/inference.py \
    --model_dir $CLIFF_A_DIR \
    --fusion_method intermediate \
    > $REPO/results/l4_cliff_a_wpg8_eval.log 2>&1
echo "[GPU0 chain] cliff_a eval done rc=$?"

### cliff_b_wpg8 finetune ###
echo "[GPU0 chain] Starting cliff_b_wpg8 finetune..."
CLIFF_B_DIR=$CKROOT/Pyramid_DAIR_m1_l4_cliff_b_wpg8_2026_06_21
$PY -m torch.distributed.launch \
    --nproc_per_node=1 --use_env --master_port=$PORT \
    $HEAL/opencood/tools/train_ddp.py \
    --hypes_yaml $CLIFF_B_DIR/config.yaml \
    --model_dir $CLIFF_B_DIR \
    --half \
    >> $REPO/results/l4_cliff_b_wpg8_finetune.log 2>&1
echo "[GPU0 chain] cliff_b finetune done rc=$?"

### eval cliff_b_wpg8 ###
echo "[GPU0 chain] Evaluating cliff_b_wpg8..."
rm -f $CLIFF_B_DIR/eval*.yaml
$PY opencood/tools/inference.py \
    --model_dir $CLIFF_B_DIR \
    --fusion_method intermediate \
    > $REPO/results/l4_cliff_b_wpg8_eval.log 2>&1
echo "[GPU0 chain] cliff_b eval done rc=$?"

### parse and write results ###
$PY -c "
import json, yaml, re
from pathlib import Path
RESULT_JSON = Path('$RESULT_JSON')
results = json.loads(RESULT_JSON.read_text()) if RESULT_JSON.exists() else {}

for tag, ckpt_dir in [
    ('cliff_a_wpg8', '$CLIFF_A_DIR'),
    ('cliff_b_wpg8', '$CLIFF_B_DIR'),
]:
    d = Path(ckpt_dir)
    ap = None
    for ev in d.glob('eval*.yaml'):
        y = yaml.safe_load(ev.read_text()) or {}
        ap30 = float(y.get('ap30') or y.get('ap_30') or 0)
        ap50 = float(y.get('ap_50') or y.get('ap50') or 0)
        ap70 = float(y.get('ap_70') or y.get('ap70') or 0)
        ap = {'ap30': round(ap30,4), 'ap50': round(ap50,4), 'ap70': round(ap70,4)}
        break
    if ap is None:
        log = Path('$REPO/results/l4_{}_eval.log'.format(tag))
        if log.exists():
            m = re.search(r'0\.3 is ([0-9.]+).*?0\.5 is ([0-9.]+).*?0\.7 is ([0-9.]+)', log.read_text(), re.DOTALL)
            if m:
                ap = {'ap30': round(float(m.group(1)),4), 'ap50': round(float(m.group(2)),4), 'ap70': round(float(m.group(3)),4)}
    # best ckpt
    bests = list(d.glob('net_epoch_bestval_at*.pth'))
    best = max(bests, key=lambda p: int(re.search(r'at(\d+)\.pth', p.name).group(1))).name if bests else None
    nf = {'cliff_a_wpg8': [8,16,32], 'cliff_b_wpg8': [8,8,16]}[tag]
    wpg = 8
    results[tag] = {
        'num_filters': nf, 'wpg': wpg, 'groups': 32,
        'ap70': ap['ap70'] if ap else None,
        'ap50': ap['ap50'] if ap else None,
        'ap30': ap['ap30'] if ap else None,
        'bestval_epoch': best,
        'ckpt': str(d),
        'status': 'ok' if ap else 'eval_failed',
    }
    print(f'{tag}: AP70={results[tag][\"ap70\"]}')
RESULT_JSON.write_text(json.dumps(results, indent=2))
print('Written to', RESULT_JSON)
" 2>&1

echo "[GPU0 chain] DONE $(date)"
