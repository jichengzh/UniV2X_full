#!/bin/bash
# GPU 1 chain: wg_pair4 finetune already running
# After it completes: eval pair4, then pair5 finetune+eval, then pair6 finetune+eval
# Usage: bash l4_chain_gpu1.sh >> /home/jichengzhi/V2X/results/l4_gpu1_chain.log 2>&1 &

set -e
REPO=/home/jichengzhi/V2X
PY=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python
HEAL=/home/jichengzhi/heal_research/HEAL
CKROOT=/home/jichengzhi/heal_research/checkpoints/stage1
GPU=1
PORT=29822
RESULT_JSON=$REPO/results/l4_b2_ap_finetune.json

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$HEAL

echo "[GPU1 chain] START $(date)"

### wg_pair4: wait for current finetune ###
echo "[GPU1 chain] Waiting for wg_pair4 finetune (PID=2676558)..."
wait 2676558 2>/dev/null || true
PAIR4_DIR=$CKROOT/Pyramid_DAIR_m1_l4_wg_pair4_2026_06_21
while true; do
    RUNNING=$(pgrep -f "model_dir.*wg_pair4" 2>/dev/null | head -1)
    if [ -z "$RUNNING" ]; then
        echo "[GPU1 chain] wg_pair4 finetune done"
        break
    fi
    sleep 60
    echo "[GPU1 chain] wg_pair4 still running... $(date)"
done

### eval wg_pair4 ###
echo "[GPU1 chain] Evaluating wg_pair4..."
rm -f $PAIR4_DIR/eval*.yaml
$PY opencood/tools/inference.py \
    --model_dir $PAIR4_DIR \
    --fusion_method intermediate \
    > $REPO/results/l4_wg_pair4_eval.log 2>&1
echo "[GPU1 chain] pair4 eval done rc=$?"

### wg_pair5 finetune ###
PAIR5_DIR=$CKROOT/Pyramid_DAIR_m1_l4_wg_pair5_2026_06_21
echo "[GPU1 chain] Starting wg_pair5 finetune..."
$PY -m torch.distributed.launch \
    --nproc_per_node=1 --use_env --master_port=$PORT \
    $HEAL/opencood/tools/train_ddp.py \
    --hypes_yaml $PAIR5_DIR/config.yaml \
    --model_dir $PAIR5_DIR \
    --half \
    >> $REPO/results/l4_wg_pair5_finetune.log 2>&1
echo "[GPU1 chain] pair5 finetune done rc=$?"

### eval wg_pair5 ###
echo "[GPU1 chain] Evaluating wg_pair5..."
rm -f $PAIR5_DIR/eval*.yaml
$PY opencood/tools/inference.py \
    --model_dir $PAIR5_DIR \
    --fusion_method intermediate \
    > $REPO/results/l4_wg_pair5_eval.log 2>&1
echo "[GPU1 chain] pair5 eval done rc=$?"

### wg_pair6 finetune ###
PAIR6_DIR=$CKROOT/Pyramid_DAIR_m1_l4_wg_pair6_2026_06_21
echo "[GPU1 chain] Starting wg_pair6 finetune..."
$PY -m torch.distributed.launch \
    --nproc_per_node=1 --use_env --master_port=$PORT \
    $HEAL/opencood/tools/train_ddp.py \
    --hypes_yaml $PAIR6_DIR/config.yaml \
    --model_dir $PAIR6_DIR \
    --half \
    >> $REPO/results/l4_wg_pair6_finetune.log 2>&1
echo "[GPU1 chain] pair6 finetune done rc=$?"

### eval wg_pair6 ###
echo "[GPU1 chain] Evaluating wg_pair6..."
rm -f $PAIR6_DIR/eval*.yaml
$PY opencood/tools/inference.py \
    --model_dir $PAIR6_DIR \
    --fusion_method intermediate \
    > $REPO/results/l4_wg_pair6_eval.log 2>&1
echo "[GPU1 chain] pair6 eval done rc=$?"

### parse and write results ###
$PY -c "
import json, yaml, re
from pathlib import Path
RESULT_JSON = Path('$RESULT_JSON')
results = json.loads(RESULT_JSON.read_text()) if RESULT_JSON.exists() else {}

pairs = [
    ('wg_pair4', '$PAIR4_DIR', [48,96,256], 4),
    ('wg_pair5', '$PAIR5_DIR', [48,32,128], 4),
    ('wg_pair6', '$PAIR6_DIR', [48,64,192], 4),
]
for tag, ckpt_dir, nf, wpg in pairs:
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
    bests = list(d.glob('net_epoch_bestval_at*.pth'))
    best = max(bests, key=lambda p: int(re.search(r'at(\d+)\.pth', p.name).group(1))).name if bests else None
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

echo "[GPU1 chain] DONE $(date)"
