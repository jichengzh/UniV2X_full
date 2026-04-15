#!/bin/bash
# FFN 剪枝比例扫描实验
# 顺序跑 20% / 40% / 50% / 60%, 每个约 17-20 分钟
# 每个实验都保存 pkl + 评估输出

set -e
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/jichengzhi/UniV2X
PYBIN=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python

CONFIG=projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py
CKPT=ckpts/univ2x_coop_e2e_stg2.pth

cd /home/jichengzhi/UniV2X

for RATIO in 20 40 50 60; do
    echo ""
    echo "====================================="
    echo "  Running FFN ${RATIO}% pruning experiment"
    echo "====================================="
    date

    PRUNE_CFG=prune_configs/p1_ffn_${RATIO}pct.json
    PKL_OUT=output/p1_ffn_${RATIO}pct_results.pkl
    METRICS_OUT=output/p1_ffn_${RATIO}pct_metrics.json
    LOG=output/p1_ffn_${RATIO}pct_run.log

    # Step 1: inference + save pkl
    $PYBIN tools/test_with_pruning.py \
        $CONFIG $CKPT \
        --prune-config $PRUNE_CFG \
        --out $PKL_OUT \
        2>&1 | tee $LOG

    # Step 2: evaluate using eval_pkl_amota (避开 tmp_dir GC 竞态)
    echo ""
    echo "--- Evaluating ${RATIO}% pkl ---"
    $PYBIN tools/eval_pkl_amota.py \
        $CONFIG $PKL_OUT \
        2>&1 | tail -25 | tee -a $LOG

    date
    echo "FFN ${RATIO}% DONE"
done

echo ""
echo "All sweep experiments finished!"
