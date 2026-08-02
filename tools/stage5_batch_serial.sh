#!/usr/bin/env bash
# Run a list of configs serially on a single GPU pair.
# Avoids work_dir race conditions from parallel test.py runs.
#
# Usage:
#   bash tools/stage5_batch_serial.sh "B2 B3 C1 C2 C3" "1,2"

set -euo pipefail
CONFIGS="${1:?Usage: $0 \"<CONFIG_IDS>\" \"<GPUS>\"}"
GPUS="${2:-1,2}"

cd /home/jichengzhi/UniV2X
mkdir -p data/phase4/stage5_metrics

for cid in $CONFIGS; do
    echo ""
    echo "######################################################################"
    echo "###  Running $cid on GPU $GPUS  ###"
    echo "######################################################################"
    bash tools/stage5_run_one_config.sh "$cid" "$GPUS" 2>&1 | tail -30 || {
        echo "WARN: $cid failed, continuing..."
        echo "{\"label\": \"stage5_$cid\", \"error\": \"failed\"}" > "data/phase4/stage5_metrics/${cid}.json"
    }
done

echo ""
echo "=== BATCH DONE ==="
ls -1 data/phase4/stage5_metrics/*.json | wc -l
