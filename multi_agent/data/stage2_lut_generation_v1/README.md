# Stage2 LUT Generation Data Directory v1

This directory stores immutable seed data from `multi_agent/data/dataset_v2.csv`, validation outputs, generated LUT rows, registry snapshots, logs, and raw artifacts.

Quick checks:

```bash
wc -l existing/dataset_v2_ap_valid_63.csv
python -m json.tool existing/dataset_v2_summary.json | sed -n "1,120p"
find . -maxdepth 2 -type f | sort
```
