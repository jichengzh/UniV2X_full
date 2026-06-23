# Plan v5 Phase C — TRT engine real INT8 AP

Subset size: 1789 DAIR-V2X val samples (full val: 1789)

## Per anchor AP

| tag | Q | n_ok/n | AP30 | AP50 | AP70 | eval_secs | status |
|---|---|---|---|---|---|---|---|
| p64_baseline | fp32 | 1789/1789 | 0.8349 | 0.7925 | 0.6333 | 193.2 | ok |
| p48 | fp32 | 1789/1789 | 0.8312 | 0.7903 | 0.6331 | 193.1 | ok |
| p32 | fp32 | 1789/1789 | 0.8363 | 0.79 | 0.615 | 196.9 | ok |
| p16 | fp32 | 1789/1789 | 0.8277 | 0.784 | 0.625 | 186.3 | ok |
| p8 | fp32 | 1789/1789 | 0.8309 | 0.7839 | 0.6137 | 191.9 | ok |

## Gate G_C analysis
Compare INT8 AP vs FP32 AP per anchor; |gap| <= 0.02 → H3 PASS