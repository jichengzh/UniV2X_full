# Existing H800+TVM Evidence Inventory

All normalized latency columns in these CSV files are in `ms`. Raw source files often use `us`.

Files:

- `pyramid_h800_tvm_backbone_fp16_ms.csv`: Pyramid backbone-only FP16 H800 TVM LUT seed from `results/latency_lut_pyramid.json`.
- `pyramid_h800_tvm_gap1_corrected_ms.csv`: corrected Gap1 rows and W_g/P_g evidence from `results/gap1_grid_corrected.json`.
- `pyramid_h800_tvm_int8_scope_limited_ms.csv`: scope-limited Pyramid INT8/stage0 evidence; not a full-backbone INT8 LUT.
- `codriving_h800_tvm_backbone_ms.csv`: CoDriving backbone H800 TVM measured rows from `results/codriving_tvm_p25_p75.csv`.
- `codriving_h800_tvm_s0_and_int8_probes_ms.csv`: CoDriving s0 and INT8 micro-probes.
- `h800_tvm_existing_summary.json`: inventory summary and caveats.

Quick audit:

```bash
cd /home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/existing_h800_tvm
python -m json.tool h800_tvm_existing_summary.json | sed -n '1,120p'
for f in *.csv; do echo "$f"; tail -n +2 "$f" | wc -l; done
```
