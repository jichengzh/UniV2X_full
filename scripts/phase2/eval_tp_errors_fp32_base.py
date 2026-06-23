"""Quantization-axis 3rd level: base anchor pure-FP32 TRT, full DAIR val 1789.
Reuses run_one() from eval_tp_errors_corrected_full.py (same pipeline/criteria)."""
import importlib.util, json, csv
from pathlib import Path

SRC = Path("/home/jichengzhi/UniV2X/scripts/phase2/eval_tp_errors_corrected_full.py")
spec = importlib.util.spec_from_file_location("eval_full", SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

r = m.run_one("base",
              "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29",
              None, "fp32")
assert r is not None, "run_one returned None (engine missing?)"

out = Path("/home/jichengzhi/UniV2X/results/tp_errors_fp32_base.json")
out.write_text(json.dumps(r, indent=2, default=float))
with open("/home/jichengzhi/UniV2X/results/tp_errors_fp32_base.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(r.keys()))
    w.writeheader(); w.writerow(r)
print(json.dumps({k: r[k] for k in r if not k.startswith("n_") or k == "n_tp"}, indent=2, default=float))
