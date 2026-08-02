"""RealQLookup: real-per-width int8_tc calibration for the P×Q×S ablation.

Loads results/q_int8tc_real_ratios_v1.json (built by
scripts/phase2/build_q_int8tc_real_ratios_v1.py from the original60 SMBO loop's
real H800 int8_tc measurements) and applies each width's REAL, MEASURED
fp16->int8_tc speedup ratio to whatever fp16 latency the ablation's own
LatencyLUT already resolves for that width -- instead of the single global
UNIFORM_INT8_SPEEDUP=1.449 proxy constant used by the (untouched) proxy version
in framework/run_pqs_ablation.py.

Design rationale (why a subclass + ratio, not "just populate _h800_int8"):
  * QLookup.int8_lat() already supports injecting an ABSOLUTE h800_tvm_int8_*_us
    value per (width, sched) via the Q-LUT JSON schema
    ("h800_tvm_int8_default_us"/"h800_tvm_int8_tuned_us"), which takes precedence
    over the uniform proxy. That mechanism was considered and rejected here:
    the SMBO loop's real int8_tc measurements are numerically consistent with its
    OWN same-batch fp16 companion measurement, but NOT with the older
    results/gap1_grid_corrected.json seed-grid fp16 baseline that LatencyLUT()
    still serves to the rest of the ablation (up to ~4x absolute discrepancy
    observed for some widths -- almost certainly a different/older TVM backend
    revision; see project memory "精度轴双重口径不对等"). Injecting the SMBO's
    absolute int8_tc microsecond value directly against the OLD fp16 baseline
    would silently produce a nonsensical speedup number.
  * The fix: store and apply the REAL RATIO (fp16_smbo / int8tc_smbo), computed
    entirely within the SAME internally-consistent SMBO batch, and multiply it
    onto the ablation's own (unchanged) fp16 latency. This keeps the rest of the
    Pareto axis exactly as the proxy version computes it; only the int8 branch's
    MAGNITUDE is corrected from "one constant everywhere" to "per-width real
    measurement where available, else explicitly unmeasured".

has_direct_h800() is overridden so CostModelPQS.evaluate() (unmodified, in
search_three_arm.py) correctly tags these records with lat_source =
"H800_TVM_int8_real" via its existing logic -- no changes to search_three_arm.py
are needed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from framework.search_three_arm import QLookup, Width

REAL_RATIO_JSON = Path(__file__).resolve().parent.parent / "results" / "q_int8tc_real_ratios_v1.json"


class RealQLookup(QLookup):
    """QLookup that prefers a per-width REAL int8_tc/fp16 ratio (H800, SMBO loop,
    original60_quant_20260627) over the uniform proxy. Widths without a real
    ratio fall through to whatever `uniform_int8_speedup`/`enforce_int8_buildable`
    the caller configured on the base QLookup (default: no proxy fallback -> the
    honest "unmeasured, neutral fp16 latency" path in CostModelPQS.evaluate()).
    """

    def __init__(self, *args, real_ratio_path: Path = REAL_RATIO_JSON, **kwargs):
        super().__init__(*args, **kwargs)
        self._real_ratio: dict[Width, dict] = {}
        self._real_ratio_meta: dict = {}
        self._real_ratio_source = str(real_ratio_path)
        if real_ratio_path.exists():
            self._load_real_ratio(real_ratio_path)

    def _load_real_ratio(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self._real_ratio_meta = {
            k: v for k, v in data.items()
            if k not in ("covered_widths", "missing_grid_widths")
        }
        for e in data.get("covered_widths", []):
            w = tuple(int(x) for x in e["num_filters"])
            self._real_ratio[w] = e

    # -- lookups used by the report/driver, not by CostModelPQS -------------
    def missing_widths(self) -> list[dict]:
        data = json.loads(Path(self._real_ratio_source).read_text())
        return data.get("missing_grid_widths", [])

    def real_ratio_record(self, width: Width) -> Optional[dict]:
        return self._real_ratio.get(tuple(int(x) for x in width))

    # -- QLookup overrides ----------------------------------------------------
    def has_direct_h800(self, width: Width, sched: str = "tuned") -> bool:
        w = tuple(int(x) for x in width)
        if sched == "tuned" and w in self._real_ratio:
            return True
        return super().has_direct_h800(width, sched)

    def int8_lat(self, h800_tvm_us: float, width: Width, sched: str = "tuned") -> float:
        w = tuple(int(x) for x in width)
        if sched == "tuned" and w in self._real_ratio:
            ratio = self._real_ratio[w]["ratio_int8tc_over_fp16"]
            return h800_tvm_us / ratio
        # sched="default" (no int8_tc data exists at default schedule for any
        # width yet) or a width outside the 4 real-covered ones: defer to
        # whatever the driver configured (proxy on/off) on the base class.
        return super().int8_lat(h800_tvm_us, width, sched)

    def energy_ratio(self, width: Width) -> Optional[float]:
        """fp16_energy_j / int8tc_energy_j, real-measured, else None."""
        rec = self.real_ratio_record(width)
        return rec.get("energy_ratio_fp16_over_int8tc") if rec else None
