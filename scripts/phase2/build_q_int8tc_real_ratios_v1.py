"""Build results/q_int8tc_real_ratios_v1.json from real H800 int8_tc measurements.

Part of the P×Q×S ablation "real caliber" upgrade (framework/run_pqs_ablation_real_v1.py).
Replaces the UNIFORM_INT8_SPEEDUP=1.449 proxy (a stage0-submodule-only dp4a
microbenchmark, results/q_int8_ms_stage0_result.json) with PER-WIDTH, FULL-BACKBONE
real measurements from the original60 SMBO loop
(multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/smbo_loop/).

For every width in the P×Q×S ablation grid (results/gap1_grid_corrected.json,
9 widths, labels base/p50/p75/trap25/pad64/mix_b/s1_64/mix_d/s2_128), this script:
  1. Looks for a same-batch (fp16, int8_tc) MEASURED pair at schedule="tuned" in the
     SMBO loop's diag_anchors / round{1,2,3}_measured / round{1,2,3}_int8tc_measured
     directories (files are per-(width,precision), num_filters-keyed, NOT the
     different "cur_width" units used by the dense 1029-point joint-search grid).
  2. If found: records the empirical ratio_int8tc_over_fp16 = fp16_lat_tuned_ms /
     int8tc_lat_tuned_ms (computed WITHIN the same measurement batch/backend, so it
     is backend-consistent even though the absolute values are NOT directly
     comparable to the older results/gap1_grid_corrected.json seed-grid fp16 numbers
     -- see CAVEATS below) plus the real energy_j for both precisions.
  3. If NOT found: the width is emitted under "missing_grid_widths" with the exact
     measure_config.py commands (fp16 + int8_tc) needed to close the gap on H800.

CAVEATS (read before using the output):
  * The int8_tc measurements are "h800_tvm_int8_rewritten_tensorcore" (im2col+MMA,
    genuinely tensorized: tensorcore_gate=true, wmma_count>0) -- NOT the old NCHWc
    dp4a path (which the ORIGINAL run_pqs_ablation.py's structural buildability
    rule, in_per_g%4==0, was modeling). 2026-07-03 finding (see
    framework/run_pqs_ablation.py's --int8-buildable-all flag / comment): under
    int8_tc ALL widths build (including s0=48, in_per_g=3, historically
    "NOT_APPLICABLE" under dp4a per results/q_int8_dp4a_pairs.csv). Confirmed here:
    trap25 (s0=48) has a real, tensorized, build_success=true int8_tc measurement.
    => The real-caliber run should use --int8-buildable-all (qlut.enforce_int8_buildable
       = False), NOT the legacy dp4a wall. This changes the ablation's headline
       claim from "categorical buildability wall" to "quantitative speedup /
       rank-order", which is only as strong as the coverage below.
  * The fp16 numbers used for the ratio come from the SAME SMBO batch
    ("h800_tvm_fp16_rewritten_tensorcore" backend), which is INTERNALLY CONSISTENT
    with the int8_tc numbers but NOT numerically consistent with the older
    results/gap1_grid_corrected.json seed-grid "tuned_us" fp16 baseline that
    LatencyLUT()/the rest of the ablation still uses (observed up to ~4x absolute
    discrepancy for trap25 -- almost certainly an older/different TVM backend or
    scope). => This script stores a RATIO, not an absolute latency, so it can be
    applied multiplicatively to whatever fp16 baseline the ablation's LatencyLUT
    already resolves (keeps the rest of the Pareto axis unchanged/comparable to the
    shipped proxy run; only the int8 branch's magnitude is corrected using a real,
    per-width factor instead of a single global constant).
  * The AP delta for int8 (_INT8_AP_DELTA_MEDIAN=-0.008 in search_three_arm.py) is
    STILL a historical-TRT-derived proxy. This script does NOT touch it -- AP under
    the real int8_tc backbone has not been separately re-evaluated. Flagged as an
    open gap in the "real caliber" driver's meta, not fixed here.

Run: python -m scripts.phase2.build_q_int8tc_real_ratios_v1
Output: results/q_int8tc_real_ratios_v1.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
SMBO_DIR = (ROOT / "multi_agent" / "data" / "stage2_lut_generation_v1" / "generated"
            / "original60_quant_20260627" / "smbo_loop")
OUT_JSON = RESULTS / "q_int8tc_real_ratios_v1.json"

sys.path.insert(0, str(ROOT))

GLOBS = [
    "diag_anchors/*.json",
    "round*_measured/*.json",
    "round*_int8tc_measured/*.json",
]


def _load_smbo_measurements() -> dict[tuple[int, int, int], dict[str, dict]]:
    """width -> {precision: (record, source_path)}."""
    out: dict[tuple, dict] = {}
    for pattern in GLOBS:
        for p in glob.glob(str(SMBO_DIR / pattern)):
            try:
                d = json.loads(Path(p).read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if "width" not in d or "precision" not in d:
                continue
            w = tuple(int(x) for x in d["width"])
            out.setdefault(w, {})[d["precision"]] = (d, p)
    return out


def _measure_cmd(width: tuple[int, int, int], precision: str) -> str:
    w = ",".join(str(x) for x in width)
    return (f"python framework/measure_config.py --width {w} "
            f"--precision {precision} --gpu <idle_gpu>")


def _real_grid_widths() -> list[tuple[tuple[int, int, int], str]]:
    """Reproduce the EXACT width grid run_pqs_ablation.py's run() uses:
    candidate_widths(lut, apm) over the real B1 LatencyLUT + B2 APModel, not the
    raw results/gap1_grid_corrected.json 'grid' list (which is a superset/subset
    with different labels -- iso_s0/iso_s1/iso_s2 have no ap70 and are NOT part of
    the actual searchable grid; mix_b/s1_64/mix_d/s2_128 ARE, via the B2 AP table).
    """
    from framework.search_three_arm import APModel, LatencyLUT, candidate_widths

    lut = LatencyLUT()
    apm = APModel()
    widths = candidate_widths(lut, apm)
    # SMOKE_WIDTHS_PQS labels (framework/search_three_arm.py) for readability only.
    label_map = {
        (64, 128, 256): "base", (32, 64, 128): "p50", (16, 32, 64): "p75",
        (48, 96, 192): "trap25", (64, 96, 192): "pad64", (48, 64, 256): "mix_b",
        (64, 64, 256): "s1_64", (48, 128, 128): "mix_d", (64, 128, 128): "s2_128",
    }
    out = []
    for w in widths:
        w = tuple(int(x) for x in w)
        out.append((w, label_map.get(w, str(w))))
    return out


def main() -> None:
    grid = _real_grid_widths()
    smbo = _load_smbo_measurements()

    covered: list[dict] = []
    missing: list[dict] = []

    for w, label in grid:
        entry = smbo.get(w, {})
        fp16 = entry.get("fp16")
        int8tc = entry.get("int8_tc")
        if fp16 and int8tc:
            fr, fp = fp16
            ir, ip = int8tc
            fp16_ms = fr["lat_tuned_ms"]
            int8_ms = ir["lat_tuned_ms"]
            fp16_e = fr.get("energy_j")
            int8_e = ir.get("energy_j")
            covered.append({
                "label": label,
                "num_filters": list(w),
                "sched": "tuned",
                "fp16_lat_tuned_ms": fp16_ms,
                "int8tc_lat_tuned_ms": int8_ms,
                "ratio_int8tc_over_fp16": round(fp16_ms / int8_ms, 4),
                "fp16_energy_j": fp16_e,
                "int8tc_energy_j": int8_e,
                "energy_ratio_fp16_over_int8tc": (
                    round(fp16_e / int8_e, 4) if fp16_e and int8_e else None),
                "int8tc_build_success": ir.get("build_success"),
                "int8tc_tensorcore_gate": ir.get("tensorcore_gate"),
                "int8tc_quant_method": ir.get("quant_method"),
                "int8tc_rewrite_max_abs_err": ir.get("rewrite_max_abs_err"),
                "source_files": [str(Path(fp).relative_to(ROOT)),
                                 str(Path(ip).relative_to(ROOT))],
            })
        else:
            need = []
            if not fp16:
                need.append("fp16")
            if not int8tc:
                need.append("int8_tc")
            missing.append({
                "label": label,
                "num_filters": list(w),
                "missing_precisions": need,
                "measure_cmds": [_measure_cmd(w, p) for p in need],
                "reason": ("no matching (width,precision) record found under "
                           f"{SMBO_DIR.relative_to(ROOT)}"),
            })

    out = {
        "schema": "q_int8tc_real_ratio_v1",
        "generated_by": "scripts/phase2/build_q_int8tc_real_ratios_v1.py",
        "source_dir": str(SMBO_DIR.relative_to(ROOT)),
        "grid_source": ("framework.search_three_arm.candidate_widths(LatencyLUT(), "
                         "APModel()) -- the exact grid run_pqs_ablation.py searches"),
        "n_grid_widths": len(grid),
        "n_covered": len(covered),
        "n_missing": len(missing),
        "caveats": [
            "int8_tc = h800_tvm_int8_rewritten_tensorcore (im2col+MMA, genuinely "
            "tensorized: tensorcore_gate=true); NOT the old NCHWc dp4a path.",
            "2026-07-03 finding: under int8_tc ALL widths build (incl. s0=48/32/16, "
            "historically dp4a-'NOT_APPLICABLE'). Real-caliber runs should set "
            "enforce_int8_buildable=False (--int8-buildable-all), not the legacy wall.",
            "ratio_int8tc_over_fp16 is computed WITHIN the same SMBO measurement "
            "batch (backend-consistent); the absolute fp16_lat_tuned_ms here is NOT "
            "numerically consistent with the older gap1_grid_corrected.json seed-grid "
            "fp16 'tuned_us' (observed up to ~4x discrepancy, e.g. trap25). Apply the "
            "RATIO multiplicatively to the ablation's own LatencyLUT fp16 value; do "
            "not swap in the absolute SMBO fp16 number.",
            "AP delta for int8 (_INT8_AP_DELTA_MEDIAN=-0.008) remains a historical-TRT "
            "proxy, NOT re-measured under int8_tc. Open gap, not addressed here.",
            "int8_tc measured ONLY at sched='tuned' (metaschedule-tuned build); no "
            "sched='default' int8_tc data exists yet for any width.",
        ],
        "covered_widths": covered,
        "missing_grid_widths": missing,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON.relative_to(ROOT)}: {len(covered)}/{len(grid)} grid widths "
          f"covered by real int8_tc, {len(missing)} missing")
    for m in missing:
        print(f"  MISSING {m['label']:8s} {m['num_filters']} needs "
              f"{m['missing_precisions']}")


if __name__ == "__main__":
    main()
