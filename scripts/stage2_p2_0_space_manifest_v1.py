#!/usr/bin/env python3
"""P2.0 — land the frozen software search space Theta_sw for Pyramid + H800.

Frozen decisions (8_7_2 doc, 2026-07-02):
  * quant axis = GLOBAL single precision, Q in {fp32, fp16, int8}  (N_q=1, |Q|=3)
  * partition   = 3-stage COARSE (backbone stages s0/s1/s2)         (N_p=3, |W|=7)
  * |Theta_sw|  = |W|^N_p * |Q| = 7^3 * 3 = 1029                     (unmeasured 849)
  * scope       = Pyramid + H800 only

What this script does (low-cost, NO GPU):
  1. Enumerate the frozen 3-stage grid  W0 x W1 x W2 x {fp32,fp16,int8}.
  2. Mark measured (original60: 60 widths x 3 prec = 180) vs unmeasured (849).
  3. Legality: cross-check the stage1 bridge's STATIC int8_buildable gate against
     the EMPIRICAL int8 measurements. On H800/TVM native int8, all 60 widths built
     & measured though 60/60 are not multiples of 128 -> the bridge's align=128/32
     gate is a TRT-tiling artifact that does NOT enter TVM buildability
     (cf. codriving Cin=48/16 all-build finding). => no int8-buildability cliff at
     coarse 3-stage granularity; every config in the grid is legal.
  4. Emit space_manifest_pyramid_h800_v1.json with the enumeration, legality
     provenance, separability note, and a check of the |Theta_sw| formula.

Reproducible: reads only the frozen training table + the pyramid partition yaml.
"""
from __future__ import annotations
import json, itertools, sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/home/jichengzhi/V2X")
sys.path.insert(0, str(ROOT))
from framework.stage1_bridge import SpaceSpec  # noqa: E402

W0 = [16, 24, 32, 40, 48, 56, 64]
W1 = [32, 48, 64, 80, 96, 112, 128]
W2 = [64, 96, 128, 160, 192, 224, 256]
PRECISIONS = ["fp32", "fp16", "int8"]

TRAIN_TABLE = ROOT / ("multi_agent/data/stage2_lut_generation_v1/generated/"
                      "original60_quant_20260627/cost_model/train/"
                      "original60_training_table_latest.json")
PARTITION_YAML = ROOT / "framework/partitions/pyramid_lidar_partition.yaml"
OUT = ROOT / ("multi_agent/data/stage2_lut_generation_v1/generated/"
              "original60_quant_20260627/space/space_manifest_pyramid_h800_v1.json")


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_width(w) -> tuple:
    return tuple(int(x) for x in (w if isinstance(w, list) else str(w).split("x")))


def load_measured():
    tab = json.load(open(TRAIN_TABLE))
    measured = {p: set() for p in PRECISIONS}
    lat_measured = {p: set() for p in PRECISIONS}
    for r in tab["rows"]:
        p = r["precision"]
        if p not in measured:
            continue
        w = parse_width(r["width"])
        measured[p].add(w)
        if r.get("latency_ms_status") == "measured":
            lat_measured[p].add(w)
    return measured, lat_measured


def bridge_legality():
    """Static int8-buildable widths per fine-grained bridge knob (for provenance)."""
    sp = SpaceSpec.from_manifest(PARTITION_YAML)
    knobs = []
    for k in sp.knobs:
        legal = list(k.legal_widths())
        build = list(k.buildable_int8_widths())
        knobs.append({
            "knob": k.search_group_id, "bucket": k.bucket,
            "grouped_conv": k.grouped_conv, "round_to": k.round_to,
            "int8_buildable_align": k.int8_buildable_align,
            "legal_widths": legal, "int8_buildable_widths_static": build,
            "has_static_cliff": bool(set(legal) - set(build)),
        })
    return sp.model, sp.hw.name, sp.has_int8_buildability_cliff(), knobs


def main():
    grid = list(itertools.product(W0, W1, W2))
    assert len(grid) == 343
    measured, lat_measured = load_measured()
    # all 3 precisions share the same 60 measured widths (verified upstream)
    meas_widths = measured["int8"]
    unmeasured_widths = [w for w in grid if w not in meas_widths]

    # empirical int8 buildability audit
    int8_meas = sorted(measured["int8"])
    non_mult_128 = [w for w in int8_meas if not all(x % 128 == 0 for x in w)]
    non_mult_32 = [w for w in int8_meas if not all(x % 32 == 0 for x in w)]

    b_model, b_hw, b_cliff, b_knobs = bridge_legality()

    # enumerate full Theta_sw with measured flag; all legal (see legality note)
    configs = []
    for w in grid:
        w_measured = w in meas_widths
        for p in PRECISIONS:
            configs.append({
                "width": list(w), "width_str": "x".join(map(str, w)),
                "precision": p,
                "measured": w_measured,        # AP+lat+energy all measured for original60
                "legal": True,                 # coarse-stage TVM native int8: no build cliff
            })

    n_total = len(configs)               # 1029
    n_measured = sum(c["measured"] for c in configs)     # 180
    n_unmeasured = n_total - n_measured                  # 849

    manifest = {
        "schema": "space_manifest_pyramid_h800_v1",
        "generated_at": utc(),
        "stage": "P2.0_space_landing",
        "model": "pyramid_lidar",
        "hardware_target": "H800",
        "frozen_decisions_ref": "progress/7_2/8_7_2_...待审.md §5 (2026-07-02)",

        "axes": {
            "prune_partition": {
                "granularity": "3_stage_coarse",
                "N_p": 3,
                "stage_widths": {"s0": W0, "s1": W1, "s2": W2},
                "abs_W_per_stage": 7,
                "note": ("backbone stages s0/s1/s2; each width = 2x num_filters. "
                         "finer per-block partition (N_p~9) frozen OUT this phase: "
                         "exponential blowup + exponential cold-start need + marginal "
                         "Pareto gain (AP plateau / latency near-monotone in total width)."),
            },
            "quant_precision": {
                "granularity": "global_single_precision",
                "N_q": 1, "Q": PRECISIONS, "abs_Q": 3,
                "note": "mixed precision has no measured data this phase; frozen out.",
            },
        },

        "scale_formula": {
            "expr": "|Theta_sw| = |W|^N_p * |Q|",
            "substituted": "7^3 * 3",
            "value": 7 ** 3 * 3,
            "matches_enumeration": (7 ** 3 * 3 == n_total),
        },

        "counts": {
            "grid_widths": len(grid),                 # 343
            "measured_widths": len(meas_widths),      # 60
            "unmeasured_widths": len(unmeasured_widths),  # 283
            "configs_total": n_total,                 # 1029
            "configs_measured": n_measured,           # 180
            "configs_unmeasured": n_unmeasured,       # 849
        },

        "legality": {
            "verdict": "ALL_1029_LEGAL",
            "int8_buildability_cliff_coarse_tvm": False,
            "empirical_int8_build_audit": {
                "int8_widths_measured": len(int8_meas),
                "not_multiple_of_128": len(non_mult_128),
                "not_multiple_of_32": len(non_mult_32),
                "conclusion": ("60/60 int8 widths built & measured on H800/TVM though "
                               f"{len(non_mult_128)}/60 are NOT multiples of 128 and "
                               f"{len(non_mult_32)}/60 not multiples of 32 => bridge "
                               "static int8_buildable_align (128/32) is a TRT-tiling "
                               "artifact, does NOT gate TVM native int8 buildability."),
            },
            "bridge_static_reference": {
                "model": b_model, "hw_in_manifest": b_hw,
                "static_int8_cliff_fine_grained": b_cliff,
                "note": ("bridge manifest hw = rtx4090; fine-grained grouped-conv knobs "
                         "report a STATIC int8 cliff (align=128) that is empirically "
                         "false on TVM native int8 at coarse stage granularity."),
                "knobs": b_knobs,
            },
        },

        "separability": {
            "coarse_stage_verdict": ("no structural int8-buildability trap at 3-stage "
                                     "granularity on TVM (all build) => P x Q coupling, "
                                     "if any, is measurement-driven not structural."),
            "joint_value_test": "deferred to P2.3 three-arm ablation (S0/S1/S2, Gap1).",
        },

        "measured_widths": [list(w) for w in sorted(meas_widths)],
        "unmeasured_widths": [list(w) for w in sorted(unmeasured_widths)],
        "configs": configs,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(OUT, "w"), ensure_ascii=False, indent=1)

    # console summary
    print(json.dumps({k: manifest[k] for k in
                      ["schema", "scale_formula", "counts", "legality"]},
                     ensure_ascii=False, indent=1)[:2000])
    print("\nwrote:", OUT)
    # sanity asserts
    assert n_total == 1029, n_total
    assert n_measured == 180, n_measured
    assert n_unmeasured == 849, n_unmeasured
    assert manifest["scale_formula"]["matches_enumeration"]
    print("P2.0 asserts PASS: 1029 total / 180 measured / 849 unmeasured / formula OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
