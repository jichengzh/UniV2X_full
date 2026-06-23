"""Plan v5 Phase 0 — Pre-flight + pre-register thresholds.

Verifies g8/g32 ckpt, calib data, TRT version + sparsity support,
sparsity tool (apex.contrib.sparsity preferred, torch.nn.utils.prune fallback),
DAIR val n=1789 split. Writes plan5_state.json with pre-registered thresholds.

Run:
    python scripts/phase2/plan5_phase0_preflight.py
Output:
    paper_learning/2. AAAI最终故事/data/plan5_state.json
    paper_learning/2. AAAI最终故事/data/plan5_phase0_preflight.json
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
DATA_DIR = REPO / "paper_learning" / "2. AAAI最终故事" / "data"
STATE_FILE = DATA_DIR / "plan5_state.json"
PREFLIGHT_FILE = DATA_DIR / "plan5_phase0_preflight.json"

G8_CKPT = Path(
    "/home/jichengzhi/heal_research/HEAL/opencood/logs/"
    "Pyramid_DAIR_m1_base_g8_2026_05_21_22_07_45/net_epoch_bestval_at19.pth"
)
G32_CKPT_DIR = Path(
    "/home/jichengzhi/heal_research/checkpoints/stage1/"
    "Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
)
CALIB_DIR = REPO / "calibration"
DAIR_VAL_JSON = Path(
    "/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/"
    "cooperative-vehicle-infrastructure/val.json"
)


def check_ckpt(path: Path, label: str) -> dict:
    res = {"label": label, "path": str(path), "exists": path.exists()}
    if not res["exists"]:
        res["loadable"] = False
        res["size_mb"] = 0.0
        return res
    res["size_mb"] = round(path.stat().st_size / 1e6, 2)
    try:
        import torch
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict):
            n_keys = len(sd.get("model_state_dict", sd))
            res["loadable"] = n_keys > 0
            res["n_state_keys"] = n_keys
        else:
            res["loadable"] = True
    except Exception as e:
        res["loadable"] = False
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def check_g32_ckpt(ckpt_dir: Path) -> dict:
    candidates = sorted(ckpt_dir.glob("net_epoch*.pth"))
    target = None
    for p in candidates:
        if "bestval_at23" in p.name or p.name == "net_epoch23.pth":
            target = p
            break
    if target is None and candidates:
        target = candidates[-1]
    if target is None:
        return {"label": "g32_baseline", "exists": False, "loadable": False}
    return check_ckpt(target, "g32_baseline_epoch23")


def check_calib_data(calib_dir: Path) -> dict:
    if not calib_dir.exists():
        return {"label": "calibration_data", "exists": False, "files": []}
    files = [p.name for p in calib_dir.iterdir() if p.is_file()]
    has_minmax = any("dair" in f.lower() and "minmax" in f.lower() for f in files)
    has_entropy = any("entropy" in f.lower() for f in files)
    return {
        "label": "calibration_data",
        "exists": True,
        "files": files,
        "has_dair_minmax_cache": has_minmax,
        "has_entropy_cache": has_entropy,
    }


def check_trt() -> dict:
    res = {"label": "tensorrt"}
    try:
        import tensorrt as trt
        res["version"] = trt.__version__
        res["installed"] = True
        flags = [f for f in dir(trt.BuilderFlag) if "SPARSE" in f.upper()]
        res["sparse_weights_flag"] = "SPARSE_WEIGHTS" in flags
        res["available_sparse_flags"] = flags
        int8_flag = "INT8" in [f for f in dir(trt.BuilderFlag) if f.isupper()]
        res["int8_flag"] = int8_flag
    except Exception as e:
        res["installed"] = False
        res["error"] = f"{type(e).__name__}: {e}"
    trtexec = None
    for cand in ("/usr/src/tensorrt/bin/trtexec", "/usr/local/bin/trtexec"):
        if Path(cand).exists():
            trtexec = cand
            break
    if trtexec is None:
        try:
            out = subprocess.check_output(["which", "trtexec"], stderr=subprocess.DEVNULL).decode().strip()
            trtexec = out or None
        except Exception:
            trtexec = None
    res["trtexec_binary"] = trtexec or "NOT_FOUND"
    res["trtexec_available"] = trtexec is not None
    return res


def check_sparsity_tools() -> dict:
    res = {"label": "sparsity_tools"}
    try:
        from apex.contrib.sparsity import ASP  # noqa: F401
        res["apex_asp_available"] = True
    except Exception as e:
        res["apex_asp_available"] = False
        res["apex_asp_error"] = f"{type(e).__name__}: {e}"
    try:
        import torch.nn.utils.prune as prune  # noqa: F401
        res["torch_prune_available"] = True
    except Exception:
        res["torch_prune_available"] = False
    res["fallback_strategy"] = (
        "apex_asp" if res.get("apex_asp_available")
        else ("manual_24_via_torch_prune" if res.get("torch_prune_available") else "NONE")
    )
    return res


def check_dair_val(val_json: Path) -> dict:
    res = {"label": "dair_val", "path": str(val_json), "exists": val_json.exists()}
    if not res["exists"]:
        return res
    try:
        with open(val_json) as f:
            data = json.load(f)
        res["n_samples"] = len(data) if hasattr(data, "__len__") else None
        res["expected_n"] = 1789
        res["matches_expected"] = res["n_samples"] == 1789
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        res["matches_expected"] = False
    return res


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[Plan v5 Phase 0] Pre-flight checks ==========================")
    checks = {
        "g8_baseline_ckpt": check_ckpt(G8_CKPT, "g8_baseline_epoch19"),
        "g32_baseline_ckpt": check_g32_ckpt(G32_CKPT_DIR),
        "calibration_data": check_calib_data(CALIB_DIR),
        "tensorrt": check_trt(),
        "sparsity_tools": check_sparsity_tools(),
        "dair_val": check_dair_val(DAIR_VAL_JSON),
    }

    pass_criteria = {
        "g8_ckpt_ok": checks["g8_baseline_ckpt"].get("loadable", False),
        "g32_ckpt_ok": checks["g32_baseline_ckpt"].get("loadable", False),
        "calib_ok": (
            checks["calibration_data"].get("has_dair_minmax_cache", False)
            or len(checks["calibration_data"].get("files", [])) > 0
        ),
        "trt_ok": (
            checks["tensorrt"].get("installed", False)
            and checks["tensorrt"].get("sparse_weights_flag", False)
            and checks["tensorrt"].get("int8_flag", False)
        ),
        "sparsity_tool_ok": checks["sparsity_tools"].get("fallback_strategy") != "NONE",
        "dair_val_ok": checks["dair_val"].get("matches_expected", False),
    }

    all_pass = all(pass_criteria.values())

    print("--- check results ---")
    for k, v in checks.items():
        print(f"  {k}: {json.dumps(v, ensure_ascii=False)[:200]}")
    print("--- pass criteria ---")
    for k, v in pass_criteria.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"--- all_pass: {all_pass} ---")

    soft_warnings = []
    if not checks["sparsity_tools"].get("apex_asp_available"):
        soft_warnings.append(
            "apex.contrib.sparsity NOT available; falling back to torch.nn.utils.prune manual 2:4 mask. "
            "Functionally equivalent but ASP-style auto-finetune loop must be hand-rolled in Phase B."
        )
    if not checks["tensorrt"].get("trtexec_available"):
        soft_warnings.append(
            "trtexec binary NOT found; all engine build + lat bench must use TRT Python API "
            "(IBuilderConfig + IExecutionContext.execute_v2 + cuda events). "
            "Plan §5/§7 commands referencing trtexec need to be wrapped in Python helper."
        )

    payload = {
        "phase": "phase_0",
        "ran_at_iso_date": "2026-05-29",
        "checks": checks,
        "pass_criteria": pass_criteria,
        "all_pass": all_pass,
        "soft_warnings": soft_warnings,
    }
    PREFLIGHT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[Plan v5 Phase 0] wrote {PREFLIGHT_FILE}")

    state = {
        "plan_version": "v5",
        "created_iso_date": "2026-05-29",
        "current_phase": "phase_A" if all_pass else "phase_0_blocked",
        "last_completed_phase": "phase_0" if all_pass else None,
        "pre_registered_thresholds": {
            "G_A_smooth_r2_min": 0.85,
            "G_B_sparsity_reduction_min": 0.30,
            "G_B_sparsity_pass_planes_min": 3,
            "G_C_ap_gap_max": 0.02,
            "G_D_dominate_lat_factor": 0.70,
            "G_D_dominate_ap_factor": 0.92,
        },
        "hypotheses": {
            "H1": "g8 lat vs plane is smooth (poly3 R^2 >= 0.85)",
            "H2": "2:4 sparsity gives independent >=30% extra lat reduction on >=3/5 plane",
            "H3": "TRT real INT8 AP vs fake quant proxy median gap <= 0.02",
            "D1_derived": "g8 anchor dominates g32 best (lat<0.7x AND ap>=0.92x)",
        },
        "design_constraints_inherited": [
            "C1: FT=8 locked across all anchors (g8 epoch 19->27, g32 epoch 23->31)",
            "C5: pruning anchor MUST include finetune (no mask-only zero-shot AP)",
            "C6: INT8 AP must come from TRT engine forward (Phase C closes this loop)",
            "C7: 2:4 sparsity must be TRT --sparsity=enable real build (no simulation)",
        ],
        "key_paths": {
            "g8_baseline_ckpt": str(G8_CKPT),
            "g32_baseline_ckpt_dir": str(G32_CKPT_DIR),
            "calib_dir": str(CALIB_DIR),
            "dair_val_json": str(DAIR_VAL_JSON),
        },
        "phase_status": {
            "phase_0": "completed" if all_pass else "blocked",
            "phase_A": "pending",
            "phase_B": "pending",
            "phase_C": "pending",
            "phase_D": "pending",
        },
        "soft_warnings": soft_warnings,
        "preflight_summary": pass_criteria,
    }
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    print(f"[Plan v5 Phase 0] wrote {STATE_FILE}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
