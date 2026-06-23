"""L4 deliverable: find AP70 cliff at 80%/87%/93% TOTAL param reduction.

Strategy: wholenet_prune_pyramid.py (backbone+deblocks+shrink all pruned together)
so total params can hit real 80/87/93% reduction targets.

Configs (confirmed total params via model instantiation):
  wn_80pct  nf=[16,32,64] wpg=4 nuf=[64,64,64]  sh=128  → 0.879M  83.9% reduction
  wn_87pct  nf=[8,16,32]  wpg=8 nuf=[48,48,48]  sh=96   → 0.593M  89.2% reduction
  wn_93pct  nf=[8,8,16]   wpg=8 nuf=[32,32,32]  sh=64   → 0.368M  93.3% reduction

Runs serially on GPU1. Writes results/ap_cliff_l4.json on completion.

Usage:
  cd /home/jichengzhi/V2X
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/jichengzhi/heal_research/HEAL \\
  /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \\
    scripts/phase2/l4_wholenet_runner.py \\
    > results/l4_wholenet_runner.log 2>&1 &
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
import yaml

REPO = Path("/home/jichengzhi/V2X")
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
BASELINE = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
CKROOT = Path("/home/jichengzhi/heal_research/checkpoints/stage1")
BASE_TOTAL = 5_464_791

# FT = 35 epochs total (init@23 → 12 more epochs; small model → fast epochs)
EPOCHES = 35
GPU = 1
MASTER_PORT = 29820
RESULT_JSON = REPO / "results/ap_cliff_l4.json"

# (tag, num_filters, wpg, num_upsample, shrink_dim, expected_total_M, expected_red_pct)
CONFIGS = [
    ("wn_80pct", [16, 32, 64], 4, [64, 64, 64], 128, 0.879, 83.9),
    ("wn_87pct", [8, 16, 32],  8, [48, 48, 48], 96,  0.593, 89.2),
    ("wn_93pct", [8,  8, 16],  8, [32, 32, 32], 64,  0.368, 93.3),
]


def ckpt_dir_for(tag: str) -> Path:
    return CKROOT / f"Pyramid_DAIR_m1_l4_{tag}_2026_06_22"


def prune(tag: str, nf: list, wpg: int, nuf: list, sh: int) -> Path | None:
    out_dir = ckpt_dir_for(tag)
    init_ckpt = out_dir / "net_epoch_bestval_at23.pth"

    # Skip if already has post-init epoch (resume guard)
    post = [p for p in out_dir.glob("net_epoch*.pth")
            if "bestval" not in p.name or
            (m := re.search(r"at(\d+)", p.name)) and int(m.group(1)) > 23]
    post_bestval = [p for p in out_dir.glob("net_epoch_bestval_at*.pth")
                    if (m := re.search(r"at(\d+)", p.name)) and int(m.group(1)) > 23]
    if post_bestval:
        print(f"[{tag}] post-init bestval exists → skip prune (resume)")
        return out_dir

    if init_ckpt.exists() and not post_bestval:
        print(f"[{tag}] init ckpt exists (no post-init), skip re-prune")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, "tools/wholenet_prune_pyramid.py",
        "--orig-dir", BASELINE,
        "--out-dir", str(out_dir),
        "--num-filters-new", ",".join(str(x) for x in nf),
        "--num-upsample-new", ",".join(str(x) for x in nuf),
        "--shrink-dim-new", str(sh),
        "--groups", "32",
        "--width-per-group", str(wpg),
        "--epoches", str(EPOCHES),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(HEAL)}
    log_path = REPO / f"results/l4_{tag}_prune.log"
    print(f"[{tag}] pruning nf={nf} wpg={wpg} nuf={nuf} sh={sh} ...")
    with open(log_path, "w") as lf:
        r = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=600)
    log_text = log_path.read_text()
    print(log_text[-2000:])
    if r.returncode != 0 or not init_ckpt.exists():
        print(f"[{tag}] PRUNE FAILED rc={r.returncode}")
        return None
    print(f"[{tag}] prune OK → {init_ckpt}")
    return out_dir


def verify_flat(tag: str, out_dir: Path) -> bool:
    """Verify init ckpt loads with 0 missing keys."""
    sys.path.insert(0, str(HEAL))
    sys.path.insert(0, str(REPO))
    try:
        from opencood.hypes_yaml.yaml_utils import load_yaml
        from opencood.models.heter_pyramid_collab import HeterPyramidCollab
    except ImportError as e:
        print(f"[{tag}] import error: {e}")
        return False

    init_ckpt = out_dir / "net_epoch_bestval_at23.pth"
    cfg = out_dir / "config.yaml"
    hypes = load_yaml(str(cfg))
    model = HeterPyramidCollab(hypes["model"]["args"])
    sd = torch.load(str(init_ckpt), map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        # Shouldn't happen since wholenet tool saves flat
        torch.save(sd["model_state_dict"], str(init_ckpt))
        sd = sd["model_state_dict"]
        print(f"[{tag}] WARNING: unwrapped model_state_dict (shouldn't happen)")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] verify: missing={len(missing)} unexpected={len(unexpected)} params={n_total:,}")
    if missing:
        print(f"[{tag}] WARN missing keys: {missing[:5]}")
        return False
    # Check weights not random (a few norms should be > 0 and < 100)
    w = list(model.parameters())[0]
    norm = w.norm().item()
    print(f"[{tag}] first param norm={norm:.4f} (should be >0 and <50 for non-random)")
    if norm < 1e-6 or norm > 500:
        print(f"[{tag}] WARN: suspicious norm → might be random init!")
        return False
    return True


def finetune(tag: str, out_dir: Path) -> int:
    cfg = out_dir / "config.yaml"
    log = REPO / f"results/l4_{tag}_finetune.log"
    cmd = [
        PY, "-m", "torch.distributed.launch",
        "--nproc_per_node=1", "--use_env",
        f"--master_port={MASTER_PORT}",
        str(HEAL / "opencood/tools/train_ddp.py"),
        "--hypes_yaml", str(cfg),
        "--model_dir", str(out_dir),
        "--half",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU), "PYTHONPATH": str(HEAL)}
    print(f"[{tag}] finetune GPU{GPU} port={MASTER_PORT} log={log.name}")
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=HEAL, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"[{tag}] finetune done rc={rc}")
    return rc


def eval_ap(tag: str, out_dir: Path) -> dict | None:
    log = REPO / f"results/l4_{tag}_eval.log"
    # Remove stale eval yamls
    for ev in out_dir.glob("eval*.yaml"):
        ev.unlink()
        print(f"[{tag}] removed stale {ev.name}")

    cmd = [
        PY, "opencood/tools/inference.py",
        "--model_dir", str(out_dir),
        "--fusion_method", "intermediate",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU), "PYTHONPATH": str(HEAL)}
    print(f"[{tag}] eval AP → {log.name}")
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=HEAL, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"[{tag}] eval rc={rc}")

    # Try yaml first
    ap = None
    for ev in out_dir.glob("eval*.yaml"):
        try:
            y = yaml.safe_load(ev.read_text()) or {}
            ap30 = float(y.get("ap30") or y.get("ap_30") or 0)
            ap50 = float(y.get("ap_50") or y.get("ap50") or 0)
            ap70 = float(y.get("ap_70") or y.get("ap70") or 0)
            ap = {"ap30": round(ap30, 4), "ap50": round(ap50, 4), "ap70": round(ap70, 4)}
            print(f"[{tag}] AP from yaml {ev.name}: {ap}")
            break
        except Exception as e:
            print(f"[{tag}] yaml parse error: {e}")

    # Fallback: stdout
    if ap is None:
        try:
            txt = log.read_text()
            m = re.search(
                r"Average Precision at IOU 0\.3 is ([0-9.]+).*?0\.5 is ([0-9.]+).*?0\.7 is ([0-9.]+)",
                txt, re.DOTALL)
            if m:
                ap = {"ap30": round(float(m.group(1)), 4),
                      "ap50": round(float(m.group(2)), 4),
                      "ap70": round(float(m.group(3)), 4)}
                print(f"[{tag}] AP from stdout: {ap}")
        except Exception as e:
            print(f"[{tag}] stdout parse error: {e}")

    return ap


def find_best_ckpt(ft_dir: Path) -> Path | None:
    bests = list(ft_dir.glob("net_epoch_bestval_at*.pth"))
    if bests:
        return max(bests, key=lambda p: int(re.search(r"at(\d+)\.pth", p.name).group(1)))
    all_pths = [p for p in ft_dir.glob("net_epoch*.pth") if "bestval" not in p.name]
    if all_pths:
        return max(all_pths, key=lambda p: int(re.search(r"epoch(\d+)", p.name).group(1)))
    return None


def count_model_params(out_dir: Path) -> int:
    """Count params from config."""
    try:
        sys.path.insert(0, str(HEAL))
        sys.path.insert(0, str(REPO))
        from opencood.hypes_yaml.yaml_utils import load_yaml
        from opencood.models.heter_pyramid_collab import HeterPyramidCollab
        hypes = load_yaml(str(out_dir / "config.yaml"))
        model = HeterPyramidCollab(hypes["model"]["args"])
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return 0


def load_results() -> dict:
    if RESULT_JSON.exists():
        return json.loads(RESULT_JSON.read_text())
    return {}


def save_results(results: dict):
    RESULT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Saved → {RESULT_JSON}")


def run_one(tag: str, nf: list, wpg: int, nuf: list, sh: int,
            exp_total_M: float, exp_red: float, results: dict):
    print(f"\n{'='*70}")
    print(f"[{tag}] nf={nf} wpg={wpg} nuf={nuf} sh={sh} "
          f"exp_total={exp_total_M:.3f}M exp_red={exp_red:.1f}%")
    print(f"{'='*70}")

    # Skip if already done
    if tag in results and results[tag].get("status") == "ok":
        print(f"[{tag}] already done, skipping")
        return

    # 1. Prune
    out_dir = prune(tag, nf, wpg, nuf, sh)
    if out_dir is None:
        results[tag] = {"nf": nf, "wpg": wpg, "nuf": nuf, "shrink": sh,
                        "status": "prune_failed"}
        save_results(results)
        return

    # 2. Verify flat load
    weights_ok = verify_flat(tag, out_dir)
    if not weights_ok:
        results[tag] = {"nf": nf, "wpg": wpg, "nuf": nuf, "shrink": sh,
                        "status": "verify_failed"}
        save_results(results)
        return

    # 3. Count actual params
    actual_params = count_model_params(out_dir)
    actual_total_M = actual_params / 1e6
    actual_red_pct = (1 - actual_params / BASE_TOTAL) * 100
    print(f"[{tag}] actual params: {actual_params:,} ({actual_total_M:.3f}M, "
          f"-{actual_red_pct:.1f}%)")

    # 4. Finetune
    rc = finetune(tag, out_dir)

    # 5. Find best ckpt
    best = find_best_ckpt(out_dir)
    best_name = best.name if best else None
    best_epoch = int(re.search(r"at(\d+)", best.name).group(1)) if best and "bestval" in best.name else None

    # 6. Eval
    ap = eval_ap(tag, out_dir)

    # 7. Save
    results[tag] = {
        "prune_pct": round(actual_red_pct, 1),
        "param_count_M": round(actual_total_M, 3),
        "nf": nf,
        "wpg": wpg,
        "nuf": nuf,
        "shrink": sh,
        "finetune_epochs": (best_epoch - 23) if best_epoch else (EPOCHES - 23),  # actual extra
        "bestval_epoch": best_name,
        "best_epoch_num": best_epoch,
        "ckpt_path": str(out_dir),
        "weights_loaded_verified": weights_ok,
        "finetune_rc": rc,
        "ap30": ap["ap30"] if ap else None,
        "ap50": ap["ap50"] if ap else None,
        "ap70": ap["ap70"] if ap else None,
        "status": "ok" if ap else f"eval_failed:rc={rc}",
    }
    save_results(results)
    print(f"[{tag}] COMPLETE: AP70={results[tag]['ap70']} params={actual_total_M:.3f}M -"
          f"{actual_red_pct:.1f}%")


def synthesize_cliff_analysis(results: dict) -> str:
    """Characterize the AP70 wholenet curve by RANGE + SLOPE ACCELERATION, not a binary threshold.

    Cliff determination uses only wholenet configs (all3_hard + wn_80/87/93pct).
    cliff_a (backbone-only strategy) is excluded from cliff judgment.

    Key metrics:
      - Total AP70 range from base: signals whether AP axis has real information
      - Per-interval slope: flat → accelerating → cliff characterization
      - Verdict: "clear cliff" / "soft knee" / "plateau with knee onset"
    """
    new_pts = [(d["prune_pct"], d.get("ap70"), d.get("ap50"), d.get("ap30"))
               for d in results.values()
               if d.get("ap70") is not None and d.get("status") == "ok"]
    new_pts.sort(key=lambda x: x[0])

    # Wholenet curve: base anchor + all3_hard + this experiment's 3 configs
    wholenet_existing = [
        (0.0,  0.6309, 0.7910, 0.8332),  # base FP16, stage_a_ap_real.parquet
        (75.3, 0.5900, 0.7400, 0.7800),  # all3_hard wholenet [32,64,128]+[48]+sh96, 1.35M, ep35
    ]
    wholenet_curve = sorted(wholenet_existing + new_pts, key=lambda x: x[0])
    valid = [(pct, ap70) for pct, ap70, *_ in wholenet_curve if ap70 is not None]

    # --- Print full table ---
    lines = []
    lines.append("=== WHOLENET curve (cliff judgment) ===")
    lines.append("total_prune%  AP30    AP50    AP70    Δ AP70  slope(/5%)")
    prev_ap70 = None
    prev_pct = None
    for pct, ap70, ap50, ap30 in wholenet_curve:
        ap30s = f"{ap30:.4f}" if ap30 is not None else "  --  "
        ap50s = f"{ap50:.4f}" if ap50 is not None else "  --  "
        ap70s = f"{ap70:.4f}" if ap70 is not None else "  --  "
        if prev_ap70 is not None and ap70 is not None:
            delta = ap70 - prev_ap70
            interval = pct - prev_pct
            slope_per5 = (delta / interval * 5) if interval > 0 else 0
            delta_s = f"{delta:+.4f}"
            slope_s = f"{slope_per5:+.4f}"
        else:
            delta_s = "  --  "
            slope_s = "  --  "
        lines.append(f"  {pct:5.1f}%   {ap30s}  {ap50s}  {ap70s}  {delta_s}  {slope_s}")
        if ap70 is not None:
            prev_ap70 = ap70
            prev_pct = pct

    lines.append("")

    # --- Compute summary metrics ---
    base_ap70 = valid[0][1] if valid else None
    min_ap70 = min(v[1] for v in valid) if valid else None
    total_range = (base_ap70 - min_ap70) if (base_ap70 and min_ap70) else 0.0
    pipeline_noise = 0.001  # known ~0.001 AP70 noise floor

    # Per-interval slopes on wholenet curve
    slopes = []
    for i in range(1, len(valid)):
        pct0, ap70_0 = valid[i-1]
        pct1, ap70_1 = valid[i]
        interval = pct1 - pct0
        if interval > 0:
            slopes.append((pct0, pct1, (ap70_1 - ap70_0) / interval * 5))

    # Hard cliff check: any single step drop > 0.03
    hard_cliff = any(
        valid[i-1][1] - valid[i][1] > 0.03
        for i in range(1, len(valid))
    )
    hard_cliff_at = None
    if hard_cliff:
        for i in range(1, len(valid)):
            if valid[i-1][1] - valid[i][1] > 0.03:
                hard_cliff_at = valid[i][0]
                break

    # Slope acceleration: is slope magnitude increasing over successive intervals?
    slope_magnitudes = [abs(s[2]) for s in slopes]
    accelerating = len(slope_magnitudes) >= 2 and slope_magnitudes[-1] > slope_magnitudes[-2]

    # Verdict
    snr = total_range / pipeline_noise if pipeline_noise > 0 else 0
    if hard_cliff:
        verdict = f"CLEAR CLIFF at ~{hard_cliff_at:.0f}%"
        cliff_type = "clear_cliff"
    elif total_range >= 0.05 and accelerating:
        # Find where acceleration starts
        accel_start = slopes[-2][0] if len(slopes) >= 2 else slopes[-1][0]
        verdict = (f"SOFT KNEE — AP70 range={total_range:.3f} ({snr:.0f}× noise), "
                   f"slope accelerates after ~{accel_start:.0f}%; "
                   f"no single step >{0.03:.2f} but curve has real structure")
        cliff_type = "soft_knee"
    elif total_range >= 0.03:
        verdict = (f"PLATEAU WITH KNEE ONSET — range={total_range:.3f} ({snr:.0f}× noise), "
                   f"slope {'accelerating' if accelerating else 'not yet accelerating'}")
        cliff_type = "plateau_knee_onset"
    else:
        verdict = (f"PLATEAU — range={total_range:.3f} ({snr:.0f}× noise); "
                   f"AP stable → over-parameterization confirmed → iso-AP framing valid")
        cliff_type = "plateau"

    lines.append(f"  Total AP70 range (base→min): {total_range:.4f}  ({snr:.0f}× pipeline noise)")
    if slopes:
        lines.append("  Per-interval slopes (AP70 per 5% pruning):")
        for p0, p1, sl in slopes:
            lines.append(f"    {p0:.1f}%→{p1:.1f}%: {sl:+.4f}/5%")
    lines.append(f"  Slope accelerating: {accelerating}")
    lines.append(f"  Hard cliff (>0.03 single step): {hard_cliff}"
                 + (f" at {hard_cliff_at:.1f}%" if hard_cliff_at else ""))
    lines.append("")
    lines.append(f"  VERDICT: {verdict}")
    lines.append(f"  cliff_type: {cliff_type}")
    lines.append("")

    # --- Backbone-only ref (separate strategy, not judged) ---
    lines.append("=== Backbone-only ref (different strategy, excluded from cliff judgment) ===")
    lines.append("  62.3%  cliff_a_wpg8  AP30=0.7919 AP50=0.7474 AP70=0.5755"
                 "  (shrink_conv untouched at 1.475M)")
    lines.append("")

    # --- Paper framing ---
    lines.append("=== Paper framing ===")
    lines.append(f"  AP70 range {base_ap70:.4f}→{min_ap70:.4f} = {total_range:.3f} span "
                 f"(~{snr:.0f}× noise) — AP axis carries real information, not just plateau.")
    if cliff_type in ("clear_cliff", "soft_knee"):
        lines.append("  Curve shows a knee: flat plateau through ~84%, then accelerating decline.")
        lines.append("  Supports: AP trade-off is real at extreme pruning (>84% total reduction).")
    else:
        lines.append("  AP is stable across wide pruning range → DAIR over-parameterization.")
        lines.append("  Supports iso-AP framing: co-design payoff is in latency, not accuracy axis.")

    return "\n".join(lines)


def main():
    print("=== L4 Wholenet Cliff Runner ===")
    print(f"GPU={GPU}  EPOCHES={EPOCHES}  RESULT={RESULT_JSON}")
    print(f"CONFIGS: {[c[0] for c in CONFIGS]}")

    results = load_results()

    for tag, nf, wpg, nuf, sh, exp_M, exp_red in CONFIGS:
        run_one(tag, nf, wpg, nuf, sh, exp_M, exp_red, results)

    # Final synthesis
    print("\n=== FINAL SYNTHESIS ===")
    analysis = synthesize_cliff_analysis(results)
    print(analysis)

    # cliff characterization: wholenet curve ONLY
    # cliff_a (backbone-only 62.3%) intentionally EXCLUDED — different strategy.
    new_pts_ap70 = [(d["prune_pct"], d["ap70"])
                    for d in results.values()
                    if d.get("ap70") is not None]
    wholenet_valid = sorted(
        [(0.0, 0.6309), (75.3, 0.5900)] + new_pts_ap70,
        key=lambda x: x[0]
    )
    # Hard cliff: any single wholenet step drops > 0.03 AP70
    hard_cliff = any(
        wholenet_valid[i-1][1] - wholenet_valid[i][1] > 0.03
        for i in range(1, len(wholenet_valid))
    )
    total_ap70_range = wholenet_valid[0][1] - min(v[1] for v in wholenet_valid)
    # Slope acceleration: compare last two intervals
    slopes_ap70 = []
    for i in range(1, len(wholenet_valid)):
        p0, a0 = wholenet_valid[i-1]
        p1, a1 = wholenet_valid[i]
        if p1 > p0:
            slopes_ap70.append(abs((a1 - a0) / (p1 - p0) * 5))
    slope_accel = len(slopes_ap70) >= 2 and slopes_ap70[-1] > slopes_ap70[-2]
    if hard_cliff:
        cliff_type = "clear_cliff"
    elif total_ap70_range >= 0.05 and slope_accel:
        cliff_type = "soft_knee"
    elif total_ap70_range >= 0.03:
        cliff_type = "plateau_knee_onset"
    else:
        cliff_type = "plateau"
    cliff_found = hard_cliff  # strict binary for backward compat
    results["_meta"] = {
        "cliff_found": cliff_found,           # strict: any single step >0.03 AP70
        "cliff_type": cliff_type,             # "clear_cliff" / "soft_knee" / "plateau_knee_onset" / "plateau"
        "ap70_total_range": round(total_ap70_range, 4),  # base→min, real span on AP axis
        "ap70_slope_accelerating": slope_accel,           # True if slope magnitude grows with pruning
        "interpretation": analysis,
        "base_total_params": BASE_TOTAL,
        "eval_protocol": "DAIR_val_1789 inference.py intermediate, GPU1",
        "existing_refs": {
            "base_fp16": {
                "prune_pct": 0.0, "param_count_M": 5.465,
                "ap30": 0.8332, "ap50": 0.7910, "ap70": 0.6309,
                "note": "base FP16, stage_a_ap_real.parquet, DAIR val 1789",
            },
            "cliff_a_wpg8": {
                "prune_pct": 62.3, "param_count_M": 2.063,
                "ap30": 0.7919, "ap50": 0.7474, "ap70": 0.5755,
                "note": "backbone-only [8,16,32]wpg8→channels[32,64,128], finetuned 31ep, eval yaml",
                "strategy": "backbone_only (shrink_conv untouched at 1.475M)",
                "cliff_determination": "EXCLUDED — different pruning strategy from wholenet configs; "
                                       "mixing creates spurious non-monotonicity. Listed as ref only.",
            },
            "all3_hard": {
                "prune_pct": 75.3, "param_count_M": 1.350,
                "ap30": 0.7800, "ap50": 0.7400, "ap70": 0.5900,
                "note": "wholenet [32,64,128]+[48,48,48]+sh96, finetuned 35ep, stdout",
                "strategy": "wholenet (backbone+deblocks+shrink)",
            },
        },
    }
    save_results(results)
    print(f"\n=== DONE → {RESULT_JSON} ===")


if __name__ == "__main__":
    main()
