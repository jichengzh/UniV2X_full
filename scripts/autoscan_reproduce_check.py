"""Phase A0 验证脚本: auto-scan 复现 4 个手写 adapter 的 manifest.

用法:
  PYTHONPATH=/home/jichengzhi/V2X \
  /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \
      scripts/autoscan_reproduce_check.py [--models all] [--device cpu] [--save-yaml]

输出: results/autoscan_reproduce_check.json
  {model: {match: bool, diffs: [...], new_manifest: {...}}}

比对字段 (Phase A0 金标准):
  n_b1_groups_structural, n_b1_search_knobs, n_b2_quant_units
  view_b1_search_groups: 每组 {search_group_id, n_b1_groups, widths, max_rate,
                                int8_buildable_align, grouped_conv}
  params_total, param_dist (桶分布 ≈ 2%)
  dryrun_prune05.status
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path("/home/jichengzhi/V2X")
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import yaml

from framework.stage1.auto_trace import AUTO_REGISTRY, get_auto_adapter
from framework.stage1.hardware_scan import HwCapability
from framework.stage1 import graph_scan

_A0_MODELS = ["codriving", "pyramid_lidar", "v2xvit"]  # pyramid_camera 可选 (仅 OPV2V ckpt)
_A1_MODELS = ["fcooper", "attfuse"]
_A2_MODELS = ["where2comm", "v2vnet", "disconet"]
_PARTITION_DIR = _REPO / "framework/partitions"
_RESULTS_DIR = _REPO / "results"
_HW_YAML = _REPO / "configs/hardware/rtx4090.yaml"


def load_ref_yaml(model: str) -> dict | None:
    p = _PARTITION_DIR / f"{model}_partition.yaml"
    if not p.exists():
        return None
    with open(p) as f:
        return yaml.safe_load(f)


def compare_manifests(ref: dict, new: dict, model: str) -> dict:
    """比对两个 manifest 的关键字段, 返回 diff 报告。"""
    diffs = []
    ok = True

    def chk(field_path: str, ref_val, new_val, tol=None):
        nonlocal ok
        if tol is not None:
            if abs(float(ref_val or 0) - float(new_val or 0)) > tol:
                diffs.append({"field": field_path, "ref": ref_val, "new": new_val})
                ok = False
        else:
            if ref_val != new_val:
                diffs.append({"field": field_path, "ref": ref_val, "new": new_val})
                ok = False

    # search_space_summary
    ss_ref = ref.get("search_space_summary", {})
    ss_new = new.get("search_space_summary", {})
    for key in ("n_b1_groups_structural", "n_b1_search_knobs", "n_b2_quant_units"):
        chk(f"search_space_summary.{key}", ss_ref.get(key), ss_new.get(key))

    # params
    chk("stats.params_total", ref.get("stats", {}).get("params_total"),
        new.get("stats", {}).get("params_total"))

    # param_dist 各桶 (容差 2%)
    ref_dist = ref.get("stats", {}).get("param_dist", {})
    new_dist = new.get("stats", {}).get("param_dist", {})
    all_buckets = set(ref_dist) | set(new_dist)
    for b in sorted(all_buckets):
        chk(f"stats.param_dist.{b}", ref_dist.get(b, 0.0), new_dist.get(b, 0.0), tol=0.02)

    # dryrun status
    ref_dr = ref.get("checks", {}).get("dryrun_prune05", {}).get("status")
    new_dr = new.get("checks", {}).get("dryrun_prune05", {}).get("status")
    chk("checks.dryrun_prune05.status", ref_dr, new_dr)

    # view_b1_search_groups 每组比对
    ref_sg = {g["search_group_id"]: g for g in ref.get("view_b1_search_groups", [])}
    new_sg = {g["search_group_id"]: g for g in new.get("view_b1_search_groups", [])}
    missing_grps = set(ref_sg) - set(new_sg)
    extra_grps = set(new_sg) - set(ref_sg)
    if missing_grps:
        diffs.append({"field": "view_b1_search_groups.missing", "ref": sorted(missing_grps), "new": None})
        ok = False
    if extra_grps:
        diffs.append({"field": "view_b1_search_groups.extra", "ref": None, "new": sorted(extra_grps)})
        ok = False
    for gid in sorted(set(ref_sg) & set(new_sg)):
        rg, ng = ref_sg[gid], new_sg[gid]
        for f in ("n_b1_groups", "widths", "max_rate", "int8_buildable_align", "grouped_conv"):
            chk(f"view_b1_search_groups[{gid}].{f}", rg.get(f), ng.get(f))

    return {"model": model, "match": ok, "n_diffs": len(diffs), "diffs": diffs}


def run_scan_one(model: str, hw: HwCapability, device: str) -> dict:
    adapter = get_auto_adapter(model)
    print(f"\n{'='*60}")
    print(f"[auto-scan] {model}")
    manifest = graph_scan.scan(adapter, hw, device=device, profile_latency_mode="off")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="a0",
                    help="a0 | a1 | a2 | all | comma-sep model names")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hw", default=str(_HW_YAML))
    ap.add_argument("--save-yaml", action="store_true",
                    help="将 auto-scan manifest 写到 results/autoscan_<model>_partition.yaml")
    args = ap.parse_args()

    if args.models == "a0":
        models = _A0_MODELS
    elif args.models == "a1":
        models = _A1_MODELS
    elif args.models == "a2":
        models = _A2_MODELS
    elif args.models == "all":
        models = _A0_MODELS + _A1_MODELS + _A2_MODELS
    else:
        models = [m.strip() for m in args.models.split(",")]

    hw = HwCapability.from_yaml(args.hw)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for model in models:
        if model not in AUTO_REGISTRY:
            print(f"[SKIP] {model} not in AUTO_REGISTRY")
            continue
        try:
            manifest = run_scan_one(model, hw, args.device)
        except Exception as e:
            import traceback
            print(f"[FAIL] {model}: {e}")
            traceback.print_exc()
            results[model] = {"model": model, "match": False, "error": str(e)}
            continue

        if args.save_yaml:
            out = _RESULTS_DIR / f"autoscan_{model}_partition.yaml"
            with open(out, "w") as f:
                yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)
            print(f"  [saved] {out}")

        ref = load_ref_yaml(model)
        if ref is None:
            print(f"  [INFO] {model}: 无 ref partition yaml (Phase A1 新模型, 跳过比对)")
            results[model] = {"model": model, "match": None,
                              "note": "no_ref_yaml",
                              "scan_status": manifest.get("scan_status"),
                              "search_space_summary": manifest.get("search_space_summary"),
                              "stats": manifest.get("stats")}
        else:
            cmp = compare_manifests(ref, manifest, model)
            results[model] = cmp
            status = "✅ MATCH" if cmp["match"] else f"❌ {cmp['n_diffs']} diffs"
            print(f"  [compare] {model}: {status}")
            if cmp["diffs"]:
                for d in cmp["diffs"][:8]:
                    print(f"    DIFF {d['field']}: ref={d['ref']} vs new={d['new']}")

    out_json = _RESULTS_DIR / "autoscan_reproduce_check.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[done] → {out_json}")

    # 汇总
    n_match = sum(1 for r in results.values() if r.get("match") is True)
    n_fail = sum(1 for r in results.values() if r.get("match") is False and "error" not in r)
    n_new = sum(1 for r in results.values() if r.get("match") is None)
    print(f"  A0 复现: {n_match}/{n_match+n_fail} MATCH  |  A1 新模型: {n_new} 扫通")


if __name__ == "__main__":
    main()
