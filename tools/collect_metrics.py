"""
collect_metrics.py — 一键聚合实验指标到单 JSON

设计：不重新跑任何实验，仅扫描 work_dir + output_dir 下的 **已有**产物，
      提取全部指标到 `{experiment}_metrics.json`。

支持提取：
  1. 参数量 / 参数缩减率     — 从 train.log 的 [prune] 行
  2. 模型体积 (.pth/.onnx/.engine MB)  — 文件 stat
  3. 精度 (AMOTA, mAP, NDS, seg IoU)   — 从 eval_pkl_amota.py 输出的
                                          {experiment}_results_metrics.json
  4. PyTorch latency (模块级 + ego_forward) — 从 benchmark_latency.py 的 JSON
  5. TRT latency                       — 从 benchmark_trt_engine.py 的 JSON

用法:
  python tools/collect_metrics.py \\
      --name ft_p1_60_q2 \\
      --work-dir work_dirs/ft_p1_60_q2 \\
      --eval-json output/p1_ffn_60pct_q2_results_metrics.json \\
      --latency-json output/latency_ft_p1_60_q2.json \\
      --trt-json output/trt_bench_pruned_60_fp32.json \\
      --onnx onnx/univ2x_ego_bev_pruned_60_50.onnx \\
      --engine trt_engines/univ2x_ego_bev_pruned_60_50_fp32.trt \\
      --out output/summary_ft_p1_60_q2.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate experiment metrics into single JSON")
    p.add_argument("--name", required=True,
                   help="Experiment ID / label (e.g., ft_p1_60_q2)")
    p.add_argument("--work-dir", default=None,
                   help="Training work_dir containing train.log + epoch_*.pth")
    p.add_argument("--eval-json", default=None,
                   help="AMOTA metrics JSON from eval_pkl_amota.py")
    p.add_argument("--latency-json", default=None,
                   help="PyTorch latency JSON from benchmark_latency.py")
    p.add_argument("--trt-json", default=None,
                   help="TRT engine latency JSON from benchmark_trt_engine.py")
    p.add_argument("--onnx", default=None, help="ONNX file (for size)")
    p.add_argument("--engine", default=None, help="TRT engine file (for size)")
    p.add_argument("--seg-log", default=None,
                   help="Optional: raw eval log file (to fallback-extract seg IoU "
                        "if not in eval-json)")
    p.add_argument("--out", required=True, help="Output aggregated JSON")
    return p.parse_args()


def _extract_params(train_log: str):
    """从 train.log 里找 '[prune] 参数: X -> Y (-Z.ZZ%)' 行。"""
    if not os.path.exists(train_log):
        return None
    pat = re.compile(r"\[prune\] 参数: ([\d,]+) -> ([\d,]+) \(-([\d.]+)%\)")
    with open(train_log, "r", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                before = int(m.group(1).replace(",", ""))
                after = int(m.group(2).replace(",", ""))
                return {
                    "params_before": before,
                    "params_after": after,
                    "params_after_M": round(after / 1e6, 3),
                    "reduction_pct": float(m.group(3)),
                }
    return None


def _file_size_mb(path: str):
    if path and os.path.exists(path):
        return round(os.path.getsize(path) / 1e6, 3)
    return None


def _extract_seg_from_log(log_path: str):
    """Fallback: 从 eval 日志里 grep 出 drivable/lanes/crossing/contour IoU。"""
    if not os.path.exists(log_path):
        return {}
    pat = re.compile(r"'(drivable_iou|lanes_iou|divider_iou|crossing_iou|contour_iou)': "
                     r"([\d.]+)")
    out = {}
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            for m in pat.finditer(line):
                key, val = m.group(1), float(m.group(2))
                if key not in out:  # keep first occurrence
                    out[key] = round(val, 4)
    return out


def main() -> int:
    args = parse_args()

    summary = {
        "experiment": args.name,
        "accuracy": {},
        "params": {},
        "size_mb": {},
        "latency_pytorch_ms": {},
        "latency_trt_ms": {},
    }

    # --- 1. accuracy (AMOTA, mAP, NDS, MT, ML, IDS ...)
    if args.eval_json and os.path.exists(args.eval_json):
        with open(args.eval_json, "r") as f:
            em = json.load(f)
        # Pick commonly-reported keys
        core_keys = [
            "pts_bbox_NuScenes/amota", "pts_bbox_NuScenes/amotp",
            "pts_bbox_NuScenes/mAP", "pts_bbox_NuScenes/NDS",
            "pts_bbox_NuScenes/recall", "pts_bbox_NuScenes/motar",
            "pts_bbox_NuScenes/mota", "pts_bbox_NuScenes/motp",
            "pts_bbox_NuScenes/tp", "pts_bbox_NuScenes/fp",
            "pts_bbox_NuScenes/fn", "pts_bbox_NuScenes/ids",
            "pts_bbox_NuScenes/mt", "pts_bbox_NuScenes/ml",
            "drivable_iou", "lanes_iou", "divider_iou",
            "crossing_iou", "contour_iou",
        ]
        for k in core_keys:
            if k in em:
                v = em[k]
                if isinstance(v, float):
                    v = round(v, 4)
                summary["accuracy"][k.replace("pts_bbox_NuScenes/", "")] = v

    # Fallback seg IoU from raw log
    if args.seg_log:
        seg_fallback = _extract_seg_from_log(args.seg_log)
        for k, v in seg_fallback.items():
            summary["accuracy"].setdefault(k, v)

    # --- 2. params
    if args.work_dir:
        train_log = os.path.join(args.work_dir, "train.log")
        if os.path.exists(train_log):
            p = _extract_params(train_log)
            if p:
                summary["params"] = p

    # --- 3. file sizes
    if args.work_dir:
        for f in ("epoch_3.pth", "pruned_before_ft.pth", "latest.pth"):
            full = os.path.join(args.work_dir, f)
            if os.path.exists(full):
                summary["size_mb"][f] = _file_size_mb(full)
    if args.onnx:
        summary["size_mb"]["onnx"] = _file_size_mb(args.onnx)
    if args.engine:
        summary["size_mb"]["trt_engine"] = _file_size_mb(args.engine)

    # --- 4. PyTorch latency
    if args.latency_json and os.path.exists(args.latency_json):
        with open(args.latency_json, "r") as f:
            lat = json.load(f)
        summary["latency_pytorch_ms"] = {
            "e2e_multiagent": {
                "mean": lat.get("e2e_ms_mean"),
                "std": lat.get("e2e_ms_std"),
            },
        }
        mods = lat.get("modules", {})
        for name in ("ego_agent_forward", "backbone", "neck", "bev_encoder",
                     "track_head_decoder", "track_head", "seg_head",
                     "seg_transformer"):
            if name in mods:
                summary["latency_pytorch_ms"][name] = {
                    "mean": mods[name].get("mean"),
                    "p50": mods[name].get("p50"),
                    "p90": mods[name].get("p90"),
                }

    # --- 5. TRT latency
    if args.trt_json and os.path.exists(args.trt_json):
        with open(args.trt_json, "r") as f:
            tj = json.load(f)
        summary["latency_trt_ms"] = {
            "engine": tj.get("engine"),
            "engine_size_mb": tj.get("engine_size_mb"),
            **{k: v for k, v in tj.get("latency_ms", {}).items()},
        }

    # --- save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    # --- print compact summary
    print(f"\n===== {args.name} =====")
    acc = summary["accuracy"]
    print(f"  AMOTA        : {acc.get('amota', '-')}")
    print(f"  mAP / NDS    : {acc.get('mAP', '-')} / {acc.get('NDS', '-')}")
    print(f"  drivable IoU : {acc.get('drivable_iou', '-')}")
    print(f"  lanes IoU    : {acc.get('lanes_iou', '-')}")
    params = summary["params"]
    if params:
        print(f"  params       : {params.get('params_after_M', '-')} M "
              f"(-{params.get('reduction_pct', '-')}%)")
    sz = summary["size_mb"]
    for k in ("epoch_3.pth", "onnx", "trt_engine"):
        if k in sz:
            print(f"  size {k:<12}: {sz[k]} MB")
    lat_pt = summary["latency_pytorch_ms"]
    if lat_pt.get("ego_agent_forward"):
        print(f"  ego fwd (PyT): {lat_pt['ego_agent_forward']['mean']} ms")
    if lat_pt.get("bev_encoder"):
        print(f"  bev_enc (PyT): {lat_pt['bev_encoder']['mean']} ms")
    lat_trt = summary["latency_trt_ms"]
    if lat_trt.get("mean"):
        print(f"  TRT engine   : {lat_trt['mean']} ms")
    print(f"\n[out] {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
