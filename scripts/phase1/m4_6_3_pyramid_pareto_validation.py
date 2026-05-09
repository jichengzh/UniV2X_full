"""M4.6.3: Framework M4.7 Pareto validation — 抽样 Pareto + 非 Pareto 候选实测.

从 results/phase2_pareto_pyramid.csv 选 5 个 configs:
  - 1 Pareto-optimal: m4_7_pyramid_0040 (INT8+80% encoder+75% decoder+70% heads, 唯一 Pareto)
  - 2 channel pruning 候选 (非 Pareto, 但有 channel-only): 0013, 0027
  - 2 全 quant-only 候选 (非 Pareto): 0001, 0010

注意 INT8 在 PyTorch GPU 上不能直接做 (要 TRT), 简化为 FP16 等价 (auto-cast).
重点验证 channel pruning 维度的 framework ranking 是否成立.

输出:
  results/m4_6_3_pyramid_pareto_validation.csv
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
os.chdir(HEAL_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils  # type: ignore
from opencood.tools import train_utils  # type: ignore
from opencood.data_utils.datasets import build_dataset  # type: ignore
from opencood.utils import eval_utils  # type: ignore
from opencood.utils.common_utils import update_dict  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
PARETO_CSV = RESULTS_DIR / "phase2_pareto_pyramid.csv"
OUT_CSV = RESULTS_DIR / "m4_6_3_pyramid_pareto_validation.csv"

# 复用 M4.6.2 的 module mapping
MODULE_PATTERNS = {
    "backbone":  ["encoder_m1.", "backbone_m1."],
    "encoder":   ["pyramid_backbone.resnet.layer0."],
    "decoder":   ["pyramid_backbone.resnet.layer1.", "pyramid_backbone.resnet.layer2."],
    "heads":     ["cls_head", "reg_head", "dir_head", "pyramid_backbone.single_head_"],
    "v2x_comm":  ["pyramid_backbone.shrink_conv"],
}

# 选 5 个 candidates 验证 framework ranking
SELECTED_IDS = [
    "m4_7_pyramid_0040",  # Pareto (INT8 + 80% encoder + 75% decoder + 70% heads)
    "m4_7_pyramid_0013",  # channel pruning 候选 (非 Pareto)
    "m4_7_pyramid_0027",  # channel pruning 候选 (非 Pareto)
    "m4_7_pyramid_0001",  # quant-only 候选 (非 Pareto, negative control)
    "m4_7_pyramid_0010",  # quant-only 候选 (非 Pareto, negative control)
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--bn_recal_batches", type=int, default=50)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--range", default="102.4,102.4")
    p.add_argument("--note", default="")
    return p.parse_args()


def load_candidates() -> pd.DataFrame:
    df = pd.read_csv(PARETO_CSV)
    sub = df[df["config_id"].isin(SELECTED_IDS)].copy()
    print(f"Loaded {len(sub)} candidates from {PARETO_CSV.name}:")
    for _, r in sub.iterrows():
        print(f"  {r['config_id']:<25} pareto={r['is_pareto']}  prune_object={r['prune_object']}  "
              f"q_bits_decoder={r.get('q_bits__decoder', 'N/A')}  "
              f"prune_decoder={r.get('prune_rate__decoder', 0):.2f}")
    return sub


def make_hypes(opt):
    hypes = yaml_utils.load_yaml(None, opt)
    if "heter" in hypes:
        x_min, x_max = -float(opt.range.split(",")[0]), float(opt.range.split(",")[0])
        y_min, y_max = -float(opt.range.split(",")[1]), float(opt.range.split(",")[1])
        opt.note += f"_{x_max}_{y_max}"
        new_cav_range = [
            x_min, y_min,
            hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
            x_max, y_max,
            hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
        ]
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range,
        })
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                hypes = func(hypes)
    hypes["validate_dir"] = hypes["test_dir"]
    if "box_align" in hypes:
        hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]
    return hypes


def get_convs_for_framework_module(model, fw_module: str) -> list:
    patterns = MODULE_PATTERNS.get(fw_module, [])
    found = []
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.Conv2d):
            continue
        if m.groups != 1:
            continue
        if m.weight.shape[0] <= 4:
            continue
        if any(name.startswith(p) or p in name for p in patterns):
            found.append((name, m))
    return found


def apply_candidate(model, cand_row: pd.Series) -> dict:
    """根据 csv 行的 prune_rate__{module} 应用 multi-module pruning."""
    rates = {}
    for m in ("backbone", "encoder", "decoder", "heads", "v2x_comm"):
        col = f"prune_rate__{m}"
        rate = float(cand_row.get(col, 0.0) or 0.0)
        if rate > 0:
            rates[m] = rate
    n_layers = 0
    n_pruned = 0
    n_total = 0
    for m, rate in rates.items():
        convs = get_convs_for_framework_module(model, m)
        for name, conv in convs:
            prune.ln_structured(conv, name="weight", amount=rate, n=1, dim=0)
        n_layers += len(convs)
        n_total += sum(c.weight.shape[0] for _, c in convs)
        n_pruned += int(sum(c.weight.shape[0] for _, c in convs) * rate)
    return {"rates": rates, "n_layers": n_layers, "n_total": n_total, "n_pruned": n_pruned}


def maybe_apply_quant(model, cand_row: pd.Series) -> str:
    """处理 candidate 的 q_bits — INT8 在 PyTorch GPU 不可直接, 用 FP16 autocast 替代.

    Returns: precision tag "fp32" | "fp16_autocast" | "int8_proxy_fp16"
    """
    bits_set = set()
    for m in ("backbone", "encoder", "decoder", "heads", "v2x_comm"):
        b = cand_row.get(f"q_bits__{m}", "FP32")
        if b: bits_set.add(b)

    # 简化逻辑: 只要任一模块用 INT8 就标 INT8 proxy (用 fp16 autocast)
    if "INT8" in bits_set:
        return "int8_proxy_fp16"
    if "FP16" in bits_set:
        return "fp16_autocast"
    return "fp32"


def make_model(opt, hypes):
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.cuda().eval()
    return model


def bn_recalibrate(model, dataset, n_batches: int):
    if n_batches <= 0: return
    print(f"  BN recal: {n_batches} batches")
    model.train()
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=True, pin_memory=False, drop_last=False)
    seen = 0
    with torch.no_grad():
        for batch_data in loader:
            if seen >= n_batches: break
            if batch_data is None: continue
            try:
                batch_data = train_utils.to_device(batch_data, "cuda")
                _ = model(batch_data["ego"])
                seen += 1
            except Exception:
                continue
    model.eval()
    print(f"  BN recal done: {seen}/{n_batches}")


@torch.no_grad()
def eval_one(model, dataset, prec_tag: str, max_samples: int = 0) -> dict:
    loader = DataLoader(dataset, batch_size=1, num_workers=4,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, pin_memory=False, drop_last=False)
    result_stat = {0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.7: {"tp": [], "fp": [], "gt": 0, "score": []}}
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []
    n_done = 0
    from collections import OrderedDict as _OD
    use_amp = (prec_tag != "fp32")
    for i, batch_data in enumerate(loader):
        if max_samples > 0 and i >= max_samples: break
        if batch_data is None: continue
        batch_data = train_utils.to_device(batch_data, "cuda")
        torch.cuda.synchronize()
        starter.record()
        ctx = torch.cuda.amp.autocast(dtype=torch.float16) if use_amp \
              else torch.cuda.amp.autocast(enabled=False)
        with ctx:
            out = model(batch_data["ego"])
        if use_amp:
            out = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                   for k, v in out.items()}
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))
        try:
            pred_box, pred_score, gt_box = dataset.post_process(batch_data, _OD([("ego", out)]))
        except Exception as e:
            print(f"  [post_process error i={i}: {e}]")
            continue
        for iou in (0.3, 0.5, 0.7):
            eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stat, iou)
        n_done += 1
        if i % 200 == 0 and i > 0:
            print(f"  [{n_done}/{max_samples or len(dataset)}] cum_lat_p50={np.percentile(timings,50):.2f}ms")
        torch.cuda.empty_cache()
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, "/tmp", "m4_6_3")
    timings_arr = np.array(timings)
    return {
        "n_samples": n_done,
        "AP30": float(ap30), "AP50": float(ap50), "AP70": float(ap70),
        "lat_mean_ms": float(np.mean(timings_arr)),
        "lat_p50_ms": float(np.percentile(timings_arr, 50)),
        "lat_p99_ms": float(np.percentile(timings_arr, 99)),
    }


def main():
    opt = parse_args()
    candidates = load_candidates()

    print(f"\n{'='*60}")
    print(f"M4.6.3: framework M4.7 Pareto validation")
    print(f"  N candidates: {len(candidates)}, BN recal: {opt.bn_recal_batches}")
    print(f"{'='*60}")

    hypes = make_hypes(opt)
    np.random.seed(303)
    dataset = build_dataset(hypes, visualize=True, train=False)

    rows = []
    for _, cand in candidates.iterrows():
        cid = cand["config_id"]
        is_pareto = bool(cand.get("is_pareto", False))
        print(f"\n=== {cid} (pareto={is_pareto}) ===")
        model = make_model(opt, hypes)

        prune_stats = apply_candidate(model, cand)
        prec_tag = maybe_apply_quant(model, cand)
        print(f"  prune: {prune_stats['rates']}")
        print(f"  prec_tag: {prec_tag}")
        if prune_stats["n_pruned"] > 0:
            bn_recalibrate(model, dataset, opt.bn_recal_batches)

        eval_res = eval_one(model, dataset, prec_tag, opt.max_samples)
        row = {
            "config_id": cid,
            "is_pareto_predicted": is_pareto,
            "predicted_lat_4090_ms": float(cand.get("est_lat_4090_pytorch_ms", np.nan)),
            "predicted_params_M": float(cand.get("est_params_M", np.nan)),
            "prec_tag": prec_tag,
            "prune_object": cand.get("prune_object", "none"),
            **{f"prune_rate__{k}": float(cand.get(f"prune_rate__{k}", 0) or 0)
               for k in ("backbone", "encoder", "decoder", "heads", "v2x_comm")},
            **eval_res,
            "bn_recal_batches": opt.bn_recal_batches,
        }
        rows.append(row)
        print(f"  → measured AP30/50/70: {row['AP30']:.4f}/{row['AP50']:.4f}/{row['AP70']:.4f} "
              f"| lat_p50: {row['lat_p50_ms']:.2f}ms (predicted: {row['predicted_lat_4090_ms']:.2f}ms)")
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Wrote {OUT_CSV}")

    print(f"\n=== M4.6.3 framework Pareto validation 摘要 ===")
    show = df[["config_id", "is_pareto_predicted", "prune_object", "prec_tag",
               "predicted_lat_4090_ms", "lat_p50_ms",
               "AP30", "AP50", "AP70"]]
    print(show.to_string(index=False))

    # Pareto 真实性判定
    print(f"\n=== Framework Pareto 真实性判定 ===")
    pareto_rows = df[df["is_pareto_predicted"] == True]
    nonpareto_rows = df[df["is_pareto_predicted"] == False]
    if len(pareto_rows) > 0 and len(nonpareto_rows) > 0:
        for _, p in pareto_rows.iterrows():
            print(f"  Pareto candidate {p['config_id']}: AP50={p['AP50']:.4f} lat={p['lat_p50_ms']:.2f}ms")
            for _, np_r in nonpareto_rows.iterrows():
                p_dom = (p["AP50"] >= np_r["AP50"] and p["lat_p50_ms"] <= np_r["lat_p50_ms"]
                         and (p["AP50"] > np_r["AP50"] or p["lat_p50_ms"] < np_r["lat_p50_ms"]))
                np_dom = (np_r["AP50"] >= p["AP50"] and np_r["lat_p50_ms"] <= p["lat_p50_ms"]
                          and (np_r["AP50"] > p["AP50"] or np_r["lat_p50_ms"] < p["lat_p50_ms"]))
                if np_dom:
                    print(f"    ⚠️ {np_r['config_id']} 实际 dominate Pareto candidate (framework 排序错)")
                elif p_dom:
                    print(f"    ✅ {p['config_id']} dominate {np_r['config_id']} (framework 排序对)")
                else:
                    print(f"    ~ {np_r['config_id']} 跟 Pareto candidate 互不 dominate (一致 trade-off)")


if __name__ == "__main__":
    main()
