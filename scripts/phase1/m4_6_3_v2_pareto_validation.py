"""M4.6.3 v2: 用 M4.7 v3 (LGB v6 amota + rule-based lat) 的 Pareto 候选实测 AP.

抽样 15 个 candidates (从 results/phase2_pareto_pyramid_v3.csv):
  - 10 个 Pareto 候选 (覆盖前沿不同位置)
  - 5 个 非 Pareto 候选 (negative control)

每个候选用 framework adapter 5 模块 → HEAL 实际模块 mapping 应用 mask-based pruning
+ BN recal 50 batches + autocast(fp16) (INT8 在 PyTorch GPU 不支持, 用 FP16 替代)
跑完整 OPV2V test 2170 samples 测 AP30/50/70 + lat_p50.

输出:
  results/m4_6_3_v2_pareto_validation.csv (15 行实测)
  → 跟 v3 预测对比, 验证 v6 + rule-based 是否有效

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
PARETO_CSV = RESULTS_DIR / "phase2_pareto_pyramid_v3.csv"
OUT_CSV = RESULTS_DIR / "m4_6_3_v2_pareto_validation.csv"

MODULE_PATTERNS = {
    "backbone":  ["encoder_m1.", "backbone_m1."],
    "encoder":   ["pyramid_backbone.resnet.layer0."],
    "decoder":   ["pyramid_backbone.resnet.layer1.", "pyramid_backbone.resnet.layer2."],
    "heads":     ["cls_head", "reg_head", "dir_head", "pyramid_backbone.single_head_"],
    "v2x_comm":  ["pyramid_backbone.shrink_conv"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--bn_recal_batches", type=int, default=50)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--range", default="102.4,102.4")
    p.add_argument("--note", default="")
    p.add_argument("--n_pareto", type=int, default=10)
    p.add_argument("--n_nonpareto", type=int, default=5)
    return p.parse_args()


def load_candidates(opt) -> pd.DataFrame:
    df = pd.read_csv(PARETO_CSV)
    pareto = df[df["is_pareto"]].sort_values("predicted_lat_ms").head(opt.n_pareto)
    # 非 Pareto: 选 lat × amota 上中下三档非 Pareto, 共 5 个
    non = df[~df["is_pareto"]].sort_values("predicted_lat_ms")
    n_non = min(opt.n_nonpareto, len(non))
    if n_non > 0:
        # 取 evenly spaced 5 个
        idx = np.linspace(0, len(non) - 1, n_non).astype(int)
        nonpareto = non.iloc[idx]
        sel = pd.concat([pareto, nonpareto])
    else:
        sel = pareto
    print(f"Selected {len(sel)} candidates ({len(pareto)} pareto + {len(sel)-len(pareto)} non-pareto)")
    return sel.reset_index(drop=True)


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
    rates = {}
    for m in ("backbone", "encoder", "decoder", "heads", "v2x_comm"):
        rate = float(cand_row.get(f"prune_rate__{m}", 0.0) or 0.0)
        if rate > 0:
            rates[m] = rate
    n_total = 0
    n_pruned = 0
    n_layers = 0
    for m, rate in rates.items():
        convs = get_convs_for_framework_module(model, m)
        for name, conv in convs:
            prune.ln_structured(conv, name="weight", amount=rate, n=1, dim=0)
        n_layers += len(convs)
        n_total += sum(c.weight.shape[0] for _, c in convs)
        n_pruned += int(sum(c.weight.shape[0] for _, c in convs) * rate)
    return {"rates": rates, "n_layers": n_layers, "n_total": n_total, "n_pruned": n_pruned}


def maybe_prec(cand: pd.Series) -> str:
    bits = set()
    for m in ("backbone", "encoder", "decoder", "heads", "v2x_comm"):
        b = cand.get(f"q_bits__{m}", "FP32")
        if b: bits.add(b)
    if "INT8" in bits: return "int8_proxy_fp16"
    if "FP16" in bits: return "fp16_autocast"
    return "fp32"


def make_model(opt, hypes):
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.cuda().eval()
    return model


def bn_recalibrate(model, dataset, n_batches: int):
    if n_batches <= 0: return
    print(f"  BN recal: {n_batches}")
    model.train()
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test, shuffle=True,
                        pin_memory=False, drop_last=False)
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


@torch.no_grad()
def eval_one(model, dataset, prec_tag: str, max_samples: int = 0) -> dict:
    loader = DataLoader(dataset, batch_size=1, num_workers=4,
                        collate_fn=dataset.collate_batch_test, shuffle=False,
                        pin_memory=False, drop_last=False)
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
            print(f"  [post err i={i}: {e}]"); continue
        for iou in (0.3, 0.5, 0.7):
            eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stat, iou)
        n_done += 1
        if i % 400 == 0 and i > 0:
            print(f"  [{n_done}/{max_samples or len(dataset)}] cum_lat_p50={np.percentile(timings,50):.2f}ms")
        torch.cuda.empty_cache()
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, "/tmp", "m4_6_3_v2")
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
    candidates = load_candidates(opt)

    print(f"\n{'='*60}\nM4.6.3 v2: 验证 {len(candidates)} 候选 (M4.7 v3 输出)\n{'='*60}")
    hypes = make_hypes(opt)
    np.random.seed(303)
    dataset = build_dataset(hypes, visualize=True, train=False)

    rows = []
    for ci, cand in candidates.iterrows():
        cid = cand["config_id"]
        is_pareto = bool(cand.get("is_pareto", False))
        print(f"\n=== {ci+1}/{len(candidates)}: {cid} (pareto={is_pareto}) ===")
        model = make_model(opt, hypes)
        stats = apply_candidate(model, cand)
        prec = maybe_prec(cand)
        print(f"  prune: {stats['rates']} | prec: {prec}")
        if stats["n_pruned"] > 0:
            bn_recalibrate(model, dataset, opt.bn_recal_batches)
        eval_res = eval_one(model, dataset, prec, opt.max_samples)
        row = {
            "config_id": cid,
            "is_pareto_predicted": is_pareto,
            "predicted_lat_ms": float(cand.get("predicted_lat_ms", np.nan)),
            "predicted_amota": float(cand.get("predicted_amota", np.nan)),
            "prec_tag": prec,
            "prune_object": cand.get("prune_object", "none"),
            **{f"prune_rate__{m}": float(cand.get(f"prune_rate__{m}", 0) or 0)
               for m in ("backbone", "encoder", "decoder", "heads", "v2x_comm")},
            **eval_res,
            "bn_recal_batches": opt.bn_recal_batches,
        }
        rows.append(row)
        print(f"  → measured AP50={row['AP50']:.4f} (pred amota={row['predicted_amota']:.4f}) "
              f"| measured lat={row['lat_p50_ms']:.2f}ms (pred {row['predicted_lat_ms']:.2f}ms)")

        # increment commit per row (避 long run loss)
        df = pd.DataFrame(rows)
        df["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(OUT_CSV, index=False)
        del model
        torch.cuda.empty_cache()

    # 最终摘要 + framework 排序判定
    df = pd.read_csv(OUT_CSV)
    print(f"\n=== M4.6.3 v2 摘要 (N={len(df)}) ===")
    show = df[["config_id", "is_pareto_predicted", "prune_object",
               "predicted_amota", "AP50", "predicted_lat_ms", "lat_p50_ms"]]
    print(show.to_string(index=False))

    print(f"\n=== Framework Pareto 验证: 实测 vs v6 预测 ===")
    pareto_rows = df[df["is_pareto_predicted"]]
    nonpareto_rows = df[~df["is_pareto_predicted"]]
    correct = 0; wrong = 0; tradeoff = 0
    for _, p in pareto_rows.iterrows():
        for _, np_r in nonpareto_rows.iterrows():
            p_dom = (p["AP50"] >= np_r["AP50"] and p["lat_p50_ms"] <= np_r["lat_p50_ms"]
                     and (p["AP50"] > np_r["AP50"] or p["lat_p50_ms"] < np_r["lat_p50_ms"]))
            np_dom = (np_r["AP50"] >= p["AP50"] and np_r["lat_p50_ms"] <= p["lat_p50_ms"]
                      and (np_r["AP50"] > p["AP50"] or np_r["lat_p50_ms"] < p["lat_p50_ms"]))
            if p_dom: correct += 1
            elif np_dom: wrong += 1
            else: tradeoff += 1
    total = correct + wrong + tradeoff
    print(f"  pareto vs non-pareto pairwise: {correct} 正确 / {wrong} 错位 / {tradeoff} tradeoff")
    print(f"  framework 排序正确率 (excl tradeoff): {correct}/{correct+wrong} = "
          f"{correct/max(correct+wrong,1)*100:.1f}%")


if __name__ == "__main__":
    main()
