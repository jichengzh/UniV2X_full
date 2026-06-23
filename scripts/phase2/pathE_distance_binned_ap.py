"""Path E (AP 激活, 免训, 探前沿级 AP trade-off) — DAIR val 距离分箱重算 AP.

动机: 全集 AP 上剪枝/INT8 的 ΔAP 太小(过参数化)。难样本(远距)上量化/剪枝
可能掉得更多 → 唯一可能产生真前沿 AP trade-off 的路径(A/B 是消融非前沿)。

★ 难度切分定义(落盘可复现, ISS-015):
  每个 box 取 8 角点中心 (mean over corners) 的 ego 距离 d = sqrt(cx^2 + cy^2)。
  分箱(DAIR range 102.4×51.2): near [0,30) / mid [30,50) / far [50,∞)。
  对 pred 与 gt 各自按 d 过滤到同一箱, 箱内 caluclate_tp_fp → 箱内 AP(range-conditioned AP)。
  n_gt(每箱真值框数)= 显著性判据, 随 AP 一起报。

ISS-015 四判据落实:
  1. 平移≠trade-off: 报告同时给 full + near/mid/far, 由 data 判"相对关系是否变"(非仅整体下移)。
  2. 噪声尺度: far 箱样本少 → 噪声更大, 用 far 箱自身 n_gt 评显著性, 勿套全集 0.001。
  3. 每 AP 带 n_gt。  4. 主轴 AP70。

复用现成引擎(免 build): base_fp16/base_int8/pruned50_int8/pruned75_int8 (collab N=2)。
口径: DAIR val 1789, body_subnet_collab2, finetuned ckpt。
GPU: CUDA_VISIBLE_DEVICES(默认 0)。SMOKE_N 环境变量设小样本数先验证。
输出: results/pathE_distance_binned_ap.json
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np
import torch

GPU = os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
SMOKE_N = int(os.environ.get("SMOKE_N", "0"))  # >0 则只跑这么多样本(验证用)

REPO_ROOT = Path("/home/jichengzhi/UniV2X")
sys.path.insert(0, str(REPO_ROOT / "scripts/phase1"))
# 导入 hybrid 模块(import 期会 chdir 到 HEAL + sys.path 加 HEAL)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "hyb", str(REPO_ROOT / "scripts/phase1/m4_8_hybrid_infer_ap.py"))
hyb = importlib.util.module_from_spec(spec); spec.loader.exec_module(hyb)

from torch.utils.data import DataLoader
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.utils.common_utils import update_dict

# 距离分箱阈值(可复现, 改这里即改定义)
# data-orchestrator 要求: far(>50m)是主体非尾部(smoke 559/870), 加 60/80m 细分隔离真难尾。
BINS = [("near", 0.0, 30.0), ("mid", 30.0, 50.0),
        ("r50_60", 50.0, 60.0), ("r60_80", 60.0, 80.0), ("r80plus", 80.0, 1e9),
        ("far50plus", 50.0, 1e9), ("full", 0.0, 1e9)]
CKB = "/home/jichengzhi/heal_research/checkpoints/stage1"
CONFIGS = [
    ("base_fp16",     f"{CKB}/Pyramid_DAIR_m1_base_2023_08_14_11_42_29", "models/stage_a_cache/base_fp16.engine"),
    ("base_int8",     f"{CKB}/Pyramid_DAIR_m1_base_2023_08_14_11_42_29", "models/stage_a_cache/base_int8.engine"),
    ("pruned50_int8", f"{CKB}/Pyramid_DAIR_m1_pruned50_2026_05_10",      "models/stage_a_cache/pruned50_int8.engine"),
    ("pruned75_int8", f"{CKB}/Pyramid_DAIR_m1_pruned75_2026_05_10",      "models/stage_a_cache/pruned75_int8.engine"),
]
RANGE = "102.4,51.2"
N_SAMPLES = SMOKE_N if SMOKE_N > 0 else 1789


def box_center_dist(boxes):
    """boxes: (N,8,3) torch/np -> (N,) ego 距离 sqrt(cx^2+cy^2)."""
    if boxes is None or len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    b = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
    c = b.mean(axis=1)  # (N,3)
    return np.sqrt(c[:, 0] ** 2 + c[:, 1] ** 2)


def filter_by_bin(boxes, scores, lo, hi):
    """按中心距离 [lo,hi) 过滤 boxes(+scores)。"""
    if boxes is None or len(boxes) == 0:
        return boxes, scores
    d = box_center_dist(boxes)
    m = (d >= lo) & (d < hi)
    idx = np.nonzero(m)[0]
    if len(idx) == len(boxes):
        return boxes, scores
    bsel = boxes[idx]
    ssel = scores[idx] if scores is not None else None
    return bsel, ssel


def build_hypes_model_dataset(model_dir):
    hypes = yaml_utils.load_yaml(str(Path(model_dir) / "config.yaml"))
    if "heter" in hypes:
        x_max, y_max = (float(v) for v in RANGE.split(","))
        new_range = [-x_max, -y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
                     x_max, y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5]]
        hypes = update_dict(hypes, {"cav_lidar_range": new_range, "lidar_range": new_range, "gt_range": new_range})
        import importlib
        yu = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        hypes = getattr(yu, hypes["yaml_parser"])(hypes)
    hypes["validate_dir"] = hypes["test_dir"]
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(model_dir, model)
    model.cuda().eval()
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test, shuffle=False)
    return model, dataset, loader


def empty_stat():
    return {th: {"tp": [], "fp": [], "gt": 0, "score": []} for th in (0.3, 0.5, 0.7)}


def run_config(tag, model_dir, engine_rel):
    engine = str(REPO_ROOT / engine_rel)
    print(f"\n=== {tag} | engine={Path(engine).name} | model_dir={Path(model_dir).name} ===")
    model, dataset, loader = build_hypes_model_dataset(model_dir)
    trt_collab = hyb.TrtCollabN2(engine, spatial_shape=(2, 64, 128, 256), tego_shape=(2, 2, 3))
    # 每个箱一份 result_stat
    stats = {name: empty_stat() for name, _, _ in BINS}
    n_done = n_trt = n_fb = 0
    t0 = time.time()
    with torch.inference_mode():
        for batch_data in loader:
            if batch_data is None:
                continue
            if n_done >= N_SAMPLES:
                break
            batch_data = train_utils.to_device(batch_data, "cuda")
            out = hyb.hybrid_forward(model, batch_data, None, trt_collab)
            n_trt += (out["_path"] == "trt_collab"); n_fb += (out["_path"] == "pytorch_fallback")
            pred_box, pred_score, gt_box = dataset.post_process(batch_data, {"ego": out})
            for name, lo, hi in BINS:
                pb, ps = filter_by_bin(pred_box, pred_score, lo, hi)
                gb, _ = filter_by_bin(gt_box, None, lo, hi)
                gb = gb if gb is not None else gt_box
                for th in (0.3, 0.5, 0.7):
                    eval_utils.caluclate_tp_fp(pb, ps, gb if gb is not None else gt_box, stats[name], th)
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {n_done}/{N_SAMPLES} trt={n_trt} fb={n_fb} {time.time()-t0:.0f}s")
    # 每箱算 AP + 取 n_gt
    res = {}
    for name, _, _ in BINS:
        out_dir = REPO_ROOT / f"results/pathE_eval/{tag}_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        ap30, ap50, ap70 = eval_utils.eval_final_results(stats[name], str(out_dir))
        res[name] = {"ap30": float(ap30), "ap50": float(ap50), "ap70": float(ap70),
                     "n_gt": int(stats[name][0.7]["gt"])}
    res["_meta"] = {"n_samples": n_done, "n_trt_collab": int(n_trt), "n_pytorch_fb": int(n_fb),
                    "elapsed_min": round((time.time()-t0)/60, 1)}
    print(f"  {tag}: r80plus ap70={res['r80plus']['ap70']:.4f}(n_gt={res['r80plus']['n_gt']}) "
          f"far50plus={res['far50plus']['ap70']:.4f} full={res['full']['ap70']:.4f}(n_gt={res['full']['n_gt']})")
    return res


def main():
    all_res = {"bins_def": {n: [lo, hi] for n, lo, hi in BINS},
               "range": RANGE, "n_samples_target": N_SAMPLES, "smoke": SMOKE_N > 0,
               "configs": {}}
    for tag, model_dir, engine_rel in CONFIGS:
        try:
            all_res["configs"][tag] = run_config(tag, model_dir, engine_rel)
        except Exception as e:
            import traceback; traceback.print_exc()
            all_res["configs"][tag] = {"error": str(e)}
    suffix = "_smoke" if SMOKE_N > 0 else ""
    out = REPO_ROOT / f"results/pathE_distance_binned_ap{suffix}.json"
    out.write_text(json.dumps(all_res, indent=2))
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
