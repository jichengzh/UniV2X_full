"""评估已保存的 pkl 计算 AMOTA (对单 GPU 模式 pkl 做 custom post-processing)"""
import sys, os, argparse
sys.path.insert(0, '/home/jichengzhi/UniV2X')
import warnings
warnings.filterwarnings("ignore")

import mmcv
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet.datasets import replace_ImageToTensor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("pkl")
    return p.parse_args()


def main():
    args = parse_args()

    # 加载 config + plugin
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib
        plugin_dir = cfg.get("plugin_dir", "projects/mmdet3d_plugin/")
        importlib.import_module(plugin_dir.rstrip("/").replace("/", "."))

    cfg.data.test.test_mode = True
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    dataset = build_dataset(cfg.data.test)
    print(f"Dataset: {len(dataset)} samples")

    # 加载 pkl
    outputs = mmcv.load(args.pkl)

    # 关键: baseline (custom_multi_gpu_test) 的 pkl 是 {'bbox_results': [...]}
    # 而 single_gpu_test 的 pkl 是直接的 [...]
    # 需要 wrap 成前者格式以兼容 dataset.evaluate()
    if isinstance(outputs, list):
        print(f"Wrapping list ({len(outputs)} samples) into dict with 'bbox_results' key")
        outputs = {"bbox_results": outputs}

    print(f"Loaded, structure: {type(outputs).__name__}")
    if isinstance(outputs, dict):
        print(f"  keys: {list(outputs.keys())}")
        if "bbox_results" in outputs:
            print(f"  bbox_results len: {len(outputs['bbox_results'])}")

    # 避免 spd 数据集的 tmp_dir GC 竞态: 显式指定 jsonfile_prefix
    persistent_dir = args.pkl.replace(".pkl", "_eval_tmp")
    os.makedirs(persistent_dir, exist_ok=True)
    print(f"\nRunning dataset.evaluate() with jsonfile_prefix={persistent_dir}")
    metrics = dataset.evaluate(outputs, jsonfile_prefix=persistent_dir)

    print("\n" + "=" * 60)
    print("  最终指标")
    print("=" * 60)
    key_metrics = [
        "pts_bbox_NuScenes/amota", "pts_bbox_NuScenes/amotp",
        "pts_bbox_NuScenes/mAP", "pts_bbox_NuScenes/NDS",
        "pts_bbox_NuScenes/recall", "pts_bbox_NuScenes/motar",
        "pts_bbox_NuScenes/mota", "pts_bbox_NuScenes/motp",
        "pts_bbox_NuScenes/tp", "pts_bbox_NuScenes/fp",
        "pts_bbox_NuScenes/fn", "pts_bbox_NuScenes/ids",
        "pts_bbox_NuScenes/mt", "pts_bbox_NuScenes/ml",
    ]
    for k in key_metrics:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                print(f"  {k:<40} {v:.4f}")
            else:
                print(f"  {k:<40} {v}")

    # Save clean metrics
    out_json = args.pkl.replace(".pkl", "_metrics.json")
    import json
    clean = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v == v}
    with open(out_json, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    print(f"\n指标已保存: {out_json}")


if __name__ == "__main__":
    main()
