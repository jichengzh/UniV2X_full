"""
评估已保存的 .pkl 推理结果, 计算 AMOTA 等指标。

用法:
    PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=29504 \
        tools/eval_pkl.py \
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
        output/dcn_baseline_results.pkl
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet.datasets import replace_ImageToTensor


def parse_args():
    p = argparse.ArgumentParser(description="评估保存的推理结果")
    p.add_argument("config", help="mmdet3d 配置文件")
    p.add_argument("pkl", help="推理结果 .pkl")
    p.add_argument("--launcher", default="none", choices=["none", "pytorch"])
    p.add_argument("--local_rank", "--local-rank", type=int, default=0)
    p.add_argument("--eval-options", default=None,
                   help="评估额外选项 (json string)")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # 加载 plugin
    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib
        plugin_dir = cfg.get("plugin_dir", "projects/mmdet3d_plugin/")
        _module_path = plugin_dir.rstrip("/").replace("/", ".")
        importlib.import_module(_module_path)

    # 构建 test 数据集
    cfg.data.test.test_mode = True
    if isinstance(cfg.data.test, dict):
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)

    # 加载推理结果
    print(f"加载推理结果: {args.pkl}")
    outputs = mmcv.load(args.pkl)
    print(f"  样本数: {len(outputs)}")
    if len(outputs) > 0:
        print(f"  样本 0 keys: {list(outputs[0].keys()) if isinstance(outputs[0], dict) else type(outputs[0])}")

    # 评估
    print("\n开始评估...")
    eval_kwargs = {}
    if args.eval_options:
        import json
        eval_kwargs = json.loads(args.eval_options)

    metrics = dataset.evaluate(outputs, **eval_kwargs)

    print("\n=== 评估结果 ===")
    if isinstance(metrics, dict):
        for k, v in sorted(metrics.items()):
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        print(metrics)


if __name__ == "__main__":
    main()
