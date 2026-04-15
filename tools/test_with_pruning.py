"""
test.py 的带剪枝变体: 加载模型 → 应用剪枝 → 评估 AMOTA

用法:
    PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=29506 \
        tools/test_with_pruning.py \
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
        ckpts/univ2x_coop_e2e_stg2.pth \
        --prune-config prune_configs/p1_ffn_30pct.json \
        --out output/p1_ffn_30pct_results.pkl \
        --eval bbox \
        --launcher pytorch
"""
import argparse
import copy
import json
import os
import time
import warnings

import torch
import mmcv
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)

from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.univ2x.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent
from mmdet.datasets import replace_ImageToTensor

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="test.py + pruning")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--prune-config", required=True, help="剪枝配置 JSON 文件")
    parser.add_argument("--out", default="output/pruned_results.pkl", help="结果保存路径")
    parser.add_argument("--save-pruned", default=None, help="可选: 保存剪枝后的 checkpoint")
    parser.add_argument("--finetuned-ckpt", default=None,
                        help="可选: 剪枝+微调后的 checkpoint, 评估时会先按 prune-config 剪枝"
                             "生成架构, 再 load 这个 ckpt 覆盖权重 (即评估微调结果)")
    parser.add_argument("--fuse-conv-bn", action="store_true")
    parser.add_argument("--eval", type=str, nargs="+", help="评估指标, 如 'bbox'")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="pytorch")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    return parser.parse_args()


def load_prune_config(json_path):
    with open(json_path, "r") as f:
        cfg = json.load(f)
    locked = cfg.setdefault("locked", {})
    locked.setdefault("importance_criterion", "l1_norm")
    locked.setdefault("pruning_granularity", "local")
    locked.setdefault("iterative_steps", 5)
    locked.setdefault("round_to", 8)
    cfg.setdefault("encoder", {})
    cfg.setdefault("decoder", {})
    cfg.setdefault("heads", {})
    return cfg


def main():
    args = parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    assert args.out or args.eval, "必须指定 --out 或 --eval"

    # 1. 加载 config
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib
        plugin_dir = cfg.get("plugin_dir", "projects/mmdet3d_plugin/")
        _module_path = plugin_dir.rstrip("/").replace("/", ".")
        importlib.import_module(_module_path)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model_ego_agent.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    # 2. 非分布式模式 (剪枝后模型与 custom_multi_gpu_test 分布式路径不兼容)
    distributed = False

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # 3. 构建数据加载 (workers_per_gpu=0 避免 pickle 问题)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=samples_per_gpu,
        workers_per_gpu=0,
        dist=distributed, shuffle=False,
    )

    # 4. 构建 multi-agent 模型
    other_agent_names = [k for k in cfg.keys() if "model_other_agent" in k]
    model_other_agents = {}
    for name in other_agent_names:
        cfg.get(name).train_cfg = None
        other = build_model(cfg.get(name), test_cfg=cfg.get("test_cfg"))
        load_from = cfg.get(name).get("load_from")
        if load_from:
            load_checkpoint(other, load_from, map_location="cpu",
                            revise_keys=[(r"^model_ego_agent\.", "")])
        model_other_agents[name] = other

    cfg.model_ego_agent.train_cfg = None
    model_ego = build_model(cfg.model_ego_agent, test_cfg=cfg.get("test_cfg"))
    load_from = cfg.model_ego_agent.get("load_from")
    if load_from:
        load_checkpoint(model_ego, load_from, map_location="cpu",
                        revise_keys=[(r"^model_ego_agent\.", "")])

    model_multi = MultiAgent(model_ego, model_other_agents)
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model_multi)
    checkpoint = load_checkpoint(model_multi, args.checkpoint, map_location="cpu")

    if "CLASSES" in checkpoint.get("meta", {}):
        model_multi.model_ego_agent.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model_multi.model_ego_agent.CLASSES = dataset.CLASSES
    if "PALETTE" in checkpoint.get("meta", {}):
        model_multi.model_ego_agent.PALETTE = checkpoint["meta"]["PALETTE"]
    elif hasattr(dataset, "PALETTE"):
        model_multi.model_ego_agent.PALETTE = dataset.PALETTE

    # 5. 应用剪枝 (仅作用于 ego_agent)
    prune_cfg = load_prune_config(args.prune_config)
    print(f"\n{'='*60}")
    print(f"  应用剪枝配置: {args.prune_config}")
    print(f"  locked: {prune_cfg['locked']}")
    print(f"  encoder: {prune_cfg['encoder']}")
    print(f"  decoder: {prune_cfg['decoder']}")
    print(f"{'='*60}\n")

    params_before = sum(p.numel() for p in model_multi.model_ego_agent.parameters())

    from projects.mmdet3d_plugin.univ2x.pruning.prune_univ2x import apply_prune_config
    t0 = time.time()
    apply_prune_config(model_multi.model_ego_agent, prune_cfg, dataloader=None)
    prune_elapsed = time.time() - t0

    params_after = sum(p.numel() for p in model_multi.model_ego_agent.parameters())
    reduction = 1.0 - params_after / params_before

    print(f"\n{'='*60}")
    print(f"  剪枝完成 ({prune_elapsed:.1f}s)")
    print(f"  参数: {params_before:,d} -> {params_after:,d}")
    print(f"  缩减: {(params_before-params_after):,d} ({reduction*100:.2f}%)")
    print(f"{'='*60}\n")

    # 可选: 加载微调后的 checkpoint 覆盖剪枝后的随机权重
    if args.finetuned_ckpt:
        print(f"\n{'='*60}")
        print(f"  加载微调后 checkpoint: {args.finetuned_ckpt}")
        print(f"{'='*60}\n")
        # MMDataParallel 包装前 load, 保持 state_dict key 和模型结构一致
        ft_ckpt = load_checkpoint(model_multi, args.finetuned_ckpt, map_location="cpu")
        print(f"  已加载微调权重 (meta keys: {list(ft_ckpt.get('meta', {}).keys())})")

    # 可选: 保存剪枝后的 checkpoint
    if args.save_pruned:
        os.makedirs(os.path.dirname(args.save_pruned) or ".", exist_ok=True)
        torch.save({
            "state_dict": model_multi.state_dict(),
            "meta": {
                "source_ckpt": args.checkpoint,
                "prune_config": prune_cfg,
                "params_before": params_before,
                "params_after": params_after,
            },
        }, args.save_pruned)
        print(f"已保存剪枝后 checkpoint: {args.save_pruned}")

    if args.fuse_conv_bn:
        model_multi = fuse_conv_bn(model_multi)

    # 6. 评估 (单 GPU + MMDataParallel + single_gpu_test)
    from mmdet3d.apis import single_gpu_test
    model_multi = MMDataParallel(model_multi.cuda(), device_ids=[0])
    model_multi.eval()

    print(f"\n开始评估（workers=0, 预计 ~3-5 min）...")
    t0 = time.time()
    outputs = single_gpu_test(model_multi, data_loader)
    eval_elapsed = time.time() - t0
    print(f"推理完成 ({eval_elapsed:.1f}s)")

    # 7. 保存 + 评估指标
    if True:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)

        if args.eval:
            print(f"\n评估指标: {args.eval}")
            # 显式传 jsonfile_prefix 避免 tmp_dir GC race (tracking eval 需要同一个目录)
            jfp = args.out.replace(".pkl", "")
            os.makedirs(os.path.dirname(jfp) or ".", exist_ok=True)
            metrics = dataset.evaluate(outputs, jsonfile_prefix=jfp)
            print(f"\n=== 最终指标 ===")
            key_metrics = ["pts_bbox_NuScenes/amota", "pts_bbox_NuScenes/amotp",
                           "pts_bbox_NuScenes/mAP", "pts_bbox_NuScenes/NDS",
                           "pts_bbox_NuScenes/recall", "pts_bbox_NuScenes/motar",
                           "pts_bbox_NuScenes/mota", "pts_bbox_NuScenes/motp",
                           "pts_bbox_NuScenes/tp", "pts_bbox_NuScenes/fp",
                           "pts_bbox_NuScenes/fn", "pts_bbox_NuScenes/ids",
                           "pts_bbox_NuScenes/mt", "pts_bbox_NuScenes/ml"]
            for k in key_metrics:
                if k in metrics:
                    v = metrics[k]
                    if isinstance(v, float):
                        print(f"  {k:<40} {v:.4f}")
                    else:
                        print(f"  {k:<40} {v}")

            # 保存指标 JSON
            metrics_path = args.out.replace(".pkl", "_metrics.json")
            # 过滤 float 以外的值 (有 nan)
            clean_metrics = {k: (v if isinstance(v, (int, float)) and v == v else None)
                            for k, v in metrics.items()
                            if isinstance(v, (int, float))}
            clean_metrics["_experiment_meta"] = {
                "prune_config": prune_cfg,
                "params_before": params_before,
                "params_after": params_after,
                "reduction_ratio": reduction,
                "prune_elapsed_sec": prune_elapsed,
                "eval_elapsed_sec": eval_elapsed,
            }
            with open(metrics_path, "w") as f:
                json.dump(clean_metrics, f, indent=2, default=str)
            print(f"\n指标已保存: {metrics_path}")


if __name__ == "__main__":
    main()
