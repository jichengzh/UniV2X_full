"""
统一配置驱动剪枝 + 评估管线

用法:
    python tools/prune_and_eval.py \
        --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
        --checkpoint work_dirs/latest.pth \
        --prune-config prune_configs/moderate.json \
        --output work_dirs/pruned_moderate.pth \
        [--eval]
"""
import argparse
import json
import logging
import os
import time
import warnings

import torch
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prune_and_eval")


def parse_args():
    parser = argparse.ArgumentParser(description="配置驱动剪枝 + 评估管线")
    parser.add_argument("--config", required=True, help="mmdet3d 模型配置 (.py)")
    parser.add_argument("--checkpoint", required=True, help="模型 checkpoint (.pth)")
    parser.add_argument("--prune-config", required=True, help="剪枝配置 (.json)")
    parser.add_argument("--output", required=True, help="剪枝后模型保存路径 (.pth)")
    parser.add_argument("--eval", action="store_true", help="剪枝后评估 AMOTA")
    parser.add_argument("--eval-out", default="output/pruned_results.pkl", help="评估结果路径")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    parser.add_argument("--calib-batches", type=int, default=32, help="梯度收集 batch 数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    return parser.parse_args()


# ============================================================================
# 配置加载
# ============================================================================

def load_prune_config(json_path):
    """加载并校验 prune_config.json"""
    with open(json_path, "r") as f:
        config = json.load(f)

    # locked 段默认值
    locked = config.setdefault("locked", {})
    locked.setdefault("importance_criterion", "taylor")
    locked.setdefault("pruning_granularity", "local")
    locked.setdefault("iterative_steps", 5)
    locked.setdefault("round_to", 8)

    # 搜索维度默认值
    config.setdefault("encoder", {})
    config.setdefault("decoder", {})
    config.setdefault("heads", {})
    config.setdefault("finetune", {"epochs": 10})
    config.setdefault("constraints", {})

    # ratio 范围校验
    for section_name in ("encoder", "decoder", "heads"):
        section = config[section_name]
        for key, val in section.items():
            if "ratio" in key and isinstance(val, (int, float)):
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"{section_name}.{key}={val} 超出 [0.0, 1.0]")

    # constraints 默认值
    constraints = config["constraints"]
    constraints.setdefault("skip_layers", ["sampling_offsets", "attention_weights"])
    constraints.setdefault("min_channels", 64)
    constraints.setdefault("channel_alignment", 8)

    return config


# ============================================================================
# 模型加载
# ============================================================================

def load_model_from_config(cfg_path, ckpt_path, device="cuda:0"):
    """从 mmdet3d config + checkpoint 加载模型"""
    cfg = Config.fromfile(cfg_path)

    # 导入插件
    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib
        if hasattr(cfg, "plugin_dir"):
            _module_path = os.path.dirname(cfg.plugin_dir).replace("/", ".")
            importlib.import_module(_module_path)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )

    checkpoint = load_checkpoint(model, ckpt_path, map_location="cpu")

    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]

    model = model.to(device)
    model.eval()
    return model, cfg


# ============================================================================
# 统计与报告
# ============================================================================

def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def report_model_stats(model, prefix=""):
    """打印模型统计"""
    total_params, trainable_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"  {prefix}")
    print(f"  总参数量:   {total_params:>12,d} ({total_params/1e6:.2f}M)")
    print(f"  可训练参数: {trainable_params:>12,d} ({trainable_params/1e6:.2f}M)")

    # 按类别汇总
    categories = {
        "FFN": lambda n: "ffns" in n and isinstance(dict(model.named_modules()).get(n), torch.nn.Linear),
        "Attn Proj": lambda n: ("value_proj" in n or "output_proj" in n),
        "Det Heads": lambda n: ("cls_branches" in n or "reg_branches" in n),
        "Coord-sensitive": lambda n: ("sampling_offsets" in n or "attention_weights" in n),
    }

    print(f"  {'类别':<18} {'参数量':>12}")
    print(f"  {'-'*30}")
    for cat_name, matcher in categories.items():
        cat_params = 0
        for name, module in model.named_modules():
            if matcher(name) and isinstance(module, torch.nn.Linear):
                cat_params += module.weight.numel()
                if module.bias is not None:
                    cat_params += module.bias.numel()
        if cat_params > 0:
            print(f"  {cat_name:<18} {cat_params:>12,d}")

    print(f"{'='*60}")
    return total_params


def verify_constraints(model, prune_config):
    """校验剪枝后模型是否满足约束"""
    constraints = prune_config.get("constraints", {})
    alignment = constraints.get("channel_alignment", 8)
    min_ch = constraints.get("min_channels", 64)
    skip_patterns = constraints.get("skip_layers", [])

    violations = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        is_skip = any(pat in name for pat in skip_patterns)
        if not is_skip:
            if module.out_features % alignment != 0:
                violations.append(
                    f"[ALIGN] {name}: out={module.out_features} 不是 {alignment} 的倍数"
                )
            if min_ch < module.out_features < min_ch and module.out_features > 1:
                violations.append(
                    f"[MIN_CH] {name}: out={module.out_features} < {min_ch}"
                )

    if violations:
        print(f"\n  [WARN] {len(violations)} 个约束违规:")
        for v in violations[:10]:  # 最多显示 10 个
            print(f"    {v}")
    else:
        print(f"\n  [OK] 约束校验通过 (alignment={alignment}, min_ch={min_ch})")

    return violations


# ============================================================================
# 评估
# ============================================================================

def evaluate_amota(model, cfg, output_path):
    """评估 AMOTA"""
    from mmcv.parallel import MMDataParallel
    from mmdet3d.apis import single_gpu_test
    from mmdet3d.datasets import build_dataset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from mmdet.datasets import replace_ImageToTensor

    cfg.data.test.test_mode = True
    if isinstance(cfg.data.test, dict):
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1,
        workers_per_gpu=cfg.data.get("workers_per_gpu", 4),
        dist=False, shuffle=False,
    )

    model_parallel = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model_parallel, data_loader)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mmcv.dump(outputs, output_path)

    metrics = dataset.evaluate(outputs, jsonfile_prefix=output_path.replace(".pkl", ""))
    amota = metrics.get("AMOTA", metrics.get("amota", None))
    if amota is not None:
        print(f"\n  [EVAL] AMOTA = {amota:.4f}")
    else:
        print(f"\n  [EVAL] 指标: {metrics}")

    return metrics


# ============================================================================
# 主流程
# ============================================================================

def main():
    args = parse_args()

    torch.cuda.set_device(args.gpu_id)
    device = f"cuda:{args.gpu_id}"
    from mmdet.apis import set_random_seed
    set_random_seed(args.seed, deterministic=True)

    # 1. 加载剪枝配置
    print(f"[1/6] 加载剪枝配置: {args.prune_config}")
    prune_config = load_prune_config(args.prune_config)
    print(f"  locked: {prune_config['locked']}")
    print(f"  encoder: {prune_config['encoder']}")
    print(f"  decoder: {prune_config['decoder']}")
    print(f"  heads: {prune_config['heads']}")

    # 2. 加载模型
    print(f"\n[2/6] 加载模型: {args.checkpoint}")
    model, cfg = load_model_from_config(args.config, args.checkpoint, device)
    original_params = report_model_stats(model, prefix="原始模型")

    # 3. 准备 dataloader
    print(f"\n[3/6] 准备数据")
    criterion = prune_config["locked"].get("importance_criterion", "taylor")
    dataloader = None
    if criterion in ("taylor", "hessian"):
        from mmdet3d.datasets import build_dataset
        from projects.mmdet3d_plugin.datasets.builder import build_dataloader as build_dl
        try:
            dataset = build_dataset(cfg.data.train)
            dataloader = build_dl(
                dataset, samples_per_gpu=1,
                workers_per_gpu=cfg.data.get("workers_per_gpu", 4),
                dist=False, shuffle=True,
            )
            print(f"  criterion={criterion}, 将收集 {args.calib_batches} batches 梯度")
        except Exception as e:
            logger.warning("无法构建训练 dataloader: %s, 降级到 l1_norm", e)
            prune_config["locked"]["importance_criterion"] = "l1_norm"
    else:
        print(f"  criterion={criterion}, 无需梯度收集")

    # 4. 执行剪枝
    print(f"\n[4/6] 执行剪枝...")
    t0 = time.time()
    from projects.mmdet3d_plugin.univ2x.pruning.prune_univ2x import apply_prune_config
    model = apply_prune_config(model, prune_config, dataloader=dataloader)
    elapsed = time.time() - t0
    print(f"  剪枝完成, 耗时 {elapsed:.1f}s")

    # 5. 校验约束 + 报告
    print(f"\n[5/6] 校验约束 + 统计")
    pruned_params = report_model_stats(model, prefix="剪枝后模型")
    verify_constraints(model, prune_config)

    reduction = 1.0 - pruned_params / original_params
    print(f"  参数缩减: {reduction*100:.1f}% ({original_params/1e6:.2f}M -> {pruned_params/1e6:.2f}M)")

    # 6. 保存
    print(f"\n[6/6] 保存: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "meta": {
            "prune_config": prune_config,
            "original_params": original_params,
            "pruned_params": pruned_params,
            "reduction_ratio": reduction,
        },
    }, args.output)
    print(f"  已保存 ({os.path.getsize(args.output)/1e6:.1f}MB)")

    # 可选评估
    if args.eval:
        print(f"\n[EVAL] 评估 AMOTA...")
        evaluate_amota(model, cfg, args.eval_out)

    print(f"\n{'='*60}")
    print(f"  管线完成! 参数缩减 {reduction*100:.1f}%, 输出: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
