"""
剪枝后微调 (Phase B.0 / B.1 / B.3 共用入口)

与 tools/train.py 的关系:
    - 保留 train.py 的完整流程: config 解析 → model build → MultiAgent 包装
    - 新增: 在 custom_train_model 之前插入 apply_prune_config
    - 新增: CLI 控制 lr 缩放、epochs、warmup、是否冻结 backbone

用法:
    # B.0.1 冒烟: P1 FFN 30% + 3 epoch 微调（只训 pts_bbox_head）
    PYTHONPATH=/home/jichengzhi/UniV2X python tools/finetune_pruned.py \
        projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
        ckpts/univ2x_coop_e2e_stg2.pth \
        --prune-config prune_configs/p1_ffn_30pct.json \
        --epochs 3 --lr-scale 0.1 \
        --train-modules pts_bbox_head \
        --work-dir work_dirs/ft_p1_30

    # seg_head 剪枝微调（只训 seg_head, 内存最省）
    PYTHONPATH=/home/jichengzhi/UniV2X python tools/finetune_pruned.py \
        ... --prune-config prune_configs/p1_with_seg_mlp_30pct.json \
        --train-modules seg_head

    # 联合剪枝微调
    PYTHONPATH=/home/jichengzhi/UniV2X python tools/finetune_pruned.py \
        ... --train-modules pts_bbox_head,seg_head

通用冻结规则:
    - 被剪模块 + 其下游: 必训 (通过 --train-modules 指定)
    - 未剪上游: 可冻 (上游未变, 下游自适应)
    - other_agent_*: 永久冻结 (V2X 对端不在本工作剪枝范围)

成功标准:
    - loss 从初始 > 2.0 下降到 < 1.5 (3 epoch)
    - 第 3 epoch 不发散
    - 输出 checkpoint 可供评估，理想 AMOTA ≥ 剪枝零微调 baseline
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
import warnings
from os import path as osp

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/jichengzhi/UniV2X")

import torch
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import init_dist, get_dist_info, load_checkpoint
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent


logger_name = "finetune_pruned"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="剪枝后微调 (Phase B.0/B.1/B.3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("config", help="mmdet3d 配置文件")
    p.add_argument("checkpoint", help="原始(未剪枝) checkpoint")
    p.add_argument("--prune-config", required=True,
                   help="剪枝配置 JSON (prune_configs/*.json)")
    p.add_argument("--epochs", type=int, default=3,
                   help="微调 epoch 数 (默认 3, B.0 冒烟)")
    p.add_argument("--lr-scale", type=float, default=0.1,
                   help="lr 缩放系数, 乘以 cfg 原 lr (默认 0.1)")
    p.add_argument("--warmup-iters", type=int, default=None,
                   help="warmup 步数 (默认继承 cfg)")
    p.add_argument("--train-modules", type=str, default="pts_bbox_head",
                   help="可训模块 (逗号分隔, 名字匹配 ego_agent 的直接子模块); "
                        "'all' 表示训全部 ego_agent. "
                        "规则: 被剪模块 + 其下游必训, 上游可冻结. "
                        "例: 'pts_bbox_head' (P1/P3/P9), 'seg_head' (seg_mlp 剪枝), "
                        "'pts_bbox_head,seg_head' (联合剪枝)")
    p.add_argument("--skip-aux-heads", action="store_true",
                   help="训练时跳过 seg/motion/occ/planning head 的 forward "
                        "(显著省激活内存, 适合 P1/P3/P9 剪枝 track-only 微调). "
                        "不影响 state_dict, 权重依然保存. "
                        "联合剪枝涉及 seg_head 时不应开启.")
    p.add_argument("--queue-length", type=int, default=None,
                   help="覆盖配置的 queue_length (时序帧数). 默认继承 cfg. "
                        "设为 1 可大幅省激活内存 (去除 temporal context). "
                        "注意: queue=1 与原训练目标略有差异 (无时序监督), "
                        "但对短期微调 (3 epoch) 影响有限.")
    p.add_argument("--work-dir", default=None,
                   help="工作目录 (默认 work_dirs/finetune_<prune_config名>)")
    p.add_argument("--save-pruned-ckpt", default=None,
                   help="微调前先保存剪枝 checkpoint 的路径")
    p.add_argument("--no-validate", action="store_true",
                   help="训练中不做 validation")
    p.add_argument("--data-subset", type=float, default=None,
                   help="训练集采样比例 (0.1 = 10%%, 用于 pilot)")
    p.add_argument("--dry-run", action="store_true",
                   help="只加载 + 剪枝 + 打印统计, 不启动训练 (冒烟验证用)")
    p.add_argument("--gpu-ids", type=int, nargs="+", default=[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--launcher", choices=["none", "pytorch", "slurm"],
                   default="none")
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    args = p.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


# ---------------------------------------------------------------------------
# 剪枝
# ---------------------------------------------------------------------------

def apply_pruning(
    model_multi_agents: MultiAgent,
    prune_config_path: str,
    cfg,
    calib_batches: int = 16,
) -> dict:
    """对 MultiAgent.model_ego_agent 应用剪枝配置。

    返回 prune_config 字典 (带填充默认值)。
    """
    from projects.mmdet3d_plugin.univ2x.pruning.prune_univ2x import apply_prune_config

    with open(prune_config_path, "r") as f:
        prune_config = json.load(f)

    # 填充默认值 (复用 prune_and_eval.py 的逻辑, 内联以减耦合)
    locked = prune_config.setdefault("locked", {})
    locked.setdefault("importance_criterion", "l1_norm")
    locked.setdefault("pruning_granularity", "local")
    locked.setdefault("iterative_steps", 5)
    locked.setdefault("round_to", 8)
    prune_config.setdefault("encoder", {})
    prune_config.setdefault("decoder", {})
    prune_config.setdefault("heads", {})
    prune_config.setdefault("constraints", {
        "skip_layers": ["sampling_offsets", "attention_weights"],
        "min_channels": 64,
        "channel_alignment": 8,
    })

    criterion = locked["importance_criterion"]
    dataloader = None
    if criterion in ("taylor", "hessian"):
        from projects.mmdet3d_plugin.datasets.builder import build_dataloader
        dataset = build_dataset(cfg.data.train)
        dataloader = build_dataloader(
            dataset, samples_per_gpu=1,
            workers_per_gpu=0,
            dist=False, shuffle=True,
            nonshuffler_sampler=cfg.data.get("nonshuffler_sampler"),
        )
        print(f"[prune] 将为 {criterion} 收集 {calib_batches} batch 梯度")

    ego_agent = model_multi_agents.model_ego_agent
    params_before = sum(p.numel() for p in ego_agent.parameters())

    print(f"[prune] 应用剪枝: criterion={criterion}, "
          f"encoder={prune_config['encoder']}, decoder={prune_config['decoder']}")

    apply_prune_config(ego_agent, prune_config, dataloader=dataloader)

    params_after = sum(p.numel() for p in ego_agent.parameters())
    print(f"[prune] 参数: {params_before:,d} -> {params_after:,d} "
          f"(-{(1-params_after/params_before)*100:.2f}%)")

    # 清除收集梯度 (避免影响后续训练)
    ego_agent.zero_grad(set_to_none=True)
    return prune_config


# ---------------------------------------------------------------------------
# 冻结策略: 基于 --train-modules 白名单
# ---------------------------------------------------------------------------

def apply_train_modules(
    model_multi_agents: MultiAgent,
    train_modules_str: str,
) -> dict:
    """根据 --train-modules 白名单设置 requires_grad。

    规则:
        - 全模型默认冻结
        - other_agent (V2X 对端): **永远冻结** (V2X peers 不在本工作剪枝)
        - ego_agent 的直接子模块: 名字在 train_modules 白名单里才解冻
        - 'all' 关键字: 解冻整个 ego_agent (内存允许时, 最保守不过度冻结)

    Args:
        model_multi_agents: MultiAgent 包装的顶层模型
        train_modules_str: 逗号分隔的模块名, 或 'all'

    Returns:
        dict: {"allowed": list[str], "trainable_params": int, "frozen_params": int}
    """
    allowed = [m.strip() for m in train_modules_str.split(",") if m.strip()]

    # Step 1: 全冻
    for p in model_multi_agents.parameters():
        p.requires_grad = False

    # Step 2: other_agents 永久冻结 (再次确认, 不碰)
    other_names = getattr(model_multi_agents, "other_agent_names", [])
    other_n = 0
    for name in other_names:
        mod = getattr(model_multi_agents, name, None)
        if mod is None:
            continue
        for p in mod.parameters():
            p.requires_grad = False
            other_n += p.numel()

    # Step 3: 按白名单解冻 ego_agent 的子模块
    ego = model_multi_agents.model_ego_agent
    trainable_n = 0

    if "all" in allowed:
        for p in ego.parameters():
            p.requires_grad = True
            trainable_n += p.numel()
        unfrozen_modules = ["<ego_agent:*>"]
    else:
        unfrozen_modules = []
        for mod_name in allowed:
            # 名字匹配 ego_agent 的直接子属性
            if not hasattr(ego, mod_name):
                print(f"[train-modules] ⚠ ego_agent 没有子模块 '{mod_name}', 跳过")
                continue
            sub_mod = getattr(ego, mod_name)
            for p in sub_mod.parameters():
                p.requires_grad = True
                trainable_n += p.numel()
            unfrozen_modules.append(mod_name)

    total_n = sum(p.numel() for p in model_multi_agents.parameters())
    frozen_n = total_n - trainable_n

    print(f"[train-modules] allowed={allowed}")
    print(f"[train-modules] 解冻: {unfrozen_modules}")
    print(f"[train-modules] 可训: {trainable_n/1e6:.2f}M / {total_n/1e6:.2f}M "
          f"({100*trainable_n/max(total_n,1):.1f}%)")
    print(f"[train-modules]   - other_agents 冻结: {other_n/1e6:.2f}M")
    print(f"[train-modules]   - ego_agent 被冻结部分: "
          f"{(frozen_n-other_n)/1e6:.2f}M")

    return {
        "allowed": allowed,
        "unfrozen_modules": unfrozen_modules,
        "trainable_params": trainable_n,
        "frozen_params": frozen_n,
    }


# ---------------------------------------------------------------------------
# 配置调整
# ---------------------------------------------------------------------------

def adjust_training_cfg(cfg, args: argparse.Namespace) -> None:
    """根据 CLI 参数修改 cfg 的训练超参。"""
    # queue_length 覆盖 (在 lr/epochs 之前, 因为它会影响 dataset 构建)
    if args.queue_length is not None:
        orig_ql = None
        # ego_agent
        if "model_ego_agent" in cfg and "queue_length" in cfg.model_ego_agent:
            orig_ql = cfg.model_ego_agent.queue_length
            cfg.model_ego_agent.queue_length = args.queue_length
        # other_agent_*
        for key in list(cfg.keys()):
            if key.startswith("model_other_agent") and \
                    "queue_length" in cfg[key]:
                cfg[key].queue_length = args.queue_length
        # data.train
        if "data" in cfg and "train" in cfg.data and \
                "queue_length" in cfg.data.train:
            cfg.data.train.queue_length = args.queue_length
        print(f"[cfg] queue_length: {orig_ql} -> {args.queue_length} "
              f"(时序帧数减少, 激活内存大降)")

    # lr 缩放
    orig_lr = cfg.optimizer["lr"]
    new_lr = orig_lr * args.lr_scale
    cfg.optimizer["lr"] = new_lr
    print(f"[cfg] lr: {orig_lr} -> {new_lr} (scale={args.lr_scale})")

    # epoch 数
    orig_epochs = cfg.get("total_epochs", None)
    cfg.total_epochs = args.epochs
    if "runner" in cfg and isinstance(cfg.runner, dict):
        cfg.runner["max_epochs"] = args.epochs
    print(f"[cfg] epochs: {orig_epochs} -> {args.epochs}")

    # warmup
    if args.warmup_iters is not None and "lr_config" in cfg:
        cfg.lr_config["warmup_iters"] = args.warmup_iters
        print(f"[cfg] warmup_iters: -> {args.warmup_iters}")

    # data subset (pilot 用)
    if args.data_subset is not None and 0 < args.data_subset < 1.0:
        # V2X-Seq-SPD dataset 假设有 ann_file; 这里先打 flag, 由 dataset 读取决定
        print(f"[cfg] 数据子集: {args.data_subset*100:.0f}%%")
        # 简单做法: 减少 workflow 中训练迭代数 (更彻底的做法见 sampler)
        # 这里不做实际裁剪 - 用户可通过 cfg-options 覆盖

    # 保存间隔: 至少每 epoch 存一次, 以便观察
    if "checkpoint_config" in cfg:
        cfg.checkpoint_config["interval"] = 1

    # evaluation 关闭或降频 (微调主要看 loss)
    if args.no_validate and "evaluation" in cfg:
        cfg.evaluation["interval"] = args.epochs + 1  # 永不触发


# ---------------------------------------------------------------------------
# 主流程 (复制 train.py 的骨架, 中间插入剪枝)
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    # 导入插件 (复制 train.py 的逻辑)
    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib
        if hasattr(cfg, "plugin_dir"):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir).split("/")
            _module_path = ".".join(_module_dir)
            importlib.import_module(_module_path)
        from projects.mmdet3d_plugin.univ2x.apis.train import custom_train_model
    else:
        from projects.mmdet3d_plugin.univ2x.apis.train import custom_train_model

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        prune_name = osp.splitext(osp.basename(args.prune_config))[0]
        cfg.work_dir = osp.join("./work_dirs", f"finetune_{prune_name}")

    cfg.gpu_ids = args.gpu_ids
    if digit_version(TORCH_VERSION) == digit_version("1.8.1") and \
            cfg.optimizer["type"] == "AdamW":
        cfg.optimizer["type"] = "AdamW2"

    # 分布式
    distributed = args.launcher != "none"
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # 调整训练超参 (在 build 之前, lr 会进到 optimizer 构造)
    adjust_training_cfg(cfg, args)

    # work_dir + logger
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"finetune_{timestamp}.log")
    log_name = "mmseg" if cfg.model_ego_agent.type == "EncoderDecoder3D" else "mmdet"
    mm_logger = get_root_logger(log_file=log_file, log_level=cfg.log_level,
                                name=log_name)

    # 环境信息
    meta = dict()
    env_info = "\n".join([f"{k}: {v}" for k, v in collect_env().items()])
    mm_logger.info("Env:\n%s\n%s", "-" * 60, env_info)
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text
    meta["seed"] = args.seed
    meta["exp_name"] = osp.basename(args.config)
    meta["prune_config"] = args.prune_config

    set_random_seed(args.seed, deterministic=False)
    cfg.seed = args.seed

    # 构建 other_agent
    other_agent_names = [k for k in cfg.keys() if "model_other_agent" in k]
    model_other_agents = {}
    for other_agent_name in other_agent_names:
        model_other_agent = build_model(
            cfg.get(other_agent_name),
            train_cfg=cfg.get("train_cfg"),
            test_cfg=cfg.get("test_cfg"),
        )
        model_other_agent.init_weights()
        load_from = cfg.get(other_agent_name).get("load_from")
        if load_from:
            load_checkpoint(model_other_agent, load_from, map_location="cpu",
                            revise_keys=[(r"^model_ego_agent\.", "")])
        model_other_agents[other_agent_name] = model_other_agent

    # 构建 ego_agent
    model_ego_agent = build_model(
        cfg.model_ego_agent,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    model_ego_agent.init_weights()
    load_from = cfg.model_ego_agent.get("load_from")
    if load_from:
        load_checkpoint(model_ego_agent, load_from, map_location="cpu",
                        revise_keys=[(r"^model_ego_agent\.", "")])

    # 包装 MultiAgent
    model_multi_agents = MultiAgent(model_ego_agent, model_other_agents)

    # 加载完整 stg2 checkpoint (覆盖上面 ego load_from)
    mm_logger.info("加载 checkpoint: %s", args.checkpoint)
    checkpoint = load_checkpoint(model_multi_agents, args.checkpoint,
                                 map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model_multi_agents.model_ego_agent.CLASSES = checkpoint["meta"]["CLASSES"]

    # ==========================================================================
    # 关键新增: 应用剪枝
    # ==========================================================================
    mm_logger.info("=" * 60)
    mm_logger.info("应用剪枝: %s", args.prune_config)
    mm_logger.info("=" * 60)
    model_multi_agents = model_multi_agents.cuda()
    prune_config = apply_pruning(model_multi_agents, args.prune_config, cfg)

    # 保存剪枝 checkpoint (微调前)
    if args.save_pruned_ckpt:
        torch.save({
            "state_dict": model_multi_agents.state_dict(),
            "meta": {
                "prune_config": prune_config,
                "note": "pruned-only, not finetuned",
            },
        }, args.save_pruned_ckpt)
        mm_logger.info("剪枝前 checkpoint 保存到: %s", args.save_pruned_ckpt)

    # 冻结策略: 基于 --train-modules 白名单
    mm_logger.info("=" * 60)
    mm_logger.info("冻结策略: --train-modules=%s", args.train_modules)
    mm_logger.info("=" * 60)
    freeze_info = apply_train_modules(model_multi_agents, args.train_modules)
    meta["train_modules"] = freeze_info

    # 统计可训练参数
    n_trainable = sum(p.numel() for p in model_multi_agents.parameters()
                      if p.requires_grad)
    n_total = sum(p.numel() for p in model_multi_agents.parameters())
    mm_logger.info("可训练参数: %d / %d (%.1f%%)",
                   n_trainable, n_total, 100 * n_trainable / max(n_total, 1))

    # ==========================================================================
    # 构建数据集 + 启动训练
    # ==========================================================================
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        if "dataset" in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE if hasattr(datasets[0], "PALETTE") else None,
            prune_config=prune_config,
        )
    model_multi_agents.model_ego_agent.CLASSES = datasets[0].CLASSES

    # 应用 skip_aux_heads (省激活内存)
    if args.skip_aux_heads:
        model_multi_agents.model_ego_agent._skip_aux_heads_during_train = True
        mm_logger.info("[skip-aux-heads] 训练时将跳过 seg/motion/occ/planning "
                       "head forward, 只算 track loss")
        meta["skip_aux_heads"] = True

    if args.dry_run:
        mm_logger.info("=" * 60)
        mm_logger.info("DRY RUN 完成 — 剪枝后模型已在 GPU, 未启动训练")
        mm_logger.info("  模型总参数: %d (%.2fM)", n_total, n_total / 1e6)
        mm_logger.info("  可训练参数: %d (%.2fM)", n_trainable, n_trainable / 1e6)
        mm_logger.info("  训练数据集样本数: %d", len(datasets[0]))
        mm_logger.info("  work_dir: %s", cfg.work_dir)
        mm_logger.info("  下一步: 去掉 --dry-run 启动真实微调")
        mm_logger.info("=" * 60)
        return

    mm_logger.info("启动微调训练: epochs=%d, lr=%f, work_dir=%s",
                   args.epochs, cfg.optimizer["lr"], cfg.work_dir)

    custom_train_model(
        model_multi_agents,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )

    mm_logger.info("=" * 60)
    mm_logger.info("微调完成, checkpoint 位于: %s", cfg.work_dir)
    mm_logger.info("下一步: 使用 tools/test_with_pruning.py 评估 AMOTA")
    mm_logger.info("=" * 60)


if __name__ == "__main__":
    main()
