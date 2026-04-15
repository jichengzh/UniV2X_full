#!/usr/bin/env python3
"""
Phase 0 实验自动化脚本: 锁定维度预选 + 搜索维度敏感度分析

用法:
  # Phase 0-A: 锁定 P4/P5/P6
  python tools/pruning_sensitivity_analysis.py \
      --mode lock-dims \
      --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
      --checkpoint work_dirs/latest.pth \
      --output-dir work_dirs/phase0

  # Phase 0-B: 搜索维度敏感度
  python tools/pruning_sensitivity_analysis.py \
      --mode sensitivity \
      --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
      --checkpoint work_dirs/latest.pth \
      --locked-config work_dirs/phase0/locked_config.json \
      --output-dir work_dirs/phase0

  # 快速验证模式 (仅检查脚本逻辑, 不做真实评估)
  python tools/pruning_sensitivity_analysis.py \
      --mode lock-dims \
      --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
      --checkpoint work_dirs/latest.pth \
      --output-dir work_dirs/phase0_debug \
      --fast
"""
import argparse
import copy
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase0")


# ============================================================
# 工具函数
# ============================================================

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(data, path):
    """原子写入 JSON（先写 .tmp 再 rename）"""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, str(path))


def load_json(path):
    """加载 JSON，不存在返回 None"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_model_fresh(cfg_path, ckpt_path):
    """每次实验加载全新模型（避免剪枝状态污染）

    支持两种 config 格式:
      1. 单 agent: cfg.model (传统 mmdet3d)
      2. multi-agent: cfg.model_ego_agent + cfg.model_other_agent_* (UniV2X)

    返回:
      (model, cfg): model 是可直接剪枝的 nn.Module
        - 单 agent: 直接返回 model
        - multi-agent: 返回 MultiAgent wrapper, 但剪枝作用于 model_ego_agent
    """
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model
    import importlib

    cfg = Config.fromfile(cfg_path)

    # 导入插件
    if hasattr(cfg, "plugin") and cfg.plugin:
        plugin_dir = cfg.get("plugin_dir", "projects/mmdet3d_plugin/")
        _module_path = plugin_dir.rstrip("/").replace("/", ".")
        importlib.import_module(_module_path)

    # 判断 config 格式
    is_multi_agent = "model_ego_agent" in cfg

    if is_multi_agent:
        from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent

        # 构建 other_agent 模型
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

        # 构建 ego_agent 模型
        cfg.model_ego_agent.train_cfg = None
        cfg.model_ego_agent.pretrained = None
        model_ego = build_model(cfg.model_ego_agent, test_cfg=cfg.get("test_cfg"))
        load_from = cfg.model_ego_agent.get("load_from")
        if load_from:
            load_checkpoint(model_ego, load_from, map_location="cpu",
                            revise_keys=[(r"^model_ego_agent\.", "")])

        # 包装为 MultiAgent
        model = MultiAgent(model_ego, model_other_agents)

        # 加载完整 checkpoint (覆盖上面的 load_from)
        checkpoint = load_checkpoint(model, ckpt_path, map_location="cpu")
        if "CLASSES" in checkpoint.get("meta", {}):
            model.model_ego_agent.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        # 单 agent 路径 (兼容性保留)
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        load_checkpoint(model, ckpt_path, map_location="cpu")

    model.cuda().eval()
    return model, cfg


def get_prune_target(model):
    """从模型中提取剪枝目标 (处理 MultiAgent wrapper)

    返回实际承载 BEVFormer/FFN 权重的子模型。
    """
    from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent
    if isinstance(model, MultiAgent):
        return model.model_ego_agent
    return model


def evaluate_model(model, cfg, max_samples=None):
    """评估模型 AMOTA

    Args:
        model: 剪枝后模型 (MultiAgent 或单 agent)
        cfg: mmdet3d Config
        max_samples: 最大评估样本数 (None=全量)

    Returns:
        float: AMOTA 分数
    """
    from mmcv.parallel import MMDataParallel
    from mmdet3d.datasets import build_dataset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from mmdet.datasets import replace_ImageToTensor

    cfg = copy.deepcopy(cfg)
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    dataset = build_dataset(cfg.data.test)

    data_loader = build_dataloader(
        dataset, samples_per_gpu=1,
        workers_per_gpu=cfg.data.get("workers_per_gpu", 2),
        dist=False, shuffle=False,
        nonshuffler_sampler=cfg.data.get("nonshuffler_sampler"),
    )

    # 优先使用 univ2x 的 custom test
    try:
        from projects.mmdet3d_plugin.univ2x.apis.test import custom_multi_gpu_test
        from mmcv.parallel import MMDataParallel
        # MultiAgent 需要 MMDataParallel 包装
        model_parallel = MMDataParallel(model, device_ids=[0])
        outputs = custom_multi_gpu_test(
            model_parallel, data_loader,
            max_samples=max_samples,
        ) if "max_samples" in custom_multi_gpu_test.__code__.co_varnames else \
            custom_multi_gpu_test(model_parallel, data_loader)
    except ImportError:
        # 回退到 mmdet3d 标准 single_gpu_test
        from mmdet3d.apis import single_gpu_test
        model_parallel = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model_parallel, data_loader)

    # 限制样本数 (手动截断, 如果测试函数不原生支持)
    if max_samples is not None and max_samples > 0 and len(outputs) > max_samples:
        outputs = outputs[:max_samples]

    # 评估
    try:
        metrics = dataset.evaluate(outputs)
        amota = metrics.get("AMOTA", metrics.get("amota", 0.0))
        if hasattr(amota, "item"):
            amota = amota.item()
    except Exception as e:
        logger.warning("评估失败: %s, 返回 0.0", e)
        amota = 0.0

    return float(amota)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_single_experiment(args, prune_cfg, exp_name):
    """运行单次剪枝 + 评估实验

    Returns:
        dict: {amota, params, elapsed_sec, ...}
    """
    logger.info("  [%s] 加载模型...", exp_name)
    t0 = time.time()
    model, cfg = load_model_fresh(args.config, args.checkpoint)

    # 获取实际剪枝目标 (处理 MultiAgent wrapper)
    prune_target = get_prune_target(model)
    params_before = count_params(prune_target)

    # 准备 dataloader (Taylor/Hessian 需要)
    dataloader = None
    criterion = prune_cfg.get("locked", {}).get("importance_criterion", "taylor")
    if criterion in ("taylor", "hessian"):
        try:
            from mmdet3d.datasets import build_dataset
            from projects.mmdet3d_plugin.datasets.builder import build_dataloader
            ds = build_dataset(cfg.data.train)
            dataloader = build_dataloader(
                ds, samples_per_gpu=1,
                workers_per_gpu=2, dist=False, shuffle=True,
            )
        except Exception as e:
            logger.warning("  构建训练 dataloader 失败: %s, 降级到 l1_norm", e)
            prune_cfg["locked"]["importance_criterion"] = "l1_norm"

    # 剪枝 (作用于 ego_agent, 不是 MultiAgent wrapper)
    from projects.mmdet3d_plugin.univ2x.pruning.prune_univ2x import apply_prune_config
    apply_prune_config(prune_target, prune_cfg, dataloader=dataloader)
    params_after = count_params(prune_target)

    # 评估
    if args.fast:
        # fast 模式: 不做真实评估, 用参数量变化作为代理指标
        amota = -1.0  # 占位
        logger.info("  [%s] fast 模式, 跳过评估 (params: %d -> %d)", exp_name, params_before, params_after)
    else:
        logger.info("  [%s] 评估中...", exp_name)
        amota = evaluate_model(model, cfg, max_samples=args.max_samples)
        logger.info("  [%s] AMOTA=%.4f, params: %d->%d", exp_name, amota, params_before, params_after)

    elapsed = time.time() - t0

    # 释放
    del model
    if dataloader is not None:
        del dataloader
    torch.cuda.empty_cache()

    return {
        "amota": amota,
        "params_before": params_before,
        "params_after": params_after,
        "param_reduction": round(1.0 - params_after / params_before, 4),
        "elapsed_sec": round(elapsed, 1),
        "timestamp": get_timestamp(),
    }


# ============================================================
# Phase 0-A: 锁定维度预选
# ============================================================

def run_lock_dims(args):
    """Phase 0-A: 依次锁定 P4, P5, P6"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "lock_dims_checkpoint.json"
    locked_path = output_dir / "locked_config.json"

    results = load_json(ckpt_path) or {
        "meta": {"start_time": get_timestamp(), "config": args.config},
        "set1_criterion": {},
        "set2_granularity": {},
        "set3_iterative_steps": {},
        "locked": {"round_to": 8},
    }

    # ── 实验集 1: P4 重要性准则 ──
    logger.info("=" * 60)
    logger.info("实验集 1: P4 重要性准则对比 (固定 FFN 30%%)")
    logger.info("=" * 60)

    criteria = ["l1_norm", "taylor", "fpgm"]
    # hessian 容易 OOM, 放最后, 失败可跳过
    if not args.fast:
        criteria.append("hessian")

    for criterion in criteria:
        key = f"criterion_{criterion}"
        if key in results["set1_criterion"]:
            logger.info("  跳过: %s (已完成)", criterion)
            continue

        logger.info("  测试: %s", criterion)
        prune_cfg = {
            "locked": {
                "importance_criterion": criterion,
                "pruning_granularity": "local",
                "iterative_steps": 5,
                "round_to": 8,
            },
            "encoder": {"ffn_mid_ratio": 0.7},
            "decoder": {"ffn_mid_ratio": 0.7, "num_layers": 6},
            "heads": {"head_mid_ratio": 1.0},
        }

        try:
            result = run_single_experiment(args, prune_cfg, criterion)
            result["criterion"] = criterion
            results["set1_criterion"][key] = result
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("  %s OOM, 跳过", criterion)
                results["set1_criterion"][key] = {
                    "criterion": criterion, "amota": -1.0, "error": "OOM",
                }
                torch.cuda.empty_cache()
            else:
                raise

        save_json(results, ckpt_path)

    # 选 P4 最优
    valid = {k: v for k, v in results["set1_criterion"].items() if v.get("amota", -1) > 0}
    if valid:
        best_key = max(valid, key=lambda k: valid[k]["amota"])
        best_criterion = valid[best_key]["criterion"]
    else:
        best_criterion = "l1_norm"  # fallback
    results["locked"]["importance_criterion"] = best_criterion
    logger.info("P4 锁定: %s", best_criterion)

    # ── 实验集 2: P5 剪枝粒度 ──
    logger.info("=" * 60)
    logger.info("实验集 2: P5 剪枝粒度对比 (P4=%s)", best_criterion)
    logger.info("=" * 60)

    for granularity in ["global", "local", "isomorphic"]:
        key = f"granularity_{granularity}"
        if key in results["set2_granularity"]:
            logger.info("  跳过: %s (已完成)", granularity)
            continue

        logger.info("  测试: %s", granularity)
        prune_cfg = {
            "locked": {
                "importance_criterion": best_criterion,
                "pruning_granularity": granularity,
                "iterative_steps": 5,
                "round_to": 8,
            },
            "encoder": {"ffn_mid_ratio": 0.7},
            "decoder": {"ffn_mid_ratio": 0.7, "num_layers": 6},
            "heads": {"head_mid_ratio": 1.0},
        }

        result = run_single_experiment(args, prune_cfg, granularity)
        result["granularity"] = granularity
        results["set2_granularity"][key] = result
        save_json(results, ckpt_path)

    # 选 P5 最优
    valid = results["set2_granularity"]
    best_key = max(valid, key=lambda k: valid[k]["amota"])
    best_granularity = valid[best_key]["granularity"]
    results["locked"]["pruning_granularity"] = best_granularity
    logger.info("P5 锁定: %s", best_granularity)

    # ── 实验集 3: P6 迭代步数 ──
    logger.info("=" * 60)
    logger.info("实验集 3: P6 迭代步数 (P4=%s, P5=%s)", best_criterion, best_granularity)
    logger.info("=" * 60)

    for steps in [1, 3, 5, 10]:
        key = f"steps_{steps}"
        if key in results["set3_iterative_steps"]:
            logger.info("  跳过: steps=%d (已完成)", steps)
            continue

        logger.info("  测试: steps=%d", steps)
        prune_cfg = {
            "locked": {
                "importance_criterion": best_criterion,
                "pruning_granularity": best_granularity,
                "iterative_steps": steps,
                "round_to": 8,
            },
            "encoder": {"ffn_mid_ratio": 0.6},  # 40% 剪枝, 放大差异
            "decoder": {"ffn_mid_ratio": 0.6, "num_layers": 6},
            "heads": {"head_mid_ratio": 1.0},
        }

        result = run_single_experiment(args, prune_cfg, f"steps_{steps}")
        result["iterative_steps"] = steps
        results["set3_iterative_steps"][key] = result
        save_json(results, ckpt_path)

    # 选 P6: 如果 steps=1 与 steps=5 差距 < 0.002, 选 1 以加速
    amota_1 = results["set3_iterative_steps"].get("steps_1", {}).get("amota", 0)
    amota_5 = results["set3_iterative_steps"].get("steps_5", {}).get("amota", 0)
    if abs(amota_5 - amota_1) < 0.002:
        best_steps = 1
        logger.info("P6: steps=1 vs 5 差距 < 0.002, 选 1 加速")
    else:
        best_key = max(results["set3_iterative_steps"],
                       key=lambda k: results["set3_iterative_steps"][k]["amota"])
        best_steps = results["set3_iterative_steps"][best_key]["iterative_steps"]
    results["locked"]["iterative_steps"] = best_steps
    logger.info("P6 锁定: %d", best_steps)

    # 保存 locked_config.json
    locked_config = {
        "version": "1.2",
        "phase": "Phase 0-A",
        "timestamp": get_timestamp(),
        "locked": results["locked"],
        "evidence": {
            "set1_criterion": results["set1_criterion"],
            "set2_granularity": results["set2_granularity"],
            "set3_iterative_steps": results["set3_iterative_steps"],
        },
    }
    save_json(locked_config, locked_path)
    logger.info("locked_config.json 已保存: %s", locked_path)
    logger.info("锁定结果: %s", json.dumps(results["locked"], indent=2))

    # 可视化
    try:
        _plot_lock_dims(results, output_dir)
    except Exception as e:
        logger.warning("可视化失败: %s", e)

    return results["locked"]


# ============================================================
# Phase 0-B: 搜索维度敏感度分析
# ============================================================

def run_sensitivity(args):
    """Phase 0-B: P1/P2/P8/P9 敏感度 + B1×B2 交互"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载锁定配置
    locked_path = args.locked_config or str(output_dir / "locked_config.json")
    locked_data = load_json(locked_path)
    if locked_data is None:
        logger.error("locked_config.json 不存在: %s, 请先执行 --mode lock-dims", locked_path)
        sys.exit(1)
    locked = locked_data["locked"]
    logger.info("锁定配置: %s", json.dumps(locked))

    ckpt_path = output_dir / "sensitivity_checkpoint.json"
    report_path = output_dir / "pruning_sensitivity_report.json"

    results = load_json(ckpt_path) or {
        "meta": {"start_time": get_timestamp()},
        "baseline_amota": None,
        "set4_ffn_per_layer": {},
        "set5_attn_proj": {},
        "set6_head_pruning": {},
        "set7_decoder_layers": {},
        "set8_prune_quant_interaction": {},
    }

    # 基线
    if results["baseline_amota"] is None:
        logger.info("获取基线 AMOTA (未剪枝)...")
        if args.fast:
            results["baseline_amota"] = 0.338  # PyTorch 基线
            logger.info("fast 模式, 使用基线 0.338")
        else:
            model, cfg = load_model_fresh(args.config, args.checkpoint)
            results["baseline_amota"] = evaluate_model(model, cfg, args.max_samples)
            logger.info("基线 AMOTA: %.4f", results["baseline_amota"])
            del model
            torch.cuda.empty_cache()
        save_json(results, ckpt_path)

    baseline = results["baseline_amota"]

    # ── 实验集 4: 逐层 FFN 敏感度 (P1) ──
    logger.info("=" * 60)
    logger.info("实验集 4: 逐层 FFN 敏感度 (P1)")
    logger.info("=" * 60)

    ffn_layers = []
    for i in range(6):
        ffn_layers.append(f"encoder.layers.{i}.ffns")
    for i in range(6):
        ffn_layers.append(f"decoder.layers.{i}.ffns")

    ratios_to_test = [0.8, 0.6, 0.4]  # 保留比例: 80%, 60%, 40%

    for layer_name in ffn_layers:
        for ratio in ratios_to_test:
            pct = int((1.0 - ratio) * 100)
            key = f"{layer_name}_ratio{ratio}"

            if key in results["set4_ffn_per_layer"]:
                logger.info("  跳过: %s %d%% (已完成)", layer_name, pct)
                continue

            logger.info("  测试: %s 剪枝 %d%%", layer_name, pct)

            # 仅剪枝目标层: 全局 ffn_mid_ratio=1.0, per_layer_override 指定目标层
            prune_cfg = {
                "locked": locked,
                "encoder": {"ffn_mid_ratio": 1.0},
                "decoder": {"ffn_mid_ratio": 1.0, "num_layers": 6},
                "heads": {"head_mid_ratio": 1.0},
                "per_layer_override": {
                    layer_name: {"ffn_mid_ratio": ratio},
                },
            }

            result = run_single_experiment(args, prune_cfg, f"{layer_name}_{pct}pct")
            result["layer"] = layer_name
            result["ffn_mid_ratio"] = ratio
            result["prune_pct"] = pct
            result["delta_amota"] = round(result["amota"] - baseline, 4) if result["amota"] > 0 else None
            results["set4_ffn_per_layer"][key] = result
            save_json(results, ckpt_path)

    # 分类
    _classify_ffn_sensitivity(results)
    save_json(results, ckpt_path)

    # ── 实验集 5: 注意力投影敏感度 (P2) ──
    logger.info("=" * 60)
    logger.info("实验集 5: 注意力投影敏感度 (P2)")
    logger.info("=" * 60)

    for attn_ratio in [0.1, 0.2]:
        key = f"attn_proj_{attn_ratio}"
        if key in results["set5_attn_proj"]:
            logger.info("  跳过: attn_proj_ratio=%.1f (已完成)", attn_ratio)
            continue

        logger.info("  测试: attn_proj_ratio=%.1f", attn_ratio)
        prune_cfg = {
            "locked": locked,
            "encoder": {"ffn_mid_ratio": 1.0, "attn_proj_ratio": attn_ratio},
            "decoder": {"ffn_mid_ratio": 1.0, "attn_proj_ratio": attn_ratio, "num_layers": 6},
            "heads": {"head_mid_ratio": 1.0},
        }
        result = run_single_experiment(args, prune_cfg, f"attn_{attn_ratio}")
        result["attn_proj_ratio"] = attn_ratio
        result["delta_amota"] = round(result["amota"] - baseline, 4) if result["amota"] > 0 else None
        results["set5_attn_proj"][key] = result
        save_json(results, ckpt_path)

    # ── 实验集 6: 注意力头剪枝 (P8) ──
    logger.info("=" * 60)
    logger.info("实验集 6: 注意力头剪枝 (P8)")
    logger.info("=" * 60)

    for head_ratio in [0.0, 0.125]:
        key = f"head_prune_{head_ratio}"
        if key in results["set6_head_pruning"]:
            logger.info("  跳过: head_pruning_ratio=%.3f (已完成)", head_ratio)
            continue

        logger.info("  测试: head_pruning_ratio=%.3f", head_ratio)
        prune_cfg = {
            "locked": locked,
            "encoder": {"ffn_mid_ratio": 1.0, "head_pruning_ratio": head_ratio},
            "decoder": {"ffn_mid_ratio": 1.0, "head_pruning_ratio": head_ratio, "num_layers": 6},
            "heads": {"head_mid_ratio": 1.0},
        }
        result = run_single_experiment(args, prune_cfg, f"head_{head_ratio}")
        result["head_pruning_ratio"] = head_ratio
        result["delta_amota"] = round(result["amota"] - baseline, 4) if result["amota"] > 0 else None
        results["set6_head_pruning"][key] = result
        save_json(results, ckpt_path)

    # ── 实验集 7: 解码器层数 (P9) ──
    logger.info("=" * 60)
    logger.info("实验集 7: 解码器层数 (P9)")
    logger.info("=" * 60)

    for num_layers in [5, 6]:
        key = f"decoder_layers_{num_layers}"
        if key in results["set7_decoder_layers"]:
            logger.info("  跳过: num_layers=%d (已完成)", num_layers)
            continue

        logger.info("  测试: num_layers=%d", num_layers)
        prune_cfg = {
            "locked": locked,
            "encoder": {"ffn_mid_ratio": 1.0},
            "decoder": {"ffn_mid_ratio": 1.0, "num_layers": num_layers},
            "heads": {"head_mid_ratio": 1.0},
        }
        result = run_single_experiment(args, prune_cfg, f"layers_{num_layers}")
        result["num_layers"] = num_layers
        result["delta_amota"] = round(result["amota"] - baseline, 4) if result["amota"] > 0 else None
        results["set7_decoder_layers"][key] = result
        save_json(results, ckpt_path)

    # ── 实验集 8: B1×B2 交互项 ──
    logger.info("=" * 60)
    logger.info("实验集 8: B1×B2 交互 (剪枝 × 量化)")
    logger.info("=" * 60)

    repr_configs = {
        "conservative": {
            "encoder": {"ffn_mid_ratio": 0.8},
            "decoder": {"ffn_mid_ratio": 0.8, "num_layers": 6},
            "heads": {"head_mid_ratio": 1.0},
        },
        "moderate": {
            "encoder": {"ffn_mid_ratio": 0.6, "attn_proj_ratio": 0.1},
            "decoder": {"ffn_mid_ratio": 0.6, "attn_proj_ratio": 0.1, "num_layers": 6},
            "heads": {"head_mid_ratio": 0.7},
        },
        "aggressive": {
            "encoder": {"ffn_mid_ratio": 0.4, "attn_proj_ratio": 0.2, "head_pruning_ratio": 0.125},
            "decoder": {"ffn_mid_ratio": 0.4, "attn_proj_ratio": 0.2, "head_pruning_ratio": 0.125, "num_layers": 5},
            "heads": {"head_mid_ratio": 0.5},
        },
    }

    for config_name, repr_cfg in repr_configs.items():
        # (a) prune only
        key_prune = f"{config_name}_prune_only"
        if key_prune not in results["set8_prune_quant_interaction"]:
            logger.info("  [%s] prune only...", config_name)
            prune_cfg = {**repr_cfg, "locked": locked}
            result = run_single_experiment(args, prune_cfg, f"{config_name}_prune")
            result["config"] = config_name
            result["mode"] = "prune_only"
            result["delta_amota"] = round(result["amota"] - baseline, 4) if result["amota"] > 0 else None
            results["set8_prune_quant_interaction"][key_prune] = result
            save_json(results, ckpt_path)

        # (b) quant only — 使用已知的 INT8 TRT 基线
        key_quant = f"{config_name}_quant_only"
        if key_quant not in results["set8_prune_quant_interaction"]:
            # 量化需要完整的 QuantModel pipeline, 这里使用已知基线值
            # INT8 TRT AMOTA = 0.364 (来自 1.1 可配置量化实验)
            results["set8_prune_quant_interaction"][key_quant] = {
                "config": config_name,
                "mode": "quant_only",
                "amota": 0.364,
                "delta_amota": round(0.364 - baseline, 4),
                "note": "使用 1.1 实验已知值 (INT8 TRT AMOTA=0.364)",
            }
            logger.info("  [%s] quant only: 使用已知基线 0.364", config_name)
            save_json(results, ckpt_path)

        # (c) both — 先剪枝, 再标注 (实际量化需要 TRT, 这里只做剪枝部分)
        key_both = f"{config_name}_both"
        if key_both not in results["set8_prune_quant_interaction"]:
            logger.info("  [%s] prune + quant (先剪枝, 量化效果待 Phase B1 验证)...", config_name)
            # 暂时只记录剪枝后的结果, 量化需要 TRT 管线 (Phase B1)
            results["set8_prune_quant_interaction"][key_both] = {
                "config": config_name,
                "mode": "both",
                "amota": None,
                "delta_amota": None,
                "note": "需要 Phase B1 TRT 管线完成后补充",
            }
            save_json(results, ckpt_path)

    # 计算交互效应
    _compute_interaction_effects(results, baseline)

    # 保存最终报告
    report = {
        "version": "1.2",
        "phase": "Phase 0-B",
        "timestamp": get_timestamp(),
        "baseline_amota": baseline,
        "per_layer_sensitivity": results["set4_ffn_per_layer"],
        "ffn_classification": results.get("ffn_classification", {}),
        "attn_proj_sensitivity": results["set5_attn_proj"],
        "head_pruning_impact": results["set6_head_pruning"],
        "decoder_layer_impact": results["set7_decoder_layers"],
        "prune_quant_interaction": results["set8_prune_quant_interaction"],
        "interaction_effects": results.get("interaction_effects", {}),
    }
    save_json(report, report_path)
    logger.info("报告已保存: %s", report_path)

    # 可视化
    try:
        _plot_sensitivity(results, output_dir)
    except Exception as e:
        logger.warning("可视化失败: %s", e)

    return report


# ============================================================
# 辅助分析函数
# ============================================================

def _classify_ffn_sensitivity(results):
    """按 40% 剪枝时的 delta_AMOTA 分三档"""
    classification = {}
    layer_deltas = {}

    for key, data in results["set4_ffn_per_layer"].items():
        layer = data["layer"]
        ratio = data["ffn_mid_ratio"]
        delta = data.get("delta_amota")
        if delta is None:
            continue
        if layer not in layer_deltas:
            layer_deltas[layer] = {}
        layer_deltas[layer][ratio] = delta

    for layer, deltas in layer_deltas.items():
        delta_40 = abs(deltas.get(0.6, 0))
        if delta_40 < 0.005:
            category = "safe_aggressive"
        elif delta_40 < 0.015:
            category = "moderate"
        else:
            category = "sensitive"

        classification[layer] = {
            "category": category,
            "delta_at_20pct": deltas.get(0.8),
            "delta_at_40pct": deltas.get(0.6),
            "delta_at_60pct": deltas.get(0.4),
        }

    results["ffn_classification"] = classification
    logger.info("FFN 敏感度分类:")
    for layer, info in sorted(classification.items()):
        logger.info("  %s: %s (delta@40%%=%s)", layer, info["category"], info["delta_at_40pct"])


def _compute_interaction_effects(results, baseline):
    """计算 B1×B2 交互效应"""
    effects = {}
    data = results["set8_prune_quant_interaction"]

    for config_name in ["conservative", "moderate", "aggressive"]:
        prune_key = f"{config_name}_prune_only"
        quant_key = f"{config_name}_quant_only"
        both_key = f"{config_name}_both"

        if all(k in data for k in [prune_key, quant_key, both_key]):
            delta_prune = data[prune_key].get("delta_amota", 0) or 0
            delta_quant = data[quant_key].get("delta_amota", 0) or 0
            delta_joint = data[both_key].get("delta_amota")

            if delta_joint is not None:
                expected = delta_prune + delta_quant
                interaction = delta_joint - expected

                if abs(interaction) < 0.003:
                    effect = "independent"
                elif interaction < 0:
                    effect = "cancellation"
                else:
                    effect = "amplification"

                effects[config_name] = {
                    "delta_prune": delta_prune,
                    "delta_quant": delta_quant,
                    "delta_joint": delta_joint,
                    "expected_additive": round(expected, 4),
                    "interaction": round(interaction, 4),
                    "effect_type": effect,
                }
            else:
                effects[config_name] = {
                    "delta_prune": delta_prune,
                    "delta_quant": delta_quant,
                    "delta_joint": None,
                    "note": "joint 实验待 Phase B1 完成",
                }

    results["interaction_effects"] = effects


# ============================================================
# 可视化
# ============================================================

def _plot_lock_dims(results, output_dir):
    """Phase 0-A 可视化: 准则/粒度/步数对比"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Phase 0-A: Lock Dimensions", fontsize=14)

    # P4
    ax = axes[0]
    data = results["set1_criterion"]
    names, vals = [], []
    for k in sorted(data.keys()):
        if data[k].get("amota", -1) > 0:
            names.append(data[k]["criterion"])
            vals.append(data[k]["amota"])
    if names:
        bars = ax.bar(names, vals, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(names)])
        ax.set_title("P4: Importance Criterion")
        ax.set_ylabel("AMOTA")
        best = np.argmax(vals)
        bars[best].set_edgecolor("black")
        bars[best].set_linewidth(2)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

    # P5
    ax = axes[1]
    data = results["set2_granularity"]
    names, vals = [], []
    for k in sorted(data.keys()):
        names.append(data[k]["granularity"])
        vals.append(data[k]["amota"])
    if names:
        bars = ax.bar(names, vals, color=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_title("P5: Pruning Granularity")
        ax.set_ylabel("AMOTA")
        best = np.argmax(vals)
        bars[best].set_edgecolor("black")
        bars[best].set_linewidth(2)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

    # P6
    ax = axes[2]
    data = results["set3_iterative_steps"]
    names, vals = [], []
    for k in sorted(data.keys(), key=lambda x: data[x]["iterative_steps"]):
        names.append(str(data[k]["iterative_steps"]))
        vals.append(data[k]["amota"])
    if names:
        ax.plot(names, vals, "o-", color="#4C72B0", linewidth=2, markersize=8)
        ax.set_title("P6: Iterative Steps")
        ax.set_ylabel("AMOTA")
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points",
                        xytext=(0, 10), fontsize=9, ha="center")

    plt.tight_layout()
    save_path = output_dir / "phase0a_lock_dims.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("可视化已保存: %s", save_path)


def _plot_sensitivity(results, output_dir):
    """Phase 0-B 可视化: FFN 热力图 + 交互效应"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # FFN 敏感度热力图
    fig, ax = plt.subplots(figsize=(8, 8))

    layer_names = [f"enc.{i}" for i in range(6)] + [f"dec.{i}" for i in range(6)]
    ratios = [0.8, 0.6, 0.4]
    ratio_labels = ["20%", "40%", "60%"]
    heatmap = np.zeros((len(layer_names), len(ratios)))

    ffn_data = results.get("set4_ffn_per_layer", {})
    for key, entry in ffn_data.items():
        layer = entry["layer"]
        ratio = entry["ffn_mid_ratio"]
        for idx, prefix in enumerate(layer_names):
            full = (f"encoder.layers.{idx}.ffns" if idx < 6
                    else f"decoder.layers.{idx - 6}.ffns")
            if layer == full and ratio in ratios:
                col = ratios.index(ratio)
                delta = entry.get("delta_amota", 0) or 0
                heatmap[idx, col] = delta

    im = ax.imshow(heatmap, cmap="RdYlGn", aspect="auto", vmin=-0.05, vmax=0.005)
    ax.set_xticks(range(len(ratio_labels)))
    ax.set_xticklabels(ratio_labels)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names)
    ax.set_xlabel("Pruning Ratio")
    ax.set_ylabel("FFN Layer")
    ax.set_title("Per-Layer FFN Sensitivity (delta AMOTA)")
    plt.colorbar(im, label="delta AMOTA")

    for i in range(len(layer_names)):
        for j in range(len(ratios)):
            v = heatmap[i, j]
            color = "white" if abs(v) > 0.02 else "black"
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    save_path = output_dir / "phase0b_ffn_sensitivity.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("FFN 热力图已保存: %s", save_path)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 0: 锁定维度预选 + 敏感度分析")
    parser.add_argument("--mode", choices=["lock-dims", "sensitivity", "all"],
                        required=True, help="运行模式")
    parser.add_argument("--config", required=True, help="mmdet3d 配置文件")
    parser.add_argument("--checkpoint", required=True, help="模型权重")
    parser.add_argument("--locked-config", default=None,
                        help="锁定配置 (sensitivity 模式需要)")
    parser.add_argument("--output-dir", default="work_dirs/phase0", help="输出目录")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大评估样本数 (None=全量)")
    parser.add_argument("--fast", action="store_true",
                        help="快速模式: 跳过评估, 仅验证脚本逻辑")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Phase 0: mode=%s", args.mode)
    logger.info("config: %s", args.config)
    logger.info("checkpoint: %s", args.checkpoint)
    logger.info("output: %s", args.output_dir)

    if args.mode in ("lock-dims", "all"):
        run_lock_dims(args)
        logger.info("Phase 0-A 完成!")

    if args.mode in ("sensitivity", "all"):
        run_sensitivity(args)
        logger.info("Phase 0-B 完成!")

    logger.info("Phase 0 完成!")


if __name__ == "__main__":
    main()
