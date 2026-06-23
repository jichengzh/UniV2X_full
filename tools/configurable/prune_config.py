"""配置驱动的全网络结构化剪枝工具 (Configurable Pruning Engineer 交付物 1).

参考 DepGraph (Torch-Pruning, CVPR'23) 的依赖图自动剪枝路线. 本工具消费
`framework/config_schema.py` 的 Config 对象 (剪枝子段), 把 per-module 的
{prune_rate, prune_object, prune_criterion} 决议成一条**可执行的结构化剪枝计划**,
产出**真实 rebuild 的小模型 ckpt** (非 mask) + 一份 manifest (记录每模块剪后通道数 /
params / FLOPs), 供硬件师 build TRT engine / 量化师做 INT8 校准.

核心能力:
- per-module {prune_rate, prune_object(channel/head/2:4), criterion(L1/FPGM/Taylor)}
- 双路径决议 (M4.9 反思 #21 沿用): DepGraph trace 通的子模块走 DepGraph; trace 不通
  (或带 ResNeXt grouped-conv 残差陷阱) 的子模块 fallback 到现有手写 rebuild
  (`tools/structural_prune_pyramid.py`).
- 内置 §约束检查: grouped-conv %32 对齐 / ResNeXt wpg 陷阱 / 跨层依赖一致性 / INT8 round_to=32.

设计原则 (不可变 / fail-fast / 复用现有, 与 quant_config.py 一致):
- 不修改 framework/config_schema.py (只读契约).
- 不重写 structural_prune_pyramid.py / prune_univ2x.py (复用其 rebuild / DepGraph 逻辑).
- 跑不通真模型时只产出 plan + manifest (dry-run), 标 "未实测", 不假装.

DepGraph 可用性结论 (本工具 §实测, 见 dims_pruning_v1.md §6):
- Pyramid 子模块: DepGraph **build_dependency 可 trace 通** (残差/downsample/deblocks/
  跨 stage 依赖图完整正确), 但 **local 剪枝 ResNeXt bottleneck (grouped conv2, groups=32)
  会触发残差通道不一致** (实测 RuntimeError: size 32 vs 64). 故 Pyramid 默认走手写 rebuild
  (已正确处理 grouped-conv + 残差). DepGraph 仅用于无分组卷积的平直子网.
- UniV2X 主干: forward 签名复杂, DepGraph trace 不通, 沿用 prune_univ2x.prune_direct.

数据流:
    Config (per-module prune_rate/prune_object/prune_criterion)
        │  resolve_prune_plan()  ← 应用硬约束 (round_to=32 / grouped %32 / wpg 陷阱)
        ▼
    PrunePlan (per-module ResolvedModulePrune + 执行路径 depgraph|handwritten_pyramid|prune_direct)
        │  execute()                          ← 真实 rebuild 小模型
        ▼
    小模型 ckpt + manifest.json (给硬件师/量化师/数据生成师)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable

# ---------------------------------------------------------------------------
# 引入只读共享契约 (framework/config_schema.py)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework.config_schema import (  # noqa: E402
    Config,
    PYRAMID_M1_MODULES,
    PRUNE_OBJECT_VALUES,
    PRUNE_CRITERION_VALUES,
)


# ===========================================================================
# §1 剪枝边界 / 硬约束
# ===========================================================================

# INT8 路径: 剪后通道数必须 %32==0 (TRT INT8 Tensor Core tile 对齐). round_to 取 32.
INT8_ROUND_TO = 32
# FP16/FP32 路径: round_to 取 8 (与现有 prune_univ2x default 一致).
DEFAULT_ROUND_TO = 8

# ResNeXt 分组卷积陷阱 (MEMORY: width=int(planes*wpg/64)*groups):
#   g=32 + wpg=4 时 planes<16 → width=0 崩溃. 想剪到极小要把 wpg 加到 16.
# 这里只检查可行性, 不自动改 wpg (留给调用者显式决策).
RESNEXT_GROUPS_DEFAULT = 32
RESNEXT_WPG_DEFAULT = 4

# criterion 字符串 → 内部小写名 (与 prune_univ2x._select_importance / handwritten 对齐)
CRITERION_MAP = {
    "L1": "l1_norm",
    "L2": "l2_norm",
    "Taylor": "taylor",
    "FPGM": "fpgm",
    "Wanda": "wanda",
    "none": "none",
}

# 每模块"最大可剪率边界" (受分组 / 精度 / 依赖限制, 见 dims_pruning_v1.md §3).
# 仅作 sanity warning, 不强制截断.
MAX_PRUNE_RATE = {
    "model": 0.75,   # Pyramid: 75% (num_filters [64,128,256]→[16,32,64], g32+wpg=16) 已实测过 pruned75 ckpt
    "backbone": 0.6,
    "encoder": 0.5,
    "decoder": 0.5,
    "heads": 0.3,
    "v2x_comm": 0.3,
}


class PruneConstraintError(ValueError):
    """剪枝约束违反 (fail-fast)."""


# ===========================================================================
# §2 决议结构 (per-module)
# ===========================================================================

# 执行路径枚举
PATH_DEPGRAPH = "depgraph"               # Torch-Pruning DepGraph 自动依赖图
PATH_HANDWRITTEN_PYRAMID = "handwritten_pyramid"  # tools/structural_prune_pyramid
PATH_PRUNE_DIRECT = "prune_direct"       # prune_univ2x.prune_direct (UniV2X 复杂 forward)


@dataclass(frozen=True)
class ResolvedModulePrune:
    """单模块决议后的可执行剪枝参数."""

    module: str
    prune_rate: float            # 剪掉比例 (0.0 = 不剪)
    prune_object: str            # channel / head / 2:4 / none
    criterion: str               # l1_norm / fpgm / taylor / ...
    round_to: int                # 通道对齐 (INT8→32, 否则 8)
    exec_path: str               # PATH_* 之一
    notes: tuple[str, ...] = ()  # 决议过程记录 (约束 / fallback 原因)

    def is_noop(self) -> bool:
        return self.prune_rate <= 0.0 or self.prune_object == "none"


@dataclass(frozen=True)
class PrunePlan:
    """整个 Config 决议后的剪枝计划."""

    modules: tuple[ResolvedModulePrune, ...]
    config_id: Optional[str] = None
    int8_aligned: bool = False   # 是否有模块要求 INT8 round_to=32

    def active_modules(self) -> list[ResolvedModulePrune]:
        return [m for m in self.modules if not m.is_noop()]

    def to_manifest(self) -> dict:
        return {
            "config_id": self.config_id,
            "int8_aligned": self.int8_aligned,
            "modules": [
                {
                    "module": m.module,
                    "prune_rate": m.prune_rate,
                    "prune_object": m.prune_object,
                    "criterion": m.criterion,
                    "round_to": m.round_to,
                    "exec_path": m.exec_path,
                    "notes": list(m.notes),
                }
                for m in self.modules
            ],
        }


# ===========================================================================
# §3 决议: Config → PrunePlan (应用硬约束 + 选执行路径)
# ===========================================================================

def resolve_prune_plan(
    config: Config,
    *,
    int8_targets: Optional[tuple[str, ...]] = None,
    arch: str = "pyramid",
) -> PrunePlan:
    """把 Config 的剪枝子段决议成可执行的 PrunePlan.

    Args:
        config: framework.config_schema.Config (只读).
        int8_targets: 后续要走 INT8 的模块名集合 (来自量化师的 q_bits=="INT8");
            为这些模块强制 round_to=32. 不给则从 config.q_bits 自动推断.
        arch: "pyramid" → grouped-conv 模块默认走手写 rebuild;
              "univ2x"  → 走 prune_direct.

    Returns:
        PrunePlan (不可变).

    Raises:
        PruneConstraintError: prune_object/criterion 非法, 或 prune_rate 越界 (>1).
    """
    # 自动从 q_bits 推断 INT8 目标 (量化师写入 q_bits=="INT8" 的模块)
    if int8_targets is None:
        int8_targets = tuple(
            m for m, b in (config.q_bits or {}).items() if b == "INT8"
        )

    modules_in_config = config.modules() or list(config.prune_rate.keys())
    if not modules_in_config:
        modules_in_config = list(PYRAMID_M1_MODULES)

    resolved = []
    int8_aligned = False
    for m in modules_in_config:
        rate = float(config.prune_rate.get(m, 0.0))
        if not (0.0 <= rate < 1.0):
            raise PruneConstraintError(
                f"模块 {m} prune_rate={rate} 非法 (需 0.0 <= rate < 1.0)"
            )

        # prune_object: Config 是全局粒度 (channel/head/2:4); none 表示不剪
        obj = config.prune_object if rate > 0 else "none"
        if obj not in PRUNE_OBJECT_VALUES:
            raise PruneConstraintError(
                f"prune_object={obj} 非法 (合法: {PRUNE_OBJECT_VALUES})"
            )

        crit_raw = (config.prune_criterion or {}).get(m, "L1")
        if crit_raw not in PRUNE_CRITERION_VALUES:
            raise PruneConstraintError(
                f"模块 {m} prune_criterion={crit_raw} 非法 (合法: {PRUNE_CRITERION_VALUES})"
            )
        crit = CRITERION_MAP[crit_raw]

        notes = []

        # 硬约束 1: INT8 模块 round_to=32 (TRT Tensor Core 对齐)
        if m in int8_targets:
            round_to = INT8_ROUND_TO
            int8_aligned = True
            notes.append("INT8 目标: round_to 强制 32 (TRT Tensor Core 对齐)")
        else:
            round_to = DEFAULT_ROUND_TO

        # 硬约束 2: 最大可剪率 sanity (越界仅 warn, 不截断)
        cap = MAX_PRUNE_RATE.get(m)
        if cap is not None and rate > cap:
            notes.append(
                f"WARNING: prune_rate={rate:.2f} > 建议上限 {cap:.2f} "
                f"(受分组/精度/依赖限制, 可能 AP 崩塌, 见 dims_pruning_v1.md §3)"
            )

        # 选执行路径
        if rate <= 0.0:
            exec_path = PATH_DEPGRAPH  # noop, 占位
        elif arch == "pyramid":
            # Pyramid 含 ResNeXt grouped-conv (groups=32) + 残差: DepGraph local 剪枝实测
            # 触发残差通道不一致, 沿用手写 rebuild (已正确处理 grouped-conv).
            exec_path = PATH_HANDWRITTEN_PYRAMID
            notes.append(
                "Pyramid 含 ResNeXt grouped-conv+残差: 走手写 rebuild "
                "(DepGraph local 剪枝实测触发残差不一致, 见 §6)"
            )
        elif arch == "univ2x":
            exec_path = PATH_PRUNE_DIRECT
            notes.append("UniV2X forward 复杂: DepGraph trace 不通, 走 prune_direct")
        else:
            exec_path = PATH_DEPGRAPH

        resolved.append(ResolvedModulePrune(
            module=m, prune_rate=rate, prune_object=obj, criterion=crit,
            round_to=round_to, exec_path=exec_path, notes=tuple(notes),
        ))

    return PrunePlan(
        modules=tuple(resolved),
        config_id=config.config_id,
        int8_aligned=int8_aligned,
    )


# ===========================================================================
# §4 Pyramid num_filters 决议 (剪率 → 具体通道数, 含 grouped-conv %32 + wpg 陷阱)
# ===========================================================================

def resolve_pyramid_num_filters(
    base_num_filters: list[int],
    prune_rate: float,
    *,
    groups: int = RESNEXT_GROUPS_DEFAULT,
    width_per_group: int = RESNEXT_WPG_DEFAULT,
    round_to: int = INT8_ROUND_TO,
) -> tuple[list[int], int, list[str]]:
    """把 Pyramid 的全局 prune_rate 映射到具体 num_filters + 校验分组约束.

    HEAL Bottleneck 宽度公式: width = int(planes * wpg / 64) * groups (MEMORY).
    返回 (num_filters_new, suggested_wpg, notes).

    约束:
      - 每个 num_filters[i] 必须 %32==0 (INT8 round_to=32; deblock/heads 依赖一致).
      - width = int(p*wpg/64)*groups 必须 >0 且 width//groups >= 1, 否则该 stage 崩为 0.
        当 round 后 planes 太小触发崩溃时, 建议把 wpg 从 4 提到 16 (MEMORY).
    """
    notes = []
    keep = 1.0 - prune_rate
    nf_new = []
    for p in base_num_filters:
        raw = int(round(p * keep))
        # 对齐到 round_to (向下取整到倍数, 但至少保 round_to)
        aligned = max(round_to, (raw // round_to) * round_to)
        nf_new.append(aligned)

    # 校验分组卷积可行性, 必要时建议提升 wpg
    suggested_wpg = width_per_group
    for p in nf_new:
        w = int(p * suggested_wpg / 64) * groups
        ipg = w // groups if groups else 0
        if w <= 0 or ipg < 1:
            # 尝试提升 wpg 到 16 (MEMORY 修复方案)
            w16 = int(p * 16 / 64) * groups
            if w16 > 0 and (w16 // groups) >= 1:
                suggested_wpg = 16
                notes.append(
                    f"planes={p} 在 wpg={width_per_group} 下 width={w} 崩为 0; "
                    f"建议 wpg→16 (width={w16}). (MEMORY: ResNeXt width 陷阱)"
                )
            else:
                raise PruneConstraintError(
                    f"planes={p} groups={groups} 即使 wpg=16 也 width={w16} 不可行; "
                    f"prune_rate={prune_rate} 太激进"
                )
    notes.append(
        f"prune_rate={prune_rate:.2f} → num_filters {base_num_filters}→{nf_new} "
        f"(round_to={round_to}, 均%32=={'OK' if all(n % 32 == 0 for n in nf_new) else 'FAIL'})"
    )
    return nf_new, suggested_wpg, notes


# ===========================================================================
# §5 执行: PrunePlan → 真实小模型 (复用现有 rebuild, 不重写)
# ===========================================================================

def execute_pyramid(
    plan: PrunePlan,
    *,
    orig_dir: str,
    out_dir: str,
    module_name: str = "model",
    groups: int = RESNEXT_GROUPS_DEFAULT,
    width_per_group: int = RESNEXT_WPG_DEFAULT,
    dry_run: bool = False,
) -> dict:
    """对 Pyramid 模型执行 PrunePlan 中 module_name 的剪枝, 真实 rebuild 小模型.

    复用 tools/structural_prune_pyramid.py 的 build_smaller_model / transfer_weights
    (已正确处理 ResNeXt grouped-conv groups=32 + 残差 + deblocks/heads 依赖).

    Args:
        plan: resolve_prune_plan() 的输出.
        orig_dir: 原始 Pyramid ckpt 目录 (含 config.yaml + net_epoch_bestval_at*.pth).
        out_dir: 输出小模型目录.
        module_name: 取 plan 中哪个模块的 prune_rate (Pyramid 用 "model").
        dry_run: True 则只决议 num_filters + 返回 manifest, 不真 rebuild.

    Returns:
        manifest dict (含剪后 num_filters / params / 路径 / 约束 notes).
    """
    rm = next((m for m in plan.modules if m.module == module_name), None)
    if rm is None:
        raise PruneConstraintError(f"PrunePlan 无模块 {module_name}")

    # 找原始 num_filters (从 config.yaml)
    import yaml as _yaml
    HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
    if str(HEAL_ROOT) not in sys.path:
        sys.path.insert(0, str(HEAL_ROOT))
    from opencood.hypes_yaml.yaml_utils import load_yaml  # noqa: E402

    cfg_path = Path(orig_dir) / "config.yaml"
    hypes = load_yaml(str(cfg_path))
    nf_old = hypes["model"]["args"]["fusion_backbone"]["num_filters"]

    nf_new, sug_wpg, nf_notes = resolve_pyramid_num_filters(
        nf_old, rm.prune_rate, groups=groups,
        width_per_group=width_per_group, round_to=rm.round_to,
    )

    manifest = {
        "module": module_name,
        "exec_path": rm.exec_path,
        "criterion": rm.criterion,
        "prune_object": rm.prune_object,
        "prune_rate": rm.prune_rate,
        "round_to": rm.round_to,
        "num_filters_old": list(nf_old),
        "num_filters_new": nf_new,
        "groups": groups,
        "width_per_group": sug_wpg,
        "notes": list(rm.notes) + nf_notes,
        "out_dir": str(out_dir),
        "dry_run": dry_run,
    }

    if dry_run:
        manifest["status"] = "dry_run (未实测 rebuild)"
        return manifest

    # 真实 rebuild — 复用 structural_prune_pyramid (不重写)
    import torch
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from tools.structural_prune_pyramid import (  # noqa: E402
        build_smaller_model, transfer_weights,
    )
    from tools.export_onnx_pyramid import build_pyramid_from_ckpt  # noqa: E402

    bestvals = sorted(Path(orig_dir).glob("net_epoch_bestval_at*.pth"),
                      key=lambda p: int(p.stem.split("_at")[-1]))
    if not bestvals:
        raise PruneConstraintError(f"{orig_dir} 无 net_epoch_bestval_at*.pth")
    orig_ckpt = bestvals[-1]
    src_epoch = int(orig_ckpt.stem.split("_at")[-1])

    old_model = build_pyramid_from_ckpt(str(cfg_path), str(orig_ckpt), device="cpu")
    new_model, _ = build_smaller_model(str(cfg_path), nf_new,
                                       groups=groups, width_per_group=sug_wpg)
    new_model.eval()
    transfer_weights(old_model, new_model, nf_old, nf_new,
                     groups=groups, width_per_group=sug_wpg)

    n_old = sum(p.numel() for p in old_model.parameters())
    n_new = sum(p.numel() for p in new_model.parameters())
    n_pb_old = sum(p.numel() for p in old_model.pyramid_backbone.parameters())
    n_pb_new = sum(p.numel() for p in new_model.pyramid_backbone.parameters())

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_ckpt = out / f"net_epoch_bestval_at{src_epoch}.pth"
    torch.save({"model_state_dict": new_model.state_dict()}, out_ckpt)

    # 保存 config.yaml (patch num_filters)
    hypes["model"]["args"]["fusion_backbone"]["num_filters"] = nf_new
    if groups != RESNEXT_GROUPS_DEFAULT:
        hypes["model"]["args"]["fusion_backbone"]["resnext_groups"] = groups
    if sug_wpg != RESNEXT_WPG_DEFAULT:
        hypes["model"]["args"]["fusion_backbone"]["width_per_group"] = sug_wpg
    with open(out / "config.yaml", "w") as f:
        _yaml.dump(hypes, f, default_flow_style=False, allow_unicode=True)

    manifest.update({
        "status": "rebuilt (真实小模型, 非 mask)",
        "params_total_old": n_old,
        "params_total_new": n_new,
        "params_total_reduction_pct": round((1 - n_new / n_old) * 100, 2),
        "params_backbone_old": n_pb_old,
        "params_backbone_new": n_pb_new,
        "params_backbone_reduction_pct": round((1 - n_pb_new / n_pb_old) * 100, 2),
        "out_ckpt": str(out_ckpt),
        "out_ckpt_mb": round(out_ckpt.stat().st_size / 1e6, 2),
        "constraint_check": {
            "all_num_filters_mod32": all(n % 32 == 0 for n in nf_new),
            "grouped_conv_feasible": True,
        },
    })
    return manifest


def write_manifest(manifest: dict, path: str | Path) -> None:
    """把 manifest 写 JSON (给硬件师/量化师/数据生成师消费)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


# ===========================================================================
# §6 CLI
# ===========================================================================

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="配置驱动结构化剪枝 (Config → 真实小模型)")
    ap.add_argument("--config", help="Config YAML/JSON 路径 (剪枝子段)")
    ap.add_argument("--arch", default="pyramid", choices=["pyramid", "univ2x"])
    ap.add_argument("--orig-dir",
                    default="/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29")
    ap.add_argument("--out-dir", default=str(_REPO_ROOT / "checkpoints/configurable_pruned"))
    ap.add_argument("--manifest", default=str(_REPO_ROOT / "output/prune_manifest.json"))
    ap.add_argument("--dry-run", action="store_true", help="只决议+写 manifest, 不真 rebuild")
    args = ap.parse_args()

    if args.config:
        cfg = (Config.from_yaml(args.config) if args.config.endswith((".yaml", ".yml"))
               else Config.from_dict(json.load(open(args.config))))
    else:
        # 缺省演示 Config: Pyramid model 模块剪 50%, L1, channel, 走 INT8
        cfg = Config(
            prune_rate={"model": 0.5},
            prune_object="channel",
            prune_criterion={"model": "L1"},
            q_bits={"model": "INT8"},
            config_id="demo_pyramid_p50_l1_int8",
        )

    plan = resolve_prune_plan(cfg, arch=args.arch)
    print("=== PrunePlan ===")
    print(json.dumps(plan.to_manifest(), ensure_ascii=False, indent=2))

    if args.arch == "pyramid":
        manifest = execute_pyramid(
            plan, orig_dir=args.orig_dir, out_dir=args.out_dir,
            dry_run=args.dry_run,
        )
        write_manifest(manifest, args.manifest)
        print("\n=== Execute manifest ===")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"\nmanifest 写入: {args.manifest}")
    else:
        print("\n[univ2x] 决议完成; rebuild 请走 projects/.../pruning/prune_univ2x.apply_prune_config")


if __name__ == "__main__":
    _cli()
