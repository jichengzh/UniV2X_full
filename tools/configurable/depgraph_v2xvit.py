"""【A-2 正式授权 — team-lead 2026-06-04】A-2 批准启动
授权原文引用 (ISS-030 回执协议):
  "授权范围 = ①depgraph_v2xvit wrapper 开发(纯 CPU) ②backbone_m1 L1 剪枝 p50+p75"

DepGraph 剪枝工具 — HEAL V2X-ViT (HeterModelBaseline / heter_model_baseline).

目标:
  对 V2X-ViT 的 backbone_m1 (BaseBEVBackbone) + shrinker_m1 进行 L1 通道剪枝。
  fusion_net (transformer) 不剪枝 (依据 A-2 授权范围)。

关键设计决策:
  fusion_net forward 需要 multi-agent inputs + com_mask，不可 trace 。
  → 创建 V2XViTBackboneTraceNet wrapper: 仅 trace backbone_m1 → shrinker_m1 → heads。
  → shrinker_m1 输出 (256ch) 直接接 cls/reg/dir heads (跳过 fusion)。
  → DepGraph 依赖图覆盖: backbone_m1 内部 + backbone→shrinker 接口通道。
  → shrinker 输出 256ch 标记为 ignored (不剪, 保持 fusion_net 输入维度不变)。

纪律 (ISS-005/009/024):
  - 输出 flat state_dict (无 model_state_dict 包裹)
  - 记录 epoch_used + 实际剪枝率供后续 finetune 锁定
  - A-2 剪枝后须 finetune 25ep 才算有效数据

用法:
  python tools/configurable/depgraph_v2xvit.py \\
      --ratio 0.5 \\
      --out-dir output/a2_prune/v2xvit_bb_p50 \\
      [--device cpu]

  python tools/configurable/depgraph_v2xvit.py \\
      --ratio 0.75 \\
      --out-dir output/a2_prune/v2xvit_bb_p75 \\
      [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2]
_HEAL = Path("/home/jichengzhi/heal_research/HEAL")
for _p in (str(_REPO), str(_HEAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch_pruning as tp  # noqa: E402
from opencood.models.heter_model_baseline import HeterModelBaseline  # noqa: E402
from opencood.hypes_yaml.yaml_utils import load_yaml  # noqa: E402
from opencood.tools import train_utils  # noqa: E402

CKPT_DIR = Path("/home/jichengzhi/heal_research/checkpoints/baselines_hf/"
                "HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26")
CONFIG_YAML = CKPT_DIR / "config.yaml"
CKPT_FILE = "net_epoch_bestval_at17.pth"  # ISS-024: epoch17 lock


# ---------------------------------------------------------------------------
# 1. 全网 trace wrapper (backbone_m1 → shrinker_m1 → heads, 跳过 fusion)
# ---------------------------------------------------------------------------

class V2XViTBackboneTraceNet(nn.Module):
    """把 HeterModelBaseline 的可剪稠密路径收进固定计算图。

    入口: spatial_features (1, 64, H, W) = PointPillar scatter 的稠密输出
    路径: backbone_m1 → shrinker_m1 → cls/reg/dir heads
    说明: 跳过 fusion_net (transformer)，因其需要 multi-agent inputs 不可 trace。
          shrinker 输出直接接 heads，仅用于建立 backbone↔shrinker 通道依赖图。
    """

    def __init__(self, model: HeterModelBaseline):
        super().__init__()
        self.backbone_m1 = model.backbone_m1
        self.shrinker_m1 = model.shrinker_m1
        # 是否有 top-level shrink_conv (与 fusion 无关)
        self.has_shrink = getattr(model, "shrink_flag", False)
        if self.has_shrink:
            self.shrink_conv = model.shrink_conv
        self.cls_head = model.cls_head
        self.reg_head = model.reg_head
        self.dir_head = model.dir_head

    def forward(self, spatial_features: torch.Tensor):
        # backbone_m1: BaseBEVBackbone dict-style API
        feat = self.backbone_m1({"spatial_features": spatial_features})[
            "spatial_features_2d"
        ]
        # shrinker_m1: double_conv stride=2
        feat = self.shrinker_m1(feat)
        # optional top-level shrink_conv
        if self.has_shrink:
            feat = self.shrink_conv(feat)
        # 预测头 (输出通道 fixed → ignored by DepGraph)
        cls  = self.cls_head(feat)
        reg  = self.reg_head(feat)
        dir_ = self.dir_head(feat)
        return cls, reg, dir_


# ---------------------------------------------------------------------------
# 2. 模型加载
# ---------------------------------------------------------------------------

def build_model(device: str) -> HeterModelBaseline:
    """加载 V2X-ViT DAIR baseline (epoch17 bestval)。"""
    os.chdir(str(_HEAL))  # 让相对数据路径正常解析
    hypes = load_yaml(str(CONFIG_YAML))
    from opencood.hypes_yaml.yaml_utils import load_general_params
    hypes = load_general_params(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    model = train_utils.create_model(hypes)
    ckpt_path = str(CKPT_DIR / CKPT_FILE)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]  # ISS-005: unwrap if wrapped
        print("[ckpt] WARNING: wrapped format detected, unwrapped")
    model.load_state_dict(state)
    print(f"[ckpt] loaded {CKPT_FILE} (epoch17, ISS-024 lock)")
    return model.to(device).eval()


def get_scatter_shape(hypes: dict) -> tuple[int, int]:
    """从 DAIR config 计算 PointPillar scatter 输出的 H, W。"""
    r = hypes["cav_lidar_range"]  # [xmin, ymin, zmin, xmax, ymax, zmax]
    vx = hypes["model"]["args"]["m1"]["encoder_args"]["voxel_size"][0]
    vy = hypes["model"]["args"]["m1"]["encoder_args"]["voxel_size"][1]
    nx = round((r[3] - r[0]) / vx)  # W
    ny = round((r[4] - r[1]) / vy)  # H
    return ny, nx  # (H, W)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# 3. DepGraph 剪枝
# ---------------------------------------------------------------------------

def build_pruner(net: V2XViTBackboneTraceNet, x: torch.Tensor,
                 ratio: float, device: str):
    """构建 MetaPruner (L1 重要性, 全局 False, round_to=32)。"""
    # 忽略: 任务头输出通道固定 (预测格式不可改)
    ignored = [net.cls_head, net.reg_head, net.dir_head]
    # shrinker_m1 输出通道也 ignored: 保持 fusion_net 输入维度 256 不变
    # 通过 ignore shrinker 的最后一个 Conv 层实现
    for name, mod in net.shrinker_m1.named_modules():
        if isinstance(mod, nn.Conv2d) and "double_conv.2" in name:
            ignored.append(mod)
            print(f"[pruner] ignored shrinker output layer: shrinker_m1.{name}")
    if net.has_shrink:
        ignored.append(net.shrink_conv)

    return tp.pruner.MetaPruner(
        net, x,
        importance=tp.importance.MagnitudeImportance(p=1),   # L1 norm
        pruning_ratio=ratio,
        round_to=32,         # 保持通道数为 32 的倍数
        global_pruning=False,
        iterative_steps=1,
        ignored_layers=ignored,
    )


def prune_and_save(ratio: float, out_dir: Path, device: str) -> dict:
    """执行剪枝并保存 flat state_dict + 元信息。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"V2X-ViT backbone_m1 L1 剪枝  ratio={ratio:.2f}  device={device}")
    print(f"{'='*72}")

    # ── 加载模型 ──────────────────────────────────────────────────────────
    model = build_model(device)
    net   = V2XViTBackboneTraceNet(model).to(device).eval()

    # ── dummy 输入 (scatter 输出) ─────────────────────────────────────────
    hypes_raw = load_yaml(str(CONFIG_YAML))
    ny, nx    = get_scatter_shape(hypes_raw)
    x = torch.randn(1, 64, ny, nx, device=device)
    print(f"[trace] dummy scatter input: (1, 64, {ny}, {nx})")

    # ── forward 健检 (剪前) ───────────────────────────────────────────────
    with torch.no_grad():
        outs_before = net(x)
    p_before = n_params(net)
    pb_before = n_params(net.backbone_m1)
    ps_before = n_params(net.shrinker_m1)
    print(f"[pre ] total={p_before:,}  backbone={pb_before:,}  shrinker={ps_before:,}")
    print(f"[pre ] output shapes: {[tuple(o.shape) for o in outs_before]}")

    # ── DepGraph build ────────────────────────────────────────────────────
    print("[depgraph] building dependency graph...")
    DG = tp.DependencyGraph()
    DG.build_dependency(net, example_inputs=x)
    n_groups = len(list(DG.get_all_groups()))
    print(f"[depgraph] dependency groups: {n_groups}")

    # ── pruner step ───────────────────────────────────────────────────────
    print(f"[prune] MetaPruner L1 ratio={ratio} round_to=32 global=False ...")
    pr = build_pruner(net, x, ratio, device)
    pr.step()
    print("[prune] step() done")

    # ── forward 健检 (剪后) ───────────────────────────────────────────────
    with torch.no_grad():
        outs_after = net(x)
    p_after   = n_params(net)
    pb_after  = n_params(net.backbone_m1)
    ps_after  = n_params(net.shrinker_m1)
    reduction = (1 - p_after / p_before) * 100
    bb_reduction = (1 - pb_after / pb_before) * 100
    print(f"[post] total={p_after:,} ({reduction:.1f}% off)")
    print(f"[post] backbone={pb_after:,} ({bb_reduction:.1f}% off)")
    print(f"[post] shrinker={ps_after:,}")
    print(f"[post] output shapes: {[tuple(o.shape) for o in outs_after]}")

    # 验证输出头 shape 不变
    for i, (b, a) in enumerate(zip(outs_before, outs_after)):
        assert b.shape == a.shape, \
            f"head[{i}] shape changed! {b.shape} → {a.shape}"
    print("[post] ✓ 输出头 shape 不变 (anchor format 保持)")

    # 验证 shrinker 输出通道 = 256 (fusion_net 输入兼容)
    shrinker_out_ch = outs_after[0].shape[1] if False else None
    # 直接查 shrinker 最后 conv 输出通道
    for nm, mod in net.shrinker_m1.named_modules():
        if isinstance(mod, nn.Conv2d) and "double_conv.2" in nm:
            assert mod.out_channels == 256, \
                f"shrinker output channels changed! {mod.out_channels}"
            print(f"[post] ✓ shrinker_m1 output ch = {mod.out_channels} (fusion 兼容)")
            break

    # ── 保存 flat state_dict (ISS-005) ────────────────────────────────────
    # 需要将剪枝后的 backbone/shrinker 权重写回原始 model 对象
    # V2XViTBackboneTraceNet 共享同一 param objects → state_dict 直接含剪枝后权重
    full_sd = {}
    for k, v in net.backbone_m1.state_dict().items():
        full_sd[f"backbone_m1.{k}"] = v
    for k, v in net.shrinker_m1.state_dict().items():
        full_sd[f"shrinker_m1.{k}"] = v
    # 保留 fusion_net 等其余模块 (未剪, 直接从原始 model 取)
    for k, v in model.state_dict().items():
        if not k.startswith("backbone_m1.") and not k.startswith("shrinker_m1."):
            full_sd[k] = v

    ckpt_out = out_dir / f"v2xvit_pruned_{int(ratio*100)}_epoch17_depgraph.pth"
    torch.save(full_sd, str(ckpt_out))
    print(f"\n[save] flat state_dict → {ckpt_out}")
    print(f"       keys: {len(full_sd)}, "
          f"size: {sum(v.numel() for v in full_sd.values()):,} params")

    # ── 保存剪枝元信息 ────────────────────────────────────────────────────
    # 记录实际 backbone num_filters 供 finetune YAML 适配
    actual_filters = []
    for i, blk in enumerate(net.backbone_m1.blocks):
        for mod in blk.modules():
            if isinstance(mod, nn.Conv2d) and mod.groups == 1:
                actual_filters.append(mod.out_channels)
                break
    print(f"[meta] backbone actual output channels per stage: {actual_filters}")

    meta = {
        "tool":          "depgraph_v2xvit.py",
        "authorization": "A-2 正式授权 — team-lead 2026-06-04",
        "epoch_src":     "bestval_at17",   # ISS-024
        "prune_target":  "backbone_m1 (L1 channel, round_to=32)",
        "ratio_requested":   ratio,
        "params_before": p_before,
        "params_after":  p_after,
        "reduction_pct": round(reduction, 2),
        "backbone_before": pb_before,
        "backbone_after":  pb_after,
        "backbone_reduction_pct": round(bb_reduction, 2),
        "actual_backbone_stage_filters": actual_filters,
        "shrinker_output_ch": 256,  # verified, fusion_net compat
        "ckpt_out":      str(ckpt_out),
        "forward_ok":    True,
        "note": ("ISS-005: flat state_dict, no wrapper. "
                 "ISS-009: finetune required before AP eval. "
                 "ISS-024: epoch_src=bestval_at17, finetune ckpt must lock epoch."),
    }
    meta_out = out_dir / "prune_meta.json"
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] meta → {meta_out}")
    print(f"\n[DONE] backbone reduction: {bb_reduction:.1f}%  "
          f"total: {reduction:.1f}%  forward: OK")
    return meta


# ---------------------------------------------------------------------------
# 4. Finetune YAML 生成辅助
# ---------------------------------------------------------------------------

def write_finetune_yaml(out_dir: Path, meta: dict):
    """在 out_dir 生成 finetune 用的 config.yaml (修改 backbone num_filters)。"""
    import shutil, re
    src_yaml = CONFIG_YAML
    dst_yaml = out_dir / "config_finetune.yaml"
    shutil.copy(src_yaml, dst_yaml)
    # 更新 model_dir hint
    txt = dst_yaml.read_text()
    # 写入注释说明
    hdr = (f"# A-2 finetune config — pruned {int(meta['ratio_requested']*100)}%\n"
           f"# backbone stage filters: {meta['actual_backbone_stage_filters']}\n"
           f"# prune_meta: {out_dir/'prune_meta.json'}\n"
           f"# ckpt: {meta['ckpt_out']}\n\n")
    dst_yaml.write_text(hdr + txt)
    print(f"[yaml] finetune config → {dst_yaml}")
    print(f"       Note: manually adjust num_filters in yaml if train.py "
          f"instantiates model from scratch (needed if pruned arch ≠ yaml arch)")


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="V2X-ViT backbone L1 DepGraph pruning")
    ap.add_argument("--ratio", type=float, required=True,
                    help="Pruning ratio (e.g. 0.5 for 50%%)")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Output directory for pruned ckpt + meta")
    ap.add_argument("--device", type=str, default="cpu",
                    help="Device (cpu recommended for pruning)")
    args = ap.parse_args()

    # resolve 到绝对路径，防 os.chdir(_HEAL) 后相对路径失效
    out_dir = Path(args.out_dir).resolve()
    meta    = prune_and_save(args.ratio, out_dir, args.device)
    write_finetune_yaml(out_dir, meta)

    print("\n" + "="*72)
    print("NEXT STEP (ISS-009): finetune 必须在剪枝后运行才算有效数据")
    print(f"  参考命令:")
    print(f"  CUDA_VISIBLE_DEVICES=1 python opencood/tools/train.py \\")
    print(f"      --hypes_yaml {out_dir}/config_finetune.yaml \\")
    print(f"      --model_dir <finetune_output_dir>")
    print(f"  注: 若 train.py 从 YAML 重建模型, 须先确认 YAML num_filters 与剪枝后一致")
    print(f"      (或使用 load_saved_model strict=False + resume)")
    print("="*72)


if __name__ == "__main__":
    main()
