"""Stage1 自动扫描器 — AutoTraceAdapter (Phase A0 产物).

目标 (v2 计划 §2):
  给 (build_fn, ckpt_path, bev_shape), stage1 自动扫描 —— 不写 _Net 子类、
  不手列 skipped_modules / ignored_layers。

每模型薄胶水 (~8 行):
  {name, model_class, config_path, ckpt_path, build_fn, bev_shape}

自动化:
  - ignored_layers : 按名称模式 (cls_head/reg_head/dir_head/single_head_*) 探头
  - skipped_modules: 由 build_fn 记录 (auto-skip 在 build_fn 捕获异常后填)
  - semantic_bucket: 复用 _generic_bucket()

通用 wrapper (只写一次, 不再 per-model 重写 forward):
  _CoDrivingTraceNet   : V2Xverse CoDriving backbone + optional shrink + heads
  _HeterBaselineTraceNet: HEAL HeterModelBaseline 的稠密核
                           (F-Cooper / AttFuse / Where2comm / DiscoNet 共用)

Phase A0 验证: 对 4 个手写 adapter 模型用 AutoTraceAdapter 跑出 manifest,
  与 framework/partitions/*.yaml 逐项比对。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn

_REPO = Path("/home/jichengzhi/V2X")
_HEAL = Path("/home/jichengzhi/heal_research/HEAL")
_V2XVERSE = Path("/home/jichengzhi/V2Xverse")

# 复用 adapters.py 的工具函数
from framework.stage1.adapters import (
    TraceAdapter, _add_path, _generic_bucket, _first_conv_in_channels
)


def _use_heal_opencood():
    """清除 V2Xverse opencood 缓存, 确保接下来 import opencood 用 HEAL 版本。"""
    keys = [k for k in sys.modules if k == "opencood" or k.startswith("opencood.")]
    for k in keys:
        del sys.modules[k]
    # 把 HEAL 放到 sys.path 最前
    heal_str = str(_HEAL)
    if heal_str in sys.path:
        sys.path.remove(heal_str)
    sys.path.insert(0, heal_str)


def _use_v2xverse_opencood():
    """清除 HEAL opencood 缓存, 确保接下来 import opencood 用 V2Xverse 版本。"""
    keys = [k for k in sys.modules if k == "opencood" or k.startswith("opencood.")]
    for k in keys:
        del sys.modules[k]
    v2xv_str = str(_V2XVERSE)
    if v2xv_str in sys.path:
        sys.path.remove(v2xv_str)
    sys.path.insert(0, v2xv_str)


# ---------------------------------------------------------------------------
# AutoTraceAdapter 基类
# ---------------------------------------------------------------------------

class AutoTraceAdapter(TraceAdapter):
    """通用自动扫描 adapter。

    Args:
        name            : 模型唯一标识 (注册键)
        model_class     : 人可读的模型类名
        config_path     : 配置文件路径
        ckpt_path       : 权重路径
        build_fn        : build_fn(config_path, ckpt_path, device) -> nn.Module
                          返回 trace-ready 的网络 (已处理好 dict I/O、已 to(device).eval())
        bev_shape       : BEV 特征张量形状 (1, C, H, W), 作为 dummy 输入
        ckpt_status     : "ok" | "opv2v_only_no_dair" | "missing"
        skipped_desc    : 人可读的 skipped_modules 描述 (由 build_fn 填写)
        trace_note      : 人可读的 trace 说明
        ignored_attr_names : 可选, 显式指定 ignored 层的点分路径 (如 "pyramid_backbone.single_head_0")
                             为 None 时按名称模式自动探测
    """

    # 自动识别头层的名称模式 (含点路径任意片段)
    _HEAD_PATTERNS = ("cls_head", "reg_head", "dir_head", "single_head")

    def __init__(
        self,
        name: str,
        model_class: str,
        config_path: str,
        ckpt_path: str,
        build_fn: Callable,
        bev_shape: tuple,
        ckpt_status: str = "ok",
        skipped_desc: Optional[list] = None,
        trace_note: str = "",
        ignored_attr_names: Optional[list] = None,
    ):
        self.name = name
        self.model_class = model_class
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self._build_fn = build_fn
        self._bev_shape = bev_shape
        self.ckpt_status = ckpt_status
        self.skipped_modules = skipped_desc or []
        self.trace_note = trace_note
        self._ignored_attr_names = ignored_attr_names

    # -----------------------------------------------------------------------
    # TraceAdapter 接口实现
    # -----------------------------------------------------------------------

    def build_trace_net(self, device: str):
        net = self._build_fn(self.config_path, self.ckpt_path, device)
        x = torch.zeros(self._bev_shape, device=device)
        return net, x

    def ignored_layers(self, net: nn.Module) -> list:
        """自动探测 ignored 层 (输出头), 或按 ignored_attr_names 显式取。"""
        if self._ignored_attr_names is not None:
            return self._resolve_ignored(net)
        return self._auto_detect_heads(net)

    def semantic_bucket(self, layer_name: str) -> str:
        return _generic_bucket(layer_name)

    # -----------------------------------------------------------------------
    # 辅助方法
    # -----------------------------------------------------------------------

    def _resolve_ignored(self, net: nn.Module) -> list:
        """按 ignored_attr_names 列表用点分路径取模块。"""
        result = []
        for attr in self._ignored_attr_names:
            m = net
            ok = True
            for part in attr.split("."):
                # 处理带下标的属性如 "single_head_0"
                if hasattr(m, part):
                    m = getattr(m, part)
                elif part.isdigit():
                    try:
                        m = m[int(part)]
                    except (IndexError, TypeError):
                        ok = False
                        break
                else:
                    ok = False
                    break
            if ok and isinstance(m, nn.Module) and m is not net:
                result.append(m)
            elif not ok:
                print(f"  [auto_trace] WARN: ignored_attr '{attr}' not found in {type(net).__name__}")
        return result

    def _auto_detect_heads(self, net: nn.Module) -> list:
        """按名称模式自动探测头层 (cls/reg/dir/single_head_*)。

        策略:
          - 遍历 net.named_modules(), 找名称片段含 _HEAD_PATTERNS 的模块
          - 只取最浅层级 (不要已被加的模块的子模块)
        """
        result = []
        seen_ids: set = set()
        added_names: list = []  # 已加模块的名称 (用于跳过子模块)

        for n, m in net.named_modules():
            # 跳过已加模块的子模块
            if any(n.startswith(prev + ".") for prev in added_names):
                continue
            n_lower = n.lower()
            if any(pat in n_lower for pat in self._HEAD_PATTERNS):
                if id(m) not in seen_ids and isinstance(m, nn.Module) and m is not net:
                    result.append(m)
                    seen_ids.add(id(m))
                    added_names.append(n)

        return result


# ---------------------------------------------------------------------------
# 通用 wrapper (一次, 不再 per-model 重写)
# ---------------------------------------------------------------------------

class _CoDrivingTraceNet(nn.Module):
    """V2Xverse CoDriving 稠密核 wrapper.

    backbone 的 forward 期望 {"spatial_features": x} 并返回 {"spatial_features_2d": y}。
    此 wrapper 将其包装成接受裸张量、输出 (cls, reg) 的 nn.Module, 使
    torch_pruning.DependencyGraph 可以 trace。
    """

    def __init__(self, full: nn.Module):
        super().__init__()
        self.backbone = full.backbone
        self.shrink_flag = bool(getattr(full, "shrink_flag", False))
        if self.shrink_flag:
            self.shrink_conv = full.shrink_conv
        self.cls_head = full.cls_head
        self.reg_head = full.reg_head

    def forward(self, spatial_features: torch.Tensor):
        feat = self.backbone({"spatial_features": spatial_features})["spatial_features_2d"]
        if self.shrink_flag:
            feat = self.shrink_conv(feat)
        return self.cls_head(feat), self.reg_head(feat)


class _HeterBaselineTraceNet(nn.Module):
    """HEAL HeterModelBaseline 稠密核 wrapper.

    适用: F-Cooper / AttFuse / Where2comm / DiscoNet 等所有使用
    HeterModelBaseline 且 ego 模态为 m1 的模型。
    自动查找 backbone_{m1/m2/m3/m4}, shrink_conv, cls/reg/dir heads。
    """

    def __init__(self, full: nn.Module):
        super().__init__()
        # 自动找 ego backbone
        self._backbone_found = False
        for cand in ("m1", "m2", "m3", "m4"):
            b = getattr(full, f"backbone_{cand}", None)
            if b is not None:
                self.backbone = b
                self._backbone_modality = cand
                self._backbone_found = True
                break
        if not self._backbone_found:
            raise ValueError("_HeterBaselineTraceNet: 找不到 backbone_m? 属性")

        # HeterModelBaseline 使用 shrinker_m{x} (非 shrink_flag+shrink_conv)
        # 优先检查 shrinker_m1/m2, 再回退到旧式 shrink_flag+shrink_conv
        shrinker_name = None
        for cand in (f"shrinker_{self._backbone_modality}", "shrinker_m1"):
            if hasattr(full, cand):
                shrinker_name = cand
                break
        if shrinker_name:
            self.shrinker = getattr(full, shrinker_name)
            self._shrinker_style = "module"
        elif getattr(full, "shrink_flag", False):
            self.shrinker = full.shrink_conv
            self._shrinker_style = "module"
        else:
            self._shrinker_style = "none"

        self.cls_head = full.cls_head
        self.reg_head = full.reg_head
        self._has_dir = hasattr(full, "dir_head")
        if self._has_dir:
            self.dir_head = full.dir_head

    def forward(self, spatial_features: torch.Tensor):
        feat = self.backbone(
            {"spatial_features": spatial_features}
        )["spatial_features_2d"]
        if self._shrinker_style == "module":
            feat = self.shrinker(feat)
        out = [self.cls_head(feat), self.reg_head(feat)]
        if self._has_dir:
            out.append(self.dir_head(feat))
        return tuple(out)


# ---------------------------------------------------------------------------
# Build functions: 每模型 ~8 行, 无 _Net 子类, 无源码 surgery
# ---------------------------------------------------------------------------

def _build_codriving(config_path: str, ckpt_path: str, device: str) -> nn.Module:
    _use_v2xverse_opencood()
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.models.center_point_codriving import centerpointcodriving
    hypes = load_yaml(config_path)
    full = centerpointcodriving(hypes["model"]["args"])
    raw = torch.load(ckpt_path, map_location="cpu")
    sd = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    miss, unexp = full.load_state_dict(sd, strict=False)
    print(f"  [ckpt codriving] missing={len(miss)} unexpected={len(unexp)}")
    full = full.to(device).eval()
    return _CoDrivingTraceNet(full).to(device).eval()


def _build_pyramid_lidar(config_path: str, ckpt_path: str, device: str) -> nn.Module:
    _use_heal_opencood()
    _add_path(str(_REPO))
    from tools.configurable.depgraph_pyramid import build_full, PyramidFullTraceNet, find_ckpt
    ckpt = ckpt_path if Path(ckpt_path).is_file() else find_ckpt(str(Path(ckpt_path).parent))
    full = build_full(config_path, ckpt, device)
    return PyramidFullTraceNet(full).to(device).eval()


def _build_pyramid_camera(config_path: str, ckpt_path: str, device: str) -> nn.Module:
    _use_heal_opencood()
    _add_path(str(_REPO))
    os.chdir(str(_HEAL))
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.models.heter_pyramid_single import HeterPyramidSingle

    class _PyCamNet(nn.Module):
        """Camera Pyramid: LSS 不可 trace → 从 backbone_m2 起."""
        def __init__(self, full_model, modality="m2"):
            super().__init__()
            self.backbone = getattr(full_model, f"backbone_{modality}")
            self.aligner = getattr(full_model, f"aligner_{modality}")
            self.pyramid_backbone = full_model.pyramid_backbone
            self.shrink_flag = bool(getattr(full_model, "shrink_flag", False))
            if self.shrink_flag:
                self.shrink_conv = full_model.shrink_conv
            self.cls_head = full_model.cls_head
            self.reg_head = full_model.reg_head
            self.dir_head = full_model.dir_head

        def forward(self, spatial_features: torch.Tensor):
            feat = self.backbone({"spatial_features": spatial_features})["spatial_features_2d"]
            feat = self.aligner(feat)
            feat, occ = self.pyramid_backbone.forward_single(feat)
            if self.shrink_flag:
                feat = self.shrink_conv(feat)
            cls = self.cls_head(feat)
            reg = self.reg_head(feat)
            dir_ = self.dir_head(feat)
            extras = [o for o in occ] if isinstance(occ, (list, tuple)) and occ else []
            return (cls, reg, dir_, *extras)

    hypes = load_yaml(config_path)
    full = HeterPyramidSingle(hypes["model"]["args"])
    raw = torch.load(ckpt_path, map_location="cpu")
    sd = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    miss, unexp = full.load_state_dict(sd, strict=False)
    print(f"  [ckpt pyramid_camera] missing={len(miss)} unexpected={len(unexp)}")
    full = full.to(device).eval()
    # 自动探测 modality
    modality = "m2"
    for cand in ("m2", "m1", "m3", "m4"):
        if hasattr(full, f"backbone_{cand}"):
            modality = cand
            break
    return _PyCamNet(full, modality).to(device).eval()


def _build_v2xvit(config_path: str, ckpt_path: str, device: str) -> nn.Module:
    _use_heal_opencood()
    _add_path(str(_REPO))
    from tools.configurable.depgraph_v2xvit import build_model, V2XViTBackboneTraceNet
    full = build_model(device)
    return V2XViTBackboneTraceNet(full).to(device).eval()


def _build_heter_baseline(config_path: str, ckpt_path: str, device: str) -> nn.Module:
    """通用 HEAL HeterModelBaseline builder (F-Cooper / AttFuse 共用)."""
    _use_heal_opencood()
    os.chdir(str(_HEAL))
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.models.heter_model_baseline import HeterModelBaseline
    hypes = load_yaml(config_path)
    full = HeterModelBaseline(hypes["model"]["args"])
    raw = torch.load(ckpt_path, map_location="cpu")
    sd = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    miss, unexp = full.load_state_dict(sd, strict=False)
    print(f"  [ckpt {Path(config_path).parent.name}] missing={len(miss)} unexpected={len(unexp)}")
    full = full.to(device).eval()
    return _HeterBaselineTraceNet(full).to(device).eval()


def _infer_heter_bev_shape(config_path: str) -> tuple:
    """从 HEAL hetes_yaml 推断 BEV scatter 输出形状 (1, C, H, W)."""
    _add_path(str(_HEAL))
    from opencood.hypes_yaml.yaml_utils import load_yaml
    hypes = load_yaml(config_path)
    # 找 lidar 模态的 encoder_args
    enc = None
    for modality_cfg in (hypes.get("heter") or {}).get("modality_setting", {}).values():
        if modality_cfg.get("sensor_type") == "lidar":
            enc = modality_cfg.get("encoder_args", {})
            break
    if enc is None:
        # 非 heter 结构, 直接拿 model.args 或 preprocess
        enc = hypes.get("model", {}).get("args", {}).get("encoder_args", {})
    # 取 pillar_vfe num_filters 最后一个作为 C
    pfe = enc.get("pillar_vfe", {})
    C = pfe.get("num_filters", [64])[-1]
    # 取 lidar_range + voxel_size 推 H, W
    lr = enc.get("lidar_range", hypes.get("preprocess", {}).get("args", {}).get("cav_lidar_range",
         [-51.2, -51.2, -3, 51.2, 51.2, 1]))
    vs = enc.get("voxel_size", [0.4, 0.4, 4])
    H = round((lr[4] - lr[1]) / vs[1])
    W = round((lr[3] - lr[0]) / vs[0])
    return (1, C, H, W)


# ---------------------------------------------------------------------------
# AUTO_REGISTRY
# ---------------------------------------------------------------------------

_CKPT_FCOOPER = ("/home/jichengzhi/heal_research/checkpoints/baselines_hf/"
                 "HeterBaseline_opv2v_lidar_fcooper_2023_08_06_19_53_10")
_CKPT_ATTFUSE = ("/home/jichengzhi/heal_research/checkpoints/baselines_hf/"
                 "HeterBaseline_opv2v_lidar_attfuse_2023_08_06_19_58_00")
_CKPT_PYRAMID_LIDAR = ("/home/jichengzhi/heal_research/checkpoints/stage1/"
                        "Pyramid_DAIR_m1_base_2023_08_14_11_42_29")
_CKPT_PYRAMID_CAM = "/home/jichengzhi/heal_research/checkpoints/stage2/m2_alignto_m1"
_CKPT_V2XVIT = ("/home/jichengzhi/heal_research/checkpoints/baselines_hf/"
                "HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26")
_CKPT_CODRIVING = ("/home/jichengzhi/V2Xverse/checkpoints/codriving/perception/"
                   "net_epoch_bestval_at16.pth")

AUTO_REGISTRY: dict[str, AutoTraceAdapter] = {

    # ── Phase A0: 复现 4 个手写 adapter ──────────────────────────────────
    "codriving": AutoTraceAdapter(
        name="codriving",
        model_class="centerpointcodriving",
        config_path=str(_V2XVERSE / "opencood/hypes_yaml/v2xverse/codriving_multiclass_config.yaml"),
        ckpt_path=_CKPT_CODRIVING,
        build_fn=_build_codriving,
        bev_shape=(1, 64, 192, 704),
        ckpt_status="ok",
        skipped_desc=[
            "pillar_vfe (sparse VFE, auto-skip: 不在 trace 核内)",
            "scatter (sparse scatter, auto-skip)",
            "fusion_net (CoDriving 0-param channel-preserving, auto-skip)",
        ],
        trace_note=(
            "auto_trace: _CoDrivingTraceNet (backbone + optional shrink + heads); "
            "backbone dict I/O 自动包装; 无 _Net 子类"
        ),
    ),

    "pyramid_lidar": AutoTraceAdapter(
        name="pyramid_lidar",
        model_class="HeterPyramidCollab",
        config_path=_CKPT_PYRAMID_LIDAR + "/config.yaml",
        ckpt_path=_CKPT_PYRAMID_LIDAR + "/net_epoch_bestval_at23.pth",
        build_fn=_build_pyramid_lidar,
        bev_shape=(1, 64, 128, 256),
        ckpt_status="ok",
        skipped_desc=[
            "encoder_m1 (PointPillar VFE, sparse, 不在 trace 核内)",
            "collab fusion warp_affine (channel-preserving, auto-skip)",
        ],
        trace_note=(
            "auto_trace: 复用 PyramidFullTraceNet (已有工具, 无新 _Net); "
            "pyramid_backbone(ResNeXt g=32)+deblocks+shrink+heads"
        ),
        # 显式指定头层路径 (pyramid 有多级 single_head_*)
        ignored_attr_names=[
            "cls_head", "reg_head", "dir_head",
            "pyramid_backbone.single_head_0",
            "pyramid_backbone.single_head_1",
            "pyramid_backbone.single_head_2",
        ],
    ),

    "pyramid_camera": AutoTraceAdapter(
        name="pyramid_camera",
        model_class="HeterPyramidSingle (m2)",
        config_path=_CKPT_PYRAMID_CAM + "/config.yaml",
        ckpt_path=_CKPT_PYRAMID_CAM + "/net_epoch25.pth",
        build_fn=_build_pyramid_camera,
        bev_shape=(1, 128, 256, 512),
        ckpt_status="opv2v_only_no_dair",
        skipped_desc=[
            "encoder_m2 (LiftSplatShoot: geometry proj+voxel scatter+QuickCumsum, 不可 trace, auto-skip)",
        ],
        trace_note=(
            "auto_trace: _PyCamNet (backbone_m2 起 + aligner + pyramid_backbone.forward_single + heads); "
            "aligner(ConvNeXt+LayerNorm) 在 ignored_layers 内冻结"
        ),
        ignored_attr_names=[
            "cls_head", "reg_head", "dir_head", "aligner",
            "pyramid_backbone.single_head_0",
            "pyramid_backbone.single_head_1",
            "pyramid_backbone.single_head_2",
        ],
    ),

    "v2xvit": AutoTraceAdapter(
        name="v2xvit",
        model_class="HeterModelBaseline (+ V2XTransformer)",
        config_path=_CKPT_V2XVIT + "/config.yaml",
        ckpt_path=_CKPT_V2XVIT + "/net_epoch_bestval_at17.pth",
        build_fn=_build_v2xvit,
        bev_shape=(1, 64, 256, 512),
        ckpt_status="ok",
        skipped_desc=[
            "encoder_m1 (PointPillar VFE, sparse, auto-skip)",
            "fusion_net (V2XTransformer: HMSA+MSwin, multi-agent, 不可 trace → 不剪)",
        ],
        trace_note=(
            "auto_trace: 复用 V2XViTBackboneTraceNet; 仅 backbone_m1+shrinker_m1+heads; "
            "transformer 整体不入图"
        ),
        # V2X-ViT 头层 + shrinker 输出层 (保持 fusion 输入 256ch 不被剪)
        ignored_attr_names=[
            "cls_head", "reg_head", "dir_head",
            "shrinker_m1.layers.0.double_conv.2",  # shrinker 输出 Conv2d (256ch → fusion 输入)
        ],
    ),

    # ── Phase A1: push-button 新模型 ─────────────────────────────────────
    "fcooper": AutoTraceAdapter(
        name="fcooper",
        model_class="HeterModelBaseline (MaxFusion)",
        config_path=_CKPT_FCOOPER + "/config.yaml",
        ckpt_path=_CKPT_FCOOPER + "/net_epoch_bestval_at23.pth",
        build_fn=_build_heter_baseline,
        bev_shape=(1, 64, 512, 512),   # OPV2V lidar: [-102.4,102.4]×[0.4] → 512; C=64
        ckpt_status="ok",
        skipped_desc=[
            "pillar_vfe (sparse VFE, auto-skip)",
            "scatter (sparse scatter, auto-skip)",
            "fusion_net (MaxFusion: 无参数 max pooling, auto-skip)",
        ],
        trace_note=(
            "auto_trace: _HeterBaselineTraceNet (backbone_m1 + optional shrink + heads); "
            "push-button: 零源码 surgery, 零 _Net 子类"
        ),
    ),

    "attfuse": AutoTraceAdapter(
        name="attfuse",
        model_class="HeterModelBaseline (AttFusion)",
        config_path=_CKPT_ATTFUSE + "/config.yaml",
        ckpt_path=_CKPT_ATTFUSE + "/net_epoch_bestval_at15.pth",
        build_fn=_build_heter_baseline,
        bev_shape=(1, 64, 512, 512),
        ckpt_status="ok",
        skipped_desc=[
            "pillar_vfe (sparse VFE, auto-skip)",
            "scatter (sparse scatter, auto-skip)",
            "fusion_net (AttFusion: 多 agent 注意力融合, auto-skip)",
        ],
        trace_note=(
            "auto_trace: _HeterBaselineTraceNet (backbone_m1 + optional shrink + heads); "
            "push-button: 零源码 surgery, 零 _Net 子类"
        ),
    ),
}


def get_auto_adapter(name: str) -> AutoTraceAdapter:
    if name not in AUTO_REGISTRY:
        raise KeyError(f"unknown model '{name}', choices={list(AUTO_REGISTRY)}")
    return AUTO_REGISTRY[name]


__all__ = [
    "AutoTraceAdapter", "AUTO_REGISTRY", "get_auto_adapter",
    "_CoDrivingTraceNet", "_HeterBaselineTraceNet",
    "_build_codriving", "_build_pyramid_lidar", "_build_pyramid_camera",
    "_build_v2xvit", "_build_heter_baseline",
]
