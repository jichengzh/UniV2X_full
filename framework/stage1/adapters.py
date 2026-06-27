"""Stage1 子流程 B — 每模型 trace adapter (薄, 声明 trace 边界).

设计文档: §1 (trace_adapter 形式) + §3 S0。
每个 adapter 声明: 稠密 trace 核入口 / 跳过的不可 trace 模块 / 输出头(冻结) / 语义桶。
模型无关核 graph_scan.py 只认 TraceAdapter 接口, 不认具体模型。

4 个模型 (事实来自 multi_agent/model/*_structure_audit_v1.md + 源码核实):
  - codriving       : centerpointcodriving (V2Xverse), ResNetBEV(BasicBlock, 无 grouped), cls3/reg24
  - pyramid_lidar   : HeterPyramidCollab m1 (HEAL), ResNeXt g=32, cls2/reg14/dir4, 复用 depgraph_pyramid
  - pyramid_camera  : HeterPyramidSingle m2 (HEAL), LSS 不可 trace → 从 backbone_m2(128ch) 起 trace, 仅 OPV2V ckpt
  - v2xvit          : HeterModelBaseline (HEAL), 仅剪 backbone(transformer 不入图), 复用 depgraph_v2xvit
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path("/home/jichengzhi/V2X")
_HEAL = Path("/home/jichengzhi/heal_research/HEAL")
_V2XVERSE = Path("/home/jichengzhi/V2Xverse")


def _add_path(p: str):
    if p not in sys.path:
        sys.path.insert(0, p)


def _first_conv_in_channels(module: nn.Module, default: int = 64) -> int:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m.in_channels
    return default


def _generic_bucket(name: str) -> str:
    """层名 → 语义桶 (B2 量化视图用). 顺序敏感: 先判 neck/heads 再 backbone。"""
    n = name.lower()
    if "deblock" in n or "shrink" in n:          # 上采样 neck + shrink
        return "neck"
    if any(k in n for k in ("single_head", "cls_head", "reg_head", "dir_head")):
        return "heads"
    if "pyramid_backbone" in n:                  # HEAL ResNeXt BEV encoder
        return "bev_encoder"
    if "backbone" in n or "resnet" in n or "shrinker" in n:
        if "shrinker" in n:
            return "neck"
        return "backbone"
    return "other"


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class TraceAdapter:
    name: str = "base"
    model_class: str = ""
    config_path: str = ""
    ckpt_path: str = ""
    ckpt_status: str = "ok"        # ok / opv2v_only_no_dair / missing
    skipped_modules: list[str] = []
    skipped_subgraphs: list[dict] = []
    trace_note: str = ""

    def build_trace_net(self, device: str) -> tuple[nn.Module, torch.Tensor]:
        raise NotImplementedError

    def ignored_layers(self, net: nn.Module) -> list[nn.Module]:
        raise NotImplementedError

    def semantic_bucket(self, layer_name: str) -> str:
        return _generic_bucket(layer_name)

    def typed_skipped_subgraphs(self) -> list[dict]:
        """Return typed skipped trace boundaries while keeping legacy text skips."""

        if self.skipped_subgraphs:
            return [dict(item) for item in self.skipped_subgraphs]
        out = []
        for idx, desc in enumerate(self.skipped_modules):
            text = str(desc)
            name = text.split("(", 1)[0].strip() or f"skipped_{idx}"
            low = text.lower()
            if any(k in low for k in ("vfe", "scatter", "sparse", "quicksum", "cumsum")):
                typ = "sparse_or_geometry_preprocess"
            elif any(k in low for k in ("attention", "transformer", "where2comm", "v2vnet", "disco")):
                typ = "attention_or_routing_fusion"
            elif "fusion" in low or "warp" in low:
                typ = "fusion_or_alignment"
            else:
                typ = "custom_untraced_subgraph"
            out.append({
                "name": name,
                "type": typ,
                "description": text,
                "full_model_verdict_blocker": True,
                "blocker_gate": "trace_closure_required",
                "source": "trace_adapter.skipped_modules",
            })
        return out


# ---------------------------------------------------------------------------
# 1. CoDriving (V2Xverse)
# ---------------------------------------------------------------------------

class CoDrivingAdapter(TraceAdapter):
    name = "codriving"
    model_class = "centerpointcodriving"
    config_path = "/home/jichengzhi/V2Xverse/opencood/hypes_yaml/v2xverse/codriving_multiclass_config.yaml"
    ckpt_path = "/home/jichengzhi/V2Xverse/checkpoints/codriving/perception/net_epoch_bestval_at16.pth"
    ckpt_status = "ok"
    skipped_modules = ["pillar_vfe (sparse VFE)", "scatter (sparse)", "fusion_net (CoDriving, 0-param, channel-preserving)"]
    trace_note = "trace 核 = backbone(ResNetBEV) → shrink_conv → cls/reg heads; 入口 = scatter 稠密输出"

    class _Net(nn.Module):
        def __init__(self, full):
            super().__init__()
            self.backbone = full.backbone
            self.shrink_flag = getattr(full, "shrink_flag", False)
            if self.shrink_flag:
                self.shrink_conv = full.shrink_conv
            self.cls_head = full.cls_head
            self.reg_head = full.reg_head

        def forward(self, spatial_features):
            feat = self.backbone({"spatial_features": spatial_features})["spatial_features_2d"]
            if self.shrink_flag:
                feat = self.shrink_conv(feat)
            return self.cls_head(feat), self.reg_head(feat)

    def build_trace_net(self, device):
        _add_path(str(_V2XVERSE))
        from opencood.hypes_yaml.yaml_utils import load_yaml
        from opencood.models.center_point_codriving import centerpointcodriving
        hypes = load_yaml(self.config_path)
        full = centerpointcodriving(hypes["model"]["args"])
        raw = torch.load(self.ckpt_path, map_location="cpu")
        sd = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        miss, unexp = full.load_state_dict(sd, strict=False)
        print(f"  [ckpt codriving] missing={len(miss)} unexpected={len(unexp)}")
        full = full.to(device).eval()
        net = self._Net(full).to(device).eval()
        cin = _first_conv_in_channels(net.backbone, 64)
        x = torch.randn(1, cin, 192, 704, device=device)
        return net, x

    def ignored_layers(self, net):
        return [net.cls_head, net.reg_head]


# ---------------------------------------------------------------------------
# 2. Pyramid-LiDAR (HEAL m1) — 复用 depgraph_pyramid
# ---------------------------------------------------------------------------

class PyramidLidarAdapter(TraceAdapter):
    name = "pyramid_lidar"
    model_class = "HeterPyramidCollab"
    _ckpt_dir = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
    config_path = _ckpt_dir + "/config.yaml"
    ckpt_path = _ckpt_dir + "/net_epoch_bestval_at23.pth"
    ckpt_status = "ok"
    skipped_modules = ["encoder_m1 (PointPillar VFE, sparse)", "collab fusion warp_affine (channel-preserving)"]
    trace_note = "复用 PyramidFullTraceNet: backbone_m1→aligner→pyramid_backbone(ResNeXt g=32)→single_head→deblocks→shrink→cls/reg/dir"

    def build_trace_net(self, device):
        _add_path(str(_REPO))
        _add_path(str(_HEAL))
        from tools.configurable.depgraph_pyramid import build_full, PyramidFullTraceNet, find_ckpt
        ckpt = self.ckpt_path if Path(self.ckpt_path).is_file() else find_ckpt(self._ckpt_dir)
        full = build_full(self.config_path, ckpt, device)
        net = PyramidFullTraceNet(full).to(device).eval()
        cin = _first_conv_in_channels(net.backbone_m1, 64)
        x = torch.randn(1, cin, 128, 256, device=device)
        return net, x

    def ignored_layers(self, net):
        ign = [net.cls_head, net.reg_head, net.dir_head]
        for i in range(net.num_levels):
            ign.append(getattr(net.pyramid_backbone, f"single_head_{i}"))
        return ign


# ---------------------------------------------------------------------------
# 3. Pyramid-Camera (HEAL m2) — LSS 不可 trace, 从 backbone_m2 起
# ---------------------------------------------------------------------------

class PyramidCameraAdapter(TraceAdapter):
    name = "pyramid_camera"
    model_class = "HeterPyramidSingle (m2)"
    _ckpt_dir = "/home/jichengzhi/heal_research/checkpoints/stage2/m2_alignto_m1"
    config_path = _ckpt_dir + "/config.yaml"
    ckpt_path = _ckpt_dir + "/net_epoch25.pth"
    ckpt_status = "opv2v_only_no_dair"   # ⚠️ 无 DAIR camera ckpt, 仅 OPV2V
    skipped_modules = ["encoder_m2 (LiftSplatShoot: geometry proj + voxel scatter + QuickCumsum, 不可 trace)"]
    trace_note = ("LSS encoder 不可 trace → trace 核从 backbone_m2(BEV 入口)起: "
                  "backbone_m2→aligner_m2→pyramid_backbone.forward_single→shrink→cls/reg/dir; "
                  "⚠️ ckpt 仅 OPV2V(无 DAIR camera); "
                  "⚠️ aligner_m2(channel_align ConvNeXt 含 LayerNorm)排除出可剪集 —— "
                  "torch_pruning 结构化剪枝不更新 ConvNeXt LayerNorm normalized_shape(实测崩), "
                  "且其仅占 ~1.9% 参数, v0 冻结(其上游 backbone_m2 因耦合一并冻结, 占~4.7%); "
                  "主可剪路径 = pyramid_backbone(~56%)+neck(~37%)")

    class _Net(nn.Module):
        def __init__(self, full, modality="m2"):
            super().__init__()
            self.backbone = getattr(full, f"backbone_{modality}")
            self.aligner = getattr(full, f"aligner_{modality}")
            self.pyramid_backbone = full.pyramid_backbone
            self.shrink_flag = getattr(full, "shrink_flag", False)
            if self.shrink_flag:
                self.shrink_conv = full.shrink_conv
            self.cls_head = full.cls_head
            self.reg_head = full.reg_head
            self.dir_head = full.dir_head

        def forward(self, spatial_features):
            feat = self.backbone({"spatial_features": spatial_features})["spatial_features_2d"]
            feat = self.aligner(feat)
            feat, occ = self.pyramid_backbone.forward_single(feat)
            if self.shrink_flag:
                feat = self.shrink_conv(feat)
            cls, reg, dir_ = self.cls_head(feat), self.reg_head(feat), self.dir_head(feat)
            if isinstance(occ, (list, tuple)) and len(occ) > 0:
                return (cls, reg, dir_, *[o for o in occ])
            return cls, reg, dir_

    def build_trace_net(self, device):
        _add_path(str(_REPO))
        _add_path(str(_HEAL))
        os.chdir(str(_HEAL))
        from opencood.hypes_yaml.yaml_utils import load_yaml
        from opencood.models.heter_pyramid_single import HeterPyramidSingle
        hypes = load_yaml(self.config_path)
        full = HeterPyramidSingle(hypes["model"]["args"])
        raw = torch.load(self.ckpt_path, map_location="cpu")
        sd = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
        miss, unexp = full.load_state_dict(sd, strict=False)
        print(f"  [ckpt pyramid_camera] missing={len(miss)} unexpected={len(unexp)}")
        full = full.to(device).eval()
        # 探测 modality 名 (camera 一般是 m2)
        modality = "m2"
        if not hasattr(full, f"backbone_{modality}"):
            for cand in ("m2", "m1", "m3", "m4"):
                if hasattr(full, f"backbone_{cand}"):
                    modality = cand
                    break
        net = self._Net(full, modality).to(device).eval()
        cin = _first_conv_in_channels(net.backbone, 128)
        x = torch.randn(1, cin, 256, 512, device=device)
        return net, x

    def ignored_layers(self, net):
        ign = [net.cls_head, net.reg_head, net.dir_head]
        # aligner_m2 (ConvNeXt channel_align + LayerNorm): torch_pruning 不更新
        # LayerNorm normalized_shape → 结构化剪枝后 forward 崩; v0 整体冻结。
        ign.append(net.aligner)
        for i in range(getattr(net.pyramid_backbone, "num_levels", 3)):
            sh = getattr(net.pyramid_backbone, f"single_head_{i}", None)
            if sh is not None:
                ign.append(sh)
        return ign


# ---------------------------------------------------------------------------
# 4. V2X-ViT (HEAL) — 复用 depgraph_v2xvit, 仅剪 backbone (transformer 不入图)
# ---------------------------------------------------------------------------

class V2XViTAdapter(TraceAdapter):
    name = "v2xvit"
    model_class = "HeterModelBaseline (+ V2XTransformer)"
    _ckpt_dir = "/home/jichengzhi/heal_research/checkpoints/baselines_hf/HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26"
    config_path = _ckpt_dir + "/config.yaml"
    ckpt_path = _ckpt_dir + "/net_epoch_bestval_at17.pth"
    ckpt_status = "ok"
    skipped_modules = ["encoder_m1 (PointPillar VFE, sparse)", "fusion_net (V2XTransformer: HMSA+MSwin, multi-agent, 不可 trace → 不剪)"]
    trace_note = "复用 V2XViTBackboneTraceNet: 仅 backbone_m1→shrinker_m1→cls/reg/dir; transformer 融合整体不入图(仅剪 backbone, 依据 A-2 授权 + dims_pruning §8.6)"

    def build_trace_net(self, device):
        _add_path(str(_REPO))
        _add_path(str(_HEAL))
        from tools.configurable.depgraph_v2xvit import build_model, V2XViTBackboneTraceNet
        full = build_model(device)
        net = V2XViTBackboneTraceNet(full).to(device).eval()
        cin = _first_conv_in_channels(net.backbone_m1, 64)
        x = torch.randn(1, cin, 256, 512, device=device)
        return net, x

    def ignored_layers(self, net):
        ign = [net.cls_head, net.reg_head, net.dir_head]
        # shrinker 输出层 ignored (保持 fusion 输入 256ch)
        for nm, m in net.shrinker_m1.named_modules():
            if isinstance(m, nn.Conv2d) and "double_conv.2" in nm:
                ign.append(m)
        if getattr(net, "has_shrink", False):
            ign.append(net.shrink_conv)
        return ign


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

REGISTRY = {
    "codriving": CoDrivingAdapter,
    "pyramid_lidar": PyramidLidarAdapter,
    "pyramid_camera": PyramidCameraAdapter,
    "v2xvit": V2XViTAdapter,
}


def get_adapter(name: str) -> TraceAdapter:
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}', choices={list(REGISTRY)}")
    return REGISTRY[name]()


__all__ = ["TraceAdapter", "REGISTRY", "get_adapter"]
