"""Stage1 子流程 B — 网络计算图扫描与分区 (模型无关核).

设计文档: multi_agent/methods/design/stage1_graph_scan_partition_v1.md §3
原型: tools/configurable/depgraph_pyramid.py (PyramidFullTraceNet / build_dependency /
      module_bucket / run_global_prune 的 forward sanity) 抽象上提为此模型无关核。

流程 S1-S6:
  S1 build_dependency (DepGraph)
  S2 extract_prune_groups  (视图① B1 — 以依赖组为准, 抓跨模块残差耦合)
  S3 extract_quant_units   (视图② B2 — 语义桶 = 完整依赖组并集)
  S4 tag_routing           (视图③ D  — 逐节点 op-type × 硬件 op 白名单, D↔B2 传播边)
  S5 stats_and_validate    (参数分布 + dry-run 0.5 剪 forward sanity)
  S6 assemble + 写 partition yaml

剪枝对象锁定 channel (设计裁剪: 不做 2:4/element)。
"""
from __future__ import annotations

import math
import re
from typing import Optional

import torch
import torch.nn as nn

import torch_pruning as tp

from framework.stage1.hardware_scan import HwCapability
from framework.stage1.adapters import TraceAdapter
from framework.stage1.trace_plan import attach_runtime_validation

_PRUNABLE_TYPES = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.BatchNorm2d)
_ROOT_TYPES = [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _name_map(net: nn.Module) -> dict[int, str]:
    return {id(m): n for n, m in net.named_modules()}


def _dep_module(item):
    """从 torch_pruning Group item 取被剪 module (兼容 GroupItem / tuple)。"""
    dep = getattr(item, "dep", None)
    if dep is None:
        dep = item[0]
    return dep.target.module


def _width_of(m: nn.Module) -> Optional[int]:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return m.out_channels
    if isinstance(m, nn.Linear):
        return m.out_features
    if isinstance(m, nn.BatchNorm2d):
        return m.num_features
    return None


def _group_members(group, name_map) -> list[tuple[str, nn.Module]]:
    out, seen = [], set()
    for item in group:
        try:
            m = _dep_module(item)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(m, _PRUNABLE_TYPES):
            continue
        nm = name_map.get(id(m))
        if nm is None or id(m) in seen:
            continue
        seen.add(id(m))
        out.append((nm, m))
    return out


def _criterion_pool(members) -> list[str]:
    mods = [m for _, m in members]
    has_conv = any(isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) for m in mods)
    has_linear = any(isinstance(m, nn.Linear) for m in mods)
    if has_linear and not has_conv:
        return ["Taylor", "L1"]
    if has_conv:
        return ["L1", "FPGM"]
    return ["L1"]


def _grouped_conv_g(members) -> int:
    g = 1
    for _, m in members:
        if isinstance(m, nn.Conv2d) and m.groups > 1:
            g = max(g, m.groups)
    return g


def _module_feature(m: nn.Module) -> dict:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        groups = int(getattr(m, "groups", 1) or 1)
        cin = int(getattr(m, "in_channels", 0) or 0)
        return {
            "cin": cin,
            "cout": int(getattr(m, "out_channels", 0) or 0),
            "groups": groups,
            "ic_bn": round(cin / groups, 4) if groups else None,
            "kernel": list(getattr(m, "kernel_size", ())) or None,
            "stride": list(getattr(m, "stride", ())) or None,
            "op_type": type(m).__name__,
        }
    if isinstance(m, nn.Linear):
        return {
            "cin": int(getattr(m, "in_features", 0) or 0),
            "cout": int(getattr(m, "out_features", 0) or 0),
            "groups": 1,
            "ic_bn": int(getattr(m, "in_features", 0) or 0),
            "kernel": None,
            "stride": None,
            "op_type": type(m).__name__,
        }
    if isinstance(m, nn.BatchNorm2d):
        return {
            "cin": int(getattr(m, "num_features", 0) or 0),
            "cout": int(getattr(m, "num_features", 0) or 0),
            "groups": 1,
            "ic_bn": int(getattr(m, "num_features", 0) or 0),
            "kernel": None,
            "stride": None,
            "op_type": type(m).__name__,
        }
    return {
        "cin": None,
        "cout": None,
        "groups": 1,
        "ic_bn": None,
        "kernel": None,
        "stride": None,
        "op_type": type(m).__name__,
    }


def _group_feature(members, coupled_buckets) -> dict:
    rows = [_module_feature(m) for _, m in members]
    root = rows[0] if rows else {}
    groups = max([int(row.get("groups") or 1) for row in rows] or [1])
    ic_bn_values = [
        float(row["ic_bn"])
        for row in rows
        if row.get("ic_bn") is not None
    ]
    return {
        "cin": root.get("cin"),
        "cout": root.get("cout"),
        "groups": groups,
        "ic_bn": min(ic_bn_values) if ic_bn_values else None,
        "kernel": root.get("kernel"),
        "stride": root.get("stride"),
        "op_types": sorted({str(row.get("op_type")) for row in rows if row.get("op_type")}),
        "fanout_buckets": sorted({str(item) for item in (coupled_buckets or []) if item}),
    }


def _rollup_search_feature(features) -> dict:
    ic_bn_values = [
        float(feature["ic_bn"])
        for feature in features
        if feature.get("ic_bn") is not None
    ]
    return {
        "min_ic_bn": min(ic_bn_values) if ic_bn_values else None,
        "max_groups": max([int(feature.get("groups") or 1) for feature in features] or [1]),
        "op_types": sorted({
            str(op)
            for feature in features
            for op in (feature.get("op_types") or [])
            if op
        }) or ["unknown"],
        "fanout_buckets": sorted({
            str(bucket)
            for feature in features
            for bucket in (feature.get("fanout_buckets") or [])
            if bucket
        }),
    }


def _trace_latency_coverage(status: str | None, skipped_subgraphs: list[dict]) -> dict:
    return {
        "coverage_scope": "trace_net_only",
        "trace_net_latency_pct": None if status in {None, "skipped"} else 100.0,
        "full_model_latency_pct": None,
        "skipped_subgraphs_accounted_separately": True,
        "n_skipped_subgraphs": len(skipped_subgraphs),
        "note": (
            "Trace-net latency coverage only; this is not full-model coverage. "
            "Sparse, fusion, attention, routing, and custom skipped subgraphs "
            "remain separate blockers until traced or gated."
        ),
    }


# ---------------------------------------------------------------------------
# S2 视图① B1 剪枝单元
# ---------------------------------------------------------------------------

def extract_prune_groups(DG, net, hw: HwCapability, adapter: TraceAdapter,
                         ignored: list[nn.Module]) -> list[dict]:
    name_map = _name_map(net)
    ignored_ids = {id(m) for m in ignored}
    groups_out, gid = [], 0
    for group in DG.get_all_groups(root_module_types=_ROOT_TYPES,
                                   ignored_layers=ignored):
        members = _group_members(group, name_map)
        if not members:
            continue
        root_name, root_mod = members[0]
        if id(root_mod) in ignored_ids:
            continue
        cur_w = _width_of(root_mod)
        if not cur_w:
            continue
        g = _grouped_conv_g(members)
        round_int8 = hw.int8_align
        round_fp16 = hw.fp16_align
        # 剪枝结构对齐 (prune granularity): grouped-conv 取 lcm(round_to, groups)。
        # 决定"哪些剪后宽度结构上合法可剪" → 驱动 width_floor/max_rate。
        align_int8 = round_int8 if g == 1 else math.lcm(round_int8, g)
        # int8 dp4a/WMMA 可建对齐 (buildability): 沿 per-group 输入轴打包 pack_factor
        # 个 int8 → in_per_g % pack_factor == 0 ⟺ width % (pack_factor×groups) == 0。
        # = pack_factor × groups。这是 **TVM dp4a/WMMA 路径** 的 per-group 打包约束,
        # 不叠加 TRT 张量对齐 int8_align (那是张量 tiling, 进 round_to 不进可建性)。
        # 与 round_to 区分: 它是搜索期 int8 可建谓词 (合法宽度的可建子集), 非剪枝粒度
        # (故不进 max_rate)。g=1: 4 (标准卷积全可建, 实测 codriving_int8_verify.json
        # Cin=48/64 均 build); g=32: 128 (实测 q_int8_dp4a_pairs.csv: cur_width 96 NOT_
        # APPLICABLE / 128 PASS → 仅 %128 可建)。★ 两模型实测交叉定标: 若叠 int8_align=32
        # 取 lcm, g=1 会误判 48/16 不可建 (lcm(32,4)=32) — 与 codriving 实测矛盾。
        int8_buildable_align = hw.int8_pack_factor * max(1, g)
        floor = min(align_int8, cur_w) if cur_w >= align_int8 else cur_w
        max_rate = round((cur_w - floor) / cur_w, 4) if cur_w > floor else 0.0
        bucket = adapter.semantic_bucket(root_name)
        coupled = sorted({adapter.semantic_bucket(n) for n, _ in members})
        feature = _group_feature(members, coupled)
        groups_out.append({
            "group_id": f"g{gid}",
            "bucket": bucket,
            "root_layer": root_name,
            "member_layers": [n for n, _ in members],
            "n_members": len(members),
            "prune_dim": "out",
            "cur_width": int(cur_w),
            "grouped_conv_g": g,
            "round_to_int8": round_int8,
            "round_to_fp16": round_fp16,
            "round_to_default": align_int8,   # 保守取 INT8 对齐 (含 grouped lcm); = 剪枝粒度
            "int8_buildable_align": int(int8_buildable_align),  # dp4a int8 可建对齐 (lcm(int8_align, pack×g))
            "width_floor": int(floor),
            "enforcement": hw.alignment_enforcement,
            "max_rate": max_rate,
            "criterion_pool": _criterion_pool(members),
            "prunable": True,
            "coupled_buckets": coupled if len(coupled) > 1 else None,
            "feature": feature,
        })
        gid += 1
    return groups_out


# ---------------------------------------------------------------------------
# S3 视图② B2 量化单元
# ---------------------------------------------------------------------------

def extract_quant_units(net, prune_groups, hw: HwCapability,
                        adapter: TraceAdapter) -> list[dict]:
    # 桶 → 该桶内的依赖组
    by_bucket: dict[str, list[str]] = {}
    for grp in prune_groups:
        by_bucket.setdefault(grp["bucket"], []).append(grp["group_id"])
    # 桶 → 参数量 / op 组成 (覆盖全部 named_modules, 含 BN/未入组层)
    bucket_params: dict[str, int] = {}
    bucket_ops: dict[str, dict[str, int]] = {}
    for n, m in net.named_modules():
        if not isinstance(m, _PRUNABLE_TYPES):
            continue
        b = adapter.semantic_bucket(n)
        bucket_params[b] = bucket_params.get(b, 0) + n_params(m)
        bucket_ops.setdefault(b, {})
        t = type(m).__name__
        bucket_ops[b][t] = bucket_ops[b].get(t, 0) + 1
    units = []
    for b in sorted(set(list(by_bucket) + list(bucket_params))):
        quantizable = b != "heads"  # 输出头精度敏感, 默认候选锁 FP16
        units.append({
            "unit": b,
            "param_count": int(bucket_params.get(b, 0)),
            "op_composition": bucket_ops.get(b, {}),
            "member_groups": by_bucket.get(b, []),
            "quantizable": quantizable,
            "legal_bits": hw.legal_bits if quantizable else ["FP16"],
            "legal_granularity_w": hw.legal_granularity_w,
            "per_channel_act_ok": hw.per_channel_act,
        })
    return units


# ---------------------------------------------------------------------------
# S2b 视图①b — B1 搜索旋钮 (把 N 个耦合组按 stage/角色绑成少数搜索变量)
# ---------------------------------------------------------------------------
# 动机 (用户 Q): DepGraph 的 N 个依赖组是"结构上能独立剪的最小单元"(事实),
#   但不等于"该搜的旋钮"。43 个独立 rate = 10^43 不可行且无必要(项目实测剪枝
#   Pareto 退化, AP 由总容量主导非逐层分配)。
# 整合逻辑 (参考 UPAQ 每组统一率 / MetaPruning-AMC per-stage / HALP 全局预算):
#   按 "语义桶 + stage 索引" 把耦合组绑成一个搜索旋钮 (多组共享同一剪枝率)。
#   tying 永远合法 (只放弃自由度); max_rate 取成员最小 (最紧约束), 准则取交集。

def _stage_idx(name: str) -> Optional[int]:
    for pat in (r"resnet\.layer(\d+)", r"\blayer(\d+)", r"\bblocks?\.(\d+)",
                r"stage(\d+)"):
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def _search_key(root_layer: str, bucket: str) -> str:
    # backbone / bev_encoder 按 stage 细分; neck/heads/other 整桶绑一个旋钮
    if bucket in ("backbone", "bev_encoder"):
        idx = _stage_idx(root_layer)
        if idx is not None:
            return f"{bucket}.s{idx}"
    return bucket


def consolidate_search_groups(prune_groups) -> list[dict]:
    knobs: dict[str, dict] = {}
    for g in prune_groups:
        key = _search_key(g["root_layer"], g["bucket"])
        e = knobs.setdefault(key, {
            "search_group_id": key, "bucket": g["bucket"],
            "member_b1_groups": [], "widths": [], "max_rates": [],
            "round_to": 0, "int8_buildable_align": 1,
            "criterion_pools": [], "grouped": False,
            "features": [],
        })
        e["member_b1_groups"].append(g["group_id"])
        e["widths"].append(g["cur_width"])
        e["max_rates"].append(g["max_rate"])
        e["round_to"] = max(e["round_to"], g["round_to_default"])
        # 旋钮的 int8 可建对齐 = 成员对齐的 lcm (一个共享宽度须满足全部成员的 dp4a 约束)
        e["int8_buildable_align"] = math.lcm(
            e["int8_buildable_align"], int(g.get("int8_buildable_align", g["round_to_int8"])))
        e["criterion_pools"].append(set(g["criterion_pool"]))
        e["features"].append(g.get("feature", {}))
        if g["grouped_conv_g"] > 1:
            e["grouped"] = True
    out = []
    for key, e in knobs.items():
        crit = set.intersection(*e["criterion_pools"]) if e["criterion_pools"] else set()
        out.append({
            "search_group_id": key,
            "bucket": e["bucket"],
            "n_b1_groups": len(e["member_b1_groups"]),
            "member_b1_groups": e["member_b1_groups"],
            "widths": sorted(set(e["widths"])),
            "max_rate": round(min(e["max_rates"]), 4),   # 最紧约束 = 成员 min
            "round_to": e["round_to"],
            "int8_buildable_align": e["int8_buildable_align"],  # dp4a int8 可建对齐 (= round_to 时无新增约束)
            "grouped_conv": e["grouped"],
            "criterion_pool": sorted(crit) or ["L1"],     # 交集 → 全成员都支持
            "feature": _rollup_search_feature(e["features"]),
        })
    return out


# ---------------------------------------------------------------------------
# S4 视图③ D 路由候选 (op-type tag × A 算子白名单; D↔B2 传播边)
# ---------------------------------------------------------------------------

# torch op_type → DLA 白名单常用名 (粗映射, 仅 has_dla 时启用)
_DLA_OPNAME = {
    "Conv2d": "Conv", "ConvTranspose2d": "Deconv", "Linear": "FullyConnected",
    "BatchNorm2d": "Scale", "ReLU": "Activation",
}


def tag_routing(net, quant_units, hw: HwCapability, adapter: TraceAdapter) -> list[dict]:
    wl = set(hw.dla_op_whitelist or []) if hw.has_dla else set()
    unit_of = {}  # bucket → unit dict (for propagate target)
    for u in quant_units:
        unit_of[u["unit"]] = u["unit"]
    out = []
    for n, m in net.named_modules():
        if not isinstance(m, _PRUNABLE_TYPES):
            continue
        t = type(m).__name__
        if not hw.has_dla:
            dla_able = False  # 无 DLA → D.L1 仅 GPU (退化)
            prop = None
        else:
            dla_name = _DLA_OPNAME.get(t)
            dla_able = dla_name in wl if wl else None
            # D→B2 传播: 若该节点可路由 DLA, 其所属量化单元被锁 INT8/per-tensor/HWC4
            prop = ({"unit": adapter.semantic_bucket(n),
                     "force": "INT8/per-tensor/HWC4"} if dla_able else None)
        out.append({
            "node": n, "op_type": t,
            "bucket": adapter.semantic_bucket(n),
            "dla_able": dla_able,
            "propagate_b2": prop,
        })
    return out


# ---------------------------------------------------------------------------
# S4b 视图③b — D 路由段 (逐节点 tag 折叠成连续可路由子图 = 真决策粒度)
# ---------------------------------------------------------------------------
# 动机 (用户 Q): 只有 2 DLA + 1 GPU, 逐节点(64-137)不是决策空间。真决策粒度 =
#   "最大连续可路由子图"(GPU↔DLA 每切换插一次 reformat → 只在几个切点决策)。
# caveat: 用 named_modules 顺序近似拓扑序; 精确连续性需 fx 拓扑(路线 C, 延后)。

def routing_segments(d_nodes, hw: HwCapability) -> dict:
    if not hw.has_dla:
        return {
            "n_segments": 1,
            "note": "硬件无 DLA → 整图单 GPU 段, D.L1 退化, 无路由决策 (逐节点 dla_able 全 False)",
            "segments": [{"device": "gpu", "n_nodes": len(d_nodes)}],
        }
    segs, cur = [], None
    for n in d_nodes:
        dev = "dla" if n.get("dla_able") else "gpu"
        if cur is None or cur["device"] != dev:
            cur = {"device": dev, "n_nodes": 0, "first": n["node"], "last": n["node"]}
            segs.append(cur)
        cur["n_nodes"] += 1
        cur["last"] = n["node"]
    n_dla = sum(1 for s in segs if s["device"] == "dla")
    return {
        "n_segments": len(segs),
        "n_dla_routable_segments": n_dla,
        "note": ("路由决策 = 选哪些 DLA-able 段真放 DLA0/DLA1 vs 留 GPU; "
                 "连续性用 named_modules 顺序近似(精确需 fx 拓扑, 路线 C)"),
        "segments": segs,
    }


# ---------------------------------------------------------------------------
# S5 全局统计 + 一致性校验 (dry-run 0.5 剪 forward)
# ---------------------------------------------------------------------------

def _build_pruner(net, x, ratio, ignored):
    return tp.pruner.MetaPruner(
        net, x,
        importance=tp.importance.MagnitudeImportance(p=1),
        pruning_ratio=ratio,
        round_to=32,
        global_pruning=False,
        iterative_steps=1,
        ignored_layers=ignored,
    )


def stats_and_validate(adapter: TraceAdapter, prune_groups, hw: HwCapability,
                       device: str) -> dict:
    # 参数分布 (按桶)
    net, x = adapter.build_trace_net(device)
    total = n_params(net)
    dist: dict[str, int] = {}
    for n, m in net.named_modules():
        if isinstance(m, _PRUNABLE_TYPES):
            dist[adapter.semantic_bucket(n)] = dist.get(adapter.semantic_bucket(n), 0) + n_params(m)
    param_dist = {k: round(v / total, 4) for k, v in sorted(dist.items())}

    checks = {"params_total": int(total), "param_dist": param_dist}

    # 一致性: floor 后 %round_to
    bad = [g["group_id"] for g in prune_groups
           if g["width_floor"] % g["round_to_int8"] != 0 and g["grouped_conv_g"] == 1]
    checks["all_floor_mod_round_to"] = (len(bad) == 0)
    checks["floor_violations"] = bad or None

    # dry-run 0.5 剪 + forward sanity (复用 depgraph run_global_prune 逻辑)
    net2, x2 = adapter.build_trace_net(device)
    ignored2 = adapter.ignored_layers(net2)
    p_before = n_params(net2)
    try:
        pr = _build_pruner(net2, x2, 0.5, ignored2)
        pr.step()
        with torch.no_grad():
            outs = net2(x2)
        p_after = n_params(net2)
        g_ok = True
        for _, m in net2.named_modules():
            if isinstance(m, nn.Conv2d) and m.groups > 1:
                if m.out_channels % m.groups != 0 or m.in_channels % m.groups != 0:
                    g_ok = False
        checks["dryrun_prune05"] = {
            "status": "ok",
            "params_before": int(p_before), "params_after": int(p_after),
            "reduction_pct": round((1 - p_after / p_before) * 100, 1),
            "grouped_conv_consistent": g_ok,
            "out_shapes": [tuple(o.shape) for o in outs] if isinstance(outs, (list, tuple)) else [tuple(outs.shape)],
        }
    except Exception as e:  # noqa: BLE001
        checks["dryrun_prune05"] = {"status": "fail", "err": f"{type(e).__name__}: {str(e)[:200]}"}
    return checks


# ---------------------------------------------------------------------------
# S1-S6 主流程
# ---------------------------------------------------------------------------

def _resolve_profile(mode: str, device: str) -> bool:
    """profile-latency 模式解析: auto → 仅 cuda 时开; on → 总开(cpu 估算); off → 关。"""
    if mode == "off":
        return False
    if mode == "on":
        return True
    return str(device).startswith("cuda")   # auto


def _output_shapes(out) -> list[list[int]]:
    if torch.is_tensor(out):
        return [list(out.shape)]
    if isinstance(out, dict):
        shapes: list[list[int]] = []
        for item in out.values():
            shapes.extend(_output_shapes(item))
        return shapes
    if isinstance(out, (list, tuple)):
        shapes = []
        for item in out:
            shapes.extend(_output_shapes(item))
        return shapes
    return []


def scan(adapter: TraceAdapter, hw: HwCapability, device: str = "cpu",
         profile_latency_mode: str = "auto",
         lat_warmup: int = 30, lat_measure: int = 100) -> dict:
    """跑完整 Stage1 子流程 B + 汇合 A → 返回 partition manifest dict。"""
    print(f"[scan] {adapter.name}  device={device}  hw={hw.name}")
    print(f"       torch {torch.__version__}  tp {tp.__version__}")

    # S0 + S1
    net, x = adapter.build_trace_net(device)
    trace_plan = getattr(adapter, "trace_plan", None)
    if trace_plan is None and hasattr(adapter, "build_trace_plan"):
        try:
            trace_plan = adapter.build_trace_plan(device=device)
        except Exception as e:  # noqa: BLE001
            trace_plan = {
                "schema": "stage1_trace_plan_v1",
                "model": adapter.name,
                "detector": "trace_plan_builder_failed",
                "manual_override_used": True,
                "trace_confidence": "low",
                "coverage_scope": "unknown",
                "review_required": True,
                "review_reasons": ["trace_plan_builder_failed"],
                "selected_candidate": {
                    "candidate_id": "trace_plan_builder_failed",
                    "status": "rejected",
                    "validation": {"full_model_module_tree_scan": "fail"},
                },
                "included_modules": [],
                "ignored_layers": [],
                "skipped_subgraphs": [],
                "rejected_candidates": [
                    {
                        "candidate_id": "trace_plan_builder_failed",
                        "status": "rejected",
                        "failed_at": "trace_plan_builder",
                        "error": f"{type(e).__name__}: {str(e)[:200]}",
                        "suggested_override": "fall back to legacy TraceAdapter registry",
                    }
                ],
                "module_inventory": [],
            }
    ignored = adapter.ignored_layers(net)
    with torch.no_grad():
        out0 = net(x)
    print(f"  [S0] forward OK, entry={tuple(x.shape)}, "
          f"outs={[tuple(o.shape) for o in out0] if isinstance(out0,(list,tuple)) else tuple(out0.shape)}")

    DG = tp.DependencyGraph().build_dependency(net, example_inputs=x)
    n_groups_raw = len(list(DG.get_all_groups(root_module_types=_ROOT_TYPES,
                                              ignored_layers=ignored)))
    print(f"  [S1] build_dependency OK, prunable groups={n_groups_raw}")

    # S2 / S2b / S3 / S4 / S4b
    b1 = extract_prune_groups(DG, net, hw, adapter, ignored)
    b1_search = consolidate_search_groups(b1)
    b2 = extract_quant_units(net, b1, hw, adapter)
    d = tag_routing(net, b2, hw, adapter)
    d_seg = routing_segments(d, hw)
    print(f"  [S2] B1 groups={len(b1)} → 搜索旋钮={len(b1_search)}  "
          f"[S3] B2 units={len(b2)}  [S4] D nodes={len(d)} → 路由段={d_seg['n_segments']}")

    # S5
    checks = stats_and_validate(adapter, b1, hw, device)
    trace_plan = attach_runtime_validation(
        trace_plan,
        forward_status="ok",
        depgraph_status="ok",
        prune_status=str(checks["dryrun_prune05"].get("status") or "unknown"),
        n_prunable_groups=n_groups_raw,
        output_shapes=_output_shapes(out0),
    )
    print(f"  [S5] dryrun_prune05={checks['dryrun_prune05'].get('status')}  "
          f"param_dist={checks['param_dist']}")

    # S5b 逐层时延 profiling (内建工作流步骤; 复用 S0 的 net/x, 描述性旁注非约束)
    if _resolve_profile(profile_latency_mode, device):
        from framework.stage1.latency_profile import profile_latency
        view_latency = profile_latency(net, x, adapter, device,
                                       warmup=lat_warmup, measure=lat_measure)
        print(f"  [S5b] latency profile status={view_latency['status']}  "
              f"e2e_forward={view_latency['e2e_forward_ms']}ms  "
              f"top_bucket={view_latency['by_b2_quant_unit'][0] if view_latency['by_b2_quant_unit'] else None}")
    else:
        view_latency = {"status": "skipped",
                        "reason": (f"profile_latency_mode={profile_latency_mode}, device={device} "
                                   "(auto 仅 cuda 开; --profile-latency on 可强制 cpu 估算)")}
        print(f"  [S5b] latency profile skipped ({view_latency['reason']})")
    skipped_subgraphs = adapter.typed_skipped_subgraphs()
    view_latency.setdefault(
        "coverage",
        _trace_latency_coverage(view_latency.get("status"), skipped_subgraphs),
    )

    # S6 汇合 manifest
    manifest = {
        "stage": "stage1_partition",
        "model": adapter.name,
        "model_class": adapter.model_class,
        "config": adapter.config_path,
        "ckpt": adapter.ckpt_path,
        "ckpt_status": adapter.ckpt_status,
        "hw_capability": hw.summary(),
        "trace": {
            "entry_shape": list(x.shape),
            "skipped_modules": adapter.skipped_modules,
            "skipped_subgraphs": skipped_subgraphs,
            "note": adapter.trace_note,
        },
        "trace_plan": trace_plan,
        "prune_object": "channel",   # 设计裁剪: 锁定, 不搜 2:4/element
        "stats": {"params_total": checks["params_total"], "param_dist": checks["param_dist"]},
        "search_space_summary": {
            "n_b1_groups_structural": len(b1),     # 结构真相 (DepGraph 最小耦合单元)
            "n_b1_search_knobs": len(b1_search),   # 整合后真正要搜的剪枝率旋钮
            "n_b2_quant_units": len(b2),
            "n_d_routing_segments": d_seg["n_segments"],
            "latency_status": view_latency.get("status"),
            "latency_e2e_forward_ms": view_latency.get("e2e_forward_ms"),
        },
        "view_b1_search_groups": b1_search,   # ★ 搜索器用这个 (整合旋钮)
        "view_b1_prune_groups": b1,           # 结构真相 (保留, 不直接搜)
        "view_b2_quant_units": b2,
        "view_d_routing_segments": d_seg,     # ★ 搜索器用这个 (路由段)
        "view_d_routing": d,                  # 逐节点 tag (素材, 保留)
        "view_latency": view_latency,         # S5b 时延旁注 (描述性, 非约束; 回答"动哪里最值")
        "checks": {k: v for k, v in checks.items() if k not in ("params_total", "param_dist")},
        "scan_status": "ok" if checks["dryrun_prune05"].get("status") == "ok" else "partial",
    }
    return manifest


__all__ = ["scan", "extract_prune_groups", "extract_quant_units", "tag_routing",
           "stats_and_validate", "n_params"]
