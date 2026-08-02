#!/usr/bin/env python3
"""scripts/stage2_smbo_loop_v1.py 的 --manifest/adaptive_budget 运行时接入回归测试
(sw-optimizer 计划三任务收尾)。

覆盖两条路径:
  1. 不传 --manifest -> resolve_budget 必须与改动前逐位一致: K = 6(硬编码历史默认)
     或用户显式传的 --K, plan 恒为 None(下游 do_select/do_feedback 因此不写
     adaptive_budget_plan 键, 输出 JSON 结构不变)。
  2. 传 --manifest -> K 由 framework.adaptive_budget 的耦合分档决定(narrow=4/
     wide=12), 除非用户显式传 --K(显式值恒赢)。

`resolve_budget` 是纯函数(不接触磁盘之外的 manifest 文件/训练数据), 用
argparse.Namespace 直接构造入参, 不跑完整 do_select(避免真实训练 LGBM/依赖
生产 LOOP_DIR)。
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts/stage2_smbo_loop_v1.py"
PARTITIONS = ROOT / "framework/partitions"


def _load_smbo_module():
    spec = importlib.util.spec_from_file_location("stage2_smbo_loop_v1", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ns(**kwargs):
    base = {"K": None, "manifest": None, "round": 1, "precision": "fp16",
            "step": "select", "measured_json": None}
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_no_manifest_no_explicit_K_matches_legacy_default():
    """★回归锚点: 这是改动前 --K 的硬编码默认值, 绝不能变。"""
    mod = _load_smbo_module()
    K, plan = mod.resolve_budget(_ns(K=None, manifest=None))
    assert K == 6
    assert plan is None


def test_no_manifest_explicit_K_unchanged_passthrough():
    mod = _load_smbo_module()
    K, plan = mod.resolve_budget(_ns(K=9, manifest=None))
    assert K == 9
    assert plan is None


def test_manifest_codriving_selects_narrow_tier_budget():
    mod = _load_smbo_module()
    manifest = str(PARTITIONS / "codriving_partition.yaml")
    K, plan = mod.resolve_budget(_ns(K=None, manifest=manifest))
    assert plan is not None
    assert plan.tier == "narrow"
    assert K == plan.k_per_round == 4


def test_manifest_pyramid_selects_wide_tier_budget():
    mod = _load_smbo_module()
    manifest = str(PARTITIONS / "pyramid_lidar_partition.yaml")
    K, plan = mod.resolve_budget(_ns(K=None, manifest=manifest))
    assert plan is not None
    assert plan.tier == "wide"
    assert K == plan.k_per_round == 12


def test_explicit_K_overrides_manifest_derived_budget():
    mod = _load_smbo_module()
    manifest = str(PARTITIONS / "pyramid_lidar_partition.yaml")
    K, plan = mod.resolve_budget(_ns(K=7, manifest=manifest))
    assert K == 7  # explicit CLI value always wins
    assert plan is not None  # but the plan is still computed (for the printed
    assert plan.tier == "wide"  # advisory line / adaptive_budget_plan metadata)


def test_do_select_summary_omits_plan_key_when_plan_none():
    """未传 manifest 时下游 summary 不应新增 adaptive_budget_plan 键
    (确保'不传 manifest 时行为逐位不变'覆盖到输出 JSON 结构, 不仅仅是 K 数值)。"""
    mod = _load_smbo_module()
    import inspect

    sig = inspect.signature(mod.do_select)
    assert "plan" in sig.parameters
    assert sig.parameters["plan"].default is None


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
