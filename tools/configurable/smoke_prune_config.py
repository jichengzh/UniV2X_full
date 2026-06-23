"""Smoke test: prune_config.py 端到端跑通 (真实 Pyramid 子模块 rebuild).

验证:
  1. resolve_prune_plan: Config → PrunePlan 决议 (含 INT8 round_to=32).
  2. resolve_pyramid_num_filters: 剪率→通道数 + grouped-conv %32 约束 + wpg 陷阱.
  3. execute_pyramid: 真实 rebuild 小模型, params 确实下降, forward 通, 通道%32.
  4. DepGraph 可行性实测结论 (build trace 通 / local 剪枝残差陷阱).

跑: /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python tools/configurable/smoke_prune_config.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_HEAL = Path("/home/jichengzhi/heal_research/HEAL")
for p in (str(_REPO), str(_HEAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from framework.config_schema import Config  # noqa: E402
from tools.configurable.prune_config import (  # noqa: E402
    resolve_prune_plan, resolve_pyramid_num_filters, execute_pyramid,
    PATH_HANDWRITTEN_PYRAMID, INT8_ROUND_TO,
)

ORIG = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
OUT = str(_REPO / "checkpoints/configurable_pruned_smoke")


def test_resolve_plan():
    print("\n[T1] resolve_prune_plan — Config → PrunePlan (INT8 round_to)")
    cfg = Config(
        prune_rate={"model": 0.5},
        prune_object="channel",
        prune_criterion={"model": "L1"},
        q_bits={"model": "INT8"},
        config_id="smoke_p50_l1_int8",
    )
    plan = resolve_prune_plan(cfg, arch="pyramid")
    rm = plan.modules[0]
    assert rm.module == "model"
    assert rm.criterion == "l1_norm", rm.criterion
    assert rm.round_to == INT8_ROUND_TO, f"INT8 应 round_to=32, 得 {rm.round_to}"
    assert rm.exec_path == PATH_HANDWRITTEN_PYRAMID, rm.exec_path
    assert plan.int8_aligned is True
    print(f"     OK  rate={rm.prune_rate} crit={rm.criterion} round_to={rm.round_to} "
          f"path={rm.exec_path}")
    return cfg


def test_num_filters_constraint():
    print("\n[T2] resolve_pyramid_num_filters — %32 对齐 + wpg 陷阱")
    # 50% on [64,128,256] → [32,64,128], all %32, wpg=4 OK
    nf, wpg, notes = resolve_pyramid_num_filters([64, 128, 256], 0.5)
    assert nf == [32, 64, 128], nf
    assert all(n % 32 == 0 for n in nf)
    assert wpg == 4
    print(f"     50%: {nf} wpg={wpg} (all %32 OK)")
    # 75% → [16,32,64]; planes=16 在 wpg=4 下 width=int(16*4/64)*32=32 OK; 16 不%32!
    nf2, wpg2, notes2 = resolve_pyramid_num_filters([64, 128, 256], 0.75)
    print(f"     75%: {nf2} wpg={wpg2}  ({'all %32 OK' if all(n%32==0 for n in nf2) else 'NOTE: 含非%32'})")
    for n in notes2:
        print(f"        - {n}")


def test_execute_real():
    print("\n[T3] execute_pyramid — 真实 rebuild 小模型 (非 mask)")
    if not Path(ORIG).exists():
        print(f"     SKIP (未实测): 原始 ckpt 不存在 {ORIG}")
        return
    cfg = Config(
        prune_rate={"model": 0.5},
        prune_object="channel",
        prune_criterion={"model": "L1"},
        q_bits={"model": "INT8"},
        config_id="smoke_p50_l1_int8",
    )
    plan = resolve_prune_plan(cfg, arch="pyramid")
    manifest = execute_pyramid(plan, orig_dir=ORIG, out_dir=OUT, dry_run=False)

    assert manifest["status"].startswith("rebuilt")
    assert manifest["num_filters_new"] == [32, 64, 128], manifest["num_filters_new"]
    assert manifest["constraint_check"]["all_num_filters_mod32"] is True
    assert manifest["params_total_new"] < manifest["params_total_old"]
    print(f"     params total {manifest['params_total_old']:,} → "
          f"{manifest['params_total_new']:,} "
          f"(-{manifest['params_total_reduction_pct']}%)")
    print(f"     backbone     {manifest['params_backbone_old']:,} → "
          f"{manifest['params_backbone_new']:,} "
          f"(-{manifest['params_backbone_reduction_pct']}%)")
    print(f"     num_filters  {manifest['num_filters_old']} → {manifest['num_filters_new']} "
          f"(all %32 = {manifest['constraint_check']['all_num_filters_mod32']})")
    print(f"     ckpt  {manifest['out_ckpt']}  ({manifest['out_ckpt_mb']} MB)")

    # forward 通 (重新加载小模型跑一遍)
    from tools.export_onnx_pyramid import build_pyramid_from_ckpt, PyramidSubnet
    sub = PyramidSubnet(
        build_pyramid_from_ckpt(str(Path(OUT) / "config.yaml"),
                                manifest["out_ckpt"], device="cpu")
    ).eval()
    x = torch.randn(1, 64, 128, 256)
    with torch.no_grad():
        outs = sub(x)
    print(f"     forward 通 (小模型): outputs={[tuple(o.shape) for o in outs]}")
    assert outs[0].shape[1] == 2 and outs[1].shape[1] == 14 and outs[2].shape[1] == 4
    return manifest


def test_depgraph_feasibility():
    print("\n[T4] DepGraph 可行性实测 (Pyramid 子模块)")
    if not Path(ORIG).exists():
        print("     SKIP (未实测)")
        return
    import torch_pruning as tp
    from tools.export_onnx_pyramid import build_pyramid_from_ckpt, PyramidSubnet
    sub = PyramidSubnet(
        build_pyramid_from_ckpt(
            str(Path(ORIG) / "config.yaml"),
            str(sorted(Path(ORIG).glob("net_epoch_bestval_at*.pth"))[-1]),
            device="cpu")
    ).eval()
    x = torch.randn(1, 64, 128, 256)
    # build_dependency 能否 trace 通?
    try:
        tp.DependencyGraph().build_dependency(sub, example_inputs=x)
        print("     build_dependency: trace 通 (残差/downsample/deblocks 依赖图完整)")
    except Exception as e:
        print(f"     build_dependency: FAIL {type(e).__name__}: {str(e)[:80]}")
        return
    # local 剪枝是否触发残差陷阱?
    import torch.nn as nn
    sub2 = PyramidSubnet(
        build_pyramid_from_ckpt(
            str(Path(ORIG) / "config.yaml"),
            str(sorted(Path(ORIG).glob("net_epoch_bestval_at*.pth"))[-1]),
            device="cpu")
    ).eval()
    ignored = [m for n, m in sub2.named_modules()
               if n in ("cls_head", "reg_head", "dir_head")]
    pr = tp.pruner.MetaPruner(sub2, x, importance=tp.importance.MagnitudeImportance(p=1),
                              pruning_ratio=0.5, round_to=32, global_pruning=False,
                              iterative_steps=1, ignored_layers=ignored)
    pr.step()
    try:
        with torch.no_grad():
            sub2(x)
        print("     local 剪枝 forward: OK (本 ckpt 偶然通过)")
    except Exception as e:
        print(f"     local 剪枝 forward: FAIL {type(e).__name__}: {str(e)[:90]}")
        print("     → 结论: DepGraph local 剪枝 ResNeXt bottleneck 触发残差不一致, "
              "Pyramid 默认走手写 rebuild (已验证正确).")


if __name__ == "__main__":
    print("=" * 70)
    print("SMOKE TEST: tools/configurable/prune_config.py")
    print("=" * 70)
    test_resolve_plan()
    test_num_filters_constraint()
    test_execute_real()
    test_depgraph_feasibility()
    print("\n" + "=" * 70)
    print("SMOKE TEST DONE")
    print("=" * 70)
