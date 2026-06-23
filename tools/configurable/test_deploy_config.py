"""单元测试: deploy_config.build_plan 的契约门控逻辑 (无需 GPU/TRT build)。

运行:
    pytest tools/configurable/test_deploy_config.py -v
或:
    python tools/configurable/test_deploy_config.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.config_schema import Config  # noqa: E402
from tools.configurable.deploy_config import build_plan  # noqa: E402

HW_4090 = str(REPO_ROOT / "configs/hardware/rtx4090.yaml")
HW_ORIN = str(REPO_ROOT / "configs/hardware/orin_agx.yaml")
# 任意已存在 ONNX 用于 plan 推导 (build_plan 只读 Q/DQ 节点数, 不 build)
ONNX_QDQ = str(REPO_ROOT / "onnx/univ2x_ego_bev_encoder_qdq.onnx")
ONNX_FP = str(REPO_ROOT / "onnx/univ2x_ego_bev_encoder_200_1cam.onnx")


def test_qdq_onnx_triggers_explicit_int8_on_4090():
    plan = build_plan(ONNX_QDQ, "/tmp/x.engine", HW_4090, precision="auto")
    assert plan.has_qdq is True
    assert plan.qdq_count > 0
    assert plan.want_int8 is True
    assert plan.int8_mode == "explicit"
    assert plan.use_dla is False  # 4090 无 DLA


def test_fp_onnx_no_qdq_defaults_fp16():
    plan = build_plan(ONNX_FP, "/tmp/x.engine", HW_4090, precision="auto")
    assert plan.has_qdq is False
    assert plan.want_int8 is False
    assert plan.want_fp16 is True


def test_dla_routing_degraded_to_gpu_on_4090():
    # Config 要求 DLA, 但 4090 无 DLA → 应降级为 GPU
    cfg = Config(d_routing={"model": "DLA0"})
    plan = build_plan(ONNX_FP, "/tmp/x.engine", HW_4090, config=cfg, precision="fp16")
    assert plan.use_dla is False
    assert all(not r.startswith("DLA") for r in plan.routing.values())


def test_dla_routing_kept_on_orin():
    cfg = Config(d_routing={"model": "DLA1"})
    plan = build_plan(ONNX_FP, "/tmp/x.engine", HW_ORIN, config=cfg, precision="fp16")
    assert plan.use_dla is True
    assert plan.dla_core == 1


def test_strongly_typed_only_on_4090():
    p4090 = build_plan(ONNX_FP, "/tmp/x.engine", HW_4090, precision="fp16")
    porin = build_plan(ONNX_FP, "/tmp/x.engine", HW_ORIN, precision="fp16")
    assert p4090.strongly_typed is True   # TRT10 4090
    assert porin.strongly_typed is False  # TRT8.5 Orin


def test_alignment_enforcement_from_yaml():
    p4090 = build_plan(ONNX_FP, "/tmp/x.engine", HW_4090, precision="fp16")
    porin = build_plan(ONNX_FP, "/tmp/x.engine", HW_ORIN, precision="fp16")
    assert p4090.alignment_enforcement == "hard"  # 4090 未实证
    assert porin.alignment_enforcement == "soft"  # N2v2 实证


def test_workspace_and_tactics_passthrough():
    plan = build_plan(ONNX_FP, "/tmp/x.engine", HW_4090, precision="fp16",
                      workspace_gb=8.0, tactic_sources=("CUBLAS", "CUDNN"))
    assert plan.workspace_gb == 8.0
    assert plan.tactic_sources == ("CUBLAS", "CUDNN")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
