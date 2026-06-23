"""Smoke test (可直接 python 运行, 也兼容 pytest) — 证明 quant_config 工具:

1. 给两个不同 Config (全 INT8 vs 敏感层 FP16 混精), 产出**不同的 manifest**.
2. 硬约束生效: entropy 禁用→minmax; 敏感层 INT8→FP16; activation per-tensor.
3. 分模块混合精度: mixed 模式下 INT8/FP16 模块共存于同一引擎.

不依赖真 TRT / 真模型 (纯 plan/manifest 接口验证, 标注 [接口验证 非真 build]).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework.config_schema import Config, UNIV2X_MODULES, PYRAMID_M1_MODULES  # noqa: E402
from tools.configurable.quant_config import resolve_quant_plan  # noqa: E402


def _cfg_full_int8() -> Config:
    """配置 A: Pyramid 单 "model" 模块全 INT8 + minmax + W+A + per-channel weight.

    "model" 是 backbone+heads 合并模块 (99.9% 是 Conv), 非整体敏感 →
    可真正走纯 int8 路径 (cls/reg head 子层仍在 build 时由全局敏感 substr 保 FP16,
    但模块级 effective_bits=INT8).
    """
    mods = PYRAMID_M1_MODULES
    return Config(
        prune_rate={m: 0.0 for m in mods},
        prune_object="none",
        prune_criterion={m: "none" for m in mods},
        q_bits={m: "INT8" for m in mods},
        q_granularity={m: "per-channel" for m in mods},
        q_object={m: "W+A" for m in mods},
        q_calibrator={m: "minmax" for m in mods},
        d_routing={m: "GPU" for m in mods},
        config_id="A_full_int8",
        source="smoke",
    )


def _cfg_sensitive_fp16_mixed() -> Config:
    """配置 B: backbone/encoder INT8, decoder/heads/v2x_comm FP16 (敏感层混精).

    且 heads 故意请求 entropy + INT8 — 应被约束改写为 FP16 (敏感) 而非崩塌.
    decoder 请求 INT8 — 含 attention/MSDA, 应被强制 FP16.
    """
    return Config(
        prune_rate={m: 0.0 for m in UNIV2X_MODULES},
        prune_object="none",
        prune_criterion={m: "none" for m in UNIV2X_MODULES},
        q_bits={
            "backbone": "INT8", "encoder": "INT8",
            "decoder": "INT8",  # 含 attention → 应强制 FP16
            "heads": "INT8",    # cls/reg → 应强制 FP16
            "v2x_comm": "FP16",
        },
        q_granularity={
            "backbone": "per-channel", "encoder": "per-channel",
            "decoder": "per-channel", "heads": "per-channel", "v2x_comm": "none",
        },
        q_object={
            "backbone": "W+A", "encoder": "W-only",  # 测 W-only flag
            "decoder": "W+A", "heads": "W+A", "v2x_comm": "none",
        },
        q_calibrator={
            "backbone": "minmax",
            "encoder": "entropy",  # 禁用 → 应回退 minmax
            "decoder": "none", "heads": "entropy", "v2x_comm": "none",
        },
        d_routing={m: "GPU" for m in UNIV2X_MODULES},
        config_id="B_sensitive_fp16_mixed",
        source="smoke",
    )


def run_smoke() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="quant_smoke_"))
    onnx = "models/pyramid_backbone_fp32.onnx"  # 占位路径, 不实际读取

    # ---- 配置 A: 全 INT8 ----
    plan_a = resolve_quant_plan(_cfg_full_int8(), onnx, str(tmp / "engine_a.trt"))
    man_a_path = tmp / "manifest_a.json"
    plan_a.write_manifest(man_a_path)
    man_a = json.loads(man_a_path.read_text())

    # ---- 配置 B: 敏感层 FP16 混精 ----
    plan_b = resolve_quant_plan(_cfg_sensitive_fp16_mixed(), onnx, str(tmp / "engine_b.trt"))
    man_b_path = tmp / "manifest_b.json"
    plan_b.write_manifest(man_b_path)
    man_b = json.loads(man_b_path.read_text())

    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {msg}")
        if not cond:
            fails.append(msg)

    print("=== 配置 A (全 INT8) manifest ===")
    print(f"  trt_precision = {man_a['trt_precision']}")
    print(f"  int8_modules  = {man_a['int8_modules']}")
    print(f"  fp16_modules  = {man_a['fp16_modules']}")
    print(f"  calibrator    = {man_a['calibrator']}, w_only = {man_a['w_only']}")

    print("\n=== 配置 B (敏感层 FP16 混精) manifest ===")
    print(f"  trt_precision = {man_b['trt_precision']}")
    print(f"  int8_modules  = {man_b['int8_modules']}")
    print(f"  fp16_modules  = {man_b['fp16_modules']}")
    print(f"  calibrator    = {man_b['calibrator']}, w_only = {man_b['w_only']}")
    print(f"  mixed_fp16_substr = {man_b['mixed_fp16_substr']}")
    for w in man_b["warnings"]:
        print(f"  [WARN] {w}")

    print("\n=== 断言 ===")
    # 1. 两个 manifest 必须不同
    check(man_a != man_b, "两个 Config 产出不同 manifest")
    check(man_a["trt_precision"] == "int8",
          "配置 A 全 INT8 → trt_precision=int8")
    check(man_b["trt_precision"] == "mixed",
          "配置 B 混精 → trt_precision=mixed")

    # 2. 分模块混合精度: B 同时有 INT8 和 FP16 模块
    check(len(man_b["int8_modules"]) > 0 and len(man_b["fp16_modules"]) > 0,
          "配置 B 实现分模块混合精度 (INT8+FP16 共存)")

    # 3. 敏感层约束: decoder(含attention) / heads(cls/reg) INT8→FP16
    check("decoder" in man_b["fp16_modules"],
          "敏感层约束: decoder (含 attention/MSDA) 被强制 FP16")
    check("heads" in man_b["fp16_modules"],
          "敏感层约束: heads (cls/reg) 被强制 FP16")
    check("backbone" in man_b["int8_modules"],
          "纯 Conv backbone 保持 INT8")

    # 4. entropy 禁用: encoder 请求 entropy → 实际 minmax
    enc = next(m for m in man_b["modules"] if m["module"] == "encoder")
    check(enc["calibrator"] == "minmax",
          "校准器约束: encoder entropy → 回退 minmax")
    check(man_b["calibrator"] != "entropy",
          "全局 calibrator 不含 entropy (主路径 minmax)")

    # 5. activation per-tensor: 任何 INT8 W+A 模块 activation 恒 per-tensor
    bb = next(m for m in man_b["modules"] if m["module"] == "backbone")
    check(bb["granularity_a"] == "per-tensor",
          "activation 粒度恒 per-tensor (GPU GEMM 约束)")
    check(bb["granularity_w"] == "per-channel",
          "weight 粒度可 per-channel")

    # 6. W-only: encoder 请求 W-only → 全局 w_only flag 开
    check(man_b["w_only"] is True,
          "W-only: encoder 请求 W-only → 全局 w_only=True")

    # 7. build 命令可生成且含 mixed-fp16-substr
    cmd_b = plan_b.to_build_command()
    check("--mixed-fp16-substr" in cmd_b and "--precision" in cmd_b,
          "配置 B build 命令含 --precision mixed --mixed-fp16-substr")
    check("--w-only" in cmd_b, "配置 B build 命令含 --w-only")
    print("\n[配置 B build 命令] " + " ".join(cmd_b))

    print(f"\n=== 结果: {len(fails)} 个失败 ===")
    print(f"[manifest A] {man_a_path}")
    print(f"[manifest B] {man_b_path}")
    return 1 if fails else 0


# pytest 入口
def test_two_configs_produce_different_manifests():
    assert run_smoke() == 0


if __name__ == "__main__":
    sys.exit(run_smoke())
