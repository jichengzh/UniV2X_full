"""配置驱动的全网络混合精度量化工具 (Configurable Quantization Engineer 交付物 1).

参考 QuantV2X (ICCV'25) PTQ 路线: MinMax-init + per-channel weight + AdaRound (PyTorch 侧 reference).
本工具消费 `framework/config_schema.py` 的 Config 对象, 把 per-module 的
{q_bits, q_granularity, q_object, q_calibrator} 决议成一条**可执行的 TRT build 计划**
+ 一份 manifest (记录每个模块实际生效的精度), 供硬件师 build TRT engine.

核心能力 (用户明确要的): **分模块混合精度** — 某些模块 INT8, 某些 FP16, 同一引擎内共存.
实现机制不重造: 复用 `scripts/phase1/m4_8_trt_build_bench.py` 的
`--precision mixed --mixed-fp16-substr <substrs>` (per-layer-name substring 精度覆盖) +
`--w-only` (W-only INT8) + `--calibrator minmax`.

设计原则 (不可变 / fail-fast / 复用现有):
- 不修改 framework/config_schema.py (只读契约)
- 不重写 m4_8_trt_build_bench.py / inject_qdq_from_config.py (复用其 CLI)
- 跑不通真 TRT 时只产出 plan + manifest (dry-run), 不假装 build 成功

数据流:
    Config (per-module dict)
        │  resolve_quant_plan()  ← 应用硬约束 (entropy 禁用 / activation per-tensor / 敏感层 FP16)
        ▼
    QuantPlan (per-module ResolvedModuleQuant + build CLI)
        │  to_manifest() / to_build_command()
        ▼
    manifest.json (给硬件师) + trtexec/m4_8 命令 (给硬件师执行)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 引入只读共享契约 (framework/config_schema.py)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework.config_schema import (  # noqa: E402
    Config,
    PYRAMID_M1_MODULES,
    Q_BITS_VALUES,
    Q_GRAN_VALUES,
    Q_OBJ_VALUES,
    Q_CALIBRATOR_VALUES,
)


# ===========================================================================
# §1 量化边界 / 硬约束 (来自 data/问题.md + 量化和剪枝方法.md, 全部 [实测/论文])
# ===========================================================================

# 禁用校准器: entropy 在自训 ckpt 长尾激活上触发 AP 崩塌 (0.02-0.41) [实测 问题1].
# 主路径 minmax 稳 (0.54-0.65); percentile_99_99 备选.
FORBIDDEN_CALIBRATORS = ("entropy",)

# activation 量化只能 per-tensor (GPU GEMM / Orin DLA 硬约束) [TRT 约束].
# per-channel 仅对 weight 合法.
ACTIVATION_GRANULARITY = "per-tensor"

# 模块 → ONNX/TRT layer-name 子串映射 (用于 mixed-precision 逐层匹配).
# Pyramid_DAIR_m1 当前简化为单 "model" 模块 (PYRAMID_M1_MODULES), backbone 99.9% params.
# 若未来拆分 UniV2X 5 模块, 在此扩展 substr.
MODULE_LAYER_SUBSTR: dict[str, tuple[str, ...]] = {
    # Pyramid 简化单模块: backbone + heads 合并
    "model": ("resnet", "conv", "shrink", "deblock", "cls", "reg", "dir"),
    # ResNeXt per-stage (Q1' per-stage mixed precision)
    "stage0": ("layer0", "resnet.0", "blocks.0"),
    "stage1": ("layer1", "resnet.1", "blocks.1"),
    "stage2": ("layer2", "resnet.2", "blocks.2"),
    # UniV2X 5 模块 (未来扩展, substr 为占位/约定)
    "backbone": ("backbone", "resnet", "img_backbone"),
    "encoder": ("encoder", "bev_encoder", "vfe", "scatter"),
    "decoder": ("decoder", "transformer", "attn", "attention", "msda"),
    "heads": ("cls", "reg", "dir", "head", "bbox"),
    "v2x_comm": ("v2x", "comm", "fusion", "agent"),
}

# 敏感层 (FP16-only, 不应 INT8) [论文 + 实测 问题1]:
#   - cls/reg head (logit 量化噪声 → score 边界失真 → AP 崩)
#   - attention / MSDA (自定义 plugin, TRT 无 INT8 实现, 已默认 FP16)
#   - grid_sample fusion (跨 agent warp, 高动态范围)
# 这些 substr 命中的层在 build 时强制保留 FP16.
SENSITIVE_FP16_SUBSTR = ("cls", "reg", "dir", "msda", "attn", "attention", "grid_sample")

# 纯 Conv backbone 可走 INT8 (INT8-able).
INT8_ABLE_SUBSTR = ("resnet", "conv", "layer", "shrink", "deblock", "backbone")


# ===========================================================================
# §2 数据结构: 单模块解析结果 + 完整量化计划
# ===========================================================================

@dataclass(frozen=True)
class ResolvedModuleQuant:
    """单个模块经硬约束解析后的实际量化配置."""

    module: str
    requested_bits: str            # Config 里请求的 (INT8/FP16/FP32)
    effective_bits: str            # 应用约束后实际生效的
    granularity_w: str             # weight 粒度 (per-channel / per-tensor / none)
    granularity_a: str             # activation 粒度 (恒 per-tensor 或 none)
    q_object: str                  # W+A / W-only / none
    calibrator: str                # minmax / percentile_99_99 / none
    layer_substr: tuple[str, ...]  # 该模块匹配的 TRT layer-name 子串
    forced_fp16: bool              # 是否因敏感层约束被强制 FP16
    notes: tuple[str, ...]         # 约束应用记录 (审计用)

    def is_int8(self) -> bool:
        return self.effective_bits == "INT8"

    def is_fp16(self) -> bool:
        return self.effective_bits == "FP16"


@dataclass(frozen=True)
class QuantPlan:
    """全网络量化计划 — 多模块混合精度的最终决议 + build 入口."""

    modules: tuple[ResolvedModuleQuant, ...]
    onnx_path: str
    engine_path: str
    # TRT precision 模式: fp32 / fp16 / int8 / mixed (由各模块组合推断)
    trt_precision: str
    # mixed 模式下保持 FP16 的 layer 子串 (其余 INT8)
    mixed_fp16_substr: tuple[str, ...]
    # W-only 是否对全网络生效 (任一 INT8 模块请求 W-only 即开)
    w_only: bool
    calibrator: str
    config_id: Optional[str] = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    # ---- 序列化: manifest (给硬件师) ----
    def to_manifest(self) -> dict:
        return {
            "config_id": self.config_id,
            "onnx_path": self.onnx_path,
            "engine_path": self.engine_path,
            "trt_precision": self.trt_precision,
            "mixed_fp16_substr": list(self.mixed_fp16_substr),
            "w_only": self.w_only,
            "calibrator": self.calibrator,
            "warnings": list(self.warnings),
            "modules": [asdict(m) for m in self.modules],
            "is_mixed_precision": self.trt_precision == "mixed",
            "int8_modules": [m.module for m in self.modules if m.is_int8()],
            "fp16_modules": [m.module for m in self.modules if m.is_fp16()],
        }

    def write_manifest(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_manifest(), f, ensure_ascii=False, indent=2)

    # ---- build 命令 (复用 m4_8_trt_build_bench.py, 不重造) ----
    def to_build_command(
        self,
        builder: str = "scripts/phase1/m4_8_trt_build_bench.py",
        python: str = sys.executable,
        report_path: Optional[str] = None,
        calib_data: Optional[str] = None,
        calib_cache: Optional[str] = None,
        input_shape: str = "1,64,256,256",
        extra: Optional[list[str]] = None,
    ) -> list[str]:
        """生成可执行的 TRT build 命令 (argv list).

        硬件师可直接 subprocess.run(cmd) 或拷贝到 shell.
        """
        cmd = [
            python, builder,
            "--onnx", self.onnx_path,
            "--engine", self.engine_path,
            "--report", report_path or (self.engine_path + ".report.json"),
            "--precision", self.trt_precision,
            "--input-shape", input_shape,
        ]
        if self.trt_precision in ("int8", "mixed"):
            cmd += ["--calibrator", self.calibrator]
            if calib_data:
                cmd += ["--calib-data", calib_data]
            if calib_cache:
                cmd += ["--calib-cache", calib_cache]
        if self.trt_precision == "mixed" and self.mixed_fp16_substr:
            cmd += ["--mixed-fp16-substr", ",".join(self.mixed_fp16_substr)]
        if self.w_only:
            cmd += ["--w-only"]
        if extra:
            cmd += list(extra)
        return cmd


# ===========================================================================
# §3 验证 + 解析 (核心逻辑)
# ===========================================================================

def _validate_module_request(module: str, bits: str, gran: str,
                             obj: str, calib: str) -> None:
    """系统边界输入验证 — fail fast (coding-style: 不信任外部数据)."""
    if bits not in Q_BITS_VALUES:
        raise ValueError(
            f"模块 {module!r}: q_bits={bits!r} 非法, 合法值 {Q_BITS_VALUES}")
    if gran not in Q_GRAN_VALUES:
        raise ValueError(
            f"模块 {module!r}: q_granularity={gran!r} 非法, 合法值 {Q_GRAN_VALUES}")
    if obj not in Q_OBJ_VALUES:
        raise ValueError(
            f"模块 {module!r}: q_object={obj!r} 非法, 合法值 {Q_OBJ_VALUES}")
    if calib not in Q_CALIBRATOR_VALUES:
        raise ValueError(
            f"模块 {module!r}: q_calibrator={calib!r} 非法, 合法值 {Q_CALIBRATOR_VALUES}")


def _module_is_sensitive(module: str, layer_substr: tuple[str, ...]) -> bool:
    """模块是否包含敏感层 (应 FP16-only)."""
    name_hit = any(s in module.lower() for s in SENSITIVE_FP16_SUBSTR)
    substr_hit = any(
        any(sens in ls.lower() for sens in SENSITIVE_FP16_SUBSTR)
        for ls in layer_substr
    )
    # "model" 是合并模块 (含 cls/reg head 但 99.9% 是 Conv backbone),
    # 不视为整体敏感 — 仅 head 子层在 mixed 模式下用 SENSITIVE_FP16_SUBSTR 保 FP16.
    if module == "model":
        return False
    return name_hit or substr_hit


def resolve_module_quant(
    module: str,
    bits: str,
    gran: str,
    obj: str,
    calib: str,
) -> ResolvedModuleQuant:
    """单模块: 把 Config 请求解析为应用所有硬约束后的实际量化决议."""
    _validate_module_request(module, bits, gran, obj, calib)

    notes: list[str] = []
    layer_substr = MODULE_LAYER_SUBSTR.get(module, (module.lower(),))
    forced_fp16 = False
    effective_bits = bits

    # ---- 约束 1: 敏感模块强制 FP16 (cls/reg/attention/MSDA/grid_sample) ----
    if bits == "INT8" and _module_is_sensitive(module, layer_substr):
        effective_bits = "FP16"
        forced_fp16 = True
        notes.append(
            f"敏感层约束: 模块 {module!r} 含 cls/reg/attention/MSDA, INT8→FP16 [论文+问题1]")

    # ---- 约束 2: entropy 校准禁用 → 回退 minmax ----
    eff_calib = calib
    if effective_bits == "INT8":
        if calib in FORBIDDEN_CALIBRATORS:
            eff_calib = "minmax"
            notes.append(
                f"校准器约束: {calib!r} 禁用 (自训 ckpt AP 崩塌), 回退 minmax [实测 问题1]")
        elif calib == "none":
            eff_calib = "minmax"
            notes.append("INT8 未指定校准器, fallback minmax (config_schema 约定)")
    else:
        eff_calib = "none"  # FP16/FP32 不需要校准器

    # ---- 约束 3: granularity. weight per-channel 合法; activation 恒 per-tensor ----
    if effective_bits == "INT8":
        gran_w = gran if gran in ("per-channel", "per-tensor") else "per-channel"
        if gran == "none":
            gran_w = "per-channel"  # TRT INT8 weight 默认 per-channel
            notes.append("INT8 weight 粒度未指定, 用 TRT 默认 per-channel")
        if obj in ("W+A", "W-only"):
            gran_a = ACTIVATION_GRANULARITY  # 恒 per-tensor
            if gran == "per-channel":
                notes.append(
                    "约束: activation 强制 per-tensor (GPU GEMM 不支持 per-channel A) [TRT 约束]")
        else:
            gran_a = "none"
    else:
        gran_w = "none"
        gran_a = "none"

    # ---- 约束 4: q_object. INT8 默认 W+A; W-only 走 --w-only flag ----
    if effective_bits == "INT8":
        eff_obj = obj if obj in ("W+A", "W-only") else "W+A"
        if obj == "none":
            eff_obj = "W+A"
            notes.append("INT8 未指定 q_object, 默认 W+A")
    else:
        eff_obj = "none"

    return ResolvedModuleQuant(
        module=module,
        requested_bits=bits,
        effective_bits=effective_bits,
        granularity_w=gran_w,
        granularity_a=gran_a,
        q_object=eff_obj,
        calibrator=eff_calib,
        layer_substr=layer_substr,
        forced_fp16=forced_fp16,
        notes=tuple(notes),
    )


def resolve_quant_plan(
    config: Config,
    onnx_path: str,
    engine_path: str,
    modules: Optional[tuple[str, ...]] = None,
) -> QuantPlan:
    """把整个 Config 解析为全网络量化计划 (支持分模块混合精度).

    Args:
        config: framework/config_schema.py 的 Config 对象
        onnx_path: FP32 ONNX 输入 (硬件师 build 用)
        engine_path: 目标 TRT engine 路径
        modules: 要量化的模块列表 (默认取 config 涉及的所有模块)
    """
    if modules is None:
        modules = tuple(config.modules()) or PYRAMID_M1_MODULES

    resolved: list[ResolvedModuleQuant] = []
    warnings: list[str] = []

    for m in modules:
        bits = config.q_bits.get(m, "FP16")
        gran = config.q_granularity.get(m, "none")
        obj = config.q_object.get(m, "none")
        calib = config.q_calibrator.get(m, "minmax")
        rmq = resolve_module_quant(m, bits, gran, obj, calib)
        resolved.append(rmq)
        for note in rmq.notes:
            if "禁用" in note or "强制" in note:
                warnings.append(f"[{m}] {note}")

    # ---- 推断全网络 TRT precision 模式 ----
    int8_mods = [r for r in resolved if r.is_int8()]
    fp16_mods = [r for r in resolved if r.is_fp16()]
    fp32_mods = [r for r in resolved if r.effective_bits == "FP32"]

    if int8_mods and (fp16_mods or fp32_mods):
        trt_precision = "mixed"  # 混合精度: 部分 INT8 部分 FP16
    elif int8_mods and not fp16_mods and not fp32_mods:
        trt_precision = "int8"
    elif fp32_mods and not int8_mods and not fp16_mods:
        trt_precision = "fp32"
    else:
        trt_precision = "fp16"

    # ---- mixed 模式: 收集应保持 FP16 的 layer 子串 ----
    #  = (被强制 FP16 的模块的 substr) ∪ (显式 FP16 模块的 substr) ∪ (全局敏感层 substr)
    mixed_fp16_substr: list[str] = []
    if trt_precision == "mixed":
        for r in resolved:
            if r.is_fp16():
                mixed_fp16_substr.extend(r.layer_substr)
        # 即使整个模块 INT8, 其内部 cls/reg/attention 子层也要 FP16
        mixed_fp16_substr.extend(SENSITIVE_FP16_SUBSTR)
        # 去重保序
        seen: set[str] = set()
        mixed_fp16_substr = [
            s for s in mixed_fp16_substr if not (s in seen or seen.add(s))
        ]

    # ---- W-only: 任一 INT8 模块请求 W-only 即开 (m4_8 是全局 flag) ----
    w_only = any(r.is_int8() and r.q_object == "W-only" for r in resolved)
    if w_only and any(r.is_int8() and r.q_object == "W+A" for r in resolved):
        warnings.append(
            "混合 W-only/W+A 请求: m4_8 --w-only 是全局 flag, 当前对全部 INT8 层生效 "
            "(per-module W-only 需 inject_qdq_from_config.py 路径, 见 dims 文档 §未完成项)")

    # ---- 全局校准器: 取 INT8 模块的 (m4_8 calibrator 是全局参数) ----
    calibrators = {r.calibrator for r in resolved if r.is_int8()}
    if len(calibrators) > 1:
        warnings.append(
            f"多个不同校准器 {calibrators}: m4_8 calibrator 是全局参数, "
            f"统一用 minmax (主路径)")
        global_calib = "minmax"
    elif calibrators:
        global_calib = calibrators.pop()
    else:
        global_calib = "none"

    return QuantPlan(
        modules=tuple(resolved),
        onnx_path=onnx_path,
        engine_path=engine_path,
        trt_precision=trt_precision,
        mixed_fp16_substr=tuple(mixed_fp16_substr),
        w_only=w_only,
        calibrator=global_calib,
        config_id=config.config_id,
        warnings=tuple(warnings),
    )


# ===========================================================================
# §4 CLI 入口
# ===========================================================================

def _load_config(path: str) -> Config:
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        return Config.from_yaml(p)
    if p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            return Config.from_dict(json.load(f))
    raise ValueError(f"不支持的 config 格式: {p.suffix} (用 .yaml / .json)")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="配置驱动的全网络混合精度量化工具 — 消费 Config, 产出 TRT build plan + manifest")
    ap.add_argument("--config", required=True,
                    help="framework Config (.yaml / .json), 含 per-module q_bits/q_granularity/...")
    ap.add_argument("--onnx", required=True, help="FP32 ONNX 输入路径")
    ap.add_argument("--engine", required=True, help="目标 TRT engine 路径")
    ap.add_argument("--manifest", default=None,
                    help="输出 manifest JSON (默认 <engine>.manifest.json)")
    ap.add_argument("--modules", nargs="+", default=None,
                    help="要量化的模块 (默认取 config 涉及的所有模块)")
    ap.add_argument("--calib-data", default=None, help="INT8 校准 numpy (N,C,H,W)")
    ap.add_argument("--calib-cache", default=None, help="校准 cache 文件")
    ap.add_argument("--input-shape", default="1,64,256,256")
    ap.add_argument("--print-cmd", action="store_true",
                    help="打印 TRT build 命令 (给硬件师执行)")
    args = ap.parse_args()

    config = _load_config(args.config)
    plan = resolve_quant_plan(
        config,
        onnx_path=args.onnx,
        engine_path=args.engine,
        modules=tuple(args.modules) if args.modules else None,
    )

    manifest_path = args.manifest or (args.engine + ".manifest.json")
    plan.write_manifest(manifest_path)

    print(f"[quant_config] config_id = {plan.config_id}")
    print(f"[quant_config] TRT precision 模式 = {plan.trt_precision}"
          + (" (混合精度)" if plan.trt_precision == "mixed" else ""))
    print(f"[quant_config] INT8 模块 = {[m.module for m in plan.modules if m.is_int8()]}")
    print(f"[quant_config] FP16 模块 = {[m.module for m in plan.modules if m.is_fp16()]}")
    print(f"[quant_config] W-only = {plan.w_only}, calibrator = {plan.calibrator}")
    if plan.mixed_fp16_substr:
        print(f"[quant_config] mixed FP16 保留层 substr = {list(plan.mixed_fp16_substr)}")
    for w in plan.warnings:
        print(f"[quant_config][WARN] {w}")
    print(f"[quant_config] manifest 已写 → {manifest_path}")

    if args.print_cmd:
        cmd = plan.to_build_command(
            calib_data=args.calib_data,
            calib_cache=args.calib_cache,
            input_shape=args.input_shape,
        )
        print("\n[quant_config] 给硬件师的 TRT build 命令:")
        print("  " + " ".join(cmd))


if __name__ == "__main__":
    main()
