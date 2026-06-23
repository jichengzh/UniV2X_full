"""deploy_config.py — 配置驱动的 TensorRT 部署工具 (D 维度 / hardware target).

角色: 硬件/TRT 部署工程师交付物 #1。

输入:
  - framework/config_schema.py 的 Config (d_routing / tactic / workspace + hardware target)
  - 量化师产出的 ONNX (FP32 或 explicit Q/DQ INT8)
  - configs/hardware/*.yaml 描述的目标硬件能力 (HardwareCapability)

能力:
  - ONNX → TRT engine build:
      * FP16 主路径 (至少跑通)
      * explicit QDQ INT8 路径 (检测 ONNX 内 QuantizeLinear/DequantizeLinear 节点自动启用)
      * implicit calibrator INT8 路径 (复用 build_trt_int8_univ2x 的 calibrator, 可选)
  - 按 Config.d_routing 配置 GPU / DLA 路由 (DLA 仅 Orin 等含 DLA 硬件可用)
  - 设置 tactic source / workspace
  - 自定义 plugin (MSDA / Rotate / Inverse / DCNv4) 强制 FP16, 不被 INT8 量化破坏
  - alignment_enforcement 由硬件 YAML 决定 (hard/soft/auto), 仅做记录不改图

输出:
  - TRT engine 文件
  - build report (JSON): 各 layer precision、INT8 layer 占比、fallback、engine size、build 时间

注意:
  - 本工具不重写 build_trt_int8_univ2x.py / build_trt_fp_univ2x.py, 而是复用其 calibrator
    与 _force_msda_fp16 思路, 收敛到单一配置驱动入口。
  - 真 build 不通时, report["status"]="failed" 并记录 parser/build 错误, 不假装成功。

CLI 示例 (FP16 explicit QDQ smoke):
    python tools/configurable/deploy_config.py \\
        --onnx onnx/univ2x_ego_bev_encoder_qdq.onnx \\
        --hw configs/hardware/rtx4090.yaml \\
        --plugin plugins/build/libuniv2x_plugins.so \\
        --engine trt_engines/bev_encoder_qdq_int8.engine \\
        --report results/deploy_bev_encoder_qdq.json \\
        --precision int8

也可从一个 Config YAML 驱动 (--config):
    python tools/configurable/deploy_config.py \\
        --config configs/deploy_example.yaml \\
        --onnx ... --hw ... --engine ... --report ...
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# 部署计划 (从 Config + HardwareCapability + ONNX 推导出的可执行 build 计划)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeployPlan:
    """配置驱动的 build 计划 — 不可变, 由 build_plan() 从契约推导。"""

    onnx_path: str
    engine_path: str
    hw_name: str
    arch: str
    # 精度
    want_fp16: bool = True
    want_int8: bool = False
    int8_mode: str = "explicit"          # "explicit" (Q/DQ in ONNX) | "implicit" (calibrator)
    has_qdq: bool = False
    qdq_count: int = 0
    # 路由
    routing: dict[str, str] = field(default_factory=dict)   # module -> GPU/DLA0/DLA1/CPU
    use_dla: bool = False
    dla_core: int = 0
    gpu_fallback: bool = True
    # tactic / workspace
    tactic_sources: tuple[str, ...] = ()                    # e.g. ("CUBLAS","CUDNN","EDGE_MASK_CONVOLUTIONS")
    workspace_gb: float = 4.0
    strongly_typed: bool = False
    # 对齐 (仅记录)
    alignment_enforcement: str = "hard"
    plugin_path: Optional[str] = None
    calib_data_path: Optional[str] = None
    calib_cache_path: Optional[str] = None


# ---------------------------------------------------------------------------
# 计划推导: Config + hw yaml + onnx → DeployPlan
# ---------------------------------------------------------------------------

def _detect_qdq(onnx_path: str) -> int:
    """统计 ONNX 内 QuantizeLinear / DequantizeLinear 节点数 (>0 → explicit INT8)。"""
    try:
        import onnx
    except ImportError:
        return 0
    m = onnx.load(onnx_path, load_external_data=False)
    return sum(1 for n in m.graph.node
               if n.op_type in ("QuantizeLinear", "DequantizeLinear"))


def build_plan(
    onnx_path: str,
    engine_path: str,
    hw_yaml: str,
    config=None,                 # framework.config_schema.Config | None
    precision: str = "auto",     # auto | fp16 | int8 | fp32
    workspace_gb: Optional[float] = None,
    tactic_sources: tuple[str, ...] = (),
    plugin_path: Optional[str] = None,
    calib_data_path: Optional[str] = None,
    calib_cache_path: Optional[str] = None,
) -> DeployPlan:
    """把共享契约 (Config + HardwareCapability YAML + ONNX) 推导成可执行 DeployPlan。

    精度决议:
      - precision=="auto": ONNX 有 Q/DQ → explicit int8; 否则 fp16
      - 若硬件不支持 INT8 (capability precisions 无 INT8) → 降级 fp16 并记录
    路由决议:
      - 取 Config.d_routing 的非 GPU 路由; 若硬件无 DLA → 强制 GPU 并记录
    """
    from framework.capability_schema import HardwareCapability

    cap = HardwareCapability.from_yaml(hw_yaml)
    supported = {p.upper() for p in cap.supported_precisions}
    has_dla = cap.has_dla

    qdq = _detect_qdq(onnx_path)
    has_qdq = qdq > 0

    # ---- 精度 ----
    want_int8 = False
    int8_mode = "explicit"
    if precision == "auto":
        want_int8 = has_qdq and ("INT8" in supported)
        int8_mode = "explicit" if has_qdq else "implicit"
    elif precision == "int8":
        want_int8 = "INT8" in supported
        int8_mode = "explicit" if has_qdq else "implicit"
    elif precision == "fp32":
        want_int8 = False
    # fp16: want_int8 False
    want_fp16 = precision != "fp32"  # fp16 always on except pure fp32 request

    # ---- 路由 (Config.d_routing) ----
    routing: dict[str, str] = {}
    if config is not None and getattr(config, "d_routing", None):
        routing = dict(config.d_routing)
    use_dla = any(r.startswith("DLA") for r in routing.values())
    dla_core = 0
    if use_dla:
        for r in routing.values():
            if r == "DLA1":
                dla_core = 1
                break
    if use_dla and not has_dla:
        # 硬件无 DLA → 强制 GPU
        routing = {k: ("GPU" if v.startswith("DLA") else v) for k, v in routing.items()}
        use_dla = False

    # ---- workspace ----
    if workspace_gb is None:
        workspace_gb = 4.0

    # ---- alignment ----
    align = getattr(cap.alignment, "alignment_enforcement", "hard")

    # ---- strongly typed (TRT 10 only, 4090) ----
    feat_extra = cap.features.model_extra or {}
    strongly_typed = bool(feat_extra.get("trt_strongly_typed", False))

    return DeployPlan(
        onnx_path=onnx_path,
        engine_path=engine_path,
        hw_name=cap.name,
        arch=cap.arch,
        want_fp16=want_fp16,
        want_int8=want_int8,
        int8_mode=int8_mode,
        has_qdq=has_qdq,
        qdq_count=qdq,
        routing=routing,
        use_dla=use_dla,
        dla_core=dla_core,
        gpu_fallback=True,
        tactic_sources=tuple(tactic_sources),
        workspace_gb=float(workspace_gb),
        strongly_typed=strongly_typed,
        alignment_enforcement=align,
        plugin_path=plugin_path,
        calib_data_path=calib_data_path,
        calib_cache_path=calib_cache_path,
    )


# ---------------------------------------------------------------------------
# Plugin 加载
# ---------------------------------------------------------------------------

def load_plugins(plugin_path: Optional[str], logger) -> list[str]:
    """加载自定义 plugin .so 并初始化 TRT plugin registry。返回可见的自定义 creator 名列表。"""
    import tensorrt as trt

    loaded = []
    if plugin_path and os.path.exists(plugin_path):
        ctypes.CDLL(os.path.abspath(plugin_path), mode=ctypes.RTLD_GLOBAL)
        loaded.append(plugin_path)
    trt.init_libnvinfer_plugins(logger, "")
    reg = trt.get_plugin_registry()
    visible = []
    interesting = ("MSDA", "MultiscaleDeform", "ModulatedDeformConv", "Rotate",
                   "Inverse", "DCNv4", "GridSample")
    for c in reg.all_creators:
        if any(k in c.name for k in interesting):
            visible.append(f"{c.name} v{getattr(c, 'plugin_version', '?')}")
    return sorted(set(visible))


# ---------------------------------------------------------------------------
# 核心 build
# ---------------------------------------------------------------------------

_TACTIC_MAP = {
    "CUBLAS": "CUBLAS",
    "CUBLAS_LT": "CUBLAS_LT",
    "CUDNN": "CUDNN",
    "EDGE_MASK_CONVOLUTIONS": "EDGE_MASK_CONVOLUTIONS",
    "JIT_CONVOLUTIONS": "JIT_CONVOLUTIONS",
}


def _apply_tactic_sources(config, plan: DeployPlan, trt) -> Optional[int]:
    if not plan.tactic_sources:
        return None
    mask = 0
    applied = []
    for name in plan.tactic_sources:
        attr = _TACTIC_MAP.get(name.upper())
        if attr and hasattr(trt.TacticSource, attr):
            mask |= 1 << int(getattr(trt.TacticSource, attr))
            applied.append(attr)
    if applied:
        config.set_tactic_sources(mask)
    return mask


def _force_plugins_fp16(network, trt) -> int:
    """统计自定义 plugin 层 (MSDA/Rotate/Inverse 等)。

    与 build_trt_int8_univ2x._force_msda_fp16 一致: TRT 默认不会把无 INT8 实现的
    PLUGIN_V2 层量化, 自动留在 FP16/FP32, 无需显式 layer.precision (显式反而插 Q/DQ 降精度)。
    """
    return sum(1 for i in range(network.num_layers)
               if network.get_layer(i).type == trt.LayerType.PLUGIN_V2)


def _collect_layer_precisions(engine_or_inspector, network, trt) -> dict:
    """从 network (build 前) 统计 layer 类型分布; precision 细节由 trtexec --dumpLayerInfo 补充。"""
    by_type: dict[str, int] = {}
    plugin = 0
    for i in range(network.num_layers):
        ly = network.get_layer(i)
        t = str(ly.type).split(".")[-1]
        by_type[t] = by_type.get(t, 0) + 1
        if ly.type == trt.LayerType.PLUGIN_V2:
            plugin += 1
    return {"num_layers": network.num_layers, "by_type": by_type, "plugin_layers": plugin}


def build_engine(plan: DeployPlan, verbose: bool = False) -> dict:
    """按 DeployPlan build TRT engine, 返回 build report dict。"""
    import tensorrt as trt

    report: dict = {
        "status": "unknown",
        "onnx": plan.onnx_path,
        "engine": plan.engine_path,
        "hardware": plan.hw_name,
        "arch": plan.arch,
        "trt_version": trt.__version__,
        "plan": {
            "want_fp16": plan.want_fp16,
            "want_int8": plan.want_int8,
            "int8_mode": plan.int8_mode,
            "has_qdq": plan.has_qdq,
            "qdq_count": plan.qdq_count,
            "routing": plan.routing,
            "use_dla": plan.use_dla,
            "dla_core": plan.dla_core,
            "tactic_sources": list(plan.tactic_sources),
            "workspace_gb": plan.workspace_gb,
            "strongly_typed": plan.strongly_typed,
            "alignment_enforcement": plan.alignment_enforcement,
        },
    }

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    report["plugins_visible"] = load_plugins(plan.plugin_path, logger)

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 int(plan.workspace_gb * 1024 ** 3))
    # DETAILED profiling verbosity → engine inspector 才能给出 per-layer precision
    try:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    except Exception:  # noqa: BLE001
        pass

    # ---- strongly-typed 决议 ----
    # 仅在 explicit-QDQ INT8 build 上启用 strongly_typed (TRT 10+)。
    # 根因修复: weakly-typed + FP16 flag + explicit Q/DQ 会让 TRT 在 MatMul 一侧
    # 插入 FP16 cast → 两个操作数的 DQ-scale 解析成不同 float 类型 (FP16 vs FP32) →
    # Myelin "type mismatch between matmul operand scales (LHS:2 RHS:1)" → INT8 tactic
    # 被跳过, 回退 FP16 (真实 INT8 覆盖率仅 10.8%)。
    # strongly-typed 让 TRT 完全按 ONNX 图内类型 (全 FP32 scale) 走, 不自动插 cast,
    # scale 类型一致 → INT8 tactic 不再被 reject。
    # NVIDIA 官方建议见 TensorRT issue #4050 (kSTRONGLY_TYPED 解 matmul scale 类型不匹配)。
    use_strongly_typed = bool(
        plan.strongly_typed
        and plan.want_int8
        and plan.int8_mode == "explicit"
        and plan.has_qdq
        and hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED")
    )
    report["strongly_typed_active"] = use_strongly_typed

    # ---- 精度 flags ----
    # strongly-typed 模式下类型完全由 ONNX 图决定, 不可 (也不应) set FP16/INT8 flag —
    # 否则 TRT 报错或忽略, 且重新引入 cast 不一致。
    int8_on = False
    if use_strongly_typed:
        # explicit Q/DQ 已编码 INT8; FP16 fallback 由图内 FP32 + plugin FP16 自然给出。
        report["fp16_enabled"] = bool(builder.platform_has_fast_fp16)
        int8_on = bool(plan.want_int8 and builder.platform_has_fast_int8)
        report["int8_enabled"] = int8_on
        if plan.want_int8 and not int8_on:
            report.setdefault("warnings", []).append(
                "INT8 requested but platform_has_fast_int8 == False — degraded to FP16")
    else:
        if plan.want_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            report["fp16_enabled"] = True
        else:
            report["fp16_enabled"] = False

        if plan.want_int8 and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            int8_on = True
        report["int8_enabled"] = int8_on
        if plan.want_int8 and not int8_on:
            report.setdefault("warnings", []).append(
                "INT8 requested but platform_has_fast_int8 == False — degraded to FP16")

    # ---- DLA 路由 ----
    if plan.use_dla:
        try:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = plan.dla_core
            if plan.gpu_fallback:
                config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            report["dla_configured"] = True
        except Exception as e:  # noqa: BLE001
            report.setdefault("warnings", []).append(f"DLA config failed: {e}")
            report["dla_configured"] = False

    # ---- tactic sources ----
    mask = _apply_tactic_sources(config, plan, trt)
    if mask is not None:
        report["tactic_mask"] = mask

    # ---- network 创建 (strongly typed 走不同路径) ----
    # strongly-typed: 类型完全由 ONNX 图决定, TRT 不自动插 FP16 cast,
    # explicit Q/DQ 的 matmul 两侧 DQ-scale 类型保持一致 (全 FP32) →
    # 消除 Myelin scale 类型不匹配, INT8 tactic 可被选中。
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if use_strongly_typed:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    t_parse = time.time()
    with open(plan.onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    report["parse_sec"] = round(time.time() - t_parse, 2)
    if not ok:
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        report["status"] = "failed"
        report["stage"] = "onnx_parse"
        report["errors"] = errs
        return report

    report["inputs"] = [network.get_input(i).name for i in range(network.num_inputs)]
    report["outputs"] = [network.get_output(i).name for i in range(network.num_outputs)]
    report["layer_stats"] = _collect_layer_precisions(None, network, trt)
    report["plugin_layers"] = _force_plugins_fp16(network, trt)

    # ---- INT8 implicit calibrator (仅当 ONNX 无 Q/DQ 且要 INT8) ----
    if int8_on and plan.int8_mode == "implicit":
        if plan.calib_data_path and os.path.exists(plan.calib_data_path):
            try:
                from tools.build_trt_int8_univ2x import _make_calibrator_class
                import pickle
                with open(plan.calib_data_path, "rb") as f:
                    raw = pickle.load(f)
                CalibCls = _make_calibrator_class()
                cache = plan.calib_cache_path or (
                    os.path.splitext(plan.engine_path)[0] + "_int8.cache")
                config.int8_calibrator = CalibCls(
                    cali_tensors=raw, input_names=report["inputs"], cache_file=cache)
                report["calibrator"] = "IInt8EntropyCalibrator2 (implicit)"
            except Exception as e:  # noqa: BLE001
                report.setdefault("warnings", []).append(
                    f"implicit calibrator setup failed: {e}; INT8 disabled")
                config.clear_flag(trt.BuilderFlag.INT8)
                report["int8_enabled"] = False
                int8_on = False
        else:
            report.setdefault("warnings", []).append(
                "INT8 implicit mode but no calib-data — INT8 disabled, FP16 only")
            config.clear_flag(trt.BuilderFlag.INT8)
            report["int8_enabled"] = False
            int8_on = False
    elif int8_on and plan.int8_mode == "explicit":
        report["calibrator"] = "NONE (explicit Q/DQ from ONNX)"

    # ---- build ----
    t_build = time.time()
    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except Exception as e:  # noqa: BLE001
        report["status"] = "failed"
        report["stage"] = "build"
        report["errors"] = [str(e)]
        return report
    report["build_sec"] = round(time.time() - t_build, 1)

    if engine_bytes is None:
        report["status"] = "failed"
        report["stage"] = "build"
        report["errors"] = ["build_serialized_network returned None"]
        return report

    os.makedirs(os.path.dirname(os.path.abspath(plan.engine_path)), exist_ok=True)
    with open(plan.engine_path, "wb") as f:
        f.write(memoryview(engine_bytes))
    size_mb = os.path.getsize(plan.engine_path) / 1024 ** 2
    report["engine_size_mb"] = round(size_mb, 1)
    report["status"] = "ok"

    # ---- 反序列化做 per-layer precision inspection ----
    try:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        inspector = engine.create_engine_inspector()
        info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        prec_count: dict[str, int] = {}
        import json as _json
        try:
            parsed = _json.loads(info)
            layers = parsed.get("Layers", []) if isinstance(parsed, dict) else parsed
            for ly in layers:
                if isinstance(ly, dict):
                    # TRT 10 DETAILED 格式: precision 在 Outputs[].Format/Datatype
                    # (例: "Half" / "Int8" / "Float" / "FP8"); 顶层 Precision 多为缺省。
                    p = ly.get("Precision")
                    if not p:
                        outs = ly.get("Outputs") or []
                        fmts = [o.get("Format/Datatype") or o.get("Format")
                                for o in outs if isinstance(o, dict)]
                        fmts = [f for f in fmts if f]
                        p = fmts[0] if fmts else "UNKNOWN"
                else:
                    p = "STRING_FORMAT"  # 非 DETAILED 时只有字符串名
                prec_count[p] = prec_count.get(p, 0) + 1
        except Exception:  # noqa: BLE001
            pass
        if prec_count:
            total = sum(prec_count.values())
            report["layer_precision_count"] = prec_count
            # Int8 在 TRT10 DETAILED JSON 里写作 "Int8"; 兼容大小写变体
            int8_layers = sum(v for k, v in prec_count.items()
                              if str(k).lower() in ("int8", "int8 (channel)"))
            report["int8_layer_pct"] = round(100.0 * int8_layers / total, 1) if total else 0.0
        report["engine_layers"] = engine.num_layers if hasattr(engine, "num_layers") else None
    except Exception as e:  # noqa: BLE001
        report.setdefault("warnings", []).append(f"inspector failed: {e}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="配置驱动 TRT 部署 build (D 维度)")
    p.add_argument("--onnx", required=True)
    p.add_argument("--hw", required=True, help="configs/hardware/*.yaml")
    p.add_argument("--engine", required=True, help="输出 TRT engine 路径")
    p.add_argument("--report", required=True, help="输出 build report JSON")
    p.add_argument("--config", default=None, help="可选: framework Config YAML (含 d_routing)")
    p.add_argument("--precision", choices=["auto", "fp16", "int8", "fp32"], default="auto")
    p.add_argument("--workspace-gb", type=float, default=None)
    p.add_argument("--tactic", default="", help="逗号分隔: CUBLAS,CUDNN,EDGE_MASK_CONVOLUTIONS")
    p.add_argument("--plugin", default="plugins/build/libuniv2x_plugins.so")
    p.add_argument("--calib-data", default=None, help="implicit INT8 calibration pkl")
    p.add_argument("--calib-cache", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = None
    if args.config:
        from framework.config_schema import Config
        cfg = Config.from_yaml(args.config)

    tactics = tuple(t.strip() for t in args.tactic.split(",") if t.strip())

    plan = build_plan(
        onnx_path=args.onnx,
        engine_path=args.engine,
        hw_yaml=args.hw,
        config=cfg,
        precision=args.precision,
        workspace_gb=args.workspace_gb,
        tactic_sources=tactics,
        plugin_path=args.plugin,
        calib_data_path=args.calib_data,
        calib_cache_path=args.calib_cache,
    )
    print(f"[plan] hw={plan.hw_name} arch={plan.arch} "
          f"fp16={plan.want_fp16} int8={plan.want_int8}({plan.int8_mode}) "
          f"qdq={plan.qdq_count} dla={plan.use_dla} ws={plan.workspace_gb}GB")

    report = build_engine(plan, verbose=args.verbose)

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[report] {args.report}  status={report['status']}")
    if report["status"] != "ok":
        print(f"[report] stage={report.get('stage')} errors={report.get('errors')}")
        return 1
    print(f"[report] engine={report.get('engine_size_mb')}MB "
          f"build={report.get('build_sec')}s "
          f"int8_layer_pct={report.get('int8_layer_pct')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
