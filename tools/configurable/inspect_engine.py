"""inspect_engine.py — TRT engine layer precision 报告 (trtexec --dumpLayerInfo 等价物)。

trtexec 二进制本环境未安装; 用 TRT Python EngineInspector + DETAILED JSON 复现
per-layer precision 统计。读取 Outputs[].Format/Datatype 字段 (TRT10 格式)。

用法:
    python tools/configurable/inspect_engine.py \\
        --engine trt_engines/smoke_bev_encoder_fp16.engine \\
        --plugin plugins/build/libuniv2x_plugins.so
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys


def inspect(engine_path: str, plugin_path: str | None) -> dict:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    if plugin_path and os.path.exists(plugin_path):
        ctypes.CDLL(os.path.abspath(plugin_path), mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(logger, "")

    rt = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = rt.deserialize_cuda_engine(f.read())
    insp = engine.create_engine_inspector()
    info = insp.get_engine_information(trt.LayerInformationFormat.JSON)
    data = json.loads(info)
    layers = data["Layers"] if isinstance(data, dict) else data

    prec: dict[str, int] = {}
    ltype: dict[str, int] = {}
    for ly in layers:
        if isinstance(ly, dict):
            lt = ly.get("LayerType", "?")
            ltype[lt] = ltype.get(lt, 0) + 1
            outs = ly.get("Outputs") or []
            fmts = [o.get("Format/Datatype") or o.get("Format")
                    for o in outs if isinstance(o, dict)]
            fmts = [f for f in fmts if f]
            p = ly.get("Precision") or (fmts[0] if fmts else "UNKNOWN")
        else:
            p = "STRING_FORMAT"
            ltype["string"] = ltype.get("string", 0) + 1
        prec[p] = prec.get(p, 0) + 1

    total = sum(prec.values())
    int8 = sum(v for k, v in prec.items() if str(k).lower().startswith("int8"))
    return {
        "engine": engine_path,
        "engine_size_mb": round(os.path.getsize(engine_path) / 1024 ** 2, 1),
        "total_layers": total,
        "precision_count": dict(sorted(prec.items(), key=lambda x: -x[1])),
        "int8_layer_pct": round(100.0 * int8 / total, 1) if total else 0.0,
        "layer_type_count": dict(sorted(ltype.items(), key=lambda x: -x[1])),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    p.add_argument("--plugin", default="plugins/build/libuniv2x_plugins.so")
    p.add_argument("--out", default=None, help="可选: 写 JSON")
    a = p.parse_args()
    rep = inspect(a.engine, a.plugin)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
