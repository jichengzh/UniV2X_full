"""Generate model-independent ONNX probes for the Stage 2 S1 gate."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any


OPSET = 13
PRECISIONS = ("fp16", "int8")
MANIFEST_NAME = "probe_manifest.json"

PROBE_SPECS: dict[str, dict[str, Any]] = {
    "P1": {"description": "stride1", "input_shape": [1, 16, 16, 16], "output_shape": [1, 16, 16, 16]},
    "P2": {"description": "stride2", "input_shape": [1, 16, 16, 16], "output_shape": [1, 16, 8, 8]},
    "P3": {"description": "two_conv_chain", "input_shape": [1, 16, 16, 16], "output_shape": [1, 16, 16, 16]},
    "P4": {"description": "conv_residual_add", "input_shape": [1, 16, 16, 16], "output_shape": [1, 16, 16, 16]},
    "P5": {"description": "group_conv", "input_shape": [1, 16, 16, 16], "output_shape": [1, 16, 16, 16], "group": 4},
    "P6": {"description": "qdq_conv_chain", "input_shape": [1, 16, 16, 16], "output_shape": [1, 16, 16, 16]},
}


def load_onnx_package() -> Any:
    """Import ONNX even when the repository's ``onnx/`` asset dir shadows it."""
    repo_root = Path(__file__).resolve().parents[2]
    loaded = sys.modules.get("onnx")
    if loaded is not None and hasattr(loaded, "helper"):
        return loaded
    if loaded is not None:
        del sys.modules["onnx"]
    original_path = list(sys.path)
    try:
        sys.path[:] = [
            entry
            for entry in sys.path
            if entry and Path(entry).resolve() != repo_root
        ]
        return importlib.import_module("onnx")
    finally:
        sys.path[:] = original_path


def _tensor_dimensions(value_info: Any) -> list[int]:
    return [int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim]


def _make_initializer(onnx: Any, name: str, shape: list[int], dtype: str) -> Any:
    import numpy as np

    np_dtype = np.float16 if dtype == "fp16" else np.float32
    values = np.full(shape, 0.03125, dtype=np_dtype)
    return onnx.numpy_helper.from_array(values, name=name)


def _append_qdq(
    onnx: Any,
    nodes: list[Any],
    initializers: list[Any],
    source: str,
    prefix: str,
) -> str:
    import numpy as np

    scale_name = f"{prefix}_scale"
    zero_name = f"{prefix}_zero_point"
    quantized = f"{prefix}_quantized"
    dequantized = f"{prefix}_dequantized"
    initializers.extend(
        [
            onnx.numpy_helper.from_array(np.asarray(0.125, dtype=np.float32), name=scale_name),
            onnx.numpy_helper.from_array(np.asarray(0, dtype=np.int8), name=zero_name),
        ]
    )
    nodes.extend(
        [
            onnx.helper.make_node(
                "QuantizeLinear", [source, scale_name, zero_name], [quantized], name=f"{prefix}_Q"
            ),
            onnx.helper.make_node(
                "DequantizeLinear", [quantized, scale_name, zero_name], [dequantized], name=f"{prefix}_DQ"
            ),
        ]
    )
    return dequantized


def _append_conv(
    onnx: Any,
    nodes: list[Any],
    initializers: list[Any],
    *,
    source: str,
    output: str,
    name: str,
    precision: str,
    stride: int = 1,
    group: int = 1,
) -> None:
    channels = 16
    weight_name = f"{name}_weight"
    weight_shape = [channels, channels // group, 3, 3]
    initializers.append(_make_initializer(onnx, weight_name, weight_shape, precision))
    conv_source = source
    conv_weight = weight_name
    if precision == "int8":
        conv_source = _append_qdq(onnx, nodes, initializers, source, f"{name}_activation")
        conv_weight = _append_qdq(onnx, nodes, initializers, weight_name, f"{name}_weight")
    nodes.append(
        onnx.helper.make_node(
            "Conv",
            [conv_source, conv_weight],
            [output],
            name=name,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[stride, stride],
            group=group,
        )
    )


def build_probe_model(probe_id: str, precision: str) -> Any:
    """Build and checker-validate one fixed-shape neutral probe."""
    if probe_id not in PROBE_SPECS:
        raise ValueError(f"unknown probe_id: {probe_id}")
    precision = "int8" if precision == "int8_qdq" else precision
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision: {precision}")

    onnx = load_onnx_package()
    spec = PROBE_SPECS[probe_id]
    elem_type = onnx.TensorProto.FLOAT16 if precision == "fp16" else onnx.TensorProto.FLOAT
    nodes: list[Any] = []
    initializers: list[Any] = []

    if probe_id == "P2":
        _append_conv(onnx, nodes, initializers, source="input", output="output", name="conv", precision=precision, stride=2)
    elif probe_id == "P3":
        _append_conv(onnx, nodes, initializers, source="input", output="hidden", name="conv1", precision=precision)
        _append_conv(onnx, nodes, initializers, source="hidden", output="output", name="conv2", precision=precision)
    elif probe_id == "P4":
        _append_conv(onnx, nodes, initializers, source="input", output="conv_output", name="conv", precision=precision)
        nodes.append(onnx.helper.make_node("Add", ["conv_output", "input"], ["output"], name="residual_add"))
    elif probe_id == "P5":
        _append_conv(
            onnx, nodes, initializers, source="input", output="output", name="group_conv",
            precision=precision, group=int(spec["group"]),
        )
    elif probe_id == "P6" and precision == "int8":
        _append_conv(
            onnx, nodes, initializers, source="input", output="conv_output", name="conv",
            precision=precision,
        )
        qdq_output = _append_qdq(onnx, nodes, initializers, "conv_output", "output_boundary")
        nodes.append(onnx.helper.make_node("Identity", [qdq_output], ["output"], name="output_identity"))
    else:
        _append_conv(onnx, nodes, initializers, source="input", output="output", name="conv", precision=precision)

    graph = onnx.helper.make_graph(
        nodes,
        f"stage2_s1_{probe_id.lower()}_{precision}",
        [onnx.helper.make_tensor_value_info("input", elem_type, spec["input_shape"])],
        [onnx.helper.make_tensor_value_info("output", elem_type, spec["output_shape"])],
        initializer=initializers,
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="stage2_s1_neutral_probes_v3",
        opset_imports=[onnx.helper.make_opsetid("", OPSET)],
    )
    model.ir_version = min(int(model.ir_version), 8)
    onnx.checker.check_model(model)
    return model


def generate_neutral_probes(output_dir: str | Path) -> Path:
    """Write all probe variants and their content-addressed manifest."""
    onnx = load_onnx_package()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for probe_id, spec in PROBE_SPECS.items():
        for precision in PRECISIONS:
            filename = f"{probe_id.lower()}_{precision}.onnx"
            path = destination / filename
            model = build_probe_model(probe_id, precision)
            onnx.save(model, str(path))
            onnx.checker.check_model(onnx.load(str(path)))
            records.append(
                {
                    "probe_id": probe_id,
                    "description": spec["description"],
                    "precision": precision,
                    "path": filename,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                    "input_shape": list(spec["input_shape"]),
                    "output_shape": list(spec["output_shape"]),
                    "checker_status": "passed",
                    "opset": OPSET,
                }
            )

    manifest = {"schema": "stage2_s1_neutral_probes_v3", "probe_count": len(records), "probes": records}
    manifest_path = destination / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
