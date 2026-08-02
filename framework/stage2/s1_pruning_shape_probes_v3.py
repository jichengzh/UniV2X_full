"""Generate model-independent ONNX probes for S1-P pruning shape effects."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any


OPSET = 13
ALIGNMENT_MULTIPLES = {"fp16": 8, "int8_qdq": 16}
PRECISIONS = ("fp16", "int8_qdq")

# These dimensions are synthetic constants. They are not derived from model assets,
# partition manifests, measurements, or prior experiment outputs.
PROBE_SPECS: dict[str, dict[str, Any]] = {
    "aligned_channels": {
        "input_shape": [1, 64, 8, 8],
        "output_shape": [1, 96, 8, 8],
        "group": 1,
    },
    "misaligned_channels": {
        "input_shape": [1, 30, 8, 8],
        "output_shape": [1, 46, 8, 8],
        "group": 1,
    },
    "small_channels": {
        "input_shape": [1, 4, 8, 8],
        "output_shape": [1, 6, 8, 8],
        "group": 1,
    },
    "group_packed": {
        "input_shape": [1, 64, 8, 8],
        "output_shape": [1, 96, 8, 8],
        "group": 4,
    },
    "group_unpacked": {
        "input_shape": [1, 24, 8, 8],
        "output_shape": [1, 40, 8, 8],
        "group": 4,
    },
    "off_diagonal_two_stage_boundary": {
        "input_shape": [1, 32, 8, 8],
        "boundary_shape": [1, 48, 8, 8],
        "output_shape": [1, 64, 8, 8],
        "group": 1,
    },
}


def load_onnx_package() -> Any:
    """Import ONNX when the repository's asset directory shadows the package."""
    repo_root = Path(__file__).resolve().parents[2]
    loaded = sys.modules.get("onnx")
    if loaded is not None and hasattr(loaded, "helper"):
        return loaded
    if loaded is not None:
        del sys.modules["onnx"]
    original_path = list(sys.path)
    try:
        sys.path[:] = [
            entry for entry in sys.path if entry and Path(entry).resolve() != repo_root
        ]
        return importlib.import_module("onnx")
    finally:
        sys.path[:] = original_path


def _initializer(onnx: Any, name: str, shape: list[int], precision: str) -> Any:
    import numpy as np

    dtype = np.float16 if precision == "fp16" else np.float32
    return onnx.numpy_helper.from_array(
        np.full(shape, 0.03125, dtype=dtype), name=name
    )


def _append_qdq(
    onnx: Any,
    nodes: list[Any],
    initializers: list[Any],
    source: str,
    prefix: str,
) -> str:
    import numpy as np

    scale = f"{prefix}_scale"
    zero_point = f"{prefix}_zero_point"
    quantized = f"{prefix}_quantized"
    dequantized = f"{prefix}_dequantized"
    initializers.extend(
        [
            onnx.numpy_helper.from_array(np.asarray(0.125, dtype=np.float32), name=scale),
            onnx.numpy_helper.from_array(np.asarray(0, dtype=np.int8), name=zero_point),
        ]
    )
    nodes.extend(
        [
            onnx.helper.make_node(
                "QuantizeLinear", [source, scale, zero_point], [quantized], name=f"{prefix}_Q"
            ),
            onnx.helper.make_node(
                "DequantizeLinear",
                [quantized, scale, zero_point],
                [dequantized],
                name=f"{prefix}_DQ",
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
    input_channels: int,
    output_channels: int,
    group: int,
    precision: str,
) -> None:
    weight = f"{name}_weight"
    initializers.append(
        _initializer(
            onnx,
            weight,
            [output_channels, input_channels // group, 3, 3],
            precision,
        )
    )
    conv_source = source
    conv_weight = weight
    if precision == "int8_qdq":
        conv_source = _append_qdq(onnx, nodes, initializers, source, f"{name}_activation")
        conv_weight = _append_qdq(onnx, nodes, initializers, weight, f"{name}_weight")
    nodes.append(
        onnx.helper.make_node(
            "Conv",
            [conv_source, conv_weight],
            [output],
            name=name,
            group=group,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    )


def build_probe_model(probe_id: str, precision: str) -> Any:
    """Build and ONNX-check one fixed synthetic probe."""
    if probe_id not in PROBE_SPECS:
        raise ValueError(f"unknown probe_id: {probe_id}")
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision: {precision}")

    onnx = load_onnx_package()
    spec = PROBE_SPECS[probe_id]
    input_channels = int(spec["input_shape"][1])
    output_channels = int(spec["output_shape"][1])
    elem_type = onnx.TensorProto.FLOAT16 if precision == "fp16" else onnx.TensorProto.FLOAT
    nodes: list[Any] = []
    initializers: list[Any] = []

    if "boundary_shape" in spec:
        boundary_channels = int(spec["boundary_shape"][1])
        _append_conv(
            onnx,
            nodes,
            initializers,
            source="input",
            output="stage1_output",
            name="stage1_conv",
            input_channels=input_channels,
            output_channels=boundary_channels,
            group=1,
            precision=precision,
        )
        nodes.append(
            onnx.helper.make_node("Relu", ["stage1_output"], ["stage_boundary"], name="boundary_relu")
        )
        _append_conv(
            onnx,
            nodes,
            initializers,
            source="stage_boundary",
            output="output",
            name="stage2_conv",
            input_channels=boundary_channels,
            output_channels=output_channels,
            group=1,
            precision=precision,
        )
    else:
        _append_conv(
            onnx,
            nodes,
            initializers,
            source="input",
            output="output",
            name="conv",
            input_channels=input_channels,
            output_channels=output_channels,
            group=int(spec["group"]),
            precision=precision,
        )

    graph = onnx.helper.make_graph(
        nodes,
        f"stage2_s1_pruning_shape_{probe_id}_{precision}",
        [onnx.helper.make_tensor_value_info("input", elem_type, spec["input_shape"])],
        [onnx.helper.make_tensor_value_info("output", elem_type, spec["output_shape"])],
        initializer=initializers,
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="stage2_s1_pruning_shape_probes_v3",
        opset_imports=[onnx.helper.make_opsetid("", OPSET)],
    )
    model.ir_version = min(int(model.ir_version), 8)
    onnx.checker.check_model(model)
    return model


def _alignment_metadata(spec: dict[str, Any], precision: str) -> dict[str, Any]:
    group = int(spec["group"])
    input_channels = int(spec["input_shape"][1])
    output_channels = int(spec["output_shape"][1])
    alignment_multiple = ALIGNMENT_MULTIPLES[precision]
    metadata: dict[str, Any] = {
        "channel_multiple": alignment_multiple,
        "input_channels_aligned": input_channels % alignment_multiple == 0,
        "output_channels_aligned": output_channels % alignment_multiple == 0,
        "group": group,
        "input_channels_per_group": input_channels // group,
        "output_channels_per_group": output_channels // group,
        "input_per_group_aligned": (input_channels // group) % alignment_multiple == 0,
        "output_per_group_aligned": (output_channels // group) % alignment_multiple == 0,
    }
    if "boundary_shape" in spec:
        boundary_channels = int(spec["boundary_shape"][1])
        metadata.update(
            {
                "stage_boundary_channels_aligned": boundary_channels % alignment_multiple == 0,
                "off_diagonal": len({input_channels, boundary_channels, output_channels}) == 3,
            }
        )
    return metadata


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_pruning_shape_probes(output_dir: str | Path) -> Path:
    """Write twelve checked models and return the content-addressed manifest."""
    onnx = load_onnx_package()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for probe_id, spec in PROBE_SPECS.items():
        for precision in PRECISIONS:
            model = build_probe_model(probe_id, precision)
            serialized = model.SerializeToString()
            digest = hashlib.sha256(serialized).hexdigest()
            filename = f"{probe_id}-{precision}-{digest}.onnx"
            artifact_path = destination / filename
            artifact_path.write_bytes(serialized)
            onnx.checker.check_model(onnx.load(str(artifact_path)))
            shape = {
                "input": list(spec["input_shape"]),
                "output": list(spec["output_shape"]),
            }
            if "boundary_shape" in spec:
                shape["stage_boundary"] = list(spec["boundary_shape"])
            records.append(
                {
                    "probe_id": probe_id,
                    "precision": precision,
                    "artifact": {
                        "path": filename,
                        "sha256": digest,
                        "size_bytes": len(serialized),
                    },
                    "checker": {"name": "onnx.checker", "status": "passed"},
                    "shape": shape,
                    "alignment": _alignment_metadata(spec, precision),
                    "opset": OPSET,
                }
            )

    manifest = {
        "schema": "stage2_s1_pruning_shape_probes_v3",
        "source": "fixed_synthetic_specs",
        "probe_count": len(records),
        "probes": records,
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_digest = hashlib.sha256(payload).hexdigest()
    manifest_path = destination / f"manifest-{manifest_digest}.json"
    manifest_path.write_bytes(payload)
    return manifest_path


def check_probe_manifest(manifest_path: str | Path) -> bool:
    """Verify manifest addressing, artifact hashes, and ONNX checker validity."""
    onnx = load_onnx_package()
    path = Path(manifest_path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if path.name != f"manifest-{digest}.json":
        raise ValueError("manifest sha256 mismatch")
    manifest = json.loads(payload)
    if manifest.get("schema") != "stage2_s1_pruning_shape_probes_v3":
        raise ValueError("unsupported manifest schema")
    probes = manifest.get("probes")
    if not isinstance(probes, list) or manifest.get("probe_count") != len(probes):
        raise ValueError("invalid probe count")

    for record in probes:
        artifact = record["artifact"]
        artifact_path = path.parent / artifact["path"]
        if _sha256(artifact_path) != artifact["sha256"]:
            raise ValueError(f"sha256 mismatch: {artifact_path.name}")
        if artifact_path.stat().st_size != artifact["size_bytes"]:
            raise ValueError(f"size mismatch: {artifact_path.name}")
        onnx.checker.check_model(onnx.load(str(artifact_path)))
    return True
