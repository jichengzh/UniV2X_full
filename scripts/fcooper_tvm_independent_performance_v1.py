#!/usr/bin/env python3
"""Repeat F-Cooper TVM latency and energy on one frozen compiled artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.fcooper_tvm_evidence_pools_v1 import _database_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(path: Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def resolve_paths(feedback_path: Path) -> dict[str, Path]:
    feedback_path = _require(feedback_path)
    row = json.loads(feedback_path.read_text(encoding="utf-8"))
    module = _require(Path(str(row["tvm_artifact_path"])))
    if file_sha256(module) != row["tvm_artifact_sha256"]:
        raise ValueError("TVM module SHA drift")
    label_dir = module.parent
    q_mode = str(row["q_mode"])
    if q_mode == "int8":
        result = label_dir / "route_b_int8_auto_decomp_result.json"
        database = _database_path(row, module)
        runtime_weights = label_dir / "runtime_weights_int8.npz"
        quant_contract = Path(str(row["quant_contract_path"]))
        for path in (result, database, runtime_weights, quant_contract):
            _require(path)
        return {
            "feedback": feedback_path,
            "module": module,
            "database": database,
            "route_result": result,
            "runtime_weights": runtime_weights,
            "quant_contract": quant_contract.resolve(),
        }
    if q_mode not in {"fp16", "fp32"}:
        raise ValueError(f"unsupported q_mode: {q_mode}")
    result = _require(label_dir / f"route_b_{q_mode}_auto_result.json")
    database = _database_path(row, module)
    return {
        "feedback": feedback_path,
        "module": module,
        "database": database,
        "route_result": result,
    }


def _tvm_array(tvm: Any, dev: Any, value: np.ndarray) -> Any:
    constructor = getattr(tvm.runtime, "tensor", None)
    if constructor is not None:
        return constructor(value, device=dev)
    return tvm.nd.array(value, dev)


def _fp_runtime_args(tvm: Any, dev: Any, seed: int) -> list[Any]:
    generator = np.random.default_rng(seed)
    value = generator.normal(0.0, 0.5, size=(5, 64, 512, 512)).astype(np.float32)
    return [_tvm_array(tvm, dev, value)]


def _int8_runtime_args(
    tvm: Any,
    dev: Any,
    *,
    route_result: Path,
    runtime_weights: Path,
    seed: int,
) -> list[Any]:
    from scripts.fcooper_tvm_relax_worker_v1 import (
        runtime_weight_archive_key,
        validate_runtime_tensor,
    )

    result = json.loads(route_result.read_text(encoding="utf-8"))
    plan = [validate_runtime_tensor(item) for item in result["runtime_arg_plan"]]
    declared = result.get("runtime_weight_archive_keys") or {}
    generator = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    with np.load(runtime_weights, allow_pickle=False) as archive:
        archive_keys = set(archive.files)
        for item in plan:
            role = item["role"]
            if role == "graph_output":
                continue
            if role == "graph_input":
                value = generator.integers(
                    0,
                    256,
                    size=tuple(item["shape"]),
                    dtype=np.uint8,
                )
            else:
                key = runtime_weight_archive_key(
                    item,
                    declared=declared,
                    archive_keys=archive_keys,
                )
                value = np.ascontiguousarray(archive[key])
            values.append(value)
    return [_tvm_array(tvm, dev, value) for value in values]


def _path_sha(path: Path) -> str:
    if path.is_file():
        return file_sha256(path)
    rows = [
        {
            "path": str(item.relative_to(path)),
            "sha256": file_sha256(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def measure(args: argparse.Namespace) -> dict[str, Any]:
    import tvm
    from tvm import relax

    from scripts.fcooper_tvm_relax_worker_v1 import load_compiled_module
    from scripts.stage2_route_b_fp16_auto_runner import measure_compiled_vm_energy

    paths = resolve_paths(args.feedback_row)
    row = json.loads(paths["feedback"].read_text(encoding="utf-8"))
    dev = tvm.cuda(args.gpu_id)
    if hasattr(dev, "exist") and not dev.exist:
        raise RuntimeError(f"TVM CUDA device {args.gpu_id} is unavailable")
    library = load_compiled_module(tvm, paths["module"])
    vm = relax.VirtualMachine(library, dev)
    if row["q_mode"] == "int8":
        runtime_args = _int8_runtime_args(
            tvm,
            dev,
            route_result=paths["route_result"],
            runtime_weights=paths["runtime_weights"],
            seed=args.seed,
        )
    else:
        runtime_args = _fp_runtime_args(tvm, dev, args.seed)
    for _ in range(args.warmup):
        vm["main"](*runtime_args)
        dev.sync()
    evaluator = vm.time_evaluator(
        "main",
        dev,
        number=args.number,
        repeat=args.repeat,
    )
    samples_ms = [float(value) * 1000.0 for value in evaluator(*runtime_args).results]
    if not samples_ms or any(not math.isfinite(value) or value <= 0 for value in samples_ms):
        raise RuntimeError("latency evaluator returned invalid samples")
    energy = measure_compiled_vm_energy(
        label_dir=args.output_json.parent / "energy",
        vm=vm,
        dev=dev,
        runtime_args=runtime_args,
        gpu=str(args.physical_gpu_id),
        measure_iters=args.energy_iters,
    )
    if energy.get("status") != "success":
        raise RuntimeError(f"energy measurement failed: {energy}")
    energy_j = float(energy["joules_per_inference"])
    if not math.isfinite(energy_j) or energy_j <= 0:
        raise RuntimeError("energy measurement returned an invalid value")
    report = {
        "schema_version": "fcooper_tvm_independent_performance_v1",
        "row_id": row["row_id"],
        "gpu_index": int(args.physical_gpu_id),
        "q_mode": row["q_mode"],
        "latency_ms": float(statistics.median(samples_ms)),
        "latency_samples_ms": samples_ms,
        "energy_j": energy_j,
        "energy": energy,
        "warmup": args.warmup,
        "number": args.number,
        "repeat": args.repeat,
        "module_sha256": _path_sha(paths["module"]),
        "database_sha256": _path_sha(paths["database"]),
        "checkpoint_sha256": row["checkpoint_sha256"],
        "onnx_sha256": row["graph_features"]["onnx_sha256"],
    }
    if row["q_mode"] == "int8":
        report["quant_contract_sha256"] = _path_sha(paths["quant_contract"])
    write_exclusive(args.output_json, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-row", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--physical-gpu-id", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--number", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--energy-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    if min(args.warmup, args.number, args.repeat, args.energy_iters) <= 0:
        parser.error("measurement counts must be positive")
    return args


def main() -> int:
    report = measure(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
