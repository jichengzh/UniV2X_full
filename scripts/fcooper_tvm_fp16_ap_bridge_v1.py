#!/usr/bin/env python3
"""Evaluate F-Cooper OPV2V AP through a compiled TVM Relax VM FP16 dense body."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.fcooper_prediction_artifact_v1 import PredictionArtifactWriter

from scripts.fcooper_tvm_subprocess_runner_v1 import (
    SharedMemoryTvmClient,
    build_worker_env,
)


SCHEMA_BY_COMPUTE_DTYPE = {
    "float16": "fcooper_tvm_fp16_ap_report_v1",
    "float32": "fcooper_tvm_fp32_ap_report_v1",
}
PIPELINE_SCOPE = "fcooper_post_scatter_backbone_m1_plus_shrinker_m1"


def sha256_path(path: Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    root = path.parent if path.is_file() else path
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}") from exc
    if not shape or any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError("shape must contain positive static dimensions")
    return shape


@dataclass(frozen=True)
class StaticTensorContract:
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]

    @classmethod
    def from_shapes(
        cls,
        *,
        input_shapes: Iterable[Sequence[int]],
        output_shapes: Iterable[Sequence[int]],
    ) -> "StaticTensorContract":
        inputs = [tuple(int(value) for value in shape) for shape in input_shapes]
        outputs = [tuple(int(value) for value in shape) for shape in output_shapes]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("F-Cooper TVM artifact must expose exactly one input and one output")
        if any(value <= 0 for shape in (*inputs, *outputs) for value in shape):
            raise ValueError("F-Cooper TVM AP bridge requires positive static dimensions")
        if len(inputs[0]) != 4 or len(outputs[0]) != 4:
            raise ValueError("F-Cooper dense-body input and output must be rank-4 NCHW tensors")
        if inputs[0][0] != outputs[0][0]:
            raise ValueError("F-Cooper dense-body input and output batch dimensions must match")
        return cls(input_shape=inputs[0], output_shape=outputs[0])


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    if hasattr(value, "asnumpy"):
        return np.asarray(value.asnumpy())
    return np.asarray(value)


def _single_vm_output(value: Any) -> np.ndarray:
    if hasattr(value, "numpy") or hasattr(value, "asnumpy") or isinstance(value, np.ndarray):
        return _as_numpy(value)
    if isinstance(value, (list, tuple)):
        outputs = list(value)
    else:
        try:
            outputs = [value[index] for index in range(len(value))]
        except TypeError as exc:
            raise ValueError("TVM Relax VM result is not a tensor or output tuple") from exc
    if len(outputs) != 1:
        raise ValueError(f"F-Cooper TVM artifact returned {len(outputs)} outputs; expected exactly one")
    return _as_numpy(outputs[0])


class TvmRelaxVmRunner:
    """Strict static-shape runner for one compiled Relax VM ``main`` function."""

    def __init__(
        self,
        artifact_path: Path,
        contract: StaticTensorContract,
        *,
        gpu_id: int = 0,
        output_dtype: str = "float16",
    ) -> None:
        artifact_path = Path(artifact_path)
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        self.artifact_path = artifact_path
        self.input_shape = contract.input_shape
        self.output_shape = contract.output_shape
        if output_dtype not in {"float16", "float32"}:
            raise ValueError(f"unsupported TVM artifact output dtype: {output_dtype}")
        self.output_dtype = output_dtype
        self.gpu_id = int(gpu_id)
        self.call_count = 0
        self._tvm, self._device, self._main = self._load_backend()

    def _load_backend(self) -> tuple[Any, Any, Any]:
        try:
            import tvm
            from tvm import relax
        except ImportError as exc:
            raise RuntimeError("TVM is required to execute the F-Cooper Relax VM artifact") from exc
        device = tvm.cuda(self.gpu_id)
        if hasattr(device, "exist") and not device.exist:
            raise RuntimeError(f"TVM CUDA device {self.gpu_id} is unavailable")
        library = tvm.runtime.load_module(str(self.artifact_path))
        vm = relax.VirtualMachine(library, device)
        return tvm, device, vm["main"]

    def _make_tvm_array(self, value: np.ndarray) -> Any:
        tensor_constructor = getattr(self._tvm.runtime, "tensor", None)
        if tensor_constructor is not None:
            return tensor_constructor(value, device=self._device)
        return self._tvm.nd.array(value, self._device)

    def _execute_host_input(self, host_input: np.ndarray) -> np.ndarray:
        if host_input.dtype != np.float32:
            raise ValueError(f"TVM input dtype drift: {host_input.dtype} != float32")
        self.call_count += 1
        result = self._main(self._make_tvm_array(host_input))
        self._device.sync()
        host_output = np.ascontiguousarray(_single_vm_output(result))
        if tuple(host_output.shape) != self.output_shape:
            raise ValueError(
                f"TVM output shape drift: {tuple(host_output.shape)} != {self.output_shape}"
            )
        expected_dtype = getattr(self, "output_dtype", "float16")
        if str(host_output.dtype) != expected_dtype:
            raise ValueError(
                f"TVM output dtype drift: {host_output.dtype} != {expected_dtype}"
            )
        return host_output

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        if tuple(source.shape) != self.input_shape:
            raise ValueError(f"TVM input shape drift: {tuple(source.shape)} != {self.input_shape}")
        if not source.is_cuda:
            raise ValueError("F-Cooper TVM AP bridge requires a CUDA torch input tensor")
        host_input = np.ascontiguousarray(source.detach().to("cpu", dtype=torch.float32).numpy())
        host_output = self._execute_host_input(host_input)
        return torch.from_numpy(host_output).to(device=source.device)


class SubprocessTvmRelaxVmRunner:
    """Run TVM in its Python 3.10 ABI while evaluation remains in Python 3.9."""

    def __init__(
        self,
        artifact_path: Path,
        contract: StaticTensorContract,
        *,
        precision: str,
        artifact_output_dtype: str,
        gpu_id: int,
        worker_python: Path,
        tvm_site: Path,
        tvm_lib_dirs: Sequence[Path],
        workspace: Path,
    ) -> None:
        self.input_shape = contract.input_shape
        self.output_shape = contract.output_shape
        output_dtype = artifact_output_dtype
        worker = Path(__file__).with_name("fcooper_tvm_relax_worker_v1.py")
        command = [
            str(worker_python),
            str(worker),
            "--mode",
            precision,
            "--artifact",
            str(artifact_path),
            "--input-shape",
            ",".join(map(str, self.input_shape)),
            "--output-shape",
            ",".join(map(str, self.output_shape)),
            "--input-dtype",
            "float32",
            "--output-dtype",
            output_dtype,
            "--gpu-id",
            "0",
        ]
        self._client = SharedMemoryTvmClient(
            worker_command=command,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            input_dtype="float32",
            output_dtype=output_dtype,
            workspace=workspace,
            env=build_worker_env(
                tvm_site=tvm_site,
                tvm_lib_dirs=tvm_lib_dirs,
                gpu_id=gpu_id,
            ),
        )

    @property
    def call_count(self) -> int:
        return self._client.call_count

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        if tuple(source.shape) != self.input_shape:
            raise ValueError(f"TVM input shape drift: {tuple(source.shape)} != {self.input_shape}")
        if not source.is_cuda:
            raise ValueError("F-Cooper TVM AP bridge requires a CUDA torch input tensor")
        host_input = np.ascontiguousarray(source.detach().to("cpu", dtype=torch.float32).numpy())
        host_output = self._client.execute(host_input)
        return torch.from_numpy(host_output).to(device=source.device)

    def close(self) -> None:
        self._client.close()


class TvmDenseBody(nn.Module):
    """Match the TRT bridge's zero-pad, execute, unpad dense-body protocol."""

    def __init__(self, runner: Any) -> None:
        super().__init__()
        self.runner = runner

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        source = data["spatial_features"]
        agents = int(source.shape[0])
        expected_tail = tuple(self.runner.input_shape[1:])
        if tuple(source.shape[1:]) != expected_tail:
            raise ValueError(
                f"dense-body input shape drift: {tuple(source.shape[1:])} != {expected_tail}"
            )
        artifact_batch = int(self.runner.input_shape[0])
        if agents > artifact_batch:
            raise ValueError(f"record has {agents} agents, artifact supports {artifact_batch}")
        padded = torch.zeros(
            self.runner.input_shape,
            device=source.device,
            dtype=source.dtype,
        )
        padded[:agents] = source
        encoded = self.runner(padded)
        if tuple(encoded.shape) != tuple(self.runner.output_shape):
            raise ValueError(
                f"dense-body output shape drift: {tuple(encoded.shape)} "
                f"!= {tuple(self.runner.output_shape)}"
            )
        return {"spatial_features_2d": encoded[:agents].float()}


class IdentityShrinker(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def evaluate_execution_gates(
    *,
    dataset_samples: int,
    requested_samples: int,
    processed_samples: int,
    failed_samples: int,
    backend_calls: int,
    fallback_samples: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    if fallback_samples:
        blockers.append("fallback_forbidden")
    if failed_samples:
        blockers.append("failed_samples_present")
    if backend_calls != processed_samples:
        blockers.append("backend_calls_mismatch")
    if processed_samples != requested_samples:
        blockers.append("requested_samples_not_processed")
    full = (
        requested_samples == dataset_samples
        and processed_samples == dataset_samples
        and not blockers
    )
    sanity = processed_samples > 0 and processed_samples == requested_samples and not blockers
    return {
        "status": "success_full" if full else "success_sanity" if sanity else "failure",
        "sanity": sanity,
        "full": full,
        "blockers": blockers,
    }


def _optional_provenance(paths: dict[str, Path | None]) -> tuple[dict[str, str], dict[str, str]]:
    recorded_paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for name, candidate in paths.items():
        if candidate is None:
            continue
        path = Path(candidate)
        recorded_paths[name] = str(path)
        if path.exists():
            hashes[name] = sha256_path(path)
    return recorded_paths, hashes


def build_report(
    *,
    dataset_samples: int,
    requested_samples: int,
    processed_samples: int,
    failed_samples: int,
    backend_calls: int,
    fallback_samples: int,
    ap30: float,
    ap50: float,
    ap70: float,
    elapsed_seconds: float,
    artifact_path: Path,
    checkpoint_path: Path | None,
    config_path: Path,
    prediction_path: Path | None,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    artifact_compute_dtype: str = "float16",
    artifact_output_dtype: str | None = None,
) -> dict[str, Any]:
    gates = evaluate_execution_gates(
        dataset_samples=dataset_samples,
        requested_samples=requested_samples,
        processed_samples=processed_samples,
        failed_samples=failed_samples,
        backend_calls=backend_calls,
        fallback_samples=fallback_samples,
    )
    paths, hashes = _optional_provenance(
        {
            "artifact": artifact_path,
            "checkpoint": checkpoint_path,
            "config": config_path,
            "prediction": prediction_path,
        }
    )
    resolved_output_dtype = artifact_output_dtype or artifact_compute_dtype
    schema_version = SCHEMA_BY_COMPUTE_DTYPE.get(artifact_compute_dtype)
    if schema_version is None:
        raise ValueError(
            f"unsupported artifact compute dtype: {artifact_compute_dtype}"
        )
    report = {
        "schema_version": schema_version,
        "status": gates["status"],
        "dataset": "OPV2V",
        "split": "test",
        "pipeline_scope": PIPELINE_SCOPE,
        "dataset_samples": int(dataset_samples),
        "requested_samples": int(requested_samples),
        "processed_samples": int(processed_samples),
        "failed_samples": int(failed_samples),
        "backend_samples": int(processed_samples),
        "backend_calls": int(backend_calls),
        "fallback_samples": int(fallback_samples),
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "elapsed_seconds": float(elapsed_seconds),
        "execution_device": {
            "physical_gpu_id": (
                int(os.environ["CUDA_VISIBLE_DEVICES"])
                if os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit()
                else None
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "paths": paths,
        "sha256": hashes,
        "gates": gates,
        "numerical_contract": {
            "io_cardinality": {"inputs": 1, "outputs": 1},
            "input_shape": [int(value) for value in input_shape],
            "output_shape": [int(value) for value in output_shape],
            "shapes_are_static": True,
            "agent_batch_padding": "zero_pad_to_artifact_batch_then_unpad_output",
            "artifact_compute_dtype": artifact_compute_dtype,
            "artifact_input_dtype": "float32",
            "artifact_input_conversion": "cuda_torch_to_host_fp32_to_tvm_cuda",
            "artifact_output_conversion": (
                f"tvm_cuda_to_host_{resolved_output_dtype}_to_cuda_torch"
            ),
            "artifact_output_dtype": resolved_output_dtype,
            "model_boundary_output_dtype": "float32",
            "host_roundtrip_permitted_for_ap_correctness": True,
            "one_backend_call_per_processed_sample": True,
            "silent_fallback_forbidden": True,
            "fallback_samples": int(fallback_samples),
        },
    }
    if "artifact" in paths:
        report["artifact_path"] = paths["artifact"]
    for name, digest in hashes.items():
        report[f"{name}_sha256"] = digest
    return report


def _checkpoint_for_hash(args: argparse.Namespace) -> Path | None:
    if args.checkpoint is not None:
        return args.checkpoint
    best_validation = sorted(args.checkpoint_dir.glob("net_epoch_bestval_at*.pth"))
    if len(best_validation) > 1:
        raise ValueError("checkpoint directory contains multiple best-validation checkpoints")
    if best_validation:
        return best_validation[0]
    epochs: list[tuple[int, Path]] = []
    for path in args.checkpoint_dir.glob("net_epoch*.pth"):
        match = re.fullmatch(r"net_epoch(\d+)\.pth", path.name)
        if match:
            epochs.append((int(match.group(1)), path))
    return max(epochs, default=(0, None), key=lambda item: item[0])[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--input-shape", type=parse_shape, required=True)
    parser.add_argument("--output-shape", type=parse_shape, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prediction-path", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--artifact-output-dtype",
        choices=("float16", "float32"),
    )
    parser.add_argument("--tvm-worker-python", type=Path)
    parser.add_argument("--tvm-site", type=Path)
    parser.add_argument("--tvm-lib-dir", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    StaticTensorContract.from_shapes(
        input_shapes=[args.input_shape],
        output_shapes=[args.output_shape],
    )
    worker_fields = (args.tvm_worker_python, args.tvm_site, args.tvm_lib_dir)
    if any(worker_fields) and not all(worker_fields):
        parser.error(
            "--tvm-worker-python, --tvm-site, and at least one --tvm-lib-dir "
            "must be provided together"
        )
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import inference_utils, train_utils
    from opencood.utils import eval_utils

    hypes = yaml_utils.load_yaml(str(args.config), SimpleNamespace(model_dir=None))
    hypes["validate_dir"] = hypes["test_dir"]
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    model = train_utils.create_model(hypes)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
    else:
        _, model = train_utils.load_saved_model(str(args.checkpoint_dir), model)
    torch.cuda.set_device(args.gpu_id)
    model.cuda().eval()

    contract = StaticTensorContract.from_shapes(
        input_shapes=[args.input_shape],
        output_shapes=[args.output_shape],
    )
    output_dtype = args.artifact_output_dtype or (
        "float16" if args.precision == "fp16" else "float32"
    )
    if args.tvm_worker_python is not None:
        runner = SubprocessTvmRelaxVmRunner(
            args.artifact,
            contract,
            precision=args.precision,
            artifact_output_dtype=output_dtype,
            gpu_id=args.gpu_id,
            worker_python=args.tvm_worker_python,
            tvm_site=args.tvm_site,
            tvm_lib_dirs=args.tvm_lib_dir,
            workspace=args.output_json.parent / "tvm_worker_ipc",
        )
    else:
        runner = TvmRelaxVmRunner(
            args.artifact,
            contract,
            gpu_id=args.gpu_id,
            output_dtype=output_dtype,
        )
    close_runner = getattr(runner, "close", None)
    if close_runner is not None:
        atexit.register(close_runner)
    model.backbone_m1 = TvmDenseBody(runner)
    model.shrinker_m1 = IdentityShrinker()
    stats = {
        threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
        for threshold in (0.3, 0.5, 0.7)
    }
    processed = failed = 0
    started = time.time()
    prediction_writer = (
        PredictionArtifactWriter(args.prediction_path)
        if args.prediction_path is not None
        else nullcontext(None)
    )
    with prediction_writer as writer:
        for batch in loader:
            if batch is None:
                continue
            try:
                batch = train_utils.to_device(batch, torch.device("cuda"))
                with torch.no_grad():
                    result = inference_utils.inference_intermediate_fusion(
                        batch, model, dataset
                    )
                for threshold in stats:
                    eval_utils.caluclate_tp_fp(
                        result["pred_box_tensor"],
                        result["pred_score"],
                        result["gt_box_tensor"],
                        stats,
                        threshold,
                    )
                if writer is not None:
                    writer.append(sample_index=processed, result=result)
                processed += 1
                if args.max_samples is not None and processed >= args.max_samples:
                    break
            except Exception:
                failed += 1
                raise

    ap30, ap50, ap70 = (
        eval_utils.calculate_ap(stats, threshold)[0]
        for threshold in (0.3, 0.5, 0.7)
    )
    requested = min(len(dataset), args.max_samples) if args.max_samples is not None else len(dataset)
    report = build_report(
        dataset_samples=len(dataset),
        requested_samples=requested,
        processed_samples=processed,
        failed_samples=failed,
        backend_calls=runner.call_count,
        fallback_samples=0,
        ap30=ap30,
        ap50=ap50,
        ap70=ap70,
        elapsed_seconds=time.time() - started,
        artifact_path=args.artifact,
        checkpoint_path=_checkpoint_for_hash(args),
        config_path=args.config,
        prediction_path=args.prediction_path,
        input_shape=contract.input_shape,
        output_shape=contract.output_shape,
        artifact_compute_dtype="float16" if args.precision == "fp16" else "float32",
        artifact_output_dtype=output_dtype,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if close_runner is not None:
        close_runner()
        atexit.unregister(close_runner)


if __name__ == "__main__":
    main()
