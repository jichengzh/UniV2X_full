#!/usr/bin/env python3
"""Capture real OPV2V-test post-scatter tensors for numeric gates only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np


EXPECTED_OPV2V_TEST_MANIFEST_SHA256 = (
    "e70afc9c82d29d405d86aa75fc6651a0fec77e88b825934c6464650f3416aafb"
)
EXPECTED_DENSE_SHAPE = (64, 512, 512)
SCOPE = "post_scatter_backbone_shrinker"
PURPOSE = "heldout_numeric_only_not_calibration"
DEFAULT_SAMPLE_INDICES = (37, 941, 1733)
MAX_HELDOUT_SAMPLES = 4


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def build_stable_sample_id(sample_idx: int, cav_id_list: Sequence[Any]) -> str:
    if isinstance(sample_idx, bool) or not isinstance(sample_idx, (int, np.integer)):
        raise ValueError("sample_idx must be an integer")
    cav_ids = [str(value) for value in cav_id_list]
    if not cav_ids or any(not value for value in cav_ids):
        raise ValueError("cav_id_list must contain stable non-empty IDs")
    return f"opv2v-test:{int(sample_idx)}:{','.join(cav_ids)}"


def _json_ids(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _json_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from _json_ids(child)


def load_forbid_manifests(
    paths: Sequence[Path | str],
) -> tuple[set[str], list[dict[str, str]]]:
    forbidden: set[str] = set()
    receipts: list[dict[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path)
        content = path.read_text(encoding="utf-8")
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError:
            values = [line.strip() for line in content.splitlines() if line.strip()]
        else:
            values = list(_json_ids(decoded))
        forbidden.update(value for value in values if value)
        receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    return forbidden, receipts


def validate_capture_request(
    *,
    dataset_samples: int,
    expected_dataset_samples: int,
    sample_indices: Sequence[int],
    stable_ids: Sequence[str] | None = None,
    forbidden_ids: set[str] | None = None,
) -> list[int]:
    if dataset_samples != expected_dataset_samples or dataset_samples != 2170:
        raise ValueError(
            f"dataset length must be exactly 2170, got {dataset_samples}"
        )
    if not sample_indices:
        raise ValueError("explicit stable sample indices are required")
    if len(set(sample_indices)) > MAX_HELDOUT_SAMPLES:
        raise ValueError("held-out capture accepts at most 4 unique samples")
    indices = sorted(set(sample_indices))
    if any(
        isinstance(index, bool)
        or not isinstance(index, (int, np.integer))
        or index < 0
        or index >= dataset_samples
        for index in indices
    ):
        raise ValueError("sample index out of range")
    overlap = set(stable_ids or ()) & set(forbidden_ids or ())
    if overlap:
        raise ValueError(
            "held-out sample ID is forbidden by training/tuning/calibration "
            f"manifest: {sorted(overlap)}"
        )
    return [int(index) for index in indices]


def pad_and_validate_capture(
    value: np.ndarray, *, engine_agent_batch: int
) -> tuple[np.ndarray, int]:
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise ValueError(f"capture must be float32, got {array.dtype}")
    if array.ndim != 4 or tuple(array.shape[1:]) != EXPECTED_DENSE_SHAPE:
        raise ValueError(
            "capture shape must be [agents,64,512,512], "
            f"got {list(array.shape)}"
        )
    agents = int(array.shape[0])
    if engine_agent_batch != 5 or not 1 <= agents <= engine_agent_batch:
        raise ValueError(
            f"capture agent count must be in [1,5], got {agents}"
        )
    if not np.isfinite(array).all():
        raise ValueError("capture must contain only finite values")
    padded = np.zeros(
        (engine_agent_batch, *EXPECTED_DENSE_SHAPE), dtype=np.float32
    )
    padded[:agents] = array
    return padded, agents


def build_capture_manifest(
    *,
    array: np.ndarray,
    output_npy: Path | str,
    dataset_samples: int,
    sample_indices: Sequence[int],
    stable_ids: Sequence[str],
    cav_id_lists: Sequence[Sequence[str]],
    agent_counts: Sequence[int],
    config_sha256: str,
    checkpoint_sha256: str,
    test_manifest_sha256: str,
    forbid_manifest_receipts: Sequence[dict[str, str]],
    host: str,
    device: str,
) -> dict[str, Any]:
    expected_shape = (len(sample_indices), 5, *EXPECTED_DENSE_SHAPE)
    if (
        array.dtype != np.float32
        or tuple(array.shape) != expected_shape
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            f"stacked NPY contract must be float32 {list(expected_shape)} and finite"
        )
    fields = (stable_ids, cav_id_lists, agent_counts)
    if any(len(values) != len(sample_indices) for values in fields):
        raise ValueError("capture provenance lengths do not match")
    items = [
        {
            "sample_idx": int(sample_idx),
            "stable_id": stable_id,
            "cav_id_list": list(cav_ids),
            "agent_count": int(agent_count),
            "tensor_sha256": sha256_array(array[position]),
        }
        for position, (sample_idx, stable_id, cav_ids, agent_count) in enumerate(
            zip(sample_indices, stable_ids, cav_id_lists, agent_counts)
        )
    ]
    return {
        "schema_version": "fcooper_heldout_capture_v2",
        "status": "ready",
        "dataset": "OPV2V",
        "split": "test",
        "dataset_samples": dataset_samples,
        "sample_batch": 1,
        "scope": SCOPE,
        "purpose": PURPOSE,
        "calibration_eligible": False,
        "tensor_contract": {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "layout": "NCHW",
            "engine_agent_batch": 5,
        },
        "items": items,
        "npy_path": str(Path(output_npy).resolve()),
        "npy_sha256": sha256_file(output_npy),
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "opv2v_test_manifest_sha256": test_manifest_sha256,
        "forbid_manifests": list(forbid_manifest_receipts),
        "host": host,
        "device": device,
    }


def build_heldout_identity(
    *,
    output_npy: Path | str,
    capture_manifest: Path | str,
    test_manifest_sha256: str,
    host: str,
    platform_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": "fcooper_heldout_identity_v1",
        "input_sha256": sha256_file(output_npy),
        "opv2v_test_manifest_sha256": test_manifest_sha256,
        "capture_provenance_sha256": sha256_file(capture_manifest),
        "capture_manifest": str(Path(capture_manifest).resolve()),
        "host": host,
        "platform": platform_name,
        "scope": SCOPE,
        "purpose": PURPOSE,
        "calibration_eligible": False,
    }


def _unwrap_singleton(value: Any) -> Any:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def sample_identity_from_batch(batch: dict[str, Any]) -> tuple[int, list[str]]:
    record = batch.get("ego", batch)
    if "sample_idx" not in record or "cav_id_list" not in record:
        raise ValueError("batch lacks sample_idx or cav_id_list provenance")
    sample_idx = _unwrap_singleton(record["sample_idx"])
    cav_ids = record["cav_id_list"]
    if (
        isinstance(cav_ids, (list, tuple))
        and len(cav_ids) == 1
        and isinstance(cav_ids[0], (list, tuple))
    ):
        cav_ids = cav_ids[0]
    return int(sample_idx), [str(value) for value in cav_ids]


def _checkpoint_state(torch: Any, checkpoint: Path) -> dict[str, Any]:
    loaded = torch.load(checkpoint, map_location="cpu")
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint must contain a state dictionary")
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = loaded.get(key)
        if isinstance(candidate, dict):
            return candidate
    return loaded


def _capture_one(
    *,
    dataset: Any,
    model: Any,
    sample_index: int,
    forbidden_ids: set[str],
    train_utils: Any,
    inference_utils: Any,
    torch: Any,
) -> tuple[np.ndarray, int, list[str]]:
    batch = dataset.collate_batch_test([dataset[sample_index]])
    actual_index, cav_ids = sample_identity_from_batch(batch)
    if actual_index != sample_index:
        raise ValueError(
            f"dataset sample_idx drift: requested {sample_index}, got {actual_index}"
        )
    stable_id = build_stable_sample_id(actual_index, cav_ids)
    if stable_id in forbidden_ids:
        raise ValueError(
            f"held-out sample ID is forbidden by supplied manifest: {stable_id}"
        )
    captures: list[np.ndarray] = []

    def capture_spatial_features(
        _module: Any, inputs: tuple[dict[str, Any], ...]
    ) -> None:
        tensor = inputs[0]["spatial_features"]
        captures.append(
            tensor.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
        )

    hook = model.backbone_m1.register_forward_pre_hook(capture_spatial_features)
    try:
        batch = train_utils.to_device(batch, torch.device("cuda"))
        with torch.no_grad():
            inference_utils.inference_intermediate_fusion(batch, model, dataset)
    finally:
        hook.remove()
    if len(captures) != 1:
        raise ValueError(f"expected exactly one backbone capture, got {len(captures)}")
    return captures[0], actual_index, cav_ids


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--expected-dataset-samples", type=int, default=2170)
    parser.add_argument("--engine-agent-batch", type=int, default=5)
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=list(DEFAULT_SAMPLE_INDICES),
    )
    parser.add_argument(
        "--forbid-manifest", type=Path, action="append", default=[]
    )
    parser.add_argument("--output-npy", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    test_manifest_sha = sha256_file(args.test_manifest)
    if test_manifest_sha != EXPECTED_OPV2V_TEST_MANIFEST_SHA256:
        raise ValueError("OPV2V test manifest SHA-256 mismatch")
    if args.engine_agent_batch != 5:
        raise ValueError("engine agent batch must be exactly 5")
    forbidden_ids, forbid_receipts = load_forbid_manifests(
        args.forbid_manifest
    )

    import torch

    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import inference_utils, train_utils

    hypes = yaml_utils.load_yaml(
        str(args.config), SimpleNamespace(model_dir=None)
    )
    if "test_dir" not in hypes:
        raise ValueError("configuration lacks OPV2V test_dir")
    hypes["validate_dir"] = hypes["test_dir"]
    dataset = build_dataset(hypes, visualize=False, train=False)
    indices = validate_capture_request(
        dataset_samples=len(dataset),
        expected_dataset_samples=args.expected_dataset_samples,
        sample_indices=args.sample_indices,
    )
    model = train_utils.create_model(hypes)
    model.load_state_dict(
        _checkpoint_state(torch, args.checkpoint), strict=True
    )
    model.cuda().eval()
    tensors: list[np.ndarray] = []
    stable_ids: list[str] = []
    cav_id_lists: list[list[str]] = []
    agent_counts: list[int] = []
    for index in indices:
        capture, actual_index, cav_ids = _capture_one(
            dataset=dataset,
            model=model,
            sample_index=index,
            forbidden_ids=forbidden_ids,
            train_utils=train_utils,
            inference_utils=inference_utils,
            torch=torch,
        )
        stable_id = build_stable_sample_id(actual_index, cav_ids)
        validate_capture_request(
            dataset_samples=len(dataset),
            expected_dataset_samples=args.expected_dataset_samples,
            sample_indices=[index],
            stable_ids=[stable_id],
            forbidden_ids=forbidden_ids,
        )
        padded, agents = pad_and_validate_capture(
            capture, engine_agent_batch=args.engine_agent_batch
        )
        tensors.append(padded)
        stable_ids.append(stable_id)
        cav_id_lists.append(cav_ids)
        agent_counts.append(agents)

    stacked = np.stack(tensors).astype(np.float32, copy=False)
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, stacked, allow_pickle=False)
    manifest = build_capture_manifest(
        array=stacked,
        output_npy=args.output_npy,
        dataset_samples=len(dataset),
        sample_indices=indices,
        stable_ids=stable_ids,
        cav_id_lists=cav_id_lists,
        agent_counts=agent_counts,
        config_sha256=sha256_file(args.config),
        checkpoint_sha256=sha256_file(args.checkpoint),
        test_manifest_sha256=test_manifest_sha,
        forbid_manifest_receipts=forbid_receipts,
        host=socket.gethostname(),
        device=torch.cuda.get_device_name(torch.cuda.current_device()),
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar = build_heldout_identity(
        output_npy=args.output_npy,
        capture_manifest=args.manifest,
        test_manifest_sha256=test_manifest_sha,
        host=socket.gethostname(),
        platform_name=platform.platform(),
    )
    sidecar_path = args.output_npy.with_suffix(".heldout.json")
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
