from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.fcooper_collect_heldout_post_scatter_v2 import (
    EXPECTED_OPV2V_TEST_MANIFEST_SHA256,
    build_capture_manifest,
    build_heldout_identity,
    build_stable_sample_id,
    load_forbid_manifests,
    pad_and_validate_capture,
    _capture_one,
    validate_capture_request,
)


def test_stable_sample_id_and_forbid_overlap(tmp_path: Path) -> None:
    json_forbid = tmp_path / "calibration.json"
    json_forbid.write_text(json.dumps({"sample_ids": ["opv2v-test:37:1,2"]}))
    text_forbid = tmp_path / "training.txt"
    text_forbid.write_text("opv2v-test:941:3,4\n")

    forbidden, receipts = load_forbid_manifests([json_forbid, text_forbid])

    assert build_stable_sample_id(37, ["1", 2]) == "opv2v-test:37:1,2"
    assert forbidden == {"opv2v-test:37:1,2", "opv2v-test:941:3,4"}
    assert [receipt["sha256"] for receipt in receipts] == [
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (json_forbid, text_forbid)
    ]
    with pytest.raises(ValueError, match="forbidden"):
        validate_capture_request(
            dataset_samples=2170,
            expected_dataset_samples=2170,
            sample_indices=[37],
            stable_ids=["opv2v-test:37:1,2"],
            forbidden_ids=forbidden,
        )


@pytest.mark.parametrize(
    ("dataset_samples", "indices", "message"),
    [
        (2169, [37], "dataset length"),
        (2170, [-1], "out of range"),
        (2170, [2170], "out of range"),
        (2170, [1, 2, 3, 4, 5], "at most 4"),
    ],
)
def test_request_rejects_wrong_length_indices_and_excess_samples(
    dataset_samples: int, indices: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_capture_request(
            dataset_samples=dataset_samples,
            expected_dataset_samples=2170,
            sample_indices=indices,
        )


def test_request_deduplicates_without_mutating_and_sorts() -> None:
    requested = [1733, 37, 941, 37]

    validated = validate_capture_request(
        dataset_samples=2170,
        expected_dataset_samples=2170,
        sample_indices=requested,
    )

    assert validated == [37, 941, 1733]
    assert requested == [1733, 37, 941, 37]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.zeros((5, 63, 512, 512), dtype=np.float32), "shape"),
        (np.zeros((5, 64, 512, 512), dtype=np.float16), "float32"),
        (np.zeros((0, 64, 512, 512), dtype=np.float32), "agent count"),
        (np.zeros((6, 64, 512, 512), dtype=np.float32), "agent count"),
    ],
)
def test_capture_rejects_shape_dtype_and_agent_count(
    value: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pad_and_validate_capture(value, engine_agent_batch=5)


def test_capture_rejects_nonfinite_and_pads_to_five() -> None:
    bad = np.zeros((2, 64, 512, 512), dtype=np.float32)
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pad_and_validate_capture(bad, engine_agent_batch=5)

    value = np.ones((2, 64, 512, 512), dtype=np.float32)
    padded, agents = pad_and_validate_capture(value, engine_agent_batch=5)
    assert padded.shape == (5, 64, 512, 512)
    assert padded.dtype == np.float32
    assert agents == 2
    np.testing.assert_array_equal(padded[:2], value)
    assert np.count_nonzero(padded[2:]) == 0


def test_manifest_binds_numeric_only_purpose_and_exact_npy_shape(tmp_path: Path) -> None:
    npy = tmp_path / "heldout_inputs.npy"
    array = np.zeros((2, 5, 64, 512, 512), dtype=np.float32)
    np.save(npy, array, allow_pickle=False)
    manifest = build_capture_manifest(
        array=array,
        output_npy=npy,
        dataset_samples=2170,
        sample_indices=[37, 941],
        stable_ids=["opv2v-test:37:1", "opv2v-test:941:2,3"],
        cav_id_lists=[["1"], ["2", "3"]],
        agent_counts=[1, 2],
        config_sha256="config",
        checkpoint_sha256="checkpoint",
        test_manifest_sha256=EXPECTED_OPV2V_TEST_MANIFEST_SHA256,
        forbid_manifest_receipts=[],
        host="host",
        device="cuda:0",
    )

    assert manifest["purpose"] == "heldout_numeric_only_not_calibration"
    assert manifest["scope"] == "post_scatter_backbone_shrinker"
    assert manifest["tensor_contract"]["shape"] == [2, 5, 64, 512, 512]
    assert manifest["tensor_contract"]["dtype"] == "float32"
    assert manifest["dataset_samples"] == 2170
    assert manifest["npy_sha256"] == hashlib.sha256(npy.read_bytes()).hexdigest()
    assert len(manifest["items"]) == 2
    assert manifest["calibration_eligible"] is False


def test_heldout_sidecar_matches_task1_runner_contract(tmp_path: Path) -> None:
    npy = tmp_path / "heldout_inputs.npy"
    np.save(npy, np.zeros((1, 5, 64, 512, 512), dtype=np.float32))
    provenance = tmp_path / "heldout_manifest.json"
    provenance.write_text('{"status":"ready"}\n')

    identity = build_heldout_identity(
        output_npy=npy,
        capture_manifest=provenance,
        test_manifest_sha256=EXPECTED_OPV2V_TEST_MANIFEST_SHA256,
        host="orin",
        platform_name="Linux-aarch64",
    )

    assert identity["input_sha256"] == hashlib.sha256(npy.read_bytes()).hexdigest()
    assert identity["capture_provenance_sha256"] == hashlib.sha256(
        provenance.read_bytes()
    ).hexdigest()
    assert (
        identity["opv2v_test_manifest_sha256"]
        == EXPECTED_OPV2V_TEST_MANIFEST_SHA256
    )
    assert identity["purpose"] == "heldout_numeric_only_not_calibration"


def test_forbidden_id_is_rejected_before_model_forward() -> None:
    batch = {"ego": {"sample_idx": [37], "cav_id_list": [["1", "2"]]}}

    class Dataset:
        def __getitem__(self, index):
            return {"index": index}

        def collate_batch_test(self, records):
            return batch

    class Backbone:
        def register_forward_pre_hook(self, hook):
            raise AssertionError("hook must not be registered for forbidden ID")

    class Model:
        backbone_m1 = Backbone()

    class Inference:
        @staticmethod
        def inference_intermediate_fusion(*args):
            raise AssertionError("inference must not run for forbidden ID")

    with pytest.raises(ValueError, match="forbidden"):
        _capture_one(
            dataset=Dataset(),
            model=Model(),
            sample_index=37,
            forbidden_ids={"opv2v-test:37:1,2"},
            train_utils=object(),
            inference_utils=Inference(),
            torch=object(),
        )
