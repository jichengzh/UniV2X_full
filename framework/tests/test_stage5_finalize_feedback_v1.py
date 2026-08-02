from __future__ import annotations

import unittest

from scripts.stage5_finalize_feedback_v1 import (
    canonical_sha256,
    validate_manifest_request_binding,
)


def _request_and_manifest() -> tuple[dict, dict]:
    rows = []
    for dispatch_key, profile in (("tvm_auto", "tvm"), ("trt_engine", "trt")):
        for q_mode in ("fp16", "int8"):
            row_id = f"pyramid|16x32x64|q={q_mode}|profile={profile}"
            rows.append(
                {
                    "manifest_job_id": row_id,
                    "group_id": "pyramid|16x32x64",
                    "model": "pyramid",
                    "width": [16, 32, 64],
                    "q_mode": q_mode,
                    "capability_profile_id": profile,
                    "dispatch_key": dispatch_key,
                    "genome": [16, 32, 64, q_mode],
                    "strategy_id": f"q={q_mode}",
                    "capability_digest": "c" * 64,
                    "source_status": "materializable",
                    "source_contract": {"checkpoint": "/planned/checkpoint"},
                    "source_evidence_sha256": "p" * 64,
                }
            )
    request = {
        "schema_version": "stage5_measurement_request_v1",
        "round_index": 0,
        "group_count": 1,
        "row_count": 4,
        "required_metrics": ["latency_ms", "energy_j", "ap70"],
        "offline_replay_allowed": False,
        "rows": rows,
    }
    jobs = [
        {
            **row,
            "source_status": "ready",
            "source_contract": {
                **row["source_contract"],
                "checkpoint_sha256": "d" * 64,
            },
            "source_evidence_sha256": "e" * 64,
        }
        for row in rows
    ]
    manifest = {
        "source_request_sha256": canonical_sha256(request),
        "group_count": 1,
        "row_count": 4,
        "jobs": jobs,
    }
    return request, manifest


class Stage5FinalizeFeedbackV1Tests(unittest.TestCase):
    def test_manifest_must_be_derived_from_exact_request_subset(self) -> None:
        request, manifest = _request_and_manifest()
        validate_manifest_request_binding(manifest, request)

        drifted_sha = {**manifest, "source_request_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "source request SHA mismatch"):
            validate_manifest_request_binding(drifted_sha, request)

        drifted_jobs = {**manifest, "jobs": [dict(row) for row in manifest["jobs"]]}
        drifted_jobs["jobs"][0] = {**drifted_jobs["jobs"][0], "genome": [99, 32, 64, "fp16"]}
        with self.assertRaisesRegex(ValueError, "manifest job identity drift"):
            validate_manifest_request_binding(drifted_jobs, request)


if __name__ == "__main__":
    unittest.main()
