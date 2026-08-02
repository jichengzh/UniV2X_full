from __future__ import annotations

import copy
import hashlib
import json
import unittest

from framework.stage5.closure_v3 import build_stage5_closure_audit


TASK_ID = "S5-PYR-TVM"
TASK_SHA = "t" * 64
MODEL = "pyramid"
HARDWARE = "h800"
PROFILE = "h800-tvm-probe-conditioned-v3"
DISPATCH = "tvm_auto"


def _identity(index: int) -> dict:
    width = [16 + index, 32 + index, 64 + index]
    q_mode = "fp16" if index % 2 == 0 else "int8"
    row_id = f"{MODEL}|{'x'.join(map(str, width))}|q={q_mode}|profile={PROFILE}"
    return {
        "task_id": TASK_ID,
        "task_sha256": TASK_SHA,
        "row_id": row_id,
        "manifest_job_id": row_id,
        "group_id": f"{MODEL}|{'x'.join(map(str, width))}",
        "model": MODEL,
        "hardware_id": HARDWARE,
        "capability_profile_id": PROFILE,
        "dispatch_key": DISPATCH,
        "width": width,
        "q_mode": q_mode,
        "genome": [*width, q_mode],
    }


def _success(identity: dict, latency: float, energy: float, ap70: float) -> dict:
    return {
        **identity,
        "terminal_status": "measured_success_gold",
        "latency_ms": latency,
        "energy_j": energy,
        "ap70": ap70,
    }


def _artifacts() -> tuple[list[dict], dict, list[dict], list[dict], list[dict]]:
    initial_identities = [_identity(100), _identity(101)]
    gold = [
        _success(initial_identities[0], 10.0, 10.0, 0.5),
        _success(initial_identities[1], 8.0, 12.0, 0.4),
    ]
    gold.extend(
        {
            "manifest_job_id": f"unrelated-gold-{index}",
            "terminal_status": "measured_success_gold",
        }
        for index in range(174)
    )

    candidates = [_identity(index) for index in range(16)]
    manifest = {
        "schema_version": "stage5_task_candidate_manifest_v2",
        "task_id": TASK_ID,
        "task_sha256": TASK_SHA,
        "target_model": MODEL,
        "capability_profile_id": PROFILE,
        "eligible_row_count": 16,
        "excluded": [
            {"row_id": row["row_id"], "reason": "already_measured"}
            for row in initial_identities
        ]
        + [{"row_id": "held-out", "reason": "frozen_independent_holdout"}],
        "rows": candidates,
    }

    requests = []
    feedback_batches = []
    audits = []
    for round_index in range(4):
        request_rows = copy.deepcopy(candidates[round_index * 4 : (round_index + 1) * 4])
        request = {
            "schema_version": "stage5_measurement_request_v2",
            "task_id": TASK_ID,
            "task_sha256": TASK_SHA,
            "round_index": round_index,
            "batch_size": 4,
            "sample_budget": 16,
            "atomic_feedback": True,
            "rows": request_rows,
        }
        request["row_sha256"] = {
            row["row_id"]: hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for row in request_rows
        }
        request["measurement_request_sha256"] = hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        requests.append(request)
        feedback = [
            _success(row, 20.0 + index, 20.0 + index, 0.1)
            for index, row in enumerate(request_rows, start=round_index * 4)
        ]
        feedback_batches.append({"rows": feedback})
        audits.append(
            {
                "schema_version": "stage5_atomic_batch_audit_v2",
                "feedback_released": True,
                "batch_quarantined": False,
                "budget_consumed": 4,
                "released_feedback_rows": copy.deepcopy(feedback),
            }
        )

    failed = feedback_batches[3]["rows"][2]
    feedback_batches[3]["rows"][2] = {
        **failed,
        "terminal_status": "feasibility_failure",
        "failure_reason": "unsupported generated engine",
    }
    for metric in ("latency_ms", "energy_j", "ap70"):
        feedback_batches[3]["rows"][2].pop(metric)
    feedback_batches[3]["rows"][3] = _success(
        feedback_batches[3]["rows"][3], 5.0, 5.0, 0.9
    )
    audits[3]["released_feedback_rows"] = copy.deepcopy(feedback_batches[3]["rows"])
    return gold, manifest, requests, feedback_batches, audits


class Stage5ClosureV3Tests(unittest.TestCase):
    def test_derives_initial_genome_from_width_and_q_mode(self) -> None:
        gold, manifest, requests, feedback, audits = _artifacts()
        for row in gold[:2]:
            row.pop("genome")
        gold[0].update({"latency_ms": 1.0, "energy_j": 1.0, "ap70": 0.99})

        report = build_stage5_closure_audit(
            gold, manifest, requests, feedback, audits
        )

        self.assertEqual(report["initial_count"], 2)
        self.assertEqual(report["online_count"], 16)
        self.assertIn(gold[0]["manifest_job_id"], report["frontier_ids"])

    def test_builds_json_serializable_closed_report_with_comparable_hv(self) -> None:
        artifacts = _artifacts()
        original = copy.deepcopy(artifacts)

        report = build_stage5_closure_audit(*artifacts)

        self.assertEqual(artifacts, original)
        self.assertEqual(report["schema_version"], "stage5_task_closure_audit_v3")
        self.assertEqual(report["initial_count"], 2)
        self.assertEqual(report["online_count"], 16)
        self.assertEqual(
            report["terminal_counts"],
            {
                "feasibility_failure": 1,
                "measured_success_gold": 17,
                "numerical_feasibility_failure": 0,
            },
        )
        self.assertEqual(report["frontier_ids"], [_identity(15)["row_id"]])
        self.assertEqual(
            [point["manifest_job_id"] for point in report["frontier_points"]],
            report["frontier_ids"],
        )
        self.assertEqual(
            report["frontier_points"][0]["objectives"],
            {"latency_ms": 5.0, "energy_j": 5.0, "ap70": 0.9},
        )
        self.assertEqual(report["frontier_points"][0]["round_index"], 3)
        self.assertEqual(len(report["frontier_after_rounds"]), 4)
        self.assertEqual(
            report["frontier_after_rounds"][-1]["frontier_ids"],
            report["frontier_ids"],
        )
        self.assertEqual(
            report["independent_validation_ids"], report["frontier_ids"][:4]
        )
        self.assertEqual(len(report["hv_after_rounds"]), 4)
        self.assertEqual(
            report["hv_curve"], [report["hv_initial"], *report["hv_after_rounds"]]
        )
        self.assertTrue(
            all(left <= right for left, right in zip(report["hv_curve"], report["hv_curve"][1:]))
        )
        self.assertGreater(report["hv_after_rounds"][-1], report["hv_initial"])
        self.assertEqual(report["hv_normalization"]["derived_from_successful_count"], 17)
        self.assertEqual(len(report["hv_normalization"]["reference"]), 3)
        self.assertTrue(report["closure"])
        json.dumps(report, allow_nan=False)

    def test_does_not_treat_non_already_measured_exclusions_as_initial(self) -> None:
        gold, manifest, requests, feedback, audits = _artifacts()
        manifest["excluded"].append(
            {"row_id": "unrelated-gold-0", "reason": "source_not_ready"}
        )

        report = build_stage5_closure_audit(gold, manifest, requests, feedback, audits)

        self.assertEqual(report["initial_count"], 2)

    def test_rejects_budget_duplicate_overlap_context_terminal_and_atomic_drift(self) -> None:
        cases = {}

        artifacts = list(_artifacts())
        artifacts[2] = artifacts[2][:-1]
        cases["exactly four measurement requests"] = artifacts

        artifacts = list(_artifacts())
        artifacts[2][1]["rows"][0]["genome"] = copy.deepcopy(
            artifacts[2][0]["rows"][0]["genome"]
        )
        cases["genome/width/q_mode identity drift"] = artifacts

        artifacts = list(_artifacts())
        artifacts[2][0]["rows"][0]["row_id"] = artifacts[1]["excluded"][0]["row_id"]
        artifacts[2][0]["rows"][0]["manifest_job_id"] = artifacts[1]["excluded"][0]["row_id"]
        cases["overlaps initial observed IDs"] = artifacts

        artifacts = list(_artifacts())
        artifacts[3][1]["rows"][0]["hardware_id"] = "rtx4090"
        cases["fixed task context drift"] = artifacts

        artifacts = list(_artifacts())
        artifacts[3][0]["rows"][0]["terminal_status"] = "public_runner_failure"
        cases["true feasibility terminal"] = artifacts

        artifacts = list(_artifacts())
        artifacts[4][2]["feedback_released"] = False
        cases["released atomic batch"] = artifacts

        artifacts = list(_artifacts())
        artifacts[4][0]["released_feedback_rows"][0]["terminal_status"] = (
            "feasibility_failure"
        )
        cases["released atomic batch audit"] = artifacts

        artifacts = list(_artifacts())
        artifacts[2][0]["rows"][0]["strategy_id"] = "drifted-after-request-freeze"
        cases["measurement request row SHA mismatch"] = artifacts

        for message, case in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_stage5_closure_audit(*case)


if __name__ == "__main__":
    unittest.main()
