from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from framework.stage5 import measurement_plan_v2 as plan


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Stage5MeasurementPlanV2Tests(unittest.TestCase):
    def test_fcooper_request_accepts_named_four_axis_genome(self) -> None:
        width = [64, 128, 256, 128, 256]
        width_schema = [
            "backbone.s0",
            "backbone.s1",
            "backbone.s2",
            "neck.deblock",
            "neck.output",
        ]
        group_id = (
            "fcooper|backbone.s0=64|backbone.s1=128|backbone.s2=256|"
            "neck.deblock=128|neck.output=256"
        )
        row_id = f"{group_id}|q=fp16|profile=h800-trt"
        source_contract = {"onnx_path": "/tmp/fcooper.onnx"}
        source_plan_sha = hashlib.sha256(
            json.dumps(
                {
                    "kind": "fcooper_scanner_materialize_export",
                    "width": width,
                    "contract": source_contract,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        row = {
            "row_id": row_id,
            "manifest_job_id": row_id,
            "task_id": "S5-FCO-TRT",
            "task_sha256": "a" * 64,
            "group_id": group_id,
            "model": "fcooper",
            "width": width,
            "width_schema": width_schema,
            "structure_widths": dict(zip(width_schema, width)),
            "genome": [*width, "fp16"],
            "q_mode": "fp16",
            "hardware_id": "h800",
            "capability_profile_id": "h800-trt",
            "capability_digest": "b" * 64,
            "dispatch_key": "trt_engine",
            "source_contract": source_contract,
            "source_evidence_sha256": source_plan_sha,
        }
        request = {
            "schema_version": "stage5_independent_validation_request_v1",
            "task_id": "S5-FCO-TRT",
            "task_sha256": "a" * 64,
            "batch_size": 1,
            "independent_from_search_measurement": True,
            "real_h800_measurement_required": True,
            "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
            "rows": [row],
            "row_sha256": {
                row_id: hashlib.sha256(
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            },
        }
        request["measurement_request_sha256"] = hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        validated = plan._validate_request(request)

        self.assertEqual(validated[0]["width_schema"], width_schema)
        self.assertEqual(len(validated[0]["genome"]), 6)

    def test_four_independent_genomes_emit_four_jobs_without_arm_product(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            evidence_paths = {}
            for index, width in enumerate(([16, 48, 64], [24, 48, 96], [32, 64, 96], [48, 128, 224])):
                group_id = f"pyramid|{'x'.join(map(str, width))}"
                trt_dir = root / f"trt_{index}"
                trt_dir.mkdir()
                (trt_dir / "sample.npy").write_bytes(b"calib")
                contract = {
                    "checkpoint_path": str(root / f"checkpoint_{index}.bin"),
                    "onnx_path": str(root / f"model_{index}.onnx"),
                    "calibration_npz": str(root / f"calibration_{index}.npz"),
                    "calibration_root": str(root / f"calibration_root_{index}"),
                    "calibration_summary": str(root / f"summary_{index}.json"),
                    "trt_calibration_dir": str(trt_dir),
                }
                source_plan = hashlib.sha256(
                    json.dumps(
                        {"kind": "pyramid_checkpoint_export", "width": width, "contract": contract},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                row_id = f"{group_id}|q={'int8' if index < 2 else 'fp16'}|profile=tvm"
                rows.append(
                    {
                        "row_id": row_id,
                        "manifest_job_id": row_id,
                        "task_id": "S5-PYR-TVM",
                        "task_sha256": "b" * 64,
                        "group_id": group_id,
                        "model": "pyramid",
                        "width": width,
                        "genome": [*width, "int8" if index < 2 else "fp16"],
                        "q_mode": "int8" if index < 2 else "fp16",
                        "hardware_id": "h800",
                        "capability_profile_id": "h800-tvm-probe-conditioned-v3",
                        "capability_digest": "c" * 64,
                        "dispatch_key": "tvm_auto",
                        "source_contract": contract,
                        "source_evidence_sha256": source_plan,
                    }
                )
                artifacts = {}
                for kind in ("checkpoint", "onnx", "calibration", "summary"):
                    artifact = root / f"{kind}_{index}.bin"
                    artifact.write_bytes(f"{kind}-{index}".encode())
                    artifacts[kind] = artifact
                evidence = {
                    "schema_version": "stage5_source_materialization_evidence_v1",
                    "group_id": group_id,
                    "source_plan_sha256": source_plan,
                    "status": "ready",
                    **{
                        f"{kind}_path" if kind != "summary" else "calibration_summary_path": str(path)
                        for kind, path in artifacts.items()
                    },
                    **{
                        f"{kind}_sha256" if kind != "summary" else "calibration_summary_sha256": _sha(path)
                        for kind, path in artifacts.items()
                    },
                }
                evidence_path = root / f"evidence_{index}.json"
                evidence_path.write_text(json.dumps(evidence))
                evidence_paths[group_id] = evidence_path
            request = {
                "schema_version": "stage5_measurement_request_v2",
                "task_id": "S5-PYR-TVM",
                "task_sha256": "b" * 64,
                "batch_size": 4,
                "sample_budget": 16,
                "atomic_feedback": True,
                "real_h800_measurement_required": True,
                "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
                "rows": rows,
            }
            request["row_sha256"] = {
                row["row_id"]: hashlib.sha256(
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                for row in rows
            }
            request["measurement_request_sha256"] = hashlib.sha256(
                json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            quant_contract_paths = {}
            for row in rows:
                if row["q_mode"] != "int8":
                    continue
                contract = root / f"quant_{'x'.join(map(str, row['width']))}.json"
                contract.write_text(
                    json.dumps(
                        {
                            "schema": "stage3_tvm_int8_quant_contract_v3",
                            "params": {"spatial_features": {"scale": 0.1, "zero_point": 128}},
                        }
                    )
                )
                quant_contract_paths[row["row_id"]] = contract

            result = plan.build_performance_plan(
                request,
                source_evidence_paths=evidence_paths,
                quant_contract_paths=quant_contract_paths,
                remote_artifact_root=root / "execution",
                gpus=[4, 5, 6, 7],
            )
            independent = {
                **request,
                "schema_version": "stage5_independent_validation_request_v1",
                "batch_size": 2,
                "independent_from_search_measurement": True,
                "rows": rows[:2],
                "row_sha256": {
                    row["row_id"]: hashlib.sha256(
                        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
                    for row in rows[:2]
                },
            }
            independent.pop("measurement_request_sha256", None)
            independent["measurement_request_sha256"] = hashlib.sha256(
                json.dumps(independent, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            independent_result = plan.build_performance_plan(
                independent,
                source_evidence_paths={
                    row["group_id"]: evidence_paths[row["group_id"]]
                    for row in rows[:2]
                },
                quant_contract_paths={
                    row["row_id"]: quant_contract_paths[row["row_id"]]
                    for row in rows[:2]
                },
                remote_artifact_root=root / "independent_execution",
                gpus=[4, 5],
            )
            downgraded = json.loads(json.dumps(independent))
            downgraded["independent_from_search_measurement"] = False
            downgraded.pop("measurement_request_sha256")
            downgraded["measurement_request_sha256"] = hashlib.sha256(
                json.dumps(downgraded, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "independent-validation contract"):
                plan.build_performance_plan(
                    downgraded,
                    source_evidence_paths={
                        row["group_id"]: evidence_paths[row["group_id"]]
                        for row in rows[:2]
                    },
                    quant_contract_paths={
                        row["row_id"]: quant_contract_paths[row["row_id"]]
                        for row in rows[:2]
                    },
                    remote_artifact_root=root / "downgraded",
                    gpus=[4, 5],
                )
            tampered = json.loads(json.dumps(request))
            tampered["rows"][0]["q_mode"] = "fp16"
            with self.assertRaisesRegex(ValueError, "measurement request SHA mismatch"):
                plan.build_performance_plan(
                    tampered,
                    source_evidence_paths=evidence_paths,
                    quant_contract_paths=quant_contract_paths,
                    remote_artifact_root=root / "execution_tampered",
                    gpus=[4, 5, 6, 7],
                )

        self.assertEqual(result["manifest"]["row_count"], 4)
        self.assertEqual(result["manifest"]["genome_count"], 4)
        self.assertEqual(len(result["performance_jobs"]), 4)
        self.assertEqual({job["runner_key"] for job in result["performance_jobs"]}, {"tvm_fp16", "tvm_int8"})
        for job in result["performance_jobs"]:
            if job["runner_key"] == "tvm_int8":
                self.assertIn("--tensor-quant-params-json", job["command"])
        self.assertEqual(independent_result["manifest"]["row_count"], 2)
        self.assertEqual(
            independent_result["manifest"]["source_request_schema"],
            "stage5_independent_validation_request_v1",
        )


if __name__ == "__main__":
    unittest.main()
