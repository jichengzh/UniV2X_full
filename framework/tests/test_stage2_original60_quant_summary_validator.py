from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_summary_rows(int8_ap_quality: str = "native_int8_full_ap_eval") -> list[dict[str, object]]:
    labels = [f"label_{index:02d}" for index in range(55)] + [
        "s0_024",
        "lhc_07",
        "lhc_20",
        "s0_056",
        "frontier_25",
    ]
    rows: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        for precision in ("fp32", "fp16", "int8"):
            latency_status = "measured"
            latency_ms: float | None = 12.0 + index
            latency_source = "true_measurement_smoke"
            latency_schedule = "metaschedule_tuned"
            energy_status = "measured"
            energy_source = "true_measurement_smoke"
            energy_schedule = "metaschedule_tuned"
            ap_status = "no_claim"
            ap70: float | None = None
            ap_source = "no_claim"
            quality = "base_gate"
            if precision == "fp32":
                latency_status = "no_claim" if index < 2 else "measured"
                latency_ms = None if index < 2 else 20.0 + index
                latency_source = "no_claim" if index < 2 else "historical_true_measurement_reclassified"
                latency_schedule = "default" if index < 2 else "metaschedule_tuned"
                energy_source = "true_measurement"
                energy_schedule = "default" if index < 4 else "metaschedule_tuned"
                ap_status = "measured" if index < 5 else "no_claim"
                ap70 = 0.5 if index < 5 else None
                ap_source = "stage2_original60_ap_source_map_reference" if index < 5 else "no_claim"
                quality = "fp32_reclassified_from_suspect_fp16_tagged_row;fp32_energy_threaded_window_h800"
            elif precision == "fp16":
                ap_status = "measured"
                ap70 = 0.6
                ap_source = "true_eval"
                quality = "true_fp16_onnx_smoke_only;true_fp16_onnx_energy;true_fp16_original60_ap_eval"
            else:
                latency_schedule = "native_int8_full_onnx_topology_direct"
                energy_schedule = "native_int8_full_onnx_topology_direct"
                if label == "s0_024":
                    ap_status = "measured"
                    ap70 = 0.4
                    ap_source = "true_eval"
                    quality = f"native_int8_full_onnx_topology_latency;native_int8_full_onnx_topology_energy;{int8_ap_quality}"
                else:
                    quality = "native_int8_full_onnx_topology_latency;native_int8_full_onnx_topology_energy;tvm_int8_backbone_subnet_not_ready"
            rows.append(
                {
                    "label": label,
                    "candidate_id": f"{label}:{precision}",
                    "original60_candidate_id": label,
                    "width": "24x128x256",
                    "precision": precision,
                    "latency_ms": latency_ms,
                    "latency_status": latency_status,
                    "latency_measurement_source": latency_source,
                    "latency_schedule_policy": latency_schedule,
                    "latency_claim_status": "claimable" if latency_status == "measured" else "no_claim",
                    "latency_evidence_scope": "backbone_only",
                    "energy_j_per_inference": 1.0 + index,
                    "energy_status": energy_status,
                    "energy_measurement_source": energy_source,
                    "energy_schedule_policy": energy_schedule,
                    "energy_claim_status": "claimable",
                    "energy_evidence_scope": "backbone_only",
                    "ap70": ap70,
                    "ap_status": ap_status,
                    "ap_measurement_source": ap_source,
                    "ap_source_kind": ap_source,
                    "ap_schedule_policy": "not_applicable",
                    "ap_claim_status": "claimable" if ap_status == "measured" else "no_claim",
                    "ap_evidence_scope": "model_eval",
                    "quality_gate_status": quality,
                    "failure_reasons": "",
                    "full_network_claim": False,
                    "quant_method": precision,
                    "quant_scope": "backbone_only",
                }
            )
    return rows


def write_fixture(output_root: Path, int8_ap_quality: str = "native_int8_full_ap_eval") -> None:
    exports = output_root / "exports"
    rows_dir = output_root / "rows"
    summary_rows = make_summary_rows(int8_ap_quality=int8_ap_quality)
    write_json(
        exports / "original60_quant_three_metric_summary_latest.json",
        {
            "schema": "original60_quant_three_metric_summary_v1",
            "total_cells": 180,
            "candidate_count": 60,
            "rows": summary_rows,
        },
    )
    (exports / "original60_quant_three_metric_summary_latest.csv").write_text("label,precision\n", encoding="utf-8")
    (exports / "original60_quant_three_metric_summary_latest.md").write_text("# Summary\n", encoding="utf-8")
    write_json(
        exports / "original60_quant_measurement_source_audit_latest.json",
        {
            "schema": "original60_quant_measurement_source_audit_v3",
            "conclusion": {
                "all_cells_are_direct_same_level_measurements": False,
                "fp16_latency_energy": "FP16 dtype exists but no tensorcore/wmma workdir evidence.",
            },
            "fp32_energy_threaded60": {
                "canonical_fp32_energy_rows": 60,
                "dynamic_watt_lt_50_count": 0,
            },
        },
    )
    (exports / "original60_quant_measurement_source_audit_latest.md").write_text("# Audit\n", encoding="utf-8")
    write_json(
        exports / "fp32_threaded_window_energy_60label_review_latest.json",
        {
            "schema": "fp32_threaded_window_energy_60label_review_v1",
            "row_count": 60,
            "unique_label_count": 60,
            "low_new_dynamic_w_lt_50_count": 0,
            "schedule_counts": {"default": 4, "metaschedule_tuned": 56},
        },
    )
    tir_rows = []
    for label in sorted({"s0_024", "lhc_07", "lhc_20", "s0_056", "frontier_25"}):
        tir_rows.append(
            {
                "label": label,
                "classification": "fp16_dtype_present_no_tensorcore_evidence",
                "input_dtypes": {"spatial_features": "float16"},
                "initializer_dtype_counts": {"float16": 10},
                "keyword_file_counts": {
                    "float32": 5,
                    "float16": 0,
                    "wmma": 0,
                    "tensorcore": 0,
                    "tensor_core": 0,
                },
            }
        )
    write_json(
        exports / "fp16_tir_lowering_5label_audit_latest.json",
        {"schema": "fp16_tir_lowering_5label_audit_v1", "rows": tir_rows},
    )
    (exports / "fp16_tir_lowering_5label_audit_latest.md").write_text("# FP16 TIR Audit\n", encoding="utf-8")
    energy_rows = []
    for index in range(60):
        energy_rows.append(
            {
                "precision": "fp32",
                "measurement_status": "measured",
                "quality_gate_status": "fp32_energy_threaded_window_h800",
                "raw_artifact": f"raw/label_{index:02d}",
                "run_id": f"fp32_energy_threaded60_label_{index:02d}_20260629_metaschedule_tuned_gpu0",
                "software_point_id": f"original60:label_{index:02d}:24x128x256:fp32:energy",
                "source_files": [f"raw/label_{index:02d}/energy_result.json"],
                "watt_avg": 200.0,
                "idle_watt_avg": 120.0,
                "schedule_policy": "default" if index < 4 else "metaschedule_tuned",
            }
        )
    write_jsonl(rows_dir / "fp32_original60_energy_threaded60_rows_v1.jsonl", energy_rows)


class Stage2Original60QuantSummaryValidatorTest(unittest.TestCase):
    def run_validator(self, output_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/stage2_validate_original60_quant_summary.py"),
                "--output-root",
                str(output_root),
                *extra,
            ],
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
        )

    def test_validator_passes_and_writes_freeze_for_trusted_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "original60_quant"
            write_fixture(output_root)

            result = self.run_validator(output_root, "--write-freeze")

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(payload["summary"]["fp32_energy_schedule"], {"default": 4, "metaschedule_tuned": 56})
            self.assertEqual(payload["fp32_energy_rows"]["dynamic_watt_lt_50_count"], 0)
            self.assertIn("当前 FP16 route 不是 tensor-core optimized", payload["fp16_tir_audit"]["closure_conclusion"])
            self.assertTrue((output_root / "exports/original60_quant_trusted_freeze_latest.json").exists())
            self.assertTrue((output_root / "exports/original60_quant_trusted_freeze_latest.md").exists())

    def test_validator_rejects_int8_smoke_ap_marked_as_measured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "original60_quant"
            write_fixture(output_root, int8_ap_quality="native_int8_5sample_smoke")

            result = self.run_validator(output_root)

            self.assertNotEqual(result.returncode, 0)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "FAIL")
            self.assertTrue(any("INT8 AP measured row s0_024 must not come from smoke gate" in item for item in payload["errors"]))


if __name__ == "__main__":
    unittest.main()
