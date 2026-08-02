from __future__ import annotations

import json
import os
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from framework.stage2.original60_quant_completion import (
    build_completion_jobs,
    fp16_label_widths_from_jobs,
    native_int8_label_widths_from_jobs,
    summarize_completion_jobs,
    write_completion_outputs,
)
from scripts.stage2_generate_original60_quant_state_coverage import measured_ap_index


ROOT = Path(__file__).resolve().parents[2]
FP16_RUNNER = (
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    "raw/fp16_true_smoke/stage2_h800_fp16_true_smoke.py"
)


def row(
    *,
    label: str,
    precision: str,
    axis: str,
    status: str,
    quality_gate_status: str = "",
    quant_method: str = "",
    failure_reason: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "label": label,
        "precision": precision,
        "width": [24, 128, 256] if label == "s0_024" else [64, 48, 256],
        "measurement_status": status,
        "quality_gate_status": quality_gate_status,
        "quant_method": quant_method,
        "full_network_claim": False,
        "failure_reason": failure_reason,
        "original60_candidate_id": f"coverage:pyramid_lidar:{label}",
        "raw_artifact": f"raw/{axis}/{label}" if status == "measured" else None,
    }
    if axis == "latency" and status == "measured":
        payload["latency_p50_us"] = 12345.0
    if axis == "energy" and status == "measured":
        payload["joule_per_inference"] = 1.25
    if axis == "ap" and status == "measured":
        payload["metric_value"] = 0.6
    return payload


class Stage2Original60QuantCompletionTest(unittest.TestCase):
    def test_measured_ap_index_rejects_width_mismatched_true_fp16_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows_path = tmp_path / "rows/fp16_true_original60_ap_rows_v1.jsonl"
            rows_path.parent.mkdir(parents=True, exist_ok=True)
            row_payload = {
                "schema": "ap_anchor_row_v1",
                "label": "s2_160",
                "precision": "fp16",
                "quant_policy": "fp16",
                "width": [64, 160, 256],
                "backend": "model_eval",
                "measurement_status": "measured",
                "metric": "AP70",
                "metric_value": 0.5925,
                "secondary_metrics": {"AP30": 0.7927, "AP50": 0.7513},
                "dataset": "DAIR-V2X",
                "eval_split": "val",
                "ckpt_path": "/tmp/ckpts/s2_160/net_epoch_bestval_at25.pth",
                "ckpt_digest": "fake_ckpt_digest",
                "eval_command": "python scripts/stage2_h800_true_fp16_ap_eval.py --label s2_160 --width 64,160,256",
                "run_id": "bad_width_s2_160",
                "source_files": ["/tmp/raw/s2_160/ap_eval_report.json"],
                "raw_artifact": "/tmp/raw/s2_160",
                "quant_method": "h800_tvm_true_fp16_onnx_relax",
                "quant_scope": "full_model_ap_eval_true_fp16",
                "layer_precision_summary": "/tmp/raw/s2_160/layer_precision_summary.json",
                "engine_kind": "model_eval",
                "measurement_source": "true_eval",
                "claim_status": "claimable_true_eval",
                "quality_gate_status": "true_fp16_original60_ap_eval",
                "full_network_claim": False,
                "created_at": "2026-06-28T00:00:00Z",
            }
            rows_path.write_text(json.dumps(row_payload, sort_keys=True) + "\n", encoding="utf-8")

            measured = measured_ap_index(
                [str(rows_path)],
                "fp16",
                {"s2_160": [64, 128, 160]},
            )

            self.assertNotIn("s2_160", measured)

    def test_completion_jobs_require_120_style_fp16_int8_cells_and_replace_old_int8_route(
        self,
    ) -> None:
        latency_rows = [
            row(
                label="s0_024",
                precision="fp16",
                axis="latency",
                status="measured",
                quality_gate_status="true_fp16_onnx_smoke_only",
                quant_method="h800_tvm_true_fp16_onnx_relax",
            ),
            row(
                label="s0_024",
                precision="int8",
                axis="latency",
                status="measured",
                quality_gate_status="int8_tvm_vm_latency_smoke_only",
                quant_method="h800_tvm_int8_backbone_subnet_experimental",
            ),
            row(
                label="s1_048",
                precision="fp16",
                axis="latency",
                status="no_claim",
                failure_reason="historical_fp16_tagged_latency_not_true_fp16_evidence",
            ),
            row(
                label="s1_048",
                precision="int8",
                axis="latency",
                status="no_claim",
                failure_reason="tvm_int8_backbone_subnet_not_ready",
            ),
        ]
        energy_rows = [
            row(
                label="s0_024",
                precision="fp16",
                axis="energy",
                status="measured",
                quality_gate_status="true_fp16_onnx_energy_smoke",
                quant_method="h800_tvm_true_fp16_onnx_relax",
            ),
            row(
                label="s0_024",
                precision="int8",
                axis="energy",
                status="measured",
                quality_gate_status="int8_tvm_vm_energy_smoke_only",
                quant_method="h800_tvm_int8_backbone_subnet_experimental",
            ),
            row(
                label="s1_048",
                precision="fp16",
                axis="energy",
                status="no_claim",
                failure_reason="fp16_energy_requires_true_fp16_h800_power_telemetry",
            ),
            row(
                label="s1_048",
                precision="int8",
                axis="energy",
                status="no_claim",
                failure_reason="tvm_int8_energy_backend_missing",
            ),
        ]
        ap_rows = [
            row(
                label="s0_024",
                precision="fp16",
                axis="ap",
                status="no_claim",
                failure_reason="fp16_ap_requires_true_fp16_eval_source_revalidation",
            ),
            row(
                label="s0_024",
                precision="int8",
                axis="ap",
                status="no_claim",
                failure_reason="tvm_int8_ap_eval_backend_missing",
            ),
            row(
                label="s1_048",
                precision="fp16",
                axis="ap",
                status="no_claim",
                failure_reason="fp16_ap_requires_true_fp16_eval_source_revalidation",
            ),
            row(
                label="s1_048",
                precision="int8",
                axis="ap",
                status="no_claim",
                failure_reason="tvm_int8_ap_eval_backend_missing",
            ),
        ]

        jobs = build_completion_jobs(
            latency_rows=latency_rows,
            energy_rows=energy_rows,
            ap_rows=ap_rows,
            created_at="2026-06-28T00:00:00Z",
        )

        self.assertEqual(len(jobs), 4)
        self.assertEqual({job["precision"] for job in jobs}, {"fp16", "int8"})
        self.assertTrue(all(job["schema"] == "original60_fp16_int8_completion_job_v1" for job in jobs))
        self.assertTrue(all(job["full_network_claim"] is False for job in jobs))
        self.assertTrue(all("onnx_backbone_path" in job for job in jobs))
        self.assertTrue(all("workdir" in job for job in jobs))

        fp16_s0 = next(job for job in jobs if job["label"] == "s0_024" and job["precision"] == "fp16")
        self.assertEqual(fp16_s0["latency_status"], "measured")
        self.assertEqual(fp16_s0["energy_status"], "measured")
        self.assertEqual(fp16_s0["ap_status"], "no_claim")
        self.assertEqual(fp16_s0["required_actions"], ["run_true_fp16_ap_eval"])

        int8_s0 = next(job for job in jobs if job["label"] == "s0_024" and job["precision"] == "int8")
        self.assertEqual(int8_s0["latency_status"], "needs_native_int8_replace")
        self.assertEqual(int8_s0["energy_status"], "needs_native_int8_replace")
        self.assertEqual(int8_s0["ap_status"], "no_claim")
        self.assertEqual(
            int8_s0["required_actions"],
            [
                "run_native_int8_full_onnx_latency",
                "run_native_int8_full_onnx_energy",
                "run_native_int8_ap_eval",
            ],
        )
        self.assertIn("old_int8_route", int8_s0["last_failure_reason"])

        summary = summarize_completion_jobs(jobs)
        self.assertEqual(summary["total_jobs"], 4)
        self.assertEqual(summary["precision_counts"], {"fp16": 2, "int8": 2})
        self.assertEqual(summary["jobs_requiring_action"], 4)
        self.assertEqual(summary["axis_status_counts"]["latency"]["needs_native_int8_replace"], 1)

    def test_write_completion_outputs_creates_queue_review_and_gap_files(self) -> None:
        jobs = [
            {
                "schema": "original60_fp16_int8_completion_job_v1",
                "job_id": "original60_completion:s0_024:int8",
                "label": "s0_024",
                "precision": "int8",
                "width": [24, 128, 256],
                "original60_candidate_id": "coverage:pyramid_lidar:s0_024",
                "onnx_backbone_path": "/exdata/jichengzhi/s2_tvm/models/s0_024_backbone.onnx",
                "workdir": "/exdata/jichengzhi/s2_tvm/workdirs/s0_024",
                "latency_status": "needs_native_int8_replace",
                "energy_status": "needs_native_int8_replace",
                "ap_status": "no_claim",
                "artifact_ready": False,
                "artifact_status": "pending_measurement",
                "required_actions": [
                    "run_native_int8_full_onnx_latency",
                    "run_native_int8_full_onnx_energy",
                    "run_native_int8_ap_eval",
                ],
                "last_failure_reason": "old_int8_route_requires_native_full_onnx_replacement",
                "full_network_claim": False,
                "axis_rows": {},
                "created_at": "2026-06-28T00:00:00Z",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            written = write_completion_outputs(
                output_root=output_root,
                jobs=jobs,
                created_at="2026-06-28T00:00:00Z",
            )

            self.assertEqual(
                written["queue"],
                output_root / "jobs/fp16_int8_original60_completion_queue_v1.jsonl",
            )
            queue_lines = written["queue"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(queue_lines), 1)
            self.assertEqual(json.loads(queue_lines[0])["job_id"], "original60_completion:s0_024:int8")

            review = json.loads(written["review_json"].read_text(encoding="utf-8"))
            self.assertEqual(review["summary"]["total_jobs"], 1)
            self.assertEqual(review["summary"]["jobs_requiring_action"], 1)
            self.assertEqual(review["jobs"][0]["required_actions"][0], "run_native_int8_full_onnx_latency")

            gap = json.loads(written["gap_json"].read_text(encoding="utf-8"))
            self.assertEqual(gap["gap_count"], 1)
            self.assertEqual(gap["rows"][0]["job_id"], "original60_completion:s0_024:int8")

            self.assertIn("s0_024", written["review_md"].read_text(encoding="utf-8"))
            self.assertIn("needs_native_int8_replace", written["gap_md"].read_text(encoding="utf-8"))

    def test_native_int8_label_widths_from_jobs_selects_only_native_int8_work(self) -> None:
        jobs = [
            {
                "label": "frontier_01",
                "precision": "int8",
                "width": [24, 64, 128],
                "required_actions": ["run_native_int8_full_onnx_latency"],
            },
            {
                "label": "s0_024",
                "precision": "int8",
                "width": [24, 128, 256],
                "required_actions": ["run_native_int8_full_onnx_energy"],
            },
            {
                "label": "s1_048",
                "precision": "fp16",
                "width": [64, 48, 256],
                "required_actions": ["run_true_fp16_latency"],
            },
            {
                "label": "already_done",
                "precision": "int8",
                "width": [64, 128, 256],
                "required_actions": [],
            },
        ]

        label_widths = native_int8_label_widths_from_jobs(jobs)

        self.assertEqual(
            label_widths,
            {
                "frontier_01": [24, 64, 128],
                "s0_024": [24, 128, 256],
            },
        )

    def test_fp16_label_widths_from_jobs_selects_only_true_fp16_work(self) -> None:
        jobs = [
            {
                "label": "frontier_01",
                "precision": "fp16",
                "width": [24, 64, 128],
                "required_actions": ["run_true_fp16_latency"],
            },
            {
                "label": "frontier_02",
                "precision": "fp16",
                "width": [40, 64, 128],
                "required_actions": ["run_true_fp16_energy"],
            },
            {
                "label": "s0_024",
                "precision": "int8",
                "width": [24, 128, 256],
                "required_actions": ["run_native_int8_full_onnx_latency"],
            },
            {
                "label": "already_done",
                "precision": "fp16",
                "width": [64, 128, 256],
                "required_actions": [],
            },
        ]

        label_widths = fp16_label_widths_from_jobs(jobs)

        self.assertEqual(
            label_widths,
            {
                "frontier_01": [24, 64, 128],
                "frontier_02": [40, 64, 128],
            },
        )

    def test_completion_queue_cli_reads_axis_rows_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "original60_quant"
            latency_path = tmp_path / "latency.jsonl"
            energy_path = tmp_path / "energy.jsonl"
            ap_path = tmp_path / "ap.jsonl"
            latency_path.write_text(
                json.dumps(
                    row(
                        label="s0_024",
                        precision="int8",
                        axis="latency",
                        status="measured",
                        quality_gate_status="int8_tvm_vm_latency_smoke_only",
                        quant_method="h800_tvm_int8_backbone_subnet_experimental",
                    ),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            energy_path.write_text(
                json.dumps(
                    row(
                        label="s0_024",
                        precision="int8",
                        axis="energy",
                        status="no_claim",
                        failure_reason="tvm_int8_energy_backend_missing",
                    ),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            ap_path.write_text(
                json.dumps(
                    row(
                        label="s0_024",
                        precision="int8",
                        axis="ap",
                        status="no_claim",
                        failure_reason="tvm_int8_ap_eval_backend_missing",
                    ),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_fp16_int8_original60_completion_queue.py"),
                    "--output-root",
                    str(output_root),
                    "--latency-rows",
                    str(latency_path),
                    "--energy-rows",
                    str(energy_path),
                    "--ap-rows",
                    str(ap_path),
                    "--created-at",
                    "2026-06-28T00:00:00Z",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            queue_path = output_root / "jobs/fp16_int8_original60_completion_queue_v1.jsonl"
            rows = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            int8_row = next(row for row in rows if row["precision"] == "int8")
            self.assertEqual(int8_row["label"], "s0_024")
            self.assertEqual(int8_row["latency_status"], "needs_native_int8_replace")
            self.assertTrue(
                (output_root / "exports/fp16_int8_original60_completion_review_latest.md").exists()
            )
            self.assertTrue(
                (output_root / "exports/fp16_int8_original60_gap_report_latest.json").exists()
            )

    def test_fp16_runner_uses_stage2_v2x_root_env_and_help_is_lightweight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env = {
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "STAGE2_V2X_ROOT": str(tmp_path),
            }
            help_result = subprocess.run(
                [sys.executable, str(FP16_RUNNER), "--help"],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("--emit-schedules", help_result.stdout)
            self.assertIn("--emit-energy", help_result.stdout)
            self.assertIn("--energy-out-jsonl", help_result.stdout)
            self.assertIn("--energy-iters", help_result.stdout)

            root_result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import importlib.util;"
                        f"spec=importlib.util.spec_from_file_location('fp16_runner', {str(FP16_RUNNER)!r});"
                        "module=importlib.util.module_from_spec(spec);"
                        "spec.loader.exec_module(module);"
                        "print(module.ROOT)"
                    ),
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(root_result.returncode, 0, root_result.stderr)
            self.assertEqual(Path(root_result.stdout.strip()), tmp_path)

    def test_fp16_runner_builds_true_fp16_energy_row_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env = {
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "STAGE2_V2X_ROOT": str(tmp_path),
            }
            script = (
                "import importlib.util;"
                f"spec=importlib.util.spec_from_file_location('fp16_runner', {str(FP16_RUNNER)!r});"
                "module=importlib.util.module_from_spec(spec);"
                "spec.loader.exec_module(module);"
                "cmd=module.build_energy_row_command("
                "label='frontier_01',"
                "width=[24,64,128],"
                "result={"
                "'layer_precision_summary_path':'/exdata/layer_precision_summary.json',"
                "'energy_rows':[{'schedule':'metaschedule_tuned','payload':{'run_id':'energy_run','latency_run_id':'latency_run'},'payload_path':'/tmp/telemetry_payload.json'}]"
                "},"
                "row={'schedule':'metaschedule_tuned','payload':{'run_id':'energy_run','latency_run_id':'latency_run'},'payload_path':'/tmp/telemetry_payload.json'},"
                "out_jsonl=module.Path('/tmp/fp16_energy_rows.jsonl')"
                ");"
                "print('\\n'.join(str(item) for item in cmd))"
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            command_text = result.stdout
            self.assertIn("scripts/stage2_generate_energy_lut.py", command_text)
            self.assertIn("--precision\nfp16", command_text)
            self.assertIn("--backend\nh800_tvm_power_telemetry", command_text)
            self.assertIn("--full-network-claim\nfalse", command_text)
            self.assertIn("--quality-gate-status\ntrue_fp16_onnx_energy", command_text)
            self.assertIn(
                "--software-point-id\ntrue_fp16_original60:frontier_01:24x64x128:fp16:energy",
                command_text,
            )
            self.assertIn("--out-jsonl\n/tmp/fp16_energy_rows.jsonl", command_text)

    def test_ap_true_eval_queue_cli_marks_compliant_sources_ready_and_blocks_bad_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "original60_quant"
            queue_path = output_root / "jobs/fp16_int8_original60_completion_queue_v1.jsonl"
            fp16_sources = tmp_path / "fp16_ap_sources.jsonl"
            int8_sources = tmp_path / "int8_ap_sources.jsonl"
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_rows = [
                {
                    "schema": "original60_fp16_int8_completion_job_v1",
                    "job_id": "original60_completion:s0_024:fp16",
                    "label": "s0_024",
                    "precision": "fp16",
                    "width": [24, 128, 256],
                    "onnx_backbone_path": "/exdata/jichengzhi/s2_tvm/models/s0_024_backbone.onnx",
                    "workdir": "/exdata/jichengzhi/s2_tvm/workdirs/s0_024",
                    "latency_status": "measured",
                    "energy_status": "measured",
                    "ap_status": "no_claim",
                    "required_actions": ["run_true_fp16_ap_eval"],
                    "full_network_claim": False,
                    "axis_rows": {},
                    "created_at": "2026-06-28T00:00:00Z",
                },
                {
                    "schema": "original60_fp16_int8_completion_job_v1",
                    "job_id": "original60_completion:s0_024:int8",
                    "label": "s0_024",
                    "precision": "int8",
                    "width": [24, 128, 256],
                    "onnx_backbone_path": "/exdata/jichengzhi/s2_tvm/models/s0_024_backbone.onnx",
                    "workdir": "/exdata/jichengzhi/s2_tvm/workdirs/s0_024",
                    "latency_status": "measured",
                    "energy_status": "measured",
                    "ap_status": "no_claim",
                    "required_actions": ["run_native_int8_ap_eval"],
                    "full_network_claim": False,
                    "axis_rows": {},
                    "created_at": "2026-06-28T00:00:00Z",
                },
            ]
            queue_path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in queue_rows),
                encoding="utf-8",
            )
            fp16_sources.write_text(
                json.dumps(
                    {
                        "schema": "ap_anchor_row_v1",
                        "label": "s0_024",
                        "candidate_id": "s0_024",
                        "precision": "fp16",
                        "quant_policy": "fp16",
                        "width": [24, 128, 256],
                        "backend": "model_eval",
                        "measurement_status": "measured",
                        "metric": "AP70",
                        "metric_value": 0.612345,
                        "secondary_metrics": {"AP30": 0.812345, "AP50": 0.712345},
                        "dataset": "DAIR-V2X",
                        "eval_split": "val",
                        "ckpt_path": "checkpoints/pyramid_lidar_true_fp16_s0_024.ckpt",
                        "ckpt_digest": "fp16_ckpt_digest",
                        "eval_command": "python tools/eval.py --precision true_fp16 --label s0_024",
                        "run_id": "true_fp16_ap_s0_024",
                        "source_files": ["raw/ap_eval_original60/fp16_true_s0_024/metrics.json"],
                        "raw_artifact": "raw/ap_eval_original60/fp16_true_s0_024",
                        "quant_method": "h800_tvm_true_fp16_onnx_relax",
                        "quant_scope": "backbone_only",
                        "layer_precision_summary": "raw/ap_eval_original60/fp16_true_s0_024/layer_precision_summary.json",
                        "engine_kind": "model_eval",
                        "measurement_source": "true_eval",
                        "claim_status": "claimable_true_eval",
                        "quality_gate_status": "true_fp16_original60_ap_eval",
                        "full_network_claim": False,
                        "created_at": "2026-06-28T00:00:00Z",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            int8_sources.write_text(
                json.dumps(
                    {
                        "schema": "ap_anchor_row_v1",
                        "label": "s0_024",
                        "candidate_id": "s0_024",
                        "precision": "int8",
                        "backend": "model_eval",
                        "measurement_status": "measured",
                        "metric": "AP70",
                        "metric_value": 0.9,
                        "secondary_metrics": {"AP30": 0.9, "AP50": 0.9},
                        "dataset": "DAIR-V2X",
                        "eval_split": "val",
                        "ckpt_path": "checkpoints/native_int8.ckpt",
                        "ckpt_digest": "int8_digest",
                        "eval_command": "python tools/eval.py --use-predicted-ap",
                        "source_files": ["raw/predicted/metrics.json"],
                        "raw_artifact": "raw/predicted",
                        "quant_method": "h800_tvm_native_int8_backbone_subnet",
                        "quant_scope": "backbone_subnet_native_int8",
                        "engine_kind": "model_eval",
                        "measurement_source": "predicted",
                        "claim_status": "claimable_true_eval",
                        "quality_gate_status": "predicted_model_fit_ap",
                        "full_network_claim": False,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_original60_quant_ap_true_eval_queue.py"),
                    "--output-root",
                    str(output_root),
                    "--completion-queue",
                    str(queue_path),
                    "--fp16-ap-source-rows",
                    str(fp16_sources),
                    "--int8-ap-source-rows",
                    str(int8_sources),
                    "--created-at",
                    "2026-06-28T00:00:00Z",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            ap_queue = [
                json.loads(line)
                for line in (
                    output_root / "jobs/fp16_int8_original60_ap_true_eval_queue_v1.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(ap_queue), 2)
            fp16_job = next(job for job in ap_queue if job["precision"] == "fp16")
            int8_job = next(job for job in ap_queue if job["precision"] == "int8")
            self.assertEqual(fp16_job["ap_eval_status"], "ready_for_import")
            self.assertEqual(fp16_job["next_action"], "import_compliant_ap_row")
            self.assertEqual(fp16_job["metric_value"], 0.612345)
            self.assertEqual(int8_job["ap_eval_status"], "blocked")
            self.assertEqual(int8_job["blocker"], "no_compliant_native_int8_model_eval_source")
            self.assertIn("build_or_connect_native_int8_eval_backend", int8_job["next_action"])

            blockers = [
                json.loads(line)
                for line in (
                    output_root / "quarantine/fp16_int8_original60_ap_true_eval_blockers_v1.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(blockers), 1)
            self.assertEqual(blockers[0]["label"], "s0_024")
            self.assertEqual(blockers[0]["precision"], "int8")

            audit = json.loads(
                (
                    output_root / "exports/fp16_int8_original60_ap_true_eval_source_audit_latest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(audit["summary"]["ready_for_import"], 1)
            self.assertEqual(audit["summary"]["blocked"], 1)
            self.assertIn("source_rejections", int8_job)

    def test_ap_true_eval_queue_rejects_width_mismatched_fp16_source_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_root = tmp_path / "generated/original60_quant_20260627"
            output_root.mkdir(parents=True)
            jobs_dir = output_root / "jobs"
            jobs_dir.mkdir(parents=True, exist_ok=True)
            queue_path = jobs_dir / "fp16_int8_original60_completion_queue_v1.jsonl"
            fp16_sources = tmp_path / "sources/fp16_width_mismatch.jsonl"

            queue_rows = [
                {
                    "schema": "original60_fp16_int8_completion_job_v1",
                    "job_id": "original60_completion:s2_160:fp16",
                    "label": "s2_160",
                    "precision": "fp16",
                    "width": [64, 128, 160],
                    "onnx_backbone_path": "/exdata/jichengzhi/s2_tvm/models/s2_160_backbone.onnx",
                    "workdir": "/exdata/jichengzhi/s2_tvm/workdirs/s2_160",
                    "latency_status": "measured",
                    "energy_status": "measured",
                    "ap_status": "no_claim",
                    "required_actions": ["run_true_fp16_ap_eval"],
                    "full_network_claim": False,
                    "axis_rows": {},
                    "created_at": "2026-06-28T00:00:00Z",
                }
            ]
            queue_path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in queue_rows),
                encoding="utf-8",
            )
            fp16_sources.parent.mkdir(parents=True, exist_ok=True)
            fp16_sources.write_text(
                json.dumps(
                    {
                        "schema": "ap_anchor_row_v1",
                        "label": "s2_160",
                        "candidate_id": "s2_160",
                        "precision": "fp16",
                        "quant_policy": "fp16",
                        "width": [64, 160, 256],
                        "backend": "model_eval",
                        "measurement_status": "measured",
                        "metric": "AP70",
                        "metric_value": 0.59252789126425,
                        "secondary_metrics": {"AP30": 0.7927004774506324, "AP50": 0.7513990229771785},
                        "dataset": "DAIR-V2X",
                        "eval_split": "val",
                        "ckpt_path": "checkpoints/pyramid_lidar_true_fp16_s2_160.ckpt",
                        "ckpt_digest": "fp16_ckpt_digest",
                        "eval_command": "python scripts/stage2_h800_true_fp16_ap_eval.py --precision true_fp16 --label s2_160 --width 64,160,256",
                        "run_id": "true_fp16_ap_s2_160_bad_width",
                        "source_files": ["raw/ap_eval_original60/fp16_true_s2_160/metrics.json"],
                        "raw_artifact": "raw/ap_eval_original60/fp16_true_s2_160",
                        "quant_method": "h800_tvm_true_fp16_onnx_relax",
                        "quant_scope": "full_model_ap_eval_true_fp16",
                        "layer_precision_summary": "raw/ap_eval_original60/fp16_true_s2_160/layer_precision_summary.json",
                        "engine_kind": "model_eval",
                        "measurement_source": "true_eval",
                        "claim_status": "claimable_true_eval",
                        "quality_gate_status": "true_fp16_original60_ap_eval",
                        "full_network_claim": False,
                        "created_at": "2026-06-28T00:00:00Z",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_generate_original60_quant_ap_true_eval_queue.py"),
                    "--output-root",
                    str(output_root),
                    "--completion-queue",
                    str(queue_path),
                    "--fp16-ap-source-rows",
                    str(fp16_sources),
                    "--no-default-source-rows",
                    "--created-at",
                    "2026-06-28T00:00:00Z",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            ap_queue = [
                json.loads(line)
                for line in (
                    output_root / "jobs/fp16_int8_original60_ap_true_eval_queue_v1.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(ap_queue), 1)
            fp16_job = ap_queue[0]
            self.assertEqual(fp16_job["label"], "s2_160")
            self.assertEqual(fp16_job["ap_eval_status"], "blocked")
            self.assertEqual(fp16_job["blocker"], "no_compliant_true_fp16_model_eval_source")
            self.assertTrue(
                any(
                    "width_mismatch_against_completion_queue" in reason
                    for rejection in fp16_job["source_rejections"]
                    for reason in rejection["reject_reasons"]
                )
            )

    def test_true_fp16_ap_eval_cli_converts_full_report_to_compliant_ap_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw/ap_eval_original60/fp16_true_s0_024"
            ckpt_dir = tmp_path / "ckpt/s0_024"
            rows_out = tmp_path / "rows/fp16_true_original60_ap_rows_v1.jsonl"
            raw_dir.mkdir(parents=True)
            ckpt_dir.mkdir(parents=True)
            ckpt = ckpt_dir / "net_epoch_bestval_at29.pth"
            config = ckpt_dir / "config.yaml"
            ckpt.write_bytes(b"fake-ckpt")
            config.write_text("name: fake\n", encoding="utf-8")
            report = {
                "status": "success",
                "label": "s0_024",
                "precision_mode": "model_half",
                "dataset": "DAIR-V2X",
                "eval_split": "val",
                "num_samples": 1789,
                "ap30": 0.8123,
                "ap50": 0.7123,
                "ap70": 0.6123,
                "ckpt_path": str(ckpt),
                "config_path": str(config),
                "eval_command": "python scripts/stage2_h800_true_fp16_ap_eval.py --execute ...",
            }
            report_path = raw_dir / "ap_eval_report.json"
            report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_h800_true_fp16_ap_eval.py"),
                    "--label",
                    "s0_024",
                    "--width",
                    "24,128,256",
                    "--ckpt-dir",
                    str(ckpt_dir),
                    "--raw-dir",
                    str(raw_dir),
                    "--report-json",
                    str(report_path),
                    "--rows-out",
                    str(rows_out),
                    "--created-at",
                    "2026-06-28T00:00:00Z",
                    "--emit-row-only",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in rows_out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["label"], "s0_024")
            self.assertEqual(row["precision"], "fp16")
            self.assertEqual(row["measurement_status"], "measured")
            self.assertEqual(row["metric"], "AP70")
            self.assertEqual(row["metric_value"], 0.6123)
            self.assertEqual(row["secondary_metrics"], {"AP30": 0.8123, "AP50": 0.7123})
            self.assertEqual(row["quant_method"], "h800_tvm_true_fp16_onnx_relax")
            self.assertEqual(row["measurement_source"], "true_eval")
            self.assertEqual(row["engine_kind"], "model_eval")
            self.assertEqual(row["quality_gate_status"], "true_fp16_original60_ap_eval")
            self.assertFalse(row["full_network_claim"])
            self.assertIn(str(report_path), row["source_files"])
            self.assertNotIn("trt", json.dumps(row).lower())

    def test_native_int8_full_ap_import_cli_converts_gated_report_to_compliant_ap_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw/int8_native_route/s0_024/ap_full"
            route_dir = tmp_path / "raw/int8_native_route/s0_024"
            rows_out = tmp_path / "rows/native_int8_original60_ap_rows_v1.jsonl"
            raw_dir.mkdir(parents=True)
            route_dir.mkdir(parents=True, exist_ok=True)
            tensor_quant = route_dir / "tensor_quant_params_calibration_v2_to_pyramid_level2.json"
            route_manifest = route_dir / "native_int8_route_manifest.json"
            output_dequant = raw_dir / "output_dequant_summary.json"
            worker_response = raw_dir / "worker_response_summary.json"
            postprocess_summary = raw_dir / "postprocess_summary.json"
            for path, payload in [
                (
                    tensor_quant,
                    {
                        "params": {
                            "pyramid_level0": {"scale": 0.2, "zero_point": 128},
                            "pyramid_level1": {"scale": 0.1, "zero_point": 128},
                            "pyramid_level2": {"scale": 0.05, "zero_point": 128},
                        }
                    },
                ),
                (route_manifest, {"route_spec": "full_onnx_topology_conv_relu_add_identity_v1"}),
                (
                    output_dequant,
                    {
                        "outputs": [
                            {"tensor_name": "pyramid_level0", "scheme": "tensor_quant_params_v2"},
                            {"tensor_name": "pyramid_level1", "scheme": "tensor_quant_params_v2"},
                            {"tensor_name": "pyramid_level2", "scheme": "tensor_quant_params_v2"},
                        ]
                    },
                ),
                (worker_response, {"status": "success"}),
                (postprocess_summary, {"pred_nonempty_count": 25}),
            ]:
                path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            report = {
                "label": "s0_024",
                "dataset": "DAIR-V2X",
                "eval_split": "val",
                "processed_samples": 1789,
                "failed_samples": 0,
                "pred_nonempty_count": 25,
                "pred_total_count": 77,
                "ap30": 0.4,
                "ap50": 0.3,
                "ap70": 0.2,
                "smoke_gate_passed": True,
                "ap_row_allowed": True,
                "ap_row_min_samples": 1789,
                "output_dequant_policy": "per_output_tensor_quant_params_v2_with_activation_quant_fallback",
            }
            report_path = raw_dir / "full_ap_eval_report.json"
            report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_import_native_int8_full_ap_row.py"),
                    "--label",
                    "s0_024",
                    "--width",
                    "24,128,256",
                    "--raw-dir",
                    str(raw_dir),
                    "--route-dir",
                    str(route_dir),
                    "--tensor-quant-params-path",
                    str(tensor_quant),
                    "--rows-out",
                    str(rows_out),
                    "--created-at",
                    "2026-06-28T00:00:00Z",
                    "--run-id",
                    "20260628_native_int8_s0_024_full_ap_v1",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in rows_out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["label"], "s0_024")
            self.assertEqual(row["precision"], "int8")
            self.assertEqual(row["measurement_status"], "measured")
            self.assertEqual(row["metric"], "AP70")
            self.assertEqual(row["metric_value"], 0.2)
            self.assertEqual(row["secondary_metrics"], {"AP30": 0.4, "AP50": 0.3})
            self.assertEqual(row["num_samples"], 1789)
            self.assertEqual(row["quant_method"], "h800_tvm_native_int8_backbone_subnet")
            self.assertEqual(row["quant_scope"], "backbone_subnet_native_int8")
            self.assertEqual(row["engine_kind"], "tvm_graph_executor")
            self.assertEqual(row["measurement_source"], "true_eval")
            self.assertEqual(row["quality_gate_status"], "native_int8_full_ap_eval")
            self.assertFalse(row["full_network_claim"])
            self.assertIn(str(report_path), row["source_files"])
            self.assertIn(str(output_dequant), row["source_files"])
            self.assertNotIn("trt", json.dumps(row).lower())

    def test_native_int8_full_ap_import_cli_accepts_items_output_dequant_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw/int8_native_route/s0_024/ap_full"
            route_dir = tmp_path / "raw/int8_native_route/s0_024"
            rows_out = tmp_path / "rows/native_int8_original60_ap_rows_v1.jsonl"
            raw_dir.mkdir(parents=True)
            route_dir.mkdir(parents=True, exist_ok=True)
            tensor_quant = route_dir / "tensor_quant_params_calibration_v2_to_pyramid_level2.json"
            route_manifest = route_dir / "native_int8_route_manifest.json"
            output_dequant = raw_dir / "output_dequant_summary.json"
            tensor_quant.write_text(
                json.dumps(
                    {
                        "params": {
                            "pyramid_level0": {"scale": 0.2, "zero_point": 128},
                            "pyramid_level1": {"scale": 0.1, "zero_point": 128},
                            "pyramid_level2": {"scale": 0.05, "zero_point": 128},
                        }
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            route_manifest.write_text(
                json.dumps({"route_spec": "full_onnx_topology_scale_aware_tensor_quant_params_v2"}, sort_keys=True),
                encoding="utf-8",
            )
            output_dequant.write_text(
                json.dumps(
                    {
                        "items": [
                            {"tensor_name": "pyramid_level0", "scheme": "tensor_quant_params_v2", "agent_index": 0},
                            {"tensor_name": "pyramid_level1", "scheme": "tensor_quant_params_v2", "agent_index": 0},
                            {"tensor_name": "pyramid_level2", "scheme": "tensor_quant_params_v2", "agent_index": 0},
                        ],
                        "tensor_quant_params_path": str(tensor_quant),
                        "full_network_claim": False,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            report_path = raw_dir / "full_ap_eval_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "label": "s0_024",
                        "dataset": "DAIR-V2X",
                        "eval_split": "val",
                        "processed_samples": 1789,
                        "failed_samples": 0,
                        "pred_nonempty_count": 1787,
                        "pred_total_count": 2500,
                        "ap30": 0.0016761650535494891,
                        "ap50": 0.0011790254856126326,
                        "ap70": 0.00018487973338387518,
                        "smoke_gate_passed": True,
                        "ap_row_allowed": True,
                        "ap_row_min_samples": 1789,
                        "output_dequant_policy": "per_output_tensor_quant_params_v2_with_activation_quant_fallback",
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_import_native_int8_full_ap_row.py"),
                    "--label",
                    "s0_024",
                    "--width",
                    "24,128,256",
                    "--raw-dir",
                    str(raw_dir),
                    "--route-dir",
                    str(route_dir),
                    "--tensor-quant-params-path",
                    str(tensor_quant),
                    "--rows-out",
                    str(rows_out),
                    "--created-at",
                    "2026-06-28T00:00:00Z",
                    "--run-id",
                    "20260628_native_int8_s0_024_full_ap_items_schema_v1",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in rows_out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["precision"], "int8")
            self.assertEqual(rows[0]["pred_nonempty_count"], 1787)
            self.assertEqual(rows[0]["metric_value"], 0.00018487973338387518)

    def test_native_int8_full_ap_import_row_is_counted_by_original60_state_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw/int8_native_route/s0_024/ap_full"
            route_dir = tmp_path / "raw/int8_native_route/s0_024"
            rows_out = tmp_path / "rows/native_int8_original60_ap_rows_v1.jsonl"
            raw_dir.mkdir(parents=True)
            route_dir.mkdir(parents=True, exist_ok=True)
            tensor_quant = route_dir / "tensor_quant_params_calibration_v2_to_pyramid_level2.json"
            route_manifest = route_dir / "native_int8_route_manifest.json"
            output_dequant = raw_dir / "output_dequant_summary.json"
            worker_response = raw_dir / "worker_response_summary.json"
            postprocess_summary = raw_dir / "postprocess_summary.json"
            ckpt_path = route_dir / "net_epoch_bestval_at29.pth"
            ckpt_path.write_bytes(b"fake-ckpt")
            for path, payload in [
                (
                    tensor_quant,
                    {
                        "params": {
                            "pyramid_level0": {"scale": 0.2, "zero_point": 128},
                            "pyramid_level1": {"scale": 0.1, "zero_point": 128},
                            "pyramid_level2": {"scale": 0.05, "zero_point": 128},
                        }
                    },
                ),
                (route_manifest, {"route_spec": "full_onnx_topology_scale_aware_tensor_quant_params_v2"}),
                (
                    output_dequant,
                    {
                        "items": [
                            {"tensor_name": "pyramid_level0", "scheme": "tensor_quant_params_v2"},
                            {"tensor_name": "pyramid_level1", "scheme": "tensor_quant_params_v2"},
                            {"tensor_name": "pyramid_level2", "scheme": "tensor_quant_params_v2"},
                        ]
                    },
                ),
                (worker_response, {"status": "success"}),
                (postprocess_summary, {"pred_nonempty_count": 1787}),
            ]:
                path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            report_path = raw_dir / "full_ap_eval_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "label": "s0_024",
                        "dataset": "DAIR-V2X",
                        "eval_split": "val",
                        "processed_samples": 1789,
                        "pred_nonempty_count": 1787,
                        "pred_total_count": 54235,
                        "ap30": 0.0016761650535494891,
                        "ap50": 0.0011790254856126326,
                        "ap70": 0.00018487973338387518,
                        "ap_row_allowed": True,
                        "ap_row_min_samples": 1789,
                        "output_dequant_policy": "per_output_tensor_quant_params_v2_with_activation_quant_fallback",
                        "ckpt_path": str(ckpt_path),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_import_native_int8_full_ap_row.py"),
                    "--label",
                    "s0_024",
                    "--width",
                    "24,128,256",
                    "--raw-dir",
                    str(raw_dir),
                    "--route-dir",
                    str(route_dir),
                    "--tensor-quant-params-path",
                    str(tensor_quant),
                    "--rows-out",
                    str(rows_out),
                    "--created-at",
                    "2026-06-28T00:00:00Z",
                    "--run-id",
                    "20260628_native_int8_s0_024_full_ap_state_coverage_v1",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            measured = measured_ap_index([str(rows_out)], "int8")
            self.assertIn("s0_024", measured)
            self.assertEqual(measured["s0_024"]["measurement_source"], "true_eval")

    def test_native_int8_full_ap_import_cli_rejects_smoke_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw/int8_native_route/s0_024/ap_smoke"
            route_dir = tmp_path / "raw/int8_native_route/s0_024"
            rows_out = tmp_path / "rows/native_int8_original60_ap_rows_v1.jsonl"
            raw_dir.mkdir(parents=True)
            route_dir.mkdir(parents=True, exist_ok=True)
            tensor_quant = route_dir / "tensor_quant_params_calibration_v2_to_pyramid_level2.json"
            output_dequant = raw_dir / "output_dequant_summary.json"
            tensor_quant.write_text(json.dumps({"params": {}}), encoding="utf-8")
            output_dequant.write_text(
                json.dumps(
                    {
                        "outputs": [
                            {"tensor_name": "pyramid_level0", "scheme": "tensor_quant_params_v2"},
                            {"tensor_name": "pyramid_level1", "scheme": "tensor_quant_params_v2"},
                            {"tensor_name": "pyramid_level2", "scheme": "tensor_quant_params_v2"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report_path = raw_dir / "full_ap_eval_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "processed_samples": 1,
                        "pred_nonempty_count": 1,
                        "ap30": 0.0,
                        "ap50": 0.0,
                        "ap70": 0.0,
                        "smoke_gate_passed": True,
                        "ap_row_allowed": False,
                        "ap_row_block_reason": "full_eval_num_samples_1_lt_1789",
                        "ap_row_min_samples": 1789,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_import_native_int8_full_ap_row.py"),
                    "--label",
                    "s0_024",
                    "--width",
                    "24,128,256",
                    "--raw-dir",
                    str(raw_dir),
                    "--route-dir",
                    str(route_dir),
                    "--tensor-quant-params-path",
                    str(tensor_quant),
                    "--rows-out",
                    str(rows_out),
                    "--created-at",
                    "2026-06-28T00:00:00Z",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertFalse(rows_out.exists())
            blocker = json.loads(
                (raw_dir / "native_int8_ap_row_import_blocker.json").read_text(encoding="utf-8")
            )
            self.assertEqual(blocker["label"], "s0_024")
            self.assertIn("partial eval", blocker["failure_reason"])

    def test_true_fp16_ap_eval_execute_failure_writes_blocker_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw/ap_eval_original60/fp16_true_s0_024_missing_ckpt"
            rows_out = tmp_path / "rows/fp16_true_original60_ap_rows_v1.jsonl"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_h800_true_fp16_ap_eval.py"),
                    "--label",
                    "s0_024",
                    "--width",
                    "24,128,256",
                    "--ckpt-dir",
                    str(tmp_path / "missing_ckpt"),
                    "--raw-dir",
                    str(raw_dir),
                    "--rows-out",
                    str(rows_out),
                    "--created-at",
                    "2026-06-28T00:00:00Z",
                    "--execute",
                    "--precision-mode",
                    "amp_fp16",
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 2)
            blocker_path = raw_dir / "ap_eval_blocker.json"
            self.assertTrue(blocker_path.exists())
            blocker = json.loads(blocker_path.read_text(encoding="utf-8"))
            self.assertEqual(blocker["label"], "s0_024")
            self.assertEqual(blocker["precision"], "fp16")
            self.assertEqual(blocker["failure_type"], "configuration_error")
            self.assertIn("missing_ckpt", blocker["failure_reason"])
            self.assertEqual(blocker["stdout_path"], str(raw_dir / "stdout.txt"))
            self.assertEqual(blocker["stderr_path"], str(raw_dir / "stderr.txt"))
            self.assertTrue((raw_dir / "stdout.txt").exists())
            self.assertTrue((raw_dir / "stderr.txt").exists())
            self.assertIn("missing_ckpt", (raw_dir / "stderr.txt").read_text(encoding="utf-8"))
            self.assertFalse(rows_out.exists())

    def test_true_fp16_ap_eval_keeps_relative_raw_dir_under_launch_cwd_after_chdir_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt_dir = tmp_path / "ckpt/s0_024"
            heal_root = tmp_path / "empty_heal_root"
            ckpt_dir.mkdir(parents=True)
            heal_root.mkdir(parents=True)
            (ckpt_dir / "config.yaml").write_text("name: fake\n", encoding="utf-8")
            (ckpt_dir / "net_epoch_bestval_at29.pth").write_bytes(b"fake-ckpt")
            raw_dir = Path("raw/ap_eval_original60/fp16_true_s0_024_relative_failure")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2_h800_true_fp16_ap_eval.py"),
                    "--label",
                    "s0_024",
                    "--width",
                    "24,128,256",
                    "--ckpt-dir",
                    str(ckpt_dir),
                    "--raw-dir",
                    str(raw_dir),
                    "--rows-out",
                    str(tmp_path / "rows/fp16_true_original60_ap_rows_v1.jsonl"),
                    "--heal-root",
                    str(heal_root),
                    "--created-at",
                    "2026-06-28T00:00:00Z",
                    "--execute",
                    "--precision-mode",
                    "amp_fp16",
                ],
                cwd=tmp_path,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 2)
            expected_blocker = tmp_path / raw_dir / "ap_eval_blocker.json"
            wrong_blocker = heal_root / raw_dir / "ap_eval_blocker.json"
            self.assertTrue(expected_blocker.exists())
            self.assertFalse(wrong_blocker.exists())
            blocker = json.loads(expected_blocker.read_text(encoding="utf-8"))
            self.assertEqual(blocker["raw_artifact"], str(tmp_path / raw_dir))
            self.assertIn("ModuleNotFoundError", blocker["failure_reason"])

    def test_true_fp16_runner_casts_model_outputs_to_fp32_before_postprocess(self) -> None:
        import torch

        spec = importlib.util.spec_from_file_location(
            "stage2_h800_true_fp16_ap_eval",
            ROOT / "scripts/stage2_h800_true_fp16_ap_eval.py",
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        class DummyModel:
            def __call__(self, cav_content):
                self.last_dtype = cav_content["x"].dtype
                return {
                    "reg": torch.ones((1, 2), dtype=torch.float16),
                    "nested": [torch.ones((1,), dtype=torch.float16)],
                    "count": torch.tensor([1], dtype=torch.int64),
                }

        class DummyDataset:
            def post_process(self, batch_data, output_dict):
                ego = output_dict["ego"]
                self.output_dtype = ego["reg"].dtype
                self.nested_dtype = ego["nested"][0].dtype
                self.count_dtype = ego["count"].dtype
                return (
                    torch.ones((1, 8, 3), dtype=torch.float32),
                    torch.ones((1,), dtype=torch.float32),
                    torch.ones((1, 8, 3), dtype=torch.float32),
                )

        batch = {"ego": {"x": torch.ones((1,), dtype=torch.float16)}}
        model = DummyModel()
        dataset = DummyDataset()

        result = module.inference_intermediate_fusion_fp32_postprocess(
            batch,
            model,
            dataset,
        )

        self.assertEqual(model.last_dtype, torch.float16)
        self.assertEqual(dataset.output_dtype, torch.float32)
        self.assertEqual(dataset.nested_dtype, torch.float32)
        self.assertEqual(dataset.count_dtype, torch.int64)
        self.assertEqual(result["pred_box_tensor"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
