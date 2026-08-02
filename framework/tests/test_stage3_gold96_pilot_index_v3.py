from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "stage3_gold96_pilot_index_v3.py"
SPEC = importlib.util.spec_from_file_location("stage3_gold96_pilot_index_v3", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)


def _manifest() -> dict:
    jobs = []
    pilots = {"pyramid": "32x32x128", "codriving": "32x64x128"}
    for model, pilot_width in pilots.items():
        for index in range(12):
            width = pilot_width if index == 0 else f"{index}x{index + 1}x{index + 2}"
            for dispatch, q_mode in (("tvm_auto", "fp16"), ("tvm_auto", "int8"), ("trt_engine", "fp16"), ("trt_engine", "int8")):
                profile = f"h800-{'tvm' if dispatch == 'tvm_auto' else 'trt'}-probe-conditioned-v3"
                jobs.append({
                    "job_id": f"{model}|{width}|q={q_mode}|profile={profile}",
                    "group_id": f"{model}|{width}",
                    "model": model,
                    "width": [int(value) for value in width.split("x")],
                    "width_key": width,
                    "q_mode": q_mode,
                    "dispatch_key": dispatch,
                    "capability_profile_id": profile,
                })
    return {
        "schema_version": "stage3_gold_coldstart96_manifest_v3",
        "pilot_group_ids": ["pyramid|32x32x128", "codriving|32x64x128"],
        "jobs": jobs,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_evidence(root: Path) -> None:
    for model, width in (("pyramid", "32x32x128"), ("codriving", "32x64x128")):
        group = f"{model}|{width}"
        label = f"{model}_{width}"
        for runner in ("tvm_fp16", "tvm_int8", "trt_fp16", "trt_int8"):
            result, artifact = MODULE._performance_paths(root, model, width, runner)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f"{group}:{runner}".encode())
            if runner.startswith("trt"):
                payload = {"artifact_sha256": {"compiled_engine": _sha(artifact)}}
            elif runner == "tvm_fp16":
                payload = {"artifact_path": str(artifact)}
            else:
                payload = {"artifact_path": str(artifact)}
            result.write_text(json.dumps(payload), encoding="utf-8")

    reports = (
        ("ap/pyramid_trt_fp16_full/stage3_trt_multiscale_ap_bridge_report.json", {"engine_ap_claim": True, "processed_samples": 1789, "ap30": 0.61, "ap50": 0.52, "ap70": 0.31}),
        ("ap/pyramid_trt_int8_full/stage3_trt_multiscale_ap_bridge_report.json", {"engine_ap_claim": True, "processed_samples": 1789, "ap30": 0.60, "ap50": 0.51, "ap70": 0.30}),
        ("ap/pyramid_tvm_fp16_full_v4/fp16_rewritten_activation_bridge_report.json", {"ap_measured": True, "smoke_gate_passed": True, "processed_samples": 1789, "ap30": 0.59, "ap50": 0.50, "ap70": 0.29}),
        ("ap/pyramid_tvm_int8_sanity_v4/output_dequant_calibration_summary.json", {"status": "blocked", "processed_samples": 16, "outputs": {"pyramid_level0": {"passed": False}}}),
    )
    for relative_path, payload in reports:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


class Stage3Gold96PilotIndexV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        SPEC.loader.exec_module(MODULE)

    def test_builds_eight_strict_performance_jobs_and_success_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_evidence(root)
            jobs, states, seeds = MODULE.build_indexes(_manifest(), root)

        self.assertEqual(len(jobs), 8)
        self.assertEqual(len(states), 8)
        self.assertEqual({row["status"] for row in states}, {"success"})
        self.assertEqual({row["group_id"] for row in jobs}, {"pyramid|32x32x128", "codriving|32x64x128"})
        self.assertEqual({row["runner_key"] for row in jobs}, {"tvm_fp16", "tvm_int8", "trt_fp16", "trt_int8"})
        self.assertTrue(all(len(row["result_sha256"]) == 64 and len(row["artifact_sha256"]) == 64 for row in states))
        self.assertEqual(len(seeds), 4)

    def test_ap_seed_contains_only_real_pyramid_terminal_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_evidence(root)
            _, _, seeds = MODULE.build_indexes(_manifest(), root)
            reports = {Path(row["report_json"]): row for row in seeds}
            sha_matches = all(row["report_sha256"] == _sha(path) for path, row in reports.items())

        self.assertEqual([row["status"] for row in seeds], ["success", "success", "success", "failed"])
        self.assertEqual([row["stage"] for row in seeds], ["full", "full", "full", "sanity"])
        self.assertTrue(all(row["model"] == "pyramid" for row in seeds))
        self.assertTrue(all("ap" in row for row in seeds[:3]))
        self.assertEqual(seeds[3]["failure_class"], "feasibility_failure")
        self.assertEqual(seeds[3]["blockers"], ["pyramid_level0"])
        self.assertTrue(all(row["job_id"] == row["manifest_job_id"] for row in seeds))
        self.assertTrue(all(row["performance_job_id"] != row["job_id"] for row in seeds))
        self.assertTrue(sha_matches)

    def test_rejects_manifest_with_wrong_pilot_groups(self) -> None:
        manifest = _manifest()
        manifest["pilot_group_ids"] = ["pyramid|32x64x128", "codriving|32x64x128"]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "pilot_group_ids"):
                MODULE.build_indexes(manifest, tmp)

    def test_missing_known_result_does_not_fall_back_to_similar_or_v2_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_evidence(root)
            required, _ = MODULE._performance_paths(root, "pyramid", "32x32x128", "trt_fp16")
            required.unlink()
            decoy = root / "v2_results" / "pyramid|32x32x128" / "trt_fp16" / "trt_profile_result.json"
            decoy.parent.mkdir(parents=True)
            decoy.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing performance result"):
                MODULE.build_indexes(_manifest(), root)

    def test_rejects_tvm_result_that_points_away_from_fixed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_evidence(root)
            result, _ = MODULE._performance_paths(root, "pyramid", "32x32x128", "tvm_fp16")
            result.write_text(json.dumps({"artifact_path": str(root / "elsewhere.so")}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixed compiled artifact"):
                MODULE.build_indexes(_manifest(), root)

    def test_rejects_trt_result_whose_compiled_engine_sha_does_not_match_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_evidence(root)
            result, _ = MODULE._performance_paths(root, "codriving", "32x64x128", "trt_fp16")
            result.write_text(json.dumps({"artifact_sha256": {"compiled_engine": "0" * 64}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "compiled_engine SHA256"):
                MODULE.build_indexes(_manifest(), root)

    def test_cli_writes_three_planner_readable_jsonl_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pilot"
            out = Path(tmp) / "index"
            manifest_path = Path(tmp) / "manifest.json"
            _write_evidence(root)
            manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--manifest-json", str(manifest_path), "--pilot-root", str(root), "--output-dir", str(out)],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(len((out / "pilot_jobs.jsonl").read_text().splitlines()), 8)
            self.assertEqual(len((out / "pilot_state.jsonl").read_text().splitlines()), 8)
            self.assertEqual(len((out / "ap_seed_state.jsonl").read_text().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
