from __future__ import annotations

import argparse
import unittest
from unittest import mock
from pathlib import Path

from tools.orin_deploy.lane_c_orin_runner import (
    _start_tegrastats,
    _stop_tegrastats,
    classify_inspector_precisions,
    validate_output_paths,
    validate_repeat_contract,
)


class LaneCOrinRunnerTest(unittest.TestCase):
    @mock.patch("tools.orin_deploy.lane_c_orin_runner.subprocess.Popen")
    @mock.patch("pathlib.Path.open")
    def test_start_tegrastats_scrubs_password_from_child_environment(
        self, path_open: mock.Mock, popen: mock.Mock
    ) -> None:
        process = popen.return_value
        process.stdin = mock.Mock()
        with mock.patch.dict("os.environ", {"ORIN_SUDO_PW": "temporary"}):
            _start_tegrastats(mock.Mock())
        child_env = popen.call_args.kwargs["env"]
        self.assertNotIn("ORIN_SUDO_PW", child_env)

    def test_classify_inspector_precisions_reports_mixed_engine(self) -> None:
        inspector = {
            "Layers": [
                {"Name": "conv0", "Inputs": [{"Format/Datatype": "Int8"}]},
                {"Name": "relu0", "Outputs": [{"Format/Datatype": "Half"}]},
                {"Name": "softmax", "Outputs": [{"Format/Datatype": "Float"}]},
                {"Name": "shuffle", "LayerType": "Shuffle"},
            ]
        }
        self.assertEqual(
            classify_inspector_precisions(inspector),
            {"int8": 1, "fp16": 1, "fp32": 1, "other": 1},
        )

    def test_validate_repeat_contract_accepts_both_count_and_duration(self) -> None:
        validate_repeat_contract(
            warmup_count=200,
            warmup_seconds=10.0,
            inference_count=500,
            measurement_seconds=10.0,
            power_sample_count=20,
        )

    def test_validate_repeat_contract_rejects_short_measurement(self) -> None:
        with self.assertRaisesRegex(ValueError, "measurement_seconds"):
            validate_repeat_contract(
                warmup_count=200,
                warmup_seconds=10.0,
                inference_count=500,
                measurement_seconds=9.99,
                power_sample_count=20,
            )

    def test_validate_repeat_contract_rejects_missing_power_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "power_sample_count"):
            validate_repeat_contract(
                warmup_count=200,
                warmup_seconds=10.0,
                inference_count=500,
                measurement_seconds=10.0,
                power_sample_count=0,
            )

    @mock.patch("tools.orin_deploy.lane_c_orin_runner.subprocess.run")
    def test_stop_tegrastats_uses_privileged_stop_command(self, run: mock.Mock) -> None:
        process = mock.Mock()
        handle = mock.Mock()
        with mock.patch.dict("os.environ", {"ORIN_SUDO_PW": "temporary"}):
            _stop_tegrastats(process, handle)
        run.assert_called_once_with(
            ["sudo", "-S", "-p", "", "tegrastats", "--stop"],
            input="temporary\n",
            text=True,
            check=True,
            timeout=5,
            env=mock.ANY,
        )
        self.assertNotIn("ORIN_SUDO_PW", run.call_args.kwargs["env"])
        process.wait.assert_called_once_with(timeout=5)
        handle.close.assert_called_once_with()

    @mock.patch("tools.orin_deploy.lane_c_orin_runner.subprocess.run")
    def test_stop_tegrastats_falls_back_to_privileged_kill(
        self, run: mock.Mock
    ) -> None:
        run.side_effect = [
            OSError("stop failed"),
            mock.Mock(returncode=0),
        ]
        process = mock.Mock()
        handle = mock.Mock()
        _stop_tegrastats(process, handle)
        self.assertEqual(
            run.call_args_list[1].args[0],
            ["sudo", "-S", "-p", "", "pkill", "-x", "tegrastats"],
        )
        process.wait.assert_called_once_with(timeout=5)
        handle.close.assert_called_once_with()

    @mock.patch("tools.orin_deploy.lane_c_orin_runner.subprocess.run")
    def test_stop_tegrastats_waits_even_when_stop_and_fallback_fail(
        self, run: mock.Mock
    ) -> None:
        run.side_effect = [OSError("stop failed"), OSError("pkill failed")]
        process = mock.Mock()
        handle = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "failed to stop tegrastats"):
            _stop_tegrastats(process, handle)
        process.wait.assert_called_once_with(timeout=5)
        handle.close.assert_called_once_with()

    def test_output_paths_must_stay_inside_artifact_root(self) -> None:
        root = Path("/tmp/lane-c-root")
        valid = argparse.Namespace(
            command="numerical",
            artifact_root=root,
            output_npz=root / "numerical/output.npz",
            report_json=root / "numerical/report.json",
        )
        validate_output_paths(valid)

        escaped = argparse.Namespace(
            command="numerical",
            artifact_root=root,
            output_npz=Path("/tmp/outside.npz"),
            report_json=root / "numerical/report.json",
        )
        with self.assertRaisesRegex(ValueError, "outside artifact root"):
            validate_output_paths(escaped)


if __name__ == "__main__":
    unittest.main()
