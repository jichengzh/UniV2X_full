from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import stage2_original60_fp16_lane_runner as lane_runner


class Stage2Original60Fp16LaneRunnerTest(unittest.TestCase):
    def _args(self, root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            gpu_id=3,
            queue=str(root / "queue.jsonl"),
            rows_out=str(root / "rows.jsonl"),
            ckpt_root=str(root / "ckpts"),
            raw_root=str(root / "raw"),
            env_python="/usr/bin/python3",
            master_port_base=29730,
            labels="frontier_01,frontier_02",
            poll_seconds=1,
            watcher_poll_seconds=60,
        )

    def test_lane_runner_continues_after_blocked_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root)
            queue = {
                "frontier_01": {"width": [24, 64, 128]},
                "frontier_02": {"width": [32, 64, 128]},
            }
            call_order: list[str] = []

            def fake_wait(_args: argparse.Namespace, label: str, raw_dir: Path) -> None:
                call_order.append(label)
                if label == "frontier_01":
                    raise RuntimeError(f"watcher ended with status=blocked for {label}")
                (Path(_args.rows_out)).write_text(
                    json.dumps({"label": label, "measurement_status": "measured"}) + "\n",
                    encoding="utf-8",
                )

            with mock.patch.object(lane_runner, "completion_jobs", return_value=queue), mock.patch.object(
                lane_runner, "load_measured_labels", side_effect=[set(), set(), set(), {"frontier_02"}]
            ), mock.patch.object(lane_runner, "ps_alive", return_value=False), mock.patch.object(
                lane_runner, "launch_train", side_effect=[111, 222]
            ), mock.patch.object(lane_runner, "ensure_watcher", side_effect=[333, 444]), mock.patch.object(
                lane_runner, "wait_for_label", side_effect=fake_wait
            ):
                result = lane_runner.run_lane(args)

            self.assertEqual(result, 0)
            self.assertEqual(call_order, ["frontier_01", "frontier_02"])
            status = json.loads((root / "raw/fp16_lane_gpu3_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "lane_complete")
            failure = json.loads(
                (root / "raw/fp16_ckpt_gen_frontier_01_gpu3_v1/lane_failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["label"], "frontier_01")
            self.assertEqual(failure["status"], "label_failed")

    def test_launch_train_allows_checkpoint_reuse_without_train_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root)
            ckpt_dir = root / "ckpts/frontier_16"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            raw_dir = root / "raw/fp16_ckpt_gen_frontier_16_gpu3_v1"
            raw_dir.mkdir(parents=True, exist_ok=True)

            def fake_run(*_args, **_kwargs):
                (raw_dir / "train_skip.json").write_text(
                    json.dumps({"reason": "post_init_checkpoint_exists"}),
                    encoding="utf-8",
                )
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch("subprocess.run", side_effect=fake_run):
                train_pid = lane_runner.launch_train(args, "frontier_16", [48, 80, 192], ckpt_dir, raw_dir, 0)

            self.assertEqual(train_pid, 0)

    def test_ensure_watcher_uses_dedicated_watcher_poll_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root)
            raw_dir = root / "raw_dir"
            raw_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir = root / "ckpt_dir"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            with mock.patch.object(lane_runner, "ps_alive", return_value=False), mock.patch(
                "subprocess.Popen"
            ) as popen:
                popen.return_value.pid = 777
                lane_runner.ensure_watcher(args, "frontier_01", [24, 64, 128], ckpt_dir, raw_dir, 123)

            command = popen.call_args.kwargs["args"] if "args" in popen.call_args.kwargs else popen.call_args.args[0]
            self.assertIn("--poll-seconds", command)
            poll_index = command.index("--poll-seconds")
            self.assertEqual(command[poll_index + 1], "60")

    def test_ps_alive_returns_false_for_zombie_process(self) -> None:
        proc = mock.Mock(returncode=0, stdout=" 123 Z\n")
        with mock.patch("subprocess.run", return_value=proc):
            self.assertFalse(lane_runner.ps_alive(123))

    def test_ensure_watcher_restarts_when_pid_file_points_to_zombie(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root)
            raw_dir = root / "raw_dir"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "eval_watcher.pid").write_text("555\n", encoding="utf-8")
            ckpt_dir = root / "ckpt_dir"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            with mock.patch.object(lane_runner, "ps_alive", return_value=False), mock.patch(
                "subprocess.Popen"
            ) as popen:
                popen.return_value.pid = 777
                watcher_pid = lane_runner.ensure_watcher(args, "frontier_01", [24, 64, 128], ckpt_dir, raw_dir, 123)

            self.assertEqual(watcher_pid, 777)
            self.assertEqual((raw_dir / "eval_watcher.pid").read_text(encoding="utf-8").strip(), "777")


if __name__ == "__main__":
    unittest.main()
