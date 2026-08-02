from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fcooper_tvm_subprocess_runner_v1 import SharedMemoryTvmClient, build_worker_env


class SharedMemoryTvmClientTests(unittest.TestCase):
    def test_persistent_worker_exchanges_exact_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / "fake_worker.py"
            worker.write_text(
                textwrap.dedent(
                    """
                    import argparse
                    import json
                    import sys
                    import numpy as np

                    parser = argparse.ArgumentParser()
                    parser.add_argument("--input-buffer")
                    parser.add_argument("--output-buffer")
                    args = parser.parse_args()
                    source = np.memmap(
                        args.input_buffer, mode="r+", dtype=np.float32, shape=(2, 3)
                    )
                    target = np.memmap(
                        args.output_buffer, mode="r+", dtype=np.float16, shape=(2, 3)
                    )
                    print(json.dumps({"status": "ready"}), flush=True)
                    for command in sys.stdin:
                        if command.strip() == "close":
                            print(json.dumps({"status": "closed"}), flush=True)
                            break
                        target[:] = source * 2
                        print(json.dumps({"status": "success"}), flush=True)
                    """
                )
            )
            client = SharedMemoryTvmClient(
                worker_command=[sys.executable, str(worker)],
                input_shape=(2, 3),
                output_shape=(2, 3),
                input_dtype="float32",
                output_dtype="float16",
                workspace=root / "ipc",
            )
            try:
                first = np.arange(6, dtype=np.float32).reshape(2, 3)
                second = first + 10
                np.testing.assert_array_equal(client.execute(first), first * 2)
                np.testing.assert_array_equal(client.execute(second), second * 2)
                self.assertEqual(client.call_count, 2)
            finally:
                client.close()

    def test_rejects_input_shape_or_dtype_drift_before_dispatch(self) -> None:
        client = SharedMemoryTvmClient.__new__(SharedMemoryTvmClient)
        client.input_shape = (2, 3)
        client.input_dtype = np.dtype("float32")
        client.call_count = 0

        with self.assertRaisesRegex(ValueError, "shape drift"):
            client.execute(np.zeros((1, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "dtype drift"):
            client.execute(np.zeros((2, 3), dtype=np.float16))
        self.assertEqual(client.call_count, 0)

    def test_worker_preserves_parent_physical_gpu_mapping(self) -> None:
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "6"}, clear=True):
            env = build_worker_env(
                tvm_site=Path("/tvm/site"),
                tvm_lib_dirs=(Path("/tvm/lib"),),
                gpu_id=0,
            )

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "6")

    def test_silent_worker_times_out_instead_of_deadlocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / "silent_worker.py"
            worker.write_text(
                textwrap.dedent(
                    """
                    import argparse
                    import json
                    import sys
                    import time

                    parser = argparse.ArgumentParser()
                    parser.add_argument("--input-buffer")
                    parser.add_argument("--output-buffer")
                    parser.parse_args()
                    print(json.dumps({"status": "ready"}), flush=True)
                    for command in sys.stdin:
                        if command.strip() == "run":
                            time.sleep(60)
                    """
                )
            )
            client = SharedMemoryTvmClient(
                worker_command=[sys.executable, str(worker)],
                input_shape=(1,),
                output_shape=(1,),
                input_dtype="float32",
                output_dtype="float32",
                workspace=root / "ipc",
                execution_timeout_seconds=0.05,
            )
            try:
                with self.assertRaisesRegex(TimeoutError, "TVM worker"):
                    client.execute(np.zeros((1,), dtype=np.float32))
            finally:
                client.close(force=True)


if __name__ == "__main__":
    unittest.main()
