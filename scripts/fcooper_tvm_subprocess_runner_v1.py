#!/usr/bin/env python3
"""Persistent shared-memory client for a TVM worker in a separate Python ABI."""

from __future__ import annotations

import json
import os
import select
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


class SharedMemoryTvmClient:
    def __init__(
        self,
        *,
        worker_command: Sequence[str],
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        input_dtype: str,
        output_dtype: str,
        workspace: Path,
        env: Mapping[str, str] | None = None,
        startup_timeout_seconds: float = 120.0,
        execution_timeout_seconds: float = 300.0,
    ) -> None:
        self.input_shape = tuple(int(value) for value in input_shape)
        self.output_shape = tuple(int(value) for value in output_shape)
        self.input_dtype = np.dtype(input_dtype)
        self.output_dtype = np.dtype(output_dtype)
        if execution_timeout_seconds <= 0:
            raise ValueError("execution_timeout_seconds must be positive")
        self.execution_timeout_seconds = float(execution_timeout_seconds)
        self.call_count = 0
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.input_path = self.workspace / "input.buffer"
        self.output_path = self.workspace / "output.buffer"
        self.stderr_path = self.workspace / "worker.stderr.log"
        self._input = np.memmap(
            self.input_path,
            mode="w+",
            dtype=self.input_dtype,
            shape=self.input_shape,
        )
        self._output = np.memmap(
            self.output_path,
            mode="w+",
            dtype=self.output_dtype,
            shape=self.output_shape,
        )
        self._stderr = self.stderr_path.open("w", encoding="utf-8")
        command = [
            *(str(value) for value in worker_command),
            "--input-buffer",
            str(self.input_path),
            "--output-buffer",
            str(self.output_path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
            env=dict(env) if env is not None else None,
        )
        try:
            ready = self._read_response(startup_timeout_seconds)
            if ready.get("status") != "ready":
                raise RuntimeError(f"TVM worker did not become ready: {ready}")
        except Exception:
            self.close(force=True)
            raise

    def _read_response(self, timeout_seconds: float | None = None) -> dict[str, object]:
        stdout = self._process.stdout
        if stdout is None:
            raise RuntimeError("TVM worker stdout is unavailable")
        if timeout_seconds is not None:
            readable, _, _ = select.select([stdout], [], [], timeout_seconds)
            if not readable:
                raise TimeoutError("timed out waiting for TVM worker")
        line = stdout.readline()
        if not line:
            returncode = self._process.poll()
            details = ""
            if self.stderr_path.exists():
                details = self.stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"TVM worker exited before responding (returncode={returncode}): {details}"
            )
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError("TVM worker response must be a JSON object")
        return payload

    def execute(self, value: np.ndarray) -> np.ndarray:
        source = np.asarray(value)
        if tuple(source.shape) != self.input_shape:
            raise ValueError(
                f"TVM worker input shape drift: {tuple(source.shape)} != {self.input_shape}"
            )
        if source.dtype != self.input_dtype:
            raise ValueError(
                f"TVM worker input dtype drift: {source.dtype} != {self.input_dtype}"
            )
        if self._process.poll() is not None:
            raise RuntimeError(f"TVM worker is not running (returncode={self._process.returncode})")
        self._input[:] = source
        stdin = self._process.stdin
        if stdin is None:
            raise RuntimeError("TVM worker stdin is unavailable")
        stdin.write("run\n")
        stdin.flush()
        response = self._read_response(self.execution_timeout_seconds)
        if response.get("status") != "success":
            raise RuntimeError(f"TVM worker execution failed: {response}")
        self.call_count += 1
        return np.array(self._output, copy=True)

    def close(self, *, force: bool = False) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            if not force and process.stdin is not None:
                try:
                    process.stdin.write("close\n")
                    process.stdin.flush()
                    self._read_response(5.0)
                except Exception:
                    force = True
            if force and process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream_name in ("stdin", "stdout"):
            stream = getattr(process, stream_name, None) if process is not None else None
            if stream is not None:
                stream.close()
        stderr = getattr(self, "_stderr", None)
        if stderr is not None and not stderr.closed:
            stderr.close()
        for array_name in ("_input", "_output"):
            array = getattr(self, array_name, None)
            if array is not None:
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()

    def __enter__(self) -> "SharedMemoryTvmClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close(force=True)
        except Exception:
            pass


def build_worker_env(
    *,
    tvm_site: Path,
    tvm_lib_dirs: Sequence[Path],
    gpu_id: int,
) -> dict[str, str]:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(tvm_site)),
        "LD_LIBRARY_PATH": ":".join(
            [*(str(Path(value)) for value in tvm_lib_dirs), existing]
        ).rstrip(":"),
        "PYTHONUNBUFFERED": "1",
    }
    env.setdefault("CUDA_VISIBLE_DEVICES", str(int(gpu_id)))
    return env
