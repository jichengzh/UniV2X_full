#!/usr/bin/env python3
"""Stream F-Cooper prediction tensors into an atomic JSONL artifact."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PREDICTION_KEYS = ("pred_box_tensor", "pred_score", "gt_box_tensor")


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    current = value
    for method in ("detach", "cpu"):
        callback = getattr(current, method, None)
        if callback is not None:
            current = callback()
    callback = getattr(current, "numpy", None)
    if callback is not None:
        current = callback()
    array = np.asarray(current)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"prediction tensor has non-numeric dtype: {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError("prediction tensor contains a non-finite value")
    return array


class PredictionArtifactWriter:
    """Publish only a complete, fsynced prediction artifact."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = Path(output_path)
        self._temporary: Path | None = None
        self._handle: Any | None = None
        self.row_count = 0

    def __enter__(self) -> "PredictionArtifactWriter":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.output_path.name}.",
            dir=self.output_path.parent,
        )
        self._temporary = Path(temporary)
        self._handle = os.fdopen(descriptor, "w", encoding="utf-8")
        return self

    def append(self, *, sample_index: int, result: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("prediction writer is not open")
        row = {"sample_index": int(sample_index)}
        for key in PREDICTION_KEYS:
            array = _array(result.get(key))
            row[key] = None if array is None else array.tolist()
        self._handle.write(
            json.dumps(
                row,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        self.row_count += 1

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> None:
        if self._handle is not None:
            if error_type is None:
                self._handle.flush()
                os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None
        if self._temporary is None:
            return
        if error_type is None:
            os.replace(self._temporary, self.output_path)
        elif self._temporary.exists():
            self._temporary.unlink()
        self._temporary = None
