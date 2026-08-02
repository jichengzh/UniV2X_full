#!/usr/bin/env python3
"""Prepare CoDriving arm-scoped Stage6 independent performance reruns."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.stage6_prepare_independent_validation_v1 as base


base.TASKS = {"tvm": "S5-COD-TVM", "trt": "S5-COD-TRT"}


if __name__ == "__main__":
    raise SystemExit(base.main())
