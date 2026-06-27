#!/usr/bin/env python3
"""Generate calibrated Stage1 coupling predictions v1 and S6 report."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage1.calibrated_predictor import write_reports


def main() -> int:
    write_reports()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
