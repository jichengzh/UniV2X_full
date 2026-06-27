#!/usr/bin/env python3
"""Prepare a Stage2 evidence registry fixture for a known model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.evidence_registry import write_default_registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=("pyramid_lidar", "codriving"),
        help="Model fixture to prepare.",
    )
    parser.add_argument(
        "--out-json",
        required=True,
        help="Output evidence_registry.json path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    registry = write_default_registry(args.model, args.out_json, root=ROOT)
    coverage = registry.coverage_summary()
    print(
        "stage2_prepare_evidence_ok "
        f"model={registry.model} "
        f"out={Path(args.out_json)} "
        f"expected_cells={coverage['total_expected_cells']} "
        f"measured_cells={coverage['total_measured_cells']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
