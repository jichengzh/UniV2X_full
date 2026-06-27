#!/usr/bin/env python3
"""Generate end-to-end Stage1 model classification reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage1.model_classifier import (
    DEFAULT_MANIFESTS,
    DEFAULT_OUT_JSON,
    DEFAULT_OUT_MD,
    STAGE1_DIR,
    write_classification_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        help=(
            "Stage1 partition YAML path. Repeatable. Defaults to the nine-model "
            "classifier manifest set."
        ),
    )
    parser.add_argument(
        "--evidence-dir",
        default=str(STAGE1_DIR),
        help="Directory containing stage1_model_predict evidence JSON files.",
    )
    parser.add_argument(
        "--out-json",
        default=str(DEFAULT_OUT_JSON),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--out-md",
        default=str(DEFAULT_OUT_MD),
        help="Output Markdown report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifests = [Path(path) for path in args.manifest] if args.manifest else DEFAULT_MANIFESTS
    report = write_classification_report(
        manifests=manifests,
        evidence_dir=Path(args.evidence_dir),
        out_json=Path(args.out_json),
        out_md=Path(args.out_md),
    )
    print(
        "stage1_model_classifier_ok "
        f"models={len(report['models'])} "
        f"json={Path(args.out_json)} "
        f"md={Path(args.out_md)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
