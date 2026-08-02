#!/usr/bin/env python3
"""Verify real Stage5 feedback and advance the two-model search by one round."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.production_search_v1 import advance_search_round


DEFAULT_STAGE4 = REPO_ROOT / "results/stage4_p1_p3_closure_v1_20260716"
DEFAULT_OUTPUT = REPO_ROOT / "results/stage5_two_model_search_v1_20260717"


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_previous_request(previous: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    expected = str(previous.get("measurement_request_sha256") or "")
    if len(expected) != 64:
        raise ValueError("previous measurement request SHA missing from round state")
    if canonical_sha256(request) != expected:
        raise ValueError("previous measurement request SHA mismatch")


def _rows(payload: Any, field: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get(field), list):
        return [dict(item) for item in payload[field]]
    raise ValueError(f"expected list or object containing {field}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-json", type=Path, default=DEFAULT_STAGE4 / "stage4_p1_p3_closure_audit.json")
    parser.add_argument("--training-rows-json", type=Path, default=DEFAULT_STAGE4 / "data/gold176_feedback16_rows.json")
    parser.add_argument("--graph-features-json", type=Path, default=DEFAULT_STAGE4 / "data/gold176_feedback16_graph_features.json")
    parser.add_argument("--capability-profiles-json", type=Path, default=REPO_ROOT / "results/s1_profile_final_v3_20260711/capability_profiles_v3.json")
    parser.add_argument("--source-registry-json", type=Path, default=DEFAULT_OUTPUT / "candidate_source_registry.json")
    parser.add_argument("--frozen-holdout-json", type=Path, default=DEFAULT_STAGE4 / "stage5_independent_holdout_manifest.json")
    parser.add_argument("--feedback-rows-json", type=Path, required=True)
    parser.add_argument("--previous-round-state-json", type=Path, required=True)
    parser.add_argument("--previous-measurement-request-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--round-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    previous = json.loads(args.previous_round_state_json.read_text(encoding="utf-8"))
    if previous.get("status") != "awaiting_real_measurement":
        raise ValueError("previous round is not awaiting real measurement")
    previous_request = json.loads(
        args.previous_measurement_request_json.read_text(encoding="utf-8")
    )
    validate_previous_request(previous, previous_request)
    result = advance_search_round(
        closure=json.loads(args.closure_json.read_text(encoding="utf-8")),
        initial_training_rows=_rows(json.loads(args.training_rows_json.read_text(encoding="utf-8")), "rows"),
        initial_graph_features=_rows(json.loads(args.graph_features_json.read_text(encoding="utf-8")), "graph_features"),
        capability_profiles=_rows(json.loads(args.capability_profiles_json.read_text(encoding="utf-8")), "capability_profiles"),
        source_registry=json.loads(args.source_registry_json.read_text(encoding="utf-8")),
        frozen_holdout=json.loads(args.frozen_holdout_json.read_text(encoding="utf-8")),
        feedback_rows=_rows(json.loads(args.feedback_rows_json.read_text(encoding="utf-8")), "rows"),
        measurement_request=previous_request,
        selected_group_ids={str(group_id) for group_id in previous.get("selected_group_ids") or []},
        output_dir=args.output_dir,
        round_index=args.round_index,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
