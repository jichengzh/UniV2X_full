#!/usr/bin/env python3
"""Build the transparent Stage7 scanner-admission union and A4 blocker."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SUCCESS_STATUSES = {
    "success",
    "measured_success",
    "measured_success_gold",
}
CANDIDATE_FAILURE_STATUSES = {
    "feasibility_failure",
    "numerical_feasibility_failure",
}
CANDIDATE_FAILURE_KINDS = {
    "backend_failure",
    "build_failure",
    "unsupported_precision",
    "quantization_failure",
    "numerical_failure",
    "candidate_runtime_capability_failure",
}
TASK_FILTER = {
    "model": "pyramid",
    "dispatch_key": "tvm_auto",
    "capability_profile_id": "h800-tvm-probe-conditioned-v3",
}
FORMAL_SCANNER_SHA256 = {
    "rule": "9d045f1c205d6f843fb148370b09227de7dfeeb38fbc5357f6b49af76421c20a",
    "decision": "0fcfd29d10ac52caf63a290aa4eb3610207528aafb654d2131231358f69cbfcd",
}
EXPECTED_FORMAL_COUNTS = {
    "source_count": 8,
    "raw_instance_count": 100,
    "unique_row_id_count": 82,
    "success_instance_count": 97,
    "candidate_failure_instance_count": 3,
    "conflicting_row_id_count": 1,
    "candidate_failure_unique_count": 3,
    "outside_frozen_pool_unique_count": 6,
}


@dataclass(frozen=True)
class EvidenceSource:
    label: str
    path: Path
    sha256: str
    selected_count: int


FORMAL_SOURCES = (
    EvidenceSource(
        "gold176",
        REPO_ROOT
        / (
            "results/stage35_gold144_targeted_supplement_v2_20260714/"
            "final_gold176_v1/gold176_final.json"
        ),
        "9880d625e1ac2c5e336a5de3bc1d861072d58e05d4b1bea6c79ef1cd0e93ca19",
        48,
    ),
    EvidenceSource(
        "stage5_v2_final_history",
        REPO_ROOT
        / (
            "results/stage5_single_target_search_v2_gold176_20260718/"
            "S5-PYR-TVM/feedback_history_through_round_03.json"
        ),
        "ed6e246625d821539f0fd2de359268a27d95240470e48d776bdbc2d167c9096b",
        16,
    ),
    EvidenceSource(
        "stage5_actual_v3_final_history",
        REPO_ROOT
        / (
            "results/stage5_pyramid_actual_v3_20260720/"
            "S5-PYR-TVM/feedback_history_through_round_03.json"
        ),
        "7539467edef9379976adb842de6ff8c8351123365db2f968ccdc0ec3be3bd8e2",
        16,
    ),
    EvidenceSource(
        "stage6_compress_then_tune_batch_00",
        REPO_ROOT
        / (
            "results/stage6_pyramid_formal_20260720/tvm/compress_then_tune/"
            "formal_batch_00/S5-PYR-TVM/round_00/final/"
            "stage5_feedback_v2_final.json"
        ),
        "4e1744f7a243b1544bca9e1b619538f947be521fc52d0ac00f2f45212b10acec",
        4,
    ),
    EvidenceSource(
        "stage6_compression_only_batch_00",
        REPO_ROOT
        / (
            "results/stage6_pyramid_formal_20260720/tvm/compression_only/"
            "formal_batch_00/S5-PYR-TVM/round_00/final/"
            "stage5_feedback_v2_final.json"
        ),
        "af11b265cd45b4de93d27a723345c71c04194910798e7fc273370cae96225847",
        4,
    ),
    EvidenceSource(
        "stage6_compression_only_batch_01",
        REPO_ROOT
        / (
            "results/stage6_pyramid_formal_20260720/tvm/compression_only/"
            "formal_batch_01/S5-PYR-TVM/round_01/final/"
            "stage5_feedback_v2_final.json"
        ),
        "ecbe89a58c4de486b3793f358e73f9792f302a922d935f654c360fe85cb29a8b",
        4,
    ),
    EvidenceSource(
        "stage6_compression_only_batch_02",
        REPO_ROOT
        / (
            "results/stage6_pyramid_formal_20260720/tvm/compression_only/"
            "formal_batch_02/S5-PYR-TVM/round_02/final/"
            "stage5_feedback_v2_final.json"
        ),
        "2bd8bfb5242097677162709cdd35f0692d79796ae5529972625af32603b96156",
        4,
    ),
    EvidenceSource(
        "stage6_compression_only_batch_03",
        REPO_ROOT
        / (
            "results/stage6_pyramid_formal_20260720/tvm/compression_only/"
            "formal_batch_03/S5-PYR-TVM/round_03/final/"
            "stage5_feedback_v2_final.json"
        ),
        "142c9f926ce6e2c21aa9aaf0e78fb0daecec63e70dd6bf3b7fb887a980b4eb30",
        4,
    ),
)


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(payload: Any) -> list[dict[str, Any]]:
    value = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(value, list) or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise ValueError("historical source must contain a row list")
    return [dict(row) for row in value]


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("row_id") or row.get("manifest_job_id") or "")


def _terminal_class(row: Mapping[str, Any]) -> str:
    status = str(row.get("terminal_status") or "")
    kind = str(row.get("failure_kind") or "")
    if status in SUCCESS_STATUSES:
        return "success"
    if status in CANDIDATE_FAILURE_STATUSES or kind in CANDIDATE_FAILURE_KINDS:
        return "candidate_failure"
    raise ValueError(
        f"unclassified formal terminal row: {_row_id(row) or '<empty>'}"
    )


def _load_decisions(
    output_root: Path, *, enforce_formal_sha: bool
) -> tuple[dict[str, str], dict[str, str]]:
    contracts = output_root / "contracts"
    paths = {
        "rule": contracts / "scanner_rule_manifest.json",
        "decision": contracts / "scanner_decision_by_candidate.csv",
    }
    for label, path in paths.items():
        sidecar = path.with_suffix(".sha256")
        if not path.is_file() or not sidecar.is_file():
            raise ValueError(f"missing frozen scanner {label} evidence")
        recorded = sidecar.read_text(encoding="ascii").strip()
        if recorded != _file_sha256(path):
            raise ValueError(f"frozen scanner {label} SHA drift")
    actual_sha = {
        "rule": _file_sha256(paths["rule"]),
        "decision": _file_sha256(paths["decision"]),
    }
    if enforce_formal_sha:
        for label, expected in FORMAL_SCANNER_SHA256.items():
            if actual_sha[label] != expected:
                raise ValueError(f"formal scanner {label} SHA drift")
    rows = list(
        csv.DictReader(
            io.StringIO(paths["decision"].read_text(encoding="utf-8"))
        )
    )
    decisions = {str(row["row_id"]): str(row["decision"]) for row in rows}
    if (
        len(rows) != 686
        or len(decisions) != 686
        or set(decisions.values()) - {"pass", "reject"}
    ):
        raise ValueError("frozen scanner decision table is invalid")
    return decisions, {
        "scanner_rule_file_sha256": actual_sha["rule"],
        "scanner_decision_file_sha256": actual_sha["decision"],
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if path.is_file():
        if path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite drifted blocker evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def build_scanner_blocker(
    output_root: Path,
    *,
    sources: Sequence[EvidenceSource] = FORMAL_SOURCES,
    enforce_formal_counts: bool = True,
) -> dict[str, Any]:
    if not output_root.is_absolute():
        raise ValueError("output root must be absolute")
    decisions, scanner_sha = _load_decisions(
        output_root, enforce_formal_sha=enforce_formal_counts
    )
    instances: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for source in sources:
        path = source.path.resolve()
        if not path.is_file() or _file_sha256(path) != source.sha256:
            raise ValueError(f"historical source SHA mismatch: {source.label}")
        selected = [
            row
            for row in _rows(json.loads(path.read_text(encoding="utf-8")))
            if all(row.get(key) == value for key, value in TASK_FILTER.items())
        ]
        if len(selected) != source.selected_count:
            raise ValueError(f"historical source count drift: {source.label}")
        source_records.append(
            {
                "label": source.label,
                "path": str(path),
                "sha256": source.sha256,
                "selected_count": len(selected),
            }
        )
        for source_index, row in enumerate(selected):
            row_id = _row_id(row)
            if not row_id:
                raise ValueError(f"historical source row lacks identity: {source.label}")
            terminal_class = _terminal_class(row)
            instances.append(
                {
                    "source_label": source.label,
                    "source_path": str(path),
                    "source_sha256": source.sha256,
                    "source_index": source_index,
                    "row_id": row_id,
                    "terminal_class": terminal_class,
                    "terminal_status": row.get("terminal_status"),
                    "failure_kind": row.get("failure_kind"),
                    "failure_reason": row.get("failure_reason"),
                    "row_sha256": _canonical_sha256(row),
                    "row": row,
                }
            )

    statuses_by_id: dict[str, set[str]] = {}
    for instance in instances:
        row_id = instance["row_id"]
        statuses_by_id[row_id] = {
            *statuses_by_id.get(row_id, set()),
            instance["terminal_class"],
        }
    conflicts = sorted(
        row_id for row_id, statuses in statuses_by_id.items() if len(statuses) > 1
    )
    success_ids = {
        instance["row_id"]
        for instance in instances
        if instance["terminal_class"] == "success"
    }
    failure_ids = {
        instance["row_id"]
        for instance in instances
        if instance["terminal_class"] == "candidate_failure"
    }
    known_success_rejected = sorted(
        row_id for row_id in success_ids if decisions.get(row_id) == "reject"
    )
    true_failure_passed = sorted(
        row_id for row_id in failure_ids if decisions.get(row_id) == "pass"
    )
    outside_pool = sorted(set(statuses_by_id) - set(decisions))
    counts = {
        "source_count": len(source_records),
        "raw_instance_count": len(instances),
        "unique_row_id_count": len(statuses_by_id),
        "success_instance_count": sum(
            item["terminal_class"] == "success" for item in instances
        ),
        "candidate_failure_instance_count": sum(
            item["terminal_class"] == "candidate_failure" for item in instances
        ),
        "conflicting_row_id_count": len(conflicts),
        "candidate_failure_unique_count": len(failure_ids),
        "outside_frozen_pool_unique_count": len(outside_pool),
    }
    if enforce_formal_counts and counts != EXPECTED_FORMAL_COUNTS:
        raise ValueError(f"formal historical union count drift: {counts}")

    union_unsigned = {
        "schema_version": "stage7_scanner_terminal_evidence_union_v1",
        "task_filter": TASK_FILTER,
        "admission_scope": "frozen_686_pool_intersection",
        "outside_frozen_pool_excluded_from_admission": True,
        "source_allowlist": source_records,
        "counts": counts,
        "conflicting_row_ids": conflicts,
        "outside_frozen_pool_row_ids": outside_pool,
        "instances": instances,
    }
    union = {
        **union_unsigned,
        "union_sha256": _canonical_sha256(union_unsigned),
    }
    union_path = output_root / "audits/scanner_terminal_evidence_union_v1.json"
    _atomic_write_json(union_path, union)

    admitted = not known_success_rejected and not true_failure_passed
    audit_unsigned = {
        "schema_version": "stage7_scanner_admission_expanded_v1",
        "status": (
            "scanner_ready_expanded"
            if admitted
            else "blocked_scanner_admission_failed"
        ),
        "admission_passed": admitted,
        "admission_scope": "frozen_686_pool_intersection",
        "outside_frozen_pool_excluded_from_admission": True,
        **scanner_sha,
        "union_path": str(union_path),
        "union_file_sha256": _file_sha256(union_path),
        "known_success_rejected": known_success_rejected,
        "true_candidate_capability_failure_passed": true_failure_passed,
        "false_negative_unique_count": len(known_success_rejected),
        "false_positive_unique_count": len(true_failure_passed),
        "outside_frozen_pool_row_ids": outside_pool,
        "conflicting_row_ids": conflicts,
    }
    audit = {
        **audit_unsigned,
        "expanded_admission_sha256": _canonical_sha256(audit_unsigned),
    }
    audit_path = output_root / "audits/scanner_admission_expanded_v1.json"
    _atomic_write_json(audit_path, audit)

    blocker_unsigned = {
        "schema_version": "stage7_variant_blocker_v1",
        "variant": "without_capability_scan",
        "status": (
            "not_blocked" if admitted else "blocked_missing_candidate_level_scanner"
        ),
        "reason": (
            None
            if admitted
            else "frozen scanner passed real candidate capability failures"
        ),
        "selected_events_consumed": 0,
        "gpu_jobs_launched": 0,
        "trajectory_initialized": False,
        "expanded_admission_path": str(audit_path),
        "expanded_admission_file_sha256": _file_sha256(audit_path),
        "frozen_scanner_modified_after_label_review": False,
        "admission_scope": "frozen_686_pool_intersection",
    }
    blocker = {
        **blocker_unsigned,
        "blocker_sha256": _canonical_sha256(blocker_unsigned),
    }
    blocker_path = (
        output_root / "status/without_capability_scan_blocked.json"
    )
    _atomic_write_json(blocker_path, blocker)
    return {
        "union": union,
        "audit": audit,
        "blocker": blocker,
        "paths": {
            "union": str(union_path),
            "audit": str(audit_path),
            "blocker": str(blocker_path),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_scanner_blocker(args.output_root.resolve())
    print(
        json.dumps(
            {
                "status": result["audit"]["status"],
                "admission_passed": result["audit"]["admission_passed"],
                "false_positive_unique_count": result["audit"][
                    "false_positive_unique_count"
                ],
                "a4_status": result["blocker"]["status"],
                "paths": result["paths"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["audit"]["admission_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
