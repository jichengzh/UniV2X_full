from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import stage7_finalize_scanner_blocker_v1 as blocker


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _candidate(row_id: str, status: str, **extra: object) -> dict:
    return {
        "row_id": row_id,
        "manifest_job_id": row_id,
        "model": "pyramid",
        "dispatch_key": "tvm_auto",
        "capability_profile_id": "h800-tvm-probe-conditioned-v3",
        "terminal_status": status,
        **extra,
    }


def _scanner_root(tmp_path: Path) -> Path:
    root = (tmp_path / "formal").resolve()
    contracts = root / "contracts"
    rule = _write_json(contracts / "scanner_rule_manifest.json", {"rule": "frozen"})
    decision = contracts / "scanner_decision_by_candidate.csv"
    decision.write_text(
        "row_id,decision,reasons\n"
        + "".join(
            f"candidate-{index},pass,[]\n" for index in range(686)
        ),
        encoding="utf-8",
    )
    rule.with_suffix(".sha256").write_text(_sha(rule) + "\n", encoding="ascii")
    decision.with_suffix(".sha256").write_text(
        _sha(decision) + "\n", encoding="ascii"
    )
    return root


def _source(path: Path, label: str, rows: list[dict]) -> blocker.EvidenceSource:
    _write_json(path, rows)
    return blocker.EvidenceSource(label, path, _sha(path), len(rows))


def test_union_preserves_conflicts_and_emits_explicit_a4_blocker(
    tmp_path: Path,
) -> None:
    root = _scanner_root(tmp_path)
    sources = (
        _source(
            tmp_path / "gold.json",
            "gold",
            [
                _candidate("candidate-1", "measured_success_gold"),
                _candidate(
                    "candidate-2",
                    "feasibility_failure",
                    failure_reason="illegal memory access",
                ),
            ],
        ),
        _source(
            tmp_path / "actual.json",
            "actual",
            [_candidate("candidate-2", "measured_success_gold")],
        ),
    )

    result = blocker.build_scanner_blocker(
        root, sources=sources, enforce_formal_counts=False
    )

    assert result["union"]["counts"] == {
        "source_count": 2,
        "raw_instance_count": 3,
        "unique_row_id_count": 2,
        "success_instance_count": 2,
        "candidate_failure_instance_count": 1,
        "conflicting_row_id_count": 1,
        "candidate_failure_unique_count": 1,
        "outside_frozen_pool_unique_count": 0,
    }
    assert result["union"]["conflicting_row_ids"] == ["candidate-2"]
    assert result["audit"]["admission_passed"] is False
    assert result["audit"]["true_candidate_capability_failure_passed"] == [
        "candidate-2"
    ]
    assert result["audit"]["admission_scope"] == "frozen_686_pool_intersection"
    assert result["audit"]["outside_frozen_pool_excluded_from_admission"] is True
    assert result["blocker"]["status"] == "blocked_missing_candidate_level_scanner"
    assert result["blocker"]["selected_events_consumed"] == 0
    assert result["blocker"]["gpu_jobs_launched"] == 0
    for path in result["paths"].values():
        assert Path(path).is_file()


def test_source_sha_drift_is_fail_closed(tmp_path: Path) -> None:
    root = _scanner_root(tmp_path)
    path = _write_json(
        tmp_path / "source.json",
        [_candidate("candidate-1", "measured_success_gold")],
    )
    source = blocker.EvidenceSource("source", path, "0" * 64, 1)

    with pytest.raises(ValueError, match="historical source SHA mismatch"):
        blocker.build_scanner_blocker(
            root, sources=(source,), enforce_formal_counts=False
        )


def test_unclassified_terminal_is_not_silently_dropped(tmp_path: Path) -> None:
    root = _scanner_root(tmp_path)
    source = _source(
        tmp_path / "source.json",
        "source",
        [_candidate("candidate-1", "pending")],
    )

    with pytest.raises(ValueError, match="unclassified formal terminal row"):
        blocker.build_scanner_blocker(
            root, sources=(source,), enforce_formal_counts=False
        )


def test_existing_drifted_blocker_artifact_is_not_overwritten(
    tmp_path: Path,
) -> None:
    root = _scanner_root(tmp_path)
    source = _source(
        tmp_path / "source.json",
        "source",
        [_candidate("candidate-1", "measured_success_gold")],
    )
    blocker.build_scanner_blocker(
        root, sources=(source,), enforce_formal_counts=False
    )
    path = root / "status/without_capability_scan_blocked.json"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite drifted"):
        blocker.build_scanner_blocker(
            root, sources=(source,), enforce_formal_counts=False
        )


def test_formal_scanner_sha_is_pinned_beyond_mutable_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _scanner_root(tmp_path)
    rule = root / "contracts/scanner_rule_manifest.json"
    decision = root / "contracts/scanner_decision_by_candidate.csv"
    monkeypatch.setattr(
        blocker,
        "FORMAL_SCANNER_SHA256",
        {"rule": _sha(rule), "decision": _sha(decision)},
    )
    decision.write_text(
        decision.read_text(encoding="utf-8").replace(
            "candidate-1,pass", "candidate-1,reject"
        ),
        encoding="utf-8",
    )
    decision.with_suffix(".sha256").write_text(
        _sha(decision) + "\n", encoding="ascii"
    )

    with pytest.raises(ValueError, match="formal scanner decision SHA drift"):
        blocker.build_scanner_blocker(root, sources=(), enforce_formal_counts=True)


def test_outside_pool_rows_are_preserved_with_explicit_scope_exclusion(
    tmp_path: Path,
) -> None:
    root = _scanner_root(tmp_path)
    source = _source(
        tmp_path / "source.json",
        "source",
        [_candidate("candidate-outside", "feasibility_failure")],
    )

    result = blocker.build_scanner_blocker(
        root, sources=(source,), enforce_formal_counts=False
    )

    assert result["union"]["outside_frozen_pool_row_ids"] == [
        "candidate-outside"
    ]
    assert result["audit"]["outside_frozen_pool_row_ids"] == [
        "candidate-outside"
    ]
    assert result["audit"]["admission_scope"] == "frozen_686_pool_intersection"
    assert result["audit"]["outside_frozen_pool_excluded_from_admission"] is True


def test_direct_cli_bootstraps_repository_imports(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(blocker.__file__).resolve()), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--output-root" in completed.stdout
