from __future__ import annotations

import json
from pathlib import Path

from scripts.fcooper_replay_search_timing_v2 import (
    build_round_commands,
    verify_replayed_request,
)


def test_build_round_commands_uses_only_formal_feedback(tmp_path: Path) -> None:
    root = tmp_path / "formal_v2"
    commands = build_round_commands(
        root=root,
        code_root=Path("/repo"),
        python=Path("/env/python"),
        stage4_dir=Path("/stage4"),
        coldstart_root=Path("/gold176"),
        profiles_json=Path("/profiles.json"),
        replay_root=tmp_path / "replay",
    )
    assert [item["round_index"] for item in commands] == [0, 1, 2, 3]
    serialized = json.dumps(commands, default=str)
    assert "fcooper_workpackage_a_20260723" not in serialized
    assert "feedback_history_through_round_00.json" in serialized
    assert "feedback_history_through_round_02.json" in serialized


def test_verify_replayed_request_binds_selection_and_request_sha(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected.json"
    replayed = tmp_path / "replayed.json"
    payload = {
        "selected_row_ids": ["a", "b", "c", "d"],
        "measurement_request_sha256": "f" * 64,
    }
    expected.write_text(json.dumps(payload))
    replayed.write_text(json.dumps(payload))
    audit = verify_replayed_request(expected, replayed)
    assert audit["request_semantics_match"] is True
    assert audit["selected_row_ids"] == ["a", "b", "c", "d"]


def test_verify_replayed_request_derives_ids_from_request_rows(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected.json"
    replayed = tmp_path / "replayed.json"
    payload = {
        "rows": [{"row_id": value} for value in ("a", "b", "c", "d")],
        "measurement_request_sha256": "f" * 64,
    }
    expected.write_text(json.dumps(payload))
    replayed.write_text(json.dumps(payload))
    audit = verify_replayed_request(expected, replayed)
    assert audit["selected_row_ids"] == ["a", "b", "c", "d"]


def test_verify_replayed_request_rejects_semantic_drift(tmp_path: Path) -> None:
    expected = tmp_path / "expected.json"
    replayed = tmp_path / "replayed.json"
    expected.write_text(
        json.dumps(
            {
                "selected_row_ids": ["a", "b", "c", "d"],
                "measurement_request_sha256": "a" * 64,
            }
        )
    )
    replayed.write_text(
        json.dumps(
            {
                "selected_row_ids": ["a", "b", "c", "e"],
                "measurement_request_sha256": "b" * 64,
            }
        )
    )
    try:
        verify_replayed_request(expected, replayed)
    except ValueError as exc:
        assert "semantic drift" in str(exc)
    else:
        raise AssertionError("replayed request drift must be rejected")


def test_verify_replayed_request_rejects_equal_self_reported_sha_with_byte_drift(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected.json"
    replayed = tmp_path / "replayed.json"
    common = {
        "selected_row_ids": ["a", "b", "c", "d"],
        "measurement_request_sha256": "f" * 64,
    }
    expected.write_text(json.dumps({**common, "rows": [{"row_id": "a"}]}))
    replayed.write_text(json.dumps({**common, "rows": [{"row_id": "b"}]}))

    try:
        verify_replayed_request(expected, replayed)
    except ValueError as exc:
        assert "byte drift" in str(exc)
    else:
        raise AssertionError("equal self-reported SHA must not hide request byte drift")
