#!/usr/bin/env python3
"""Measure deterministic F-Cooper cost-model/refit replay time with SHA checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_ID = "S5-FCO-TRT-V2"
PILOT_FRAGMENT = "fcooper_workpackage_a_20260723"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_pilot(values: Sequence[object]) -> None:
    for value in values:
        if PILOT_FRAGMENT in str(value):
            raise ValueError(f"replay timing rejects pilot input: {value}")


def build_round_commands(
    *,
    root: Path,
    code_root: Path,
    python: Path,
    stage4_dir: Path,
    coldstart_root: Path,
    profiles_json: Path,
    replay_root: Path,
) -> list[dict[str, object]]:
    search = root / "search" / TASK_ID
    replay_search = replay_root / "search" / TASK_ID
    common = {
        "source_registry": root / "contracts/candidate_source_registry.json",
        "probe_isolation": root / "probes/probe_isolation_audit.json",
        "profiles": profiles_json,
    }
    initialization = [
        str(python),
        str(code_root / "scripts/stage5_initialize_fcooper_actual_v2.py"),
        "--stage4-dir",
        str(stage4_dir),
        "--coldstart-root",
        str(coldstart_root),
        "--profiles-json",
        str(profiles_json),
        "--source-registry-json",
        str(common["source_registry"]),
        "--formal-contract-json",
        str(root / "contracts/frozen_contract.json"),
        "--probe-audit-json",
        str(root / "probes/probe_audit.json"),
        "--probe-isolation-audit-json",
        str(common["probe_isolation"]),
        "--numeric-gate-summary-json",
        str(root / "gates/recovery_numeric_gate_summary.json"),
        "--scanner-execution-json",
        str(root / "scanner/scanner_execution.json"),
        "--output-dir",
        str(replay_search),
    ]
    commands: list[dict[str, object]] = [
        {
            "round_index": 0,
            "command": initialization,
            "expected_request": search / "round_00/measurement_request.json",
            "replayed_request": replay_search / "round_00/measurement_request.json",
        }
    ]
    for round_index in (1, 2, 3):
        previous = round_index - 1
        command = [
            str(python),
            str(code_root / "scripts/stage5_advance_fcooper_round_v2.py"),
            "--feedback-json",
            str(search / f"feedback_history_through_round_{previous:02d}.json"),
            "--atomic-audit-json",
            str(search / f"round_{previous:02d}/atomic_batch_audit.json"),
            "--output-dir",
            str(replay_search),
            "--round-index",
            str(round_index),
            "--source-registry-json",
            str(common["source_registry"]),
            "--probe-isolation-audit-json",
            str(common["probe_isolation"]),
            "--coldstart-rows-json",
            str(coldstart_root / "gold176_final.json"),
            "--coldstart-graph-features-json",
            str(coldstart_root / "graph_features.json"),
            "--profiles-json",
            str(common["profiles"]),
        ]
        commands.append(
            {
                "round_index": round_index,
                "command": command,
                "expected_request": search
                / f"round_{round_index:02d}/measurement_request.json",
                "replayed_request": replay_search
                / f"round_{round_index:02d}/measurement_request.json",
            }
        )
    _reject_pilot(
        [
            root,
            code_root,
            python,
            stage4_dir,
            coldstart_root,
            profiles_json,
            replay_root,
            *[argument for item in commands for argument in item["command"]],
        ]
    )
    return commands


def verify_replayed_request(
    expected_path: Path,
    replayed_path: Path,
) -> dict[str, object]:
    expected = _read(expected_path)
    replayed = _read(replayed_path)
    expected_ids = list(expected.get("selected_row_ids") or [])
    replayed_ids = list(replayed.get("selected_row_ids") or [])
    if not expected_ids:
        expected_ids = [
            str(row.get("row_id") or "")
            for row in list(expected.get("rows") or [])
        ]
    if not replayed_ids:
        replayed_ids = [
            str(row.get("row_id") or "")
            for row in list(replayed.get("rows") or [])
        ]
    expected_sha = expected.get("measurement_request_sha256")
    replayed_sha = replayed.get("measurement_request_sha256")
    if (
        len(expected_ids) != 4
        or expected_ids != replayed_ids
        or not isinstance(expected_sha, str)
        or expected_sha != replayed_sha
    ):
        raise ValueError("deterministic replay request semantic drift")
    if expected_path.read_bytes() != replayed_path.read_bytes():
        raise ValueError("deterministic replay request byte drift")
    return {
        "request_semantics_match": True,
        "selected_row_ids": expected_ids,
        "measurement_request_sha256": expected_sha,
        "expected_request_path": str(expected_path.resolve()),
        "expected_file_sha256": _sha256(expected_path),
        "replayed_request_path": str(replayed_path.resolve()),
        "replayed_file_sha256": _sha256(replayed_path),
    }


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--stage4-dir", type=Path, required=True)
    parser.add_argument("--coldstart-root", type=Path, required=True)
    parser.add_argument("--profiles-json", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_output = (
        args.root.resolve()
        / "controls/resource_audit/cost_model_replay_timing_v2.json"
    )
    if args.output_json.resolve() != expected_output:
        raise ValueError(f"--output-json must be {expected_output}")
    if args.replay_root.exists():
        raise FileExistsError(f"replay root already exists: {args.replay_root}")
    commands = build_round_commands(
        root=args.root.resolve(),
        code_root=args.code_root.resolve(),
        python=args.python.resolve(),
        stage4_dir=args.stage4_dir.resolve(),
        coldstart_root=args.coldstart_root.resolve(),
        profiles_json=args.profiles_json.resolve(),
        replay_root=args.replay_root.resolve(),
    )
    args.replay_root.mkdir(parents=True)
    records: list[dict[str, object]] = []
    total_started = time.monotonic()
    try:
        for item in commands:
            round_index = int(item["round_index"])
            log = args.replay_root / f"round_{round_index:02d}.log"
            started = time.monotonic()
            with log.open("wb") as handle:
                completed = subprocess.run(
                    list(item["command"]),
                    cwd=args.code_root,
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            elapsed = time.monotonic() - started
            if completed.returncode != 0:
                raise RuntimeError(
                    f"round {round_index} replay failed with "
                    f"return code {completed.returncode}: {log}"
                )
            request_audit = verify_replayed_request(
                Path(item["expected_request"]), Path(item["replayed_request"])
            )
            records.append(
                {
                    "round_index": round_index,
                    "elapsed_seconds": elapsed,
                    "command": item["command"],
                    "log_path": str(log),
                    "log_sha256": _sha256(log),
                    **request_audit,
                }
            )
        payload = {
            "schema_version": "fcooper_cost_model_replay_timing_v2",
            "task_id": TASK_ID,
            "status": "passed",
            "timing_kind": "deterministic_same_input_replay_monotonic",
            "original_online_refit_timing_available": False,
            "replay_matches_original_requests": True,
            "started_at_utc": _utc_now(),
            "total_elapsed_seconds": time.monotonic() - total_started,
            "rounds": records,
        }
        _atomic_write(args.output_json, payload)
    except Exception as error:
        failure_path = args.output_json.with_name(
            f"{args.output_json.stem}.failure.json"
        )
        failure_payload = {
            "schema_version": "fcooper_cost_model_replay_timing_failure_v2",
            "task_id": TASK_ID,
            "status": "failed",
            "timing_kind": "deterministic_same_input_replay_monotonic",
            "completed_rounds": records,
            "error": f"{type(error).__name__}: {error}",
            "replay_root": str(args.replay_root),
            "failed_at_utc": _utc_now(),
        }
        _atomic_write(failure_path, failure_payload)
        raise
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
