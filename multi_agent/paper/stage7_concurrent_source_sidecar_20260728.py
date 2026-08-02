from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"non-object JSON: {path}")
    return payload


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def process_command(pid: int) -> list[str] | None:
    try:
        return [
            value.decode(errors="replace")
            for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if value
        ]
    except OSError:
        return None


def _iter_numeric_proc_pids() -> tuple[int, ...]:
    return tuple(
        sorted(
            int(entry.name)
            for entry in Path("/proc").iterdir()
            if entry.name.isdigit()
        )
    )


def _is_canonical_orchestrator_command(
    command: list[str] | None, root: Path
) -> bool:
    return bool(
        command
        and "stage7_speed_priority_takeover_v1.py" in " ".join(command)
        and str(root) in command
    )


def find_canonical_orchestrator_pid(root: Path, recorded_pid: int) -> int:
    if _is_canonical_orchestrator_command(process_command(recorded_pid), root):
        return recorded_pid
    matches = tuple(
        pid
        for pid in _iter_numeric_proc_pids()
        if _is_canonical_orchestrator_command(process_command(pid), root)
    )
    if len(matches) != 1:
        raise RuntimeError(
            "canonical orchestrator identity is not unique: "
            f"recorded_pid={recorded_pid}, live_matches={list(matches)}"
        )
    return matches[0]


def limit_ready_uuids(
    uuids: tuple[str, ...], *, max_gpus: int
) -> tuple[str, ...]:
    if isinstance(max_gpus, bool) or max_gpus <= 0:
        raise ValueError("max_gpus must be a positive integer")
    return tuple(uuids[:max_gpus])


def install_sidecar_gpu_cap(scheduler_class, *, max_gpus: int) -> None:
    original_ready_uuids = scheduler_class._ready_uuids
    admitted_once = False

    def lock_available_ready_uuids(self):
        nonlocal admitted_once
        if admitted_once:
            return []
        available = []
        for uuid in original_ready_uuids(self):
            lock = self.lock_factory(uuid)
            if lock is None:
                continue
            lock.release()
            available.append(uuid)
        limited = limit_ready_uuids(tuple(available), max_gpus=max_gpus)
        if limited:
            admitted_once = True
        return list(limited)

    scheduler_class._ready_uuids = lock_available_ready_uuids


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--variant", required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--round-index", type=int, required=True)
    result.add_argument("--audit-tag", required=True)
    result.add_argument("--max-gpus", type=int, default=1)
    return result


def main() -> int:
    args = parser().parse_args()
    root = args.root.resolve(strict=True)
    prepare = read_json(root / "prepare_state.json")
    frozen = Path(str(prepare["executor_repo_root"])).resolve(strict=True)
    code = (root / "deployment/code").resolve(strict=True)
    recorded_orchestrator_pid = int(prepare["orchestrator_pid"])
    orchestrator_pid = find_canonical_orchestrator_pid(
        root, recorded_orchestrator_pid
    )
    orchestrator_command = process_command(orchestrator_pid)
    if not _is_canonical_orchestrator_command(orchestrator_command, root):
        raise RuntimeError("canonical orchestrator identity changed during admission")
    round_dir = (
        root
        / "variants"
        / args.variant
        / f"seed_{args.seed}"
        / f"round_{args.round_index:02d}"
    )
    if not (round_dir / "source_resolution_plan.json").is_file():
        raise RuntimeError("target source plan is not frozen")
    if (round_dir / "source_resolution_result.json").exists():
        raise RuntimeError("target source result is already canonical")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["STAGE7_FROZEN_REPO_ROOT"] = str(frozen)
    os.environ["PYTHONPATH"] = os.pathsep.join((str(code), str(frozen)))
    for runtime_root in (str(frozen), str(code)):
        if runtime_root in sys.path:
            sys.path.remove(runtime_root)
        sys.path.insert(0, runtime_root)
    takeover_path = root / "tools/stage7_speed_priority_takeover_v1.py"
    specification = importlib.util.spec_from_file_location(
        "stage7_source_sidecar_takeover_policy", takeover_path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("speed-priority takeover policy cannot be loaded")
    takeover = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(takeover)
    from scripts import stage7_core_ablation_scheduler_v2 as scheduler_module
    from scripts import stage7_source_lease_controller_v2 as controller

    speed_policy = takeover.install_speed_priority_policy(scheduler_module)
    scheduler_class = scheduler_module.V2Stage7Scheduler
    install_sidecar_gpu_cap(scheduler_class, max_gpus=args.max_gpus)
    speed_policy = {
        **speed_policy,
        "active_uuid_lock_filter": True,
        "max_gpus_per_sidecar": args.max_gpus,
    }
    audit = (
        root
        / "audits/speed_priority_recovery/concurrent_source_sidecars"
        / args.audit_tag
    )
    if audit.exists():
        raise RuntimeError("sidecar audit tag already exists")
    start = {
        "schema_version": "stage7_concurrent_source_sidecar_v1",
        "status": "running",
        "wall_time": time.time(),
        "pid": os.getpid(),
        "sidecar_file_sha256": sha256(Path(__file__).resolve()),
        "orchestrator_pid": orchestrator_pid,
        "recorded_orchestrator_pid": recorded_orchestrator_pid,
        "orchestrator_pid_rebound": orchestrator_pid != recorded_orchestrator_pid,
        "variant": args.variant,
        "seed": args.seed,
        "round_index": args.round_index,
        "round_dir": str(round_dir),
        "logical_request_sha256": read_json(
            round_dir / "logical_request.json"
        ).get("measurement_request_sha256"),
        "source_resolution_plan_file_sha256": sha256(
            round_dir / "source_resolution_plan.json"
        ),
        "deployment_release_sha256": prepare["deployment_release_sha256"],
        "deployment_manifest_file_sha256": prepare[
            "deployment_manifest_file_sha256"
        ],
        "frozen_repo_root": str(frozen),
        "speed_policy": speed_policy,
        "selected_event_budget_delta": 0,
        "scientific_contract_changed": False,
        "selected_ids_changed": False,
        "logical_request_changed": False,
    }
    write_json(audit / "start.json", start)
    capacity_probes = []
    try:
        for probe_index in range(4):
            result = controller.run_source_lease_controller(
                v2_root=root,
                repo_root=frozen,
                variant=args.variant,
                seed=args.seed,
                round_index=args.round_index,
                orchestrator_pid=orchestrator_pid,
                expected_release_sha256=str(prepare["deployment_release_sha256"]),
                expected_manifest_file_sha256=str(
                    prepare["deployment_manifest_file_sha256"]
                ),
            )
            capacity_probes.append(
                {
                    "probe_index": probe_index,
                    "wall_time": time.time(),
                    "status": result.get("status"),
                    "reason": result.get("reason"),
                }
            )
            if result.get("status") != "waiting_for_source_gpu_capacity":
                break
            time.sleep(5)
    except BaseException as error:
        write_json(
            audit / "failure.json",
            {
                **start,
                "status": "failed",
                "completed_wall_time": time.time(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    write_json(
        audit / "completion.json",
        {
            **start,
            "status": "completed",
            "completed_wall_time": time.time(),
            "capacity_probes": capacity_probes,
            "result": result,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
