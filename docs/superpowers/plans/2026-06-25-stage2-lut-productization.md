# Stage2 LUT Productization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build productized latency, AP, and energy LUT generation/import routes with stable row schemas, queue-based long-running supplement jobs, and registry coverage updates.

**Architecture:** Keep `Stage2Input` thin and do not expose `evidence_registry_path` as a public Stage2 optimization input. Add a focused LUT production layer under `framework/stage2/lut_productization.py` plus small CLI wrappers under `scripts/`; canonical data is JSONL row tables, summaries are generated from rows, and `Stage2EvidenceRegistry` consumes the resulting registry.

**Tech Stack:** Python stdlib (`dataclasses`, `json`, `csv`, `argparse`, `hashlib`, `pathlib`, `subprocess`, `time`), existing `framework/stage2/evidence_registry.py`, `framework/stage1_bridge.py`, `unittest`.

---

## File Structure

- Create: `framework/stage2/lut_productization.py`
  - Row dataclasses, validators, JSONL helpers, coverage aggregation, job planning, state-log parsing.
- Create: `framework/tests/test_stage2_lut_productization.py`
  - Unit tests for row schema, config_id stability, coverage, job queue resume, registry update safety.
- Create: `scripts/stage2_plan_lut_jobs.py`
  - Generates `lut_job_plan_v1.jsonl` from manifest/search space/registry; primary jobs are `generate_latency_lut`, `generate_ap_lut`, and `generate_energy_lut`.
- Create: `scripts/stage2_generate_latency_lut.py`
  - Executes one latency measurement job and appends a canonical `latency_lut_row_v1`.
- Create: `scripts/stage2_generate_ap_lut.py`
  - Executes one AP eval job and appends a canonical `ap_anchor_row_v1`.
- Create: `scripts/stage2_generate_energy_lut.py`
  - Executes one energy telemetry job and appends a canonical `energy_lut_row_v1`.
- Create: `scripts/stage2_import_latency_lut.py`
  - Converts existing latency JSON into canonical `latency_lut_rows_v1.jsonl`.
- Create: `scripts/stage2_import_ap_anchors.py`
  - Converts existing AP model/anchor JSON into canonical `ap_anchor_rows_v1.jsonl`.
- Create: `scripts/stage2_import_energy_lut.py`
  - Imports measured energy CSV/JSONL into canonical `energy_lut_rows_v1.jsonl`.
- Create: `scripts/stage2_lut_worker.py`
  - Queue worker for long-running jobs; first version executes `generate_*_lut` commands and records state append-only.
- Create: `scripts/stage2_update_registry_from_luts.py`
  - Updates registry evidence source paths and coverage from canonical row tables.
- Modify: `framework/stage2/__init__.py`
  - Export public helper classes/functions from `lut_productization.py`.
- Modify: `framework/stage2/evidence_registry.py`
  - Add a small `update_source_coverage()` helper only if needed by registry updater; keep public Stage2 input unchanged.
- Create: `multi_agent/methods/progress/HANDOFF_stage2_lut_productization_plan_v1_zh.md`
  - Already created as the human-facing handoff plan; keep it in sync when implementation changes schema fields.

---

### Task 1: Canonical LUT Row Dataclasses And Validators

**Files:**
- Create: `framework/stage2/lut_productization.py`
- Test: `framework/tests/test_stage2_lut_productization.py`

- [ ] **Step 1: Write the failing tests**

Add these tests to `framework/tests/test_stage2_lut_productization.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from framework.stage2.lut_productization import (
    LutProductizationError,
    ap_anchor_row,
    energy_lut_row,
    latency_lut_row,
    stable_config_id,
    validate_lut_row,
    write_jsonl,
    read_jsonl,
)


class Stage2LutProductizationTest(unittest.TestCase):
    def test_stable_config_id_is_shared_across_latency_ap_and_energy(self):
        config_id = stable_config_id(
            model="pyramid_lidar",
            candidate_id="bev_encoder.s2",
            software_point_id="bev_encoder.s2:w128:fp16",
            quant_policy="fp16",
            schedule_policy="metaschedule_tuned",
        )

        self.assertEqual(
            config_id,
            "pyramid_lidar__bev_encoder.s2__bev_encoder.s2-w128-fp16__q_fp16__s_metaschedule_tuned",
        )

    def test_latency_ap_energy_rows_validate_required_fields(self):
        common = {
            "config_id": "pyramid_lidar__neck__neck-w64-fp16__q_fp16__s_default",
            "model": "pyramid_lidar",
            "manifest_digest": "a" * 64,
            "candidate_id": "neck",
            "software_point_id": "neck:w64:fp16",
            "dense_stage": "neck",
            "width": [64, 128, 256],
            "quant_policy": "fp16",
            "run_id": "run_001",
            "created_at": "2026-06-25T00:00:00+08:00",
        }
        latency = latency_lut_row(
            **common,
            schedule_policy="default",
            backend="h800_tvm",
            measurement_status="measured",
            latency_p50_us=123.4,
            warmup_iters=50,
            measure_iters=200,
            repeat=5,
        )
        ap = ap_anchor_row(
            **common,
            schedule_policy="not_applicable",
            backend="model_eval",
            measurement_status="measured",
            metric="AP70",
            metric_value=0.6369,
            dataset="DAIR-V2X",
            eval_split="val",
            ckpt_path="checkpoints/pyramid.ckpt",
            finetune_protocol="none",
        )
        energy = energy_lut_row(
            **common,
            schedule_policy="default",
            backend="h800_tvm_power_telemetry",
            measurement_status="measured",
            joule_per_inference=1.2,
            watt_avg=240.0,
            telemetry_source="nvidia_smi",
            idle_baseline_policy="subtract_idle_avg",
            sample_window_ms=5000,
            latency_run_id="run_001",
        )

        for row in (latency, ap, energy):
            validate_lut_row(row)

    def test_measured_latency_requires_h800_backend(self):
        row = latency_lut_row(
            config_id="bad",
            model="pyramid_lidar",
            manifest_digest="a" * 64,
            candidate_id="neck",
            software_point_id="neck:w64:fp16",
            dense_stage="neck",
            width=[64],
            quant_policy="fp16",
            schedule_policy="default",
            backend="historical_trt",
            measurement_status="measured",
            latency_p50_us=1.0,
            warmup_iters=1,
            measure_iters=1,
            repeat=1,
            run_id="run_bad",
            created_at="2026-06-25T00:00:00+08:00",
        )

        with self.assertRaisesRegex(LutProductizationError, "measured latency"):
            validate_lut_row(row)

    def test_jsonl_round_trip_preserves_rows(self):
        row = latency_lut_row(
            config_id="cfg",
            model="codriving",
            manifest_digest="b" * 64,
            candidate_id="backbone.s2",
            software_point_id="backbone.s2:w128:fp16",
            dense_stage="stage3",
            width=[32, 64, 128],
            quant_policy="fp16",
            schedule_policy="default",
            backend="h800_tvm",
            measurement_status="measured",
            latency_p50_us=1609.1,
            warmup_iters=50,
            measure_iters=200,
            repeat=5,
            run_id="run_jsonl",
            created_at="2026-06-25T00:00:00+08:00",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latency_lut_rows_v1.jsonl"
            write_jsonl(path, [row])
            self.assertEqual(read_jsonl(path), [row])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m unittest framework.tests.test_stage2_lut_productization
```

Expected: FAIL with `ModuleNotFoundError: No module named 'framework.stage2.lut_productization'`.

- [ ] **Step 3: Implement minimal row helpers**

Create `framework/stage2/lut_productization.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LutProductizationError(ValueError):
    pass


def _safe(value: str) -> str:
    return str(value).replace(":", "-").replace("/", "-").replace(" ", "_")


def stable_config_id(
    *,
    model: str,
    candidate_id: str,
    software_point_id: str,
    quant_policy: str,
    schedule_policy: str,
) -> str:
    return (
        f"{_safe(model)}__{_safe(candidate_id)}__{_safe(software_point_id)}"
        f"__q_{_safe(quant_policy)}__s_{_safe(schedule_policy)}"
    )


def _base_row(
    *,
    schema: str,
    config_id: str,
    model: str,
    manifest_digest: str,
    candidate_id: str,
    software_point_id: str,
    dense_stage: str,
    width: list[int],
    quant_policy: str,
    schedule_policy: str,
    backend: str,
    measurement_status: str,
    run_id: str,
    created_at: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "schema": schema,
        "row_id": f"{schema.removesuffix('_row_v1')}:{model}:{config_id}:{backend}:{run_id}",
        "config_id": config_id,
        "model": model,
        "manifest_digest": manifest_digest,
        "search_space_schema": "stage2_search_space_v1",
        "candidate_id": candidate_id,
        "software_point_id": software_point_id,
        "dense_stage": dense_stage,
        "optimized_scope": "rsu_dense_core",
        "width": [int(item) for item in width],
        "quant_policy": quant_policy,
        "schedule_policy": schedule_policy,
        "hardware_target": "H800 Hopper" if backend.startswith("h800") else "not_hardware_specific",
        "backend": backend,
        "measurement_status": measurement_status,
        "provenance": extra.pop("provenance", "stage2_lut_productization"),
        "run_id": run_id,
        "created_at": created_at,
        "source_files": extra.pop("source_files", []),
        "raw_artifact": extra.pop("raw_artifact", None),
        "failure_reason": extra.pop("failure_reason", None),
        "notes": extra.pop("notes", ""),
        **extra,
    }


def latency_lut_row(**kwargs: Any) -> dict[str, Any]:
    return _base_row(schema="latency_lut_row_v1", latency_unit="us", **kwargs)


def ap_anchor_row(**kwargs: Any) -> dict[str, Any]:
    return _base_row(schema="ap_anchor_row_v1", **kwargs)


def energy_lut_row(**kwargs: Any) -> dict[str, Any]:
    return _base_row(schema="energy_lut_row_v1", energy_unit="joule_per_inference", **kwargs)


def validate_lut_row(row: dict[str, Any]) -> None:
    required = [
        "schema", "row_id", "config_id", "model", "manifest_digest",
        "candidate_id", "software_point_id", "dense_stage", "width",
        "quant_policy", "schedule_policy", "backend", "measurement_status",
        "run_id", "created_at",
    ]
    missing = [key for key in required if key not in row]
    if missing:
        raise LutProductizationError(f"missing required fields: {missing}")
    if row["measurement_status"] == "measured":
        if row["schema"] == "latency_lut_row_v1" and row["backend"] != "h800_tvm":
            raise LutProductizationError("measured latency rows must use backend=h800_tvm")
        if row["schema"] == "energy_lut_row_v1" and not str(row["backend"]).startswith("h800"):
            raise LutProductizationError("measured energy rows must use an h800 telemetry backend")
    if row["measurement_status"] == "failed" and not row.get("failure_reason"):
        raise LutProductizationError("failed rows must include failure_reason")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            validate_lut_row(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m unittest framework.tests.test_stage2_lut_productization
```

Expected: `Ran 4 tests` and `OK`.

- [ ] **Step 5: Export helpers**

Modify `framework/stage2/__init__.py` to import and expose:

```python
from .lut_productization import (
    LutProductizationError,
    ap_anchor_row,
    energy_lut_row,
    latency_lut_row,
    stable_config_id,
    validate_lut_row,
)
```

Add the same names to `__all__`.

- [ ] **Step 6: Commit**

```bash
git add framework/stage2/lut_productization.py framework/stage2/__init__.py framework/tests/test_stage2_lut_productization.py
git commit -m "feat: define stage2 lut row schemas"
```

---

### Task 2: Coverage Aggregation And Registry-Safe Summaries

**Files:**
- Modify: `framework/stage2/lut_productization.py`
- Modify: `framework/tests/test_stage2_lut_productization.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
from framework.stage2.lut_productization import coverage_from_rows


class Stage2LutCoverageTest(unittest.TestCase):
    def test_coverage_counts_measured_proxy_failed_and_expected_rows(self):
        rows = [
            {"schema": "latency_lut_row_v1", "measurement_status": "measured"},
            {"schema": "latency_lut_row_v1", "measurement_status": "proxy"},
            {"schema": "latency_lut_row_v1", "measurement_status": "estimated"},
            {"schema": "latency_lut_row_v1", "measurement_status": "failed", "failure_reason": "build_failed"},
        ]

        self.assertEqual(
            coverage_from_rows(rows, expected_cells=6),
            {
                "expected_cells": 6,
                "measured_cells": 1,
                "failed_cells": 1,
                "proxy_cells": 2,
            },
        )
```

- [ ] **Step 2: Run tests to verify failure**

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m unittest framework.tests.test_stage2_lut_productization
```

Expected: FAIL with `ImportError: cannot import name 'coverage_from_rows'`.

- [ ] **Step 3: Implement coverage aggregation**

Add:

```python
def coverage_from_rows(rows: list[dict[str, Any]], *, expected_cells: int | None = None) -> dict[str, int]:
    measured = sum(1 for row in rows if row.get("measurement_status") == "measured")
    failed = sum(1 for row in rows if row.get("measurement_status") == "failed")
    proxy = sum(1 for row in rows if row.get("measurement_status") in {"proxy", "estimated", "historical"})
    expected = len(rows) if expected_cells is None else int(expected_cells)
    return {
        "expected_cells": expected,
        "measured_cells": measured,
        "failed_cells": failed,
        "proxy_cells": proxy,
    }
```

Export it from `__all__`.

- [ ] **Step 4: Run tests**

Expected: all `test_stage2_lut_productization` tests pass.

- [ ] **Step 5: Commit**

```bash
git add framework/stage2/lut_productization.py framework/tests/test_stage2_lut_productization.py
git commit -m "feat: aggregate stage2 lut coverage"
```

---

### Task 3: Job Plan And Resume State Format

**Files:**
- Modify: `framework/stage2/lut_productization.py`
- Modify: `framework/tests/test_stage2_lut_productization.py`
- Create: `scripts/stage2_plan_lut_jobs.py`

- [ ] **Step 1: Write failing tests for job plan rows**

Append:

```python
from framework.stage2.lut_productization import (
    job_plan_row,
    latest_job_status,
    next_queued_jobs,
)


class Stage2LutJobPlanTest(unittest.TestCase):
    def test_job_plan_and_state_support_resume(self):
        plan = [
            job_plan_row(
                job_id="latency:cfg_a",
                model="pyramid_lidar",
                lut_kind="latency",
                job_type="generate_latency_lut",
                priority=10,
                config_id="cfg_a",
                manifest_path="framework/partitions/pyramid_lidar_partition.yaml",
                registry_path="results/stage2/pyramid/evidence_registry.json",
                expected_output="results/stage2/pyramid/latency/latency_lut_rows_v1.jsonl",
                command=["python", "scripts/stage2_generate_latency_lut.py", "--job-id", "latency:cfg_a"],
                max_attempts=2,
                timeout_s=21600,
            ),
            job_plan_row(
                job_id="ap:cfg_a",
                model="pyramid_lidar",
                lut_kind="ap",
                job_type="generate_ap_lut",
                priority=5,
                config_id="cfg_a",
                manifest_path="framework/partitions/pyramid_lidar_partition.yaml",
                registry_path="results/stage2/pyramid/evidence_registry.json",
                expected_output="results/stage2/pyramid/ap/ap_anchor_rows_v1.jsonl",
                command=["python", "scripts/stage2_generate_ap_lut.py", "--job-id", "ap:cfg_a"],
                max_attempts=1,
                timeout_s=21600,
            ),
        ]
        state = [
            {"schema": "lut_job_state_row_v1", "job_id": "latency:cfg_a", "status": "succeeded", "attempt": 1},
        ]

        self.assertEqual(latest_job_status(state, "latency:cfg_a"), "succeeded")
        self.assertEqual([job["job_id"] for job in next_queued_jobs(plan, state)], ["ap:cfg_a"])
```

- [ ] **Step 2: Verify failure**

Expected: missing imports.

- [ ] **Step 3: Implement job helpers**

Add:

```python
VALID_LUT_KINDS = {"latency", "ap", "energy"}
VALID_JOB_TYPES = {
    "generate_latency_lut",
    "generate_ap_lut",
    "generate_energy_lut",
    "import_existing_latency_lut",
    "import_existing_ap_lut",
    "import_existing_energy_lut",
}


def job_plan_row(
    *,
    job_id: str,
    model: str,
    lut_kind: str,
    job_type: str,
    priority: int,
    config_id: str,
    manifest_path: str,
    registry_path: str,
    expected_output: str,
    command: list[str],
    max_attempts: int,
    timeout_s: int,
) -> dict[str, Any]:
    if lut_kind not in VALID_LUT_KINDS:
        raise LutProductizationError(f"unknown lut_kind: {lut_kind}")
    if job_type not in VALID_JOB_TYPES:
        raise LutProductizationError(f"unknown job_type: {job_type}")
    return {
        "schema": "lut_job_plan_row_v1",
        "job_id": job_id,
        "model": model,
        "lut_kind": lut_kind,
        "job_type": job_type,
        "priority": int(priority),
        "config_id": config_id,
        "manifest_path": manifest_path,
        "registry_path": registry_path,
        "expected_output": expected_output,
        "command": command,
        "max_attempts": int(max_attempts),
        "timeout_s": int(timeout_s),
    }


def latest_job_status(state_rows: list[dict[str, Any]], job_id: str) -> str:
    matches = [row for row in state_rows if row.get("job_id") == job_id]
    return str(matches[-1].get("status", "queued")) if matches else "queued"


def next_queued_jobs(plan_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending = [
        row for row in plan_rows
        if latest_job_status(state_rows, str(row["job_id"])) not in {"succeeded", "skipped", "running"}
    ]
    return sorted(pending, key=lambda row: (-int(row["priority"]), str(row["job_id"])))
```

- [ ] **Step 4: Create CLI skeleton**

Create `scripts/stage2_plan_lut_jobs.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import job_plan_row, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--phase", choices=("smoke", "calibration", "paper"), default="smoke")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out_jsonl)
    base_output = out.parent.parent
    rows = [
        job_plan_row(
            job_id=f"latency:{args.model}:smoke",
            model=args.model,
            lut_kind="latency",
            job_type="generate_latency_lut",
            priority=10,
            config_id=f"{args.model}__smoke__q_fp16__s_default",
            manifest_path=args.manifest,
            registry_path=args.registry,
            expected_output=str(base_output / "latency/latency_lut_rows_v1.jsonl"),
            command=[
                "python",
                "scripts/stage2_generate_latency_lut.py",
                "--job-id",
                f"latency:{args.model}:smoke",
                "--out-jsonl",
                str(base_output / "latency/latency_lut_rows_v1.jsonl"),
            ],
            max_attempts=2,
            timeout_s=21600,
        ),
        job_plan_row(
            job_id=f"ap:{args.model}:smoke",
            model=args.model,
            lut_kind="ap",
            job_type="generate_ap_lut",
            priority=9,
            config_id=f"{args.model}__smoke__q_fp16__s_not_applicable",
            manifest_path=args.manifest,
            registry_path=args.registry,
            expected_output=str(base_output / "ap/ap_anchor_rows_v1.jsonl"),
            command=[
                "python",
                "scripts/stage2_generate_ap_lut.py",
                "--job-id",
                f"ap:{args.model}:smoke",
                "--out-jsonl",
                str(base_output / "ap/ap_anchor_rows_v1.jsonl"),
            ],
            max_attempts=1,
            timeout_s=21600,
        ),
        job_plan_row(
            job_id=f"energy:{args.model}:smoke",
            model=args.model,
            lut_kind="energy",
            job_type="generate_energy_lut",
            priority=8,
            config_id=f"{args.model}__smoke__q_fp16__s_default",
            manifest_path=args.manifest,
            registry_path=args.registry,
            expected_output=str(base_output / "energy/energy_lut_rows_v1.jsonl"),
            command=[
                "python",
                "scripts/stage2_generate_energy_lut.py",
                "--job-id",
                f"energy:{args.model}:smoke",
                "--out-jsonl",
                str(base_output / "energy/energy_lut_rows_v1.jsonl"),
            ],
            max_attempts=1,
            timeout_s=21600,
        ),
    ]
    if not args.dry_run:
        write_jsonl(out, rows)
    print(json.dumps({"schema": "lut_job_plan_summary_v1", "jobs": len(rows), "out": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests and CLI dry-run**

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m unittest framework.tests.test_stage2_lut_productization
PYTHONPATH=/home/jichengzhi/V2X python scripts/stage2_plan_lut_jobs.py \
  --model pyramid_lidar \
  --manifest framework/partitions/pyramid_lidar_partition.yaml \
  --registry results/stage2/pyramid/evidence_registry.json \
  --phase smoke \
  --out-jsonl /tmp/lut_job_plan_v1.jsonl \
  --dry-run
```

Expected: tests pass; CLI prints `{"schema": "lut_job_plan_summary_v1", "jobs": 3, ...}`.

- [ ] **Step 6: Commit**

```bash
git add framework/stage2/lut_productization.py framework/tests/test_stage2_lut_productization.py scripts/stage2_plan_lut_jobs.py
git commit -m "feat: plan resumable stage2 lut jobs"
```

---

### Task 4: Productized LUT Generation Driver CLIs

**Files:**
- Create: `scripts/stage2_generate_latency_lut.py`
- Create: `scripts/stage2_generate_ap_lut.py`
- Create: `scripts/stage2_generate_energy_lut.py`
- Modify: `framework/stage2/lut_productization.py`
- Modify: `framework/tests/test_stage2_lut_productization.py`

- [ ] **Step 1: Write generation smoke tests**

Add a test that runs all three generator scripts with command JSON arrays. Each command prints one JSON object to stdout; each generator parses stdout and appends one canonical row.

```python
class Stage2LutGeneratorCliTest(unittest.TestCase):
    def test_latency_ap_energy_generators_append_rows(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            latency_out = tmp_path / "latency_lut_rows_v1.jsonl"
            ap_out = tmp_path / "ap_anchor_rows_v1.jsonl"
            energy_out = tmp_path / "energy_lut_rows_v1.jsonl"

            latency_cmd = [
                sys.executable,
                "-c",
                "import json; print(json.dumps({'latency_p50_us': 123.4, 'latency_p90_us': 140.0, 'latency_mean_us': 125.0}))",
            ]
            latency = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/stage2_generate_latency_lut.py"),
                    "--model", "pyramid_lidar",
                    "--config-id", "cfg_latency",
                    "--candidate-id", "smoke",
                    "--software-point-id", "smoke:w64x128x256:fp16",
                    "--dense-stage", "neck",
                    "--width", "64,128,256",
                    "--quant-policy", "fp16",
                    "--schedule-policy", "default",
                    "--backend", "h800_tvm",
                    "--measurement-command-json", json.dumps(latency_cmd),
                    "--out-jsonl", str(latency_out),
                ],
                cwd=root,
                env={"PYTHONPATH": str(root)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(latency.returncode, 0, latency.stderr)
            latency_row = json.loads(latency_out.read_text().splitlines()[0])
            self.assertEqual(latency_row["schema"], "latency_lut_row_v1")
            self.assertEqual(latency_row["measurement_status"], "measured")
            self.assertEqual(latency_row["latency_p50_us"], 123.4)

            ap_cmd = [
                sys.executable,
                "-c",
                "import json; print(json.dumps({'metric': 'AP70', 'metric_value': 0.612, 'dataset': 'DAIR-V2X', 'eval_split': 'val', 'ckpt_path': 'ckpts/smoke.pt'}))",
            ]
            ap = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/stage2_generate_ap_lut.py"),
                    "--model", "pyramid_lidar",
                    "--config-id", "cfg_latency",
                    "--candidate-id", "smoke",
                    "--software-point-id", "smoke:w64x128x256:fp16",
                    "--dense-stage", "model",
                    "--width", "64,128,256",
                    "--quant-policy", "fp16",
                    "--schedule-policy", "not_applicable",
                    "--eval-command-json", json.dumps(ap_cmd),
                    "--out-jsonl", str(ap_out),
                ],
                cwd=root,
                env={"PYTHONPATH": str(root)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(ap.returncode, 0, ap.stderr)
            ap_row = json.loads(ap_out.read_text().splitlines()[0])
            self.assertEqual(ap_row["schema"], "ap_anchor_row_v1")
            self.assertEqual(ap_row["config_id"], "cfg_latency")
            self.assertEqual(ap_row["metric_value"], 0.612)

            energy_cmd = [
                sys.executable,
                "-c",
                "import json; print(json.dumps({'joule_per_inference': 1.2, 'watt_avg': 240.0, 'telemetry_source': 'nvidia_smi', 'idle_baseline_policy': 'subtract_idle_avg', 'raw_artifact': 'logs/energy_smoke.json'}))",
            ]
            energy = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/stage2_generate_energy_lut.py"),
                    "--model", "pyramid_lidar",
                    "--config-id", "cfg_latency",
                    "--candidate-id", "smoke",
                    "--software-point-id", "smoke:w64x128x256:fp16",
                    "--dense-stage", "neck",
                    "--width", "64,128,256",
                    "--quant-policy", "fp16",
                    "--schedule-policy", "default",
                    "--backend", "h800_tvm_power_telemetry",
                    "--telemetry-command-json", json.dumps(energy_cmd),
                    "--out-jsonl", str(energy_out),
                ],
                cwd=root,
                env={"PYTHONPATH": str(root)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(energy.returncode, 0, energy.stderr)
            energy_row = json.loads(energy_out.read_text().splitlines()[0])
            self.assertEqual(energy_row["schema"], "energy_lut_row_v1")
            self.assertEqual(energy_row["measurement_status"], "measured")
            self.assertEqual(energy_row["telemetry_source"], "nvidia_smi")
```

- [ ] **Step 2: Run tests to verify missing generator scripts**

Expected: FAIL with `can't open file ... stage2_generate_latency_lut.py`.

- [ ] **Step 3: Add shared command-output helper**

Add to `framework/stage2/lut_productization.py`:

```python
def parse_width_csv(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise LutProductizationError("width must contain at least one integer")
    return [int(part) for part in parts]


def run_json_command(command_json: str) -> dict[str, Any]:
    command = json.loads(command_json)
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise LutProductizationError("command json must be a JSON string array")
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise LutProductizationError(proc.stderr.strip() or f"command failed with exit code {proc.returncode}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise LutProductizationError(f"command stdout is not JSON: {proc.stdout[:200]}") from exc
    if not isinstance(payload, dict):
        raise LutProductizationError("command stdout JSON must be an object")
    return payload
```

Add imports:

```python
import subprocess
```

- [ ] **Step 4: Implement latency generator**

Create `scripts/stage2_generate_latency_lut.py` with CLI args from the test. It must call `run_json_command(args.measurement_command_json)`, require `latency_p50_us`, and write one `latency_lut_row()` with `measurement_status="measured"`.

- [ ] **Step 5: Implement AP generator**

Create `scripts/stage2_generate_ap_lut.py` with CLI args from the test. It must call `run_json_command(args.eval_command_json)`, require `metric_value`, and write one `ap_anchor_row()` with the same `config_id` used by latency/energy.

- [ ] **Step 6: Implement energy generator**

Create `scripts/stage2_generate_energy_lut.py` with CLI args from the test. It must call `run_json_command(args.telemetry_command_json)`, require `joule_per_inference`, `telemetry_source`, and `raw_artifact`, and write one `energy_lut_row()` with `measurement_status="measured"`.

- [ ] **Step 7: Run tests**

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m unittest framework.tests.test_stage2_lut_productization
```

Expected: generator smoke test passes and each output JSONL contains one canonical row.

- [ ] **Step 8: Commit**

```bash
git add framework/stage2/lut_productization.py framework/tests/test_stage2_lut_productization.py scripts/stage2_generate_latency_lut.py scripts/stage2_generate_ap_lut.py scripts/stage2_generate_energy_lut.py
git commit -m "feat: generate stage2 lut rows from measurement commands"
```

---

### Task 5: Historical Latency And AP Importer CLIs

**Files:**
- Create: `scripts/stage2_import_latency_lut.py`
- Create: `scripts/stage2_import_ap_anchors.py`
- Modify: `framework/tests/test_stage2_lut_productization.py`

- [ ] **Step 1: Write integration tests**

Add tests that run both scripts against existing Pyramid fixtures and assert JSONL rows exist:

```python
import json
import subprocess
import sys


class Stage2LutImporterCliTest(unittest.TestCase):
    def test_latency_and_ap_importers_write_canonical_rows(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            latency_out = tmp_path / "latency_lut_rows_v1.jsonl"
            ap_out = tmp_path / "ap_anchor_rows_v1.jsonl"

            latency = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/stage2_import_latency_lut.py"),
                    "--model", "pyramid_lidar",
                    "--source-json", str(root / "results/latency_lut_pyramid.json"),
                    "--out-jsonl", str(latency_out),
                    "--backend", "h800_tvm",
                    "--measurement-status", "measured",
                ],
                cwd=root,
                env={"PYTHONPATH": str(root)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(latency.returncode, 0, latency.stderr)
            first_latency = json.loads(latency_out.read_text().splitlines()[0])
            self.assertEqual(first_latency["schema"], "latency_lut_row_v1")
            self.assertEqual(first_latency["backend"], "h800_tvm")

            ap = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/stage2_import_ap_anchors.py"),
                    "--model", "pyramid_lidar",
                    "--source-json", str(root / "results/ap70_model_pyramid.json"),
                    "--out-jsonl", str(ap_out),
                    "--metric", "AP70",
                    "--dataset", "DAIR-V2X",
                    "--eval-split", "val",
                    "--finetune-protocol", "mixed_existing_anchors",
                ],
                cwd=root,
                env={"PYTHONPATH": str(root)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(ap.returncode, 0, ap.stderr)
            first_ap = json.loads(ap_out.read_text().splitlines()[0])
            self.assertEqual(first_ap["schema"], "ap_anchor_row_v1")
            self.assertEqual(first_ap["metric"], "AP70")
```

- [ ] **Step 2: Run tests to verify scripts are missing**

Expected: FAIL with `can't open file ... stage2_import_latency_lut.py`.

- [ ] **Step 3: Implement `scripts/stage2_import_latency_lut.py`**

Use this complete first version:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import latency_lut_row, stable_config_id, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--backend", default="h800_tvm")
    parser.add_argument("--measurement-status", default="measured")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source_json)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    created_at = datetime.now(timezone.utc).isoformat()
    for item in data.get("widths", data.get("grid", [])):
        width = [int(v) for v in item["num_filters"]]
        label = str(item.get("label", "_".join(map(str, width))))
        for sched, key in (("default", "default_us"), ("metaschedule_tuned", "tuned_us")):
            config_id = stable_config_id(
                model=args.model,
                candidate_id=label,
                software_point_id=f"{label}:w{'x'.join(map(str, width))}:fp16",
                quant_policy="fp16",
                schedule_policy=sched,
            )
            rows.append(
                latency_lut_row(
                    config_id=config_id,
                    model=args.model,
                    manifest_digest=digest,
                    candidate_id=label,
                    software_point_id=f"{label}:w{'x'.join(map(str, width))}:fp16",
                    dense_stage=str(item.get("dense_stage", "unknown")),
                    width=width,
                    quant_policy="fp16",
                    schedule_policy=sched,
                    backend=args.backend,
                    measurement_status=args.measurement_status if item.get("_source") != "estimated_vol_power_law" else "estimated",
                    latency_p50_us=float(item[key]),
                    latency_p90_us=None,
                    latency_mean_us=None,
                    latency_std_us=None,
                    latency_min_us=None,
                    latency_max_us=None,
                    warmup_iters=0,
                    measure_iters=0,
                    repeat=0,
                    run_id=f"import_latency_{source.stem}",
                    created_at=created_at,
                    source_files=[str(source)],
                    provenance=str(data.get("_source", data.get("_source_real", "imported_latency_lut"))),
                    notes="imported from existing Stage2 latency LUT",
                )
            )
    write_jsonl(args.out_jsonl, rows)
    print(f"stage2_import_latency_lut_ok rows={len(rows)} out={args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Implement `scripts/stage2_import_ap_anchors.py`**

Use this complete first version:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import ap_anchor_row, stable_config_id, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--metric", default="AP70")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eval-split", required=True)
    parser.add_argument("--finetune-protocol", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source_json)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    created_at = datetime.now(timezone.utc).isoformat()
    for item in data.get("table", data.get("anchors", data.get("fit_points", []))):
        width = [int(v) for v in item.get("num_filters", [item.get("s0"), item.get("s1"), item.get("s2")]) if v is not None]
        label = str(item.get("label", item.get("tag", "_".join(map(str, width)))))
        value = item.get("ap70")
        if value is None:
            continue
        config_id = stable_config_id(
            model=args.model,
            candidate_id=label,
            software_point_id=f"{label}:w{'x'.join(map(str, width))}:fp16",
            quant_policy="fp16",
            schedule_policy="not_applicable",
        )
        rows.append(
            ap_anchor_row(
                config_id=config_id,
                model=args.model,
                manifest_digest=digest,
                candidate_id=label,
                software_point_id=f"{label}:w{'x'.join(map(str, width))}:fp16",
                dense_stage=str(item.get("dense_stage", "model")),
                width=width,
                quant_policy="fp16",
                schedule_policy="not_applicable",
                backend="model_eval",
                measurement_status="measured",
                metric=args.metric,
                metric_value=float(value),
                secondary_metrics={},
                dataset=args.dataset,
                eval_split=args.eval_split,
                num_samples=None,
                ckpt_path=str(item.get("ckpt_path", "unknown")),
                ckpt_digest=None,
                finetune_protocol=args.finetune_protocol,
                training_budget=str(item.get("training_budget", "unknown")),
                eval_command="imported_existing_ap_anchor",
                run_id=f"import_ap_{source.stem}",
                created_at=created_at,
                source_files=[str(source)],
                provenance=str(item.get("source", item.get("_source", data.get("_source", "imported_ap_anchor")))),
                notes="imported from existing Stage2 AP anchor/model file",
            )
        )
    write_jsonl(args.out_jsonl, rows)
    print(f"stage2_import_ap_anchors_ok rows={len(rows)} out={args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests**

Expected: importer CLI test passes.

- [ ] **Step 6: Commit**

```bash
git add scripts/stage2_import_latency_lut.py scripts/stage2_import_ap_anchors.py framework/tests/test_stage2_lut_productization.py
git commit -m "feat: import stage2 latency and ap lut rows"
```

---

### Task 6: Energy Importer And No-Claim/Claim Gate

**Files:**
- Create: `scripts/stage2_import_energy_lut.py`
- Modify: `framework/tests/test_stage2_lut_productization.py`

- [ ] **Step 1: Write failing energy importer test**

Create a temp CSV with columns:

```csv
config_id,model,candidate_id,software_point_id,dense_stage,width,quant_policy,schedule_policy,joule_per_inference,watt_avg,telemetry_source,latency_run_id
cfg,pyramid_lidar,neck,neck:w64:fp16,neck,64|128|256,fp16,default,1.2,240.0,nvidia_smi,run_001
```

Assert output row has `schema=energy_lut_row_v1`, `measurement_status=measured`, `backend=h800_tvm_power_telemetry`.

- [ ] **Step 2: Run test to verify missing script failure**

Expected: `can't open file ... stage2_import_energy_lut.py`.

- [ ] **Step 3: Implement importer**

Implement `scripts/stage2_import_energy_lut.py` with `csv.DictReader`, `energy_lut_row()`, and `write_jsonl()`. Parse `width` with `row["width"].split("|")`. Set `idle_baseline_policy="subtract_idle_avg"` unless a CSV column overrides it.

- [ ] **Step 4: Run tests**

Expected: energy importer test passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/stage2_import_energy_lut.py framework/tests/test_stage2_lut_productization.py
git commit -m "feat: import stage2 energy lut rows"
```

---

### Task 7: Registry Update From LUT Rows

**Files:**
- Create: `scripts/stage2_update_registry_from_luts.py`
- Modify: `framework/tests/test_stage2_lut_productization.py`

- [ ] **Step 1: Write failing registry update test**

Use `scripts/stage2_prepare_evidence.py` to create a temp registry, create temp latency/AP/energy JSONL files, run updater, and assert:

```python
updated["latency_lut"]["path"].endswith("latency_lut_rows_v1.jsonl")
updated["latency_lut"]["coverage"]["measured_cells"] == 1
updated["energy_lut"]["coverage"]["measured_cells"] == 1
updated["energy_lut"]["measurement_status"] == "measured"
```

- [ ] **Step 2: Run test to verify missing script failure**

Expected: missing updater script.

- [ ] **Step 3: Implement updater**

Read registry JSON, read row JSONL files, compute coverage with `coverage_from_rows()`, replace `path`, `measurement_status`, `backend`, and `coverage` for each provided source. Energy `measurement_status` becomes `measured` only if at least one energy row is measured.

- [ ] **Step 4: Run tests**

Expected: updater test passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/stage2_update_registry_from_luts.py framework/tests/test_stage2_lut_productization.py
git commit -m "feat: update stage2 registry from lut rows"
```

---

### Task 8: Long-Running Worker With Resume

**Files:**
- Create: `scripts/stage2_lut_worker.py`
- Modify: `framework/tests/test_stage2_lut_productization.py`

- [ ] **Step 1: Write worker resume test**

Create a plan JSONL with two jobs:

```json
{"schema":"lut_job_plan_row_v1","job_id":"latency:done","model":"pyramid_lidar","lut_kind":"latency","job_type":"generate_latency_lut","priority":10,"config_id":"done","manifest_path":"m","registry_path":"r","expected_output":"/tmp/out.jsonl","command":["python","-c","print('done')"],"max_attempts":1,"timeout_s":30}
{"schema":"lut_job_plan_row_v1","job_id":"latency:new","model":"pyramid_lidar","lut_kind":"latency","job_type":"generate_latency_lut","priority":9,"config_id":"new","manifest_path":"m","registry_path":"r","expected_output":"/tmp/out.jsonl","command":["python","-c","print('new')"],"max_attempts":1,"timeout_s":30}
```

Create state JSONL with `latency:done` succeeded. Run worker with `--max-jobs 1`. Assert state contains `latency:new` succeeded and does not rerun `latency:done`.

- [ ] **Step 2: Run test to verify missing worker failure**

Expected: missing worker script.

- [ ] **Step 3: Implement worker**

Implement:

- read plan via `read_jsonl()`
- read state via `read_jsonl()`
- select `next_queued_jobs()`
- append running state
- run command via `subprocess.run`
- append succeeded or failed state
- stop at `--max-jobs` or `--max-hours`

- [ ] **Step 4: Run tests**

Expected: worker resume test passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/stage2_lut_worker.py framework/tests/test_stage2_lut_productization.py
git commit -m "feat: run resumable stage2 lut jobs"
```

---

### Task 9: Final Verification And Documentation Sync

**Files:**
- Modify: `multi_agent/methods/progress/HANDOFF_stage2_lut_productization_plan_v1_zh.md`
- Create: `docs/stage2-evidence-registry.zh-CN.md`

- [ ] **Step 1: Run full targeted tests**

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m unittest \
  framework.tests.test_stage2_lut_productization \
  framework.tests.test_stage2_evidence_registry \
  framework.tests.test_stage2_integration_contract
```

Expected: all tests pass.

- [ ] **Step 2: Run py_compile**

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m py_compile \
  framework/stage2/lut_productization.py \
  scripts/stage2_plan_lut_jobs.py \
  scripts/stage2_generate_latency_lut.py \
  scripts/stage2_generate_ap_lut.py \
  scripts/stage2_generate_energy_lut.py \
  scripts/stage2_import_latency_lut.py \
  scripts/stage2_import_ap_anchors.py \
  scripts/stage2_import_energy_lut.py \
  scripts/stage2_update_registry_from_luts.py \
  scripts/stage2_lut_worker.py
```

Expected: exit code 0.

- [ ] **Step 3: Run dry-run and smoke queue**

```bash
tmp=$(mktemp -d)
PYTHONPATH=/home/jichengzhi/V2X python scripts/stage2_plan_lut_jobs.py \
  --model pyramid_lidar \
  --manifest framework/partitions/pyramid_lidar_partition.yaml \
  --registry "$tmp/evidence_registry.json" \
  --phase smoke \
  --out-jsonl "$tmp/jobs/lut_job_plan_v1.jsonl"
PYTHONPATH=/home/jichengzhi/V2X python scripts/stage2_lut_worker.py \
  --job-plan "$tmp/jobs/lut_job_plan_v1.jsonl" \
  --job-state "$tmp/jobs/lut_job_state_v1.jsonl" \
  --max-jobs 1 \
  --resume
```

Expected: worker appends one succeeded or failed state row; no traceback.

- [ ] **Step 4: Run background command smoke**

```bash
mkdir -p logs
nohup env PYTHONPATH=/home/jichengzhi/V2X \
  python scripts/stage2_lut_worker.py \
    --job-plan "$tmp/jobs/lut_job_plan_v1.jsonl" \
    --job-state "$tmp/jobs/lut_job_state_v1.jsonl" \
    --max-hours 12 \
    --resume \
  > "logs/stage2_lut_worker_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

Expected: command starts in the background, writes state rows incrementally, and can be stopped and resumed without duplicating succeeded jobs.

- [ ] **Step 5: Update docs**

Document:

- canonical JSONL row tables
- job plan/state format
- background command
- measured/proxy/historical rules
- energy no-claim rule

- [ ] **Step 6: Commit**

```bash
git add framework/stage2/lut_productization.py framework/tests/test_stage2_lut_productization.py scripts/stage2_*.py docs/stage2-evidence-registry.zh-CN.md multi_agent/methods/progress/HANDOFF_stage2_lut_productization_plan_v1_zh.md
git commit -m "docs: plan stage2 lut productization"
```

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-25-stage2-lut-productization.md`. Two execution options:

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.
