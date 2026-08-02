#!/usr/bin/env python3
"""Generate Stage2 productized LUT job plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import (  # noqa: E402
    LATENCY_ROW_SCHEMA,
    default_quant_contract,
    job_plan_row,
    parse_width_csv,
    stable_config_id,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--phase", choices=("smoke", "calibration", "paper"), default="smoke")
    parser.add_argument("--config-id")
    parser.add_argument("--candidate-id", default="smoke")
    parser.add_argument("--software-point-id", default="smoke:w64x128x256:fp16")
    parser.add_argument("--dense-stage", default="neck")
    parser.add_argument("--optimized-scope", default="rsu_dense_core")
    parser.add_argument("--width", default="64,128,256")
    parser.add_argument("--quant-policy", default="fp16")
    parser.add_argument("--precision")
    parser.add_argument("--quant-scheme")
    parser.add_argument("--quant-method")
    parser.add_argument("--quant-scope")
    parser.add_argument("--calibration-source")
    parser.add_argument("--calibration-digest")
    parser.add_argument("--calibrator")
    parser.add_argument("--calibration-inputs", default="")
    parser.add_argument("--fallback-policy")
    parser.add_argument("--layer-precision-summary")
    parser.add_argument("--full-network-claim", choices=("true", "false"))
    parser.add_argument("--engine-kind")
    parser.add_argument("--engine-digest")
    parser.add_argument("--measurement-source")
    parser.add_argument("--claim-status")
    parser.add_argument("--quality-gate-status")
    parser.add_argument("--schedule-profile")
    parser.add_argument("--tune-budget")
    parser.add_argument("--schedule-policy", default="default")
    parser.add_argument("--manifest-digest", default="unknown")
    parser.add_argument("--latency-measurement-command-json", required=True)
    parser.add_argument("--ap-eval-command-json", required=True)
    parser.add_argument("--energy-telemetry-command-json", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _base_output(out_jsonl: Path) -> Path:
    return out_jsonl.parent.parent


def _generator_command(
    *,
    script: str,
    model: str,
    config_id: str,
    candidate_id: str,
    software_point_id: str,
    dense_stage: str,
    optimized_scope: str,
    width: str,
    quant_policy: str,
    quant_contract: dict[str, object],
    schedule_policy: str,
    manifest_digest: str,
    command_flag: str,
    command_json: str,
    out_jsonl: Path,
    backend: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        script,
        "--model",
        model,
        "--config-id",
        config_id,
        "--candidate-id",
        candidate_id,
        "--software-point-id",
        software_point_id,
        "--dense-stage",
        dense_stage,
        "--optimized-scope",
        optimized_scope,
        "--width",
        width,
        "--quant-policy",
        quant_policy,
        "--precision",
        str(quant_contract["precision"]),
        "--quant-scheme",
        str(quant_contract["quant_scheme"]),
        "--quant-method",
        str(quant_contract["quant_method"]),
        "--quant-scope",
        str(quant_contract["quant_scope"]),
        "--calibration-source",
        str(quant_contract["calibration_source"]),
        "--calibration-digest",
        str(quant_contract["calibration_digest"]),
        "--calibrator",
        str(quant_contract["calibrator"]),
        "--calibration-inputs",
        ",".join(str(item) for item in quant_contract["calibration_inputs"]),
        "--fallback-policy",
        str(quant_contract["fallback_policy"]),
        "--layer-precision-summary",
        str(quant_contract["layer_precision_summary"]),
        "--full-network-claim",
        "true" if quant_contract["full_network_claim"] else "false",
        "--engine-kind",
        str(quant_contract["engine_kind"]),
        "--engine-digest",
        str(quant_contract["engine_digest"]),
        "--measurement-source",
        str(quant_contract["measurement_source"]),
        "--claim-status",
        str(quant_contract["claim_status"]),
        "--quality-gate-status",
        str(quant_contract["quality_gate_status"]),
        "--schedule-profile",
        str(quant_contract["schedule_profile"]),
        "--tune-budget",
        str(quant_contract["tune_budget"]),
        "--schedule-policy",
        schedule_policy,
        "--manifest-digest",
        manifest_digest,
        command_flag,
        command_json,
        "--out-jsonl",
        str(out_jsonl),
    ]
    if backend is not None:
        command.extend(["--backend", backend])
    return command


def _quant_contract(args: argparse.Namespace) -> dict[str, object]:
    contract = default_quant_contract(
        quant_policy=args.quant_policy,
        backend="h800_tvm",
        optimized_scope=args.optimized_scope,
        schedule_policy=args.schedule_policy,
        measurement_status="not_done",
        schema=LATENCY_ROW_SCHEMA,
    )
    overrides = {
        "precision": args.precision,
        "quant_scheme": args.quant_scheme,
        "quant_method": args.quant_method,
        "quant_scope": args.quant_scope,
        "calibration_source": args.calibration_source,
        "calibration_digest": args.calibration_digest,
        "calibrator": args.calibrator,
        "fallback_policy": args.fallback_policy,
        "layer_precision_summary": args.layer_precision_summary,
        "engine_kind": args.engine_kind,
        "engine_digest": args.engine_digest,
        "measurement_source": args.measurement_source,
        "claim_status": args.claim_status,
        "quality_gate_status": args.quality_gate_status,
        "schedule_profile": args.schedule_profile,
        "tune_budget": args.tune_budget,
    }
    for key, value in overrides.items():
        if value is not None:
            contract[key] = value
    if args.calibration_inputs:
        contract["calibration_inputs"] = [
            item.strip() for item in args.calibration_inputs.split(",") if item.strip()
        ]
    if args.full_network_claim is not None:
        contract["full_network_claim"] = args.full_network_claim == "true"
    return contract


def main() -> int:
    args = parse_args()
    parse_width_csv(args.width)
    out = Path(args.out_jsonl)
    base_output = _base_output(out)
    config_id = args.config_id or stable_config_id(
        model=args.model,
        candidate_id=args.candidate_id,
        software_point_id=args.software_point_id,
        quant_policy=args.quant_policy,
        schedule_policy=args.schedule_policy,
    )
    quant_contract = _quant_contract(args)

    latency_out = base_output / "latency/latency_lut_rows_v1.jsonl"
    ap_out = base_output / "ap/ap_anchor_rows_v1.jsonl"
    energy_out = base_output / "energy/energy_lut_rows_v1.jsonl"
    rows = [
        job_plan_row(
            job_id=f"latency:{config_id}",
            model=args.model,
            lut_kind="latency",
            job_type="generate_latency_lut",
            priority=30,
            config_id=config_id,
            manifest_path=args.manifest,
            registry_path=args.registry,
            candidate_id=args.candidate_id,
            software_point_id=args.software_point_id,
            expected_output=str(latency_out),
            command=_generator_command(
                script="scripts/stage2_generate_latency_lut.py",
                model=args.model,
                config_id=config_id,
                candidate_id=args.candidate_id,
                software_point_id=args.software_point_id,
                dense_stage=args.dense_stage,
                optimized_scope=args.optimized_scope,
                width=args.width,
                quant_policy=args.quant_policy,
                quant_contract=quant_contract,
                schedule_policy=args.schedule_policy,
                manifest_digest=args.manifest_digest,
                command_flag="--measurement-command-json",
                command_json=args.latency_measurement_command_json,
                out_jsonl=latency_out,
                backend="h800_tvm",
            ),
            max_attempts=2,
            timeout_s=21600,
            resource={"gpu": "any_h800", "exclusive": True},
        ),
        job_plan_row(
            job_id=f"ap:{config_id}",
            model=args.model,
            lut_kind="ap",
            job_type="generate_ap_lut",
            priority=20,
            config_id=config_id,
            manifest_path=args.manifest,
            registry_path=args.registry,
            candidate_id=args.candidate_id,
            software_point_id=args.software_point_id,
            expected_output=str(ap_out),
            command=_generator_command(
                script="scripts/stage2_generate_ap_lut.py",
                model=args.model,
                config_id=config_id,
                candidate_id=args.candidate_id,
                software_point_id=args.software_point_id,
                dense_stage=args.dense_stage,
                optimized_scope=args.optimized_scope,
                width=args.width,
                quant_policy=args.quant_policy,
                quant_contract=quant_contract,
                schedule_policy="not_applicable",
                manifest_digest=args.manifest_digest,
                command_flag="--eval-command-json",
                command_json=args.ap_eval_command_json,
                out_jsonl=ap_out,
            ),
            max_attempts=1,
            timeout_s=21600,
            resource={"gpu": "any_h800", "exclusive": False},
        ),
        job_plan_row(
            job_id=f"energy:{config_id}",
            model=args.model,
            lut_kind="energy",
            job_type="generate_energy_lut",
            priority=10,
            config_id=config_id,
            manifest_path=args.manifest,
            registry_path=args.registry,
            candidate_id=args.candidate_id,
            software_point_id=args.software_point_id,
            expected_output=str(energy_out),
            command=_generator_command(
                script="scripts/stage2_generate_energy_lut.py",
                model=args.model,
                config_id=config_id,
                candidate_id=args.candidate_id,
                software_point_id=args.software_point_id,
                dense_stage=args.dense_stage,
                optimized_scope=args.optimized_scope,
                width=args.width,
                quant_policy=args.quant_policy,
                quant_contract=quant_contract,
                schedule_policy=args.schedule_policy,
                manifest_digest=args.manifest_digest,
                command_flag="--telemetry-command-json",
                command_json=args.energy_telemetry_command_json,
                out_jsonl=energy_out,
                backend="h800_tvm_power_telemetry",
            ),
            max_attempts=1,
            timeout_s=21600,
            resource={"gpu": "any_h800", "exclusive": True},
        ),
    ]
    if not args.dry_run:
        write_jsonl(out, rows)
    print(
        json.dumps(
            {
                "schema": "lut_job_plan_summary_v1",
                "phase": args.phase,
                "jobs": len(rows),
                "job_types": [row["job_type"] for row in rows],
                "out": str(out),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
