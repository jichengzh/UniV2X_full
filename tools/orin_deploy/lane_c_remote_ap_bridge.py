#!/usr/bin/env python3
"""Run the exact full_1789 bridge with only the backbone executed on Orin."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from tools.orin_deploy.lane_c_backbone_rpc import OrinBackboneRpcClient


def augment_report(
    report: dict[str, Any],
    *,
    rpc_audit: dict[str, Any],
    orin_host: str,
) -> dict[str, Any]:
    return {
        **report,
        "replacement_scope": "get_multiscale_feature_only",
        "backbone_execution_location": str(orin_host),
        "downstream_execution_location": "server_pytorch_checkpoint_precision",
        "transport": "persistent_RPC_over_SSH_local_forward",
        "local_engine_file_role": "sha_anchor_not_local_execution",
        "orin_rpc_client_audit": dict(rpc_audit),
    }


def _argument_value(name: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError as error:
        raise ValueError(f"{name} is required") from error
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{name} requires a value")
    return sys.argv[index + 1]


def main() -> int:
    host = os.environ.get("LANE_C_ORIN_HOST", "").strip()
    if not host:
        raise ValueError("LANE_C_ORIN_HOST must identify the Orin execution host")
    rpc_host = os.environ.get("LANE_C_RPC_HOST", "127.0.0.1")
    rpc_port = int(os.environ["LANE_C_RPC_PORT"])
    report_path = Path(_argument_value("--report-json")).expanduser().resolve()
    artifact_root = Path(os.environ["LANE_C_ARTIFACT_ROOT"]).expanduser().resolve()
    if report_path != artifact_root and artifact_root not in report_path.parents:
        raise ValueError(
            f"--report-json must remain under artifact root {artifact_root}"
        )
    client = OrinBackboneRpcClient(rpc_host, rpc_port)
    try:
        import scripts.stage3_trt_multiscale_ap_bridge_v3 as bridge_module

        bridge_module.TrtMultiscaleRunner = lambda _engine_path: client
        exit_code = int(bridge_module.main())
        if exit_code == 0:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            augmented = augment_report(
                report, rpc_audit=client.audit(), orin_host=host
            )
            report_path.write_text(
                json.dumps(augmented, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return exit_code
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
