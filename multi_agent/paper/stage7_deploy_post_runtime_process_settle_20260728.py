#!/usr/bin/env python3
"""Deploy the reviewed Stage7 post-runtime process-settle fix."""

from __future__ import annotations

import importlib.util
from pathlib import Path


BASE = Path(__file__).with_name(
    "stage7_deploy_replicate_dependency_pins_20260727.py"
)
spec = importlib.util.spec_from_file_location("_stage7_transactional_deploy", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load transactional Stage7 deployment helper")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)

deploy.TARGETS = {
    "framework/stage7/physical_runtime_v2.py": {
        "old": "e1f9bb23b414ce628c3508ddf94fb3058ecc55f96ae3e078c792d880ae0e02ff",
        "new": "0506b8f624f3f31b286d32555f9e3b141358424e71cd31f31dccee031ed0381c",
    },
}
deploy.TRANSACTION_SCHEMA = "stage7_post_runtime_process_settle_transaction_v1"
deploy.main()
