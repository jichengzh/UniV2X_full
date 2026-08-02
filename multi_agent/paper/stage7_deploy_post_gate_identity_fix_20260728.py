#!/usr/bin/env python3
"""Deploy reviewed post-launch identity and source-import fixes."""

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
    "scripts/stage7_scheduler_requests_v2.py": {
        "old": "898caac149985a3eb4646e7795c8c8d13ec739ea3b1f59781942d73a0ecc3a64",
        "new": "6ada2be3c0b9ce6c09af40f50a2a4760e097ee823dcf51c156412fc5ae108572",
    },
    "scripts/stage7_resolve_round_sources_v2.py": {
        "old": "fc8dd87e70eecfc89a8c59eea135e8a2d33f892ad1d1207d7ca30f8552669ea3",
        "new": "0c629f834116c3292c5a8aa4df4b41eca187f7c771250e5e4966188d7d887b99",
    },
}
deploy.TRANSACTION_SCHEMA = "stage7_post_gate_identity_source_import_fix_transaction_v1"
deploy.main()
