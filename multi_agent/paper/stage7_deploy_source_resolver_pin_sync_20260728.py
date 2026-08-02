#!/usr/bin/env python3
"""Deploy the Stage7 source scheduler pin for the reviewed resolver."""

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
    "scripts/stage7_source_scheduler_v2.py": {
        "old": "c17437bfbcade17d19e25f0f3329516324bfb2834ad67aaa71117124bac8bf03",
        "new": "d68350ac9cc540c8c14faf595643274accebac205e88f38e3ce2bc7fa4d76fc8",
    },
}
deploy.TRANSACTION_SCHEMA = "stage7_source_resolver_pin_sync_transaction_v1"
deploy.main()
