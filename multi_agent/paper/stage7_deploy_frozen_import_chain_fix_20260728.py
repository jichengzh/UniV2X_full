#!/usr/bin/env python3
"""Deploy the trusted frozen-import chain used by Stage7 source workers."""

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
    "scripts/stage7_resolve_round_sources_v2.py": {
        "old": "0c629f834116c3292c5a8aa4df4b41eca187f7c771250e5e4966188d7d887b99",
        "new": "6960ec5e3624cafaf8638773470e6d777d76018f4a9f4024cf52d3a3c0b62cdb",
    },
    "scripts/stage7_source_scheduler_v2.py": {
        "old": "d68350ac9cc540c8c14faf595643274accebac205e88f38e3ce2bc7fa4d76fc8",
        "new": "4bb567769192c42463760c7b5325550a0a13fa7ef33b7d3f64627c75951ed740",
    },
}
deploy.TRANSACTION_SCHEMA = "stage7_frozen_import_chain_fix_transaction_v1"
deploy.main()
