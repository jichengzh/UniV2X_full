#!/usr/bin/env python3
"""Deploy the Stage7 batched measurement submission scheduler fix."""

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
    "scripts/stage7_core_ablation_orchestrator_v2.py": {
        "old": "4e06cd07e9e2d3c477397241f8cf02fff256e65086a875e5e318bc47c7f63583",
        "new": "c61c7615988415d048150a2adbc92d41369d692558f9e4e897de959bd7c7fbe2",
    },
    "scripts/stage7_orchestrator_scheduler_adapter_v2.py": {
        "old": "664fa48413ae0d28fe0f9c72691587b53c2e4751a61af923041570354df61229",
        "new": "d1a4531a58db2f1de332b76702a7814647b475466feb9ecfb7636db0c8e1e391",
    },
}
deploy.TRANSACTION_SCHEMA = "stage7_batch_measurement_submit_transaction_v1"
deploy.main()
