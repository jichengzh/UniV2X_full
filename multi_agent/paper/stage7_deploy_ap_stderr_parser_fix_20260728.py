#!/usr/bin/env python3
"""Deploy the reviewed warning-prefixed AP stderr classifier fix."""

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
    "framework/stage7/actual_pipeline_support_v2.py": {
        "old": "07b0b471e086b119be8ae07a65ae1b1a55632ce9bfe7f1ca62442ea186c448b5",
        "new": "0a55694accc4df4cc5969265da69581a8ff78d88aa25e86b97f38c13863753c9",
    }
}
deploy.TRANSACTION_SCHEMA = "stage7_ap_stderr_parser_fix_transaction_v1"
deploy.main()
