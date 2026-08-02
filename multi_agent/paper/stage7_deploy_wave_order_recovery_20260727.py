#!/usr/bin/env python3
"""Deploy the reviewed Stage7 isolated-wave ordering recovery patch."""

from __future__ import annotations

import importlib.util
from pathlib import Path


BASE = Path(__file__).with_name(
    "stage7_deploy_replicate_dependency_pins_20260727.py"
)
SPEC = importlib.util.spec_from_file_location("stage7_transactional_deploy", BASE)
if SPEC is None or SPEC.loader is None:
    raise SystemExit("transactional deployment helper is unavailable")
DEPLOY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEPLOY)

DEPLOY.TARGETS = {
    "framework/stage7/actual_pipeline_v2.py": {
        "old": "9f27b9a3b3c398dc019e3dcb3a7235ad7fadaea50b7d6e97719c01e26c9a6060",
        "new": "8fecb701e462662aa966745acd85acc6aa5605d15b41bb7349e1b0b1e2a33597",
    },
    "framework/stage7/actual_pipeline_support_v2.py": {
        "old": "b52999801a6335b99a0f7bc21a069c30deb4cb07777af68c7a3f2d4a10d50fed",
        "new": "07b0b471e086b119be8ae07a65ae1b1a55632ce9bfe7f1ca62442ea186c448b5",
    },
}
DEPLOY.TRANSACTION_SCHEMA = "stage7_wave_and_process_probe_recovery_transaction_v1"


if __name__ == "__main__":
    DEPLOY.main()
