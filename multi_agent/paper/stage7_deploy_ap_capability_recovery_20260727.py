#!/usr/bin/env python3
"""Deploy the reviewed Stage7 AP-capability recovery patch transactionally."""

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
        "old": "c9d896704123f668c5f10ca81ddde11fb720d509d67fa6266e32fe51cfb30fd3",
        "new": "9f27b9a3b3c398dc019e3dcb3a7235ad7fadaea50b7d6e97719c01e26c9a6060",
    },
    "framework/stage7/actual_pipeline_support_v2.py": {
        "old": "a4e75ea7bfc2e82092c7fe6770d6fb02b503f84651d08ec2f4a7ad3e19f28213",
        "new": "b52999801a6335b99a0f7bc21a069c30deb4cb07777af68c7a3f2d4a10d50fed",
    },
    "framework/stage7/physical_feedback_v2.py": {
        "old": "023019fe5fc2f8ad19711989f64fa64841709815e40fb691d409df8220cb71d2",
        "new": "c8e6186489add7559072ff414ff20f8c7fe6d77c5285f3e6fe6526e5b24994d4",
    },
}
DEPLOY.TRANSACTION_SCHEMA = "stage7_ap_capability_recovery_transaction_v1"


if __name__ == "__main__":
    DEPLOY.main()
