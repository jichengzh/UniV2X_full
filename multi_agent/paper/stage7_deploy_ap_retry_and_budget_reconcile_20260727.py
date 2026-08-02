#!/usr/bin/env python3
"""Deploy reviewed AP retry evidence and scheduler-budget reconciliation."""

from __future__ import annotations

import os

import stage7_deploy_replicate_dependency_pins_20260727 as deploy


deploy.TARGETS = {
    "framework/stage7/actual_pipeline_v2.py": {
        "old": "db5b355ecfa8ebc273f23a70841c148d1dbb931cfc9356e8dac25d9229bcc420",
        "new": "4d13e0343acb55c94822dfc51866752239c921fb6c6c2f35847b7eeae1ce66bc",
    },
    "scripts/stage7_orchestrator_scheduler_adapter_v2.py": {
        "old": "ac511d8ac4cdc030e25e7bb35327a5f7f8e751bc407c29ccf3136e4744f7b673",
        "new": "664fa48413ae0d28fe0f9c72691587b53c2e4751a61af923041570354df61229",
    },
}
deploy.TRANSACTION_SCHEMA = "stage7_ap_retry_budget_reconcile_transaction_v1"
os.environ.setdefault(
    "TRANSACTION_NAME",
    ".stage7_ap_retry_budget_reconcile_transaction",
)


if __name__ == "__main__":
    deploy.main()
