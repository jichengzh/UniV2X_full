#!/usr/bin/env python3
"""Deploy the reviewed Stage7 adapter budget-baseline reconciliation."""

from __future__ import annotations

import os

import stage7_deploy_replicate_dependency_pins_20260727 as deploy


deploy.TARGETS = {
    "scripts/stage7_orchestrator_scheduler_adapter_v2.py": {
        "old": "ac511d8ac4cdc030e25e7bb35327a5f7f8e751bc407c29ccf3136e4744f7b673",
        "new": "664fa48413ae0d28fe0f9c72691587b53c2e4751a61af923041570354df61229",
    }
}
deploy.TRANSACTION_SCHEMA = "stage7_adapter_budget_reconcile_transaction_v1"
os.environ.setdefault(
    "TRANSACTION_NAME",
    ".stage7_adapter_budget_reconcile_transaction",
)


if __name__ == "__main__":
    deploy.main()
