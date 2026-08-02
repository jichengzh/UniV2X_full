#!/usr/bin/env python3
"""Deploy the reviewed Stage7 exact-GPU cooldown recovery fix."""

from __future__ import annotations

import os

import stage7_deploy_replicate_dependency_pins_20260727 as deploy


deploy.TARGETS = {
    "scripts/stage7_core_ablation_scheduler_v2.py": {
        "old": "aa8bd0191db2873ec48d62c372a5a29a6b5eab7aea2cc211a367d96c58523d20",
        "new": "229a558f29dfa4f33a8452b7d45f5ac97b2e2b6d63310a81e490bd463ded0504",
    },
}
deploy.TRANSACTION_SCHEMA = "stage7_exact_gpu_cooldown_transaction_v1"
os.environ.setdefault(
    "TRANSACTION_NAME",
    ".stage7_exact_gpu_cooldown_transaction",
)


if __name__ == "__main__":
    deploy.main()
