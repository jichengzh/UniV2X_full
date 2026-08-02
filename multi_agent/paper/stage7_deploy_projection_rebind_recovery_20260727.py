#!/usr/bin/env python3
"""Deploy the reviewed Stage7 projection-SHA recovery classification fix."""

from __future__ import annotations

import os

import stage7_deploy_replicate_dependency_pins_20260727 as deploy


deploy.TARGETS = {
    "scripts/stage7_execute_actual_v3_misses_v2.sh": {
        "old": "0f87b1f851e8e9625c0818f45600bd28e189ec2bb596cef01b65b92e7594072c",
        "new": "73d4897773f3ac485025e550411fde640790dbf02acf29741257f4fe004e4157",
    },
}
deploy.TRANSACTION_SCHEMA = "stage7_projection_rebind_recovery_transaction_v1"
os.environ.setdefault(
    "TRANSACTION_NAME",
    ".stage7_projection_rebind_recovery_transaction",
)


if __name__ == "__main__":
    deploy.main()
