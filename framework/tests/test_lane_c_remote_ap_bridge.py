from __future__ import annotations

import unittest

from tools.orin_deploy.lane_c_remote_ap_bridge import augment_report


class LaneCRemoteApBridgeTest(unittest.TestCase):
    def test_augment_report_marks_orin_rpc_execution_boundary(self) -> None:
        report = augment_report(
            {"engine_ap_claim": True},
            rpc_audit={"call_count": 1789},
            orin_host="orin-lab",
        )
        self.assertEqual(report["backbone_execution_location"], "orin-lab")
        self.assertEqual(
            report["replacement_scope"], "get_multiscale_feature_only"
        )
        self.assertEqual(
            report["local_engine_file_role"], "sha_anchor_not_local_execution"
        )
        self.assertEqual(report["orin_rpc_client_audit"]["call_count"], 1789)


if __name__ == "__main__":
    unittest.main()
