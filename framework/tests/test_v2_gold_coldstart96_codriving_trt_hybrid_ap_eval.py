from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_v2_gold_coldstart96_codriving_trt_hybrid_ap_eval as ap_eval  # noqa: E402


class V2GoldColdstart96CoDrivingTrtHybridApEvalTests(unittest.TestCase):
    def test_build_report_records_ap_and_evidence_paths(self) -> None:
        report = ap_eval.build_report(
            width="16x32x64",
            tag="trt_fp16",
            model_dir=Path("/tmp/model"),
            engine=Path("/tmp/model.engine"),
            ap30=0.6,
            ap50=0.5,
            ap70=0.4,
            n_done=10,
            n_trt_path=9,
            n_fallback_path=1,
            n_skipped=0,
            elapsed_secs=12.3,
        )

        self.assertEqual(report["width"], "16x32x64")
        self.assertEqual(report["tag"], "trt_fp16")
        self.assertEqual(report["ap70"], 0.4)
        self.assertEqual(report["n_trt_path"], 9)
        self.assertEqual(report["model_dir"], "/tmp/model")
        self.assertEqual(report["engine"], "/tmp/model.engine")


if __name__ == "__main__":
    unittest.main()
