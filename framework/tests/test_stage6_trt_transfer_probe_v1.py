from __future__ import annotations

import ast
import unittest
from pathlib import Path


class Stage6TrtTransferProbeV1Tests(unittest.TestCase):
    def test_probe_forbids_cache_miss_profiling(self) -> None:
        source = Path("scripts/stage6_trt_timing_cache_transfer_probe_v1.py").read_text(
            encoding="utf-8"
        )
        ast.parse(source)
        self.assertIn("BuilderFlag.ERROR_ON_TIMING_CACHE_MISS", source)
        self.assertIn('"compressed_shape_retuned": False', source)
        self.assertIn('"fallback_used": False', source)
        self.assertIn('"q_dispatch_reached": False', source)
        self.assertIn('"blocked_stage": "frozen_base_policy_transfer"', source)
        self.assertIn("--cache-in", source)


if __name__ == "__main__":
    unittest.main()
