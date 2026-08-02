from __future__ import annotations

import ast
import unittest
from pathlib import Path


class Stage6Int8DispatchGuardV1Tests(unittest.TestCase):
    def test_measure_config_cannot_silently_route_int8_to_legacy_dp4a(self) -> None:
        source = Path("framework/measure_config.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('precision == "int8_legacy_dp4a"', source)
        self.assertIn('precision == "int8"', source)
        self.assertIn("stage5_route_b_int8_auto_decomp", source)
        self.assertIn('"int8_legacy_dp4a"', source)


if __name__ == "__main__":
    unittest.main()
