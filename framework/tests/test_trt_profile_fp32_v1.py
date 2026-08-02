from __future__ import annotations

import ast
import unittest
from pathlib import Path


class TrtProfileFp32V1Tests(unittest.TestCase):
    def test_cli_and_builder_accept_fp32_without_precision_flags(self) -> None:
        source = Path("framework/trt_baseline/trt_profile_v1.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        choices = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "choices" and isinstance(keyword.value, (ast.List, ast.Tuple)):
                        choices.extend(
                            item.value for item in keyword.value.elts if isinstance(item, ast.Constant)
                        )
        self.assertIn("fp32", choices)
        self.assertIn('if precision == "fp32":', source)
        self.assertIn('"builder_flags": builder_flags_for_precision(precision)', source)


if __name__ == "__main__":
    unittest.main()
