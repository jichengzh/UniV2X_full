from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from framework.stage5.coldstart_validation_source_v1 import (
    quant_contract_from_gold_result,
)


class Stage5ColdstartValidationSourceV1Tests(unittest.TestCase):
    def test_recovers_exact_quant_contract_from_gold_performance_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            quant = root / "percentile_quant.json"
            quant.write_text('{"schema":"quant"}')
            digest = hashlib.sha256(quant.read_bytes()).hexdigest()
            result = root / "performance.json"
            result.write_text(
                json.dumps(
                    {
                        "tensor_quant_params_path": str(quant),
                        "tensor_quant_params_sha256": digest,
                    }
                )
            )

            resolved = quant_contract_from_gold_result(
                {"performance_result_json": str(result)}
            )

        self.assertEqual(resolved, quant)

    def test_rejects_quant_contract_sha_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            quant = root / "quant.json"
            quant.write_text("{}")
            result = root / "performance.json"
            result.write_text(
                json.dumps(
                    {
                        "tensor_quant_params_path": str(quant),
                        "tensor_quant_params_sha256": "a" * 64,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                quant_contract_from_gold_result(
                    {"performance_result_json": str(result)}
                )


if __name__ == "__main__":
    unittest.main()
