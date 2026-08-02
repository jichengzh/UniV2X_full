from __future__ import annotations

import unittest

from scripts.stage5_advance_round_v1 import canonical_sha256, validate_previous_request


class Stage5AdvanceRoundV1Tests(unittest.TestCase):
    def test_previous_state_must_bind_exact_measurement_request(self) -> None:
        request = {"schema_version": "stage5_measurement_request_v1", "rows": [{"id": 1}]}
        previous = {
            "status": "awaiting_real_measurement",
            "measurement_request_sha256": canonical_sha256(request),
        }
        validate_previous_request(previous, request)

        drifted = {**request, "rows": [{"id": 2}]}
        with self.assertRaisesRegex(ValueError, "previous measurement request SHA mismatch"):
            validate_previous_request(previous, drifted)

        with self.assertRaisesRegex(ValueError, "previous measurement request SHA missing"):
            validate_previous_request({"status": "awaiting_real_measurement"}, request)


if __name__ == "__main__":
    unittest.main()
