import unittest

from scripts.fcooper_native_scope_benchmark_v1 import (
    gpu_binding,
    summarize_measurements,
)


class FCooperNativeScopeBenchmarkTest(unittest.TestCase):
    def test_summary_uses_median_latency_and_mean_power(self) -> None:
        summary = summarize_measurements([3.0, 1.0, 2.0], [100.0, 110.0])

        self.assertEqual(summary["latency_ms"], 2.0)
        self.assertEqual(summary["power_w"], 105.0)
        self.assertAlmostEqual(summary["energy_j"], 0.21)

    def test_summary_rejects_empty_measurements(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            summarize_measurements([], [100.0])

    def test_gpu_binding_records_physical_and_visible_indices(self) -> None:
        self.assertEqual(
            gpu_binding(7, "7"),
            {"gpu_abs": 7, "cuda_visible_devices": "7", "cuda_device_index": 0},
        )


if __name__ == "__main__":
    unittest.main()
