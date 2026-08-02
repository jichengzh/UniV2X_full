import unittest

from tools.orin_deploy.lane_c_backbone_finalize import build_comparison


class LaneCBackboneFinalizeTests(unittest.TestCase):
    @staticmethod
    def h800(median, watts):
        return {
            "lat_p50_ms": median,
            "lat_p90_ms": median,
            "lat_p99_ms": median,
            "lat_mean_ms": median,
            "watt_avg": watts,
            "warmup": 20,
            "iters": 300,
            "repeat": 5,
            "input_shape": [2, 64, 128, 256],
            "caliber": (
                "backbone-subnet, input_hw=128x256, batch=2; latency p50 "
                "warmup20/iters300/repeat5 CUDA-event; engine-compute noDataTransfers"
            ),
        }

    @staticmethod
    def orin(median, watts, engine_sha):
        return {
            "median_ms": median,
            "p90_ms": median,
            "p99_ms": median,
            "mean_ms": median,
            "engine_sha256": engine_sha,
            "scope": "pyramid_get_multiscale_feature_engine_compute_no_data_transfer",
            "protocol": {
                "agent_batch": 2,
                "warmup": 20,
                "iters": 300,
                "repeat": 5,
                "timing": "CUDA_event",
                "data_transfer_inside_timed_region": False,
            },
            "power_measurement": {"vin_sys_5v0_mean_w": watts},
        }

    def test_degraded_calibration_omits_cross_hardware_speedup(self):
        comparison = build_comparison(
            calibration_status="calibration_not_identical",
            h800_fp16=self.h800(0.45, 300.0),
            h800_int8=self.h800(0.44, 295.0),
            orin_fp16=self.orin(10.5, 7.8, "f" * 64),
            orin_int8=self.orin(9.3, 7.4, "i" * 64),
            fp16_ap=None,
            int8_ap=None,
        )

        self.assertEqual(comparison["evidence_grade"], "degraded")
        self.assertNotIn("speedup", str(comparison).lower())
        self.assertEqual(
            comparison["latency"]["orin_int8"]["median_ms"],
            9.3,
        )

    def test_power_interfaces_remain_separate(self):
        fp16_ap = self.ap_report("fp16", "f" * 64, 0.78, 0.62)
        int8_ap = self.ap_report("int8", "i" * 64, 0.70, 0.50)
        comparison = build_comparison(
            calibration_status="identical",
            h800_fp16=self.h800(0.45, 300.0),
            h800_int8=self.h800(0.44, 295.0),
            orin_fp16=self.orin(10.5, 7.8, "f" * 64),
            orin_int8=self.orin(9.3, 7.4, "i" * 64),
            fp16_ap=fp16_ap,
            int8_ap=int8_ap,
            execution_audit=self.execution_audit(fp16_ap, int8_ap),
        )

        self.assertEqual(
            comparison["power"]["h800"]["measurement_interface"],
            "NVML board power",
        )
        self.assertEqual(
            comparison["power"]["orin"]["measurement_interface"],
            "tegrastats VIN_SYS_5V0 rail",
        )
        self.assertEqual(comparison["ap"]["delta_int8_minus_fp16"]["ap50"], -0.08)

    @staticmethod
    def ap_report(precision, engine_sha, ap50, ap70):
        return {
            "schema": "stage3_trt_multiscale_ap_bridge_v3",
            "status": "success",
            "precision_tag": precision,
            "checkpoint_sha256": (
                "d08fb16e778c6701aef7172e8d9609f1e4d73f20c419ee27bb8f158acc24a279"
            ),
            "engine_sha256": engine_sha,
            "dataset_split_sha256": "v" * 64,
            "processed_samples": 1789,
            "failed_samples": 0,
            "fallback_samples": 0,
            "full_ap_min_samples": 1789,
            "engine_ap_claim": True,
            "protocol": {
                "bridge": "stage3_trt_multiscale_ap_bridge_v3",
                "engine_batch": 2,
                "full_ap_min_samples": 1789,
                "input_route": (
                    "HEAL spatial_features -> TensorRT multiscale outputs -> "
                    "PyTorch fusion/head/postprocess"
                ),
            },
            "ap50": ap50,
            "ap70": ap70,
            "_report_sha256": precision[0] * 64,
        }

    @staticmethod
    def execution_audit(fp16_ap, int8_ap):
        return {
            "schema": "lane_c_direct_orin_ap_execution_audit_v1",
            "execution_mode": "direct_on_orin_same_process",
            "execution_host": {
                "address": "orin-lab",
                "hostname": "orin-lab",
                "architecture": "aarch64",
            },
            "replacement_scope": "get_multiscale_feature_only",
            "engine_file_role": "local_orin_execution",
            "backbone_execution_location": "orin-lab_orin_tensorrt",
            "downstream_execution_location": "orin-lab_same_process_pytorch",
            "rpc_used": False,
            "ap_reports": {
                "fp16": {
                    "report_sha256": fp16_ap["_report_sha256"],
                    "engine_sha256": fp16_ap["engine_sha256"],
                },
                "int8": {
                    "report_sha256": int8_ap["_report_sha256"],
                    "engine_sha256": int8_ap["engine_sha256"],
                },
            },
            "_audit_sha256": "a" * 64,
        }

    def test_rejects_malformed_latency_protocol(self):
        invalid = self.orin(10.5, 7.8, "f" * 64)
        invalid["protocol"]["warmup"] = 200
        with self.assertRaisesRegex(ValueError, "warmup"):
            build_comparison(
                calibration_status="identical",
                h800_fp16=self.h800(0.45, 300.0),
                h800_int8=self.h800(0.44, 295.0),
                orin_fp16=invalid,
                orin_int8=self.orin(9.3, 7.4, "i" * 64),
                fp16_ap=None,
                int8_ap=None,
            )

    def test_rejects_ap_with_wrong_engine(self):
        with self.assertRaisesRegex(ValueError, "engine_sha256"):
            build_comparison(
                calibration_status="identical",
                h800_fp16=self.h800(0.45, 300.0),
                h800_int8=self.h800(0.44, 295.0),
                orin_fp16=self.orin(10.5, 7.8, "f" * 64),
                orin_int8=self.orin(9.3, 7.4, "i" * 64),
                fp16_ap=self.ap_report("fp16", "x" * 64, 0.78, 0.62),
                int8_ap=self.ap_report("int8", "i" * 64, 0.70, 0.50),
                execution_audit=self.execution_audit(
                    self.ap_report("fp16", "x" * 64, 0.78, 0.62),
                    self.ap_report("int8", "i" * 64, 0.70, 0.50),
                ),
            )

    def test_rejects_ap_without_direct_orin_execution_audit(self):
        with self.assertRaisesRegex(ValueError, "execution audit"):
            build_comparison(
                calibration_status="identical",
                h800_fp16=self.h800(0.45, 300.0),
                h800_int8=self.h800(0.44, 295.0),
                orin_fp16=self.orin(10.5, 7.8, "f" * 64),
                orin_int8=self.orin(9.3, 7.4, "i" * 64),
                fp16_ap=self.ap_report("fp16", "f" * 64, 0.78, 0.62),
                int8_ap=self.ap_report("int8", "i" * 64, 0.70, 0.50),
            )


if __name__ == "__main__":
    unittest.main()
