import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.fcooper_tvm_admit_original_v1 import admit_original


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AdmitOriginalTests(unittest.TestCase):
    def test_admits_only_native_gpu7_evidence_from_paper_ready_trt_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _write(root / "checkpoint.pth", b"checkpoint")
            config = _write(root / "config.yaml", b"config")
            evaluation = _write(root / "evaluation.yaml", b"evaluation")
            ap_report = _write(
                root / "ap.json",
                {
                    "status": "success_full",
                    "dataset_samples": 2170,
                    "ap30": 0.91,
                    "ap50": 0.82,
                    "ap70": 0.63,
                    "checkpoint_sha256": _sha(checkpoint),
                    "config_sha256": _sha(config),
                    "raw_report_path": str(evaluation),
                    "raw_report_sha256": _sha(evaluation),
                },
            )
            preflight = _write(
                root / "preflight.json",
                {
                    "passed": True,
                    "checkpoint_sha256": _sha(checkpoint),
                    "config_sha256": _sha(config),
                    "ap_reference_report_sha256": _sha(ap_report),
                },
            )
            repeat_paths = []
            repeat_shas = []
            for index, (latency, energy) in enumerate(
                ((11.7, 6.8), (11.8, 6.7), (11.9, 6.6))
            ):
                repeat = _write(
                    root / f"repeat_{index}.json",
                    {
                        "backend": "pytorch_cuda_cudnn",
                        "gpu_abs": 7,
                        "optimized_scope": "post_scatter_backbone_shrinker",
                        "precision": "fp32",
                        "checkpoint_sha256": _sha(checkpoint),
                        "config_sha256": _sha(config),
                        "latency_ms": latency,
                        "energy_j": energy,
                    },
                )
                repeat_paths.append(str(repeat))
                repeat_shas.append(_sha(repeat))
            audit = _write(
                root / "trt_audit.json",
                {
                    "paper_ready": True,
                    "backend": "trt",
                    "performance_gpu_index": 7,
                    "rows": [
                        {
                            "method": "Original/default",
                            "terminal_status": "measured_success_gold",
                            "configuration": "(64,128,256,128,256,fp32)",
                            "ap30": 0.91,
                            "ap50": 0.82,
                            "ap70": 0.63,
                            "latency_ms": 11.8,
                            "energy_j": 6.7,
                        }
                    ],
                    "selected_evidence": {
                        "Original/default": {
                            "row_id": "fcooper-original-default",
                            "checkpoint_sha256": _sha(checkpoint),
                            "ap_report_sha256": _sha(ap_report),
                            "repeat_paths": repeat_paths,
                            "repeat_sha256": repeat_shas,
                        }
                    },
                },
            )

            outputs = admit_original(
                trt_audit_path=audit,
                preflight_path=preflight,
                checkpoint_path=checkpoint,
                config_path=config,
                ap_report_path=ap_report,
                output_dir=root / "out",
            )

            contract = json.loads(outputs["original_contract"].read_text())
            validation = json.loads(outputs["validation_manifest"].read_text())
            self.assertTrue(contract["same_scope_sha_admission"])
            self.assertEqual(
                contract["row"]["backend"], "pytorch_cuda_cudnn"
            )
            self.assertEqual(len(validation["performance_repeats"]), 3)
            self.assertEqual(validation["gpu_index"], 7)
            self.assertNotIn("engine", json.dumps(contract).lower())


if __name__ == "__main__":
    unittest.main()
