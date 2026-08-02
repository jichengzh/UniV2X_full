import json
from pathlib import Path

from scripts.fcooper_tvm_gpu7_validation_v1 import (
    _tvm_environment,
    build_ap_command,
    feedback_with_resolved_source,
    wait_for_exclusive_gpu,
)


def test_build_fp16_ap_command_includes_prediction_output(tmp_path: Path) -> None:
    feedback = {
        "q_mode": "fp16",
        "tvm_artifact_path": str(tmp_path / "model.so"),
        "source_contract": {
            "config_path": str(tmp_path / "config.yaml"),
            "checkpoint_path": str(tmp_path / "checkpoint.pth"),
        },
        "width": [32, 64, 128, 64, 160],
    }

    command = build_ap_command(
        feedback,
        output_dir=tmp_path / "validation",
        python=Path("/python"),
        tvm_python=Path("/tvm-python"),
        tvm_site=Path("/tvm-site"),
        tvm_lib_dirs=[Path("/tvm-lib")],
    )

    assert "fcooper_tvm_fp16_ap_bridge_v1.py" in command[1]
    assert command[command.index("--output-shape") + 1] == "5,160,256,256"
    assert "--prediction-path" in command


def test_build_int8_ap_command_binds_existing_sanity(tmp_path: Path) -> None:
    execution = tmp_path / "execution"
    feedback_path = execution / "feedback_row.json"
    feedback = {
        "q_mode": "int8",
        "tvm_artifact_path": str(execution / "route/row_int8/model.vmexec"),
        "quant_contract_path": str(execution / "quant/quant_contract.json"),
        "source_contract": {
            "config_path": str(tmp_path / "config.yaml"),
            "checkpoint_path": str(tmp_path / "checkpoint.pth"),
        },
        "width": [32, 64, 128, 64, 160],
    }
    feedback_path.parent.mkdir(parents=True)
    feedback_path.write_text(json.dumps(feedback))
    sanity = execution / "numeric_sanity.json"
    sanity.write_text("{}")

    command = build_ap_command(
        feedback,
        output_dir=tmp_path / "validation",
        python=Path("/python"),
        tvm_python=Path("/tvm-python"),
        tvm_site=Path("/tvm-site"),
        tvm_lib_dirs=[Path("/tvm-lib")],
        execution_dir=execution,
    )

    assert "fcooper_tvm_int8_ap_bridge_v1.py" in command[1]
    assert command[command.index("--mode") + 1] == "full"
    assert command[command.index("--sanity-report") + 1] == str(sanity)


def test_tvm_environment_excludes_python39_site_packages() -> None:
    environment = _tvm_environment(
        physical_gpu=7,
        tvm_site=Path("/tvm/lib/python3.10/site-packages"),
        tvm_lib_dirs=[Path("/tvm/lib")],
        base_environment={"PYTHONPATH": "/old/python3.9/site-packages"},
    )

    assert "/old/python3.9/site-packages" not in environment["PYTHONPATH"]
    assert environment["PYTHONPATH"].split(":") == [
        "/home/jichengzhi/V2X",
        "/tvm/lib/python3.10/site-packages",
    ]


def test_ap_feedback_uses_resolved_recovered_checkpoint(tmp_path: Path) -> None:
    execution = tmp_path / "execution"
    checkpoint = execution / "recovered_checkpoint.pth"
    config = execution / "config.yaml"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("config")
    provenance = {
        "passed": True,
        "resolved_source_contract": {
            "checkpoint_path": str(checkpoint),
            "config_path": str(config),
        },
    }
    (execution / "source_reuse_audit.json").write_text(json.dumps(provenance))
    feedback = {
        "source_contract": {
            "checkpoint_path": str(tmp_path / "stale.pth"),
            "config_path": str(tmp_path / "stale.yaml"),
        }
    }

    resolved = feedback_with_resolved_source(feedback, execution_dir=execution)

    assert resolved["source_contract"]["checkpoint_path"] == str(checkpoint)
    assert resolved["source_contract"]["config_path"] == str(config)


def test_wait_for_exclusive_gpu_records_bounded_wait() -> None:
    states = [[101], [101], []]
    clock = [100.0]

    def probe(_gpu):
        return states.pop(0)

    def sleep(seconds):
        clock[0] += seconds

    audit = wait_for_exclusive_gpu(
        7,
        timeout_seconds=30,
        poll_interval_seconds=5,
        process_probe=probe,
        monotonic=lambda: clock[0],
        sleep=sleep,
    )

    assert audit == {
        "physical_gpu_id": 7,
        "checks": 3,
        "waited_seconds": 10.0,
        "exclusive": True,
    }


def test_wait_for_exclusive_gpu_times_out_without_creating_evidence() -> None:
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    try:
        wait_for_exclusive_gpu(
            7,
            timeout_seconds=10,
            poll_interval_seconds=5,
            process_probe=lambda _gpu: [101],
            monotonic=lambda: clock[0],
            sleep=sleep,
        )
    except TimeoutError as exc:
        assert "GPU7" in str(exc)
    else:
        raise AssertionError("expected bounded exclusivity timeout")
