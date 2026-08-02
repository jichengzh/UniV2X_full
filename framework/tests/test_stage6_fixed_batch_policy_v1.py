from scripts.stage5_build_performance_plan_v2 import apply_tvm_fp16_trial_policy


def test_apply_tvm_fp16_trial_policy_rewrites_only_tvm_fp16() -> None:
    jobs = [
        {"runner_key": "tvm_fp16", "command": ["python3", "runner.py", "--max-trials", "64"]},
        {"runner_key": "tvm_int8", "command": ["python3", "int8.py"]},
        {"runner_key": "trt_fp16", "command": ["python3", "trt.py"]},
    ]
    updated = apply_tvm_fp16_trial_policy(jobs, 0)
    assert updated[0]["command"][-1] == "0"
    assert updated[0]["stage6_tvm_fp16_max_trials"] == 0
    assert updated[1] == jobs[1]
    assert updated[2] == jobs[2]
    assert jobs[0]["command"][-1] == "64"
