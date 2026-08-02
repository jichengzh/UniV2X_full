from scripts.fcooper_remeasure_recovery_initialization_timing_v2 import (
    missing_widths,
)


def test_missing_widths_deduplicates_only_successful_nonbase_rows() -> None:
    missing = {
        "terminal_status": "measured_success_gold",
        "width": [32, 64, 64, 32, 64],
        "phase_timings_seconds": {},
    }
    measured = {
        **missing,
        "width": [64, 64, 64, 32, 64],
        "phase_timings_seconds": {"recovery_initialization_seconds": 1.0},
    }
    base = {
        **missing,
        "width": [64, 128, 256, 128, 256],
    }

    assert missing_widths(
        [{"rows": [missing, missing, measured, base]}]
    ) == [(32, 64, 64, 32, 64)]
