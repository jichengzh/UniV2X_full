from __future__ import annotations

from types import SimpleNamespace

import pytest

from framework.stage1.adapters import available_models, get_adapter
from framework.stage1.run_scan import apply_adapter_overrides


def test_adapter_overrides_are_copy_on_write(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.pth"
    config.write_text("name: fcooper\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    adapter = SimpleNamespace(
        name="fcooper",
        config_path="/old/config.yaml",
        ckpt_path="/old/model.pth",
        ckpt_status="ok",
        trace_plan={"stale": True},
    )

    overridden = apply_adapter_overrides(
        adapter,
        config_path=config,
        checkpoint_path=checkpoint,
    )

    assert overridden is not adapter
    assert adapter.config_path == "/old/config.yaml"
    assert overridden.config_path == str(config.resolve())
    assert overridden.ckpt_path == str(checkpoint.resolve())
    assert overridden.ckpt_status == "ok"
    assert overridden.trace_plan is None


def test_adapter_override_rejects_missing_inputs(tmp_path) -> None:
    adapter = SimpleNamespace(
        name="fcooper",
        config_path="/old/config.yaml",
        ckpt_path="/old/model.pth",
        ckpt_status="ok",
        trace_plan=None,
    )

    with pytest.raises(FileNotFoundError):
        apply_adapter_overrides(
            adapter,
            config_path=tmp_path / "missing.yaml",
            checkpoint_path=None,
        )


def test_push_button_registry_exposes_fcooper() -> None:
    assert "fcooper" in available_models()
    assert get_adapter("fcooper").name == "fcooper"
