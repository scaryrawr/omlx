# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen4-Exp multimodal admission in the mlx-vlm load path."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

from omlx.engine import vlm as vlm_module
from omlx.engine.vlm import VLMBatchedEngine, _force_qwen4_exp_sanitize_on_load
from omlx.exceptions import InvalidRequestError
from omlx.model_settings import ModelSettings
from omlx.utils.model_loading import maybe_apply_pre_load_patches


def test_qwen4_exp_runtime_rejects_audio_only():
    engine = VLMBatchedEngine("qwen4")
    engine._vlm_model = SimpleNamespace(
        config=SimpleNamespace(model_type=vlm_module.QWEN4_EXP_MODEL_TYPE)
    )

    with pytest.raises(InvalidRequestError, match="not audio"):
        engine._prepare_vision_inputs(
            [{"role": "user", "content": "hello"}],
            images=[],
            audio=[("samples", 16000)],
        )


def test_qwen4_exp_mlx_metadata_is_hidden_during_load(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp"}), encoding="utf-8"
    )
    weight_file = tmp_path / "model.safetensors"
    weight_file.touch()

    class FakeHandle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def metadata(self):
            return {"format": "mlx", "source": "test"}

    import safetensors

    original = lambda *_args, **_kwargs: FakeHandle()
    monkeypatch.setattr(safetensors, "safe_open", original)

    with vlm_module._force_qwen4_exp_sanitize_on_load(tmp_path):
        with safetensors.safe_open(weight_file) as handle:
            assert handle.metadata() == {"source": "test"}

    assert safetensors.safe_open is original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_type", "expected_lazy"),
    [("qwen4_exp", True), ("qwen2_vl", None)],
)
async def test_only_qwen4_exp_loader_defers_parameter_eval_to_materialize(
    tmp_path, monkeypatch, model_type, expected_lazy
):
    import mlx_vlm.utils as vlm_utils

    from omlx.utils import model_loading

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": model_type}), encoding="utf-8"
    )
    captured = {}

    def stop_after_load(model_name, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after load")

    monkeypatch.setattr(vlm_utils, "load", stop_after_load)
    monkeypatch.setattr(vlm_module, "_patch_video_processor_bug", lambda: None)
    monkeypatch.setattr(vlm_module, "_patch_torch_free_image_processor", lambda: None)
    monkeypatch.setattr(vlm_module, "apply_pixtral_torch_free_patch", lambda: None)
    monkeypatch.setattr(
        model_loading, "maybe_apply_pre_load_patches", lambda *a, **k: None
    )
    monkeypatch.setattr(
        model_loading, "maybe_load_custom_quantization", lambda *a, **k: None
    )

    with pytest.raises(RuntimeError, match="stop after load"):
        await VLMBatchedEngine(model_name=str(tmp_path)).start()

    if expected_lazy is None:
        assert "lazy" not in captured
    else:
        assert captured["lazy"] is expected_lazy


def test_qwen4_exp_loader_defaults_to_depth_three_without_sidecar_contract(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "mtp_num_hidden_layers": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.fc_hidden.weight": "model.safetensors"}}),
        encoding="utf-8",
    )
    settings = SimpleNamespace(mtp_enabled=True, mtp_num_draft_tokens=None)

    maybe_apply_pre_load_patches(str(tmp_path), settings, for_vlm=True)

    from mlx_vlm.models.qwen4_exp.language import get_mtp_runtime

    from omlx.patches.mlx_lm_mtp import get_mtp_depth, is_mtp_active

    assert get_mtp_runtime().enabled is True
    assert get_mtp_runtime().checkpoint_prefix == "mtp."
    assert get_mtp_depth() == 3
    assert is_mtp_active() is True

    maybe_apply_pre_load_patches(
        str(tmp_path),
        SimpleNamespace(mtp_enabled=False),
        for_vlm=True,
    )
    assert get_mtp_runtime().enabled is False
    assert is_mtp_active() is False


def test_qwen4_exp_loader_detects_and_loads_standalone_mtp(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "mtp_num_hidden_layers": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    parent_weights = tmp_path / "model.safetensors"
    parent_weights.touch()
    mtp_dir = tmp_path / "mtp"
    mtp_dir.mkdir()
    (mtp_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp_mtp", "block_size": 2}),
        encoding="utf-8",
    )
    mtp_weights = mtp_dir / "model.safetensors"
    mtp_weights.touch()

    import mlx_vlm.utils as vlm_utils

    def fake_load_safetensors(path):
        if str(path).endswith("/mtp/model.safetensors"):
            return {"fc_embedding.weight": "mtp-weight"}
        return {"language_model.lm_head.weight": "parent-weight"}

    monkeypatch.setattr(vlm_utils, "_load_safetensors", fake_load_safetensors)
    monkeypatch.setattr(
        "omlx.utils.model_loading._checkpoint_weight_prefix",
        lambda path, prefixes: (
            "fc_embedding."
            if path == mtp_dir and "fc_embedding." in prefixes
            else None
        ),
    )
    settings = SimpleNamespace(mtp_enabled=True, mtp_num_draft_tokens=None)

    maybe_apply_pre_load_patches(str(tmp_path), settings, for_vlm=True)

    from mlx_vlm.models.qwen4_exp.language import get_mtp_runtime

    from omlx.patches.mlx_lm_mtp import get_mtp_depth

    assert get_mtp_runtime().checkpoint_prefix == "mtp/"
    assert get_mtp_depth() == 1

    maybe_apply_pre_load_patches(
        str(tmp_path),
        ModelSettings(mtp_enabled=True, mtp_num_draft_tokens=3),
        for_vlm=True,
    )
    assert get_mtp_depth() == 3

    with _force_qwen4_exp_sanitize_on_load(tmp_path):
        loaded = vlm_utils._load_safetensors(str(parent_weights))

    assert loaded == {
        "language_model.lm_head.weight": "parent-weight",
        "mtp.fc_embedding.weight": "mtp-weight",
    }


@pytest.mark.parametrize(
    ("sidecar_content", "warning"),
    [
        ("{", None),
        ("[]", "must contain an object"),
        (
            json.dumps({"model_type": "qwen4_exp_mtp", "block_size": "2"}),
            "invalid block_size",
        ),
    ],
)
def test_qwen4_exp_loader_safely_falls_back_for_invalid_sidecar_contract(
    tmp_path, caplog, sidecar_content, warning
):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "mtp_num_hidden_layers": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.fc_hidden.weight": "model.safetensors"}}),
        encoding="utf-8",
    )
    mtp_dir = tmp_path / "mtp"
    mtp_dir.mkdir()
    (mtp_dir / "config.json").write_text(sidecar_content, encoding="utf-8")

    maybe_apply_pre_load_patches(
        str(tmp_path),
        ModelSettings(mtp_enabled=True),
        for_vlm=True,
    )

    from omlx.patches.mlx_lm_mtp import get_mtp_depth

    assert get_mtp_depth() == 3
    if warning is not None:
        assert warning in caplog.text


def test_qwen4_exp_loader_uses_explicit_ple_ssd_offload_setting(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp"}), encoding="utf-8"
    )

    maybe_apply_pre_load_patches(
        str(tmp_path),
        SimpleNamespace(mtp_enabled=False, qwen4_ple_ssd_offload=False),
        for_vlm=True,
    )
    from mlx_vlm.models.qwen4_exp.language import get_ple_runtime_mode

    assert get_ple_runtime_mode() == "resident"

    maybe_apply_pre_load_patches(
        str(tmp_path),
        SimpleNamespace(mtp_enabled=False, qwen4_ple_ssd_offload=True),
        for_vlm=True,
    )
    assert get_ple_runtime_mode() == "mmap"
