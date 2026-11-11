# SPDX-License-Identifier: Apache-2.0
"""DFlash/DSpark target allow-list in omlx.utils.model_loading."""

import pytest

from omlx.utils.model_loading import dflash_batched_supported, dflash_compatibility


@pytest.mark.parametrize(
    ("model_type", "engine_type"),
    [
        ("qwen3_5_moe", "vlm"),
        ("qwen4_exp", "vlm"),
        ("minicpmv4_7", "vlm"),
        ("qwen3_moe", "batched"),
    ],
)
def test_supported_targets(model_type, engine_type):
    assert dflash_batched_supported(model_type, engine_type)
    assert dflash_compatibility(model_type, engine_type) == (True, "")


def test_minicpm_v47_is_a_vlm_target_only():
    # MiniCPM-V 4.7 loads on the VLM engine (its text backbone is a Qwen3.5 MoE
    # behind a minicpmv4_7 wrapper); it is not a text-engine DFlash target.
    assert not dflash_batched_supported("minicpmv4_7", "batched")


def test_qwen4_is_a_vlm_target_only():
    assert not dflash_batched_supported("qwen4_exp", "batched")
    assert not dflash_batched_supported("qwen4_exp_text", "batched")
    assert not dflash_batched_supported("qwen4_exp_text", "vlm")


@pytest.mark.parametrize(
    ("model_type", "engine_type"),
    [("llama", "vlm"), (None, "vlm"), ("minicpmv4_7", None)],
)
def test_unsupported_targets_report_their_model_type(model_type, engine_type):
    supported, reason = dflash_compatibility(model_type, engine_type)
    assert not supported
    assert f"model_type='{model_type or ''}'" in reason
    assert "MiniCPM-V 4.7" in reason


def test_dflash2_attachment_requests_the_hc_mean_capture():
    """Qwen3.8-Flash-Next-DFlash2 was trained on the target's hc-stream mean;
    v1/Laguna/Muse drafters keep the mixer representation."""
    from types import SimpleNamespace

    from omlx.speculative.dflash import attach_drafter, detach_drafter

    language_model = SimpleNamespace(model=SimpleNamespace())
    drafter = SimpleNamespace(kind="dflash2", depth=3, clear=lambda: None)
    attach_drafter(language_model, drafter)
    assert language_model._omlx_dflash_capture_mean is True
    assert language_model.model._omlx_dflash_capture_mean is True
    detach_drafter(language_model)
    assert language_model.model._omlx_dflash_capture_mean is False

    language_model = SimpleNamespace(model=SimpleNamespace())
    drafter = SimpleNamespace(kind="dflash", depth=3, clear=lambda: None)
    attach_drafter(language_model, drafter)
    assert language_model._omlx_dflash_capture_mean is False
