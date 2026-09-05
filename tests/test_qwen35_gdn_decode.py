# SPDX-License-Identifier: Apache-2.0
"""Parity coverage for the fused Qwen3.5-family GDN decode convolution."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches.qwen35_gdn_decode import (
    apply_qwen35_gdn_decode_patch,
    fused_conv_silu,
)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("batch", [1, 3])
def test_fused_decode_conv_is_bit_exact(batch):
    mx.random.seed(23)
    channels = 256
    state = (mx.random.normal((batch, 3, channels)) * 0.5).astype(mx.bfloat16)
    x = (mx.random.normal((batch, 1, channels)) * 0.5).astype(mx.bfloat16)
    weight = (mx.random.normal((channels, 4)) * 0.2).astype(mx.bfloat16)
    conv = nn.Conv1d(channels, channels, 4, groups=channels, bias=False)
    conv.weight = weight[:, :, None]

    conv_input = mx.concatenate([state, x], axis=1)
    expected_state = mx.contiguous(conv_input[:, -3:, :])
    expected_output = nn.silu(conv(conv_input))
    actual_state, actual_output = fused_conv_silu(state, x, weight)
    mx.eval(expected_state, expected_output, actual_state, actual_output)

    assert mx.array_equal(actual_state, expected_state).item()
    assert mx.array_equal(actual_output, expected_output).item()


def _qwen35_config():
    from mlx_vlm.models.qwen3_5 import TextConfig

    return TextConfig(
        model_type="qwen3_5",
        hidden_size=32,
        intermediate_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_hidden_layers=2,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )


def _qwen4_config():
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import TextConfig

    return TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        hc_count=2,
        hc_lowrank=8,
        head_dim=8,
        layer_types=["linear_attention", "qwen_sparse_attention"],
    )


def _assert_vlm_gdn_call_matches_upstream(
    monkeypatch, family, shape, use_mask, conv_kernel_size
):
    from mlx_vlm.models.qwen3_5 import language as q35

    import omlx.patches.qwen35_gdn_decode as patch

    if family == "qwen3_5":
        cls = q35.Qwen3_5GatedDeltaNet
        config = _qwen35_config()
    else:
        from omlx.patches.mlx_vlm_qwen4_exp_compat import (
            apply_mlx_vlm_qwen4_exp_compat_patch,
        )

        apply_mlx_vlm_qwen4_exp_compat_patch()
        from mlx_vlm.models.qwen4_exp import language as qwen4

        cls = qwen4.Qwen4ExpGatedDeltaNet
        config = _qwen4_config()

    upstream_call = getattr(q35.Qwen3_5GatedDeltaNet.__call__, "_omlx_original", None)
    if upstream_call is None:
        upstream_call = q35.Qwen3_5GatedDeltaNet.__call__
    monkeypatch.setattr(q35.Qwen3_5GatedDeltaNet, "__call__", upstream_call)
    monkeypatch.setattr(patch, "_VLM_PATCHED", False)
    monkeypatch.setattr(patch, "_ENABLED", True)
    assert patch._patch_vlm()

    config.linear_conv_kernel_dim = conv_kernel_size
    mx.random.seed(31)
    module = cls(config)
    module.set_dtype(mx.bfloat16)
    inputs = mx.random.normal(shape).astype(mx.bfloat16)
    mask = None
    if use_mask:
        mask = mx.array(
            [[True, False, True] for _ in range(shape[0])],
            dtype=mx.bool_,
        )

    reference_cache = [None, None]
    patched_cache = [None, None]
    expected = upstream_call(module, inputs, mask=mask, cache=reference_cache)
    actual = module(inputs, mask=mask, cache=patched_cache)
    mx.eval(expected, actual, *reference_cache, *patched_cache)

    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(patched_cache[0], reference_cache[0]).item()
    assert mx.array_equal(patched_cache[1], reference_cache[1]).item()

    next_inputs = mx.random.normal((shape[0], 1, shape[-1])).astype(mx.bfloat16)
    expected = upstream_call(module, next_inputs, cache=reference_cache)
    actual = module(next_inputs, cache=patched_cache)
    mx.eval(expected, actual, *reference_cache, *patched_cache)

    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(patched_cache[0], reference_cache[0]).item()
    assert mx.array_equal(patched_cache[1], reference_cache[1]).item()


@pytest.mark.parametrize("family", ["qwen3_5", "qwen4_exp"])
@pytest.mark.parametrize("conv_kernel_size", [3, 4])
@pytest.mark.parametrize(
    ("shape", "use_mask"),
    [
        ((1, 1, 32), False),
        ((1, 3, 32), True),
        ((2, 3, 32), True),
    ],
)
def test_vlm_gdn_call_matches_current_upstream(
    monkeypatch, family, shape, use_mask, conv_kernel_size
):
    _assert_vlm_gdn_call_matches_upstream(
        monkeypatch, family, shape, use_mask, conv_kernel_size
    )


def test_patch_installs_for_lm_and_vlm():
    assert apply_qwen35_gdn_decode_patch()
    from mlx_lm.models.qwen3_5 import GatedDeltaNet
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

    assert getattr(GatedDeltaNet.__call__, "_omlx_fused_decode_conv", False) or getattr(
        GatedDeltaNet.__call__, "_omlx_mtp_call_marker", False
    )
    assert getattr(Qwen3_5GatedDeltaNet.__call__, "_omlx_fused_decode_conv", False)
