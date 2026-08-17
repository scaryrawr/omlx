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


def test_patch_installs_for_lm_and_vlm():
    assert apply_qwen35_gdn_decode_patch()
    from mlx_lm.models.qwen3_5 import GatedDeltaNet
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

    assert getattr(GatedDeltaNet.__call__, "_omlx_fused_decode_conv", False)
    assert getattr(Qwen3_5GatedDeltaNet.__call__, "_omlx_fused_decode_conv", False)
