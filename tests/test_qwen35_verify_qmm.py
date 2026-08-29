# SPDX-License-Identifier: Apache-2.0

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches.qwen35_verify_qmm import vk_eligible, vk_qmm


@pytest.mark.parametrize("mode", ["affine", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("rows", [3, 6])
def test_verify_qmm_matches_quantized_linear(mode, rows):
    bits = 4 if mode != "mxfp8" else 8
    group_size = 64 if mode == "affine" else 32
    linear = nn.Linear(256, 128, bias=True)
    quantized = nn.QuantizedLinear.from_linear(
        linear,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    inputs = mx.random.normal((rows, 256), dtype=mx.bfloat16)
    expected = quantized(inputs)
    actual = vk_qmm(
        inputs,
        quantized.weight,
        quantized.scales,
        quantized.biases
        if quantized.biases is not None
        else quantized.scales,
        bits=bits,
        group_size=group_size,
        mode=mode,
    )
    if "bias" in quantized:
        actual = actual + quantized.bias
    mx.eval(expected, actual)

    error = mx.max(
        mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32))
    ).item()
    assert error <= 1.0


@pytest.mark.parametrize("mode", ["mxfp4", "mxfp8"])
def test_verify_qmm_mxfp_lm_head_tile_matches_quantized_linear(mode):
    bits = 4 if mode == "mxfp4" else 8
    linear = nn.Linear(64, 100000, bias=False)
    quantized = nn.QuantizedLinear.from_linear(
        linear,
        group_size=32,
        bits=bits,
        mode=mode,
    )
    inputs = mx.random.normal((3, 64), dtype=mx.bfloat16)
    expected = quantized(inputs)
    actual = vk_qmm(
        inputs,
        quantized.weight,
        quantized.scales,
        quantized.scales,
        bits=bits,
        group_size=32,
        mode=mode,
    )
    mx.eval(expected, actual)

    error = mx.max(
        mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32))
    ).item()
    assert error <= 1.0


@pytest.mark.parametrize(
    ("mode", "bits", "group_size", "expected"),
    [
        ("affine", 4, 64, True),
        ("mxfp4", 4, 32, True),
        ("mxfp8", 8, 32, True),
        ("mxfp4", 8, 32, False),
        ("mxfp8", 8, 64, False),
    ],
)
def test_verify_qmm_mode_eligibility(mode, bits, group_size, expected):
    assert (
        vk_eligible(
            3,
            256,
            16384,
            bits,
            group_size,
            mx.bfloat16,
            mode,
        )
        is expected
    )
