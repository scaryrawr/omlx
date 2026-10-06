# SPDX-License-Identifier: Apache-2.0

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches.qwen35_verify_qmm import (
    _verify_route,
    set_verify_qmm_armed,
    takes_verify_route,
    vk_eligible,
    vk_qmm,
)


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


@pytest.mark.parametrize(
    ("mode", "rows", "expected_route"),
    [
        ("affine", 4, "sg8"),
        ("affine", 16, "mma"),
        ("mxfp4", 3, "vk"),
        ("mxfp4", 4, "vk"),
        ("mxfp8", 4, "vk"),
        ("mxfp4", 8, None),
        ("mxfp8", 16, None),
    ],
)
def test_verify_route_prediction_respects_quantization_mode(mode, rows, expected_route):
    bits = 8 if mode == "mxfp8" else 4
    group_size = 64 if mode == "affine" else 32
    linear = nn.QuantizedLinear(1024, 16384, group_size=group_size, bits=bits, mode=mode)
    assert (
        _verify_route(rows, 1024, 16384, bits, group_size, mx.bfloat16, mode)
        == expected_route
    )
    from omlx.patches.qwen35_verify_qmm import apply_verify_qmm_patch

    apply_verify_qmm_patch()
    set_verify_qmm_armed(True)
    try:
        assert takes_verify_route(linear, rows, mx.bfloat16) is (
            expected_route is not None
        )
    finally:
        set_verify_qmm_armed(False)
