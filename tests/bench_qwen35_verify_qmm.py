#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from omlx.patches.qwen35_verify_qmm import vk_qmm


def _time_ms(fn, warmup, iterations):
    for _ in range(warmup):
        mx.eval(fn())
    started = time.perf_counter()
    for _ in range(iterations):
        mx.eval(fn())
    return (time.perf_counter() - started) * 1000.0 / iterations


def _stock(linear, inputs):
    return linear(inputs)


def _custom(linear, inputs, bits, group_size, mode):
    return vk_qmm(
        inputs,
        linear.weight,
        linear.scales,
        linear.biases if linear.biases is not None else linear.scales,
        bits=bits,
        group_size=group_size,
        mode=mode,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--input-dims", type=int, default=2048)
    parser.add_argument("--output-dims", type=int, default=16384)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    inputs = mx.random.normal(
        (args.rows, args.input_dims), dtype=mx.bfloat16
    )
    for mode, bits, group_size in (
        ("affine", 4, 64),
        ("mxfp4", 4, 32),
        ("mxfp8", 8, 32),
    ):
        linear = nn.QuantizedLinear.from_linear(
            nn.Linear(args.input_dims, args.output_dims, bias=False),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        fallback = partial(_stock, linear, inputs)
        custom = partial(
            _custom,
            linear,
            inputs,
            bits,
            group_size,
            mode,
        )
        fallback_ms = _time_ms(fallback, args.warmup, args.iterations)
        custom_ms = _time_ms(custom, args.warmup, args.iterations)
        print(
            f"{mode}: stock={fallback_ms:.3f}ms custom={custom_ms:.3f}ms "
            f"speedup={fallback_ms / custom_ms:.2f}x"
        )


if __name__ == "__main__":
    main()
