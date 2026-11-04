# SPDX-License-Identifier: Apache-2.0
"""MLX conversions for Ling's published FP8 and mixed FP4 checkpoints."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def convert_ling_fp8_weights(
    weights: dict[str, mx.array],
    quantization_config: dict[str, Any] | None = None,
) -> dict[str, mx.array]:
    """Convert Ling block-FP8 and routed MXFP4 pairs into MLX weight layouts.

    Ling stores standard projections as E4M3 bytes with 128-by-128 inverse
    scale grids, while routed experts use two E2M1 values per int8 with E8M0
    scales. MLX natively consumes the latter after reinterpreting packed bytes;
    its affine 8-bit linears require the former to be restored and requantized.
    """
    quantization_config = quantization_config or {}
    block_size = quantization_config.get("weight_block_size", (128, 128))
    if not isinstance(block_size, (list, tuple)) or len(block_size) != 2:
        block_size = (128, 128)
    block_rows, block_cols = (int(block_size[0]), int(block_size[1]))

    scale_keys = [key for key in weights if key.endswith(".weight_scale_inv")]
    for scale_key in scale_keys:
        weight_key = scale_key[: -len("_scale_inv")]
        if weight_key not in weights:
            continue

        source_weight = weights[weight_key]
        is_routed_mxfp4 = (
            ".mlp.switch_mlp." in weight_key and source_weight.dtype == mx.int8
        )
        if is_routed_mxfp4:
            scale = weights.pop(scale_key)
            packed = weights.pop(weight_key).view(mx.uint32)
            base = weight_key[: -len("weight")]
            weights[weight_key] = packed
            weights[f"{base}scales"] = scale
            mx.eval(packed, scale)
            continue

        if source_weight.dtype != mx.uint8:
            continue

        scale = weights.pop(scale_key)
        if scale.dtype == mx.uint8:
            scale = mx.power(
                mx.array(2.0, dtype=mx.float32),
                scale.astype(mx.float32) - 127.0,
            )
        else:
            scale = scale.astype(mx.float32)
        weight = mx.from_fp8(weights.pop(weight_key), dtype=mx.float32)
        out_dim, in_dim = weight.shape[-2:]
        target_out = scale.shape[-2] * block_rows
        target_in = scale.shape[-1] * block_cols
        if target_out < out_dim or target_in < in_dim:
            raise ValueError(
                f"Invalid FP8 block scale for {weight_key}: weight "
                f"{weight.shape}, scale {scale.shape}, block {block_size}"
            )

        pad_out = target_out - out_dim
        pad_in = target_in - in_dim
        if pad_out or pad_in:
            padding = [(0, 0)] * (weight.ndim - 2)
            padding.extend(((0, pad_out), (0, pad_in)))
            weight = mx.pad(weight, padding)

        lead = weight.shape[:-2]
        weight = weight.reshape(
            *lead,
            scale.shape[-2],
            block_rows,
            scale.shape[-1],
            block_cols,
        )
        weight = weight * scale[..., :, None, :, None]
        weight = weight.reshape(*lead, target_out, target_in)
        weight = weight[..., :out_dim, :in_dim].astype(mx.bfloat16)

        quantized, scales, biases = mx.quantize(weight, group_size=64, bits=8)
        weights[weight_key] = quantized
        base = weight_key[: -len("weight")]
        weights[f"{base}scales"] = scales
        weights[f"{base}biases"] = biases
        mx.eval(quantized, scales, biases)
        mx.clear_cache()

    return weights
