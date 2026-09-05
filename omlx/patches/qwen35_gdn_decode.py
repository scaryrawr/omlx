# SPDX-License-Identifier: Apache-2.0
"""Fuse Qwen3.5-family decode convolution state handling into one Metal launch."""

from __future__ import annotations

import logging
import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_ENABLED = False
_LM_PATCHED = False
_VLM_PATCHED = False
_KERNEL = None

_SOURCE = r"""
    uint gid = thread_position_in_grid.x;
    uint batch = gid / uint(C);
    uint c = gid - batch * uint(C);
    uint state_base = batch * 3 * uint(C);
    uint input_base = batch * uint(C);
    float s0 = float(state[state_base + c]);
    float s1 = float(state[state_base + uint(C) + c]);
    float s2 = float(state[state_base + 2 * uint(C) + c]);
    float s3 = float(x[input_base + c]);
    uint weight_base = c * 4;
    float acc = float(weight[weight_base]) * s0
              + float(weight[weight_base + 1]) * s1
              + float(weight[weight_base + 2]) * s2
              + float(weight[weight_base + 3]) * s3;
    T y = T(acc);
    T sigmoid_tail = T(1) / (T(1) + metal::exp(metal::abs(y)));
    T sigmoid = y < T(0) ? sigmoid_tail : T(1) - sigmoid_tail;
    output[input_base + c] = y * sigmoid;
    new_state[state_base + c] = T(s1);
    new_state[state_base + uint(C) + c] = T(s2);
    new_state[state_base + 2 * uint(C) + c] = T(s3);
"""


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen35_fused_decode_conv",
            input_names=["state", "x", "weight"],
            output_names=["new_state", "output"],
            source=_SOURCE,
        )
    return _KERNEL


def fused_conv_silu(state, x, weight):
    """Return bit-exact decode convolution output and shifted state."""

    batch, _, channels = state.shape
    return _kernel()(
        inputs=[state, x, weight],
        template=[("T", state.dtype), ("C", channels)],
        grid=(batch * channels, 1, 1),
        threadgroup=(min(channels, 256), 1, 1),
        output_shapes=[state.shape, (batch, 1, channels)],
        output_dtypes=[state.dtype, state.dtype],
    )


def try_fused_conv_silu(
    module: Any,
    state,
    x,
    *,
    mask=None,
    lengths=None,
):
    """Use the fused path when the call has safe decode geometry."""

    if (
        not _ENABLED
        or mask is not None
        or lengths is not None
        or x.ndim != 3
        or x.shape[1] != 1
        or state is None
        or state.ndim != 3
        or state.shape[0] != x.shape[0]
        or state.shape[1] != 3
        or state.shape[2] != x.shape[2]
        or getattr(module, "conv_kernel_size", 0) != 4
        or module.conv1d.weight.dtype != x.dtype
        or state.dtype != x.dtype
    ):
        return None

    cached = getattr(module, "_omlx_decode_conv_weight", None)
    cache_key = (id(module.conv1d.weight), x.dtype)
    if cached is None or cached[0] != cache_key:
        weight = mx.contiguous(module.conv1d.weight.reshape(x.shape[-1], 4))
        mx.eval(weight)
        cached = (cache_key, weight)
        module._omlx_decode_conv_weight = cached
    return fused_conv_silu(state, x, cached[1])


def _patch_lm() -> bool:
    global _LM_PATCHED
    if _LM_PATCHED:
        return True
    try:
        from mlx.nn.layers.distributed import sum_gradients
        from mlx_lm.models import qwen3_5 as q35
        from mlx_lm.models.gated_delta import gated_delta_update
    except ImportError:
        return False

    cls = q35.GatedDeltaNet
    if getattr(cls.__call__, "_omlx_fused_decode_conv", False):
        _LM_PATCHED = True
        return True
    # The native-MTP patch owns this method but calls try_fused_conv_silu
    # directly from its shared chunk helper.
    if getattr(cls.__call__, "_omlx_mtp_call_marker", False):
        _LM_PATCHED = True
        return True

    def call(self, inputs, mask=None, cache=None):
        batch, seq_len, _ = inputs.shape
        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(
            batch, seq_len, self.num_v_heads, self.head_v_dim
        )
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (batch, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        lengths = cache.lengths if cache is not None else None
        fused = try_fused_conv_silu(self, conv_state, qkv, mask=mask, lengths=lengths)
        if fused is not None:
            new_conv_state, conv_out = fused
        else:
            if mask is not None:
                qkv = mx.where(mask[..., None], qkv, 0)
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            n_keep = self.conv_kernel_size - 1
            if lengths is not None:
                ends = mx.clip(lengths, 0, seq_len)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                new_conv_state = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                new_conv_state = mx.contiguous(conv_input[:, -n_keep:, :])
            conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            tensor.reshape(batch, seq_len, heads, dims)
            for tensor, heads, dims in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[0] = new_conv_state
            cache[1] = state
            cache.advance(seq_len)
        out = self.norm(out, z)
        out = self.out_proj(out.reshape(batch, seq_len, -1))
        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)
        return out

    call._omlx_fused_decode_conv = True
    call._omlx_original = cls.__call__
    cls.__call__ = call
    _LM_PATCHED = True
    return True


def _patch_vlm() -> bool:
    global _VLM_PATCHED
    if _VLM_PATCHED:
        return True
    try:
        from mlx_vlm.models.qwen3_5 import language as q35
    except ImportError:
        return False

    cls = q35.Qwen3_5GatedDeltaNet
    if getattr(cls.__call__, "_omlx_fused_decode_conv", False):
        _VLM_PATCHED = True
        return True
    original = cls.__call__

    def call(self, inputs, mask=None, cache=None):
        batch, seq_len, _ = inputs.shape
        mixed_qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(batch, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != batch:
                conv_state = mx.zeros(
                    (batch, self.conv_kernel_size - 1, self.conv_dim),
                    dtype=inputs.dtype,
                )
        else:
            conv_state = mx.zeros(
                (batch, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        lengths = getattr(cache, "lengths", None) if cache is not None else None
        if mask is not None and mask.shape[0] != batch:
            mask = None
        fused = try_fused_conv_silu(
            self, conv_state, mixed_qkv, mask=mask, lengths=lengths
        )
        if fused is not None:
            new_conv_state, conv_out = fused
        else:
            if mask is not None:
                mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
            conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
            n_keep = self.conv_kernel_size - 1
            if lengths is not None:
                ends = mx.clip(lengths, 0, seq_len)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                new_conv_state = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                new_conv_state = mx.contiguous(conv_input[:, -n_keep:, :])
            if (
                seq_len == 1
                and conv_input.shape[1] == self.conv_kernel_size
                and self.conv1d.weight.dtype in (mx.bfloat16, mx.float16)
            ):
                conv_out = nn.silu(self._causal_conv1d_decode(conv_input))
            else:
                conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            tensor.reshape(batch, seq_len, heads, dims)
            for tensor, heads, dims in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        state = cache[1] if cache else None
        if state is not None and state.shape[0] != batch:
            state = None
        q, k = self._normalize_qk(q, k)
        out, state = q35.gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[0] = new_conv_state
            cache[1] = state
            if hasattr(cache, "advance"):
                cache.advance(seq_len)
                q35._qwen3_5_advance_left_padding_info(cache, seq_len)
                q35._qwen3_5_advance_lengths_info(cache, seq_len)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(batch, seq_len, -1))

    call._omlx_fused_decode_conv = True
    call._omlx_original = original
    cls.__call__ = call
    _VLM_PATCHED = True
    return True


def apply_qwen35_gdn_decode_patch() -> bool:
    """Install the default-on, opt-out fused decode path."""

    global _ENABLED
    if os.environ.get("OMLX_QWEN35_GDN_DECODE_FUSION", "1") == "0":
        return False
    if not mx.metal.is_available():
        return False
    _ENABLED = True
    applied = _patch_lm() | _patch_vlm()
    if applied:
        logger.info("Qwen3.5-family fused GDN decode convolution enabled")
    return applied


__all__ = [
    "apply_qwen35_gdn_decode_patch",
    "fused_conv_silu",
    "try_fused_conv_silu",
]
