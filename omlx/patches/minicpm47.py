# SPDX-License-Identifier: Apache-2.0
# Portions adapted from mlx-vlm; see ../MLX_VLM_LICENSE.txt.
"""MiniCPM serving hooks over the official mlx-vlm model and exact verifier."""

from __future__ import annotations

import threading
from types import MethodType

import mlx.core as mx

_LOCK = threading.Lock()
_APPLIED = False


def validate_target_config(config: dict) -> None:
    if (config.get("text_config") or {}).get("norm_topk_prob", True) is not True:
        raise ValueError("MiniCPM-V 4.7 requires normalized top-k routing.")


def _normalize_qk(layer, q, k):
    inv_scale = layer.head_k_dim**-0.5
    eps = 1e-6 / layer.head_k_dim
    return (
        inv_scale**2 * mx.fast.rms_norm(q, None, eps),
        inv_scale * mx.fast.rms_norm(k, None, eps),
    )


def _prepare_padding(inputs, embeds, positions, mask, cache):
    from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache

    if bool(mx.all(mask).item()):
        return inputs, embeds, positions, None
    valid = mask.astype(mx.bool_)
    attention = next(
        (entry for entry in cache if isinstance(entry, BatchKVCache)), None
    )
    if isinstance(attention, BatchKVCache):
        expected = (
            mx.arange(inputs.shape[1])[None] + attention._idx
            >= attention.left_padding[:, None]
        )
        if bool(mx.all(valid == expected).item()):
            return inputs, embeds, positions, None
    lengths = mask.sum(axis=1)
    if bool(mx.any(lengths == 0).item()):
        raise ValueError("Every prompt must contain at least one unmasked token")
    starts = mx.argmax(valid, axis=1)
    axis = mx.arange(inputs.shape[1])[None]
    contiguous = (axis >= starts[:, None]) & (axis < (starts + lengths)[:, None])
    if not bool(mx.all(valid == contiguous).item()):
        raise ValueError("attention_mask must describe contiguous tokens")
    if any(not entry.empty() for entry in cache):
        raise ValueError("Padded cached prefill requires scheduler-prepared windows")
    if any(type(entry) not in (ArraysCache, KVCache, BatchKVCache) for entry in cache):
        raise ValueError("Padded prefill requires plain hybrid KV/Arrays caches")
    padding = (inputs.shape[1] - lengths).tolist()
    for index, entry in enumerate(cache):
        if isinstance(entry, ArraysCache):
            entry.lengths = None
            entry.left_padding = mx.array(padding)
        elif type(entry) in (KVCache, BatchKVCache):
            cache[index] = BatchKVCache(padding)
    order = mx.argsort(mx.where(valid, axis + inputs.shape[1], axis), axis=1)
    inputs = mx.take_along_axis(inputs, order, axis=1)
    positions = mx.take_along_axis(
        positions, mx.broadcast_to(order[None], positions.shape), axis=2
    )
    if embeds is not None:
        embeds = mx.take_along_axis(
            embeds, mx.broadcast_to(order[..., None], embeds.shape), axis=1
        )
    return inputs, embeds, positions, mx.argsort(order, axis=1)


def apply_minicpm47_serving_patch() -> bool:
    """Keep numerical and request-local serving contracts, not model copies."""
    global _APPLIED
    with _LOCK:
        if _APPLIED:
            return False
        from mlx_vlm.models.minicpmv4_7 import (
            MiniCPMV4_7Processor,
            language,
            minicpmv4_7,
        )
        from mlx_vlm.models.qwen3_5.speculative_verifier import (
            Qwen3_5BatchInvariantForward,
        )

        class MiniCPMForward(Qwen3_5BatchInvariantForward):
            @staticmethod
            def _normalize_gated_delta_qk(layer, q, k):
                return _normalize_qk(layer, q, k)

            def _attention(self, attention, x, mask, *args):
                if isinstance(mask, mx.array) and mask.ndim == 2:
                    mask = mask[None, None]
                return super()._attention(attention, x, mask, *args)

        forward = MiniCPMForward()

        def process(
            self,
            text=None,
            images=None,
            videos=None,
            downsample_mode=None,
            max_slice_nums=None,
            max_num_frames=None,
            **kwargs,
        ):
            # mlx-vlm's generic dispatcher forwards only explicitly named parameters.
            options = {
                "downsample_mode": downsample_mode,
                "max_slice_nums": max_slice_nums,
                "max_num_frames": max_num_frames,
            }
            return self(
                text=text,
                images=images,
                videos=videos,
                **{key: value for key, value in options.items() if value is not None},
                **kwargs,
            )

        MiniCPMV4_7Processor.process = process  # type: ignore[attr-defined]

        class ServingLanguage:
            requires_explicit_rope_deltas = True

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                for layer in self.layers:
                    if layer.is_linear:
                        layer.linear_attn._normalize_qk = MethodType(
                            _normalize_qk, layer.linear_attn
                        )

            def __call__(
                self, inputs, inputs_embeds=None, mask=None, cache=None, **kwargs
            ):
                batch, length = inputs.shape
                if mask is None:
                    mask = kwargs.pop("attention_mask", None)
                if (
                    mask is not None
                    and mask.ndim == 2
                    and (
                        mask.shape != inputs.shape
                        or not bool(mx.all((mask == 0) | (mask == 1)).item())
                    )
                ):
                    raise ValueError(
                        "attention_mask must contain one binary value per input token"
                    )
                positions = kwargs.pop("position_ids", None)
                deltas = kwargs.pop("rope_deltas", None)
                offset = cache[self.model.fa_idx].offset if cache is not None else 0
                offsets = mx.array(offset, dtype=mx.int32)
                if offsets.ndim == 0:
                    offsets = mx.broadcast_to(offsets, (batch,))
                if offsets.shape != (batch,):
                    raise ValueError("Cache offsets must have one entry per request")
                if deltas is not None and deltas.shape != (batch, 1):
                    raise ValueError("rope_deltas must have shape [batch, 1]")
                if positions is None:
                    if deltas is None and bool(mx.any(offsets > 0).item()):
                        raise ValueError(
                            "Cached MiniCPM-V 4.7 decode requires request-local rope_deltas"
                        )
                    positions = offsets[:, None] + mx.arange(length)[None]
                    if mask is not None and mask.ndim == 2:
                        positions = mx.maximum(mx.cumsum(mask, axis=-1) - 1, 0)
                        positions = positions + mx.maximum(offsets[:, None], 0)
                    if deltas is not None:
                        positions = positions + deltas
                    positions = mx.broadcast_to(positions[None], (3, batch, length))
                elif positions.shape != (3, batch, length):
                    raise ValueError("position_ids must match the current input window")
                verify = kwargs.pop("speculative_verify", False)
                restore = None
                if mask is not None and mask.ndim == 2:
                    if verify and not bool(mx.all(mask).item()):
                        raise ValueError(
                            "Speculative verification cannot accept padded proposals"
                        )
                    if cache is None:
                        cache = self.make_cache()
                    inputs, inputs_embeds, positions, restore = _prepare_padding(
                        inputs, inputs_embeds, positions, mask, cache
                    )
                if verify or (cache is not None and length == 1):
                    call = forward.verify if verify else forward
                    return call(
                        self,
                        inputs,
                        cache=cache,
                        inputs_embeds=inputs_embeds,
                        position_ids=positions,
                        capture_layer_ids=kwargs.get("capture_layer_ids"),
                        return_hidden=kwargs.get("return_hidden", False),
                        return_shared_kv=kwargs.get("return_shared_kv", False),
                        skip_logits=kwargs.get("skip_logits", False),
                    )
                output = super().__call__(
                    inputs,
                    inputs_embeds=inputs_embeds,
                    mask=mask,
                    cache=cache,
                    position_ids=positions,
                    rope_deltas=deltas,
                    **kwargs,
                )
                if restore is not None:

                    def restore_rows(value):
                        return mx.take_along_axis(
                            value,
                            mx.broadcast_to(restore[..., None], value.shape),
                            axis=1,
                        )

                    if output.logits is not None:
                        output.logits = restore_rows(output.logits)
                    if output.hidden_states is not None:
                        output.hidden_states = [
                            restore_rows(value) for value in output.hidden_states
                        ]
                return output

        class LanguageModel(ServingLanguage, language.LanguageModel):
            pass

        class MoELanguageModel(ServingLanguage, language.MoELanguageModel):
            pass

        minicpmv4_7.LanguageModel = LanguageModel  # type: ignore[misc,assignment]
        minicpmv4_7.MoELanguageModel = MoELanguageModel  # type: ignore[misc,assignment]
        _APPLIED = True
        return True
