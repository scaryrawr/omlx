from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput
from ..cache import ArraysCache, BatchKVCache, KVCache
from ..qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
from ..qwen3_5_moe.language import (
    LanguageModel as QwenMoeLanguageModel,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeModel,
)
from .canvas import canvas_positions


class GatedDeltaNet(Qwen3_5MoeGatedDeltaNet):
    def _normalize_qk(self, q, k):
        # HF v5.18 uses sum-of-squares + epsilon, not mean-of-squares + epsilon.
        inv_scale = self.head_k_dim**-0.5
        eps = 1e-6 / self.head_k_dim
        return (
            inv_scale**2 * mx.fast.rms_norm(q, None, eps),
            inv_scale * mx.fast.rms_norm(k, None, eps),
        )


class ExactForward(Qwen3_5BatchInvariantForward):
    def _attention(self, attention, x, mask, cache, position_ids, position_embeddings):
        # The row-exact path slices a four-axis mask; rotating caches emit 2D.
        if isinstance(mask, mx.array) and mask.ndim == 2:
            mask = mask[None, None]
        return super()._attention(
            attention, x, mask, cache, position_ids, position_embeddings
        )

    @staticmethod
    def _normalize_gated_delta_qk(layer, q, k):
        return layer._normalize_qk(q, k)


_FORWARD = ExactForward()


class LanguageModel(QwenMoeLanguageModel):
    requires_explicit_rope_deltas = True

    def __init__(self, args, config=None):
        nn.Module.__init__(self)
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Qwen3_5MoeModel(args)
        for layer in self.model.layers:
            if layer.is_linear:
                layer.linear_attn = GatedDeltaNet(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def validate_dspark_compatibility(self, config):
        if config.block_size_policy != "fixed":
            raise ValueError("MiniCPM-V 4.7 DSpark requires fixed-block verification")
        if (
            self.args.hidden_size == 2048
            and self.args.num_hidden_layers == 40
            and self.args.vocab_size == 248144
        ):
            required = {
                "target_layer_ids": [3, 11, 19, 27, 35],
                "num_hidden_layers": 5,
                "head_dim": 256,
                "markov_rank": 256,
                "mask_token_id": 248076,
                "proposal_length": 15,
                "enable_confidence_head": True,
                "sample_from_anchor": True,
                "target_model_type": "minicpmv4_7",
            }
            if any(getattr(config, name) != value for name, value in required.items()):
                raise ValueError(
                    "DSpark helper does not match the published MiniCPM-V 4.7 pairing"
                )
            if self.config is None or self.config.model_type != "minicpmv4_7":
                raise ValueError("MiniCPM-V 4.7 DSpark requires complete target config")

    def get_rope_index(
        self,
        input_ids,
        *,
        target_sizes=None,
        target_sizes_videos=None,
        mm_token_type_ids=None,
        attention_mask=None,
        downsample_mode=None,
    ):
        if self.config is None:
            raise ValueError("Canvas positions require complete MiniCPM-V 4.7 config")
        return canvas_positions(
            input_ids,
            self.config,
            target_sizes=target_sizes,
            target_sizes_videos=target_sizes_videos,
            mm_token_type_ids=mm_token_type_ids,
            attention_mask=attention_mask,
            downsample_mode=downsample_mode,
        )

    def _positions(self, inputs, cache, position_ids, rope_deltas, mask):
        batch, length = inputs.shape
        offset = 0
        if cache is not None:
            fa_cache = cache[self.model.fa_idx]
            if fa_cache is not None:
                offset = fa_cache.offset
        offsets = mx.array(offset, dtype=mx.int32)
        if offsets.ndim == 0:
            offsets = mx.broadcast_to(offsets, (batch,))
        elif offsets.shape != (batch,):
            raise ValueError("Cache offsets must have one entry per request")

        if position_ids is not None:
            if position_ids.ndim == 2:
                position_ids = mx.broadcast_to(
                    position_ids[None], (3, batch, position_ids.shape[-1])
                )
            if position_ids.ndim != 3 or position_ids.shape[:2] != (3, batch):
                raise ValueError("position_ids must have shape [3, batch, sequence]")
            if position_ids.shape[-1] == length:
                return position_ids
            if position_ids.shape[-1] < length:
                raise ValueError("position_ids do not cover this input window")
            if offsets.size > 1 and not bool(mx.all(offsets == offsets[0]).item()):
                raise ValueError("Batched prompt positions must be sliced per request")
            start = int(mx.maximum(offsets[0], 0).item())
            if start + length > position_ids.shape[-1]:
                raise ValueError("Prompt positions do not cover the cached prefix")
            return position_ids[..., start : start + length]

        if rope_deltas is not None and rope_deltas.shape != (batch, 1):
            raise ValueError("rope_deltas must have shape [batch, 1]")
        if rope_deltas is None and bool(mx.any(offsets > 0).item()):
            raise ValueError(
                "Cached MiniCPM-V 4.7 decode requires request-local rope_deltas"
            )
        positions = mx.arange(length)[None] + offsets[:, None]
        if mask is not None and mask.ndim == 2 and mask.shape == inputs.shape:
            positions = mx.maximum(mx.cumsum(mask, axis=-1) - 1, 0) + mx.maximum(
                offsets[:, None], 0
            )
        if rope_deltas is not None:
            positions = positions + rope_deltas
        return mx.broadcast_to(positions[None], (3, batch, length))

    def prepare_prefill_inputs(self, inputs, embeds, positions, mask, cache):
        if mask is None:
            return inputs, embeds, positions, cache, None
        if mask.shape != inputs.shape or mask.ndim != 2:
            raise ValueError("attention_mask must match the current input window")
        if not bool(mx.all((mask == 0) | (mask == 1)).item()):
            raise ValueError("attention_mask must contain only zero or one")
        if bool(mx.all(mask).item()):
            return inputs, embeds, positions, cache, None
        valid = mask.astype(mx.bool_)
        fa_cache = None if cache is None else cache[self.model.fa_idx]
        if isinstance(fa_cache, BatchKVCache):
            expected = (
                mx.arange(inputs.shape[1])[None] + fa_cache._idx
                >= fa_cache.left_padding[:, None]
            )
            if bool(mx.all(valid == expected).item()):
                identity = mx.broadcast_to(
                    mx.arange(inputs.shape[1])[None], inputs.shape
                )
                return inputs, embeds, positions, cache, identity
        lengths = mx.sum(valid, axis=1)
        if bool(mx.any(lengths == 0).item()):
            raise ValueError("Every prompt must contain at least one unmasked token")
        starts = mx.argmax(valid, axis=1)
        axis = mx.arange(inputs.shape[1])[None]
        contiguous = (axis >= starts[:, None]) & (axis < (starts + lengths)[:, None])
        if not bool(mx.all(valid == contiguous).item()):
            raise ValueError("Prompt attention_mask must describe contiguous tokens")
        if cache is None:
            cache = self.make_cache()
        if any(not entry.empty() for entry in cache):
            raise ValueError(
                "Padded cached prefill requires scheduler-prepared windows"
            )
        if any(
            type(entry) not in (ArraysCache, KVCache, BatchKVCache) for entry in cache
        ):
            raise ValueError("Padded prefill requires plain hybrid KV/Arrays caches")
        padding = (inputs.shape[1] - lengths).tolist()
        for index, entry in enumerate(cache):
            if isinstance(entry, ArraysCache):
                entry.lengths = None
                entry.left_padding = mx.array(padding)
            else:
                cache[index] = BatchKVCache(padding)
        # Normalize padding at the input boundary; the reused hybrid cache core
        # expects left padding, including when the tokenizer pads on the right.
        order = mx.argsort(mx.where(valid, axis + inputs.shape[1], axis), axis=1)
        inputs = mx.take_along_axis(inputs, order, axis=1)
        positions = mx.take_along_axis(
            positions, mx.broadcast_to(order[None], positions.shape), axis=2
        )
        if embeds is not None:
            embeds = mx.take_along_axis(
                embeds, mx.broadcast_to(order[..., None], embeds.shape), axis=1
            )
        return inputs, embeds, positions, cache, mx.argsort(order, axis=1)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        attention_mask = kwargs.get("attention_mask")
        if mask is None and attention_mask is not None:
            mask = attention_mask
        capture_ids = kwargs.get("capture_layer_ids")
        if capture_ids is not None:
            if len(set(capture_ids)) != len(capture_ids) or any(
                type(i) is not int or not 0 <= i < len(self.layers) for i in capture_ids
            ):
                raise ValueError("capture_layer_ids must be unique valid layer indices")
        positions = self._positions(
            inputs, cache, kwargs.get("position_ids"), kwargs.get("rope_deltas"), mask
        )
        if (
            kwargs.get("speculative_verify", False)
            and mask is not None
            and not bool(mx.all(mask).item())
        ):
            raise ValueError(
                "Speculative verification does not accept padded proposals"
            )
        inputs, inputs_embeds, positions, cache, restore = self.prepare_prefill_inputs(
            inputs, inputs_embeds, positions, mask, cache
        )
        forward_kwargs = dict(
            cache=cache,
            inputs_embeds=inputs_embeds,
            position_ids=positions,
            capture_layer_ids=capture_ids,
            return_hidden=kwargs.get("return_hidden", False),
            return_shared_kv=kwargs.get("return_shared_kv", False),
            skip_logits=kwargs.get("skip_logits", False),
        )
        if kwargs.get("speculative_verify", False):
            return _FORWARD.verify(self, inputs, **forward_kwargs)
        if inputs.shape[-1] == 1 and cache is not None:
            return _FORWARD(self, inputs, **forward_kwargs)
        sink = [] if capture_ids is not None or restore is not None else None
        hidden = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            cache=cache,
            mask=mask,
            position_ids=positions,
            capture_layer_ids=capture_ids,
            hidden_sink=sink,
        )
        if forward_kwargs["return_hidden"]:
            if sink is None:
                sink = []
            sink.append(hidden)
        logits = None
        if not forward_kwargs["skip_logits"]:
            logits = self.speculative_logits_from_hidden(hidden)
        if restore is not None:

            def restore_rows(value):
                return mx.take_along_axis(
                    value, mx.broadcast_to(restore[..., None], value.shape), axis=1
                )

            if logits is not None:
                logits = restore_rows(logits)
            if sink is not None:
                sink = [restore_rows(value) for value in sink]
        return LanguageModelOutput(
            logits=logits,
            hidden_states=(
                sink
                if capture_ids is not None or forward_kwargs["return_hidden"]
                else None
            ),
            shared_kv_states={} if forward_kwargs["return_shared_kv"] else None,
        )

    def speculative_verify_logits(self, inputs, cache, sampler, **kwargs):
        output = self(
            inputs,
            cache=cache,
            capture_layer_ids=[],
            speculative_verify=True,
            return_hidden=True,
            return_shared_kv=True,
            **kwargs,
        )
        try:
            return (
                output.hidden_states[-1],
                output.shared_kv_states,
                output.gdn_states,
                sampler(output.logits),
            )
        except BaseException:
            output.gdn_states.abort()
            raise

    def speculative_verify_hidden(self, inputs, cache, **kwargs):
        output = self(
            inputs,
            cache=cache,
            capture_layer_ids=[],
            speculative_verify=True,
            return_hidden=True,
            return_shared_kv=True,
            skip_logits=True,
            **kwargs,
        )
        return output.hidden_states[-1], output.shared_kv_states, output.gdn_states
