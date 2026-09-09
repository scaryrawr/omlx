import re
from functools import partial

import mlx.core as mx
from mlx import nn

from ..bailing_moe.language import aggregate_expert_outputs, group_expert_select
from ..base import create_attention_mask, create_ssm_mask, scaled_dot_product_attention
from ..cache import ArraysCache, BatchKVCache, KVCache
from ..gated_delta import gated_delta_update
from ..kimi_k3.language import ShortConv1d
from ..mla import MultiLinear, latent_length, max_absorbed_queries
from ..mlp import SwiGLUMLP
from ..qwen3_vl.language import (
    LanguageModel as Qwen3VLLanguageModel,
)
from ..qwen3_vl.language import (
    Qwen3VLRotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
)
from ..switch_layers import SwiGLU, SwitchGLU
from .config import TextConfig


@partial(mx.compile, shapeless=True)
def _clamped_swiglu(x, gate, limit):
    return mx.clip(nn.silu(gate), a_min=None, a_max=limit) * mx.clip(
        x, a_min=-limit, a_max=limit
    )


class ClampedSwiGLU(nn.Module):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):
        return _clamped_swiglu(x, gate, self.limit)


class DenseMLP(nn.Module):
    def __init__(self, args: TextConfig, intermediate_size: int, limit: float = 0):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)
        self.activation = ClampedSwiGLU(limit) if limit > 0 else SwiGLU()

    def __call__(self, x):
        return self.down_proj(self.activation(self.up_proj(x), self.gate_proj(x)))


class KDAAttention(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.projection_dim = self.num_heads * self.head_dim
        self.conv_kernel = args.short_conv_kernel_size
        self.scale = self.head_dim**-0.5
        self.lower_bound = args.kda_lower_bound if args.kda_safe_gate else None

        hidden = args.hidden_size
        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.q_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.f_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.g_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.A_log = mx.zeros((self.num_heads,))
        self.dt_bias = mx.zeros((self.projection_dim,))
        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
        **kwargs,
    ) -> mx.array:
        batch, length, _ = x.shape
        states = [None, None, None, None] if cache is None else list(cache)
        lengths = None if cache is None else cache.lengths
        q, states[0] = self.q_conv(self.q_proj(x), states[0], mask, lengths)
        k, states[1] = self.k_conv(self.k_proj(x), states[1], mask, lengths)
        v, states[2] = self.v_conv(self.v_proj(x), states[2], mask, lengths)

        q = q.reshape(batch, length, self.num_heads, self.head_dim)
        k = k.reshape(batch, length, self.num_heads, self.head_dim)
        v = v.reshape(batch, length, self.num_heads, self.head_dim)
        eps = 1e-6 / self.head_dim
        q = (self.scale**2) * mx.fast.rms_norm(q, None, eps)
        k = self.scale * mx.fast.rms_norm(k, None, eps)
        gate = self.f_proj(x).reshape(batch, length, self.num_heads, self.head_dim)
        beta = self.b_proj(x).reshape(batch, length, self.num_heads)
        out, states[3] = gated_delta_update(
            q,
            k,
            v,
            gate,
            beta,
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            state=states[3],
            mask=mask,
            use_kernel=not self.training,
            lower_bound=self.lower_bound,
        )
        if cache is not None:
            for index, state in enumerate(states):
                cache[index] = state
            cache.advance(length)
        output_gate = mx.sigmoid(self.g_proj(x)).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        out = (self.o_norm(out) * output_gate).reshape(batch, length, -1)
        return self.o_proj(out)


class MLAAttention(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.scale = self.q_head_dim**-0.5
        hidden = args.hidden_size

        self.q_proj = nn.Linear(hidden, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.g_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.dense = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=False)
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_theta,
            rope_scaling=args.rope_scaling,
        )
        self._absorbed_dims = (
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.v_head_dim,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
        position_ids: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.num_heads, self.q_head_dim)
        q = q.transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed = self.kv_a_proj_with_mqa(x)
        compressed, k_pe = mx.split(compressed, [self.kv_lora_rank], axis=-1)
        kv_latent = mx.expand_dims(self.kv_a_layernorm(compressed), axis=1)
        k_pe = k_pe.reshape(batch, length, 1, self.qk_rope_head_dim).transpose(
            0, 2, 1, 3
        )

        if position_embeddings is None:
            if position_ids is None:
                offset = cache.offset if cache is not None else 0
                position_ids = mx.arange(offset, offset + length)[None, :]
            q_pe, k_pe = self.rotary_emb.apply_rotary(
                q_pe, k_pe, position_ids, unsqueeze_dim=1
            )
        else:
            q_pe, k_pe = apply_multimodal_rotary_pos_emb(
                q_pe, k_pe, *position_embeddings
            )

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        absorbed = length == 1 or length <= max_absorbed_queries(
            *self._absorbed_dims, latent_length(kv_latent)
        )
        if absorbed:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)
        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if absorbed:
            output = self.unembed_out(output)
        gate = mx.sigmoid(self.g_proj(x)).transpose(0, 2, 1)[..., None]
        output = output * gate
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.dense(output)


class BailingMoeGate(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.expert_bias = (
            mx.zeros((args.num_experts,), dtype=mx.float32)
            if args.moe_router_enable_expert_bias
            else None
        )
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.score_function = args.score_function

    def __call__(self, x):
        return group_expert_select(
            x @ self.weight.T,
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            self.score_function,
        )


class SparseMoE(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.gate = BailingMoeGate(args)
        routed_limit = args.expert_swiglu_limit_list[layer_idx]
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            activation=(ClampedSwiGLU(routed_limit) if routed_limit > 0 else SwiGLU()),
            bias=False,
        )
        self.shared_experts = DenseMLP(
            args,
            args.moe_shared_expert_intermediate_size,
            limit=args.share_expert_swiglu_limit_list[layer_idx],
        )

    def __call__(self, x):
        indices, scores = self.gate(x)
        routed = aggregate_expert_outputs(self.switch_mlp(x, indices), scores)
        return routed + self.shared_experts(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.kind = args.layer_plan[layer_idx]
        self.attention = (
            KDAAttention(args) if self.kind == "kda" else MLAAttention(args)
        )
        self.mlp = (
            SwiGLUMLP(args.hidden_size, args.intermediate_size, bias=False)
            if layer_idx < args.first_k_dense_replace
            else SparseMoE(args, layer_idx)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None, **kwargs):
        h = x + self.attention(
            self.input_layernorm(x), mask=mask, cache=cache, **kwargs
        )
        return h + self.mlp(self.post_attention_layernorm(h))


class BailingMoeV3Model(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, index) for index in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs,
        inputs_embeds=None,
        mask=None,
        cache=None,
        position_ids=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
    ):
        h = self.word_embeddings(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        kda_index = next(
            (i for i, kind in enumerate(self.args.layer_plan) if kind == "kda"), None
        )
        mla_index = next(
            (i for i, kind in enumerate(self.args.layer_plan) if kind == "mla"), None
        )
        kda_mask = (
            create_ssm_mask(h, cache[kda_index]) if kda_index is not None else None
        )
        mla_mask = (
            create_attention_mask(h, cache[mla_index], return_array=True)
            if mla_index is not None
            else None
        )
        if mask is not None:
            if mask.ndim == 2:
                kda_mask = mask
                key_mask = mask.astype(mx.bool_)[:, None, None, :]
                mla_mask = (
                    key_mask
                    if mla_mask is None
                    else mla_mask.astype(mx.bool_) & key_mask
                )
            else:
                mla_mask = mask
        position_embeddings = None
        if (
            position_ids is not None
            and mla_index is not None
            and not self.layers[mla_index].attention.rotary_emb.fused_apply
        ):
            position_embeddings = self.layers[mla_index].attention.rotary_emb(
                h, position_ids
            )
        for layer, layer_cache in zip(self.layers, cache):
            layer_mask = kda_mask if layer.kind == "kda" else mla_mask
            h = layer(
                h,
                mask=layer_mask,
                cache=layer_cache,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        return self.norm(h)


class LanguageModel(Qwen3VLLanguageModel):
    def __init__(self, args: TextConfig, config=None):
        nn.Module.__init__(self)
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = BailingMoeV3Model(args)
        self._rope_deltas = None
        self._position_ids = None
        self.lm_head = (
            None
            if args.tie_word_embeddings
            else nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        )

    def get_rope_index(
        self,
        input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
    ):
        video_start = self.config.video_start_token_id
        vision_start = self.config.vision_start_token_id
        if video_start != vision_start:
            input_ids = mx.where(input_ids == video_start, vision_start, input_ids)
        return super().get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, attention_mask
        )

    def __call__(
        self,
        inputs,
        inputs_embeds=None,
        mask=None,
        cache=None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        rope_deltas = kwargs.pop("rope_deltas", None)
        if (
            pixel_values is not None
            or image_grid_thw is not None
            or video_grid_thw is not None
        ):
            self._rope_deltas = None
            self._position_ids = None
        if rope_deltas is not None:
            self._rope_deltas = rope_deltas

        position_cache = next(
            (item for item in cache or [] if isinstance(item, (KVCache, BatchKVCache))),
            None,
        )
        cache_offset = 0
        cache_offset_array = None
        if position_cache is not None:
            cache_offset = (
                position_cache._idx
                if hasattr(position_cache, "_idx")
                else position_cache.offset
            )
            if (
                isinstance(position_cache.offset, mx.array)
                and position_cache.offset.ndim > 0
            ):
                cache_offset_array = position_cache.offset

        if (
            position_ids is not None
            and cache_offset_array is None
            and position_ids.shape[-1] > inputs.shape[-1]
        ):
            position_ids = position_ids[
                ..., cache_offset : cache_offset + inputs.shape[-1]
            ]

        if position_ids is None:
            if cache_offset == 0 or self._rope_deltas is None:
                position_ids, new_deltas = self.get_rope_index(
                    inputs,
                    image_grid_thw,
                    video_grid_thw,
                    mask if mask is None or mask.ndim == 2 else None,
                )
                self._position_ids = position_ids
                self._rope_deltas = new_deltas
            else:
                batch, length = inputs.shape
                delta = self._rope_deltas[:batch]
                position_ids = mx.arange(length).reshape(1, -1)
                position_ids = mx.broadcast_to(position_ids, (batch, length))
                position_ids = position_ids + cache_offset + delta
                if self._position_ids is not None and self._position_ids.ndim == 3:
                    position_ids = mx.broadcast_to(
                        position_ids[None, ...], (3, batch, length)
                    )

        out = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
            visual_pos_masks=kwargs.get("visual_pos_masks"),
            deepstack_visual_embeds=kwargs.get("deepstack_visual_embeds"),
        )
        logits = (
            self.model.word_embeddings.as_linear(out)
            if self.lm_head is None
            else self.lm_head(out)
        )
        from ..base import LanguageModelOutput

        return LanguageModelOutput(logits=logits)

    def make_cache(self):
        return [
            ArraysCache(size=4) if kind == "kda" else KVCache()
            for kind in self.args.layer_plan
        ]

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        prefix = "language_model."
        other = {
            key: value for key, value in weights.items() if not key.startswith(prefix)
        }
        local = {
            key[len(prefix) :]: value
            for key, value in weights.items()
            if key.startswith(prefix)
        }
        if self.args.tie_word_embeddings:
            local.pop("lm_head.weight", None)

        for layer_index, kind in enumerate(self.args.layer_plan):
            base = f"model.layers.{layer_index}"
            if layer_index >= self.args.first_k_dense_replace:
                expert_re = re.compile(
                    rf"^{re.escape(base)}\.mlp\.experts\.(\d+)\."
                    r"(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
                )
                grouped = {}
                for key in list(local):
                    match = expert_re.match(key)
                    if match:
                        expert, projection, suffix = match.groups()
                        grouped.setdefault((projection, suffix), {})[int(expert)] = key
                for (projection, suffix), entries in grouped.items():
                    expected = set(range(self.args.num_experts))
                    found = set(entries)
                    if found != expected:
                        missing = sorted(expected - found)
                        extra = sorted(found - expected)
                        raise ValueError(
                            f"Incomplete expert set for {base} {projection}.{suffix}: "
                            f"missing={missing[:8]}, extra={extra[:8]}"
                        )
                    target = f"{base}.mlp.switch_mlp.{projection}.{suffix}"
                    if target in local:
                        raise ValueError(
                            f"Both stacked and per-expert weights exist: {target}"
                        )
                    local[target] = mx.stack(
                        [
                            local.pop(entries[index])
                            for index in range(self.args.num_experts)
                        ]
                    )

            attention = f"{base}.attention"
            if kind == "kda":
                for name in ("q", "k", "v"):
                    source = f"{attention}.{name}_conv1d.weight"
                    target = f"{attention}.{name}_conv.conv.weight"
                    if source in local:
                        if target in local:
                            raise ValueError(
                                f"Both source and sanitized weights exist: {source}"
                            )
                        value = local.pop(source)
                        local[target] = (
                            value.moveaxis(2, 1) if value.ndim == 3 else value
                        )
                for name in ("A_log", "dt_bias"):
                    key = f"{attention}.{name}"
                    if key in local and local[key].ndim > 1:
                        local[key] = local[key].reshape(-1)
            else:
                key = f"{attention}.kv_b_proj.weight"
                if key in local:
                    targets = (
                        f"{attention}.embed_q.weight",
                        f"{attention}.unembed_out.weight",
                    )
                    if any(target in local for target in targets):
                        raise ValueError(
                            f"Both fused and absorbed MLA weights exist: {attention}"
                        )
                    scale_key = f"{attention}.kv_b_proj.scales"
                    bias_key = f"{attention}.kv_b_proj.biases"
                    quantized = scale_key in local
                    if bias_key in local and not quantized:
                        raise ValueError(f"Missing scales for {attention}.kv_b_proj")
                    value = local.pop(key)
                    bits = group_size = mode = None
                    if quantized:
                        scales = local.pop(scale_key)
                        biases = local.pop(bias_key, None)
                        bits = (value.shape[-1] * 32) // self.args.kv_lora_rank
                        group_size = self.args.kv_lora_rank // scales.shape[-1]
                        if biases is not None:
                            mode = "affine"
                        elif bits == 4:
                            mode = "mxfp4"
                        elif bits == 8:
                            mode = "mxfp8"
                        else:
                            raise ValueError(
                                f"Unsupported bias-free {bits}-bit quantized "
                                f"layout for {attention}.kv_b_proj"
                            )
                        value = mx.dequantize(
                            value,
                            scales,
                            biases,
                            bits=bits,
                            group_size=group_size,
                            mode=mode,
                        )
                    head_width = self.args.qk_nope_head_dim + self.args.v_head_dim
                    value = value.reshape(self.args.num_attention_heads, head_width, -1)
                    embed_q = mx.contiguous(
                        value[:, : self.args.qk_nope_head_dim].swapaxes(-1, -2)
                    )
                    unembed_out = mx.contiguous(value[:, self.args.qk_nope_head_dim :])
                    if quantized:
                        embed_quantized = mx.quantize(
                            embed_q, bits=bits, group_size=group_size, mode=mode
                        )
                        output_quantized = mx.quantize(
                            unembed_out, bits=bits, group_size=group_size, mode=mode
                        )
                        embed_q, embed_scales, *embed_biases = embed_quantized
                        unembed_out, out_scales, *out_biases = output_quantized
                        local[f"{attention}.embed_q.scales"] = embed_scales
                        local[f"{attention}.unembed_out.scales"] = out_scales
                        if embed_biases:
                            local[f"{attention}.embed_q.biases"] = embed_biases[0]
                        if out_biases:
                            local[f"{attention}.unembed_out.biases"] = out_biases[0]
                    local[f"{attention}.embed_q.weight"] = embed_q
                    local[f"{attention}.unembed_out.weight"] = unembed_out

        other.update({f"{prefix}{key}": value for key, value in local.items()})
        return other

    @property
    def quant_predicate(self):
        def predicate(path, module):
            if path.endswith(".gate"):
                return False
            if not (hasattr(module, "to_quantized") or isinstance(module, nn.Linear)):
                return False
            weight = getattr(module, "weight", None)
            return weight is None or weight.shape[-1] % 32 == 0

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path):
            return not path.endswith(("A_log", "dt_bias", "expert_bias"))

        return predicate
