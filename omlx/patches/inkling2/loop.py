# SPDX-License-Identifier: Apache-2.0
"""Looped Inkling 2 text execution used by Inkling 2 Mini."""

from dataclasses import dataclass

import mlx.core as mx

from .model import Inkling2Layer, Inkling2TextModel, TextArgs, _unit_rms

MODEL_TYPE = "inkling2_loop"


@dataclass(frozen=True)
class LoopPlan:
    start: int
    end: int
    passes: int
    rank: int

    @classmethod
    def from_config(cls, config: dict, args: TextArgs):
        plan = cls(
            int(config["core_start"]),
            int(config["core_end"]),
            int(config["num_passes"]),
            int(config["gate_rank"]),
        )
        block = args.n_layers_per_attn_res_block
        if (
            config.get("is_varloop")
            or not 0 < plan.start < plan.end <= args.num_hidden_layers
            or plan.start % block
            or plan.end % block
            or plan.passes < 1
            or plan.rank < 1
        ):
            raise ValueError(f"Unsupported Inkling 2 loop configuration: {config}")
        return plan

    def visits(self, num_layers: int):
        return (
            range(self.start),
            *((range(self.start, self.end),) * self.passes),
            range(self.end, num_layers),
        )


class LoopTextModel(Inkling2TextModel):
    def __init__(self, args, quant, plan: LoopPlan):
        self._plan = plan
        super().__init__(args, quant)
        self._visits = plan.visits(len(self.layers))
        self.cache_count = sum(map(len, self._visits))
        self.message_query = mx.zeros(
            (plan.end - plan.start, 4, args.hidden_size), dtype=mx.float32
        )

    def _make_layer(self, args, index, w13, w2):
        rank = self._plan.rank if self._plan.start <= index < self._plan.end else None
        return Inkling2Layer(args, index, w13, w2, message_rank=rank)

    def __call__(self, inputs, cache=None, input_embeddings=None, capture=None):
        if capture:
            raise ValueError("Inkling 2 Mini does not support speculative decoding.")
        h = (
            self.embed_norm(self.embed(inputs))
            if input_embeddings is None
            else input_embeddings
        )
        if cache is not None and len(cache) != self.cache_count:
            raise ValueError("Inkling 2 Mini requires one cache per layer visit.")
        blocks = []
        index = 0
        pre_h, pre_blocks = h, blocks
        for stage, layers in enumerate(self._visits):
            previous = None
            if stage == 1:
                pre_h, pre_blocks = h, list(blocks)
            elif 1 < stage < len(self._visits) - 1:
                # Exclude the loop-entry block, which every pass already sees.
                previous = mx.stack(
                    [*blocks[len(pre_blocks) + 1 :], _unit_rms(h.astype(mx.float32))],
                    axis=-2,
                )
                h, blocks = pre_h, list(pre_blocks)
            for layer in layers:
                messages = None
                if previous is not None:
                    query = self.message_query[layer - self._plan.start]
                    scores = (query @ previous.swapaxes(-1, -2)) * (
                        self.args.hidden_size**-0.5
                    )
                    messages = mx.softmax(scores, axis=-1) @ previous
                entry = cache[index] if cache is not None else None
                h = self.layers[layer](h, blocks, cache=entry, messages=messages)
                index += 1
        x = self.final_input(blocks, _unit_rms(h.astype(mx.float32)))[:, :, 0]
        return self.norm(x).astype(h.dtype)
