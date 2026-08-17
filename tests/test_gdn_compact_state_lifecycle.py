# SPDX-License-Identifier: Apache-2.0
"""Lock the compact recurrent-state lifecycle used by continuous decode."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import GenerationBatch, StopSequenceMatcher
from mlx_lm.models.cache import ArraysCache


class _CompactStateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache_ids: list[int] = []
        self.state_shapes: list[tuple[int, ...]] = []

    def __call__(self, inputs, cache):
        recurrent = cache[0]
        self.cache_ids.append(id(recurrent))
        for slot, shape in enumerate(((3, 8), (2, 3, 4))):
            state = recurrent[slot]
            if state is None:
                state = mx.zeros((inputs.shape[0], *shape), dtype=mx.float32)
            state = state + 1
            recurrent[slot] = state
        self.state_shapes.append(recurrent[1].shape)
        logits = mx.zeros((inputs.shape[0], 1, 8))
        logits[:, :, 1] = 1
        return logits


def test_generation_batch_keeps_gdn_state_compact_between_decode_steps(
    monkeypatch,
):
    """Steady decode must not extract, merge, filter, or scatter GDN state."""

    def unexpected_cache_transition(*_args, **_kwargs):
        raise AssertionError("steady decode must retain the compact batch cache")

    monkeypatch.setattr(ArraysCache, "extract", unexpected_cache_transition)
    monkeypatch.setattr(ArraysCache, "filter", unexpected_cache_transition)
    monkeypatch.setattr(ArraysCache, "merge", unexpected_cache_transition)

    model = _CompactStateModel()
    cache = ArraysCache(2)
    batch = GenerationBatch(
        model=model,
        uids=[10, 11, 12],
        inputs=mx.array([1, 1, 1]),
        prompt_cache=[cache],
        tokens=[[1], [1], [1]],
        samplers=[None, None, None],
        fallback_sampler=lambda logits: mx.argmax(logits, axis=-1),
        logits_processors=[[], [], []],
        stop_matchers=[StopSequenceMatcher() for _ in range(3)],
        max_tokens=[8, 8, 8],
    )

    for _ in range(3):
        batch._step()

    assert model.cache_ids == [id(cache)] * 4
    assert model.state_shapes == [(3, 2, 3, 4)] * 4
    assert cache[0].shape == (3, 3, 8)
    assert cache[1].shape == (3, 2, 3, 4)
    assert mx.all(cache[0] == 4).item()
    assert mx.all(cache[1] == 4).item()
