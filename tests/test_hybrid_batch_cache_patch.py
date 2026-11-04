# SPDX-License-Identifier: Apache-2.0
"""Regressions for mixed-length hybrid continuous-batching caches."""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, BatchKVCache

from omlx.patches.hybrid_batch_cache import apply_hybrid_batch_cache_fixes


def test_arrays_cache_mask_combines_left_and_right_padding():
    assert apply_hybrid_batch_cache_fixes()
    merged = ArraysCache.merge([ArraysCache(1) for _ in range(3)])
    merged.prepare(lengths=[4, 2, 1])
    merged.left_padding = mx.array([1, 0, 0])

    mask = merged.make_mask(4)

    expected = mx.array(
        [
            [False, True, True, True],
            [True, True, False, False],
            [True, False, False, False],
        ]
    )
    assert mx.array_equal(mask, expected).item()


def test_empty_batch_kv_filter_preserves_zero_offset():
    assert apply_hybrid_batch_cache_fixes()
    cache = BatchKVCache(left_padding=[2, 1])

    cache.filter([1])

    assert cache._idx == 0
    assert cache.make_mask(1).shape[-1] == 1
