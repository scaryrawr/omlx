# SPDX-License-Identifier: Apache-2.0
"""Compatibility fixes for mlx-lm hybrid continuous-batching caches."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def _arrays_mask_is_correct(arrays_cache_cls, mx) -> bool:
    caches = [arrays_cache_cls(1), arrays_cache_cls(1)]
    merged = arrays_cache_cls.merge(caches)
    merged.prepare(lengths=[2, 1])
    mask = merged.make_mask(2)
    expected = mx.array([[True, True], [True, False]])
    return mask is not None and mx.array_equal(mask, expected).item()


def _empty_filter_is_correct(batch_kv_cache_cls) -> bool:
    cache = batch_kv_cache_cls(left_padding=[2, 1])
    cache.filter([1])
    return cache._idx == 0 and cache.make_mask(1).shape[-1] == 1


def apply_hybrid_batch_cache_fixes() -> bool:
    """Install self-retiring fixes for mixed-length hybrid batches."""

    global _APPLIED
    if _APPLIED:
        return True
    try:
        import mlx.core as mx
        from mlx_lm.models.cache import ArraysCache, BatchKVCache
    except ImportError:
        return False

    if not _arrays_mask_is_correct(ArraysCache, mx):
        original_make_mask = ArraysCache.make_mask

        def make_mask(self, N: int):  # noqa: N803 - matches mlx-lm API
            mask = None
            if self.left_padding is not None:
                pos = mx.arange(N)
                mask = pos >= self.left_padding[:, None]
            if self.lengths is not None:
                pos = mx.arange(N)
                right = pos < self.lengths[:, None]
                mask = right if mask is None else (mask & right)
            return mask

        make_mask._omlx_original = original_make_mask
        make_mask._omlx_hybrid_batch_fix = True
        ArraysCache.make_mask = make_mask
        logger.info("Installed ArraysCache mixed-padding mask fix")

    if not _empty_filter_is_correct(BatchKVCache):
        original_filter = BatchKVCache.filter

        def filter_cache(self, batch_indices):
            if self.keys is None:
                self.offset = self.offset[batch_indices]
                self.left_padding = self.left_padding[batch_indices]
                return
            original_filter(self, batch_indices)

        filter_cache._omlx_original = original_filter
        filter_cache._omlx_hybrid_batch_fix = True
        BatchKVCache.filter = filter_cache
        logger.info("Installed empty BatchKVCache filter fix")

    _APPLIED = True
    return True


__all__ = ["apply_hybrid_batch_cache_fixes"]
