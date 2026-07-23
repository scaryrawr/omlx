# SPDX-License-Identifier: Apache-2.0
"""Benchmarks for MLX-resident VLM preprocessing and SpecPrefill selection.

Run with:
    pytest tests/bench_mlx_vectorization.py -m slow -v -s
"""

import math
import statistics
import time
from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from omlx.patches.mlx_vlm_pixtral_torch_free.vendor.image_processing_pixtral import (
    PixtralImageProcessor,
)
from omlx.patches.specprefill import select_chunks

pytestmark = pytest.mark.slow


def _median_ms(operation: Callable[[], object], repeat: int = 5) -> float:
    """Return warmed median wall time while materializing any MLX result."""
    for _ in range(2):
        result = operation()
        if isinstance(result, mx.array):
            mx.eval(result)

    durations = []
    for _ in range(repeat):
        started = time.perf_counter()
        result = operation()
        if isinstance(result, mx.array):
            mx.eval(result)
        durations.append(time.perf_counter() - started)
    return statistics.median(durations) * 1000


def _legacy_select_chunks(
    importance: mx.array,
    keep_pct: float = 0.3,
    chunk_size: int = 32,
) -> mx.array:
    """Reproduce the former per-chunk synchronization baseline."""
    token_count = importance.shape[0]
    chunk_count = math.ceil(token_count / chunk_size)
    keep_count = max(1, math.ceil(chunk_count * keep_pct))
    scores = [
        mx.mean(
            importance[
                chunk * chunk_size : min((chunk + 1) * chunk_size, token_count)
            ]
        ).item()
        for chunk in range(chunk_count)
    ]
    kept_chunks = sorted(
        range(chunk_count),
        key=lambda chunk: scores[chunk],
        reverse=True,
    )[:keep_count]
    kept_chunks.sort()
    return mx.array(
        [
            token
            for chunk in kept_chunks
            for token in range(
                chunk * chunk_size,
                min((chunk + 1) * chunk_size, token_count),
            )
        ]
    )


@pytest.mark.parametrize("token_count", [4096, 32768, 131071])
def test_specprefill_chunk_selection(token_count: int):
    """Compare per-chunk synchronization with one vectorized reduction."""
    importance = mx.random.uniform(shape=(token_count,))
    mx.eval(importance)

    expected = _legacy_select_chunks(importance)
    actual = select_chunks(importance)
    assert actual.tolist() == expected.tolist()

    baseline_ms = _median_ms(lambda: _legacy_select_chunks(importance))
    vectorized_ms = _median_ms(lambda: select_chunks(importance))
    print(
        f"\n  tokens={token_count} baseline={baseline_ms:.3f} ms "
        f"vectorized={vectorized_ms:.3f} ms "
        f"speedup={baseline_ms / vectorized_ms:.1f}x"
    )


@pytest.mark.parametrize("batch_size", [1, 4])
def test_pixtral_preprocessing(batch_size: int):
    """Compare NumPy preprocessing plus conversion with MLX-resident output."""
    rng = np.random.default_rng(42)
    processor = PixtralImageProcessor(
        size={"longest_edge": 1024},
        patch_size=16,
    )
    images = [
        Image.fromarray(
            rng.integers(
                0,
                256,
                (900 - index * 73, 1200 - index * 91, 3),
                dtype=np.uint8,
            )
        )
        for index in range(batch_size)
    ]

    numpy_output = processor(images)["pixel_values"]
    mlx_output = processor(images, return_tensors="mlx")["pixel_values"]
    mx.eval(mlx_output)
    np.testing.assert_array_equal(np.asarray(mlx_output), numpy_output)

    baseline_ms = _median_ms(
        lambda: mx.array(processor(images)["pixel_values"]),
    )
    mlx_ms = _median_ms(
        lambda: processor(images, return_tensors="mlx")["pixel_values"],
    )
    print(
        f"\n  batch={batch_size} baseline={baseline_ms:.2f} ms "
        f"mlx={mlx_ms:.2f} ms speedup={baseline_ms / mlx_ms:.2f}x"
    )
