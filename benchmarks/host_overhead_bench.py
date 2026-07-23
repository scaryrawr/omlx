#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark host-side decode and cache-metadata overhead without loading a model.

Use this microbenchmark to identify Python allocation costs before considering a
native extension. Per-token native crossings are not justified by small isolated
gains; a native candidate must improve a representative end-to-end workload.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil

from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDBlockMetadata, PagedSSDCacheIndex
from omlx.output_collector import RequestOutputCollector
from omlx.request import Request, RequestStatus, SamplingParams
from omlx.scheduler import Scheduler


class _Detokenizer:
    """Minimal streaming detokenizer for exercising scheduler output handling."""

    def __init__(self) -> None:
        self.last_segment = ""
        self.text = ""

    def add_token(self, token: int) -> None:
        self.last_segment = "x"
        self.text += self.last_segment

    def finalize(self) -> None:
        self.last_segment = ""


class _Tokenizer:
    """Minimal tokenizer surface used when a request reaches its terminal output."""

    def decode(self, token_ids: list[int]) -> str:
        return "x" * len(token_ids)


def _make_output_scheduler() -> Scheduler:
    """Create only the scheduler state needed by response processing."""
    scheduler = Scheduler.__new__(Scheduler)
    request = Request(
        request_id="bench",
        prompt=[1],
        prompt_token_ids=[1],
        num_prompt_tokens=1,
        sampling_params=SamplingParams(max_tokens=1_000_000),
        status=RequestStatus.RUNNING,
        batch_uid=1,
    )
    scheduler.uid_to_request_id = {1: request.request_id}
    scheduler.running = {request.request_id: request}
    scheduler._output_parser_factory = None
    scheduler._output_parser_sessions = {}
    scheduler._request_detokenizers = {request.request_id: _Detokenizer()}
    scheduler.tokenizer = _Tokenizer()
    scheduler.block_aware_cache = None
    scheduler.total_completion_tokens = 0
    scheduler.num_requests_processed = 0
    scheduler._maybe_capture_boundary_snapshot = lambda _request, _uid: None
    return scheduler


def _run_output_pipeline(num_tokens: int, burst_size: int) -> int:
    """Process and aggregate a decode stream, returning a result checksum."""
    scheduler = _make_output_scheduler()
    collector = RequestOutputCollector()
    checksum = 0

    for start in range(0, num_tokens, burst_size):
        stop = min(start + burst_size, num_tokens)
        responses = [
            SimpleNamespace(
                uid=1,
                token=token,
                finish_reason="length" if token == num_tokens - 1 else None,
                logprobs=None,
                prompt_cache=None,
            )
            for token in range(start, stop)
        ]
        outputs, _ = scheduler._process_batch_responses(responses)
        for output in outputs:
            collector.put(output)
        aggregated = collector.get_nowait()
        if aggregated is not None:
            checksum += len(aggregated.new_token_ids)
            checksum += len(aggregated.output_token_ids)

    return checksum


def _build_cache_metadata(num_blocks: int) -> tuple[PagedCacheManager, PagedSSDCacheIndex]:
    """Build representative in-memory paged and SSD metadata indexes."""
    paged = PagedCacheManager(
        block_size=256,
        max_blocks=num_blocks,
        initial_blocks=num_blocks,
        model_name="benchmark",
    )
    index = PagedSSDCacheIndex(max_size_bytes=1 << 60)
    cache_root = Path("/tmp/omlx-host-overhead")
    now = time.time()

    for block_id in range(num_blocks):
        block_hash = block_id.to_bytes(32, "little")
        index.add(
            PagedSSDBlockMetadata(
                block_hash=block_hash,
                file_path=cache_root / f"{block_hash.hex()}.safetensors",
                file_size=4096,
                token_count=256,
                created_at=now,
                last_access=now,
                num_layers=32,
                model_name="benchmark",
                block_size=256,
                cache_signature="benchmark",
                layer_cache_types=["KVCache"] * 32,
                layer_meta_states=[(0,)] * 32,
            )
        )

    return paged, index


def _measure(
    name: str,
    operation: Callable[[], Any],
    result_size: Callable[[Any], int],
    repeat: int,
) -> dict[str, int | float | str]:
    """Measure median duration and traced peak allocation for one operation."""
    durations = []
    checksum = 0

    operation()
    for _ in range(repeat):
        gc.collect()
        started = time.perf_counter()
        result = operation()
        durations.append(time.perf_counter() - started)
        checksum ^= result_size(result)
        del result

    gc.collect()
    rss_before = psutil.Process().memory_info().rss
    tracemalloc.start()
    result = operation()
    checksum ^= result_size(result)
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = psutil.Process().memory_info().rss
    del result
    gc.collect()

    return {
        "name": name,
        "median_ms": statistics.median(durations) * 1000.0,
        "traced_peak_bytes": traced_peak,
        "rss_delta_bytes": max(0, rss_after - rss_before),
        "checksum": checksum,
    }


def main() -> None:
    """Run selected host-overhead benchmarks and print JSON results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--burst-size", type=int, default=64)
    parser.add_argument("--metadata-blocks", type=int, default=10_000)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    results = [
        _measure(
            "decode_output_pipeline",
            lambda: _run_output_pipeline(args.tokens, args.burst_size),
            int,
            args.repeat,
        ),
        _measure(
            "cache_metadata",
            lambda: _build_cache_metadata(args.metadata_blocks),
            lambda value: len(value[0].blocks) + value[1].count,
            args.repeat,
        ),
    ]
    print(
        json.dumps(
            {
                "tokens": args.tokens,
                "burst_size": args.burst_size,
                "metadata_blocks": args.metadata_blocks,
                "repeat": args.repeat,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
