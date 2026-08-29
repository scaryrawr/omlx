#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone benchmark script using omlx's native benchmark machinery.

Runs the same single-request and continuous-batching tests as the UI benchmark,
but directly in Python — no server or HTTP needed.  Pass multiple model paths
to run each in sequence and print a side-by-side comparison table.

Usage
-----
    # Single model
    ~/.venv/bin/python scripts/bench.py ~/models/Bonsai-27B

    # Compare two variants
    ~/.venv/bin/python scripts/bench.py ~/models/bonsai-27b ~/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-27B-mlx-2bit/snapshots/70f75f3ad081ab840a42f3304c02c27e7f89bfb7

    # With batch tests
    ~/.venv/bin/python scripts/bench.py model-a model-b --pp 1024 4096 --batch 2 4

Metrics (single-request)
--------------------------
  pp       prompt tokens
  ttft     time-to-first-token (ms)
  tpot     time-per-output-token (ms)
  gen_tps  decode tokens/sec
  pp_tps   prefill tokens/sec
  mem      peak GPU memory

Metrics (batch)
----------------
  bs       batch size
  pp_tps   aggregate prefill tokens/sec
  tg_tps   aggregate decode tokens/sec
  ttft     average time-to-first-token (ms)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from omlx.model_settings import ModelSettings


class NativeMTPMode(StrEnum):
    OFF = "off"
    ON = "on"
    BOTH = "both"


@dataclass(frozen=True)
class BenchmarkVariant:
    name: Literal["native-mtp=off", "native-mtp=on"]
    model_settings: ModelSettings


@dataclass(frozen=True)
class BenchmarkCase:
    model_path: str
    variant: BenchmarkVariant


@dataclass
class BenchmarkResult:
    label: str
    single_results: list[dict[str, object]]
    batch_results: list[dict[str, object]]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="omlx native benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("models", nargs="+", help="Path(s) to model directory")
    p.add_argument(
        "--pp",
        metavar="N",
        type=int,
        nargs="+",
        default=[1024, 4096, 8192],
        help="Prompt token lengths to test (default: 1024 4096 8192)",
    )
    p.add_argument(
        "--gen",
        metavar="N",
        type=int,
        default=128,
        help="Tokens to generate per request (default: 128)",
    )
    p.add_argument(
        "--batch",
        metavar="N",
        type=int,
        nargs="+",
        default=[],
        help="Batch sizes for continuous-batching tests (default: none)",
    )
    p.add_argument(
        "--warmup",
        metavar="N",
        type=int,
        default=1,
        help="Warmup runs before timing (default: 1)",
    )
    p.add_argument(
        "--native-mtp",
        type=NativeMTPMode,
        choices=tuple(NativeMTPMode),
        default=NativeMTPMode.OFF,
        help="Native MTP variant(s): off (default), on, or both.",
    )
    return p


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


# ── formatting helpers ────────────────────────────────────────────────────────

def _fmt_mem(peak_bytes: int) -> str:
    if peak_bytes <= 0:
        return "—"
    return f"{peak_bytes / 1e9:.1f}G"


def _fmt_metric(value: float | None, decimals: int = 1, width: int = 0) -> str:
    """Format a metric, rendering unmeasured (None) values as an em dash.

    Timing-derived metrics come back as None when the run could not observe
    the phase they describe, e.g. an endpoint that never streamed a content
    delta (omlx/admin/benchmark.py::_compute_single_metrics).
    """
    if value is None:
        return f"{'—':>{width}}"
    return f"{value:>{width}.{decimals}f}"


def _short_name(path: str) -> str:
    """Return a short display label for a model path."""
    p = Path(path)
    name = p.name
    # HF snapshot paths: …/models--org--name/snapshots/<hash> → org/name
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "snapshots" and i >= 1:
            repo = parts[i - 1]  # models--org--name
            label = repo.removeprefix("models--").replace("--", "/")
            return label
    return name


def _native_mtp_variants(mode: NativeMTPMode) -> tuple[BenchmarkVariant, ...]:
    if mode is NativeMTPMode.ON:
        return (
            BenchmarkVariant(
                name="native-mtp=on",
                model_settings=ModelSettings(mtp_enabled=True),
            ),
        )
    off = BenchmarkVariant(
        name="native-mtp=off",
        model_settings=ModelSettings(mtp_enabled=False),
    )
    if mode is NativeMTPMode.OFF:
        return (off,)
    return (
        off,
        BenchmarkVariant(
            name="native-mtp=on",
            model_settings=ModelSettings(mtp_enabled=True),
        ),
    )


def _expand_cases(
    model_paths: Sequence[str], mode: NativeMTPMode
) -> list[BenchmarkCase]:
    return [
        BenchmarkCase(model_path, variant)
        for model_path in model_paths
        for variant in _native_mtp_variants(mode)
    ]


def _allocate_labels(
    cases: Sequence[BenchmarkCase], *, include_variant: bool
) -> list[str]:
    bases = [
        (
            f"{'baseline' if case.variant.name == 'native-mtp=off' else 'native-mtp'} "
            f"{_short_name(case.model_path)}"
            if include_variant
            else _short_name(case.model_path)
        )
        for case in cases
    ]
    totals: dict[str, int] = defaultdict(int)
    for base in bases:
        totals[base] += 1

    ordinals: dict[str, int] = defaultdict(int)
    labels: list[str] = []
    used: set[str] = set()
    for base in bases:
        if totals[base] == 1:
            label = base
        else:
            ordinals[base] += 1
            label = f"{base} #{ordinals[base]}"

        collision_ordinal = 1
        candidate = label
        while candidate in used:
            candidate = f"{label} #{collision_ordinal}"
            collision_ordinal += 1
        labels.append(candidate)
        used.add(candidate)
    return labels


def _truncate_label(label: str, width: int) -> str:
    return label[:width] if len(label) > width else label


def _label_with_ordinal(label: str, width: int, ordinal: int) -> str:
    suffix = f"#{ordinal}"
    prefix_width = width - len(suffix)
    if len(label) <= prefix_width:
        return f"{label}{suffix}"
    return f"{label[:prefix_width - 1]}…{suffix}"


def _display_labels(labels: Sequence[str], width: int) -> list[str]:
    if width < 3:
        raise ValueError("display label width must leave room for an ordinal")

    displays = [_truncate_label(label, width) for label in labels]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, display in enumerate(displays):
        groups[display].append(index)

    for indexes in groups.values():
        if len(indexes) > 1:
            for ordinal, index in enumerate(indexes, start=1):
                displays[index] = _label_with_ordinal(labels[index], width, ordinal)

    while True:
        collisions: dict[str, list[int]] = defaultdict(list)
        for index, display in enumerate(displays):
            collisions[display].append(index)
        duplicate_groups = [
            indexes for indexes in collisions.values() if len(indexes) > 1
        ]
        if not duplicate_groups:
            return displays
        for indexes in duplicate_groups:
            for index in indexes:
                displays[index] = _label_with_ordinal(labels[index], width, index + 1)


# ── per-model benchmark runner ────────────────────────────────────────────────

async def _bench_model(
    case: BenchmarkCase,
    pp_lengths: list[int],
    gen_tokens: int,
    batch_sizes: list[int],
    warmup: int,
) -> tuple[list[dict], list[dict]]:
    """Load one model, run all tests, unload.  Returns (single_results, batch_results)."""
    from omlx.admin.benchmark import (
        _generate_prompt,
        _run_batch_test,
        _run_single_test,
    )
    from omlx.engine.vlm import VLMBatchedEngine

    print(f"\nLoading {case.model_path} …")
    t0 = time.perf_counter()
    engine = VLMBatchedEngine(
        model_name=case.model_path,
        model_settings=case.variant.model_settings,
    )
    await engine.start()
    try:
        print(f"Loaded in {time.perf_counter() - t0:.1f}s")

        tokenizer = engine.tokenizer
        prompts: dict[int, str] = {
            pp: _generate_prompt(tokenizer, pp) for pp in sorted(set(pp_lengths))
        }

        if warmup > 0 and pp_lengths:
            warmup_pp = min(pp_lengths)
            print(f"Warming up ({warmup}× pp={warmup_pp}) …")
            for _ in range(warmup):
                await _run_single_test(engine, prompts[warmup_pp], gen_tokens, warmup_pp)

        single_results: list[dict[str, object]] = []
        for pp in sorted(pp_lengths):
            print(f"  pp={pp} gen={gen_tokens} …", end="", flush=True)
            r = await _run_single_test(engine, prompts[pp], gen_tokens, pp)
            single_results.append(r)
            print(
                f"  ttft={_fmt_metric(r['ttft_ms'], 0)}ms  "
                f"{_fmt_metric(r['gen_tps'])} t/s"
            )

        batch_results: list[dict[str, object]] = []
        batch_pp = sorted(pp_lengths)[0] if pp_lengths else 1024
        for bs in sorted(batch_sizes):
            batch_prompts = [_generate_prompt(tokenizer, batch_pp) for _ in range(bs)]
            print(f"  batch={bs} pp={batch_pp} gen={gen_tokens} …", end="", flush=True)
            r = await _run_batch_test(engine, batch_prompts, batch_pp, gen_tokens, bs)
            batch_results.append(r)
            print(
                f"  pp={_fmt_metric(r['pp_tps'], 0)}/s  "
                f"tg={_fmt_metric(r['tg_tps'], 0)}/s"
            )
        return single_results, batch_results
    finally:
        await engine.stop()


# ── table printers ────────────────────────────────────────────────────────────

def _print_single_comparison(
    results: Sequence[BenchmarkResult],
    pp_lengths: list[int],
) -> None:
    """Print a side-by-side comparison table for single-request results."""
    # Column widths: fixed per metric, repeated per model
    col = 9  # width of one model's metric block
    n = len(results)

    # Header: model names spanning their columns
    metrics = ["ttft", "gen_tps", "pp_tps", "mem"]
    block_w = col * len(metrics) + len(metrics) - 1  # e.g. 4*9+3 = 39
    labels = _display_labels([result.label for result in results], block_w)

    print()
    print("  Single-request")
    # Model name header row
    name_row = f"  {'pp':>6}  "
    for label in labels:
        name_row += f"{label:^{block_w}}  "
    print(name_row.rstrip())

    # Sub-header: metric names per model
    sub_row = f"  {'':>6}  "
    for _ in labels:
        sub_row += f"{'ttft':>{col}} {'gen_tps':>{col}} {'pp_tps':>{col}} {'mem':>{col}}  "
    print(sub_row.rstrip())

    sep = "─" * (8 + (block_w + 2) * n)
    print("  " + sep)

    # Data rows
    for pp in sorted(pp_lengths):
        row = f"  {pp:>6}  "
        for result in results:
            model_results = result.single_results
            r = next((x for x in model_results if x["prompt_tokens"] == pp), None)
            if r is None:
                row += f"{'—':>{col}} {'—':>{col}} {'—':>{col}} {'—':>{col}}  "
            else:
                row += (
                    f"{_fmt_metric(r['ttft_ms'], 0, col - 2)}ms "
                    f"{_fmt_metric(r['gen_tps'], 1, col - 2)}/s "
                    f"{_fmt_metric(r['processing_tps'], 0, col - 2)}/s "
                    f"{_fmt_mem(r['peak_memory_bytes']):>{col}}  "
                )
        print(row.rstrip())

    print("  " + sep)


def _print_batch_comparison(
    results: Sequence[BenchmarkResult],
    batch_sizes: list[int],
) -> None:
    """Print a side-by-side comparison table for batch results."""
    col = 9
    n = len(results)
    metrics = ["pp_tps", "tg_tps", "ttft"]
    block_w = col * len(metrics) + len(metrics) - 1
    labels = _display_labels([result.label for result in results], block_w)

    print()
    print("  Continuous-batching")
    name_row = f"  {'bs':>4}  "
    for label in labels:
        name_row += f"{label:^{block_w}}  "
    print(name_row.rstrip())

    sub_row = f"  {'':>4}  "
    for _ in labels:
        sub_row += f"{'pp_tps':>{col}} {'tg_tps':>{col}} {'ttft':>{col}}  "
    print(sub_row.rstrip())

    sep = "─" * (6 + (block_w + 2) * n)
    print("  " + sep)

    for bs in sorted(batch_sizes):
        row = f"  {bs:>4}  "
        for result in results:
            model_results = result.batch_results
            r = next((x for x in model_results if x["batch_size"] == bs), None)
            if r is None:
                row += f"{'—':>{col}} {'—':>{col}} {'—':>{col}}  "
            else:
                row += (
                    f"{_fmt_metric(r['pp_tps'], 0, col - 2)}/s "
                    f"{_fmt_metric(r['tg_tps'], 0, col - 2)}/s "
                    f"{_fmt_metric(r['avg_ttft_ms'], 0, col - 2)}ms  "
                )
        print(row.rstrip())

    print("  " + sep)


# ── main ──────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    model_paths = [str(Path(m).expanduser().resolve()) for m in args.models]
    pp_lengths = sorted(set(args.pp))
    cases = _expand_cases(model_paths, args.native_mtp)
    labels = _allocate_labels(
        cases,
        include_variant=args.native_mtp is not NativeMTPMode.OFF,
    )
    results: list[BenchmarkResult] = []

    for case, label in zip(cases, labels, strict=True):
        single, batch = await _bench_model(
            case, pp_lengths, args.gen, sorted(args.batch), args.warmup
        )
        results.append(BenchmarkResult(label, single, batch))

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  gen_tokens={args.gen}")

    if len(results) == 1:
        # Single model: original compact table
        _print_single_comparison(results, pp_lengths)
        if results[0].batch_results:
            _print_batch_comparison(results, sorted(args.batch))
    else:
        # Multiple models: side-by-side
        _print_single_comparison(results, pp_lengths)
        if any(result.batch_results for result in results):
            _print_batch_comparison(results, sorted(args.batch))

    print()


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
