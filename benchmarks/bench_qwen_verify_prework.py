# SPDX-License-Identifier: Apache-2.0
"""Paired benchmark for the Qwen verifier GDN prework candidate."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import bench_qwen_decode as bench  # noqa: E402

PROMPT_TEXT = bench.DEFAULT_PROMPT_TEXT
PREWORK_ENV = "OMLX_QWEN35_VERIFY_PREWORK"
MISSING = object()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--pp", type=int, default=512)
    parser.add_argument("--gen", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--fixed-depth", type=int, choices=(1, 2), default=2)
    parser.add_argument("--prompt-text", default=PROMPT_TEXT)
    args = parser.parse_args(argv)
    for name in ("pp", "gen", "pairs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    return args


def _schedule(pairs: int) -> list[tuple[bool, bool]]:
    return [(True, False), (True, True)] + [
        (False, treatment)
        for _ in range(pairs)
        for treatment in (False, True, True, False)
    ]


def _head_metadata(head: Any, tree_flatten: Any) -> list[dict[str, Any]]:
    return [
        {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        for name, value in tree_flatten(head.parameters())
    ]


def _validate_engagement(record: dict[str, Any]) -> None:
    errors = []
    if record["draft_calls"] <= 0:
        errors.append("native MTP draft path was not engaged")
    if record["treatment"] and record["prework_calls"] <= 0:
        errors.append("prework treatment was not engaged")
    if not record["treatment"] and record["prework_calls"] != 0:
        errors.append("prework treatment engaged in the baseline")
    record["engagement_errors"] = errors
    if errors:
        raise bench.BenchmarkError("; ".join(errors))


async def _run(args: argparse.Namespace, deps: dict[str, Any] | None = None) -> dict:
    if deps is None:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        from omlx.engine.vlm import VLMBatchedEngine
        from omlx.patches import qwen35_gdn_prework as prework
        from omlx.patches.mlx_lm_mtp.batch_generator import _DepthController
        from omlx.scheduler import SchedulerConfig

        deps = {
            "mx": mx,
            "tree_flatten": tree_flatten,
            "engine_cls": VLMBatchedEngine,
            "prework": prework,
            "depth_controller": _DepthController,
            "scheduler_config": SchedulerConfig,
        }
    mx = deps["mx"]
    controller = deps["depth_controller"]
    prework = deps["prework"]
    old_env = os.environ.get(PREWORK_ENV)
    old_observe = controller.observe
    old_should_exit = controller.should_exit
    old_prework = prework.gdn_prework_fused
    engine = None
    original_draft = None
    language_model = None
    original_depth = MISSING
    depth_changed = False
    before = None
    result = {
        "schema_version": 1,
        "status": "running",
        "model": args.model,
        "candidate": "qwen35_gdn_prework",
        "fixed_depth": args.fixed_depth,
        "config": {"pp": args.pp, "gen": args.gen, "pairs": args.pairs},
        "trials": [],
    }
    prework_calls = 0
    draft_calls = 0

    def observe_fixed(self, *values, **kwargs):
        old_observe(self, *values, **kwargs)
        self.cur = min(self.max_depth, args.fixed_depth)

    def counted_prework(*values, **kwargs):
        nonlocal prework_calls
        prework_calls += 1
        return old_prework(*values, **kwargs)

    try:
        before = bench._model_fingerprint(args.model)
        result["model_fingerprint_before"] = before
        result["runtime"] = bench._runtime_info(mx)
        result["runner_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        os.environ[PREWORK_ENV] = "1"
        controller.observe = observe_fixed
        controller.should_exit = lambda self: False
        prework.gdn_prework_fused = counted_prework
        settings = bench._model_settings(True)
        settings.mtp_num_draft_tokens = args.fixed_depth
        engine = deps["engine_cls"](
            model_name=args.model,
            model_settings=settings,
            scheduler_config=deps["scheduler_config"](
                paged_ssd_cache_dir=None, hot_cache_max_size=0
            ),
        )
        await engine.start()
        language_model = engine._adapter._language_model
        original_depth = getattr(language_model, "_omlx_mtp_depth", MISSING)
        head = (
            language_model.get_mtp_module()
            if hasattr(language_model, "get_mtp_module")
            else getattr(language_model, "mtp", None)
        )
        decode_enabled = bool(
            getattr(language_model, "_omlx_mtp_decode_enabled", False)
        )
        if head is None or not decode_enabled:
            raise bench.BenchmarkError("native MTP did not become ready")
        language_model._omlx_mtp_depth = args.fixed_depth
        depth_changed = True
        head_tensors = _head_metadata(head, deps["tree_flatten"])
        result["mtp"] = {
            "decode_enabled": decode_enabled,
            "head_tensors": head_tensors,
            "all_bfloat16": bool(head_tensors)
            and all("bfloat16" in item["dtype"] for item in head_tensors),
        }
        if not result["mtp"]["all_bfloat16"]:
            raise bench.BenchmarkError("MTP head parameters are not all bfloat16")
        original_draft = engine._adapter.mtp_forward

        def counted_draft(*values, **kwargs):
            nonlocal draft_calls
            draft_calls += 1
            return original_draft(*values, **kwargs)

        engine._adapter.mtp_forward = counted_draft
        prompt = bench._build_prompt(engine.tokenizer, args.pp, args.prompt_text)
        result["prompt"] = {
            "text": args.prompt_text,
            "token_ids": list(prompt),
            "sha256": bench._token_digest(prompt),
        }
        expected = None
        for index, (warmup, treatment) in enumerate(_schedule(args.pairs)):
            os.environ[PREWORK_ENV] = "1" if treatment else "0"
            mx.random.seed(0)
            draft_before, prework_before = draft_calls, prework_calls
            trial = await bench._run_trial(
                engine, mx, prompt, args.gen, index=index, warmup=warmup
            )
            record = asdict(trial)
            record.update(
                treatment=treatment,
                draft_calls=draft_calls - draft_before,
                prework_calls=prework_calls - prework_before,
            )
            result["trials"].append(record)
            expected = bench._validate_trial(trial, args.pp, expected)
            _validate_engagement(record)
        result["outputs_identical"] = (
            len({record["output_sha256"] for record in result["trials"]}) == 1
        )
        result["summary"] = {
            arm: statistics.median(
                record["generation_tps"]
                for record in result["trials"]
                if not record["warmup"] and record["treatment"] is (arm == "candidate")
            )
            for arm in ("baseline", "candidate")
        }
        result["status"] = "ok"
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        controller.observe = old_observe
        controller.should_exit = old_should_exit
        prework.gdn_prework_fused = old_prework
        if engine is not None and original_draft is not None:
            engine._adapter.mtp_forward = original_draft
        if depth_changed:
            if original_depth is MISSING:
                delattr(language_model, "_omlx_mtp_depth")
            else:
                language_model._omlx_mtp_depth = original_depth
        if old_env is None:
            os.environ.pop(PREWORK_ENV, None)
        else:
            os.environ[PREWORK_ENV] = old_env
        if before is not None:
            try:
                after = bench._model_fingerprint(args.model)
                result["model_fingerprint_after"] = after
                bench._assert_model_unchanged(before, after)
            except Exception as exc:
                result["status"] = "error"
                fingerprint_error = (
                    f"post-run fingerprint failed: {type(exc).__name__}: {exc}"
                )
                result["error"] = (
                    f"{result['error']}; {fingerprint_error}"
                    if "error" in result
                    else fingerprint_error
                )
        if engine is not None:
            try:
                await engine.stop()
            except Exception as exc:
                result["status"] = "error"
                cleanup = f"engine cleanup failed: {type(exc).__name__}: {exc}"
                result["error"] = (
                    f"{result['error']}; {cleanup}" if "error" in result else cleanup
                )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    with redirect_stdout(sys.stderr):
        result = asyncio.run(_run(args))
    print(json.dumps(result, separators=(",", ":")))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
