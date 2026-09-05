# SPDX-License-Identifier: Apache-2.0
"""Deterministic serving-path decode benchmark for supported Qwen models."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROMPT_TEXT = (
    r"Complete this Python module. Continue with code only and do not stop "
    r"until the implementation is complete.\n\n"
)
SUPPORTED_MODEL_TYPES = frozenset(
    {"qwen3_5", "qwen3_5_moe", "qwen3_6", "qwen3_8", "qwen4_exp"}
)
PROMPT_BUILD_MAX_ATTEMPTS = 16
MAX_CONFIG_BYTES = 4 * 1024 * 1024
MODEL_METADATA_NAMES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
)
OMLX_CANDIDATE_ENV_PREFIXES = (
    "OMLX_FA256_",
    "OMLX_GDN_",
    "OMLX_M5_",
    "OMLX_MTP_",
    "OMLX_QWEN35_",
    "OMLX_QWEN4_",
    "OMLX_SDPA256_",
)


@dataclass(frozen=True)
class Trial:
    index: int
    warmup: bool
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    finish_reason: str | None
    output_token_ids: tuple[int, ...]
    output_sha256: str
    producer_first_s: float | None
    producer_last_s: float | None
    producer_duration_s: float | None
    consumer_duration_s: float
    generation_tps: float | None
    end_to_end_tps: float
    peak_memory_bytes: int
    errors: tuple[str, ...]


class BenchmarkError(RuntimeError):
    pass


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--pp", type=int, default=512)
    parser.add_argument("--gen", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--native-mtp", choices=("off", "on"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT)
    args = parser.parse_args(argv)
    for name in ("pp", "gen", "repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def _token_digest(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _config_model_type(config: dict[str, Any]) -> str:
    return str(config.get("model_type", "")).replace("-", "_").lower()


def _is_model_artifact(path: Path) -> bool:
    if path.name in MODEL_METADATA_NAMES:
        return True
    if path.name.endswith(".safetensors.index.json"):
        return True
    if path.suffix in {".safetensors", ".npz"}:
        return True
    return any(
        "mtp" in part.lower() or "sidecar" in part.lower() for part in path.parts
    )


def _model_fingerprint(model: str) -> dict[str, Any]:
    model_path = Path(model).expanduser().resolve(strict=True)
    if not model_path.is_dir():
        raise BenchmarkError(f"model path is not a directory: {model_path}")

    config_path = model_path / "config.json"
    config_stat = config_path.stat()
    if config_stat.st_size > MAX_CONFIG_BYTES:
        raise BenchmarkError(
            f"config.json is unexpectedly large: {config_stat.st_size} bytes"
        )
    config_bytes = config_path.read_bytes()
    config_stat_after = config_path.stat()
    if (
        config_stat_after.st_size != config_stat.st_size
        or config_stat_after.st_mtime_ns != config_stat.st_mtime_ns
    ):
        raise BenchmarkError("config.json changed while fingerprinting")
    try:
        config = json.loads(config_bytes)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid config.json: {exc}") from exc
    if not isinstance(config, dict):
        raise BenchmarkError("config.json root must be an object")
    model_type = _config_model_type(config)
    if model_type not in SUPPORTED_MODEL_TYPES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        raise BenchmarkError(
            f"unsupported model_type {model_type!r}; expected one of: {supported}"
        )

    files = []
    for path in sorted(model_path.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(model_path)
        if not _is_model_artifact(relative):
            continue
        stat = path.stat()
        files.append(
            {
                "path": relative.as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    if not any(item["path"].endswith(".safetensors") for item in files):
        raise BenchmarkError(f"model has no safetensors weights: {model_path}")

    fingerprint_payload = {
        "resolved_path": str(model_path),
        "model_type": model_type,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "files": files,
    }
    encoded = json.dumps(
        fingerprint_payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        **fingerprint_payload,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _assert_model_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    if after["sha256"] != before["sha256"]:
        raise BenchmarkError("model files changed during benchmark")


def _candidate_environment() -> dict[str, str]:
    return {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith(OMLX_CANDIDATE_ENV_PREFIXES)
    }


def _source_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "revision": revision,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _runtime_info(mx: Any) -> dict[str, Any]:
    from omlx._version import __version__ as omlx_version

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mlx_version": str(mx.__version__),
        "omlx_version": omlx_version,
        "mlx_device": json.loads(json.dumps(mx.device_info(), default=str)),
        "omlx_environment": _candidate_environment(),
        "source": _source_revision(),
    }


def _build_prompt(
    tokenizer: Any,
    target_tokens: int,
    prompt_text: str = DEFAULT_PROMPT_TEXT,
) -> tuple[int, ...]:
    from omlx.admin.benchmark import BenchmarkContextProfile, _load_bench_corpus

    corpus = _load_bench_corpus(BenchmarkContextProfile.CODE_PYTHON)
    target_chars = max(target_tokens * 4, 1)
    for _ in range(PROMPT_BUILD_MAX_ATTEMPTS):
        repeats = (target_chars + len(corpus) - 1) // len(corpus)
        body = (corpus * repeats)[:target_chars]
        token_ids = tuple(int(token) for token in tokenizer.encode(prompt_text + body))
        if len(token_ids) >= target_tokens:
            return token_ids[:target_tokens]
        if not token_ids:
            raise BenchmarkError("benchmark prompt tokenized to zero tokens")
        target_chars = max(
            target_chars + 1,
            (target_chars * target_tokens + len(token_ids) - 1) // len(token_ids) + 1,
        )
    raise BenchmarkError(
        f"could not build an exact {target_tokens}-token prompt after "
        f"{PROMPT_BUILD_MAX_ATTEMPTS} attempts"
    )


def _model_settings(native_mtp: bool) -> Any:
    from omlx.model_settings import ModelSettings

    return ModelSettings(
        mtp_enabled=native_mtp,
        vlm_mtp_enabled=False,
        dflash_enabled=False,
        dflash_in_memory_cache=False,
        dflash_ssd_cache=False,
        specprefill_enabled=False,
        turboquant_kv_enabled=False,
    )


async def _run_trial(
    engine: Any,
    mx: Any,
    prompt: tuple[int, ...],
    max_tokens: int,
    *,
    index: int,
    warmup: bool,
) -> Trial:
    mx.reset_peak_memory()

    started = time.perf_counter()
    first_generated_at: float | None = None
    last_generated_until: float | None = None
    terminal = None
    async for output in engine.stream_generate(
        prompt=list(prompt),
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        skip_cache_store=True,
    ):
        if output.generated_at is not None and first_generated_at is None:
            first_generated_at = float(output.generated_at)
        if output.generated_until is not None:
            last_generated_until = float(output.generated_until)
        terminal = output
    ended = time.perf_counter()

    peak_memory = int(mx.get_peak_memory())
    consumer_duration = ended - started
    if terminal is None:
        return Trial(
            index=index,
            warmup=warmup,
            prompt_tokens=0,
            completion_tokens=0,
            cached_tokens=0,
            finish_reason=None,
            output_token_ids=(),
            output_sha256=_token_digest(()),
            producer_first_s=first_generated_at,
            producer_last_s=last_generated_until,
            producer_duration_s=None,
            consumer_duration_s=consumer_duration,
            generation_tps=None,
            end_to_end_tps=0.0,
            peak_memory_bytes=peak_memory,
            errors=("produced no output",),
        )

    output_ids = tuple(int(token) for token in terminal.tokens)
    errors = []
    if not output_ids:
        errors.append("produced no terminal token IDs")
    if first_generated_at is None or last_generated_until is None:
        producer_duration = None
        errors.append("omitted producer timestamps")
    else:
        producer_duration = last_generated_until - first_generated_at
        if producer_duration <= 0:
            errors.append("has a non-positive producer duration")
    if not terminal.finished:
        errors.append("did not produce a terminal output")
    if terminal.completion_tokens != max_tokens or len(output_ids) != max_tokens:
        errors.append(
            f"ended early: completion_tokens={terminal.completion_tokens}, "
            f"token_ids={len(output_ids)}, expected={max_tokens}, "
            f"finish_reason={terminal.finish_reason!r}"
        )
    if terminal.cached_tokens:
        errors.append(f"reported {terminal.cached_tokens} cached prompt tokens")

    generation_tps = None
    if producer_duration is not None and producer_duration > 0:
        generation_tps = max(terminal.completion_tokens - 1, 1) / producer_duration
    return Trial(
        index=index,
        warmup=warmup,
        prompt_tokens=int(terminal.prompt_tokens),
        completion_tokens=int(terminal.completion_tokens),
        cached_tokens=int(terminal.cached_tokens),
        finish_reason=terminal.finish_reason,
        output_token_ids=output_ids,
        output_sha256=_token_digest(output_ids),
        producer_first_s=first_generated_at,
        producer_last_s=last_generated_until,
        producer_duration_s=producer_duration,
        consumer_duration_s=consumer_duration,
        generation_tps=generation_tps,
        end_to_end_tps=terminal.completion_tokens / consumer_duration,
        peak_memory_bytes=peak_memory,
        errors=tuple(errors),
    )


def _validate_trial(
    trial: Trial,
    prompt_tokens: int,
    expected_output: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if trial.errors:
        raise BenchmarkError(f"trial {trial.index} " + "; ".join(trial.errors))
    if trial.prompt_tokens != prompt_tokens:
        raise BenchmarkError(
            f"trial {trial.index} reported {trial.prompt_tokens} prompt tokens; "
            f"expected {prompt_tokens}"
        )
    if expected_output is not None and trial.output_token_ids != expected_output:
        raise BenchmarkError(f"trial {trial.index} output differs from the first trial")
    return trial.output_token_ids


async def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "label": args.label,
        "model": args.model,
        "native_mtp": args.native_mtp,
        "config": {
            "prompt_tokens": args.pp,
            "generation_tokens": args.gen,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "temperature": 0.0,
            "seed": 0,
            "prompt_text": args.prompt_text,
            "cache_store": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "trials": [],
    }
    engine = None
    trials: list[Trial] = []
    fingerprint_before = None
    try:
        fingerprint_before = _model_fingerprint(args.model)
        result["model_type"] = fingerprint_before["model_type"]
        result["model_fingerprint_before"] = fingerprint_before

        import mlx.core as mx

        from omlx.engine.vlm import VLMBatchedEngine
        from omlx.scheduler import SchedulerConfig

        result["runtime"] = _runtime_info(mx)
        engine = VLMBatchedEngine(
            model_name=args.model,
            model_settings=_model_settings(args.native_mtp == "on"),
            scheduler_config=SchedulerConfig(
                paged_ssd_cache_dir=None,
                hot_cache_max_size=0,
            ),
        )
        await engine.start()
        loaded_model_type = str(engine.model_type or "").replace("-", "_").lower()
        if loaded_model_type != fingerprint_before["model_type"]:
            raise BenchmarkError(
                f"loaded model_type {loaded_model_type!r} differs from config "
                f"{fingerprint_before['model_type']!r}"
            )

        prompt = _build_prompt(engine.tokenizer, args.pp, args.prompt_text)
        result["prompt"] = {
            "token_ids": list(prompt),
            "sha256": _token_digest(prompt),
        }
        expected_output: tuple[int, ...] | None = None
        for index in range(args.warmup + args.repeats):
            trial = await _run_trial(
                engine,
                mx,
                prompt,
                args.gen,
                index=index,
                warmup=index < args.warmup,
            )
            trials.append(trial)
            result["trials"] = [asdict(item) for item in trials]
            expected_output = _validate_trial(trial, args.pp, expected_output)
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if fingerprint_before is not None:
            try:
                fingerprint_after = _model_fingerprint(args.model)
                result["model_fingerprint_after"] = fingerprint_after
                _assert_model_unchanged(fingerprint_before, fingerprint_after)
            except Exception as exc:
                result["status"] = "error"
                result["error"] = (
                    f"{result['error']}; post-run fingerprint failed: "
                    f"{type(exc).__name__}: {exc}"
                    if "error" in result
                    else f"post-run fingerprint failed: {type(exc).__name__}: {exc}"
                )
        if engine is not None:
            try:
                await engine.stop()
            except Exception as exc:
                result["status"] = "error"
                cleanup_error = f"engine cleanup failed: {type(exc).__name__}: {exc}"
                result["error"] = (
                    f"{result['error']}; {cleanup_error}"
                    if "error" in result
                    else cleanup_error
                )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    with redirect_stdout(sys.stderr):
        result = asyncio.run(_benchmark(args))
    json.dump(result, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
