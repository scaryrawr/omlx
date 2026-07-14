# SPDX-License-Identifier: Apache-2.0
"""Canonical DeepSeek V4 chat encoder shim.

DSV4 bundles can ship ``encoding/encoding_dsv4.py`` with prompt-rendering
logic that is stricter than the tokenizer's Jinja template.  In particular it
handles reasoning effort tiers and prior-reasoning stripping.  This module
keeps that behavior scoped to DeepSeek V4 tokenizer loads.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

_encoding_cache: dict[str, Any] = {}


def _load_encoding_dsv4_module(
    *,
    model_path: str | Path | None = None,
    encoding_dir: str | Path | None = None,
) -> Any:
    """Locate and import the DSV4 ``encoding_dsv4.py`` module.

    Lookup order is intentionally narrow and DSV4-specific:
    1. explicit ``encoding_dir``;
    2. ``DSV4_ENCODING_DIR``;
    3. ``{model_path}/encoding``.
    """
    candidates: list[Path] = []
    if encoding_dir is not None:
        candidates.append(Path(encoding_dir).expanduser())
    if env_dir := os.environ.get("DSV4_ENCODING_DIR"):
        candidates.append(Path(env_dir).expanduser())
    if model_path is not None:
        candidates.append(Path(model_path).expanduser() / "encoding")

    seen: set[Path] = set()
    for directory in candidates:
        if directory in seen:
            continue
        seen.add(directory)
        source = directory / "encoding_dsv4.py"
        if not source.exists():
            continue
        spec = importlib.util.spec_from_file_location("encoding_dsv4", str(source))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["encoding_dsv4"] = module
        spec.loader.exec_module(module)
        logger.info("Loaded canonical DSV4 chat encoder from %s", source)
        return module

    raise RuntimeError(
        "DSV4 canonical chat encoder unavailable: no encoding_dsv4.py found "
        "via an explicit directory, DSV4_ENCODING_DIR, or {model_path}/encoding."
    )


def _get_encoding(model_path: str | Path | None = None) -> Any:
    env_dir = os.environ.get("DSV4_ENCODING_DIR")
    if model_path is not None:
        key = f"model:{Path(model_path).expanduser().resolve()}"
    elif env_dir:
        key = f"env:{Path(env_dir).expanduser().resolve()}"
    else:
        key = "default"
    if key not in _encoding_cache:
        _encoding_cache[key] = _load_encoding_dsv4_module(model_path=model_path)
    return _encoding_cache[key]


def _resolve_mode_and_effort(
    enable_thinking: bool | None,
    reasoning_effort: str | None,
    thinking_mode: str | None = None,
) -> tuple[str, str | None]:
    """Map oMLX/vMLX reasoning kwargs to DSV4 encoder kwargs."""
    if thinking_mode is not None:
        if thinking_mode not in {"chat", "thinking"}:
            raise ValueError(f"Invalid DSV4 thinking_mode: {thinking_mode!r}")
        if thinking_mode == "chat":
            return "chat", None
        if reasoning_effort == "max":
            return "thinking", "max"
        effort = "high" if reasoning_effort in {"low", "medium", "high"} else None
        return "thinking", effort

    if reasoning_effort == "max":
        return "thinking", "max"
    if enable_thinking is False:
        return "chat", None
    if enable_thinking is True:
        effort = "high" if reasoning_effort in {"low", "medium", "high"} else None
        return "thinking", effort
    if reasoning_effort in {"low", "medium", "high"}:
        return "thinking", "high"
    return "chat", None


def _normalize_tool_call_arguments(messages: list[dict[str, Any]]) -> None:
    for msg in messages:
        for tool_call in msg.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if isinstance(function, dict) and isinstance(function.get("arguments"), dict):
                function["arguments"] = json.dumps(
                    function["arguments"], ensure_ascii=False
                )


def _inject_tools(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    if not tools or not messages:
        return messages

    if any(m.get("role") == "system" and m.get("tools") for m in messages):
        return messages

    if messages[0].get("role") == "system":
        first = dict(messages[0])
        first["tools"] = tools
        return [first, *messages[1:]]

    return [{"role": "system", "content": "", "tools": tools}, *messages]


def apply_chat_template(
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    thinking_mode: str | None = None,
    drop_earlier_reasoning: bool = True,
    tools: list[dict[str, Any]] | None = None,
    add_default_bos_token: bool = True,
    context: list[dict[str, Any]] | None = None,
    model_path: str | Path | None = None,
    **kwargs: Any,
) -> str:
    """Render a DSV4 prompt with the bundle's canonical encoder."""
    if "drop_thinking" in kwargs and "drop_earlier_reasoning" not in kwargs:
        drop_earlier_reasoning = bool(kwargs.pop("drop_thinking"))
    kwargs.pop("add_generation_prompt", None)
    kwargs.pop("continue_final_message", None)
    kwargs.pop("chat_template", None)
    kwargs.pop("tokenize", None)

    chat_template_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
    if enable_thinking is None and "enable_thinking" in chat_template_kwargs:
        enable_thinking = chat_template_kwargs["enable_thinking"]
    if reasoning_effort is None and "reasoning_effort" in chat_template_kwargs:
        reasoning_effort = chat_template_kwargs["reasoning_effort"]
    if thinking_mode is None and "thinking_mode" in chat_template_kwargs:
        thinking_mode = chat_template_kwargs["thinking_mode"]

    mode, effort = _resolve_mode_and_effort(
        enable_thinking, reasoning_effort, thinking_mode
    )
    prepared = copy.deepcopy(messages)
    _normalize_tool_call_arguments(prepared)
    prepared = _inject_tools(prepared, tools)

    encoder = _get_encoding(model_path)
    return cast(str, encoder.encode_messages(
        prepared,
        thinking_mode=mode,
        context=context,
        drop_thinking=drop_earlier_reasoning,
        add_default_bos_token=add_default_bos_token,
        reasoning_effort=effort,
    ))


def install_canonical_chat_template(tokenizer: Any, model_path: str | Path) -> bool:
    """Install the canonical encoder on a DSV4 TokenizerWrapper once."""
    if getattr(tokenizer, "_omlx_dsv4_chat_template_shim", False):
        return False

    _get_encoding(model_path)
    original = getattr(tokenizer, "_chat_template", None)

    def _canonical_template(messages, *args, **kwargs):
        if args:
            # mlx-lm calls custom templates with kwargs, but tolerate the
            # common positional add_generation_prompt value used by HF callers.
            kwargs.setdefault("add_generation_prompt", args[0])
        return apply_chat_template(messages, model_path=model_path, **kwargs)

    tokenizer._chat_template = _canonical_template
    tokenizer.has_chat_template = True
    tokenizer._omlx_dsv4_chat_template_shim = True
    tokenizer._omlx_dsv4_chat_template_orig = original
    logger.info(
        "Installed canonical DSV4 chat-template shim for %s", model_path
    )
    return True
