# SPDX-License-Identifier: Apache-2.0
"""Codex (OpenAI Codex CLI) integration."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from omlx.integrations.base import (
    Integration,
    IntegrationContext,
    IntegrationModel,
    chat_model_ids,
)
from omlx.model_classification import EMBEDDING_MODEL_TYPES, is_helper_config_model_type
from omlx.utils.install import get_cli_command_prefix

CODEX_CONFIG_PATH = (
    Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml"
)

_CODEX_INSTRUCTIONS = (
    "You are a coding agent working in the user's local workspace. "
    "Follow the user's instructions and applicable AGENTS.md files. "
    "Inspect the workspace before making focused, maintainable changes. "
    "Preserve unrelated work, verify changes with relevant tests, and report limitations."
)


def _is_reasoning_model(model_id: str, reasoning: bool | None) -> bool:
    if reasoning is not None:
        return reasoning
    return bool(re.search(r"\b(thinking|o1|o3|r1)\b", model_id.lower()))


def is_codex_chat_model(info: dict) -> bool:
    """Exclude non-chat checkpoints even if the server reports a chat engine."""
    config_type = info.get("config_model_type") or ""
    normalized_type = config_type.lower().replace("-", "_")
    return (
        info.get("model_type") in (None, "llm", "vlm")
        and info.get("engine_type") in (None, "batched", "vlm")
        and not info.get("is_helper")
        and not info.get("is_hidden")
        and not info.get("unavailable_reason")
        and not is_helper_config_model_type(config_type)
        and normalized_type not in EMBEDDING_MODEL_TYPES
    )


def codex_model_catalog(
    ctx: IntegrationContext,
    model_ids: list[str] | None = None,
    instructions: str = _CODEX_INSTRUCTIONS,
) -> dict:
    """Build Codex's ModelInfo catalog, not an OpenAI /v1/models response.

    Keep capabilities conservative: thinking does not imply adjustable effort,
    reasoning-summary generation, verbosity control, or server-side search.
    """
    if model_ids is not None:
        known_models = {model.id: model for model in ctx.models}
        models_from_status = []
        for model_id in model_ids:
            info = ctx.models_status_map.get(model_id, {})
            models_from_status.append(
                known_models.get(model_id)
                or IntegrationModel(
                    id=model_id,
                    context_window=info.get("max_context_window"),
                    model_type=info.get("model_type"),
                    reasoning=info.get("reasoning"),
                    reasoning_effort_options=tuple(
                        level
                        for level in info.get("reasoning_levels") or ()
                        if level not in ("off", "on")
                    ),
                )
            )
        ctx = replace(ctx, models=tuple(models_from_status))
    models = {
        model.id: model
        for model in ctx.models
        if is_codex_chat_model(
            {"model_type": model.model_type, **ctx.models_status_map.get(model.id, {})}
        )
    }
    if (
        ctx.model
        and ctx.model not in models
        and is_codex_chat_model(
            {"model_type": ctx.model_type, **ctx.models_status_map.get(ctx.model, {})}
        )
    ):
        models[ctx.model] = IntegrationModel(
            id=ctx.model,
            context_window=ctx.context_window,
            model_type=ctx.model_type,
            reasoning=ctx.reasoning,
            reasoning_effort_options=ctx.reasoning_effort_options,
            reasoning_effort_default=ctx.reasoning_effort_default,
        )
    ordered = sorted(models.values(), key=lambda model: model.id != ctx.model)
    entries = []
    for priority, model in enumerate(ordered):
        reasoning = _is_reasoning_model(model.id, model.reasoning)
        # Codex accepts a fixed effort enum. Do not invent levels from a
        # thinking boolean, or advertise custom template values it cannot send.
        efforts = (
            [
                effort
                for effort in (
                    "none",
                    "minimal",
                    "low",
                    "medium",
                    "high",
                    "xhigh",
                    "max",
                    "ultra",
                )
                if effort in model.reasoning_effort_options
            ]
            if model.reasoning is not False
            else []
        )
        default_effort = (
            "medium" if "medium" in efforts else model.reasoning_effort_default
        )
        if efforts:
            if default_effort not in efforts:
                default_effort = efforts[0]
            reasoning_levels = [
                {"effort": effort, "description": f"Use {effort} reasoning effort"}
                for effort in efforts
            ]
        else:
            default_effort = "high" if reasoning else None
            reasoning_levels = (
                [{"effort": "high", "description": "Use model thinking"}]
                if reasoning
                else []
            )
        # Status can be unavailable for inference-only API keys. Do not let
        # Codex's large GPT context default overstate an unknown local limit.
        context_window = (
            model.context_window
            if model.context_window is not None and model.context_window > 0
            else 32768
        )
        entries.append(
            {
                "slug": model.id,
                "display_name": model.id,
                "description": "Local model served by oMLX",
                "visibility": "list",
                "supported_in_api": True,
                "priority": priority,
                "base_instructions": instructions,
                "shell_type": "shell_command",
                "context_window": context_window,
                "input_modalities": (
                    ["text", "image"] if model.model_type == "vlm" else ["text"]
                ),
                "default_reasoning_level": default_effort,
                "supported_reasoning_levels": reasoning_levels,
                "supports_reasoning_summaries": False,
                "supports_reasoning_summary_parameter": False,
                "support_verbosity": False,
                "supports_parallel_tool_calls": False,
                "supports_search_tool": False,
                "prefer_websockets": False,
                "experimental_supported_tools": [],
                "truncation_policy": {
                    "mode": "tokens",
                    "limit": min(10000, context_window // 4),
                },
            }
        )
    return {"models": entries}


# A slug Codex does not know, so it renders its built-in instructions.
_PROBE_MODEL = "omlx-instructions-probe"
_PROBE_TIMEOUT_SECONDS = 10.0
# Older Codex releases reject these catalog entries and do not start.
_MIN_CODEX_VERSION = (0, 148)

FALLBACK_INSTRUCTIONS = """\
You are a coding agent running in the Codex CLI on the user's machine. A local model served by oMLX powers you.

Keep working until the user's request is fully handled, using the tools available to you. Look at the relevant files before you change anything, and do not guess at facts you can check.

AGENTS.md files give instructions for the directory tree that contains them. Follow every AGENTS.md whose tree covers a file you change. A deeper AGENTS.md overrides a higher one, and direct instructions from the user override both.

To edit files, run apply_patch through the shell tool with the patch in a heredoc:

apply_patch <<'EOF'
*** Begin Patch
*** Update File: src/greeting.py
@@ def greet(name):
-    return "hi"
+    return f"Hello, {name}!"
*** End Patch
EOF

In an Update File section, each @@ line names the code the change sits in. Lines starting with a space are unchanged context, "-" removes a line, and "+" adds one. Use "*** Add File: <path>" with every line starting with "+" to create a file, and "*** Delete File: <path>" to remove one. One patch can hold several file sections.

Guidelines:
- Fix the root cause. Leave unrelated bugs and failing tests alone, and mention them instead.
- Keep changes small and in the style of the surrounding code.
- Do not commit, create branches, or add license headers unless the user asks.
- If the project has tests, run the most specific relevant ones first.
- Before a group of tool calls, tell the user in one short sentence what you are about to do.

When you finish, reply briefly (about 10 lines unless the task needs more detail). Refer to files as `path` or `path:line` so the user can open them. Use short "-" bullets only when they help, and do not print ANSI escape codes.
"""


def _provider_overrides(base_url: str) -> list[tuple[str, str]]:
    return [
        ("model_provider", json.dumps("omlx")),
        ("model_providers.omlx.name", json.dumps("oMLX")),
        ("model_providers.omlx.base_url", json.dumps(base_url)),
        ("model_providers.omlx.env_key", json.dumps("OMLX_API_KEY")),
        ("model_providers.omlx.wire_api", json.dumps("responses")),
        ("model_providers.omlx.requires_openai_auth", "false"),
        ("model_providers.omlx.supports_websockets", "false"),
    ]


def _as_cli_args(overrides: list[tuple[str, str]]) -> list[str]:
    return [arg for key, value in overrides for arg in ("-c", f"{key}={value}")]


def capture_codex_instructions(codex_bin: str, env: dict[str, str]) -> str | None:
    """Return the base instructions ``codex_bin`` sends, or None.

    Codex runs once with an empty CODEX_HOME against a loopback listener, so
    the probe needs no login, loads no model, and leaves no user session.
    """
    bodies: list[bytes] = []
    received = threading.Event()

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            bodies.append(self.rfile.read(length))
            received.set()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "oMLX instructions probe"}}')

        def log_message(self, format: str, *args) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    overrides = _provider_overrides(f"http://127.0.0.1:{server.server_port}/v1") + [
        ("model_providers.omlx.request_max_retries", "0"),
        ("model_providers.omlx.stream_max_retries", "0"),
    ]
    try:
        with tempfile.TemporaryDirectory(
            prefix="omlx-codex-probe-", ignore_cleanup_errors=True
        ) as home:
            try:
                proc = subprocess.Popen(
                    [
                        codex_bin,
                        "exec",
                        "--skip-git-repo-check",
                        *_as_cli_args(overrides),
                        "-m",
                        _PROBE_MODEL,
                        "probe",
                    ],
                    cwd=home,
                    env={**env, "CODEX_HOME": home, "OMLX_API_KEY": "probe"},
                    # codex exec waits for more prompt text on an open stdin.
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError:
                return None
            deadline = time.monotonic() + _PROBE_TIMEOUT_SECONDS
            while not received.wait(0.05):
                if proc.poll() is not None or time.monotonic() > deadline:
                    break
            with contextlib.suppress(subprocess.TimeoutExpired):
                # Codex exits on the error answer.
                proc.wait(timeout=5 if received.is_set() else 0)
            # The npm codex shim does not forward SIGKILL, so kill the whole
            # group before the temporary home is removed.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    finally:
        server.shutdown()
        server.server_close()

    if not bodies:
        return None
    try:
        instructions = json.loads(bodies[0]).get("instructions")
    except (ValueError, AttributeError):
        return None
    if isinstance(instructions, str) and instructions.strip():
        return instructions
    return None


def check_codex_version(codex_bin: str | None) -> None:
    """Exit with an update hint when ``codex_bin`` cannot read the catalog."""
    if codex_bin is None:
        return
    try:
        version = subprocess.run(
            [codex_bin, "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return
    match = re.search(r"(\d+)\.(\d+)\.\d+", version)
    if match and tuple(map(int, match.groups())) < _MIN_CODEX_VERSION:
        print(
            f"{version.strip()} cannot read oMLX's model catalog. "
            "Update Codex to 0.148 or newer."
        )
        sys.exit(1)


def codex_instructions(codex_bin: str | None, env: dict[str, str]) -> str:
    instructions = capture_codex_instructions(codex_bin, env) if codex_bin else None
    if instructions is None:
        print(
            f"Warning: could not read the built-in instructions of "
            f"{codex_bin or 'codex'}; using oMLX's fallback coding instructions."
        )
        return FALLBACK_INSTRUCTIONS
    return instructions


def resolve_codex_models(ctx: IntegrationContext) -> tuple[list[str], str]:
    """Return the catalog model ids and the model Codex starts with."""
    model_ids = [
        model_id
        for model_id in chat_model_ids(ctx)
        if is_codex_chat_model(ctx.models_status_map.get(model_id, {}))
    ]
    if not model_ids:
        print("oMLX server reported no chat models. Load a model first.")
        sys.exit(1)

    def served_id(model_id: str) -> str:
        return ctx.models_status_map.get(model_id, {}).get("model_alias") or model_id

    model = served_id(ctx.model)
    if model and model not in model_ids:
        print(f"Warning: {ctx.model} is not a chat model on the server; ignoring it.")
        model = ""
    if not model:
        default = served_id(ctx.default_model)
        model = default if default in model_ids else model_ids[0]
    return model_ids, model


def write_codex_model_catalog(
    codex_dir: Path,
    tool: str,
    ctx: IntegrationContext,
    model_ids: list[str],
    instructions: str,
) -> Path:
    # Per tool and endpoint, so a CLI launch or another server does not
    # rewrite the catalog a saved App config points to.
    endpoint_id = hashlib.sha256(ctx.openai_base_url.encode()).hexdigest()[:12]
    catalog_path = codex_dir / f"omlx-{tool}-models-{endpoint_id}.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog = codex_model_catalog(ctx, model_ids, instructions)
    # Codex does not start on a partially written catalog.
    temporary = catalog_path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, catalog_path)
    return catalog_path


def write_codex_config(
    config_path: Path, ctx: IntegrationContext, catalog_path: Path | None = None
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing_content = ""
    if config_path.exists():
        # Create backup
        timestamp = time.time_ns()
        backup = config_path.with_suffix(f".{timestamp}.bak")
        shutil.copy2(config_path, backup)
        existing_content = config_path.read_text(encoding="utf-8")
        print(f"Backup: {backup}")

    # Refuse to rewrite invalid TOML, and back up before changing the catalog.
    tomllib.loads(existing_content)
    if catalog_path is None:
        catalog_path = write_codex_model_catalog(
            config_path.parent, "codex-app", ctx, [ctx.model], _CODEX_INSTRUCTIONS
        )

    # Parse existing config lines to preserve other settings
    lines = existing_content.splitlines()
    new_lines = []
    in_any_section = False
    in_omlx_section = False

    # Model-specific limits and capabilities belong in the catalog so switching
    # models does not retain the launch model's context or thinking settings.
    top_level_overrides = {
        "model": json.dumps(ctx.model or "select-a-model", ensure_ascii=False),
        "model_provider": '"omlx"',
        "model_catalog_json": json.dumps(str(catalog_path), ensure_ascii=False),
    }
    managed_keys = {
        "model_context_window",
        "model_auto_compact_token_limit",
        "model_reasoning_effort",
        "model_reasoning_summary",
        "model_supports_reasoning_summaries",
    }

    seen_keys = set()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_any_section = True
            in_omlx_section = bool(
                re.fullmatch(
                    r"\[model_providers\.omlx(?:\.[^\]]+)?\]\s*(?:#.*)?", stripped
                )
            )

        # Handle top-level keys
        if not in_any_section and "=" in stripped:
            key = stripped.split("=")[0].strip()
            if key in top_level_overrides:
                new_lines.append(f"{key} = {top_level_overrides[key]}")
                seen_keys.add(key)
                continue
            if key in managed_keys:
                continue

        # Skip old oMLX section
        if in_omlx_section:
            continue

        new_lines.append(line)

    # Add missing top-level keys
    for key, val in top_level_overrides.items():
        if key not in seen_keys:
            new_lines.insert(0, f"{key} = {val}")

    # Append new oMLX provider section
    new_lines.append("\n[model_providers.omlx]")
    new_lines.append('name = "oMLX"')
    new_lines.append(f"base_url = {json.dumps(ctx.openai_base_url)}")
    new_lines.append('env_key = "OMLX_API_KEY"')
    new_lines.append('wire_api = "responses"')
    new_lines.append("requires_openai_auth = false")
    new_lines.append("supports_websockets = false")

    new_content = "\n".join(new_lines) + "\n"
    tomllib.loads(new_content)
    config_path.write_text(new_content, encoding="utf-8")
    print(f"Config updated: {config_path}")


def codex_config_args(ctx: IntegrationContext, catalog_path: Path) -> list[str]:
    """Build process-scoped Codex config overrides for an oMLX launch."""
    overrides = _provider_overrides(ctx.openai_base_url)
    # No global model_context_window: it would override every model's limit.
    overrides.append(("model_catalog_json", json.dumps(str(catalog_path))))
    return _as_cli_args(overrides)


class CodexIntegration(Integration):
    """Codex integration using process-scoped configuration for oMLX."""

    def __init__(self):
        super().__init__(
            name="codex",
            display_name="Codex",
            type="env_var",
            install_check="codex",
            install_hint="npm install -g @openai/codex",
            # Codex switches models with /model; --model picks the first one.
            requires_model_selection=False,
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        cmd = f"{get_cli_command_prefix()} launch codex"
        return f"{cmd} --model {ctx.model}" if ctx.model else cmd

    def configure(self, ctx: IntegrationContext) -> None:
        # Launch-time arguments carry the oMLX settings. Keeping this a no-op
        # ensures normal Codex sessions continue to use the user's config.
        return None

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)

        env = self._scrubbed_env()
        model_ids, model = resolve_codex_models(ctx)
        codex_bin = shutil.which("codex", path=env.get("PATH"))
        check_codex_version(codex_bin)
        catalog_path = write_codex_model_catalog(
            CODEX_CONFIG_PATH.parent,
            "codex",
            ctx,
            model_ids,
            codex_instructions(codex_bin, env),
        )
        env["OMLX_API_KEY"] = ctx.auth_token

        args = ["codex", *codex_config_args(ctx, catalog_path), "-m", model]
        args.extend(ctx.extra_args)
        print(f"Starting with {model}; /model switches among {len(model_ids)} models")

        os.execvpe("codex", args, env)
