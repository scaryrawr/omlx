# SPDX-License-Identifier: Apache-2.0
"""GitHub Copilot CLI integration."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path

from omlx.integrations.base import Integration, IntegrationContext, IntegrationModel
from omlx.integrations.codex import is_codex_chat_model
from omlx.utils.install import get_cli_command_prefix


def _ignore_interrupt(signum, frame):
    pass


def copilot_provider_config(ctx: IntegrationContext) -> dict:
    """Build a process-scoped provider registry for available chat models."""
    models = {
        model.id: model
        for model in ctx.models
        if is_codex_chat_model(
            {"model_type": model.model_type, **ctx.models_status_map.get(model.id, {})}
        )
    }
    if ctx.model and ctx.model not in models and is_codex_chat_model(
        {"model_type": ctx.model_type, **ctx.models_status_map.get(ctx.model, {})}
    ):
        models[ctx.model] = IntegrationModel(
            id=ctx.model,
            context_window=ctx.context_window,
            model_type=ctx.model_type,
            reasoning=ctx.reasoning,
            reasoning_effort_options=ctx.reasoning_effort_options,
        )

    entries = []
    for model in models.values():
        status = ctx.models_status_map.get(model.id, {})
        context_window = model.context_window
        if not isinstance(context_window, int) or not 0 < context_window <= 4294967295:
            context_window = 32768
        supports: dict[str, bool | list[str]] = {"vision": model.model_type == "vlm"}
        entry: dict[str, object] = {
            "id": model.id,
            "provider": "omlx",
            "maxPromptTokens": context_window,
            "maxContextWindowTokens": context_window,
            "capabilities": {"supports": supports},
        }
        max_tokens = status.get("max_tokens")
        if model.id == ctx.model and ctx.max_tokens is not None:
            max_tokens = ctx.max_tokens
        if isinstance(max_tokens, int) and 0 < max_tokens <= 4294967295:
            entry["maxOutputTokens"] = max_tokens
        efforts = [
            effort
            for effort in model.reasoning_effort_options
            if effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max")
        ]
        if efforts and model.reasoning is not False:
            supports["reasoning_effort"] = efforts
        entries.append(entry)

    return {
        "providers": [
            {
                "name": "omlx",
                "type": "openai",
                "baseUrl": ctx.openai_base_url,
                "wireApi": "responses",
                "bearerToken": ctx.auth_token,
            }
        ],
        "models": entries,
    }


class CopilotIntegration(Integration):
    """Copilot integration using a private provider registry in offline mode."""

    def __init__(self):
        super().__init__(
            name="copilot",
            display_name="Copilot CLI",
            type="env_var",
            install_check="copilot",
            install_hint="npm install -g @github/copilot",
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        command = f"{get_cli_command_prefix()} launch copilot"
        return f"{command} --model {ctx.model}" if ctx.model else command

    def launch(self, ctx: IntegrationContext) -> None:
        env = self._scrubbed_env()
        for key in list(env):
            if key.startswith("COPILOT_PROVIDER_") or key == "COPILOT_MODEL":
                del env[key]
        env["COPILOT_OFFLINE"] = "true"

        args = ["copilot"]
        if ctx.model:
            args.extend(["--model", f"omlx/{ctx.model}"])
        args.extend(ctx.extra_args)

        # The registry contains a literal bearer token; retain it only while
        # the CLI and its subagents may need to read it.
        with tempfile.TemporaryDirectory(prefix="omlx-copilot-") as directory:
            config_path = Path(directory) / "providers.json"
            fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as config:
                json.dump(copilot_provider_config(ctx), config, ensure_ascii=False)
            env["COPILOT_PROVIDERS_CONFIG"] = str(config_path)
            # A caught handler resets on exec, so Copilot receives SIGINT while
            # the launcher waits without treating Ctrl+C as its own interrupt.
            previous_handler = signal.signal(signal.SIGINT, _ignore_interrupt)
            try:
                result = subprocess.run(args, env=env, check=False)
            finally:
                signal.signal(signal.SIGINT, previous_handler)
        if result.returncode:
            raise SystemExit(
                128 - result.returncode if result.returncode < 0 else result.returncode
            )
