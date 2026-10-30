# SPDX-License-Identifier: Apache-2.0
"""GitHub Copilot CLI integration."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from collections.abc import Mapping
from pathlib import Path

from omlx.integrations.base import (
    Integration,
    IntegrationContext,
    IntegrationModel,
    backup_then_write,
)
from omlx.integrations.codex import is_codex_chat_model
from omlx.utils.install import get_cli_command_prefix

PROVIDER_NAME = "omlx"


def _ignore_interrupt(signum, frame):
    pass


def copilot_providers_config_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the CLI providers file (``COPILOT_PROVIDERS_CONFIG`` or default)."""
    resolved: Mapping[str, str] = os.environ if env is None else env
    override = resolved.get("COPILOT_PROVIDERS_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".copilot" / "providers.json"


def _clobber_omlx_entries(entries: list, key: str, replacement: list) -> list:
    """Drop omlx-owned entries, inserting ``replacement`` at the first slot.

    Entries owned by other providers keep their original order untouched.
    """
    result: list = []
    inserted = False
    for entry in entries:
        if isinstance(entry, dict) and entry.get(key) == PROVIDER_NAME:
            if not inserted:
                result.extend(replacement)
                inserted = True
            continue
        result.append(entry)
    if not inserted:
        result.extend(replacement)
    return result


def _restrict_permissions(path: Path) -> None:
    # The registry stores a bearer token in plaintext (per CLI docs).
    try:
        os.chmod(path, 0o600)
    except OSError as e:
        print(f"Warning: could not restrict permissions on {path}: {e}")


def copilot_provider_config(ctx: IntegrationContext) -> dict:
    """Build the omlx provider/model entries for the CLI providers registry."""
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
            "provider": PROVIDER_NAME,
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
                "name": PROVIDER_NAME,
                "type": "openai",
                "baseUrl": ctx.openai_base_url,
                "wireApi": "responses",
                "bearerToken": ctx.auth_token,
            }
        ],
        "models": entries,
    }


class CopilotIntegration(Integration):
    """Copilot integration launching offline via the persistent provider registry."""

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

    def configure(self, ctx: IntegrationContext) -> None:
        """Merge the omlx provider into the CLI providers file.

        Foreign providers/models are preserved verbatim; only entries owned
        by omlx (provider name/id match) are replaced so relaunches keep the
        registry correct as models, limits, or the API key change.
        """
        path = copilot_providers_config_path()
        registry = copilot_provider_config(ctx)
        existing_text: str | None = None
        existing: object = None
        if path.is_file():
            try:
                existing_text = path.read_text(encoding="utf-8")
                existing = json.loads(existing_text)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                print(f"Warning: could not parse {path}: {e}")
        if existing is not None and not isinstance(existing, dict):
            print(f"Warning: {path} does not contain a JSON object; replacing it.")
        doc = dict(existing) if isinstance(existing, dict) else {}
        providers = doc.get("providers")
        models = doc.get("models")
        doc["providers"] = _clobber_omlx_entries(
            providers if isinstance(providers, list) else [],
            "name",
            registry["providers"],
        )
        doc["models"] = _clobber_omlx_entries(
            models if isinstance(models, list) else [],
            "provider",
            registry["models"],
        )
        payload = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
        if existing_text == payload:
            # Already up to date: avoid a backup file on every relaunch.
            _restrict_permissions(path)
            return
        backup_then_write(
            path, payload, note="Copilot providers config", mode=0o600
        )
        # An existing file keeps its own mode; the bearer token demands 0600.
        _restrict_permissions(path)

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)
        env = self._scrubbed_env()
        config_path = copilot_providers_config_path(env)
        for key in list(env):
            if key.startswith("COPILOT_PROVIDER_") or key == "COPILOT_MODEL":
                del env[key]
        env["COPILOT_OFFLINE"] = "true"
        # configure() rewrote the registry at this path; point the CLI at it
        # explicitly so a prior override value stays honored.
        env["COPILOT_PROVIDERS_CONFIG"] = str(config_path)

        args = ["copilot"]
        if ctx.model:
            args.extend(["--model", f"{PROVIDER_NAME}/{ctx.model}"])
        args.extend(ctx.extra_args)

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
