# SPDX-License-Identifier: Apache-2.0
"""OpenCode integration."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import json5

from omlx.integrations.base import (
    Integration,
    IntegrationContext,
    IntegrationModel,
    backup_then_write,
)
from omlx.integrations.codex import is_codex_chat_model
from omlx.utils.install import get_cli_command_prefix


def _major_version() -> int:
    try:
        result = subprocess.run(
            ["opencode", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        match = re.search(r"\b(?:v)?(\d+)\.\d+", result.stdout)
        if match:
            return int(match.group(1))
    except (OSError, subprocess.SubprocessError):
        pass
    raise RuntimeError("Cannot determine OpenCode version; config was not changed.")


def _config_path() -> Path:
    config_directory = os.environ.get("OPENCODE_CONFIG_DIR")
    if config_directory is not None:
        directory = Path(config_directory)
    else:
        config_home = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        directory = config_home / "opencode"
    jsonc = directory / "opencode.jsonc"
    return jsonc if jsonc.exists() or jsonc.is_symlink() else directory / "opencode.json"


def opencode_provider_config(ctx: IntegrationContext) -> dict:
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
        )
    entries = {}
    for model in models.values():
        inputs = ["text", "image"] if model.model_type == "vlm" else ["text"]
        if ctx.models_status_map.get(model.id, {}).get("supports_audio_input") is True:
            inputs.append("audio")
        entry: dict = {
            "name": model.id,
            "capabilities": {"input": inputs, "output": ["text"], "tools": True},
        }
        if model.reasoning is not None:
            entry["capabilities"]["reasoning"] = {"supported": model.reasoning}
        window = model.context_window
        max_tokens = ctx.models_status_map.get(model.id, {}).get("max_tokens")
        if model.id == ctx.model and ctx.max_tokens is not None:
            max_tokens = ctx.max_tokens
        if isinstance(window, int) and not isinstance(window, bool) and window > 1:
            output_limit = window * 3 // 4
            if isinstance(max_tokens, int) and not isinstance(max_tokens, bool):
                requested_output = max_tokens if max_tokens > 0 else 2048
            else:
                requested_output = 2048
            entry["limit"] = {
                "context": window,
                "output": min(requested_output, output_limit),
            }
        elif window is not None:
            print(
                f"Warning: context window for {model.id} is unusable; "
                "preserving any existing OpenCode limits."
            )
        entries[model.id] = entry
    return {
        "name": "oMLX",
        "package": "@opencode/ai/providers/openai-compatible",
        "settings": {"baseURL": ctx.openai_base_url, "apiKey": ctx.auth_token},
        "models": entries,
    }


def _merge_config(existing: dict, updates: dict) -> dict:
    merged = existing.copy()
    for key, value in updates.items():
        if isinstance(value, dict):
            previous = existing.get(key, {})
            if not isinstance(previous, dict):
                raise ValueError(f"{key} must be an object")
            merged[key] = _merge_config(previous, value)
        else:
            merged[key] = value
    return merged


def _read_config(path: Path) -> str | None:
    try:
        entry = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(entry.st_mode):
        raise ValueError(f"Refusing to overwrite symlink: {path}")
    if (
        not stat.S_ISREG(entry.st_mode)
        or entry.st_uid != os.geteuid()
        or entry.st_nlink != 1
    ):
        raise ValueError(
            f"OpenCode config must be an owned, singly linked regular file: {path}"
        )
    return path.read_text(encoding="utf-8")


def _write_config(path: Path, text: str, original: str | None) -> None:
    if original is not None:
        path.chmod(0o600)
    if original == text:
        return
    backup_then_write(path, text, note="OpenCode config", mode=0o600)
    path.chmod(0o600)


def _preserve_model_metadata(
    existing: dict, provider: dict, ctx: IntegrationContext
) -> None:
    providers = existing.get("providers")
    if not isinstance(providers, dict) or not isinstance(providers.get("omlx"), dict):
        return
    old_models = providers["omlx"].get("models")
    if not isinstance(old_models, dict):
        return
    model_types = {model.id: model.model_type for model in ctx.models}
    if ctx.model and ctx.model not in model_types:
        model_types[ctx.model] = ctx.model_type
    for model_id, update in provider["models"].items():
        old = old_models.get(model_id)
        if not isinstance(old, dict):
            continue
        capabilities = update["capabilities"]
        previous = old.get("capabilities")
        retained = []
        if "limit" not in update and "limit" in old:
            retained.append("limit")
        if (
            "reasoning" not in capabilities
            and isinstance(previous, dict)
            and "reasoning" in previous
        ):
            retained.append("capabilities.reasoning")
        if retained:
            print(
                f"Warning: no current metadata for {model_id} "
                f"({', '.join(retained)}); preserving configured values, which may be stale."
            )
        if not isinstance(previous, dict):
            continue
        if isinstance(previous.get("tools"), bool):
            capabilities.pop("tools")
        audio_input = ctx.models_status_map.get(model_id, {}).get("supports_audio_input")
        if model_types.get(model_id) is not None:
            if (
                not isinstance(audio_input, bool)
                and isinstance(previous.get("input"), list)
                and "audio" in previous["input"]
            ):
                capabilities["input"].append("audio")
            continue
        if "input" not in previous:
            continue
        inputs = previous["input"]
        if isinstance(inputs, list) and isinstance(audio_input, bool):
            capabilities["input"] = [item for item in inputs if item != "audio"]
            if audio_input:
                capabilities["input"].append("audio")
        else:
            capabilities.pop("input")
        print(
            f"Warning: no current model type for {model_id}; "
            "preserving configured input capabilities, which may be stale."
        )


class OpenCodeIntegration(Integration):
    """Register the oMLX model catalog in OpenCode's global config."""

    CONFIG_PATH: Path | None = None

    def __init__(self):
        super().__init__(
            name="opencode",
            display_name="OpenCode",
            type="config_file",
            install_check="opencode",
            install_hint="curl -fsSL https://opencode.ai/install | bash",
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        command = f"{get_cli_command_prefix()} launch opencode"
        return f"{command} --model {ctx.model}" if ctx.model else command

    def configure(self, ctx: IntegrationContext) -> None:
        major = _major_version()
        if major != 2:
            raise ValueError(
                f"Unsupported OpenCode major version {major}; "
                "OpenCode 2 is required. Config was not changed."
            )
        path = self.CONFIG_PATH or _config_path()
        original = _read_config(path)
        try:
            existing = json5.loads(
                original if original is not None else "{}", allow_duplicate_keys=False
            )
            if not isinstance(existing, dict):
                raise ValueError("config must contain one JSON object")
            provider = opencode_provider_config(ctx)
            updates: dict[str, object] = {"providers": {"omlx": provider}}
            if ctx.model and ctx.model in provider["models"]:
                updates["model"] = f"omlx/{ctx.model}"
            _preserve_model_metadata(existing, provider, ctx)
            updated = json.dumps(
                _merge_config(existing, updates),
                indent=2,
                allow_nan=False,
            ) + "\n"
        except ValueError as exc:
            raise ValueError(
                f"Cannot update {path}: {exc}. Config was not changed."
            ) from exc
        if ctx.models_discovery_succeeded is False:
            print(
                "Warning: model discovery failed; preserving existing OpenCode models."
            )
        elif ctx.model and ctx.model not in {model.id for model in ctx.models}:
            print(
                f"Warning: model '{ctx.model}' was not in the chat catalog; "
                "registering it with available metadata."
            )
        if any(
            ctx.models_status_map.get(model_id, {}).get("model_type") == "vlm"
            and "supports_audio_input" not in ctx.models_status_map[model_id]
            for model_id in provider["models"]
        ):
            print(
                "Warning: the running oMLX server does not report audio-input "
                "capabilities. Restart it with the updated oMLX code, then run "
                "'omlx launch opencode' again; updating the launcher alone is not enough."
            )
        _write_config(path, updated, original)
        options = (
            ctx.extra_args[: ctx.extra_args.index("--")]
            if "--" in ctx.extra_args
            else ctx.extra_args
        )
        if any(arg == "--server" or arg.startswith("--server=") for arg in options):
            print(
                "The selected OpenCode server may not use this local configuration "
                "or its updated catalog; no server reload was attempted."
            )
        elif "--standalone" not in options:
            print(
                "OpenCode's running shared service may use a cached catalog. "
                "Run 'opencode reload' to refresh it, or launch with -- --standalone. "
                "Reload cancels pending permissions and forms across all loaded projects."
            )

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)
        env = self._scrubbed_env()
        args = ["opencode", *ctx.extra_args]
        sys.stdout.flush()
        sys.stderr.flush()
        os.execvpe("opencode", args, env)
