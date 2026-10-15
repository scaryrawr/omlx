# SPDX-License-Identifier: Apache-2.0
"""Guarded serving-only bridge to the reviewed native MiniCPM-V 4.7 source.

The model, processor, exact verifier and helper are authoritative native code.
oMLX supplies request-local positions, captures and sampling transport; native
mlx-vlm CLI generation is deliberately not upgraded by this bridge.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import sys
import threading
from contextvars import ContextVar
from pathlib import Path
from types import FunctionType, ModuleType

from .vendor_sync import EXPERT_SYMBOLS, UTILS_FUNCTIONS

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).parent
_VENDOR = _ROOT / "vendor"
_MODEL_PREFIX = "mlx_vlm.models.minicpmv4_7"
_HELPER_PREFIX = "mlx_vlm.speculative.drafters._omlx_minicpm47_dspark"
_LOCK = threading.RLock()
_HELPER_LOADING = ContextVar("omlx_minicpm47_helper_loading", default=False)
_STATE: str | None = None
_OWNED_MODULES: dict[str, ModuleType] = {}
_OWNED_SYMBOLS: tuple[tuple[ModuleType, str, object], ...] = ()


def _hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _inventory() -> dict:
    manifest = json.loads((_ROOT / "source_manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise RuntimeError("MiniCPM source manifest must be a JSON object")
    for path, expected in {
        **manifest["files"],
        **manifest["fragments"],
    }.items():
        if _hash(_VENDOR / path) != expected:
            raise RuntimeError(f"MiniCPM compatibility payload is incomplete: {path}")
    return manifest


def _runtime_state(manifest: dict) -> str:
    distribution = importlib.metadata.distribution("mlx-vlm")
    observed = {
        path: _hash(Path(str(distribution.locate_file(path))))
        for path in manifest["guards"]
    }
    for state in ("native", "legacy", "upgraded"):
        if all(state in hashes for hashes in manifest["guards"].values()) and all(
            observed[path] == hashes[state]
            for path, hashes in manifest["guards"].items()
        ):
            return state
    differences = [
        path
        for path, hashes in manifest["guards"].items()
        if observed[path] != hashes["legacy"]
    ]
    raise RuntimeError(
        "Unrecognized or partial mlx-vlm MiniCPM implementation; refusing to "
        "replace framework code. Review a coordinated dependency upgrade. "
        f"Differing files: {', '.join(differences[:8])}"
    )


def validate_target_config(config: dict) -> None:
    """Reject a routing mode outside the official target's proven contract."""
    text = config.get("text_config") or {}
    if text.get("norm_topk_prob", True) is not True:
        raise ValueError(
            "MiniCPM-V 4.7 requires normalized top-k routing " "(norm_topk_prob=True)."
        )


def _native_loader_functions(utils):
    namespace = dict(vars(utils))
    source = _VENDOR / "loader_functions.py"
    exec(compile(source.read_bytes(), str(source), "exec"), namespace)
    return {
        name: FunctionType(
            namespace[name].__code__,
            vars(utils),
            name,
            namespace[name].__defaults__,
            namespace[name].__closure__,
        )
        for name in UTILS_FUNCTIONS
    }


def _check_bridged_installation() -> None:
    for name, module in _OWNED_MODULES.items():
        if sys.modules.get(name) is not module:
            raise RuntimeError(f"MiniCPM runtime module changed or disappeared: {name}")
    for owner, name, value in _OWNED_SYMBOLS:
        if getattr(owner, name, None) is not value:
            raise RuntimeError(f"MiniCPM owned runtime symbol changed: {name}")
    models = _OWNED_MODULES["mlx_vlm.models"]
    prompts = _OWNED_MODULES["mlx_vlm.prompt_utils"]
    if str(_VENDOR / "mlx_vlm/models") not in models.__path__:
        raise RuntimeError("MiniCPM model registration disappeared")
    if prompts.MODEL_CONFIG.get("minicpmv4_7") is not (
        prompts.MessageFormat.LIST_WITH_IMAGE_FIRST
    ):
        raise RuntimeError("MiniCPM media prompt registration changed")


def apply_mlx_vlm_minicpm47_compat_patch() -> bool:
    """Bridge exact reviewed runtimes; the complete reviewed native is a no-op."""
    global _STATE, _OWNED_MODULES, _OWNED_SYMBOLS
    with _LOCK:
        manifest = _inventory()
        state = _runtime_state(manifest)
        if _STATE is not None:
            if state != _STATE:
                raise RuntimeError("mlx-vlm changed during MiniCPM compatibility use")
            if state != "native":
                _check_bridged_installation()
            return False
        if state == "native":
            _STATE = state
            return False
        if any(
            name in (_MODEL_PREFIX, _HELPER_PREFIX)
            or name.startswith((_MODEL_PREFIX + ".", _HELPER_PREFIX + "."))
            for name in sys.modules
        ):
            raise RuntimeError("Partial MiniCPM modules are already imported")

        import mlx_vlm.models as models
        import mlx_vlm.models.switch_layers as switch_layers
        import mlx_vlm.prompt_utils as prompt_utils
        import mlx_vlm.utils as utils

        if state == "legacy" and any(
            hasattr(switch_layers, name) for name in EXPERT_SYMBOLS
        ):
            raise RuntimeError("Partial MiniCPM expert helpers are already installed")
        original_experts = {
            name: getattr(switch_layers, name)
            for name in EXPERT_SYMBOLS
            if hasattr(switch_layers, name)
        }
        native = _native_loader_functions(utils)
        original = {
            name: getattr(utils, name)
            for name in (*UTILS_FUNCTIONS, "get_model_and_args")
        }
        model_path = str(_VENDOR / "mlx_vlm" / "models")
        helpers_path = _VENDOR / "expert_helpers.py"
        original_formatter = prompt_utils.MessageFormatter.format_message
        original_prompt = prompt_utils.MODEL_CONFIG.get("minicpmv4_7")
        if state == "legacy" and original_prompt is not None:
            raise RuntimeError(
                "Partial MiniCPM prompt registration is already installed"
            )
        formatter_path = _VENDOR / "message_formatter.py"
        namespace = dict(vars(prompt_utils))
        exec(
            compile(formatter_path.read_bytes(), str(formatter_path), "exec"), namespace
        )
        native_formatter = FunctionType(
            namespace["format_message"].__code__,
            vars(prompt_utils),
            "format_message",
            namespace["format_message"].__defaults__,
        )

        def format_message(self, *args, **kwargs):
            implementation = (
                native_formatter
                if self.model_name == "minicpmv4_7"
                else original_formatter
            )
            return implementation(self, *args, **kwargs)

        def update_module_configs(model_config, model_class, config, modules):
            implementation = (
                native["update_module_configs"]
                if config.get("model_type") == "minicpmv4_7"
                else original["update_module_configs"]
            )
            return implementation(model_config, model_class, config, modules)

        def load_processor(model_path, *args, **kwargs):
            config_path = Path(model_path) / "config.json"
            if not config_path.is_file():
                return original["load_processor"](model_path, *args, **kwargs)
            with config_path.open(encoding="utf-8") as stream:
                config = json.load(stream)
            implementation = (
                native["load_processor"]
                if config.get("model_type") == "minicpmv4_7"
                else original["load_processor"]
            )
            return implementation(model_path, *args, **kwargs)

        def get_model_and_args(config, model_path=None):
            if _HELPER_LOADING.get():
                return _helper_module(), "dspark"
            if config.get("model_type") == "minicpmv4_7":
                validate_target_config(config)
            return original["get_model_and_args"](config, model_path=model_path)

        try:
            if state == "legacy":
                exec(
                    compile(helpers_path.read_bytes(), str(helpers_path), "exec"),
                    vars(switch_layers),
                )
            models.__path__.insert(0, model_path)
            utils.update_module_configs = update_module_configs
            utils.load_processor = load_processor
            utils.get_model_and_args = get_model_and_args
            prompt_utils.MODEL_CONFIG["minicpmv4_7"] = (
                prompt_utils.MessageFormat.LIST_WITH_IMAGE_FIRST
            )
            prompt_utils.MessageFormatter.format_message = format_message
            importlib.invalidate_caches()
            importlib.import_module(_MODEL_PREFIX)
            _helper_module()
            owned_modules = {
                name: module
                for name, module in tuple(sys.modules.items())
                if name in ("mlx_vlm.models", "mlx_vlm.prompt_utils")
                or name in (_MODEL_PREFIX, _HELPER_PREFIX)
                or name.startswith((_MODEL_PREFIX + ".", _HELPER_PREFIX + "."))
            }
            owned_symbols = tuple(
                (switch_layers, name, getattr(switch_layers, name))
                for name in EXPERT_SYMBOLS
            ) + tuple(
                (module, name, getattr(module, name))
                for module in (
                    owned_modules[_MODEL_PREFIX],
                    owned_modules[_HELPER_PREFIX],
                )
                for name in tuple(vars(module))
                if not name.startswith("_")
            )
        except BaseException:
            for name, value in original.items():
                setattr(utils, name, value)
            for name in EXPERT_SYMBOLS:
                if name in original_experts:
                    setattr(switch_layers, name, original_experts[name])
                else:
                    vars(switch_layers).pop(name, None)
            if original_prompt is None:
                prompt_utils.MODEL_CONFIG.pop("minicpmv4_7", None)
            else:
                prompt_utils.MODEL_CONFIG["minicpmv4_7"] = original_prompt
            prompt_utils.MessageFormatter.format_message = original_formatter
            if model_path in models.__path__:
                models.__path__.remove(model_path)
            for name in tuple(sys.modules):
                if name in (_MODEL_PREFIX, _HELPER_PREFIX) or name.startswith(
                    (_MODEL_PREFIX + ".", _HELPER_PREFIX + ".")
                ):
                    del sys.modules[name]
            vars(models).pop("minicpmv4_7", None)
            raise
        _OWNED_MODULES = owned_modules
        _OWNED_SYMBOLS = owned_symbols
        _STATE = state
        logger.info(
            "MiniCPM-V 4.7 serving compatibility applied: %s", manifest["revision"]
        )
        return True


def _helper_module():
    if _STATE == "native":
        return importlib.import_module("mlx_vlm.speculative.drafters.dspark")
    if _HELPER_PREFIX in sys.modules:
        return sys.modules[_HELPER_PREFIX]
    # A private package keeps the generic DSpark class and its aliases
    # unchanged. Relative imports still resolve against the native parent.
    source = _VENDOR / "mlx_vlm/speculative/drafters/dspark/__init__.py"
    importlib.import_module("mlx_vlm.speculative.drafters")
    spec = importlib.util.spec_from_file_location(
        _HELPER_PREFIX, source, submodule_search_locations=[str(source.parent)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the authoritative MiniCPM DSpark package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_HELPER_PREFIX] = module
    spec.loader.exec_module(module)
    return module


def dspark_model_class():
    apply_mlx_vlm_minicpm47_compat_patch()
    return _helper_module().DSparkDraftModel


def load_dspark_model(path: str):
    """Reuse the real strict loader with a request-local native helper resolver."""
    apply_mlx_vlm_minicpm47_compat_patch()
    directory = Path(path)
    with (directory / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    module = _helper_module()
    helper_config = module.ModelConfig.from_dict(config)
    if helper_config.target_model_type != "minicpmv4_7":
        raise ValueError("MiniCPM-V 4.7 requires its explicitly paired DSpark helper")
    from mlx_vlm.utils import load_model

    token = _HELPER_LOADING.set(True)
    try:
        return load_model(directory)
    finally:
        _HELPER_LOADING.reset(token)
