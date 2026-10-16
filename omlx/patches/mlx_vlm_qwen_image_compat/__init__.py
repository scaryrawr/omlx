# SPDX-License-Identifier: Apache-2.0
"""Exact temporary Qwen Image 2.1 backport, activated before Qwen model load."""

from __future__ import annotations

import importlib
import logging
import threading
from pathlib import Path
from types import ModuleType

from .generate_backport import SYMBOLS, selected_source, source_hash

logger = logging.getLogger(__name__)
_APPLIED = False
_LOCK = threading.Lock()


def _prepare_sources(
    sources: dict[str, str],
    backports: dict[str, tuple[str, str, tuple[tuple[str, str], ...]]],
) -> dict[str, str]:
    states = {
        name: source_hash(selected_source(source, SYMBOLS.get(name)))
        for name, source in sources.items()
    }
    if all(states[name] == spec[1] for name, spec in backports.items()):
        return {}
    unknown = [name for name, spec in backports.items() if states.get(name) != spec[0]]
    if unknown:
        raise RuntimeError(
            "Unsupported or partially fixed mlx-vlm Qwen image source: "
            + ", ".join(unknown)
            + ". Update the verified Qwen image compatibility backport."
        )
    patched = {}
    for name, (_, native_hash, edits) in backports.items():
        source = selected_source(sources[name], SYMBOLS.get(name))
        for before, after in edits:
            if source.count(before) != 1:
                raise RuntimeError(f"Qwen image backport context mismatch: {name}")
            source = source.replace(before, after, 1)
        if source_hash(source) != native_hash:
            raise RuntimeError(f"Qwen image backport result mismatch: {name}")
        patched[name] = source
    return patched


def _read_source(module: ModuleType) -> str:
    if module.__file__ is None:
        raise RuntimeError(f"Qwen image module has no Python source: {module.__name__}")
    return Path(module.__file__).read_text()


def apply_qwen_image_compat_patch() -> bool:
    """Backport only verified legacy code; complete native fixes are a no-op."""
    global _APPLIED
    with _LOCK:
        if _APPLIED:
            return False
        from ._backport import BACKPORTS, SOURCE_REVISION

        modules = {name: importlib.import_module(name) for name in BACKPORTS}
        patched = _prepare_sources(
            {name: _read_source(module) for name, module in modules.items()}, BACKPORTS
        )
        if not patched:
            return False
        package = importlib.import_module("mlx_vlm.models.qwen_image")
        image = modules["mlx_vlm.generate.image"]
        edit_image = importlib.import_module("mlx_vlm.generate.edit_image")
        caches = (
            getattr(image, "_image_model_class_for_type", None),
            getattr(edit_image, "_image_edit_model_class_for_type", None),
        )
        if not isinstance(getattr(package, "__all__", None), list) or any(
            not callable(getattr(cache, "cache_clear", None)) for cache in caches
        ):
            raise RuntimeError("Unsupported mlx-vlm Qwen image dispatch/cache shape")
        # Compile every checked source before changing any module.
        codes = {
            name: compile(source, f"<omlx-qwen-image:{name}>", "exec")
            for name, source in patched.items()
        }
        for name, code in codes.items():
            exec(code, modules[name].__dict__)

        config = modules["mlx_vlm.models.qwen_image.config"]
        model = modules["mlx_vlm.models.qwen_image.model"]
        for name in package.__all__:
            if hasattr(model, name):
                setattr(package, name, getattr(model, name))
            elif hasattr(config, name):
                setattr(package, name, getattr(config, name))
        image._image_model_class_for_type.cache_clear()
        edit_image._image_edit_model_class_for_type.cache_clear()
        _APPLIED = True
        logger.info(
            "Qwen image compatibility backport applied from %s", SOURCE_REVISION
        )
        return True
