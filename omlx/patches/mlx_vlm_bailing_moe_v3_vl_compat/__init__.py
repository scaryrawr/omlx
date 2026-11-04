# SPDX-License-Identifier: Apache-2.0
"""Ling 3.0 Flash VL compatibility overlay for the pinned mlx-vlm release."""

from __future__ import annotations

import importlib
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SOURCE_COMMIT = "9a89d53fcd698edd6414dd9ea155d6482c721fc8"
_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"
_APPLIED = False


def _prepend_package_path(package: Any, path: Path) -> None:
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    path_string = str(path)
    if path_string in package_path:
        package_path.remove(path_string)
    package_path.insert(0, path_string)


def apply_mlx_vlm_bailing_moe_v3_vl_compat_patch() -> bool:
    """Expose oMLX's vendored Ling 3.0 Flash VL model to mlx-vlm."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_vlm
        import mlx_vlm.models
        from mlx_vlm.prompt_utils import MODEL_CONFIG, MessageFormat

        _prepend_package_path(mlx_vlm, _VENDOR_MLX_VLM)
        _prepend_package_path(mlx_vlm.models, _VENDOR_MLX_VLM / "models")
        importlib.import_module("mlx_vlm.models.bailing_moe_v3_vl")
        MODEL_CONFIG.setdefault(
            "bailing_moe_v3_vl", MessageFormat.LIST_WITH_IMAGE_FIRST
        )
        _patch_load_config()
        _patch_video_prompt_format()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Ling 3.0 Flash VL mlx-vlm registration failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("Ling 3.0 Flash VL mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


def _patch_load_config() -> None:
    """Retain Ling's native FP8 base plus per-module FP4 overrides."""
    import mlx_vlm.utils as vlm_utils

    original = vlm_utils.load_config
    if getattr(original, "_omlx_bailing_moe_v3_vl_compat", False):
        return

    def patched_load_config(model_path, **kwargs):
        config = original(model_path, **kwargs)
        _normalize_ling_quantization(config)
        return config

    patched_load_config._omlx_bailing_moe_v3_vl_compat = True
    patched_load_config._omlx_original = original
    vlm_utils.load_config = patched_load_config


def _normalize_ling_quantization(config: Any) -> None:
    if not isinstance(config, dict) or config.get("model_type") != "bailing_moe_v3_vl":
        return

    quantization = config.get("quantization")
    quantization_config = config.get("quantization_config")
    runtime_quantization = _runtime_quantization(quantization)
    if runtime_quantization is None:
        runtime_quantization = _runtime_quantization(quantization_config)
    if runtime_quantization is None:
        runtime_quantization = _legacy_ling_quantization(config, quantization_config)
    if runtime_quantization is None:
        return

    # mlx-vlm needs ``quantization`` before ModelConfig construction to avoid
    # collapsing a mixed map into a global mxfp4 default. Mirror the complete
    # map so ModelConfig's two metadata fields agree through load_model().
    config["quantization"] = runtime_quantization
    config["quantization_config"] = deepcopy(runtime_quantization)


def _runtime_quantization(metadata: Any) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    if not {"bits", "group_size"}.issubset(metadata):
        return None
    return deepcopy(metadata)


def _legacy_ling_quantization(
    config: dict[str, Any], metadata: Any
) -> dict[str, Any] | None:
    """Convert only Ling's explicit FP8-plus-routed-MXFP4 declaration."""
    if not isinstance(metadata, dict) or metadata.get("quant_method") != "fp8":
        return None

    quantization: dict[str, Any] = {
        "group_size": int(metadata.get("group_size", 64)),
        "bits": 8,
    }
    for path, spec in metadata.items():
        if isinstance(spec, dict) and {"bits", "group_size"}.issubset(spec):
            quantization[path] = deepcopy(spec)

    if metadata.get("routed_experts_quant_method") != "mxfp4":
        return quantization

    text_config = config.get("text_config")
    text_config = text_config if isinstance(text_config, dict) else {}
    first_sparse_layer = int(text_config.get("first_k_dense_replace", 0))
    num_hidden_layers = int(text_config.get("num_hidden_layers", 0))
    expert_quantization = {
        "group_size": int(metadata.get("routed_experts_group_size", 32)),
        "bits": 4,
        "mode": "mxfp4",
    }
    for layer_index in range(first_sparse_layer, num_hidden_layers):
        base = f"language_model.model.layers.{layer_index}.mlp.switch_mlp"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            quantization.setdefault(f"{base}.{projection}", deepcopy(expert_quantization))
    return quantization


def _patch_video_prompt_format() -> None:
    """Keep Ling on mlx-vlm's Qwen video-message path."""
    from mlx_vlm.prompt_utils import MessageFormatter

    original = MessageFormatter.format_message
    if getattr(original, "_omlx_bailing_moe_v3_vl_compat", False):
        return

    def patched_format_message(self, *args, **kwargs):
        if self.model_name == "bailing_moe_v3_vl" and kwargs.get("video"):
            return self._format_video_message(*args, **kwargs)
        return original(self, *args, **kwargs)

    patched_format_message._omlx_bailing_moe_v3_vl_compat = True
    patched_format_message._omlx_original = original
    MessageFormatter.format_message = patched_format_message


__all__ = [
    "SOURCE_COMMIT",
    "apply_mlx_vlm_bailing_moe_v3_vl_compat_patch",
    "is_applied",
]
