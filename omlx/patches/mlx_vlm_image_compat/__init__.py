# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import logging
from functools import wraps
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_mlx_vlm_image_compat_patch() -> bool:
    global _APPLIED
    if _APPLIED:
        return False

    _patch_z_image_tokenizer_security()
    _patch_local_generation_model_selection()

    _APPLIED = True
    logger.info("mlx-vlm image compatibility patch applied")
    return True


def is_applied() -> bool:
    """Return whether the compatibility layer has been registered."""
    return _APPLIED


def _patch_z_image_tokenizer_security() -> None:
    """Keep native Z-Image tokenizer loading local and remote-code-free."""
    pipeline = importlib.import_module("mlx_vlm.models.z_image.pipeline")
    tokenizer = pipeline.AutoTokenizer
    if getattr(tokenizer, "_omlx_safe_z_image_tokenizer", False):
        return

    class SafeAutoTokenizer:
        _omlx_safe_z_image_tokenizer = True

        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            kwargs["local_files_only"] = True
            kwargs["trust_remote_code"] = False
            return tokenizer.from_pretrained(*args, **kwargs)

    pipeline.AutoTokenizer = SafeAutoTokenizer


def _patch_local_generation_model_selection() -> None:
    image = importlib.import_module("mlx_vlm.generate.image")
    original = image.load_image_generation_model

    @wraps(original)
    def load_image_generation_model(model: str | None, **kwargs: Any) -> Any:
        if model is not None:
            root = Path(model).expanduser()
            if root.exists():
                for model_type in image._local_image_model_types(str(root)):
                    model_class = image._supported_image_model_class(
                        model_type, str(root)
                    )
                    if model_class is not None:
                        kwargs.pop("force_download", None)
                        kwargs.pop("revision", None)
                        kwargs.setdefault("model_path", root)
                        return model_class.from_model_id(model, **kwargs)
        return original(model, **kwargs)

    image.load_image_generation_model = load_image_generation_model


__all__ = ["apply_mlx_vlm_image_compat_patch", "is_applied"]
