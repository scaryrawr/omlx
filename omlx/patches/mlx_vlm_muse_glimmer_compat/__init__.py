# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer (Meta) compatibility layer for mlx-vlm.

mlx-vlm now ships Muse Glimmer natively. oMLX retains narrow compatibility
hooks for its vision-feature cache and DFlash target:

- keeps prompt registration idempotent across supported mlx-vlm revisions,
- adds the public `encode_image` alias used by oMLX's vision-feature cache,
- keeps rotary numerics aligned with dflash-mlx's target implementation.

oMLX deltas against the PR head are marked with `oMLX:` comments in the
vendored files; see `vendor/mlx_vlm/models/muse_glimmer/README.md` for
the pin-bump checklist.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"

_MODEL_TYPE = "muse_glimmer"

_APPLIED = False


def apply_mlx_vlm_muse_glimmer_compat_patch() -> bool:
    """Install Muse Glimmer compatibility and discovery hooks."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        _install_vendor_namespace()
        _import_vendor_modules()

        import mlx_vlm.prompt_utils as prompt_utils

        _register_prompt_format(prompt_utils)
        _patch_encode_image()
        _patch_dflash_rope_parity()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Muse Glimmer mlx-vlm compat patch failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("Muse Glimmer mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


def _install_vendor_namespace() -> None:
    import mlx_vlm
    import mlx_vlm.models

    _append_package_path(mlx_vlm, _VENDOR_MLX_VLM)
    _append_package_path(mlx_vlm.models, _VENDOR_MLX_VLM / "models")


def _append_package_path(package: Any, path: Path) -> None:
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    path_str = str(path)
    if path_str not in package_path:
        package_path.append(path_str)


def _import_vendor_modules() -> None:
    # The shared activations shim must be importable before the model
    # package, whose language module does `from ..activations import swiglu`.
    # Importing the package also imports processing_muse_glimmer, which
    # registers MuseGlimmerProcessor with AutoProcessor as a side effect.
    importlib.import_module("mlx_vlm.models.activations")
    importlib.import_module(f"mlx_vlm.models.{_MODEL_TYPE}")


def _register_prompt_format(prompt_utils: Any) -> None:
    # apply_chat_template short-circuits to text-only formatting for any
    # model_type missing from MODEL_CONFIG, dropping image parts entirely.
    # The pinned get_message_json handles LIST_WITH_IMAGE_FIRST generically,
    # so the MODEL_CONFIG entry is the only registration needed (it is the
    # same one-line registration PR #1838 makes upstream).
    model_config = getattr(prompt_utils, "MODEL_CONFIG", None)
    message_format = getattr(prompt_utils, "MessageFormat", None)
    if isinstance(model_config, dict) and message_format is not None:
        placeholder = getattr(message_format, "LIST_WITH_IMAGE_FIRST", None)
        if placeholder is not None:
            model_config.setdefault(_MODEL_TYPE, placeholder)


def _patch_encode_image() -> None:
    module = importlib.import_module(f"mlx_vlm.models.{_MODEL_TYPE}")
    model_class = getattr(module, "Model", None)
    if model_class is None or hasattr(model_class, "encode_image"):
        return

    def encode_image(self, pixel_values, image_grid_thw=None, **kwargs):
        del kwargs
        return self._encode_image(pixel_values, image_grid_thw=image_grid_thw)

    model_class.encode_image = encode_image


def _patch_dflash_rope_parity() -> None:
    language = importlib.import_module(f"mlx_vlm.models.{_MODEL_TYPE}.language")
    attention = getattr(language, "Attention", None)
    if attention is None or getattr(attention, "_omlx_dflash_rope_parity", False):
        return

    from mlx_lm.models.rope_utils import initialize_rope

    original_init = attention.__init__

    def patched_init(self, args, layer_idx):
        original_init(self, args, layer_idx)
        theta = (
            float(args.layer_rope_theta[layer_idx])
            if self.use_rope
            else float(args.rope_parameters.get("rope_theta", 500000.0))
        )
        self.rope = initialize_rope(
            self.head_dim,
            base=theta,
            traditional=False,
            scaling_config={"rope_type": "default", "rope_theta": theta},
            max_position_embeddings=args.max_position_embeddings,
        )

    attention.__init__ = patched_init
    attention._omlx_dflash_rope_parity = True


__all__ = ["apply_mlx_vlm_muse_glimmer_compat_patch", "is_applied"]
