# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer (Meta) compatibility layer for mlx-vlm.

mlx-vlm now ships Muse Glimmer natively. oMLX retains narrow compatibility
hooks for its vision-feature cache and DFlash target:

- adds the public `encode_image` alias used by oMLX's vision-feature cache,
- keeps rotary numerics aligned with dflash-mlx's target implementation.

oMLX deltas against the PR head are marked with `oMLX:` comments in the
runtime wrappers below.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

_MODEL_TYPE = "muse_glimmer"

_APPLIED = False


def apply_mlx_vlm_muse_glimmer_compat_patch() -> bool:
    """Install Muse Glimmer compatibility and discovery hooks."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        importlib.import_module(f"mlx_vlm.models.{_MODEL_TYPE}")

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
