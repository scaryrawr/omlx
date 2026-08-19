# SPDX-License-Identifier: Apache-2.0
"""Shared metadata for mlx-vlm-backed image model aliases."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ImageTask = Literal["generation", "edit"]

IMAGE_UNKNOWN_FALLBACK_SIZE = 24 * 1024**3


@dataclass(frozen=True)
class ImageModelSpec:
    """Metadata used to identify and constrain a supported image family."""

    base_model: str
    tasks: tuple[ImageTask, ...]
    estimated_size: int
    discovery_aliases: tuple[str, ...]
    allows_multiple_edit_images: bool = False


def normalize_image_alias(value: object) -> str:
    """Normalize model ids and local-folder aliases without importing mlx-vlm."""
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower().replace("_", "-").replace(".", "-")
    normalized = re.sub(r"\s+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    if normalized.startswith("flux-2-"):
        normalized = "flux2-" + normalized[len("flux-2-") :]
    return normalized


def _aliases(*values: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(normalize_image_alias(value) for value in values))


IMAGE_MODEL_SPECS: tuple[ImageModelSpec, ...] = (
    ImageModelSpec(
        base_model="flux2-klein-4b",
        tasks=("generation", "edit"),
        estimated_size=8 * 1024**3,
        discovery_aliases=_aliases(
            "flux2-klein-4b",
            "flux2-klein",
            "klein-4b",
            "black-forest-labs/FLUX.2-klein-4B",
        ),
        allows_multiple_edit_images=True,
    ),
    ImageModelSpec(
        base_model="flux2-klein-9b",
        tasks=("generation", "edit"),
        estimated_size=18 * 1024**3,
        discovery_aliases=_aliases(
            "flux2-klein-9b",
            "klein-9b",
            "black-forest-labs/FLUX.2-klein-9B",
        ),
        allows_multiple_edit_images=True,
    ),
    ImageModelSpec(
        base_model="flux2-klein-base-4b",
        tasks=("generation", "edit"),
        estimated_size=8 * 1024**3,
        discovery_aliases=_aliases(
            "flux2-klein-base-4b",
            "flux2-base-4b",
            "klein-base-4b",
            "black-forest-labs/FLUX.2-klein-base-4B",
        ),
        allows_multiple_edit_images=True,
    ),
    ImageModelSpec(
        base_model="flux2-klein-base-9b",
        tasks=("generation", "edit"),
        estimated_size=18 * 1024**3,
        discovery_aliases=_aliases(
            "flux2-klein-base-9b",
            "flux2-base-9b",
            "klein-base-9b",
            "black-forest-labs/FLUX.2-klein-base-9B",
        ),
        allows_multiple_edit_images=True,
    ),
    ImageModelSpec(
        base_model="flux2-klein-9b-kv",
        tasks=("generation", "edit"),
        estimated_size=18 * 1024**3,
        discovery_aliases=_aliases(
            "flux2-klein-9b-kv",
            "klein-9b-kv",
            "black-forest-labs/FLUX.2-klein-9b-kv",
        ),
        allows_multiple_edit_images=True,
    ),
    ImageModelSpec(
        base_model="mage-flow-base",
        tasks=("generation",),
        estimated_size=10 * 1024**3,
        discovery_aliases=_aliases(
            "mage-flow-base",
            "mage-flow-4b-base",
            "microsoft/Mage-Flow-Base",
        ),
    ),
    ImageModelSpec(
        base_model="mage-flow",
        tasks=("generation",),
        estimated_size=10 * 1024**3,
        discovery_aliases=_aliases(
            "mage-flow",
            "mage-flow-aligned",
            "mage-flow-4b",
            "mage-flow-4b-aligned",
            "microsoft/Mage-Flow",
        ),
    ),
    ImageModelSpec(
        base_model="mage-flow-turbo",
        tasks=("generation",),
        estimated_size=10 * 1024**3,
        discovery_aliases=_aliases(
            "mage-flow-turbo",
            "mage-flow-4b-turbo",
            "microsoft/Mage-Flow-Turbo",
        ),
    ),
    ImageModelSpec(
        base_model="mage-flow-edit-base",
        tasks=("edit",),
        estimated_size=10 * 1024**3,
        discovery_aliases=_aliases(
            "mage-flow-edit-base",
            "mage-flow-edit-4b-base",
            "microsoft/Mage-Flow-Edit-Base",
        ),
        allows_multiple_edit_images=True,
    ),
    ImageModelSpec(
        base_model="mage-flow-edit",
        tasks=("edit",),
        estimated_size=10 * 1024**3,
        discovery_aliases=_aliases(
            "mage-flow-edit",
            "mage-flow-edit-aligned",
            "mage-flow-edit-4b",
            "mage-flow-edit-4b-aligned",
            "microsoft/Mage-Flow-Edit",
        ),
        allows_multiple_edit_images=True,
    ),
    ImageModelSpec(
        base_model="mage-flow-edit-turbo",
        tasks=("edit",),
        estimated_size=10 * 1024**3,
        discovery_aliases=_aliases(
            "mage-flow-edit-turbo",
            "mage-flow-edit-4b-turbo",
            "microsoft/Mage-Flow-Edit-Turbo",
        ),
        allows_multiple_edit_images=True,
    ),
    ImageModelSpec(
        base_model="z-image",
        tasks=("generation", "edit"),
        estimated_size=12 * 1024**3,
        discovery_aliases=_aliases(
            "z-image",
            "zimage",
            "tongyi-mai/Z-Image",
        ),
    ),
    ImageModelSpec(
        base_model="z-image-turbo",
        tasks=("generation", "edit"),
        estimated_size=12 * 1024**3,
        discovery_aliases=_aliases(
            "z-image-turbo",
            "zimage-turbo",
            "tongyi-mai/Z-Image-Turbo",
        ),
    ),
    ImageModelSpec(
        base_model="ernie-image",
        tasks=("generation", "edit"),
        estimated_size=22 * 1024**3,
        discovery_aliases=_aliases(
            "ernie-image",
            "ernie-image-base",
            "baidu/ERNIE-Image",
        ),
    ),
    ImageModelSpec(
        base_model="ernie-image-turbo",
        tasks=("generation", "edit"),
        estimated_size=22 * 1024**3,
        discovery_aliases=_aliases(
            "ernie-image-turbo",
            "ernie-turbo",
            "baidu/ERNIE-Image-Turbo",
        ),
    ),
    ImageModelSpec(
        base_model="ideogram-4-fp8",
        tasks=("generation",),
        estimated_size=28 * 1024**3,
        discovery_aliases=_aliases(
            "ideogram-4-fp8",
            "ideogram4-fp8",
            "ideogram4",
            "ideogram-4",
            "ideogram",
            "ideogram-ai/ideogram-4-fp8",
        ),
    ),
    ImageModelSpec(
        base_model="ternary",
        tasks=("generation",),
        estimated_size=8 * 1024**3,
        discovery_aliases=_aliases(
            "ternary",
            "bonsai",
            "bonsai-ternary",
            "ternary-mlx",
            "bonsai-ternary-mlx",
            "2bit",
            "prism-ml/bonsai-image-ternary-4B-mlx-2bit",
        ),
    ),
)


def _build_aliases() -> dict[tuple[ImageTask, str], tuple[str, ...]]:
    aliases: dict[tuple[ImageTask, str], tuple[str, ...]] = {}
    for spec in IMAGE_MODEL_SPECS:
        values = _aliases(spec.base_model, *spec.discovery_aliases)
        for task in spec.tasks:
            aliases[(task, spec.base_model)] = values
    return aliases


IMAGE_ENGINE_ALIASES = _build_aliases()
_SPEC_BY_BASE_MODEL = {spec.base_model: spec for spec in IMAGE_MODEL_SPECS}
_RUNTIME_MODEL_REFERENCES = {
    "ideogram-4-fp8": "ideogram-ai/ideogram-4-fp8",
    "z-image": "Tongyi-MAI/Z-Image",
    "z-image-turbo": "Tongyi-MAI/Z-Image-Turbo",
}


def image_engine_aliases(task: ImageTask, base_model: str) -> tuple[str, ...]:
    """Return supported aliases for a canonical base model and image task."""
    return IMAGE_ENGINE_ALIASES.get((task, _canonical_base_model(base_model)), ())


def _canonical_base_model(base_model: str) -> str:
    """Return the registry base model for a user-facing image alias."""
    normalized = normalize_image_alias(base_model)
    for (_, canonical), aliases in IMAGE_ENGINE_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    return normalized


def get_image_model_spec(base_model: str) -> ImageModelSpec | None:
    """Return the registered model family for an alias, if it is supported."""
    return _SPEC_BY_BASE_MODEL.get(_canonical_base_model(base_model))


def get_image_model_reference(base_model: str) -> str:
    """Return the mlx-vlm loader reference for a supported family alias."""
    canonical = _canonical_base_model(base_model)
    return _RUNTIME_MODEL_REFERENCES.get(canonical, canonical)


def image_edit_accepts_multiple_inputs(base_model: str) -> bool:
    """Return whether an edit family accepts multiple reference images."""
    spec = get_image_model_spec(base_model)
    return bool(spec and spec.allows_multiple_edit_images)


IMAGE_DEFAULT_ESTIMATED_SIZES = {
    alias: spec.estimated_size
    for spec in IMAGE_MODEL_SPECS
    for alias in _aliases(spec.base_model, *spec.discovery_aliases)
}

# Defaults documented by mlx-vlm model configurations. Request values and
# manifest defaults always override these family defaults.
IMAGE_DEFAULTS: dict[str, dict[str, int | float]] = {
    "flux2-klein-4b": {"default_steps": 4, "default_guidance": 1.0},
    "flux2-klein-9b": {"default_steps": 4, "default_guidance": 1.0},
    "flux2-klein-base-4b": {"default_steps": 4, "default_guidance": 1.0},
    "flux2-klein-base-9b": {"default_steps": 4, "default_guidance": 1.0},
    "flux2-klein-9b-kv": {"default_steps": 4, "default_guidance": 1.0},
    "mage-flow-base": {"default_steps": 30, "default_guidance": 5.0},
    "mage-flow": {"default_steps": 20, "default_guidance": 5.0},
    "mage-flow-turbo": {"default_steps": 4, "default_guidance": 1.0},
    "mage-flow-edit-base": {"default_steps": 30, "default_guidance": 5.0},
    "mage-flow-edit": {"default_steps": 30, "default_guidance": 5.0},
    "mage-flow-edit-turbo": {"default_steps": 4, "default_guidance": 1.0},
    "z-image": {
        "default_steps": 50,
        "default_guidance": 4.0,
        "default_image_strength": 0.6,
    },
    "z-image-turbo": {
        "default_steps": 9,
        "default_guidance": 0.0,
        "default_image_strength": 0.6,
    },
    "ernie-image": {
        "default_steps": 50,
        "default_guidance": 4.0,
        "default_image_strength": 0.6,
    },
    "ernie-image-turbo": {
        "default_steps": 8,
        "default_guidance": 1.0,
        "default_image_strength": 0.6,
    },
}

IMAGE_TASK_DEFAULTS: dict[tuple[str, ImageTask], dict[str, int | float]] = {
    # The Z-Image Turbo adapter uses eight denoising steps for img2img while
    # its text-to-image configuration uses nine.
    ("z-image-turbo", "edit"): {"default_steps": 8},
}


def get_image_defaults(
    base_model: str, task: ImageTask | None = None
) -> dict[str, int | float]:
    """Return family and optional task-specific defaults for an image alias."""
    canonical = _canonical_base_model(base_model)
    defaults = dict(IMAGE_DEFAULTS.get(canonical, {}))
    if task is not None:
        defaults.update(IMAGE_TASK_DEFAULTS.get((canonical, task), {}))
    return defaults


def _matches_image_model_prefix(normalized_name: str, alias: str) -> bool:
    if normalized_name == alias or normalized_name.startswith(f"{alias}-"):
        return True

    compact_name = normalized_name.replace("-", "")
    compact_alias = alias.replace("-", "")
    return compact_name == compact_alias or compact_name.startswith(compact_alias)


def infer_image_model_spec_from_name(name: str) -> ImageModelSpec | None:
    """Return the supported image spec implied by a local model folder name."""
    normalized_name = normalize_image_alias(name)
    candidates = [
        (len(alias.replace("-", "")), spec, alias)
        for spec in IMAGE_MODEL_SPECS
        for alias in spec.discovery_aliases
    ]
    for _, spec, alias in sorted(
        candidates,
        key=lambda candidate: candidate[0],
        reverse=True,
    ):
        if _matches_image_model_prefix(normalized_name, alias):
            return spec
    return None
