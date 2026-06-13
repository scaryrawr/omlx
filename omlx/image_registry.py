# SPDX-License-Identifier: Apache-2.0
"""Shared metadata for mflux-backed image model aliases."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ImageTask = Literal["generation", "edit"]

IMAGE_UNKNOWN_FALLBACK_SIZE = 24 * 1024**3


@dataclass(frozen=True)
class ImageModelSpec:
    """Metadata needed to identify supported mflux image model families."""

    base_model: str
    tasks: tuple[ImageTask, ...]
    estimated_size: int
    discovery_aliases: tuple[str, ...]


IMAGE_ENGINE_ALIASES: dict[tuple[ImageTask, str], tuple[str, ...]] = {
    ("generation", "flux2-klein-4b"): (
        "flux2-klein-4b",
        "flux2-klein",
        "klein-4b",
        "klein",
    ),
    ("generation", "flux2-klein-9b"): ("flux2-klein-9b", "klein-9b"),
    ("generation", "z-image"): ("z-image", "zimage"),
    ("generation", "z-image-turbo"): ("z-image-turbo", "zimage-turbo"),
    ("generation", "qwen-image"): ("qwen-image", "qwen"),
    ("generation", "fibo"): ("fibo",),
    ("generation", "ernie-image-turbo"): ("ernie-image-turbo",),
    ("generation", "ernie-image"): ("ernie-image",),
    ("generation", "ideogram-4-fp8"): (
        "ideogram-4-fp8",
        "ideogram4-fp8",
        "ideogram4",
        "ideogram-4",
        "ideogram",
    ),
    ("edit", "flux2-klein-4b"): (
        "flux2-klein-4b",
        "flux2-klein-4b-edit",
        "flux2-klein-edit",
        "klein-4b",
        "klein-4b-edit",
    ),
    ("edit", "flux2-klein-9b"): (
        "flux2-klein-9b",
        "flux2-klein-9b-edit",
        "klein-9b",
        "klein-9b-edit",
    ),
    ("edit", "qwen-image-edit"): (
        "qwen-image-edit",
        "qwen-edit",
        "qwen-edit-plus",
        "qwen-edit-2509",
    ),
    ("edit", "fibo-edit"): ("fibo-edit", "fiboedit"),
    ("edit", "ernie-image-turbo"): ("ernie-image-turbo",),
    ("edit", "ernie-image"): ("ernie-image",),
}


def normalize_image_alias(value: object) -> str:
    """Normalize user/model-folder aliases without depending on mflux imports."""
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower().replace("_", "-").replace(".", "-")
    normalized = re.sub(r"\s+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    if normalized.startswith("flux-2-"):
        normalized = "flux2-" + normalized[len("flux-2-"):]
    return normalized


IMAGE_MODEL_SPECS: tuple[ImageModelSpec, ...] = (
    ImageModelSpec(
        base_model="flux2-klein-4b",
        tasks=("generation", "edit"),
        estimated_size=8 * 1024**3,
        discovery_aliases=("flux2-klein-4b", "flux2-klein"),
    ),
    ImageModelSpec(
        base_model="flux2-klein-9b",
        tasks=("generation", "edit"),
        estimated_size=18 * 1024**3,
        discovery_aliases=("flux2-klein-9b",),
    ),
    ImageModelSpec(
        base_model="z-image-turbo",
        tasks=("generation",),
        estimated_size=12 * 1024**3,
        discovery_aliases=("z-image-turbo", "zimage-turbo"),
    ),
    ImageModelSpec(
        base_model="z-image",
        tasks=("generation",),
        estimated_size=12 * 1024**3,
        discovery_aliases=("z-image", "zimage"),
    ),
    ImageModelSpec(
        base_model="qwen-image-edit",
        tasks=("edit",),
        estimated_size=20 * 1024**3,
        discovery_aliases=("qwen-image-edit",),
    ),
    ImageModelSpec(
        base_model="qwen-image",
        tasks=("generation",),
        estimated_size=20 * 1024**3,
        discovery_aliases=("qwen-image",),
    ),
    ImageModelSpec(
        base_model="fibo-edit",
        tasks=("edit",),
        estimated_size=12 * 1024**3,
        discovery_aliases=("fibo-edit", "fiboedit"),
    ),
    ImageModelSpec(
        base_model="fibo",
        tasks=("generation",),
        estimated_size=12 * 1024**3,
        discovery_aliases=("fibo",),
    ),
    ImageModelSpec(
        base_model="ernie-image-turbo",
        tasks=("generation", "edit"),
        estimated_size=22 * 1024**3,
        discovery_aliases=("ernie-image-turbo",),
    ),
    ImageModelSpec(
        base_model="ernie-image",
        tasks=("generation", "edit"),
        estimated_size=22 * 1024**3,
        discovery_aliases=("ernie-image",),
    ),
    ImageModelSpec(
        base_model="ideogram-4-fp8",
        tasks=("generation",),
        estimated_size=28 * 1024**3,
        discovery_aliases=(
            "ideogram-4-fp8",
            "ideogram4-fp8",
            "ideogram4",
            "ideogram-4",
            "ideogram",
        ),
    ),
)

IMAGE_DEFAULT_ESTIMATED_SIZES = {
    normalize_image_alias(alias): spec.estimated_size
    for spec in IMAGE_MODEL_SPECS
    for alias in (
        spec.base_model,
        *(
            alias
            for (task, base_model), aliases in IMAGE_ENGINE_ALIASES.items()
            if base_model == spec.base_model
            for alias in aliases
        ),
    )
}

# Per-model quality defaults (mflux defaults are too low for good quality).
# These values are used when the model has no manifest-level defaults.
IMAGE_DEFAULTS: dict[str, dict[str, int | float]] = {
    "flux2-klein-4b": {
        "default_steps": 8,
    },
    "flux2-klein-9b": {
        "default_steps": 8,
    },
    "z-image-turbo": {
        "default_steps": 8,
    },
    "z-image": {
        "default_steps": 50,
    },
    "qwen-image": {
        "default_steps": 50,
        "default_guidance": 10.0,
    },
    "qwen-image-edit": {
        "default_steps": 50,
        "default_guidance": 10.0,
    },
    "ernie-image-turbo": {
        "default_steps": 8,
        "default_guidance": 1.0,
        "default_image_strength": 0.4,
    },
    "ernie-image": {
        "default_steps": 50,
        "default_guidance": 4.0,
        "default_image_strength": 0.4,
    },
}


def image_engine_aliases(task: ImageTask, base_model: str) -> tuple[str, ...]:
    """Return supported engine aliases for a base model/task pair."""
    return IMAGE_ENGINE_ALIASES.get((task, normalize_image_alias(base_model)), ())


def get_image_defaults(base_model: str) -> dict[str, int | float]:
    """Return quality-adjusted defaults for a given base model.

    Falls back to empty dict when no model-specific defaults exist.
    """
    return IMAGE_DEFAULTS.get(normalize_image_alias(base_model), {})



def _matches_image_model_prefix(normalized_name: str, alias: str) -> bool:
    normalized_alias = normalize_image_alias(alias)
    if normalized_name == normalized_alias or normalized_name.startswith(
        f"{normalized_alias}-"
    ):
        return True

    compact_name = normalized_name.replace("-", "")
    compact_alias = normalized_alias.replace("-", "")
    return compact_name == compact_alias or compact_name.startswith(compact_alias)


def infer_image_model_spec_from_name(name: str) -> ImageModelSpec | None:
    """Return the supported image spec implied by a local model folder name."""
    normalized_name = normalize_image_alias(name)
    candidates = [
        (len(normalize_image_alias(alias).replace("-", "")), spec, alias)
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
