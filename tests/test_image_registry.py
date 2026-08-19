# SPDX-License-Identifier: Apache-2.0
"""Tests for mlx-vlm image model registry and discovery."""

from __future__ import annotations

import json

import pytest

from omlx.image_registry import (
    get_image_defaults,
    get_image_model_spec,
    image_edit_accepts_multiple_inputs,
    infer_image_model_spec_from_name,
)
from omlx.model_discovery import (
    _load_image_manifest,
    detect_model_type,
    estimate_image_model_size,
)


@pytest.mark.parametrize(
    ("alias", "base_model", "tasks", "defaults"),
    [
        (
            "microsoft/Mage-Flow-Base",
            "mage-flow-base",
            ("generation",),
            {"default_steps": 30, "default_guidance": 5.0},
        ),
        (
            "Mage-Flow-Aligned",
            "mage-flow",
            ("generation",),
            {"default_steps": 20, "default_guidance": 5.0},
        ),
        (
            "Mage-Flow-Turbo",
            "mage-flow-turbo",
            ("generation",),
            {"default_steps": 4, "default_guidance": 1.0},
        ),
        (
            "Mage-Flow-Edit-Base",
            "mage-flow-edit-base",
            ("edit",),
            {"default_steps": 30, "default_guidance": 5.0},
        ),
        (
            "Mage-Flow-Edit-Aligned",
            "mage-flow-edit",
            ("edit",),
            {"default_steps": 30, "default_guidance": 5.0},
        ),
        (
            "Mage-Flow-Edit-Turbo",
            "mage-flow-edit-turbo",
            ("edit",),
            {"default_steps": 4, "default_guidance": 1.0},
        ),
    ],
)
def test_mage_flow_aliases_expose_expected_tasks_and_defaults(
    alias, base_model, tasks, defaults
):
    spec = get_image_model_spec(alias)

    assert spec is not None
    assert spec.base_model == base_model
    assert spec.tasks == tasks
    assert get_image_defaults(alias) == defaults
    assert image_edit_accepts_multiple_inputs(alias) is (tasks == ("edit",))


@pytest.mark.parametrize(
    ("name", "base_model", "tasks"),
    [
        ("FLUX.2-klein-base-9B", "flux2-klein-base-9b", ("generation", "edit")),
        ("Z-Image-Turbo", "z-image-turbo", ("generation", "edit")),
        ("ERNIE-Image", "ernie-image", ("generation", "edit")),
        ("ideogram-4-fp8", "ideogram-4-fp8", ("generation",)),
        ("bonsai-image-ternary-4B-mlx-2bit", "ternary", ("generation",)),
    ],
)
def test_supported_image_names_infer_registry_specs(name, base_model, tasks):
    spec = infer_image_model_spec_from_name(name)

    assert spec is not None
    assert spec.base_model == base_model
    assert spec.tasks == tasks


def test_z_image_turbo_has_task_specific_img2img_steps():
    assert get_image_defaults("z-image-turbo")["default_steps"] == 9
    assert get_image_defaults("z-image-turbo", "edit")["default_steps"] == 8


def test_local_mage_edit_layout_is_discovered_as_mlx_vlm_image(tmp_path):
    model_dir = tmp_path / "Mage-Flow-Edit-Turbo"
    model_dir.mkdir()
    (model_dir / "tokenizer").mkdir()
    (model_dir / "tokenizer" / "tokenizer.json").write_text("{}")
    for component in ("transformer", "vae"):
        component_dir = model_dir / component
        component_dir.mkdir()
        (component_dir / "model.safetensors").write_bytes(b"weights")

    manifest = _load_image_manifest(model_dir)

    assert manifest is not None
    assert manifest.backend == "mlx-vlm"
    assert manifest.base_model == "mage-flow-edit-turbo"
    assert manifest.tasks == ["edit"]
    assert manifest.metadata["model_path"] == "."
    assert detect_model_type(model_dir) == "image"


def test_manifest_keeps_quantize_for_clear_engine_validation(tmp_path):
    model_dir = tmp_path / "flux"
    model_dir.mkdir()
    (model_dir / "omlx-image-model.json").write_text(
        json.dumps(
            {
                "backend": "mlx-vlm",
                "base_model": "flux2-klein-4b",
                "task": ["generation"],
                "quantize": 4,
            }
        )
    )

    manifest = _load_image_manifest(model_dir)

    assert manifest is not None
    assert manifest.metadata["quantize"] == 4
    assert manifest.backend == "mlx-vlm"


def test_unrecognized_vlm_layout_is_not_inferred_as_an_image_model(tmp_path):
    model_dir = tmp_path / "qwen-vl"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2_vl",
                "vision_config": {"hidden_size": 128},
                "architectures": ["Qwen2VLForConditionalGeneration"],
            }
        )
    )
    (model_dir / "tokenizer").mkdir()
    (model_dir / "tokenizer" / "tokenizer.json").write_text("{}")
    for component in ("transformer", "vae"):
        component_dir = model_dir / component
        component_dir.mkdir()
        (component_dir / "model.safetensors").write_bytes(b"weights")

    assert _load_image_manifest(model_dir) is None
    assert detect_model_type(model_dir) == "vlm"


def test_estimate_uses_unadjusted_family_fallback_without_quantize(tmp_path):
    model_dir = tmp_path / "z-image"
    model_dir.mkdir()
    (model_dir / "omlx-image-model.json").write_text(
        json.dumps(
            {
                "backend": "mlx-vlm",
                "base_model": "z-image",
                "task": "generation",
                "quantize": 4,
            }
        )
    )
    manifest = _load_image_manifest(model_dir)

    assert manifest is not None
    assert estimate_image_model_size(model_dir, manifest) == 12 * 1024**3
