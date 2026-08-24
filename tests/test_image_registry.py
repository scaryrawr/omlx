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
    discover_models,
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


@pytest.mark.parametrize(
    ("name", "tasks"),
    [
        ("Mage-Flow", ["generation"]),
        ("Mage-Flow-Turbo", ["generation"]),
        ("Mage-Flow-Edit", ["edit"]),
        ("Mage-Flow-Edit-Turbo", ["edit"]),
    ],
)
def test_nested_text_encoder_tokenizer_layout_is_discovered(tmp_path, name, tasks):
    model_dir = tmp_path / name
    text_encoder = model_dir / "text_encoder"
    text_encoder.mkdir(parents=True)
    (text_encoder / "tokenizer.json").write_text("{}")
    for component in ("transformer", "vae"):
        component_dir = model_dir / component
        component_dir.mkdir()
        (component_dir / "model.safetensors").write_bytes(b"weights")

    manifest = _load_image_manifest(model_dir)

    assert manifest is not None
    assert manifest.backend == "mlx-vlm"
    assert manifest.tasks == tasks
    assert manifest.metadata["model_path"] == "."
    assert detect_model_type(model_dir) == "image"


def test_explicit_manifest_preserves_engine_validation_fields(tmp_path):
    model_dir = tmp_path / "flux"
    model_dir.mkdir()
    (model_dir / "omlx-image-model.json").write_text(
        json.dumps(
            {
                "backend": "mlx-vlm",
                "base_model": "flux2-klein-4b",
                "task": ["generation", "edit"],
                "quantize": 4,
            }
        )
    )

    manifest = _load_image_manifest(model_dir)

    assert manifest is not None
    assert manifest.tasks == ["generation", "edit"]
    assert manifest.metadata["quantize"] == 4
    assert estimate_image_model_size(model_dir, manifest) == 8 * 1024**3


def test_unrecognized_vlm_layout_is_not_inferred_as_image(tmp_path):
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


def test_hf_cache_image_uses_repository_id_for_safe_inference(tmp_path):
    cache = tmp_path / "models--microsoft--Mage-Flow"
    snapshot = cache / "snapshots" / "abc123"
    (cache / "refs").mkdir(parents=True)
    (cache / "refs" / "main").write_text("abc123")
    text_encoder = snapshot / "text_encoder"
    text_encoder.mkdir(parents=True)
    (text_encoder / "tokenizer.json").write_text("{}")
    for component in ("transformer", "vae"):
        component_dir = snapshot / component
        component_dir.mkdir()
        (component_dir / "model.safetensors").write_bytes(b"weights")

    models = discover_models(tmp_path)

    assert models["microsoft--Mage-Flow"].model_type == "image"
    assert models["microsoft--Mage-Flow"].source_repo_id == "microsoft/Mage-Flow"
