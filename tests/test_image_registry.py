# SPDX-License-Identifier: Apache-2.0
"""Tests for mlx-vlm image model registry and discovery."""

from __future__ import annotations

import json

import pytest

from omlx.image_registry import (
    get_image_model_reference,
    get_image_model_spec,
    image_edit_accepts_multiple_inputs,
    infer_image_model_spec_from_name,
)
from omlx.model_discovery import (
    _load_image_manifest,
    detect_image_tasks,
    detect_model_type,
    discover_models,
    estimate_image_model_size,
)

TURBO_SIGMAS = [1, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]


def _qwen21_layout(tmp_path, name="renamed-image-model", *, sigmas=None):
    model_dir = tmp_path / name
    (model_dir / "tokenizer").mkdir(parents=True)
    index = {
        "_class_name": "QwenImage21Pipeline",
        "_diffusers_version": "0.41.0.dev0",
        "processor": ["transformers", "Qwen3VLProcessor"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLQwenImage21"],
    }
    if sigmas is not None:
        index["sample_sigmas"] = sigmas
    (model_dir / "model_index.json").write_text(json.dumps(index))
    for component in ("transformer", "vae", "text_encoder"):
        directory = model_dir / component
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(b"weights")
        (directory / "config.json").write_text("{}")
    (model_dir / "processor").mkdir()
    (model_dir / "processor" / "tokenizer.json").write_text("{}")
    (model_dir / "transformer" / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImage21Transformer2DModel",
                "attention_head_dim": 128,
                "num_attention_heads": 32,
                "num_layers": 32,
                "in_channels": 64,
                "out_channels": 64,
                "patch_size": 1,
                "causal_condition": True,
            }
        )
    )
    return model_dir


@pytest.mark.parametrize(
    ("sigmas", "base_model", "steps"),
    [
        (None, "qwen-image-2-1", None),
        (TURBO_SIGMAS, "qwen-image-2-1-turbo", 8),
        ([1, 0.5], "qwen-image-2-1", 2),
    ],
)
def test_qwen21_intrinsic_discovery_survives_renaming(
    tmp_path, sigmas, base_model, steps
):
    model_dir = _qwen21_layout(tmp_path, sigmas=sigmas)
    manifest = _load_image_manifest(model_dir)
    assert manifest is not None
    assert manifest.base_model == base_model
    assert manifest.tasks == ["generation", "edit"]
    assert manifest.metadata.get("default_steps") == steps
    assert manifest.metadata.get("checkpoint_steps") == steps
    assert manifest.metadata["uses_checkpoint_sigmas"] is (sigmas is not None)
    assert detect_model_type(model_dir) == "image"
    assert image_edit_accepts_multiple_inputs(base_model)


@pytest.mark.parametrize(
    "pipeline", ["QwenImagePipeline", "QwenImageEditPipeline", "OtherPipeline"]
)
def test_qwen_folder_name_does_not_override_component_identity(tmp_path, pipeline):
    model_dir = _qwen21_layout(tmp_path, "Qwen-Image-2.1-Turbo")
    index_path = model_dir / "model_index.json"
    index = json.loads(index_path.read_text())
    index["_class_name"] = pipeline
    index_path.write_text(json.dumps(index))
    assert _load_image_manifest(model_dir) is None


@pytest.mark.parametrize(
    "sigmas", [[], [1, 0], [0.5, 1], [True, 0.5], [float("nan")], "bad"]
)
def test_qwen21_invalid_saved_grid_is_not_inferred(tmp_path, sigmas):
    assert _load_image_manifest(_qwen21_layout(tmp_path, sigmas=sigmas)) is None


def test_qwen21_explicit_manifest_keeps_effective_recipe_metadata(tmp_path):
    model_dir = _qwen21_layout(tmp_path, sigmas=TURBO_SIGMAS)
    (model_dir / "omlx-image-model.json").write_text(
        json.dumps(
            {
                "backend": "mlx-vlm",
                "base_model": "qwen-image-2.1-turbo",
                "task": ["generation", "edit"],
                "model_path": ".",
                "default_steps": 25,
            }
        )
    )
    manifest = _load_image_manifest(model_dir)
    assert manifest is not None
    assert manifest.metadata["default_steps"] == 25
    assert manifest.metadata["checkpoint_steps"] == 8


def test_qwen21_aliases_and_runtime_identity():
    for alias in ("qwen-image-2.1-turbo", "Qwen/Qwen-Image-2.1-Turbo"):
        assert get_image_model_reference(alias) == "Qwen/Qwen-Image-2.1-Turbo"
    assert get_image_model_spec("qwen-image-edit") is None


@pytest.mark.parametrize(
    ("alias", "base_model", "tasks"),
    [
        (
            "microsoft/Mage-Flow-Base",
            "mage-flow-base",
            ("generation",),
        ),
        (
            "Mage-Flow-Aligned",
            "mage-flow",
            ("generation",),
        ),
        (
            "Mage-Flow-Turbo",
            "mage-flow-turbo",
            ("generation",),
        ),
        (
            "Mage-Flow-Edit-Base",
            "mage-flow-edit-base",
            ("edit",),
        ),
        (
            "Mage-Flow-Edit-Aligned",
            "mage-flow-edit",
            ("edit",),
        ),
        (
            "Mage-Flow-Edit-Turbo",
            "mage-flow-edit-turbo",
            ("edit",),
        ),
    ],
)
def test_mage_flow_aliases_expose_metadata_hints(alias, base_model, tasks):
    spec = get_image_model_spec(alias)

    assert spec is not None
    assert spec.base_model == base_model
    assert spec.tasks == tasks
    assert image_edit_accepts_multiple_inputs(alias) is (tasks == ("edit",))


@pytest.mark.parametrize(
    ("name", "base_model", "tasks"),
    [
        ("FLUX.2-klein-base-9B", "flux2-klein-base-9b", ("generation", "edit")),
        ("Z-Image-Turbo", "z-image-turbo", ("generation", "edit")),
        ("ERNIE-Image", "ernie-image", ("generation", "edit")),
        ("Ming-Image-0.1-Design-mxfp8", "ming-image-0-1-design", ("generation",)),
        ("ideogram-4-fp8", "ideogram-4-fp8", ("generation",)),
        ("bonsai-image-ternary-4B-mlx-2bit", "ternary", ("generation",)),
    ],
)
def test_supported_image_names_infer_registry_specs(name, base_model, tasks):
    spec = infer_image_model_spec_from_name(name)

    assert spec is not None
    assert spec.base_model == base_model
    assert spec.tasks == tasks


def test_local_mage_edit_layout_is_discovered_as_mlx_vlm_image(tmp_path):
    model_dir = tmp_path / "Mage-Flow-Edit-Turbo"
    _write_mage_checkpoint(model_dir)

    manifest = _load_image_manifest(model_dir)

    assert manifest is not None
    assert manifest.backend == "mlx-vlm"
    assert manifest.base_model == "mage-flow-edit-turbo"
    assert manifest.tasks == ["edit"]
    assert manifest.metadata["model_path"] == "."
    assert detect_model_type(model_dir) == "image"


@pytest.mark.parametrize(
    ("folder", "weight_file"),
    [
        ("Ming-Image-0.1-Design", "diffusion_pytorch_model.safetensors"),
        ("Ming-Image-0.1-Design-mxfp8", "model.safetensors"),
    ],
)
def test_local_ming_layout_is_discovered_as_generation_only(
    tmp_path, folder, weight_file
):
    model_dir = tmp_path / folder
    tokenizer_dir = model_dir / "mllm"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer.json").write_text("{}")
    (tokenizer_dir / "config.json").write_text(
        json.dumps({"architectures": ["BailingMMForConditionalGeneration"]})
    )
    for component in ("transformer", "vae"):
        component_dir = model_dir / component
        component_dir.mkdir()
        (component_dir / weight_file).write_bytes(b"weights")
        (component_dir / "config.json").write_text(
            json.dumps({"_class_name": "DiffusionTransformer"})
            if component == "transformer"
            else "{}"
        )

    manifest = _load_image_manifest(model_dir)

    assert manifest is not None
    assert manifest.backend == "mlx-vlm"
    assert manifest.base_model == "ming-image-0-1-design"
    assert manifest.tasks == ["generation"]
    assert manifest.metadata["model_path"] == "."
    assert detect_model_type(model_dir) == "image"


@pytest.mark.parametrize(
    ("name", "tasks"),
    [
        ("Mage-Flow", ["generation"]),
        ("Mage-Flow-Turbo", ["generation"]),
        ("Mage-Flow-Edit", ["edit"]),
        ("Mage-Flow-Edit-Turbo", ["edit"]),
    ],
)
def test_mage_flow_text_encoder_tokenizer_layout_is_discovered(tmp_path, name, tasks):
    """Mage-Flow community checkpoints keep tokenizer files in ``text_encoder/``."""
    model_dir = tmp_path / name
    _write_mage_checkpoint(model_dir)

    manifest = _load_image_manifest(model_dir)

    assert manifest is not None
    assert manifest.backend == "mlx-vlm"
    assert manifest.tasks == tasks
    assert manifest.metadata["model_path"] == "."
    assert detect_model_type(model_dir) == "image"


def _write_mage_checkpoint(model_dir):
    text_encoder = model_dir / "text_encoder"
    text_encoder.mkdir(parents=True)
    (text_encoder / "tokenizer.json").write_text("{}")
    (text_encoder / "tokenizer_config.json").write_text("{}")
    for component in ("transformer", "vae"):
        component_dir = model_dir / component
        component_dir.mkdir()
    for component in ("transformer", "vae", "text_encoder"):
        component_dir = model_dir / component
        (component_dir / "config.json").write_text("{}")
        (component_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    (model_dir / "model_index.json").write_text('{"_class_name": "MageFlowPipeline"}')


@pytest.mark.parametrize(
    ("variant", "tasks"),
    [
        ("mage-flow-turbo", ["generation"]),
        ("mage-flow-edit-turbo", ["edit"]),
    ],
)
def test_backend_recipe_overrides_folder_name_and_registry_tasks(
    tmp_path, variant, tasks
):
    model_dir = tmp_path / (
        "Mage-Flow-Edit-Turbo" if tasks == ["generation"] else "Mage-Flow-Turbo"
    )
    _write_mage_checkpoint(model_dir)
    (model_dir / "mlx_mage_flow.json").write_text(json.dumps({"variant": variant}))

    assert detect_image_tasks(model_dir) == tasks
    model = discover_models(tmp_path)[model_dir.name]
    assert model.model_type == model.engine_type == "image"
    assert model.tasks == model.capabilities == tasks


def test_renamed_checkpoint_tasks_do_not_require_registry_alias(tmp_path):
    model_dir = tmp_path / "my-custom-checkpoint"
    _write_mage_checkpoint(model_dir)
    (model_dir / "mlx_mage_flow.json").write_text(
        json.dumps({"variant": "mage-flow-edit-turbo"})
    )
    (model_dir / "transformer" / "config.json").write_text(
        '{"_class_name": "MageFlowTransformer2DModel"}'
    )

    assert infer_image_model_spec_from_name(model_dir.name) is None
    model = discover_models(tmp_path)[model_dir.name]
    assert model.model_type == model.engine_type == "image"
    assert model.tasks == model.capabilities == ["edit"]


def test_qwen_folder_name_does_not_hide_supported_mage_checkpoint(tmp_path):
    model_dir = tmp_path / "Qwen-Image-2.1-Turbo"
    _write_mage_checkpoint(model_dir)
    (model_dir / "mlx_mage_flow.json").write_text(
        json.dumps({"variant": "mage-flow-edit-turbo"})
    )

    assert detect_image_tasks(model_dir) == ["edit"]
    model = discover_models(tmp_path)[model_dir.name]
    assert model.tasks == model.capabilities == ["edit"]
    assert model.image_metadata["base_model"] == "MageFlowPipeline"


def test_incomplete_named_checkpoint_is_not_discovered(tmp_path):
    model_dir = tmp_path / "Mage-Flow-Edit-Turbo"
    _write_mage_checkpoint(model_dir)
    (model_dir / "text_encoder" / "config.json").unlink()
    assert detect_image_tasks(model_dir) == []
    assert model_dir.name not in discover_models(tmp_path)


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
