# SPDX-License-Identifier: Apache-2.0
"""Tests for the vendored mlx-vlm Z-Image and ERNIE-Image registration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from omlx.patches.mlx_vlm_image_compat import apply_mlx_vlm_image_compat_patch


@pytest.fixture(scope="module")
def image_apis():
    pytest.importorskip("mlx_vlm")
    apply_mlx_vlm_image_compat_patch()
    from mlx_vlm.generate import edit_image, image

    return image, edit_image


@pytest.mark.parametrize(
    ("alias", "generation_name", "edit_name"),
    [
        ("z-image", "ZImageGenerationModel", "ZImageEditModel"),
        ("ernie-image-turbo", "ErnieImageGenerationModel", "ErnieImageEditModel"),
    ],
)
def test_vendored_image_families_register_with_generic_dispatch(
    image_apis, alias, generation_name, edit_name
):
    image, edit_image = image_apis

    generation = image.image_generation_model_class(alias)
    edit = edit_image.image_edit_model_class(alias)

    assert generation is not None
    assert edit is not None
    assert generation.__name__ == generation_name
    assert edit.__name__ == edit_name


def test_existing_image_families_receive_backported_default_properties(image_apis):
    from mlx_vlm.models.flux2.model import (
        Flux2ImageEditModel,
        Flux2ImageGenerationModel,
    )
    from mlx_vlm.models.mage_flow.model import (
        MageFlowImageEditModel,
        MageFlowImageGenerationModel,
    )
    from mlx_vlm.models.z_image.model import ZImageEditModel, ZImageGenerationModel

    for model_class in (
        Flux2ImageGenerationModel,
        Flux2ImageEditModel,
        MageFlowImageGenerationModel,
        MageFlowImageEditModel,
        ZImageGenerationModel,
        ZImageEditModel,
    ):
        assert isinstance(model_class.default_steps, property)
        assert isinstance(model_class.default_guidance, property)

    z_generation = object.__new__(ZImageGenerationModel)
    z_generation.pipeline = SimpleNamespace(
        config=SimpleNamespace(default_steps=9, default_guidance=0.0)
    )
    assert z_generation.default_steps == 9
    assert z_generation.default_guidance == 0.0
    assert z_generation.default_width == 1024
    assert z_generation.default_height == 1024

    z_edit = object.__new__(ZImageEditModel)
    z_edit.pipeline = SimpleNamespace(
        config=SimpleNamespace(variant="turbo", default_guidance=0.0)
    )
    assert z_edit.default_steps == 8
    assert z_edit.default_guidance == 0.0


@pytest.mark.parametrize("alias", ["z-image", "ernie-image-turbo"])
def test_generic_loader_dispatches_vendored_families_without_weights(
    image_apis, alias, monkeypatch
):
    image, edit_image = image_apis
    generation_class = image.image_generation_model_class(alias)
    edit_class = edit_image.image_edit_model_class(alias)
    generation_sentinel = object()
    edit_sentinel = object()

    def fake_generation_load(cls, model, **kwargs):
        assert model == alias
        return generation_sentinel

    def fake_edit_load(cls, model, **kwargs):
        assert model == alias
        return edit_sentinel

    monkeypatch.setattr(
        generation_class,
        "from_model_id",
        classmethod(fake_generation_load),
    )
    monkeypatch.setattr(edit_class, "from_model_id", classmethod(fake_edit_load))
    monkeypatch.setattr(image, "_resolve_image_model_path", lambda *args, **kwargs: None)

    assert image.load_image_model(alias, task="generate") is generation_sentinel
    assert image.load_image_model(alias, task="edit") is edit_sentinel


def test_generic_api_resolves_nullable_request_defaults(image_apis):
    image, edit_image = image_apis

    @dataclass
    class GenerationModel:
        default_steps = 11
        default_guidance = 2.5
        default_width = 768
        default_height = 512

        def generate(self, request):
            return request

    @dataclass
    class EditModel:
        default_steps = 9
        default_guidance = 1.5

        def edit(self, request):
            return request

    generation_request = image.ImageGenerationRequest(
        prompt="a fox",
        seed=1,
    )
    assert generation_request.steps is None
    assert generation_request.width is None
    assert generation_request.height is None
    assert generation_request.guidance is None
    generation_result = image.generate_image(GenerationModel(), generation_request)
    assert generation_result.steps == 11
    assert generation_result.guidance == 2.5
    assert generation_result.width == 768
    assert generation_result.height == 512

    edit_request = edit_image.ImageEditRequest(
        prompt="make it blue",
        image_paths=("input.png",),
        seed=1,
    )
    assert edit_request.steps is None
    assert edit_request.guidance is None
    edit_result = edit_image.edit_image(EditModel(), edit_request)
    assert edit_result.steps == 9
    assert edit_result.guidance == 1.5


@pytest.mark.parametrize(
    ("markers", "expected_model_type"),
    [
        (
            {
                "layers.0.feed_forward.w1.weight": "weights.safetensors",
                "context_refiner.0.attention.to_q.weight": "weights.safetensors",
                "noise_refiner.0.adaLN_modulation.0.weight": "weights.safetensors",
            },
            "z_image",
        ),
        (
            {
                "adaln_modulation.weight": "weights.safetensors",
                "final_norm.linear.weight": "weights.safetensors",
                "layers.0.adaLN_sa_ln.weight": "weights.safetensors",
            },
            "ernie_image",
        ),
    ],
)
def test_component_indexes_dispatch_to_vendored_models(
    image_apis, markers, expected_model_type, tmp_path
):
    image, _ = image_apis
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    (transformer / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": markers})
    )

    assert image._image_model_type_from_component_indexes(tmp_path) == expected_model_type


def test_ernie_legacy_mlx_checkpoint_layout_is_preserved(image_apis):
    from mlx_vlm.models.ernie_image.weights import _tensor_layout

    assert _tensor_layout({"mflux_version": "0.1"}) == "mlx_nhwc"
