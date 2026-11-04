# SPDX-License-Identifier: Apache-2.0
"""Tests for native mlx-vlm image models and oMLX compatibility."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from omlx.patches.mlx_vlm_image_compat import (
    apply_mlx_vlm_image_compat_patch,
)


@pytest.mark.parametrize(
    ("family", "expected_steps", "expected_guidance"),
    [("mage-flow", 4, 1.0), ("z-image", 9, 0.0), ("ming-image", 12, 1.0)],
)
@pytest.mark.parametrize("overrides", ["backend", "manifest", "request"])
async def test_engine_uses_native_sampling_with_manifest_and_request_precedence(
    image_apis,
    monkeypatch,
    tmp_path,
    family,
    expected_steps,
    expected_guidance,
    overrides,
):
    import mlx.core as mx

    import omlx.engine.image as engine_module
    from omlx.engine.image import ImageEngine

    image, edit_image = image_apis
    calls = []

    def generate_array(prompt, **kwargs):
        calls.append(kwargs)
        return mx.zeros((2, 2, 3), dtype=mx.uint8)

    pipeline = SimpleNamespace(
        model_path=tmp_path,
        generate_array=generate_array,
        count_prompt_tokens=lambda prompt: 0,
        quantization_config=None,
    )
    if family == "mage-flow":
        from mlx_vlm.models.mage_flow.config import get_variant
        from mlx_vlm.models.mage_flow.model import MageFlowImageGenerationModel

        pipeline.variant = get_variant("mage-flow-turbo")
        model = MageFlowImageGenerationModel(pipeline, "local-model")
    elif family == "z-image":
        from mlx_vlm.models.z_image.model import ZImageGenerationModel

        pipeline.config = SimpleNamespace(
            variant="turbo", default_steps=9, default_guidance=0.0
        )
        model = ZImageGenerationModel(pipeline, "local-model")
    else:
        from mlx_vlm.models.ming_image.model import MingImageGenerationModel

        pipeline.config = SimpleNamespace(default_steps=12, default_guidance=1.0)
        model = MingImageGenerationModel(pipeline, "local-model")

    monkeypatch.setattr(
        engine_module,
        "_image_api",
        lambda: SimpleNamespace(
            load_image_model=lambda *args, **kwargs: model,
            generate_image=image.generate_image,
            ImageGenerationRequest=image.ImageGenerationRequest,
            ImageEditRequest=edit_image.ImageEditRequest,
        ),
    )
    metadata = {"backend": "mlx-vlm", "base_model": family}
    if overrides != "backend":
        metadata.update(default_steps=7, default_guidance=0.75)
        expected_steps, expected_guidance = 7, 0.75
    request = {}
    if overrides == "request":
        request = {"steps": 3, "guidance": 0.5}
        expected_steps, expected_guidance = 3, 0.5
    engine = ImageEngine(
        model_name="local-model",
        model_path=str(tmp_path),
        image_metadata=metadata,
        tasks=["generation"],
    )
    await engine.start()
    try:
        result = await engine.generate("a fox", **request)
        assert calls[-1]["steps"] == expected_steps
        assert calls[-1]["guidance"] == expected_guidance
        assert calls[-1]["width"] == calls[-1]["height"] == 512
        assert result.metadata["steps"] == expected_steps
        assert result.metadata["guidance"] == expected_guidance
        assert result.image.size == (2, 2)
    finally:
        await engine.stop()


@pytest.fixture(scope="module")
def image_apis():
    pytest.importorskip("mlx_vlm")
    apply_mlx_vlm_image_compat_patch()
    image = importlib.import_module("mlx_vlm.generate.image")
    edit_image = importlib.import_module("mlx_vlm.generate.edit_image")

    return image, edit_image


@pytest.mark.parametrize("name", ["Qwen-Image-2.1-Turbo", "custom-checkpoint"])
def test_native_generation_loader_prefers_supported_checkpoint_metadata(
    image_apis, monkeypatch, tmp_path, name
):
    from mlx_vlm.models.mage_flow.model import MageFlowImageGenerationModel

    image, _ = image_apis
    root = tmp_path / name
    for component in ("transformer", "vae", "text_encoder"):
        directory = root / component
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}")
        (directory / "model.safetensors").write_bytes(b"weights")
    (root / "text_encoder" / "tokenizer.json").write_text("{}")
    (root / "model_index.json").write_text('{"_class_name":"MageFlowPipeline"}')
    (root / "mlx_mage_flow.json").write_text('{"variant":"mage-flow-turbo"}')
    calls = []
    model = object()

    def from_model_id(cls, model_id, **kwargs):
        calls.append((model_id, kwargs))
        return model

    monkeypatch.setattr(
        MageFlowImageGenerationModel, "from_model_id", classmethod(from_model_id)
    )

    assert image.is_image_generation_model(str(root))
    assert image.load_image_model(str(root), task="generate") is model
    assert calls == [(str(root), {"model_path": root})]


@pytest.mark.parametrize(
    ("alias", "generation_name", "edit_name"),
    [
        ("z-image", "ZImageGenerationModel", "ZImageEditModel"),
        ("ernie-image-turbo", "ErnieImageGenerationModel", "ErnieImageEditModel"),
    ],
)
def test_image_families_register_with_generic_dispatch(
    image_apis, alias, generation_name, edit_name
):
    image, edit_image = image_apis

    generation = image.image_generation_model_class(alias)
    edit = edit_image.image_edit_model_class(alias)

    assert generation is not None
    assert edit is not None
    assert generation.__name__ == generation_name
    assert edit.__name__ == edit_name


def test_z_image_uses_native_mlx_vlm_package(image_apis):
    import mlx_vlm.models.z_image as z_image

    assert "omlx/patches" not in str(z_image.__file__)


@pytest.mark.parametrize("family", ["z-image", "ernie-image"])
def test_native_single_source_models_reject_multiple_references(image_apis, family):
    _, edit_image = image_apis
    if family == "z-image":
        from mlx_vlm.models.z_image.model import ZImageEditModel
        from mlx_vlm.models.z_image.pipeline import ZImagePipeline

        pipeline = object.__new__(ZImagePipeline)
        pipeline.config = SimpleNamespace(
            variant="turbo", default_steps=9, default_guidance=0.0
        )
        model = ZImageEditModel(pipeline, "local-model")
    else:
        from mlx_vlm.models.ernie_image.model import ErnieImageEditModel

        model = ErnieImageEditModel(SimpleNamespace(), "local-model")
    request = edit_image.ImageEditRequest(
        prompt="combine", image_paths=("first.png", "second.png")
    )
    with pytest.raises(ValueError, match="exactly one source image"):
        edit_image.edit_image(model, request)


def test_ernie_image_uses_native_mlx_vlm_package(image_apis):
    import mlx_vlm.models.ernie_image as ernie_image

    assert "omlx/patches" not in str(ernie_image.__file__)


def test_ming_image_registers_generation_only_and_uses_native_package(image_apis):
    image, edit_image = image_apis
    from mlx_vlm.models.ming_image.model import MingImageGenerationModel

    assert "omlx/patches" not in str(
        importlib.import_module("mlx_vlm.models.ming_image").__file__
    )
    assert (
        image.image_generation_model_class("ming-image-0.1-design")
        is MingImageGenerationModel
    )
    assert edit_image.image_edit_model_class("ming-image-0.1-design") is None

    ming = object.__new__(MingImageGenerationModel)
    ming.pipeline = SimpleNamespace(
        config=SimpleNamespace(default_steps=12, default_guidance=1.0)
    )
    assert ming.default_sampling.steps == 12
    assert ming.default_sampling.guidance == 1.0

    request = image.ImageGenerationRequest(prompt="a poster")
    assert request.width == request.height == 512
    assert request.steps is request.guidance is None


def test_unlimited_ocr_uses_native_mlx_vlm_support():
    import mlx_vlm.models.unlimited_ocr as unlimited_ocr
    from mlx_vlm.prompt_utils import get_message_json
    from mlx_vlm.utils import MODEL_REMAPPING, get_model_and_args

    assert "omlx/patches" not in str(unlimited_ocr.__file__)
    assert MODEL_REMAPPING["unlimited-ocr"] == "unlimited_ocr"

    module, model_type = get_model_and_args(
        {
            "model_type": "unlimited-ocr",
            "architectures": ["UnlimitedOCRForCausalLM"],
            "vision_config": {},
            "text_config": {},
        }
    )
    assert module is unlimited_ocr
    assert model_type == "unlimited_ocr"

    message = get_message_json(
        "unlimited-ocr",
        "Multi page parsing.",
        num_images=3,
    )
    assert message == {"role": "user", "content": "<image>Multi page parsing."}


def test_z_image_tokenizer_disables_remote_code(image_apis, monkeypatch):
    import transformers
    from mlx_vlm.models.z_image import pipeline

    received = {}

    def fake_from_pretrained(*args, **kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )

    pipeline.AutoTokenizer.from_pretrained(
        "/tmp/tokenizer",
        local_files_only=False,
        trust_remote_code=True,
    )

    assert received["local_files_only"] is True
    assert received["trust_remote_code"] is False


def test_existing_image_families_provide_native_sampling_defaults(image_apis):
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
        assert isinstance(model_class.default_sampling, property)

    z_generation = object.__new__(ZImageGenerationModel)
    z_generation.pipeline = SimpleNamespace(
        config=SimpleNamespace(default_steps=9, default_guidance=0.0)
    )
    assert z_generation.default_sampling.steps == 9
    assert z_generation.default_sampling.guidance == 0.0

    z_edit = object.__new__(ZImageEditModel)
    z_edit.pipeline = SimpleNamespace(
        config=SimpleNamespace(variant="turbo", default_guidance=0.0)
    )
    assert z_edit.default_sampling.steps == 8
    assert z_edit.default_sampling.guidance == 0.0


@pytest.mark.parametrize("alias", ["z-image", "ernie-image-turbo"])
def test_generic_loader_dispatches_image_families_without_weights(
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
    monkeypatch.setattr(
        image, "_resolve_image_model_path", lambda *args, **kwargs: None
    )

    assert image.load_image_model(alias, task="generate") is generation_sentinel
    assert image.load_image_model(alias, task="edit") is edit_sentinel


def test_generic_api_preserves_native_requests_for_model_default_resolution(image_apis):
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
    assert generation_request.width == 512
    assert generation_request.height == 512
    assert generation_request.guidance is None
    generation_result = image.generate_image(GenerationModel(), generation_request)
    assert generation_result is generation_request
    assert generation_result.steps is None
    assert generation_result.guidance is None
    assert generation_result.width == 512
    assert generation_result.height == 512

    edit_request = edit_image.ImageEditRequest(
        prompt="make it blue",
        image_paths=("input.png",),
        seed=1,
    )
    assert edit_request.steps is None
    assert edit_request.guidance is None
    edit_result = edit_image.edit_image(EditModel(), edit_request)
    assert edit_result is edit_request
    assert edit_result.steps is None
    assert edit_result.guidance is None


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
def test_component_indexes_dispatch_to_image_models(
    image_apis, markers, expected_model_type, tmp_path
):
    image, _ = image_apis
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    (transformer / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": markers})
    )

    assert (
        image._image_model_type_from_component_indexes(tmp_path) == expected_model_type
    )


def test_ernie_legacy_mlx_checkpoint_layout_is_preserved(image_apis):
    from mlx_vlm.models.ernie_image.weights import _tensor_layout

    assert _tensor_layout({"mflux_version": "0.1"}) == "mlx_nhwc"


def test_ernie_edit_defaults_and_converted_source_metadata(image_apis, tmp_path):
    from mlx_vlm.models.ernie_image.config import get_variant, variant_from_local_path
    from mlx_vlm.models.ernie_image.model import ErnieImageEditModel

    turbo = get_variant("ernie-image-turbo")
    assert turbo.edit_default_guidance == 3.0

    edit = object.__new__(ErnieImageEditModel)
    edit.pipeline = SimpleNamespace(variant=turbo)
    assert edit.default_guidance == 3.0

    (tmp_path / "mlx_ernie_image.json").write_text(
        json.dumps({"source": "baidu/ERNIE-Image-Turbo", "variant": "ernie-image"})
    )
    assert variant_from_local_path(tmp_path) is turbo
