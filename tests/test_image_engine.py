# SPDX-License-Identifier: Apache-2.0
"""Tests for the mlx-vlm-backed image engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

import omlx.engine.image as image_module
from omlx.engine.image import ImageEngine, ImageEngineResult
from omlx.utils.optional_deps import MLX_VLM_MISSING_MESSAGE


class _FakeImageGenerationRequest:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _FakeImageEditRequest:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


@dataclass
class _FakeModel:
    reference: str
    task: str
    calls: list[tuple[str, Any]] = field(default_factory=list)


class _FakeImageAPI:
    ImageGenerationRequest = _FakeImageGenerationRequest
    ImageEditRequest = _FakeImageEditRequest

    def __init__(self) -> None:
        self.loads: list[tuple[str, str]] = []
        self.models: list[_FakeModel] = []

    def load_image_model(self, reference: str, *, task: str) -> _FakeModel:
        self.loads.append((reference, task))
        model = _FakeModel(reference=reference, task=task)
        self.models.append(model)
        return model

    def generate_image(
        self, model: _FakeModel, request: Any, *, task: str
    ) -> SimpleNamespace:
        model.calls.append((task, request))
        return SimpleNamespace(
            image={"task": task, "prompt": request.prompt},
            metadata={"revised_prompt": f"revised: {request.prompt}"},
            seed=request.seed,
            steps=request.steps,
            guidance=request.guidance,
            width=request.width,
            height=request.height,
            model=model.reference,
            family="fake",
            variant="fake-variant",
        )


@pytest.fixture
def fake_mlx_vlm(monkeypatch):
    api = _FakeImageAPI()
    cleanup_calls: list[str] = []
    monkeypatch.setattr(image_module, "_image_api", lambda: api)
    monkeypatch.setattr(
        image_module.mx, "synchronize", lambda: cleanup_calls.append("synchronize")
    )
    monkeypatch.setattr(
        image_module.mx, "clear_cache", lambda: cleanup_calls.append("clear_cache")
    )
    return SimpleNamespace(api=api, cleanup_calls=cleanup_calls)


async def test_generate_loads_local_manifest_path_with_request_defaults(
    fake_mlx_vlm, tmp_path
):
    model_root = tmp_path / "flux-model"
    model_root.mkdir()
    (model_root / "weights").mkdir()
    engine = ImageEngine(
        model_name="api-image-model",
        model_id="image-model",
        model_path=str(model_root),
        image_metadata={
            "backend": "mlx-vlm",
            "base_model": "flux2-klein-9b",
            "model_path": "weights",
            "default_steps": "7",
            "default_guidance": "1.5",
        },
        tasks=["generation"],
    )

    await engine.start()
    result = await engine.generate(
        "a cat",
        width=512,
        height=768,
        seed=42,
        output_format="jpeg",
        negative_prompt="low quality",
    )

    assert fake_mlx_vlm.api.loads == [
        (str(model_root / "weights"), "generate"),
    ]
    model = fake_mlx_vlm.api.models[0]
    task, request = model.calls[-1]
    assert task == "generate"
    assert request.prompt == "a cat"
    assert request.seed == 42
    assert request.width == 512
    assert request.height == 768
    assert request.steps == 7
    assert request.guidance == 1.5
    assert request.extra == {"negative_prompt": "low quality"}

    assert isinstance(result, ImageEngineResult)
    assert result.image == {"task": "generate", "prompt": "a cat"}
    assert result.metadata["output_format"] == "jpeg"
    assert result.metadata["steps"] == 7
    assert result.metadata["guidance"] == 1.5
    assert result.metadata["model_path"] == str(model_root / "weights")
    assert result.metadata["revised_prompt"] == "revised: a cat"

    await engine.stop()
    assert engine.get_stats()["loaded"] is False
    assert "synchronize" in fake_mlx_vlm.cleanup_calls
    assert "clear_cache" in fake_mlx_vlm.cleanup_calls


async def test_stop_can_defer_global_mlx_cleanup(fake_mlx_vlm):
    engine = ImageEngine(
        model_name="image-model",
        image_metadata={"backend": "mlx-vlm", "base_model": "flux2-klein-4b"},
        tasks=["generation"],
    )
    await engine.start()

    await engine.stop_without_global_cleanup()

    assert engine.get_stats()["loaded"] is False
    assert fake_mlx_vlm.cleanup_calls == []


@pytest.mark.parametrize(
    ("base_model", "expected_steps", "expected_guidance"),
    [
        ("flux2-klein-4b", 4, 1.0),
        ("mage-flow-base", 30, 5.0),
        ("mage-flow", 20, 5.0),
        ("mage-flow-turbo", 4, 1.0),
        ("z-image-turbo", 9, 0.0),
        ("ernie-image-turbo", 8, 1.0),
    ],
)
async def test_generation_uses_family_defaults(
    fake_mlx_vlm, base_model, expected_steps, expected_guidance
):
    engine = ImageEngine(
        model_name=base_model,
        image_metadata={"backend": "mlx-vlm", "base_model": base_model},
        tasks=["generation"],
    )

    await engine.start()
    result = await engine.generate("a cat")

    _, request = fake_mlx_vlm.api.models[-1].calls[-1]
    assert request.steps == expected_steps
    assert request.guidance == expected_guidance
    assert result.metadata["steps"] == expected_steps
    assert result.metadata["guidance"] == expected_guidance


async def test_request_defaults_override_manifest_and_family_defaults(fake_mlx_vlm):
    engine = ImageEngine(
        model_name="mage",
        image_metadata={
            "backend": "mlx-vlm",
            "base_model": "mage-flow",
            "default_steps": 7,
            "default_guidance": 2.0,
        },
        tasks=["generation"],
    )

    await engine.start()
    await engine.generate("a cat", steps=12, guidance=3.5)

    _, request = fake_mlx_vlm.api.models[-1].calls[-1]
    assert request.steps == 12
    assert request.guidance == 3.5


async def test_edit_routes_multiple_mage_inputs_and_strength_aliases(fake_mlx_vlm):
    engine = ImageEngine(
        model_name="mage-edit",
        image_metadata={
            "backend": "mlx-vlm",
            "base_model": "mage-flow-edit",
            "default_image_strength": 0.7,
        },
        tasks=["edit"],
    )

    await engine.start()
    result = await engine.edit(
        "make it cinematic",
        image_paths=["input-a.png", "input-b.png"],
        steps=9,
        guidance=2.5,
        seed=123,
        image_strength=0.55,
        scheduler="euler",
    )

    model = fake_mlx_vlm.api.models[-1]
    task, request = model.calls[-1]
    assert task == "edit"
    assert request.image_paths == ("input-a.png", "input-b.png")
    assert request.seed == 123
    assert request.steps == 9
    assert request.guidance == 2.5
    assert request.extra == {
        "scheduler": "euler",
        "image_strength": 0.55,
        "strength": 0.55,
    }
    assert result.metadata["task"] == "edit"
    assert result.metadata["input_image_count"] == 2


@pytest.mark.parametrize("base_model", ["z-image-turbo", "ernie-image-turbo"])
async def test_edit_uses_family_image_strength_default(fake_mlx_vlm, base_model):
    engine = ImageEngine(
        model_name=base_model,
        image_metadata={"backend": "mlx-vlm", "base_model": base_model},
        tasks=["edit"],
    )
    await engine.start()

    await engine.edit("make it brighter", image_paths=["input.png"])

    _, request = fake_mlx_vlm.api.models[-1].calls[-1]
    assert request.extra["image_strength"] == 0.6
    assert request.extra["strength"] == 0.6
    assert request.steps == 8


async def test_manifest_image_strength_overrides_family_default(fake_mlx_vlm):
    engine = ImageEngine(
        model_name="ernie-image",
        image_metadata={
            "backend": "mlx-vlm",
            "base_model": "ernie-image",
            "default_image_strength": 0.4,
        },
        tasks=["edit"],
    )
    await engine.start()

    await engine.edit("make it brighter", image_paths=["input.png"])

    _, request = fake_mlx_vlm.api.models[-1].calls[-1]
    assert request.extra["image_strength"] == 0.4
    assert request.extra["strength"] == 0.4


@pytest.mark.parametrize("base_model", ["z-image", "z-image-turbo", "ernie-image"])
async def test_single_source_edit_families_reject_multiple_images(
    fake_mlx_vlm, base_model
):
    engine = ImageEngine(
        model_name=base_model,
        image_metadata={"backend": "mlx-vlm", "base_model": base_model},
        tasks=["edit"],
    )
    await engine.start()

    with pytest.raises(ValueError, match="exactly one input image"):
        await engine.edit("combine", image_paths=["a.png", "b.png"])


async def test_all_current_image_families_reject_masks(fake_mlx_vlm):
    engine = ImageEngine(
        model_name="flux-edit",
        image_metadata={"backend": "mlx-vlm", "base_model": "flux2-klein-4b"},
        tasks=["edit"],
    )
    await engine.start()

    with pytest.raises(ValueError, match="do not support masks"):
        await engine.edit(
            "remove background",
            image_paths=["input.png"],
            mask_path="mask.png",
        )


async def test_dual_task_engine_keeps_one_loaded_variant(fake_mlx_vlm):
    engine = ImageEngine(
        model_name="dual-task",
        image_metadata={"backend": "mlx-vlm", "base_model": "flux2-klein-4b"},
        tasks=["generation", "edit"],
    )
    await engine.start()

    await engine.generate("a cat")
    assert engine.get_stats()["loaded_tasks"] == ["generation"]
    assert len(engine._models) == 1

    await engine.edit("make it brighter", image_paths=["input.png"])
    assert engine.get_stats()["loaded_tasks"] == ["edit"]
    assert len(engine._models) == 1
    assert len(fake_mlx_vlm.api.loads) == 2

    await engine.generate("a dog")
    assert engine.get_stats()["loaded_tasks"] == ["generation"]
    assert len(engine._models) == 1
    assert len(fake_mlx_vlm.api.loads) == 3


async def test_model_path_override_is_a_distinct_loaded_variant(
    fake_mlx_vlm, tmp_path
):
    model_root = tmp_path / "models"
    model_root.mkdir()
    override = tmp_path / "other-model"
    override.mkdir()
    engine = ImageEngine(
        model_name="flux",
        model_path=str(model_root),
        image_metadata={"backend": "mlx-vlm", "base_model": "flux2-klein-4b"},
        tasks=["generation"],
    )
    await engine.start()

    await engine.generate("first")
    await engine.generate("second", model_path=str(override))

    assert fake_mlx_vlm.api.loads == [
        (str(model_root), "generate"),
        (str(override), "generate"),
    ]
    assert len(engine._models) == 1


@pytest.mark.parametrize(
    ("base_model", "expected_reference"),
    [
        ("mage-flow-aligned", "mage-flow"),
        ("ideogram4", "ideogram-ai/ideogram-4-fp8"),
        ("z-image", "Tongyi-MAI/Z-Image"),
        ("z-image-turbo", "Tongyi-MAI/Z-Image-Turbo"),
    ],
)
async def test_direct_engine_uses_loader_reference_without_a_local_model_path(
    fake_mlx_vlm, base_model, expected_reference
):
    engine = ImageEngine(
        model_name="public-alias-engine",
        image_metadata={"backend": "mlx-vlm", "base_model": base_model},
        tasks=["generation"],
    )
    await engine.start()

    await engine.generate("a cat")

    assert fake_mlx_vlm.api.loads == [(expected_reference, "generate")]


async def test_manifest_model_path_must_resolve_locally(fake_mlx_vlm, tmp_path):
    engine = ImageEngine(
        model_name="flux",
        model_path=str(tmp_path),
        image_metadata={
            "backend": "mlx-vlm",
            "base_model": "flux2-klein-4b",
            "model_path": "missing-weights",
        },
        tasks=["generation"],
    )
    with pytest.raises(ValueError, match="manifest model_path does not exist"):
        await engine.start()
    assert fake_mlx_vlm.api.loads == []


async def test_quantize_is_rejected_in_manifest_and_request(fake_mlx_vlm):
    manifest_engine = ImageEngine(
        model_name="flux",
        image_metadata={
            "backend": "mlx-vlm",
            "base_model": "flux2-klein-4b",
            "quantize": 4,
        },
        tasks=["generation"],
    )
    with pytest.raises(ValueError, match="manifest quantize"):
        await manifest_engine.start()

    engine = ImageEngine(
        model_name="flux",
        image_metadata={"backend": "mlx-vlm", "base_model": "flux2-klein-4b"},
        tasks=["generation"],
    )
    await engine.start()
    with pytest.raises(ValueError, match="request quantize"):
        await engine.generate("a cat", quantize=4)


async def test_start_rejects_unsupported_backend_or_model(fake_mlx_vlm):
    bad_backend = ImageEngine(
        model_name="unsupported",
        image_metadata={"backend": "other", "base_model": "flux2-klein-4b"},
        tasks=["generation"],
    )
    with pytest.raises(ValueError, match="Unsupported image backend"):
        await bad_backend.start()

    bad_model = ImageEngine(
        model_name="unsupported",
        image_metadata={"backend": "mlx-vlm", "base_model": "schnell"},
        tasks=["generation"],
    )
    with pytest.raises(ValueError, match="Unsupported mlx-vlm image base_model"):
        await bad_model.start()


@pytest.mark.parametrize(
    ("base_model", "task"),
    [
        ("FLUX.2-klein-base-4B", "generation"),
        ("flux2-klein-9b-kv", "edit"),
        ("Mage_Flow_Aligned", "generation"),
        ("Mage_Flow_Edit_Turbo", "edit"),
        ("Z_Image_Turbo", "generation"),
        ("ERNIE_Image_Turbo", "edit"),
        ("Ideogram4", "generation"),
        ("bonsai-ternary", "generation"),
    ],
)
async def test_start_accepts_supported_aliases(fake_mlx_vlm, base_model, task):
    engine = ImageEngine(
        model_name="image-alias",
        image_metadata={"backend": "mlx-vlm", "base_model": base_model},
        tasks=[task],
    )
    await engine.start()
    assert engine.get_stats()["loaded_tasks"] == [task]


async def test_missing_mlx_vlm_produces_core_dependency_error(monkeypatch):
    def unavailable_api() -> Any:
        raise ImportError(MLX_VLM_MISSING_MESSAGE)

    monkeypatch.setattr(image_module, "_image_api", unavailable_api)
    monkeypatch.setattr(image_module.mx, "synchronize", lambda: None)
    monkeypatch.setattr(image_module.mx, "clear_cache", lambda: None)
    engine = ImageEngine(
        model_name="missing-runtime",
        image_metadata={"backend": "mlx-vlm", "base_model": "flux2-klein-4b"},
        tasks=["generation"],
    )
    with pytest.raises(ImportError, match="mlx-vlm image support is unavailable"):
        await engine.start()
