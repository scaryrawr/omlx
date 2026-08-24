# SPDX-License-Identifier: Apache-2.0
"""Tests for the mlx-vlm-backed image engine."""

from __future__ import annotations

import concurrent.futures
import threading
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
        self.thread_ids: list[int] = []

    def load_image_model(self, reference: str, *, task: str) -> _FakeModel:
        self.thread_ids.append(threading.get_ident())
        self.loads.append((reference, task))
        model = _FakeModel(reference=reference, task=task)
        self.models.append(model)
        return model

    def generate_image(
        self, model: _FakeModel, request: Any, *, task: str
    ) -> SimpleNamespace:
        self.thread_ids.append(threading.get_ident())
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


async def test_generate_uses_manifest_path_and_defaults(fake_mlx_vlm, tmp_path):
    model_root = tmp_path / "flux-model"
    weights = model_root / "weights"
    weights.mkdir(parents=True)
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

    assert fake_mlx_vlm.api.loads == [(str(weights), "generate")]
    _, request = fake_mlx_vlm.api.models[0].calls[-1]
    assert request.seed == 42
    assert request.width == 512
    assert request.height == 768
    assert request.steps == 7
    assert request.guidance == 1.5
    assert request.extra == {"negative_prompt": "low quality"}
    assert isinstance(result, ImageEngineResult)
    assert result.metadata["output_format"] == "jpeg"
    assert result.metadata["revised_prompt"] == "revised: a cat"


async def test_edit_uses_task_defaults_and_multiple_mage_inputs(fake_mlx_vlm):
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
        image_strength=0.55,
        scheduler="euler",
    )

    assert fake_mlx_vlm.api.loads == [("mage-flow-edit", "edit")]
    task, request = fake_mlx_vlm.api.models[-1].calls[-1]
    assert task == "edit"
    assert request.steps == 30
    assert request.guidance == 5.0
    assert request.extra == {
        "scheduler": "euler",
        "image_strength": 0.55,
        "strength": 0.55,
    }
    assert result.metadata["input_image_count"] == 2


async def test_ernie_turbo_edit_uses_native_task_guidance(fake_mlx_vlm):
    engine = ImageEngine(
        model_name="ernie-image-turbo",
        image_metadata={"backend": "mlx-vlm", "base_model": "ernie-image-turbo"},
        tasks=["edit"],
    )

    await engine.start()
    await engine.edit("make it brighter", image_paths=["input.png"])

    _, request = fake_mlx_vlm.api.models[-1].calls[-1]
    assert request.width == 512
    assert request.height == 512
    assert request.guidance == 3.0


async def test_dual_task_engine_keeps_only_current_variant(fake_mlx_vlm):
    engine = ImageEngine(
        model_name="dual-task",
        image_metadata={"backend": "mlx-vlm", "base_model": "flux2-klein-4b"},
        tasks=["generation", "edit"],
    )
    await engine.start()

    await engine.generate("a cat")
    assert engine.get_stats()["loaded_tasks"] == ["generation"]
    _, first_request = fake_mlx_vlm.api.models[0].calls[-1]
    assert first_request.width == 512
    assert first_request.height == 512
    await engine.edit("make it brighter", image_paths=["input.png"])
    assert engine.get_stats()["loaded_tasks"] == ["edit"]
    assert len(engine._models) == 1
    await engine.generate("a dog")
    assert fake_mlx_vlm.api.loads == [
        ("flux2-klein-4b", "generate"),
        ("flux2-klein-4b", "edit"),
        ("flux2-klein-4b", "generate"),
    ]


async def test_calls_run_on_mlx_executor(fake_mlx_vlm, monkeypatch):
    main_thread = threading.get_ident()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(image_module, "get_mlx_executor", lambda: executor)
        engine = ImageEngine(
            model_name="flux",
            image_metadata={"backend": "mlx-vlm", "base_model": "flux2-klein-4b"},
            tasks=["generation"],
        )
        await engine.start()
        await engine.generate("a cat")

    assert fake_mlx_vlm.api.thread_ids
    assert all(thread_id != main_thread for thread_id in fake_mlx_vlm.api.thread_ids)


async def test_stop_clears_variant_and_mlx_cache(fake_mlx_vlm):
    engine = ImageEngine(
        model_name="flux",
        image_metadata={"backend": "mlx-vlm", "base_model": "flux2-klein-4b"},
        tasks=["generation"],
    )
    await engine.start()

    await engine.stop()

    assert engine.get_stats()["loaded"] is False
    assert engine.get_stats()["loaded_tasks"] == []
    assert "synchronize" in fake_mlx_vlm.cleanup_calls
    assert "clear_cache" in fake_mlx_vlm.cleanup_calls


async def test_stop_can_defer_global_cleanup(fake_mlx_vlm):
    engine = ImageEngine(
        model_name="flux",
        image_metadata={"backend": "mlx-vlm", "base_model": "flux2-klein-4b"},
        tasks=["generation"],
    )
    await engine.start()
    fake_mlx_vlm.cleanup_calls.clear()

    await engine.stop_without_global_cleanup()

    assert engine.get_stats()["loaded"] is False
    assert fake_mlx_vlm.cleanup_calls == []


async def test_validation_rejects_unsupported_inputs(fake_mlx_vlm):
    engine = ImageEngine(
        model_name="ernie",
        image_metadata={"backend": "mlx-vlm", "base_model": "ernie-image"},
        tasks=["edit"],
    )
    await engine.start()

    with pytest.raises(ValueError, match="exactly one input image"):
        await engine.edit("combine", image_paths=["a.png", "b.png"])
    with pytest.raises(ValueError, match="do not support masks"):
        await engine.edit("remove", image_paths=["a.png"], mask_path="mask.png")
    with pytest.raises(ValueError, match="request quantize"):
        await engine.edit("change", image_paths=["a.png"], quantize=4)


async def test_unset_seed_is_delegated_to_native_randomization(fake_mlx_vlm):
    engine = ImageEngine(
        model_name="flux",
        image_metadata={"backend": "mlx-vlm", "base_model": "flux2-klein-4b"},
        tasks=["generation"],
    )
    await engine.start()

    await engine.generate("a cat")

    _, request = fake_mlx_vlm.api.models[-1].calls[-1]
    assert request.seed is None


async def test_z_image_requires_converted_local_weights(fake_mlx_vlm):
    engine = ImageEngine(
        model_name="z-image",
        image_metadata={"backend": "mlx-vlm", "base_model": "z-image"},
        tasks=["generation"],
    )

    with pytest.raises(ValueError, match="requires a converted local MLX model path"):
        await engine.start()
    assert fake_mlx_vlm.api.loads == []


async def test_missing_mlx_vlm_produces_core_dependency_error(monkeypatch):
    def unavailable_api() -> Any:
        raise ImportError(MLX_VLM_MISSING_MESSAGE)

    monkeypatch.setattr(image_module, "_image_api", unavailable_api)
    engine = ImageEngine(
        model_name="missing-runtime",
        image_metadata={"backend": "mlx-vlm", "base_model": "flux2-klein-4b"},
        tasks=["generation"],
    )

    with pytest.raises(ImportError, match="mlx-vlm image support is unavailable"):
        await engine.start()
