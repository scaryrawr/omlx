# SPDX-License-Identifier: Apache-2.0
"""Tests for the mflux-backed image engine."""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

import omlx.engine.image as image_module
from omlx.engine.image import ImageEngine, ImageEngineResult


@dataclass
class _FakeConfig:
    name: str

    @property
    def model_name(self) -> str:
        return self.name


class _FakeModelConfig:
    @staticmethod
    def from_name(model_name: str) -> _FakeConfig:
        return _FakeConfig(model_name)


class _FakeGeneratedImage:
    def __init__(self, image: Any, model_config: _FakeConfig, **metadata: Any) -> None:
        self.image = image
        self.model_config = model_config
        for key, value in metadata.items():
            setattr(self, key, value)


def _make_fake_model_class(class_name: str):
    class _FakeModel:
        instances: list[_FakeModel] = []

        def __init__(
            self,
            *,
            model_config: _FakeConfig | None = None,
            quantize: int | None = None,
            model_path: str | None = None,
            **kwargs: Any,
        ) -> None:
            self.model_config = model_config
            self.quantize = quantize
            self.model_path = model_path
            self.init_kwargs = kwargs
            self.calls: list[dict[str, Any]] = []
            type(self).instances.append(self)

        def generate_image(self, **kwargs: Any) -> _FakeGeneratedImage:
            self.calls.append(dict(kwargs))
            return _FakeGeneratedImage(
                image={"class": class_name, "prompt": kwargs.get("prompt")},
                model_config=self.model_config or _FakeConfig("missing"),
                seed=kwargs.get("seed"),
                prompt=kwargs.get("prompt"),
                width=kwargs.get("width"),
                height=kwargs.get("height"),
                guidance=kwargs.get("guidance"),
                steps=kwargs.get("num_inference_steps"),
                generation_time=0.1,
            )

    _FakeModel.__name__ = class_name
    return _FakeModel


@pytest.fixture
def fake_mflux(monkeypatch):
    classes: dict[str, type] = {}
    cleanup_calls: list[str] = []

    def ensure_package(name: str) -> types.ModuleType:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = []  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, name, module)
        return module

    def install_module(module_name: str, class_name: str | None = None) -> types.ModuleType:
        parts = module_name.split(".")
        for index in range(1, len(parts)):
            ensure_package(".".join(parts[:index]))
        module = types.ModuleType(module_name)
        if class_name is not None:
            cls = _make_fake_model_class(class_name)
            setattr(module, class_name, cls)
            classes[class_name] = cls
        monkeypatch.setitem(sys.modules, module_name, module)
        return module

    config_module = install_module("mflux.models.common.config.model_config")
    config_module.ModelConfig = _FakeModelConfig

    install_module("mflux.models.flux2.variants.txt2img.flux2_klein", "Flux2Klein")
    install_module("mflux.models.flux2.variants.edit.flux2_klein_edit", "Flux2KleinEdit")
    install_module("mflux.models.z_image.variants.z_image", "ZImage")
    install_module("mflux.models.qwen.variants.txt2img.qwen_image", "QwenImage")
    install_module("mflux.models.qwen.variants.edit.qwen_image_edit", "QwenImageEdit")
    install_module("mflux.models.fibo.variants.txt2img.fibo", "FIBO")
    install_module("mflux.models.fibo.variants.edit.fibo_edit", "FIBOEdit")
    install_module("mflux.models.ernie_image.variants.txt2img.ernie_image", "ErnieImage")
    install_module("mflux.models.ideogram4.variants.txt2img.ideogram4", "Ideogram4")

    monkeypatch.setattr(
        image_module.mx, "synchronize", lambda: cleanup_calls.append("synchronize")
    )
    monkeypatch.setattr(
        image_module.mx, "clear_cache", lambda: cleanup_calls.append("clear_cache")
    )

    return types.SimpleNamespace(classes=classes, cleanup_calls=cleanup_calls)


async def test_generate_loads_flux2_with_manifest_defaults(fake_mflux, tmp_path):
    model_root = tmp_path / "flux-model"
    model_root.mkdir()

    engine = ImageEngine(
        model_name="api-image-model",
        model_id="image-model",
        model_path=str(model_root),
        image_metadata={
            "backend": "mflux",
            "base_model": "flux2-klein-9b",
            "model_path": "weights",
            "quantize": "8",
            "default_steps": "7",
            "default_guidance": "1.5",
        },
        tasks=["generation"],
    )

    await engine.start()

    flux_cls = fake_mflux.classes["Flux2Klein"]
    assert flux_cls.instances == []

    result = await engine.generate(
        "a cat",
        width=512,
        height=768,
        seed=42,
        output_format="jpeg",
        negative_prompt="low quality",
    )

    assert len(flux_cls.instances) == 1
    model = flux_cls.instances[0]
    assert model.model_config == _FakeConfig("flux2-klein-9b")
    assert model.quantize == 8
    assert model.model_path == str(model_root / "weights")

    assert isinstance(result, ImageEngineResult)
    assert result.image == {"class": "Flux2Klein", "prompt": "a cat"}
    assert model.calls[-1] == {
        "prompt": "a cat",
        "seed": 42,
        "width": 512,
        "height": 768,
        "num_inference_steps": 7,
        "guidance": 1.5,
        "negative_prompt": "low quality",
    }
    assert result.metadata["output_format"] == "jpeg"
    assert result.metadata["steps"] == 7
    assert result.metadata["guidance"] == 1.5
    assert result.metadata["mflux_model_name"] == "flux2-klein-9b"

    await engine.stop()
    assert engine.get_stats()["loaded"] is False
    assert "synchronize" in fake_mflux.cleanup_calls
    assert "clear_cache" in fake_mflux.cleanup_calls


async def test_edit_uses_qwen_image_paths_and_request_overrides(fake_mflux):
    engine = ImageEngine(
        model_name="qwen-edit",
        image_metadata={
            "backend": "mflux",
            "base_model": "qwen-image-edit",
            "default_steps": 4,
            "default_guidance": 4.0,
        },
        tasks=["edit"],
    )

    await engine.start()
    result = await engine.edit(
        "make it brighter",
        image_paths=["input-a.png", "input-b.png"],
        steps=9,
        guidance=2.5,
        seed=123,
    )

    qwen_cls = fake_mflux.classes["QwenImageEdit"]
    model = qwen_cls.instances[0]
    assert model.model_config == _FakeConfig("qwen-image-edit")
    assert model.calls[-1] == {
        "prompt": "make it brighter",
        "seed": 123,
        "num_inference_steps": 9,
        "guidance": 2.5,
        "image_paths": ["input-a.png", "input-b.png"],
    }
    assert result.metadata["task"] == "edit"
    assert result.metadata["input_image_count"] == 2

    with pytest.raises(ValueError, match="does not support mask_path"):
        await engine.edit("mask this", image_paths=["input.png"], mask_path="mask.png")


async def test_fibo_edit_routes_single_image_and_mask(fake_mflux):
    engine = ImageEngine(
        model_name="fibo-edit",
        image_metadata={"backend": "mflux", "base_model": "fibo-edit"},
        tasks=["edit"],
    )

    await engine.start()
    await engine.edit(
        "remove background",
        image_paths=["input.png"],
        mask_path="mask.png",
        width=1024,
        height=1024,
    )

    fibo_cls = fake_mflux.classes["FIBOEdit"]
    assert fibo_cls.instances[0].calls[-1] == {
        "prompt": "remove background",
        "seed": 0,
        "width": 1024,
        "height": 1024,
        "image_path": "input.png",
        "mask_path": "mask.png",
    }

    with pytest.raises(ValueError, match="exactly one input image"):
        await engine.edit("remove background", image_paths=["a.png", "b.png"])


async def test_ernie_edit_routes_single_image_to_image_path(fake_mflux):
    engine = ImageEngine(
        model_name="ernie-edit",
        image_metadata={"backend": "mflux", "base_model": "ernie-image-turbo"},
        tasks=["edit"],
    )

    await engine.start()
    await engine.edit(
        "make it watercolor",
        image_paths=["input.png"],
        image_strength=0.6,
        width=1024,
        height=576,
    )

    ernie_cls = fake_mflux.classes["ErnieImage"]
    model = ernie_cls.instances[0]
    assert model.model_config == _FakeConfig("ernie-image-turbo")
    assert model.calls[-1] == {
        "prompt": "make it watercolor",
        "seed": 0,
        "width": 1024,
        "height": 576,
        "num_inference_steps": 8,
        "guidance": 1.0,
        "image_path": "input.png",
        "image_strength": 0.6,
    }

    with pytest.raises(ValueError, match="does not support mask_path"):
        await engine.edit("mask this", image_paths=["input.png"], mask_path="mask.png")

    with pytest.raises(ValueError, match="exactly one input image"):
        await engine.edit("combine these", image_paths=["a.png", "b.png"])


async def test_ideogram_generation_uses_fp8_config(fake_mflux):
    engine = ImageEngine(
        model_name="ideogram",
        image_metadata={"backend": "mflux", "base_model": "ideogram4"},
        tasks=["generation"],
    )

    await engine.start()
    await engine.generate("a typographic poster", seed=99, width=1024, height=1024)

    ideogram_cls = fake_mflux.classes["Ideogram4"]
    model = ideogram_cls.instances[0]
    assert model.model_config == _FakeConfig("ideogram-4-fp8")
    assert model.calls[-1] == {
        "prompt": "a typographic poster",
        "seed": 99,
        "width": 1024,
        "height": 1024,
    }


async def test_ideogram_edit_is_not_supported(fake_mflux):
    engine = ImageEngine(
        model_name="ideogram",
        image_metadata={"backend": "mflux", "base_model": "ideogram4"},
        tasks=["edit"],
    )

    with pytest.raises(ValueError, match="Unsupported mflux image base_model"):
        await engine.start()


async def test_start_rejects_unsupported_base_model(fake_mflux):
    engine = ImageEngine(
        model_name="unsupported",
        image_metadata={"backend": "mflux", "base_model": "schnell"},
        tasks=["generation"],
    )

    with pytest.raises(ValueError, match="Unsupported mflux image base_model"):
        await engine.start()

    assert fake_mflux.classes["Flux2Klein"].instances == []


@pytest.mark.parametrize(
    ("base_model", "class_name"),
    [
        ("FLUX.2-klein-4B", "Flux2Klein"),
        ("Z_Image_Turbo", "ZImage"),
        ("Ernie_Image_Turbo", "ErnieImage"),
        ("Ideogram4", "Ideogram4"),
    ],
)
async def test_start_accepts_lmstudio_style_base_aliases(
    fake_mflux, base_model, class_name
):
    engine = ImageEngine(
        model_name="lmstudio-image",
        image_metadata={"backend": "mflux", "base_model": base_model},
        tasks=["generation"],
    )

    await engine.start()

    assert fake_mflux.classes[class_name].instances == []


async def test_generate_missing_mflux_raises_import_error_with_install_hint(monkeypatch):
    real_import_module = image_module.importlib.import_module

    def missing_mflux_import(name: str):
        if name.startswith("mflux."):
            raise ImportError("No module named 'mflux'")
        return real_import_module(name)

    monkeypatch.setattr(image_module.importlib, "import_module", missing_mflux_import)

    engine = ImageEngine(
        model_name="missing-mflux",
        image_metadata={"backend": "mflux", "base_model": "flux2-klein-4b"},
        tasks=["generation"],
    )
    await engine.start()

    with pytest.raises(ImportError) as exc_info:
        await engine.generate("a cat")

    message = str(exc_info.value)
    assert "mflux is required for image inference" in message
    assert "pip install 'omlx[image]'" in message


async def test_dual_task_engine_retains_one_loaded_model(fake_mflux):
    engine = ImageEngine(
        model_name="dual-task",
        image_metadata={"backend": "mflux", "base_model": "flux2-klein-4b"},
        tasks=["generation", "edit"],
    )

    await engine.start()
    assert engine.get_stats()["loaded"] is True
    assert engine.get_stats()["loaded_tasks"] == []

    await engine.generate("a cat")
    assert engine.get_stats()["loaded_tasks"] == ["generation"]
    assert len(engine._models) == 1

    await engine.edit("make it brighter", image_paths=["input.png"])
    assert engine.get_stats()["loaded_tasks"] == ["edit"]
    assert len(engine._models) == 1
    assert "synchronize" in fake_mflux.cleanup_calls
    assert "clear_cache" in fake_mflux.cleanup_calls

    await engine.generate("a dog")
    assert engine.get_stats()["loaded_tasks"] == ["generation"]
    assert len(engine._models) == 1


async def test_lazy_load_counts_as_active_request(monkeypatch):
    load_started = asyncio.Event()
    release_load = asyncio.Event()

    class FakeModel:
        def generate_image(self, **kwargs):
            return _FakeGeneratedImage(
                image={"prompt": kwargs.get("prompt")},
                model_config=_FakeConfig("flux2-klein-4b"),
            )

    async def fake_load_model_for_task(self, *args, **kwargs):
        load_started.set()
        await release_load.wait()
        return FakeModel()

    monkeypatch.setattr(
        ImageEngine,
        "_load_model_for_task",
        fake_load_model_for_task,
    )
    monkeypatch.setattr(image_module.mx, "synchronize", lambda: None)
    monkeypatch.setattr(image_module.mx, "clear_cache", lambda: None)

    engine = ImageEngine(
        model_name="active-load",
        image_metadata={"backend": "mflux", "base_model": "flux2-klein-4b"},
        tasks=["generation"],
    )
    await engine.start()

    task = asyncio.create_task(engine.generate("a cat"))
    await load_started.wait()
    assert engine.has_active_requests() is True

    release_load.set()
    await task
    assert engine.has_active_requests() is False


def test_task_normalization_accepts_tuple_tasks():
    engine = ImageEngine(
        model_name="tuple-tasks",
        image_metadata={"backend": "mflux", "base_model": "flux2-klein-4b"},
        tasks=("txt2img", "edit"),
    )

    assert engine.get_stats()["tasks"] == ["generation", "edit"]


def test_task_normalization_rejects_invalid_task():
    with pytest.raises(ValueError, match="Unsupported image model task"):
        ImageEngine(
            model_name="bad-task",
            image_metadata={"backend": "mflux", "base_model": "flux2-klein-4b"},
            tasks=["upscale"],
        )
