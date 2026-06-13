# SPDX-License-Identifier: Apache-2.0
"""
MFLUX-backed image engine for oMLX.

This module intentionally keeps image inference non-streaming. Routes prepare
request inputs (including temporary edit image paths) and call ``generate`` or
``edit``; the engine owns only model loading, mflux invocation, and cleanup.
"""

from __future__ import annotations

import asyncio
import gc
import importlib
import inspect
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx

from ..engine_core import get_mlx_executor
from ..image_registry import (
    get_image_defaults,
    image_engine_aliases,
    normalize_image_alias,
)
from ..utils.optional_deps import MFLUX_MISSING_MESSAGE
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

ImageTask = Literal["generation", "edit"]
EditInputStyle = Literal["image_paths", "image_path", "image_path_mask"]


def _cleanup_mlx_cache() -> None:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def _clear_mlx_cache() -> None:
    mx.synchronize()
    mx.clear_cache()


@dataclass(frozen=True)
class _MfluxModelSpec:
    task: ImageTask
    module: str
    class_name: str
    config_name: str
    edit_input_style: EditInputStyle | None = None


@dataclass(frozen=True)
class _ModelKey:
    task: ImageTask
    quantize: int | None
    model_path: str | None


@dataclass(slots=True)
class ImageEngineResult:
    """Result returned by image engine calls.

    ``image`` is the generated PIL Image object returned by mflux (or extracted
    from mflux's ``GeneratedImage`` wrapper). ``metadata`` is intentionally
    small and route-friendly; API adapters can serialize/extend it as needed.
    """

    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_base_model(value: object) -> str:
    return normalize_image_alias(value)


def _coerce_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Image manifest {field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Image manifest {field_name} must be an integer")


def _coerce_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Image manifest {field_name} must be a number")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    raise ValueError(f"Image manifest {field_name} must be a number")


def _build_alias_map() -> dict[tuple[ImageTask, str], _MfluxModelSpec]:
    specs: dict[tuple[ImageTask, str], _MfluxModelSpec] = {}

    def add(spec: _MfluxModelSpec, *aliases: str) -> None:
        for alias in aliases:
            specs[(spec.task, _normalize_base_model(alias))] = spec

    flux2_4b = _MfluxModelSpec(
        task="generation",
        module="mflux.models.flux2.variants.txt2img.flux2_klein",
        class_name="Flux2Klein",
        config_name="flux2-klein-4b",
    )
    add(flux2_4b, *image_engine_aliases("generation", "flux2-klein-4b"))

    flux2_9b = _MfluxModelSpec(
        task="generation",
        module="mflux.models.flux2.variants.txt2img.flux2_klein",
        class_name="Flux2Klein",
        config_name="flux2-klein-9b",
    )
    add(flux2_9b, *image_engine_aliases("generation", "flux2-klein-9b"))

    z_image = _MfluxModelSpec(
        task="generation",
        module="mflux.models.z_image.variants.z_image",
        class_name="ZImage",
        config_name="z-image",
    )
    add(z_image, *image_engine_aliases("generation", "z-image"))

    z_image_turbo = _MfluxModelSpec(
        task="generation",
        module="mflux.models.z_image.variants.z_image",
        class_name="ZImage",
        config_name="z-image-turbo",
    )
    add(z_image_turbo, *image_engine_aliases("generation", "z-image-turbo"))

    qwen_image = _MfluxModelSpec(
        task="generation",
        module="mflux.models.qwen.variants.txt2img.qwen_image",
        class_name="QwenImage",
        config_name="qwen-image",
    )
    add(qwen_image, *image_engine_aliases("generation", "qwen-image"))

    fibo = _MfluxModelSpec(
        task="generation",
        module="mflux.models.fibo.variants.txt2img.fibo",
        class_name="FIBO",
        config_name="fibo",
    )
    add(fibo, *image_engine_aliases("generation", "fibo"))

    ernie_image_turbo = _MfluxModelSpec(
        task="generation",
        module="mflux.models.ernie_image.variants.txt2img.ernie_image",
        class_name="ErnieImage",
        config_name="ernie-image-turbo",
    )
    add(ernie_image_turbo, *image_engine_aliases("generation", "ernie-image-turbo"))

    ernie_image = _MfluxModelSpec(
        task="generation",
        module="mflux.models.ernie_image.variants.txt2img.ernie_image",
        class_name="ErnieImage",
        config_name="ernie-image",
    )
    add(ernie_image, *image_engine_aliases("generation", "ernie-image"))

    ideogram4 = _MfluxModelSpec(
        task="generation",
        module="mflux.models.ideogram4.variants.txt2img.ideogram4",
        class_name="Ideogram4",
        config_name="ideogram-4-fp8",
    )
    add(ideogram4, *image_engine_aliases("generation", "ideogram-4-fp8"))

    flux2_4b_edit = _MfluxModelSpec(
        task="edit",
        module="mflux.models.flux2.variants.edit.flux2_klein_edit",
        class_name="Flux2KleinEdit",
        config_name="flux2-klein-4b",
        edit_input_style="image_paths",
    )
    add(flux2_4b_edit, *image_engine_aliases("edit", "flux2-klein-4b"))

    flux2_9b_edit = _MfluxModelSpec(
        task="edit",
        module="mflux.models.flux2.variants.edit.flux2_klein_edit",
        class_name="Flux2KleinEdit",
        config_name="flux2-klein-9b",
        edit_input_style="image_paths",
    )
    add(flux2_9b_edit, *image_engine_aliases("edit", "flux2-klein-9b"))

    qwen_image_edit = _MfluxModelSpec(
        task="edit",
        module="mflux.models.qwen.variants.edit.qwen_image_edit",
        class_name="QwenImageEdit",
        config_name="qwen-image-edit",
        edit_input_style="image_paths",
    )
    add(qwen_image_edit, *image_engine_aliases("edit", "qwen-image-edit"))

    fibo_edit = _MfluxModelSpec(
        task="edit",
        module="mflux.models.fibo.variants.edit.fibo_edit",
        class_name="FIBOEdit",
        config_name="fibo-edit",
        edit_input_style="image_path_mask",
    )
    add(fibo_edit, *image_engine_aliases("edit", "fibo-edit"))

    ernie_image_turbo_edit = _MfluxModelSpec(
        task="edit",
        module="mflux.models.ernie_image.variants.txt2img.ernie_image",
        class_name="ErnieImage",
        config_name="ernie-image-turbo",
        edit_input_style="image_path",
    )
    add(ernie_image_turbo_edit, *image_engine_aliases("edit", "ernie-image-turbo"))

    ernie_image_edit = _MfluxModelSpec(
        task="edit",
        module="mflux.models.ernie_image.variants.txt2img.ernie_image",
        class_name="ErnieImage",
        config_name="ernie-image",
        edit_input_style="image_path",
    )
    add(ernie_image_edit, *image_engine_aliases("edit", "ernie-image"))

    return specs


_MFLUX_SPECS = _build_alias_map()


class ImageEngine(BaseNonStreamingEngine):
    """Non-streaming mflux-backed image generation/editing engine."""

    def __init__(
        self,
        model_name: str,
        *,
        model_id: str | None = None,
        model_path: str | None = None,
        config_model_type: str = "",
        image_metadata: dict[str, object] | None = None,
        capabilities: list[str] | None = None,
        tasks: list[str] | None = None,
        model_settings: object | None = None,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._model_id = model_id or model_name
        self._model_path = model_path or model_name
        self._config_model_type = config_model_type
        self._image_metadata = dict(image_metadata or {})
        self._capabilities = list(capabilities or [])
        self._tasks = self._normalize_tasks(tasks or self._image_metadata.get("tasks"))
        self._model_settings = model_settings
        self._models: dict[_ModelKey, Any] = {}
        self._started = False
        self._call_lock = asyncio.Lock()
        self._load_lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        """Get the API/model name associated with this engine."""
        return self._model_name

    @property
    def base_model(self) -> str:
        """Return the mflux base model alias from the image manifest."""
        return str(self._image_metadata.get("base_model", "")).strip()

    async def start(self) -> None:
        """Validate image model configuration and reserve the engine slot."""
        if self._started:
            return

        backend = str(self._image_metadata.get("backend", "mflux")).strip().lower()
        if backend != "mflux":
            raise ValueError(f"Unsupported image backend for {self._model_id}: {backend!r}")

        if not self._tasks:
            raise ValueError(f"Image model {self._model_id} declares no supported tasks")
        for task in self._tasks:
            self._resolve_spec(task)

        logger.info(
            "Starting image engine: model=%s, base_model=%s, tasks=%s",
            self._model_id,
            self.base_model,
            ",".join(self._tasks),
        )
        self._started = True
        logger.info("Image engine started: %s", self._model_id)

    async def stop(self) -> None:
        """Drop loaded mflux models and clear MLX resources."""
        if not self._models and not self._started:
            return

        async with self._call_lock:
            logger.info("Stopping image engine: %s", self._model_id)
            self._models.clear()
            self._started = False

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                get_mlx_executor(),
                _cleanup_mlx_cache,
            )
        logger.info("Image engine stopped: %s", self._model_id)

    async def generate(
        self,
        prompt: str,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        output_format: str = "png",
        **kwargs: Any,
    ) -> ImageEngineResult:
        """Generate one image from a text prompt.

        Args:
            prompt: Text prompt.
            width: Optional output width; mflux defaults are used when omitted.
            height: Optional output height; mflux defaults are used when omitted.
            seed: Optional seed; defaults to 0 for deterministic route behavior.
            steps: Optional request override for manifest ``default_steps``.
            guidance: Optional request override for manifest ``default_guidance``.
            output_format: Route-facing desired encoding format; no file is written here.
            **kwargs: Extra mflux ``generate_image`` parameters such as
                ``negative_prompt``, ``scheduler``, ``image_path``, or
                ``image_strength``. ``quantize`` and ``model_path`` can be used
                to load a separate in-memory variant for this request.
        """
        self._ensure_started()
        self._ensure_task_supported("generation")

        activity_id = self._begin_activity(
            "generating image",
            detail="Generating image",
            metadata={"prompt_length": len(prompt), "width": width, "height": height},
        )
        try:
            async with self._call_lock:
                quantize_override = kwargs.pop("quantize", None)
                model_path_override = kwargs.pop("model_path", None)
                key = self._model_key("generation", quantize_override, model_path_override)
                model = await self._load_model_for_task(
                    "generation",
                    quantize_override=quantize_override,
                    model_path_override=model_path_override,
                )

                resolved_steps = self._resolve_steps(steps)
                resolved_guidance = self._resolve_guidance(guidance)
                resolved_seed = 0 if seed is None else int(seed)

                gen_kwargs: dict[str, Any] = {
                    "prompt": prompt,
                    "seed": resolved_seed,
                }
                if width is not None:
                    gen_kwargs["width"] = int(width)
                if height is not None:
                    gen_kwargs["height"] = int(height)
                if resolved_steps is not None:
                    gen_kwargs["num_inference_steps"] = resolved_steps
                if resolved_guidance is not None:
                    gen_kwargs["guidance"] = resolved_guidance
                gen_kwargs.update(kwargs)

                return await self._run_image_call(
                    model=model,
                    task="generation",
                    call_kwargs=gen_kwargs,
                    result_metadata={
                        "output_format": output_format,
                        "quantize": key.quantize,
                        "model_path": key.model_path,
                    },
                )
        finally:
            await self._finish_activity(activity_id)

    async def edit(
        self,
        prompt: str,
        image_paths: list[str],
        mask_path: str | None = None,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        output_format: str = "png",
        **kwargs: Any,
    ) -> ImageEngineResult:
        """Edit one or more input images using an mflux edit model.

        Routes should parse/upload API image inputs and pass local file paths.
        This method does not decode API payloads or write output files.
        """
        self._ensure_started()
        self._ensure_task_supported("edit")
        if not image_paths:
            raise ValueError("edit requires at least one input image path")

        activity_id = self._begin_activity(
            "editing image",
            detail="Editing image",
            metadata={"prompt_length": len(prompt), "width": width, "height": height},
        )
        try:
            async with self._call_lock:
                quantize_override = kwargs.pop("quantize", None)
                model_path_override = kwargs.pop("model_path", None)
                key = self._model_key("edit", quantize_override, model_path_override)
                model = await self._load_model_for_task(
                    "edit",
                    quantize_override=quantize_override,
                    model_path_override=model_path_override,
                )

                spec = self._resolve_spec("edit")
                resolved_steps = self._resolve_steps(steps)
                resolved_guidance = self._resolve_guidance(guidance)
                request_image_strength = kwargs.pop("image_strength", None)
                resolved_seed = 0 if seed is None else int(seed)

                gen_kwargs: dict[str, Any] = {
                    "prompt": prompt,
                    "seed": resolved_seed,
                }
                if width is not None:
                    gen_kwargs["width"] = int(width)
                if height is not None:
                    gen_kwargs["height"] = int(height)
                if resolved_steps is not None:
                    gen_kwargs["num_inference_steps"] = resolved_steps
                if resolved_guidance is not None:
                    gen_kwargs["guidance"] = resolved_guidance

                if spec.edit_input_style == "image_path_mask":
                    if len(image_paths) != 1:
                        raise ValueError(f"{self.base_model} edit supports exactly one input image")
                    gen_kwargs["image_path"] = image_paths[0]
                    if mask_path is not None:
                        gen_kwargs["mask_path"] = mask_path
                elif spec.edit_input_style == "image_path":
                    if len(image_paths) != 1:
                        raise ValueError(f"{self.base_model} edit supports exactly one input image")
                    if mask_path is not None:
                        raise ValueError(f"{self.base_model} edit does not support mask_path")
                    gen_kwargs["image_path"] = image_paths[0]
                    resolved_image_strength = self._resolve_image_strength(request_image_strength)
                    if resolved_image_strength is not None:
                        gen_kwargs["image_strength"] = resolved_image_strength
                else:
                    if mask_path is not None:
                        raise ValueError(f"{self.base_model} edit does not support mask_path")
                    gen_kwargs["image_paths"] = image_paths
                    if request_image_strength is not None:
                        gen_kwargs["image_strength"] = _coerce_float(request_image_strength, "image_strength")

                gen_kwargs.update(kwargs)

                return await self._run_image_call(
                    model=model,
                    task="edit",
                    call_kwargs=gen_kwargs,
                    result_metadata={
                        "output_format": output_format,
                        "input_image_count": len(image_paths),
                        "mask_path": mask_path,
                        "quantize": key.quantize,
                        "model_path": key.model_path,
                    },
                )
        finally:
            await self._finish_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        """Get image engine statistics."""
        return {
            "model_name": self._model_name,
            "model_id": self._model_id,
            "loaded": self._started,
            "backend": self._image_metadata.get("backend", "mflux"),
            "base_model": self.base_model,
            "tasks": list(self._tasks),
            "loaded_tasks": sorted({key.task for key in self._models}),
        }

    def _primary_task(self) -> ImageTask:
        if self._tasks == ["edit"]:
            return "edit"
        if "generation" in self._tasks:
            return "generation"
        if "edit" in self._tasks:
            return "edit"
        raise ValueError(f"Image model {self._model_id} declares no supported tasks")

    @staticmethod
    def _normalize_tasks(raw_tasks: object) -> list[ImageTask]:
        if raw_tasks is None:
            return []
        items: Iterable[object]
        if isinstance(raw_tasks, str):
            items = [raw_tasks]
        elif isinstance(raw_tasks, Iterable):
            items = raw_tasks
        else:
            raise ValueError("Image model tasks must be a string or iterable of strings")
        normalized: list[ImageTask] = []
        for item in items:
            if not isinstance(item, str):
                raise ValueError("Image model tasks must contain only strings")
            task = _normalize_base_model(item)
            if task in {"generation", "generate", "text-to-image", "txt2img"}:
                normalized.append("generation")
            elif task in {"edit", "editing", "image-to-image", "img2img", "inpaint", "inpainting"}:
                normalized.append("edit")
            else:
                raise ValueError(f"Unsupported image model task: {item!r}")
        return sorted(set(normalized), key=("generation", "edit").index)

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError("Engine not started. Call start() first.")

    def _ensure_task_supported(self, task: ImageTask) -> None:
        if task not in self._tasks:
            raise ValueError(f"Image model {self._model_id} does not support task {task!r}")

    def _resolve_spec(self, task: ImageTask) -> _MfluxModelSpec:
        base_model = _normalize_base_model(self._image_metadata.get("base_model"))
        spec = _MFLUX_SPECS.get((task, base_model))
        if spec is None:
            raise ValueError(
                f"Unsupported mflux image base_model {self.base_model!r} for task {task!r}"
            )
        return spec

    def _default_quantize(self) -> int | None:
        return _coerce_int(self._image_metadata.get("quantize"), "quantize")

    def _default_model_path(self) -> str | None:
        value = self._image_metadata.get("model_path")
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Image manifest model_path must be a non-empty string")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(self._model_path).expanduser() / path
        return str(path)

    def _resolve_model_path_override(self, value: object) -> str | None:
        if value is None:
            return self._default_model_path()
        if not isinstance(value, str) or not value.strip():
            raise ValueError("image model_path override must be a non-empty string")
        path = Path(value).expanduser()
        if not path.is_absolute():
            candidate = Path(self._model_path).expanduser() / path
            path = candidate if candidate.exists() else path
        return str(path)

    def _model_key(
        self,
        task: ImageTask,
        quantize_override: object = None,
        model_path_override: object = None,
    ) -> _ModelKey:
        quantize = (
            self._default_quantize()
            if quantize_override is None
            else _coerce_int(quantize_override, "quantize")
        )
        model_path = self._resolve_model_path_override(model_path_override)
        return _ModelKey(task=task, quantize=quantize, model_path=model_path)

    def _resolve_steps(self, steps: int | None) -> int | None:
        if steps is not None:
            return int(steps)
        # Model-level manifest default.
        manifest_steps = _coerce_int(
            self._image_metadata.get("default_steps"), "default_steps"
        )
        if manifest_steps is not None:
            return manifest_steps
        # Per-model quality defaults (applied when manifest has none).
        base_model = self.base_model
        model_defaults = get_image_defaults(base_model)
        return _coerce_int(model_defaults.get("default_steps"), "default_steps")

    def _resolve_guidance(
        self, guidance: float | None
    ) -> float | None:
        if guidance is not None:
            return float(guidance)
        # Model-level manifest default.
        manifest_guidance = _coerce_float(
            self._image_metadata.get("default_guidance"), "default_guidance"
        )
        if manifest_guidance is not None:
            return manifest_guidance
        # Per-model quality defaults (applied when manifest has none).
        base_model = self.base_model
        model_defaults = get_image_defaults(base_model)
        return model_defaults.get("default_guidance")

    def _resolve_image_strength(self, image_strength: object) -> float | None:
        if image_strength is not None:
            return _coerce_float(image_strength, "image_strength")
        manifest_image_strength = _coerce_float(
            self._image_metadata.get("default_image_strength"), "default_image_strength"
        )
        if manifest_image_strength is not None:
            return manifest_image_strength
        model_defaults = get_image_defaults(self.base_model)
        return _coerce_float(
            model_defaults.get("default_image_strength"), "default_image_strength"
        )

    async def _load_model_for_task(
        self,
        task: ImageTask,
        *,
        quantize_override: object = None,
        model_path_override: object = None,
    ) -> Any:
        key = self._model_key(task, quantize_override, model_path_override)
        model = self._models.get(key)
        if model is not None:
            return model

        async with self._load_lock:
            model = self._models.get(key)
            if model is not None:
                return model

            if self._models:
                logger.info(
                    "Unloading previous image model variant before loading: model=%s, task=%s",
                    self._model_id,
                    task,
                )
                self._models.clear()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(get_mlx_executor(), _cleanup_mlx_cache)

            spec = self._resolve_spec(task)
            base_model = self.base_model

            def _load_sync() -> Any:
                try:
                    config_module = importlib.import_module(
                        "mflux.models.common.config.model_config"
                    )
                    model_module = importlib.import_module(spec.module)
                except ImportError as exc:
                    raise ImportError(MFLUX_MISSING_MESSAGE) from exc

                model_config = config_module.ModelConfig.from_name(spec.config_name)
                model_cls = getattr(model_module, spec.class_name)
                init_kwargs = {
                    "model_config": model_config,
                    "quantize": key.quantize,
                    "model_path": key.model_path,
                }
                return model_cls(**self._filter_supported_kwargs(model_cls, init_kwargs))

            logger.info(
                "Loading image model: model=%s, base_model=%s, task=%s, quantize=%s, model_path=%s",
                self._model_id,
                base_model,
                task,
                key.quantize,
                key.model_path,
            )
            loop = asyncio.get_running_loop()
            model = await loop.run_in_executor(get_mlx_executor(), _load_sync)
            self._models[key] = model
            return model

    async def _run_image_call(
        self,
        *,
        model: Any,
        task: ImageTask,
        call_kwargs: dict[str, Any],
        result_metadata: dict[str, Any],
    ) -> ImageEngineResult:
        loop = asyncio.get_running_loop()

        def _call_sync() -> Any:
            kwargs = self._filter_supported_kwargs(model.generate_image, call_kwargs)
            return model.generate_image(**kwargs)

        raw_result = await loop.run_in_executor(get_mlx_executor(), _call_sync)
        image = getattr(raw_result, "image", raw_result)
        metadata = self._build_result_metadata(raw_result, task, call_kwargs)
        metadata.update({k: v for k, v in result_metadata.items() if v is not None})
        return ImageEngineResult(image=image, metadata=metadata)

    async def _finish_activity(self, activity_id: str) -> None:
        self._end_activity(activity_id)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(),
            _clear_mlx_cache,
        )

    def _build_result_metadata(
        self,
        raw_result: Any,
        task: ImageTask,
        call_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = {
            "model_id": self._model_id,
            "model_name": self._model_name,
            "base_model": self.base_model,
            "task": task,
            "prompt": call_kwargs.get("prompt"),
            "seed": call_kwargs.get("seed"),
            "steps": call_kwargs.get("num_inference_steps"),
            "guidance": call_kwargs.get("guidance"),
            "width": call_kwargs.get("width"),
            "height": call_kwargs.get("height"),
        }
        for attr in (
            "seed",
            "prompt",
            "steps",
            "guidance",
            "width",
            "height",
            "generation_time",
            "quantization",
            "negative_prompt",
        ):
            if hasattr(raw_result, attr):
                metadata[attr] = getattr(raw_result, attr)
        model_config = getattr(raw_result, "model_config", None)
        if model_config is not None and hasattr(model_config, "model_name"):
            metadata["mflux_model_name"] = model_config.model_name
        return {k: v for k, v in metadata.items() if v is not None}

    @staticmethod
    def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return kwargs
        params = signature.parameters
        if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in params}

    def __repr__(self) -> str:
        status = "running" if self._started else "stopped"
        return f"<ImageEngine model={self._model_id} base={self.base_model!r} status={status}>"
