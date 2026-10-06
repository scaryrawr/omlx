# SPDX-License-Identifier: Apache-2.0
"""mlx-vlm-backed image generation and editing engine for oMLX."""

from __future__ import annotations

import asyncio
import gc
import importlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx

from ..engine_core import get_mlx_executor
from ..image_registry import (
    get_image_defaults,
    get_image_model_reference,
    get_image_model_spec,
    image_edit_accepts_multiple_inputs,
    normalize_image_alias,
)
from ..utils.optional_deps import MLX_VLM_MISSING_MESSAGE
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

ImageTask = Literal["generation", "edit"]


def _cleanup_mlx_cache() -> None:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()


def _clear_mlx_cache() -> None:
    mx.synchronize()
    mx.clear_cache()


@dataclass(frozen=True)
class _ModelKey:
    task: ImageTask
    model_reference: str


@dataclass(frozen=True)
class _ImageAPI:
    load_image_model: Any
    generate_image: Any
    ImageGenerationRequest: Any
    ImageEditRequest: Any


@dataclass(slots=True)
class ImageEngineResult:
    """A route-friendly image result returned by an mlx-vlm model."""

    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)


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


def _image_api() -> _ImageAPI:
    """Import mlx-vlm's generic image API after registering oMLX shims."""
    try:
        from ..patches.mlx_vlm_image_compat import (
            apply_mlx_vlm_image_compat_patch,
        )

        apply_mlx_vlm_image_compat_patch()
        image = importlib.import_module("mlx_vlm.generate.image")
        edit_image = importlib.import_module("mlx_vlm.generate.edit_image")
    except (AttributeError, ImportError) as exc:
        raise ImportError(MLX_VLM_MISSING_MESSAGE) from exc

    return _ImageAPI(
        load_image_model=image.load_image_model,
        generate_image=image.generate_image,
        ImageGenerationRequest=image.ImageGenerationRequest,
        ImageEditRequest=edit_image.ImageEditRequest,
    )


class ImageEngine(BaseNonStreamingEngine):
    """Non-streaming image engine using mlx-vlm model loaders and requests."""

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
        self._has_explicit_model_path = model_path is not None
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
        """Return the manifest's image family alias."""
        return str(self._image_metadata.get("base_model", "")).strip()

    async def start(self) -> None:
        """Validate the manifest and load the primary image task."""
        if self._started:
            return

        backend = str(self._image_metadata.get("backend", "mlx-vlm")).strip().lower()
        if backend != "mlx-vlm":
            raise ValueError(
                f"Unsupported image backend for {self._model_id}: {backend!r}"
            )
        if "quantize" in self._image_metadata:
            raise ValueError(
                "Image manifest quantize is not supported by the mlx-vlm image backend"
            )
        if not self._tasks:
            raise ValueError(f"Image model {self._model_id} declares no supported tasks")
        for task in self._tasks:
            self._resolve_spec(task)

        logger.info(
            "Starting mlx-vlm image engine: model=%s, base_model=%s, tasks=%s",
            self._model_id,
            self.base_model,
            ",".join(self._tasks),
        )
        await self._load_model_for_task(self._primary_task())
        self._started = True

    def _primary_task(self) -> ImageTask:
        """Choose the task whose weights are accounted during pool admission."""
        if "generation" in self._tasks:
            return "generation"
        return "edit"

    async def stop(self) -> None:
        """Drop the loaded image variant and release MLX buffers."""
        await self._stop(clear_mlx_cache=True)

    async def stop_without_global_cleanup(self) -> None:
        """Drop this model without synchronizing another active MLX workload."""
        await self._stop(clear_mlx_cache=False)

    async def _stop(self, *, clear_mlx_cache: bool) -> None:
        if not self._models and not self._started:
            return

        async with self._call_lock:
            logger.info("Stopping image engine: %s", self._model_id)
            self._models.clear()
            self._started = False
            if clear_mlx_cache:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(get_mlx_executor(), _cleanup_mlx_cache)
            else:
                gc.collect()

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
        """Generate one image through mlx-vlm's generic generation API."""
        self._ensure_started()
        self._ensure_task_supported("generation")
        self._reject_quantize(kwargs)

        activity_id = self._begin_activity(
            "generating image",
            detail="Generating image",
            metadata={"prompt_length": len(prompt), "width": width, "height": height},
        )
        try:
            async with self._call_lock:
                model_path_override = kwargs.pop("model_path", None)
                key, model = await self._load_model_for_task(
                    "generation",
                    model_path_override=model_path_override,
                )
                api = _image_api()
                request = api.ImageGenerationRequest(
                    prompt=prompt,
                    seed=0 if seed is None else int(seed),
                    steps=self._resolve_steps(steps, "generation"),
                    width=None if width is None else int(width),
                    height=None if height is None else int(height),
                    guidance=self._resolve_guidance(guidance, "generation"),
                    output_format="png",
                    extra=self._request_extra(kwargs),
                )
                raw_result = await self._run_api_call(
                    api.generate_image,
                    model,
                    request,
                    task="generate",
                )
                return self._result_from_raw(
                    raw_result,
                    task="generation",
                    output_format=output_format,
                    model_key=key,
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
        """Edit input images through mlx-vlm's generic edit API."""
        self._ensure_started()
        self._ensure_task_supported("edit")
        self._reject_quantize(kwargs)
        if not image_paths:
            raise ValueError("edit requires at least one input image path")
        if mask_path is not None:
            raise ValueError("mlx-vlm image models do not support masks")
        if not image_edit_accepts_multiple_inputs(self.base_model) and len(
            image_paths
        ) != 1:
            raise ValueError(
                f"{self.base_model} edit supports exactly one input image"
            )

        activity_id = self._begin_activity(
            "editing image",
            detail="Editing image",
            metadata={"prompt_length": len(prompt), "width": width, "height": height},
        )
        try:
            async with self._call_lock:
                model_path_override = kwargs.pop("model_path", None)
                key, model = await self._load_model_for_task(
                    "edit",
                    model_path_override=model_path_override,
                )
                api = _image_api()
                image_strength = self._resolve_image_strength(
                    kwargs.pop("image_strength", None),
                    "edit",
                )
                request = api.ImageEditRequest(
                    prompt=prompt,
                    image_paths=tuple(image_paths),
                    seed=0 if seed is None else int(seed),
                    steps=self._resolve_steps(steps, "edit"),
                    width=None if width is None else int(width),
                    height=None if height is None else int(height),
                    guidance=self._resolve_guidance(guidance, "edit"),
                    output_format="png",
                    extra=self._request_extra(
                        kwargs,
                        image_strength=image_strength,
                    ),
                )
                raw_result = await self._run_api_call(
                    api.generate_image,
                    model,
                    request,
                    task="edit",
                )
                return self._result_from_raw(
                    raw_result,
                    task="edit",
                    output_format=output_format,
                    model_key=key,
                    input_image_count=len(image_paths),
                )
        finally:
            await self._finish_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        """Get image engine statistics."""
        return {
            "model_name": self._model_name,
            "model_id": self._model_id,
            "loaded": self._started,
            "backend": self._image_metadata.get("backend", "mlx-vlm"),
            "base_model": self.base_model,
            "tasks": list(self._tasks),
            "loaded_tasks": sorted({key.task for key in self._models}),
        }

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
            task = normalize_image_alias(item)
            if task in {"generation", "generate", "text-to-image", "txt2img"}:
                normalized.append("generation")
            elif task in {
                "edit",
                "editing",
                "image-to-image",
                "img2img",
                "inpaint",
                "inpainting",
            }:
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

    def _resolve_spec(self, task: ImageTask):
        spec = get_image_model_spec(self.base_model)
        if spec is None or task not in spec.tasks:
            raise ValueError(
                f"Unsupported mlx-vlm image base_model {self.base_model!r} "
                f"for task {task!r}"
            )
        return spec

    def _resolve_model_reference(self, override: object = None) -> str:
        value = self._image_metadata.get("model_path") if override is None else override
        root = Path(self._model_path).expanduser()
        if value is None:
            if not root.exists() and not self._has_explicit_model_path:
                # Programmatic callers can construct an engine from a known
                # mlx-vlm alias without a discovered local model directory.
                spec = get_image_model_spec(self.base_model)
                return (
                    get_image_model_reference(spec.base_model)
                    if spec is not None
                    else str(root)
                )
            return str(root)
        if not isinstance(value, str) or not value.strip():
            field = "Image manifest model_path" if override is None else "image model_path override"
            raise ValueError(f"{field} must be a non-empty string")
        path = Path(value).expanduser()
        if not path.is_absolute():
            candidate = root / path
            path = candidate if override is None or candidate.exists() else path
        if override is None and not path.exists():
            raise ValueError(f"Image manifest model_path does not exist: {path}")
        return str(path)

    def _model_key(
        self,
        task: ImageTask,
        model_path_override: object = None,
    ) -> _ModelKey:
        return _ModelKey(
            task=task,
            model_reference=self._resolve_model_reference(model_path_override),
        )

    def _resolve_steps(self, steps: int | None, task: ImageTask) -> int | None:
        if steps is not None:
            return int(steps)
        manifest_steps = _coerce_int(
            self._image_metadata.get("default_steps"), "default_steps"
        )
        if manifest_steps is not None:
            return manifest_steps
        return _coerce_int(
            get_image_defaults(self.base_model, task).get("default_steps"),
            "default_steps",
        )

    def _resolve_guidance(
        self, guidance: float | None, task: ImageTask
    ) -> float | None:
        if guidance is not None:
            return float(guidance)
        manifest_guidance = _coerce_float(
            self._image_metadata.get("default_guidance"), "default_guidance"
        )
        if manifest_guidance is not None:
            return manifest_guidance
        return _coerce_float(
            get_image_defaults(self.base_model, task).get("default_guidance"),
            "default_guidance",
        )

    def _resolve_image_strength(
        self, image_strength: object, task: ImageTask
    ) -> float | None:
        if image_strength is not None:
            return _coerce_float(image_strength, "image_strength")
        manifest_strength = _coerce_float(
            self._image_metadata.get("default_image_strength"),
            "default_image_strength",
        )
        if manifest_strength is not None:
            return manifest_strength
        return _coerce_float(
            get_image_defaults(self.base_model, task).get("default_image_strength"),
            "default_image_strength",
        )

    async def _load_model_for_task(
        self,
        task: ImageTask,
        *,
        model_path_override: object = None,
    ) -> tuple[_ModelKey, Any]:
        key = self._model_key(task, model_path_override)
        model = self._models.get(key)
        if model is not None:
            return key, model

        async with self._load_lock:
            model = self._models.get(key)
            if model is not None:
                return key, model

            if self._models:
                logger.info(
                    "Unloading previous mlx-vlm image variant before loading: "
                    "model=%s, task=%s",
                    self._model_id,
                    task,
                )
                self._models.clear()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(get_mlx_executor(), _cleanup_mlx_cache)

            self._resolve_spec(task)
            api = _image_api()
            loader_task = "generate" if task == "generation" else "edit"

            def load_sync() -> Any:
                return api.load_image_model(key.model_reference, task=loader_task)

            logger.info(
                "Loading mlx-vlm image model: model=%s, base_model=%s, task=%s, "
                "model_path=%s",
                self._model_id,
                self.base_model,
                task,
                key.model_reference,
            )
            loop = asyncio.get_running_loop()
            try:
                model = await loop.run_in_executor(get_mlx_executor(), load_sync)
            except ImportError as exc:
                raise ImportError(MLX_VLM_MISSING_MESSAGE) from exc
            self._models[key] = model
            return key, model

    @staticmethod
    async def _run_api_call(
        generate_image: Any,
        model: Any,
        request: Any,
        *,
        task: str,
    ) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_mlx_executor(),
            lambda: generate_image(model, request, task=task),
        )

    def _request_extra(
        self,
        kwargs: dict[str, Any],
        *,
        image_strength: float | None = None,
    ) -> dict[str, Any]:
        extra = {key: value for key, value in kwargs.items() if value is not None}
        request_strength = extra.pop("image_strength", None)
        if image_strength is None and request_strength is not None:
            image_strength = _coerce_float(request_strength, "image_strength")
        if image_strength is not None:
            # Z-Image consumes ``strength`` while ERNIE-Image consumes
            # ``image_strength``. Other mlx-vlm families safely ignore extras.
            extra["image_strength"] = image_strength
            extra["strength"] = image_strength
        return extra

    @staticmethod
    def _reject_quantize(kwargs: dict[str, Any]) -> None:
        if "quantize" in kwargs:
            raise ValueError(
                "Image request quantize is not supported by the mlx-vlm image backend"
            )

    def _result_from_raw(
        self,
        raw_result: Any,
        *,
        task: ImageTask,
        output_format: str,
        model_key: _ModelKey,
        input_image_count: int | None = None,
    ) -> ImageEngineResult:
        raw_metadata = getattr(raw_result, "metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        metadata.update(
            {
                "model_id": self._model_id,
                "model_name": self._model_name,
                "base_model": self.base_model,
                "task": task,
                "output_format": output_format,
                "model_path": model_key.model_reference,
            }
        )
        if input_image_count is not None:
            metadata["input_image_count"] = input_image_count
        for attribute in (
            "seed",
            "prompt",
            "steps",
            "guidance",
            "width",
            "height",
            "model",
            "family",
            "variant",
            "prompt_tokens",
            "peak_memory",
        ):
            value = getattr(raw_result, attribute, None)
            if value is not None:
                metadata[attribute] = value
        return ImageEngineResult(
            image=getattr(raw_result, "image", raw_result),
            metadata=metadata,
        )

    async def _finish_activity(self, activity_id: str) -> None:
        self._end_activity(activity_id)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_mlx_executor(), _clear_mlx_cache)

    def __repr__(self) -> str:
        status = "running" if self._started else "stopped"
        return (
            f"<ImageEngine model={self._model_id} base={self.base_model!r} "
            f"status={status}>"
        )
