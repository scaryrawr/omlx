# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible image generation routes for oMLX."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

from fastapi import APIRouter, HTTPException

from omlx.exceptions import (
    EnginePoolError,
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
    ModelUnavailableError,
)
from omlx.utils.optional_deps import require_mlx_vlm_available

from .image_models import (
    ImageData,
    ImageGenerationRequest,
    ImageResponse,
    ImageUsage,
)

router = APIRouter()

_SAFE_ENGINE_KWARG_FIELDS = {
    "negative_prompt",
    "scheduler",
}


def _get_engine_pool():
    """Return the active EnginePool from server state."""
    from omlx.server import _server_state

    pool = _server_state.engine_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return pool


def _resolve_model(model_id: str) -> str:
    """Resolve aliases the same way the server's other v1 routes do."""
    from omlx.server import resolve_model_id

    return resolve_model_id(model_id) or model_id


def _require_image_dependency() -> None:
    try:
        require_mlx_vlm_available()
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _normalize_task_names(tasks: object) -> set[str]:
    if tasks is None:
        return set()

    items: Iterable[object]
    if isinstance(tasks, str):
        items = [tasks]
    elif isinstance(tasks, Iterable):
        items = tasks
    else:
        return set()

    normalized: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        task = item.strip().lower().replace("_", "-")
        if task in {"generation", "generate", "text-to-image", "txt2img"}:
            normalized.add("generation")
        elif task in {
            "edit",
            "editing",
            "image-to-image",
            "img2img",
            "inpaint",
            "inpainting",
        }:
            normalized.add("edit")
    return normalized


def _validate_image_entry(entry: Any | None, model_id: str, task: str) -> bool:
    """Validate discovered model metadata when available."""
    if entry is None:
        return False

    model_type = getattr(entry, "model_type", None)
    engine_type = getattr(entry, "engine_type", None)
    entry_is_image = model_type == "image" or engine_type == "image"
    if not entry_is_image:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not an image model",
        )

    tasks = _normalize_task_names(getattr(entry, "tasks", None))
    if not tasks:
        metadata = getattr(entry, "image_metadata", None)
        if isinstance(metadata, dict):
            tasks = _normalize_task_names(metadata.get("tasks"))
    if tasks and task not in tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Image model '{model_id}' does not support task '{task}'",
        )
    return True


def _validate_image_engine(
    engine: Any,
    model_id: str,
    task: str,
    *,
    entry_is_image: bool,
) -> None:
    get_stats = getattr(engine, "get_stats", None)
    stats = get_stats() if callable(get_stats) else {}
    stats = stats if isinstance(stats, dict) else {}
    stats_tasks = _normalize_task_names(stats.get("tasks"))
    if stats_tasks and task not in stats_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Image model '{model_id}' does not support task '{task}'",
        )

    method_name = "generate" if task == "generation" else "edit"
    looks_like_image = (
        entry_is_image
        or type(engine).__name__ == "ImageEngine"
        or bool(stats_tasks)
        or stats.get("backend") == "mlx-vlm"
    )
    if not looks_like_image or not callable(getattr(engine, method_name, None)):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not an image-capable model",
        )


def _map_pool_error(exc: EnginePoolError, model_id: str) -> HTTPException:
    if isinstance(exc, ModelNotFoundError):
        available = (
            ", ".join(exc.available_models) if exc.available_models else "(none)"
        )
        return HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Available: {available}",
        )
    if isinstance(exc, (ModelTooLargeError, InsufficientMemoryError)):
        return HTTPException(status_code=507, detail=str(exc))
    if isinstance(exc, (ModelLoadingError, ModelBusyError)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ModelUnavailableError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@asynccontextmanager
async def _acquire_image_engine(
    model: str,
    task: str,
) -> AsyncIterator[tuple[str, Any]]:
    """Acquire and validate an image engine lease for one request."""
    pool = _get_engine_pool()
    resolved_model = _resolve_model(model)
    get_entry = getattr(pool, "get_entry", None)
    entry = get_entry(resolved_model) if callable(get_entry) else None
    entry_is_image = _validate_image_entry(entry, resolved_model, task)

    try:
        engine = await pool.get_engine(resolved_model, _lease=True)
    except EnginePoolError as exc:
        raise _map_pool_error(exc, resolved_model) from exc

    try:
        _validate_image_engine(
            engine,
            resolved_model,
            task,
            entry_is_image=entry_is_image,
        )
        yield resolved_model, engine
    finally:
        await pool.release_engine(resolved_model)


def _validate_generation_request(
    request: ImageGenerationRequest,
) -> tuple[int | None, int | None]:
    if request.lora_paths is not None or request.lora_scales is not None:
        raise HTTPException(
            status_code=400,
            detail="Request-level LoRA fields are not supported for image endpoints yet",
        )
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="stream is not supported for image endpoints",
        )
    if request.partial_images is not None:
        raise HTTPException(
            status_code=400,
            detail="partial_images is not supported for image endpoints",
        )
    if request.response_format == "url":
        raise HTTPException(
            status_code=400,
            detail="response_format='url' is not supported; use 'b64_json'",
        )
    try:
        return request.parsed_size()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _request_engine_kwargs(request: ImageGenerationRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for field in _SAFE_ENGINE_KWARG_FIELDS:
        value = getattr(request, field)
        if value is not None:
            kwargs[field] = value
    return kwargs


def _seed_for_output(seed: int | None, n: int, index: int) -> int | None:
    if seed is not None:
        return seed + index
    if n > 1:
        return index
    return None


def _encode_image_b64(image: Any, output_format: str) -> str:
    if not hasattr(image, "save"):
        raise HTTPException(
            status_code=500, detail="Image engine returned no PIL image"
        )

    pil_image = image
    image_format = output_format.upper()
    if image_format == "JPEG" and getattr(pil_image, "mode", None) in {
        "RGBA",
        "LA",
        "P",
    }:
        pil_image = pil_image.convert("RGB")

    buffer = BytesIO()
    try:
        pil_image.save(buffer, format=image_format)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode image as {output_format}: {exc}",
        ) from exc
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_data_from_result(
    result: Any,
    request: ImageGenerationRequest,
) -> ImageData:
    image = getattr(result, "image", result)
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    return ImageData(
        b64_json=_encode_image_b64(image, request.output_format),
        revised_prompt=metadata.get("revised_prompt"),
        size=request.size,
        quality=request.quality,
        output_format=request.output_format,
        background=request.background,
    )


async def _run_generation(
    resolved_model: str,
    engine: Any,
    request: ImageGenerationRequest,
    width: int | None,
    height: int | None,
) -> ImageResponse:
    data: list[ImageData] = []
    # ImageEngine serializes MLX calls, so n > 1 must remain sequential.
    for index in range(request.n):
        result = await engine.generate(
            request.prompt,
            width=width,
            height=height,
            seed=_seed_for_output(request.seed, request.n, index),
            steps=request.steps,
            guidance=request.guidance,
            output_format=request.output_format,
            **_request_engine_kwargs(request),
        )
        data.append(_image_data_from_result(result, request))

    return ImageResponse(
        data=data,
        model=resolved_model,
        usage=ImageUsage(image_count=len(data)),
    )


@router.post("/v1/images/generations", response_model=ImageResponse)
async def create_image_generation(request: ImageGenerationRequest) -> ImageResponse:
    """Create images from a prompt using a discovered ImageEngine model."""
    width, height = _validate_generation_request(request)
    _require_image_dependency()
    async with _acquire_image_engine(request.model, "generation") as (
        resolved_model,
        engine,
    ):
        try:
            return await _run_generation(
                resolved_model,
                engine,
                request,
                width,
                height,
            )
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
