# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible asynchronous video REST endpoints."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import ValidationError
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile
from starlette.responses import StreamingResponse

from ..video_jobs import VideoJobs
from .image_routes import _image_reference_to_path, _upload_to_path
from .video_models import Video, VideoGenerationRequest, VideoList

router = APIRouter(prefix="/v1/videos", tags=["Videos"])


def _jobs(request: Request) -> VideoJobs:
    service = request.app.state.video_jobs
    if not isinstance(service, VideoJobs):
        raise HTTPException(503, "Video job service is not initialized")
    return service


@router.post("", response_model=Video)
async def create_video(request: Request) -> Video:
    """Create a video job from JSON or SDK-compatible multipart input."""
    jobs = _jobs(request)
    with tempfile.TemporaryDirectory(prefix="omlx-video-input-") as temporary:
        directory = Path(temporary)
        reference = None
        try:
            content_type = request.headers.get("content-type", "").split(";")[0]
            if content_type == "application/json":
                body = VideoGenerationRequest.model_validate(await request.json())
                if body.input_reference is not None:
                    reference = Path(
                        await _image_reference_to_path(body.input_reference, directory)
                    )
            elif content_type in {
                "multipart/form-data",
                "application/x-www-form-urlencoded",
            }:
                async with request.form() as form:
                    fields: dict[str, str] = {}
                    upload = None
                    for key, value in form.multi_items():
                        if key in fields or (
                            key == "input_reference" and upload is not None
                        ):
                            raise HTTPException(400, f"Duplicate field '{key}'")
                        if key == "input_reference" and isinstance(value, UploadFile):
                            upload = value
                        elif isinstance(value, str):
                            fields[key] = value
                        else:
                            raise HTTPException(
                                400, f"Unsupported upload field '{key}'"
                            )
                    body = VideoGenerationRequest.model_validate(fields)
                    if upload is not None:
                        reference = Path(
                            await _upload_to_path(upload, "input_reference", directory)
                        )
            else:
                raise HTTPException(415, "Use JSON or multipart/form-data")
        except ValidationError as exc:
            raise HTTPException(
                422, exc.errors(include_url=False, include_context=False)
            ) from exc
        except json.JSONDecodeError as exc:
            raise HTTPException(400, "Invalid JSON body") from exc
        return await jobs.submit(body, reference)


@router.get("", response_model=VideoList)
async def list_videos(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    order: Literal["asc", "desc"] = "desc",
    after: str | None = None,
) -> VideoList:
    """List video jobs with OpenAI cursor pagination."""
    return _jobs(request).list(limit=limit, order=order, after=after)


@router.get("/{video_id}", response_model=Video)
async def retrieve_video(video_id: str, request: Request) -> Video:
    """Retrieve the current state of a video job."""
    return _jobs(request).get(video_id)


@router.get("/{video_id}/content")
async def download_video(
    video_id: str, request: Request, variant: str = "video"
) -> StreamingResponse:
    """Download a completed MP4 while retaining its open file handle."""
    if variant != "video":
        raise HTTPException(400, "Only the video content variant is supported")
    file = _jobs(request).content_path(video_id).open("rb")

    def chunks():
        while data := file.read(64 * 1024):
            yield data

    return StreamingResponse(
        chunks(),
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{video_id}.mp4"'},
        background=BackgroundTask(file.close),
    )


@router.delete("/{video_id}")
async def delete_video(video_id: str, request: Request) -> dict[str, str | bool]:
    """Delete a completed or failed video and its stored assets."""
    _jobs(request).delete(video_id)
    return {"id": video_id, "object": "video.deleted", "deleted": True}
