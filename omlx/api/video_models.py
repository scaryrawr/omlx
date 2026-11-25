# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible video job request and response models."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .image_models import ImageReference


class VideoGenerationRequest(BaseModel):
    """Create one asynchronous local video generation job."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=32000)
    seconds: Literal["4", "8", "12"] = "4"
    size: str = "1280x704"
    input_reference: ImageReference | None = None
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    steps: int | None = Field(default=None, ge=1, le=1000)
    guidance: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    negative_prompt: str | None = Field(default=None, max_length=32000)

    @field_validator("model", "prompt")
    @classmethod
    def nonempty(cls, value: str) -> str:
        """Reject blank names and prompts without modifying their contents."""
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("size")
    @classmethod
    def valid_size(cls, value: str) -> str:
        """Validate the local resolution extension at the request boundary."""
        if not re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", value):
            raise ValueError("size must be WIDTHxHEIGHT")
        width, height = (int(part) for part in value.split("x"))
        if width * height > 4 * 1024 * 1024:
            raise ValueError("size must not exceed 4194304 pixels")
        return value


class VideoError(BaseModel):
    """A terminal video generation error."""

    code: str
    message: str


class Video(BaseModel):
    """Historical OpenAI video job representation."""

    id: str
    object: Literal["video"] = "video"
    model: str
    status: Literal["queued", "in_progress", "completed", "failed"] = "queued"
    progress: int = 0
    created_at: int
    completed_at: int | None = None
    expires_at: int | None = None
    error: VideoError | None = None
    prompt: str
    seconds: str
    size: str
    remixed_from_video_id: str | None = None


class VideoList(BaseModel):
    """A cursor-paginated list of video jobs."""

    object: Literal["list"] = "list"
    data: list[Video]
    has_more: bool
    first_id: str | None
    last_id: str | None
