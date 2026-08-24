# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the OpenAI-compatible image generation API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .shared_models import get_unix_timestamp

ImageOutputFormat = Literal["png", "jpeg", "webp"]
ImageResponseFormat = Literal["b64_json", "url"]


def parse_image_size(size: str | None) -> tuple[int | None, int | None]:
    """Parse an OpenAI image size string into width and height."""
    if size is None:
        return None, None

    normalized = size.strip().lower()
    if normalized == "auto":
        return None, None

    parts = normalized.split("x")
    if len(parts) != 2:
        raise ValueError("size must be 'auto' or WIDTHxHEIGHT")

    try:
        width, height = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError("size must be 'auto' or WIDTHxHEIGHT") from exc

    if width <= 0 or height <= 0:
        raise ValueError("size dimensions must be positive")
    return width, height


class _ImageRequestBase(BaseModel):
    """Fields shared by OpenAI-compatible image requests."""

    prompt: str
    model: str
    n: int = Field(default=1, ge=1, le=10)
    size: str | None = "auto"
    quality: str | None = "auto"
    output_format: ImageOutputFormat = "png"
    response_format: ImageResponseFormat | None = "b64_json"
    stream: bool = False
    partial_images: int | None = None
    user: str | None = None

    # oMLX/mlx-vlm extension fields.
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    negative_prompt: str | None = None
    scheduler: str | None = None
    lora_paths: list[str] | None = None
    lora_scales: list[float] | None = None

    model_config = {"extra": "allow"}

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return value

    @field_validator("output_format", mode="before")
    @classmethod
    def normalize_output_format(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip().lower()
        if value not in {"png", "jpeg", "webp"}:
            raise ValueError("output_format must be one of: png, jpeg, webp")
        return value

    @field_validator("response_format", mode="before")
    @classmethod
    def normalize_response_format(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, str):
            value = value.strip().lower()
        if value not in {"b64_json", "url"}:
            raise ValueError("response_format must be one of: b64_json, url")
        return value

    def parsed_size(self) -> tuple[int | None, int | None]:
        return parse_image_size(self.size)


class ImageGenerationRequest(_ImageRequestBase):
    """Request body for POST /v1/images/generations."""

    background: str | None = "auto"
    style: str | None = None
    moderation: str | None = None


class ImageUsage(BaseModel):
    """Optional image response usage accounting."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    image_count: int | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.total_tokens == 0 and (
            self.prompt_tokens > 0 or self.completion_tokens > 0
        ):
            object.__setattr__(
                self,
                "total_tokens",
                self.prompt_tokens + self.completion_tokens,
            )
        if self.input_tokens == 0 and self.prompt_tokens > 0:
            object.__setattr__(self, "input_tokens", self.prompt_tokens)
        if self.output_tokens == 0 and self.completion_tokens > 0:
            object.__setattr__(self, "output_tokens", self.completion_tokens)


class ImageData(BaseModel):
    """A single OpenAI-style image result."""

    b64_json: str
    revised_prompt: str | None = None
    size: str | None = None
    quality: str | None = None
    output_format: ImageOutputFormat | None = None
    background: str | None = None

    @field_validator("output_format", mode="before")
    @classmethod
    def normalize_output_format(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, str):
            value = value.strip().lower()
        if value not in {"png", "jpeg", "webp"}:
            raise ValueError("output_format must be one of: png, jpeg, webp")
        return value


class ImageResponse(BaseModel):
    """Response body shared by OpenAI-compatible image endpoints."""

    created: int = Field(default_factory=get_unix_timestamp)
    data: list[ImageData]
    model: str | None = None
    usage: ImageUsage | None = None
