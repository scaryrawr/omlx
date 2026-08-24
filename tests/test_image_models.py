# SPDX-License-Identifier: Apache-2.0
"""Tests for OpenAI-compatible image API models."""

import pytest
from pydantic import ValidationError

from omlx.api.image_models import (
    ImageData,
    ImageGenerationRequest,
    ImageResponse,
    ImageUsage,
    parse_image_size,
)


def test_generation_request_defaults_and_extensions():
    request = ImageGenerationRequest(
        prompt="a watercolor fox",
        model="image-model",
        seed=123,
        steps=12,
        guidance=3.5,
        negative_prompt="blurry",
        scheduler="euler",
        lora_paths=["style.safetensors"],
        lora_scales=[0.8],
    )

    assert request.n == 1
    assert request.size == "auto"
    assert request.parsed_size() == (None, None)
    assert request.response_format == "b64_json"
    assert request.output_format == "png"
    assert request.background == "auto"
    assert request.stream is False

    dumped = request.model_dump(exclude_none=True)
    assert dumped["negative_prompt"] == "blurry"
    assert dumped["lora_paths"] == ["style.safetensors"]
    assert dumped["lora_scales"] == [0.8]


@pytest.mark.parametrize("n", [0, 11])
def test_generation_request_rejects_n_outside_openai_range(n):
    with pytest.raises(ValidationError):
        ImageGenerationRequest(prompt="a fox", model="image-model", n=n)


def test_generation_request_rejects_empty_prompt():
    with pytest.raises(ValidationError, match="prompt must not be empty"):
        ImageGenerationRequest(prompt="   ", model="image-model")


def test_output_format_is_normalized_and_validated():
    request = ImageGenerationRequest(
        prompt="a fox",
        model="image-model",
        output_format="WEBP",
    )

    assert request.output_format == "webp"

    with pytest.raises(ValidationError, match="output_format"):
        ImageGenerationRequest(
            prompt="a fox",
            model="image-model",
            output_format="gif",
        )


def test_parse_image_size_helper():
    assert parse_image_size(None) == (None, None)
    assert parse_image_size("auto") == (None, None)
    assert parse_image_size("1024x768") == (1024, 768)
    assert parse_image_size(" 1536 X 1024 ") == (1536, 1024)
    assert ImageGenerationRequest(
        prompt="a fox",
        model="image-model",
        size="512x512",
    ).parsed_size() == (512, 512)

    with pytest.raises(ValueError, match="WIDTHxHEIGHT"):
        parse_image_size("wide")
    with pytest.raises(ValueError, match="positive"):
        parse_image_size("0x512")


def test_image_response_serialization_and_usage():
    response = ImageResponse(
        model="image-model",
        data=[
            ImageData(
                b64_json="aW1hZ2U=",
                revised_prompt="a better fox",
                size="1024x1024",
                quality="high",
                output_format="PNG",
                background="transparent",
            )
        ],
        usage=ImageUsage(prompt_tokens=3, completion_tokens=4, image_count=1),
    )

    assert response.data[0].output_format == "png"
    assert response.usage is not None
    assert response.usage.total_tokens == 7
    assert response.usage.input_tokens == 3
    assert response.usage.output_tokens == 4

    dumped = response.model_dump(mode="json", exclude_none=True)
    assert isinstance(dumped["created"], int)
    assert dumped["data"][0]["b64_json"] == "aW1hZ2U="
    assert dumped["usage"]["image_count"] == 1
