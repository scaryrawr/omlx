# SPDX-License-Identifier: Apache-2.0
"""Tests for OpenAI-compatible image API models."""

import pytest
from pydantic import ValidationError

from omlx.api.image_models import (
    ImageData,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageMultipartEditRequest,
    ImageReference,
    ImageResponse,
    ImageURLReference,
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
        image_strength=0.4,
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
    assert dumped["image_strength"] == 0.4
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


def test_edit_request_accepts_file_id_and_image_url_refs():
    request = ImageEditRequest(
        prompt="make it cinematic",
        model="edit-model",
        images=[
            {"file_id": "file-abc123"},
            {"image_url": {"url": "https://example.test/input.png"}},
        ],
        mask={"image_url": "data:image/png;base64,AAAA"},
        output_format="jpeg",
        input_fidelity="high",
        partial_images=2,
    )

    assert request.images[0].file_id == "file-abc123"
    assert isinstance(request.images[1].image_url, ImageURLReference)
    assert request.mask is not None
    assert request.mask.image_url == "data:image/png;base64,AAAA"
    assert request.output_format == "jpeg"
    assert request.input_fidelity == "high"
    assert request.parsed_size() == (None, None)


def test_multipart_edit_request_does_not_require_json_images():
    request = ImageMultipartEditRequest(
        prompt="make it cinematic",
        model="edit-model",
        seed=7,
    )

    assert request.prompt == "make it cinematic"
    assert request.model == "edit-model"
    assert request.seed == 7
    assert not hasattr(request, "images")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"file_id": "file-abc123", "image_url": "https://example.test/input.png"},
    ],
)
def test_image_reference_requires_exactly_one_shape(payload):
    with pytest.raises(ValidationError, match="exactly one"):
        ImageReference(**payload)


def test_edit_request_caps_input_image_count():
    with pytest.raises(ValidationError, match="at most 16"):
        ImageEditRequest(
            prompt="combine",
            model="edit-model",
            images=[
                {"image_url": f"https://example.test/{index}.png"}
                for index in range(17)
            ],
        )


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
