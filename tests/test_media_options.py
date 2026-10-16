# SPDX-License-Identifier: Apache-2.0
"""Shared request and model controls for MiniCPM preprocessing."""

import pytest
from pydantic import ValidationError

from omlx.api.anthropic_models import MessagesRequest, TokenCountRequest
from omlx.api.media_options import (
    MultimodalProcessorOptions,
    merge_mm_processor_kwargs,
)
from omlx.api.openai_models import ChatCompletionRequest
from omlx.api.responses_models import ResponsesRequest
from omlx.model_profiles import filter_profile_fields, filter_universal_fields
from omlx.model_settings import ModelSettings


@pytest.mark.parametrize(
    "request_type,fields",
    [
        (ChatCompletionRequest, {"messages": []}),
        (MessagesRequest, {"messages": [], "max_tokens": 1}),
        (TokenCountRequest, {"messages": []}),
        (ResponsesRequest, {"input": "Describe the image."}),
    ],
)
def test_existing_protocols_share_processor_option_validation(request_type, fields):
    request = request_type(
        model="mini",
        mm_processor_kwargs={"downsample_mode": "4x", "max_slice_nums": 9},
        **fields,
    )
    assert request.mm_processor_kwargs == MultimodalProcessorOptions(
        downsample_mode="4x", max_slice_nums=9
    )
    assert request_type(model="mini", **fields).mm_processor_kwargs is None


@pytest.mark.parametrize(
    "options",
    [
        {"downsample_mode": "8x"},
        {"max_slice_nums": 0},
        {"max_slice_nums": -1},
        {"max_slice_nums": True},
        {"max_slice_nums": 1.5},
        {"max_slice_nums": "9"},
        {"stack_frames": 1},
        {"arbitrary": True},
    ],
)
def test_invalid_or_deferred_controls_are_rejected(options):
    with pytest.raises(ValidationError):
        MultimodalProcessorOptions.model_validate(options)
    with pytest.raises(ValidationError):
        ModelSettings(mm_processor_kwargs=options)


def test_request_overrides_model_without_mutation_or_default_changes():
    model_options = {"downsample_mode": "16x", "max_slice_nums": 9}
    request_options = MultimodalProcessorOptions(downsample_mode="4x")
    assert merge_mm_processor_kwargs(model_options, request_options) == {
        "downsample_mode": "4x",
        "max_slice_nums": 9,
    }
    assert model_options == {"downsample_mode": "16x", "max_slice_nums": 9}
    assert merge_mm_processor_kwargs(None, None) == {}
    assert merge_mm_processor_kwargs(model_options, {"max_slice_nums": None}) == (
        model_options
    )


def test_processor_options_round_trip_only_through_model_profiles():
    options = {"downsample_mode": "4x", "max_slice_nums": 36}
    settings = ModelSettings.from_dict({"mm_processor_kwargs": options})
    assert settings.to_dict()["mm_processor_kwargs"] == options
    assert settings.mm_processor_kwargs is not options
    assert filter_profile_fields(settings.to_dict())["mm_processor_kwargs"] == options
    assert "mm_processor_kwargs" not in filter_universal_fields(settings.to_dict())
    assert "mm_processor_kwargs" not in ModelSettings().to_dict()
