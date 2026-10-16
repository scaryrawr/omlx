# SPDX-License-Identifier: Apache-2.0
"""Validated request-local multimodal preprocessing controls."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MultimodalProcessorOptions(BaseModel):
    """MiniCPM crop controls, separate from chat-template arguments."""

    model_config = ConfigDict(extra="forbid")

    downsample_mode: Literal["4x", "16x"] | None = None
    max_slice_nums: int | None = Field(default=None, strict=True, ge=1)


def merge_mm_processor_kwargs(
    model_options: dict | None,
    request_options: MultimodalProcessorOptions | dict | None,
) -> dict[str, str | int]:
    model: MultimodalProcessorOptions = MultimodalProcessorOptions.model_validate(
        model_options or {}
    )
    request: MultimodalProcessorOptions = MultimodalProcessorOptions.model_validate(
        {} if request_options is None else request_options
    )
    mode = (
        request.downsample_mode
        if request.downsample_mode is not None
        else model.downsample_mode
    )
    slices = (
        request.max_slice_nums
        if request.max_slice_nums is not None
        else model.max_slice_nums
    )
    options: dict[str, str | int] = {}
    if mode is not None:
        options["downsample_mode"] = mode
    if slices is not None:
        options["max_slice_nums"] = slices
    return options
