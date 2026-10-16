# SPDX-License-Identifier: Apache-2.0
"""Native Nemotron media preparation with explicit token/embedding alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from mlx_vlm.models.nemotron_h_nano_omni import Model


@dataclass(frozen=True)
class PreparedNemotronPrompt:
    source_token_ids: mx.array
    retained_positions: mx.array
    token_ids: mx.array
    inputs_embeds: mx.array
    attention_mask: mx.array | None


class _VideoFeatureCapture:
    def __init__(self, model: Model):
        self.model = model
        self.video_features: mx.array | None = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def extract_video_feature(self, pixels):
        self.video_features = self.model.extract_video_feature(pixels)
        return self.video_features


def prepare_nemotron_prompt(
    model: Model,
    input_ids: mx.array,
    pixel_values: mx.array | None = None,
    mask: mx.array | None = None,
    **kwargs,
) -> PreparedNemotronPrompt:
    from mlx_vlm.models.nemotron_h_nano_omni.nemotron_h_nano_omni import (
        Model as NativeModel,
    )
    from mlx_vlm.models.nemotron_h_nano_omni.nemotron_h_nano_omni import (
        compute_retention_mask,
    )

    has_video = kwargs.get("pixel_values_videos") is not None
    pruning_rate = kwargs.get("video_pruning_rate", model.video_pruning_rate)
    if has_video and not 0 <= pruning_rate <= 1:
        raise ValueError("video_pruning_rate must be between 0 and 1.")
    view = _VideoFeatureCapture(model)
    # Let native merging run once, retaining the original pre-cast features.
    # Apply its pruning decision to both modalities and IDs, not embeddings alone.
    output = NativeModel.get_input_embeddings(
        view,  # type: ignore[arg-type]  # Read-through proxy for the native method.
        input_ids,
        pixel_values,
        mask=mask,
        **{**kwargs, "video_pruning_rate": 0},
    )
    retained = np.arange(input_ids.shape[1])
    if has_video and pruning_rate > 0:
        features = view.video_features
        if features is None:
            raise RuntimeError(
                "Native Nemotron preparation returned no video features."
            )
        keep = np.ones(input_ids.shape[1], dtype=bool)
        positions = np.flatnonzero(np.array(input_ids[0] == model.img_context_token_id))
        keep[positions] = np.array(compute_retention_mask(features, pruning_rate))
        retained = np.flatnonzero(keep)
    indices = mx.array(retained, dtype=mx.int32)
    embeddings = output.inputs_embeds[:, indices]
    tokens = input_ids[:, indices]
    attention_mask = None if mask is None else mask[..., indices]
    if tokens.shape[:2] != embeddings.shape[:2]:
        raise ValueError("Nemotron prompt tokens and embeddings are not aligned.")
    return PreparedNemotronPrompt(
        source_token_ids=input_ids,
        retained_positions=indices,
        token_ids=tokens,
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
    )
