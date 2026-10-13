from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from ..qwen3_5.qwen3_5 import NORM_WEIGHT_SUFFIXES
from ..switch_layers import (
    move_expert_projection,
    split_expert_projection,
    stack_expert_projection,
)
from .config import ModelConfig, validate_downsample_mode
from .language import LanguageModel
from .vision import Merger, VisionModel


@dataclass(frozen=True)
class VisualFeatures:
    values: mx.array
    target_sizes: tuple
    downsample_mode: str
    owner: int
    revision: int
    max_slice_nums: int | None
    parameter_identity: tuple


def _slice_limit(value):
    if isinstance(value, mx.array):
        if value.ndim != 0 or not mx.issubdtype(value.dtype, mx.integer):
            raise ValueError("max_slice_nums must be a strictly positive integer")
        value = value.item()
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError("max_slice_nums must be a strictly positive integer")
    return None if value is None else int(value)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config, config)
        self.vision_tower = VisionModel(config.vision_config)
        self.merger = Merger(config)
        self._visual_revision = 0

    def load_weights(self, *args, **kwargs):
        result = super().load_weights(*args, **kwargs)
        self._visual_revision += 1
        return result

    def _visual_parameter_identity(self):
        return tuple(
            (prefix, name, id(value))
            for prefix, module in (
                ("vision", self.vision_tower),
                ("merger", self.merger),
            )
            for name, value in tree_flatten(module.parameters())
        )

    def get_image_features(
        self, pixel_values, target_sizes, downsample_mode=None, max_slice_nums=None
    ):
        mode = validate_downsample_mode(
            self.config.downsample_mode if downsample_mode is None else downsample_mode
        )
        max_slice_nums = _slice_limit(max_slice_nums)
        if target_sizes is None:
            raise ValueError("Visual inputs require target_sizes")
        hidden, grids = self.vision_tower(
            pixel_values, target_sizes, use_vit_merger=(mode == "16x")
        )
        features = self.merger(hidden, grids)
        return VisualFeatures(
            mx.concatenate(features, axis=0),
            tuple(tuple(row) for row in np.asarray(target_sizes).tolist()),
            mode,
            id(self),
            self._visual_revision,
            max_slice_nums,
            self._visual_parameter_identity(),
        )

    def _features(self, pixels, grids, cached, mode, max_slice_nums):
        if cached is not None:
            if pixels is not None:
                raise ValueError("Specify visual pixels or cached features, not both")
            if not isinstance(cached, VisualFeatures):
                raise ValueError(
                    "Cached visuals must retain native geometry and model identity"
                )
            if grids is None:
                raise ValueError("Cached visuals require their target_sizes")
            if (
                cached.owner != id(self)
                or cached.revision != self._visual_revision
                or cached.downsample_mode != mode
                or cached.max_slice_nums != max_slice_nums
                or cached.parameter_identity != self._visual_parameter_identity()
                or cached.target_sizes
                != tuple(tuple(row) for row in np.asarray(grids).tolist())
            ):
                raise ValueError(
                    "Cached visual features do not match model, weights, geometry or controls"
                )
            return cached.values
        if pixels is not None:
            return self.get_image_features(pixels, grids, mode, max_slice_nums).values
        return None

    def get_input_embeddings(
        self, input_ids=None, pixel_values=None, **kwargs
    ) -> InputEmbeddingsFeatures:
        if input_ids is None:
            raise ValueError("MiniCPM-V 4.7 requires input_ids to locate visual spans")
        requested_mode = kwargs.get("downsample_mode")
        mode = validate_downsample_mode(
            self.config.downsample_mode if requested_mode is None else requested_mode
        )
        if "stack_frames" in kwargs:
            raise ValueError("Advanced video frame stacking is not supported")
        image_grids = kwargs.get("target_sizes")
        video_grids = kwargs.get("target_sizes_videos")
        mask = kwargs.get("attention_mask", kwargs.get("mask"))
        if kwargs.get("position_ids") is not None and all(
            value is None
            for value in (
                pixel_values,
                kwargs.get("pixel_values_videos"),
                kwargs.get("cached_image_features"),
                kwargs.get("cached_video_features"),
            )
        ):
            positions, deltas = kwargs["position_ids"], kwargs.get("rope_deltas")
            if (
                positions.ndim != 3
                or positions.shape[:2] != (3, input_ids.shape[0])
                or positions.shape[-1] < input_ids.shape[-1]
                or deltas is None
                or deltas.shape != (input_ids.shape[0], 1)
            ):
                raise ValueError(
                    "Explicit prompt positions require complete request-local deltas"
                )
            if bool(
                mx.any(
                    (input_ids == self.config.image_token_id)
                    | (input_ids == self.config.video_token_id)
                ).item()
            ):
                raise ValueError(
                    "Visual placeholders require pixels or validated features"
                )
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids),
                position_ids=positions,
                rope_deltas=deltas,
            )
        positions, deltas = self.language_model.get_rope_index(
            input_ids,
            mm_token_type_ids=kwargs.get("mm_token_type_ids"),
            target_sizes=image_grids,
            target_sizes_videos=video_grids,
            attention_mask=mask,
            downsample_mode=mode,
        )
        embeds = self.language_model.model.embed_tokens(input_ids)
        for token_id, pixels, grids, cached, limit in (
            (
                self.config.image_token_id,
                pixel_values,
                image_grids,
                kwargs.get("cached_image_features"),
                kwargs.get("image_max_slice_nums", kwargs.get("max_slice_nums")),
            ),
            (
                self.config.video_token_id,
                kwargs.get("pixel_values_videos"),
                video_grids,
                kwargs.get("cached_video_features"),
                kwargs.get("video_max_slice_nums", kwargs.get("max_slice_nums")),
            ),
        ):
            visual = self._features(pixels, grids, cached, mode, _slice_limit(limit))
            indices = np.flatnonzero(np.asarray(input_ids).reshape(-1) == token_id)
            if visual is None:
                if len(indices):
                    raise ValueError(
                        "Visual placeholders require pixels or validated features"
                    )
                continue
            if visual.shape != (len(indices), embeds.shape[-1]):
                raise ValueError(
                    "Visual feature shape does not match placeholder tokens"
                )
            flat = embeds.reshape(-1, embeds.shape[-1])
            flat[mx.array(indices)] = visual.astype(embeds.dtype)
            embeds = flat.reshape(embeds.shape)
        return InputEmbeddingsFeatures(
            inputs_embeds=embeds, position_ids=positions, rope_deltas=deltas
        )

    def __call__(
        self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs
    ) -> LanguageModelOutput:
        if kwargs.get("inputs_embeds") is None and (
            pixel_values is not None
            or kwargs.get("pixel_values_videos") is not None
            or kwargs.get("cached_image_features") is not None
            or kwargs.get("cached_video_features") is not None
        ):
            features = self.get_input_embeddings(
                input_ids, pixel_values, mask=mask, **kwargs
            )
            kwargs.update(
                {
                    key: value
                    for key, value in features.to_dict().items()
                    if value is not None
                }
            )
        return self.language_model(input_ids, mask=mask, cache=cache, **kwargs)

    def sanitize(self, weights):
        normalized = {}
        for original, value in weights.items():
            key = original
            hf_text = original.startswith("model.language_model.")
            hf_vision = original.startswith("model.vision_tower.")
            if hf_text:
                key = "language_model.model." + original[len("model.language_model.") :]
            elif hf_vision:
                key = "vision_tower." + original[len("model.vision_tower.") :]
            elif original.startswith("model.merger."):
                key = original[len("model.") :]
            elif original.startswith("lm_head."):
                key = "language_model." + original
            if key in normalized:
                raise ValueError(f"Duplicate checkpoint mapping for {key}")
            if hf_text:
                if key.endswith("conv1d.weight"):
                    if value.ndim != 3 or value.shape[1] != 1:
                        raise ValueError(
                            f"Invalid HF GDN convolution shape: {value.shape}"
                        )
                    value = value.moveaxis(2, 1)
                elif any(key.endswith(suffix) for suffix in NORM_WEIGHT_SUFFIXES):
                    value = value + 1.0
            if hf_vision and key.endswith("patch_embedding.weight"):
                if value.ndim != 4:
                    raise ValueError("HF visual patch convolution must be OIHW")
                value = value.transpose(0, 2, 3, 1)
            normalized[key] = value

        for layer in range(self.config.text_config.num_hidden_layers):
            prefix = f"language_model.model.layers.{layer}.mlp"
            source = f"{prefix}.experts"
            target = f"{prefix}.switch_mlp"
            fused = f"{source}.gate_up_proj"
            if fused in normalized or f"{fused}.weight" in normalized:
                if any(key.startswith(target + ".") for key in normalized):
                    raise ValueError(
                        "Checkpoint mixes fused and canonical expert projections"
                    )
                for projection in ("gate_up_proj", "down_proj"):
                    old = f"{source}.{projection}_scales"
                    if old in normalized:
                        normalized[f"{source}.{projection}.scales"] = normalized.pop(
                            old
                        )
                split_expert_projection(
                    normalized, fused, [f"{target}.gate_proj", f"{target}.up_proj"]
                )
                move_expert_projection(
                    normalized, f"{source}.down_proj", f"{target}.down_proj"
                )
            elif f"{source}.0.up_proj.weight" in normalized:
                if any(key.startswith(target + ".") for key in normalized):
                    raise ValueError(
                        "Checkpoint mixes per-expert and canonical projections"
                    )
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    stack_expert_projection(
                        normalized,
                        source,
                        projection,
                        f"{target}.{projection}",
                        self.config.text_config.num_experts,
                    )
        if self.config.tie_word_embeddings:
            normalized.pop("language_model.lm_head.weight", None)
        return normalized

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
