import mlx.core as mx
import numpy as np
from mlx import nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


def masked_scatter(final_embedding, mask, features):
    shape = final_embedding.shape
    flat = mx.flatten(final_embedding)
    positions = mx.array(np.where(mx.flatten(mask))[0], mx.uint32)
    flat[positions] = mx.flatten(features)
    return flat.reshape(shape)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = (
            None if config.skip_vision else VisionModel(config.vision_config)
        )
        packed_size = (
            config.vision_config.hidden_size
            * config.vision_config.spatial_merge_size**2
        )
        self.linear_proj = [
            nn.Linear(packed_size, config.text_config.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(
                config.text_config.hidden_size,
                config.text_config.hidden_size,
                bias=True,
            ),
        ]
        self.language_model = LanguageModel(config.text_config, config)

    def _project(self, features):
        for layer in self.linear_proj:
            features = layer(features)
        return features

    def get_input_embeddings(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        **kwargs,
    ):
        image_grid_thw = kwargs.get("image_grid_thw")
        video_grid_thw = kwargs.get("video_grid_thw")
        mask = kwargs.get("mask")
        inputs_embeds = self.language_model.model.word_embeddings(input_ids)
        cached = kwargs.get("cached_image_features")
        if (
            pixel_values is None
            and kwargs.get("pixel_values_videos") is None
            and cached is None
        ):
            if kwargs.get("cache") is not None:
                return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, attention_mask=mask
            )
            return InputEmbeddingsFeatures(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                rope_deltas=rope_deltas,
            )

        if cached is not None:
            features = cached
            visual_mask = (input_ids == self.config.image_token_index) | (
                input_ids == self.config.video_token_index
            )
            expanded = mx.broadcast_to(visual_mask[..., None], inputs_embeds.shape)
            if expanded.sum().item() != features.size:
                raise ValueError(
                    "Cached visual features and visual tokens do not match: "
                    f"tokens={visual_mask.sum().item()}, features={features.shape[0]}"
                )
            inputs_embeds = masked_scatter(inputs_embeds, expanded, features)
        else:
            if self.vision_tower is None:
                raise ValueError("bailing_moe_v3_vl was loaded without a vision tower")
            dtype = self.vision_tower.patch_embed.proj.weight.dtype
            media = (
                (
                    pixel_values,
                    image_grid_thw,
                    self.config.image_token_index,
                    "Image",
                ),
                (
                    kwargs.get("pixel_values_videos"),
                    video_grid_thw,
                    self.config.video_token_index,
                    "Video",
                ),
            )
            for pixels, grid, token_id, label in media:
                if pixels is None:
                    continue
                features, _ = self.vision_tower(pixels.astype(dtype), grid)
                features = self._project(features)
                media_mask = input_ids == token_id
                expanded = mx.broadcast_to(media_mask[..., None], inputs_embeds.shape)
                if expanded.sum().item() != features.size:
                    raise ValueError(
                        f"{label} features and {label.lower()} tokens do not match: "
                        f"tokens={media_mask.sum().item()}, features={features.shape[0]}"
                    )
                inputs_embeds = masked_scatter(inputs_embeds, expanded, features)

        visual_mask = (input_ids == self.config.image_token_index) | (
            input_ids == self.config.video_token_index
        )
        position_ids, rope_deltas = self.language_model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, mask
        )
        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_mask,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
        )

    def __call__(
        self,
        input_ids,
        pixel_values=None,
        mask=None,
        cache=None,
        **kwargs,
    ):
        features = self.get_input_embeddings(
            input_ids, pixel_values, mask=mask, cache=cache, **kwargs
        )
        kwargs.update(features.to_dict())
        kwargs["pixel_values"] = pixel_values
        return self.language_model(input_ids, mask=mask, cache=cache, **kwargs)

    @property
    def layers(self):
        return self.language_model.model.layers

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("model.visual."):
                key = "vision_tower." + key[len("model.visual.") :]
            elif key.startswith("model."):
                key = "language_model.model." + key[len("model.") :]
            elif key.startswith("lm_head."):
                key = "language_model." + key
            sanitized[key] = value
        return sanitized

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
