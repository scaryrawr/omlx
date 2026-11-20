# SPDX-License-Identifier: Apache-2.0
"""Serve the mlx-lm Inkling 2 module on the VLM engine through mlx-vlm's model
interface."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx_vlm.prompt_utils as prompt_utils
from mlx_lm.utils import load_model
from mlx_vlm.models.base import InputEmbeddingsFeatures, install_auto_processor_patch
from mlx_vlm.utils import get_model_path, load_processor

from . import apply_inkling2_patch
from . import model as inkling2
from .loop import MODEL_TYPE as LOOP_MODEL_TYPE
from .loop import LoopPlan, LoopTextModel
from .processing import Inkling2Processor

_MODEL_TYPE = "inkling2"
MODEL_TYPES = frozenset({_MODEL_TYPE, LOOP_MODEL_TYPE})
_UNSUPPORTED_INPUTS = ("pixel_values_videos", "input_values", "input_features")
_REGISTERED = False


def _register_mlx_vlm() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    # Without a MODEL_CONFIG entry apply_chat_template drops the image parts.
    model_config = getattr(prompt_utils, "MODEL_CONFIG", None)
    message_format = getattr(prompt_utils, "MessageFormat", None)
    for model_type in MODEL_TYPES:
        install_auto_processor_patch(model_type, Inkling2Processor)
        if isinstance(model_config, dict) and message_format is not None:
            model_config.setdefault(model_type, message_format.LIST_WITH_IMAGE_FIRST)
    _REGISTERED = True


class _LanguageModel:
    """The text forward of ``Model`` with the mlx-vlm call signature.

    It is not a module, so the model tree holds the text layers once. A block
    drafter attached to this object is passed to the forward explicitly.
    """

    def __init__(self, model: Model):
        self._model = model

    def __call__(self, inputs, cache=None, inputs_embeds=None, **kwargs):
        drafter = self.__dict__.get("_omlx_drafter")
        return self._model.forward(
            inputs, cache, inputs_embeds, drafter=drafter, **kwargs
        )

    @property
    def args(self):
        return self._model.args.text

    def __getattr__(self, name):
        return getattr(self._model, name)


class Model(inkling2.Model):
    @property
    def language_model(self) -> _LanguageModel:
        return _LanguageModel(self)

    def encode_image(self, pixel_values, image_grid_thw=None, **kwargs):
        return super().encode_image(pixel_values, image_grid_thw)

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        for name in _UNSUPPORTED_INPUTS:
            if kwargs.get(name) is not None:
                raise ValueError(f"Inkling 2 does not support {name} in oMLX.")
        embeds = self.embed_inputs(
            input_ids,
            pixel_values,
            kwargs.get("image_grid_thw"),
            features=kwargs.get("cached_image_features"),
        )
        return InputEmbeddingsFeatures(inputs_embeds=embeds)


class LoopModel(Model):
    def _text_model(self, args, quant):
        plan = LoopPlan.from_config(self.args.text_config["loop_config"], args)
        return LoopTextModel(args, quant, plan)

    @property
    def _cache_count(self):
        return self.model.cache_count


def _model_classes(config):
    cls = LoopModel if config.get("model_type") == LOOP_MODEL_TYPE else Model
    return cls, inkling2.ModelArgs


def load(model_path: str | Path, *, lazy: bool = False):
    """Load an Inkling 2 checkpoint with mlx-lm and its processor."""
    apply_inkling2_patch()
    _register_mlx_vlm()
    path = get_model_path(str(model_path))
    with (path / "config.json").open() as config_file:
        is_loop = json.load(config_file).get("model_type") == LOOP_MODEL_TYPE
    model, config = load_model(
        path,
        lazy=lazy,
        get_model_classes=_model_classes,
        # Mini's sanitizer installs the mixed expert carriers directly.
        # Its global ModelOpt declaration describes only one of those formats.
        model_config={"quantization_config": None} if is_loop else None,
    )
    model.config = SimpleNamespace(
        model_type=model.model_type,
        image_token_id=model.args.image_token_id,
        eos_token_id=config.get("eos_token_id"),
        vocab_size=model.args.text.vocab_size,
        text_config=model.args.text,
        vision_config=model.args.vision,
    )
    processor = load_processor(path, True, eos_token_ids=model.config.eos_token_id)
    return model, processor
