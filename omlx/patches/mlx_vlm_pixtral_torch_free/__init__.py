from __future__ import annotations

import functools
import logging
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_vlm.models.base import to_mlx
from mlx_vlm.models.pixtral import image_processing_pixtral
from mlx_vlm.models.pixtral.image_processing_pixtral import (
    split_image_sizes_by_sample,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import is_valid_image, load_image

logger = logging.getLogger(__name__)

_MARKER = "_omlx_pixtral_torch_free"


def _patch_image_preprocess() -> None:
    processor_class = image_processing_pixtral.PixtralImageProcessor
    if getattr(processor_class.preprocess, _MARKER, False):
        return

    original = processor_class.preprocess

    @functools.wraps(original)
    def patched_preprocess(self, images, **kwargs):
        if kwargs.get("return_tensors") != "mlx":
            return original(self, images, **kwargs)

        kwargs.pop("return_tensors")
        size = image_processing_pixtral._size_to_pair(kwargs.pop("size", self.size))
        patch_size = image_processing_pixtral._patch_size_to_pair(
            kwargs.pop("patch_size", self.patch_size)
        )
        do_resize = kwargs.pop("do_resize", self.do_resize)
        do_rescale = kwargs.pop("do_rescale", self.do_rescale)
        rescale_factor = kwargs.pop("rescale_factor", self.rescale_factor)
        do_normalize = kwargs.pop("do_normalize", self.do_normalize)
        image_mean = kwargs.pop("image_mean", self.image_mean)
        image_std = kwargs.pop("image_std", self.image_std)
        resample = kwargs.pop("resample", self.resample)
        mean = mx.array(image_mean, dtype=mx.float32)
        std = mx.array(image_std, dtype=mx.float32)

        pixel_values = []
        image_sizes = []
        for image in image_processing_pixtral._flatten_images(images):
            pil_image = image_processing_pixtral._to_pil_image(image)
            target_size = (pil_image.height, pil_image.width)
            if do_resize:
                target_size = image_processing_pixtral.get_resize_output_image_size(
                    target_size,
                    size=size,
                    patch_size=patch_size,
                )
                pil_image = pil_image.resize(
                    (target_size[1], target_size[0]),
                    resample=resample,
                )

            image_array = mx.array(np.asarray(pil_image)).astype(mx.float32)
            if do_rescale:
                image_array *= rescale_factor
            if do_normalize:
                image_array = (image_array - mean) / std
            pixel_values.append(mx.transpose(image_array, (2, 0, 1)))
            image_sizes.append(target_size)

        if not pixel_values:
            raise ValueError("You must provide at least one image.")

        max_height = max(image.shape[1] for image in pixel_values)
        max_width = max(image.shape[2] for image in pixel_values)
        padded_images = [
            mx.pad(
                image,
                (
                    (0, 0),
                    (0, max_height - image.shape[1]),
                    (0, max_width - image.shape[2]),
                ),
            )
            for image in pixel_values
        ]
        return BatchFeature(
            data={
                "pixel_values": mx.stack(padded_images),
                "image_sizes": image_sizes,
            }
        )

    setattr(patched_preprocess, _MARKER, True)
    processor_class.preprocess = patched_preprocess


def _patch_from_pretrained(processor_class: type) -> None:
    if getattr(processor_class, _MARKER, False):
        return

    original = processor_class.from_pretrained.__func__

    @functools.wraps(original)
    def patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs.setdefault("fix_mistral_regex", True)
        return original(cls, pretrained_model_name_or_path, **kwargs)

    processor_class.from_pretrained = classmethod(patched_from_pretrained)
    setattr(processor_class, _MARKER, True)


def _is_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("http")


def _is_image_or_url(value: Any) -> bool:
    return _is_url(value) or is_valid_image(value)


def _patch_pixtral_call(processor_class: type) -> None:
    if getattr(processor_class.__call__, _MARKER, False):
        return

    def patched_call(self, images=None, text=None, **kwargs) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("You must provide either text or images.")

        if isinstance(text, str):
            text = [text]
        elif text is not None and not isinstance(text, list):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        image_inputs = {}
        if images is not None:
            if _is_image_or_url(images):
                images = [[images]]
            elif isinstance(images, (list, tuple)) and _is_image_or_url(images[0]):
                images = [images]

            images = [
                [load_image(image) if _is_url(image) else image for image in sample]
                for sample in images
            ]
            image_inputs = self.image_processor(
                images,
                patch_size=self.patch_size * self.spatial_merge_size,
                return_tensors="mlx",
            )

            if text is not None:
                sizes_by_sample = split_image_sizes_by_sample(
                    image_inputs.get("image_sizes", []),
                    images,
                )
                prompts = []
                for batch_index, sample in enumerate(text):
                    if self.image_token not in sample:
                        prompts.append(sample)
                        continue

                    sample_sizes = (
                        sizes_by_sample[batch_index]
                        if batch_index < len(sizes_by_sample)
                        else []
                    )
                    parts = sample.split(self.image_token)
                    prompt = parts[0]
                    for image_index in range(len(parts) - 1):
                        if image_index < len(sample_sizes):
                            height, width = sample_sizes[image_index]
                            rows = height // (
                                self.patch_size * self.spatial_merge_size
                            )
                            columns = width // (
                                self.patch_size * self.spatial_merge_size
                            )
                            tokens = self.image_break_token.join(
                                self.image_token * columns for _ in range(rows)
                            )
                            prompt += tokens + self.image_end_token
                        else:
                            prompt += self.image_token
                        prompt += parts[image_index + 1]
                    prompts.append(prompt)
                text = prompts

        kwargs.pop("return_tensors", None)
        if text is not None:
            data = {**self.tokenizer(text, **kwargs), **image_inputs}
        else:
            data = image_inputs
        return BatchFeature(data=to_mlx(data))

    setattr(patched_call, _MARKER, True)
    processor_class.__call__ = patched_call


def apply_pixtral_torch_free_patch() -> bool:
    try:
        from mlx_vlm.models.mistral3 import processing_mistral3 as pin_m3
        from mlx_vlm.models.pixtral import processing_pixtral as pin_px

        _patch_image_preprocess()
        _patch_from_pretrained(pin_m3.Mistral3Processor)
        _patch_from_pretrained(pin_px.PixtralProcessor)
        _patch_pixtral_call(pin_px.PixtralProcessor)
    except Exception:
        logger.warning(
            "Pixtral compatibility patch not installed",
            exc_info=True,
        )
        return False

    logger.info("Pixtral MLX preprocessing and tokenizer compatibility installed")
    return True
