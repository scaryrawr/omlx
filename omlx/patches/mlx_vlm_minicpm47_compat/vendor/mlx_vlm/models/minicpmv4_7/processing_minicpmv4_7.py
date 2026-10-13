# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Torch-free MiniCPM-V 4.7 processing.

Slicing, patch packing and placeholders follow Transformers 5.18.0, revision
a906d3c4b65095f2308b6a6a193e934d03b8eb5d (the inherited 4.6 PIL/video
processors and the native 4.7 processor). Controls are resolved per call.
Advanced frame stacking and audio are intentionally unsupported.
"""

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AddedToken, Qwen2TokenizerFast
from transformers.image_processing_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

from ..base import load_chat_template
from .config import ModelConfig


def _positive_int(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a strictly positive integer.")
    return int(value)


@dataclass(frozen=True)
class _Options:
    max_slice_nums: int = 9
    scale_resolution: int = 448
    patch_size: int = 14
    slice_mode: bool = True
    use_image_id: bool = True
    downsample_mode: str = "16x"
    image_mean: tuple = (0.5, 0.5, 0.5)
    image_std: tuple = (0.5, 0.5, 0.5)

    def __post_init__(self):
        for name in ("max_slice_nums", "scale_resolution", "patch_size"):
            _positive_int(name, getattr(self, name))
        if self.downsample_mode not in ("4x", "16x"):
            raise ValueError("downsample_mode must be '4x' or '16x'.")
        for name in ("slice_mode", "use_image_id"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        for name in ("image_mean", "image_std"):
            value = tuple(np.broadcast_to(getattr(self, name), (3,)).tolist())
            if not np.isfinite(value).all() or (
                name == "image_std" and any(v <= 0 for v in value)
            ):
                raise ValueError(
                    f"{name} must contain three finite values; std must be positive."
                )
            object.__setattr__(self, name, value)

    @property
    def divisor(self):
        return 4 if self.downsample_mode == "4x" else 16


def _read_json(directory, filename):
    path = Path(directory) / filename
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _options(defaults, overrides):
    values = dict(overrides)
    if "stack_frames" in values:
        raise ValueError("stack_frames is not supported by native MiniCPM-V 4.7.")
    if values.pop("pack_image_by_patch", True) is not True:
        raise ValueError("Native MiniCPM-V 4.7 requires patch-packed pixels.")
    return replace(defaults, **values)


def _token_config(value):
    if isinstance(value, dict):
        return AddedToken(
            **{
                key: item
                for key, item in value.items()
                if key
                in (
                    "content",
                    "single_word",
                    "lstrip",
                    "rstrip",
                    "normalized",
                    "special",
                )
            }
        )
    if isinstance(value, list):
        return [_token_config(item) for item in value]
    return value


def _pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if (
        array.ndim == 3
        and array.shape[0] in (1, 3, 4)
        and array.shape[-1] not in (1, 3, 4)
    ):
        array = array.transpose(1, 2, 0)
    if array.ndim not in (2, 3) or (
        array.ndim == 3 and array.shape[-1] not in (1, 3, 4)
    ):
        raise ValueError("Images must be PIL images or HWC/CHW numpy arrays.")
    if not np.isfinite(array).all():
        raise ValueError("Images must contain finite pixels.")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array).convert("RGB")


def _divide(length, divisor):
    return max(round(length / divisor) * divisor, divisor)


def _resize(size, options, upscale=False):
    width, height = size
    if width * height > options.scale_resolution**2 or upscale:
        ratio = width / height
        height = int(options.scale_resolution / math.sqrt(ratio))
        width = int(height * ratio)
    return tuple(_divide(length, options.patch_size * 4) for length in (width, height))


def _grid(size, options):
    width, height = size
    multiple = min(
        math.ceil(width * height / options.scale_resolution**2), options.max_slice_nums
    )
    if multiple <= 1 or not options.slice_mode:
        return (0, 0)
    best, min_error = (1, 1), float("inf")
    log_ratio = math.log(width / height)
    for count in (multiple - 1, multiple, multiple + 1):
        if count == 1 or count > options.max_slice_nums:
            continue
        for rows in range(1, count + 1):
            if count % rows:
                continue
            cols = count // rows
            error = abs(log_ratio - math.log(cols / rows))
            if error < min_error or (error == min_error and rows > best[0]):
                best, min_error = (rows, cols), error
    return best


def _crops(image, options):
    image = _pil(image)
    rows, cols = grid = _grid(image.size, options)
    crops = [
        image.resize(_resize(image.size, options, not rows), Image.Resampling.BICUBIC)
    ]
    if rows:
        width, height = image.size
        cell_size = (_divide(width, cols) / cols, _divide(height, rows) / rows)
        cell_w, cell_h = _resize(cell_size, options, True)
        refined = image.resize((cell_w * cols, cell_h * rows), Image.Resampling.BICUBIC)
        crops.extend(
            refined.crop(
                (col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h)
            )
            for row in range(rows)
            for col in range(cols)
        )
    return crops, grid


def _pack(crops, options):
    pixels, sizes = [], []
    patch = options.patch_size
    for image in crops:
        array = np.asarray(image, dtype=np.float32) / np.float32(255)
        array = (array - np.asarray(options.image_mean, dtype=np.float32)) / np.asarray(
            options.image_std, dtype=np.float32
        )
        array = array.transpose(2, 0, 1)
        channels, height, width = array.shape
        sizes.append((height // patch, width // patch))
        pixels.append(
            array.reshape(channels, height // patch, patch, width // patch, patch)
            .transpose(0, 2, 1, 3, 4)
            .reshape(channels, patch, -1)
        )
    return pixels, sizes


def _packed_output(pixels, sizes, options, suffix=""):
    return {
        "pixel_values"
        + suffix: (
            np.concatenate(pixels, axis=-1)[None]
            if pixels
            else np.empty((1, 3, options.patch_size, 0), dtype=np.float32)
        ),
        "target_sizes" + suffix: np.asarray(sizes, dtype=np.int32).reshape(-1, 2),
    }


class MiniCPMV4_7ImageProcessor:
    model_input_names = ["pixel_values", "target_sizes"]

    def __init__(self, **kwargs):
        kwargs = dict(kwargs)
        if "stack_frames" in kwargs:
            raise ValueError("stack_frames is not supported by native MiniCPM-V 4.7.")
        if kwargs.pop("pack_image_by_patch", True) is not True:
            raise ValueError("Native MiniCPM-V 4.7 requires patch-packed pixels.")
        for name in (
            "processor_class",
            "image_processor_type",
            "video_processor_type",
            "version",
        ):
            kwargs.pop(name, None)
        # Checkpoint metadata may contain framework bookkeeping, not controls.
        values = {
            name: kwargs[name]
            for name in _Options.__dataclass_fields__
            if name in kwargs
        }
        self.options = _Options(**values)

    def __getattr__(self, name):
        if name in _Options.__dataclass_fields__:
            return getattr(self.options, name)
        raise AttributeError(name)

    @classmethod
    def from_pretrained(cls, directory, **kwargs):
        config = _read_json(directory, "preprocessor_config.json")
        config.update(kwargs)
        return cls(**config)

    def preprocess(self, images, **kwargs):
        kwargs.pop("return_tensors", None)
        options = _options(self.options, kwargs)
        images = _flatten_images(images)
        pixels, sizes, grids, counts = [], [], [], []
        for image in images:
            crops, grid = _crops(image, options)
            pv, ts = _pack(crops, options)
            pixels.extend(pv)
            sizes.extend(ts)
            grids.append(grid)
            counts.append(len(crops))
        return BatchFeature(
            {
                **_packed_output(pixels, sizes, options),
                "grids": np.asarray(grids, dtype=np.int32).reshape(-1, 2),
                "num_patches_per_image": counts,
            }
        )

    __call__ = preprocess


def _is_frame(value):
    return (
        isinstance(value, Image.Image)
        or isinstance(value, np.ndarray)
        and value.ndim in (2, 3)
    )


def _flatten_images(images):
    if images is None:
        return []
    if _is_frame(images):
        return [images]
    if isinstance(images, np.ndarray) and images.ndim == 4:
        return list(images)
    if isinstance(images, (list, tuple)):
        return [image for item in images for image in _flatten_images(item)]
    raise ValueError("Images must be PIL images or numpy arrays.")


def _flatten_videos(videos):
    if videos is None:
        return []
    if isinstance(videos, dict) or hasattr(videos, "frames"):
        return [videos]
    if isinstance(videos, np.ndarray) and videos.ndim == 4:
        return [videos]
    if isinstance(videos, (list, tuple)):
        if videos and all(_is_frame(frame) for frame in videos):
            return [videos]
        return [video for item in videos for video in _flatten_videos(item)]
    raise ValueError("Videos must be decoded frame sequences or numpy arrays.")


class MiniCPMV4_7VideoProcessor(MiniCPMV4_7ImageProcessor):
    model_input_names = ["pixel_values_videos", "target_sizes_videos"]

    def __init__(self, max_num_frames=128, **kwargs):
        if "stack_frames" in kwargs and kwargs.pop("stack_frames") != 1:
            raise ValueError("stack_frames is not supported by native MiniCPM-V 4.7.")
        self.max_num_frames = _positive_int("max_num_frames", max_num_frames)
        super().__init__(**kwargs)

    def video_sampling_defaults(self):
        return {"max_frames": self.max_num_frames}

    @classmethod
    def from_pretrained(cls, directory, **kwargs):
        config = _read_json(directory, "video_preprocessor_config.json")
        if not config:
            config = _read_json(directory, "preprocessor_config.json")
        config.update(kwargs)
        return cls(**config)

    def preprocess(self, videos, **kwargs):
        kwargs.pop("return_tensors", None)
        metadata = kwargs.pop("video_metadata", None)
        kwargs.pop("fps", None)  # Frames were decoded/sampled by the existing loader.
        sample = kwargs.pop("do_sample_frames", False)
        if sample:
            raise ValueError(
                "Sample videos with the existing video loader before processing."
            )
        maximum = _positive_int(
            "max_num_frames", kwargs.pop("max_num_frames", self.max_num_frames)
        )
        options = _options(self.options, kwargs)
        videos = _flatten_videos(videos)
        if metadata is not None and len(metadata) != len(videos):
            raise ValueError("Expected one video_metadata entry per video.")
        pixels, sizes, grids, counts, frames_per_video, timestamps = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for index, video in enumerate(videos):
            times = None
            if isinstance(video, dict):
                times = video.get("timestamps")
                video = video.get("frames")
            elif hasattr(video, "frames"):
                times = getattr(video, "timestamps", None)
                video = video.frames
            frames = list(video)
            if not frames or not all(_is_frame(frame) for frame in frames):
                raise ValueError(
                    "A video must contain at least one decoded image frame."
                )
            if times is None and metadata is not None:
                item = metadata[index]
                times = (
                    item.get("timestamps")
                    if isinstance(item, dict)
                    else getattr(item, "timestamps", None)
                )
                if times is None:
                    indices = (
                        item.get("frames_indices")
                        if isinstance(item, dict)
                        else getattr(item, "frames_indices", None)
                    )
                    fps = (
                        item.get("fps")
                        if isinstance(item, dict)
                        else getattr(item, "fps", None)
                    )
                    if indices is not None and fps:
                        times = [frame / fps for frame in indices]
            if times is not None:
                if len(times) != len(frames) or not np.isfinite(times).all():
                    raise ValueError(
                        "Video timestamps must be finite and match the frames."
                    )
            selected = np.linspace(
                0, len(frames) - 1, min(len(frames), maximum), dtype=int
            )
            frames_per_video.append(len(selected))
            timestamps.append(
                None if times is None else [float(times[i]) for i in selected]
            )
            for frame_index in selected:
                crops, grid = _crops(frames[frame_index], options)
                pv, ts = _pack(crops, options)
                pixels.extend(pv)
                sizes.extend(ts)
                grids.append(grid)
                counts.append(len(crops))
        return BatchFeature(
            {
                **_packed_output(pixels, sizes, options, "_videos"),
                "grids_videos": np.asarray(grids, dtype=np.int32).reshape(-1, 2),
                "num_patches_per_frame": counts,
                "num_frames_per_video": frames_per_video,
                "frame_timestamps": timestamps,
            }
        )

    __call__ = preprocess


class MiniCPMV4_7Processor(ProcessorMixin):
    """Native HF-style processor with immutable per-request media controls."""

    attributes = ["image_processor", "video_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "AutoTokenizer"
    _markers = re.compile(
        r"<image>\./</image>|<\|image_pad\|>|<image>|<\|video_pad\|>|<video>"
    )
    _image_markers = re.compile(r"<image>\./</image>|<\|image_pad\|>|<image>")
    _video_markers = re.compile(r"<\|video_pad\|>|<video>")
    _control_names = set(_Options.__dataclass_fields__) | {
        "pack_image_by_patch",
        "stack_frames",
    }

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        if tokenizer is None:
            raise ValueError("MiniCPM-V 4.7 requires a tokenizer.")
        self.image_processor = image_processor or MiniCPMV4_7ImageProcessor()
        self.video_processor = video_processor or MiniCPMV4_7VideoProcessor(
            **self.image_processor.options.__dict__
        )
        self.tokenizer = tokenizer
        self.additional_eos_token_ids = ()
        self.chat_template = (
            chat_template
            if chat_template is not None
            else getattr(tokenizer, "chat_template", None)
        )
        for name, token in {
            "image_token": "<|image_pad|>",
            "video_token": "<|video_pad|>",
            "image_start_token": "<image>",
            "image_end_token": "</image>",
            "slice_start_token": "<slice>",
            "slice_end_token": "</slice>",
            "image_id_start_token": "<image_id>",
            "image_id_end_token": "</image_id>",
        }.items():
            setattr(self, name, token)
            setattr(self, name + "_id", tokenizer.convert_tokens_to_ids(token))
        for name in (
            "image_token",
            "video_token",
            "image_start_token",
            "image_end_token",
            "slice_start_token",
            "slice_end_token",
        ):
            token_id = getattr(self, name + "_id")
            if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
                raise ValueError(
                    f"Tokenizer is missing MiniCPM-V token {getattr(self, name)}."
                )

    @property
    def model_input_names(self):
        return [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "target_sizes",
            "pixel_values_videos",
            "target_sizes_videos",
            "mm_token_type_ids",
            "downsample_mode",
            "image_max_slice_nums",
            "video_max_slice_nums",
        ]

    def _replacement(self, sizes, grid, modality, options):
        token = self.image_token if modality == "image" else self.video_token
        counts = np.prod(sizes, axis=-1) // options.divisor
        replacement = (
            self.image_start_token + token * int(counts[0]) + self.image_end_token
        )
        rows, cols = grid
        if rows:
            slices = [
                self.slice_start_token + token * int(count) + self.slice_end_token
                for count in counts[1:]
            ]
            replacement += "\n".join(
                "".join(slices[row * cols : (row + 1) * cols]) for row in range(rows)
            )
        return replacement

    def replace_image_token(self, image_inputs, image_idx, **kwargs):
        options = _options(self.image_processor.options, kwargs)
        counts = image_inputs["num_patches_per_image"]
        start = sum(counts[:image_idx])
        return self._replacement(
            image_inputs["target_sizes"][start : start + counts[image_idx]],
            image_inputs["grids"][image_idx],
            "image",
            options,
        )

    def replace_video_token(self, video_inputs, video_idx, **kwargs):
        options = _options(self.video_processor.options, kwargs)
        frame_start = sum(video_inputs["num_frames_per_video"][:video_idx])
        count = video_inputs["num_frames_per_video"][video_idx]
        counts = video_inputs["num_patches_per_frame"]
        parts = []
        for frame in range(frame_start, frame_start + count):
            start = sum(counts[:frame])
            parts.append(
                self._replacement(
                    video_inputs["target_sizes_videos"][start : start + counts[frame]],
                    video_inputs["grids_videos"][frame],
                    "video",
                    options,
                )
            )
        return "".join(parts)

    def _validate_spans(self, rows):
        starts = {
            self.image_start_token_id: self.image_end_token_id,
            self.slice_start_token_id: self.slice_end_token_id,
        }
        ends = set(starts.values())
        for row in rows:
            closing, modality, length = None, None, 0
            for token in row:
                if token in starts:
                    if closing is not None:
                        raise ValueError("Nested visual spans are not supported.")
                    closing, modality, length = starts[token], None, 0
                elif token in ends:
                    if token != closing or not length:
                        raise ValueError("Visual spans must be complete and nonempty.")
                    closing = None
                elif token in (self.image_token_id, self.video_token_id):
                    if closing is None or modality not in (None, token):
                        raise ValueError(
                            "Visual tokens must belong to a single-modality span."
                        )
                    modality, length = token, length + 1
                elif closing is not None:
                    raise ValueError(
                        "Visual spans may contain only modality pad tokens."
                    )
            if closing is not None:
                raise ValueError("Visual spans must have a closing marker.")

    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        *,
        downsample_mode=None,
        max_slice_nums=None,
        slice_mode=None,
        use_image_id=None,
        pack_image_by_patch=None,
        max_num_frames=None,
        stack_frames=None,
        options=None,
        images_kwargs=None,
        videos_kwargs=None,
        text_kwargs=None,
        common_kwargs=None,
        video_metadata=None,
        fps=None,
        do_sample_frames=None,
        audio=None,
        audios=None,
        padding=None,
        padding_side=None,
        add_special_tokens=None,
        return_tensors=None,
        max_length=None,
        truncation=None,
        **kwargs,
    ):
        # The existing input adapter forwards explicitly named arguments only.
        explicit = {
            "downsample_mode": downsample_mode,
            "max_slice_nums": max_slice_nums,
            "slice_mode": slice_mode,
            "use_image_id": use_image_id,
            "pack_image_by_patch": pack_image_by_patch,
            "max_num_frames": max_num_frames,
            "stack_frames": stack_frames,
            "options": options,
            "images_kwargs": images_kwargs,
            "videos_kwargs": videos_kwargs,
            "text_kwargs": text_kwargs,
            "common_kwargs": common_kwargs,
            "video_metadata": video_metadata,
            "fps": fps,
            "do_sample_frames": do_sample_frames,
            "audio": audio,
            "audios": audios,
            "padding": padding,
            "padding_side": padding_side,
            "add_special_tokens": add_special_tokens,
            "return_tensors": return_tensors,
            "max_length": max_length,
            "truncation": truncation,
        }
        kwargs = {
            **kwargs,
            **{key: value for key, value in explicit.items() if value is not None},
        }
        if text is None:
            raise ValueError("You have to specify text input to process.")
        texts = [text] if isinstance(text, str) else list(text)
        if not texts or not all(isinstance(prompt, str) for prompt in texts):
            raise ValueError("text must contain one or more strings.")
        kwargs = dict(kwargs)
        if (
            kwargs.pop("audio", None) is not None
            or kwargs.pop("audios", None) is not None
        ):
            raise ValueError("MiniCPM-V 4.7 does not support audio.")
        common = dict(kwargs.pop("common_kwargs", {}))
        text_kwargs = dict(kwargs.pop("text_kwargs", {}))
        image_kwargs = dict(kwargs.pop("images_kwargs", {}))
        video_kwargs = dict(kwargs.pop("videos_kwargs", {}))
        global_options = dict(kwargs.pop("options", {}))
        unknown_options = set(global_options) - self._control_names - {"max_num_frames"}
        if unknown_options:
            raise ValueError(f"Unknown MiniCPM-V options: {sorted(unknown_options)}.")
        kwargs = {**global_options, **kwargs}
        for name in self._control_names | {"max_num_frames"}:
            if name in kwargs:
                value = kwargs.pop(name)
                if name != "max_num_frames":
                    image_kwargs.setdefault(name, value)
                if name != "use_image_id":
                    video_kwargs.setdefault(name, value)
        for name in ("fps", "video_metadata", "do_sample_frames"):
            if name in kwargs:
                video_kwargs.setdefault(name, kwargs.pop(name))
        text_kwargs = {**common, **text_kwargs, **kwargs}
        tensors = text_kwargs.pop("return_tensors", None)
        if tensors not in (None, "np", "numpy", "mlx"):
            raise ValueError(
                "Native MiniCPM-V 4.7 returns numpy arrays, not torch tensors."
            )
        image_kwargs.pop("return_tensors", None)
        video_kwargs.pop("return_tensors", None)
        video_kwargs.pop("use_image_id", None)
        image_options = _options(self.image_processor.options, image_kwargs)
        video_options = _options(
            self.video_processor.options,
            {
                k: v
                for k, v in video_kwargs.items()
                if k
                not in ("max_num_frames", "fps", "video_metadata", "do_sample_frames")
            },
        )
        flat_images, flat_videos = _flatten_images(images), _flatten_videos(videos)
        if "max_num_frames" in video_kwargs:
            _positive_int("max_num_frames", video_kwargs["max_num_frames"])
        if (
            flat_images
            and flat_videos
            and image_options.downsample_mode != video_options.downsample_mode
        ):
            raise ValueError("Images and videos must use the same downsample_mode.")
        effective_mode = (
            video_options.downsample_mode
            if flat_videos and not flat_images
            else image_options.downsample_mode
        )
        for modality, pattern, values in (
            ("image", self._image_markers, flat_images),
            ("video", self._video_markers, flat_videos),
        ):
            if sum(len(pattern.findall(prompt)) for prompt in texts) != len(values):
                raise ValueError(
                    f"Number of {modality} placeholders does not match {modality} inputs."
                )
        # Preserve nested batch ownership, not just the total input count.
        for values, pattern, modality in (
            (images, self._image_markers, "image"),
            (videos, self._video_markers, "video"),
        ):
            if (
                isinstance(values, (list, tuple))
                and len(values) == len(texts)
                and len(texts) > 1
            ):
                flatten = _flatten_images if modality == "image" else _flatten_videos
                if any(
                    len(flatten(value)) != len(pattern.findall(prompt))
                    for value, prompt in zip(values, texts)
                ):
                    raise ValueError(
                        f"Per-sample {modality} placeholders do not match inputs."
                    )
        image_inputs = (
            self.image_processor(flat_images, **image_kwargs) if flat_images else {}
        )
        video_inputs = (
            self.video_processor(flat_videos, **video_kwargs) if flat_videos else {}
        )
        rendered, image_index, video_index = [], 0, 0
        for prompt in texts:
            local_id = 0

            def substitute(match):
                nonlocal image_index, video_index, local_id
                if self._image_markers.fullmatch(match.group()):
                    replacement = self.replace_image_token(
                        image_inputs, image_index, **image_kwargs
                    )
                    image_index += 1
                    if image_options.use_image_id:
                        replacement = (
                            f"{self.image_id_start_token}{local_id}{self.image_id_end_token}"
                            + replacement
                        )
                    local_id += 1
                    return replacement
                replacement = self.replace_video_token(
                    video_inputs,
                    video_index,
                    **{
                        k: v
                        for k, v in video_kwargs.items()
                        if k
                        not in (
                            "max_num_frames",
                            "fps",
                            "video_metadata",
                            "do_sample_frames",
                        )
                    },
                )
                video_index += 1
                return replacement

            rendered.append(self._markers.sub(substitute, prompt))
        padding = text_kwargs.pop("padding", True)
        side = text_kwargs.pop("padding_side", "left")
        if side not in ("left", "right"):
            raise ValueError("padding_side must be 'left' or 'right'.")
        maximum = text_kwargs.pop("max_length", None)
        truncation = text_kwargs.pop("truncation", False)
        return_mm = text_kwargs.pop("return_mm_token_type_ids", True)
        if text_kwargs.pop("return_text_replacement_offsets", False):
            raise ValueError(
                "Text replacement offsets are not supported by the native processor."
            )
        text_kwargs.setdefault("add_special_tokens", False)
        rows = [
            list(self.tokenizer.encode(prompt, **text_kwargs)) for prompt in rendered
        ]
        self._validate_spans(rows)
        if maximum is not None:
            maximum = _positive_int("max_length", maximum)
        if truncation and maximum is not None:
            for before in rows:
                if any(
                    token in (self.image_token_id, self.video_token_id)
                    for token in before[maximum:]
                ):
                    raise ValueError("Truncation would remove visual tokens.")
                # Closing markers are part of the complete visual span.
                if any(
                    token in (self.image_start_token_id, self.slice_start_token_id)
                    for token in before[:maximum]
                ):
                    cut = before[:maximum]
                    if cut.count(self.image_start_token_id) != cut.count(
                        self.image_end_token_id
                    ) or cut.count(self.slice_start_token_id) != cut.count(
                        self.slice_end_token_id
                    ):
                        raise ValueError("Truncation would split a visual span.")
            rows = [row[:maximum] for row in rows]
        if not padding and len({len(row) for row in rows}) > 1:
            raise ValueError("Unequal batches require padding.")
        width = max(map(len, rows))
        if padding == "max_length":
            if maximum is None or width > maximum:
                raise ValueError(
                    "padding='max_length' requires max_length at least the encoded length."
                )
            width = maximum
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is None:
            raise ValueError("Tokenizer needs a pad_token_id or eos_token_id.")
        ids = np.full((len(rows), width), pad_id, dtype=np.int32)
        mask = np.zeros_like(ids)
        for index, row in enumerate(rows):
            offset = width - len(row) if side == "left" else 0
            ids[index, offset : offset + len(row)] = row
            mask[index, offset : offset + len(row)] = 1
        for modality, token_id, inputs, key, options in (
            ("image", self.image_token_id, image_inputs, "target_sizes", image_options),
            (
                "video",
                self.video_token_id,
                video_inputs,
                "target_sizes_videos",
                video_options,
            ),
        ):
            expected = (
                int((inputs[key].prod(-1) // options.divisor).sum()) if inputs else 0
            )
            if int(((ids == token_id) & (mask == 1)).sum()) != expected:
                raise ValueError(
                    f"Encoded {modality} tokens do not match visual features."
                )
        data = {
            "input_ids": ids,
            "attention_mask": mask,
            "downsample_mode": effective_mode,
            "image_max_slice_nums": image_options.max_slice_nums,
            "video_max_slice_nums": video_options.max_slice_nums,
        }
        for inputs in (image_inputs, video_inputs):
            data.update(
                {
                    key: value
                    for key, value in inputs.items()
                    if key.startswith(("pixel_values", "target_sizes"))
                }
            )
        if return_mm:
            data["mm_token_type_ids"] = np.where(
                (ids == self.image_token_id) & mask,
                1,
                np.where((ids == self.video_token_id) & mask, 2, 0),
            ).astype(np.int32)
        return BatchFeature(data)

    def apply_chat_template(self, *args, **kwargs):
        kwargs = dict(kwargs)
        if self.chat_template is not None:
            kwargs.setdefault("chat_template", self.chat_template)
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, **kwargs
    ):
        return [
            text.strip()
            for text in self.batch_decode(
                generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs
            )
        ]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        directory = Path(pretrained_model_name_or_path)
        if not directory.is_dir():
            raise ValueError(
                "Load MiniCPM-V 4.7 from a local model directory; remote code is never executed."
            )
        kwargs = dict(kwargs)
        chat_template = kwargs.pop("chat_template", None)
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("use_fast", None)
        processor_config = _read_json(directory, "processor_config.json")
        image_config = _read_json(directory, "preprocessor_config.json")
        image_config.update(processor_config.get("image_processor") or {})
        video_config = dict(image_config)
        video_config.update(_read_json(directory, "video_preprocessor_config.json"))
        video_config.update(processor_config.get("video_processor") or {})
        config = _read_json(directory, "config.json")
        default_mode = config.get("downsample_mode", "16x")
        image_config.setdefault("downsample_mode", default_mode)
        video_config.setdefault("downsample_mode", default_mode)
        controls = {
            key: kwargs.pop(key) for key in list(kwargs) if key in cls._control_names
        }
        image_config.update(controls)
        video_config.update(
            {key: value for key, value in controls.items() if key != "use_image_id"}
        )
        tokenizer_config = _read_json(directory, "tokenizer_config.json")
        token_kwargs = {
            name: _token_config(tokenizer_config[name])
            for name in (
                "bos_token",
                "eos_token",
                "unk_token",
                "pad_token",
                "additional_special_tokens",
            )
            if name in tokenizer_config
        }
        token_kwargs.update(
            {
                name: _token_config(value)
                for name, value in _read_json(
                    directory, "special_tokens_map.json"
                ).items()
            }
        )
        # Legacy lists are incompatible with HF's mapping API.
        extra_tokens = tokenizer_config.get("extra_special_tokens", {})
        token_kwargs["extra_special_tokens"] = (
            {name: _token_config(value) for name, value in extra_tokens.items()}
            if isinstance(extra_tokens, dict)
            else {}
        )
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            directory,
            trust_remote_code=False,
            local_files_only=True,
            **token_kwargs,
        )
        load_chat_template(tokenizer, directory)
        processor = cls(
            MiniCPMV4_7ImageProcessor(**image_config),
            MiniCPMV4_7VideoProcessor(**video_config),
            tokenizer,
            chat_template,
        )
        eos = config.get(
            "eos_token_id", config.get("text_config", {}).get("eos_token_id")
        )
        if eos is not None:
            processor.additional_eos_token_ids = tuple(
                eos if isinstance(eos, list) else [eos]
            )
        vocab_size = config.get("text_config", {}).get("vocab_size")
        if vocab_size is not None:
            if len(tokenizer) != vocab_size:
                raise ValueError(
                    "MiniCPM-V 4.7 tokenizer vocabulary does not match target config."
                )
            for field, token_field in (
                ("image_token_id", "image_token_id"),
                ("video_token_id", "video_token_id"),
                ("image_start_id", "image_start_token_id"),
                ("image_end_id", "image_end_token_id"),
                ("slice_start_id", "slice_start_token_id"),
                ("slice_end_id", "slice_end_token_id"),
            ):
                expected = config.get(field, getattr(ModelConfig, field))
                if getattr(processor, token_field) != expected:
                    raise ValueError(
                        f"MiniCPM-V 4.7 tokenizer does not match target {field}."
                    )
        return processor


__all__ = [
    "MiniCPMV4_7Processor",
    "MiniCPMV4_7ImageProcessor",
    "MiniCPMV4_7VideoProcessor",
]
