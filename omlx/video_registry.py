# SPDX-License-Identifier: Apache-2.0
"""Configuration and render plans for local MLX video checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

VideoFamily = Literal["wan", "ltx2"]
LtxPipeline = Literal["distilled", "dev"]
VideoTask = Literal["text-to-video", "image-to-video"]


class WanVideoConfig(BaseModel):
    """Converted mlx-video Wan checkpoint configuration."""

    model_type: Literal["t2v", "i2v", "ti2v"]
    model_version: Literal["2.1", "2.2"]
    dual_model: bool
    patch_size: tuple[int, int, int] = (1, 2, 2)
    vae_stride: tuple[int, int, int] = (4, 8, 8)
    sample_fps: int = Field(default=16, ge=1, le=120)
    max_area: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_geometry(self) -> WanVideoConfig:
        """Reject invalid strides before request planning."""
        if any(value <= 0 for value in (*self.patch_size, *self.vae_stride)):
            raise ValueError("Video checkpoint strides must be positive")
        if self.vae_stride[0] != 4:
            raise ValueError("Wan requires a temporal stride of 4")
        if self.model_type == "ti2v" and (
            self.vae_stride != (4, 16, 16) or self.sample_fps != 24
        ):
            raise ValueError("TI2V requires a converted Wan2.2 config")
        return self


class VideoModelSpec(BaseModel):
    """Family-neutral generation limits of a servable video checkpoint."""

    family: VideoFamily
    tasks: list[VideoTask]
    fps: int
    align_width: int
    align_height: int
    temporal_stride: int
    max_area: int = 0
    pipeline: LtxPipeline | None = None
    accepts_sampling: bool = True


class VideoRenderPlan(BaseModel):
    """Validated parameters passed to an isolated video worker."""

    family: VideoFamily
    pipeline: LtxPipeline | None = None
    model_path: str
    prompt: str
    width: int
    height: int
    fps: int
    output_frames: int
    generation_frames: int
    seed: int
    steps: int | None = None
    guidance: float | None = None
    negative_prompt: str | None = None
    has_reference: bool = False


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def detect_video_family(model_path: Path) -> VideoFamily | None:
    """Recognize video checkpoints without importing inference dependencies."""
    config = _read_json(model_path / "config.json")
    if (
        config is not None
        and config.get("model_type") in {"t2v", "i2v", "ti2v"}
        and "dim" in config
        and ("text_dim" in config or config.get("_class_name") == "WanModel")
    ):
        return "wan"
    transformer = _read_json(model_path / "transformer" / "config.json")
    if transformer is not None and str(transformer.get("model_type", "")).startswith(
        "ltx"
    ):
        return "ltx2"
    return None


def _require_files(model_path: Path, required: list[str], hint: str) -> None:
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise ValueError(f"{hint}; missing " + ", ".join(missing))


def _load_wan(model_path: Path) -> VideoModelSpec:
    hint = "Video checkpoint requires mlx-video conversion"
    _require_files(model_path, ["t5_encoder.safetensors", "vae.safetensors"], hint)
    config = WanVideoConfig.model_validate_json(
        (model_path / "config.json").read_text()
    )
    required = (
        ["low_noise_model.safetensors", "high_noise_model.safetensors"]
        if config.dual_model
        else ["model.safetensors"]
    )
    _require_files(model_path, required, hint)
    tasks: dict[str, list[VideoTask]] = {
        "t2v": ["text-to-video"],
        "i2v": ["image-to-video"],
        "ti2v": ["text-to-video", "image-to-video"],
    }
    return VideoModelSpec(
        family="wan",
        tasks=tasks[config.model_type],
        fps=config.sample_fps,
        align_width=config.patch_size[2] * config.vae_stride[2],
        align_height=config.patch_size[1] * config.vae_stride[1],
        temporal_stride=config.vae_stride[0],
        max_area=config.max_area,
    )


def _load_ltx2(model_path: Path) -> VideoModelSpec:
    # Distilled and dev LTX-2 checkpoints share one transformer config; the
    # converted repositories identify the variant only in their names.
    pipeline: LtxPipeline = (
        "distilled" if "distilled" in model_path.name.lower() else "dev"
    )
    required = [
        "transformer/config.json",
        "vae/decoder/config.json",
        "vae/encoder/config.json",
        "text_projections/model.safetensors",
        "text_encoder/config.json",
    ]
    _require_files(
        model_path,
        required,
        "LTX-2 checkpoint must be a converted mlx-video directory with its "
        "Gemma text encoder in text_encoder/",
    )
    if pipeline == "distilled" and not any(
        model_path.glob("*spatial-upscaler-x2*.safetensors")
    ):
        raise ValueError("Distilled LTX-2 checkpoint is missing its x2 spatial upscaler")
    alignment = 64 if pipeline == "distilled" else 32
    return VideoModelSpec(
        family="ltx2",
        tasks=["text-to-video", "image-to-video"],
        fps=24,
        align_width=alignment,
        align_height=alignment,
        temporal_stride=8,
        pipeline=pipeline,
        accepts_sampling=pipeline == "dev",
    )


_FAMILY_LOADERS = {"wan": _load_wan, "ltx2": _load_ltx2}


def load_video_spec(model_path: Path) -> VideoModelSpec:
    """Require a complete converted checkpoint of a supported video family."""
    family = detect_video_family(model_path)
    if family is None:
        raise ValueError(f"{model_path} is not a supported video checkpoint")
    return _FAMILY_LOADERS[family](model_path)


def plan_video(
    model_path: Path,
    *,
    prompt: str,
    size: str,
    seconds: str,
    seed: int,
    has_reference: bool,
    steps: int | None = None,
    guidance: float | None = None,
    negative_prompt: str | None = None,
) -> VideoRenderPlan:
    """Resolve and validate requested geometry against a video checkpoint."""
    spec = load_video_spec(model_path)
    width, height = (int(part) for part in size.split("x"))
    if width % spec.align_width or height % spec.align_height:
        raise ValueError(
            f"size must be divisible by {spec.align_width}x{spec.align_height}"
        )
    if spec.max_area and width * height > spec.max_area:
        raise ValueError(f"size exceeds checkpoint max_area {spec.max_area}")
    if has_reference and "image-to-video" not in spec.tasks:
        raise ValueError("This model does not support input_reference")
    if not has_reference and "text-to-video" not in spec.tasks:
        raise ValueError("This model requires input_reference")
    if not spec.accepts_sampling and (
        steps is not None or guidance is not None or negative_prompt is not None
    ):
        raise ValueError(
            "This model uses a fixed distilled schedule; steps, guidance, and "
            "negative_prompt are unsupported"
        )
    output_frames = int(seconds) * spec.fps
    stride = spec.temporal_stride
    generation_frames = ((output_frames - 1 + stride - 1) // stride) * stride + 1
    return VideoRenderPlan(
        family=spec.family,
        pipeline=spec.pipeline,
        model_path=str(model_path),
        prompt=prompt,
        width=width,
        height=height,
        fps=spec.fps,
        output_frames=output_frames,
        generation_frames=generation_frames,
        seed=seed,
        steps=steps,
        guidance=guidance,
        negative_prompt=negative_prompt,
        has_reference=has_reference,
    )
