# SPDX-License-Identifier: Apache-2.0
"""Helpers for runtime dependencies with optional feature surfaces."""

from __future__ import annotations

from importlib.util import find_spec
from shutil import which

MLX_VLM_PACKAGE = "mlx_vlm"
MLX_VLM_MISSING_MESSAGE = (
    "mlx-vlm image support is unavailable or incompatible. Reinstall oMLX to "
    "restore its required mlx-vlm dependency."
)
MLX_VIDEO_MISSING_MESSAGE = (
    "Video generation requires mlx-video. Install oMLX with the video extra."
)


def require_video_runtime() -> None:
    """Raise an installation hint when video generation prerequisites are absent."""
    if find_spec("mlx_video") is None:
        raise ImportError(MLX_VIDEO_MISSING_MESSAGE)
    for tool in ("ffmpeg", "ffprobe"):
        if which(tool) is None:
            raise RuntimeError(f"Video generation requires {tool} on PATH")


def is_mlx_vlm_available() -> bool:
    """Return whether the required mlx-vlm package can be imported."""
    return find_spec(MLX_VLM_PACKAGE) is not None


def require_mlx_vlm_available() -> None:
    """Raise a clear error when the core mlx-vlm dependency is unavailable."""
    if not is_mlx_vlm_available():
        raise ImportError(MLX_VLM_MISSING_MESSAGE)
