# SPDX-License-Identifier: Apache-2.0
"""Isolated MLX video generation and exact MP4 publication."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import ParamSpec

import mlx.core as mx
from pydantic import BaseModel

try:
    from mlx_video.models.ltx_2.generate import PipelineType
    from mlx_video.models.ltx_2.generate import generate_video as generate_ltx2
    from mlx_video.models.wan_2 import vae22
    from mlx_video.models.wan_2.generate import generate_video as generate_wan
except ImportError:
    generate_wan = generate_ltx2 = PipelineType = None

from .utils.optional_deps import MLX_VIDEO_MISSING_MESSAGE
from .video_registry import VideoRenderPlan

P = ParamSpec("P")


def _reclaim_after_forward(forward: Callable[P, mx.array]) -> Callable[P, mx.array]:
    def run(*args: P.args, **kwargs: P.kwargs) -> mx.array:
        output = forward(*args, **kwargs)
        mx.eval(output)
        mx.synchronize()
        mx.clear_cache()
        return output

    return run


class _VideoStream(BaseModel):
    width: int
    height: int
    nb_frames: int
    avg_frame_rate: str
    duration: str


def finalize_video(plan: VideoRenderPlan, directory: Path) -> None:
    """Trim padded temporal frames and verify the published MP4."""
    output = directory / "video.tmp.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(directory / "generated.mp4"),
            "-vf",
            f"trim=end_frame={plan.output_frames},setpts=PTS-STARTPTS",
            "-r",
            str(plan.fps),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,avg_frame_rate,duration",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = _VideoStream.model_validate(json.loads(result.stdout)["streams"][0])
    actual = (
        stream.width,
        stream.height,
        stream.nb_frames,
        Fraction(stream.avg_frame_rate),
        Fraction(stream.duration),
    )
    expected = (
        plan.width,
        plan.height,
        plan.output_frames,
        Fraction(plan.fps),
        Fraction(plan.output_frames, plan.fps),
    )
    if actual != expected:
        raise RuntimeError(
            f"Encoded video does not match requested media: {actual} != {expected}"
        )
    with output.open("rb") as file:
        os.fsync(file.fileno())
    output.replace(directory / "video.mp4")


def _watch_parent() -> None:
    if not os.read(sys.stdin.fileno(), 1):
        os.killpg(os.getpgrp(), signal.SIGTERM)


def main() -> None:
    """Run the pinned backend without downloads or access to server credentials."""
    threading.Thread(target=_watch_parent, daemon=True).start()
    if generate_wan is None:
        raise ImportError(MLX_VIDEO_MISSING_MESSAGE)
    spec_path = Path(sys.argv[1])
    plan = VideoRenderPlan.model_validate_json(spec_path.read_text())
    directory = spec_path.parent
    reference = str(directory / "reference.png") if plan.has_reference else None
    output = str(directory / "generated.mp4")
    if plan.family == "wan":
        mx.set_cache_limit(int(mx.device_info()["memory_size"]))
        for layer in (vae22.ResidualBlock, vae22.Resample):
            layer.__call__ = _reclaim_after_forward(layer.__call__)
        generate_wan(
            model_dir=plan.model_path,
            prompt=plan.prompt,
            negative_prompt=plan.negative_prompt,
            image=reference,
            width=plan.width,
            height=plan.height,
            num_frames=plan.generation_frames,
            steps=plan.steps,
            guide_scale=plan.guidance,
            seed=plan.seed,
            output_path=output,
        )
    else:
        sampling = {
            key: value
            for key, value in {
                "num_inference_steps": plan.steps,
                "cfg_scale": plan.guidance,
                "negative_prompt": plan.negative_prompt,
            }.items()
            if value is not None
        }
        generate_ltx2(
            model_repo=plan.model_path,
            text_encoder_repo=None,
            prompt=plan.prompt,
            pipeline=PipelineType(plan.pipeline),
            image=reference,
            width=plan.width,
            height=plan.height,
            num_frames=plan.generation_frames,
            fps=plan.fps,
            seed=plan.seed,
            output_path=output,
            **sampling,
        )
    finalize_video(plan, directory)
    (directory / "generated.mp4").unlink()


if __name__ == "__main__":
    main()
