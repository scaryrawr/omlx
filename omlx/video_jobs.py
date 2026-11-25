# SPDX-License-Identifier: Apache-2.0
"""Durable, single-owner local video jobs."""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import re
import secrets
import shutil
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from io import BufferedRandom
from pathlib import Path
from typing import Protocol

from fastapi import HTTPException
from PIL import Image, ImageOps
from pydantic import BaseModel

from .api.video_models import Video, VideoError, VideoGenerationRequest, VideoList
from .engine.video import VideoEngine
from .utils.optional_deps import require_mlx_video_available
from .video_registry import VideoRenderPlan, plan_video

logger = logging.getLogger(__name__)


class VideoModelEntry(Protocol):
    """Discovery fields needed to admit a video job."""

    @property
    def model_path(self) -> str: ...

    @property
    def engine_type(self) -> str: ...


class VideoPool(Protocol):
    """Existing pool operations used by the video dispatcher."""

    def get_entry(self, model_id: str) -> VideoModelEntry | None: ...

    def acquire(self, model_id: str) -> AbstractAsyncContextManager[object]: ...

    def _wake_process_memory_enforcer(self, *, active: bool = False) -> None: ...


class _StoredVideo(BaseModel):
    video: Video
    plan: VideoRenderPlan


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _normalize_reference(reference: Path, destination: Path) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with Image.open(reference) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        normalized.save(destination)
    with destination.open("rb") as file:
        os.fsync(file.fileno())


class VideoJobs:
    """Persist video jobs and execute them serially through the model pool."""

    def __init__(
        self,
        directory: Path,
        pool: VideoPool,
        resolve_model: Callable[[str], str | None],
    ) -> None:
        self.directory = directory
        self.pool = pool
        self.resolve_model = resolve_model
        self._jobs: dict[str, _StoredVideo] = {}
        self._pending: asyncio.Queue[_StoredVideo] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._accepting = False
        self._owner: BufferedRandom | None = None

    async def start(self) -> None:
        """Recover interrupted records before accepting new work."""
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._owner = (self.directory / ".owner").open("a+b")
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for path in sorted(self.directory.glob("video_*/job.json")):
                stored = _StoredVideo.model_validate_json(path.read_text())
                if (
                    not re.fullmatch(r"video_[0-9a-f]{32}", stored.video.id)
                    or stored.video.id != path.parent.name
                ):
                    raise ValueError(f"Invalid video record at {path}")
                if stored.video.status in {"queued", "in_progress"}:
                    stored.video.status = "failed"
                    stored.video.error = VideoError(
                        code="server_restarted",
                        message="Generation was interrupted by a server restart.",
                    )
                    self._save(stored)
                else:
                    self._jobs[stored.video.id] = stored
            self._accepting = True
            self._worker = asyncio.create_task(self._run(self._owner.fileno()))
        except BaseException:
            self._owner.close()
            self._owner = None
            raise

    def _save(self, stored: _StoredVideo) -> None:
        directory = self.directory / stored.video.id
        temporary = directory / "job.tmp"
        with temporary.open("w") as file:
            file.write(stored.model_dump_json())
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(directory / "job.json")
        _sync_directory(directory)
        _sync_directory(self.directory)
        self._jobs[stored.video.id] = stored.model_copy(deep=True)

    async def submit(
        self, request: VideoGenerationRequest, reference: Path | None
    ) -> Video:
        """Validate, persist, and enqueue a video before returning its ID."""
        if not self._accepting:
            raise HTTPException(503, "Video job service is stopping")
        model = self.resolve_model(request.model) or request.model
        entry = self.pool.get_entry(model)
        if entry is None:
            raise HTTPException(404, f"Model '{request.model}' not found")
        if entry.engine_type != "video":
            raise HTTPException(400, "The selected model is not a video model")
        try:
            require_mlx_video_available()
            plan = plan_video(
                Path(entry.model_path),
                prompt=request.prompt,
                size=request.size,
                seconds=request.seconds,
                seed=request.seed if request.seed is not None else secrets.randbits(32),
                has_reference=reference is not None,
                steps=request.steps,
                guidance=request.guidance,
                negative_prompt=request.negative_prompt,
            )
        except ImportError as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if self._pending.qsize() >= 32:
            raise HTTPException(429, "Video job queue is full")
        video = Video(
            id="video_" + uuid.uuid4().hex,
            model=model,
            prompt=request.prompt,
            seconds=request.seconds,
            size=request.size,
            created_at=int(time.time()),
        )
        directory = self.directory / video.id
        if reference is not None:
            normalization = asyncio.create_task(
                asyncio.to_thread(
                    _normalize_reference, reference, directory / "reference.png"
                )
            )
            cancelled = False
            try:
                while not normalization.done():
                    try:
                        await asyncio.shield(normalization)
                    except asyncio.CancelledError:
                        cancelled = True
                normalization.result()
                if cancelled:
                    raise asyncio.CancelledError
            except BaseException:
                if directory.exists():
                    shutil.rmtree(directory)
                raise
        if not self._accepting:
            if directory.exists():
                shutil.rmtree(directory)
            raise HTTPException(503, "Video job service is stopping")
        if self._pending.qsize() >= 32:
            if directory.exists():
                shutil.rmtree(directory)
            raise HTTPException(429, "Video job queue is full")
        directory.mkdir(mode=0o700, exist_ok=True)
        stored = _StoredVideo(video=video, plan=plan)
        try:
            self._save(stored)
        except Exception:
            self._jobs.pop(video.id, None)
            shutil.rmtree(directory)
            raise
        self._pending.put_nowait(stored)
        return video.model_copy(deep=True)

    async def _run(self, owner_fd: int) -> None:
        try:
            while True:
                stored = await self._pending.get()
                try:
                    stored.video.status = "in_progress"
                    self._save(stored)
                    async with self.pool.acquire(stored.video.model) as engine:
                        if not isinstance(engine, VideoEngine):
                            raise TypeError("Pool returned a non-video engine")
                        if Path(engine.model_name).resolve() != Path(
                            stored.plan.model_path
                        ).resolve():
                            raise RuntimeError(
                                "Video checkpoint changed after the job was queued"
                            )
                        self.pool._wake_process_memory_enforcer(active=True)
                        await engine.generate(
                            stored.plan,
                            self.directory / stored.video.id,
                            owner_fd,
                        )
                    stored.video.status = "completed"
                    stored.video.progress = 100
                    stored.video.completed_at = int(time.time())
                except asyncio.CancelledError:
                    stored.video.status = "failed"
                    stored.video.error = VideoError(
                        code="server_shutdown",
                        message="Generation stopped during shutdown.",
                    )
                    self._save(stored)
                    raise
                except Exception:
                    logger.exception("Video generation failed for %s", stored.video.id)
                    stored.video.status = "failed"
                    stored.video.error = VideoError(
                        code="generation_failed",
                        message="Video generation failed. See the server and job logs.",
                    )
                    if not self._accepting:
                        self._save(stored)
                        raise
                finally:
                    self._pending.task_done()
                self._save(stored)
        finally:
            self._accepting = False

    def get(self, video_id: str) -> Video:
        """Return a snapshot of a persisted video job."""
        stored = self._jobs.get(video_id)
        if stored is None:
            raise HTTPException(404, "Video not found")
        if stored.video.status in {"queued", "in_progress"} and not self._accepting:
            raise HTTPException(
                503, "Video dispatcher is unavailable. See server logs."
            )
        return stored.video.model_copy(deep=True)

    def list(self, *, limit: int, order: str, after: str | None) -> VideoList:
        """Return stable cursor pagination using creation time and ID."""
        videos = sorted(
            (stored.video for stored in self._jobs.values()),
            key=lambda video: (video.created_at, video.id),
            reverse=order == "desc",
        )
        if after is not None:
            self.get(after)
            index = next(i for i, video in enumerate(videos) if video.id == after)
            videos = videos[index + 1 :]
        page = videos[:limit]
        return VideoList(
            data=[video.model_copy(deep=True) for video in page],
            has_more=len(videos) > limit,
            first_id=page[0].id if page else None,
            last_id=page[-1].id if page else None,
        )

    def content_path(self, video_id: str) -> Path:
        """Return only completed MP4 content."""
        if self.get(video_id).status != "completed":
            raise HTTPException(409, "Video is not completed")
        path = self.directory / video_id / "video.mp4"
        if not path.is_file():
            raise HTTPException(410, "Video content is missing")
        return path

    def delete(self, video_id: str) -> None:
        """Delete a terminal job without invalidating open download handles."""
        if self.get(video_id).status not in {"completed", "failed"}:
            raise HTTPException(409, "Cannot delete a pending video")
        directory = self.directory / video_id
        (directory / "job.json").unlink()
        _sync_directory(directory)
        del self._jobs[video_id]
        shutil.rmtree(directory)
        _sync_directory(self.directory)

    async def close(self) -> None:
        """Stop workers before the pool and memory enforcer are shut down."""
        self._accepting = False
        if self._worker is not None:
            self._worker.cancel()
            result = (await asyncio.gather(self._worker, return_exceptions=True))[0]
            if isinstance(result, Exception):
                logger.error(
                    "Video dispatcher stopped with an error",
                    exc_info=(type(result), result, result.__traceback__),
                )
        try:
            while not self._pending.empty():
                stored = self._pending.get_nowait()
                stored.video.status = "failed"
                stored.video.error = VideoError(
                    code="server_shutdown",
                    message="Generation stopped during shutdown.",
                )
                self._save(stored)
                self._pending.task_done()
        finally:
            if self._owner is not None:
                self._owner.close()
                self._owner = None
