# SPDX-License-Identifier: Apache-2.0
"""Pool-managed isolated MLX video generation."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from contextlib import suppress
from pathlib import Path

from ..utils.optional_deps import require_video_runtime
from ..utils.proc_memory import get_phys_footprint, list_process_group_pids
from ..video_registry import VideoRenderPlan, load_video_spec
from .base import BaseNonStreamingEngine

_WORKER_STOP_TIMEOUT = 5.0
_WORKER_GROUP_POLL_INTERVAL = 0.05


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class VideoEngine(BaseNonStreamingEngine):
    """Supervise one-shot video workers under the model pool's lease."""

    def __init__(self, model_name: str, *, estimated_size: int = 0) -> None:
        super().__init__()
        self._model_name = model_name
        self._estimated_size = estimated_size
        self._process: asyncio.subprocess.Process | None = None
        self._call_lock = asyncio.Lock()
        self._termination_lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        """Return the local checkpoint path."""
        return self._model_name

    async def start(self) -> None:
        """Validate converted assets and runtime prerequisites."""
        require_video_runtime()
        load_video_spec(Path(self._model_name))

    async def stop(self) -> None:
        """Terminate owned workers and reap them before releasing admission."""
        await self._terminate_worker()
        async with self._call_lock:
            return

    async def abort_all_requests(self) -> int:
        """Stop the active worker when memory enforcement aborts requests."""
        process = self._process
        if process is None or (
            process.returncode is not None and not _process_group_exists(process.pid)
        ):
            return 0
        await self._terminate_worker()
        return 1

    async def _terminate_worker(self) -> None:
        process = self._process
        if process is None:
            return

        async def terminate() -> None:
            async with self._termination_lock:
                if process.returncode is not None and not _process_group_exists(process.pid):
                    return
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)

                async def wait_for_group() -> None:
                    await process.wait()
                    while _process_group_exists(process.pid):
                        await asyncio.sleep(_WORKER_GROUP_POLL_INTERVAL)

                try:
                    await asyncio.wait_for(wait_for_group(), _WORKER_STOP_TIMEOUT)
                except TimeoutError:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    try:
                        await asyncio.wait_for(wait_for_group(), _WORKER_STOP_TIMEOUT)
                    except TimeoutError as error:
                        raise RuntimeError(
                            f"Video worker group {process.pid} did not exit after SIGKILL"
                        ) from error

        waiter = asyncio.create_task(terminate())
        cancelled = False
        while not waiter.done():
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError:
                cancelled = True
        waiter.result()
        if cancelled:
            raise asyncio.CancelledError

    def process_memory_bytes(self) -> int:
        """Read the video process group's physical memory footprint."""
        process = self._process
        if process is None or (
            process.returncode is not None and not _process_group_exists(process.pid)
        ):
            return 0
        footprint = sum(
            get_phys_footprint(pid) for pid in list_process_group_pids(process.pid)
        )
        return max(footprint, self._estimated_size)

    def get_stats(self) -> dict[str, object]:
        """Report worker readiness without claiming resident video weights."""
        return {
            "model_name": self.model_name,
            "backend": "mlx-video",
            "resident": False,
            "active_requests": self.active_request_count(),
            "process_memory_bytes": self.process_memory_bytes(),
        }

    async def generate(
        self, plan: VideoRenderPlan, directory: Path, owner_fd: int
    ) -> None:
        """Run an isolated backend and retain the lease through process exit."""
        async with self._call_lock:
            if self._process is not None and (
                self._process.returncode is None
                or _process_group_exists(self._process.pid)
            ):
                raise RuntimeError("Previous video worker has not exited")
            spec_path = directory / "render.json"
            spec_path.write_text(plan.model_dump_json())
            activity = self._begin_activity("video", detail=plan.prompt)
            environment = {
                key: value
                for key, value in os.environ.items()
                if key
                in {
                    "PATH",
                    "HOME",
                    "TMPDIR",
                    "LANG",
                    "LC_ALL",
                    "HF_HOME",
                    "HF_HUB_CACHE",
                    "XDG_CACHE_HOME",
                    "PYTHONHOME",
                    "PYTHONPATH",
                    "PYTHONDONTWRITEBYTECODE",
                }
            }
            environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            try:
                with (directory / "render.log").open("wb") as log:
                    spawning = asyncio.create_task(
                        asyncio.create_subprocess_exec(
                            sys.executable,
                            "-m",
                            "omlx.video_worker",
                            str(spec_path),
                            stdin=asyncio.subprocess.PIPE,
                            stdout=log,
                            stderr=asyncio.subprocess.STDOUT,
                            env=environment,
                            start_new_session=True,
                            pass_fds=(owner_fd,),
                        )
                    )
                    cancelled = False
                    while not spawning.done():
                        try:
                            await asyncio.shield(spawning)
                        except asyncio.CancelledError:
                            cancelled = True
                    self._process = spawning.result()
                    if cancelled:
                        raise asyncio.CancelledError
                    returncode = await asyncio.shield(self._process.wait())
                    await self._terminate_worker()
                if returncode != 0:
                    raise RuntimeError(
                        f"Video worker exited with code {returncode}; see {directory / 'render.log'}"
                    )
                if not (directory / "video.mp4").is_file():
                    raise RuntimeError("Video worker produced no MP4")
            except asyncio.CancelledError:
                await self._terminate_worker()
                raise
            finally:
                if (
                    self._process is not None
                    and self._process.returncode is not None
                    and not _process_group_exists(self._process.pid)
                ):
                    self._process = None
                self._end_activity(activity)
