# SPDX-License-Identifier: Apache-2.0
"""PCM framing and executor ownership for one Haru duplex session."""

import asyncio
import base64
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import anyio
import numpy as np

from ..engine_core import get_mlx_executor
from .realtime_voice import SAMPLE_RATE, RealtimeVoiceError

if TYPE_CHECKING:
    from mlx_audio.sts.models.haru import DuplexSession, Frame, Model

    from .sts import STSEngine

_FRAME_BYTES = 3200 * 2
_MAX_APPEND_BYTES = SAMPLE_RATE * 2
_MAX_APPEND_BASE64 = ((_MAX_APPEND_BYTES + 2) // 3) * 4
HARU_REALTIME_CAPABILITIES = {"input_image": False, "tools": False, "duplex": True}


def haru_default_instructions() -> str:
    from mlx_audio.sts.models.haru import DEFAULT_SYSTEM_PROMPT

    prompt: str = DEFAULT_SYSTEM_PROMPT
    return prompt


class HaruRealtimeSession:
    def __init__(self, engine: "STSEngine", model: "Model"):
        self._engine = engine
        self._model = model
        self._executor = get_mlx_executor()
        self._upstream: DuplexSession | None = None
        self._started = False
        self._closing: asyncio.Future | None = None
        self._remainder = b""
        self.instructions = haru_default_instructions()

    def update_instructions(self, instructions: str) -> None:
        if self._started:
            raise RealtimeVoiceError(
                "instructions_locked",
                "instructions cannot change after the first audio append",
            )
        if not isinstance(instructions, str) or len(instructions) > 8192:
            raise RealtimeVoiceError(
                "invalid_value",
                "instructions must be a string of at most 8192 characters",
            )
        self.instructions = instructions

    def _start(self) -> "Frame":
        # Retain the upstream session before start() in case start fails or
        # the socket cancels the executor await.
        upstream: DuplexSession = self._model.create_session(
            system_prompt=self.instructions
        )
        self._upstream = upstream
        frame: Frame = upstream.start()
        return frame

    def _step(self, samples: np.ndarray) -> "Frame":
        upstream = self._upstream
        if upstream is None:
            raise RuntimeError("Haru has not started")
        frame: Frame = upstream.step(samples)
        return frame

    async def append(self, audio: str) -> AsyncIterator["Frame"]:
        if self._closing is not None:
            raise RealtimeVoiceError("session_closed", "The duplex session has closed")
        if not isinstance(audio, str) or not audio:
            raise RealtimeVoiceError(
                "invalid_value", "audio must be nonempty base64 PCM16"
            )
        if len(audio) > _MAX_APPEND_BASE64:
            raise RealtimeVoiceError(
                "audio_buffer_limit", "send at most one second per append"
            )
        try:
            pcm = base64.b64decode(audio, validate=True)
        except ValueError as exc:
            raise RealtimeVoiceError(
                "invalid_value", "audio must be base64 PCM16"
            ) from exc
        loop = asyncio.get_running_loop()
        if not self._started:
            self._started = True
            frame = await loop.run_in_executor(self._executor, self._start)
            yield frame
            if frame.done:
                return
        pcm = self._remainder + pcm
        consumed = len(pcm) // _FRAME_BYTES * _FRAME_BYTES
        self._remainder = pcm[consumed:]
        for offset in range(0, consumed, _FRAME_BYTES):
            if self._closing is not None:
                return
            samples = np.frombuffer(pcm, dtype="<i2", count=3200, offset=offset)
            samples = samples.astype(np.float32) / 32768.0
            frame = await loop.run_in_executor(self._executor, self._step, samples)
            yield frame
            if frame.done:
                return

    def _close(self) -> None:
        try:
            if self._upstream is not None:
                self._upstream.synchronize_decoder()
        finally:
            self._upstream = None
            self._model = None

    async def close(self) -> None:
        if self._closing is None:
            loop = asyncio.get_running_loop()
            # The global single-worker executor drains any running frame before
            # touching its decoder. Cancelling the await cannot stop MLX work.
            self._closing = loop.run_in_executor(self._executor, self._close)
        cancelled = False
        try:
            with anyio.CancelScope(shield=True):
                while True:
                    try:
                        await asyncio.shield(self._closing)
                        break
                    except asyncio.CancelledError:
                        if self._closing.cancelled():
                            raise
                        cancelled = True
        finally:
            self._engine.release_haru_session(self)
        if cancelled:
            raise asyncio.CancelledError
