# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for OpenAI-compatible audio API.

These models define the request and response schemas for:
- Audio transcription (speech-to-text)
- Audio speech synthesis (text-to-speech)
"""

from pydantic import BaseModel, Field


class AudioTranscriptionRequest(BaseModel):
    """OpenAI-compatible audio transcription request."""

    model: str
    language: str | None = None
    prompt: str | None = None
    response_format: str | None = "json"
    temperature: float | None = 0.0


class AudioTranscriptionResponse(BaseModel):
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[dict] | None = None


class AudioSpeechRequest(BaseModel):
    model: str
    input: str
    voice: str | None = None
    language: str | None = None
    instructions: str | None = None
    speed: float | None = 1.0
    response_format: str | None = "wav"
    ref_audio: str | None = None
    ref_text: str | None = None
    # Sampling params must be non-negative and max_tokens positive; negative
    # values otherwise flow silently into the TTS engine's sampler.
    temperature: float | None = Field(default=None, ge=0)
    top_k: int | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0)
    repetition_penalty: float | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool | None = False
    streaming_interval: float | None = None


class AudioProcessRequest(BaseModel):
    """Request model for audio processing (speech enhancement / STS).

    Used by POST /v1/audio/process — the audio file is submitted as a
    multipart upload alongside this model field.
    """

    model: str
