# SPDX-License-Identifier: Apache-2.0
"""
Image processing utilities for VLM (Vision-Language Model) support.

This module provides functions for loading images from base64 data URIs,
extracting images from OpenAI-format messages, and computing image
hashes for prefix cache deduplication.
"""

import base64
import binascii
import hashlib
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ..exceptions import InvalidRequestError

logger = logging.getLogger(__name__)


class TemporaryMediaPath(str):
    """String path for decoded media that should be removed after preprocessing."""

    def cleanup(self) -> None:
        try:
            os.unlink(str(self))
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("Failed to remove temporary media file %s: %s", self, exc)


_IMAGE_INPUT_ERROR = (
    "Image inputs must be base64 data URIs "
    "(data:image/...;base64,...). Remote URLs and local file paths are not supported."
)
_AUDIO_INPUT_ERROR = (
    "input_audio.data must be a base64 string or base64 data URI. "
    "Local file paths are not supported."
)


def _decode_base64_data_uri(value: str, *, field: str) -> bytes:
    """Decode a base64 data URI, mapping malformed input to a request error."""
    if not isinstance(value, str):
        raise InvalidRequestError(_IMAGE_INPUT_ERROR, field=field)

    stripped = value.strip()
    if not stripped.startswith("data:"):
        raise InvalidRequestError(_IMAGE_INPUT_ERROR, field=field)

    prefix, separator, encoded = stripped.partition(",")
    prefix_lower = prefix.lower()
    if (
        separator != ","
        or not prefix_lower.startswith("data:image/")
        or ";base64" not in prefix_lower
    ):
        raise InvalidRequestError(
            f"{field} must use a base64 image data URI.",
            field=field,
        )

    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequestError(
            f"{field} contains invalid base64 data.",
            field=field,
        ) from exc


def _decode_input_audio_data(data: str, *, field: str = "input_audio.data") -> bytes:
    """Decode input_audio.data without falling back to filesystem paths."""
    stripped = data.strip()
    if stripped.startswith("data:"):
        prefix, separator, encoded = stripped.partition(",")
        if separator != "," or ";base64" not in prefix.lower():
            raise InvalidRequestError(
                f"{field} must use a base64 data URI.",
                field=field,
            )
    else:
        encoded = stripped

    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequestError(_AUDIO_INPUT_ERROR, field=field) from exc


def validate_image_data_uri(value: str, *, field: str = "image") -> str:
    """Validate that a request-facing image reference is an inline data URI."""
    _decode_base64_data_uri(value, field=field)
    return value


def load_image(url_or_base64: str, *, field: str = "image_url") -> Image.Image:
    """
    Load an image from a base64 data URI.

    Supports:
    - Data URIs: "data:image/jpeg;base64,..." format

    Args:
        url_or_base64: Image base64 data URI string

    Returns:
        PIL Image object

    Raises:
        InvalidRequestError: If the input is not a valid image data URI
    """
    img_bytes = _decode_base64_data_uri(url_or_base64, field=field)
    try:
        img = Image.open(io.BytesIO(img_bytes))
    except Exception as exc:
        raise InvalidRequestError(
            f"{field} does not contain a decodable image.",
            field=field,
        ) from exc

    # Apply EXIF orientation (phone photos etc.) before processing.
    # Matches mlx-vlm's load_image which calls ImageOps.exif_transpose().
    img = ImageOps.exif_transpose(img)
    # Ensure RGB format (RGBA/P/L etc. cause broadcast errors in vision processors)
    return img.convert("RGB")


def extract_media_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Image.Image], list, list[str]]:
    """
    Extract images, audio, and video from OpenAI-format messages.

    Processes messages containing content arrays with image_url or input_audio
    parts, loads the media, and returns cleaned text-only messages alongside
    the loaded images and audio files.

    Args:
        messages: List of OpenAI-format chat messages. Each message may have
            content as a string or a list of content parts
            (text/image_url/input_audio).

    Returns:
        Tuple of (text_messages, images, audio, videos):
        - text_messages: Messages with media parts removed, text parts joined
        - images: List of loaded PIL Image objects in order of appearance
        - audio: List of BytesIO audio buffers
    """
    text_messages = []
    images = []
    audio = []
    videos = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if not isinstance(content, list):
            # Simple string content — pass through
            text_messages.append({"role": role, "content": content or ""})
            # Preserve extra fields (tool_calls, tool_call_id, etc.)
            for key in msg:
                if key not in ("role", "content"):
                    text_messages[-1][key] = msg[key]
            continue

        # Content array with text, image_url, input_audio, and/or video parts
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
            else:
                # Pydantic model (ContentPart)
                part_type = getattr(part, "type", None)

            if part_type == "text":
                text = (
                    part.get("text")
                    if isinstance(part, dict)
                    else getattr(part, "text", None)
                )
                if text:
                    text_parts.append(text)

            elif part_type in ("image_url", "input_image"):
                # OpenAI chat format: {"type":"image_url","image_url":{"url":"..."}}
                # Responses-style format: {"type":"input_image","image_url":"..."}
                image_url_obj = (
                    part.get("image_url")
                    if isinstance(part, dict)
                    else getattr(part, "image_url", None)
                )
                if image_url_obj is None and isinstance(part, dict):
                    image_url_obj = part.get("input_image")

                url = None
                if isinstance(image_url_obj, str):
                    url = image_url_obj
                elif isinstance(image_url_obj, dict):
                    url = image_url_obj.get("url")
                elif image_url_obj is not None:
                    url = getattr(image_url_obj, "url", None)

                if url:
                    images.append(load_image(url, field="image_url"))

            elif part_type == "input_audio":
                # OpenAI audio format: {"type":"input_audio","input_audio":{"data":"...","format":"wav"}}
                input_audio = (
                    part.get("input_audio")
                    if isinstance(part, dict)
                    else getattr(part, "input_audio", None)
                )
                if input_audio and isinstance(input_audio, dict):
                    url = input_audio.get("url")
                    if isinstance(url, str) and url.strip():
                        raise InvalidRequestError(_AUDIO_INPUT_ERROR, field="input_audio.url")
                    data = input_audio.get("data", "")
                    if isinstance(data, str):
                        audio.append(io.BytesIO(_decode_input_audio_data(data)))
                    elif isinstance(data, bytes):
                        audio.append(io.BytesIO(data))
                    elif data is not None:
                        audio.append(data)

            elif part_type in ("input_video", "video"):
                input_video = (
                    (part.get("input_video") or part.get("video"))
                    if isinstance(part, dict)
                    else (
                        getattr(part, "input_video", None)
                        or getattr(part, "video", None)
                    )
                )
                if isinstance(input_video, str):
                    source = input_video.strip()
                    if source:
                        videos.append(_validate_media_source_url(source, "video URL"))
                elif input_video and isinstance(input_video, dict):
                    source = (
                        input_video.get("url")
                        or input_video.get("file_url")
                        or input_video.get("path")
                    )
                    if isinstance(source, str) and source.strip():
                        videos.append(_validate_media_source_url(source.strip(), "video URL"))
                    else:
                        data = input_video.get("data") or input_video.get("file_data")
                        if isinstance(data, str) and data.strip():
                            try:
                                videos.append(
                                    _video_data_to_temp_path(
                                        data,
                                        input_video.get("format"),
                                        input_video.get("filename"),
                                    )
                                )
                            except (binascii.Error, ValueError) as exc:
                                logger.warning(
                                    "Failed to decode input_video base64: %s", exc
                                )

        new_msg = {"role": role, "content": "\n".join(text_parts) if text_parts else ""}
        # Preserve extra fields
        for key in msg:
            if key not in ("role", "content"):
                new_msg[key] = msg[key]
        text_messages.append(new_msg)

    return text_messages, images, audio, videos


def extract_images_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Image.Image], list]:
    """Backward-compatible wrapper returning text, images, and audio."""
    text_messages, images, audio, _ = extract_media_from_messages(messages)
    return text_messages, images, audio


def cleanup_temporary_media(paths: list[Any]) -> None:
    for path in paths or []:
        cleanup = getattr(path, "cleanup", None)
        if callable(cleanup):
            cleanup()


def _validate_media_source_url(value: str, field_name: str) -> str:
    if value.startswith(("http://", "https://")):
        from .url_security import validate_public_http_url

        return validate_public_http_url(value, field_name=field_name)
    return value


def compute_audio_video_hash(audio: list[Any], videos: list[Any]) -> str | None:
    """Compute a stable cache key component for audio/video VLM inputs."""
    if not audio and not videos:
        return None

    hasher = hashlib.sha256()
    for label, items in (("audio", audio), ("video", videos)):
        for item in items or []:
            hasher.update(label.encode())
            _update_hash_for_media_item(hasher, item)
    return hasher.hexdigest()


def _update_hash_for_media_item(hasher: Any, item: Any) -> None:
    if isinstance(item, io.BytesIO):
        pos = item.tell()
        try:
            item.seek(0)
            hasher.update(item.read())
        finally:
            item.seek(pos)
        return

    if isinstance(item, (bytes, bytearray, memoryview)):
        hasher.update(bytes(item))
        return

    if isinstance(item, str):
        path = Path(item[7:] if item.startswith("file://") else item)
        if path.exists() and path.is_file():
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    hasher.update(chunk)
        else:
            hasher.update(item.encode())
        return

    hasher.update(repr(item).encode())


def _video_data_to_temp_path(
    value: str,
    media_format: Any = None,
    filename: Any = None,
) -> TemporaryMediaPath:
    data = value.strip()
    suffix = Path(str(filename or "")).suffix
    if not suffix:
        fmt = str(media_format or "mp4").strip().lower().lstrip(".")
        suffix = f".{fmt or 'mp4'}"
    if data.startswith("data:"):
        marker = ";base64,"
        idx = data.find(marker)
        if idx < 0:
            raise ValueError("Only base64 data URIs are supported for input_video.")
        data = data[idx + len(marker) :]
    decoded = base64.b64decode(data, validate=True)
    fd, path = tempfile.mkstemp(prefix="omlx-video-", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(decoded)
    return TemporaryMediaPath(path)


def compute_image_hash(images: list[Image.Image]) -> str | None:
    """
    Compute a SHA256 hash from a list of images for prefix cache deduplication.

    Uses image size and raw pixel data to produce a deterministic hash.
    Returns None if images list is empty.

    Args:
        images: List of PIL Image objects

    Returns:
        Hex-encoded SHA256 hash string, or None if no images
    """
    if not images:
        return None

    hasher = hashlib.sha256()
    for img in images:
        # Include image dimensions
        hasher.update(f"{img.size[0]}x{img.size[1]}".encode())
        # Include raw pixel data (convert to RGB for consistency)
        rgb_img = img.convert("RGB")
        hasher.update(rgb_img.tobytes())

    return hasher.hexdigest()


def compute_per_image_hashes(images: list[Image.Image]) -> list[str]:
    """Compute individual SHA256 hashes for each image.

    Returns a list of hex-encoded hash strings, one per image.
    """
    hashes = []
    for img in images:
        hasher = hashlib.sha256()
        hasher.update(f"{img.size[0]}x{img.size[1]}".encode())
        rgb_img = img.convert("RGB")
        hasher.update(rgb_img.tobytes())
        hashes.append(hasher.hexdigest())
    return hashes
