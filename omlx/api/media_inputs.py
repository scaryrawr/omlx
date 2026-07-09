# SPDX-License-Identifier: Apache-2.0
"""Helpers for OpenAI-compatible audio/video file inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

AUDIO_EXTENSIONS = {".wav", ".mp3", ".mpeg", ".mpga", ".m4a", ".webm"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}


def normalize_media_file_part(
    part: dict[str, Any],
    *,
    allow_file_url: bool = False,
) -> dict[str, Any] | None:
    """Convert audio/video file parts to internal media content parts.

    Returns ``None`` for document/text files so callers can keep routing those
    through MarkItDown. Video is an oMLX extension transported via OpenAI file
    part shapes.
    """
    file_obj = _file_object_from_part(part)
    if file_obj is None:
        return None

    mime_type = _normalized_mime_type(file_obj)
    extension = _file_extension(file_obj, mime_type)
    if mime_type.startswith("audio/"):
        is_audio = True
        is_video = False
    elif mime_type.startswith("video/"):
        is_audio = False
        is_video = True
    else:
        is_video = extension in VIDEO_EXTENSIONS
        is_audio = extension in AUDIO_EXTENSIONS and not is_video
    if not is_audio and not is_video:
        return None

    data = file_obj.get("file_data") or file_obj.get("data")
    url = file_obj.get("file_url")
    if url and not allow_file_url:
        raise ValueError("file_url is not supported for Chat Completions file parts.")
    if file_obj.get("file_id") and not data and not url:
        raise ValueError(
            "File IDs for audio/video inputs are not supported. "
            "Send base64 file_data or a Responses file_url instead."
        )
    if not data and not url:
        raise ValueError("Audio/video file parts require file_data or file_url.")

    filename = str(file_obj.get("filename") or "").strip() or None
    media_format = extension[1:] if extension else _format_from_mime_type(mime_type)
    payload: dict[str, Any] = {
        "format": media_format or ("mp4" if is_video else "wav"),
    }
    if data:
        payload["data"] = data
    if url:
        payload["url"] = url
    if filename:
        payload["filename"] = filename
    if mime_type:
        payload["mime_type"] = mime_type

    if is_video:
        return {"type": "input_video", "input_video": payload}
    return {"type": "input_audio", "input_audio": payload}


def normalize_media_file_parts_in_messages(
    messages: list[Any],
    *,
    allow_file_url: bool = False,
) -> list[Any]:
    """Replace media file parts in Pydantic/dict messages with media parts."""
    changed_any = False
    normalized = []
    for msg in messages:
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if not isinstance(content, list):
            normalized.append(msg)
            continue

        changed = False
        new_content = []
        for part in content:
            part_dict = _part_to_dict(part)
            media_part = normalize_media_file_part(
                part_dict,
                allow_file_url=allow_file_url,
            )
            if media_part is not None:
                new_content.append(media_part)
                changed = True
            else:
                new_content.append(part)
        if changed:
            changed_any = True
            if hasattr(msg, "model_copy"):
                normalized.append(msg.model_copy(update={"content": new_content}))
            else:
                new_msg = dict(msg)
                new_msg["content"] = new_content
                normalized.append(new_msg)
        else:
            normalized.append(msg)

    return normalized if changed_any else messages


def has_audio_video_parts(messages: list[Any]) -> bool:
    for msg in messages:
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if not isinstance(content, list):
            continue
        for part in content:
            part_dict = _part_to_dict(part)
            if part_dict.get("type") in {"input_audio", "input_video", "video"}:
                return True
    return False


def responses_input_file_to_file_part(part: dict[str, Any]) -> dict[str, Any]:
    """Convert a Responses ``input_file`` part to Chat-style ``file`` shape."""
    file_obj = {
        key: part[key]
        for key in ("filename", "file_data", "file_id", "file_url", "mime_type")
        if key in part and part[key] is not None
    }
    return {"type": "file", "file": file_obj}


def _file_object_from_part(part: dict[str, Any]) -> dict[str, Any] | None:
    if part.get("type") == "file" and isinstance(part.get("file"), dict):
        return part["file"]
    if part.get("type") == "input_file":
        return responses_input_file_to_file_part(part)["file"]
    return None


def _part_to_dict(part: Any) -> dict[str, Any]:
    if hasattr(part, "model_dump"):
        return part.model_dump(exclude_none=True)
    if hasattr(part, "dict"):
        return part.dict(exclude_none=True)
    return part if isinstance(part, dict) else {}


def _normalized_mime_type(file_obj: dict[str, Any]) -> str:
    explicit = str(file_obj.get("mime_type") or "").strip().lower()
    if explicit:
        return explicit
    data = str(file_obj.get("file_data") or file_obj.get("data") or "").strip()
    if data.startswith("data:"):
        marker = ";base64,"
        idx = data.find(marker)
        if idx >= 0:
            return data[5:idx].strip().lower()
    return ""


def _file_extension(file_obj: dict[str, Any], mime_type: str) -> str:
    filename = str(file_obj.get("filename") or "").strip()
    extension = Path(filename).suffix.lower()
    if extension:
        return extension
    if mime_type.startswith("audio/"):
        subtype = mime_type.split("/", 1)[1].split(";", 1)[0]
        return ".mp3" if subtype == "mpeg" else f".{subtype}"
    if mime_type.startswith("video/"):
        subtype = mime_type.split("/", 1)[1].split(";", 1)[0]
        return ".m4v" if subtype == "x-m4v" else f".{subtype}"
    return ""


def _format_from_mime_type(mime_type: str) -> str:
    if "/" not in mime_type:
        return ""
    return mime_type.split("/", 1)[1].split(";", 1)[0]
