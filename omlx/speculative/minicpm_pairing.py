# SPDX-License-Identifier: Apache-2.0
"""Source provenance admission for the published MiniCPM target/helper pair."""

import hashlib
import json
from pathlib import Path

TARGET = (
    "openbmb/MiniCPM-V-4.7-35B-A3B",
    "4c7623a9326e0f74313bbc7292d852a9e3018a70",
)
HELPER = (
    "openbmb/MiniCPM-V-4.7-35B-A3B-DSpark",
    "5facb75c251ffd524524a544c1f5fd97ba0d6c5c",
)


def _validate_source(directory: Path, expected: tuple[str, str]) -> None:
    config_path = directory / "config.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    declared = config.get("omlx_source")
    if declared is not None:
        if (
            not isinstance(declared, dict)
            or (declared.get("repo_id"), declared.get("revision")) != expected
        ):
            raise ValueError(f"Checkpoint source does not match {expected[0]}")
        return
    metadata = directory / ".cache/huggingface/download/config.json.metadata"
    if not metadata.is_file():
        raise ValueError(
            f"Cannot establish checkpoint source for {expected[0]}. Retain the "
            "original Hugging Face download metadata, or declare the conversion's "
            "omlx_source.repo_id and omlx_source.revision in config.json."
        )
    lines = metadata.read_text().splitlines()
    git_blob = b"blob " + str(len(config_bytes)).encode() + b"\0" + config_bytes
    if (
        len(lines) < 2
        or lines[0] != expected[1]
        or lines[1] != hashlib.sha1(git_blob, usedforsecurity=False).hexdigest()
    ):
        raise ValueError(f"Stale or mismatched source metadata for {expected[0]}")


def validate_model_pair(target_path: str | None, helper_path: str) -> None:
    """Check origin metadata before loading helper tensors; shapes are separate."""
    if target_path is None:
        raise ValueError("MiniCPM DSpark requires the target checkpoint source path")
    _validate_source(Path(target_path), TARGET)
    _validate_source(Path(helper_path), HELPER)
