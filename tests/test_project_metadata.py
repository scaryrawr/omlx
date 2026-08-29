# SPDX-License-Identifier: Apache-2.0
"""Tests for packaging metadata."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


def _dependency_name(requirement: str) -> str:
    return re.split(r"\s*(?:[<>=!~]=?|@|\[)", requirement.strip(), maxsplit=1)[
        0
    ].lower().replace("_", "-")


def test_mlx_vlm_is_core_and_image_extra_is_compatibility_only():
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )

    core_dependencies = {
        _dependency_name(requirement)
        for requirement in pyproject["project"]["dependencies"]
    }
    image_extra = pyproject["project"]["optional-dependencies"]["image"]

    assert "mlx-vlm" in core_dependencies
    assert image_extra == []
