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


def test_mflux_is_only_in_image_extra():
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )

    core_dependencies = {
        _dependency_name(requirement)
        for requirement in pyproject["project"]["dependencies"]
    }
    image_extra_dependencies = {
        _dependency_name(requirement)
        for requirement in pyproject["project"]["optional-dependencies"]["image"]
    }

    assert "mflux" not in core_dependencies
    assert "mflux" in image_extra_dependencies
