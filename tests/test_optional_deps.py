# SPDX-License-Identifier: Apache-2.0
"""Tests for optional dependency helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from omlx.utils import optional_deps


def test_mlx_vlm_availability_uses_find_spec_without_importing(monkeypatch):
    calls: list[str] = []

    def fake_find_spec(name: str):
        calls.append(name)
        return SimpleNamespace()

    monkeypatch.setattr(optional_deps, "find_spec", fake_find_spec)

    assert optional_deps.is_mlx_vlm_available() is True
    assert calls == ["mlx_vlm"]


def test_require_mlx_vlm_available_raises_core_dependency_error(monkeypatch):
    monkeypatch.setattr(optional_deps, "find_spec", lambda name: None)

    with pytest.raises(ImportError) as exc_info:
        optional_deps.require_mlx_vlm_available()

    message = str(exc_info.value)
    assert "mlx-vlm image support is unavailable" in message
    assert "required mlx-vlm dependency" in message
    assert "omlx[image]" not in message
