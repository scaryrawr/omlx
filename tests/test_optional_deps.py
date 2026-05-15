# SPDX-License-Identifier: Apache-2.0
"""Tests for optional dependency helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from omlx.utils import optional_deps


def test_mflux_availability_uses_find_spec_without_importing(monkeypatch):
    calls: list[str] = []

    def fake_find_spec(name: str):
        calls.append(name)
        return SimpleNamespace()

    monkeypatch.setattr(optional_deps, "find_spec", fake_find_spec)

    assert optional_deps.is_mflux_available() is True
    assert calls == ["mflux"]


def test_mflux_availability_false_when_spec_missing(monkeypatch):
    monkeypatch.setattr(optional_deps, "find_spec", lambda name: None)

    assert optional_deps.is_mflux_available() is False


def test_require_mflux_available_raises_install_hint(monkeypatch):
    monkeypatch.setattr(optional_deps, "find_spec", lambda name: None)

    with pytest.raises(ImportError) as exc_info:
        optional_deps.require_mflux_available()

    message = str(exc_info.value)
    assert "mflux is required for image inference" in message
    assert "pip install 'omlx[image]'" in message
    assert "pip install -e '.[image]'" in message
