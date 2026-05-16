# SPDX-License-Identifier: Apache-2.0
"""Tests for admin dashboard engine/package metadata."""

from __future__ import annotations

import importlib.metadata

from omlx.admin import routes as admin_routes


class _FakeDistribution:
    version = "9.9.9"

    @staticmethod
    def read_text(name: str) -> str | None:
        return None


def test_engine_info_reports_mflux_version(monkeypatch):
    """Image support should surface mflux in the dashboard Engine Versions card."""

    def fake_distribution(pkg_name: str):
        if pkg_name == "mflux":
            return _FakeDistribution()
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)
    monkeypatch.setattr(admin_routes, "_load_fallback_commits", lambda packages: {})

    engines = admin_routes._get_engine_info()

    assert engines["mflux"] == {
        "name": "mflux",
        "version": "9.9.9",
        "commit": None,
        "url": None,
    }
