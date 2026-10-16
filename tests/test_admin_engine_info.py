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


def test_engine_info_reports_mlx_vlm_version(monkeypatch):
    """Image support should share the mlx-vlm Engine Versions entry."""

    def fake_distribution(pkg_name: str):
        if pkg_name == "mlx-vlm":
            return _FakeDistribution()
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)
    monkeypatch.setattr(admin_routes, "_load_fallback_commits", lambda packages: {})

    engines = admin_routes._get_engine_info()

    assert engines["mlx-vlm"] == {
        "name": "mlx-vlm",
        "version": "9.9.9",
        "commit": None,
        "url": None,
    }


def test_parse_commits_uses_mlx_embeddings_source_repo(tmp_path):
    """Fallback commit links should point at the pinned package source repo."""
    pyproject = tmp_path / "pyproject.toml"
    sha = "71329b8a18f02273279ca766afb66ed0665a97f9"
    pyproject.write_text(
        "[project]\n"
        "dependencies = [\n"
        f'  "mlx-embeddings @ git+https://github.com/scaryrawr/mlx-embeddings@{sha}",\n'
        "]\n"
    )
    packages = {
        "mlx-embeddings": "https://github.com/scaryrawr/mlx-embeddings",
    }

    commits = admin_routes._parse_commits_from_pyproject(pyproject, packages)

    assert commits["mlx-embeddings"] == {
        "commit": sha,
        "url": f"https://github.com/scaryrawr/mlx-embeddings/commit/{sha}",
    }
