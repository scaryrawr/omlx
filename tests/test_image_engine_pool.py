# SPDX-License-Identifier: Apache-2.0
"""Focused EnginePool coverage for image models."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

from omlx.engine_pool import EngineEntry, EnginePool
from omlx.exceptions import ModelLoadingError


def _pool() -> EnginePool:
    pool = EnginePool()
    pool._get_final_ceiling = lambda: 0
    return pool


def test_discovery_preserves_image_metadata_in_pool_status(tmp_path):
    model_dir = tmp_path / "flux-klein"
    model_dir.mkdir()
    (model_dir / "omlx-image-model.json").write_text(
        json.dumps(
            {
                "backend": "mlx-vlm",
                "base_model": "flux2-klein-4b",
                "task": ["txt2img", "edit"],
                "estimated_size": 4096,
            }
        )
    )

    pool = _pool()
    pool.discover_models(str(tmp_path))

    entry = pool.get_entry("flux-klein")
    assert entry is not None
    assert entry.model_type == "image"
    assert entry.engine_type == "image"
    assert entry.config_model_type == "mlx-vlm"
    assert entry.capabilities == ["generation", "edit"]
    assert entry.tasks == ["generation", "edit"]
    assert entry.image_metadata is not None
    assert entry.image_metadata["base_model"] == "flux2-klein-4b"

    status = pool.get_status()["models"][0]
    assert status["capabilities"] == ["generation", "edit"]
    assert status["tasks"] == ["generation", "edit"]
    assert status["image_metadata"]["backend"] == "mlx-vlm"


async def test_pool_load_acquire_release_and_unload_image_engine(
    monkeypatch, tmp_path
):
    calls: list[str] = []
    created_kwargs: dict[str, object] = {}

    class FakeImageEngine:
        def __init__(self, **kwargs):
            created_kwargs.update(kwargs)

        async def start(self):
            calls.append("start")

        async def stop(self):
            calls.append("stop")

        def has_active_requests(self):
            return False

    fake_module = types.ModuleType("omlx.engine.image")
    fake_module.ImageEngine = FakeImageEngine
    monkeypatch.setitem(sys.modules, "omlx.engine.image", fake_module)
    dflash_module = types.ModuleType("omlx.engine.dflash")

    class UnexpectedDFlashEngine:
        def __init__(self, **kwargs):
            raise AssertionError("image models must bypass DFlash dispatch")

    dflash_module.DFlashEngine = UnexpectedDFlashEngine
    monkeypatch.setitem(sys.modules, "omlx.engine.dflash", dflash_module)
    monkeypatch.setattr("omlx.engine_pool.is_mlx_vlm_available", lambda: True)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.synchronize", lambda: None)
    monkeypatch.setattr("omlx.engine_pool.mx.clear_cache", lambda: None)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)

    model_dir = tmp_path / "z-image"
    model_dir.mkdir()
    pool = _pool()
    pool._entries["z-image"] = EngineEntry(
        model_id="z-image",
        model_path=str(model_dir),
        model_type="image",
        engine_type="image",
        estimated_size=4096,
        config_model_type="mlx-vlm",
        capabilities=["generation"],
        tasks=["generation"],
        image_metadata={"backend": "mlx-vlm", "base_model": "z-image"},
    )

    settings = SimpleNamespace(
        dflash_enabled=True,
        dflash_draft_model="draft",
    )
    await pool._load_engine("z-image", runtime_settings=settings)
    async with pool.acquire("z-image") as engine:
        assert isinstance(engine, FakeImageEngine)
        assert pool.get_entry("z-image").in_use == 1

    assert pool.get_entry("z-image").in_use == 0
    assert await pool.unload_if_idle_unpinned("z-image") is True
    assert calls == ["start", "stop"]
    assert created_kwargs["model_id"] == "z-image"
    assert created_kwargs["tasks"] == ["generation"]
    assert pool.current_model_memory == 0


async def test_missing_mlx_vlm_fails_before_image_pool_admission(
    monkeypatch, tmp_path
):
    model_dir = tmp_path / "z-image"
    model_dir.mkdir()
    pool = _pool()
    pool._entries["z-image"] = EngineEntry(
        model_id="z-image",
        model_path=str(model_dir),
        model_type="image",
        engine_type="image",
        estimated_size=4096,
        image_metadata={"backend": "mlx-vlm", "base_model": "z-image"},
        tasks=["generation"],
    )
    monkeypatch.setattr("omlx.engine_pool.is_mlx_vlm_available", lambda: False)

    with pytest.raises(
        ModelLoadingError, match="mlx-vlm image support is unavailable"
    ):
        await pool.get_engine("z-image")

    assert pool.current_model_memory == 0
    assert pool.get_entry("z-image").engine is None
