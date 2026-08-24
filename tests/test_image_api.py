# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the OpenAI-compatible image generation route."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from omlx.api import image_routes
from omlx.exceptions import (
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
    ModelUnavailableError,
)


class FakeImageEngine:
    def __init__(self, tasks: list[str]) -> None:
        self.tasks = tasks
        self.pool: FakePool | None = None
        self.generate_calls: list[dict] = []
        self.active_calls = 0
        self.max_active_calls = 0

    def get_stats(self) -> dict:
        return {"backend": "mlx-vlm", "tasks": self.tasks}

    async def generate(self, prompt: str, **kwargs):
        assert self.pool is not None
        assert self.pool.in_use > 0
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        self.generate_calls.append({"prompt": prompt, **kwargs})
        self.active_calls -= 1
        return SimpleNamespace(
            image=Image.new("RGBA", (2, 2), color=(0, 255, 0, 255)),
            metadata={"revised_prompt": "revised"},
        )


class FakePool:
    def __init__(
        self,
        engine: FakeImageEngine,
        tasks: list[str],
        *,
        model_type: str = "image",
        engine_type: str = "image",
        get_engine_error: Exception | None = None,
    ) -> None:
        self.engine = engine
        self.engine.pool = self
        self.get_engine_error = get_engine_error
        self.get_engine_calls: list[str] = []
        self.release_engine_calls: list[str] = []
        self.in_use = 0
        self.entry = SimpleNamespace(
            model_type=model_type,
            engine_type=engine_type,
            tasks=tasks,
            image_metadata={"tasks": tasks},
        )

    def get_entry(self, model_id: str):
        return self.entry if model_id == "resolved-image-model" else None

    async def get_engine(self, model_id: str, *, _lease: bool = False):
        assert model_id == "resolved-image-model"
        assert _lease is True
        self.get_engine_calls.append(model_id)
        if self.get_engine_error is not None:
            raise self.get_engine_error
        self.in_use += 1
        return self.engine

    async def release_engine(self, model_id: str):
        assert model_id == "resolved-image-model"
        self.release_engine_calls.append(model_id)
        self.in_use -= 1


@pytest.fixture
def image_client(monkeypatch):
    app = FastAPI()
    app.include_router(image_routes.router)
    monkeypatch.setattr(
        image_routes, "_resolve_model", lambda model: "resolved-image-model"
    )
    monkeypatch.setattr(image_routes, "require_mlx_vlm_available", lambda: None)

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _install_pool(
    monkeypatch,
    tasks: list[str],
    **kwargs,
) -> tuple[FakeImageEngine, FakePool]:
    engine = FakeImageEngine(tasks)
    pool = FakePool(engine, tasks, **kwargs)
    monkeypatch.setattr(image_routes, "_get_engine_pool", lambda: pool)
    return engine, pool


@pytest.mark.parametrize(
    ("output_format", "magic"),
    [
        ("png", b"\x89PNG"),
        ("jpeg", b"\xff\xd8"),
        ("webp", b"RIFF"),
    ],
)
def test_generation_encodes_supported_output_formats(
    image_client,
    monkeypatch,
    output_format,
    magic,
):
    _, _ = _install_pool(monkeypatch, ["generation"])

    response = image_client.post(
        "/v1/images/generations",
        json={
            "model": "alias",
            "prompt": "a green square",
            "output_format": output_format,
        },
    )

    assert response.status_code == 200
    encoded = base64.b64decode(response.json()["data"][0]["b64_json"])
    assert encoded.startswith(magic)


def test_generation_resolves_alias_and_runs_outputs_sequentially(
    image_client,
    monkeypatch,
):
    engine, pool = _install_pool(monkeypatch, ["generation"])

    response = image_client.post(
        "/v1/images/generations",
        json={
            "model": "alias",
            "prompt": "a green square",
            "n": 3,
            "seed": 41,
            "size": "640x480",
            "response_format": "b64_json",
            "negative_prompt": "blurry",
            "scheduler": "euler",
            "style": "vivid",
            "moderation": "low",
            "user": "user-123",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "resolved-image-model"
    assert body["usage"]["image_count"] == 3
    assert len(body["data"]) == 3
    assert all("url" not in item for item in body["data"])
    assert [call["seed"] for call in engine.generate_calls] == [41, 42, 43]
    assert {(call["width"], call["height"]) for call in engine.generate_calls} == {
        (640, 480)
    }
    assert engine.generate_calls[0]["negative_prompt"] == "blurry"
    assert engine.generate_calls[0]["scheduler"] == "euler"
    assert engine.max_active_calls == 1
    assert pool.in_use == 0
    assert pool.release_engine_calls == ["resolved-image-model"]
    for ignored_field in ("response_format", "style", "moderation", "user"):
        assert ignored_field not in engine.generate_calls[0]


@pytest.mark.parametrize(
    ("payload", "expected_detail"),
    [
        ({"response_format": "url"}, "response_format='url' is not supported"),
        ({"stream": True}, "stream is not supported"),
        ({"partial_images": 1}, "partial_images is not supported"),
        ({"lora_paths": ["style.safetensors"]}, "LoRA"),
        ({"lora_scales": [0.8]}, "LoRA"),
        ({"size": "wide"}, "WIDTHxHEIGHT"),
    ],
)
def test_generation_rejects_unsupported_requests_before_engine_pool(
    image_client,
    monkeypatch,
    payload,
    expected_detail,
):
    def get_engine_pool():
        raise AssertionError("engine pool should not be touched")

    monkeypatch.setattr(image_routes, "_get_engine_pool", get_engine_pool)

    response = image_client.post(
        "/v1/images/generations",
        json={"model": "alias", "prompt": "a square", **payload},
    )

    assert response.status_code == 400
    assert expected_detail in response.json()["detail"]


def test_generation_missing_dependency_returns_503_before_engine_pool(
    image_client,
    monkeypatch,
):
    def missing_mlx_vlm() -> None:
        raise ImportError("mlx-vlm image support is unavailable")

    def get_engine_pool():
        raise AssertionError("engine pool should not be touched")

    monkeypatch.setattr(image_routes, "require_mlx_vlm_available", missing_mlx_vlm)
    monkeypatch.setattr(image_routes, "_get_engine_pool", get_engine_pool)

    response = image_client.post(
        "/v1/images/generations",
        json={"model": "alias", "prompt": "a square"},
    )

    assert response.status_code == 503
    assert "mlx-vlm image support is unavailable" in response.json()["detail"]


def test_generation_releases_engine_when_inference_fails(image_client, monkeypatch):
    engine, pool = _install_pool(monkeypatch, ["generation"])

    async def fail_generate(prompt: str, **kwargs):
        raise ImportError("mlx-vlm image support is incompatible")

    monkeypatch.setattr(engine, "generate", fail_generate)

    response = image_client.post(
        "/v1/images/generations",
        json={"model": "alias", "prompt": "a square"},
    )

    assert response.status_code == 503
    assert pool.in_use == 0
    assert pool.release_engine_calls == ["resolved-image-model"]


def test_generation_rejects_non_image_model_without_acquiring(
    image_client,
    monkeypatch,
):
    engine, pool = _install_pool(
        monkeypatch,
        ["generation"],
        model_type="llm",
        engine_type="batched",
    )

    response = image_client.post(
        "/v1/images/generations",
        json={"model": "alias", "prompt": "a square"},
    )

    assert response.status_code == 400
    assert "not an image model" in response.json()["detail"]
    assert engine.generate_calls == []
    assert pool.get_engine_calls == []


def test_generation_rejects_wrong_task_model_without_acquiring(
    image_client,
    monkeypatch,
):
    engine, pool = _install_pool(monkeypatch, ["edit"])

    response = image_client.post(
        "/v1/images/generations",
        json={"model": "alias", "prompt": "a square"},
    )

    assert response.status_code == 400
    assert "does not support task 'generation'" in response.json()["detail"]
    assert engine.generate_calls == []
    assert pool.get_engine_calls == []


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (ModelNotFoundError("resolved-image-model", ["other"]), 404),
        (ModelTooLargeError("resolved-image-model", 10, 5), 507),
        (InsufficientMemoryError(10, 5, "not enough memory"), 507),
        (ModelLoadingError("resolved-image-model"), 409),
        (ModelBusyError("resolved-image-model", "reload"), 409),
        (ModelUnavailableError("resolved-image-model"), 503),
    ],
)
def test_generation_maps_engine_pool_errors(
    image_client,
    monkeypatch,
    error,
    status_code,
):
    _, pool = _install_pool(
        monkeypatch,
        ["generation"],
        get_engine_error=error,
    )

    response = image_client.post(
        "/v1/images/generations",
        json={"model": "alias", "prompt": "a square"},
    )

    assert response.status_code == status_code
    assert pool.release_engine_calls == []


def test_server_image_route_inherits_api_key_auth(monkeypatch):
    from omlx import server

    engine, pool = _install_pool(monkeypatch, ["generation"])
    monkeypatch.setattr(
        image_routes, "_resolve_model", lambda model: "resolved-image-model"
    )
    monkeypatch.setattr(image_routes, "require_mlx_vlm_available", lambda: None)
    monkeypatch.setattr(server._server_state, "api_key", "secret")
    monkeypatch.setattr(server._server_state, "global_settings", None)
    monkeypatch.setattr(server._server_state, "engine_pool", None)
    monkeypatch.setattr(server._server_state, "process_memory_enforcer", None)

    with TestClient(server.app, raise_server_exceptions=False) as client:
        unauthorized = client.post(
            "/v1/images/generations",
            json={"model": "alias", "prompt": "a square"},
        )
        authorized = client.post(
            "/v1/images/generations",
            json={"model": "alias", "prompt": "a square"},
            headers={"Authorization": "Bearer secret"},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert len(engine.generate_calls) == 1
    assert pool.release_engine_calls == ["resolved-image-model"]
