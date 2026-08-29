# SPDX-License-Identifier: Apache-2.0
"""Focused tests for OpenAI-compatible image route helpers."""

from __future__ import annotations

import base64
import tempfile
from contextlib import suppress
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from omlx.api import image_routes


def _png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeImageEngine:
    def __init__(self, tasks: list[str]) -> None:
        self._tasks = tasks
        self.pool: FakePool | None = None
        self.generate_calls: list[dict] = []
        self.edit_calls: list[dict] = []
        self.seen_paths: list[str] = []

    def get_stats(self) -> dict:
        return {"backend": "mlx-vlm", "tasks": self._tasks}

    async def generate(self, prompt: str, **kwargs):
        assert self.pool is not None
        assert self.pool.in_use > 0
        self.generate_calls.append({"prompt": prompt, **kwargs})
        return SimpleNamespace(
            image=Image.new("RGB", (2, 2), color=(0, 255, 0)),
            metadata={},
        )

    async def edit(self, prompt: str, image_paths: list[str], **kwargs):
        assert self.pool is not None
        assert self.pool.in_use > 0
        self.edit_calls.append({"prompt": prompt, "image_paths": image_paths, **kwargs})
        self.seen_paths.extend(image_paths)
        mask_path = kwargs.get("mask_path")
        if mask_path is not None:
            self.seen_paths.append(mask_path)
        assert all(Path(path).exists() for path in self.seen_paths)
        return SimpleNamespace(
            image=Image.new("RGB", (2, 2), color=(0, 0, 255)),
            metadata={},
        )


class FakePool:
    def __init__(
        self,
        engine: FakeImageEngine,
        tasks: list[str],
        *,
        model_type: str = "image",
        engine_type: str = "image",
    ) -> None:
        self.engine = engine
        self.engine.pool = self
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

    temp_dir = Path(".omlx_image_inputs_test")
    monkeypatch.setenv("OMLX_IMAGE_TMPDIR", str(temp_dir))
    monkeypatch.setattr(
        image_routes, "_resolve_model", lambda model: "resolved-image-model"
    )
    monkeypatch.setattr(image_routes, "require_mlx_vlm_available", lambda: None)

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    for path in temp_dir.glob("*"):
        path.unlink()
    with suppress(OSError):
        temp_dir.rmdir()


def _install_pool(monkeypatch, tasks: list[str]) -> FakeImageEngine:
    engine = FakeImageEngine(tasks)
    pool = FakePool(engine, tasks)
    monkeypatch.setattr(image_routes, "_get_engine_pool", lambda: pool)
    return engine


def test_generation_returns_b64_json_and_increments_seed(image_client, monkeypatch):
    engine = _install_pool(monkeypatch, ["generation"])

    response = image_client.post(
        "/v1/images/generations",
        json={
            "model": "alias",
            "prompt": "a green square",
            "n": 2,
            "seed": 41,
            "response_format": "b64_json",
            "output_format": "png",
            "style": "vivid",
            "moderation": "low",
            "user": "user-123",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "resolved-image-model"
    assert len(body["data"]) == 2
    assert "url" not in body["data"][0]
    assert base64.b64decode(body["data"][0]["b64_json"]).startswith(b"\x89PNG")
    assert [call["seed"] for call in engine.generate_calls] == [41, 42]
    for ignored_field in (
        "response_format",
        "style",
        "moderation",
        "user",
    ):
        assert ignored_field not in engine.generate_calls[0]


def test_generation_rejects_url_response_format_before_engine_pool(
    image_client, monkeypatch
):
    def get_engine_pool():
        raise AssertionError("engine pool should not be touched")

    monkeypatch.setattr(image_routes, "_get_engine_pool", get_engine_pool)

    response = image_client.post(
        "/v1/images/generations",
        json={
            "model": "alias",
            "prompt": "a green square",
            "response_format": "url",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "response_format='url' is not supported; use 'b64_json'"
    )


def test_generation_missing_mlx_vlm_returns_503_before_engine_pool(
    image_client, monkeypatch
):
    pool_touched = False

    def missing_mlx_vlm() -> None:
        raise ImportError(
            "mlx-vlm image support is unavailable. Reinstall oMLX to restore "
            "its required mlx-vlm dependency."
        )

    def get_engine_pool():
        nonlocal pool_touched
        pool_touched = True
        raise AssertionError("engine pool should not be touched")

    monkeypatch.setattr(
        image_routes, "require_mlx_vlm_available", missing_mlx_vlm
    )
    monkeypatch.setattr(image_routes, "_get_engine_pool", get_engine_pool)

    response = image_client.post(
        "/v1/images/generations",
        json={"model": "alias", "prompt": "a square"},
    )

    assert response.status_code == 503
    assert "mlx-vlm image support is unavailable" in response.json()["detail"]
    assert pool_touched is False


def test_generation_incompatible_mlx_vlm_returns_503_during_inference(
    image_client, monkeypatch
):
    engine = _install_pool(monkeypatch, ["generation"])

    async def fail_generate(prompt: str, **kwargs):
        raise ImportError(
            "mlx-vlm image support is unavailable or incompatible. "
            "Reinstall oMLX to restore its required mlx-vlm dependency."
        )

    monkeypatch.setattr(engine, "generate", fail_generate)

    response = image_client.post(
        "/v1/images/generations",
        json={"model": "alias", "prompt": "a square"},
    )

    assert response.status_code == 503
    assert "mlx-vlm image support is unavailable or incompatible" in response.json()[
        "detail"
    ]
    assert engine.pool is not None
    assert engine.pool.in_use == 0
    assert engine.pool.release_engine_calls == ["resolved-image-model"]


def test_generation_rejects_invalid_size_before_engine_call(image_client, monkeypatch):
    engine = _install_pool(monkeypatch, ["generation"])

    response = image_client.post(
        "/v1/images/generations",
        json={"model": "alias", "prompt": "a square", "size": "wide"},
    )

    assert response.status_code == 400
    assert "WIDTHxHEIGHT" in response.json()["detail"]
    assert engine.generate_calls == []


def test_generation_rejects_non_image_model(image_client, monkeypatch):
    engine = FakeImageEngine(["generation"])
    pool = FakePool(engine, ["generation"], model_type="llm", engine_type="batched")
    monkeypatch.setattr(image_routes, "_get_engine_pool", lambda: pool)

    response = image_client.post(
        "/v1/images/generations",
        json={"model": "alias", "prompt": "a square"},
    )

    assert response.status_code == 400
    assert "not an image model" in response.json()["detail"]
    assert pool.get_engine_calls == []


def test_generation_rejects_wrong_task_model(image_client, monkeypatch):
    engine = _install_pool(monkeypatch, ["edit"])

    response = image_client.post(
        "/v1/images/generations",
        json={"model": "alias", "prompt": "a square"},
    )

    assert response.status_code == 400
    assert "does not support task 'generation'" in response.json()["detail"]
    assert engine.generate_calls == []


def test_generation_rejects_request_lora_fields(image_client, monkeypatch):
    engine = _install_pool(monkeypatch, ["generation"])

    response = image_client.post(
        "/v1/images/generations",
        json={
            "model": "alias",
            "prompt": "a square",
            "lora_paths": ["style.safetensors"],
        },
    )

    assert response.status_code == 400
    assert "LoRA" in response.json()["detail"]
    assert engine.generate_calls == []


@pytest.mark.parametrize(
    "payload,expected_detail",
    [
        ({"stream": True}, "stream is not supported"),
        ({"partial_images": 1}, "partial_images is not supported"),
    ],
)
def test_generation_rejects_unsupported_streaming_fields(
    image_client, monkeypatch, payload, expected_detail
):
    engine = _install_pool(monkeypatch, ["generation"])

    response = image_client.post(
        "/v1/images/generations",
        json={"model": "alias", "prompt": "a square", **payload},
    )

    assert response.status_code == 400
    assert expected_detail in response.json()["detail"]
    assert engine.generate_calls == []


def test_json_edit_accepts_data_uri_and_cleans_temp_file(image_client, monkeypatch):
    engine = _install_pool(monkeypatch, ["edit"])
    data_uri = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode("ascii")

    response = image_client.post(
        "/v1/images/edits",
        json={
            "model": "alias",
            "prompt": "make it blue",
            "images": [{"image_url": data_uri}],
            "n": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["data"]) == 1
    assert base64.b64decode(body["data"][0]["b64_json"]).startswith(b"\x89PNG")
    assert len(engine.edit_calls) == 1
    assert all(not Path(path).exists() for path in engine.seen_paths)


def test_json_edit_missing_mlx_vlm_returns_503_before_download_or_engine_pool(
    image_client, monkeypatch
):
    path_conversion_touched = False
    pool_touched = False

    def missing_mlx_vlm() -> None:
        raise ImportError(
            "mlx-vlm image support is unavailable. Reinstall oMLX to restore "
            "its required mlx-vlm dependency."
        )

    async def image_reference_to_path(reference):
        nonlocal path_conversion_touched
        path_conversion_touched = True
        raise AssertionError("image refs should not be converted")

    def get_engine_pool():
        nonlocal pool_touched
        pool_touched = True
        raise AssertionError("engine pool should not be touched")

    monkeypatch.setattr(
        image_routes, "require_mlx_vlm_available", missing_mlx_vlm
    )
    monkeypatch.setattr(image_routes, "_image_reference_to_path", image_reference_to_path)
    monkeypatch.setattr(image_routes, "_get_engine_pool", get_engine_pool)

    response = image_client.post(
        "/v1/images/edits",
        json={
            "model": "alias",
            "prompt": "edit it",
            "images": [{"image_url": "https://example.com/input.png"}],
        },
    )

    assert response.status_code == 503
    assert "mlx-vlm image support is unavailable" in response.json()["detail"]
    assert path_conversion_touched is False
    assert pool_touched is False


def test_json_edit_rejects_mask_before_image_fetch_or_engine_load(
    image_client, monkeypatch
):
    engine = _install_pool(monkeypatch, ["edit"])
    data_uri = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode("ascii")
    mask_uri = "data:image/png;base64," + base64.b64encode(
        _png_bytes((255, 255, 255))
    ).decode("ascii")

    response = image_client.post(
        "/v1/images/edits",
        json={
            "model": "alias",
            "prompt": "make it blue",
            "images": [{"image_url": {"url": data_uri}}],
            "mask": {"image_url": {"url": mask_uri}},
            "n": 2,
            "seed": 7,
            "size": "2x2",
            "response_format": "b64_json",
            "style": "natural",
            "moderation": "auto",
            "user": "user-123",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "mask is not supported by mlx-vlm image models"
    assert engine.edit_calls == []
    assert engine.seen_paths == []


def test_json_edit_rejects_url_response_format_before_image_fetch(
    image_client, monkeypatch
):
    def image_reference_to_path(_reference):
        raise AssertionError("image references should not be resolved")

    def get_engine_pool():
        raise AssertionError("engine pool should not be touched")

    monkeypatch.setattr(image_routes, "_image_reference_to_path", image_reference_to_path)
    monkeypatch.setattr(image_routes, "_get_engine_pool", get_engine_pool)

    response = image_client.post(
        "/v1/images/edits",
        json={
            "model": "alias",
            "prompt": "make it blue",
            "images": [{"image_url": {"url": "data:image/png;base64,AA=="}}],
            "response_format": "url",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "response_format='url' is not supported; use 'b64_json'"
    )


def test_json_edit_rejects_file_id_refs(image_client, monkeypatch):
    _install_pool(monkeypatch, ["edit"])

    response = image_client.post(
        "/v1/images/edits",
        json={
            "model": "alias",
            "prompt": "edit it",
            "images": [{"file_id": "file-123"}],
        },
    )

    assert response.status_code == 400
    assert "file_id" in response.json()["detail"]


def test_json_edit_rejects_unsupported_streaming_fields(image_client, monkeypatch):
    engine = _install_pool(monkeypatch, ["edit"])
    data_uri = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode("ascii")

    response = image_client.post(
        "/v1/images/edits",
        json={
            "model": "alias",
            "prompt": "edit it",
            "images": [{"image_url": data_uri}],
            "partial_images": 1,
        },
    )

    assert response.status_code == 400
    assert "partial_images is not supported" in response.json()["detail"]
    assert engine.edit_calls == []


def test_json_edit_rejects_internal_http_image_url(image_client, monkeypatch):
    _install_pool(monkeypatch, ["edit"])

    response = image_client.post(
        "/v1/images/edits",
        json={
            "model": "alias",
            "prompt": "edit it",
            "images": [{"image_url": "http://127.0.0.1:8080/private.png"}],
        },
    )

    assert response.status_code == 400
    assert "public internet address" in response.json()["detail"]


def test_http_image_download_rejects_redirect_to_internal_host(monkeypatch):
    calls: list[str] = []

    def fake_getaddrinfo(host, port, *args, **kwargs):
        if host == "example.com":
            return [(image_routes.socket.AF_INET, image_routes.socket.SOCK_STREAM, 0, "", ("93.184.216.34", port))]
        if host == "127.0.0.1":
            return [(image_routes.socket.AF_INET, image_routes.socket.SOCK_STREAM, 0, "", ("127.0.0.1", port))]
        raise AssertionError(host)

    class RedirectResponse:
        status = 302

        def getheader(self, name: str):
            return {"location": "https://127.0.0.1/private.png"}.get(name.lower())

    class FakeConnection:
        def close(self) -> None:
            calls.append("closed")

    def fake_get_pinned_response(resolved):
        calls.append(resolved.url)
        assert resolved.resolved_ip == "93.184.216.34"
        return RedirectResponse(), FakeConnection()

    monkeypatch.setattr(image_routes.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(image_routes, "_get_pinned_response", fake_get_pinned_response)

    with pytest.raises(HTTPException) as exc_info:
        image_routes._download_http_image("https://example.com/image.png")

    assert exc_info.value.status_code == 400
    assert "public internet address" in exc_info.value.detail
    assert calls == ["https://example.com/image.png", "closed"]


def test_http_image_download_uses_pinned_resolved_ip(monkeypatch):
    calls: list[str] = []
    png = _png_bytes()

    def fake_getaddrinfo(host, port, *args, **kwargs):
        calls.append(f"resolve:{host}:{port}")
        return [(image_routes.socket.AF_INET, image_routes.socket.SOCK_STREAM, 0, "", ("93.184.216.34", port))]

    class OKResponse:
        status = 200

        def __init__(self) -> None:
            self._chunks = [png, b""]

        def getheader(self, name: str):
            headers = {"content-type": "image/png", "content-length": str(len(png))}
            return headers.get(name.lower())

        def read(self, size: int) -> bytes:
            return self._chunks.pop(0)

    class FakeConnection:
        def close(self) -> None:
            calls.append("closed")

    def fake_get_pinned_response(resolved):
        calls.append(f"connect:{resolved.hostname}:{resolved.resolved_ip}")
        return OKResponse(), FakeConnection()

    monkeypatch.setattr(image_routes.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(image_routes, "_get_pinned_response", fake_get_pinned_response)

    data, suffix = image_routes._download_http_image("https://example.com/image.png")

    assert data == png
    assert suffix == ".png"
    assert calls == [
        "resolve:example.com:443",
        "connect:example.com:93.184.216.34",
        "closed",
    ]


@pytest.mark.parametrize(
    "url_host,ip",
    [
        ("127.0.0.1", "127.0.0.1"),
        ("169.254.169.254", "169.254.169.254"),
        ("100.64.0.1", "100.64.0.1"),
        ("[::1]", "::1"),
        ("mapped.test", "::ffff:10.0.0.1"),
    ],
)
def test_http_image_download_rejects_non_public_addresses(monkeypatch, url_host, ip):
    def fake_getaddrinfo(resolved_host, port, *args, **kwargs):
        return [(image_routes.socket.AF_INET, image_routes.socket.SOCK_STREAM, 0, "", (ip, port))]

    monkeypatch.setattr(image_routes.socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(HTTPException) as exc_info:
        image_routes._validate_public_http_url(f"http://{url_host}/image.png")

    assert exc_info.value.status_code == 400
    assert "public internet address" in exc_info.value.detail


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:bad/image.png",
        "http://[::1/image.png",
    ],
)
def test_http_image_download_rejects_malformed_urls(url):
    with pytest.raises(HTTPException) as exc_info:
        image_routes._validate_public_http_url(url)

    assert exc_info.value.status_code == 400
    assert "Invalid image_url" in exc_info.value.detail


def test_get_pinned_response_closes_connection_on_setup_error(monkeypatch):
    instances = []

    class FakeConnection:
        def __init__(self, *args):
            self.closed = False
            instances.append(self)

        def putrequest(self, *args, **kwargs):
            raise OSError("send failed")

        def close(self):
            self.closed = True

    monkeypatch.setattr(image_routes, "_PinnedHTTPConnection", FakeConnection)

    with pytest.raises(OSError, match="send failed"):
        image_routes._get_pinned_response(
            image_routes._ResolvedHTTPURL(
                url="http://example.com/image.png",
                hostname="example.com",
                port=80,
                resolved_ip="93.184.216.34",
                scheme="http",
            )
        )

    assert instances[0].closed is True


def test_default_image_tmpdir_allows_cwd_under_system_tmp(monkeypatch):
    monkeypatch.delenv("OMLX_IMAGE_TMPDIR", raising=False)
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as tmp:
        monkeypatch.chdir(tmp)
        path = image_routes._image_tmpdir()

        assert path == Path(tmp) / ".omlx_image_inputs"
        assert path.is_dir()


def test_configured_image_tmpdir_rejects_system_tmp(monkeypatch):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as tmp:
        monkeypatch.setenv("OMLX_IMAGE_TMPDIR", str(Path(tmp) / "images"))

        with pytest.raises(HTTPException) as exc_info:
            image_routes._image_tmpdir()

    assert exc_info.value.status_code == 500
    assert "system temporary directory" in exc_info.value.detail


def test_default_image_tmpdir_secures_existing_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("OMLX_IMAGE_TMPDIR", raising=False)
    monkeypatch.chdir(tmp_path)
    image_dir = tmp_path / ".omlx_image_inputs"
    image_dir.mkdir(mode=0o777)
    image_dir.chmod(0o777)

    path = image_routes._image_tmpdir()

    assert path == image_dir
    assert path.stat().st_mode & 0o777 == 0o700


def test_validate_image_bytes_rejects_excessive_pixels(monkeypatch):
    monkeypatch.setattr(image_routes, "MAX_IMAGE_INPUT_PIXELS", 1)

    with pytest.raises(HTTPException) as exc_info:
        image_routes._validate_image_bytes(_png_bytes())

    assert exc_info.value.status_code == 413
    assert "maximum allowed pixels" in exc_info.value.detail


def test_json_edit_rejects_invalid_and_oversized_data_uri(
    image_client, monkeypatch
):
    _install_pool(monkeypatch, ["edit"])

    invalid_response = image_client.post(
        "/v1/images/edits",
        json={
            "model": "alias",
            "prompt": "edit it",
            "images": [{"image_url": "data:image/png;base64,not-valid"}],
        },
    )

    assert invalid_response.status_code == 400
    assert "Invalid image data URI" in invalid_response.json()["detail"]

    monkeypatch.setattr(image_routes, "MAX_IMAGE_INPUT_BYTES", 8)
    oversized_uri = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode(
        "ascii"
    )
    oversized_response = image_client.post(
        "/v1/images/edits",
        json={
            "model": "alias",
            "prompt": "edit it",
            "images": [{"image_url": oversized_uri}],
        },
    )

    assert oversized_response.status_code == 413
    assert "exceeds maximum allowed size" in oversized_response.json()["detail"]


def test_multipart_edit_accepts_upload_and_cleans_temp_file(image_client, monkeypatch):
    engine = _install_pool(monkeypatch, ["edit"])

    response = image_client.post(
        "/v1/images/edits",
        data={"model": "alias", "prompt": "edit upload", "seed": "9"},
        files={"image": ("input.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert len(response.json()["data"]) == 1
    assert engine.edit_calls[0]["seed"] == 9
    assert all(not Path(path).exists() for path in engine.seen_paths)


def test_multipart_edit_missing_mlx_vlm_returns_503_before_file_processing_or_engine_pool(
    image_client, monkeypatch
):
    upload_touched = False
    pool_touched = False

    def missing_mlx_vlm() -> None:
        raise ImportError(
            "mlx-vlm image support is unavailable. Reinstall oMLX to restore "
            "its required mlx-vlm dependency."
        )

    async def upload_to_path(value, field_name):
        nonlocal upload_touched
        upload_touched = True
        raise AssertionError("uploads should not be converted")

    def get_engine_pool():
        nonlocal pool_touched
        pool_touched = True
        raise AssertionError("engine pool should not be touched")

    monkeypatch.setattr(
        image_routes, "require_mlx_vlm_available", missing_mlx_vlm
    )
    monkeypatch.setattr(image_routes, "_upload_to_path", upload_to_path)
    monkeypatch.setattr(image_routes, "_get_engine_pool", get_engine_pool)

    response = image_client.post(
        "/v1/images/edits",
        data={"model": "alias", "prompt": "edit upload"},
        files={"image": ("input.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert "mlx-vlm image support is unavailable" in response.json()["detail"]
    assert upload_touched is False
    assert pool_touched is False


def test_multipart_request_from_form_does_not_inject_placeholder_image():
    class FakeForm(dict):
        def getlist(self, name: str):
            value = self.get(name)
            if value is None:
                return []
            return value if isinstance(value, list) else [value]

    request = image_routes._multipart_request_from_form(
        FakeForm({"model": "alias", "prompt": "edit upload"})
    )

    assert not hasattr(request, "images")


def test_multipart_edit_rejects_request_lora_fields(image_client, monkeypatch):
    engine = _install_pool(monkeypatch, ["edit"])

    response = image_client.post(
        "/v1/images/edits",
        data={
            "model": "alias",
            "prompt": "edit upload",
            "lora_paths": "style.safetensors",
        },
        files={"image": ("input.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 400
    assert "LoRA" in response.json()["detail"]
    assert engine.edit_calls == []


def test_multipart_edit_rejects_unsupported_streaming_fields(image_client, monkeypatch):
    engine = _install_pool(monkeypatch, ["edit"])

    response = image_client.post(
        "/v1/images/edits",
        data={
            "model": "alias",
            "prompt": "edit upload",
            "stream": "true",
        },
        files={"image": ("input.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 400
    assert "stream is not supported" in response.json()["detail"]
    assert engine.edit_calls == []


def test_multipart_edit_rejects_mask_before_upload_processing(
    image_client, monkeypatch
):
    engine = _install_pool(monkeypatch, ["edit"])

    response = image_client.post(
        "/v1/images/edits",
        data={
            "model": "alias",
            "prompt": "edit uploads",
            "n": "2",
            "seed": "3",
            "size": "2x2",
            "response_format": "b64_json",
            "style": "vivid",
            "moderation": "low",
            "user": "user-123",
        },
        files=[
            ("image[]", ("input-a.png", _png_bytes(), "image/png")),
            ("image[]", ("input-b.png", _png_bytes((0, 255, 0)), "image/png")),
            ("mask", ("mask.png", _png_bytes((255, 255, 255)), "image/png")),
        ],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "mask is not supported by mlx-vlm image models"
    assert engine.edit_calls == []
    assert engine.seen_paths == []


def test_multipart_edit_rejects_url_response_format_before_upload_processing(
    image_client, monkeypatch
):
    def upload_to_path(_value, _field_name):
        raise AssertionError("uploads should not be processed")

    def get_engine_pool():
        raise AssertionError("engine pool should not be touched")

    monkeypatch.setattr(image_routes, "_upload_to_path", upload_to_path)
    monkeypatch.setattr(image_routes, "_get_engine_pool", get_engine_pool)

    response = image_client.post(
        "/v1/images/edits",
        data={
            "model": "alias",
            "prompt": "edit uploads",
            "response_format": "url",
        },
        files={"image": ("input.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "response_format='url' is not supported; use 'b64_json'"
    )


def test_multipart_edit_rejects_invalid_and_oversized_upload(
    image_client, monkeypatch
):
    _install_pool(monkeypatch, ["edit"])

    invalid_response = image_client.post(
        "/v1/images/edits",
        data={"model": "alias", "prompt": "edit upload"},
        files={"image": ("input.png", b"not an image", "image/png")},
    )

    assert invalid_response.status_code == 400
    assert "valid image" in invalid_response.json()["detail"]

    monkeypatch.setattr(image_routes, "MAX_IMAGE_INPUT_BYTES", 8)
    oversized_response = image_client.post(
        "/v1/images/edits",
        data={"model": "alias", "prompt": "edit upload"},
        files={"image": ("input.png", b"x" * 9, "image/png")},
    )

    assert oversized_response.status_code == 413
    assert "exceeds maximum allowed size" in oversized_response.json()["detail"]


def test_server_image_routes_inherit_api_key_auth(monkeypatch):
    from omlx import server

    engine = _install_pool(monkeypatch, ["generation"])
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
