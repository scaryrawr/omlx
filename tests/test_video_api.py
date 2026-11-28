# SPDX-License-Identifier: Apache-2.0
"""Caller-visible video job lifecycle tests."""

import asyncio
import json
from contextlib import asynccontextmanager
from io import BytesIO
from threading import Event
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image

from omlx.api.video_routes import router
from omlx.engine.video import VideoEngine
from omlx.server import ServerState, app, list_models_status
from omlx.utils.optional_deps import require_video_runtime
from omlx.video_jobs import VideoJobs


class FakeVideoEngine(VideoEngine):
    def __init__(self, model_path):
        super().__init__(str(model_path))
        self.entered = asyncio.Event()
        self.finish = asyncio.Event()
        self.plan = None
        self.reference = None
        self.leased = False
        self.failure = False

    async def generate(self, plan, directory, owner_fd):
        assert self.leased
        self.plan = plan
        if plan.has_reference:
            with Image.open(directory / "reference.png") as image:
                self.reference = image.getpixel((0, 0))
        self.entered.set()
        await self.finish.wait()
        if self.failure:
            raise RuntimeError("backend failure")
        (directory / "video.mp4").write_bytes(b"verified-mp4")


class FakePool:
    def __init__(self, model_path):
        self.engine = FakeVideoEngine(model_path)
        self.entry = type(
            "Entry", (), {"model_path": str(model_path), "engine_type": "video"}
        )()
        self.wake_calls = []

    def get_entry(self, model):
        return self.entry if model == "wan" else None

    def _wake_process_memory_enforcer(self, *, active=False):
        self.wake_calls.append(active)

    @asynccontextmanager
    async def acquire(self, model):
        assert model == "wan"
        self.engine.leased = True
        try:
            yield self.engine
        finally:
            self.engine.leased = False


@pytest.fixture
async def video_client(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "ti2v",
                "model_version": "2.2",
                "dual_model": False,
                "vae_stride": [4, 16, 16],
                "sample_fps": 24,
                "max_area": 704 * 1280,
                "dim": 3072,
                "text_dim": 4096,
            }
        )
    )
    for name in ("model.safetensors", "t5_encoder.safetensors", "vae.safetensors"):
        (model / name).write_bytes(b"weights")
    monkeypatch.setattr("omlx.video_jobs.require_video_runtime", lambda: None)
    pool = FakePool(model)
    jobs = VideoJobs(
        tmp_path / "videos", pool, lambda name: "wan" if name == "alias" else name
    )
    await jobs.start()
    app = FastAPI()
    app.state.video_jobs = jobs
    app.include_router(router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, jobs, pool
    await jobs.close()


async def test_models_status_reports_video_runtime_readiness(video_client, monkeypatch):
    _, jobs, _ = video_client
    state = ServerState()
    state.engine_pool = MagicMock()
    state.engine_pool.get_status.return_value = {"models": []}
    monkeypatch.setattr("omlx.server._server_state", state)
    monkeypatch.setattr("omlx.video_jobs.require_video_runtime", require_video_runtime)
    monkeypatch.setattr("omlx.utils.optional_deps.find_spec", lambda _: None)
    monkeypatch.setattr(app.state, "video_jobs", None)
    assert (await list_models_status())["video_generation_ready"] is False

    app.state.video_jobs = jobs
    assert (await list_models_status())["video_generation_ready"] is False

    monkeypatch.setattr("omlx.utils.optional_deps.find_spec", lambda _: object())
    for missing in ("ffmpeg", "ffprobe", None):
        paths = {
            tool: f"/bin/{tool}" for tool in ("ffmpeg", "ffprobe") if tool != missing
        }
        monkeypatch.setattr("omlx.utils.optional_deps.which", paths.get)
        assert (await list_models_status())["video_generation_ready"] is (
            missing is None
        )


async def test_create_poll_download_list_delete_and_recover(video_client):
    client, jobs, pool = video_client
    response = await client.post(
        "/v1/videos", json={"model": "alias", "prompt": "Clouds drift", "seed": 0}
    )
    assert response.status_code == 200
    queued = response.json()
    video_id = queued.pop("id")
    created_at = queued.pop("created_at")
    assert queued == {
        "object": "video",
        "model": "wan",
        "status": "queued",
        "progress": 0,
        "completed_at": None,
        "expires_at": None,
        "error": None,
        "prompt": "Clouds drift",
        "seconds": "4",
        "size": "1280x704",
        "remixed_from_video_id": None,
    }
    assert isinstance(created_at, int)
    await asyncio.wait_for(pool.engine.entered.wait(), 5)
    assert (await client.get(f"/v1/videos/{video_id}")).json()[
        "status"
    ] == "in_progress"
    assert (await client.get(f"/v1/videos/{video_id}/content")).status_code == 409
    assert (await client.delete(f"/v1/videos/{video_id}")).status_code == 409
    assert (pool.engine.plan.generation_frames, pool.engine.plan.output_frames) == (
        97,
        96,
    )
    assert pool.engine.plan.seed == 0
    pool.engine.finish.set()
    await asyncio.wait_for(jobs._pending.join(), 5)
    completed = (await client.get(f"/v1/videos/{video_id}")).json()
    assert completed["status"] == "completed"
    assert completed["progress"] == 100
    assert completed["completed_at"] >= created_at
    downloaded = await client.get(f"/v1/videos/{video_id}/content")
    assert downloaded.content == b"verified-mp4"
    assert downloaded.headers["content-type"] == "video/mp4"
    assert (await client.get("/v1/videos")).json() == {
        "object": "list",
        "data": [completed],
        "has_more": False,
        "first_id": video_id,
        "last_id": video_id,
    }
    await jobs.close()
    recovered = VideoJobs(jobs.directory, pool, lambda model: model)
    await recovered.start()
    assert recovered.get(video_id).model_dump() == completed
    await recovered.close()
    deleted = await client.delete(f"/v1/videos/{video_id}")
    assert deleted.json() == {
        "id": video_id,
        "object": "video.deleted",
        "deleted": True,
    }
    assert (await client.get(f"/v1/videos/{video_id}")).status_code == 404


async def test_multipart_reference_outlives_request_and_failure_is_durable(
    video_client,
):
    client, jobs, pool = video_client
    image = BytesIO()
    Image.new("RGB", (32, 32), color=(17, 18, 19)).save(image, format="PNG")
    pool.engine.failure = True
    response = await client.post(
        "/v1/videos",
        data={"model": "wan", "prompt": "Animate", "seconds": "8", "size": "640x352"},
        files={"input_reference": ("frame.png", image.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    await asyncio.wait_for(pool.engine.entered.wait(), 5)
    assert pool.engine.reference == (17, 18, 19)
    assert pool.engine.plan.output_frames == 192
    pool.engine.finish.set()
    await asyncio.wait_for(jobs._pending.join(), 5)
    failed = jobs.get(response.json()["id"])
    assert failed.status == "failed"
    assert failed.error.model_dump() == {
        "code": "generation_failed",
        "message": "Video generation failed. See the server and job logs.",
    }
    assert pool.engine.leased is False


async def test_failed_reference_save_removes_unsubmitted_job(video_client, monkeypatch):
    client, jobs, _ = video_client
    image = BytesIO()
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(image, format="PNG")

    def fail_save(self, destination, *args, **kwargs):
        destination.write_bytes(b"partial image")
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        await client.post(
            "/v1/videos",
            data={"model": "wan", "prompt": "Animate"},
            files={"input_reference": ("frame.png", image.getvalue(), "image/png")},
        )

    assert jobs.list(limit=20, order="asc", after=None).data == []
    assert list(jobs.directory.glob("video_*")) == []


async def test_failed_job_persistence_removes_unsubmitted_job(
    video_client, monkeypatch
):
    import omlx.video_jobs as video_jobs

    client, jobs, _ = video_client
    image = BytesIO()
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(image, format="PNG")
    sync_directory = video_jobs._sync_directory

    def fail_job_directory_sync(directory):
        if directory.parent == jobs.directory and directory.name.startswith("video_"):
            raise OSError("disk full")
        sync_directory(directory)

    monkeypatch.setattr(video_jobs, "_sync_directory", fail_job_directory_sync)
    with pytest.raises(OSError, match="disk full"):
        await client.post(
            "/v1/videos",
            data={"model": "wan", "prompt": "Animate"},
            files={"input_reference": ("frame.png", image.getvalue(), "image/png")},
        )

    assert jobs.list(limit=20, order="asc", after=None).data == []
    assert jobs._pending.empty()
    assert list(jobs.directory.glob("video_*")) == []


async def test_reference_normalization_is_offloaded_and_waited_on_cancellation(
    video_client, monkeypatch
):
    import omlx.video_jobs as video_jobs

    client, jobs, _ = video_client
    entered = Event()
    release = Event()
    source_paths = []
    source_exists_during_normalization = []
    normalize = video_jobs._normalize_reference

    def blocked_normalize(source, destination):
        source_paths.append(source)
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release image normalization")
        source_exists_during_normalization.append(source.exists())
        normalize(source, destination)

    monkeypatch.setattr(video_jobs, "_normalize_reference", blocked_normalize)
    image = BytesIO()
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(image, format="PNG")
    task = asyncio.create_task(
        client.post(
            "/v1/videos",
            data={"model": "wan", "prompt": "Animate"},
            files={"input_reference": ("frame.png", image.getvalue(), "image/png")},
        )
    )
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), 6)
        loop = asyncio.get_running_loop()
        heartbeat = asyncio.Event()
        loop.call_soon(heartbeat.set)
        await asyncio.wait_for(heartbeat.wait(), 1)
        assert source_paths[0].is_file()

        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=0.05)
        assert not done
        assert source_paths[0].is_file()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert source_exists_during_normalization == [True]
    assert not source_paths[0].exists()
    assert jobs.list(limit=20, order="asc", after=None).data == []


async def test_reference_normalization_rechecks_admission_after_await(
    video_client, monkeypatch
):
    import omlx.video_jobs as video_jobs

    client, jobs, _ = video_client
    entered = Event()
    release = Event()
    normalize = video_jobs._normalize_reference

    def blocked_normalize(source, destination):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release image normalization")
        normalize(source, destination)

    monkeypatch.setattr(video_jobs, "_normalize_reference", blocked_normalize)
    image = BytesIO()
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(image, format="PNG")
    task = asyncio.create_task(
        client.post(
            "/v1/videos",
            data={"model": "wan", "prompt": "Animate"},
            files={"input_reference": ("frame.png", image.getvalue(), "image/png")},
        )
    )
    assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), 6)
    await jobs.close()
    release.set()
    response = await task
    assert response.status_code == 503
    assert jobs._pending.empty()
    assert jobs.list(limit=20, order="asc", after=None).data == []
    assert list(jobs.directory.glob("video_*")) == []


async def test_memory_enforcer_wakes_for_each_video_dispatch(video_client):
    client, jobs, pool = video_client
    first = await client.post("/v1/videos", json={"model": "wan", "prompt": "First"})
    await asyncio.wait_for(pool.engine.entered.wait(), 5)
    assert pool.wake_calls == [True]
    pool.engine.finish.set()
    await asyncio.wait_for(jobs._pending.join(), 5)

    pool.engine.entered.clear()
    pool.engine.finish.clear()
    second = await client.post("/v1/videos", json={"model": "wan", "prompt": "Second"})
    await asyncio.wait_for(pool.engine.entered.wait(), 5)
    assert pool.wake_calls == [True, True]
    pool.engine.finish.set()
    await asyncio.wait_for(jobs._pending.join(), 5)
    assert jobs.get(first.json()["id"]).status == "completed"
    assert jobs.get(second.json()["id"]).status == "completed"


@pytest.mark.parametrize(
    "fields,status",
    [
        ({"size": "1280x720"}, 400),
        ({"size": "../video"}, 422),
        ({"size": "1280x1280"}, 400),
        ({"seconds": "3"}, 422),
        ({"steps": 0}, 422),
        ({"n": 2}, 422),
        ({"prompt": " "}, 422),
        ({"model": "missing"}, 404),
    ],
)
async def test_invalid_request_never_runs_backend(video_client, fields, status):
    client, jobs, pool = video_client
    response = await client.post(
        "/v1/videos", json={"model": "wan", "prompt": "A lake", **fields}
    )
    assert response.status_code == status
    assert pool.engine.entered.is_set() is False
    assert jobs.list(limit=20, order="asc", after=None).data == []


async def test_shutdown_and_restart_fail_interrupted_work(video_client):
    client, jobs, pool = video_client
    response = await client.post(
        "/v1/videos", json={"model": "wan", "prompt": "A lake"}
    )
    await asyncio.wait_for(pool.engine.entered.wait(), 5)
    await jobs.close()
    video_id = response.json()["id"]
    assert jobs.get(video_id).error.code == "server_shutdown"
    assert pool.engine.leased is False
    record = jobs.directory / video_id / "job.json"
    data = json.loads(record.read_text())
    data["video"]["status"] = "in_progress"
    data["video"]["error"] = None
    record.write_text(json.dumps(data))
    recovered = VideoJobs(jobs.directory, pool, lambda model: model)
    await recovered.start()
    assert recovered.get(video_id).error.code == "server_restarted"
    await recovered.close()


async def test_queued_job_rejects_a_changed_checkpoint(video_client, tmp_path):
    client, jobs, pool = video_client
    first = await client.post("/v1/videos", json={"model": "wan", "prompt": "First"})
    await asyncio.wait_for(pool.engine.entered.wait(), 5)
    second = await client.post("/v1/videos", json={"model": "wan", "prompt": "Second"})
    pool.engine._model_name = str(tmp_path / "replacement")
    pool.engine.finish.set()
    await asyncio.wait_for(jobs._pending.join(), 5)
    assert jobs.get(first.json()["id"]).status == "completed"
    failed = jobs.get(second.json()["id"])
    assert failed.status == "failed"
    assert failed.error.code == "generation_failed"
    assert pool.engine.plan.prompt == "First"


async def test_shutdown_stops_dispatch_after_teardown_failure(
    video_client, monkeypatch, caplog
):
    client, jobs, pool = video_client

    async def fail_on_cancel(plan, directory, owner_fd):
        pool.engine.entered.set()
        try:
            await pool.engine.finish.wait()
        except asyncio.CancelledError as error:
            raise RuntimeError("worker teardown failed") from error

    monkeypatch.setattr(pool.engine, "generate", fail_on_cancel)
    first = await client.post("/v1/videos", json={"model": "wan", "prompt": "First"})
    await asyncio.wait_for(pool.engine.entered.wait(), 5)
    second = await client.post("/v1/videos", json={"model": "wan", "prompt": "Second"})
    await asyncio.wait_for(jobs.close(), 5)
    assert jobs.get(first.json()["id"]).error.code == "generation_failed"
    assert jobs.get(second.json()["id"]).error.code == "server_shutdown"
    assert "worker teardown failed" in caplog.text


async def test_dispatcher_persistence_failure_stops_admission_and_releases_owner(
    video_client, monkeypatch, caplog
):
    client, jobs, pool = video_client
    response = await client.post(
        "/v1/videos", json={"model": "wan", "prompt": "A lake"}
    )
    await asyncio.wait_for(pool.engine.entered.wait(), 5)
    original_save = jobs._save

    def fail_save(stored):
        raise OSError("No space left on device")

    monkeypatch.setattr(jobs, "_save", fail_save)
    pool.engine.finish.set()
    result = (await asyncio.gather(jobs._worker, return_exceptions=True))[0]
    assert isinstance(result, OSError)
    assert response.status_code == 200
    assert (await client.get(f"/v1/videos/{response.json()['id']}")).status_code == 503
    assert (
        await client.post("/v1/videos", json={"model": "wan", "prompt": "Another lake"})
    ).status_code == 503
    await jobs.close()
    assert "Video dispatcher stopped with an error" in caplog.text
    monkeypatch.setattr(jobs, "_save", original_save)
    restarted = VideoJobs(jobs.directory, pool, lambda model: model)
    await restarted.start()
    assert restarted.get(response.json()["id"]).error.code == "server_restarted"
    await restarted.close()
