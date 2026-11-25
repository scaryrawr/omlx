# SPDX-License-Identifier: Apache-2.0
"""Video subprocess ownership and encoded output checks."""

import asyncio
import json
import shutil
import signal
import subprocess
import sys
from contextlib import suppress
from types import SimpleNamespace

import pytest

from omlx.engine.video import VideoEngine
from omlx.video_registry import VideoRenderPlan, plan_video
from omlx.video_worker import finalize_video


def _plan(tmp_path):
    return VideoRenderPlan(
        family="wan",
        model_path=str(tmp_path),
        prompt="A lake",
        width=32,
        height=32,
        fps=24,
        output_frames=96,
        generation_frames=97,
        seed=0,
    )


def _ltx_checkpoint(root):
    model = root / "LTX-2-distilled"
    for name in (
        "transformer/config.json",
        "vae/decoder/config.json",
        "vae/encoder/config.json",
        "text_encoder/config.json",
    ):
        (model / name).parent.mkdir(parents=True, exist_ok=True)
        (model / name).write_text(json.dumps({"model_type": "ltx av model"}))
    (model / "text_projections").mkdir()
    (model / "text_projections/model.safetensors").write_bytes(b"weights")
    (model / "ltx-2-spatial-upscaler-x2-1.0.safetensors").write_bytes(b"weights")
    return model


def test_distilled_ltx_plan_uses_its_frame_and_size_grid(tmp_path):
    model = _ltx_checkpoint(tmp_path)
    request = dict(prompt="A lake", seconds="4", seed=0, has_reference=False)
    plan = plan_video(model, size="1280x704", **request)
    assert (plan.family, plan.pipeline, plan.fps) == ("ltx2", "distilled", 24)
    assert (plan.output_frames, plan.generation_frames) == (96, 97)
    with pytest.raises(ValueError, match="64x64"):
        plan_video(model, size="1280x720", **request)
    with pytest.raises(ValueError, match="distilled schedule"):
        plan_video(model, size="1280x704", steps=8, **request)


def test_finalize_video_has_exact_dimensions_and_duration(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:size=32x32:rate=24",
            "-frames:v",
            "97",
            "-c:v",
            "libx264",
            str(tmp_path / "generated.mp4"),
        ],
        check=True,
    )
    finalize_video(_plan(tmp_path), tmp_path)
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,duration",
            "-of",
            "json",
            str(tmp_path / "video.mp4"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["streams"] == [
        {
            "width": 32,
            "height": 32,
            "duration": "4.000000",
            "nb_frames": "96",
        }
    ]


def test_worker_exits_cleanly_with_parent_pipe_open(tmp_path):
    spec = tmp_path / "render.json"
    spec.write_text(_plan(tmp_path).model_dump_json())
    code = """
import io
import os
import sys
import threading
from pathlib import Path

from omlx import video_worker

started = threading.Event()

class Input(io.RawIOBase):
    def readable(self):
        return True

    def fileno(self):
        started.set()
        return 0

    def readinto(self, buffer):
        started.set()
        data = os.read(0, len(buffer))
        buffer[:len(data)] = data
        return len(data)

def generate(**kwargs):
    if not started.wait(10):
        raise RuntimeError("parent watcher did not start")
    Path(kwargs["output_path"]).write_bytes(b"generated")

def finalize(plan, directory):
    (directory / "video.mp4").write_bytes(b"completed")

sys.stdin = io.TextIOWrapper(io.BufferedReader(Input()))
started.clear()
sys.argv = ["omlx.video_worker", sys.argv[1]]
video_worker.generate_wan = generate
video_worker.finalize_video = finalize
video_worker.main()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(spec)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=20)
        assert returncode == 0, process.stderr.read().decode()
        assert (tmp_path / "video.mp4").read_bytes() == b"completed"
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        process.stdin.close()
        process.stderr.close()


@pytest.mark.parametrize("layer_name", ["ResidualBlock", "Resample"])
def test_wan_worker_reclaims_decoder_cache_without_changing_output(tmp_path, layer_name):
    pytest.importorskip("mlx_video")
    spec = tmp_path / "render.json"
    spec.write_text(_plan(tmp_path).model_dump_json())
    code = """
import sys
from pathlib import Path

import mlx.core as mx
from mlx_video.models.wan_2 import vae22
from omlx import video_worker

mx.set_cache_limit(mx.device_info()["memory_size"])
if sys.argv[2] == "ResidualBlock":
    layer = vae22.ResidualBlock(4, 4)
else:
    layer = vae22.Resample(4, "upsample2d")
x = mx.random.normal((1, 2, 4, 4, 4))
expected = layer(x)
mx.eval(expected)
mx.synchronize()
mx.clear_cache()

def generate(**kwargs):
    buffer = mx.ones((1024, 1024))
    mx.eval(buffer)
    del buffer
    result = layer(x)
    mx.eval(result)
    assert mx.get_cache_memory() == 0
    assert mx.array_equal(result, expected).item()
    Path(kwargs["output_path"]).write_bytes(b"generated")

def finalize(plan, directory):
    (directory / "video.mp4").write_bytes(b"completed")

sys.argv = ["omlx.video_worker", sys.argv[1]]
video_worker.generate_wan = generate
video_worker.finalize_video = finalize
video_worker.main()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(spec), layer_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=30)
        assert returncode == 0, process.stderr.read().decode()
        assert (tmp_path / "video.mp4").read_bytes() == b"completed"
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        process.stdin.close()
        process.stderr.close()


@pytest.mark.parametrize("reason", ["cancel", "memory"])
async def test_abort_reaps_worker_before_releasing_activity(tmp_path, monkeypatch, reason):
    engine = VideoEngine(str(tmp_path))
    original = asyncio.create_subprocess_exec
    entered = asyncio.Event()
    processes = []
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("OMLX_API_KEY", "test-key-not-for-worker")

    async def spawn(*args, **kwargs):
        assert kwargs["env"]["PYTHONPATH"] == str(tmp_path)
        assert "OMLX_API_KEY" not in kwargs["env"]
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        process = await original(
            sys.executable,
            "-c",
            "import sys; sys.stdin.buffer.read()",
            **kwargs,
        )
        wait = process.wait

        async def observed_wait():
            entered.set()
            return await wait()

        monkeypatch.setattr(process, "wait", observed_wait)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    with (tmp_path / ".owner").open("a+b") as owner:
        task = asyncio.create_task(
            engine.generate(_plan(tmp_path), tmp_path, owner.fileno())
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert engine.active_request_count() == 1
            if reason == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 5)
            else:
                assert await engine.abort_all_requests() == 1
                with pytest.raises(RuntimeError, match="Video worker exited with code -15"):
                    await asyncio.wait_for(task, 5)
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
    assert processes[0].returncode == -signal.SIGTERM
    assert engine.active_request_count() == 0
    assert engine.process_memory_bytes() == 0


def test_video_workers_participate_in_external_memory_accounting(tmp_path, monkeypatch):
    from omlx.engine_pool import EnginePool

    engine = VideoEngine(str(tmp_path))
    monkeypatch.setattr(engine, "process_memory_bytes", lambda: 123456)
    pool = EnginePool.__new__(EnginePool)
    pool._entries = {"wan": SimpleNamespace(engine=engine)}
    pool._starting_external = {}
    assert pool.external_process_memory_bytes() == 123456


@pytest.mark.parametrize("fails", [False, True])
async def test_video_unload_reports_its_engine_type(tmp_path, monkeypatch, caplog, fails):
    from omlx.engine_pool import EngineEntry, EnginePool

    engine = VideoEngine(str(tmp_path))
    pool = EnginePool()
    pool._entries["wan"] = EngineEntry(
        model_id="wan",
        model_path=str(tmp_path),
        model_type="video",
        engine_type="video",
        estimated_size=1024,
        engine=engine,
    )
    pool._current_model_memory = 1024

    async def fail_stop():
        raise RuntimeError("teardown failed")

    if fails:
        monkeypatch.setattr(engine, "stop", fail_stop)
    with caplog.at_level("INFO", logger="omlx.engine_pool"):
        if fails:
            with pytest.raises(RuntimeError, match="teardown failed"):
                await pool._stop_and_unload_engine("wan")
            assert pool._entries["wan"].engine is engine
            assert (
                "Video teardown failed for wan (teardown failed); "
                "keeping the engine registered for retry"
            ) in caplog.messages
        else:
            await pool._stop_and_unload_engine("wan")
            assert pool._entries["wan"].engine is None
            assert (
                "Unloaded video model: wan, released its process (1.00KB planned)"
                in caplog.messages
            )
