# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).parents[1] / "benchmarks" / "bench_qwen_decode.py"
    spec = importlib.util.spec_from_file_location("bench_qwen_decode", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_prompt_is_exact_and_deterministic(benchmark, monkeypatch):
    monkeypatch.setattr(
        "omlx.admin.benchmark._load_bench_corpus",
        lambda _profile: "def solve(value):\n    return value + 1\n",
    )

    class Tokenizer:
        def encode(self, text):
            return [ord(char) for char in text]

    first = benchmark._build_prompt(Tokenizer(), 512, "Implement this module.\n")
    second = benchmark._build_prompt(Tokenizer(), 512, "Implement this module.\n")

    assert isinstance(first, tuple)
    assert len(first) == 512
    assert first == second
    assert benchmark._token_digest(first) == benchmark._token_digest(second)


@pytest.mark.parametrize(
    "model_type",
    ["qwen3_5", "qwen3_5_moe", "qwen3_6", "qwen3_8", "qwen4_exp"],
)
def test_model_fingerprint_accepts_supported_qwen_types(
    benchmark, tmp_path, model_type
):
    (tmp_path / "config.json").write_text(
        f'{{"model_type": "{model_type}", '
        f'"text_config": {{"model_type": "{model_type}_text"}}}}'
    )
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    fingerprint = benchmark._model_fingerprint(str(tmp_path))

    assert fingerprint["resolved_path"] == str(tmp_path.resolve())
    assert fingerprint["model_type"] == model_type
    assert fingerprint["config_sha256"]
    assert fingerprint["sha256"]


@pytest.mark.parametrize("model_type", ["inkling2", "inkling2_loop"])
def test_model_fingerprint_includes_inkling2_draft(benchmark, tmp_path, model_type):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (tmp_path / "model.safetensors").write_bytes(b"target")
    draft = tmp_path / "mtp.safetensors"
    draft.write_bytes(b"draft")
    before = benchmark._model_fingerprint(str(tmp_path))

    assert before["model_type"] == model_type
    assert "mtp.safetensors" in {item["path"] for item in before["files"]}
    draft.write_bytes(b"changed draft")

    with pytest.raises(benchmark.BenchmarkError, match="changed"):
        benchmark._assert_model_unchanged(
            before, benchmark._model_fingerprint(str(tmp_path))
        )


def test_model_fingerprint_detects_metadata_drift(benchmark, tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5"}')
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    before = benchmark._model_fingerprint(str(tmp_path))

    weights.write_bytes(b"changed weights")
    after = benchmark._model_fingerprint(str(tmp_path))

    with pytest.raises(benchmark.BenchmarkError, match="changed"):
        benchmark._assert_model_unchanged(before, after)


def test_model_fingerprint_rejects_unsupported_type_before_loading(benchmark, tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "llama"}')
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(benchmark.BenchmarkError, match="unsupported model_type"):
        benchmark._model_fingerprint(str(tmp_path))


def test_candidate_environment_records_flags_not_api_keys(benchmark, monkeypatch):
    for name in benchmark._candidate_environment():
        monkeypatch.delenv(name)
    monkeypatch.setenv("OMLX_QWEN35_GDN_DECODE_FUSION", "0")
    monkeypatch.setenv("OMLX_INKLING2_TEST_CANDIDATE", "1")
    monkeypatch.setenv("OMLX_API_KEY", "secret")

    environment = benchmark._candidate_environment()

    assert environment == {
        "OMLX_INKLING2_TEST_CANDIDATE": "1",
        "OMLX_QWEN35_GDN_DECODE_FUSION": "0",
    }


def test_runtime_info_preserves_device_telemetry_error(benchmark):
    mlx = SimpleNamespace(
        __version__="test",
        device_info=lambda: (_ for _ in ()).throw(RuntimeError("device failed")),
    )

    with pytest.raises(RuntimeError, match="device failed"):
        benchmark._runtime_info(mlx)


class _FakeMLX:
    def reset_peak_memory(self):
        pass

    def get_peak_memory(self):
        return 1234


class _FakeEngine:
    def __init__(self, outputs):
        self.outputs = outputs

    async def stream_generate(self, **kwargs):
        assert kwargs["prompt"] == [10, 11]
        assert kwargs["skip_cache_store"] is True
        for output in self.outputs:
            yield output


def _output(**overrides):
    values = {
        "tokens": [20, 21, 22],
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "cached_tokens": 0,
        "finish_reason": "length",
        "finished": True,
        "generated_at": 10.0,
        "generated_until": 10.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeSampler:
    def __init__(self, telemetry=None):
        self.telemetry = telemetry
        self.windows = []
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def window(self, t0, t1):
        assert self.started and not self.stopped
        self.windows.append((t0, t1))
        return self.telemetry


def test_trial_uses_terminal_ids_and_producer_timestamps(benchmark):
    trial = asyncio.run(
        benchmark._run_trial(
            _FakeEngine([_output()]),
            _FakeMLX(),
            (10, 11),
            3,
            index=0,
            warmup=False,
        )
    )

    assert trial.output_token_ids == (20, 21, 22)
    assert trial.producer_duration_s == 0.5
    assert trial.generation_tps == 4.0
    assert trial.errors == ()


def test_trial_ttft_ignores_prefill_updates_and_windows_use_monotonic_time(
    benchmark, monkeypatch
):
    counter = iter([100.0, 100.25, 101.0])
    monotonic = iter([200.0, 201.0])
    monkeypatch.setattr(
        benchmark,
        "time",
        SimpleNamespace(
            perf_counter=lambda: next(counter),
            monotonic=lambda: next(monotonic),
        ),
    )
    telemetry = {
        "sample_count": 4,
        "interval_s": 0.25,
        "thermal": {"start": 0, "max": 0},
        "gpu": {"util_avg": 90.0, "util_max": 100.0},
    }
    sampler = _FakeSampler(telemetry)
    sampler.start()
    outputs = [
        _output(
            tokens=[],
            completion_tokens=0,
            finished=False,
            generated_at=None,
            generated_until=None,
        ),
        _output(tokens=[], completion_tokens=1, finished=False, generated_until=10.0),
        _output(),
    ]

    trial = asyncio.run(
        benchmark._run_trial(
            _FakeEngine(outputs),
            _FakeMLX(),
            (10, 11),
            3,
            index=0,
            warmup=False,
            sampler=sampler,
        )
    )

    assert trial.consumer_ttft_s == 0.25
    assert trial.consumer_duration_s == 1.0
    assert trial.producer_duration_s == 0.5
    assert trial.generation_tps == 4.0
    assert trial.output_token_ids == (20, 21, 22)
    assert sampler.windows == [(200.0, 201.0)]
    assert trial.system_telemetry == telemetry
    assert trial.telemetry_status == "available"
    assert trial.thermal_status == "nominal"


@pytest.mark.parametrize(
    ("telemetry", "status"),
    [(None, "missing"), ({"sample_count": 2, "interval_s": 0.25}, "available")],
)
def test_trial_missing_thermals_are_not_nominal(benchmark, telemetry, status):
    sampler = _FakeSampler(telemetry)
    sampler.start()
    trial = asyncio.run(
        benchmark._run_trial(
            _FakeEngine([_output()]),
            _FakeMLX(),
            (10, 11),
            3,
            index=0,
            warmup=False,
            sampler=sampler,
        )
    )

    assert trial.telemetry_status == status
    assert trial.thermal_status == "missing"
    assert trial.system_telemetry == telemetry
    assert trial.errors == ("missing thermal observations invalidate comparison",)
    with pytest.raises(benchmark.BenchmarkError, match="missing thermal"):
        benchmark._validate_trial(trial, 2, None)


def test_empty_trial_retains_missing_ttft_and_errors(benchmark):
    trial = asyncio.run(
        benchmark._run_trial(
            _FakeEngine([]),
            _FakeMLX(),
            (10, 11),
            3,
            index=0,
            warmup=False,
        )
    )

    assert trial.consumer_ttft_s is None
    assert trial.output_token_ids == ()
    assert trial.errors == ("produced no output",)


@pytest.mark.parametrize("method", ["reset_peak_memory", "get_peak_memory"])
def test_trial_preserves_telemetry_errors(benchmark, method):
    class BrokenMLX(_FakeMLX):
        def reset_peak_memory(self):
            if method == "reset_peak_memory":
                raise RuntimeError("reset failed")

        def get_peak_memory(self):
            if method == "get_peak_memory":
                raise RuntimeError("read failed")
            return super().get_peak_memory()

    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(
            benchmark._run_trial(
                _FakeEngine([_output()]),
                BrokenMLX(),
                (10, 11),
                3,
                index=0,
                warmup=False,
            )
        )


def test_early_eos_trial_is_retained_before_failure(benchmark):
    trial = asyncio.run(
        benchmark._run_trial(
            _FakeEngine([_output(tokens=[20, 21], completion_tokens=2)]),
            _FakeMLX(),
            (10, 11),
            3,
            index=0,
            warmup=False,
        )
    )

    assert trial.output_token_ids == (20, 21)
    assert "ended early" in trial.errors[0]
    with pytest.raises(benchmark.BenchmarkError, match="ended early"):
        benchmark._validate_trial(trial, 2, None)


def test_repeated_output_mismatch_fails(benchmark):
    fields = {
        "warmup": False,
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "cached_tokens": 0,
        "finish_reason": "length",
        "producer_first_s": 1,
        "producer_last_s": 2,
        "producer_duration_s": 1,
        "consumer_duration_s": 2,
        "generation_tps": 2,
        "end_to_end_tps": 1.5,
        "peak_memory_bytes": 10,
        "errors": (),
    }
    first = benchmark.Trial(
        index=0,
        output_token_ids=(1, 2, 3),
        output_sha256="a",
        **fields,
    )
    second = benchmark.Trial(
        index=1,
        output_token_ids=(1, 2, 4),
        output_sha256="b",
        **fields,
    )

    expected = benchmark._validate_trial(first, 2, None)
    with pytest.raises(benchmark.BenchmarkError, match="differs"):
        benchmark._validate_trial(second, 2, expected)


def test_model_settings_disable_other_cache_and_speculative_paths(benchmark):
    settings = benchmark._model_settings(True)

    assert settings.mtp_enabled is True
    assert settings.vlm_mtp_enabled is False
    assert settings.dflash_enabled is False
    assert settings.specprefill_enabled is False
    assert settings.turboquant_kv_enabled is False


@pytest.mark.parametrize("depth", ["0", "-1", "1.5"])
def test_mtp_depth_requires_positive_integer(benchmark, depth):
    with pytest.raises(SystemExit):
        benchmark._parse_args(
            [
                "model",
                "--native-mtp",
                "on",
                "--label",
                "candidate",
                "--mtp-depth",
                depth,
            ]
        )


def test_mtp_depth_requires_native_mtp(benchmark):
    with pytest.raises(SystemExit):
        benchmark._parse_args(
            ["model", "--native-mtp", "off", "--label", "baseline", "--mtp-depth", "2"]
        )


@pytest.fixture
def serving_benchmark(benchmark, tmp_path, monkeypatch):
    import mlx.core as mx

    calls = []
    loader_calls = []
    target = object()
    sampler = _FakeSampler(
        {"sample_count": 2, "interval_s": 0.25, "thermal": {"start": 0, "max": 0}}
    )

    class Engine(_FakeEngine):
        def __init__(self):
            super().__init__([_output()])
            self.model_type = "inkling2"
            self.tokenizer = SimpleNamespace(encode=lambda _text: [10, 11])
            self.dflash_drafter = None

        async def start(self):
            calls.append("start")

        async def stop(self):
            calls.append("stop")

        def prepare_dflash_target(self):
            assert calls == ["start"]
            calls.append("prepare")
            return target

        def set_dflash_drafter(self, drafter):
            calls.append("attach")
            self.dflash_drafter = drafter

    engine = Engine()
    construction = {}

    def create_engine(**kwargs):
        construction.update(kwargs)
        return engine

    def load_drafter(path, model, depth, fixed):
        loader_calls.append((path, model, depth, fixed, threading.get_ident()))
        calls.append("load")
        return SimpleNamespace(
            kind="dflash", depth=min(depth or 7, 7), adaptive=not fixed
        )

    dspark = SimpleNamespace(
        bundled_inkling2_dspark_path=lambda path: Path(path) / "mtp.safetensors",
        load_inkling2_dspark_drafter=load_drafter,
    )
    monkeypatch.setitem(sys.modules, "omlx.patches.inkling2.dspark", dspark)
    monkeypatch.setattr("omlx.engine.vlm.VLMBatchedEngine", create_engine)
    monkeypatch.setattr("omlx.utils.system_sampler.SystemSampler", lambda: sampler)
    monkeypatch.setattr(
        "omlx.admin.benchmark._load_bench_corpus", lambda _profile: "code"
    )
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(mx, "get_peak_memory", lambda: 1234)
    monkeypatch.setattr(benchmark, "_runtime_info", lambda _mx: {"mocked": True})

    def args(model_type="inkling2", native_mtp="on", depth=None):
        engine.model_type = model_type
        (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
        (tmp_path / "model.safetensors").write_bytes(b"not real weights")
        argv = [
            str(tmp_path),
            "--native-mtp",
            native_mtp,
            "--label",
            "mocked",
            "--pp",
            "2",
            "--gen",
            "3",
            "--repeats",
            "2",
        ]
        if depth is not None:
            argv.extend(["--mtp-depth", str(depth)])
        return benchmark._parse_args(argv)

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr("omlx.engine_core.get_mlx_executor", lambda: executor)
        yield SimpleNamespace(
            engine=engine,
            sampler=sampler,
            dspark=dspark,
            args=args,
            calls=calls,
            loader_calls=loader_calls,
            target=target,
            construction=construction,
        )


@pytest.mark.parametrize("model_type", ["inkling2", "inkling2_loop"])
@pytest.mark.parametrize("depth", [None, 2, 99])
def test_inkling2_native_mtp_attaches_production_drafter(
    benchmark, serving_benchmark, model_type, depth
):
    fake = serving_benchmark
    args = fake.args(model_type, depth=depth)

    result = asyncio.run(benchmark._benchmark(args))

    assert result["status"] == "ok", result.get("error")
    assert fake.calls == ["start", "prepare", "load", "attach", "stop"]
    assert len(fake.loader_calls) == 1
    path, target, loaded_depth, fixed, thread_id = fake.loader_calls[0]
    assert (path, target, loaded_depth, fixed) == (
        args.model,
        fake.target,
        depth,
        depth is not None,
    )
    assert thread_id != threading.get_ident()
    assert result["speculative"]["dflash_drafter_attached"] is True
    assert result["speculative"]["draft_depth"] == min(depth or 7, 7)
    assert result["speculative"]["adaptive"] is (depth is None)
    assert result["speculative"]["activity"] == "unknown"
    assert result["speculative"]["counts"] is None
    assert result["config"]["mtp_depth"] == depth
    assert fake.construction["model_settings"].mtp_fixed_depth is None
    assert [trial["warmup"] for trial in result["trials"]] == [True, False, False]
    assert len(fake.sampler.windows) == 3
    assert all(t1 >= t0 for t0, t1 in fake.sampler.windows)
    assert fake.sampler.stopped
    assert result["trials"][0]["output_token_ids"] == (20, 21, 22)
    assert result["model_fingerprint_before"] == result["model_fingerprint_after"]


@pytest.mark.parametrize(
    ("model_type", "native_mtp"),
    [
        ("qwen3_5", "off"),
        ("qwen3_5", "on"),
        ("inkling2", "off"),
        ("inkling2_loop", "off"),
    ],
)
def test_other_serving_paths_do_not_load_dspark(
    benchmark, serving_benchmark, model_type, native_mtp
):
    fake = serving_benchmark
    result = asyncio.run(benchmark._benchmark(fake.args(model_type, native_mtp)))

    assert result["status"] == "ok", result.get("error")
    assert result["native_mtp"] == native_mtp
    assert fake.calls == ["start", "stop"]
    assert fake.loader_calls == []
    assert result["speculative"]["dflash_drafter_attached"] is False
    assert result["speculative"]["activity"] == "unknown"
    settings = fake.construction["model_settings"]
    assert settings.mtp_enabled is (native_mtp == "on")
    assert not settings.vlm_mtp_enabled
    assert not settings.dflash_enabled
    assert not settings.specprefill_enabled
    assert not settings.turboquant_kv_enabled
    assert fake.sampler.stopped


def test_qwen_rejects_mini_only_depth_before_loading(benchmark, serving_benchmark):
    fake = serving_benchmark
    result = asyncio.run(benchmark._benchmark(fake.args("qwen3_8", depth=2)))

    assert result["status"] == "error"
    assert "--mtp-depth is only supported for Inkling 2" in result["error"]
    assert fake.calls == []


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing", "requires a bundled DSpark draft"),
        ("loader", "drafter load failed"),
        ("none", "loader returned no drafter"),
        ("attachment", "attachment failed"),
        ("ignored", "drafter was not attached"),
    ],
)
def test_inkling2_cannot_silently_benchmark_baseline_when_mtp_requested(
    benchmark, serving_benchmark, monkeypatch, failure, message
):
    fake = serving_benchmark

    def fail(*_args):
        raise RuntimeError(message)

    if failure == "missing":
        monkeypatch.setattr(fake.dspark, "bundled_inkling2_dspark_path", lambda _: None)
    elif failure in ("loader", "none"):
        monkeypatch.setattr(
            fake.dspark,
            "load_inkling2_dspark_drafter",
            fail if failure == "loader" else lambda *_args: None,
        )
    else:
        monkeypatch.setattr(
            fake.engine,
            "set_dflash_drafter",
            fail if failure == "attachment" else lambda _drafter: None,
        )

    result = asyncio.run(benchmark._benchmark(fake.args()))

    assert result["status"] == "error"
    assert message in result["error"]
    assert result["trials"] == []
    assert fake.calls[-1] == "stop"
    assert not fake.sampler.started


@pytest.mark.parametrize("pressure", [1, 2, 3, 4])
def test_thermal_pressure_retains_invalid_trial_and_stops_sampler(
    benchmark, serving_benchmark, pressure
):
    fake = serving_benchmark
    fake.sampler.telemetry["thermal"] = {"start": 0, "max": pressure}

    result = asyncio.run(benchmark._benchmark(fake.args()))

    assert result["status"] == "error"
    assert len(result["trials"]) == 1
    trial = result["trials"][0]
    assert trial["thermal_status"] == "pressure"
    assert trial["system_telemetry"]["thermal"]["max"] == pressure
    assert trial["errors"] == (f"thermal pressure {pressure} invalidates comparison",)
    assert "invalidates comparison" in result["error"]
    assert fake.sampler.stopped
    assert fake.calls[-1] == "stop"


@pytest.mark.parametrize("failure", ["start", "window", "stop", "stream"])
def test_sampler_cleanup_and_errors_are_preserved(
    benchmark, serving_benchmark, monkeypatch, failure
):
    fake = serving_benchmark

    def fail(*_args):
        if failure == "stop":
            fake.sampler.stopped = True
        raise RuntimeError(f"{failure} failed")

    async def broken_stream(**_kwargs):
        raise RuntimeError("stream failed")
        yield

    if failure == "stream":
        monkeypatch.setattr(fake.engine, "stream_generate", broken_stream)
    else:
        monkeypatch.setattr(fake.sampler, failure, fail)

    result = asyncio.run(benchmark._benchmark(fake.args()))

    assert result["status"] == "error"
    assert f"{failure} failed" in result["error"]
    assert fake.sampler.stopped
    assert fake.calls[-1] == "stop"


def test_vlm_stream_projects_terminal_output_token_ids():
    from omlx.engine.vlm import VLMBatchedEngine

    engine = VLMBatchedEngine("model")
    engine._loaded = True
    engine._vlm_model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_8"))
    core = SimpleNamespace()

    async def add_request(**_kwargs):
        return "request"

    async def stream_outputs(_request_id):
        yield SimpleNamespace(
            output_text="a",
            new_text="a",
            output_token_ids=[],
            prompt_tokens=2,
            completion_tokens=1,
            finished=False,
            finish_reason=None,
            tool_calls=None,
            logprobs=None,
            cached_tokens=0,
        )
        yield SimpleNamespace(
            output_text="abc",
            new_text="bc",
            output_token_ids=[20, 21, 22],
            prompt_tokens=2,
            completion_tokens=3,
            finished=True,
            finish_reason="length",
            tool_calls=None,
            logprobs=None,
            cached_tokens=0,
        )

    async def abort_request(_request_id):
        raise AssertionError("completed request must not be aborted")

    core.add_request = add_request
    core.stream_outputs = stream_outputs
    core.abort_request = abort_request
    engine._engine = core

    async def collect():
        return [output async for output in engine.stream_generate([10, 11])]

    outputs = asyncio.run(collect())
    assert outputs[0].tokens == []
    assert outputs[-1].tokens == [20, 21, 22]


def test_main_keeps_stdout_as_json(benchmark, monkeypatch, capsys):
    async def fake_benchmark(_args):
        print("loader noise")
        return {"status": "error", "trials": [{"errors": ["early EOS"]}]}

    monkeypatch.setattr(benchmark, "_benchmark", fake_benchmark)

    exit_code = benchmark.main(["model", "--native-mtp", "off", "--label", "baseline"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ('{"status":"error","trials":[{"errors":["early EOS"]}]}\n')
    assert captured.err == "loader noise\n"


@pytest.mark.parametrize("pressure", [0, 1])
def test_main_serializes_mini_measurements_and_invalid_exit_status(
    benchmark, serving_benchmark, capsys, pressure
):
    fake = serving_benchmark
    args = fake.args()
    fake.sampler.telemetry["thermal"]["max"] = pressure

    exit_code = benchmark.main(
        [
            args.model,
            "--native-mtp",
            "on",
            "--label",
            "mini",
            "--pp",
            "2",
            "--gen",
            "3",
            "--mtp-depth",
            "2",
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert exit_code == pressure
    assert result["status"] == ("ok" if pressure == 0 else "error")
    assert result["schema_version"] == 1
    assert result["speculative"]["draft_depth"] == 2
    assert result["speculative"]["activity"] == "unknown"
    trial = result["trials"][0]
    assert trial["output_token_ids"] == [20, 21, 22]
    assert 0 <= trial["consumer_ttft_s"] <= trial["consumer_duration_s"]
    assert trial["producer_duration_s"] == 0.5
    assert trial["system_telemetry"] == fake.sampler.telemetry
    assert trial["thermal_status"] == ("nominal" if pressure == 0 else "pressure")
    assert fake.sampler.stopped
