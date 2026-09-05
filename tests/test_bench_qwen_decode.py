# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
import sys
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
    monkeypatch.setenv("OMLX_API_KEY", "secret")

    environment = benchmark._candidate_environment()

    assert environment == {"OMLX_QWEN35_GDN_DECODE_FUSION": "0"}


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
    assert settings.dflash_in_memory_cache is False
    assert settings.dflash_ssd_cache is False
    assert settings.specprefill_enabled is False
    assert settings.turboquant_kv_enabled is False


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
