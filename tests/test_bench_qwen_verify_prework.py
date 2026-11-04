# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).parents[1] / "benchmarks" / "bench_qwen_verify_prework.py"
    spec = importlib.util.spec_from_file_location("bench_qwen_verify_prework", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_schedule_warms_each_arm_then_runs_abba(benchmark):
    assert benchmark._schedule(2) == [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
        (False, True),
        (False, False),
        (False, False),
        (False, True),
        (False, True),
        (False, False),
    ]


def test_default_prompt_preserves_literal_escapes(benchmark):
    assert benchmark.PROMPT_TEXT == (
        r"Complete this Python module. Continue with code only and do not stop "
        r"until the implementation is complete.\n\n"
    )
    args = benchmark._parse_args(["model"])
    assert args.prompt_text == benchmark.PROMPT_TEXT
    assert args.pp == 512
    assert args.gen == 128
    assert args.pairs == 3
    assert args.fixed_depth == 2


@pytest.mark.parametrize("depth", [1, 2])
def test_fixed_depth_accepts_only_supported_values(benchmark, depth):
    assert (
        benchmark._parse_args(["model", "--fixed-depth", str(depth)]).fixed_depth
        == depth
    )

    with pytest.raises(SystemExit):
        benchmark._parse_args(["model", "--fixed-depth", "3"])


def test_engagement_rejection_keeps_metadata(benchmark):
    record = {"treatment": True, "draft_calls": 1, "prework_calls": 0}

    with pytest.raises(benchmark.bench.BenchmarkError, match="prework"):
        benchmark._validate_engagement(record)

    assert record["engagement_errors"] == ["prework treatment was not engaged"]


def test_baseline_cannot_engage_candidate(benchmark):
    record = {"treatment": False, "draft_calls": 1, "prework_calls": 1}
    with pytest.raises(benchmark.bench.BenchmarkError, match="baseline"):
        benchmark._validate_engagement(record)


def test_output_drift_is_rejected(benchmark):
    trial = SimpleNamespace(
        index=0, errors=(), prompt_tokens=2, output_token_ids=(1, 2)
    )
    assert benchmark.bench._validate_trial(trial, 2, None) == (1, 2)
    trial.output_token_ids = (1, 3)
    with pytest.raises(benchmark.bench.BenchmarkError, match="differs"):
        benchmark.bench._validate_trial(trial, 2, (1, 2))


@pytest.mark.parametrize("fixed_depth", [1, 2])
@pytest.mark.parametrize(
    "failure", [None, "not ready", "trial failed", "fingerprint failed", "stop failed"]
)
def test_fixed_depth_and_monkeypatches_restore(
    benchmark, monkeypatch, fixed_depth, failure
):
    monkeypatch.setenv(benchmark.PREWORK_ENV, "original")

    class Controller:
        def observe(self):
            self.cur = 1

        def should_exit(self):
            return True

    original_observe = Controller.observe
    original_should_exit = Controller.should_exit

    class Prework:
        @staticmethod
        def gdn_prework_fused():
            return None

    original_prework = Prework.gdn_prework_fused
    head = SimpleNamespace(parameters=lambda: {"weight": SimpleNamespace()})
    language_model = SimpleNamespace(
        get_mtp_module=lambda: head,
        _omlx_mtp_decode_enabled=True,
        _omlx_mtp_depth=0,
    )
    if failure == "not ready":
        language_model._omlx_mtp_decode_enabled = False
        del language_model._omlx_mtp_depth

    class Engine:
        def __init__(self, **kwargs):
            self.model_settings = kwargs["model_settings"]
            self._adapter = SimpleNamespace(
                _language_model=language_model,
                mtp_forward=lambda: None,
            )
            self.tokenizer = object()

        async def start(self):
            pass

        async def stop(self):
            self.stopped = True
            if failure == "stop failed":
                raise RuntimeError(failure)

    class MX:
        random = SimpleNamespace(seed=lambda _seed: None)

    monkeypatch.setattr(
        benchmark.bench, "_model_fingerprint", lambda _model: {"sha256": "x"}
    )
    if failure == "fingerprint failed":

        def failed_fingerprint(_model):
            raise RuntimeError(failure)

        monkeypatch.setattr(benchmark.bench, "_model_fingerprint", failed_fingerprint)
    monkeypatch.setattr(benchmark.bench, "_runtime_info", lambda _mx: {})
    monkeypatch.setattr(
        benchmark.bench, "_model_settings", lambda _enabled: SimpleNamespace()
    )
    monkeypatch.setattr(benchmark.bench, "_build_prompt", lambda *_args: (10, 11))
    monkeypatch.setattr(benchmark.bench, "_token_digest", lambda _ids: "prompt")
    monkeypatch.setattr(benchmark.bench, "_assert_model_unchanged", lambda *_args: None)

    async def run_trial(*_args, index, warmup):
        if failure == "trial failed":
            raise RuntimeError(failure)
        controller = Controller()
        controller.max_depth = 4
        controller.observe()
        assert controller.cur == fixed_depth
        assert controller.should_exit() is False
        Engine.instance._adapter.mtp_forward()
        if os.environ[benchmark.PREWORK_ENV] == "1":
            Prework.gdn_prework_fused()
        return benchmark.bench.Trial(
            index=index,
            warmup=warmup,
            prompt_tokens=2,
            completion_tokens=2,
            cached_tokens=0,
            finish_reason="length",
            output_token_ids=(1, 2),
            output_sha256="same",
            producer_first_s=1.0,
            producer_last_s=2.0,
            producer_duration_s=1.0,
            consumer_duration_s=1.0,
            generation_tps=1.0,
            end_to_end_tps=2.0,
            peak_memory_bytes=1,
            errors=(),
        )

    monkeypatch.setattr(benchmark.bench, "_run_trial", run_trial)
    original_init = Engine.__init__

    def init(self, **kwargs):
        original_init(self, **kwargs)
        Engine.instance = self

    Engine.__init__ = init
    deps = {
        "mx": MX,
        "tree_flatten": lambda _params: [
            ("weight", SimpleNamespace(dtype="bfloat16", shape=(2, 2)))
        ],
        "engine_cls": Engine,
        "prework": Prework,
        "depth_controller": Controller,
        "scheduler_config": lambda **kwargs: kwargs,
    }
    args = SimpleNamespace(
        model="model",
        pp=2,
        gen=2,
        pairs=1,
        fixed_depth=fixed_depth,
        prompt_text="p",
    )

    result = asyncio.run(benchmark._run(args, deps))

    assert os.environ[benchmark.PREWORK_ENV] == "original"
    assert Controller.observe is original_observe
    assert Controller.should_exit is original_should_exit
    assert Prework.gdn_prework_fused is original_prework
    if failure == "not ready":
        assert not hasattr(language_model, "_omlx_mtp_depth")
    else:
        assert language_model._omlx_mtp_depth == 0
    if failure != "fingerprint failed":
        assert Engine.instance.stopped
    if failure:
        assert result["status"] == "error"
        expected_error = "did not become ready" if failure == "not ready" else failure
        assert expected_error in result["error"]
        return

    assert result["status"] == "ok"
    assert result["fixed_depth"] == fixed_depth
    assert Engine.instance.model_settings.mtp_num_draft_tokens == fixed_depth
    assert result["outputs_identical"] is True
    assert [record["treatment"] for record in result["trials"]] == [
        False,
        True,
        False,
        True,
        True,
        False,
    ]
    assert all(record["draft_calls"] == 1 for record in result["trials"])
    assert [record["prework_calls"] for record in result["trials"]] == [
        0,
        1,
        0,
        1,
        1,
        0,
    ]
