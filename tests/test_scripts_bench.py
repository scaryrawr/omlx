# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def bench_module():
    path = Path(__file__).parents[1] / "scripts" / "bench.py"
    spec = importlib.util.spec_from_file_location("bench_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_parser_preserves_legacy_defaults(bench_module):
    args = bench_module._parse_args(["model-a"])

    assert args.models == ["model-a"]
    assert args.native_mtp is bench_module.NativeMTPMode.OFF
    assert args.qwen4_ple is None
    assert args.pp == [1024, 4096, 8192]
    assert args.gen == 128
    assert args.batch == []
    assert args.warmup == 1
    assert bench_module._allocate_labels(
        bench_module._expand_cases(args.models, args.native_mtp),
        include_variant=False,
    ) == ["model-a"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("resident", "RESIDENT"),
        ("ssd", "SSD"),
        ("both", "BOTH"),
    ],
)
def test_parser_accepts_qwen4_ple_modes(bench_module, value, expected):
    args = bench_module._parse_args(["model-a", "--qwen4-ple", value])

    assert args.qwen4_ple is getattr(bench_module.Qwen4PLEMode, expected)


def test_parser_rejects_invalid_native_mtp_mode(bench_module):
    with pytest.raises(SystemExit, match="2"):
        bench_module._parse_args(["model-a", "--native-mtp", "enabled"])


def test_legacy_case_expansion_and_labels_are_unchanged(bench_module):
    cases = bench_module._expand_cases(
        ["model-a", "model-b"],
        bench_module.NativeMTPMode.BOTH,
    )

    assert [
        (
            case.model_path,
            case.variant.native_mtp_enabled,
            case.variant.ple_storage,
        )
        for case in cases
    ] == [
        ("model-a", False, None),
        ("model-a", True, None),
        ("model-b", False, None),
        ("model-b", True, None),
    ]
    assert [case.variant.model_settings.mtp_enabled for case in cases] == [
        False,
        True,
        False,
        True,
    ]
    assert [
        case.variant.model_settings.vlm_mtp_enabled for case in cases
    ] == [False, False, False, False]
    assert [case.variant.model_settings for case in cases] == [
        bench_module.ModelSettings(mtp_enabled=False),
        bench_module.ModelSettings(mtp_enabled=True),
        bench_module.ModelSettings(mtp_enabled=False),
        bench_module.ModelSettings(mtp_enabled=True),
    ]
    assert bench_module._allocate_labels(cases[:2], include_variant=True) == [
        "baseline model-a",
        "native-mtp model-a",
    ]
    on_cases = bench_module._expand_cases(
        ["model-a"], bench_module.NativeMTPMode.ON
    )
    assert len(on_cases) == 1
    assert on_cases[0].variant.model_settings.mtp_enabled is True


def test_ple_both_expands_resident_before_ssd_with_native_mtp_off(
    bench_module,
):
    cases = bench_module._expand_cases(
        ["model-a"],
        bench_module.NativeMTPMode.OFF,
        bench_module.Qwen4PLEMode.BOTH,
    )

    assert [
        (
            case.variant.native_mtp_enabled,
            case.variant.ple_storage,
            case.variant.model_settings.mtp_enabled,
            case.variant.model_settings.qwen4_ple_ssd_offload,
        )
        for case in cases
    ] == [
        (False, "resident", False, False),
        (False, "ssd", False, True),
    ]
    assert bench_module._allocate_labels(
        cases, include_variant=False
    ) == ["ple=resident model-a", "ple=ssd model-a"]


def test_native_mtp_and_ple_both_expand_in_cartesian_order(bench_module):
    cases = bench_module._expand_cases(
        ["model-a"],
        bench_module.NativeMTPMode.BOTH,
        bench_module.Qwen4PLEMode.BOTH,
    )

    assert [
        (
            case.variant.native_mtp_enabled,
            case.variant.ple_storage,
            case.variant.model_settings.mtp_enabled,
            case.variant.model_settings.qwen4_ple_ssd_offload,
        )
        for case in cases
    ] == [
        (False, "resident", False, False),
        (False, "ssd", False, True),
        (True, "resident", True, False),
        (True, "ssd", True, True),
    ]
    assert bench_module._allocate_labels(cases, include_variant=True) == [
        "native-mtp=off ple=resident model-a",
        "native-mtp=off ple=ssd model-a",
        "native-mtp=on ple=resident model-a",
        "native-mtp=on ple=ssd model-a",
    ]


def test_bench_model_passes_case_settings_and_stops_engine(
    bench_module, monkeypatch
):
    benchmark = ModuleType("omlx.admin.benchmark")
    def generate_prompt(tokenizer, pp):
        assert tokenizer is not None
        assert pp == 1
        return "prompt"

    async def fail_single_test(engine, prompt, max_tokens, pp_len):
        assert engine is engines[0]
        assert prompt == "prompt"
        assert max_tokens == 1
        assert pp_len == 1
        raise RuntimeError("test failure")

    async def fail_batch_test(engine, prompts, pp_len, max_tokens, batch_size):
        raise AssertionError("batch test should not run")

    benchmark._generate_prompt = generate_prompt
    benchmark._run_single_test = fail_single_test
    benchmark._run_batch_test = fail_batch_test

    engine_module = ModuleType("omlx.engine.vlm")
    constructed = []
    engines = []
    lifecycle = []

    class FakeEngine:
        def __init__(self, model_name, model_settings):
            constructed.append(
                {
                    "model_name": model_name,
                    "model_settings": model_settings,
                }
            )
            self.tokenizer = object()
            self.stopped = False
            engines.append(self)

        async def start(self):
            lifecycle.append("start")

        async def stop(self):
            lifecycle.append("stop")
            self.stopped = True

    engine_module.VLMBatchedEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "omlx.admin.benchmark", benchmark)
    monkeypatch.setitem(sys.modules, "omlx.engine.vlm", engine_module)
    case = bench_module._expand_cases(
        ["model-a"],
        bench_module.NativeMTPMode.ON,
        bench_module.Qwen4PLEMode.SSD,
    )[0]

    with pytest.raises(RuntimeError, match="test failure"):
        asyncio.run(bench_module._bench_model(case, [1], 1, [], 0))

    assert constructed == [
        {
            "model_name": "model-a",
            "model_settings": case.variant.model_settings,
        }
    ]
    assert engines[0].stopped
    assert lifecycle == ["start", "stop"]


def test_allocate_labels_adds_stable_ordinals_for_duplicate_logical_labels(
    bench_module,
):
    cases = bench_module._expand_cases(
        ["/models/duplicate", "/other/duplicate"],
        bench_module.NativeMTPMode.OFF,
    )

    assert bench_module._allocate_labels(cases, include_variant=False) == [
        "duplicate #1",
        "duplicate #2",
    ]


def test_allocate_labels_handles_names_that_resemble_ordinals(bench_module):
    cases = bench_module._expand_cases(
        ["/models/duplicate", "/other/duplicate", "/models/duplicate #1"],
        bench_module.NativeMTPMode.OFF,
    )

    labels = bench_module._allocate_labels(cases, include_variant=False)

    assert labels == ["duplicate #1", "duplicate #2", "duplicate #1 #1"]
    assert len(set(labels)) == len(labels)


@pytest.mark.parametrize("width", [39, 29])
def test_display_labels_remain_unique_after_truncation(bench_module, width):
    labels = [
        "native-mtp Qwen3.8-Flash-Next-very-long-model-name-a",
        "native-mtp Qwen3.8-Flash-Next-very-long-model-name-b",
    ]

    displays = bench_module._display_labels(labels, width)

    assert len(set(displays)) == len(displays)
    assert all(len(display) <= width for display in displays)
    assert displays[0].endswith("#1")
    assert displays[1].endswith("#2")
