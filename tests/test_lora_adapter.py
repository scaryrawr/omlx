# SPDX-License-Identifier: Apache-2.0
"""Tests for PEFT LoRA adapter parsing, base resolution, discovery and attach."""

import json
import math
import os
import re
import struct
import subprocess
import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model

from omlx.custom_kernels.oqh import fast as oqh_fast
from omlx.exceptions import (
    AdapterError,
    AdapterShapeError,
    ModelError,
    OMLXError,
    UnsupportedAdapterError,
)
from omlx.lora_adapter import (
    AdapterSource,
    BaseCandidate,
    BaseRef,
    IssueKind,
    LoraConfig,
    LoraLinear,
    LoraModule,
    attach_lora,
    base_candidates,
    bind_adapter,
    derive_entry,
    discover_adapters,
    lora_revision_of,
    match_bases,
    parse_adapter,
    parse_adapter_config,
    resolve_adapters,
)
from omlx.model_discovery import DiscoveredModel
from omlx.oqh.loading import install_load_hook
from tests.test_oqh import _tiny_qwen

CONFIG = "adapter_config.json"
WEIGHTS = "adapter_model.safetensors"
QWEN = "Qwen/Qwen3.8-27B"
REPO_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def malformed():
    """Expect an AdapterError that is not the valid-but-unsupported subclass."""
    with pytest.raises(AdapterError) as caught:
        yield caught
    assert not isinstance(caught.value, UnsupportedAdapterError)


def config_bytes(drop=(), **fields):
    document = {
        "peft_type": "LORA",
        "base_model_name_or_path": QWEN,
        "r": 32,
        "lora_alpha": 64,
        **fields,
    }
    for key in drop:
        del document[key]
    return json.dumps(document).encode()


ITEMSIZE = {"F32": 4, "BF16": 2, "F16": 2}
HAND_MODULE = "model.layers.0.mlp.gate_proj"


def lora_tensors(hf_path, rank, in_features, out_features, dtype="F32"):
    """Tensor name -> (dtype, shape) for one module's lora_A and lora_B."""
    prefix = f"base_model.model.{hf_path}"
    return {
        f"{prefix}.lora_A.weight": (dtype, (rank, in_features)),
        f"{prefix}.lora_B.weight": (dtype, (out_features, rank)),
    }


def raw_header(tensors):
    """A safetensors header with contiguous offsets, and the payload size it needs."""
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = ITEMSIZE[dtype] * math.prod(shape)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    return header, offset


def write_raw_safetensors(path, header, payload=0):
    """Write a safetensors file by hand. ``header`` is a dict or raw header bytes."""
    blob = header if isinstance(header, bytes) else json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(payload))


def adapter_source(public_id, base_ref=QWEN, root="/adapters"):
    """A source for a directory that does not exist, so resolution can read no file."""
    path = Path(root) / public_id
    return AdapterSource(
        public_id=public_id,
        path=path,
        config=LoraConfig(base_ref, rank=4, alpha=8.0, rslora=False),
        modules=(LoraModule(HAND_MODULE, 8, 16),),
        weights_path=path / WEIGHTS,
        weight_bytes=1000,
        fingerprint=f"fingerprint-of-{public_id}-in-{root}",
    )


def base_candidate(model_id, *repo_ids):
    return BaseCandidate(model_id, Path("/models") / model_id, frozenset(repo_ids))


def discovered(model_id, **fields):
    fields.setdefault("model_type", "llm")
    fields.setdefault("engine_type", "batched")
    fields.setdefault("model_path", f"/models/{model_id}")
    fields.setdefault("estimated_size", 1000)
    return DiscoveredModel(model_id=model_id, **fields)


def resolve(sources, candidates, overrides=None, taken=()):
    return resolve_adapters(sources, candidates, overrides or {}, set(taken))


def bind(adapter, base_path="/models/tiny"):
    return bind_adapter(parse_adapter(adapter), BaseRef("tiny", Path(base_path)))


BAD_FIELD_VALUES = [
    ("base_model_name_or_path", None),
    ("base_model_name_or_path", 7),
    ("base_model_name_or_path", ""),
    ("base_model_name_or_path", "  "),
    ("r", None),
    ("r", "4"),
    ("r", 4.0),
    ("r", True),
    ("r", 0),
    ("r", -1),
    ("lora_alpha", None),
    ("lora_alpha", "8"),
    ("lora_alpha", True),
    ("lora_alpha", 0),
    ("lora_alpha", -1),
    ("lora_alpha", float("nan")),
    ("lora_alpha", float("inf")),
    ("lora_alpha", float("-inf")),
    ("lora_alpha", 10**400),
]


class TestParseAdapterConfig:
    @pytest.mark.parametrize(
        ("rank", "alpha", "rslora", "scale"),
        [
            (32, 64, False, 2.0),
            (8, 4, False, 0.5),
            (64, 16, False, 0.25),
            (16, 16, True, 4.0),
            (4, 8, True, 4.0),
            (64, 16, True, 2.0),
        ],
    )
    def test_scale(self, rank, alpha, rslora, scale):
        config = parse_adapter_config(
            config_bytes(r=rank, lora_alpha=alpha, use_rslora=rslora)
        )
        assert (config.base_ref, config.rank, config.rslora) == (QWEN, rank, rslora)
        assert config.scale == pytest.approx(scale)

    def test_alpha_is_stored_as_float(self):
        config = parse_adapter_config(config_bytes(lora_alpha=64))
        assert isinstance(config.alpha, float)
        assert config.alpha == 64.0

    @pytest.mark.parametrize("bias", ["none", None])
    def test_minimal_and_explicitly_off_configs_are_the_same(self, bias):
        explicit = config_bytes(
            use_dora=False,
            use_qalora=False,
            use_rslora=False,
            fan_in_fan_out=False,
            modules_to_save=None,
            rank_pattern={},
            alpha_pattern={},
            target_parameters=None,
            bias=bias,
        )
        assert parse_adapter_config(explicit) == parse_adapter_config(config_bytes())

    def test_size_cap_is_one_mebibyte(self):
        limit = 1024 * 1024
        raw = config_bytes()
        parse_adapter_config(raw + b" " * (limit - len(raw)))
        with malformed():
            parse_adapter_config(raw + b" " * (limit + 1 - len(raw)))

    @pytest.mark.parametrize(
        ("feature", "value"),
        [
            ("use_dora", True),
            ("use_qalora", True),
            ("modules_to_save", ["lm_head"]),
            ("rank_pattern", {"q_proj": 8}),
            ("alpha_pattern", {"q_proj": 16}),
            ("layer_replication", [[0, 2]]),
            ("target_parameters", ["experts.gate_up_proj"]),
            ("fan_in_fan_out", True),
            ("bias", "all"),
            ("bias", "lora_only"),
        ],
    )
    def test_unsupported_feature(self, feature, value):
        with pytest.raises(UnsupportedAdapterError) as caught:
            parse_adapter_config(config_bytes(**{feature: value}))
        error = caught.value
        assert error.feature == feature
        assert error.details["feature"] == feature
        assert feature in error.message
        assert "merge" in error.message.lower()

    @pytest.mark.parametrize(
        "initializer",
        [
            "pissa",
            "pissa_niter_4",
            "olora",
            "loftq",
            "corda",
            "lora_ga",
            "astra",
            "unknown",
        ],
    )
    def test_base_changing_initializers_are_rejected(self, initializer):
        with pytest.raises(UnsupportedAdapterError) as caught:
            parse_adapter_config(config_bytes(init_lora_weights=initializer))
        assert caught.value.feature == "init_lora_weights"

    @pytest.mark.parametrize(
        "initializer", [True, False, "gaussian", "eva", "orthogonal", "mica", "nora"]
    )
    def test_portable_initializers_do_not_change_inference_config(self, initializer):
        assert parse_adapter_config(
            config_bytes(init_lora_weights=initializer)
        ) == parse_adapter_config(config_bytes())

    def test_alora_is_rejected_without_merge_advice(self):
        with pytest.raises(UnsupportedAdapterError) as caught:
            parse_adapter_config(config_bytes(alora_invocation_tokens=[42]))
        assert caught.value.feature == "alora_invocation_tokens"
        assert "merge" not in caught.value.message.lower()

    @pytest.mark.parametrize("peft_type", ["IA3", "LOHA", "lora"])
    def test_only_lora_peft_type_is_supported(self, peft_type):
        with pytest.raises(UnsupportedAdapterError) as caught:
            parse_adapter_config(config_bytes(peft_type=peft_type))
        assert caught.value.feature == "peft_type"

    def test_other_peft_types_need_no_lora_fields(self):
        raw = json.dumps({"peft_type": "IA3", "target_modules": ["q_proj"]}).encode()
        with pytest.raises(UnsupportedAdapterError):
            parse_adapter_config(raw)

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param(b"", id="empty"),
            pytest.param(b"{not json", id="not-json"),
            pytest.param(b'{"r": "\xff"}', id="invalid-utf8"),
            pytest.param(b"[" * 200_000, id="nested-too-deep"),
            pytest.param(b"[]", id="list"),
            pytest.param(b"null", id="null"),
            pytest.param(b'"LORA"', id="string"),
            pytest.param(config_bytes(drop=["peft_type"]), id="no-peft-type"),
            pytest.param(config_bytes(peft_type=None), id="null-peft-type"),
            pytest.param(config_bytes(peft_type=3), id="numeric-peft-type"),
            *(
                pytest.param(config_bytes(drop=[key]), id=f"no-{key}")
                for key in ("base_model_name_or_path", "r", "lora_alpha")
            ),
            *(
                pytest.param(
                    config_bytes(**{key: value}), id=f"{key}={str(value)[:12]}"
                )
                for key, value in BAD_FIELD_VALUES
            ),
        ],
    )
    def test_malformed_config(self, raw):
        with malformed():
            parse_adapter_config(raw)


class TestAdapterExceptions:
    def test_path_is_recorded_in_details(self):
        error = AdapterError("bad", Path("/adapters/pi"))
        assert error.message == "bad"
        assert error.details == {"adapter": "/adapters/pi"}

    def test_path_wins_and_the_callers_details_are_not_changed(self):
        details = {"adapter": "stale", "other": 1}
        error = AdapterError("bad", "/adapters/pi", details)
        assert error.details == {"adapter": "/adapters/pi", "other": 1}
        assert details == {"adapter": "stale", "other": 1}

    def test_no_path_leaves_the_adapter_key_out(self):
        assert "adapter" not in AdapterError("bad").details

    def test_unsupported_exposes_its_feature(self):
        error = UnsupportedAdapterError("no", "/adapters/pi", feature="use_dora")
        assert error.feature == "use_dora"
        assert error.details == {"adapter": "/adapters/pi", "feature": "use_dora"}

    def test_hierarchy(self):
        for kind in (AdapterError, UnsupportedAdapterError, AdapterShapeError):
            assert issubclass(kind, AdapterError)
            assert issubclass(kind, ModelError)
            assert issubclass(kind, OMLXError)
        assert not issubclass(AdapterShapeError, UnsupportedAdapterError)
        assert not issubclass(UnsupportedAdapterError, AdapterShapeError)


class TestParseAdapter:
    def test_reads_identity_config_and_modules(self, write_lora_adapter, tmp_path):
        adapter = write_lora_adapter(
            tmp_path / "LFM3-1.7B-pi",
            base_ref="LiquidAI/LFM3-1.7B",
            rank=4,
            alpha=16.0,
            modules={
                "model.layers.1.mlp.down_proj": (32, 8),
                "model.layers.0.mlp.gate_proj": (8, 32),
                "model.layers.0.self_attn.q_proj": (8, 6),
            },
        )

        source = parse_adapter(adapter)

        assert source.public_id == "LFM3-1.7B-pi"
        assert source.path == adapter
        assert source.weights_path == adapter / WEIGHTS
        assert source.weight_bytes == (adapter / WEIGHTS).stat().st_size
        assert source.config == LoraConfig("LiquidAI/LFM3-1.7B", 4, 16.0, False)
        assert source.modules == (
            LoraModule("model.layers.0.mlp.gate_proj", 8, 32),
            LoraModule("model.layers.0.self_attn.q_proj", 8, 6),
            LoraModule("model.layers.1.mlp.down_proj", 32, 8),
        )

    def test_modules_are_sorted_whatever_the_header_order(
        self, write_lora_adapter, tmp_path
    ):
        # safetensors.save_file sorts names itself, so the order is written by hand.
        adapter = write_lora_adapter(tmp_path / "ad", modules={})
        tensors = {}
        for layer in (2, 0, 1):
            tensors |= lora_tensors(f"model.layers.{layer}.mlp.up_proj", 4, 8, 16)
        header, payload = raw_header(tensors)
        write_raw_safetensors(adapter / WEIGHTS, header, payload)

        paths = [module.hf_path for module in parse_adapter(adapter).modules]

        assert paths == [f"model.layers.{layer}.mlp.up_proj" for layer in (0, 1, 2)]

    def test_hand_written_file_parses(self, write_lora_adapter, tmp_path):
        """The baseline the header-defect cases below are mutations of."""
        adapter = write_lora_adapter(tmp_path / "ad", modules={})
        header, payload = raw_header(lora_tensors(HAND_MODULE, 4, 8, 16))
        write_raw_safetensors(adapter / WEIGHTS, header, payload)

        assert parse_adapter(adapter).modules == (LoraModule(HAND_MODULE, 8, 16),)

    @pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
    def test_half_precision_weights_are_accepted(
        self, write_lora_adapter, tmp_path, dtype
    ):
        adapter = write_lora_adapter(tmp_path / "ad", modules={})
        tensors = {
            name: mx.zeros(shape, dtype=dtype)
            for name, (_, shape) in lora_tensors(HAND_MODULE, 4, 8, 16).items()
        }
        mx.save_safetensors(str(adapter / WEIGHTS), tensors)

        assert parse_adapter(adapter).modules == (LoraModule(HAND_MODULE, 8, 16),)

    def test_pickled_weights_beside_safetensors_are_ignored(
        self, write_lora_adapter, tmp_path
    ):
        adapter = write_lora_adapter(tmp_path / "ad")
        (adapter / "adapter_model.bin").write_bytes(b"never loaded")

        assert parse_adapter(adapter).modules

    @pytest.mark.parametrize("name", [CONFIG, WEIGHTS])
    def test_missing_file(self, write_lora_adapter, tmp_path, name):
        adapter = write_lora_adapter(tmp_path / "ad")
        (adapter / name).unlink()

        with malformed() as caught:
            parse_adapter(adapter)

        assert name in caught.value.message

    def test_pickled_weights_alone_are_unsupported(self, write_lora_adapter, tmp_path):
        adapter = write_lora_adapter(tmp_path / "ad")
        (adapter / WEIGHTS).unlink()
        (adapter / "adapter_model.bin").write_bytes(b"never loaded")

        with pytest.raises(UnsupportedAdapterError) as caught:
            parse_adapter(adapter)

        assert caught.value.feature == "weights_format"
        assert caught.value.details["adapter"] == str(adapter)

    @pytest.mark.parametrize("name", [CONFIG, WEIGHTS])
    def test_an_unreadable_file_is_an_adapter_error(
        self, write_lora_adapter, tmp_path, name
    ):
        adapter = write_lora_adapter(tmp_path / "ad")
        (adapter / name).unlink()
        (adapter / name).mkdir()

        with malformed():
            parse_adapter(adapter)

    def test_missing_directory_and_plain_file(self, tmp_path):
        plain_file = tmp_path / "file"
        plain_file.write_text("x")

        with malformed():
            parse_adapter(tmp_path / "nowhere")
        with malformed():
            parse_adapter(plain_file)

    def test_errors_name_the_adapter_directory(self, write_lora_adapter, tmp_path):
        adapter = write_lora_adapter(tmp_path / "ad", config={"r": 8})

        with malformed() as caught:
            parse_adapter(adapter)

        assert caught.value.details["adapter"] == str(adapter)

    @pytest.mark.parametrize(
        "name",
        [
            "lm_head.weight",
            "base_model.model.model.layers.0.mlp.gate_proj.lora_B.bias",
            "base_model.model.model.layers.0.mlp.gate_proj.lora_embedding_A",
            "model.layers.0.mlp.gate_proj.lora_A.weight",
        ],
    )
    def test_unexpected_tensor_name(self, write_lora_adapter, tmp_path, name):
        adapter = write_lora_adapter(
            tmp_path / "ad", tensors={name: np.zeros((4, 8), np.float32)}
        )

        with malformed() as caught:
            parse_adapter(adapter)

        assert name in caught.value.message

    @pytest.mark.parametrize(
        ("half", "shape"), [("A", (4, 8)), ("B", (16, 4))], ids=["only-A", "only-B"]
    )
    def test_a_lone_half(self, write_lora_adapter, tmp_path, half, shape):
        name = f"base_model.model.{HAND_MODULE}.lora_{half}.weight"
        adapter = write_lora_adapter(
            tmp_path / "ad", modules={}, tensors={name: np.zeros(shape, np.float32)}
        )

        with malformed():
            parse_adapter(adapter)

    def test_rank_must_match_the_config(self, write_lora_adapter, tmp_path):
        adapter = write_lora_adapter(tmp_path / "ad", rank=4, config={"r": 8})

        with malformed():
            parse_adapter(adapter)

    def test_a_and_b_must_agree_on_rank(self, write_lora_adapter, tmp_path):
        name = f"base_model.model.{HAND_MODULE}.lora_B.weight"
        adapter = write_lora_adapter(
            tmp_path / "ad",
            modules={HAND_MODULE: (8, 16)},
            tensors={name: np.zeros((16, 2), np.float32)},
        )

        with malformed():
            parse_adapter(adapter)

    @pytest.mark.parametrize("dtype", [np.float64, np.int8, np.uint8, np.int32])
    def test_unsupported_dtype(self, write_lora_adapter, tmp_path, dtype):
        name = f"base_model.model.{HAND_MODULE}.lora_A.weight"
        adapter = write_lora_adapter(
            tmp_path / "ad",
            modules={HAND_MODULE: (8, 16)},
            tensors={name: np.zeros((4, 8), dtype)},
        )

        with malformed():
            parse_adapter(adapter)

    @pytest.mark.parametrize("shape", [(4,), (1, 4, 8)], ids=["1-D", "3-D"])
    def test_weights_must_be_matrices(self, write_lora_adapter, tmp_path, shape):
        name = f"base_model.model.{HAND_MODULE}.lora_A.weight"
        adapter = write_lora_adapter(
            tmp_path / "ad",
            modules={HAND_MODULE: (8, 16)},
            tensors={name: np.zeros(shape, np.float32)},
        )

        with malformed():
            parse_adapter(adapter)

    def test_a_file_with_no_lora_weights(self, write_lora_adapter, tmp_path):
        adapter = write_lora_adapter(tmp_path / "ad", modules={})

        with malformed():
            parse_adapter(adapter)

    @pytest.mark.parametrize(
        "keep",
        [
            pytest.param(lambda size, header: 5, id="inside-the-length-prefix"),
            pytest.param(lambda size, header: 8 + header // 2, id="inside-the-header"),
            pytest.param(lambda size, header: size - 1, id="one-byte-short"),
        ],
    )
    def test_truncated_download(self, write_lora_adapter, tmp_path, keep):
        adapter = write_lora_adapter(tmp_path / "ad")
        data = (adapter / WEIGHTS).read_bytes()
        (header_len,) = struct.unpack("<Q", data[:8])
        (adapter / WEIGHTS).write_bytes(data[: keep(len(data), header_len)])

        with malformed() as caught:
            parse_adapter(adapter)

        assert "Download it again" in caught.value.message

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param(b"", id="empty-file"),
            pytest.param(struct.pack("<Q", 10_000) + b"{}", id="length-beyond-file"),
            pytest.param(struct.pack("<Q", 2**64 - 1), id="length-no-file-could-hold"),
            pytest.param(struct.pack("<Q", 15) + b"not json at all", id="not-json"),
            pytest.param(struct.pack("<Q", 2) + b"[]", id="not-an-object"),
        ],
    )
    def test_bad_safetensors_header(self, write_lora_adapter, tmp_path, raw):
        adapter = write_lora_adapter(tmp_path / "ad")
        (adapter / WEIGHTS).write_bytes(raw)

        with malformed():
            parse_adapter(adapter)

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param("not an object", id="entry-is-not-an-object"),
            pytest.param({"dtype": "F32", "shape": [4, 8]}, id="no-offsets"),
            pytest.param({"dtype": "F32", "data_offsets": [0, 128]}, id="no-shape"),
            pytest.param({"shape": [4, 8], "data_offsets": [0, 128]}, id="no-dtype"),
            pytest.param(
                {"dtype": "F32", "shape": [4, 8], "data_offsets": [0]}, id="one-offset"
            ),
            pytest.param(
                {"dtype": "F32", "shape": [4, 8], "data_offsets": [0, 10**9]},
                id="offsets-past-the-end",
            ),
            pytest.param(
                {"dtype": "F32", "shape": [4, 8], "data_offsets": [128, 0]},
                id="offsets-reversed",
            ),
            pytest.param(
                {"dtype": "F32", "shape": [4, 8], "data_offsets": [-4, 124]},
                id="negative-offset",
            ),
            pytest.param(
                {"dtype": "F32", "shape": [4, 8], "data_offsets": [0, 64]},
                id="size-does-not-fit-the-shape",
            ),
            pytest.param(
                {"dtype": "F32", "shape": [4, -8], "data_offsets": [0, 128]},
                id="negative-dimension",
            ),
            pytest.param(
                {"dtype": "F32", "shape": [4, "8"], "data_offsets": [0, 128]},
                id="string-dimension",
            ),
        ],
    )
    def test_malformed_tensor_entry(self, write_lora_adapter, tmp_path, entry):
        adapter = write_lora_adapter(tmp_path / "ad", modules={})
        header, payload = raw_header(lora_tensors(HAND_MODULE, 4, 8, 16))
        header[f"base_model.model.{HAND_MODULE}.lora_A.weight"] = entry
        write_raw_safetensors(adapter / WEIGHTS, header, payload)

        with malformed():
            parse_adapter(adapter)


def edit_alpha_keeping_the_length(adapter):
    path = adapter / CONFIG
    text = path.read_text()
    assert '"lora_alpha": 8.0' in text
    path.write_text(text.replace('"lora_alpha": 8.0', '"lora_alpha": 9.0'))


def bump_mtime(adapter):
    weights = adapter / WEIGHTS
    stat = weights.stat()
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


def grow_weights(adapter):
    weights = adapter / WEIGHTS
    stat = weights.stat()
    with weights.open("ab") as handle:
        handle.write(b"\0")
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))


class TestBinding:
    def test_unchanged_files_keep_the_same_revision(self, write_lora_adapter, tmp_path):
        adapter = write_lora_adapter(tmp_path / "ad")

        first, second = bind(adapter), bind(adapter)

        assert first.source.fingerprint == second.source.fingerprint
        assert first.cache_revision == second.cache_revision
        assert re.fullmatch(r"lora-[0-9a-f]{12}", first.cache_revision)

    @pytest.mark.parametrize(
        "change",
        [edit_alpha_keeping_the_length, bump_mtime, grow_weights],
        ids=lambda f: f.__name__,
    )
    def test_a_changed_adapter_changes_the_revision(
        self, write_lora_adapter, tmp_path, change
    ):
        adapter = write_lora_adapter(tmp_path / "ad")
        before = bind(adapter)

        change(adapter)
        after = bind(adapter)

        assert after.source.fingerprint != before.source.fingerprint
        assert after.cache_revision != before.cache_revision

    def test_a_rebound_base_path_changes_the_revision(
        self, write_lora_adapter, tmp_path
    ):
        adapter = write_lora_adapter(tmp_path / "ad")

        first = bind(adapter, "/models/a")

        assert first.cache_revision != bind(adapter, "/models/b").cache_revision
        assert first.cache_revision == bind(adapter, "/models/a").cache_revision

    def test_status_row(self, write_lora_adapter, tmp_path):
        adapter = write_lora_adapter(tmp_path / "ad", rank=4, alpha=8.0)
        binding = bind_adapter(
            parse_adapter(adapter), BaseRef("qwen-27b", Path("/models/qwen-27b"))
        )

        assert binding.as_status() == {
            "base_model_id": "qwen-27b",
            "rank": 4,
            "alpha": 8.0,
            "weight_bytes": (adapter / WEIGHTS).stat().st_size,
        }


class TestMatchBases:
    def test_exact_beats_family(self):
        exact = base_candidate("Qwen3.8-27B", "qwen/qwen3.8-27b")
        family = base_candidate("Qwen3.8-27B-oQ4e-mtp", "someone/qwen3.8-27b-oq4e-mtp")

        assert match_bases(QWEN, [family, exact]) == (exact,)

    def test_exact_tier_ignores_case(self):
        exact = base_candidate("anything", "qwen/qwen3.8-27b")

        assert match_bases("QWEN/qwen3.8-27B", [exact]) == (exact,)

    def test_family_tier_ignores_case(self):
        family = base_candidate("QWEN3.8-27B-OQ4E")

        assert match_bases(QWEN, [family]) == (family,)

    @pytest.mark.parametrize(
        ("model_id", "matches"),
        [
            ("Qwen3.8-27B", True),
            ("Qwen3.8-27B-oQ4e-mtp", True),
            ("qwen3.8-27b_x", True),
            ("Qwen3.8-270B", False),
            ("Qwen3.8-27Bx", False),
            ("Qwen3.8-27B.5", False),
            ("Qwen3.8-27", False),
            ("My-Qwen3.8-27B", False),
        ],
    )
    def test_family_tier_needs_a_separator_after_the_stem(self, model_id, matches):
        found = match_bases(QWEN, [base_candidate(model_id)])

        assert bool(found) is matches

    def test_a_ref_without_an_owner_still_matches_the_family(self):
        family = base_candidate("Qwen3.8-27B-4bit")

        assert match_bases("Qwen3.8-27B", [family]) == (family,)

    @pytest.mark.parametrize("base_ref", ["", "Qwen/"])
    def test_a_ref_with_no_name_matches_nothing(self, base_ref):
        odd = [base_candidate("-odd"), base_candidate("_odd"), base_candidate("odd")]

        assert match_bases(base_ref, odd) == ()

    def test_input_order_is_kept(self):
        later, earlier = (
            base_candidate("qwen3.8-27b-b"),
            base_candidate("qwen3.8-27b-a"),
        )

        assert match_bases(QWEN, [later, earlier]) == (later, earlier)


class TestResolveAdapters:
    """Sources here point at directories that do not exist."""

    def test_binds_the_single_match(self):
        pi = adapter_source("pi")
        base = base_candidate("Qwen3.8-27B", "qwen/qwen3.8-27b")

        bindings, issues = resolve([pi], [base, base_candidate("other")])

        assert issues == ()
        (binding,) = bindings
        assert binding.source is pi
        assert binding.base == BaseRef("Qwen3.8-27B", base.path)
        assert binding.cache_revision == bind_adapter(pi, binding.base).cache_revision

    def test_no_match_is_base_missing(self):
        pi = adapter_source("pi")

        bindings, issues = resolve([pi], [base_candidate("Llama-3-8B", "meta/llama")])

        assert bindings == ()
        (issue,) = issues
        assert (issue.kind, issue.public_id, issue.path) == (
            IssueKind.BASE_MISSING,
            "pi",
            pi.path,
        )
        assert QWEN in issue.detail

    def test_several_matches_are_never_guessed(self):
        first, second = (
            base_candidate("qwen3.8-27b-mlx"),
            base_candidate("qwen3.8-27b-q4"),
        )

        bindings, issues = resolve([adapter_source("pi")], [first, second])

        assert bindings == ()
        (issue,) = issues
        assert issue.kind is IssueKind.BASE_AMBIGUOUS
        assert issue.candidates == ("qwen3.8-27b-mlx", "qwen3.8-27b-q4")
        assert all(name in issue.detail for name in issue.candidates)

    @pytest.mark.parametrize("base_ref", [QWEN, "Other/Unrelated-Model"])
    def test_a_valid_override_wins(self, base_ref):
        first, second = (
            base_candidate("qwen3.8-27b-mlx"),
            base_candidate("qwen3.8-27b-q4"),
        )
        overrides = {"pi": "qwen3.8-27b-q4", "someone-else": "ghost"}

        bindings, issues = resolve(
            [adapter_source("pi", base_ref)], [first, second], overrides
        )

        assert issues == ()
        assert [binding.base.model_id for binding in bindings] == ["qwen3.8-27b-q4"]

    def test_an_invalid_override_does_not_fall_back_to_matching(self):
        exact = base_candidate("Qwen3.8-27B", "qwen/qwen3.8-27b")

        bindings, issues = resolve([adapter_source("pi")], [exact], {"pi": "deleted"})

        assert bindings == ()
        (issue,) = issues
        assert issue.kind is IssueKind.OVERRIDE_INVALID
        assert "deleted" in issue.detail

    def test_an_id_held_by_a_real_model_is_a_collision(self):
        exact = base_candidate("Qwen3.8-27B", "qwen/qwen3.8-27b")

        bindings, issues = resolve(
            [adapter_source("Qwen3.8-27B")], [exact], taken={"Qwen3.8-27B"}
        )

        assert bindings == ()
        assert [issue.kind for issue in issues] == [IssueKind.ID_COLLISION]

    def test_a_colon_in_the_id_is_a_collision(self):
        exact = base_candidate("Qwen3.8-27B", "qwen/qwen3.8-27b")

        bindings, issues = resolve([adapter_source("pi:fast")], [exact])

        assert bindings == ()
        (issue,) = issues
        assert issue.kind is IssueKind.ID_COLLISION
        assert ":" in issue.detail

    def test_the_first_directory_wins_a_duplicate_basename(self):
        exact = base_candidate("Qwen3.8-27B", "qwen/qwen3.8-27b")
        first = adapter_source("pi", root="/adapters/a")
        second = adapter_source("pi", root="/adapters/b")

        bindings, issues = resolve([first, second], [exact])

        assert [binding.source for binding in bindings] == [first]
        (issue,) = issues
        assert (issue.kind, issue.path) == (IssueKind.ID_COLLISION, second.path)

    @pytest.mark.parametrize(
        ("first_ref", "first_kind", "second_ref", "candidates"),
        [
            (
                "Other/Missing-Base",
                IssueKind.BASE_MISSING,
                QWEN,
                [base_candidate("Qwen3.8-27B", "qwen/qwen3.8-27b")],
            ),
            (
                QWEN,
                IssueKind.BASE_AMBIGUOUS,
                "Other/Thing",
                [
                    base_candidate("qwen3.8-27b-mlx"),
                    base_candidate("qwen3.8-27b-q4"),
                    base_candidate("Other-Thing", "other/thing"),
                ],
            ),
        ],
        ids=["base-missing", "base-ambiguous"],
    )
    def test_an_unresolved_directory_still_holds_its_id(
        self, first_ref, first_kind, second_ref, candidates
    ):
        first = adapter_source("pi", first_ref, root="/adapters/a")
        second = adapter_source("pi", second_ref, root="/adapters/b")
        alone, _ = resolve([second], candidates)
        assert [binding.source for binding in alone] == [second]

        bindings, issues = resolve([first, second], candidates)

        assert bindings == ()
        assert [(issue.kind, issue.path) for issue in issues] == [
            (first_kind, first.path),
            (IssueKind.ID_COLLISION, second.path),
        ]

    def test_an_invalid_pin_holds_the_id_too(self):
        exact = base_candidate("Qwen3.8-27B", "qwen/qwen3.8-27b")
        first = adapter_source("pi", root="/adapters/a")
        second = adapter_source("pi", root="/adapters/b")

        bindings, issues = resolve([first, second], [exact], {"pi": "deleted"})

        assert bindings == ()
        assert [(issue.kind, issue.path) for issue in issues] == [
            (IssueKind.OVERRIDE_INVALID, first.path),
            (IssueKind.ID_COLLISION, second.path),
        ]

    def test_results_follow_source_order(self):
        exact = base_candidate("Qwen3.8-27B", "qwen/qwen3.8-27b")
        sources = [
            adapter_source("a-ok"),
            adapter_source("b-missing", "Other/Thing"),
            adapter_source("c-ok"),
            adapter_source("d-missing", "Other/Thing"),
        ]

        bindings, issues = resolve(sources, [exact])

        assert [binding.source.public_id for binding in bindings] == ["a-ok", "c-ok"]
        assert [issue.public_id for issue in issues] == ["b-missing", "d-missing"]


NON_BASES = {
    "helper": discovered("drafter", is_helper=True),
    "gguf": discovered("gguf-folder", model_format="gguf"),
    "embedding": discovered("emb", model_type="embedding", engine_type="embedding"),
    "reranker": discovered("rank", model_type="reranker", engine_type="reranker"),
    "speech-to-text": discovered(
        "stt", model_type="audio_stt", engine_type="audio_stt"
    ),
    "text-to-speech": discovered(
        "tts", model_type="audio_tts", engine_type="audio_tts"
    ),
    "image": discovered("img", model_type="image", engine_type="image"),
    "decision": discovered("dec", model_type="decision", engine_type="decision"),
    "chat-type-on-a-non-chat-engine": discovered(
        "odd-engine", model_type="llm", engine_type="embedding"
    ),
    "non-chat-type-on-a-chat-engine": discovered(
        "odd-type", model_type="embedding", engine_type="batched"
    ),
}


class TestBaseCandidates:
    def test_chat_models_in_input_order(self):
        models = [
            discovered("b-llm"),
            discovered("a-vlm", model_type="vlm", engine_type="vlm"),
        ]

        assert [c.model_id for c in base_candidates(models)] == ["b-llm", "a-vlm"]

    @pytest.mark.parametrize("excluded", NON_BASES.values(), ids=list(NON_BASES))
    def test_what_cannot_serve_chat_is_not_a_base(self, excluded):
        chat = discovered("chat")

        assert [c.model_id for c in base_candidates([excluded, chat])] == ["chat"]

    def test_adapter_entries_are_not_bases(self, write_lora_adapter, tmp_path):
        base = discovered("base")
        entry = derive_entry(base, bind(write_lora_adapter(tmp_path / "pi")))

        assert [c.model_id for c in base_candidates([base, entry])] == ["base"]

    def test_repo_ids_come_from_the_directory_and_the_hub_id(self):
        local = discovered("Qwen3.8-27B", model_path="/models/Qwen/Qwen3.8-27B")
        cached = discovered(
            "snapshot",
            model_path="/hub/models--Org--Name/snapshots/abc123",
            source_type="hf_cache",
            source_repo_id="Org/Name",
        )

        first, second = base_candidates([local, cached])

        assert first.path == Path("/models/Qwen/Qwen3.8-27B")
        assert first.repo_ids == frozenset({"qwen/qwen3.8-27b"})
        assert second.path == Path("/hub/models--Org--Name/snapshots/abc123")
        assert "org/name" in second.repo_ids

    def test_a_base_in_an_owner_directory_is_an_exact_match(self):
        models = [
            discovered("Qwen3.8-27B", model_path="/models/Qwen/Qwen3.8-27B"),
            discovered(
                "Qwen3.8-27B-oQ4e", model_path="/models/mlx-community/Qwen3.8-27B-oQ4e"
            ),
        ]

        found = match_bases(QWEN, base_candidates(models))

        assert [candidate.model_id for candidate in found] == ["Qwen3.8-27B"]


QWEN_DIR = "/models/Qwen/Qwen3.8-27B"


class TestDiscoverAdapters:
    def test_the_entry_serves_the_adapter_over_the_base(
        self, write_lora_adapter, tmp_path
    ):
        base = discovered(
            "qwen-base",
            model_path=QWEN_DIR,
            estimated_size=1000,
            text_only_size=600,
            thinking_default=True,
            source_type="hf_cache",
            source_repo_id="mlx-community/qwen-base",
            config_model_type="qwen3_5",
            model_context_length=262144,
        )
        adapter = write_lora_adapter(tmp_path / "qwen-pi")
        weight_bytes = (adapter / WEIGHTS).stat().st_size

        result = discover_adapters([adapter], {"qwen-base": base}, {})

        assert result.issues == ()
        entry = result.entries["qwen-pi"]
        assert entry.model_id == "qwen-pi"
        assert entry.model_path == QWEN_DIR
        assert entry.estimated_size == 1000 + weight_bytes
        assert entry.text_only_size == 600 + weight_bytes
        assert entry.source_repo_id is None
        assert entry.source_type == "local"
        assert entry.thinking_default is False
        assert entry.adapter is not None
        assert entry.adapter.base == BaseRef("qwen-base", Path(QWEN_DIR))
        assert (entry.model_type, entry.engine_type) == ("llm", "batched")
        assert (entry.config_model_type, entry.model_context_length) == (
            "qwen3_5",
            262144,
        )

    def test_the_entry_is_never_a_helper(self, write_lora_adapter, tmp_path):
        helper_base = discovered("drafter", is_helper=True)

        entry = derive_entry(helper_base, bind(write_lora_adapter(tmp_path / "pi")))

        assert entry.is_helper is False
        assert helper_base.is_helper is True

    def test_the_base_entry_is_left_alone(self, write_lora_adapter, tmp_path):
        base = discovered("qwen-base", model_path=QWEN_DIR, source_repo_id="a/b")

        discover_adapters(
            [write_lora_adapter(tmp_path / "pi")], {"qwen-base": base}, {}
        )

        assert base.model_id == "qwen-base"
        assert base.adapter is None
        assert base.estimated_size == 1000
        assert base.source_repo_id == "a/b"

    @pytest.mark.parametrize(
        ("base_value", "expected"), [(True, False), (False, False), (None, None)]
    )
    def test_thinking_default(self, write_lora_adapter, tmp_path, base_value, expected):
        base = discovered("qwen-base", model_path=QWEN_DIR, thinking_default=base_value)

        result = discover_adapters(
            [write_lora_adapter(tmp_path / "pi")], {"qwen-base": base}, {}
        )

        assert result.entries["pi"].thinking_default is expected

    def test_text_only_size_stays_zero_when_the_base_has_none(
        self, write_lora_adapter, tmp_path
    ):
        base = discovered("qwen-base", model_path=QWEN_DIR, text_only_size=0)

        result = discover_adapters(
            [write_lora_adapter(tmp_path / "pi")], {"qwen-base": base}, {}
        )

        assert result.entries["pi"].text_only_size == 0

    def test_an_adapter_never_replaces_a_real_model(self, write_lora_adapter, tmp_path):
        base = discovered("qwen-base", model_path=QWEN_DIR)

        result = discover_adapters(
            [write_lora_adapter(tmp_path / "qwen-base")], {"qwen-base": base}, {}
        )

        assert result.entries == {}
        assert [(i.kind, i.public_id) for i in result.issues] == [
            (IssueKind.ID_COLLISION, "qwen-base")
        ]
        assert base.adapter is None

    def test_downloading_a_base_does_not_hand_an_id_to_another_directory(
        self, write_lora_adapter, tmp_path
    ):
        first = write_lora_adapter(tmp_path / "a" / "pi", base_ref="Other/Thing")
        second = write_lora_adapter(tmp_path / "b" / "pi")
        models = {"qwen-base": discovered("qwen-base", model_path=QWEN_DIR)}

        before = discover_adapters([first, second], models, {})
        models["other-base"] = discovered(
            "other-base", model_path="/models/Other/Thing"
        )
        after = discover_adapters([first, second], models, {})

        assert before.entries == {}
        assert [(issue.path, issue.kind) for issue in before.issues] == [
            (first, IssueKind.BASE_MISSING),
            (second, IssueKind.ID_COLLISION),
        ]
        assert after.entries["pi"].adapter.source.path == first
        assert [(issue.path, issue.kind) for issue in after.issues] == [
            (second, IssueKind.ID_COLLISION)
        ]

    @pytest.mark.parametrize("unsupported", [False, True])
    def test_broken_adapter_does_not_hand_its_id_to_another_directory(
        self, write_lora_adapter, tmp_path, unsupported
    ):
        first = write_lora_adapter(tmp_path / "a" / "pi")
        second = write_lora_adapter(tmp_path / "b" / "pi")
        models = {"qwen-base": discovered("qwen-base", model_path=QWEN_DIR)}
        assert (
            discover_adapters([first, second], models, {})
            .entries["pi"]
            .adapter.source.path
            == first
        )

        if unsupported:
            write_lora_adapter(first, config={"use_dora": True})
        else:
            (first / WEIGHTS).write_bytes(b"truncated")

        result = discover_adapters([first, second], models, {})
        assert not result.entries
        assert [(issue.path, issue.kind) for issue in result.issues] == [
            (first, IssueKind.UNSUPPORTED if unsupported else IssueKind.MALFORMED),
            (second, IssueKind.ID_COLLISION),
        ]

    def test_an_override_pins_one_of_two_matching_bases(
        self, write_lora_adapter, tmp_path
    ):
        models = {
            "qwen3.8-27b-a": discovered("qwen3.8-27b-a", model_path="/models/a"),
            "qwen3.8-27b-b": discovered("qwen3.8-27b-b", model_path="/models/b"),
        }
        adapter = write_lora_adapter(tmp_path / "pi")

        ambiguous = discover_adapters([adapter], models, {})
        pinned = discover_adapters([adapter], models, {"pi": "qwen3.8-27b-b"})

        assert ambiguous.entries == {}
        assert [i.kind for i in ambiguous.issues] == [IssueKind.BASE_AMBIGUOUS]
        assert pinned.entries["pi"].model_path == "/models/b"

    def test_bad_directories_become_issues_and_never_raise(
        self, write_lora_adapter, tmp_path
    ):
        base = discovered("qwen-base", model_path=QWEN_DIR)
        good = write_lora_adapter(tmp_path / "a-good")
        no_config = write_lora_adapter(tmp_path / "b-no-config")
        (no_config / CONFIG).unlink()
        dora = write_lora_adapter(tmp_path / "c-dora", config={"use_dora": True})
        garbage = write_lora_adapter(tmp_path / "d-garbage")
        (garbage / WEIGHTS).write_bytes(b"garbage")
        pickled = write_lora_adapter(tmp_path / "e-pickled")
        (pickled / WEIGHTS).unlink()
        (pickled / "adapter_model.bin").write_bytes(b"never loaded")
        nowhere = tmp_path / "f-nowhere"
        directories = [good, no_config, dora, garbage, pickled, nowhere]

        result = discover_adapters(directories, {"qwen-base": base}, {})

        assert list(result.entries) == ["a-good"]
        assert {issue.public_id: issue.kind for issue in result.issues} == {
            "b-no-config": IssueKind.MALFORMED,
            "c-dora": IssueKind.UNSUPPORTED,
            "d-garbage": IssueKind.MALFORMED,
            "e-pickled": IssueKind.UNSUPPORTED,
            "f-nowhere": IssueKind.MALFORMED,
        }
        assert all(issue.path == tmp_path / issue.public_id for issue in result.issues)
        assert all(issue.detail for issue in result.issues)

    def test_the_detail_of_an_unsupported_adapter_names_the_feature(
        self, write_lora_adapter, tmp_path
    ):
        dora = write_lora_adapter(tmp_path / "dora", config={"use_dora": True})

        (issue,) = discover_adapters([dora], {}, {}).issues

        assert "use_dora" in issue.detail

    def test_parse_issues_come_before_resolution_issues(
        self, write_lora_adapter, tmp_path
    ):
        no_base = write_lora_adapter(tmp_path / "a-no-base", base_ref="Other/Thing")
        broken = write_lora_adapter(tmp_path / "b-broken")
        (broken / CONFIG).unlink()
        base = discovered("qwen-base", model_path=QWEN_DIR)

        result = discover_adapters([no_base, broken], {"qwen-base": base}, {})

        assert [(i.public_id, i.kind) for i in result.issues] == [
            ("b-broken", IssueKind.MALFORMED),
            ("a-no-base", IssueKind.BASE_MISSING),
        ]


DIMS, HIDDEN, RANK, ALPHA = 64, 128, 4, 8.0
PLAIN_SCALE = ALPHA / RANK
RSLORA_SCALE = ALPHA / math.sqrt(RANK)
TARGETS = {"gate_proj": (DIMS, HIDDEN), "down_proj": (HIDDEN, DIMS)}


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(DIMS, HIDDEN, bias=False)
        self.down_proj = nn.Linear(HIDDEN, DIMS, bias=True)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Mlp()

    def __call__(self, x):
        return x + self.mlp(x)


class Backbone(nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        self.layers = [Block() for _ in range(depth)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LlmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def __call__(self, x):
        return self.model(x)


class VlmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = LlmModel()
        self.vision_tower = Backbone(1)

    def __call__(self, x):
        return self.language_model(x)


class TwoRoots(nn.Module):
    """Both ``layers`` and ``model.layers`` exist, so a tail names two modules."""

    def __init__(self):
        super().__init__()
        self.layers = [Block()]
        self.model = Backbone(1)

    def __call__(self, x):
        return self.model(self.layers[0](x))


class OpaqueLinear(nn.Module):
    """Stands in for ScaledQuantizedLinear, a Module that is not a Linear."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __call__(self, x):
        return self.inner(x)


@dataclass(frozen=True)
class Layout:
    build: Callable[[], nn.Module]
    hf_prefix: str
    layers: Callable[[nn.Module], list]


LAYOUTS = {
    "llm": Layout(LlmModel, "model.layers", lambda m: m.model.layers),
    "vlm": Layout(
        VlmModel,
        "model.language_model.layers",
        lambda m: m.language_model.model.layers,
    ),
    "flat": Layout(Backbone, "model.layers", lambda m: m.layers),
}


def lora_factors(in_features, out_features, seed):
    rng = np.random.default_rng(seed)
    lora_a = rng.standard_normal((RANK, in_features)) / math.sqrt(in_features)
    lora_b = rng.standard_normal((out_features, RANK)) * 0.5
    return lora_a.astype(np.float32), lora_b.astype(np.float32)


def write_adapter(
    write_lora_adapter,
    path,
    hf_prefix,
    *,
    config=None,
    dtype=None,
    layers=(0, 1),
    seed=0,
):
    """Write an adapter over ``layers``. Returns it and {(layer, name): (A, B)}.

    Another ``seed`` gives the same shapes with other values.
    """
    factors, tensors = {}, {}
    for layer in layers:
        for name, (in_features, out_features) in TARGETS.items():
            lora_a, lora_b = lora_factors(
                in_features, out_features, seed + len(factors)
            )
            factors[layer, name] = (lora_a, lora_b)
            prefix = f"base_model.model.{hf_prefix}.{layer}.mlp.{name}"
            tensors[f"{prefix}.lora_A.weight"] = lora_a
            tensors[f"{prefix}.lora_B.weight"] = lora_b
    adapter = write_lora_adapter(
        path, rank=RANK, alpha=ALPHA, modules={}, tensors=tensors, config=config
    )
    if dtype is not None:
        shipped = {
            name: mx.array(array).astype(dtype) for name, array in tensors.items()
        }
        mx.save_safetensors(str(adapter / WEIGHTS), shipped)
    return adapter, factors


@pytest.mark.parametrize("layout", ["qwen", "gemma"])
@pytest.mark.parametrize("bits", [None, 4, 8])
def test_routed_down_projection(write_lora_adapter, tmp_path, layout, bits):
    from types import SimpleNamespace

    experts, rank, width, outputs = 3, 2, 64, 32
    layer = nn.Module()
    if layout == "qwen":
        from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

        layer.mlp = Qwen3_5MoeSparseMoeBlock(
            SimpleNamespace(
                hidden_size=outputs,
                moe_intermediate_size=width,
                shared_expert_intermediate_size=width,
                num_experts=experts,
                num_experts_per_tok=2,
                norm_topk_prob=True,
            )
        )
        switch = layer.mlp.switch_mlp
        stem = "model.language_model.layers.0.mlp.experts"
    else:
        from mlx_lm.models.gemma4_text import Experts

        layer.experts = Experts(
            SimpleNamespace(
                hidden_size=outputs, moe_intermediate_size=width, num_experts=experts
            )
        )
        switch = layer.experts.switch_glu
        stem = "model.layers.0.experts"
    model = nn.Module()
    model.layers = [layer]
    if bits is not None:
        switch.down_proj = switch.down_proj.to_quantized(group_size=32, bits=bits)
    original = switch.down_proj
    weight = effective_parts(original, None if bits is None else ("affine", bits, 32))[
        0
    ]
    rng = np.random.default_rng(42)
    a = rng.normal(0, 0.1, size=(experts, rank, width)).astype(np.float32)
    b = rng.normal(0, 0.1, size=(outputs, rank, experts)).astype(np.float32)
    dense = {"model.language_model.layers.0.mlp.gate": (outputs, experts)}
    path = write_lora_adapter(
        tmp_path / "routed",
        rank=rank,
        alpha=3,
        modules=dense if layout == "qwen" else {},
        tensors={
            f"base_model.model.{stem}.lora_A.weight": a.reshape(experts * rank, width),
            f"base_model.model.{stem}.lora_B.weight": b.reshape(
                outputs, rank * experts
            ),
        },
        config={"target_parameters": [stem + ".down_proj"]},
    )
    binding = bind_adapter(parse_adapter(path), BaseRef("base", tmp_path))
    assert attach_lora(model, binding) == (2 if layout == "qwen" else 1)
    assert lora_revision_of(model) == binding.cache_revision
    assert switch.down_proj.linear is original
    assert not hasattr(switch.down_proj, "weight")
    merged = weight + 1.5 * np.einsum("ore,eri->eoi", b, a)
    for routes, sorted_indices in (
        (np.array([[2, 0], [1, 2]]), False),
        (np.array([0, 1, 2, 2]), True),
    ):
        x = rng.standard_normal((*routes.shape, 1, width)).astype(np.float32)
        actual = switch.down_proj(mx.array(x), mx.array(routes), sorted_indices)
        expected = x @ np.swapaxes(merged[routes], -1, -2)
        np.testing.assert_allclose(np.array(actual), expected, atol=2e-5, rtol=2e-5)
    x = mx.array(rng.standard_normal((1, 33, outputs)).astype(np.float32))
    if layout == "qwen":
        from omlx.patches.qwen35_moe_routed_decode import _build_plan

        model.eval()
        assert _build_plan(layer.mlp) is None
        forward, args = layer.mlp, (x,)
    else:
        routes = mx.broadcast_to(mx.array([2, 0]), (1, 33, 2))
        scores = mx.broadcast_to(mx.array([0.75, 0.25]), routes.shape)
        forward, args = layer.experts, (x, routes, scores)
    adapted = np.array(forward(*args))
    switch.down_proj = original
    baseline = np.array(forward(*args))
    assert np.max(np.abs(adapted - baseline)) > 1e-4


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("duplicate", "duplicate expert"),
        ("missing", "needs lora_A and lora_B"),
        ("rank", "matching expert counts"),
        ("experts", "has 4 experts, but"),
    ],
)
def test_invalid_routed_adapter_is_rejected(
    write_lora_adapter, tmp_path, case, message
):
    from mlx_lm.models.switch_layers import SwitchGLU

    stem = "model.layers.0.mlp.experts"
    target = stem + ".down_proj"
    path = write_lora_adapter(
        tmp_path / "routed",
        rank=2,
        modules={"model.layers.0.mlp.dense": (8, 8)},
        config={
            "target_parameters": [target, target] if case == "duplicate" else [target]
        },
        tensors=(
            {
                f"base_model.model.{stem}.lora_A.weight": np.ones((8, 64), np.float32),
                f"base_model.model.{stem}.lora_B.weight": np.ones(
                    (32, 7 if case == "rank" else 8), np.float32
                ),
            }
            if case != "missing"
            else None
        ),
    )
    if case != "experts":
        with pytest.raises(AdapterError, match=message):
            parse_adapter(path)
        return
    model, layer = nn.Module(), nn.Module()
    model.layers = [layer]
    layer.mlp = nn.Module()
    dense = layer.mlp.dense = nn.Linear(8, 8)
    switch = layer.mlp.switch_mlp = SwitchGLU(32, 64, 3)
    down = switch.down_proj
    binding = bind_adapter(parse_adapter(path), BaseRef("base", tmp_path))
    with pytest.raises(AdapterShapeError, match=message):
        attach_lora(model, binding)
    assert layer.mlp.dense is dense
    assert switch.down_proj is down
    assert lora_revision_of(model) is None


@pytest.mark.skipif(not oqh_fast.has_native(), reason="oQh kernels not built")
@pytest.mark.parametrize("group_size", [None, 128], ids=["row", "group"])
def test_lora_model_entry_runs_on_fused_oqh_layers(
    tmp_path, write_lora_adapter, group_size
):
    path, ref, _ = _tiny_qwen(tmp_path, True, group_size)
    install_load_hook()
    model, _ = load_model(path, strict=True)
    layer = model.model.layers[0]
    rank, experts = 2, layer.mlp.num_experts
    stem = "model.layers.0.mlp.experts"
    rng = np.random.default_rng(42)
    a = rng.normal(0, 0.1, (experts, rank, 256)).astype(np.float32)
    b = rng.normal(0, 0.1, (256, rank, experts)).astype(np.float32)
    adapter = write_lora_adapter(
        tmp_path / "adapter",
        rank=rank,
        alpha=4,
        modules={
            "model.layers.0.mlp.shared_expert.down_proj": (256, 256),
            "model.layers.0.linear_attn.out_proj": (1024, 256),
        },
        tensors={
            f"base_model.model.{stem}.lora_A.weight": a.reshape(experts * rank, 256),
            f"base_model.model.{stem}.lora_B.weight": b.reshape(256, rank * experts),
        },
        config={"target_parameters": [stem + ".down_proj"]},
    )
    weights = mx.load(str(adapter / WEIGHTS))
    cases = []
    for name, projection in (
        ("mlp.shared_expert.down_proj", layer.mlp.shared_expert.down_proj),
        ("linear_attn.out_proj", layer.linear_attn.out_proj),
    ):
        x = mx.array(rng.standard_normal((1, 3, projection.input_dims))).astype(
            mx.bfloat16
        )
        prefix = f"base_model.model.model.layers.0.{name}"
        expected = np.array(projection(x).astype(mx.float32)) + folded_output(
            np.array(x.astype(mx.float32)),
            np.zeros((projection.output_dims, projection.input_dims)),
            None,
            np.array(weights[f"{prefix}.lora_A.weight"]),
            np.array(weights[f"{prefix}.lora_B.weight"]),
            2,
        )
        cases.append((name, x, expected))
    routed = []
    delta = 2 * np.einsum("ore,eri->eoi", b, a)
    for routes, sorted_indices in (
        (np.array([[2, 0], [1, 2]]), False),
        (np.array([0, 1, 2, 2]), True),
    ):
        x = mx.array(rng.standard_normal((*routes.shape, 1, 256))).astype(mx.bfloat16)
        baseline = layer.mlp.switch_mlp.down_proj(x, mx.array(routes), sorted_indices)
        expected = np.array(baseline.astype(mx.float32)) + np.array(
            x.astype(mx.float32)
        ) @ np.swapaxes(delta[routes], -1, -2)
        routed.append((x, routes, sorted_indices, expected))
    binding = bind_adapter(parse_adapter(adapter), BaseRef("base", path))
    assert attach_lora(model, binding) == attach_lora(ref, binding) == 3
    for name, x, expected in cases:
        actual = dict(layer.named_modules())[name](x)
        np.testing.assert_allclose(
            np.array(actual.astype(mx.float32)), expected, atol=0.01, rtol=0.01
        )
    for x, routes, sorted_indices, expected in routed:
        actual = layer.mlp.switch_mlp.down_proj(x, mx.array(routes), sorted_indices)
        np.testing.assert_allclose(
            np.array(actual.astype(mx.float32)), expected, atol=0.01, rtol=0.01
        )
    for length in (1, 40):
        x = mx.array(rng.standard_normal((1, length, 256))).astype(mx.bfloat16)
        actual = layer.mlp(x)
        np.testing.assert_allclose(
            np.array(actual.astype(mx.float32)),
            np.array(ref.model.layers[0].mlp(x).astype(mx.float32)),
            atol=0.01,
            rtol=0.03,
        )


def effective_parts(linear, quant):
    """The weight a Linear or QuantizedLinear applies, and its bias, in float32."""
    if quant is None:
        weight = linear.weight
    else:
        mode, bits, group_size = quant
        weight = mx.dequantize(
            linear.weight,
            linear.scales,
            linear.get("biases"),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
    bias = np.array(linear.bias.astype(mx.float32)) if "bias" in linear else None
    return np.array(weight.astype(mx.float32)), bias


def folded_output(x, weight, bias, lora_a, lora_b, scale):
    """x through W + scale * B @ A, the merged-weight form of the adapter."""
    merged = weight.astype(np.float64) + scale * (
        lora_b.astype(np.float64) @ lora_a.astype(np.float64)
    )
    out = x.astype(np.float64) @ merged.T
    return out if bias is None else out + bias


def module_types(model):
    return {name: type(module) for name, module in model.named_modules()}


class TestInkling2Lora:
    @pytest.fixture
    def model(self, request):
        from test_inkling_vlm_mtp import _inkling2_tiny

        model = _inkling2_tiny(formats=("mxfp8",) * 2, local_layer_ids=[])
        model.model_type, vlm = request.param
        if vlm:
            from omlx.patches.inkling2.vlm import Model

            model.args.model_type = model.model_type
            facade = Model(model.args)
            facade.update(model.parameters())
            model = facade
        return model

    @pytest.mark.parametrize(
        "model",
        [
            ("inkling2", False),
            ("inkling2_loop", False),
            ("inkling2", True),
            ("inkling2_loop", True),
        ],
        indirect=True,
        ids=["lm", "loop-lm", "vlm", "loop-vlm"],
    )
    def test_hf_output_projections_match_folded_lora(
        self, model, write_lora_adapter, tmp_path
    ):
        rng = np.random.default_rng(42)
        tensors, cases = {}, []
        for layer in model.model.layers:
            projection = layer.attn.o_proj
            outputs, inputs = projection.weight.shape
            a = (rng.standard_normal((64, inputs)) * 0.1).astype(np.float32)
            b = (rng.standard_normal((outputs, 64)) * 0.1).astype(np.float32)
            path = f"language_model.model.layers.{len(cases)}.self_attn.o_proj"
            stem = f"base_model.model.{path}"
            tensors[f"{stem}.lora_A.weight"] = a
            tensors[f"{stem}.lora_B.weight"] = b
            x = rng.standard_normal((3, inputs)).astype(np.float32)
            base = np.array(projection(mx.array(x)))
            expected = folded_output(x, np.array(projection.weight), None, a, b, 1.0)
            cases.append((layer, x, base, expected))
        adapter = write_lora_adapter(
            tmp_path / "ad",
            base_ref="thinkingmachines/Inkling-2-Mini",
            rank=64,
            alpha=64,
            modules={},
            tensors=tensors,
        )

        assert attach_lora(model, bind(adapter)) == len(cases)

        for layer, x, base, expected in cases:
            actual = np.array(layer.attn.o_proj(mx.array(x)))
            assert np.abs(actual - base).max() > 0.1
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        ("model", "target"),
        [
            (("unrelated", False), "self_attn.o_proj"),
            (("inkling2_loop", True), "self_attn.q_proj"),
            (("inkling2_loop", False), "self_attn.k_proj"),
            (("inkling2", False), "self_attn.v_proj"),
            (("inkling2_loop", True), "self_attn.missing_proj"),
        ],
        indirect=["model"],
    )
    def test_unknown_family_and_unsupported_targets_are_rejected(
        self, model, target, write_lora_adapter, tmp_path
    ):
        projection = getattr(model.model.layers[0].attn, target.split(".")[-1], None)
        outputs, inputs = (128, 128) if projection is None else projection.weight.shape
        path = f"language_model.model.layers.0.{target}"
        adapter = write_lora_adapter(
            tmp_path / "ad", rank=RANK, modules={path: (inputs, outputs)}
        )
        types_before = module_types(model)

        with pytest.raises(AdapterShapeError, match=re.escape(path)):
            attach_lora(model, bind(adapter))

        assert module_types(model) == types_before
        assert lora_revision_of(model) is None

    @pytest.mark.parametrize("model", [("inkling2_loop", True)], indirect=True)
    def test_a_missing_runtime_output_projection_is_rejected(
        self, model, write_lora_adapter, tmp_path
    ):
        del model.model.layers[0].attn.o_proj
        path = "language_model.model.layers.0.self_attn.o_proj"
        adapter = write_lora_adapter(
            tmp_path / "ad", rank=RANK, modules={path: (128, 128)}
        )
        types_before = module_types(model)

        with pytest.raises(AdapterShapeError, match=re.escape(path)):
            attach_lora(model, bind(adapter))

        assert module_types(model) == types_before
        assert lora_revision_of(model) is None

    @pytest.mark.parametrize("model", [("inkling2", False)], indirect=True)
    def test_an_exact_projection_takes_precedence(
        self, model, write_lora_adapter, tmp_path
    ):
        layer = model.model.layers[0]
        layer.self_attn = nn.Module()
        layer.self_attn.o_proj = nn.Linear(128, 128, bias=False)
        path = "language_model.model.layers.0.self_attn.o_proj"
        a, b = lora_factors(128, 128, seed=3)
        stem = f"base_model.model.{path}"
        adapter = write_lora_adapter(
            tmp_path / "ad",
            rank=RANK,
            alpha=ALPHA,
            modules={},
            tensors={f"{stem}.lora_A.weight": a, f"{stem}.lora_B.weight": b},
        )
        x = np.ones((2, 128), dtype=np.float32)
        unchanged = np.array(layer.attn.o_proj(mx.array(x)))
        expected = folded_output(
            x, np.array(layer.self_attn.o_proj.weight), None, a, b, PLAIN_SCALE
        )

        assert attach_lora(model, bind(adapter)) == 1

        np.testing.assert_allclose(
            np.array(layer.self_attn.o_proj(mx.array(x))), expected, atol=1e-4
        )
        np.testing.assert_array_equal(
            np.array(layer.attn.o_proj(mx.array(x))), unchanged
        )


class TestAttachLora:
    @pytest.fixture(autouse=True)
    def seed(self):
        mx.random.seed(7)

    @pytest.mark.parametrize(
        ("layout_name", "quant"),
        [
            ("llm", None),
            ("vlm", None),
            ("flat", None),
            ("llm", ("affine", 4, 32)),
            ("vlm", ("affine", 4, 32)),
            ("llm", ("affine", 8, 64)),
            ("llm", ("mxfp4", 4, 32)),
        ],
        ids=["llm", "vlm", "flat", "llm-4bit", "vlm-4bit", "llm-8bit", "llm-mxfp4"],
    )
    def test_wrapped_modules_match_the_folded_weights(
        self, write_lora_adapter, tmp_path, layout_name, quant
    ):
        layout = LAYOUTS[layout_name]
        model = layout.build()
        if quant is not None:
            mode, bits, group_size = quant
            nn.quantize(model, group_size=group_size, bits=bits, mode=mode)
        mx.eval(model.parameters())
        adapter, factors = write_adapter(
            write_lora_adapter, tmp_path / "ad", layout.hf_prefix
        )
        layers = layout.layers(model)
        rng = np.random.default_rng(11)
        cases = {}
        for (layer, name), (lora_a, lora_b) in factors.items():
            weight, bias = effective_parts(getattr(layers[layer].mlp, name), quant)
            x = rng.standard_normal((3, TARGETS[name][0])).astype(np.float32)
            without_adapter = x @ weight.T + (0 if bias is None else bias)
            expected = folded_output(x, weight, bias, lora_a, lora_b, PLAIN_SCALE)
            cases[layer, name] = (x, without_adapter, expected)

        assert attach_lora(model, bind(adapter)) == len(factors)

        for (layer, name), (x, without_adapter, expected) in cases.items():
            wrapped = getattr(layers[layer].mlp, name)
            assert isinstance(wrapped, LoraLinear)
            assert np.abs(expected - without_adapter).max() > 0.1
            np.testing.assert_allclose(
                np.array(wrapped(mx.array(x))), expected, rtol=1e-4, atol=1e-4
            )

    @pytest.mark.parametrize("layout_name", list(LAYOUTS))
    def test_model_output_matches_a_model_with_folded_weights(
        self, write_lora_adapter, tmp_path, layout_name
    ):
        layout = LAYOUTS[layout_name]
        model = layout.build()
        mx.eval(model.parameters())
        reference = layout.build()
        reference.update(model.parameters())
        adapter, factors = write_adapter(
            write_lora_adapter, tmp_path / "ad", layout.hf_prefix
        )
        for (layer, name), (lora_a, lora_b) in factors.items():
            linear = getattr(layout.layers(reference)[layer].mlp, name)
            delta = mx.array(lora_b) @ mx.array(lora_a)
            linear.weight = linear.weight + PLAIN_SCALE * delta
        x = mx.random.normal((2, 5, DIMS))

        attach_lora(model, bind(adapter))

        np.testing.assert_allclose(
            np.array(model(x)), np.array(reference(x)), rtol=1e-4, atol=1e-4
        )

    def test_rslora_scales_by_alpha_over_sqrt_rank(self, write_lora_adapter, tmp_path):
        layout = LAYOUTS["llm"]
        model = layout.build()
        mx.eval(model.parameters())
        adapter, factors = write_adapter(
            write_lora_adapter,
            tmp_path / "ad",
            layout.hf_prefix,
            config={"use_rslora": True},
        )
        linear = layout.layers(model)[0].mlp.gate_proj
        weight, bias = effective_parts(linear, None)
        lora_a, lora_b = factors[0, "gate_proj"]
        x = np.random.default_rng(3).standard_normal((3, DIMS)).astype(np.float32)
        expected = folded_output(x, weight, bias, lora_a, lora_b, RSLORA_SCALE)
        plain = folded_output(x, weight, bias, lora_a, lora_b, PLAIN_SCALE)

        attach_lora(model, bind(adapter))

        assert np.abs(expected - plain).max() > 0.1
        np.testing.assert_allclose(
            np.array(layout.layers(model)[0].mlp.gate_proj(mx.array(x))),
            expected,
            rtol=1e-4,
            atol=1e-4,
        )

    def test_output_dtype_follows_the_input_dtype(self, write_lora_adapter, tmp_path):
        layout = LAYOUTS["llm"]
        model = layout.build()
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())
        adapter, factors = write_adapter(
            write_lora_adapter, tmp_path / "ad", layout.hf_prefix
        )
        weight, bias = effective_parts(layout.layers(model)[0].mlp.gate_proj, None)
        lora_a, lora_b = factors[0, "gate_proj"]
        x = mx.array(np.random.default_rng(3).standard_normal((3, DIMS))).astype(
            mx.bfloat16
        )
        expected = folded_output(
            np.array(x.astype(mx.float32)), weight, bias, lora_a, lora_b, PLAIN_SCALE
        )

        attach_lora(model, bind(adapter))
        out = layout.layers(model)[0].mlp.gate_proj(x)

        assert out.dtype == mx.bfloat16
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)), expected, rtol=5e-2, atol=5e-2
        )

    @pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
    def test_adapter_weights_keep_the_dtype_they_shipped_in(
        self, write_lora_adapter, tmp_path, dtype
    ):
        layout = LAYOUTS["llm"]
        model = layout.build()
        adapter, _ = write_adapter(
            write_lora_adapter, tmp_path / "ad", layout.hf_prefix, dtype=dtype
        )

        attach_lora(model, bind(adapter))

        wrapped = layout.layers(model)[0].mlp.gate_proj
        assert (wrapped.lora_a.dtype, wrapped.lora_b.dtype) == (dtype, dtype)

    def test_the_revision_appears_with_the_adapter(self, write_lora_adapter, tmp_path):
        layout = LAYOUTS["llm"]
        model = layout.build()
        adapter, factors = write_adapter(
            write_lora_adapter, tmp_path / "ad", layout.hf_prefix
        )
        binding = bind(adapter)
        assert lora_revision_of(model) is None

        wrapped = attach_lora(model, binding)

        assert wrapped == len(factors)
        assert lora_revision_of(model) == binding.cache_revision
        assert lora_revision_of(object()) is None

    def test_only_the_language_model_is_wrapped(self, write_lora_adapter, tmp_path):
        layout = LAYOUTS["vlm"]
        model = layout.build()
        adapter, factors = write_adapter(
            write_lora_adapter, tmp_path / "ad", layout.hf_prefix
        )

        assert attach_lora(model, bind(adapter)) == len(factors)

        wrapped = [
            m for m in model.language_model.modules() if isinstance(m, LoraLinear)
        ]
        assert len(wrapped) == len(factors)
        assert not any(isinstance(m, LoraLinear) for m in model.vision_tower.modules())

    def test_wrapped_parameters_are_arrays_only(self, write_lora_adapter, tmp_path):
        layout = LAYOUTS["llm"]
        model = layout.build()
        adapter, _ = write_adapter(
            write_lora_adapter, tmp_path / "ad", layout.hf_prefix
        )

        attach_lora(model, bind(adapter))

        assert all(isinstance(v, mx.array) for _, v in tree_flatten(model.parameters()))
        mx.eval(model.parameters())


class TestLoraLinear:
    def test_computes_the_low_rank_update_in_the_documented_orientation(self):
        rng = np.random.default_rng(1)
        lora_a = rng.standard_normal((8, 4)).astype(np.float32)
        lora_b = rng.standard_normal((4, 16)).astype(np.float32)
        x = rng.standard_normal((3, 8)).astype(np.float32)
        linear = nn.Linear(8, 16, bias=False)
        wrapper = LoraLinear(linear, mx.array(lora_a), mx.array(lora_b), 0.5, "lora-x")

        expected = x @ np.array(linear.weight).T + 0.5 * (x @ lora_a @ lora_b)

        np.testing.assert_allclose(
            np.array(wrapper(mx.array(x))), expected, rtol=1e-4, atol=1e-4
        )

    def test_scale_and_revision_are_not_parameters(self):
        wrapper = LoraLinear(
            nn.Linear(8, 16), mx.zeros((8, 4)), mx.zeros((4, 16)), 2.0, "lora-x"
        )

        names = {name for name, _ in tree_flatten(wrapper.parameters())}

        assert names == {"linear.weight", "linear.bias", "lora_a", "lora_b"}
        assert (wrapper.scale, wrapper.revision) == (2.0, "lora-x")


VALID_MODULES = {
    "model.layers.0.mlp.down_proj": (HIDDEN, DIMS),
    "model.layers.0.mlp.gate_proj": (DIMS, HIDDEN),
    "model.layers.1.mlp.down_proj": (HIDDEN, DIMS),
}


def bind_claiming(adapter, modules):
    """A binding whose source lists ``modules``, whatever the weights file holds."""
    return bind_adapter(
        replace(parse_adapter(adapter), modules=tuple(modules)),
        BaseRef("tiny", Path("/models/tiny")),
    )


def assert_refused(model, binding):
    """attach_lora must raise AdapterShapeError and leave the model as it was."""
    x = mx.random.normal((2, DIMS))
    types_before = module_types(model)
    output_before = np.array(model(x))

    with pytest.raises(AdapterShapeError) as caught:
        attach_lora(model, binding)

    assert caught.value.details["adapter"] == str(binding.source.path)
    assert lora_revision_of(model) is None
    assert not any(isinstance(m, LoraLinear) for m in model.modules())
    assert module_types(model) == types_before
    np.testing.assert_array_equal(np.array(model(x)), output_before)
    return caught.value


def assert_rejected(model, binding, offender):
    """assert_refused, and the error names ``offender``, the module that fails.

    ``offender`` sorts last in the adapter, so an implementation that installed
    wrappers while it validated would leave some behind. Returns the error.
    """
    assert binding.source.modules[-1].hf_path == offender
    error = assert_refused(model, binding)
    assert offender in error.message
    return error


class TestAttachLoraRejections:
    @pytest.fixture(autouse=True)
    def seed(self):
        mx.random.seed(7)

    def adapter(self, write_lora_adapter, tmp_path, extra):
        modules = {**VALID_MODULES, **extra}
        return write_lora_adapter(
            tmp_path / "ad", rank=RANK, alpha=ALPHA, modules=modules
        )

    def test_feature_size_mismatch(self, write_lora_adapter, tmp_path):
        bad = "model.layers.1.mlp.gate_proj"
        trained_inputs = DIMS + 8
        adapter = self.adapter(
            write_lora_adapter, tmp_path, {bad: (trained_inputs, HIDDEN)}
        )

        error = assert_rejected(LlmModel(), bind(adapter), bad)

        assert str(trained_inputs) in error.message
        assert str(DIMS) in error.message
        assert "Reload" not in error.message

    def test_unresolved_target(self, write_lora_adapter, tmp_path):
        bad = "model.layers.5.mlp.gate_proj"
        adapter = self.adapter(write_lora_adapter, tmp_path, {bad: (DIMS, HIDDEN)})

        assert_rejected(LlmModel(), bind(adapter), bad)

    @pytest.mark.parametrize(
        "bad",
        [
            "model.visual.blocks.0.attn.qkv",
            "model.vision_tower.encoder.layers.0.mlp.gate_proj",
            "model.mtp.layers.0.mlp.gate_proj",
        ],
    )
    def test_target_outside_language_layers(self, write_lora_adapter, tmp_path, bad):
        adapter = write_lora_adapter(
            tmp_path / "ad",
            rank=RANK,
            alpha=ALPHA,
            modules={
                "model.layers.0.mlp.down_proj": (HIDDEN, DIMS),
                bad: (DIMS, HIDDEN),
            },
        )

        assert_rejected(LlmModel(), bind(adapter), bad)

    def test_a_module_that_is_not_a_linear(self, write_lora_adapter, tmp_path):
        bad = "model.layers.1.mlp.gate_proj"
        model = LlmModel()
        model.model.layers[1].mlp.gate_proj = OpaqueLinear(
            nn.Linear(DIMS, HIDDEN, bias=False)
        )
        adapter = self.adapter(write_lora_adapter, tmp_path, {bad: (DIMS, HIDDEN)})

        assert_rejected(model, bind(adapter), bad)

    def test_two_adapter_modules_for_one_target(self, write_lora_adapter, tmp_path):
        modules = {
            "model.language_model.layers.0.mlp.gate_proj": (DIMS, HIDDEN),
            "model.layers.0.mlp.down_proj": (HIDDEN, DIMS),
            "model.layers.0.mlp.gate_proj": (DIMS, HIDDEN),
        }
        adapter = write_lora_adapter(
            tmp_path / "ad", rank=RANK, alpha=ALPHA, modules=modules
        )

        assert_rejected(LlmModel(), bind(adapter), "model.layers.0.mlp.gate_proj")

    def test_a_target_that_resolves_two_ways(self, write_lora_adapter, tmp_path):
        bad = "model.layers.0.mlp.gate_proj"
        adapter = write_lora_adapter(
            tmp_path / "ad", rank=RANK, alpha=ALPHA, modules={bad: (DIMS, HIDDEN)}
        )

        assert_rejected(TwoRoots(), bind(adapter), bad)

    def test_a_module_the_weights_do_not_hold(self, write_lora_adapter, tmp_path):
        ghost = "model.layers.1.mlp.gate_proj"
        adapter = self.adapter(write_lora_adapter, tmp_path, {})
        listed = [*parse_adapter(adapter).modules, LoraModule(ghost, DIMS, HIDDEN)]

        error = assert_rejected(LlmModel(), bind_claiming(adapter, listed), ghost)

        assert "Reload" in error.message

    @pytest.mark.parametrize("half", ["A", "B"])
    def test_a_tensor_that_disagrees_with_the_listed_shape(
        self, write_lora_adapter, tmp_path, half
    ):
        last = "model.layers.1.mlp.down_proj"
        in_features, out_features = VALID_MODULES[last]
        reshaped = {"A": (RANK, in_features + 8), "B": (out_features + 8, RANK)}[half]
        adapter = write_lora_adapter(
            tmp_path / "ad",
            rank=RANK,
            alpha=ALPHA,
            modules=VALID_MODULES,
            tensors={
                f"base_model.model.{last}.lora_{half}.weight": np.zeros(
                    reshaped, np.float32
                )
            },
        )
        listed = [
            LoraModule(module.hf_path, *VALID_MODULES[module.hf_path])
            for module in parse_adapter(adapter).modules
        ]

        error = assert_rejected(LlmModel(), bind_claiming(adapter, listed), last)

        assert "Reload" in error.message

    @pytest.mark.parametrize(
        ("first_layers", "second_layers"),
        [((0, 1), (0, 1)), ((0,), (1,))],
        ids=["same-targets", "other-targets"],
    )
    def test_a_second_adapter_is_rejected_and_the_first_stays(
        self, write_lora_adapter, tmp_path, first_layers, second_layers
    ):
        layout = LAYOUTS["llm"]
        model = layout.build()
        first, _ = write_adapter(
            write_lora_adapter,
            tmp_path / "first",
            layout.hf_prefix,
            layers=first_layers,
        )
        second, _ = write_adapter(
            write_lora_adapter,
            tmp_path / "second",
            layout.hf_prefix,
            config={"lora_alpha": 16.0},
            layers=second_layers,
        )
        first_binding = bind(first)
        attach_lora(model, first_binding)
        x = mx.random.normal((2, DIMS))
        output_before = np.array(model(x))

        with pytest.raises(AdapterShapeError) as caught:
            attach_lora(model, bind(second))

        assert caught.value.details["adapter"] == str(second)
        assert lora_revision_of(model) == first_binding.cache_revision
        np.testing.assert_array_equal(np.array(model(x)), output_before)
        wrappers = [m for m in model.modules() if isinstance(m, LoraLinear)]
        assert len(wrappers) == len(TARGETS) * len(first_layers)
        assert not any(isinstance(w.linear, LoraLinear) for w in wrappers)


def retrain_in_place(adapter, write_lora_adapter):
    """New values of the same shapes. Returns {(layer, name): (A, B)}."""
    _, factors = write_adapter(
        write_lora_adapter, adapter, LAYOUTS["llm"].hf_prefix, seed=1000
    )
    bump_mtime(adapter)
    return factors


def overwrite_the_weights_with_garbage(adapter, write_lora_adapter):
    (adapter / WEIGHTS).write_bytes(b"not safetensors")


def delete_the_config(adapter, write_lora_adapter):
    (adapter / CONFIG).unlink()


def delete_the_weights(adapter, write_lora_adapter):
    (adapter / WEIGHTS).unlink()


def edit_the_config(adapter, write_lora_adapter):
    edit_alpha_keeping_the_length(adapter)


CHANGES_AFTER_DISCOVERY = {
    "retrained-weights": retrain_in_place,
    "config-edited": edit_the_config,
    "garbage-weights": overwrite_the_weights_with_garbage,
    "config-deleted": delete_the_config,
    "weights-deleted": delete_the_weights,
}


class TestAttachLoraAfterTheFilesChanged:
    """The binding came from files that have since changed on disk."""

    @pytest.fixture(autouse=True)
    def seed(self):
        mx.random.seed(7)

    @pytest.mark.parametrize(
        "change",
        list(CHANGES_AFTER_DISCOVERY.values()),
        ids=list(CHANGES_AFTER_DISCOVERY),
    )
    def test_the_model_is_left_unmodified(self, write_lora_adapter, tmp_path, change):
        layout = LAYOUTS["llm"]
        adapter, _ = write_adapter(
            write_lora_adapter, tmp_path / "ad", layout.hf_prefix
        )
        binding = bind(adapter)
        change(adapter, write_lora_adapter)

        error = assert_refused(layout.build(), binding)

        assert "Reload the model list" in error.message

    def test_a_reload_attaches_the_replaced_weights(self, write_lora_adapter, tmp_path):
        layout = LAYOUTS["llm"]
        adapter, before = write_adapter(
            write_lora_adapter, tmp_path / "ad", layout.hf_prefix
        )
        stale = bind(adapter)
        after = retrain_in_place(adapter, write_lora_adapter)
        refreshed = bind(adapter)
        assert refreshed.source.modules == stale.source.modules
        assert refreshed.source.weight_bytes == stale.source.weight_bytes
        model = layout.build()
        mx.eval(model.parameters())
        linear = layout.layers(model)[0].mlp.gate_proj
        weight, bias = effective_parts(linear, None)
        x = np.random.default_rng(3).standard_normal((3, DIMS)).astype(np.float32)
        old_a, old_b = before[0, "gate_proj"]
        new_a, new_b = after[0, "gate_proj"]
        from_old = folded_output(x, weight, bias, old_a, old_b, PLAIN_SCALE)
        expected = folded_output(x, weight, bias, new_a, new_b, PLAIN_SCALE)

        attach_lora(model, refreshed)

        assert np.abs(expected - from_old).max() > 0.1
        np.testing.assert_allclose(
            np.array(layout.layers(model)[0].mlp.gate_proj(mx.array(x))),
            expected,
            rtol=1e-4,
            atol=1e-4,
        )
        assert lora_revision_of(model) == refreshed.cache_revision
        assert refreshed.cache_revision != stale.cache_revision


@pytest.mark.parametrize(
    ("target", "forbidden"),
    [
        ("omlx.model_discovery", ("omlx.lora_adapter",)),
        (
            "omlx.lora_adapter",
            ("omlx.model_discovery", "mlx_vlm.trainer", "mlx_lm.tuner"),
        ),
    ],
)
def test_import_boundaries(target, forbidden):
    code = (
        "import sys\n"
        f"import {target}\n"
        f"loaded = sorted(n for n in sys.modules if n.startswith({forbidden!r}))\n"
        "print(loaded)\n"
        "sys.exit(bool(loaded))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
