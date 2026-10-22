# SPDX-License-Identifier: Apache-2.0
"""Native Flint loading and original-BF16 streamed oQ calibration."""

import json
from dataclasses import asdict
from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.flint.config import ModelConfig
from mlx_vlm.models.flint.flint import Model
from mlx_vlm.models.flint.language import LanguageModel, SparseMoeBlock
from mlx_vlm.utils import load_model

import omlx.oq as oq


def _tiny_config(window=3):
    return ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        no_rope_layer_indices=[0, 2],
        mlp_only_layers=[0],
        num_experts=2,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        sliding_window=window,
        eos_token_id=[2],
    )


@pytest.fixture
def checkpoint(tmp_path):
    mx.random.seed(17)
    args = _tiny_config()
    model = Model(args)
    model.eval()
    weights = {
        key: value.astype(mx.bfloat16)
        for key, value in tree_flatten(model.parameters())
    }
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    original = {}
    for key, value in weights.items():
        key = key.removeprefix("language_model.")
        if ".switch_mlp." in key:
            prefix, projection = key.split(".switch_mlp.")
            for expert in range(args.num_experts):
                original[f"{prefix}.experts.{expert}.{projection}"] = value[expert]
        elif key.endswith(".expert_bias"):
            continue
        else:
            original[key] = value
    source = tmp_path / "source"
    source.mkdir()
    config = {**asdict(args), "architectures": ["FlintForCausalLM"]}
    (source / "config.json").write_text(json.dumps(config))
    shards = [{}, {}]
    mapping = {}
    for index, (key, value) in enumerate(original.items()):
        shards[index % 2][key] = value
        mapping[key] = f"model-{index % 2}.safetensors"
    for index, shard in enumerate(shards):
        mx.save_safetensors(str(source / f"model-{index}.safetensors"), shard)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    yield source, config, model, original
    mx.synchronize()
    mx.clear_cache()


def test_sanitizer_never_constructs_model_and_matches_native(checkpoint, monkeypatch):
    source, config, model, original = checkpoint
    expected = model.sanitize(original)
    for cls in (Model, LanguageModel, SparseMoeBlock):
        monkeypatch.setattr(
            cls, "__init__", Mock(side_effect=AssertionError("weights"))
        )
    sanitizer = oq._build_model_sanitizer(config, model_path=source)
    actual = sanitizer(original)
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_array_equal(
            np.asarray(actual[key].astype(mx.float32)),
            np.asarray(expected[key].astype(mx.float32)),
        )
    plan = oq._streamed_source_plan(source, config)
    assert plan.keys() == expected.keys()


def test_original_cross_shard_checkpoint_strictly_loads(checkpoint):
    source, _, reference, _ = checkpoint
    loaded = load_model(source, lazy=False, strict=True)
    tokens = mx.array([[1, 3, 4, 5, 6, 7]])
    np.testing.assert_array_equal(
        np.asarray(loaded(tokens).logits), np.asarray(reference(tokens).logits)
    )


@pytest.mark.parametrize("prefix", ["", "language_model."])
@pytest.mark.parametrize("alias", ["expert_bias", "e_score_correction_bias"])
def test_native_sanitizer_preserves_router_bias_aliases(checkpoint, prefix, alias):
    source, config, model, original = checkpoint
    weights = {prefix + key: value for key, value in original.items()}
    bias = mx.array([0.125, -0.25], dtype=mx.bfloat16)
    key = f"{prefix}model.layers.1.mlp.gate.{alias}"
    weights[key] = bias
    sanitizer = oq._build_model_sanitizer(config, model_path=source)
    sanitized = sanitizer(weights)
    assert mx.array_equal(
        sanitized["language_model.model.layers.1.mlp.expert_bias"], bias
    ).item()
    model.load_weights(list(sanitized.items()), strict=True)


@pytest.mark.parametrize("window,length", [(3, 2), (3, 3), (3, 6), (513, 514)])
def test_layer_walk_matches_native_masks_and_logits(window, length):
    mx.random.seed(5)
    model = Model(_tiny_config(window))
    model.eval()
    model.set_dtype(mx.bfloat16)
    tokens = (mx.arange(length)[None, :] % 100).astype(mx.int32)
    inputs = model.language_model.model.embed_tokens(tokens)
    inputs, masks, state = oq._prepare_layer_inputs(model, model.layers, tokens, inputs)
    for index, layer in enumerate(model.layers):
        inputs, _ = oq._forward_layer_result(
            layer, inputs, masks[index], state, layer_idx=index
        )
    hidden = model.language_model.model.norm(inputs).astype(mx.float32)
    actual = model.language_model.lm_head(hidden)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(model(tokens).logits))


@pytest.mark.parametrize("window", [3, 513])
@pytest.mark.parametrize("quantized", [False, True])
def test_native_cache_continuation_state(window, quantized, record_property):
    mx.random.seed(5)
    model = Model(_tiny_config(window))
    model.eval()
    model.set_dtype(mx.bfloat16)
    if quantized:
        nn.quantize(model, group_size=64, bits=4)
    tokens = (mx.arange(514)[None, :] % 100).astype(mx.int32)
    full = model(tokens).logits[:, -1, :]
    empty_cache = model.make_cache()
    cached_full = model(tokens, cache=empty_cache).logits[:, -1, :]
    assert mx.array_equal(full, cached_full).item()
    cache = model.make_cache()
    mx.eval(model(tokens[:, :512], cache=cache).logits)
    continued = model(tokens[:, 512:], cache=cache).logits[:, -1, :]
    mx.eval(full, continued)
    if quantized:
        error = mx.sqrt(mx.mean((full - continued) ** 2) / mx.mean(full**2))
        record_property("chunked_logit_relative_rms", error.item())
    else:
        assert mx.array_equal(full, continued).item()
    assert mx.isfinite(continued).all().item()
    assert all(item.offset == 514 for item in cache)


def test_oq6_reuses_bf16_imatrix_but_measures_its_own_sensitivity(
    checkpoint, tmp_path, monkeypatch
):
    source, config, _, _ = checkpoint
    tokens = (mx.arange(128).reshape(8, 16) % 100).astype(mx.int32)
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **kw: tokens)
    monkeypatch.setattr("mlx_lm.tokenizer_utils.load", lambda *a, **kw: object())
    cache_path = tmp_path / "bf16.npz"
    cached = oq._load_or_collect_imatrix(
        str(source),
        config,
        cache_path=str(cache_path),
        reuse_cache=True,
        num_samples=2,
        seq_length=16,
        strict=True,
        trust_remote_code=False,
        stream_calibration=True,
        measure_sensitivity=True,
        sensitivity_oq_level=4,
        sensitivity_num_samples=2,
        sensitivity_seq_length=16,
        require_mtp_entries=False,
    )
    original_cache = cache_path.read_bytes()
    measure = oq._measure_sensitivity_streaming

    def measure_small(model_path, model_config, level, **kwargs):
        return measure(
            model_path,
            model_config,
            level,
            num_samples=2,
            seq_length=16,
            trust_remote_code=False,
        )

    measured = Mock(side_effect=measure_small)
    monkeypatch.setattr(oq, "_measure_sensitivity_streaming", measured)
    monkeypatch.setattr(
        oq,
        "_collect_imatrix_streaming",
        Mock(side_effect=AssertionError("must reuse BF16 statistics")),
    )
    output = tmp_path / "oq6"
    oq.quantize_oq_streaming(
        str(source),
        str(output),
        6,
        enhanced=True,
        stream_calibration=True,
        auto_proxy_sensitivity=False,
        imatrix_strict=True,
        imatrix_cache_path=str(cache_path),
        imatrix_reuse_cache=True,
        imatrix_num_samples=2,
        imatrix_seq_length=16,
        preserve_mtp=False,
    )
    measured.assert_called_once()
    assert measured.call_args.args[2] == 6
    assert cache_path.read_bytes() == original_cache
    report = json.loads((output / "oq_imatrix_report.json").read_text())
    assert report["cache_reused"]
    assert report["entry_count"] == len(cached.entries)
    assert report["applied"] and not report["missing"] and not report["mismatched"]
    loaded = load_model(output, lazy=False, strict=True)
    assert mx.isfinite(loaded(tokens[:1]).logits).all().item()


def test_streamed_imatrix_sensitivity_and_cache_match_resident(
    checkpoint, tmp_path, monkeypatch
):
    source, config, model, _ = checkpoint
    tokens = (mx.arange(48).reshape(8, 6) % 100).astype(mx.int32)
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **kw: tokens)
    monkeypatch.setattr("mlx_lm.tokenizer_utils.load", lambda *a, **kw: object())
    real_batch_plan = oq._oqe_calibration_batch_plan

    def batch_plan(*args, **kwargs):
        return {**real_batch_plan(*args, **kwargs), "micro_batch_size": 2}

    monkeypatch.setattr(oq, "_oqe_calibration_batch_plan", batch_plan)
    resident, resident_meta = oq._collect_imatrix_from_model(
        model,
        object(),
        config,
        calib_dataset=oq._OQE_CALIB_DATASET,
        num_samples=2,
        seq_length=6,
    )
    expected_sensitivity = oq._measure_sensitivity_from_model(
        model, object(), config, 4, num_samples=2, seq_length=6
    )
    monkeypatch.setattr(
        "mlx_vlm.utils.load_model",
        Mock(side_effect=AssertionError("must not load the whole model")),
    )
    monkeypatch.setattr(
        oq,
        "_build_proxy_for_sensitivity",
        Mock(side_effect=AssertionError("proxy forbidden")),
    )
    cache = str(tmp_path / "imatrix.npz")
    kwargs = dict(
        cache_path=cache,
        reuse_cache=True,
        num_samples=2,
        seq_length=6,
        strict=True,
        trust_remote_code=False,
        stream_calibration=True,
        measure_sensitivity=True,
        sensitivity_oq_level=4,
        sensitivity_num_samples=2,
        sensitivity_seq_length=6,
        require_mtp_entries=False,
    )
    streamed = oq._load_or_collect_imatrix(str(source), config, **kwargs)
    assert "language_model.lm_head" in streamed.entries
    assert resident.keys() == streamed.entries.keys()
    assert streamed.metadata["processed_samples"] == resident_meta["processed_samples"]
    assert streamed.metadata["collection"]["load_kind"] == "streaming"
    assert len(streamed.metadata["collection"]["rounds"]) > 1
    assert streamed.metadata["layer_walk"] == "flint_native_masks_fp32_head_v1"
    for name, entry in resident.items():
        np.testing.assert_array_equal(streamed.entries[name].counts, entry.counts)
        np.testing.assert_array_equal(streamed.entries[name].in_sum2, entry.in_sum2)
    measured = streamed.metadata["collection"]["sensitivity_map"]
    assert measured.keys() == expected_sensitivity.keys()
    for index, score in expected_sensitivity.items():
        assert measured[index] == pytest.approx(score, abs=1e-6)
    assert oq._load_or_collect_imatrix(str(source), config, **kwargs).reused
    standalone = oq._measure_sensitivity_streaming(
        str(source), config, 4, num_samples=2, seq_length=6
    )
    assert standalone == pytest.approx(expected_sensitivity, abs=1e-6)
    incompatible = {**streamed.metadata, "layer_walk": "old_generic_mask"}
    assert not oq._oqe_cache_matches(streamed, incompatible)


def test_streamed_output_strictly_reloads_without_proxy(
    checkpoint, tmp_path, monkeypatch
):
    source, config, _, _ = checkpoint
    tokens = (mx.arange(128).reshape(8, 16) % 100).astype(mx.int32)
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **kw: tokens)
    monkeypatch.setattr("mlx_lm.tokenizer_utils.load", lambda *a, **kw: object())
    monkeypatch.setattr(
        oq,
        "_build_proxy_for_sensitivity",
        Mock(side_effect=AssertionError("proxy forbidden")),
    )
    output = tmp_path / "output"
    oq.quantize_oq_streaming(
        str(source),
        str(output),
        4,
        enhanced=True,
        stream_calibration=True,
        auto_proxy_sensitivity=False,
        imatrix_strict=True,
        imatrix_num_samples=2,
        imatrix_seq_length=16,
    )
    loaded = load_model(output, lazy=False, strict=True)
    assert mx.isfinite(loaded(tokens[:1]).logits).all().item()
    assert oq._is_vlm_load(config)
    assert oq._resolve_stream_calibration(
        None, model_exceeds_ram=True, model_type="flint"
    )
    assert (
        oq._source_imatrix_signature(
            source,
            config,
            num_samples=2,
            seq_length=16,
            calib_dataset=oq._OQE_CALIB_DATASET,
        )["layer_walk"]
        == "flint_native_masks_fp32_head_v1"
    )
