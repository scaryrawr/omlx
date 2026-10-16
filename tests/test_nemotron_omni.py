# SPDX-License-Identifier: Apache-2.0
"""Small native Nemotron contracts; no downloaded model weights."""

from __future__ import annotations

import json
from copy import copy, deepcopy
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.mlx_lm_mtp.uniform import sample_uniform_prefix
from omlx.patches.nemotron_omni import prepare_nemotron_prompt


def tiny_nemotron_config():
    return {
        "model_type": "nemotron_h_omni",
        "img_context_token_id": 18,
        "projector_hidden_size": 32,
        "vit_hidden_size": 16,
        "ps_version": "v2",
        "video_pruning_rate": 0.7,
        "llm_config": {
            "model_type": "nemotron_h",
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 32,
            "max_position_embeddings": 2048,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "attention_bias": False,
            "head_dim": 8,
            "mamba_num_heads": 4,
            "mamba_head_dim": 16,
            "mamba_proj_bias": False,
            "ssm_state_size": 32,
            "conv_kernel": 4,
            "n_groups": 1,
            "use_conv_bias": True,
            "mlp_bias": False,
            "layer_norm_epsilon": 1e-5,
            "use_bias": False,
            "layers_block_type": ["mamba", "moe", "attention"],
            "time_step_limit": [0.0, float("inf")],
            "num_nextn_predict_layers": 1,
            "mtp_layers_block_type": ["attention", "moe"],
            "moe_latent_size": 32,
            "moe_intermediate_size": 32,
            "moe_shared_expert_intermediate_size": 32,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.0,
        },
        "vision_config": {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 32,
            "patch_size": 16,
            "max_resolution": 64,
            "video_temporal_patch_size": 2,
            "args": {
                "teachers": [{"name": 0}, {"name": 1}, {"name": 2}],
                "cls_token_per_teacher": True,
                "register_multiple": 10,
            },
        },
    }


@pytest.fixture
def native_target():
    from mlx_vlm.models.nemotron_h_nano_omni import Model, ModelConfig

    mx.random.seed(71)
    model = Model(ModelConfig.from_dict(tiny_nemotron_config()))
    model.eval()
    return model


@pytest.mark.parametrize(
    "limits,stop_first,replace",
    [
        ((1, 1), False, False),
        ((2, 2), False, False),
        ((3, 9), False, False),
        ((9, 9), False, False),
        ((9, 9), True, False),
        ((9, 9), False, True),
    ],
)
def test_native_mtp_batch_generator_serves_two_unequal_prompts(
    native_target, monkeypatch, limits, stop_first, replace
):
    from mlx_lm.generate import BatchGenerator, StopSequences
    from mlx_vlm.speculative.drafters.nemotron_h_mtp import Model, ModelConfig

    from omlx import scheduler
    from omlx.models.vlm import VLMModelAdapter
    from omlx.patches.mlx_lm_mtp import batch_generator

    assert batch_generator.apply()
    assert scheduler.BatchGenerator is BatchGenerator
    head = Model(ModelConfig(text_config=tiny_nemotron_config()["llm_config"]))
    adapter = VLMModelAdapter(native_target)
    adapter.enable_nemotron_mtp(head, depth=3)
    verified_batches = []
    verify = adapter.speculative_verify_logits

    def observed_verify(inputs, cache, sampler):
        verified_batches.append(inputs.shape)
        return verify(inputs, cache, sampler)

    monkeypatch.setattr(adapter, "speculative_verify_logits", observed_verify)
    generator = BatchGenerator(
        adapter,
        max_tokens=9,
        sampler=lambda lp: mx.argmax(lp, axis=-1),
        prefill_step_size=2,
        completion_batch_size=2,
    )
    prompts = [[1, 2, 3], [5, 6, 7, 8, 9]]
    stop = int(
        mx.argmax(
            native_target.language_model(mx.array([prompts[0]])).logits[:, -1]
        ).item()
    )
    uids = generator.insert(
        prompts,
        max_tokens=list(limits),
        stop_sequences=(
            [StopSequences([[stop]]), StopSequences()] if stop_first else None
        ),
    )
    prompt_by_uid = dict(zip(uids, prompts))
    limit_by_uid = dict(zip(uids, limits))
    if stop_first:
        limit_by_uid[uids[0]] = 1
    outputs = {uid: [] for uid in uids}
    terminal = {}
    cancelled = set()
    try:
        for _ in range(40):
            responses = generator.next_generated()
            if not responses:
                break
            for response in responses:
                assert response.uid not in cancelled
                outputs[response.uid].append(response.token)
                if response.finish_reason:
                    terminal[response.uid] = response
            if replace and len(outputs[uids[0]]) >= 2:
                generator.remove([uids[0]])
                cancelled.add(uids[0])
                joined_prompt = [11, 12, 13, 14]
                joined_uid = generator.insert([joined_prompt])[0]
                prompt_by_uid[joined_uid] = joined_prompt
                limit_by_uid[joined_uid] = 9
                outputs[joined_uid] = []
                replace = False
    finally:
        generator.close()
    assert set(terminal) == set(prompt_by_uid) - cancelled
    if min(limits) >= 9 and not stop_first:
        assert any(batch == 2 and width == 4 for batch, width in verified_batches)
    for uid in terminal:
        prompt = prompt_by_uid[uid]
        assert len(outputs[uid]) == limit_by_uid[uid]
        baseline = native_target.language_model.make_cache()
        expected = []
        ids = mx.array([prompt])
        for _ in range(limit_by_uid[uid]):
            logits = native_target.language_model(ids, cache=baseline).logits[:, -1]
            token = int(mx.argmax(logits, axis=-1).item())
            expected.append(token)
            ids = mx.array([[token]])
        assert outputs[uid] == expected
        native_target.language_model(ids, cache=baseline)
        probe = mx.array([[21]])
        actual = native_target.language_model(
            probe, cache=terminal[uid].prompt_cache
        ).logits
        expected_logits = native_target.language_model(probe, cache=baseline).logits
        np.testing.assert_allclose(
            np.array(actual), np.array(expected_logits), atol=1e-5, rtol=1e-5
        )


@pytest.mark.parametrize(
    "family", ["NemotronH_Nano_Omni_Reasoning_V3", "nemotron_h_omni", "bolt"]
)
@pytest.mark.parametrize("calibration", ["basic", "resident", "streamed"])
@pytest.mark.parametrize("modelopt", [False, True])
def test_complete_tiny_quantization_strict_native_reload(
    tmp_path, family, calibration, modelopt, monkeypatch
):
    from mlx.utils import tree_flatten
    from mlx_vlm.models.nemotron_h_nano_omni import Model, ModelConfig
    from mlx_vlm.utils import load_model

    from omlx import oq
    from omlx.oq import quantize_oq_streaming

    if calibration != "basic":
        from mlx_lm import tokenizer_utils

        monkeypatch.setattr(tokenizer_utils, "load", lambda *_args: SimpleNamespace())
        monkeypatch.setattr(
            oq,
            "_load_calibration_data",
            lambda *_args, **_kwargs: mx.array(
                [[1 + i % 16 for i in range(64)] for _ in range(2)]
            ),
        )

    config = tiny_nemotron_config()
    if family == "bolt":
        text = config["llm_config"]
        text["model_type"] = "nemotron_h_puzzle"
        text["num_nextn_predict_layers"] = 0
        text["block_configs"] = [
            {"block_type": "mamba"},
            {"block_type": "moe", "n_routed_experts": 3},
            {"block_type": "attention"},
        ]
    else:
        config["model_type"] = family
    config["eos_token_id"] = 127
    config["llm_config"].update(
        hidden_size=128,
        intermediate_size=128,
        head_dim=32,
        moe_latent_size=128,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=128,
    )
    model = Model(ModelConfig.from_dict(config))
    model.eval()
    source = tmp_path / "source"
    source.mkdir()
    weights = dict(tree_flatten(model.parameters()))
    source_references = {}
    if modelopt:
        config["quantization_config"] = {
            "quant_method": "modelopt_mixed",
            "quant_algo": "MIXED_PRECISION",
        }
        prefix = "language_model.backbone.layers.1.mixer"
        experts = weights.pop(prefix + ".switch_mlp.fc1.weight")
        for expert, weight in enumerate(experts):
            packed, scales = mx.quantize(
                weight.astype(mx.bfloat16), group_size=16, bits=4, mode="nvfp4"
            )
            name = f"{prefix}.experts.{expert}.up_proj"
            global_scale = mx.array(1 + expert, dtype=mx.float32)
            weights[name + ".weight"] = packed.view(mx.uint8)
            weights[name + ".weight_scale"] = scales
            weights[name + ".weight_scale_2"] = global_scale
            weights[name + ".input_scale"] = mx.array(1.0)
            source_references[name + ".weight"] = (
                mx.dequantize(
                    packed, scales, group_size=16, bits=4, mode="nvfp4"
                ).astype(mx.float32)
                * global_scale
            ).astype(mx.bfloat16)
        fp8_name = "mlp1.layers.1.weight"
        weights[fp8_name] = mx.full(weights[fp8_name].shape, 0x38, dtype=mx.uint8)
        weights["mlp1.layers.1.weight_scale"] = mx.array(0.01)
        weights["mlp1.layers.1.input_scale"] = mx.array(1.0)
        weights[prefix + ".k_scale"] = mx.array(1.0)
        source_references[fp8_name] = mx.full(
            weights[fp8_name].shape, 0.01, dtype=mx.bfloat16
        )
    (source / "config.json").write_text(json.dumps(config))
    preserve_mtp = family == "nemotron_h_omni"
    if preserve_mtp:
        from mlx_vlm.speculative.drafters.nemotron_h_mtp import (
            Model as Head,
        )
        from mlx_vlm.speculative.drafters.nemotron_h_mtp import (
            ModelConfig as HeadConfig,
        )

        head = Head(HeadConfig(text_config=config["llm_config"]))
        weights.update(
            {"mtp." + key: value for key, value in tree_flatten(head.parameters())}
        )
        if modelopt:
            name = "mtp.layers.1.mixer.switch_mlp.fc1"
            packed, scales = mx.quantize(
                weights[name + ".weight"].astype(mx.bfloat16),
                group_size=16,
                bits=4,
                mode="nvfp4",
            )
            weights[name + ".weight"] = packed.view(mx.uint8)
            weights[name + ".weight_scale"] = scales
            weights[name + ".weight_scale_2"] = mx.array([1.0, 2.0])
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    if modelopt:
        index = oq._LazyTensorIndex([source / "model.safetensors"], config=config)
        for name, expected in source_references.items():
            np.testing.assert_array_equal(
                np.array(index[name].astype(mx.float32)),
                np.array(expected.astype(mx.float32)),
            )
        assert not any(
            name.endswith(
                (".weight_scale", ".weight_scale_2", ".input_scale", ".k_scale")
            )
            for name in index
        )
    output = tmp_path / "output"
    quantize_oq_streaming(
        str(source),
        str(output),
        4,
        sensitivity_map_override=(
            None if calibration == "streamed" else {0: 1, 1: 1, 2: 1}
        ),
        preserve_mtp=preserve_mtp,
        enhanced=calibration != "basic",
        stream_calibration=calibration == "streamed",
        imatrix_num_samples=2,
        imatrix_seq_length=64,
    )
    reloaded = load_model(output, lazy=False)
    output_config = json.loads((output / "config.json").read_text())
    if calibration != "basic":
        report = json.loads((output / "oq_imatrix_report.json").read_text())
        assert not any(
            name.startswith("language_model.backbone.layers.")
            for name in report["missing"]
        ), report["missing"]
        assert not report["mismatched"]
    assert output_config["text_config"]["num_hidden_layers"] == 3
    assert output_config["text_config"]["num_nextn_predict_layers"] == int(preserve_mtp)
    assert any(
        key.endswith(".scales") for key, _ in tree_flatten(reloaded.parameters())
    )
    result = reloaded.language_model(mx.array([[1, 2, 3]])).logits
    assert result.shape == (1, 3, 128)
    assert bool(mx.all(mx.isfinite(result)).item())
    assert reloaded.language_model.backbone.layers[0].mixer.A_log.dtype == mx.float32
    embeddings = reloaded.get_input_embeddings(
        mx.array([[1, 18, 3]]), pixel_values=mx.ones((1, 3, 32, 32))
    ).inputs_embeds
    assert embeddings.shape == (1, 3, 128)
    assert bool(mx.all(mx.isfinite(embeddings)).item())
    if preserve_mtp:
        without_head = oq.estimate_bpw_and_size(str(source), 4)
        with_head = oq.estimate_bpw_and_size(str(source), 4, preserve_mtp=True)
        assert with_head["output_size_bytes"] > without_head["output_size_bytes"]
    if preserve_mtp:
        from omlx.models.vlm import VLMModelAdapter
        from omlx.patches.nemotron_omni_quantization import native_mtp_sidecar
        from omlx.utils.model_loading import _checkpoint_has_mtp_weights

        assert _checkpoint_has_mtp_weights(output)
        sidecar = native_mtp_sidecar(output, output_config)
        loaded_head = load_model(sidecar, lazy=False)
        adapter = VLMModelAdapter(reloaded)
        adapter.enable_nemotron_mtp(loaded_head, depth=3)
        prompt = mx.array([[1, 2, 3]])
        hidden = reloaded.language_model(prompt, return_hidden=True).hidden_states[-1]
        cache, offset = adapter.prime_native_mtp(prompt, hidden, mx.array([4]))
        assert offset == 3 and cache[0].offset == 3
        tampered = {**output_config, "vision_final_layernorm": False}
        with pytest.raises(ValueError, match="match its parent"):
            native_mtp_sidecar(output, tampered)


@pytest.mark.parametrize(
    "family",
    ["nemotron_h_omni", "nemotron_h_nano_omni", "NemotronH_Nano_Omni_Reasoning_V3"],
)
def test_native_streamed_calibration_selected_only_when_needed(family):
    from omlx.oq import _resolve_stream_calibration

    assert _resolve_stream_calibration(None, model_exceeds_ram=True, model_type=family)
    assert not _resolve_stream_calibration(
        None, model_exceeds_ram=False, model_type=family
    )
    assert not _resolve_stream_calibration(
        False, model_exceeds_ram=True, model_type=family
    )


def test_streamed_bolt_calibration_matches_resident(tmp_path, monkeypatch):
    from mlx.utils import tree_flatten
    from mlx_vlm.models.nemotron_h_nano_omni import Model, ModelConfig

    from omlx import oq
    from omlx.model_classification import normalize_nemotron_config

    config = tiny_nemotron_config()
    text = config["llm_config"]
    text.update(
        model_type="nemotron_h_puzzle",
        num_nextn_predict_layers=0,
        hidden_size=128,
        intermediate_size=128,
        head_dim=32,
        moe_latent_size=128,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=128,
        block_configs=[
            {"block_type": "mamba"},
            {"block_type": "moe", "n_routed_experts": 3, "num_experts_per_tok": 2},
            {"block_type": "attention"},
        ],
    )
    config = normalize_nemotron_config(config)
    mx.random.seed(42)
    model = Model(ModelConfig.from_dict(config))
    model.eval()
    weights = {
        key: value if key.endswith(".A_log") else value.astype(mx.bfloat16)
        for key, value in tree_flatten(model.parameters())
    }
    model.load_weights(list(weights.items()), strict=True)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(config))
    tokens = mx.array([[1 + (i + row) % 32 for i in range(64)] for row in range(2)])
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **kw: tokens)
    resident, _ = oq._collect_imatrix_from_model(
        model,
        object(),
        config,
        calib_dataset=oq._OQE_CALIB_DATASET,
        num_samples=2,
        seq_length=64,
    )

    def reject_full_load(*args, **kwargs):
        raise AssertionError("streamed calibration must not load the complete model")

    monkeypatch.setattr("mlx_vlm.utils.load_model", reject_full_load)
    streamed, metadata = oq._collect_imatrix_streaming(
        tmp_path,
        object(),
        config,
        calib_dataset=oq._OQE_CALIB_DATASET,
        num_samples=2,
        seq_length=64,
        measure_sensitivity=True,
        sensitivity_oq_level=4,
        sensitivity_num_samples=2,
        sensitivity_seq_length=64,
    )
    assert streamed.keys() == resident.keys()
    for key, expected in resident.items():
        np.testing.assert_array_equal(streamed[key].counts, expected.counts)
        np.testing.assert_array_equal(streamed[key].in_sum2, expected.in_sum2)
    reference = oq._measure_sensitivity_from_model(
        model, object(), config, 4, num_samples=2, seq_length=64
    )
    assert metadata["sensitivity_map"] == pytest.approx(reference, abs=1e-6)
    assert metadata["requires_expert_counts"]
    plan = oq._oqe_calibration_batch_plan(config, requested_samples=32, seq_length=64)
    assert plan["top_k"] == 2
    assert plan["num_experts"] == 3


def test_nemotron_rejects_text_only_before_creating_output(tmp_path):
    from omlx.oq import quantize_oq_streaming

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(tiny_nemotron_config()))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="multimodal"):
        quantize_oq_streaming(str(source), str(output), 4, text_only=True)
    assert not output.exists()


def test_nemotron_rejects_existing_mlx_quant_before_creating_output(tmp_path):
    from omlx.oq import quantize_oq_streaming

    source = tmp_path / "source"
    source.mkdir()
    config = {**tiny_nemotron_config(), "quantization": {"bits": 4, "group_size": 64}}
    (source / "config.json").write_text(json.dumps(config))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="existing MLX quantization"):
        quantize_oq_streaming(str(source), str(output), 4)
    assert not output.exists()


@pytest.mark.parametrize("orphan", [False, True])
def test_native_modelopt_source_rejects_invalid_packed_storage(tmp_path, orphan):
    from omlx.oq import _LazyTensorIndex

    config = {
        **tiny_nemotron_config(),
        "quantization_config": {
            "quant_method": "modelopt_mixed",
            "quant_algo": "MIXED_PRECISION",
        },
    }
    weights = {"language_model.lm_head.weight": mx.ones((128, 16), dtype=mx.uint8)}
    if not orphan:
        weights["language_model.lm_head.weight_scale"] = mx.ones(
            (128, 1), dtype=mx.uint8
        )
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), weights)
    with pytest.raises(ValueError, match="Unpaired packed|Invalid ModelOpt FP8 scale"):
        index = _LazyTensorIndex([path], config=config)
        index["language_model.lm_head.weight"]


@pytest.mark.parametrize(
    "model_type",
    [
        "NemotronH_Nano_Omni_Reasoning_V3",
        "nemotron_h_nano_omni",
        "nemotron_h_omni",
    ],
)
def test_native_mtp_admin_capability_requires_configuration_and_real_heads(
    tmp_path, model_type
):
    from omlx.utils.model_loading import lightning_mtp_supported

    config = {**tiny_nemotron_config(), "model_type": model_type}
    (tmp_path / "config.json").write_text(json.dumps(config))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {"weight_map": {"language_model.lm_head.weight": "model.safetensors"}}
        )
    )
    assert not lightning_mtp_supported(tmp_path)
    index.write_text(
        json.dumps(
            {"weight_map": {"mtp.layers.0.mixer.q_proj.weight": "model.safetensors"}}
        )
    )
    assert lightning_mtp_supported(tmp_path)
    config["llm_config"]["num_nextn_predict_layers"] = 0
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert not lightning_mtp_supported(tmp_path)


@pytest.mark.parametrize("with_audio", [False, True])
def test_engine_preparation_preserves_compacted_video_and_source_usage(
    native_target, monkeypatch, with_audio
):
    from mlx_vlm import utils

    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.patches import mlx_audio_compat
    from omlx.request import Request, SamplingParams
    from omlx.scheduler import Scheduler

    engine = VLMBatchedEngine("unused")
    engine._vlm_model = native_target
    engine._processor = SimpleNamespace(
        audio_sampling_rate=16000,
        apply_chat_template=lambda *_args, **_kwargs: "<video>",
    )
    engine._tokenizer = engine._processor
    native_target.sound_context_token_id = 19
    if with_audio:
        native_target.sound_encoder = object()
        monkeypatch.setattr(
            native_target,
            "_extract_sound_features",
            lambda **_kwargs: mx.ones((1, 1, 32)),
        )
    monkeypatch.setattr(
        mlx_audio_compat, "ensure_mlx_audio_resample_export", lambda: None
    )
    monkeypatch.setattr(utils, "load_audio", lambda waveform, _rate: waveform)
    calls = []

    def prepare(_processor, **kwargs):
        calls.append(kwargs)
        ids = [1, 18, 18, 18] + ([19] if with_audio else []) + [3]
        inputs = {
            "input_ids": mx.array([ids]),
            "attention_mask": mx.ones((1, len(ids)), dtype=mx.int32),
            "pixel_values_videos": mx.ones((6, 3, 32, 32)),
        }
        if with_audio:
            inputs["sound_clips"] = kwargs["audio"]
        return inputs

    monkeypatch.setattr(utils, "prepare_inputs", prepare)
    audio = [(np.ones((160, 2), dtype=np.float32), 16000)] if with_audio else None
    ids, embeddings, extra, media_hash, _, ranges = engine._prepare_vision_inputs(
        [{"role": "user", "content": "Describe this."}],
        [],
        audio=audio,
        videos=["bounded-test-video"],
    )
    assert len(ids) == embeddings.shape[1] == 3 + int(with_audio)
    assert extra["_omlx_source_prompt_tokens"] == 5 + int(with_audio)
    assert extra["_omlx_pruned_video"]
    assert calls[0]["max_frames"] == 64
    if with_audio:
        assert calls[0]["audio"][0].shape == (160,)
        assert ranges == [(0, media_hash)]
        changed_audio = [(np.zeros((160, 2), dtype=np.float32), 16000)]
        changed = engine._prepare_vision_inputs(
            [{"role": "user", "content": "Describe this."}],
            [],
            audio=changed_audio,
            videos=["bounded-test-video"],
        )
        assert changed[3] != media_hash
    counted = engine._prepare_vision_inputs(
        [{"role": "user", "content": "Describe this."}],
        [],
        audio=audio,
        videos=["bounded-test-video"],
        encode_vision=False,
    )
    assert len(counted[0]) == 5 + int(with_audio)
    assert counted[1] is None
    request = Request(
        "video",
        ids,
        SamplingParams(),
        prompt_token_ids=ids,
        num_prompt_tokens=len(ids),
        vlm_extra_kwargs=extra,
    )
    scheduler = SimpleNamespace(
        model=engine._vlm_model,
        _prefix_cache_prepared=set(),
        block_aware_cache=object(),
        _model_has_unreconstructible_cache=lambda: False,
        _log_prefix_divergence=lambda _request: None,
        _try_specprefill_scoring=lambda _request: None,
    )
    Scheduler._prepare_prefix_cache_for_request(scheduler, request)
    assert request.skip_cache_store
    assert request.cached_tokens == 0
    assert request.remaining_tokens == ids
    assert request.num_prompt_tokens == len(ids)
    assert request.source_prompt_tokens == 5 + int(with_audio)
    assert "_omlx_source_prompt_tokens" not in extra


@pytest.mark.parametrize("mode", ["greedy", "sampled", "head_failure"])
def test_native_uniform_commit_uses_common_frontier_atomically(
    native_target, monkeypatch, mode
):
    from mlx_lm.generate import BatchGenerator
    from mlx_vlm.speculative.drafters.nemotron_h_mtp import Model, ModelConfig

    from omlx import scheduler
    from omlx.models.vlm import VLMModelAdapter
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    assert scheduler.BatchGenerator is BatchGenerator
    adapter = VLMModelAdapter(native_target)
    adapter.enable_nemotron_mtp(
        Model(ModelConfig(text_config=tiny_nemotron_config()["llm_config"])), depth=3
    )

    class Sampler:
        temp = 0.7

        def __init__(self, seed):
            self.seed, self.calls = seed, 0

        def __call__(self, logprobs):
            key = mx.random.key(self.seed + self.calls)
            self.calls += 1
            return mx.random.categorical(logprobs / self.temp, key=key)

    generator = BatchGenerator(
        adapter,
        max_tokens=12,
        completion_batch_size=2,
        sampler=lambda lp: mx.argmax(lp, axis=-1),
    )
    uids = generator.insert([[1, 2, 3], [5, 6, 7, 8]])
    try:
        generator.next_generated()
        generator.next_generated()
        batch = generator._generation_batch
        states = batch._omlx_mtp_batch_state.states
        expected = {}
        samplers = [Sampler(91), Sampler(103)]
        if mode == "sampled":
            batch.samplers = samplers
        for index, uid in enumerate(uids):
            state = states[uid]
            cache = deepcopy(batch.extract_cache(index))
            sampler = (
                copy(samplers[index]) if mode == "sampled" else batch.fallback_sampler
            )
            ids = state.next_main.reshape(1, 1)
            tokens = []
            for _ in range(3):
                logits = native_target.language_model(ids, cache=cache).logits[:, -1]
                lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                token = int(sampler(lp).item())
                tokens.append(token)
                ids = mx.array([[token]])
            expected[uid] = tokens[:2]
            if index == 1:
                tokens[1] = (tokens[1] + 1) % 128
            state.drafts = mx.array(tokens, dtype=mx.uint32)
        rollback = adapter.rollback_speculative_cache
        accepted = []

        def observed_rollback(cache, undo, count, width):
            accepted.append((count, width))
            return rollback(cache, undo, count, width)

        monkeypatch.setattr(adapter, "rollback_speculative_cache", observed_rollback)
        if mode == "head_failure":
            chain = bg._chain_next_drafts
            calls = []
            before = {uid: deepcopy(state.stats) for uid, state in states.items()}

            def fail_second(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("injected native head failure")
                return chain(*args, **kwargs)

            monkeypatch.setattr(bg, "_chain_next_drafts", fail_second)
            with pytest.raises(RuntimeError, match="injected native head failure"):
                generator.next_generated()
            assert all(
                not state.queue and state.stats == before[uid]
                for uid, state in states.items()
            )
        else:
            responses = generator.next_generated()
            for response in responses:
                assert response.token == expected[response.uid].pop(0)
            assert all(state.stats.accepts == 1 for state in states.values())
            if mode == "sampled":
                assert [sampler.calls for sampler in samplers] == [2, 2]
        assert accepted == [(1, 4)]
    finally:
        generator.close()


def test_native_processor_pairs_odd_video_frames_without_model_download(native_target):
    from mlx_vlm.models.nemotron_h_nano_omni.image_processing_nemotron_h_nano_omni import (
        NemotronHNanoOmniImageProcessor,
    )
    from mlx_vlm.models.nemotron_h_nano_omni.processing_nemotron_h_nano_omni import (
        NemotronHNanoOmniProcessor,
    )

    class Tokenizer:
        image_token_id, video_token_id, audio_token_id = 18, 20, 19

        def __call__(self, texts, **_kwargs):
            ids = [[1] + [18] * text.count("<image>") + [3] for text in texts]
            return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}

    processor = NemotronHNanoOmniProcessor(
        NemotronHNanoOmniImageProcessor(
            norm_mean=[0.5] * 3,
            norm_std=[0.5] * 3,
            min_num_patches=4,
            max_num_patches=4,
            video_target_num_patches=4,
        ),
        Tokenizer(),
    )
    result = processor(
        text=["Describe <video>."], videos=[np.zeros((3, 3, 32, 32), dtype=np.uint8)]
    )
    assert int(mx.sum(result["input_ids"] == 18).item()) == 2
    assert result["pixel_values_videos"].shape == (3, 3, 32, 32)
    prepared = prepare_nemotron_prompt(
        native_target,
        result["input_ids"],
        pixel_values_videos=result["pixel_values_videos"],
    )
    assert prepared.token_ids.shape[1] == prepared.inputs_embeds.shape[1] == 3


@pytest.mark.parametrize("video", [False, True])
def test_media_conditioned_mtp_uses_production_chunked_prompt_capture(
    native_target, video
):
    from mlx_lm.generate import BatchGenerator
    from mlx_vlm.speculative.drafters.nemotron_h_mtp import Model, ModelConfig

    from omlx import scheduler
    from omlx.models.vlm import VLMModelAdapter

    assert scheduler.BatchGenerator is BatchGenerator
    ids = mx.array([[1, 18, 18, 18, 3]]) if video else mx.array([[1, 18, 3]])
    media = (
        {"pixel_values_videos": mx.ones((6, 3, 32, 32))}
        if video
        else {"pixel_values": mx.ones((1, 3, 32, 32))}
    )
    prepared = prepare_nemotron_prompt(native_target, ids, **media)
    adapter = VLMModelAdapter(native_target)
    adapter.enable_nemotron_mtp(
        Model(ModelConfig(text_config=tiny_nemotron_config()["llm_config"])), depth=3
    )
    adapter.set_pending_embeddings(prepared.inputs_embeds)
    generator = BatchGenerator(
        adapter,
        max_tokens=7,
        prefill_step_size=1,
        sampler=lambda lp: mx.argmax(lp, axis=-1),
    )
    generator.insert([prepared.token_ids[0].tolist()])
    actual = []
    terminal = None
    try:
        for _ in range(20):
            responses = generator.next_generated()
            if not responses:
                break
            for response in responses:
                actual.append(response.token)
                if response.finish_reason:
                    terminal = response.prompt_cache
    finally:
        generator.close()
    cache = native_target.language_model.make_cache()
    logits = native_target.language_model(
        prepared.token_ids, inputs_embeds=prepared.inputs_embeds, cache=cache
    ).logits[:, -1]
    expected = []
    for _ in range(7):
        token = int(mx.argmax(logits, axis=-1).item())
        expected.append(token)
        logits = native_target.language_model(mx.array([[token]]), cache=cache).logits[
            :, -1
        ]
    assert actual == expected
    probe = mx.array([[12]])
    np.testing.assert_allclose(
        np.array(native_target.language_model(probe, cache=terminal).logits),
        np.array(native_target.language_model(probe, cache=cache).logits),
        atol=1e-5,
        rtol=1e-5,
    )
    assert not adapter.has_pending_embeddings


@pytest.mark.parametrize("rate", [0, 0.7, 1])
@pytest.mark.parametrize("width", [32, 64])
def test_video_pruning_matches_native_and_preserves_alignment(
    native_target, rate, width
):
    spatial = width // 32
    ids = mx.array([[1] + [18] * (3 * spatial) + [3]])
    mask = mx.ones_like(ids)
    pixels = mx.random.normal((6, 3, 32, width))
    kwargs = {"pixel_values_videos": pixels, "video_pruning_rate": rate}
    expected = native_target.get_input_embeddings(ids, **kwargs).inputs_embeds
    result = prepare_nemotron_prompt(native_target, ids, mask=mask, **kwargs)
    np.testing.assert_array_equal(np.array(result.inputs_embeds), np.array(expected))
    np.testing.assert_array_equal(
        np.array(result.token_ids), np.array(ids[:, result.retained_positions])
    )
    assert result.token_ids.shape == result.attention_mask.shape
    assert result.token_ids.shape[1] == result.inputs_embeds.shape[1]
    assert result.token_ids[0, 0].item() == 1
    assert result.token_ids[0, -1].item() == 3
    assert native_target.video_pruning_rate == 0.7


def test_video_preparation_extracts_features_once(native_target, monkeypatch):
    extract = native_target.extract_video_feature
    calls = []

    def counted(pixels):
        calls.append(pixels)
        return extract(pixels)

    monkeypatch.setattr(native_target, "extract_video_feature", counted)
    prepare_nemotron_prompt(
        native_target,
        mx.array([[1, 18, 18, 18, 3]]),
        pixel_values_videos=mx.zeros((6, 3, 32, 32)),
    )
    assert len(calls) == 1


@pytest.mark.parametrize("rate", [-0.1, 1.1, float("nan")])
def test_video_preparation_rejects_invalid_pruning(native_target, rate):
    with pytest.raises(ValueError, match="between 0 and 1"):
        prepare_nemotron_prompt(
            native_target,
            mx.array([[1, 18, 3]]),
            pixel_values_videos=mx.zeros((2, 3, 32, 32)),
            video_pruning_rate=rate,
        )


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_native_batched_uniform_rollback_matches_independent_prefixes(
    native_target, accepted
):
    model = native_target.language_model
    prompts = [mx.array([[1, 2, 3]]), mx.array([[5, 6, 7, 8, 9]])]
    ordinary = [model.make_cache() for _ in prompts]
    rows = [model.make_cache() for _ in prompts]
    for prompt, baseline, candidate in zip(prompts, ordinary, rows):
        model(prompt, cache=baseline)
        model(prompt, cache=candidate)
    caches = [
        None if first is None else first.merge([first, second])
        for first, second in zip(*rows)
    ]
    block = mx.array([[4, 5, 6, 7], [10, 11, 12, 13]])
    hidden, _, states = model.speculative_verify_hidden(block, caches)
    mx.eval(hidden)
    assert hidden.shape == (2, 4, 32)
    model.rollback_speculative_cache(caches, states, accepted, 4)
    for row, baseline in enumerate(ordinary):
        for position in range(accepted + 1):
            model(block[row : row + 1, position : position + 1], cache=baseline)
        extracted = [None if c is None else c.extract(row) for c in caches]
        for actual_cache, expected_cache in zip(extracted, baseline):
            for actual_state, expected_state in zip(
                actual_cache.state, expected_cache.state
            ):
                np.testing.assert_allclose(
                    np.array(actual_state),
                    np.array(expected_state),
                    atol=1e-5,
                    rtol=1e-5,
                )
        probe = mx.array([[20 + row]])
        expected = model(probe, cache=baseline).logits
        actual = model(probe, cache=extracted).logits
        np.testing.assert_allclose(
            np.array(actual), np.array(expected), atol=1e-5, rtol=1e-5
        )


def test_native_rollback_rejects_ragged_acceptance(native_target):
    model = native_target.language_model
    with pytest.raises(ValueError, match="uniform acceptance"):
        model.rollback_speculative_cache([], [], [3, 1], 4)


@pytest.mark.parametrize("mismatch", [0, 1, 3])
def test_uniform_walk_stops_all_rows_at_first_mismatch(mismatch):
    drafts = [[10, 11, 12], [20, 21, 22]]
    calls = []

    def sample(position):
        calls.append(position)
        if position == 3:
            return [13, 23]
        return [
            drafts[0][position],
            99 if position == mismatch else drafts[1][position],
        ]

    result = sample_uniform_prefix(drafts, sample, lambda *_: False)
    assert result.accepted_drafts == mismatch
    assert calls == list(range(mismatch + 1))
    assert result.successors[0] == 10 + mismatch
    assert result.successors[1] == (23 if mismatch == 3 else 99)


def test_uniform_walk_stop_boundary_does_not_sample_future_positions():
    calls = []

    def sample(position):
        calls.append(position)
        return [10 + position, 20 + position]

    result = sample_uniform_prefix(
        [[10, 11, 12], [20, 21, 22]], sample, lambda position, _: position == 1
    )
    assert result.accepted_drafts == 1
    assert result.successors == (11, 21)
    assert calls == [0, 1]


def test_native_head_unequal_prompt_caches_merge_and_continue(native_target):
    from mlx_vlm.speculative.drafters.nemotron_h_mtp import Model, ModelConfig

    prototype = Model(ModelConfig(text_config=tiny_nemotron_config()["llm_config"]))
    prompts = [mx.array([[1, 2, 3]]), mx.array([[5, 6, 7, 8, 9]])]
    heads = [deepcopy(prototype) for _ in prompts]
    hidden_rows = []
    for head, ids in zip(heads, prompts):
        hidden = native_target.language_model(ids, return_hidden=True).hidden_states[-1]
        head.reset(native_target)
        head.prefill_from_target_hidden(ids, hidden, 7, sampler=None, greedy=True)
        hidden_rows.append(hidden[:, -1:])
    merged = [
        left.merge([left, right])
        for left, right in zip(heads[0]._cache, heads[1]._cache)
    ]
    tokens = mx.array([[10], [11]])
    hidden = mx.concatenate(hidden_rows, axis=0)
    actual, _ = prototype._forward_hidden(
        native_target.language_model.get_input_embeddings(tokens),
        hidden,
        tokens,
        merged,
    )
    expected = [
        head._forward_tokens(tokens[i : i + 1], hidden[i : i + 1], mx.int32)[0]
        for i, head in enumerate(heads)
    ]
    np.testing.assert_allclose(
        np.array(actual), np.array(mx.concatenate(expected)), atol=1e-5, rtol=1e-5
    )
    for index, head in enumerate(heads):
        for group_cache, row_cache in zip(merged, head._cache):
            for got, want in zip(group_cache.extract(index).state, row_cache.state):
                np.testing.assert_allclose(
                    np.array(got), np.array(want), atol=1e-5, rtol=1e-5
                )


def test_pruned_media_full_and_chunked_head_priming_match(native_target):
    from mlx_vlm.speculative.drafters.nemotron_h_mtp import Model, ModelConfig

    ids = mx.array([[1, 18, 18, 18, 3]])
    prompt = prepare_nemotron_prompt(
        native_target, ids, pixel_values_videos=mx.random.normal((6, 3, 32, 32))
    )
    language = native_target.language_model
    full = language(
        prompt.token_ids, inputs_embeds=prompt.inputs_embeds, return_hidden=True
    ).hidden_states[-1]
    cache = language.make_cache()
    chunks = []
    for start in range(0, prompt.token_ids.shape[1], 2):
        output = language(
            prompt.token_ids[:, start : start + 2],
            inputs_embeds=prompt.inputs_embeds[:, start : start + 2],
            cache=cache,
            return_hidden=True,
        )
        chunks.append(output.hidden_states[-1])
    chunked = mx.concatenate(chunks, axis=1)
    head = Model(ModelConfig(text_config=tiny_nemotron_config()["llm_config"]))
    heads = [head, deepcopy(head)]
    for draft, hidden in zip(heads, [full, chunked]):
        draft.reset(native_target)
        draft.prefill_from_target_hidden(
            prompt.token_ids, hidden, 7, sampler=None, greedy=True
        )
    np.testing.assert_allclose(
        np.array(heads[0]._seed_hidden),
        np.array(heads[1]._seed_hidden),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_native_full_prompt_mtp_cannot_memory_requeue(enabled, phase):
    from omlx.request import Request, SamplingParams
    from omlx.scheduler import Scheduler

    request = Request("native", [1, 2], SamplingParams())
    scheduler = SimpleNamespace(
        model=SimpleNamespace(_omlx_native_mtp_full_prompt=enabled),
        _MAX_MEMORY_REQUEUES=2,
        _specprefill_active_request_id=None,
        _prefill_states={"native": object()} if phase == "prefill" else {},
        request_id_to_uid={"native": 1},
    )
    assert Scheduler._memory_requeue_blocker(scheduler, request) == (
        "native full-prompt MTP" if enabled else None
    )


@pytest.mark.parametrize(
    "field",
    [
        "image_token_id",
        "video_token_id",
        "audio_token_id",
        "img_context_token_id",
        "video_context_token_id",
        "sound_context_token_id",
    ],
)
def test_memory_requeue_recognizes_native_media_boundaries(field):
    from omlx.scheduler import Scheduler

    scheduler = SimpleNamespace(
        model=SimpleNamespace(
            _vlm_model=SimpleNamespace(config=SimpleNamespace(**{field: 18}))
        )
    )
    assert Scheduler._media_token_end(scheduler, [1, 18, 18, 3]) == 3
    assert Scheduler._media_token_end(scheduler, [1, 3]) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("media_type", ["image_url", "input_audio", "video_url"])
async def test_native_preflight_counts_processor_expanded_media_tokens(media_type):
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import AsyncMock, Mock

    from omlx.engine.vlm import VLMBatchedEngine

    engine = VLMBatchedEngine("unused")
    engine._loaded = True
    engine._vlm_model = SimpleNamespace(
        config=SimpleNamespace(model_type="nemotron_h_omni")
    )
    engine._processor = SimpleNamespace()
    engine._tokenizer = SimpleNamespace(encode=lambda _text: [1])
    engine._apply_chat_template = lambda *_args, **_kwargs: "text only"
    engine._extract_request_media = lambda messages: ([], messages, [], [], [])
    engine._process_chat_messages = Mock(
        return_value=(list(range(501)), None, {}, None, 0, [])
    )
    engine._preflight_or_raise_with_eviction = AsyncMock()
    messages = [{"role": "user", "content": [{"type": media_type}]}]
    with ThreadPoolExecutor(max_workers=1) as executor:
        engine._engine = SimpleNamespace(
            _mlx_executor=executor, engine=SimpleNamespace(scheduler=object())
        )
        await engine.preflight_chat(messages, request_id="media")
    assert (
        engine._preflight_or_raise_with_eviction.call_args.kwargs["num_prompt_tokens"]
        == 501
    )
    engine._process_chat_messages.assert_called_once()
    assert engine._process_chat_messages.call_args.args[2]["_encode_vision"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_type",
    ["nemotron_h_omni", "nemotron_h_nano_omni", "NemotronH_Nano_Omni_Reasoning_V3"],
)
async def test_native_tokenization_does_not_encode_or_prune_media(model_type):
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import Mock

    from omlx.engine.vlm import VLMBatchedEngine

    engine = VLMBatchedEngine("unused")
    engine._loaded = True
    engine._vlm_model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    engine._process_chat_messages = Mock(
        return_value=([1, 18, 18, 18, 3], None, {}, None, 0, [])
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        engine._engine = SimpleNamespace(_mlx_executor=executor)
        tokens = await engine.tokenize_chat([{"role": "user", "content": "media"}])
    assert tokens == [1, 18, 18, 18, 3]
    assert engine._process_chat_messages.call_args.args[2]["_encode_vision"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("sidecar", [False, True])
async def test_native_head_load_runs_on_mlx_executor(
    native_target, tmp_path, monkeypatch, sidecar
):
    import threading

    from mlx_vlm import utils
    from mlx_vlm.speculative.drafters.nemotron_h_mtp.split import (
        NemotronHMTPSplitter,
    )

    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.model_settings import ModelSettings
    from omlx.models.vlm import VLMModelAdapter
    from omlx.patches import nemotron_omni_quantization
    from omlx.utils import model_loading

    class HeadEnabledError(Exception):
        pass

    (tmp_path / "config.json").write_text(json.dumps(tiny_nemotron_config()))
    mx.eval(native_target.parameters())
    loop_thread = threading.get_ident()
    head = object()
    loads = []
    expected_path = tmp_path / "head" if sidecar else tmp_path

    def load_head(path):
        assert threading.get_ident() != loop_thread
        assert path == expected_path
        loads.append(path)
        return head

    def enable_head(self, loaded, depth):
        assert loaded is head and depth > 0
        raise HeadEnabledError

    processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda _text: [1]))
    monkeypatch.setattr(
        model_loading, "maybe_apply_pre_load_patches", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(
        model_loading,
        "maybe_load_custom_quantization",
        lambda *_a, **_kw: (native_target, processor),
    )
    monkeypatch.setattr(model_loading, "_checkpoint_has_mtp_weights", lambda _p: True)
    monkeypatch.setattr(
        nemotron_omni_quantization,
        "native_mtp_sidecar",
        lambda *_a: expected_path if sidecar else None,
    )
    monkeypatch.setattr(utils, "load_model", load_head)
    monkeypatch.setattr(NemotronHMTPSplitter, "load", lambda _self, path: load_head(path))
    monkeypatch.setattr(VLMModelAdapter, "enable_nemotron_mtp", enable_head)
    engine = VLMBatchedEngine(
        str(tmp_path),
        model_settings=ModelSettings(mtp_enabled=True),
    )
    with pytest.raises(HeadEnabledError):
        await engine.start()
    assert loads == [expected_path]
