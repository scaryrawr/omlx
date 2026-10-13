import json
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from types import MethodType, SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.models.minicpmv4_7.canvas import canvas_positions
from mlx_vlm.models.minicpmv4_7.config import ModelConfig
from mlx_vlm.models.minicpmv4_7.language import LanguageModel

pinned_hf = pytest.mark.skipif(
    version("transformers") != "5.18.0",
    reason="Numerical oracle requires pinned Transformers 5.18.0",
)


def tiny_config():
    return {
        "model_type": "minicpmv4_7",
        "architectures": ["MiniCPMV4_7ForConditionalGeneration"],
        "insert_layer_id": 1,
        "downsample_mode": "16x",
        "mrope_mode": "canvas",
        "image_token_id": 8,
        "video_token_id": 9,
        "image_start_id": 10,
        "image_end_id": 11,
        "slice_start_id": 12,
        "slice_end_id": 13,
        "newline_id": 2,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "shared_expert_intermediate_size": 8,
            "moe_intermediate_size": 8,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "full_attention_interval": 2,
            "layer_types": ["linear_attention", "full_attention"] * 2,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [2, 1, 1],
                "partial_rotary_factor": 1.0,
                "rope_theta": 10000000,
                "mrope_interleaved": True,
            },
            "mtp_num_hidden_layers": 1,
            "tie_word_embeddings": False,
        },
        "vision_config": {
            "model_type": "minicpmv4_7_vision",
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_channels": 3,
            "image_size": 8,
            "patch_size": 2,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
        },
    }


def hf_weight_fixture():
    """Construct HF tensors from its declared module shapes, not the sanitizer."""
    shapes = {
        "model.language_model.embed_tokens.weight": (32, 16),
        "model.language_model.norm.weight": (16,),
        "lm_head.weight": (32, 16),
        "model.vision_tower.embeddings.patch_embedding.weight": (8, 3, 2, 2),
        "model.vision_tower.embeddings.patch_embedding.bias": (8,),
        "model.vision_tower.embeddings.position_embedding.weight": (16, 8),
        "model.vision_tower.post_layernorm.weight": (8,),
        "model.vision_tower.post_layernorm.bias": (8,),
    }

    def linear(prefix, out_dim, in_dim, bias=False):
        shapes[f"{prefix}.weight"] = (out_dim, in_dim)
        if bias:
            shapes[f"{prefix}.bias"] = (out_dim,)

    def norm(prefix, dim, bias=False):
        shapes[f"{prefix}.weight"] = (dim,)
        if bias:
            shapes[f"{prefix}.bias"] = (dim,)

    for layer in range(4):
        prefix = f"model.language_model.layers.{layer}"
        norm(f"{prefix}.input_layernorm", 16)
        norm(f"{prefix}.post_attention_layernorm", 16)
        if layer % 2 == 0:
            attn = f"{prefix}.linear_attn"
            shapes[f"{attn}.conv1d.weight"] = (16, 1, 4)
            for name, out_dim in (
                ("in_proj_qkv", 16),
                ("in_proj_z", 8),
                ("in_proj_a", 2),
                ("in_proj_b", 2),
            ):
                linear(f"{attn}.{name}", out_dim, 16)
            linear(f"{attn}.out_proj", 16, 8)
            shapes[f"{attn}.A_log"] = (2,)
            shapes[f"{attn}.dt_bias"] = (2,)
            norm(f"{attn}.norm", 4)
        else:
            attn = f"{prefix}.self_attn"
            for name, out_dim in (("q_proj", 32), ("k_proj", 8), ("v_proj", 8)):
                linear(f"{attn}.{name}", out_dim, 16)
            linear(f"{attn}.o_proj", 16, 16)
            norm(f"{attn}.q_norm", 8)
            norm(f"{attn}.k_norm", 8)
        mlp = f"{prefix}.mlp"
        shapes[f"{mlp}.experts.gate_up_proj"] = (2, 16, 16)
        shapes[f"{mlp}.experts.down_proj"] = (2, 16, 8)
        linear(f"{mlp}.gate", 2, 16)
        linear(f"{mlp}.shared_expert_gate", 1, 16)
        for name in ("gate_proj", "up_proj"):
            linear(f"{mlp}.shared_expert.{name}", 8, 16)
        linear(f"{mlp}.shared_expert.down_proj", 16, 8)
    for layer in range(3):
        prefix = f"model.vision_tower.encoder.layers.{layer}"
        for name in ("layer_norm1", "layer_norm2"):
            norm(f"{prefix}.{name}", 8, True)
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            linear(f"{prefix}.self_attn.{name}", 8, 8, True)
        linear(f"{prefix}.mlp.fc1", 12, 8, True)
        linear(f"{prefix}.mlp.fc2", 8, 12, True)
    prefix = "model.vision_tower.vit_merger"
    norm(f"{prefix}.layer_norm1", 8, True)
    norm(f"{prefix}.pre_norm", 32, True)
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        linear(f"{prefix}.self_attn.{name}", 8, 8, True)
    linear(f"{prefix}.linear_1", 48, 32, True)
    linear(f"{prefix}.linear_2", 8, 48, True)
    prefix = "model.merger.mlp.0"
    norm(f"{prefix}.pre_norm", 32, True)
    linear(f"{prefix}.linear_1", 32, 32, True)
    linear(f"{prefix}.linear_2", 16, 32, True)
    result = {}
    for index, (name, shape) in enumerate(sorted(shapes.items())):
        values = np.sin(np.arange(np.prod(shape), dtype=np.float32) + index) * 0.03
        if (
            "layer_norm" in name
            or "pre_norm.weight" in name
            or "post_layernorm.weight" in name
        ):
            if name.endswith(".weight"):
                values += 1.0
        if name.endswith("linear_attn.norm.weight"):
            values += 1.0
        result[name] = mx.array(values.reshape(shape))
    return result


def test_nested_config_is_moe_and_does_not_mutate_source():
    raw = tiny_config()
    before = deepcopy(raw)
    config = ModelConfig.from_dict(raw)
    assert raw == before
    assert config.text_config.moe_intermediate_size == 8
    assert config.vision_config.insert_layer_id == 1
    assert config.text_config.rope_parameters["type"] == "default"
    assert not hasattr(config.text_config, "mtp_num_hidden_layers")
    with pytest.raises(ValueError, match="hybrid"):
        raw["text_config"]["layer_types"][0] = "full_attention"
        ModelConfig.from_dict(raw)


def test_canvas_thumbnail_slice_markers_and_padding():
    config = ModelConfig.from_dict(tiny_config())
    ids = mx.array([[0, 3, 10, 8, 11, 12, 8, 13, 4]])
    mm = mx.array([[0, 0, 0, 1, 0, 0, 1, 0, 0]])
    positions, delta = canvas_positions(
        ids,
        config,
        mm_token_type_ids=mm,
        target_sizes=mx.array([[4, 4], [4, 4]]),
        attention_mask=mx.array([[0, 1, 1, 1, 1, 1, 1, 1, 1]]),
    )
    expected = np.array(
        [
            [0, 0, 1, 1, 1, 1, 1, 1, 3],
            [0, 0, 0, 1, 2, 1, 1, 1, 3],
            [0, 0, 0, 1, 2, 1, 1, 1, 3],
        ]
    )
    np.testing.assert_array_equal(np.array(positions[:, 0]), expected)
    np.testing.assert_array_equal(np.array(delta), [[-4]])
    with pytest.raises(ValueError, match="placeholder count"):
        canvas_positions(
            ids, config, mm_token_type_ids=mm, target_sizes=mx.array([[8, 8], [4, 4]])
        )


def test_canvas_text_and_mismatched_visual_inputs():
    config = ModelConfig.from_dict(tiny_config())
    positions, delta = canvas_positions(mx.array([[3, 4, 5]]), config)
    np.testing.assert_array_equal(np.array(positions), np.tile(np.arange(3), (3, 1, 1)))
    np.testing.assert_array_equal(np.array(delta), [[0]])
    with pytest.raises(ValueError, match="mm_token_type_ids"):
        canvas_positions(mx.array([[10, 8, 11]]), config)
    with pytest.raises(ValueError, match="without matching"):
        canvas_positions(
            mx.array([[3, 4]]),
            config,
            mm_token_type_ids=mx.zeros((1, 2)),
            target_sizes=mx.array([[4, 4]]),
        )


def test_language_request_positions_are_not_model_state():
    language = LanguageModel(ModelConfig.from_dict(tiny_config()).text_config)
    assert "_position_ids" not in language and "_rope_deltas" not in language
    a = mx.array([[3, 4, 5]])
    cache_a, cache_b = language.make_cache(), language.make_cache()
    first = language(a, cache=cache_a, rope_deltas=mx.array([[0]]))
    language(a, cache=cache_b, rope_deltas=mx.array([[17]]))
    out = language(mx.array([[6]]), cache=cache_a, rope_deltas=mx.array([[0]]))
    reference_cache = language.make_cache()
    language(a, cache=reference_cache, rope_deltas=mx.array([[0]]))
    reference = language(
        mx.array([[6]]), cache=reference_cache, rope_deltas=mx.array([[0]])
    )
    np.testing.assert_allclose(
        np.array(out.logits), np.array(reference.logits), rtol=1e-5, atol=1e-5
    )
    assert first.logits.shape == (1, 3, 32)
    assert "_position_ids" not in language and "_rope_deltas" not in language


def test_complete_original_and_canonical_checkpoint_loading():
    from mlx_vlm.models.minicpmv4_7.minicpmv4_7 import Model

    model = Model(ModelConfig.from_dict(tiny_config()))
    original = hf_weight_fixture()
    canonical = model.sanitize(original)
    model.load_weights(list(canonical.items()), strict=True)
    assert set(canonical) == set(dict(tree_flatten(model.parameters())))
    twice = model.sanitize(canonical)
    for name in canonical:
        np.testing.assert_array_equal(np.array(canonical[name]), np.array(twice[name]))
    fresh = Model(ModelConfig.from_dict(tiny_config()))
    fresh.load_weights(list(twice.items()), strict=True)
    assert not any("mtp" in name for name in canonical)
    for key in (
        "model.vision_tower.embeddings.patch_embedding.weight",
        "model.language_model.layers.0.linear_attn.conv1d.weight",
    ):
        assert (
            np.array(original[key]).shape
            != np.array(
                canonical[
                    key.replace("model.vision_tower.", "vision_tower.").replace(
                        "model.language_model.", "language_model.model."
                    )
                ]
            ).shape
        )
    with pytest.raises(ValueError, match="Duplicate"):
        model.sanitize(
            {**original, "language_model.model.norm.weight": mx.zeros((16,))}
        )
    missing = dict(canonical)
    missing.pop("merger.mlp.0.linear_2.weight")
    with pytest.raises(ValueError):
        fresh.load_weights(list(missing.items()), strict=True)


@pytest.mark.parametrize("mode", ["4x", "16x"])
@pinned_hf
def test_pinned_hf_tiny_checkpoint_numerical_parity(mode):
    torch = pytest.importorskip("torch")
    from transformers.models.minicpmv4_7.configuration_minicpmv4_7 import (
        MiniCPMV4_7Config,
    )
    from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import (
        MiniCPMV4_7ForConditionalGeneration,
    )

    from mlx_vlm.models.minicpmv4_7.minicpmv4_7 import Model

    torch.set_num_threads(1)
    raw = tiny_config()
    raw["downsample_mode"] = mode
    config = MiniCPMV4_7Config(**deepcopy(raw))
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    reference = MiniCPMV4_7ForConditionalGeneration(config).cpu().eval()
    weights = hf_weight_fixture()
    reference.load_state_dict(
        {name: torch.from_numpy(np.array(value)) for name, value in weights.items()},
        strict=True,
    )
    native = Model(ModelConfig.from_dict(raw))
    native.load_weights(list(native.sanitize(weights).items()), strict=True)
    count = 4 if mode == "4x" else 1
    ids = np.array([[3, 10] + [8] * count + [11, 4]], dtype=np.int64)
    modalities = np.array([[0, 0] + [1] * count + [0, 0]], dtype=np.int64)
    pixels = np.cos(np.arange(192, dtype=np.float32)).reshape(1, 3, 2, 32)
    grids = np.array([[4, 4]], dtype=np.int32)
    with torch.no_grad():
        expected = reference(
            input_ids=torch.from_numpy(ids),
            pixel_values=torch.from_numpy(pixels),
            target_sizes=torch.from_numpy(grids),
            mm_token_type_ids=torch.from_numpy(modalities),
            use_cache=False,
        ).logits.numpy()
        spatial, delta = reference.model.get_rope_index(
            torch.from_numpy(ids),
            mm_token_type_ids=torch.from_numpy(modalities),
            target_sizes=torch.from_numpy(grids),
        )
    prepared = native.get_input_embeddings(
        mx.array(ids),
        mx.array(pixels),
        target_sizes=mx.array(grids),
        mm_token_type_ids=mx.array(modalities),
    )
    np.testing.assert_array_equal(np.array(prepared.position_ids), spatial.numpy())
    np.testing.assert_array_equal(np.array(prepared.rope_deltas), delta.numpy())
    result = native.language_model(
        mx.array(ids), **{k: v for k, v in prepared.to_dict().items() if v is not None}
    )
    np.testing.assert_allclose(np.array(result.logits), expected, atol=4e-5, rtol=4e-4)


@pytest.mark.parametrize("mode", ["4x", "16x"])
@pytest.mark.parametrize("accepted", [0, 1, 2])
@pytest.mark.parametrize("rotating", [False, True])
def test_hybrid_chunk_decode_captures_and_exact_rollback(mode, accepted, rotating):
    from mlx_vlm.models.minicpmv4_7 import Model
    from mlx_vlm.models.cache import KVCache, RotatingKVCache

    model = Model(ModelConfig.from_dict(tiny_config()))
    model.load_weights(list(model.sanitize(hf_weight_fixture()).items()), strict=True)
    count = 4 if mode == "4x" else 1
    ids = mx.array([[3, 10] + [8] * count + [11, 4]])
    features = model.get_input_embeddings(
        ids,
        mx.arange(192, dtype=mx.float32).reshape(1, 3, 2, 32) / 192,
        target_sizes=mx.array([[4, 4]]),
        downsample_mode=mode,
        mm_token_type_ids=(ids == 8).astype(mx.int32),
    )
    lm = model.language_model
    kwargs = dict(
        position_ids=features.position_ids,
        rope_deltas=features.rope_deltas,
        capture_layer_ids=[3, 0],
        return_hidden=True,
    )
    full = lm(ids, inputs_embeds=features.inputs_embeds, **kwargs)
    chunks, caches = [], lm.make_cache()
    for start, end in ((0, 3), (3, ids.shape[1])):
        chunks.append(
            lm(
                ids[:, start:end],
                inputs_embeds=features.inputs_embeds[:, start:end],
                cache=caches,
                **kwargs,
            )
        )
    np.testing.assert_allclose(
        np.asarray(mx.concatenate([chunk.logits for chunk in chunks], axis=1)),
        np.asarray(full.logits),
        atol=3e-5,
    )
    for index in range(3):
        np.testing.assert_allclose(
            np.asarray(
                mx.concatenate([chunk.hidden_states[index] for chunk in chunks], axis=1)
            ),
            np.asarray(full.hidden_states[index]),
            atol=3e-5,
        )
    assert not np.array_equal(
        np.asarray(full.hidden_states[1]), np.asarray(full.hidden_states[2])
    )

    def prime():
        cache = lm.make_cache()
        if rotating:
            cache = [
                (
                    RotatingKVCache(max_size=4, keep=0)
                    if isinstance(entry, KVCache)
                    else entry
                )
                for entry in cache
            ]
        lm(ids, inputs_embeds=features.inputs_embeds, cache=cache, **kwargs)
        return cache

    verify_cache, replay_cache = prime(), prime()
    block = mx.array([[5, 6, 7]])
    decode_kwargs = dict(
        rope_deltas=features.rope_deltas,
        capture_layer_ids=[3, 0],
        return_hidden=True,
    )
    ordinary = [
        lm(block[:, index : index + 1], cache=replay_cache, **decode_kwargs)
        for index in range(3)
    ]
    verified = lm(
        block,
        cache=verify_cache,
        speculative_verify=True,
        **decode_kwargs,
    )
    np.testing.assert_allclose(
        np.asarray(verified.logits),
        np.asarray(mx.concatenate([output.logits for output in ordinary], axis=1)),
        atol=3e-5,
    )
    for index in range(3):
        np.testing.assert_allclose(
            np.asarray(verified.hidden_states[index]),
            np.asarray(
                mx.concatenate(
                    [output.hidden_states[index] for output in ordinary], axis=1
                )
            ),
            atol=3e-5,
        )
    lm.rollback_speculative_cache(verify_cache, verified.gdn_states, accepted, 3)
    replay_cache = prime()
    for index in range(accepted + 1):
        lm(block[:, index : index + 1], cache=replay_cache, **decode_kwargs)
    probe = mx.array([[6]])
    np.testing.assert_allclose(
        np.asarray(lm(probe, cache=verify_cache, **decode_kwargs).logits),
        np.asarray(lm(probe, cache=replay_cache, **decode_kwargs).logits),
        atol=3e-5,
    )
    assert not hasattr(lm, "_rope_deltas") and not hasattr(lm, "_position_ids")


@pytest.mark.parametrize("side", ["left", "right"])
def test_padded_hybrid_prefill_and_continuation_match_unpadded_rows(side):
    from mlx_vlm.models.minicpmv4_7 import Model

    model = Model(ModelConfig.from_dict(tiny_config()))
    model.load_weights(list(model.sanitize(hf_weight_fixture()).items()), strict=True)
    rows = [[3, 4, 5, 6], [3, 4]]
    if side == "left":
        ids, mask = [[3, 4, 5, 6], [0, 0, 3, 4]], [[1] * 4, [0, 0, 1, 1]]
    else:
        ids, mask = [[3, 4, 5, 6], [3, 4, 0, 0]], [[1] * 4, [1, 1, 0, 0]]
    lm = model.language_model
    caches = lm.make_cache()
    output = lm(
        mx.array(ids),
        attention_mask=mx.array(mask),
        cache=caches,
        rope_deltas=mx.zeros((2, 1), mx.int32),
        capture_layer_ids=[0, 3],
    )
    continuation = lm(
        mx.array([[5], [5]]),
        cache=caches,
        rope_deltas=mx.zeros((2, 1), mx.int32),
    )
    for index, row in enumerate(rows):
        cache = lm.make_cache()
        expected = lm(
            mx.array([row]),
            cache=cache,
            capture_layer_ids=[0, 3],
            rope_deltas=mx.zeros((1, 1), mx.int32),
        )
        valid = np.array(mask[index], dtype=bool)
        np.testing.assert_allclose(
            np.asarray(output.logits)[index, valid],
            np.asarray(expected.logits)[0],
            atol=3e-5,
        )
        for layer in range(2):
            np.testing.assert_allclose(
                np.asarray(output.hidden_states[layer])[index, valid],
                np.asarray(expected.hidden_states[layer])[0],
                atol=3e-5,
            )
        next_output = lm(
            mx.array([[5]]),
            cache=cache,
            rope_deltas=mx.zeros((1, 1), mx.int32),
        )
        np.testing.assert_allclose(
            np.asarray(continuation.logits)[index],
            np.asarray(next_output.logits)[0],
            atol=3e-5,
        )


def test_verify_sampler_exception_aborts_and_cached_decode_requires_delta():
    from mlx_vlm.models.minicpmv4_7 import Model

    model = Model(ModelConfig.from_dict(tiny_config()))
    model.load_weights(list(model.sanitize(hf_weight_fixture()).items()), strict=True)
    lm = model.language_model
    cache, reference = lm.make_cache(), lm.make_cache()
    for current in (cache, reference):
        lm(mx.array([[3, 4]]), cache=current, rope_deltas=mx.array([[0]]))
    with pytest.raises(ValueError, match="request-local"):
        lm(mx.array([[5]]), cache=cache)

    def fail(logits):
        raise RuntimeError("sampling failure")

    with pytest.raises(RuntimeError, match="sampling failure"):
        lm.speculative_verify_logits(
            mx.array([[5, 6, 7]]),
            cache,
            fail,
            rope_deltas=mx.array([[0]]),
        )
    actual = lm(mx.array([[5]]), cache=cache, rope_deltas=mx.array([[0]])).logits
    expected = lm(mx.array([[5]]), cache=reference, rope_deltas=mx.array([[0]])).logits
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=3e-5)


@pytest.mark.parametrize("representation", ["hf", "canonical", "quantized"])
def test_complete_tiny_checkpoint_through_native_loader(tmp_path, representation):
    import mlx.nn as nn

    from mlx_vlm.models.minicpmv4_7 import Model
    from mlx_vlm.utils import load_model

    config = tiny_config()
    if representation == "quantized":
        config["text_config"]["hidden_size"] = 32
        config["quantization"] = {"group_size": 32, "bits": 4, "mode": "affine"}
    original = Model(ModelConfig.from_dict(config))
    if representation == "hf":
        weights = hf_weight_fixture()
        original.load_weights(list(original.sanitize(weights).items()), strict=True)
    else:
        if representation == "quantized":
            nn.quantize(
                original,
                group_size=32,
                bits=4,
                class_predicate=lambda path, module: path
                in (
                    "language_model.lm_head",
                    "language_model.model.embed_tokens",
                ),
            )
        weights = dict(tree_flatten(original.parameters()))
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    loaded = load_model(tmp_path, lazy=False, strict=True)
    assert loaded.config.vision_config.insert_layer_id == config["insert_layer_id"]
    actual_keys = dict(tree_flatten(loaded.parameters()))
    expected_keys = dict(tree_flatten(original.parameters()))
    assert actual_keys.keys() == expected_keys.keys()
    for key in expected_keys:
        np.testing.assert_array_equal(
            np.asarray(actual_keys[key]), np.asarray(expected_keys[key])
        )
    ids = mx.array([[3, 4, 5]])
    np.testing.assert_allclose(
        np.asarray(loaded(ids).logits),
        np.asarray(original(ids).logits),
        atol=1e-6,
    )


@pytest.mark.parametrize("mode", ["4x", "16x"])
@pinned_hf
def test_pinned_hf_canvas_slices_video_groups_and_padding(mode):
    torch = pytest.importorskip("torch")
    from transformers import MiniCPMV4_7Config, MiniCPMV4_7ForConditionalGeneration

    raw = tiny_config()
    raw["downsample_mode"] = mode
    config = ModelConfig.from_dict(raw)
    hf_config = MiniCPMV4_7Config(**deepcopy(raw))
    for component in (hf_config, hf_config.text_config, hf_config.vision_config):
        component._attn_implementation = "eager"
    reference = MiniCPMV4_7ForConditionalGeneration(hf_config).cpu().eval()
    divisor = 4 if mode == "4x" else 16
    crop = [8] * (16 // divisor)
    thumbnail = [10] + [8] * (32 // divisor) + [11]
    detail = [12] + crop + [13]
    image = [10] + crop + [11]
    video = [10] + [9] * (16 // divisor) + [11]
    row = (
        [3]
        + thumbnail
        + detail * 2
        + [2]
        + detail * 2
        + [4]
        + video
        + [2]
        + video
        + [5]
        + image
        + [6]
    )
    short = [3] + image + [4]
    ids = np.array([row, [0] * (len(row) - len(short)) + short], dtype=np.int32)
    mask = ids != 0
    modalities = np.where(ids == 8, 1, np.where(ids == 9, 2, 0)).astype(np.int32)
    images = np.array([[4, 8]] + [[4, 4]] * 6, dtype=np.int32)
    videos = np.array([[4, 4], [4, 4]], dtype=np.int32)
    with torch.no_grad():
        expected, delta = reference.model.get_rope_index(
            torch.from_numpy(ids),
            mm_token_type_ids=torch.from_numpy(modalities),
            target_sizes=torch.from_numpy(images),
            target_sizes_videos=torch.from_numpy(videos),
            attention_mask=torch.from_numpy(mask),
        )
    actual, actual_delta = canvas_positions(
        ids,
        config,
        mm_token_type_ids=modalities,
        target_sizes=images,
        target_sizes_videos=videos,
        attention_mask=mask,
    )
    np.testing.assert_array_equal(np.asarray(actual), expected.numpy())
    np.testing.assert_array_equal(np.asarray(actual_delta), delta.numpy())
    from mlx_vlm.models.minicpmv4_7 import Model

    weights = hf_weight_fixture()
    reference.load_state_dict(
        {name: torch.from_numpy(np.asarray(value)) for name, value in weights.items()},
        strict=True,
    )
    native = Model(config)
    native.load_weights(list(native.sanitize(weights).items()), strict=True)

    def pixels(grids):
        width = int(grids.prod(-1).sum()) * 2
        return np.cos(np.arange(6 * width, dtype=np.float32)).reshape(1, 3, 2, width)

    image_pixels, video_pixels = pixels(images), pixels(videos)
    with torch.no_grad():
        expected_logits = reference(
            input_ids=torch.from_numpy(ids),
            attention_mask=torch.from_numpy(mask),
            mm_token_type_ids=torch.from_numpy(modalities),
            pixel_values=torch.from_numpy(image_pixels),
            target_sizes=torch.from_numpy(images),
            pixel_values_videos=torch.from_numpy(video_pixels),
            target_sizes_videos=torch.from_numpy(videos),
            use_cache=False,
        ).logits.numpy()
    logits = native(
        mx.array(ids),
        pixel_values=mx.array(image_pixels),
        target_sizes=mx.array(images),
        pixel_values_videos=mx.array(video_pixels),
        target_sizes_videos=mx.array(videos),
        attention_mask=mx.array(mask),
        mm_token_type_ids=mx.array(modalities),
    ).logits
    np.testing.assert_allclose(
        np.asarray(logits)[mask],
        expected_logits[mask],
        atol=4e-5,
        rtol=4e-4,
    )


def test_pinned_production_config_and_dspark_pairing_without_weight_allocation():
    from mlx_vlm.speculative.drafters.dspark.config import DSparkConfig
    from mlx_vlm.speculative.drafters.dspark.dspark import validate_dspark_target

    fixtures = Path(__file__).parent / "fixtures"
    raw = json.loads((fixtures / "minicpmv4_7_config.json").read_text())
    original = deepcopy(raw)
    config = ModelConfig.from_dict(raw)
    assert raw == original
    assert config.text_config.hidden_size == 2048
    assert config.text_config.num_hidden_layers == 40
    assert config.text_config.num_experts == 256
    assert config.vision_config.insert_layer_id == 6
    assert "intermediate_size" not in raw["text_config"]
    helper = DSparkConfig.from_dict(
        json.loads((fixtures / "minicpmv4_7_dspark_config.json").read_text())
    )
    assert (helper.proposal_length, helper.block_size, helper.runtime_block_size) == (
        15,
        16,
        8,
    )
    assert helper.target_model_type == "minicpmv4_7"
    assert helper.block_size_policy == "fixed"
    target = SimpleNamespace(
        config=config,
        args=config.text_config,
        model=SimpleNamespace(layers=[None] * 40),
        rollback_speculative_cache=lambda *args: None,
    )
    target.validate_dspark_compatibility = MethodType(
        LanguageModel.validate_dspark_compatibility,
        target,
    )
    validate_dspark_target(helper, target)
    for field, invalid in (
        ("mask_token_id", 248075),
        ("target_layer_ids", [2, 11, 19, 27, 35]),
        ("block_size_policy", "adaptive"),
    ):
        bad = deepcopy(helper)
        setattr(bad, field, invalid)
        with pytest.raises(ValueError):
            validate_dspark_target(bad, target)
    target.config = SimpleNamespace(
        model_type="qwen3_5_moe", text_config=config.text_config
    )
    with pytest.raises(ValueError, match="model_type"):
        validate_dspark_target(helper, target)


def test_visual_features_keep_controls_owner_and_weight_identity():
    import mlx.nn as nn

    from mlx_vlm.models.minicpmv4_7 import Model

    model = Model(ModelConfig.from_dict(tiny_config()))
    model.load_weights(list(model.sanitize(hf_weight_fixture()).items()), strict=True)
    pixels = mx.arange(192, dtype=mx.float32).reshape(1, 3, 2, 32) / 192
    grids = mx.array([[4, 4]])
    ids = mx.array([[3, 10, 8, 8, 8, 8, 11, 4]])
    kwargs = dict(
        target_sizes=grids,
        downsample_mode="4x",
        max_slice_nums=3,
        mm_token_type_ids=(ids == 8).astype(mx.int32),
    )
    cached = model.get_image_features(pixels, grids, "4x", 3)
    fresh = model.get_input_embeddings(ids, pixels, **kwargs)
    reused = model.get_input_embeddings(ids, cached_image_features=cached, **kwargs)
    np.testing.assert_array_equal(
        np.asarray(fresh.inputs_embeds), np.asarray(reused.inputs_embeds)
    )
    with pytest.raises(ValueError, match="do not match"):
        model.get_input_embeddings(
            ids,
            cached_image_features=cached,
            **{**kwargs, "max_slice_nums": 4},
        )
    other = Model(ModelConfig.from_dict(tiny_config()))
    with pytest.raises(ValueError, match="do not match"):
        other.get_input_embeddings(ids, cached_image_features=cached, **kwargs)
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda path, module: path == "merger.mlp.0.linear_1",
    )
    with pytest.raises(ValueError, match="do not match"):
        model.get_input_embeddings(ids, cached_image_features=cached, **kwargs)


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_invalid_attention_interval_fails_at_config_boundary(interval):
    config = tiny_config()
    config["text_config"]["full_attention_interval"] = interval
    with pytest.raises(ValueError, match="positive integer"):
        ModelConfig.from_dict(config)
