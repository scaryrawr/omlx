# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the Ling 3.0 Flash VL mlx-vlm overlay."""

from __future__ import annotations

import json
from dataclasses import asdict
from unittest.mock import patch

import pytest

from omlx.model_discovery import detect_model_type
from omlx.patches import mlx_vlm_bailing_moe_v3_vl_compat as compat

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


@pytest.fixture(scope="module", autouse=True)
def _apply_compat_patch():
    compat.apply_mlx_vlm_bailing_moe_v3_vl_compat_patch()
    assert compat.is_applied()


def _tiny_config(*, layer_plan: tuple[str, ...] = ("kda", "mla")):
    from mlx_vlm.models import bailing_moe_v3_vl

    text = bailing_moe_v3_vl.TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=len(layer_plan),
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        first_k_dense_replace=2,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32,
        n_group=2,
        topk_group=1,
        kv_lora_rank=32,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        layer_group_size=2,
        mrope_section=[2, 1, 1],
        video_start_token=61,
    )
    text.layer_plan = layer_plan
    vision = bailing_moe_v3_vl.VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=1,
        num_position_embeddings=16,
    )
    return bailing_moe_v3_vl.ModelConfig(
        text_config=text,
        vision_config=vision,
        mrope_section=[2, 1, 1],
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        skip_vision=True,
    )


def test_vendor_module_processor_and_prompt_registration(tmp_path):
    from mlx_vlm.models import bailing_moe_v3_vl
    from mlx_vlm.prompt_utils import MODEL_CONFIG, MessageFormat, get_message_json
    from mlx_vlm.utils import get_model_and_args
    from transformers import AutoProcessor

    module, model_type, *_ = get_model_and_args({"model_type": "bailing_moe_v3_vl"})
    assert module is bailing_moe_v3_vl
    assert model_type == "bailing_moe_v3_vl"
    assert MODEL_CONFIG["bailing_moe_v3_vl"] is MessageFormat.LIST_WITH_IMAGE_FIRST
    sentinel = object()
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bailing_moe_v3_vl"}))
    with patch.object(
        bailing_moe_v3_vl.BailingMoeV3VLProcessor,
        "from_pretrained",
        return_value=sentinel,
    ) as load:
        assert AutoProcessor.from_pretrained(tmp_path) is sentinel
        load.assert_called_once()

    image = get_message_json(
        "bailing_moe_v3_vl", "Describe this image.", num_images=1
    )
    assert image["content"][0] == {"type": "image"}
    assert image["content"][1]["type"] == "text"
    assert image["content"][1]["text"] == "Describe this image."
    video = get_message_json(
        "bailing_moe_v3_vl", "Describe this clip.", video=["clip.mp4"]
    )
    assert video["content"][0]["type"] == "video"
    assert video["content"][-1]["type"] == "text"
    assert video["content"][-1]["text"] == "Describe this clip."


def test_processor_normalizes_ling_image_and_video_wrappers():
    from mlx_vlm.models.bailing_moe_v3_vl import BailingMoeV3VLProcessor

    processor = object.__new__(BailingMoeV3VLProcessor)
    processor.vision_start_token = "<|vision_start|>"
    processor.vision_end_token = "<|vision_end|>"
    processor.video_start_token = "<|video_start|>"
    processor.video_end_token = "<|video_end|>"
    processor.image_token = "<|image_pad|>"
    processor.video_token = "<|video_pad|>"

    assert processor._normalize_media_wrappers(
        "<|vision_start|><|image_pad|><|vision_end|>Describe."
    ) == "<|vision_start|><|image_pad|><|vision_end|>\nDescribe."
    assert processor._normalize_media_wrappers(
        "<|video_start|><|video_pad|><|video_end|>Describe."
    ) == (
        "<|video_start|><|vision_start|><|video_pad|>"
        "<|vision_end|><|video_end|>Describe."
    )


def test_tiny_config_and_mxfp4_mla_sanitization():
    from mlx_vlm.models import bailing_moe_v3_vl

    config = _tiny_config(layer_plan=("mla",))
    config.text_config.qk_nope_head_dim = 32
    config.text_config.v_head_dim = 32
    language = bailing_moe_v3_vl.LanguageModel(config.text_config, config)
    fused = mx.random.uniform(shape=(128, 32))
    packed, scales = mx.quantize(fused, group_size=32, bits=4, mode="mxfp4")
    prefix = "language_model.model.layers.0.attention"
    sanitized = language.sanitize(
        {
            f"{prefix}.kv_b_proj.weight": packed,
            f"{prefix}.kv_b_proj.scales": scales,
        }
    )

    assert config.text_config.layer_plan == ("mla",)
    assert config.text_config.rope_scaling["mrope_section"] == [2, 1, 1]
    assert f"{prefix}.embed_q.weight" in sanitized
    assert f"{prefix}.unembed_out.weight" in sanitized
    assert f"{prefix}.embed_q.biases" not in sanitized


def test_mixed_fp4_fp8_quantization_metadata_is_preserved(tmp_path):
    from mlx_vlm.utils import load_config

    experts = {
        "group_size": 32,
        "bits": 4,
        "mode": "mxfp4",
    }
    fp8_attention = {
        "group_size": 64,
        "bits": 8,
        "mode": "mxfp8",
    }
    quantization = {
        "group_size": 64,
        "bits": 8,
        "mode": "mxfp8",
        "language_model.model.layers.2.mlp.switch_mlp.gate_proj": experts,
        "language_model.model.layers.2.attention.dense": fp8_attention,
    }
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bailing_moe_v3_vl",
                "quantization_config": quantization,
            }
        )
    )

    config = load_config(tmp_path)

    assert config["quantization"] == quantization
    assert config["quantization_config"] == quantization
    assert (
        config["quantization"][
            "language_model.model.layers.2.mlp.switch_mlp.gate_proj"
        ]["mode"]
        == "mxfp4"
    )
    assert (
        config["quantization"]["language_model.model.layers.2.attention.dense"][
            "mode"
        ]
        == "mxfp8"
    )


def test_legacy_ling_fp8_metadata_gets_only_explicit_expert_fp4_overrides(tmp_path):
    from mlx_vlm.utils import load_config

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bailing_moe_v3_vl",
                "text_config": {
                    "first_k_dense_replace": 2,
                    "num_hidden_layers": 4,
                },
                "quantization_config": {
                    "quant_method": "fp8",
                    "routed_experts_quant_method": "mxfp4",
                    "routed_experts_group_size": 16,
                    "language_model.model.layers.3.attention.dense": {
                        "group_size": 32,
                        "bits": 8,
                        "mode": "mxfp8",
                    },
                },
            }
        )
    )

    config = load_config(tmp_path)

    assert config["quantization"]["bits"] == 8
    assert config["quantization"][
        "language_model.model.layers.2.mlp.switch_mlp.gate_proj"
    ] == {"group_size": 16, "bits": 4, "mode": "mxfp4"}
    assert config["quantization"][
        "language_model.model.layers.3.attention.dense"
    ] == {"group_size": 32, "bits": 8, "mode": "mxfp8"}


def test_ling_vlm_sanitize_converts_fp8_and_mxfp4_sidecars():
    from mlx_vlm.models import bailing_moe_v3_vl

    config = _tiny_config(layer_plan=("kda", "kda", "mla"))
    language = bailing_moe_v3_vl.LanguageModel(config.text_config, config)
    fp8_weight = "model.layers.0.attention.q_proj.weight"
    expert_weight = "model.layers.2.mlp.experts"
    raw_weights = {
        f"language_model.{fp8_weight}": mx.full((128, 128), 0x38, dtype=mx.uint8),
        f"language_model.{fp8_weight}_scale_inv": mx.ones(
            (1, 1), dtype=mx.float32
        ),
    }
    for expert in range(config.text_config.num_experts):
        key = f"{expert_weight}.{expert}.gate_proj.weight"
        raw_weights[f"language_model.{key}"] = mx.zeros((1, 32), dtype=mx.int8)
        raw_weights[f"language_model.{key}_scale_inv"] = mx.full(
            (1, 2), 127, dtype=mx.uint8
        )

    converted = language.sanitize(raw_weights)
    mxfp4_weight = "model.layers.2.mlp.switch_mlp.gate_proj.weight"

    assert f"language_model.{fp8_weight}_scale_inv" not in converted
    assert f"language_model.{mxfp4_weight}_scale_inv" not in converted
    assert converted[f"language_model.{fp8_weight}"].dtype == mx.uint32
    assert f"language_model.{fp8_weight[:-len('weight')]}scales" in converted
    assert f"language_model.{fp8_weight[:-len('weight')]}biases" in converted
    assert converted[f"language_model.{mxfp4_weight}"].dtype == mx.uint32
    assert f"language_model.{mxfp4_weight[:-len('weight')]}scales" in converted


def test_ling_fp8_and_mxfp4_checkpoint_loads_strictly(tmp_path):
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_vlm.models import bailing_moe_v3_vl
    from mlx_vlm.utils import load_model

    config = _tiny_config(layer_plan=("kda", "kda", "mla"))
    text = config.text_config
    text.hidden_size = 128
    text.intermediate_size = 128
    text.num_attention_heads = 2
    text.num_key_value_heads = 2
    text.head_dim = 64
    text.moe_intermediate_size = 64
    text.moe_shared_expert_intermediate_size = 64
    source = bailing_moe_v3_vl.Model(config)
    weights = dict(tree_flatten(source.parameters()))

    fp8_weight = "language_model.model.layers.0.attention.q_proj.weight"
    weights[fp8_weight] = mx.to_fp8(weights[fp8_weight].astype(mx.float32))
    weights[f"{fp8_weight}_scale_inv"] = mx.ones((1, 1), dtype=mx.float32)

    expert_base = "language_model.model.layers.2.mlp"
    for projection in ("gate_proj", "up_proj", "down_proj"):
        runtime_key = f"{expert_base}.switch_mlp.{projection}.weight"
        expert_weights = weights.pop(runtime_key)
        for expert, expert_weight in enumerate(expert_weights):
            packed, scales = mx.quantize(
                expert_weight,
                group_size=32,
                bits=4,
                mode="mxfp4",
            )
            checkpoint_key = f"{expert_base}.experts.{expert}.{projection}.weight"
            weights[checkpoint_key] = packed.view(mx.int8)
            weights[f"{checkpoint_key}_scale_inv"] = scales

    serialized_config = asdict(config)
    serialized_config.pop("quantization", None)
    serialized_config["quantization_config"] = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "routed_experts_quant_method": "mxfp4",
        "routed_experts_group_size": 32,
    }
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(serialized_config))

    loaded = load_model(tmp_path, strict=True)

    assert isinstance(
        loaded.language_model.model.layers[0].attention.q_proj,
        nn.QuantizedLinear,
    )
    for projection in ("gate_proj", "up_proj", "down_proj"):
        module = getattr(
            loaded.language_model.model.layers[2].mlp.switch_mlp,
            projection,
        )
        assert module.mode == "mxfp4"


def test_tiny_hybrid_kda_mla_forward_and_cache_schedule():
    from mlx_vlm.models import bailing_moe_v3_vl

    config = _tiny_config()
    language = bailing_moe_v3_vl.LanguageModel(config.text_config, config)
    cache = language.make_cache()
    logits = language(mx.array([[1, 2]]), cache=cache).logits
    mx.eval(logits)

    assert config.text_config.layer_plan == ("kda", "mla")
    assert [type(item).__name__ for item in cache] == ["ArraysCache", "KVCache"]
    assert logits.shape == (1, 2, 64)


@pytest.mark.parametrize(
    ("model_type", "architectures"),
    [
        ("bailing_moe_v3_vl", []),
        ("unknown", ["BailingMoeV3VLForConditionalGeneration"]),
    ],
)
def test_ling_vl_discovery(model_type, architectures, tmp_path):
    config = {
        "model_type": model_type,
        "architectures": architectures,
    }
    if model_type == "bailing_moe_v3_vl":
        config["vision_config"] = {"model_type": "qwen3_moe_vit"}
    (tmp_path / "config.json").write_text(
        json.dumps(config)
    )
    assert detect_model_type(tmp_path) == "vlm"
