# SPDX-License-Identifier: Apache-2.0
"""Tiny HF-shaped checkpoints and tokenizer for MiniCPM integration tests."""

import json
import re

import mlx.core as mx
import numpy as np


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
            "mtp_num_hidden_layers": 1,
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
            or name.endswith("linear_attn.norm.weight")
        ) and name.endswith(".weight"):
            values += 1.0
        result[name] = mx.array(values.reshape(shape))
    return result


class Tokenizer:
    tokens = {
        "<|image_pad|>": 10,
        "<|video_pad|>": 11,
        "<image>": 12,
        "</image>": 13,
        "<slice>": 14,
        "</slice>": 15,
        "<image_id>": 16,
        "</image_id>": 17,
        "<|im_end|>": 18,
        "<|endoftext|>": 19,
        "\n": 20,
    }
    pad_token_id = 0
    unk_token_id = -1
    eos_token_id = 18
    eos_token = "<|im_end|>"
    pad_token = "<pad>"
    chat_template = "local-template"
    init_kwargs = {}

    def convert_tokens_to_ids(self, token):
        return self.tokens.get(token, self.unk_token_id)

    def encode(self, text, **kwargs):
        pattern = "|".join(
            re.escape(token) for token in sorted(self.tokens, key=len, reverse=True)
        )
        return [
            self.tokens[chunk] if chunk in self.tokens else 100 + ord(chunk)
            for chunk in re.findall(pattern + r"|[\s\S]", text)
        ]

    def decode(self, ids, **kwargs):
        inverse = {value: key for key, value in self.tokens.items()}
        return "".join(
            inverse.get(value, chr(value - 100) if value >= 100 else "")
            for value in ids
        )

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row, **kwargs) for row in rows]

    def apply_chat_template(self, messages, **kwargs):
        output = json.dumps(kwargs.get("tools", [])) + "".join(
            message["content"] for message in messages
        )
        if kwargs.get("add_generation_prompt"):
            output += (
                "<think>"
                if kwargs.get("enable_thinking", True)
                else "<think>\n\n</think>\n\n"
            )
        return self.encode(output) if kwargs.get("tokenize") else output
