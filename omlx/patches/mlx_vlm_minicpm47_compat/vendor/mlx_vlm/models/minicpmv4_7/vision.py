# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from Transformers v5.18.0 modeling_minicpmv4_7.py and vision_utils.py.

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _grids(target_sizes):
    sizes = np.asarray(target_sizes)
    if (
        sizes.ndim != 2
        or sizes.shape[1] != 2
        or not len(sizes)
        or not np.issubdtype(sizes.dtype, np.integer)
        or np.any(sizes <= 0)
    ):
        raise ValueError(
            "target_sizes must contain positive integer (height, width) grids"
        )
    return tuple((int(h), int(w)) for h, w in sizes)


def _gelu(name):
    if name in {"gelu_pytorch_tanh", "gelu_new", "gelu_fast"}:
        return nn.GELU(approx="tanh")
    if name == "gelu":
        return nn.GELU(approx="precise")
    raise ValueError(f"Unsupported vision activation: {name}")


def _windows(hidden_states, height, width, kernel):
    kh, kw = kernel
    if height % kh or width % kw:
        raise ValueError(
            f"Patch grid ({height}, {width}) must be divisible by {kernel}"
        )
    return (
        hidden_states.reshape(height // kh, kh, width // kw, kw, -1)
        .transpose(0, 2, 1, 3, 4)
        .reshape(height // kh * (width // kw), kh * kw, -1)
    )


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != config.hidden_size:
            raise ValueError(
                "Vision hidden_size must be divisible by num_attention_heads"
            )
        self.scale = self.head_dim**-0.5
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, hidden_states, lengths):
        batch, total, dim = hidden_states.shape
        if sum(lengths) != total:
            raise ValueError("Attention crop lengths do not match hidden states")
        q, k, v = (
            proj(hidden_states)
            .reshape(batch, total, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        outputs = []
        start = 0
        for length in lengths:
            end = start + length
            outputs.append(
                mx.fast.scaled_dot_product_attention(
                    q[:, :, start:end],
                    k[:, :, start:end],
                    v[:, :, start:end],
                    scale=self.scale,
                )
            )
            start = end
        output = mx.concatenate(outputs, axis=2)
        return self.out_proj(output.transpose(0, 2, 1, 3).reshape(batch, total, dim))


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation_fn = _gelu(config.hidden_act)

    def __call__(self, hidden_states):
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class VisionEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = VisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config)

    def __call__(self, hidden_states, lengths):
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states), lengths
        )
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = [
            VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]


class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.num_patches_per_side = config.image_size // config.patch_size
        self.patch_embedding = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        self.position_embedding = nn.Embedding(
            self.num_patches_per_side**2, config.hidden_size
        )

    def __call__(self, pixel_values, target_sizes):
        grids = _grids(target_sizes)
        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None]
        patches = sum(h * w for h, w in grids)
        expected = (1, self.num_channels, self.patch_size, patches * self.patch_size)
        if pixel_values.shape != expected:
            raise ValueError(
                f"Expected NaViT packed pixels {expected}, got {pixel_values.shape}"
            )
        pixels = pixel_values.transpose(0, 2, 3, 1).astype(
            self.patch_embedding.weight.dtype
        )
        hidden_states = self.patch_embedding(pixels).reshape(1, patches, -1)
        side = self.num_patches_per_side
        # Match torch.arange's float64 step calculation followed by float32
        # rounding; MLX arange accumulates rounded steps at exact bucket edges.
        boundaries = np.asarray(
            [1 / side + i / side for i in range(side - 1)], dtype=np.float32
        )
        positions = []
        for height, width in grids:
            rows = np.searchsorted(
                boundaries, np.arange(height, dtype=np.float32) / height, side="right"
            )
            cols = np.searchsorted(
                boundaries, np.arange(width, dtype=np.float32) / width, side="right"
            )
            positions.append((rows[:, None] * side + cols[None]).reshape(-1))
        ids = mx.array(np.concatenate(positions), dtype=mx.int32)
        return hidden_states + self.position_embedding(ids)[None]


class ViTWindowAttentionMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.window_kernel_size = tuple(config.window_kernel_size)
        if self.window_kernel_size != (2, 2):
            raise ValueError("MiniCPM-V 4.7 requires a 2x2 insertion window")
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = VisionAttention(config)
        self.pre_norm = nn.LayerNorm(
            config.window_hidden_size, eps=config.layer_norm_eps
        )
        self.linear_1 = nn.Linear(
            config.window_hidden_size, config.window_intermediate_size
        )
        self.linear_2 = nn.Linear(config.window_intermediate_size, config.hidden_size)
        self.act = nn.GELU(approx="tanh")

    def __call__(self, hidden_states, target_sizes):
        grids = _grids(target_sizes)
        windows = []
        start = 0
        for height, width in grids:
            end = start + height * width
            windows.append(
                _windows(
                    hidden_states[0, start:end], height, width, self.window_kernel_size
                )
            )
            start = end
        if hidden_states.shape[:2] != (1, start):
            raise ValueError("Window merger hidden states do not match target_sizes")
        residual = mx.concatenate(windows, axis=0)
        # Every batch row is one independent window, in crop-local raster order.
        patches = residual + self.self_attn(self.layer_norm1(residual), (4,))
        flat = patches.reshape(patches.shape[0], -1)
        output = self.linear_2(self.act(self.linear_1(self.pre_norm(flat))))
        return (output + mx.mean(patches, axis=1))[None]


class VisionModel(nn.Module):
    """Release-exact packed tower returning ``(hidden[1, N, H], grids[crops, 2])``."""

    def __init__(self, config):
        super().__init__()
        self.insert_layer_id = config.insert_layer_id
        self.embeddings = VisionEmbeddings(config)
        self.encoder = VisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.vit_merger = ViTWindowAttentionMerger(config)

    def __call__(self, pixel_values, target_sizes, use_vit_merger=True):
        grids = _grids(target_sizes)
        hidden_states = self.embeddings(pixel_values, target_sizes)
        lengths = tuple(h * w for h, w in grids)
        for index, layer in enumerate(self.encoder.layers):
            hidden_states = layer(hidden_states, lengths)
            if use_vit_merger and index == self.insert_layer_id:
                hidden_states = self.vit_merger(hidden_states, grids)
                grids = tuple((h // 2, w // 2) for h, w in grids)
                lengths = tuple(h * w for h, w in grids)
        return self.post_layernorm(hidden_states), mx.array(grids, dtype=mx.int32)


class DownsampleMLP(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        dim = 4 * hidden_size
        self.pre_norm = nn.LayerNorm(dim, eps=1e-6)
        self.linear_1 = nn.Linear(dim, dim)
        self.linear_2 = nn.Linear(dim, output_size)
        self.act = nn.GELU(approx="precise")

    def __call__(self, hidden_states):
        return self.linear_2(self.act(self.linear_1(self.pre_norm(hidden_states))))


class Merger(nn.Module):
    """Project post-encoder grids to a tuple of crop-local language features."""

    def __init__(self, config):
        super().__init__()
        self.merge_kernel_size = tuple(config.merge_kernel_size)
        if self.merge_kernel_size != (2, 2):
            raise ValueError("MiniCPM-V 4.7 requires a 2x2 final merge kernel")
        hidden_size = config.vision_config.hidden_size
        self.mlp = [
            DownsampleMLP(
                hidden_size,
                (
                    config.text_config.hidden_size
                    if i == config.merger_times - 1
                    else hidden_size
                ),
            )
            for i in range(config.merger_times)
        ]
        if not self.mlp:
            raise ValueError("merger_times must be positive")

    def __call__(self, hidden_states, target_sizes):
        grids = _grids(target_sizes)
        if hidden_states.ndim != 3 or hidden_states.shape[:2] != (
            1,
            sum(h * w for h, w in grids),
        ):
            raise ValueError("Final merger hidden states do not match target_sizes")
        features = []
        start = 0
        for height, width in grids:
            output = hidden_states[0, start : start + height * width]
            start += height * width
            for mlp in self.mlp:
                windows = _windows(output, height, width, self.merge_kernel_size)
                output = mlp(windows.reshape(windows.shape[0], -1))
                height, width = height // 2, width // 2
            features.append(output)
        return tuple(features)
