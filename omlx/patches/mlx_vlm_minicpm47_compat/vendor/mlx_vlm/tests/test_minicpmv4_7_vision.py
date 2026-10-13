"""Tiny CPU parity against the equations/order in Transformers v5.18.0.

The independent NumPy reference uses explicit patch/window coordinate loops.
No checkpoint, PyTorch, or accelerator is needed.
"""

import math
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.models.minicpmv4_7.vision import Merger, VisionModel


def config(**overrides):
    values = dict(
        hidden_size=4,
        intermediate_size=7,
        num_attention_heads=2,
        num_hidden_layers=3,
        layer_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
        image_size=8,
        patch_size=2,
        num_channels=3,
        insert_layer_id=1,
        window_kernel_size=(2, 2),
        window_hidden_size=16,
        window_intermediate_size=28,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def root_config(vision, merger_times=1):
    return SimpleNamespace(
        vision_config=vision,
        text_config=SimpleNamespace(hidden_size=6),
        merge_kernel_size=(2, 2),
        merger_times=merger_times,
    )


def fill(model):
    weights = {}
    for index, (name, value) in enumerate(sorted(tree_flatten(model.parameters()))):
        array = (
            np.sin(
                np.arange(value.size, dtype=np.float32) * 0.31 + index * 0.71
            ).reshape(value.shape)
            * 0.17
        )
        if name.endswith(".weight") and ("norm" in name or "layernorm" in name):
            array += 1
        weights[name] = array
    model.load_weights(
        [(name, mx.array(value)) for name, value in weights.items()], strict=True
    )
    return weights


def linear(x, weights, name):
    return x @ weights[name + ".weight"].T + weights[name + ".bias"]


def norm(x, weights, name, eps):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * weights[name + ".weight"] + weights[
        name + ".bias"
    ]


def gelu(x, tanh=False):
    if tanh:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    erf = np.vectorize(math.erf)(x / np.sqrt(2))
    return (0.5 * x * (1 + erf)).astype(np.float32)


def attention(x, weights, name, heads):
    count, dim = x.shape
    head_dim = dim // heads
    q, k, v = [
        linear(x, weights, name + "." + projection).reshape(count, heads, head_dim)
        for projection in ("q_proj", "k_proj", "v_proj")
    ]
    output = np.empty_like(q)
    for head in range(heads):
        scores = q[:, head] @ k[:, head].T / math.sqrt(head_dim)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        output[:, head] = probs @ v[:, head]
    return linear(output.reshape(count, dim), weights, name + ".out_proj")


def image_crops(grids, patch_size=2):
    return [
        np.sin(
            np.arange(h * patch_size * w * patch_size * 3, dtype=np.float32) * 0.13 + i
        ).reshape(h * patch_size, w * patch_size, 3)
        for i, (h, w) in enumerate(grids)
    ]


def pack(crops, patch_size=2):
    patches = []
    for crop in crops:
        for row in range(0, crop.shape[0], patch_size):
            for col in range(0, crop.shape[1], patch_size):
                patches.append(crop[row : row + patch_size, col : col + patch_size])
    return mx.array(np.concatenate(patches, axis=1).transpose(2, 0, 1)[None])


def ref_embeddings(crops, grids, weights, cfg):
    result = []
    side = cfg.image_size // cfg.patch_size
    boundaries = np.asarray(
        [1 / side + i / side for i in range(side - 1)], dtype=np.float32
    )
    for crop, (height, width) in zip(crops, grids):
        hidden = []
        for row in range(height):
            for col in range(width):
                patch = crop[
                    row * cfg.patch_size : (row + 1) * cfg.patch_size,
                    col * cfg.patch_size : (col + 1) * cfg.patch_size,
                ]
                projection = (
                    np.sum(
                        weights["embeddings.patch_embedding.weight"] * patch[None],
                        axis=(1, 2, 3),
                    )
                    + weights["embeddings.patch_embedding.bias"]
                )
                r = np.searchsorted(boundaries, np.float32(row) / height, side="right")
                c = np.searchsorted(boundaries, np.float32(col) / width, side="right")
                hidden.append(
                    projection
                    + weights["embeddings.position_embedding.weight"][r * side + c]
                )
        result.append(np.asarray(hidden))
    return result


def ref_windows(hidden, height, width):
    return [
        np.asarray(
            [hidden[(r + dr) * width + c + dc] for dr in range(2) for dc in range(2)]
        )
        for r in range(0, height, 2)
        for c in range(0, width, 2)
    ]


def ref_tower(crops, grids, weights, cfg, use_merger):
    hidden = ref_embeddings(crops, grids, weights, cfg)
    grids = list(grids)
    for index in range(cfg.num_hidden_layers):
        prefix = f"encoder.layers.{index}"
        for i, x in enumerate(hidden):
            x = x + attention(
                norm(x, weights, prefix + ".layer_norm1", cfg.layer_norm_eps),
                weights,
                prefix + ".self_attn",
                cfg.num_attention_heads,
            )
            z = norm(x, weights, prefix + ".layer_norm2", cfg.layer_norm_eps)
            z = gelu(linear(z, weights, prefix + ".mlp.fc1"), cfg.hidden_act != "gelu")
            hidden[i] = x + linear(z, weights, prefix + ".mlp.fc2")
        if use_merger and index == cfg.insert_layer_id:
            prefix = "vit_merger"
            for i, (height, width) in enumerate(grids):
                output = []
                for window in ref_windows(hidden[i], height, width):
                    window = window + attention(
                        norm(
                            window,
                            weights,
                            prefix + ".layer_norm1",
                            cfg.layer_norm_eps,
                        ),
                        weights,
                        prefix + ".self_attn",
                        cfg.num_attention_heads,
                    )
                    flat = norm(
                        window.reshape(-1),
                        weights,
                        prefix + ".pre_norm",
                        cfg.layer_norm_eps,
                    )
                    z = gelu(linear(flat, weights, prefix + ".linear_1"), tanh=True)
                    output.append(
                        linear(z, weights, prefix + ".linear_2") + window.mean(axis=0)
                    )
                hidden[i] = np.asarray(output)
                grids[i] = (height // 2, width // 2)
    return [
        norm(x, weights, "post_layernorm", cfg.layer_norm_eps) for x in hidden
    ], grids


def ref_merger(hidden, grids, weights, rounds):
    results = []
    for x, (height, width) in zip(hidden, grids):
        for index in range(rounds):
            flat = np.asarray(
                [window.reshape(-1) for window in ref_windows(x, height, width)]
            )
            prefix = f"mlp.{index}"
            flat = norm(flat, weights, prefix + ".pre_norm", 1e-6)
            x = linear(
                gelu(linear(flat, weights, prefix + ".linear_1")),
                weights,
                prefix + ".linear_2",
            )
            height, width = height // 2, width // 2
        results.append(x)
    return results


@pytest.mark.parametrize("use_merger", [False, True])
@pytest.mark.parametrize("hidden_act", ["gelu", "gelu_pytorch_tanh"])
def test_release_equations_packed_order_and_reduced_grids(use_merger, hidden_act):
    cfg = config(hidden_act=hidden_act)
    model = VisionModel(cfg)
    weights = fill(model)
    grids = [(4, 8), (8, 4)]
    crops = image_crops(grids)
    pixels = pack(crops)
    expected, expected_grids = ref_tower(crops, grids, weights, cfg, use_merger)
    actual, actual_grids = model(pixels, mx.array(grids), use_merger)
    np.testing.assert_array_equal(np.asarray(actual_grids), expected_grids)
    np.testing.assert_allclose(
        np.asarray(actual[0]), np.concatenate(expected), rtol=3e-5, atol=3e-5
    )
    three_dim, _ = model(pixels[0], mx.array(grids), use_merger)
    np.testing.assert_allclose(
        np.asarray(three_dim), np.asarray(actual), rtol=1e-6, atol=1e-6
    )

    merger = Merger(root_config(cfg))
    merger_weights = fill(merger)
    features = merger(actual, actual_grids)
    expected_features = ref_merger(expected, expected_grids, merger_weights, 1)
    assert isinstance(features, tuple)
    for actual_crop, expected_crop in zip(features, expected_features):
        np.testing.assert_allclose(
            np.asarray(actual_crop), expected_crop, rtol=3e-5, atol=3e-5
        )
        assert actual_crop.shape == (2 if use_merger else 8, 6)


@pytest.mark.parametrize("use_merger", [False, True])
def test_crop_isolation_and_request_local_compression(use_merger):
    cfg = config()
    model = VisionModel(cfg)
    fill(model)
    grids = [(4, 8), (4, 4)]
    crops = image_crops(grids)
    original, original_grids = model(pack(crops), mx.array(grids), use_merger)
    first, first_grid = model(pack(crops[:1]), mx.array(grids[:1]), use_merger)
    length = first.shape[1]
    np.testing.assert_allclose(
        np.asarray(original[:, :length]), np.asarray(first), atol=2e-6
    )
    modified = [crops[0], crops[1] * 91 + 14]
    altered, _ = model(pack(modified), mx.array(grids), use_merger)
    np.testing.assert_allclose(
        np.asarray(altered[:, :length]), np.asarray(first), atol=2e-6
    )
    model(pack(modified), mx.array(grids), not use_merger)
    again, again_grids = model(pack(crops), mx.array(grids), use_merger)
    np.testing.assert_allclose(np.asarray(again), np.asarray(original), atol=2e-6)
    np.testing.assert_array_equal(np.asarray(again_grids), np.asarray(original_grids))
    np.testing.assert_array_equal(
        np.asarray(first_grid), np.asarray(original_grids[:1])
    )


def test_window_mean_residual_and_coordinate_order():
    cfg = config(num_hidden_layers=1, insert_layer_id=0)
    model = VisionModel(cfg)
    zero_weights = [
        (name, mx.zeros_like(value))
        for name, value in tree_flatten(model.vit_merger.parameters())
    ]
    model.vit_merger.load_weights(zero_weights, strict=True)
    grids = [(4, 6), (2, 4)]
    values = np.arange(32 * 4, dtype=np.float32).reshape(1, 32, 4)
    actual = model.vit_merger(mx.array(values), mx.array(grids))
    expected = []
    offset = 0
    for height, width in grids:
        expected.extend(
            [
                window.mean(axis=0)
                for window in ref_windows(
                    values[0, offset : offset + height * width], height, width
                )
            ]
        )
        offset += height * width
    np.testing.assert_array_equal(np.asarray(actual[0]), expected)


def test_nearest_positions_at_bucket_boundaries_and_unequal_crops():
    cfg = config(image_size=140, patch_size=2)
    model = VisionModel(cfg)
    model.embeddings.patch_embedding.weight = mx.zeros_like(
        model.embeddings.patch_embedding.weight
    )
    model.embeddings.patch_embedding.bias = mx.zeros_like(
        model.embeddings.patch_embedding.bias
    )
    table = np.repeat(
        np.arange(70**2, dtype=np.float32)[:, None], cfg.hidden_size, axis=1
    )
    model.embeddings.position_embedding.weight = mx.array(table)
    grids = [(7, 10), (3, 14), (2, 140)]
    crops = image_crops(grids)
    weights = {
        name: np.asarray(value) for name, value in tree_flatten(model.parameters())
    }
    expected = ref_embeddings(crops, grids, weights, cfg)
    actual = model.embeddings(pack(crops), mx.array(grids))
    np.testing.assert_array_equal(np.asarray(actual[0]), np.concatenate(expected))


def test_final_merger_multiple_rounds_exact_gelu():
    cfg = config()
    merger = Merger(root_config(cfg, merger_times=2))
    weights = fill(merger)
    grids = [(4, 8), (8, 4)]
    hidden = np.cos(np.arange(64 * 4, dtype=np.float32) * 0.19).reshape(1, 64, 4)
    actual = merger(mx.array(hidden), mx.array(grids))
    expected = ref_merger([hidden[0, :32], hidden[0, 32:]], grids, weights, 2)
    for crop, reference in zip(actual, expected):
        assert crop.shape == (2, 6)
        np.testing.assert_allclose(np.asarray(crop), reference, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("insert_layer_id", [-1, 3])
def test_no_reduction_without_matching_insert_layer(insert_layer_id):
    cfg = config(insert_layer_id=insert_layer_id)
    model = VisionModel(cfg)
    fill(model)
    grids = [(3, 5)]
    hidden, actual_grids = model(pack(image_crops(grids)), mx.array(grids))
    assert hidden.shape == (1, 15, cfg.hidden_size)
    np.testing.assert_array_equal(np.asarray(actual_grids), grids)


@pytest.mark.parametrize("grids", [[[0, 2]], [[2.5, 4]], [2, 4], []])
def test_invalid_grids_fail_explicitly(grids):
    model = VisionModel(config())
    with pytest.raises(ValueError, match="target_sizes"):
        model(mx.zeros((1, 3, 2, 8)), mx.array(grids))


def test_mismatched_packing_and_indivisible_windows_fail():
    cfg = config()
    model = VisionModel(cfg)
    with pytest.raises(ValueError, match="NaViT"):
        model(mx.zeros((1, 3, 2, 8)), mx.array([[4, 4]]))
    grids = [(3, 4)]
    with pytest.raises(ValueError, match="divisible"):
        model(pack(image_crops(grids)), mx.array(grids))
    merger = Merger(root_config(cfg))
    with pytest.raises(ValueError, match="divisible"):
        merger(mx.zeros((1, 12, cfg.hidden_size)), mx.array(grids))
    with pytest.raises(ValueError, match="target_sizes"):
        merger(mx.zeros((1, 13, cfg.hidden_size)), mx.array(grids))


def test_parameter_namespace_has_all_release_merger_weights():
    model = VisionModel(config())
    names = dict(tree_flatten(model.parameters()))
    assert len(names) == 3 + 3 * 16 + 2 + 16
    for suffix in ("weight", "bias"):
        assert f"vit_merger.layer_norm1.{suffix}" in names
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            assert f"vit_merger.self_attn.{projection}.{suffix}" in names
        for layer in ("pre_norm", "linear_1", "linear_2"):
            assert f"vit_merger.{layer}.{suffix}" in names
    merger = Merger(root_config(config()))
    assert set(dict(tree_flatten(merger.parameters()))) == {
        f"mlp.0.{layer}.{suffix}"
        for layer in ("pre_norm", "linear_1", "linear_2")
        for suffix in ("weight", "bias")
    }
