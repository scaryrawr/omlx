# SPDX-License-Identifier: Apache-2.0
"""Native MXFP4 selected-row compatibility for Qwen4 PLE mmap storage."""

from __future__ import annotations

import json
import struct
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from omlx.patches.mlx_vlm_qwen4_exp_compat.ple_load_resources import (
    ple_load_resources,
)

SOURCE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
RUNTIME_PREFIX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"


@pytest.fixture
def ple():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import language

    return language


def _shard(rows, dims, offset, mode):
    columns = mx.arange(dims)
    values = (
        ((columns % 16).astype(mx.float32) - 8)
        * mx.power(2.0, columns // 32 - 2)
        * (mx.arange(offset + 1, offset + rows + 1)[:, None] / 4)
    )
    values = mx.where((columns // 32) == 2, 0, values).astype(mx.bfloat16)
    embedding = nn.Embedding(rows, dims)
    embedding.weight = values
    if mode == "dense":
        return embedding, {"weight": values}
    embedding = nn.QuantizedEmbedding.from_embedding(
        embedding, group_size=32, bits=8 if mode == "mxfp8" else 4, mode=mode
    )
    tensors = {"weight": embedding.weight, "scales": embedding.scales}
    if embedding.biases is not None:
        tensors["biases"] = embedding.biases
    return embedding, tensors


def _checkpoint(
    path, modes=("mxfp4",) * 3, *, prefix=RUNTIME_PREFIX, spelling="shards"
):
    sizes = (4, 4, 3)
    oracles = []
    weight_map = {}
    offset = 0
    for index, (rows, mode) in enumerate(zip(sizes, modes)):
        oracle, tensors = _shard(rows, 160, offset, mode)
        oracles.append(oracle)
        base = (
            f"{prefix}.shards.{index}"
            if spelling == "shards"
            else f"{prefix}.shard_{index}"
        )
        for family, tensor in tensors.items():
            key = f"{base}.{family}"
            filename = f"shard-{index}-{family}.safetensors"
            mx.save_safetensors(str(path / filename), {key: tensor})
            weight_map[key] = filename
        offset += rows
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return oracles


def _expected(oracles, host, shape, scale):
    table = mx.concatenate(
        [
            oracle(mx.arange(oracle.weight.shape[0])).astype(mx.bfloat16)
            for oracle in oracles
        ]
    )
    return (table[mx.array(host, dtype=mx.int32)] * scale).reshape(*shape, 160)


@pytest.mark.parametrize("prefix", [SOURCE_PREFIX, RUNTIME_PREFIX])
@pytest.mark.parametrize("spelling", ["shards", "shard"])
def test_native_mxfp4_ple_matches_mlx_embedding(tmp_path, ple, prefix, spelling):
    oracles = _checkpoint(tmp_path, prefix=prefix, spelling=spelling)
    with ple_load_resources():
        embedding = ple.DiskBackedShardedEmbedding(
            tmp_path, SOURCE_PREFIX, num_embeddings=11, dims=160, num_shards=3
        )
    try:
        host = [10, 0, 4, 3, 8, 4]
        indices = mx.array(host, dtype=mx.int32).reshape(2, 3)
        expected = _expected(oracles, host, indices.shape, embedding.weight_scale)
        actual = embedding(indices)
        mx.eval(actual, expected)
        assert mx.array_equal(actual, expected).item()
        assert embedding.rows_read == len(host)
        assert embedding.last_uploads == 2
        assert embedding.last_touched_shards == (0, 1, 2)
    finally:
        embedding.close()


def test_mxfp4_lookup_and_prefetch_only_process_selected_rows(
    tmp_path, ple, monkeypatch
):
    oracles = _checkpoint(tmp_path)
    reads = []
    dequantized_shapes = []
    original_read = ple._SafeTensorMMap.rows_np
    original_dequantize = mx.dequantize

    def record_read(reader, key, rows):
        reads.append((key, len(rows)))
        return original_read(reader, key, rows)

    def record_dequantize(weight, scales, biases=None, **kwargs):
        assert kwargs["mode"] == "mxfp4"
        assert biases is None
        assert weight.dtype == mx.uint32
        assert scales.dtype == mx.uint8
        dequantized_shapes.append((weight.shape, scales.shape))
        return original_dequantize(weight, scales, biases, **kwargs)

    monkeypatch.setattr(ple._SafeTensorMMap, "rows_np", record_read)
    with ple_load_resources():
        embedding = ple.DiskBackedShardedEmbedding(tmp_path, SOURCE_PREFIX, 11, 160, 3)
    try:
        assert reads == []
        assert list(embedding.parameters()) == ["weight_scale"]
        embedding.weight_scale = mx.array([1.75], dtype=mx.bfloat16)
        host = [10, 3, 4, 3]
        indices = mx.array(host).reshape(1, 2, 2)
        expected = _expected(oracles, host, indices.shape, embedding.weight_scale)
        mx.eval(expected)
        monkeypatch.setattr(mx, "dequantize", record_dequantize)

        for prefetch in (False, True, False):
            reads.clear()
            if prefetch:
                embedding.prefetch(indices)
            actual = embedding(indices)
            mx.eval(actual)
            assert mx.array_equal(actual, expected).item()
            assert embedding.last_prefetch_hit is prefetch
            assert embedding.last_uploads == 2
            assert sum(count for _, count in reads) == 2 * len(host)
            assert {key.rsplit(".", 1)[-1] for key, _ in reads} == {
                "weight",
                "scales",
            }
        assert dequantized_shapes == [((4, 20), (4, 5))] * 3
    finally:
        embedding.close()


@pytest.mark.parametrize(
    "modes",
    [
        ("mxfp4", "affine", "dense"),
        ("affine", "mxfp4", "affine"),
        ("dense", "mxfp4", "dense"),
    ],
)
def test_mixed_shards_match_native_oracles_exactly(tmp_path, ple, modes):
    oracles = _checkpoint(tmp_path, modes=modes)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "group_size": 32,
                    "bits": 4,
                    "mode": "mxfp4",
                    f"{RUNTIME_PREFIX}.shards.0": {"group_size": 32, "bits": 4},
                    f"{RUNTIME_PREFIX}.shards.2": False,
                }
            }
        )
    )
    with ple_load_resources():
        embedding = ple.DiskBackedShardedEmbedding(tmp_path, SOURCE_PREFIX, 11, 160, 3)
    try:
        embedding.weight_scale = mx.array([0.75], dtype=mx.bfloat16)
        host = [10, 0, 4, 3, 8, 4]
        indices = mx.array(host).reshape(2, 3)
        expected = _expected(oracles, host, indices.shape, embedding.weight_scale)
        for prefetch in (False, True):
            if prefetch:
                embedding.prefetch(indices)
            actual = embedding(indices)
            mx.eval(actual, expected)
            assert mx.array_equal(actual, expected).item()
            assert not embedding.last_prefetch_hit
            assert embedding.rows_read == len(host)
            assert embedding.last_touched_shards == (0, 1, 2)
    finally:
        embedding.close()


@pytest.mark.parametrize("host", [[], [0], [3, 4], [7, 8], [10, 10]])
def test_mxfp4_boundaries_and_empty_ids(tmp_path, ple, host):
    oracles = _checkpoint(tmp_path)
    with ple_load_resources():
        embedding = ple.DiskBackedShardedEmbedding(tmp_path, SOURCE_PREFIX, 11, 160, 3)
    try:
        indices = mx.array(host, dtype=mx.int32)
        actual = embedding(indices)
        expected = _expected(oracles, host, indices.shape, embedding.weight_scale)
        mx.eval(actual, expected)
        assert actual.shape == (len(host), 160)
        assert mx.array_equal(actual, expected).item()
    finally:
        embedding.close()


@pytest.mark.parametrize("host", [[-1], [11], [0, 12]])
def test_mxfp4_out_of_range_rejected_before_row_read(tmp_path, ple, monkeypatch, host):
    _checkpoint(tmp_path)
    with ple_load_resources():
        embedding = ple.DiskBackedShardedEmbedding(tmp_path, SOURCE_PREFIX, 11, 160, 3)
    try:

        def unexpected_read(*args):
            pytest.fail("out-of-range lookup read checkpoint rows")

        monkeypatch.setattr(ple._SafeTensorMMap, "rows_np", unexpected_read)
        for call in (embedding, embedding.prefetch):
            with pytest.raises(IndexError, match="outside the sharded vocabulary"):
                call(mx.array(host))
    finally:
        embedding.close()


@pytest.mark.parametrize(
    "problem,exception,message",
    [
        ("missing_scales", ValueError, "both scales and biases are required"),
        ("float_scales", ValueError, "Incomplete or unsupported"),
        ("biases", TypeError, "must have float dtypes"),
        ("weight_dtype", TypeError, "weight must be U32"),
        ("scale_rows", ValueError, "Unexpected mxfp4 PLE scale shape"),
        ("scale_rank", ValueError, "Unexpected mxfp4 PLE scale shape"),
        ("empty_scales", ValueError, "Unexpected mxfp4 PLE scale shape"),
        ("group_size", ValueError, "Unsupported mxfp4 PLE layout"),
        ("packed_width", ValueError, "Cannot infer mxfp4 PLE bits"),
        ("mxfp8", ValueError, "Unsupported mxfp4 PLE layout"),
        ("nvfp4", ValueError, "Unsupported mxfp4 PLE layout"),
    ],
)
def test_malformed_or_unsupported_packed_shards_fail_explicitly(
    tmp_path, ple, problem, exception, message
):
    _, tensors = _shard(4, 160, 0, "mxfp4")
    if problem == "missing_scales":
        del tensors["scales"]
        tensors["biases"] = mx.ones((4, 5))
    elif problem == "float_scales":
        tensors["scales"] = tensors["scales"].astype(mx.float32)
    elif problem == "biases":
        tensors["biases"] = mx.zeros((4, 5))
    elif problem == "weight_dtype":
        tensors["weight"] = tensors["weight"].astype(mx.float32)
    elif problem == "scale_rows":
        tensors["scales"] = tensors["scales"][:3]
    elif problem == "scale_rank":
        tensors["scales"] = tensors["scales"].reshape(-1)
    elif problem == "empty_scales":
        tensors["scales"] = mx.zeros((4, 0), dtype=mx.uint8)
    elif problem == "group_size":
        tensors["scales"] = mx.ones((4, 10), dtype=mx.uint8)
    elif problem == "packed_width":
        tensors["weight"] = tensors["weight"][:, :19]
    elif problem == "mxfp8":
        _, tensors = _shard(4, 160, 0, problem)
    elif problem == "nvfp4":
        weight, scales = mx.quantize(
            mx.ones((4, 160)), group_size=16, bits=4, mode="nvfp4"
        )
        tensors = {"weight": weight, "scales": scales}
    base = f"{RUNTIME_PREFIX}.shards.0"
    tensors = {f"{base}.{key}": value for key, value in tensors.items()}
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    with pytest.raises(exception, match=message), ple_load_resources():
        ple.DiskBackedShardedEmbedding(tmp_path, SOURCE_PREFIX, 4, 160, 1)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32, "fp8"])
def test_dense_storage_keeps_selected_row_behavior(tmp_path, ple, dtype):
    values = mx.arange(4 * 160).reshape(4, 160) / 97
    weight = mx.to_fp8(values) if dtype == "fp8" else values.astype(dtype)
    key = f"{RUNTIME_PREFIX}.shards.0.weight"
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), {key: weight})
    if dtype == "fp8":
        raw = path.read_bytes()
        size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + size])
        header[key]["dtype"] = "F8_E4M3"
        encoded = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + size :])
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: path.name}})
    )
    with ple_load_resources():
        embedding = ple.DiskBackedShardedEmbedding(tmp_path, SOURCE_PREFIX, 4, 160, 1)
    try:
        indices = mx.array([3, 0, 3])
        expected = (
            mx.from_fp8(weight[indices], dtype=mx.bfloat16)
            if dtype == "fp8"
            else weight[indices].astype(mx.bfloat16)
        )
        actual = embedding(indices)
        mx.eval(actual, expected)
        assert mx.array_equal(actual, expected).item()
        assert embedding.last_uploads == 1
    finally:
        embedding.close()


@pytest.mark.parametrize("scale_dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("bias_dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("per_shard", [False, True])
def test_affine_preserves_mixed_float_parameter_precision(
    tmp_path, ple, monkeypatch, scale_dtype, bias_dtype, per_shard
):
    _, tensors = _shard(4, 160, 0, "affine")
    tensors["scales"] = tensors["scales"].astype(scale_dtype)
    tensors["biases"] = tensors["biases"].astype(bias_dtype)
    indices = mx.array([3, 0, 3])
    dtype = mx.result_type(tensors["scales"], tensors["biases"])
    expected = mx.dequantize(
        tensors["weight"][indices],
        tensors["scales"][indices].astype(dtype),
        tensors["biases"][indices].astype(dtype),
        group_size=32,
        bits=4,
        mode="affine",
    ).astype(mx.bfloat16)
    base = f"{RUNTIME_PREFIX}.shards.0"
    tensors = {f"{base}.{key}": value for key, value in tensors.items()}
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    with ple_load_resources():
        embedding = ple.DiskBackedShardedEmbedding(tmp_path, SOURCE_PREFIX, 4, 160, 1)
    if per_shard:
        monkeypatch.setattr(embedding, "_plan", lambda _indices: None)
    try:
        actual = embedding(indices)
        mx.eval(actual, expected)
        assert mx.array_equal(actual, expected).item()
        if not per_shard:
            assert embedding.last_uploads == 3
    finally:
        embedding.close()


@pytest.mark.parametrize("mtp_layout", ["off", "embedded", "sidecar"])
def test_complete_mxfp4_checkpoint_loads_strictly_with_mmap_ple(
    tmp_path, ple, mtp_layout
):
    from mlx_vlm.models.qwen4_exp import Model
    from mlx_vlm.utils import load_model
    from test_mlx_vlm_qwen4_exp_compat import _tiny_config

    from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load

    config = _tiny_config()
    config.text_config.ple_embed_dim = 640
    mtp_enabled = mtp_layout != "off"
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"mtp.fc_hidden.weight": "model.safetensors"}}
            if mtp_enabled
            else {"weight_map": {}}
        )
    )
    ple.configure_ple_runtime(tmp_path, mode="resident")
    ple.configure_mtp_runtime(tmp_path, enabled=mtp_enabled)
    loaded = None
    try:
        source = Model(config)
        prefix = "language_model.model.layers.0.ple.ple_embedding.ngram_embedding"
        quantization = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
        for path, _ in source.named_modules():
            quantization[path] = (
                {"group_size": 32, "bits": 4, "mode": "mxfp4"}
                if path.startswith(f"{prefix}.shards.")
                else False
            )
        nn.quantize(
            source,
            class_predicate=lambda path, _: quantization[path],
        )
        source_embedding = source.language_model.model.layers[
            0
        ].ple.ple_embedding.ngram_embedding
        source_embedding.weight_scale = mx.array([1.5], dtype=mx.bfloat16)
        host = [
            source_embedding.shard_offsets[-1] - 1,
            0,
            source_embedding.shard_offsets[1],
        ]
        expected = source_embedding(mx.array(host))
        mx.eval(expected)
        config_dict = asdict(config)
        config_dict["quantization"] = quantization
        (tmp_path / "config.json").write_text(json.dumps(config_dict))
        weights = dict(tree_flatten(source.parameters()))
        if mtp_layout == "sidecar":
            sidecar = tmp_path / "mtp"
            sidecar.mkdir()
            (sidecar / "config.json").write_text(
                json.dumps({"model_type": "qwen4_exp_mtp", "block_size": 2})
            )
            mtp_weights = {
                path.removeprefix("mtp."): value
                for path, value in weights.items()
                if path.startswith("mtp.")
            }
            mx.save_safetensors(str(sidecar / "model.safetensors"), mtp_weights)
            weights = {
                path: value
                for path, value in weights.items()
                if not path.startswith("mtp.")
            }
        mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": dict.fromkeys(weights, "model.safetensors")})
        )

        ple.configure_ple_runtime(tmp_path, mode="mmap")
        ple.configure_mtp_runtime(tmp_path, enabled=mtp_enabled)
        if mtp_layout == "sidecar":
            assert ple.get_mtp_runtime().checkpoint_prefix == "mtp/"
        with _force_qwen4_exp_sanitize_on_load(tmp_path):
            loaded = load_model(tmp_path, strict=True)
        embedding = loaded.language_model.model.layers[
            0
        ].ple.ple_embedding.ngram_embedding
        assert isinstance(embedding, ple.DiskBackedShardedEmbedding)
        actual = embedding(mx.array(host))
        mx.eval(actual)
        assert mx.array_equal(actual, expected).item()
        assert embedding.last_uploads == 2
        assert embedding.rows_read == len(host)
        if mtp_enabled:
            assert loaded.language_model.get_mtp_module() is loaded.mtp
            assert getattr(loaded.mtp.layers[0], "ple", None) is None
    finally:
        if loaded is not None:
            loaded.language_model.model.layers[
                0
            ].ple.ple_embedding.ngram_embedding.close()
        ple.configure_mtp_runtime(tmp_path, enabled=False)
        ple.configure_ple_runtime(tmp_path, mode="resident")
