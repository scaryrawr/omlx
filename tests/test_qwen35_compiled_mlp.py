# SPDX-License-Identifier: Apache-2.0
"""Tests for compiled Qwen3.5-family decode MLP dispatch."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.qwen3_next import (
    ModelArgs,
    Qwen3NextMLP,
    Qwen3NextSparseMoeBlock,
)
from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP
from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

from omlx.patches.qwen35_compiled_mlp import (
    CompiledMLPBlock,
    CompiledMLPBlocks,
    CompiledTargetVerifyMLPBlock,
)


class _Host(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


def _dense_mlp(cls=Qwen3NextMLP):
    mx.random.seed(31)
    mlp = cls(64, 128)
    mlp.eval()
    nn.quantize(mlp, group_size=64, bits=4)
    mx.eval(mlp.parameters())
    return mlp


def _moe_block(cls=Qwen3NextSparseMoeBlock):
    mx.random.seed(31)
    args = ModelArgs(
        model_type="qwen3_next",
        hidden_size=64,
        num_hidden_layers=1,
        intermediate_size=128,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=128,
        mlp_only_layers=[],
        moe_intermediate_size=128,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        rope_theta=10000.0,
        partial_rotary_factor=0.25,
        max_position_embeddings=2048,
        head_dim=16,
    )
    block = cls(args)
    block.eval()
    nn.quantize(block.switch_mlp, group_size=64, bits=4)
    nn.quantize(block.shared_expert, group_size=64, bits=4)
    mx.eval(block.parameters())
    return block


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_install_is_explicitly_gated_and_idempotent(monkeypatch):
    host = _Host(_dense_mlp())
    monkeypatch.setenv("OMLX_QWEN35_COMPILED_MLP", "0")
    assert CompiledMLPBlocks.install(host) == 0
    assert not isinstance(host.mlp, CompiledMLPBlock)

    assert CompiledMLPBlocks.install(host, enabled=True) == 1
    wrapper = host.mlp
    assert isinstance(wrapper, CompiledMLPBlock)
    assert CompiledMLPBlocks.install(host, enabled=True) == 0
    assert host.mlp is wrapper
    assert CompiledMLPBlocks.install(wrapper, enabled=True) == 0
    assert not isinstance(wrapper.inner, CompiledMLPBlock)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_dispatch_defaults_on_with_opt_out(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_COMPILED_MLP", raising=False)
    assert CompiledMLPBlocks.enabled()
    monkeypatch.setenv("OMLX_QWEN35_COMPILED_MLP", "0")
    assert not CompiledMLPBlocks.enabled()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("cls", [Qwen3NextMLP, Qwen3_5MLP])
def test_prefill_patch_preserves_compiled_decode_policy(monkeypatch, cls):
    from omlx.patches.qwen35_q4_mlp import _make_patched_mlp

    original = getattr(cls, "_omlx_q4_mlp_original_call", cls.__call__)
    monkeypatch.setattr(cls, "__call__", _make_patched_mlp(original, 8, 128, 16384))
    inner = _dense_mlp(cls)
    host = _Host(inner)
    x = mx.random.normal((1, 1, 64)).astype(mx.float16)
    expected = inner(x)
    assert CompiledMLPBlocks.install(host, enabled=True) == 1
    actual = host.mlp(x)
    mx.eval(expected, actual)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("cls", [Qwen3NextSparseMoeBlock, Qwen3_5MoeSparseMoeBlock])
def test_eager_sparse_wrapper_does_not_block_dense_compilation(monkeypatch, cls):
    original = cls.__call__

    def eager(self, x, *args, **kwargs):
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(cls, "__call__", eager)
    sparse = _moe_block(cls)
    host = _Host(sparse)
    host.dense = _dense_mlp()
    assert CompiledMLPBlocks.install(host, enabled=True) == 1
    assert host.mlp is sparse
    assert isinstance(host.dense, CompiledMLPBlock)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("batch,seq", [(1, 1), (1, 4)])
def test_compiled_quantized_dense_output_is_bit_exact(batch, seq):
    inner = _dense_mlp()
    host = _Host(inner)
    x = mx.random.normal((batch, seq, 64)).astype(mx.float16)
    expected = inner(x)
    assert CompiledMLPBlocks.install(host, enabled=True) == 1

    actual = host.mlp(x)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("cls", [Qwen3NextSparseMoeBlock, Qwen3_5MoeSparseMoeBlock])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("batch,seq", [(1, 1), (1, 3), (4, 1), (1, 5)])
@pytest.mark.parametrize("direct", [False, True])
def test_quantized_moe_stays_eager_and_bit_exact(
    cls, dtype, batch, seq, direct, monkeypatch
):
    inner = _moe_block(cls)
    host = _Host(inner)
    model = inner if direct else host
    shared_expert = inner.shared_expert
    x = mx.random.normal((batch, seq, 64)).astype(dtype)
    expected = inner(x)
    mx.eval(expected)

    def fail_compile(*args, **kwargs):
        raise AssertionError("sparse MoE blocks and their children must stay eager")

    monkeypatch.setattr(mx, "compile", fail_compile)
    assert CompiledMLPBlocks.install(model, enabled=True) == 0
    assert host.mlp is inner
    assert host.mlp.shared_expert is shared_expert
    assert CompiledMLPBlocks.install(model, enabled=True) == 0

    actual = host.mlp(x)
    mx.eval(actual)

    assert mx.array_equal(actual, expected).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("cls", [Qwen3NextSparseMoeBlock, Qwen3_5MoeSparseMoeBlock])
def test_mixed_model_compiles_only_dense_blocks(cls):
    sparse = _moe_block(cls)
    dense = _dense_mlp()
    host = _Host(sparse)
    host.dense = dense

    assert CompiledMLPBlocks.install(host, enabled=True) == 1
    assert host.mlp is sparse
    assert not isinstance(host.mlp.shared_expert, CompiledMLPBlock)
    assert isinstance(host.dense, CompiledMLPBlock)
    assert host.dense.inner is dense
    assert CompiledMLPBlocks.install(host, enabled=True) == 0


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_batched_decode_prefill_and_target_verify_stay_eager(monkeypatch):
    inner = _dense_mlp(Qwen3_5MLP)
    host = _Host(inner)
    assert CompiledMLPBlocks.install(host, enabled=True) == 1
    assert isinstance(host.mlp, CompiledTargetVerifyMLPBlock)
    compiled_calls = 0
    original_dispatch = host.mlp.dispatch_compiled

    def record(x):
        nonlocal compiled_calls
        compiled_calls += 1
        return original_dispatch(x)

    monkeypatch.setattr(host.mlp, "dispatch_compiled", record)
    decode = mx.random.normal((1, 1, 64)).astype(mx.float16)
    batched_decode = mx.random.normal((4, 1, 64)).astype(mx.float16)
    prefill = mx.random.normal((1, 5, 64)).astype(mx.float16)

    host.mlp(decode)
    host.mlp(batched_decode)
    host.mlp(decode, target_verify=True)
    host.mlp(prefill)

    assert compiled_calls == 1


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_exact_verifier_unwraps_compiled_vlm_block(monkeypatch):
    from mlx_vlm.models.qwen3_5 import language as q35

    verifier = q35.LanguageModel.__call__.__globals__.get(
        "_EXACT_SPECULATIVE_VERIFIER"
    )
    if verifier is None:
        pytest.skip("mlx-vlm exact verifier not available")

    inner = _dense_mlp(Qwen3_5MLP)
    host = _Host(inner)
    x = mx.random.normal((1, 1, 64)).astype(mx.float16)
    expected = verifier._feed_forward(inner, x)

    assert CompiledMLPBlocks.install(host, enabled=True) == 1

    def fail_compiled_dispatch(_value):
        raise AssertionError("exact verification must bypass compiled dispatch")

    monkeypatch.setattr(host.mlp, "dispatch_compiled", fail_compiled_dispatch)
    actual = verifier._feed_forward(host.mlp, x)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()
