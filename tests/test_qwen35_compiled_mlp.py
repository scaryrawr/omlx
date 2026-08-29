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


def _moe_block():
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
    block = Qwen3NextSparseMoeBlock(args)
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


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_dispatch_defaults_on_with_opt_out(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_COMPILED_MLP", raising=False)
    assert CompiledMLPBlocks.enabled()
    monkeypatch.setenv("OMLX_QWEN35_COMPILED_MLP", "0")
    assert not CompiledMLPBlocks.enabled()


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
def test_compiled_quantized_moe_output_is_bit_exact():
    inner = _moe_block()
    host = _Host(inner)
    x = mx.random.normal((1, 3, 64)).astype(mx.float16)
    expected = inner(x)
    assert CompiledMLPBlocks.install(host, enabled=True) == 1

    actual = host.mlp(x)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


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
