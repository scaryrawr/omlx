# SPDX-License-Identifier: Apache-2.0
"""Tests for compiled Qwen3.5-family decode MLP dispatch."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.qwen3_next import Qwen3NextMLP
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
