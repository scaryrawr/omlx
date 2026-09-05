# SPDX-License-Identifier: Apache-2.0
"""Parity coverage for Qwen3.5 verifier and Qwen4 decode prework."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import qwen35_gdn_prework as prework_mod
from omlx.patches.qwen35_gdn_prework import (
    gdn_prework_fused,
    qwen4_decode_norm_gate_fused,
    qwen4_decode_prework_fused,
)

HK, HV, DK, DV = 16, 48, 128, 128
C = 2 * HK * DK + HV * DV
KEY_DIM = HK * DK


class _Cache:
    def __init__(self, conv_state, recurrent_state):
        self.values = [conv_state, recurrent_state]
        self.lengths = None
        self.advance_calls = 0

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def advance(self, count):
        self.advance_calls += count


def _assert_nested_equal(expected, actual):
    if isinstance(expected, mx.array):
        assert mx.array_equal(expected, actual).item()
    elif isinstance(expected, (tuple, list)):
        assert type(expected) is type(actual)
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _assert_nested_equal(expected_item, actual_item)
    else:
        assert expected == actual


def _composed(qkv, conv_state, conv1d):
    batch, seq, _ = qkv.shape
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    new_state = mx.contiguous(conv_input[:, -3:, :])
    co = nn.silu(conv1d(conv_input))
    q, k, v = mx.split(co, [KEY_DIM, 2 * KEY_DIM], -1)
    q = q.reshape(batch, seq, HK, DK)
    k = k.reshape(batch, seq, HK, DK)
    v = v.reshape(batch, seq, HV, DV)
    inv = DK**-0.5
    return (
        (inv**2) * mx.fast.rms_norm(q, None, 1e-6),
        inv * mx.fast.rms_norm(k, None, 1e-6),
        v,
        new_state,
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("seq", [2, 3, 4, 5, 7, 9])
def test_fused_prework_bit_exact(seq):
    mx.random.seed(11)
    conv_w = (mx.random.normal((C, 4, 1)) * 0.2).astype(mx.bfloat16)
    conv1d = nn.Conv1d(C, C, kernel_size=4, groups=C, bias=False)
    conv1d.weight = conv_w
    qkv = (mx.random.normal((1, seq, C)) * 0.5).astype(mx.bfloat16)
    state = (mx.random.normal((1, 3, C)) * 0.5).astype(mx.bfloat16)
    inv = DK**-0.5

    expected = _composed(qkv, state, conv1d)
    actual = gdn_prework_fused(
        qkv,
        state,
        conv_w,
        mx.array(inv * inv, dtype=mx.bfloat16),
        mx.array(inv, dtype=mx.bfloat16),
        HK,
        HV,
        DK,
        DV,
    )
    mx.eval(*expected, *actual)
    for expected_value, actual_value in zip(expected, actual, strict=True):
        assert mx.array_equal(expected_value, actual_value).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_qwen4_decode_prework_is_bit_exact_including_fp32_gate():
    from mlx_vlm.models.qwen3_5.gated_delta import _compute_g_beta

    mx.random.seed(29)
    conv_w = (mx.random.normal((C, 4, 1)) * 0.2).astype(mx.bfloat16)
    conv1d = nn.Conv1d(C, C, kernel_size=4, groups=C, bias=False)
    conv1d.weight = conv_w
    qkv = (mx.random.normal((1, 1, C)) * 0.5).astype(mx.bfloat16)
    state = (mx.random.normal((1, 3, C)) * 0.5).astype(mx.bfloat16)
    a = (mx.random.normal((1, 1, HV)) * 0.2).astype(mx.bfloat16)
    b = (mx.random.normal((1, 1, HV)) * 0.2).astype(mx.bfloat16)
    a_log = (mx.random.normal((HV,)) * 0.2).astype(mx.bfloat16)
    dt_bias = (mx.random.normal((HV,)) * 0.2).astype(mx.bfloat16)
    inv = DK**-0.5
    expected = (*_composed(qkv, state, conv1d), *_compute_g_beta(a_log, a, b, dt_bias))
    actual = qwen4_decode_prework_fused(
        qkv,
        state,
        conv_w,
        mx.array(inv * inv, dtype=mx.bfloat16),
        mx.array(inv, dtype=mx.bfloat16),
        b,
        a,
        a_log,
        dt_bias,
        HK,
        HV,
        DK,
        DV,
    )
    mx.eval(*expected, *actual)
    for expected_value, actual_value in zip(expected, actual, strict=True):
        assert mx.array_equal(expected_value, actual_value).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_qwen4_decode_norm_gate_is_bit_exact():
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNormGated

    mx.random.seed(31)
    y = (mx.random.normal((1, 1, HV, DV)) * 0.25).astype(mx.bfloat16)
    z = (mx.random.normal((1, 1, HV, DV)) * 0.25).astype(mx.bfloat16)
    norm = Qwen4ExpRMSNormGated(DV, eps=1e-6, activation="sigmoid")
    norm.weight = (1 + mx.random.normal((DV,)) * 0.1).astype(mx.bfloat16)
    expected = norm(y, z).reshape(1, 1, HV * DV)
    actual = qwen4_decode_norm_gate_fused(y, z, norm.weight, hv=HV, dv=DV, eps=norm.eps)
    mx.eval(expected, actual)
    assert mx.array_equal(expected, actual).item()


class _Qwen4Cache:
    def __init__(self, conv_state, recurrent_state):
        self.values = [conv_state, recurrent_state]
        self.lengths = None
        self.left_padding = None

    def __getitem__(self, index):
        return self.values[index]


def test_qwen4_decode_dynamic_gate_is_strictly_b1_t1_nonverify(monkeypatch):
    monkeypatch.setattr(prework_mod, "_qwen4_decode_static_eligible", lambda _: True)
    cache = _Qwen4Cache(
        mx.zeros((1, 3, C), dtype=mx.bfloat16),
        mx.zeros((1, HV, DV, DK), dtype=mx.float32),
    )
    inputs = mx.zeros((1, 1, 2560), dtype=mx.bfloat16)

    def eligible(**updates):
        args = dict(
            module=object(),
            inputs=inputs,
            mask=None,
            cache=cache,
            gdn_sink=None,
            target_verify=False,
        )
        args.update(updates)
        return prework_mod._qwen4_decode_dynamic_eligible(**args)

    assert eligible()
    assert not eligible(inputs=mx.zeros((2, 1, 2560), dtype=mx.bfloat16))
    assert not eligible(inputs=mx.zeros((1, 2, 2560), dtype=mx.bfloat16))
    assert not eligible(inputs=mx.zeros((1, 1, 2560), dtype=mx.float16))
    assert not eligible(mask="causal")
    assert not eligible(gdn_sink=[])
    assert not eligible(target_verify=True)
    cache.lengths = mx.array([1])
    assert not eligible()
    cache.lengths = None
    cache.left_padding = mx.array([0])
    assert not eligible()


def _fake_quantized_linear(input_dims, output_dims, bits, group_size):
    linear = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(linear)
    linear.bits, linear.group_size, linear.mode = bits, group_size, "affine"
    linear.weight = mx.zeros((output_dims, input_dims * bits // 32), dtype=mx.uint32)
    linear.scales = mx.zeros((output_dims, input_dims // group_size), dtype=mx.bfloat16)
    linear.biases = mx.zeros_like(linear.scales)
    return linear


@pytest.mark.parametrize(
    "signatures",
    [
        ((6, 64), (6, 64), (6, 64), (6, 64)),
        ((4, 64), (5, 128), (5, 128), (5, 128)),
        ((5, 64), (6, 64), (6, 64), (6, 64)),
    ],
)
def test_qwen4_decode_static_gate_accepts_canonical_oqe_allocations(signatures):
    module_type = type("Qwen4ExpGatedDeltaNet", (), {})
    module_type.__module__ = "mlx_vlm.models.qwen4_exp.language"
    module = module_type()
    module.training = False
    module.num_k_heads, module.num_v_heads = HK, HV
    module.head_k_dim, module.head_v_dim, module.conv_kernel_size = DK, DV, 4
    module.conv1d = SimpleNamespace(
        weight=mx.zeros((C, 4, 1), dtype=mx.bfloat16), bias=None
    )
    module.norm = SimpleNamespace(
        activation="sigmoid", weight=mx.ones((DV,), dtype=mx.bfloat16)
    )
    module.A_log = mx.zeros((HV,), dtype=mx.bfloat16)
    module.dt_bias = mx.zeros((HV,), dtype=mx.bfloat16)
    rows = (C, HV * DV, HV, HV)
    (
        module.in_proj_qkv,
        module.in_proj_z,
        module.in_proj_b,
        module.in_proj_a,
    ) = [
        _fake_quantized_linear(2560, output, bits, group)
        for output, (bits, group) in zip(rows, signatures, strict=True)
    ]
    module.out_proj = _fake_quantized_linear(6144, 2560, 5, 128)
    assert prework_mod._qwen4_decode_static_eligible(module)
    module.in_proj_z.group_size = 32
    assert not prework_mod._qwen4_decode_static_eligible(module)


def test_verify_prework_gate_excludes_qwen4_and_masks(monkeypatch):
    q35 = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    module_type = type("Qwen4ExpGatedDeltaNet", (q35.Qwen3_5GatedDeltaNet,), {})
    layer = module_type.__new__(module_type)
    inputs = mx.zeros((1, 3, 8), dtype=mx.bfloat16)
    cache = _Cache(
        mx.zeros((1, 3, C), dtype=mx.bfloat16),
        mx.zeros((1, HV, DV, DK), dtype=mx.float32),
    )

    assert not prework_mod._verify_prework_eligible(q35, layer, inputs, None, cache, [])
    monkeypatch.setattr(q35, "Qwen3_5GatedDeltaNet", module_type)
    assert not prework_mod._verify_prework_eligible(
        q35, layer, inputs, "causal", cache, []
    )


def test_current_verifier_hook_defaults_on_and_can_be_disabled(monkeypatch):
    q35 = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    verifier = q35.LanguageModel.__call__.__globals__["_EXACT_SPECULATIVE_VERIFIER"]
    verifier_class = type(verifier)
    original = verifier_class._gated_delta

    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    monkeypatch.setattr(verifier_class, "_gated_delta", original)
    monkeypatch.setattr(prework_mod.mx.metal, "is_available", lambda: True)
    monkeypatch.setenv("OMLX_QWEN35_VERIFY_PREWORK", "0")
    assert not prework_mod.apply_qwen35_gdn_prework_patch()

    monkeypatch.delenv("OMLX_QWEN35_VERIFY_PREWORK", raising=False)
    assert prework_mod.apply_qwen35_gdn_prework_patch()
    assert verifier_class._gated_delta is not original

    calls = []

    def fallback(*args):
        calls.append(args)
        return "original"

    monkeypatch.setattr(verifier_class, "_gated_delta", fallback)
    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    assert prework_mod.apply_qwen35_gdn_prework_patch()
    monkeypatch.setenv("OMLX_QWEN35_VERIFY_PREWORK", "0")
    assert verifier._gated_delta(None, None, None, None, None) == "original"
    assert len(calls) == 1


def test_current_verifier_hook_survives_wrapped_language_model_call(monkeypatch):
    q35 = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    from mlx_vlm.models.qwen3_5.speculative_verifier import (
        Qwen3_5BatchInvariantForward,
    )

    original = Qwen3_5BatchInvariantForward._gated_delta
    language_call = q35.LanguageModel.__call__

    def wrapped_call(self, *args, **kwargs):
        return language_call(self, *args, **kwargs)

    monkeypatch.setattr(q35.LanguageModel, "__call__", wrapped_call)
    monkeypatch.setattr(Qwen3_5BatchInvariantForward, "_gated_delta", original)
    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    monkeypatch.setattr(prework_mod.mx.metal, "is_available", lambda: True)
    monkeypatch.delenv("OMLX_QWEN35_VERIFY_PREWORK", raising=False)
    assert "_EXACT_SPECULATIVE_VERIFIER" not in wrapped_call.__globals__
    assert prework_mod.apply_qwen35_gdn_prework_patch()
    patched = Qwen3_5BatchInvariantForward._gated_delta
    assert patched is not original
    assert prework_mod.apply_qwen35_gdn_prework_patch()
    assert Qwen3_5BatchInvariantForward._gated_delta is patched


def _dense_gdn_layer(value_heads):
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

    config = TextConfig(
        model_type="qwen3_5",
        hidden_size=8,
        intermediate_size=16,
        linear_num_value_heads=value_heads,
        linear_num_key_heads=HK,
        linear_key_head_dim=DK,
        linear_value_head_dim=DV,
        linear_conv_kernel_dim=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        rms_norm_eps=1e-6,
        vocab_size=16,
        num_key_value_heads=1,
        max_position_embeddings=32,
    )
    layer = Qwen3_5GatedDeltaNet(config)
    layer.eval()
    for linear in (
        layer.in_proj_qkv,
        layer.in_proj_z,
        layer.in_proj_b,
        layer.in_proj_a,
        layer.out_proj,
    ):
        linear.weight = linear.weight.astype(mx.bfloat16)
    layer.conv1d.weight = layer.conv1d.weight.astype(mx.bfloat16)
    layer.A_log = layer.A_log.astype(mx.bfloat16)
    layer.dt_bias = layer.dt_bias.astype(mx.bfloat16)
    layer.norm.weight = layer.norm.weight.astype(mx.bfloat16)
    return layer


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("value_heads", [32, 48])
@pytest.mark.parametrize("length", [2, 3, 5])
@pytest.mark.parametrize("has_state", [False, True])
def test_current_verifier_eligible_block_and_rollback_tuple_are_exact(
    monkeypatch, value_heads, length, has_state
):
    q35 = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    verifier = q35.LanguageModel.__call__.__globals__["_EXACT_SPECULATIVE_VERIFIER"]
    verifier_class = type(verifier)
    original = verifier_class._gated_delta
    mx.random.seed(31)
    layer = _dense_gdn_layer(value_heads)
    inputs = (mx.random.normal((1, length, 8)) * 0.1).astype(mx.bfloat16)
    conv_dim = 2 * HK * DK + value_heads * DV
    conv_state = (mx.random.normal((1, 3, conv_dim)) * 0.1).astype(mx.bfloat16)
    recurrent_state = (
        mx.random.normal((1, value_heads, DV, DK)) * 0.1 if has_state else None
    )

    expected_cache = _Cache(conv_state, recurrent_state)
    expected_sink = []
    expected = original(verifier, layer, inputs, None, expected_cache, expected_sink)

    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    monkeypatch.setattr(verifier_class, "_gated_delta", original)
    monkeypatch.setenv("OMLX_QWEN35_VERIFY_PREWORK", "1")
    assert prework_mod.apply_qwen35_gdn_prework_patch()
    actual_cache = _Cache(conv_state, recurrent_state)
    actual_sink = []
    actual = verifier._gated_delta(layer, inputs, None, actual_cache, actual_sink)
    mx.eval(
        expected,
        actual,
        expected_cache[0],
        actual_cache[0],
        expected_cache[1],
        actual_cache[1],
    )
    assert mx.array_equal(expected, actual).item()
    assert mx.array_equal(expected_cache[0], actual_cache[0]).item()
    assert mx.array_equal(expected_cache[1], actual_cache[1]).item()
    assert expected_cache.advance_calls == actual_cache.advance_calls == length
    assert len(expected_sink) == len(actual_sink) == 1
    _assert_nested_equal(expected_sink, actual_sink)
