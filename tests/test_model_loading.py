# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.model_loading.maybe_load_custom_quantization."""

import json
import os
import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from omlx.utils import model_loading
from omlx.utils.model_loading import (
    materialize_lazy_state,
    maybe_apply_pre_load_patches,
    maybe_load_custom_quantization,
)


def _write_config(tmp_path, body: str) -> str:
    (tmp_path / "config.json").write_text(body)
    return str(tmp_path)


def _write_jang_config(tmp_path, body: str, name: str = "jang_config.json") -> str:
    (tmp_path / name).write_text(body)
    return str(tmp_path)


def _write_mtp_index(tmp_path, has_mtp: bool) -> None:
    """Drop a stub ``model.safetensors.index.json`` next to config.json so
    ``_checkpoint_has_mtp_weights`` resolves deterministically in tests."""
    keys = {"language_model.model.embed_tokens.weight": "model.safetensors"}
    if has_mtp:
        keys["language_model.mtp.fc.weight"] = "model.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata": {}, "weight_map": ' + str(keys).replace("'", '"') + "}"
    )


class TestNoDispatch:
    """Cases where the dispatcher should return None and let the caller
    fall back to the standard mlx-lm/mlx-vlm load path."""

    def test_missing_config_returns_none(self, tmp_path):
        # tmp_path has no config.json
        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) is None

    def test_malformed_config_returns_none(self, tmp_path):
        path = _write_config(tmp_path, "{not valid json")
        assert maybe_load_custom_quantization(path, is_vlm=False) is None

    def test_no_quantization_config_returns_none(self, tmp_path):
        path = _write_config(tmp_path, '{"model_type": "llama"}')
        assert maybe_load_custom_quantization(path, is_vlm=False) is None

    def test_empty_quant_method_returns_none(self, tmp_path):
        path = _write_config(tmp_path, '{"quantization_config": {}}')
        assert maybe_load_custom_quantization(path, is_vlm=False) is None

    def test_unknown_quant_method_returns_none(self, tmp_path):
        # Methods we don't dispatch on (mlx-lm may or may not handle them
        # natively; either way the dispatcher stays out of the way).
        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "awq"}}',
        )
        assert maybe_load_custom_quantization(path, is_vlm=False) is None


def _install_jang_loader_stub(monkeypatch, *, text_load=None, vlm_load=None):
    monkeypatch.setitem(sys.modules, "jang_tools", types.ModuleType("jang_tools"))
    loader_mod = types.ModuleType("jang_tools.loader")
    if text_load is not None:
        loader_mod.load_jang_model = text_load
    if vlm_load is not None:
        loader_mod.load_jang_vlm_model = vlm_load
    monkeypatch.setitem(sys.modules, "jang_tools.loader", loader_mod)


def _install_jangtq_loader_stub(
    monkeypatch,
    *,
    text_load=None,
    vlm_load=None,
    kimi_vlm_load=None,
    streaming_hydrate=None,
):
    monkeypatch.setitem(sys.modules, "jang_tools", types.ModuleType("jang_tools"))
    if text_load is not None or streaming_hydrate is not None:
        text_mod = types.ModuleType("jang_tools.load_jangtq")
        if text_load is not None:
            text_mod.load_jangtq_model = text_load
        if streaming_hydrate is not None:
            text_mod._hydrate_dsv4_jangtq_streaming = streaming_hydrate
        monkeypatch.setitem(sys.modules, "jang_tools.load_jangtq", text_mod)
    if vlm_load is not None:
        vlm_mod = types.ModuleType("jang_tools.load_jangtq_vlm")
        vlm_mod.load_jangtq_vlm_model = vlm_load
        monkeypatch.setitem(sys.modules, "jang_tools.load_jangtq_vlm", vlm_mod)
    if kimi_vlm_load is not None:
        kimi_mod = types.ModuleType("jang_tools.load_jangtq_kimi_vlm")
        kimi_mod.load_jangtq_kimi_vlm_model = kimi_vlm_load
        monkeypatch.setitem(sys.modules, "jang_tools.load_jangtq_kimi_vlm", kimi_mod)


def _install_dsv4_scoped_patch_stubs(
    monkeypatch,
    *,
    load_config,
    tokenizer_from_pretrained,
):
    mlx_lm_mod = types.ModuleType("mlx_lm")
    mlx_lm_mod.__path__ = []
    utils_mod = types.ModuleType("mlx_lm.utils")
    utils_mod.load_config = load_config
    mlx_lm_mod.utils = utils_mod
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils_mod)

    class AutoTokenizer:
        from_pretrained = staticmethod(tokenizer_from_pretrained)

    class PreTrainedTokenizerFast:
        def __init__(self, *, tokenizer_file):
            self.tokenizer_file = tokenizer_file

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = AutoTokenizer
    transformers_mod.PreTrainedTokenizerFast = PreTrainedTokenizerFast
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    return utils_mod, transformers_mod


def _patch_dsv4_runtime_contract_helpers(monkeypatch, calls=None):
    if calls is None:
        calls = []
    monkeypatch.setattr(
        model_loading,
        "_validate_dsv4_control_tensors",
        lambda model_path: calls.append(("control", model_path)),
    )
    monkeypatch.setattr(
        model_loading,
        "_validate_dsv4_artifact_bit_plan",
        lambda model_path: calls.append(("bitplan", model_path)),
    )
    monkeypatch.setattr(
        model_loading,
        "_configure_dsv4_runtime_defaults",
        lambda: calls.append("defaults"),
    )
    monkeypatch.setattr(
        model_loading,
        "_import_dsv4_mlx_register",
        lambda: calls.append("register"),
    )
    monkeypatch.setattr(
        model_loading,
        "_verify_dsv4_attention_contract",
        lambda: calls.append("attention"),
    )
    monkeypatch.setattr(
        model_loading,
        "_audit_dsv4_switchglu_contract",
        lambda model: calls.append(("switchglu", model)),
    )
    monkeypatch.setattr(
        model_loading,
        "_maybe_install_dsv4_fast_load_patch",
        lambda model_path: calls.append(("fastload", model_path)) or False,
    )
    return calls


def _install_dsv4_attention_stub(monkeypatch, call):
    jang_mod = types.ModuleType("jang_tools")
    jang_mod.__path__ = []
    dsv4_mod = types.ModuleType("jang_tools.dsv4")
    dsv4_mod.__path__ = []
    mlx_model_mod = types.ModuleType("jang_tools.dsv4.mlx_model")

    class DeepseekV4Attention:
        pass

    DeepseekV4Attention.__call__ = call
    mlx_model_mod.DeepseekV4Attention = DeepseekV4Attention
    monkeypatch.setitem(sys.modules, "jang_tools", jang_mod)
    monkeypatch.setitem(sys.modules, "jang_tools.dsv4", dsv4_mod)
    monkeypatch.setitem(sys.modules, "jang_tools.dsv4.mlx_model", mlx_model_mod)
    return DeepseekV4Attention


def _dsv4_attention_contract_ok(
    self,
    attn_mask=None,
    full_kv=None,
    comp_mask=None,
    selected=None,
    pooled=None,
):
    if attn_mask.shape[-1] > full_kv.shape[2]:
        attn_mask = attn_mask[..., -full_kv.shape[2] :]
    comp_mask = comp_mask & selected
    if pooled.shape[1] > self.indexer.index_topk:
        pass
    return attn_mask, comp_mask


def _dsv4_attention_contract_missing_pool_mask(
    self,
    attn_mask=None,
    full_kv=None,
    pooled=None,
):
    if attn_mask.shape[-1] > full_kv.shape[2]:
        attn_mask = attn_mask[..., -full_kv.shape[2] :]
    if pooled.shape[1] > self.indexer.index_topk:
        pass
    return attn_mask


def test_materialize_lazy_state_collects_custom_object_attributes(monkeypatch):
    """JANG model loaders keep some arrays on plain Python attributes."""
    import mlx.core as mx

    hidden = mx.array([1])
    nested = mx.array([2])

    class CustomModel:
        def __init__(self):
            self.hidden = hidden
            self.container = {"nested": [nested]}

    evaluated = []
    monkeypatch.setattr(model_loading, "tree_flatten", lambda _model: [])
    monkeypatch.setattr(mx, "eval", lambda *arrays: evaluated.extend(arrays))

    materialize_lazy_state(CustomModel())

    assert hidden in evaluated
    assert nested in evaluated


def test_materialize_lazy_state_collects_module_api_arrays(monkeypatch):
    import mlx.core as mx

    param = mx.array([1])
    leaf_param = mx.array([2])

    class Leaf:
        def parameters(self):
            return {"leaf": leaf_param}

    class CustomModel:
        def parameters(self):
            return {"param": param}

        def leaf_modules(self):
            return {"leaf": Leaf()}

    evaluated = []
    monkeypatch.setattr(model_loading, "tree_flatten", lambda _model: [])
    monkeypatch.setattr(mx, "eval", lambda *arrays: evaluated.extend(arrays))

    materialize_lazy_state(CustomModel())

    assert param in evaluated
    assert leaf_param in evaluated


def _install_dsv4_switchglu_stubs(monkeypatch):
    tq_kernel_mod = types.ModuleType("jang_tools.turboquant.tq_kernel")
    switch_layers_mod = types.ModuleType("mlx_lm.models.switch_layers")

    class TurboQuantSwitchLinear:
        pass

    class SwitchGLU:
        def __init__(self, *, gate_proj=None, up_proj=None, down_proj=None, limit=10):
            self.gate_proj = gate_proj
            self.up_proj = up_proj
            self.down_proj = down_proj
            self.activation = types.SimpleNamespace(swiglu_limit=limit)

    tq_kernel_mod.TurboQuantSwitchLinear = TurboQuantSwitchLinear
    switch_layers_mod.SwitchGLU = SwitchGLU

    jang_mod = types.ModuleType("jang_tools")
    jang_mod.__path__ = []
    turboquant_mod = types.ModuleType("jang_tools.turboquant")
    turboquant_mod.__path__ = []
    mlx_lm_mod = types.ModuleType("mlx_lm")
    mlx_lm_mod.__path__ = []
    models_mod = types.ModuleType("mlx_lm.models")
    models_mod.__path__ = []
    monkeypatch.setitem(sys.modules, "jang_tools", jang_mod)
    monkeypatch.setitem(sys.modules, "jang_tools.turboquant", turboquant_mod)
    monkeypatch.setitem(sys.modules, "jang_tools.turboquant.tq_kernel", tq_kernel_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.switch_layers", switch_layers_mod)
    return TurboQuantSwitchLinear, SwitchGLU


class _NamedModuleModel:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        return iter(self._modules)


class TestDSV4RuntimeContracts:
    def test_runtime_defaults_set_values_and_preserve_cache_default(
        self, monkeypatch
    ):
        for name in (
            "DSV4_LONG_CTX",
            "DSV4_POOL_QUANT",
            "JANGTQ_WIRED_LIMIT_GB",
            "DSV4_MLX_CACHE_LIMIT_GB",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(model_loading.sys, "platform", "darwin")
        monkeypatch.setattr(
            model_loading, "_dsv4_total_memory_gb_decimal", lambda: 128.0
        )
        set_cache_limit = MagicMock()
        monkeypatch.setattr(
            model_loading.mx, "set_cache_limit", set_cache_limit, raising=False
        )

        model_loading._configure_dsv4_runtime_defaults()

        assert os.environ["DSV4_LONG_CTX"] == "1"
        assert os.environ["DSV4_POOL_QUANT"] == "1"
        assert os.environ["JANGTQ_WIRED_LIMIT_GB"] == "66"
        assert os.environ["DSV4_MLX_CACHE_LIMIT_GB"] == "8"
        set_cache_limit.assert_called_once_with(8 * 1024**3)

    def test_runtime_defaults_preserve_explicit_overrides(self, monkeypatch):
        monkeypatch.setenv("DSV4_LONG_CTX", "0")
        monkeypatch.setenv("DSV4_POOL_QUANT", "0")
        monkeypatch.setenv("JANGTQ_WIRED_LIMIT_GB", "99")
        monkeypatch.setenv("DSV4_MLX_CACHE_LIMIT_GB", "3.5")
        monkeypatch.setattr(model_loading.sys, "platform", "darwin")
        monkeypatch.setattr(
            model_loading, "_dsv4_total_memory_gb_decimal", lambda: 128.0
        )
        set_cache_limit = MagicMock()
        monkeypatch.setattr(
            model_loading.mx, "set_cache_limit", set_cache_limit, raising=False
        )

        model_loading._configure_dsv4_runtime_defaults()

        assert os.environ["DSV4_LONG_CTX"] == "0"
        assert os.environ["DSV4_POOL_QUANT"] == "0"
        assert os.environ["JANGTQ_WIRED_LIMIT_GB"] == "99"
        assert os.environ["DSV4_MLX_CACHE_LIMIT_GB"] == "3.5"
        set_cache_limit.assert_called_once_with(int(3.5 * 1024**3))

    def test_eager_dsv4_registration_imports_module(self, monkeypatch):
        jang_mod = types.ModuleType("jang_tools")
        jang_mod.__path__ = []
        dsv4_mod = types.ModuleType("jang_tools.dsv4")
        dsv4_mod.__path__ = []
        register_mod = types.ModuleType("jang_tools.dsv4.mlx_register")
        monkeypatch.setitem(sys.modules, "jang_tools", jang_mod)
        monkeypatch.setitem(sys.modules, "jang_tools.dsv4", dsv4_mod)
        monkeypatch.setitem(sys.modules, "jang_tools.dsv4.mlx_register", register_mod)

        model_loading._import_dsv4_mlx_register()

    def test_eager_dsv4_registration_import_error_is_clear(self, monkeypatch):
        def fail_import(name, package=None):
            raise ImportError(name)

        monkeypatch.setattr(model_loading.importlib, "import_module", fail_import)

        with pytest.raises(ImportError, match="mlx_register"):
            model_loading._import_dsv4_mlx_register()

    def test_attention_contract_accepts_required_markers(self, monkeypatch):
        _install_dsv4_attention_stub(monkeypatch, _dsv4_attention_contract_ok)

        model_loading._verify_dsv4_attention_contract()

    def test_attention_contract_rejects_missing_markers(self, monkeypatch):
        _install_dsv4_attention_stub(
            monkeypatch, _dsv4_attention_contract_missing_pool_mask
        )

        with pytest.raises(RuntimeError, match="per_query_compressed_pool_mask"):
            model_loading._verify_dsv4_attention_contract()

    def test_switchglu_contract_requires_hydrated_tq_modules(self, monkeypatch):
        _, switch_glu_cls = _install_dsv4_switchglu_stubs(monkeypatch)
        model = _NamedModuleModel([("mlp", switch_glu_cls())])

        with pytest.raises(RuntimeError, match="zero TurboQuant SwitchGLU"):
            model_loading._audit_dsv4_switchglu_contract(model)

    def test_switchglu_contract_requires_limit_10(self, monkeypatch):
        tq_cls, switch_glu_cls = _install_dsv4_switchglu_stubs(monkeypatch)
        tq = tq_cls()
        model = _NamedModuleModel(
            [
                (
                    "mlp",
                    switch_glu_cls(
                        gate_proj=tq,
                        up_proj=tq,
                        down_proj=tq,
                        limit=9,
                    ),
                )
            ]
        )

        with pytest.raises(RuntimeError, match="swiglu_limit=10"):
            model_loading._audit_dsv4_switchglu_contract(model)

    def test_switchglu_contract_accepts_limit_10(self, monkeypatch):
        tq_cls, switch_glu_cls = _install_dsv4_switchglu_stubs(monkeypatch)
        tq = tq_cls()
        model = _NamedModuleModel(
            [
                (
                    "mlp",
                    switch_glu_cls(
                        gate_proj=tq,
                        up_proj=tq,
                        down_proj=tq,
                        limit=10,
                    ),
                )
            ]
        )

        model_loading._audit_dsv4_switchglu_contract(model)

    def test_fast_load_helper_leaves_streaming_hook_unpatched_and_writes_nothing(
        self, tmp_path, monkeypatch, caplog
    ):
        def streaming_hydrate(*args, **kwargs):
            raise AssertionError("helper should inspect, not call, the hydrate hook")

        _install_jangtq_loader_stub(
            monkeypatch,
            streaming_hydrate=streaming_hydrate,
        )
        monkeypatch.setattr(model_loading, "_DSV4_FAST_LOAD_NOTICE_LOGGED", False)
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": {
                        "layers.0.mlp.switch_mlp.gate_proj.tq_packed": (
                            "model-00001-of-00001.safetensors"
                        )
                    },
                }
            )
        )
        before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

        caplog.set_level("INFO", logger=model_loading.__name__)

        assert model_loading._maybe_install_dsv4_fast_load_patch(tmp_path) is False
        assert model_loading._maybe_install_dsv4_fast_load_patch(tmp_path) is False

        load_jangtq = sys.modules["jang_tools.load_jangtq"]
        assert load_jangtq._hydrate_dsv4_jangtq_streaming is streaming_hydrate
        after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
        assert after == before
        assert "sidecar fast-load patch not installed" in caplog.text

    def test_fast_load_helper_env_gate_is_noop_before_import(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("JANGTQ_DISABLE_DSV4_FAST_LOAD", "1")
        monkeypatch.setenv("HOME", str(tmp_path))

        def fail_import(name, package=None):
            raise AssertionError(f"unexpected import: {name}")

        monkeypatch.setattr(model_loading.importlib, "import_module", fail_import)
        before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

        assert model_loading._maybe_install_dsv4_fast_load_patch(tmp_path) is False
        assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


class TestJANGDispatch:
    """Cases where a JANG sidecar selects jang_tools before config fallback."""

    def test_jang_text_dispatch_without_config(self, tmp_path, monkeypatch):
        def fake_load(model_path):
            assert str(model_path) == str(tmp_path)
            return "JANG_MODEL", "JANG_TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        path = _write_jang_config(
            tmp_path,
            '{"format": "jang", "format_version": "2.0"}',
        )

        assert maybe_load_custom_quantization(path, is_vlm=False) == (
            "JANG_MODEL",
            "JANG_TOKENIZER",
        )

    def test_jang_text_dispatch_without_quant_method(self, tmp_path, monkeypatch):
        def fake_load(model_path):
            return "MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(tmp_path, '{"model_type": "llama"}')
        _write_jang_config(tmp_path, '{"format": "jang", "format_version": "2.0"}')

        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) == (
            "MODEL",
            "TOKENIZER",
        )

    def test_jang_legacy_sidecar_name_dispatches(self, tmp_path, monkeypatch):
        def fake_load(model_path):
            return "LEGACY", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        path = _write_jang_config(
            tmp_path,
            '{"format": "jjqf", "format_version": "1.1"}',
            name="jjqf_config.json",
        )

        assert maybe_load_custom_quantization(path, is_vlm=False) == (
            "LEGACY",
            "TOKENIZER",
        )

    def test_jang_vlm_dispatches_to_vlm_loader(self, tmp_path, monkeypatch):
        def fake_load(model_path):
            return "VLM_MODEL", "PROCESSOR"

        _install_jang_loader_stub(monkeypatch, vlm_load=fake_load)
        path = _write_jang_config(
            tmp_path,
            '{"format": "jang", "architecture": {"has_vision": true}}',
        )

        assert maybe_load_custom_quantization(path, is_vlm=True) == (
            "VLM_MODEL",
            "PROCESSOR",
        )

    def test_jang_vlm_rejects_text_loader_path(self, tmp_path):
        path = _write_jang_config(
            tmp_path,
            '{"format": "jang", "architecture": {"has_vision": true}}',
        )

        with pytest.raises(ValueError, match="VLM artifact as text-only"):
            maybe_load_custom_quantization(path, is_vlm=False)

    def test_jang_text_rejects_vlm_loader_path(self, tmp_path):
        path = _write_jang_config(tmp_path, '{"format": "jang"}')

        with pytest.raises(ValueError, match="text-only model"):
            maybe_load_custom_quantization(path, is_vlm=True)

    def test_jangtq_text_dispatches_to_jangtq_loader(self, tmp_path, monkeypatch):
        def fake_load(model_path):
            return "JANGTQ_MODEL", "TOKENIZER"

        _install_jangtq_loader_stub(monkeypatch, text_load=fake_load)
        path = _write_jang_config(
            tmp_path,
            '{"format": "jang", "quantization": {"mode": "affine+mxtq"}}',
        )

        assert maybe_load_custom_quantization(path, is_vlm=False) == (
            "JANGTQ_MODEL",
            "TOKENIZER",
        )

    def test_jang_profile_dispatches_basic_loader_not_jangtq(
        self, tmp_path, monkeypatch
    ):
        def basic_load(model_path):
            assert model_path == tmp_path
            return "JANG_PROFILE_MODEL", "TOKENIZER"

        def jangtq_load(model_path):
            raise AssertionError("JANGTQ loader should not handle plain JANG profiles")

        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        path = _write_jang_config(
            tmp_path,
            '{"format": "jang", "profile": "JANG_3M"}',
        )

        assert maybe_load_custom_quantization(path, is_vlm=False) == (
            "JANG_PROFILE_MODEL",
            "TOKENIZER",
        )

    @pytest.mark.parametrize(
        "body",
        [
            '{"format": "jang", "quantization": {"profile": "JANGTQ2"}}',
            '{"format": "jang", "quantization": {"profile": "JANGTQ_2L"}}',
        ],
    )
    def test_jangtq_profile_variants_dispatch_jangtq_loader(
        self, body, tmp_path, monkeypatch
    ):
        def basic_load(model_path):
            raise AssertionError("basic JANG loader should not handle JANGTQ profiles")

        def jangtq_load(model_path):
            assert model_path == tmp_path
            return "JANGTQ_PROFILE_MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        path = _write_jang_config(tmp_path, body)

        assert maybe_load_custom_quantization(path, is_vlm=False) == (
            "JANGTQ_PROFILE_MODEL",
            "TOKENIZER",
        )

    @pytest.mark.parametrize(
        "body",
        [
            '{"format": "MXTQ"}',
            '{"format": "jang", "weight_format": "mxtq"}',
        ],
    )
    def test_mxtq_markers_dispatch_jangtq_loader(
        self, body, tmp_path, monkeypatch
    ):
        def basic_load(model_path):
            raise AssertionError("basic JANG loader should not handle MXTQ markers")

        def jangtq_load(model_path):
            return "MXTQ_MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        path = _write_jang_config(tmp_path, body)

        assert maybe_load_custom_quantization(path, is_vlm=False) == (
            "MXTQ_MODEL",
            "TOKENIZER",
        )

    @pytest.mark.parametrize("format_value", ["mxfp4", "MXFP8"])
    def test_mxfp_formats_dispatch_basic_jang_loader(
        self, format_value, tmp_path, monkeypatch
    ):
        def basic_load(model_path):
            assert model_path == tmp_path
            return "MXFP_MODEL", "TOKENIZER"

        def jangtq_load(model_path):
            raise AssertionError("JANGTQ loader should not handle MXFP formats")

        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        path = _write_jang_config(tmp_path, json.dumps({"format": format_value}))

        assert maybe_load_custom_quantization(path, is_vlm=False) == (
            "MXFP_MODEL",
            "TOKENIZER",
        )

    @pytest.mark.parametrize("artifact", ["runtime_sidecar", "index_tq_packed"])
    def test_tq_artifact_markers_dispatch_jangtq_loader(
        self, artifact, tmp_path, monkeypatch
    ):
        def basic_load(model_path):
            raise AssertionError("basic JANG loader should not handle tq artifacts")

        def jangtq_load(model_path):
            assert model_path == tmp_path
            return "TQ_ARTIFACT_MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        path = _write_jang_config(tmp_path, '{"format": "jang"}')
        if artifact == "runtime_sidecar":
            (tmp_path / "jangtq_runtime.safetensors").write_text("")
        else:
            (tmp_path / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {
                            "model.layers.0.self_attn.q_proj.weight": (
                                "model-00001-of-00001.tq_packed"
                            )
                        },
                    }
                )
            )

        assert maybe_load_custom_quantization(path, is_vlm=False) == (
            "TQ_ARTIFACT_MODEL",
            "TOKENIZER",
        )

    def test_dsv4_jangtq_shim_repairs_target_and_restores_after_success(
        self, tmp_path, monkeypatch
    ):
        dsv4_calls = _patch_dsv4_runtime_contract_helpers(monkeypatch)
        base_config = {
            "model_type": "deepseek_v4",
            "compress_ratios": [0, 4, 128, 0],
            "hidden_size": 4096,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "compress_rope_theta": 160000,
            "rope_scaling": None,
        }

        def original_load_config(model_path):
            return dict(base_config)

        def original_tokenizer_from_pretrained(model_path, *args, **kwargs):
            raise ValueError("Unrecognized model type deepseek_v4")

        utils_mod, transformers_mod = _install_dsv4_scoped_patch_stubs(
            monkeypatch,
            load_config=original_load_config,
            tokenizer_from_pretrained=original_tokenizer_from_pretrained,
        )

        def fake_load(model_path):
            assert utils_mod.load_config is not original_load_config
            assert (
                transformers_mod.AutoTokenizer.from_pretrained
                is not original_tokenizer_from_pretrained
            )

            repaired = utils_mod.load_config(model_path)
            assert (
                repaired["rope_scaling"]
                == model_loading._DSV4_FLASH_REQUIRED_ROPE_SCALING
            )
            assert base_config["rope_scaling"] is None

            other = utils_mod.load_config(tmp_path / "other-model")
            assert other["rope_scaling"] is None

            with pytest.raises(ValueError, match="deepseek_v4"):
                transformers_mod.AutoTokenizer.from_pretrained(tmp_path / "other-model")

            tokenizer = transformers_mod.AutoTokenizer.from_pretrained(model_path)
            assert tokenizer.tokenizer_file == str(tmp_path / "tokenizer.json")
            return "DSV4_JANGTQ_MODEL", tokenizer

        _install_jangtq_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(tmp_path, json.dumps(base_config))
        (tmp_path / "tokenizer.json").write_text("{}")
        _write_jang_config(
            tmp_path,
            '{"format": "jang", "quantization": {"weight_format": "mxtq"}}',
        )

        model, tokenizer = maybe_load_custom_quantization(str(tmp_path), is_vlm=False)

        assert model == "DSV4_JANGTQ_MODEL"
        assert tokenizer.tokenizer_file == str(tmp_path / "tokenizer.json")
        assert dsv4_calls == [
            ("control", tmp_path),
            ("bitplan", tmp_path),
            "defaults",
            "register",
            "attention",
            ("fastload", tmp_path),
            ("switchglu", "DSV4_JANGTQ_MODEL"),
        ]
        assert utils_mod.load_config is original_load_config
        assert (
            transformers_mod.AutoTokenizer.from_pretrained
            is original_tokenizer_from_pretrained
        )

    def test_dsv4_jangtq_shim_restores_after_loader_failure(
        self, tmp_path, monkeypatch
    ):
        dsv4_calls = _patch_dsv4_runtime_contract_helpers(monkeypatch)

        def original_load_config(model_path):
            return {
                "model_type": "deepseek_v4",
                "compress_ratios": [4],
                "compress_rope_theta": 160000,
                "rope_scaling": None,
            }

        def original_tokenizer_from_pretrained(model_path, *args, **kwargs):
            return "ORIGINAL_TOKENIZER"

        utils_mod, transformers_mod = _install_dsv4_scoped_patch_stubs(
            monkeypatch,
            load_config=original_load_config,
            tokenizer_from_pretrained=original_tokenizer_from_pretrained,
        )

        def fake_load(model_path):
            assert utils_mod.load_config is not original_load_config
            assert (
                transformers_mod.AutoTokenizer.from_pretrained
                is not original_tokenizer_from_pretrained
            )
            raise RuntimeError("loader failed")

        _install_jangtq_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(tmp_path, '{"model_type": "deepseek_v4"}')
        (tmp_path / "tokenizer.json").write_text("{}")
        _write_jang_config(
            tmp_path,
            '{"format": "jang", "quantization": {"weight_format": "mxtq"}}',
        )

        with pytest.raises(RuntimeError, match="loader failed"):
            maybe_load_custom_quantization(str(tmp_path), is_vlm=False)

        assert utils_mod.load_config is original_load_config
        assert (
            transformers_mod.AutoTokenizer.from_pretrained
            is original_tokenizer_from_pretrained
        )
        assert dsv4_calls == [
            ("control", tmp_path),
            ("bitplan", tmp_path),
            "defaults",
            "register",
            "attention",
            ("fastload", tmp_path),
        ]

    def test_dsv4_jangtq_wrapper_orders_audits_contracts_and_load(
        self, tmp_path, monkeypatch
    ):
        calls = _patch_dsv4_runtime_contract_helpers(monkeypatch)

        @contextmanager
        def config_context(model_path):
            calls.append(("config_enter", model_path))
            try:
                yield
            finally:
                calls.append("config_exit")

        @contextmanager
        def tokenizer_context(model_path):
            calls.append(("tokenizer_enter", model_path))
            try:
                yield
            finally:
                calls.append("tokenizer_exit")

        monkeypatch.setattr(
            model_loading, "_scoped_dsv4_jangtq_load_config", config_context
        )
        monkeypatch.setattr(
            model_loading,
            "_scoped_dsv4_jangtq_tokenizer_fallback",
            tokenizer_context,
        )
        monkeypatch.setattr(
            model_loading,
            "_install_dsv4_canonical_chat_template",
            lambda tokenizer, model_path: calls.append(("chat", tokenizer, model_path)),
        )

        def fake_load(model_path):
            calls.append(("load_jangtq", model_path))
            return "DSV4_MODEL", "DSV4_TOKENIZER"

        _install_jangtq_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(tmp_path, '{"model_type": "deepseek_v4"}')
        _write_jang_config(
            tmp_path,
            '{"format": "jang", "quantization": {"weight_format": "mxtq"}}',
        )

        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) == (
            "DSV4_MODEL",
            "DSV4_TOKENIZER",
        )
        assert calls == [
            ("control", tmp_path),
            ("bitplan", tmp_path),
            "defaults",
            "register",
            "attention",
            ("fastload", tmp_path),
            ("config_enter", tmp_path),
            ("tokenizer_enter", tmp_path),
            ("load_jangtq", tmp_path),
            "tokenizer_exit",
            "config_exit",
            ("switchglu", "DSV4_MODEL"),
            ("chat", "DSV4_TOKENIZER", tmp_path),
        ]

    def test_dsv4_affine_wrapper_uses_basic_loader_without_jangtq_audits(
        self, tmp_path, monkeypatch
    ):
        calls = _patch_dsv4_runtime_contract_helpers(monkeypatch)

        @contextmanager
        def config_context(model_path):
            calls.append(("config_enter", model_path))
            yield
            calls.append("config_exit")

        @contextmanager
        def tokenizer_context(model_path):
            calls.append(("tokenizer_enter", model_path))
            yield
            calls.append("tokenizer_exit")

        monkeypatch.setattr(
            model_loading, "_scoped_dsv4_jangtq_load_config", config_context
        )
        monkeypatch.setattr(
            model_loading,
            "_scoped_dsv4_jangtq_tokenizer_fallback",
            tokenizer_context,
        )
        monkeypatch.setattr(
            model_loading,
            "_install_dsv4_canonical_chat_template",
            lambda tokenizer, model_path: calls.append(("chat", tokenizer, model_path)),
        )

        def basic_load(model_path):
            calls.append(("load_jang", model_path))
            return "DSV4_AFFINE_MODEL", "DSV4_AFFINE_TOKENIZER"

        def jangtq_load(model_path):
            raise AssertionError("JANGTQ loader must not handle DSV4 affine/MXFP")

        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        _write_config(tmp_path, '{"model_type": "deepseek_v4"}')
        _write_jang_config(tmp_path, '{"format": "mxfp4"}')

        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) == (
            "DSV4_AFFINE_MODEL",
            "DSV4_AFFINE_TOKENIZER",
        )
        assert calls == [
            "defaults",
            "register",
            "attention",
            ("config_enter", tmp_path),
            ("tokenizer_enter", tmp_path),
            ("load_jang", tmp_path),
            "tokenizer_exit",
            "config_exit",
            ("chat", "DSV4_AFFINE_TOKENIZER", tmp_path),
        ]

    def test_non_dsv4_jang_does_not_use_dsv4_wrapper(self, tmp_path, monkeypatch):
        def dsv4_wrapper(model_path, metadata):
            raise AssertionError("non-DSV4 JANG must not use DSV4 wrapper")

        def basic_load(model_path):
            return "GENERIC_JANG_MODEL", "GENERIC_TOKENIZER"

        monkeypatch.setattr(model_loading, "_load_deepseek_v4_jang", dsv4_wrapper)
        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        _write_config(tmp_path, '{"model_type": "llama"}')
        _write_jang_config(tmp_path, '{"format": "mxfp4"}')

        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) == (
            "GENERIC_JANG_MODEL",
            "GENERIC_TOKENIZER",
        )

    def test_jangtq_vlm_dispatches_to_jangtq_vlm_loader(self, tmp_path, monkeypatch):
        def fake_load(model_path):
            return "JANGTQ_VLM", "PROCESSOR"

        _install_jangtq_loader_stub(monkeypatch, vlm_load=fake_load)
        path = _write_jang_config(
            tmp_path,
            '{"format": "jang", "quantization": {"weight_format": "mxtq"}, '
            '"architecture": {"has_vision": true}}',
        )

        assert maybe_load_custom_quantization(path, is_vlm=True) == (
            "JANGTQ_VLM",
            "PROCESSOR",
        )

    def test_kimi_jangtq_vlm_dispatches_to_kimi_loader(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "Kimi-K2.6-VL"
        model_dir.mkdir()

        def generic_load(model_path):
            raise AssertionError("generic JANGTQ VLM loader should not be used")

        def kimi_load(model_path):
            assert model_path == model_dir
            return "KIMI_JANGTQ_VLM", "PROCESSOR"

        _install_jangtq_loader_stub(
            monkeypatch,
            vlm_load=generic_load,
            kimi_vlm_load=kimi_load,
        )
        _write_config(model_dir, '{"model_type": "kimi_vl"}')
        path = _write_jang_config(
            model_dir,
            '{"format": "jang", "quantization": {"weight_format": "mxtq"}}',
        )

        assert maybe_load_custom_quantization(path, is_vlm=True) == (
            "KIMI_JANGTQ_VLM",
            "PROCESSOR",
        )

    def test_unknown_jang_codec_raises_without_basic_fallback(
        self, tmp_path, monkeypatch
    ):
        def fake_load(model_path):
            raise AssertionError("basic JANG loader should not be used")

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        path = _write_jang_config(tmp_path, '{"format": "future-jang-family"}')

        with pytest.raises(ValueError) as exc_info:
            maybe_load_custom_quantization(path, is_vlm=False)

        message = str(exc_info.value)
        assert "Unsupported JANG quantization codec" in message
        assert "future_jang_family" in message
        assert "newer 'jang' package" in message

    def test_jang_missing_dependency_raises_install_hint(self, tmp_path, monkeypatch):
        for name in ["jang_tools", "jang_tools.loader"]:
            monkeypatch.setitem(sys.modules, name, None)
        path = _write_jang_config(tmp_path, '{"format": "jang"}')

        with pytest.raises(ImportError, match="jang"):
            maybe_load_custom_quantization(path, is_vlm=False)

    def test_malformed_jang_sidecar_raises(self, tmp_path):
        path = _write_jang_config(tmp_path, "{not valid json")

        with pytest.raises(ValueError, match="JANG config"):
            maybe_load_custom_quantization(path, is_vlm=False)

    def test_jang_sidecar_takes_precedence_over_paroquant_config(
        self, tmp_path, monkeypatch
    ):
        def fake_load(model_path):
            return "JANG_WINS", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        _write_jang_config(tmp_path, '{"format": "jang"}')

        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) == (
            "JANG_WINS",
            "TOKENIZER",
        )

    def test_jang_metadata_classifies_affine_and_extracts_bits(self, tmp_path):
        _write_config(
            tmp_path,
            '{"model_type": "deepseek_v4", "vision_config": {}}',
        )
        _write_jang_config(
            tmp_path,
            '{"format": "future", "profile": "JANG_2L", "target_bits": 4, '
            '"actual_bits": 4.25, "quantization": {"group_size": 64}}',
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "affine_jang"
        assert metadata.profile == "JANG_2L"
        assert metadata.target_bits == 4
        assert metadata.actual_bits == 4.25
        assert metadata.group_size == 64
        assert metadata.is_vlm is True
        assert metadata.is_deepseek_v4 is True

    def test_jang_metadata_classifies_mxfp_and_kimi_vlm(self, tmp_path):
        model_dir = tmp_path / "Kimi-K2.6-VL"
        model_dir.mkdir()
        _write_config(model_dir, '{"model_type": "kimi_vl"}')
        _write_jang_config(model_dir, '{"format": "mxfp4"}')

        metadata = model_loading.read_jang_metadata(model_dir)

        assert metadata is not None
        assert metadata.codec == "mxfp"
        assert metadata.is_kimi_vlm is True
        assert metadata.is_vlm is True

    def test_jang_metadata_classifies_jangtq_markers_and_plans(self, tmp_path):
        _write_jang_config(
            tmp_path,
            '{"format": "jang", "quantization": {"profile": "JANGTQ_2L", '
            '"mxtq_bits": [2, 4], "routed_expert_bits": {"expert": 2}}}',
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"
        assert metadata.mxtq_bits == [2, 4]
        assert metadata.routed_expert_bit_plans == ({"expert": 2},)

    def test_jang_metadata_classifies_jangtq_from_weight_map(self, tmp_path):
        _write_jang_config(tmp_path, '{"format": "jang"}')
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"metadata": {}, "weight_map": {"model.layers.0": '
            '"model-00001-of-00001.tq_packed"}}'
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"

    def test_jang_metadata_classifies_unknown_sidecar(self, tmp_path):
        _write_jang_config(tmp_path, '{"format": "future-jang-family"}')

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "unknown_jang"


def _install_paroquant_stub(monkeypatch, load_impl):
    """Register a minimal paroquant.inference.backends.mlx.load stub."""
    names = [
        "paroquant",
        "paroquant.inference",
        "paroquant.inference.backends",
        "paroquant.inference.backends.mlx",
    ]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    load_mod = types.ModuleType("paroquant.inference.backends.mlx.load")
    load_mod.load = load_impl
    monkeypatch.setitem(sys.modules, "paroquant.inference.backends.mlx.load", load_mod)


class TestParoquantDispatch:
    """Cases where quant_method == 'paroquant'."""

    def test_paroquant_missing_raises_install_hint(self, tmp_path, monkeypatch):
        # Force the import to fail: shadow the package with a sentinel that
        # blocks submodule resolution. setitem(..., None) makes `import X`
        # raise ImportError in the standard machinery.
        for name in [
            "paroquant",
            "paroquant.inference",
            "paroquant.inference.backends",
            "paroquant.inference.backends.mlx",
            "paroquant.inference.backends.mlx.load",
        ]:
            monkeypatch.setitem(sys.modules, name, None)

        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        with pytest.raises(ImportError, match="paroquant"):
            maybe_load_custom_quantization(path, is_vlm=False)

    def test_paroquant_text_load_returns_tuple(self, tmp_path, monkeypatch):
        def fake_load(model_path, force_text):
            assert force_text is True
            return "MODEL", "PROC", False

        _install_paroquant_stub(monkeypatch, fake_load)

        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        result = maybe_load_custom_quantization(path, is_vlm=False)
        assert result == ("MODEL", "PROC")

    def test_paroquant_vlm_load_returns_tuple(self, tmp_path, monkeypatch):
        def fake_load(model_path, force_text):
            assert force_text is False
            return "VLM_MODEL", "VLM_PROC", True

        _install_paroquant_stub(monkeypatch, fake_load)

        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        result = maybe_load_custom_quantization(path, is_vlm=True)
        assert result == ("VLM_MODEL", "VLM_PROC")

    def test_paroquant_text_only_for_vlm_load_raises(self, tmp_path, monkeypatch):
        # is_vlm=True but the loader returned (..., loaded_is_vlm=False).
        def fake_load(model_path, force_text):
            return "MODEL", "PROC", False

        _install_paroquant_stub(monkeypatch, fake_load)

        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        with pytest.raises(ValueError, match="text-only"):
            maybe_load_custom_quantization(path, is_vlm=True)

    def test_quant_method_case_insensitive(self, tmp_path, monkeypatch):
        # The dispatcher lowercases quant_method, so mixed-case configs
        # (e.g. produced by other tooling) still hit the paroquant path.
        captured = {}

        def fake_load(model_path, force_text):
            captured["called"] = True
            return "M", "P", False

        _install_paroquant_stub(monkeypatch, fake_load)
        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "ParoQuant"}}',
        )
        assert maybe_load_custom_quantization(path, is_vlm=False) == ("M", "P")
        assert captured["called"] is True


class TestLlama4PreLoadDispatch:
    @pytest.mark.parametrize(
        "body",
        [
            '{"model_type": "llama4", "text_config": {}}',
            '{"model_type": "mllama", "text_config": {"model_type": "llama4"}}',
        ],
    )
    def test_llama4_attention_patch_applies(self, tmp_path, monkeypatch, body):
        monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)
        monkeypatch.setitem(
            sys.modules,
            "omlx.patches.mlx_lm_mtp",
            MagicMock(set_mtp_active=MagicMock()),
        )
        apply_mock = MagicMock(return_value=True)
        monkeypatch.setitem(
            sys.modules,
            "omlx.patches.llama4_attention",
            MagicMock(apply_llama4_attention_patch=apply_mock),
        )

        path = _write_config(tmp_path, body)
        maybe_apply_pre_load_patches(path)

        apply_mock.assert_called_once_with()


class TestLoadTextModel:
    def test_forwards_trust_remote_code_to_mlx_lm_load(self, tmp_path, monkeypatch):
        path = _write_config(tmp_path, '{"model_type": "llama"}')
        maybe_apply = MagicMock()
        monkeypatch.setattr(model_loading, "maybe_apply_pre_load_patches", maybe_apply)

        load_mock = MagicMock(return_value=("MODEL", "TOKENIZER"))
        monkeypatch.setitem(sys.modules, "mlx_lm", MagicMock(load=load_mock))

        settings = types.SimpleNamespace(trust_remote_code=True)
        result = model_loading.load_text_model(
            path,
            tokenizer_config={"trust_remote_code": True},
            model_settings=settings,
        )

        assert result == ("MODEL", "TOKENIZER")
        maybe_apply.assert_called_once_with(path, model_settings=settings)
        load_mock.assert_called_once_with(
            path,
            tokenizer_config={"trust_remote_code": True},
            trust_remote_code=True,
        )


class TestVlmMtpPreLoadDispatch:
    """maybe_apply_pre_load_patches must wire the mlx-vlm MTP sanitize
    patch alongside the runtime patch for MTP-capable VLM checkpoints.

    The dense Qwen3.5/3.6 VLM runtime patch does not touch Model.sanitize;
    it relies on apply_mlx_vlm_mtp_patch having installed the mtp.*
    preservation first. If only the runtime patch runs, stock mlx-vlm
    sanitize strips every mtp.* key and the MTP head loads at random
    init (PR #1320).

    MoE VLMs without declared MTP heads still need the sanitize replacement
    so pre-converted switch_mlp weights load (issue #1261); runtime patch
    must not run on that path."""

    def _stub_patches(self, monkeypatch):
        """Replace the patch modules with mocks that record call order.

        Returns the recorded-order list plus the sanitize/runtime/attach
        mocks."""
        calls: list[str] = []
        sanitize_mock = MagicMock(side_effect=lambda: calls.append("sanitize") or True)
        runtime_mock = MagicMock(side_effect=lambda: calls.append("runtime") or True)
        attach_mock = MagicMock(
            side_effect=lambda enabled: calls.append(f"attach={enabled}")
        )
        # Side-step the real mlx-lm load_config monkey-patch.
        monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)
        monkeypatch.setitem(
            sys.modules,
            "omlx.patches.mlx_lm_mtp",
            MagicMock(
                set_mtp_active=MagicMock(),
                apply_mlx_lm_mtp_patch=MagicMock(return_value=True),
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "omlx.patches.mlx_vlm_mtp",
            MagicMock(
                apply_mlx_vlm_mtp_patch=sanitize_mock,
                apply_mlx_vlm_mtp_runtime_patch=runtime_mock,
                set_mtp_attach_enabled=attach_mock,
            ),
        )
        return calls, sanitize_mock, runtime_mock, attach_mock

    def test_sanitize_patch_runs_before_runtime_for_vlm_mtp(
        self, tmp_path, monkeypatch
    ):
        calls, sanitize_mock, runtime_mock, attach_mock = self._stub_patches(
            monkeypatch
        )
        # qwen3_5 (dense VLM) declaring an MTP head under text_config.
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        _write_mtp_index(tmp_path, has_mtp=True)
        settings = types.SimpleNamespace(mtp_enabled=True)

        maybe_apply_pre_load_patches(path, model_settings=settings, for_vlm=True)

        sanitize_mock.assert_called_once()
        runtime_mock.assert_called_once()
        attach_mock.assert_called_once_with(True)
        # Ordering matters: the dense runtime patch assumes sanitize was
        # already installed by apply_mlx_vlm_mtp_patch.
        assert calls == ["attach=True", "sanitize", "runtime"]

    def test_vlm_patches_applied_when_mtp_disabled_for_vlm(self, tmp_path, monkeypatch):
        # Issue #1404: persisted ``mtp.*`` weights must still get a binding
        # site on the LanguageModel tree when entering through VLMBatchedEngine
        # even with mtp_enabled=False. Otherwise mlx-vlm's strict load_weights
        # fails with "parameters not in model" and the engine falls back to
        # LLM, silently dropping vision.
        calls, sanitize_mock, runtime_mock, attach_mock = self._stub_patches(
            monkeypatch
        )
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        _write_mtp_index(tmp_path, has_mtp=True)
        settings = types.SimpleNamespace(mtp_enabled=False)

        maybe_apply_pre_load_patches(path, model_settings=settings, for_vlm=True)

        sanitize_mock.assert_called_once()
        runtime_mock.assert_called_once()
        attach_mock.assert_called_once_with(True)
        assert calls == ["attach=True", "sanitize", "runtime"]

    def test_vlm_attach_disabled_when_config_declares_mtp_but_weights_missing(
        self, tmp_path, monkeypatch
    ):
        # Issue #1426: unsloth Qwen3.6 UD MLX builds declare
        # mtp_num_hidden_layers=1 in config.json but ship no mtp.* weights.
        # Attaching MTPModule there causes mlx-vlm strict load_weights to
        # fail with "Missing N parameters: language_model.mtp.*", the
        # engine falls back to LLM, and vision is silently dropped. The
        # dispatcher must flip set_mtp_attach_enabled(False) so the runtime
        # patch's __init__ wrap skips attachment for this load.
        calls, sanitize_mock, runtime_mock, attach_mock = self._stub_patches(
            monkeypatch
        )
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5_moe", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        _write_mtp_index(tmp_path, has_mtp=False)
        settings = types.SimpleNamespace(mtp_enabled=False)

        maybe_apply_pre_load_patches(path, model_settings=settings, for_vlm=True)

        sanitize_mock.assert_called_once()
        # Runtime patch itself still applies (process-wide class wrap is
        # idempotent and harmless when there are no mtp.* weights to bind);
        # the gate is what prevents MTPModule attachment.
        runtime_mock.assert_called_once()
        attach_mock.assert_called_once_with(False)
        assert calls == ["attach=False", "sanitize", "runtime"]

    def test_vlm_patches_skipped_when_not_for_vlm(self, tmp_path, monkeypatch):
        # BatchedEngine / DFlashEngine / LLM loader paths must NOT touch
        # mlx-vlm classes even when the model declares MTP heads. for_vlm
        # defaults to False so they pass through without invoking mlx-vlm
        # patches.
        calls, sanitize_mock, runtime_mock, attach_mock = self._stub_patches(
            monkeypatch
        )
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        settings = types.SimpleNamespace(mtp_enabled=True)

        maybe_apply_pre_load_patches(path, model_settings=settings)

        sanitize_mock.assert_not_called()
        runtime_mock.assert_not_called()
        assert calls == []

    def test_qwen36_moe_vlm_sanitize_when_no_mtp_heads(self, tmp_path, monkeypatch):
        # mlx-lm Qwen3.6 MoE VLMs without MTP heads still need the mlx-vlm
        # sanitize replacement so pre-converted switch_mlp weights load.
        # Runtime MTP patch must NOT run — there is no mtp.* tree to bind.
        calls, sanitize_mock, runtime_mock, attach_mock = self._stub_patches(
            monkeypatch
        )
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5_moe", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 0}}',
        )
        settings = types.SimpleNamespace(mtp_enabled=False)

        maybe_apply_pre_load_patches(path, model_settings=settings, for_vlm=True)

        sanitize_mock.assert_called_once()
        runtime_mock.assert_not_called()
        assert calls == ["sanitize"]

    def test_qwen36_moe_vlm_sanitize_skipped_without_for_vlm(
        self, tmp_path, monkeypatch
    ):
        calls, sanitize_mock, runtime_mock, attach_mock = self._stub_patches(
            monkeypatch
        )
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5_moe", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 0}}',
        )
        settings = types.SimpleNamespace(mtp_enabled=False)

        maybe_apply_pre_load_patches(path, model_settings=settings)

        sanitize_mock.assert_not_called()
        runtime_mock.assert_not_called()
        assert calls == []


class TestCheckpointHasMtpWeights:
    """``_checkpoint_has_mtp_weights`` decides whether the mlx-vlm runtime
    patch attaches ``MTPModule`` at load time. The scan must:

    - return True when ``model.safetensors.index.json`` declares any key
      under the ``(language_model.|model.)?mtp.`` prefix family;
    - return False when no MTP-prefixed key is found;
    - return False on missing / unreadable inputs (callers treat that as
      "no MTP weights" — the conservative choice).
    """

    def _write_index(self, tmp_path, weight_map: dict) -> None:
        import json as _json

        (tmp_path / "model.safetensors.index.json").write_text(
            _json.dumps({"metadata": {}, "weight_map": weight_map})
        )

    def test_returns_true_when_index_has_language_model_mtp(self, tmp_path):
        self._write_index(
            tmp_path,
            {
                "language_model.model.embed_tokens.weight": "model.safetensors",
                "language_model.mtp.fc.weight": "model.safetensors",
            },
        )
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is True

    def test_returns_true_when_index_has_bare_mtp(self, tmp_path):
        self._write_index(
            tmp_path,
            {"mtp.layers.0.self_attn.q_proj.weight": "model.safetensors"},
        )
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is True

    def test_returns_true_when_index_has_model_language_model_mtp(self, tmp_path):
        # mlx-vlm HF-source layout before sanitize-time remap (oQ writes this).
        self._write_index(
            tmp_path,
            {"model.language_model.mtp.norm.weight": "model.safetensors"},
        )
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is True

    def test_returns_false_when_index_lacks_mtp(self, tmp_path):
        # Unsloth Qwen3.6 UD MLX layout: vision_tower + language_model.model.*
        # but no language_model.mtp.* keys despite mtp_num_hidden_layers > 0
        # in config.json (issue #1426).
        self._write_index(
            tmp_path,
            {
                "language_model.model.embed_tokens.weight": "model.safetensors",
                "vision_tower.blocks.0.attn.proj.weight": "model.safetensors",
            },
        )
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is False

    def test_returns_false_for_empty_dir(self, tmp_path):
        # No index, no shards — caller treats as "no MTP weights".
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is False

    def test_returns_false_for_nonexistent_path(self, tmp_path):
        assert (
            model_loading._checkpoint_has_mtp_weights(str(tmp_path / "does-not-exist"))
            is False
        )

    def test_returns_false_on_malformed_index(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text("{not valid")
        # Falls through to safetensors-header scan; no shards exist, so
        # the helper conservatively returns False.
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is False

    def test_returns_true_when_later_shard_has_mtp(self, tmp_path):
        import numpy as np
        from safetensors.numpy import save_file

        save_file(
            {
                "language_model.model.embed_tokens.weight": np.zeros(
                    (1,), dtype=np.float16
                )
            },
            str(tmp_path / "model-00001-of-00002.safetensors"),
        )
        save_file(
            {"language_model.mtp.fc.weight": np.zeros((1,), dtype=np.float16)},
            str(tmp_path / "model-00002-of-00002.safetensors"),
        )

        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is True
