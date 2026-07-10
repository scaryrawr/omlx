# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.model_loading.maybe_load_custom_quantization."""

import json
import os
import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock

import mlx.core as mx
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
    def test_runtime_defaults_set_values_and_preserve_cache_default(self, monkeypatch):
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
        assert (
            sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
        )


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

    def test_jang_v2_sidecar_without_format_dispatches_basic_loader(
        self, tmp_path, monkeypatch
    ):
        def fake_load(model_path):
            assert model_path == tmp_path
            return "JANG_V2_MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(tmp_path, '{"model_type": "llama"}')
        path = _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "version": 2,
                    "quantization": {"bits": 4, "group_size": 64},
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)
        assert metadata is not None
        assert metadata.markers == ("jang",)
        assert metadata.codec == "affine_jang"
        assert maybe_load_custom_quantization(path, is_vlm=False) == (
            "JANG_V2_MODEL",
            "TOKENIZER",
        )

    def test_embedded_jang_v2_sidecar_without_format_materializes(
        self, tmp_path, monkeypatch
    ):
        def fake_load(model_path):
            assert model_path == tmp_path
            assert json.loads((tmp_path / "jang_config.json").read_text()) == {
                "version": 2,
                "quantization": {"bits": 5, "block_size": 128},
            }
            return "EMBEDDED_V2_MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(
            tmp_path,
            json.dumps(
                {
                    "model_type": "llama",
                    "jang": {
                        "version": 2,
                        "quantization": {"bits": 5, "block_size": 128},
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)
        assert metadata is not None
        assert metadata.embedded_sidecar is True
        assert metadata.codec == "affine_jang"
        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) == (
            "EMBEDDED_V2_MODEL",
            "TOKENIZER",
        )

    def test_config_only_jang_v2_metadata_without_format_materializes(
        self, tmp_path, monkeypatch
    ):
        def fake_load(model_path):
            assert model_path == tmp_path
            assert json.loads((tmp_path / "jang_config.json").read_text()) == {
                "model_type": "llama",
                "format_version": "2.0",
                "quantization": {
                    "profile": "JANG_2L",
                    "block_size": 64,
                    "bit_widths_used": [2, 6, 8],
                    "quantization_backend": "mx.quantize",
                },
            }
            return "CONFIG_ONLY_V2_MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(
            tmp_path,
            json.dumps(
                {
                    "model_type": "llama",
                    "format_version": "2.0",
                    "quantization": {
                        "profile": "JANG_2L",
                        "block_size": 64,
                        "bit_widths_used": [2, 6, 8],
                        "quantization_backend": "mx.quantize",
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)
        assert metadata is not None
        assert metadata.embedded_sidecar is False
        assert metadata.markers == ("jang_2l",)
        assert metadata.codec == "affine_jang"
        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) == (
            "CONFIG_ONLY_V2_MODEL",
            "TOKENIZER",
        )

    def test_embedded_jang_sidecar_materializes_before_dispatch(
        self, tmp_path, monkeypatch
    ):
        def fake_load(model_path):
            assert model_path == tmp_path
            assert json.loads((tmp_path / "jang_config.json").read_text()) == {
                "format": "jang",
                "target_bits": 4,
            }
            return "EMBEDDED_MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(
            tmp_path,
            json.dumps(
                {
                    "model_type": "llama",
                    "jang": {"format": "jang", "target_bits": 4},
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)
        assert metadata is not None
        assert metadata.embedded_sidecar is True
        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) == (
            "EMBEDDED_MODEL",
            "TOKENIZER",
        )

    def test_config_only_jangtq_alias_materializes_loader_sidecar(
        self, tmp_path, monkeypatch
    ):
        def jangtq_load(model_path):
            assert model_path == tmp_path
            sidecar = json.loads((tmp_path / "jang_config.json").read_text())
            assert sidecar["weight_format"] == "mxtq"
            assert sidecar["mxtq_bits"] == {
                "routed_expert": {
                    "gate_proj": 2,
                    "up_proj": 2,
                    "down_proj": 4,
                }
            }
            return "CONFIG_ONLY_ALIAS_JANGTQ", "TOKENIZER"

        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        _write_config(
            tmp_path,
            json.dumps(
                {
                    "model_type": "minimax_m2",
                    "format": "jangtq",
                    "routed_expert_bits": {
                        "gate_proj": 2,
                        "up_proj": 2,
                        "down_proj": 4,
                    },
                }
            ),
        )

        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) == (
            "CONFIG_ONLY_ALIAS_JANGTQ",
            "TOKENIZER",
        )

    def test_jangtq_alias_sidecar_uses_temporary_loader_path(
        self, tmp_path, monkeypatch
    ):
        class LoadedModel:
            pass

        loaded = LoadedModel()
        original_sidecar = {
            "format": "jangtq",
            "routedExpertBits": {
                "gate_proj": 2,
                "up_proj": 2,
                "down_proj": 4,
            },
        }

        def jangtq_load(model_path):
            assert model_path != tmp_path
            assert (model_path / "config.json").is_symlink()
            normalized = json.loads((model_path / "jang_config.json").read_text())
            assert normalized["mxtq_bits"] == {
                "routed_expert": {
                    "gate_proj": 2,
                    "up_proj": 2,
                    "down_proj": 4,
                }
            }
            return loaded, "TOKENIZER"

        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        _write_config(tmp_path, '{"model_type": "minimax_m2"}')
        _write_jang_config(tmp_path, json.dumps(original_sidecar))

        result = maybe_load_custom_quantization(str(tmp_path), is_vlm=False)

        assert result == (loaded, "TOKENIZER")
        assert hasattr(loaded, "_omlx_jang_loader_temp_dir")
        assert (
            json.loads((tmp_path / "jang_config.json").read_text()) == original_sidecar
        )

    def test_jang_target_subdirectory_dispatches_payload_path(
        self, tmp_path, monkeypatch
    ):
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        def fake_load(model_path):
            assert model_path == target_dir
            return "TARGET_MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(target_dir, '{"model_type": "llama"}')
        _write_jang_config(target_dir, '{"format": "jang"}')

        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) == (
            "TARGET_MODEL",
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

    def test_jang_vlm_dispatches_from_top_level_media_flags(
        self, tmp_path, monkeypatch
    ):
        def fake_load(model_path):
            return "TOP_LEVEL_MEDIA_MODEL", "PROCESSOR"

        _install_jang_loader_stub(monkeypatch, vlm_load=fake_load)
        path = _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "mxfp4",
                    "has_vision": True,
                    "modalities": {
                        "text": True,
                        "image": True,
                        "vision": True,
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)
        assert metadata is not None
        assert metadata.is_vlm is True
        assert maybe_load_custom_quantization(path, is_vlm=True) == (
            "TOP_LEVEL_MEDIA_MODEL",
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

    def test_jang_has_vision_false_overrides_config_vision(self, tmp_path, monkeypatch):
        def fake_load(model_path):
            return "TEXT_MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(
            tmp_path,
            '{"model_type": "mistral4", "vision_config": {"hidden_size": 128}}',
        )
        path = _write_jang_config(
            tmp_path,
            '{"format": "jang", "architecture": {"has_vision": false}}',
        )

        metadata = model_loading.read_jang_metadata(tmp_path)
        assert metadata is not None
        assert metadata.is_vlm is False
        assert maybe_load_custom_quantization(path, is_vlm=False) == (
            "TEXT_MODEL",
            "TOKENIZER",
        )

    def test_jang_top_level_has_vision_false_overrides_config_vision(
        self, tmp_path, monkeypatch
    ):
        def fake_load(model_path):
            return "TOP_LEVEL_TEXT_MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(
            tmp_path,
            '{"model_type": "mistral4", "vision_config": {"hidden_size": 128}}',
        )
        path = _write_jang_config(tmp_path, '{"format": "mxfp4", "has_vision": false}')

        metadata = model_loading.read_jang_metadata(tmp_path)
        assert metadata is not None
        assert metadata.is_vlm is False
        assert maybe_load_custom_quantization(path, is_vlm=False) == (
            "TOP_LEVEL_TEXT_MODEL",
            "TOKENIZER",
        )

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

    def test_jangtq_tq_layout_marker_dispatches_jangtq_loader(
        self, tmp_path, monkeypatch
    ):
        def basic_load(model_path):
            raise AssertionError("basic JANG loader should not handle tq_layout")

        def jangtq_load(model_path):
            assert model_path == tmp_path
            return "JANGTQ_TQ_LAYOUT_MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        path = _write_jang_config(
            tmp_path,
            '{"version": 2, "tq_layout": "routed_experts"}',
        )

        assert maybe_load_custom_quantization(path, is_vlm=False) == (
            "JANGTQ_TQ_LAYOUT_MODEL",
            "TOKENIZER",
        )

    def test_jangtq_gate_down_markers_dispatch_jangtq_loader(
        self, tmp_path, monkeypatch
    ):
        def basic_load(model_path):
            raise AssertionError("basic JANG loader should not handle JANGTQ_K markers")

        def jangtq_load(model_path):
            assert model_path != tmp_path
            sidecar = json.loads((model_path / "jang_config.json").read_text())
            assert sidecar["mxtq_bits"] == {
                "routed_expert": {
                    "gate_proj": 2,
                    "up_proj": 2,
                    "down_proj": 4,
                }
            }
            return "JANGTQ_K_MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        path = _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "mxtq_gate_up_bits": 2,
                    "mxtq_down_bits": 4,
                }
            ),
        )

        assert maybe_load_custom_quantization(path, is_vlm=False) == (
            "JANGTQ_K_MODEL",
            "TOKENIZER",
        )

    def test_affine_jang_bit_policy_does_not_force_jangtq(self, tmp_path, monkeypatch):
        def basic_load(model_path):
            assert model_path == tmp_path
            return "AFFINE_MIXED_MODEL", "TOKENIZER"

        def jangtq_load(model_path):
            raise AssertionError("affine JANG bit accounting must not force JANGTQ")

        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        path = _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "quantization": {
                        "mode": "affine",
                        "mxtq_bits": {
                            "gate_proj": 2,
                            "up_proj": 2,
                            "down_proj": 4,
                        },
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "affine_jang"
        assert maybe_load_custom_quantization(path, is_vlm=False) == (
            "AFFINE_MIXED_MODEL",
            "TOKENIZER",
        )

    @pytest.mark.parametrize(
        "body",
        [
            '{"format": "MXTQ"}',
            '{"format": "jang", "weight_format": "mxtq"}',
        ],
    )
    def test_mxtq_markers_dispatch_jangtq_loader(self, body, tmp_path, monkeypatch):
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
    def test_mxfp_formats_with_bit_widths_dispatch_basic_jang_loader(
        self, format_value, tmp_path, monkeypatch
    ):
        def basic_load(model_path):
            assert model_path == tmp_path
            return "MXFP_MODEL", "TOKENIZER"

        def jangtq_load(model_path):
            raise AssertionError("JANGTQ loader should not handle MXFP formats")

        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        path = _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": format_value,
                    "quantization": {"bit_widths_used": [4, 8]},
                }
            ),
        )

        assert maybe_load_custom_quantization(path, is_vlm=False) == (
            "MXFP_MODEL",
            "TOKENIZER",
        )

    @pytest.mark.parametrize("format_value", ["mxfp4", "MXFP8"])
    def test_bare_mxfp_sidecar_falls_back_to_standard_loader(
        self, format_value, tmp_path, monkeypatch
    ):
        def fake_load(model_path):
            raise AssertionError("JANG loader should not handle bare MXFP sidecars")

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(tmp_path, '{"model_type": "llama"}')
        _write_jang_config(tmp_path, json.dumps({"format": format_value}))

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "mxfp"
        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) is None

    @pytest.mark.parametrize(
        "artifact",
        [
            "runtime_sidecar",
            "stacked_sidecar",
            "jangpress_prestacked",
            "index_tq_packed",
            "index_tq_bits",
            "index_tq_codebook",
            "index_tq_signs",
            "consolidated_index_tq_packed",
            "legacy_index_tq_hadamard",
            "safetensors_key_tq_codebook",
        ],
    )
    def test_tq_artifact_markers_dispatch_jangtq_loader(
        self, artifact, tmp_path, monkeypatch
    ):
        def basic_load(model_path):
            raise AssertionError("basic JANG loader should not handle tq artifacts")

        def jangtq_load(model_path):
            assert model_path == tmp_path or (
                model_path.name.startswith("omlx-jang-loader-")
                and (model_path / "jang_config.json").is_file()
            )
            return "TQ_ARTIFACT_MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        path = _write_jang_config(tmp_path, '{"format": "jang"}')
        if artifact == "runtime_sidecar":
            import numpy as np
            from safetensors.numpy import save_file

            save_file(
                {
                    "codebook.16.2": np.zeros((1,), dtype=np.float32),
                    "signs.16.42": np.zeros((1,), dtype=np.float32),
                },
                str(tmp_path / "jangtq_runtime.safetensors"),
            )
        elif artifact == "stacked_sidecar":
            (tmp_path / "jangtq_stacked.safetensors").write_text("")
        elif artifact == "jangpress_prestacked":
            (tmp_path / "jangpress-prestacked.safetensors").write_text("")
        elif artifact == "safetensors_key_tq_codebook":
            import numpy as np
            from safetensors.numpy import save_file

            save_file(
                {"layers.0.mlp.switch_mlp.gate_proj.tq_codebook": np.zeros((1,))},
                str(tmp_path / "model-00001-of-00001.safetensors"),
            )
        else:
            if artifact == "consolidated_index_tq_packed":
                index_name = "consolidated.safetensors.index.json"
                marker = "model-00001-of-00001.tq_packed"
            elif artifact == "legacy_index_tq_hadamard":
                index_name = "model.jang.index.json"
                marker = "model-00001-of-00001.tq_hadamard_seed"
            elif artifact == "index_tq_bits":
                index_name = "model.safetensors.index.json"
                marker = "model-00001-of-00001.tq_bits"
            elif artifact == "index_tq_codebook":
                index_name = "model.safetensors.index.json"
                marker = "model-00001-of-00001.tq_codebook"
            elif artifact == "index_tq_signs":
                index_name = "model.safetensors.index.json"
                marker = "model-00001-of-00001.tq_signs"
            else:
                index_name = "model.safetensors.index.json"
                marker = "model-00001-of-00001.tq_packed"
            (tmp_path / index_name).write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {
                            "model.layers.0.self_attn.q_proj.weight": (marker)
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
        _write_jang_config(tmp_path, '{"format": "jang"}')

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

    def test_kimi_jangtq_vlm_dispatches_from_source_model_metadata(
        self, tmp_path, monkeypatch
    ):
        def generic_load(model_path):
            raise AssertionError("generic JANGTQ VLM loader should not be used")

        def kimi_load(model_path):
            assert model_path == tmp_path
            return "KIMI_SOURCE_METADATA_VLM", "PROCESSOR"

        _install_jangtq_loader_stub(
            monkeypatch,
            vlm_load=generic_load,
            kimi_vlm_load=kimi_load,
        )
        _write_config(tmp_path, '{"model_type": "qwen3_vl"}')
        path = _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "weight_format": "mxtq",
                    "source_model": {
                        "name": "Kimi-K2.6-VL-Instruct",
                        "architecture": "kimi_vl",
                    },
                    "capabilities": {"modality": "vision-language"},
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)
        assert metadata is not None
        assert metadata.is_kimi_vlm is True
        assert metadata.is_vlm is True
        assert maybe_load_custom_quantization(path, is_vlm=True) == (
            "KIMI_SOURCE_METADATA_VLM",
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

    def test_empty_bit_widths_used_sidecar_falls_back_to_standard_loader(
        self, tmp_path, monkeypatch
    ):
        def fake_load(model_path):
            raise AssertionError("JANG loader should not handle non-JANG MXFP shim")

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(tmp_path, '{"model_type": "llama"}')
        _write_jang_config(
            tmp_path,
            '{"format": "mxfp4", "quantization": {"bit_widths_used": []}}',
        )

        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) is None

    def test_empty_camel_case_bit_widths_used_falls_back_to_standard_loader(
        self, tmp_path, monkeypatch
    ):
        def fake_load(model_path):
            raise AssertionError("JANG loader should not handle non-JANG MXFP shim")

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(tmp_path, '{"model_type": "llama"}')
        _write_jang_config(
            tmp_path,
            '{"format": "mxfp4", "quantization": {"bitWidthsUsed": []}}',
        )

        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) is None

    def test_plain_mlx_weight_format_sidecar_falls_back_to_standard_loader(
        self, tmp_path, monkeypatch
    ):
        def fake_load(model_path):
            raise AssertionError("JANG loader should not handle plain MLX weights")

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(tmp_path, '{"model_type": "llama"}')
        _write_jang_config(
            tmp_path,
            '{"format": "jang", "weight_format": "mlx"}',
        )

        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) is None

    def test_plain_mlx_format_sidecar_falls_back_to_standard_loader(
        self, tmp_path, monkeypatch
    ):
        def fake_load(model_path):
            raise AssertionError("JANG loader should not handle plain MLX weights")

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(tmp_path, '{"model_type": "llama"}')
        _write_jang_config(
            tmp_path,
            '{"format": "mlx", "capabilities": {"supports_text": true}}',
        )

        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) is None

    def test_config_only_weight_format_mxtq_dispatches_jangtq_loader(
        self, tmp_path, monkeypatch
    ):
        def basic_load(model_path):
            raise AssertionError("basic JANG loader should not handle config-only MXTQ")

        def jangtq_load(model_path):
            assert model_path == tmp_path
            return "CONFIG_ONLY_JANGTQ_MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        _write_config(
            tmp_path,
            json.dumps({"model_type": "zaya", "weight_format": "mxtq"}),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"
        assert metadata.sidecar_path == tmp_path / "config.json"
        assert metadata.embedded_sidecar is False
        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) == (
            "CONFIG_ONLY_JANGTQ_MODEL",
            "TOKENIZER",
        )

    def test_config_only_jang_method_dispatches_basic_loader(
        self, tmp_path, monkeypatch
    ):
        def basic_load(model_path):
            assert model_path == tmp_path
            return "JANG_METHOD_MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        _write_config(
            tmp_path,
            json.dumps(
                {
                    "model_type": "llama",
                    "quantization": {"method": "jang-importance"},
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "affine_jang"
        assert metadata.markers == ("jang_importance",)
        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) == (
            "JANG_METHOD_MODEL",
            "TOKENIZER",
        )

    @pytest.mark.parametrize("quant_method", ["fp8", "mxfp4", "nvfp4"])
    def test_config_only_non_jang_quant_method_uses_standard_loader(
        self, tmp_path, monkeypatch, quant_method
    ):
        def basic_load(model_path):
            raise AssertionError("non-JANG quant_method must not dispatch JANG loader")

        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        _write_config(
            tmp_path,
            json.dumps(
                {
                    "model_type": "llama",
                    "quantization_config": {"quant_method": quant_method},
                }
            ),
        )

        assert model_loading.read_jang_metadata(tmp_path) is None
        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) is None

    def test_config_only_plain_mlx_format_is_not_jang(self, tmp_path, monkeypatch):
        def basic_load(model_path):
            raise AssertionError("plain MLX config should not use JANG loader")

        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        _write_config(
            tmp_path,
            json.dumps(
                {
                    "model_type": "llama",
                    "weight_format": "mlx",
                    "capabilities": {"supports_text": True},
                }
            ),
        )

        assert model_loading.read_jang_metadata(tmp_path) is None
        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) is None

    def test_jang_audio_config_marks_loader_metadata_vlm(self, tmp_path):
        _write_config(
            tmp_path,
            json.dumps(
                {
                    "model_type": "some_jang_model",
                    "audio_config": {"feature_size": 128},
                }
            ),
        )
        _write_jang_config(tmp_path, json.dumps({"format": "jang"}))

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.is_vlm is True

    def test_jang_disabled_media_overrides_audio_config_for_loader(self, tmp_path):
        _write_config(
            tmp_path,
            json.dumps(
                {
                    "model_type": "some_jang_model",
                    "audio_config": {"feature_size": 128},
                }
            ),
        )
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "capabilities": {
                        "supportsAudio": False,
                        "supportsVision": False,
                        "supportsVideo": False,
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.is_vlm is False

    def test_jang_source_model_tokenizer_fallback_uses_hf_cache(
        self, tmp_path, monkeypatch
    ):
        hf_cache = tmp_path / "hf-cache"
        snapshot = hf_cache / "models--Qwen--Qwen3-8B" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / "tokenizer.json").write_text("{}")
        monkeypatch.setenv("HF_HUB_CACHE", str(hf_cache))

        calls = []

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model_path, *args, **kwargs):
                calls.append(model_path)
                return f"TOKENIZER:{model_path}"

        transformers_mod = types.ModuleType("transformers")
        transformers_mod.AutoTokenizer = AutoTokenizer
        monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

        def fake_load(model_path):
            tokenizer = transformers_mod.AutoTokenizer.from_pretrained(model_path)
            return "MODEL", tokenizer

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(tmp_path, '{"model_type": "llama"}')
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "source_model": {"org": "Qwen", "name": "Qwen3-8B"},
                }
            ),
        )

        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) == (
            "MODEL",
            f"TOKENIZER:{snapshot}",
        )
        assert calls == [snapshot]

    @pytest.mark.parametrize(
        "tokenizer_class",
        ["TikTokenTokenizerFast", "TokenizersBackend"],
    )
    def test_jang_source_tokenizer_fallback_for_unsupported_tokenizer_config_only(
        self, tmp_path, monkeypatch, tokenizer_class
    ):
        hf_cache = tmp_path / "hf-cache"
        snapshot = hf_cache / "models--Qwen--Qwen3-8B" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / "tokenizer.json").write_text("{}")
        monkeypatch.setenv("HF_HUB_CACHE", str(hf_cache))
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": tokenizer_class})
        )

        calls = []

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model_path, *args, **kwargs):
                calls.append(model_path)
                return f"TOKENIZER:{model_path}"

        transformers_mod = types.ModuleType("transformers")
        transformers_mod.AutoTokenizer = AutoTokenizer
        monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

        def fake_load(model_path):
            tokenizer = transformers_mod.AutoTokenizer.from_pretrained(model_path)
            return "MODEL", tokenizer

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(tmp_path, '{"model_type": "llama"}')
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "source_model": {"org": "Qwen", "name": "Qwen3-8B"},
                }
            ),
        )

        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) == (
            "MODEL",
            f"TOKENIZER:{snapshot}",
        )
        assert calls == [snapshot]

    def test_jang_source_model_tokenizer_fallback_uses_local_path(
        self, tmp_path, monkeypatch
    ):
        source_dir = tmp_path / "source-model"
        source_dir.mkdir()
        (source_dir / "tokenizer.json").write_text("{}")
        model_dir = tmp_path / "jang-model"
        model_dir.mkdir()

        calls = []

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model_path, *args, **kwargs):
                calls.append(model_path)
                return f"TOKENIZER:{model_path}"

        transformers_mod = types.ModuleType("transformers")
        transformers_mod.AutoTokenizer = AutoTokenizer
        monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

        def fake_load(model_path):
            tokenizer = transformers_mod.AutoTokenizer.from_pretrained(model_path)
            return "MODEL", tokenizer

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(model_dir, '{"model_type": "llama"}')
        _write_jang_config(
            model_dir,
            json.dumps({"format": "jang", "source_model": str(source_dir)}),
        )

        assert maybe_load_custom_quantization(str(model_dir), is_vlm=False) == (
            "MODEL",
            f"TOKENIZER:{source_dir}",
        )
        assert calls == [source_dir]

    def test_jang_source_model_tokenizer_fallback_reads_model_config(
        self, tmp_path, monkeypatch
    ):
        source_dir = tmp_path / "source-model"
        source_dir.mkdir()
        (source_dir / "tokenizer.json").write_text("{}")
        model_dir = tmp_path / "jang-model"
        model_dir.mkdir()

        calls = []

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model_path, *args, **kwargs):
                calls.append(model_path)
                return f"TOKENIZER:{model_path}"

        transformers_mod = types.ModuleType("transformers")
        transformers_mod.AutoTokenizer = AutoTokenizer
        monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

        def fake_load(model_path):
            tokenizer = transformers_mod.AutoTokenizer.from_pretrained(model_path)
            return "MODEL", tokenizer

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(
            model_dir,
            json.dumps({"model_type": "llama", "source_model": str(source_dir)}),
        )
        _write_jang_config(model_dir, json.dumps({"format": "jang"}))

        assert maybe_load_custom_quantization(str(model_dir), is_vlm=False) == (
            "MODEL",
            f"TOKENIZER:{source_dir}",
        )
        assert calls == [source_dir]

    def test_jang_source_model_tokenizer_fallback_uses_hub_id_alias(
        self, tmp_path, monkeypatch
    ):
        hf_cache = tmp_path / "hf-cache"
        snapshot = (
            hf_cache
            / "models--stepfun-ai--Step-3.7-Flash-NVFP4"
            / "snapshots"
            / "abc123"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "tokenizer.json").write_text("{}")
        monkeypatch.setenv("HF_HUB_CACHE", str(hf_cache))

        calls = []

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model_path, *args, **kwargs):
                calls.append(model_path)
                return f"TOKENIZER:{model_path}"

        transformers_mod = types.ModuleType("transformers")
        transformers_mod.AutoTokenizer = AutoTokenizer
        monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

        def fake_load(model_path):
            tokenizer = transformers_mod.AutoTokenizer.from_pretrained(model_path)
            return "MODEL", tokenizer

        _install_jang_loader_stub(monkeypatch, text_load=fake_load)
        _write_config(tmp_path, '{"model_type": "step3p7"}')
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "source_model": {"hub_id": "stepfun-ai/Step-3.7-Flash-NVFP4"},
                }
            ),
        )

        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) == (
            "MODEL",
            f"TOKENIZER:{snapshot}",
        )
        assert calls == [snapshot]

    def test_jang_metadata_classifies_affine_and_extracts_bits(self, tmp_path):
        _write_config(
            tmp_path,
            '{"model_type": "deepseek_v4", "vision_config": {}}',
        )
        _write_jang_config(
            tmp_path,
            '{"format": "future", "profile": "JANG_2L", "target_bits": 4, '
            '"actual_bits_per_weight": 4.25, "quantization": {"group_size": 64}, '
            '"capabilities": {"reasoning_parser": "qwen3"}, '
            '"chat": {"sampling_defaults": {"temperature": 0.6}}}',
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "affine_jang"
        assert metadata.profile == "JANG_2L"
        assert metadata.target_bits == 4
        assert metadata.actual_bits == 4.25
        assert metadata.group_size == 64
        assert metadata.capabilities == {"reasoning_parser": "qwen3"}
        assert metadata.chat == {"sampling_defaults": {"temperature": 0.6}}
        assert metadata.is_vlm is True
        assert metadata.is_deepseek_v4 is True

    def test_jang_metadata_accepts_camel_case_current_schema(self, tmp_path):
        _write_config(tmp_path, '{"model_type": "qwen3_vl"}')
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "weightFormat": "mxtq",
                    "tqLayout": "streaming",
                    "sourceModel": {
                        "hubID": "Kimi/Kimi-K2.6-VL-Instruct",
                        "architecture": "kimi_vl",
                    },
                    "hasVideo": True,
                    "capabilities": {
                        "supportsVision": True,
                        "reasoningParser": "qwen3",
                        "toolParser": "xml_function",
                        "thinkInTemplate": True,
                        "supportsThinking": True,
                        "supportsTools": True,
                        "cacheSubtype": "hybrid_full_swa_kv",
                        "draftStrategy": "ddtree",
                        "drafterPath": "drafter/",
                        "branchingBudget": 16,
                        "blockSize": 8,
                    },
                    "chat": {
                        "reasoning": {"dropEarlierReasoning": True},
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"
        assert metadata.model_family == "kimi_vl"
        assert metadata.cache_subtype == "hybrid_full_swa_kv"
        assert metadata.reasoning_parser == "qwen3"
        assert metadata.tool_parser == "xml_function"
        assert metadata.think_in_template is True
        assert metadata.supports_thinking is True
        assert metadata.supports_tools is True
        assert metadata.drop_earlier_reasoning is True
        assert metadata.draft_strategy == "ddtree"
        assert metadata.drafter_path == "drafter/"
        assert metadata.draft_branching_budget == 16
        assert metadata.draft_block_size == 8
        assert metadata.is_kimi_vlm is True
        assert metadata.is_vlm is True

    def test_jang_metadata_prefers_chat_tool_parser_over_capability_alias(
        self, tmp_path
    ):
        _write_config(tmp_path, '{"model_type": "deepseek_v4"}')
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "capabilities": {
                        "toolParser": "legacy_xml",
                        "cacheType": "mla",
                    },
                    "chat": {
                        "tool_calling": {"parser": "dsml"},
                        "reasoning": {"thinkInTemplate": False},
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.tool_parser == "dsml"
        assert metadata.cache_type == "mla"
        assert metadata.think_in_template is False

    def test_jang_metadata_uses_model_family_for_deepseek_v4(self, tmp_path):
        _write_config(tmp_path, '{"model_type": "llama"}')
        _write_jang_config(
            tmp_path,
            json.dumps({"format": "mxfp4", "model_family": "deepseek_v4"}),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.model_family == "deepseek_v4"
        assert metadata.is_deepseek_v4 is True

    def test_jang_metadata_uses_current_top_level_family(self, tmp_path):
        _write_config(tmp_path, '{"model_type": "llama"}')
        _write_jang_config(
            tmp_path,
            json.dumps({"format": "jang", "family": "deepseek_v4"}),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.model_family == "deepseek_v4"
        assert metadata.is_deepseek_v4 is True

    def test_jang_metadata_uses_source_architecture_for_deepseek_v4(self, tmp_path):
        _write_config(tmp_path, '{"model_type": "llama"}')
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "source_model": {"architecture": "deepseek_v4"},
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.model_family == "deepseek_v4"
        assert metadata.is_deepseek_v4 is True

    def test_jang_metadata_falls_back_to_text_config_family(self, tmp_path):
        _write_config(
            tmp_path,
            json.dumps(
                {
                    "model_type": "wrapper",
                    "text_config": {"model_type": "lfm2_moe"},
                }
            ),
        )
        _write_jang_config(tmp_path, json.dumps({"format": "jang"}))

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.model_family == "lfm2_moe"

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

    def test_jang_metadata_uses_top_level_video_flag_for_vlm(self, tmp_path):
        _write_config(tmp_path, '{"model_type": "some_jang_model"}')
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "mxfp4",
                    "has_video": True,
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.is_vlm is True

    def test_jang_metadata_uses_audio_capability_for_vlm(self, tmp_path):
        _write_config(tmp_path, '{"model_type": "some_jang_model"}')
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "mxfp4",
                    "capabilities": {
                        "supports_audio": True,
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.is_vlm is True

    def test_jang_metadata_disabled_media_capabilities_keep_text_runtime(
        self, tmp_path
    ):
        _write_config(
            tmp_path,
            json.dumps(
                {
                    "model_type": "some_jang_model",
                    "vision_config": {"hidden_size": 1024},
                }
            ),
        )
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "mxfp4",
                    "capabilities": {
                        "supportsText": True,
                        "supportsVision": False,
                        "supportsVideo": False,
                        "supportsAudio": False,
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.is_vlm is False
        assert metadata.supports_text is True
        assert metadata.supports_vision is False
        assert metadata.supports_video is False
        assert metadata.supports_audio is False

    def test_jang_metadata_uses_video_preprocessor_file_for_vlm(self, tmp_path):
        _write_config(tmp_path, '{"model_type": "some_jang_model"}')
        _write_jang_config(tmp_path, '{"format": "jang"}')
        (tmp_path / "video_preprocessor_config.json").write_text("{}")

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.is_vlm is True

    def test_jang_metadata_uses_capability_family_for_kimi_vlm(self, tmp_path):
        _write_config(tmp_path, '{"model_type": "qwen3_vl"}')
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "weight_format": "mxtq",
                    "capabilities": {
                        "family": "kimi_k26",
                        "supports_video": True,
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.model_family == "kimi_k26"
        assert metadata.codec == "jangtq"
        assert metadata.is_kimi_vlm is True
        assert metadata.is_vlm is True

    def test_jang_metadata_uses_capabilities_modalities_dict(self, tmp_path):
        _write_config(tmp_path, '{"model_type": "some_jang_model"}')
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "mxfp4",
                    "capabilities": {
                        "modalities": {
                            "text": True,
                            "vision": True,
                            "audio": False,
                        }
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
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

    def test_jang_metadata_uses_bit_widths_used_for_jangtq_bits(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "quantization": {
                        "profile": "JANGTQ4",
                        "bit_widths_used": [4],
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"
        assert metadata.mxtq_bits == 4

    def test_jang_metadata_uses_text_config_weight_format_for_jangtq(
        self, tmp_path, monkeypatch
    ):
        def basic_load(model_path):
            raise AssertionError("basic JANG loader should not handle text_config MXTQ")

        def jangtq_load(model_path):
            assert model_path == tmp_path
            return "TEXT_CONFIG_JANGTQ_MODEL", "TOKENIZER"

        _install_jang_loader_stub(monkeypatch, text_load=basic_load)
        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        _write_config(
            tmp_path,
            json.dumps(
                {"model_type": "zaya", "text_config": {"weight_format": "mxtq"}}
            ),
        )
        path = _write_jang_config(tmp_path, json.dumps({"format": "jang"}))

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"
        assert maybe_load_custom_quantization(path, is_vlm=False) == (
            "TEXT_CONFIG_JANGTQ_MODEL",
            "TOKENIZER",
        )

    def test_jang_metadata_accepts_current_routed_layer_bit_aliases(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "routedExpertLayerBits": {"0": 2, "3": 4},
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"
        assert metadata.routed_expert_bit_plans == ({"0": 2, "3": 4},)

    def test_jang_metadata_does_not_treat_affine_bit_widths_as_mxtq(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "quantization": {
                        "profile": "JANG_5K",
                        "bit_widths_used": [5, 6],
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "affine_jang"
        assert metadata.mxtq_bits is None

    def test_config_only_jang_v2_detects_actual_bits_per_weight_alias(self, tmp_path):
        _write_config(
            tmp_path,
            json.dumps(
                {
                    "model_type": "llama",
                    "format_version": "2.0",
                    "quantization": {
                        "actualBitsPerWeight": 2.85,
                        "groupSize": 64,
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "affine_jang"
        assert metadata.actual_bits == 2.85
        assert metadata.group_size == 64

    def test_dsv4_default_bits_falls_back_to_jangtq_bit_widths_used(self):
        assert (
            model_loading._dsv4_routed_default_bits(
                {},
                {"quantization": {"profile": "JANGTQ4", "bit_widths_used": [4]}},
            )
            == 4
        )

    def test_dsv4_metadata_routed_layer_bits_accepts_current_alias(self):
        assert model_loading._dsv4_metadata_routed_layer_bits(
            {},
            {"runtime": {"routedExpertBitPlan": {"routedExpertLayerBits": {"1": 4}}}},
        ) == {"1": 4}

    def test_dsv4_default_bits_reads_routed_expert_plan_default(self):
        assert (
            model_loading._dsv4_routed_default_bits(
                {},
                {
                    "routed_expert_bit_plan": {
                        "default": {
                            "gate_proj": 3,
                            "up_proj": 3,
                            "down_proj": 5,
                        }
                    }
                },
            )
            == 3
        )

    def test_patch_jangtq_artifacts_marks_text_config(self, tmp_path):
        (tmp_path / "jangtq_runtime.safetensors").write_text("")
        config = {
            "text_config": {
                "quantization": {},
                "quantization_config": {},
            }
        }

        patched = model_loading.patch_jangtq_weight_format_from_artifacts(
            config,
            tmp_path,
        )

        assert patched["weight_format"] == "mxtq"
        assert patched["text_config"]["weight_format"] == "mxtq"
        assert patched["text_config"]["quantization"]["weight_format"] == "mxtq"
        assert patched["text_config"]["quantization_config"]["weight_format"] == "mxtq"

    def test_quantization_default_pair_prefers_block_size_alias(self):
        assert model_loading._quantization_default_pair(
            {"bit_width": "5", "block_size": 96, "group_size": 64}
        ) == (5, 96)

    def test_quantization_default_pair_ignores_fractional_target_bits(self):
        assert (
            model_loading._quantization_default_pair(
                {"target_bits": 2.75, "block_size": 64}
            )
            is None
        )

    def test_jang_metadata_normalizes_current_capability_and_chat_schema(
        self, tmp_path
    ):
        _write_config(tmp_path, '{"model_type": "lfm2_moe"}')
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "capabilities": {
                        "family": "lfm2_5",
                        "cache_type": "kv",
                        "cache_subtype": "hybrid_full_swa_kv",
                        "supports_text": True,
                        "supports_vision": False,
                        "supports_video": False,
                        "supports_audio": False,
                        "reasoning": {"parser": "qwen3", "supported": True},
                        "tools": {"parser": "lfm2", "supported": True},
                        "think_in_template": False,
                        "draft_strategy": "ddtree",
                        "drafter_path": "drafter/",
                        "branching_budget": 24,
                        "block_size": 8,
                    },
                    "chat": {
                        "encoder": "deepseek_v4",
                        "has_tokenizer_chat_template": True,
                        "bos_token": "<bos>",
                        "bos_token_id": 0,
                        "eos_token": "<eos>",
                        "eos_token_id": 1,
                        "role_tokens": {
                            "user": "<user>",
                            "assistant": "<assistant>",
                        },
                        "reasoning": {
                            "parser": "think_xml",
                            "supported": True,
                            "modes": ["chat", "thinking"],
                            "default_mode": "chat",
                            "thinking_start": "<think>",
                            "thinking_end": "</think>",
                            "reasoning_effort_levels": ["max", "high", None],
                            "drop_earlier_reasoning": True,
                        },
                        "tool_calling": {
                            "parser": "xml_function",
                            "dsml_token": "<tool_calls_begin>",
                            "tool_calls_block": "tool_calls",
                            "invoke_block": "invoke",
                            "parameter_block": "parameter",
                            "tool_output_tag": "tool_outputs",
                        },
                        "sampling_defaults": {
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "max_new_tokens": 300,
                        },
                    },
                    "mxtq_bits_by_role": {
                        "routed_expert": {
                            "gate_proj": 2,
                            "up_proj": 2,
                            "down_proj": 4,
                        }
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"
        assert metadata.cache_type == "kv"
        assert metadata.cache_subtype == "hybrid_full_swa_kv"
        assert metadata.model_family == "lfm2_5"
        assert metadata.reasoning_parser is None
        assert metadata.tool_parser == "xml_function"
        assert metadata.think_in_template is False
        assert metadata.supports_thinking is True
        assert metadata.supports_tools is True
        assert metadata.supports_text is True
        assert metadata.supports_vision is False
        assert metadata.supports_video is False
        assert metadata.supports_audio is False
        assert metadata.drop_earlier_reasoning is True
        assert metadata.chat_encoder == "deepseek_v4"
        assert metadata.chat_has_tokenizer_template is True
        assert metadata.chat_bos_token == "<bos>"
        assert metadata.chat_bos_token_id == 0
        assert metadata.chat_eos_token == "<eos>"
        assert metadata.chat_eos_token_id == 1
        assert metadata.chat_role_tokens == {
            "user": "<user>",
            "assistant": "<assistant>",
        }
        assert metadata.chat_reasoning_modes == ("chat", "thinking")
        assert metadata.chat_reasoning_default_mode == "chat"
        assert metadata.chat_reasoning_thinking_start == "<think>"
        assert metadata.chat_reasoning_thinking_end == "</think>"
        assert metadata.chat_reasoning_effort_levels == ("max", "high", None)
        assert metadata.chat_tool_dsml_token == "<tool_calls_begin>"
        assert metadata.chat_tool_calls_block == "tool_calls"
        assert metadata.chat_tool_invoke_block == "invoke"
        assert metadata.chat_tool_parameter_block == "parameter"
        assert metadata.chat_tool_output_tag == "tool_outputs"
        assert metadata.sampling_temperature == 0.6
        assert metadata.sampling_top_p == 0.95
        assert metadata.sampling_max_new_tokens == 300
        assert metadata.draft_strategy == "ddtree"
        assert metadata.drafter_path == "drafter/"
        assert metadata.draft_branching_budget == 24
        assert metadata.draft_block_size == 8
        assert metadata.mxtq_bits == {
            "routed_expert": {"gate_proj": 2, "up_proj": 2, "down_proj": 4}
        }

    def test_jang_metadata_normalizes_nested_draft_capability(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "capabilities": {
                        "draft": {
                            "strategy": "dflash",
                            "path": "draft-model",
                            "blockSize": 4,
                        }
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.draft_strategy == "dflash"
        assert metadata.drafter_path == "draft-model"
        assert metadata.draft_branching_budget is None
        assert metadata.draft_block_size == 4

    def test_jang_metadata_derives_mxtq_bits_from_gate_down_fields(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "mxtq_gate_up_bits": 2,
                    "mxtq_down_bits": 4,
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"
        assert metadata.mxtq_bits == {
            "routed_expert": {"gate_proj": 2, "up_proj": 2, "down_proj": 4}
        }

    def test_jang_metadata_treats_runtime_routed_expert_bits_as_jangtq(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "runtime": {"routed_expert_bits": {"routed_expert": 3}},
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"
        assert metadata.mxtq_bits == {"routed_expert": 3}

    def test_jang_metadata_reads_routed_expert_plan_default_bits(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "routed_expert_bit_plan": {
                        "default_bits": {
                            "gate_proj": 2,
                            "up_proj": 2,
                            "down_proj": 4,
                        },
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"
        assert metadata.mxtq_bits == {
            "routed_expert": {"gate_proj": 2, "up_proj": 2, "down_proj": 4}
        }

    def test_jang_metadata_ignores_unscoped_default_key_for_mxtq_bits(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "chat": {"sampling_defaults": {"default": 2}},
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "affine_jang"
        assert metadata.mxtq_bits is None

    def test_jang_metadata_wraps_flat_projection_bit_map(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jangtq",
                    "routed_expert_bits": {
                        "gate_proj": 2,
                        "up_proj": 2,
                        "down_proj": 4,
                    },
                }
            ),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"
        assert metadata.mxtq_bits == {
            "routed_expert": {"gate_proj": 2, "up_proj": 2, "down_proj": 4}
        }

    def test_jang_metadata_treats_quantization_family_as_jangtq(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps({"format": "jang", "quantization": {"family": "jangtq"}}),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"

    def test_jang_metadata_treats_mxtq_profile_alias_as_jangtq(self, tmp_path):
        _write_config(
            tmp_path,
            json.dumps({"model_type": "llama", "quantization": {"family": "mxtq4"}}),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"
        assert metadata.markers == ("mxtq4",)

    def test_jang_metadata_infers_mxtq_bits_from_runtime_codebook(self, tmp_path):
        import numpy as np
        from safetensors.numpy import save_file

        _write_jang_config(tmp_path, json.dumps({"format": "mxtq"}))
        save_file(
            {
                "codebook.16.4": np.zeros((1,), dtype=np.float32),
                "signs.16.42": np.zeros((1,), dtype=np.float32),
            },
            str(tmp_path / "jangtq_runtime.safetensors"),
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"
        assert metadata.mxtq_bits == 4

    def test_jangtq_runtime_sidecar_preflight_rejects_unreadable_file(
        self, tmp_path, monkeypatch
    ):
        def jangtq_load(model_path):
            raise AssertionError("malformed runtime sidecar should fail before load")

        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        _write_jang_config(tmp_path, json.dumps({"format": "mxtq"}))
        (tmp_path / "jangtq_runtime.safetensors").write_text("not safetensors")

        with pytest.raises(RuntimeError, match="runtime sidecar"):
            maybe_load_custom_quantization(str(tmp_path), is_vlm=False)

    def test_jangtq_runtime_sidecar_preflight_rejects_missing_signs(
        self, tmp_path, monkeypatch
    ):
        import numpy as np
        from safetensors.numpy import save_file

        def jangtq_load(model_path):
            raise AssertionError("incomplete runtime sidecar should fail before load")

        _install_jangtq_loader_stub(monkeypatch, text_load=jangtq_load)
        _write_jang_config(tmp_path, json.dumps({"format": "mxtq"}))
        save_file(
            {"codebook.16.4": np.zeros((1,), dtype=np.float32)},
            str(tmp_path / "jangtq_runtime.safetensors"),
        )

        with pytest.raises(RuntimeError, match="no signs"):
            maybe_load_custom_quantization(str(tmp_path), is_vlm=False)

    def test_jang_metadata_normalizes_scalar_mxtq_bits(self, tmp_path):
        _write_jang_config(tmp_path, json.dumps({"format": "jang", "mxtq_bits": 2}))

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"
        assert metadata.mxtq_bits == {"routed_expert": 2}

    def test_jang_metadata_classifies_jangtq_from_weight_map(self, tmp_path):
        _write_jang_config(tmp_path, '{"format": "jang"}')
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"metadata": {}, "weight_map": {"model.layers.0": '
            '"model-00001-of-00001.tq_packed"}}'
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"

    def test_jang_metadata_classifies_jangtq_from_consolidated_weight_map(
        self, tmp_path
    ):
        _write_jang_config(tmp_path, '{"format": "jang"}')
        (tmp_path / "consolidated.safetensors.index.json").write_text(
            '{"metadata": {}, "weight_map": {"model.layers.0": '
            '"model-00001-of-00001.tq_packed"}}'
        )

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "jangtq"

    def test_jang_metadata_treats_top_level_drop_mtp_as_authoritative(self, tmp_path):
        _write_jang_config(tmp_path, '{"format": "jang", "drop_mtp": true}')

        assert model_loading._jang_sidecar_declares_mtp_dropped(tmp_path) is True

    def test_jang_metadata_treats_runtime_unwired_mtp_as_dropped(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "runtime": {
                        "bundle_has_mtp": False,
                        "mtp_mode": "runtime_unwired",
                    },
                }
            ),
        )

        assert model_loading._jang_sidecar_declares_mtp_dropped(tmp_path) is True

    def test_jang_metadata_classifies_unknown_sidecar(self, tmp_path):
        _write_jang_config(tmp_path, '{"format": "future-jang-family"}')

        metadata = model_loading.read_jang_metadata(tmp_path)

        assert metadata is not None
        assert metadata.codec == "unknown_jang"

    def test_jangspec_bundle_root_dispatches_to_bundle_loader(
        self, tmp_path, monkeypatch
    ):
        bundle = tmp_path
        target = bundle / "target"
        target.mkdir()
        (bundle / "jangspec.json").write_text('{"bundle_version": 1}')
        _write_config(target, '{"model_type": "llama"}')
        _write_jang_config(target, '{"format": "jang"}')

        calls = []
        monkeypatch.setitem(sys.modules, "jang_tools", types.ModuleType("jang_tools"))
        jangspec_mod = types.ModuleType("jang_tools.jangspec")
        jangspec_mod.__path__ = []
        monkeypatch.setitem(sys.modules, "jang_tools.jangspec", jangspec_mod)
        bundle_loader_mod = types.ModuleType("jang_tools.jangspec.bundle_loader")

        def load_jang_model_from_bundle(path):
            calls.append(path)
            return "JANGSPEC_MODEL", "TOKENIZER"

        bundle_loader_mod.load_jang_model_from_bundle = load_jang_model_from_bundle
        monkeypatch.setitem(
            sys.modules,
            "jang_tools.jangspec.bundle_loader",
            bundle_loader_mod,
        )

        metadata = model_loading.read_jang_metadata(bundle)

        assert metadata is not None
        assert metadata.model_path == target
        assert metadata.bundle_path == bundle
        assert maybe_load_custom_quantization(str(bundle), is_vlm=False) == (
            "JANGSPEC_MODEL",
            "TOKENIZER",
        )
        assert calls == [bundle]

    def test_jangspec_embedded_target_metadata_dispatches_to_bundle_loader(
        self, tmp_path, monkeypatch
    ):
        bundle = tmp_path
        target = bundle / "target"
        target.mkdir()
        (bundle / "jangspec.json").write_text('{"bundle_version": 1}')
        _write_config(target, '{"model_type": "llama", "jang": {"format": "jang"}}')

        calls = []
        monkeypatch.setitem(sys.modules, "jang_tools", types.ModuleType("jang_tools"))
        jangspec_mod = types.ModuleType("jang_tools.jangspec")
        jangspec_mod.__path__ = []
        monkeypatch.setitem(sys.modules, "jang_tools.jangspec", jangspec_mod)
        bundle_loader_mod = types.ModuleType("jang_tools.jangspec.bundle_loader")

        def load_jang_model_from_bundle(path):
            calls.append(path)
            return "EMBEDDED_JANGSPEC_MODEL", "TOKENIZER"

        bundle_loader_mod.load_jang_model_from_bundle = load_jang_model_from_bundle
        monkeypatch.setitem(
            sys.modules,
            "jang_tools.jangspec.bundle_loader",
            bundle_loader_mod,
        )

        metadata = model_loading.read_jang_metadata(bundle)

        assert metadata is not None
        assert metadata.model_path == target
        assert metadata.bundle_path == bundle
        assert metadata.embedded_sidecar is True
        assert maybe_load_custom_quantization(str(bundle), is_vlm=False) == (
            "EMBEDDED_JANGSPEC_MODEL",
            "TOKENIZER",
        )
        assert calls == [bundle]


class TestJANGQuantizationShapePatch:
    def _save_quant_shapes(self, model_dir, shapes):
        np = pytest.importorskip("numpy")
        safetensors_numpy = pytest.importorskip("safetensors.numpy")
        tensors = {}
        for base, spec in shapes.items():
            bits, group_size, in_dim = spec
            packed = (bits * in_dim + 31) // 32
            scale_groups = (in_dim + group_size - 1) // group_size
            tensors[f"{base}.weight"] = np.zeros((1, packed), dtype=np.uint32)
            tensors[f"{base}.scales"] = np.zeros((1, scale_groups), dtype=np.float16)
            tensors[f"{base}.biases"] = np.zeros((1, scale_groups), dtype=np.float16)
        safetensors_numpy.save_file(tensors, str(model_dir / "model.safetensors"))

    def test_patches_stale_mixed_bit_overrides_from_shapes(self, tmp_path):
        _write_jang_config(tmp_path, '{"format": "jang"}')
        self._save_quant_shapes(
            tmp_path,
            {
                "model.layers.0.self_attn.q_proj": (8, 32, 256),
                "model.layers.0.mlp.gate_proj": (6, 32, 384),
            },
        )
        config = {
            "model_type": "llama",
            "quantization": {
                "bits": 8,
                "group_size": 32,
                "mode": "affine",
                "model.layers.0.mlp.gate_proj": {
                    "bits": 8,
                    "group_size": 32,
                },
            },
        }

        patched = model_loading.patch_jang_quantization_from_shapes(config, tmp_path)

        quantization = patched["quantization"]
        assert quantization["bits"] == 8
        assert quantization["group_size"] == 32
        assert quantization["model.layers.0.mlp.gate_proj"] == {
            "bits": 6,
            "group_size": 32,
            "mode": "affine",
        }
        assert quantization["language_model.model.layers.0.mlp.gate_proj"] == {
            "bits": 6,
            "group_size": 32,
            "mode": "affine",
        }
        assert "model.layers.0.self_attn.q_proj" not in quantization

    def test_repairs_stale_default_quantization_from_shapes(self, tmp_path):
        _write_jang_config(tmp_path, '{"format": "jang"}')
        self._save_quant_shapes(
            tmp_path,
            {
                "model.layers.0.self_attn.q_proj": (6, 32, 384),
                "model.layers.0.self_attn.k_proj": (6, 32, 384),
            },
        )
        config = {
            "model_type": "llama",
            "quantization": {
                "bits": 8,
                "group_size": 32,
                "mode": "affine",
            },
        }

        patched = model_loading.patch_jang_quantization_from_shapes(config, tmp_path)

        quantization = patched["quantization"]
        assert quantization["bits"] == 6
        assert quantization["group_size"] == 32
        assert "model.layers.0.self_attn.q_proj" not in quantization

    def test_shape_inference_preserves_valid_explicit_module_claim(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "quantization": {"bit_widths_used": [4, 8]},
                }
            ),
        )
        self._save_quant_shapes(
            tmp_path,
            {
                "model.layers.0.self_attn.q_proj": (4, 64, 512),
                "model.layers.0.self_attn.k_proj": (8, 32, 256),
            },
        )
        config = {
            "model_type": "llama",
            "quantization": {
                "bits": 8,
                "group_size": 32,
                "mode": "affine",
                "model.layers.0.self_attn.q_proj": {
                    "bits": 4,
                    "group_size": 64,
                },
            },
        }

        patched = model_loading.patch_jang_quantization_from_shapes(config, tmp_path)

        quantization = patched["quantization"]
        assert quantization["bits"] == 8
        assert quantization["group_size"] == 32
        assert quantization["model.layers.0.self_attn.q_proj"] == {
            "bits": 4,
            "group_size": 64,
        }

    def test_shape_inference_trusts_top_level_claim_for_non_mixed_sidecar(
        self, tmp_path
    ):
        _write_jang_config(tmp_path, '{"format": "jang"}')
        self._save_quant_shapes(
            tmp_path,
            {"model.layers.0.self_attn.q_proj": (4, 64, 512)},
        )
        config = {
            "model_type": "llama",
            "quantization": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
            },
        }

        patched = model_loading.patch_jang_quantization_from_shapes(config, tmp_path)

        quantization = patched["quantization"]
        assert quantization["bits"] == 4
        assert quantization["group_size"] == 64
        assert "model.layers.0.self_attn.q_proj" not in quantization

    def test_shape_inference_resolves_routed_ambiguous_bits_from_name(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "quantization": {"bit_widths_used": [2, 4, 8]},
                }
            ),
        )
        self._save_quant_shapes(
            tmp_path,
            {
                "model.layers.0.self_attn.q_proj": (8, 32, 256),
                "model.layers.0.mlp.switch_mlp.gate_proj": (2, 128, 1024),
            },
        )
        config = {"model_type": "llama", "quantization": {"bits": 8, "group_size": 32}}

        patched = model_loading.patch_jang_quantization_from_shapes(config, tmp_path)

        quantization = patched["quantization"]
        assert quantization["bits"] == 8
        assert quantization["group_size"] == 32
        assert quantization["model.layers.0.mlp.switch_mlp.gate_proj"] == {
            "bits": 2,
            "group_size": 128,
            "mode": "affine",
        }

    def test_shape_inference_uses_uniform_group_size_from_shape_ratios(
        self, tmp_path
    ):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "quantization": {"bit_widths_used": [3, 4, 8]},
                }
            ),
        )
        self._save_quant_shapes(
            tmp_path,
            {
                "model.layers.0.self_attn.q_proj": (8, 64, 512),
                "model.layers.0.mlp.gate_proj": (4, 64, 512),
                "model.layers.0.mlp.down_proj": (3, 64, 512),
            },
        )
        config = {
            "model_type": "llama",
            "quantization": {
                "bits": 8,
                "group_size": 32,
                "mode": "affine",
            },
        }

        patched = model_loading.patch_jang_quantization_from_shapes(config, tmp_path)

        quantization = patched["quantization"]
        assert quantization["bits"] == 8
        assert quantization["group_size"] == 64
        assert quantization["model.layers.0.mlp.gate_proj"] == {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
        }
        assert quantization["model.layers.0.mlp.down_proj"] == {
            "bits": 3,
            "group_size": 64,
            "mode": "affine",
        }

    def test_shape_inference_uses_qwen_architecture_dim_for_ambiguous_shapes(
        self, tmp_path
    ):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "quantization": {"bit_widths_used": [4, 8]},
                }
            ),
        )
        self._save_quant_shapes(
            tmp_path,
            {
                "model.layers.0.linear_attn.out_proj": (4, 64, 128),
                "model.layers.0.self_attn.q_proj": (8, 32, 128),
            },
        )
        config = {
            "model_type": "qwen3_5",
            "vision_config": {},
            "hidden_size": 128,
            "num_attention_heads": 4,
            "head_dim": 32,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 32,
            "quantization": {
                "bits": 8,
                "group_size": 32,
                "mode": "affine",
            },
        }

        patched = model_loading.patch_jang_quantization_from_shapes(config, tmp_path)

        quantization = patched["quantization"]
        assert quantization["bits"] == 8
        assert quantization["group_size"] == 32
        assert quantization["model.layers.0.linear_attn.out_proj"] == {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
        }
        assert "model.layers.0.self_attn.q_proj" not in quantization

    def test_shape_inference_respects_bit_width_hints_for_jang4_routed(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "quantization": {"bit_widths_used": [4, 8]},
                }
            ),
        )
        self._save_quant_shapes(
            tmp_path,
            {
                "model.layers.0.self_attn.q_proj": (8, 32, 256),
                "model.layers.0.mlp.switch_mlp.gate_proj": (4, 64, 512),
            },
        )
        config = {"model_type": "llama", "quantization": {"bits": 8, "group_size": 32}}

        patched = model_loading.patch_jang_quantization_from_shapes(config, tmp_path)

        assert patched["quantization"]["model.layers.0.mlp.switch_mlp.gate_proj"] == {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
        }

    def test_infers_missing_jang_quantization_block_from_shapes(self, tmp_path):
        self._save_quant_shapes(
            tmp_path,
            {
                "model.layers.0.self_attn.q_proj": (8, 32, 256),
                "model.layers.0.self_attn.k_proj": (8, 32, 256),
                "model.layers.0.mlp.gate_proj": (6, 32, 384),
            },
        )
        config = {"model_type": "gemma4_unified", "weight_format": "mxfp4"}

        patched = model_loading.patch_jang_quantization_from_shapes(config, tmp_path)

        quantization = patched["quantization"]
        assert quantization["bits"] == 8
        assert quantization["group_size"] == 32
        assert quantization["mode"] == "affine"
        assert quantization["model.layers.0.mlp.gate_proj"] == {
            "bits": 6,
            "group_size": 32,
            "mode": "affine",
        }

    def test_infers_jang_5_bit_shapes_from_upstream_profile(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "quantization": {
                        "profile": "JANG_5K",
                        "bit_widths_used": [5],
                        "block_size": 64,
                    },
                }
            ),
        )
        self._save_quant_shapes(
            tmp_path,
            {
                "model.layers.0.self_attn.q_proj": (5, 64, 320),
                "model.layers.0.mlp.gate_proj": (5, 64, 320),
            },
        )
        config = {
            "model_type": "llama",
            "quantization": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
            },
        }

        patched = model_loading.patch_jang_quantization_from_shapes(config, tmp_path)

        quantization = patched["quantization"]
        assert quantization["bits"] == 5
        assert quantization["group_size"] == 64
        assert "model.layers.0.self_attn.q_proj" not in quantization

    def test_infers_non_group_aligned_jang_v2_shapes(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "format_version": "2.0",
                    "quantization": {
                        "profile": "JANG_5K",
                        "bit_widths_used": [5],
                        "block_size": 64,
                    },
                }
            ),
        )
        self._save_quant_shapes(
            tmp_path,
            {
                "model.layers.0.self_attn.q_proj": (5, 64, 130),
                "model.layers.0.mlp.gate_proj": (5, 64, 130),
            },
        )
        config = {
            "model_type": "llama",
            "quantization": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
            },
        }

        patched = model_loading.patch_jang_quantization_from_shapes(config, tmp_path)

        quantization = patched["quantization"]
        assert quantization["bits"] == 5
        assert quantization["group_size"] == 64

    def test_infers_jang_quantization_with_block_size_override(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "quantization": {
                        "profile": "JANG_5K",
                        "bitWidthsUsed": [5],
                        "blockSize": 48,
                    },
                }
            ),
        )
        self._save_quant_shapes(
            tmp_path,
            {
                "model.layers.0.self_attn.q_proj": (5, 48, 96),
                "model.layers.0.mlp.gate_proj": (5, 48, 96),
            },
        )
        config = {
            "model_type": "llama",
            "quantization": {
                "bits": 4,
                "group_size": 32,
                "mode": "affine",
            },
        }

        patched = model_loading.patch_jang_quantization_from_shapes(config, tmp_path)

        quantization = patched["quantization"]
        assert quantization["bits"] == 5
        assert quantization["group_size"] == 48

    def test_shape_patch_preserves_current_quantization_metadata_blocks(self, tmp_path):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "quantization": {
                        "bit_widths_used": [4, 8],
                        "block_size": 64,
                    },
                }
            ),
        )
        self._save_quant_shapes(
            tmp_path,
            {
                "model.layers.0.self_attn.q_proj": (8, 64, 256),
                "model.layers.0.mlp.gate_proj": (4, 64, 256),
            },
        )
        config = {
            "model_type": "llama",
            "quantization": {
                "bits": 8,
                "group_size": 64,
                "routed_experts": {"group_size": 128, "bits": 4},
                "top_level_default": {"group_size": 64, "bits": 8},
                "expert_layout": {"format": "stacked"},
                "mxtq_bits": {"routed_expert": 2},
                "routed_expert_bit_plan": {"routed_layer_bits": {"1": 4}},
                "embed_bits": 8,
                "router_bits": 8,
                "attention_bits": 8,
                "lm_head_bits": 8,
                "cca_conv_bits": 4,
                "norms_residual_bits": 8,
            },
        }

        patched = model_loading.patch_jang_quantization_from_shapes(config, tmp_path)

        quantization = patched["quantization"]
        assert quantization["routed_experts"] == {"group_size": 128, "bits": 4}
        assert quantization["top_level_default"] == {"group_size": 64, "bits": 8}
        assert quantization["expert_layout"] == {"format": "stacked"}
        assert quantization["mxtq_bits"] == {"routed_expert": 2}
        assert quantization["routed_expert_bit_plan"] == {
            "routed_layer_bits": {"1": 4}
        }
        assert quantization["embed_bits"] == 8
        assert quantization["router_bits"] == 8
        assert quantization["attention_bits"] == 8
        assert quantization["lm_head_bits"] == 8
        assert quantization["cca_conv_bits"] == 4
        assert quantization["norms_residual_bits"] == 8
        assert quantization["model.layers.0.mlp.gate_proj"] == {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
        }

    def test_shape_inference_uses_top_level_routed_expert_group_size_hint(
        self, tmp_path
    ):
        _write_jang_config(
            tmp_path,
            json.dumps(
                {
                    "format": "jang",
                    "quantization": {"bit_widths_used": [5]},
                    "routed_expert_group_size": 48,
                }
            ),
        )
        self._save_quant_shapes(
            tmp_path,
            {
                "model.layers.0.mlp.switch_mlp.gate_proj": (5, 48, 96),
                "model.layers.0.mlp.switch_mlp.up_proj": (5, 48, 96),
            },
        )
        config = {
            "model_type": "llama",
            "quantization": {"bits": 4, "group_size": 64},
        }

        patched = model_loading.patch_jang_quantization_from_shapes(config, tmp_path)

        quantization = patched["quantization"]
        assert quantization["bits"] == 5
        assert quantization["group_size"] == 48

    def test_jangtq_runtime_artifact_forces_mxtq_weight_format(self, tmp_path):
        (tmp_path / "jangtq_runtime.safetensors").write_bytes(b"")
        config = {
            "model_type": "deepseek_v4",
            "weight_format": "mlx",
            "quantization": {"weight_format": "mlx"},
        }

        patched = model_loading.patch_jangtq_weight_format_from_artifacts(
            config,
            tmp_path,
        )

        assert patched["weight_format"] == "mxtq"
        assert patched["quantization"]["weight_format"] == "mxtq"


class TestJANGGroupedConv1dSanitize:
    def test_sanitize_grouped_conv1d_layout_transposes_hf_shape(self):
        raw = mx.zeros((8, 1, 4))
        other = mx.zeros((8, 4, 1))
        weights = {
            "model.layers.0.linear_attn.conv1d.weight": raw,
            "model.layers.0.linear_attn.other.weight": other,
        }

        sanitized = model_loading._sanitize_grouped_conv1d_layout(weights)

        assert sanitized is not weights
        assert sanitized["model.layers.0.linear_attn.conv1d.weight"].shape == (8, 4, 1)
        assert sanitized["model.layers.0.linear_attn.other.weight"] is other

    def test_sanitize_grouped_conv1d_layout_leaves_multi_channel_shape(self):
        raw = mx.zeros((8, 2, 4))
        weights = {"model.layers.0.conv1d.weight": raw}

        sanitized = model_loading._sanitize_grouped_conv1d_layout(weights)

        assert sanitized is weights

    def test_scoped_load_weights_sanitizer_restores_module_method(self):
        import mlx.nn as nn

        original = nn.Module.load_weights

        with model_loading._scoped_jang_grouped_conv1d_load_weights_sanitize():
            assert nn.Module.load_weights is not original

        assert nn.Module.load_weights is original


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
    def test_forwards_trust_remote_code_when_mlx_lm_supports_it(
        self, tmp_path, monkeypatch
    ):
        path = _write_config(tmp_path, '{"model_type": "llama"}')
        maybe_apply = MagicMock()
        monkeypatch.setattr(model_loading, "maybe_apply_pre_load_patches", maybe_apply)
        # Pin the capability flag so the test is deterministic regardless of the
        # installed mlx-lm version (lm_load_compat reads this global at call time).
        monkeypatch.setattr(model_loading, "_LM_LOAD_ACCEPTS_TRC", True)

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

    def test_omits_trust_remote_code_when_mlx_lm_lacks_it(self, tmp_path, monkeypatch):
        # Some mlx-lm releases dropped ``trust_remote_code`` from ``load``.
        # lm_load_compat must omit the kwarg there rather than raise TypeError.
        path = _write_config(tmp_path, '{"model_type": "llama"}')
        monkeypatch.setattr(model_loading, "maybe_apply_pre_load_patches", MagicMock())
        monkeypatch.setattr(model_loading, "_LM_LOAD_ACCEPTS_TRC", False)

        load_mock = MagicMock(return_value=("MODEL", "TOKENIZER"))
        monkeypatch.setitem(sys.modules, "mlx_lm", MagicMock(load=load_mock))

        settings = types.SimpleNamespace(trust_remote_code=True)
        result = model_loading.load_text_model(
            path,
            tokenizer_config={"trust_remote_code": True},
            model_settings=settings,
        )

        assert result == ("MODEL", "TOKENIZER")
        load_mock.assert_called_once_with(
            path,
            tokenizer_config={"trust_remote_code": True},
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

    def test_jang_metadata_dropped_mtp_overrides_config_mtp_heads(
        self, tmp_path, monkeypatch, caplog
    ):
        calls, sanitize_mock, runtime_mock, attach_mock = self._stub_patches(
            monkeypatch
        )
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5", "text_config": {"mtp_num_hidden_layers": 1}}',
        )
        _write_jang_config(
            tmp_path,
            '{"format": "jang", "runtime": {"bundle_has_mtp": false, '
            '"mtp_mode": "drop"}}',
        )
        settings = types.SimpleNamespace(mtp_enabled=True)

        caplog.set_level("WARNING", logger=model_loading.__name__)
        maybe_apply_pre_load_patches(path, model_settings=settings, for_vlm=True)

        sanitize_mock.assert_not_called()
        runtime_mock.assert_not_called()
        attach_mock.assert_not_called()
        assert calls == []
        assert "JANG metadata marks MTP weights as dropped" in caplog.text

    def test_jang_metadata_absent_mtp_mode_overrides_config_mtp_heads(
        self, tmp_path, monkeypatch, caplog
    ):
        calls, sanitize_mock, runtime_mock, attach_mock = self._stub_patches(
            monkeypatch
        )
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5", "text_config": {"mtp_num_hidden_layers": 1}}',
        )
        _write_jang_config(
            tmp_path,
            '{"format": "jang", "runtime": {"mtp_mode": "absent"}}',
        )
        settings = types.SimpleNamespace(mtp_enabled=True)

        caplog.set_level("WARNING", logger=model_loading.__name__)
        maybe_apply_pre_load_patches(path, model_settings=settings, for_vlm=True)

        sanitize_mock.assert_not_called()
        runtime_mock.assert_not_called()
        attach_mock.assert_not_called()
        assert calls == []
        assert "JANG metadata marks MTP weights as dropped" in caplog.text

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

    - return True when a safetensors index declares any key
      under the ``(language_model.|model.)?mtp.`` prefix family;
    - return False when no MTP-prefixed key is found;
    - return False on missing / unreadable inputs (callers treat that as
      "no MTP weights" — the conservative choice).
    """

    def _write_index(
        self,
        tmp_path,
        weight_map: dict,
        *,
        name: str = "model.safetensors.index.json",
    ) -> None:
        import json as _json

        (tmp_path / name).write_text(
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

    def test_returns_true_when_consolidated_index_has_mtp(self, tmp_path):
        self._write_index(
            tmp_path,
            {"language_model.mtp.fc.weight": "model.safetensors"},
            name="consolidated.safetensors.index.json",
        )
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is True

    def test_returns_true_when_index_has_model_language_model_mtp(self, tmp_path):
        # mlx-vlm HF-source layout before sanitize-time remap (oQ writes this).
        self._write_index(
            tmp_path,
            {"model.language_model.mtp.norm.weight": "model.safetensors"},
        )
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is True

    def test_returns_true_when_index_has_model_mtp_layers(self, tmp_path):
        self._write_index(
            tmp_path,
            {"model.mtp_layers.0.norm.weight": "model.safetensors"},
        )
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is True

    def test_returns_true_when_index_has_nextn_weights(self, tmp_path):
        self._write_index(
            tmp_path,
            {"model.layers.3.nextn_predictor.weight": "model.safetensors"},
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

    def test_returns_true_for_nextn_layout(self, tmp_path):
        # DeepSeek-V3-style checkpoints (GLM-5.2) keep the MTP head as an
        # extra decoder layer past num_hidden_layers, not under mtp.*.
        import json as _json

        (tmp_path / "config.json").write_text(
            _json.dumps({"num_hidden_layers": 78, "num_nextn_predict_layers": 1})
        )
        self._write_index(
            tmp_path,
            {
                "model.layers.0.self_attn.q_a_proj.weight": "model.safetensors",
                "model.layers.78.eh_proj.weight": "model.safetensors",
            },
        )
        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is True

    def test_returns_false_for_nextn_config_without_weights(self, tmp_path):
        # Config declares nextn layers but the checkpoint stripped them.
        import json as _json

        (tmp_path / "config.json").write_text(
            _json.dumps({"num_hidden_layers": 78, "num_nextn_predict_layers": 1})
        )
        self._write_index(
            tmp_path,
            {"model.layers.0.self_attn.q_a_proj.weight": "model.safetensors"},
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

    def test_reads_legacy_jang_index_for_mtp(self, tmp_path):
        (tmp_path / "model.jjqf.index.json").write_text(
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": {
                        "language_model.mtp.fc.weight": (
                            "model-00001-of-00001.jjqf.safetensors"
                        )
                    },
                }
            )
        )

        assert model_loading._checkpoint_has_mtp_weights(str(tmp_path)) is True

    def test_select_safetensors_weight_files_uses_complete_largest_set(self, tmp_path):
        stale = tmp_path / "model-00001-of-00001.safetensors"
        stale.write_text("stale")
        selected = [
            tmp_path / "model-00001-of-00002.safetensors",
            tmp_path / "model-00002-of-00002.safetensors",
        ]
        for path in selected:
            path.write_text("new")
        (tmp_path / "jangtq_runtime.safetensors").write_text("runtime")
        (tmp_path / "jang_imatrix.safetensors").write_text("calibration")

        assert model_loading.select_safetensors_weight_files(tmp_path) == selected

    def test_select_safetensors_weight_files_accepts_legacy_jang_shards(self, tmp_path):
        selected = [
            tmp_path / "model-00001-of-00002.mxq.safetensors",
            tmp_path / "model-00002-of-00002.mxq.safetensors",
        ]
        for path in selected:
            path.write_text("legacy")
        (tmp_path / "jangtq_stacked.safetensors").write_text("runtime")

        assert model_loading.select_safetensors_weight_files(tmp_path) == selected

    def test_select_safetensors_weight_files_excludes_jangpress_prestacked(
        self, tmp_path
    ):
        selected = tmp_path / "model.safetensors"
        selected.write_text("weights")
        (tmp_path / "jangpress-prestacked.safetensors").write_text("overlay")

        assert model_loading.select_safetensors_weight_files(tmp_path) == [selected]


class TestExpandPerLayerQuantKeys:
    """expand_per_layer_quant_keys adds runtime module-tree key variants."""

    def test_adds_language_model_prefix_for_bare_key(self):
        cfg = {
            "quantization": {
                "bits": 6,
                "group_size": 64,
                "lm_head": {"bits": 8, "group_size": 64},
            }
        }

        model_loading.expand_per_layer_quant_keys(cfg)

        assert cfg["quantization"]["language_model.lm_head"] == {
            "bits": 8,
            "group_size": 64,
        }

    def test_adds_swapped_prefix_variant_for_model_language_model_key(self):
        key = "model.language_model.layers.0.linear_attn.in_proj_qkv"
        cfg = {
            "quantization": {
                "bits": 6,
                "group_size": 64,
                key: {"bits": 8, "group_size": 64},
            }
        }

        model_loading.expand_per_layer_quant_keys(cfg)

        swapped = "language_model.model.layers.0.linear_attn.in_proj_qkv"
        assert swapped in cfg["quantization"]
        assert cfg["quantization"][swapped]["bits"] == 8
