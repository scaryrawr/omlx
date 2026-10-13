# SPDX-License-Identifier: Apache-2.0
"""Immutable native CPU fixtures against the actual pinned legacy runtime."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from omlx.patches import mlx_vlm_minicpm47_compat as compat


def test_payload_matches_sealed_source():
    manifest = compat._inventory()
    assert manifest["revision"] == "70b9ddc6fad39a7fb7639c9941e2847f3cacd414"
    for path, expected in manifest["verification"].items():
        assert compat._hash(compat._VENDOR / path) == expected


@pytest.mark.parametrize("state", ["legacy", "native"])
def test_preimages_require_the_entire_contract(tmp_path, monkeypatch, state):
    paths = {"model.py": b"native model", "verifier.py": b"native verifier"}
    guards = {}
    for name, contents in paths.items():
        path = tmp_path / name
        path.write_bytes(contents if state == "native" else b"old " + contents)
        native = compat.hashlib.sha256(contents).hexdigest()
        legacy = compat.hashlib.sha256(b"old " + contents).hexdigest()
        guards[name] = {"native": native, "legacy": legacy}
    monkeypatch.setattr(
        compat.importlib.metadata,
        "distribution",
        lambda _: type(
            "Distribution", (), {"locate_file": lambda self, p: tmp_path / p}
        )(),
    )
    assert compat._runtime_state({"guards": guards}) == state
    (tmp_path / "verifier.py").write_bytes(b"partial or unknown verifier")
    with pytest.raises(RuntimeError, match="Unrecognized or partial"):
        compat._runtime_state({"guards": guards})


def test_payload_corruption_fails_before_runtime_mutation(tmp_path, monkeypatch):
    (tmp_path / "source_manifest.json").write_text(
        json.dumps(
            {
                "files": {"model.py": "not-a-valid-hash"},
                "fragments": {},
                "guards": {},
            }
        )
    )
    monkeypatch.setattr(compat, "_ROOT", tmp_path)
    monkeypatch.setattr(compat, "_VENDOR", tmp_path)
    with pytest.raises(RuntimeError, match="payload is incomplete"):
        compat.apply_mlx_vlm_minicpm47_compat_patch()


def test_manifest_requires_json_object(tmp_path, monkeypatch):
    (tmp_path / "source_manifest.json").write_text("[]")
    monkeypatch.setattr(compat, "_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="manifest must be a JSON object"):
        compat.apply_mlx_vlm_minicpm47_compat_patch()


def test_non_normalized_routing_is_explicitly_unsupported():
    compat.validate_target_config({"text_config": {}})
    with pytest.raises(ValueError, match="normalized top-k"):
        compat.validate_target_config({"text_config": {"norm_topk_prob": False}})


def test_complete_native_is_a_noop_before_any_runtime_import(monkeypatch):
    monkeypatch.setattr(compat, "_STATE", None)
    monkeypatch.setattr(compat, "_runtime_state", lambda _: "native")
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(
            compat.importlib,
            "import_module",
            lambda _: pytest.fail("A complete native implementation needs no patch"),
        )
        assert compat.apply_mlx_vlm_minicpm47_compat_patch() is False
        assert compat.apply_mlx_vlm_minicpm47_compat_patch() is False


def test_failed_install_restores_only_owned_runtime_state():
    script = """
import sys
import mlx.core as mx
mx.set_default_device(mx.cpu)
from unittest.mock import patch
from omlx.patches import mlx_vlm_minicpm47_compat as compat
import mlx_vlm.models as models
import mlx_vlm.models.switch_layers as switch
import mlx_vlm.utils as utils
import mlx_vlm.prompt_utils as prompts
before = {name: getattr(utils, name) for name in
          (*compat.UTILS_FUNCTIONS, 'get_model_and_args')}
paths = list(models.__path__)
formatter = prompts.MessageFormatter.format_message
with patch.object(compat, '_helper_module', side_effect=RuntimeError('fixture failure')):
    try:
        compat.apply_mlx_vlm_minicpm47_compat_patch()
    except RuntimeError as exc:
        assert str(exc) == 'fixture failure'
    else:
        raise AssertionError('Installation failure was swallowed')
assert compat._STATE is None
assert list(models.__path__) == paths
assert all(getattr(utils, name) is value for name, value in before.items())
assert prompts.MessageFormatter.format_message is formatter
assert 'minicpmv4_7' not in prompts.MODEL_CONFIG
assert not any(hasattr(switch, name) for name in compat.EXPERT_SYMBOLS)
assert not any(name.startswith(compat._MODEL_PREFIX) for name in sys.modules)
assert compat.apply_mlx_vlm_minicpm47_compat_patch()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("corruption", ["module", "symbol", "export", "path", "prompt"])
def test_repeated_patch_rejects_partial_owned_runtime(corruption):
    script = f"""
import sys
import mlx.core as mx
mx.set_default_device(mx.cpu)
from unittest.mock import patch
from omlx.patches import mlx_vlm_minicpm47_compat as compat
import mlx_vlm.models as models
import mlx_vlm.models.switch_layers as switch
import mlx_vlm.prompt_utils as prompts
import mlx_vlm.utils as utils
assert compat.apply_mlx_vlm_minicpm47_compat_patch()
# Other families may wrap shared dispatchers without replacing our classes.
original = utils.get_model_and_args
with patch.object(utils, 'get_model_and_args',
                  side_effect=lambda *a, **k: original(*a, **k)):
    assert not compat.apply_mlx_vlm_minicpm47_compat_patch()
corruption = {corruption!r}
if corruption == 'module':
    del sys.modules[compat._HELPER_PREFIX]
elif corruption == 'symbol':
    delattr(switch, compat.EXPERT_SYMBOLS[-1])
elif corruption == 'export':
    compat._helper_module().DSparkDraftModel = object()
elif corruption == 'path':
    models.__path__.remove(str(compat._VENDOR / 'mlx_vlm/models'))
else:
    del prompts.MODEL_CONFIG['minicpmv4_7']
try:
    compat.apply_mlx_vlm_minicpm47_compat_patch()
except RuntimeError as exc:
    assert 'MiniCPM' in str(exc)
else:
    raise AssertionError('Partially removed runtime was silently accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_authoritative_model_processor_and_vision_fixtures_on_legacy_cpu():
    script = """
import sys, types
import mlx.core as mx
mx.set_default_device(mx.cpu)
from omlx.patches import mlx_vlm_minicpm47_compat as compat
from mlx_vlm.models.switch_layers import OffloadedSwitchGLU
from mlx_vlm.speculative.drafters.dspark import DSparkDraftModel
import mlx_vlm.utils as utils
original_offload = OffloadedSwitchGLU.__init__
original_dspark = DSparkDraftModel
original_loader = utils.get_model_and_args
assert compat.apply_mlx_vlm_minicpm47_compat_patch()
assert not compat.apply_mlx_vlm_minicpm47_compat_patch()
assert OffloadedSwitchGLU.__init__ is original_offload
from mlx_vlm.speculative.drafters.dspark import DSparkDraftModel as generic_dspark
assert generic_dspark is original_dspark
assert compat.dspark_model_class() is not original_dspark
# Other architecture resolution still calls the original registry.
assert utils.get_model_and_args({'model_type':'qwen3_5_moe'}) == original_loader(
    {'model_type':'qwen3_5_moe'}
)
tests = types.ModuleType('mlx_vlm.tests')
tests.__path__ = [str(compat._VENDOR / 'mlx_vlm/tests')]
sys.modules['mlx_vlm.tests'] = tests
# The immutable fixtures import the upstream public helper spelling. Bind
# that spelling to the private serving helper only inside this subprocess.
from unittest.mock import patch
import importlib
helper_aliases = {
    'mlx_vlm.speculative.drafters.dspark.config': importlib.import_module(
        compat._HELPER_PREFIX + '.config'
    ),
    'mlx_vlm.speculative.drafters.dspark.dspark': importlib.import_module(
        compat._HELPER_PREFIX + '.dspark'
    ),
}
import pytest
with patch.dict(sys.modules, helper_aliases):
    outcome = pytest.main([
        str(compat._VENDOR / 'mlx_vlm/tests/test_minicpmv4_7.py'),
        str(compat._VENDOR / 'mlx_vlm/tests/test_minicpmv4_7_processor.py'),
        str(compat._VENDOR / 'mlx_vlm/tests/test_minicpmv4_7_vision.py'),
        'tests/minicpm47_native_cases.py',
        '-q', '--tb=short',
    ])
sys.exit(outcome)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
