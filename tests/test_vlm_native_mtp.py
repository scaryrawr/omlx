# SPDX-License-Identifier: Apache-2.0
"""Fast wiring tests for native-head MTP through ``VLMBatchedEngine``.

The VLM model and checkpoint index are lightweight stand-ins: no model weights
are loaded, and no mlx-vlm generation helper is allowed to run.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from omlx.engine.vlm import _report_native_vlm_mtp_readiness
from omlx.models.vlm import VLMModelAdapter
from omlx.patches.mlx_lm_mtp import batch_generator
from omlx.scheduler import _record_native_mtp_request_eligibility
from omlx.utils.model_loading import (
    _checkpoint_has_mtp_weights,
    _native_vlm_mtp_eligible,
)

_TARGET_CONFIG = {
    "model_type": "qwen3_5",
    "vision_config": {},
    "text_config": {"mtp_num_hidden_layers": 1},
}


def _write_checkpoint_index(tmp_path, tensor_keys: list[str]) -> None:
    weight_map = {key: "model-00001-of-00001.safetensors" for key in tensor_keys}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


def _make_adapter() -> VLMModelAdapter:
    """Build a minimal mlx-vlm-shaped model for adapter state tests."""
    language_model = SimpleNamespace(
        args=SimpleNamespace(),
        model=SimpleNamespace(layers=[]),
    )
    vlm_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5"),
        language_model=language_model,
    )
    return VLMModelAdapter(vlm_model)


# ---------------------------------------------------------------------------
# Phase 1: construction-time config/settings/tensor gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "settings_enabled", "tensor_keys", "expected"),
    [
        pytest.param(
            _TARGET_CONFIG,
            True,
            ["language_model.mtp.layers.0.self_attn.q_proj.weight"],
            True,
            id="target-positive",
        ),
        pytest.param(
            _TARGET_CONFIG,
            True,
            ["language_model.model.embed_tokens.weight"],
            False,
            id="head-stripped-negative",
        ),
        pytest.param(
            _TARGET_CONFIG,
            False,
            ["language_model.mtp.layers.0.self_attn.q_proj.weight"],
            False,
            id="settings-off-negative",
        ),
        pytest.param(
            {
                "model_type": "qwen3_5",
                "vision_config": {},
                "text_config": {},
            },
            True,
            ["language_model.mtp.layers.0.self_attn.q_proj.weight"],
            False,
            id="config-heads-missing-negative",
        ),
        pytest.param(
            {"model_type": "llama", "mtp_num_hidden_layers": 1},
            True,
            ["language_model.mtp.layers.0.self_attn.q_proj.weight"],
            False,
            id="unsupported-family-negative",
        ),
    ],
)
def test_native_vlm_mtp_eligibility_matrix(
    tmp_path,
    config,
    settings_enabled,
    tensor_keys,
    expected,
):
    """Native VLM decode requires config, setting, and persisted head tensors."""
    (tmp_path / "config.json").write_text(json.dumps(config))
    _write_checkpoint_index(tmp_path, tensor_keys)

    has_mtp_weights = _checkpoint_has_mtp_weights(tmp_path)
    settings = SimpleNamespace(mtp_enabled=settings_enabled)

    assert (
        _native_vlm_mtp_eligible(
            config,
            config.get("model_type"),
            settings,
            has_mtp_weights=has_mtp_weights,
        )
        is expected
    )


# ---------------------------------------------------------------------------
# Phase 2: per-row text/embeddings gate and UID lifecycle
# ---------------------------------------------------------------------------


def test_native_vlm_mtp_per_row_gate_and_uid_cleanup():
    """Text rows may speculate; vision/mixed batches cannot; retirement cleans up."""
    adapter = _make_adapter()
    text_uid = 101
    vision_uid = 202

    _record_native_mtp_request_eligibility(
        adapter,
        text_uid,
        SimpleNamespace(vlm_inputs_embeds=None),
    )
    _record_native_mtp_request_eligibility(
        adapter,
        vision_uid,
        SimpleNamespace(vlm_inputs_embeds=object()),
    )

    assert adapter.native_mtp_allowed_for_uids([text_uid]) is True
    assert adapter.native_mtp_allowed_for_uids([vision_uid]) is False
    assert adapter.native_mtp_allowed_for_uids([text_uid, vision_uid]) is False

    # Exercise BatchGenerator's shared singleton/row-wise eligibility choke
    # point, not only the adapter's set membership helper.
    model = SimpleNamespace(
        mtp=object(),
        mtp_forward=lambda *_args, **_kwargs: None,
        _omlx_mtp_decode_enabled=True,
        native_mtp_allowed_for_uids=adapter.native_mtp_allowed_for_uids,
    )
    generation_batch = SimpleNamespace(
        model=model,
        uids=[text_uid],
        logits_processors=[],
    )
    assert batch_generator._mtp_common_eligible(generation_batch) is True

    generation_batch.uids = [vision_uid]
    assert batch_generator._mtp_common_eligible(generation_batch) is False

    generation_batch.uids = [text_uid, vision_uid]
    assert batch_generator._mtp_common_eligible(generation_batch) is False

    adapter._uid_rope_deltas[vision_uid] = 3.0
    adapter.unregister_rope_delta(vision_uid)
    assert vision_uid not in adapter._uid_rope_deltas
    assert vision_uid not in adapter._native_mtp_disabled_uids
    assert adapter.native_mtp_allowed_for_uids([vision_uid]) is True


# ---------------------------------------------------------------------------
# Phase 3: strict post-load readiness and fail-soft fallback
# ---------------------------------------------------------------------------


def test_native_vlm_mtp_readiness_logs_ready(caplog):
    """A bound head, decode marker, adapter method, and tensor index report ready."""
    adapter = SimpleNamespace(
        _language_model=SimpleNamespace(
            mtp=object(),
            _omlx_mtp_decode_enabled=True,
        ),
        mtp_forward=lambda *_args, **_kwargs: None,
    )

    with caplog.at_level(logging.INFO, logger="omlx.engine.vlm"):
        ready = _report_native_vlm_mtp_readiness(
            "/models/qwen3.8-27b-oq6-mtp",
            adapter,
            has_mtp_weights=True,
        )

    assert ready is True
    assert "Native VLM MTP ready" in caplog.text


@pytest.mark.parametrize(
    ("has_head", "has_adapter_bridge"),
    [(False, True), (True, False)],
    ids=("missing-loaded-head", "missing-adapter-bridge"),
)
def test_native_vlm_mtp_readiness_failure_keeps_standard_decode(
    caplog,
    has_head,
    has_adapter_bridge,
):
    """An incomplete strict load warns and remains on GenerationBatch decode."""
    mtp_head = object() if has_head else None
    language_model = SimpleNamespace(
        mtp=mtp_head,
        _omlx_mtp_decode_enabled=True,
    )
    adapter_attrs = {
        "_language_model": language_model,
        "mtp": mtp_head,
    }
    if has_adapter_bridge:
        adapter_attrs["mtp_forward"] = lambda *_args, **_kwargs: None
    adapter = SimpleNamespace(**adapter_attrs)

    with caplog.at_level(logging.WARNING, logger="omlx.engine.vlm"):
        ready = _report_native_vlm_mtp_readiness(
            "/models/incomplete-native-mtp",
            adapter,
            has_mtp_weights=True,
        )

    assert ready is False
    assert "readiness gate failed" in caplog.text
    assert "standard decode remains active" in caplog.text

    generation_batch = SimpleNamespace(
        model=adapter,
        uids=[1],
        logits_processors=[],
    )
    assert batch_generator._mtp_common_eligible(generation_batch) is False
