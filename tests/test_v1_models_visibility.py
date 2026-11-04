# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/models visibility filtering (per-model hide + global helper hide)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from omlx.integrations.base import IntegrationContext
from omlx.integrations.copilot import copilot_provider_config
from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.server import ServerState, app
from omlx.settings import GlobalSettings


class _Pool:
    """Engine pool stub exposing a fixed model list via get_status()."""

    def __init__(self, models: list[dict]):
        self._models = models

    def resolve_model_id(self, model_id_or_alias: str, settings_manager) -> str:
        return model_id_or_alias

    def get_active_model_aliases(self, settings_manager) -> dict[str, str]:
        aliases: dict[str, str] = {}
        model_ids = {model["id"] for model in self._models}
        for model_id, settings in settings_manager.get_all_settings().items():
            if model_id in model_ids and settings.model_alias:
                aliases[model_id] = settings.model_alias
        return aliases

    def get_status(self) -> dict:
        return {
            "final_ceiling": 0,
            "current_model_memory": 0,
            "model_count": len(self._models),
            "loaded_count": 0,
            "models": self._models,
        }


def _model(model_id: str, config_model_type: str = "llama", **extra) -> dict:
    return {
        "id": model_id,
        "model_path": f"/models/{model_id}",
        "config_model_type": config_model_type,
        "model_type": "llm",
        "source_repo_id": None,
        **extra,
    }


def _state(models: list[dict], tmp_path, *, hide_helpers: bool = False) -> ServerState:
    state = ServerState()
    state.engine_pool = _Pool(models)
    state.settings_manager = ModelSettingsManager(base_path=tmp_path)
    gs = GlobalSettings()
    gs.model.hide_helper_models = hide_helpers
    state.global_settings = gs
    return state


def _list_ids(state, path="/v1/models") -> list[str]:
    with (
        patch("omlx.server._server_state", state),
        patch("omlx.server.get_max_context_window", return_value=None),
    ):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get(path)
    assert response.status_code == 200
    return [m["id"] for m in response.json()["data"]]


def test_lists_all_models_by_default(tmp_path):
    state = _state([_model("chat-a"), _model("chat-b")], tmp_path)
    assert _list_ids(state) == ["chat-a", "chat-b"]


def test_public_model_types_support_copilot_discovery(tmp_path):
    models = [
        _model("chat", model_type="llm"),
        _model("vision", model_type="vlm"),
        _model("embedding", model_type="embedding"),
        _model("reranker", model_type="reranker"),
        _model("image", model_type="image"),
        _model("speech", model_type="stt"),
    ]
    state = _state(models, tmp_path)
    state.settings_manager.set_settings(
        "vision", ModelSettings(model_alias="vision-alias")
    )
    state.settings_manager.save_profile(
        "vision", "fast", "Fast", None, {"temperature": 0.1}, expose_as_model=True
    )
    with (
        patch("omlx.server._server_state", state),
        patch("omlx.server.get_max_context_window", return_value=8192),
    ):
        response = TestClient(app).get("/v1/models")
    assert response.status_code == 200
    inventory = response.json()
    assert {model["id"]: model["model_type"] for model in inventory["data"]} == {
        "chat": "llm",
        "vision-alias": "vlm",
        "embedding": "embedding",
        "reranker": "reranker",
        "image": "image",
        "speech": "stt",
        "vision-alias:fast": "vlm",
    }
    with patch("omlx.integrations.base.requests.get") as get:
        get.return_value.json.return_value = inventory
        config = copilot_provider_config(
            IntegrationContext(host="127.0.0.1", port=8000)
        )
    assert [model["id"] for model in config["models"]] == [
        "chat",
        "vision-alias",
        "vision-alias:fast",
    ]
    assert config["models"][0]["maxContextWindowTokens"] == 8192
    assert config["models"][0]["capabilities"]["supports"]["vision"] is False
    assert all(
        model["capabilities"]["supports"]["vision"] for model in config["models"][1:]
    )


def test_claude_gateway_discovery_filters_and_preserves_standard_catalog(tmp_path):
    state = _state(
        [
            _model("owner/chat"),
            _model("small"),
            _model("hidden"),
            _model("embed", model_type="embedding"),
            _model("image", model_type="image"),
            _model("draft", is_helper=True),
            _model("unavailable", unavailable_reason="Cannot load model"),
        ],
        tmp_path,
    )
    state.settings_manager.set_settings(
        "owner/chat", ModelSettings(model_alias="my-chat", is_favorite=True)
    )
    state.settings_manager.set_settings("hidden", ModelSettings(is_hidden=True))
    state.settings_manager.save_profile(
        "owner/chat", "fast", "Fast", None, {}, expose_as_model=True
    )
    state.settings_manager.save_profile(
        "embed", "fast", "Fast", None, {}, expose_as_model=True
    )
    with (
        patch("omlx.server._server_state", state),
        patch(
            "omlx.server.get_max_context_window",
            side_effect=lambda name: 32768 if name == "small" else 65536,
        ),
    ):
        client = TestClient(app)
        standard_before = client.get("/v1/models").json()
        response = client.get("/claude/v1/models?limit=1000")
        standard_after = client.get("/v1/models").json()

    assert response.status_code == 200
    assert response.history == []
    assert response.json()["data"] == [
        {
            "id": "omlx-claude/my-chat",
            "display_name": "my-chat",
            "max_model_len": 65536,
        },
        {
            "id": "omlx-claude/my-chat:fast",
            "display_name": "my-chat:fast",
            "max_model_len": 65536,
        },
    ]
    assert [m["id"] for m in standard_before["data"]] == [
        "my-chat",
        "small",
        "embed",
        "image",
        "draft",
        "my-chat:fast",
        "embed:fast",
    ]
    assert [m["id"] for m in standard_after["data"]] == [
        m["id"] for m in standard_before["data"]
    ]
    assert all("display_name" not in m for m in standard_after["data"])


@pytest.mark.parametrize("context_window", [None, 49152, 65536])
def test_claude_gateway_discovers_nested_and_already_prefixed_ids(
    tmp_path, context_window
):
    state = _state(
        [_model("owner/chat"), _model("omlx-claude/chat"), _model("Claude-local")],
        tmp_path,
    )
    with (
        patch("omlx.server._server_state", state),
        patch("omlx.server.get_max_context_window", return_value=context_window),
    ):
        response = TestClient(app).get("/claude/v1/models?limit=1000")

    assert response.status_code == 200
    assert [m["id"] for m in response.json()["data"]] == [
        "omlx-claude/owner/chat",
        "omlx-claude/omlx-claude/chat",
        "omlx-claude/Claude-local",
    ]


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer test-key"}, {"x-api-key": "test-key"}],
)
@pytest.mark.parametrize(
    "path",
    ["/claude/v1/models", "/claude/v1/messages", "/claude/v1/messages/count_tokens"],
)
def test_claude_gateway_uses_inference_auth(tmp_path, headers, path):
    state = _state([], tmp_path)
    state.api_key = "test-key"
    with patch("omlx.server._server_state", state):
        client = TestClient(app)
        response = (
            client.get(path, headers=headers)
            if path.endswith("/models")
            else client.post(path, headers=headers, json={})
        )

    expected_status = 200 if path.endswith("/models") else 422
    assert response.status_code == (expected_status if headers else 401)


def test_per_model_hidden_excluded(tmp_path):
    state = _state([_model("chat-a"), _model("chat-b")], tmp_path)
    state.settings_manager.set_settings("chat-b", ModelSettings(is_hidden=True))
    ids = _list_ids(state)
    assert "chat-a" in ids
    assert "chat-b" not in ids


@pytest.mark.parametrize("path", ["/v1/models", "/claude/v1/models"])
def test_hidden_source_profile_excluded(tmp_path, path):
    state = _state([_model("chat-a")], tmp_path)
    state.settings_manager.set_settings("chat-a", ModelSettings(is_hidden=True))
    state.settings_manager.save_profile(
        "chat-a",
        "fast",
        "Fast",
        None,
        {"temperature": 0.1},
        expose_as_model=True,
    )
    assert _list_ids(state, path) == []


def test_claude_gateway_hidden_tier_source_excluded(tmp_path):
    state = _state([_model("hidden")], tmp_path)
    state.settings_manager.set_settings("hidden", ModelSettings(is_hidden=True))
    state.global_settings.claude_code.desktop_enabled = True
    state.global_settings.claude_code.opus_model = "hidden"

    assert _list_ids(state, "/claude/v1/models") == []


def test_unavailable_model_and_profiles_are_excluded(tmp_path):
    state = _state(
        [
            _model("chat-a"),
            _model(
                "qwen-next",
                config_model_type="qwen4_exp",
                unavailable_reason="QSA continuous batching is unavailable",
            ),
        ],
        tmp_path,
    )
    state.settings_manager.save_profile(
        "qwen-next",
        "fast",
        "Fast",
        None,
        {"temperature": 0.1},
        expose_as_model=True,
    )

    assert _list_ids(state) == ["chat-a"]


def test_hidden_ignored_when_helper_toggle_off(tmp_path):
    # A drafter is visible while the global toggle is off.
    models = [_model("chat-a"), _model("drafter", is_helper=True)]
    state = _state(models, tmp_path, hide_helpers=False)
    assert "drafter" in _list_ids(state)


def test_global_hide_helper_excludes_intrinsic_flag(tmp_path):
    # Discovery flags drafters (config marker / architecture) via is_helper.
    models = [_model("chat-a"), _model("drafter", is_helper=True)]
    state = _state(models, tmp_path, hide_helpers=True)
    ids = _list_ids(state)
    assert ids == ["chat-a"]


def test_global_hide_helper_excludes_helper_source_profile(tmp_path):
    models = [_model("chat-a"), _model("drafter", is_helper=True)]
    state = _state(models, tmp_path, hide_helpers=True)
    state.settings_manager.save_profile(
        "drafter",
        "fast",
        "Fast",
        None,
        {"temperature": 0.1},
        expose_as_model=True,
    )
    assert _list_ids(state) == ["chat-a"]


def test_global_hide_helper_excludes_referenced_draft(tmp_path):
    # A plain-LLM draft is a helper only because another model references it.
    models = [_model("chat-a"), _model("draft-llm", config_model_type="llama")]
    state = _state(models, tmp_path, hide_helpers=True)
    state.settings_manager.set_settings(
        "chat-a", ModelSettings(dflash_draft_model="/models/draft-llm")
    )
    ids = _list_ids(state)
    assert "chat-a" in ids
    assert "draft-llm" not in ids


def test_referenced_parent_stays_visible(tmp_path):
    # The referencing chat model itself is never treated as a helper.
    models = [_model("chat-a"), _model("draft-llm")]
    state = _state(models, tmp_path, hide_helpers=True)
    state.settings_manager.set_settings(
        "chat-a", ModelSettings(vlm_mtp_draft_model="/models/draft-llm")
    )
    assert "chat-a" in _list_ids(state)


def test_favorites_listed_first(tmp_path):
    models = [_model("chat-a"), _model("chat-b"), _model("chat-c")]
    state = _state(models, tmp_path)
    state.settings_manager.set_settings("chat-c", ModelSettings(is_favorite=True))
    assert _list_ids(state) == ["chat-c", "chat-a", "chat-b"]


def test_favorites_alphabetical_within_groups(tmp_path):
    models = [_model("chat-a"), _model("chat-b"), _model("chat-c"), _model("chat-d")]
    state = _state(models, tmp_path)
    state.settings_manager.set_settings("chat-b", ModelSettings(is_favorite=True))
    state.settings_manager.set_settings("chat-d", ModelSettings(is_favorite=True))
    assert _list_ids(state) == ["chat-b", "chat-d", "chat-a", "chat-c"]


def test_favorite_alias_listed_first(tmp_path):
    # The favorite set must track display ids, since an alias replaces the id.
    models = [_model("chat-a"), _model("chat-b")]
    state = _state(models, tmp_path)
    state.settings_manager.set_settings(
        "chat-b", ModelSettings(is_favorite=True, model_alias="zz-alias")
    )
    assert _list_ids(state) == ["zz-alias", "chat-a"]


def test_hidden_favorite_still_excluded(tmp_path):
    # Favorite is ordering-only and never overrides the hidden filter.
    models = [_model("chat-a"), _model("chat-b")]
    state = _state(models, tmp_path)
    state.settings_manager.set_settings(
        "chat-b", ModelSettings(is_hidden=True, is_favorite=True)
    )
    assert _list_ids(state) == ["chat-a"]
