# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from omlx.model_settings import ModelSettings


@pytest.mark.asyncio
async def test_models_status_exposes_model_alias_metadata():
    from omlx.engine_pool import EnginePool
    from omlx.server import ServerState, list_models_status

    state = ServerState()
    state.engine_pool = MagicMock(spec=EnginePool)
    state.engine_pool.get_status.return_value = {
        "models": [
            {
                "id": "raw-image-model",
                "engine_type": "image",
                "model_type": "llm",
            }
        ]
    }
    state.engine_pool.get_active_model_aliases.return_value = {
        "raw-image-model": "friendly-image-name",
    }
    state.settings_manager = MagicMock()
    settings = ModelSettings(model_alias="friendly-image-name", max_tokens=1234)
    state.settings_manager.get_settings.return_value = settings
    state.settings_manager.get_settings_for_request.return_value = settings
    state.settings_manager.list_exposed_profile_models.return_value = [
        {
            "source_model_id": "raw-image-model",
            "model_id": "image-profile",
        }
    ]

    with patch("omlx.server._server_state", state):
        status = await list_models_status()

    assert status["models"][0]["id"] == "raw-image-model"
    assert status["models"][0]["model_alias"] == "friendly-image-name"
    assert status["models"][0]["engine_type"] == "image"
    assert status["models"][0]["max_tokens"] == 1234
    profile = next(m for m in status["models"] if m["id"] == "image-profile")
    assert "model_alias" not in profile
    assert profile["source_model_id"] == "raw-image-model"
    assert profile["max_tokens"] == 1234


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings", "discovered_default", "expected"),
    [
        (ModelSettings(enable_thinking=False), True, False),
        (ModelSettings(enable_thinking=True), False, True),
        (ModelSettings(chat_template_kwargs={"enable_thinking": True}), False, True),
        (
            ModelSettings(
                enable_thinking=False, chat_template_kwargs={"enable_thinking": True}
            ),
            True,
            False,
        ),
        (ModelSettings(), True, True),
        (ModelSettings(), None, None),
        (None, False, False),
    ],
)
async def test_models_status_exposes_effective_thinking(
    settings, discovered_default, expected
):
    from omlx.server import ServerState, list_models_status

    state = ServerState()
    state.engine_pool = MagicMock()
    state.engine_pool.get_status.return_value = {
        "models": [
            {"id": "model", "model_type": "llm", "thinking_default": discovered_default}
        ]
    }
    state.engine_pool.get_active_model_aliases.return_value = {}
    if settings is not None:
        state.settings_manager = MagicMock()
        state.settings_manager.get_settings_for_request.return_value = settings
        state.settings_manager.get_settings.return_value = settings
    with patch("omlx.server._server_state", state):
        status = await list_models_status()
    assert status["models"][0]["enable_thinking"] is expected
