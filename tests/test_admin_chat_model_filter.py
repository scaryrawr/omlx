# SPDX-License-Identifier: Apache-2.0
"""Regression tests for admin chat model dropdown filtering."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omlx.model_settings import ModelSettings
from omlx.settings import GlobalSettings

CHAT_TEMPLATE = (
    Path(__file__).parent.parent / "omlx" / "admin" / "templates" / "chat.html"
)


@pytest.mark.asyncio
async def test_models_endpoints_omit_inactive_colliding_aliases():
    """Inactive aliases should not create duplicate /v1/models display IDs."""
    from omlx.server import ServerState, list_models, list_models_status

    state = ServerState()
    state.engine_pool = MagicMock()
    state.engine_pool.get_status.return_value = {
        "models": [
            {
                "id": "chat-model",
                "engine_type": "batched",
                "model_type": "llm",
            },
            {
                "id": "raw-image-model",
                "engine_type": "image",
                "model_type": "llm",
            },
        ]
    }
    state.engine_pool.get_active_model_aliases.return_value = {}
    state.global_settings = GlobalSettings()
    state.global_settings.integrations.markitdown_expose_model = False
    state.settings_manager = MagicMock()
    state.settings_manager.get_settings.side_effect = lambda model_id: ModelSettings(
        model_alias="chat-model" if model_id == "raw-image-model" else None
    )

    with patch("omlx.server._server_state", state):
        models = await list_models()
        status = await list_models_status()

    assert [m.id for m in models.data] == ["chat-model", "raw-image-model"]
    raw_image = next(m for m in status["models"] if m["id"] == "raw-image-model")
    assert "model_alias" not in raw_image


def test_chat_template_uses_status_metadata_for_filtering():
    """The UI filter should use /v1/models/status model and engine metadata."""
    source = CHAT_TEMPLATE.read_text()

    assert "modelTypeMap" in source
    assert "modelEngineTypeMap" in source
    assert "fetch('/v1/models/status'" in source
    assert "map[m.id] = m.model_type || 'llm';" in source
    assert "engineMap[m.id] = m.engine_type || '';" in source
    assert "map[m.model_alias] === undefined" in source
    assert source.index("await this.fetchModelTypes();") < source.index(
        "this.availableModels = this.dedupeAvailableModels("
    )
    assert "t !== 'image'" in source
    assert "e !== 'image'" in source
    assert "t !== 'embedding'" in source
    assert "t !== 'reranker'" in source
    assert "t !== 'audio_tts'" in source
    assert "t !== 'audio_sts'" in source
