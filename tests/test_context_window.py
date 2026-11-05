# SPDX-License-Identifier: Apache-2.0
"""Tests for context window validation feature."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from omlx.model_settings import ModelSettings


class TestGetMaxContextWindow:
    """Tests for get_max_context_window() priority logic."""

    def _make_server_state(
        self, global_max_ctx=32768, policy_cap=None
    ):
        """Create a mock server state with given global
        ``max_context_window`` fallback and optional
        ``max_context_window_policy`` cap."""
        from omlx.server import SamplingDefaults

        state = MagicMock()
        state.sampling = SamplingDefaults(
            max_context_window=global_max_ctx,
            max_context_window_policy=policy_cap,
        )
        state.settings_manager = None
        # Discovery-tier (#1308) lookups are exercised in TestGetMaxContextWindow
        # in test_server.py; nulling the pool here keeps these focused on the
        # per-model-setting → global fallback path.
        state.engine_pool = None
        return state

    def test_returns_global_default(self):
        """Test returns global default when no model settings."""
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=32768)
        with patch("omlx.server._server_state", state):
            result = get_max_context_window()
            assert result == 32768

    @pytest.mark.parametrize(
        ("global_limit", "model_limit", "expected"),
        [
            pytest.param(32768, 4096, 4096, id="model-overrides-global"),
            pytest.param(65536, None, 65536, id="unset-model-falls-back"),
        ],
    )
    def test_model_setting_priority(self, global_limit, model_limit, expected):
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=global_limit)
        mock_manager = MagicMock()
        mock_manager.get_settings_for_request.return_value = ModelSettings(
            max_context_window=model_limit
        )
        state.settings_manager = mock_manager

        with patch("omlx.server._server_state", state):
            result = get_max_context_window("test-model")
            assert result == expected

    def test_no_model_id_returns_global(self):
        """Test returns global when model_id is None."""
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=16384)
        with patch("omlx.server._server_state", state):
            result = get_max_context_window(None)
            assert result == 16384

    def _mount_native_and_policy(
        self, native_ctx: int | None, policy_cap: int | None
    ):
        """Mount a server state with a model that has the given native
        context length, the policy field set to ``policy_cap``, and no
        per-model override."""
        state = self._make_server_state(
            global_max_ctx=32768, policy_cap=policy_cap
        )
        mock_manager = MagicMock()
        mock_manager.get_settings_for_request.return_value = ModelSettings(
            max_context_window=None
        )
        state.settings_manager = mock_manager

        mock_pool = MagicMock()
        mock_entry = MagicMock()
        mock_entry.model_context_length = native_ctx
        mock_pool.get_entry.return_value = mock_entry
        state.engine_pool = mock_pool
        return state

    @pytest.mark.parametrize(
        ("native_context", "policy_cap", "expected"),
        [
            pytest.param(262_144, None, 262_144, id="policy-unset"),
            pytest.param(262_144, 128_000, 128_000, id="policy-clamps-native"),
            pytest.param(32_768, 128_000, 32_768, id="native-below-policy"),
            pytest.param(None, 16_000, 32_768, id="fallback-not-clamped"),
        ],
    )
    def test_policy_only_caps_native_context(
        self, native_context, policy_cap, expected
    ):
        from omlx.server import get_max_context_window

        state = self._mount_native_and_policy(
            native_ctx=native_context, policy_cap=policy_cap
        )
        with patch("omlx.server._server_state", state):
            assert get_max_context_window("test-model") == expected

    def test_per_model_override_escapes_policy(self):
        """A per-model override is the operator's explicit per-model
        choice; the global policy cap does NOT clamp it. This is the
        operator's escape hatch for individual models that should
        exceed the policy."""
        from omlx.server import get_max_context_window

        state = self._mount_native_and_policy(
            native_ctx=100_000, policy_cap=64_000
        )
        # Add a per-model override above both native and policy
        state.settings_manager = MagicMock()
        state.settings_manager.get_settings_for_request.return_value = ModelSettings(
            max_context_window=200_000
        )
        with patch("omlx.server._server_state", state):
            assert get_max_context_window("override-model") == 200_000, (
                "Per-model override must escape the policy clamp"
            )

    @pytest.mark.parametrize(
        ("engine_limit", "expected"),
        [
            pytest.param(77_817, 77_817, id="engine-lowers-native-limit"),
            pytest.param(300_000, 262_144, id="engine-cannot-raise-native-limit"),
        ],
    )
    def test_loaded_engine_plan_caps_the_limit(self, engine_limit, expected):
        """An engine that fits its context to memory (Splash) caps the limit,
        so a longer prompt gets the standard overflow error from oMLX."""
        from omlx.server import get_max_context_window

        state = self._mount_native_and_policy(native_ctx=262_144, policy_cap=None)
        engine = state.engine_pool.get_entry.return_value.engine
        engine.max_context_tokens = engine_limit
        with patch("omlx.server._server_state", state):
            assert get_max_context_window("splash-model") == expected


class TestValidateContextWindow:
    """Tests for validate_context_window()."""

    def _make_server_state(self, global_max_ctx=32768):
        from omlx.server import SamplingDefaults

        state = MagicMock()
        state.sampling = SamplingDefaults(max_context_window=global_max_ctx)
        state.settings_manager = None
        return state

    @pytest.mark.parametrize(
        "token_count",
        [
            pytest.param(500, id="under-limit"),
            pytest.param(1000, id="exact-limit"),
        ],
    )
    def test_passes_within_limit(self, token_count):
        from omlx.server import validate_context_window

        state = self._make_server_state(global_max_ctx=1000)
        with patch("omlx.server._server_state", state):
            validate_context_window(token_count)

    def test_raises_when_over_limit(self):
        """Test HTTPException raised when token count exceeds limit."""
        from omlx.server import validate_context_window

        state = self._make_server_state(global_max_ctx=1000)
        with patch("omlx.server._server_state", state):
            with pytest.raises(HTTPException) as exc_info:
                validate_context_window(1001)
            # Overflow markers that clients match before compacting.
            detail = exc_info.value.detail.lower()
            assert exc_info.value.status_code == 400
            assert exc_info.value.code == "context_length_exceeded"
            assert "prompt is too long: 1001 tokens > 1000" in detail
            assert "maximum context length" in detail

    def test_raises_with_model_specific_limit(self):
        """Test uses model-specific limit when available."""
        from omlx.server import validate_context_window

        state = self._make_server_state(global_max_ctx=32768)
        mock_manager = MagicMock()
        mock_manager.get_settings_for_request.return_value = ModelSettings(
            max_context_window=100
        )
        state.settings_manager = mock_manager

        with patch("omlx.server._server_state", state):
            with pytest.raises(HTTPException) as exc_info:
                validate_context_window(200, "test-model")
            assert exc_info.value.status_code == 400
            assert "200 tokens > 100" in exc_info.value.detail


class TestCountChatTokens:
    """Tests for BatchedEngine.count_chat_tokens()."""

    def test_count_chat_tokens(self):
        """Test token counting with mocked tokenizer."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        engine._tokenizer = mock_tokenizer

        # Mock model (not gpt_oss)
        engine._model = MagicMock(spec=[])
        engine._enable_thinking = None

        messages = [{"role": "user", "content": "Hello"}]
        count = engine.count_chat_tokens(messages)

        assert count == 5
        mock_tokenizer.apply_chat_template.assert_called_once()
        mock_tokenizer.encode.assert_called_once_with("formatted prompt")

    def test_count_chat_tokens_with_tools(self):
        """Test token counting includes tools in template."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt with tools"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7]
        engine._tokenizer = mock_tokenizer
        engine._model = MagicMock(spec=[])
        engine._enable_thinking = None

        messages = [{"role": "user", "content": "Call a tool"}]
        tools = [{"type": "function", "function": {"name": "test"}}]
        count = engine.count_chat_tokens(messages, tools)

        assert count == 7


class TestGetRerankMaxLength:
    """get_rerank_max_length() mirrors get_embedding_max_length but must not
    consult the LLM-oriented global sampling default: only an explicit request
    or the model's own per-model max_context_window override applies."""

    def test_request_value_wins(self):
        from omlx.server import get_rerank_max_length

        with patch("omlx.server.get_model_settings_for_request") as gm:
            assert get_rerank_max_length("m", 1024) == 1024
            gm.assert_not_called()

    def test_per_model_override_applies(self):
        from omlx.server import get_rerank_max_length

        settings = ModelSettings(max_context_window=4096)
        with patch(
            "omlx.server.get_model_settings_for_request", return_value=settings
        ):
            assert get_rerank_max_length("m") == 4096

    def test_no_override_defers_to_model_default(self):
        """None lets the reranker resolve its own declared model_max_length."""
        from omlx.server import get_rerank_max_length

        with patch(
            "omlx.server.get_model_settings_for_request",
            return_value=ModelSettings(max_context_window=None),
        ):
            assert get_rerank_max_length("m") is None

    def test_missing_settings_defers_to_model_default(self):
        from omlx.server import get_rerank_max_length

        with patch("omlx.server.get_model_settings_for_request", return_value=None):
            assert get_rerank_max_length("m") is None
