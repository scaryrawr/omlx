# SPDX-License-Identifier: Apache-2.0
"""Tests for the enable_thinking toggle and detect_thinking_default heuristic."""

import json

import pytest

from omlx.model_discovery import (
    detect_preserve_thinking,
    detect_reasoning_effort,
    detect_thinking_default,
)
from omlx.model_settings import ModelSettings

# ---------------------------------------------------------------------------
# detect_thinking_default
# ---------------------------------------------------------------------------


class TestDetectThinkingDefault:
    """Test chat template heuristic for thinking default detection."""

    @pytest.mark.parametrize(
        "template, expected",
        [
            pytest.param(
                "{%- if enable_thinking is false -%}\n"
                "  ... suppress thinking ...\n"
                "{%- endif -%}",
                True,
                id="qwen-default-on",
            ),
            pytest.param(
                "{%- set thinking = enable_thinking | default(false) -%}",
                False,
                id="gemma-default-off",
            ),
            pytest.param(
                "{%- set enable_thinking = enable_thinking | default(true) -%}\n"
                "{%- set preserve_thinking = preserve_thinking | default(false) -%}",
                True,
                id="explicit-default-on-with-other-default-off",
            ),
            pytest.param(
                "{%- if default(enable_thinking) -%}think{%- endif -%}",
                False,
                id="parenthesized-enable-thinking",
            ),
            pytest.param(
                "{{ messages[0].content }}", None, id="no-enable-thinking"
            ),
            pytest.param(
                "{%- if enable_thinking == 'maybe' -%}hmm{%- endif -%}",
                None,
                id="unrecognized-pattern",
            ),
        ],
    )
    def test_template_default(self, tmp_path, template, expected):
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_thinking_default(tmp_path) is expected

    def test_no_template_files_returns_none(self, tmp_path):
        """Directory without any template file returns None."""
        assert detect_thinking_default(tmp_path) is None

    def test_laguna_template_uses_recommended_serving_default(self, tmp_path):
        """Laguna's effective default follows Poolside's serving recommendation."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "laguna"}))
        (tmp_path / "chat_template.jinja").write_text(
            "{%- set enable_thinking = enable_thinking | default(false) -%}\n"
            "{%- if not enable_thinking -%}</think>{%- else -%}<think>{%- endif -%}"
        )

        assert detect_thinking_default(tmp_path) is True

    @pytest.mark.parametrize("model_type", ["laguna", "llama"])
    def test_config_without_thinking_template_returns_none(self, tmp_path, model_type):
        """Config metadata alone is not evidence that a template accepts the flag."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))

        assert detect_thinking_default(tmp_path) is None

    def test_jinja_file_takes_priority_over_tokenizer_config(self, tmp_path):
        """chat_template.jinja is preferred over tokenizer_config.json."""
        # Jinja file says Qwen pattern (True)
        (tmp_path / "chat_template.jinja").write_text(
            "{%- if enable_thinking is false -%}suppress{%- endif -%}"
        )
        # tokenizer_config says Gemma pattern (False)
        tc = {"chat_template": "{%- set t = enable_thinking | default(false) -%}"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))

        assert detect_thinking_default(tmp_path) is True

    def test_falls_back_to_tokenizer_config(self, tmp_path):
        """When no jinja file exists, reads from tokenizer_config.json."""
        tc = {"chat_template": "{%- if enable_thinking is false -%}ok{%- endif -%}"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))
        assert detect_thinking_default(tmp_path) is True

    def test_tokenizer_config_without_chat_template_key(self, tmp_path):
        """tokenizer_config.json without chat_template key returns None."""
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"model_type": "llama"}))
        assert detect_thinking_default(tmp_path) is None

    def test_malformed_tokenizer_config_returns_none(self, tmp_path):
        """Malformed JSON in tokenizer_config.json returns None gracefully."""
        (tmp_path / "tokenizer_config.json").write_text("not valid json{{{")
        assert detect_thinking_default(tmp_path) is None


# ---------------------------------------------------------------------------
# ModelSettings.enable_thinking field
# ---------------------------------------------------------------------------


class TestModelSettingsEnableThinking:
    """Test enable_thinking field on ModelSettings dataclass."""

    def test_default_is_none(self):
        ms = ModelSettings()
        assert ms.enable_thinking is None

    def test_set_to_true(self):
        ms = ModelSettings(enable_thinking=True)
        assert ms.enable_thinking is True

    def test_set_to_false(self):
        ms = ModelSettings(enable_thinking=False)
        assert ms.enable_thinking is False


# ---------------------------------------------------------------------------
# detect_preserve_thinking
# ---------------------------------------------------------------------------


class TestDetectPreserveThinking:
    """Test chat template heuristic for preserve_thinking support detection."""

    def test_qwen36_pattern_returns_true(self, tmp_path):
        """Qwen 3.6+ pattern: template references preserve_thinking kwarg."""
        template = (
            "{%- if (preserve_thinking is defined and preserve_thinking is true) "
            "or (loop.index0 > ns.last_query_index) -%}\n"
            "  <think>{{ reasoning_content }}</think>\n"
            "{%- endif -%}"
        )
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_preserve_thinking(tmp_path) is True

    def test_no_preserve_thinking_returns_none(self, tmp_path):
        """Template without preserve_thinking reference returns None."""
        template = "{%- if enable_thinking is false -%}suppress{%- endif -%}"
        (tmp_path / "chat_template.jinja").write_text(template)
        assert detect_preserve_thinking(tmp_path) is None

    def test_no_template_files_returns_none(self, tmp_path):
        """Directory without any template file returns None."""
        assert detect_preserve_thinking(tmp_path) is None

    def test_jinja_file_takes_priority_over_tokenizer_config(self, tmp_path):
        """chat_template.jinja is preferred over tokenizer_config.json."""
        (tmp_path / "chat_template.jinja").write_text(
            "{%- if preserve_thinking -%}keep{%- endif -%}"
        )
        tc = {"chat_template": "{{ messages[0].content }}"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))

        assert detect_preserve_thinking(tmp_path) is True

    def test_falls_back_to_tokenizer_config(self, tmp_path):
        """When no jinja file exists, reads from tokenizer_config.json."""
        tc = {"chat_template": "{%- if preserve_thinking -%}keep{%- endif -%}"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))
        assert detect_preserve_thinking(tmp_path) is True

    def test_tokenizer_config_without_chat_template_key(self, tmp_path):
        """tokenizer_config.json without chat_template key returns None."""
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"model_type": "llama"}))
        assert detect_preserve_thinking(tmp_path) is None

    def test_malformed_tokenizer_config_returns_none(self, tmp_path):
        """Malformed JSON in tokenizer_config.json returns None gracefully."""
        (tmp_path / "tokenizer_config.json").write_text("not valid json{{{")
        assert detect_preserve_thinking(tmp_path) is None


# ---------------------------------------------------------------------------
# ModelSettings.preserve_thinking field
# ---------------------------------------------------------------------------


class TestModelSettingsPreserveThinking:
    """Test preserve_thinking field on ModelSettings dataclass."""

    def test_default_is_none(self):
        ms = ModelSettings()
        assert ms.preserve_thinking is None

    def test_set_to_true(self):
        ms = ModelSettings(preserve_thinking=True)
        assert ms.preserve_thinking is True

    def test_set_to_false(self):
        ms = ModelSettings(preserve_thinking=False)
        assert ms.preserve_thinking is False


@pytest.mark.parametrize(
    "template, levels",
    [
        ("{{ messages }}", None),
        ("Reasoning: {{ reasoning_effort | default('medium') }}", []),
        # Qwen3.8 raises on anything outside its three levels.
        (
            "{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}"
            "{%- elif resolved_reasoning_effort == 'low' %}",
            ["low", "medium", "xhigh"],
        ),
        # GLM-5.3 maps every other value to max; Hunyuan adds no_think.
        (
            "{%- set e = reasoning_effort if reasoning_effort in ['low', 'high'] %}",
            ["low", "high"],
        ),
        (
            "{%- if reasoning_effort not in ['high', 'low', 'no_think'] %}",
            ["low", "high"],
        ),
        # Muse Spark Open checks the keys of a level-to-strength map.
        (
            "{%- set reasoning_effort_map = "
            "{'low': 8, 'medium': 32, 'high': 256, 'xhigh': 512} -%}"
            "{%- if not (reasoning_effort is string and "
            "reasoning_effort in reasoning_effort_map) -%}",
            ["low", "medium", "high", "xhigh"],
        ),
    ],
)
def test_detect_reasoning_effort_reads_the_accepted_levels(tmp_path, template, levels):
    tc = {"chat_template": template}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tc))
    assert detect_reasoning_effort(tmp_path) == levels
