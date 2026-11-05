# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.api.shared_models module."""

import json
import time

import pytest
from pydantic import ValidationError

from omlx.api.shared_models import (
    BaseUsage,
    IDPrefix,
    generate_id,
    get_unix_timestamp,
)


class TestIDPrefix:
    """Test cases for IDPrefix enum."""

    @pytest.mark.parametrize(
        "prefix, expected",
        [
            (IDPrefix.CHAT_COMPLETION, "chatcmpl"),
            (IDPrefix.COMPLETION, "cmpl"),
            (IDPrefix.MESSAGE, "msg"),
            (IDPrefix.EMBEDDING, "emb"),
            (IDPrefix.RERANK, "rerank"),
        ],
    )
    def test_prefix_value(self, prefix, expected):
        assert prefix.value == expected

    def test_all_prefixes_are_strings(self):
        """Test that all prefixes are strings."""
        for prefix in IDPrefix:
            assert isinstance(prefix.value, str)

    def test_prefix_is_str_subclass(self):
        """Test that IDPrefix is a str enum."""
        assert issubclass(IDPrefix, str)
        # Can use as string directly
        assert IDPrefix.CHAT_COMPLETION == "chatcmpl"


class TestGenerateId:
    """Test cases for generate_id function."""

    @pytest.mark.parametrize(
        "prefix, expected_prefix, suffix_length",
        [
            (IDPrefix.CHAT_COMPLETION, "chatcmpl-", 8),
            (IDPrefix.COMPLETION, "cmpl-", 8),
            (IDPrefix.MESSAGE, "msg_", 24),
            (IDPrefix.EMBEDDING, "emb-", 8),
            (IDPrefix.RERANK, "rerank-", 8),
        ],
    )
    def test_generate_default_id(self, prefix, expected_prefix, suffix_length):
        id_str = generate_id(prefix)
        assert id_str.startswith(expected_prefix)
        assert len(id_str) == len(expected_prefix) + suffix_length

    def test_generate_id_custom_length(self):
        """Test generating ID with custom length."""
        id_str = generate_id(IDPrefix.CHAT_COMPLETION, length=16)
        assert id_str.startswith("chatcmpl-")
        assert len(id_str) == len("chatcmpl-") + 16

    def test_generate_id_uniqueness(self):
        """Test that generated IDs are unique."""
        ids = [generate_id(IDPrefix.CHAT_COMPLETION) for _ in range(100)]
        assert len(set(ids)) == 100

    def test_generate_id_valid_hex(self):
        """Test that ID suffix is valid hex."""
        id_str = generate_id(IDPrefix.CHAT_COMPLETION)
        suffix = id_str.split("-")[1]
        # Should be valid hex
        int(suffix, 16)

    def test_generate_message_id_valid_hex(self):
        """Test that message ID suffix is valid hex."""
        id_str = generate_id(IDPrefix.MESSAGE)
        suffix = id_str.split("_")[1]
        # Should be valid hex
        int(suffix, 16)


class TestGetUnixTimestamp:
    """Test cases for get_unix_timestamp function."""

    def test_returns_integer(self):
        """Test that function returns an integer."""
        timestamp = get_unix_timestamp()
        assert isinstance(timestamp, int)

    def test_timestamp_is_current_time(self):
        """Test that timestamp is close to current time."""
        before = int(time.time())
        timestamp = get_unix_timestamp()
        after = int(time.time())
        assert before <= timestamp <= after

    def test_timestamp_is_positive(self):
        """Test that timestamp is a positive number."""
        timestamp = get_unix_timestamp()
        assert timestamp > 0

    def test_timestamp_is_reasonable(self):
        """Test that timestamp is in a reasonable range (after 2020)."""
        timestamp = get_unix_timestamp()
        # Unix timestamp for 2020-01-01
        year_2020 = 1577836800
        assert timestamp > year_2020


class TestBaseUsage:
    """Test cases for BaseUsage model."""

    def test_default_values(self):
        """Test default usage values."""
        usage = BaseUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_custom_values(self):
        """Test custom usage values."""
        usage = BaseUsage(
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_total_tokens_auto_calculated(self):
        """Test that total_tokens is auto-calculated."""
        usage = BaseUsage(
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert usage.total_tokens == 150

    def test_total_tokens_explicit(self):
        """Test explicit total_tokens value."""
        usage = BaseUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=200,  # Explicit, different from sum
        )
        # Explicit value should be preserved
        assert usage.total_tokens == 200

    def test_total_tokens_zero_when_all_zero(self):
        """Test total_tokens stays 0 when all tokens are 0."""
        usage = BaseUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        assert usage.total_tokens == 0

    def test_pydantic_model_dump(self):
        """Test that BaseUsage can be serialized."""
        usage = BaseUsage(
            prompt_tokens=100,
            completion_tokens=50,
        )
        data = usage.model_dump()
        assert data["prompt_tokens"] == 100
        assert data["completion_tokens"] == 50
        assert data["total_tokens"] == 150

    def test_pydantic_model_json(self):
        """Test that BaseUsage can be converted to JSON."""
        usage = BaseUsage(
            prompt_tokens=100,
            completion_tokens=50,
        )
        json_str = usage.model_dump_json()
        data = json.loads(json_str)
        assert data["prompt_tokens"] == 100
        assert data["completion_tokens"] == 50
        assert data["total_tokens"] == 150

    def test_pydantic_validation(self):
        """Test Pydantic validation."""
        # Valid creation
        usage = BaseUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.prompt_tokens == 100

        # Invalid type should raise error
        with pytest.raises(ValidationError):
            BaseUsage(prompt_tokens="not_a_number")

    def test_large_token_counts(self):
        """Test with large token counts."""
        usage = BaseUsage(
            prompt_tokens=100000,
            completion_tokens=50000,
        )
        assert usage.total_tokens == 150000

    def test_only_prompt_tokens(self):
        """Test with only prompt tokens."""
        usage = BaseUsage(prompt_tokens=100)
        assert usage.total_tokens == 100

    def test_only_completion_tokens(self):
        """Test with only completion tokens."""
        usage = BaseUsage(completion_tokens=50)
        assert usage.total_tokens == 50

    def test_input_output_tokens_in_json(self):
        """Test that input_tokens/output_tokens appear in JSON output."""
        usage = BaseUsage(prompt_tokens=100, completion_tokens=50)
        data = usage.model_dump()
        assert data["input_tokens"] == 100
        assert data["output_tokens"] == 50
