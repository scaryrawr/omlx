# SPDX-License-Identifier: Apache-2.0
"""Tests for OpenAI Responses API models and utilities."""

import copy
import json

import pytest

from omlx.api.responses_models import (
    InputItem,
    ResponseObject,
    ResponsesRequest,
    ResponsesTool,
    ResponseUsage,
    TextConfig,
    TextFormatConfig,
)
from omlx.api.responses_utils import (
    ResponseStateCorruptError,
    ResponseStateNotFoundError,
    ResponseStore,
    build_function_call_output_item,
    build_message_output_item,
    build_reasoning_output_item,
    build_response_store_record,
    build_response_usage,
    convert_responses_input_to_messages,
    convert_responses_tools,
    convert_stored_response_to_messages,
    format_sse_event,
    normalize_chat_messages_for_response_store,
    normalize_response_output_to_messages,
    split_namespace_tool_name,
)
from omlx.api.shared_models import IDPrefix, generate_id

# =============================================================================
# ID Generation Tests
# =============================================================================


class TestIDGeneration:
    def test_response_id_prefix(self):
        rid = generate_id(IDPrefix.RESPONSE)
        assert rid.startswith("resp_")
        assert len(rid) == 29  # "resp_" + 24 hex chars

    def test_function_call_id_prefix(self):
        fid = generate_id(IDPrefix.FUNCTION_CALL)
        assert fid.startswith("fc_")
        assert len(fid) == 11  # "fc_" + 8 hex chars

    def test_reasoning_id_prefix(self):
        rid = generate_id(IDPrefix.REASONING)
        assert rid.startswith("rs_")
        assert len(rid) == 27  # "rs_" + 24 hex chars

    def test_unique_ids(self):
        ids = {generate_id(IDPrefix.RESPONSE) for _ in range(100)}
        assert len(ids) == 100


# =============================================================================
# Input Conversion Tests
# =============================================================================


class TestConvertResponsesInput:
    def test_string_input(self):
        messages = convert_responses_input_to_messages("Hello world")
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello world"}

    def test_none_input(self):
        messages = convert_responses_input_to_messages(None)
        assert messages == []

    def test_instructions_prepended(self):
        messages = convert_responses_input_to_messages(
            "Hi", instructions="You are helpful"
        )
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are helpful"}
        assert messages[1] == {"role": "user", "content": "Hi"}

    def test_message_items(self):
        items = [
            InputItem(type="message", role="user", content="Hello"),
            InputItem(type="message", role="assistant", content="Hi there"),
            InputItem(type="message", role="user", content="How are you?"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["content"] == "How are you?"

    def test_easy_input_message_no_type(self):
        """EasyInputMessage has no type field — just role and content."""
        items = [
            InputItem(role="user", content="Hello"),
            InputItem(role="assistant", content="Hi"),
        ]
        assert items[0].type is None  # No type set
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"

    def test_easy_input_message_from_dict(self):
        """Verify Pydantic parses EasyInputMessage dict (no type field)."""
        raw = {"role": "user", "content": "Hello from dict"}
        item = InputItem(**raw)
        assert item.type is None
        assert item.role == "user"
        messages = convert_responses_input_to_messages([item])
        assert messages[0]["content"] == "Hello from dict"

    def test_developer_role_maps_to_system(self):
        items = [
            InputItem(type="message", role="developer", content="Be concise"),
            InputItem(type="message", role="user", content="Hi"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise"
        assert messages[1]["role"] == "user"

    def test_developer_role_no_type(self):
        """EasyInputMessage with developer role (no type field)."""
        items = [InputItem(role="developer", content="Be brief")]
        messages = convert_responses_input_to_messages(items)
        assert messages[0]["role"] == "system"

    def test_instructions_and_developer_merged(self):
        """instructions + developer role in input → single system message."""
        items = [
            InputItem(role="developer", content="Working dir: /foo"),
            InputItem(role="user", content="Hello"),
        ]
        messages = convert_responses_input_to_messages(
            items, instructions="You are helpful"
        )
        # Should be exactly 2: one merged system + one user
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "You are helpful" in messages[0]["content"]
        assert "Working dir: /foo" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_multiple_developer_messages_merged(self):
        """Multiple developer messages all merge into one system message."""
        items = [
            InputItem(role="developer", content="Rule 1"),
            InputItem(role="developer", content="Rule 2"),
            InputItem(role="user", content="Hi"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Rule 1" in messages[0]["content"]
        assert "Rule 2" in messages[0]["content"]

    def test_developer_position_can_be_deferred(self):
        """Server path can preserve developer/system position until template probe."""
        items = [
            InputItem(role="user", content="Hi"),
            InputItem(role="developer", content="Plan mode"),
        ]
        messages = convert_responses_input_to_messages(
            items,
            consolidate_system_messages=False,
        )
        assert [m["role"] for m in messages] == ["user", "system"]
        assert messages[1]["content"] == "Plan mode"

    def test_function_call_items(self):
        items = [
            InputItem(type="message", role="user", content="What's the weather?"),
            InputItem(
                type="function_call",
                call_id="call_abc",
                name="get_weather",
                arguments='{"location": "Paris"}',
            ),
            InputItem(
                type="function_call_output",
                call_id="call_abc",
                output='{"temp": 22}',
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        # function_call → assistant with tool_calls
        assert messages[1]["role"] == "assistant"
        assert len(messages[1]["tool_calls"]) == 1
        assert messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        # arguments should be parsed as dict (not string) for Jinja2 chat templates
        assert messages[1]["tool_calls"][0]["function"]["arguments"] == {
            "location": "Paris"
        }
        # function_call_output → tool message
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call_abc"
        assert messages[2]["content"] == '{"temp": 22}'

    def test_multiple_function_calls_grouped(self):
        """Multiple consecutive function_calls should be grouped into one assistant message."""
        items = [
            InputItem(type="message", role="user", content="Do two things"),
            InputItem(
                type="function_call",
                call_id="call_1",
                name="fn_a",
                arguments="{}",
            ),
            InputItem(
                type="function_call",
                call_id="call_2",
                name="fn_b",
                arguments="{}",
            ),
            InputItem(
                type="function_call_output",
                call_id="call_1",
                output="result_a",
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        # user, assistant (2 tool_calls), tool (output for call_1)
        assert len(messages) == 3
        assert messages[1]["role"] == "assistant"
        assert len(messages[1]["tool_calls"]) == 2

    def test_assistant_message_merged_with_function_call(self):
        """Assistant message followed by function_call should produce a single
        assistant message (not two), preventing duplicate assistant turns that
        confuse chat templates."""
        items = [
            InputItem(type="message", role="user", content="Do something"),
            InputItem(
                type="message",
                role="assistant",
                content=[{"type": "output_text", "text": ""}],
            ),
            InputItem(
                type="function_call",
                call_id="call_abc",
                name="exec_command",
                arguments='{"cmd": "ls"}',
            ),
            InputItem(
                type="function_call_output",
                call_id="call_abc",
                output="file1.txt\nfile2.txt",
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        # user, assistant (content + tool_calls merged), tool
        assert len(messages) == 3
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == ""
        assert len(messages[1]["tool_calls"]) == 1
        assert messages[1]["tool_calls"][0]["function"]["name"] == "exec_command"
        assert messages[2]["role"] == "tool"

    def test_assistant_with_content_merged_with_function_call(self):
        """Assistant message with non-empty content merges with tool_calls."""
        items = [
            InputItem(type="message", role="user", content="Check weather"),
            InputItem(
                type="message",
                role="assistant",
                content=[{"type": "output_text", "text": "Let me check."}],
            ),
            InputItem(
                type="function_call",
                call_id="call_1",
                name="get_weather",
                arguments='{"city": "Seoul"}',
            ),
            InputItem(
                type="function_call_output",
                call_id="call_1",
                output='{"temp": 20}',
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 3
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Let me check."
        assert len(messages[1]["tool_calls"]) == 1

    def test_multi_round_tool_calls_no_duplicate_assistant(self):
        """Multiple rounds of tool calls should not create duplicate assistant
        messages (the pattern that causes model EOS after many rounds)."""
        items = [
            InputItem(type="message", role="user", content="Explore the code"),
        ]
        # Simulate 3 rounds of tool calls (Codex pattern)
        for i in range(3):
            items.extend(
                [
                    InputItem(
                        type="message",
                        role="assistant",
                        content=[{"type": "output_text", "text": ""}],
                    ),
                    InputItem(
                        type="function_call",
                        call_id=f"call_{i}",
                        name="exec_command",
                        arguments=f'{{"cmd": "cmd_{i}"}}',
                    ),
                    InputItem(
                        type="function_call_output",
                        call_id=f"call_{i}",
                        output=f"result_{i}",
                    ),
                ]
            )
        messages = convert_responses_input_to_messages(items)
        # Count assistant messages — should be exactly 3 (one per round)
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 3
        # Each assistant message should have tool_calls
        for msg in assistant_msgs:
            assert "tool_calls" in msg
            assert len(msg["tool_calls"]) == 1

    def test_output_text_content_type(self):
        """output_text content type should be parsed correctly."""
        items = [
            InputItem(
                type="message",
                role="assistant",
                content=[{"type": "output_text", "text": "Hello world"}],
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        assert messages[0]["content"] == "Hello world"

    def test_content_parts_array(self):
        items = [
            InputItem(
                type="message",
                role="user",
                content=[
                    {"type": "input_text", "text": "Hello"},
                    {"type": "input_text", "text": "World"},
                ],
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        assert messages[0]["content"] == "Hello\nWorld"

    def test_input_image_preserved_as_content_list(self):
        """input_image parts should be preserved as list for VLM processing."""
        items = [
            InputItem(
                type="message",
                role="user",
                content=[
                    {"type": "input_text", "text": "What is in this image?"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,abc123",
                        "detail": "high",
                    },
                ],
            ),
        ]
        messages = convert_responses_input_to_messages(items, preserve_images=True)
        content = messages[0]["content"]
        # Content should be a list (not flattened to string) when images present
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "What is in this image?"}
        assert content[1] == {
            "type": "input_image",
            "image_url": "data:image/png;base64,abc123",
            "detail": "high",
        }

    def test_input_image_url_field_fallback(self):
        """input_image with 'url' field instead of 'image_url' should also work."""
        items = [
            InputItem(
                type="message",
                role="user",
                content=[
                    {"type": "input_text", "text": "Describe"},
                    {"type": "input_image", "url": "https://example.com/img.png"},
                ],
            ),
        ]
        messages = convert_responses_input_to_messages(items, preserve_images=True)
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[1]["image_url"] == "https://example.com/img.png"
        assert content[1]["detail"] == "auto"  # default

    def test_input_image_dropped_without_preserve_images(self):
        """Text engines get a string, as /v1/chat/completions does (#4069)."""
        items = [
            InputItem(
                type="message",
                role="user",
                content=[
                    {"type": "input_text", "text": "Reply OK."},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                ],
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        assert messages[0]["content"] == "Reply OK."

    def test_input_audio_preserved_as_content_list(self):
        """Responses input_audio parts should be preserved for VLM processing."""
        items = [
            InputItem(
                type="message",
                role="user",
                content=[
                    {"type": "input_text", "text": "What is in this recording?"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "abc123", "format": "wav"},
                    },
                ],
            )
        ]

        messages = convert_responses_input_to_messages(items)
        content = messages[0]["content"]

        assert isinstance(content, list)
        assert content[1] == {
            "type": "input_audio",
            "input_audio": {"data": "abc123", "format": "wav"},
        }

    def test_input_file_preserved_as_file_part(self):
        """Responses input_file maps to internal file part for preprocessing."""
        items = [
            InputItem(
                type="message",
                role="user",
                content=[
                    {"type": "input_text", "text": "Summarize this."},
                    {
                        "type": "input_file",
                        "filename": "notes.txt",
                        "file_data": "data:text/plain;base64,ZA==",
                    },
                ],
            )
        ]

        messages = convert_responses_input_to_messages(items)
        content = messages[0]["content"]

        assert isinstance(content, list)
        assert content[1] == {
            "type": "file",
            "file": {
                "filename": "notes.txt",
                "file_data": "data:text/plain;base64,ZA==",
            },
        }

    def test_text_only_content_parts_flattened(self):
        """Content with only text parts should still be flattened to string."""
        items = [
            InputItem(
                type="message",
                role="user",
                content=[
                    {"type": "input_text", "text": "Hello"},
                    {"type": "input_text", "text": "World"},
                ],
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        assert isinstance(messages[0]["content"], str)
        assert messages[0]["content"] == "Hello\nWorld"

    def test_previous_messages_prepended(self):
        prev = [{"role": "assistant", "content": "Previous answer"}]
        messages = convert_responses_input_to_messages(
            "Follow up", previous_messages=prev
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "user"

    def test_instructions_with_previous(self):
        prev = [{"role": "assistant", "content": "Previous"}]
        messages = convert_responses_input_to_messages(
            "New question",
            instructions="System prompt",
            previous_messages=prev,
        )
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_previous_system_messages_merge_with_current_instructions(self):
        prev = [{"role": "system", "content": "Previous instruction"}]
        messages = convert_responses_input_to_messages(
            "New question",
            instructions="Current instruction",
            previous_messages=prev,
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Previous instruction" in messages[0]["content"]
        assert "Current instruction" in messages[0]["content"]

    def test_current_function_call_does_not_merge_into_previous_assistant(self):
        prev = [{"role": "assistant", "content": "Previous answer"}]
        items = [
            InputItem(
                type="function_call",
                call_id="call_next",
                name="exec_command",
                arguments='{"cmd":"pwd"}',
            )
        ]
        messages = convert_responses_input_to_messages(items, previous_messages=prev)
        assert len(messages) == 2
        assert messages[0]["content"] == "Previous answer"
        assert messages[1]["tool_calls"][0]["id"] == "call_next"

    def test_reasoning_then_message_attaches_reasoning_content(self):
        items = [
            InputItem(
                type="reasoning",
                summary=[{"type": "summary_text", "text": "step-by-step"}],
            ),
            InputItem(type="message", role="assistant", content="Final answer"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Final answer"
        assert messages[0]["reasoning_content"] == "step-by-step"

    def test_reasoning_then_function_call_attaches_reasoning_content(self):
        # Tool-call sequence: reasoning → function_call → function_call_output.
        # Reasoning must survive into the synthesized assistant tool_calls
        # message instead of being dropped on the floor.
        items = [
            InputItem(
                type="reasoning",
                summary=[{"type": "summary_text", "text": "need weather"}],
            ),
            InputItem(
                type="function_call",
                call_id="call_w",
                name="get_weather",
                arguments='{"city":"Paris"}',
            ),
            InputItem(type="function_call_output", call_id="call_w", output="sunny"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert messages[0]["reasoning_content"] == "need weather"
        assert messages[1]["role"] == "tool"
        assert messages[1]["content"] == "sunny"

    def test_reasoning_with_multi_part_summary_joined(self):
        items = [
            InputItem(
                type="reasoning",
                summary=[
                    {"type": "summary_text", "text": "line1"},
                    {"type": "summary_text", "text": "line2"},
                ],
            ),
            InputItem(type="message", role="assistant", content="ok"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert messages[0]["reasoning_content"] == "line1\nline2"

    def test_reasoning_orphan_then_tool_calls_only(self):
        # reasoning followed by function_call but no function_call_output
        # in the same input — reasoning still lands on the tool_calls message
        # when the final flush runs.
        items = [
            InputItem(
                type="reasoning",
                summary=[{"type": "summary_text", "text": "thinking"}],
            ),
            InputItem(
                type="function_call",
                call_id="call_x",
                name="lookup",
                arguments="{}",
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 1
        assert messages[0]["tool_calls"][0]["function"]["name"] == "lookup"
        assert messages[0]["reasoning_content"] == "thinking"


IMAGE_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="


class TestResponseHistoryNormalization:
    @pytest.mark.parametrize("serialized", [False, True])
    def test_tool_argument_objects_survive_store_and_replay(self, serialized):
        arguments = {"cmd": "pwd", "options": {"login": False}}
        messages = [
            {
                "role": "assistant",
                "reasoning_content": "Inspect first.",
                "partial": False,
                "tool_calls": [
                    {
                        "id": "call_shell",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "namespace": "functions",
                            "arguments": (
                                json.dumps(arguments) if serialized else arguments
                            ),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_shell", "content": "/tmp"},
        ]
        original = copy.deepcopy(messages)
        normalized = normalize_chat_messages_for_response_store(messages)
        replayed = convert_responses_input_to_messages(None, previous_messages=messages)
        assert normalized == replayed
        assert normalized[0]["tool_calls"][0]["function"]["arguments"] == arguments
        assert normalized[0]["tool_calls"][0]["function"]["namespace"] == "functions"
        assert normalized[0]["reasoning_content"] == "Inspect first."
        assert "partial" not in normalized[0]
        assert normalized[1] == original[1]
        assert messages == original
        normalized[0]["tool_calls"][0]["function"]["arguments"]["cmd"] = "changed"
        assert messages == original

    def test_normalization_does_not_replace_malformed_arguments(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "exec_command", "arguments": "{bad json"}}
                ],
            }
        ]
        assert normalize_chat_messages_for_response_store(messages) == messages

    def test_null_tool_calls_are_preserved(self):
        messages = [{"role": "assistant", "content": "Done", "tool_calls": None}]
        assert normalize_chat_messages_for_response_store(messages) == messages


class TestFunctionCallOutputMultimodal:
    """Images in function_call_output lists route to VLM, not the prompt (#2989)."""

    def _tool_round(self, outputs):
        items = [InputItem(type="message", role="user", content="take screenshots")]
        for i in range(len(outputs)):
            items.append(
                InputItem(
                    type="function_call",
                    call_id=f"call_{i}",
                    name="screenshot",
                    arguments="{}",
                )
            )
        for i, output in enumerate(outputs):
            items.append(
                InputItem(
                    type="function_call_output",
                    call_id=f"call_{i}",
                    output=output,
                )
            )
        return items

    def test_image_extracted_to_user_message(self):
        output = [
            {"type": "input_text", "text": "screenshot result"},
            {"type": "input_image", "detail": "auto", "image_url": IMAGE_URI},
        ]
        messages = convert_responses_input_to_messages(
            self._tool_round([output]), preserve_images=True
        )
        assert [m["role"] for m in messages] == ["user", "assistant", "tool", "user"]
        assert messages[2]["content"] == "screenshot result"
        assert messages[3]["content"] == [
            {"type": "input_image", "image_url": IMAGE_URI, "detail": "auto"}
        ]

    def test_parallel_outputs_keep_tool_messages_contiguous(self):
        output = [
            {"type": "input_text", "text": "shot"},
            {"type": "input_image", "detail": "auto", "image_url": IMAGE_URI},
        ]
        messages = convert_responses_input_to_messages(
            self._tool_round([output, output]), preserve_images=True
        )
        # Images flush once after the tool run, never between tool messages
        assert [m["role"] for m in messages] == [
            "user",
            "assistant",
            "tool",
            "tool",
            "user",
        ]
        assert len(messages[4]["content"]) == 2
        for msg in messages[2:4]:
            assert IMAGE_URI not in msg["content"]

    def test_images_flush_before_next_tool_round(self):
        items = self._tool_round(
            [[{"type": "input_image", "detail": "auto", "image_url": IMAGE_URI}]]
        )
        items.extend(
            [
                InputItem(
                    type="function_call",
                    call_id="call_next",
                    name="lookup",
                    arguments="{}",
                ),
                InputItem(
                    type="function_call_output",
                    call_id="call_next",
                    output="done",
                ),
            ]
        )
        messages = convert_responses_input_to_messages(items, preserve_images=True)
        assert [m["role"] for m in messages] == [
            "user",
            "assistant",
            "tool",
            "user",
            "assistant",
            "tool",
        ]
        assert messages[3]["content"][0]["type"] == "input_image"
        assert messages[5]["content"] == "done"

    def test_placeholder_without_preserve_images(self):
        output = [
            {"type": "input_text", "text": "screenshot result"},
            {"type": "input_image", "detail": "auto", "image_url": IMAGE_URI},
        ]
        messages = convert_responses_input_to_messages(self._tool_round([output]))
        assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
        assert messages[2]["content"] == "screenshot result\n(see attached image)"
        assert IMAGE_URI not in json.dumps(messages)

    def test_plain_json_list_falls_back_to_dumps(self):
        messages = convert_responses_input_to_messages(
            self._tool_round([[1, 2, 3]]), preserve_images=True
        )
        assert messages[2]["content"] == "[1, 2, 3]"
        messages = convert_responses_input_to_messages(
            self._tool_round([["a", "b"]]), preserve_images=True
        )
        assert messages[2]["content"] == '["a", "b"]'

    def test_text_only_typed_list_extracted(self):
        messages = convert_responses_input_to_messages(
            self._tool_round([[{"type": "input_text", "text": "hello"}]]),
            preserve_images=True,
        )
        assert messages[2]["content"] == "hello"


# =============================================================================
# Tool Conversion Tests
# =============================================================================


class TestConvertResponsesTools:
    def test_none_tools(self):
        assert convert_responses_tools(None) is None

    def test_empty_tools(self):
        assert convert_responses_tools([]) is None

    def test_single_tool(self):
        tools = [
            ResponsesTool(
                type="function",
                name="get_weather",
                description="Get weather info",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            )
        ]
        result = convert_responses_tools(tools)
        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather info"
        assert "parameters" in result[0]["function"]

    def test_strict_field(self):
        tools = [ResponsesTool(name="fn", strict=True)]
        result = convert_responses_tools(tools)
        assert result[0]["function"]["strict"] is True

    def test_minimal_tool(self):
        tools = [ResponsesTool(name="simple_fn")]
        result = convert_responses_tools(tools)
        assert result[0]["function"]["name"] == "simple_fn"
        assert "description" not in result[0]["function"]
        assert "parameters" not in result[0]["function"]

    def test_non_function_tools_skipped(self):
        """Non-function tool types (local_shell, mcp, etc.) should be skipped."""
        tools = [
            ResponsesTool(type="local_shell"),
            ResponsesTool(type="function", name="fn_a"),
            ResponsesTool(type="mcp"),
        ]
        result = convert_responses_tools(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "fn_a"

    def test_only_non_function_tools_returns_none(self):
        tools = [ResponsesTool(type="local_shell")]
        result = convert_responses_tools(tools)
        assert result is None

    def test_namespace_tools_are_expanded(self):
        """Namespace groups hold client-executed function tools (#3371)."""
        aliases = {}
        tools = [
            ResponsesTool(
                type="namespace",
                name="mcp__demo__",
                description="Demo MCP server",
                tools=[
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                    {"type": "function", "name": "get_time"},
                ],
            )
        ]
        result = convert_responses_tools(tools, aliases)
        assert result is not None
        # Wire names join the namespace like Codex's join_tool_name().
        assert [t["function"]["name"] for t in result] == [
            "mcp__demo__get_weather",
            "mcp__demo__get_time",
        ]
        assert result[0]["type"] == "function"
        assert result[0]["function"]["description"] == (
            "Get the current weather for a city."
        )
        assert result[0]["function"]["parameters"]["required"] == ["city"]
        assert aliases == {
            "mcp__demo__get_weather": ("mcp__demo__", "get_weather"),
            "mcp__demo__get_time": ("mcp__demo__", "get_time"),
        }

    def test_namespace_wire_name_survives_collision(self):
        """A flat tool already holding the joined name must not be shadowed."""
        aliases = {}
        tools = [
            ResponsesTool(type="function", name="mcp__demo__get_weather"),
            ResponsesTool(
                type="namespace",
                name="mcp__demo__",
                tools=[{"type": "function", "name": "get_weather"}],
            ),
        ]
        result = convert_responses_tools(tools, aliases)
        assert [t["function"]["name"] for t in result] == [
            "mcp__demo__get_weather",
            "mcp__demo__get_weather_2",
        ]
        assert aliases == {"mcp__demo__get_weather_2": ("mcp__demo__", "get_weather")}

    def test_split_namespace_tool_name(self):
        aliases = {"mcp__demo__get_weather": ("mcp__demo__", "get_weather")}
        assert split_namespace_tool_name("mcp__demo__get_weather", aliases) == (
            "mcp__demo__",
            "get_weather",
        )
        # Flat calls, and calls the model invented, pass through unchanged.
        assert split_namespace_tool_name("get_weather", aliases) == (
            None,
            "get_weather",
        )
        assert split_namespace_tool_name("get_weather") == (None, "get_weather")

    def test_namespace_serializes_only_when_set(self):
        namespaced = build_function_call_output_item(
            name="get_weather",
            arguments='{"city": "Paris"}',
            call_id="call_1",
            namespace="mcp__demo__",
        ).model_dump()
        assert namespaced["name"] == "get_weather"
        assert namespaced["namespace"] == "mcp__demo__"
        flat = build_function_call_output_item(
            name="get_weather", arguments="{}", call_id="call_2"
        ).model_dump()
        assert "namespace" not in flat
        # Other optional fields keep serializing as null, as before.
        assert flat["role"] is None and flat["summary"] is None

    def test_namespace_without_usable_members_contributes_nothing(self):
        tools = [
            ResponsesTool(type="namespace", name="empty__"),
            ResponsesTool(type="namespace", name="junk__", tools=["not-a-tool"]),
            ResponsesTool(type="function", name="fn_a"),
        ]
        result = convert_responses_tools(tools)
        assert [t["function"]["name"] for t in result] == ["fn_a"]


# =============================================================================
# InputItem Validation Tests
# =============================================================================


class TestInputItemOutputSerialization:
    """InputItem should accept list/dict output; dict serializes, list survives."""

    def test_list_output_preserved(self):
        # Lists pass through so multimodal parts stay extractable (#2989)
        item = InputItem(
            type="function_call_output",
            call_id="call_123",
            output=[{"type": "input_image", "image_url": "data:image/jpeg;base64,abc"}],
        )
        assert isinstance(item.output, list)
        assert item.output[0]["type"] == "input_image"

    def test_dict_output_serialized_to_json(self):
        item = InputItem(
            type="function_call_output",
            call_id="call_456",
            output={"result": "success", "data": [1, 2, 3]},
        )
        assert isinstance(item.output, str)
        parsed = json.loads(item.output)
        assert parsed["result"] == "success"

    def test_string_output_unchanged(self):
        item = InputItem(
            type="function_call_output",
            call_id="call_789",
            output="plain text result",
        )
        assert item.output == "plain text result"

    def test_none_output_unchanged(self):
        item = InputItem(type="function_call_output", call_id="call_000")
        assert item.output is None


# =============================================================================
# Response Building Tests
# =============================================================================


class TestBuildOutputItems:
    def test_message_output_item(self):
        item = build_message_output_item("Hello!")
        assert item.type == "message"
        assert item.role == "assistant"
        assert item.content[0].text == "Hello!"
        assert item.content[0].type == "output_text"
        assert item.status == "completed"
        assert item.id.startswith("msg_")

    def test_function_call_output_item(self):
        item = build_function_call_output_item(
            name="get_weather",
            arguments='{"location": "Paris"}',
            call_id="call_abc",
        )
        assert item.type == "function_call"
        assert item.name == "get_weather"
        assert item.call_id == "call_abc"
        assert item.arguments == '{"location": "Paris"}'
        assert item.id.startswith("fc_")

    def test_response_usage(self):
        usage = build_response_usage(100, 50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.input_tokens_details.cached_tokens == 0
        assert usage.output_tokens_details.reasoning_tokens == 0

    def test_response_usage_with_cached_tokens(self):
        usage = build_response_usage(100, 50, cached_tokens=30)
        assert usage.input_tokens == 100
        assert usage.input_tokens_details.cached_tokens == 30

    def test_response_usage_with_reasoning_tokens(self):
        usage = build_response_usage(100, 50, reasoning_tokens=20, cached_tokens=30)
        assert usage.output_tokens == 50
        assert usage.output_tokens_details.reasoning_tokens == 20
        assert usage.input_tokens_details.cached_tokens == 30

    def test_build_reasoning_output_item(self):
        item = build_reasoning_output_item("Step 1: foo. Step 2: bar.")
        assert item.type == "reasoning"
        assert item.status == "completed"
        assert item.id.startswith("rs_")
        assert len(item.summary) == 1
        assert item.summary[0].type == "summary_text"
        assert item.summary[0].text == "Step 1: foo. Step 2: bar."

    def test_build_reasoning_output_item_empty_text(self):
        item = build_reasoning_output_item("")
        assert item.type == "reasoning"
        assert item.summary == []


class TestResponseObject:
    def test_defaults(self):
        resp = ResponseObject(model="test-model")
        assert resp.object == "response"
        assert resp.id.startswith("resp_")
        assert resp.status == "completed"
        assert resp.output == []

    def test_with_output(self):
        msg = build_message_output_item("Hi")
        resp = ResponseObject(
            model="test-model",
            output=[msg],
            usage=build_response_usage(10, 5),
        )
        assert len(resp.output) == 1
        assert resp.usage.total_tokens == 15

    def test_incomplete_details_on_truncation(self):
        # A max_output_tokens truncation must surface as status="incomplete"
        # with incomplete_details.reason, so clients can distinguish an
        # incomplete turn from a natural stop (the Responses API has no
        # finish_reason field).
        resp = ResponseObject(
            model="test-model",
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        )
        assert resp.status == "incomplete"
        assert resp.incomplete_details == {"reason": "max_output_tokens"}
        dumped = resp.model_dump(exclude_none=True)
        assert dumped["incomplete_details"] == {"reason": "max_output_tokens"}

    def test_completed_response_omits_incomplete_details(self):
        resp = ResponseObject(model="test-model")
        dumped = resp.model_dump(exclude_none=True)
        assert "incomplete_details" not in dumped


# =============================================================================
# SSE Event Formatting Tests
# =============================================================================


class TestFormatSSEEvent:
    def test_dict_data(self):
        result = format_sse_event("response.created", {"id": "resp_123"})
        assert result.startswith("event: response.created\n")
        assert "data: " in result
        assert result.endswith("\n\n")
        data_line = result.split("\n")[1]
        parsed = json.loads(data_line.removeprefix("data: "))
        assert parsed["id"] == "resp_123"

    def test_string_data(self):
        result = format_sse_event("response.output_text.delta", '{"delta": "hi"}')
        assert "event: response.output_text.delta\n" in result
        assert 'data: {"delta": "hi"}\n' in result

    def test_pydantic_model(self):
        usage = ResponseUsage(input_tokens=10, output_tokens=5)
        result = format_sse_event("test", usage)
        assert "event: test\n" in result
        parsed = json.loads(result.split("\n")[1].removeprefix("data: "))
        assert parsed["input_tokens"] == 10


# =============================================================================
# Response Store Tests
# =============================================================================


class TestResponseStore:
    def test_put_and_get(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"id": "resp_1", "output": []})
        assert store.get("resp_1") is not None
        assert store.get("resp_1")["id"] == "resp_1"

    def test_get_nonexistent(self):
        store = ResponseStore()
        assert store.get("nonexistent") is None

    def test_delete(self):
        store = ResponseStore()
        store.put("resp_1", {"id": "resp_1"})
        assert store.delete("resp_1") is True
        assert store.get("resp_1") is None

    def test_delete_nonexistent(self):
        store = ResponseStore()
        assert store.delete("nonexistent") is False

    def test_lru_eviction(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"id": "resp_1"})
        store.put("resp_2", {"id": "resp_2"})
        store.put("resp_3", {"id": "resp_3"})
        # This should evict resp_1 (oldest)
        store.put("resp_4", {"id": "resp_4"})
        assert store.get("resp_1") is None
        assert store.get("resp_2") is not None
        assert store.get("resp_4") is not None
        assert len(store) == 3

    def test_lru_access_refreshes(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"id": "resp_1"})
        store.put("resp_2", {"id": "resp_2"})
        store.put("resp_3", {"id": "resp_3"})
        # Access resp_1 to refresh it
        store.get("resp_1")
        # Now resp_2 should be evicted (oldest after refresh)
        store.put("resp_4", {"id": "resp_4"})
        assert store.get("resp_1") is not None  # Refreshed, still here
        assert store.get("resp_2") is None  # Evicted
        assert store.get("resp_4") is not None

    def test_update_existing(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"id": "resp_1", "v": 1})
        store.put("resp_1", {"id": "resp_1", "v": 2})
        assert store.get("resp_1")["v"] == 2
        assert len(store) == 1

    def test_disk_persistence_round_trip(self, tmp_path):
        state_dir = tmp_path / "response-state"
        store = ResponseStore(max_size=10, state_dir=state_dir)
        public = {"id": "resp_1", "created_at": 1, "output": []}
        record = build_response_store_record(
            public, [{"role": "user", "content": "Hi"}], []
        )
        store.put("resp_1", record)

        reloaded = ResponseStore(max_size=10, state_dir=state_dir)
        assert reloaded.get("resp_1")["id"] == "resp_1"
        assert reloaded.get_record("resp_1")["input_messages"][0]["content"] == "Hi"

    def test_resolve_chain_messages(self, tmp_path):
        state_dir = tmp_path / "response-state"
        store = ResponseStore(max_size=10, state_dir=state_dir)
        store.put(
            "resp_1",
            build_response_store_record(
                {"id": "resp_1", "created_at": 1, "output": []},
                [{"role": "user", "content": "First"}],
                [{"role": "assistant", "content": "One"}],
            ),
        )
        store.put(
            "resp_2",
            build_response_store_record(
                {
                    "id": "resp_2",
                    "created_at": 2,
                    "previous_response_id": "resp_1",
                    "output": [],
                },
                [{"role": "tool", "tool_call_id": "call_1", "content": "result"}],
                [{"role": "assistant", "content": "Two"}],
            ),
        )

        messages = store.resolve_chain_messages("resp_2")
        assert [m["content"] for m in messages if "content" in m] == [
            "First",
            "One",
            "result",
            "Two",
        ]

    def test_missing_response_chain_raises_not_found(self):
        store = ResponseStore(max_size=10)
        with pytest.raises(ResponseStateNotFoundError):
            store.resolve_chain_messages("resp_missing")

    def test_missing_ancestor_raises_corrupt(self):
        store = ResponseStore(max_size=10)
        store.put(
            "resp_2",
            build_response_store_record(
                {
                    "id": "resp_2",
                    "created_at": 2,
                    "previous_response_id": "resp_missing",
                    "output": [],
                },
                [],
                [],
            ),
        )
        with pytest.raises(ResponseStateCorruptError):
            store.resolve_chain_messages("resp_2")


# =============================================================================
# Previous Response Conversion Tests
# =============================================================================


class TestConvertStoredResponse:
    def test_message_output(self):
        stored = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ]
        }
        messages = convert_stored_response_to_messages(stored)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Hello!"

    def test_function_call_output(self):
        stored = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Let me check."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                },
            ]
        }
        messages = convert_stored_response_to_messages(stored)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Let me check."
        assert messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        # arguments should be parsed as dict for Jinja2 chat templates
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == {
            "location": "Paris"
        }

    def test_empty_output(self):
        stored = {"output": []}
        messages = convert_stored_response_to_messages(stored)
        assert messages == []

    def test_state_record_prefers_output_messages(self):
        stored = {
            "output_messages": [
                {"role": "assistant", "content": "Stored"},
            ]
        }
        messages = convert_stored_response_to_messages(stored)
        assert messages == [{"role": "assistant", "content": "Stored"}]

    def test_normalize_response_output_merges_assistant_tool_call_turn(self):
        output_items = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Let me check."}],
            },
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": '{"location":"Paris"}',
            },
        ]
        messages = normalize_response_output_to_messages(output_items)
        assert len(messages) == 1
        assert messages[0]["content"] == "Let me check."
        assert messages[0]["tool_calls"][0]["id"] == "call_abc"

    def test_normalize_reasoning_then_message(self):
        output_items = [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thinking"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer"}],
            },
        ]
        messages = normalize_response_output_to_messages(output_items)
        assert len(messages) == 1
        assert messages[0]["content"] == "answer"
        assert messages[0]["reasoning_content"] == "thinking"

    def test_normalize_reasoning_then_function_call(self):
        # Tool-call output sequence — reasoning must attach to the
        # synthesized assistant tool_calls message.
        output_items = [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "deciding"}],
            },
            {
                "type": "function_call",
                "call_id": "call_xyz",
                "name": "lookup",
                "arguments": '{"q":"foo"}',
            },
        ]
        messages = normalize_response_output_to_messages(output_items)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["tool_calls"][0]["function"]["name"] == "lookup"
        assert messages[0]["reasoning_content"] == "deciding"


# =============================================================================
# Request Model Tests
# =============================================================================


class TestResponsesRequest:
    def test_minimal_request(self):
        req = ResponsesRequest(model="test-model")
        assert req.model == "test-model"
        assert req.input is None
        assert req.stream is False
        assert req.store is None

    def test_string_input(self):
        req = ResponsesRequest(model="test", input="Hello")
        assert req.input == "Hello"

    def test_item_input(self):
        req = ResponsesRequest(
            model="test",
            input=[InputItem(type="message", role="user", content="Hi")],
        )
        assert len(req.input) == 1
        assert req.input[0].type == "message"

    def test_with_tools(self):
        req = ResponsesRequest(
            model="test",
            input="Hi",
            tools=[ResponsesTool(name="fn_a", description="Does A")],
        )
        assert len(req.tools) == 1
        assert req.tools[0].name == "fn_a"

    def test_text_format(self):
        req = ResponsesRequest(
            model="test",
            input="Hi",
            text=TextConfig(format=TextFormatConfig(type="json_object")),
        )
        assert req.text.format.type == "json_object"

    def test_codex_fields(self):
        """Codex CLI sends include, service_tier, prompt_cache_key, etc."""
        req = ResponsesRequest(
            model="test",
            input=[InputItem(role="user", content="Hi")],
            stream=True,
            store=False,
            include=["reasoning.encrypted_content"],
            service_tier="auto",
            prompt_cache_key="test-key",
            reasoning={"effort": "high", "summary": "auto"},
            parallel_tool_calls=True,
        )
        assert req.include == ["reasoning.encrypted_content"]
        assert req.service_tier == "auto"
        assert req.prompt_cache_key == "test-key"
        assert req.reasoning["effort"] == "high"

    def test_chat_template_kwargs(self):
        req = ResponsesRequest(
            model="test",
            input="Hi",
            chat_template_kwargs={"enable_thinking": False},
        )
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_extra_fields_allowed(self):
        """Unknown fields should not cause validation errors."""
        req = ResponsesRequest(
            model="test",
            input="Hi",
            some_future_field="value",
            another_field=42,
        )
        assert req.model == "test"

    def test_input_item_extra_fields(self):
        """InputItem should accept unknown fields for forward compatibility."""
        item = InputItem(
            type="message",
            role="user",
            content="Hi",
            phase="final_answer",
        )
        assert item.role == "user"

    def test_request_from_codex_like_json(self):
        """Parse a request body similar to what Codex CLI sends."""
        raw = {
            "model": "Qwen3.5-122B-A10B-4bit",
            "instructions": "You are a helpful assistant.",
            "input": [
                {"role": "user", "content": "Hello!"},
            ],
            "tools": [
                {"type": "local_shell"},
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                    "strict": True,
                },
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "reasoning": {"effort": "xhigh", "summary": "auto"},
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
        req = ResponsesRequest(**raw)
        assert req.model == "Qwen3.5-122B-A10B-4bit"
        assert req.stream is True
        assert len(req.input) == 1
        assert req.input[0].role == "user"
        assert req.input[0].type is None  # EasyInputMessage
        assert len(req.tools) == 2
        assert req.tools[0].type == "local_shell"
        assert req.tools[1].name == "read_file"


# =============================================================================
# Deferred tools + tool_search
# =============================================================================


def _deferred_ns_tool(name="calendar_events", description=None):
    return {
        "type": "namespace",
        "name": "planning_center",
        "description": "Tools in the planning_center namespace.",
        "tools": [
            {
                "type": "function",
                "name": name,
                "description": description
                or f"Search calendar events via {name.replace('_', ' ')}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Event name filter",
                        }
                    },
                },
                "defer_loading": True,
            }
        ],
    }


class TestDeferredToolConversion:
    def test_default_namespace_tool_search_cannot_collide_with_discovery(self):
        from omlx.exceptions import InvalidRequestError

        with pytest.raises(InvalidRequestError, match="reserved"):
            convert_responses_tools(
                [
                    ResponsesTool(
                        type="namespace",
                        name="functions",
                        tools=[{"type": "function", "name": "tool_search"}],
                    ),
                    ResponsesTool(type="tool_search"),
                ],
                tool_search_enabled=True,
            )

    @pytest.mark.parametrize("enabled", [False, True])
    @pytest.mark.parametrize("declared", [False, True])
    @pytest.mark.parametrize("deferred", [False, True])
    def test_flat_tool_search_name_cannot_collide_with_enabled_discovery(
        self, enabled, declared, deferred
    ):
        from omlx.exceptions import InvalidRequestError

        tools = [ResponsesTool(name="tool_search", defer_loading=deferred)]
        if declared:
            tools.append(ResponsesTool(type="tool_search"))
        if enabled and declared:
            with pytest.raises(InvalidRequestError, match="reserved"):
                convert_responses_tools(tools, tool_search_enabled=enabled)
        else:
            active = convert_responses_tools(tools, tool_search_enabled=enabled)
            assert [tool["function"]["name"] for tool in active] == ["tool_search"]

    @pytest.mark.parametrize("enabled", [False, True])
    def test_deferral_requires_an_enabled_search_declaration(self, enabled):
        from omlx.api.responses_tool_search import DeferredToolSet

        pool = DeferredToolSet()
        active = convert_responses_tools(
            [
                ResponsesTool(**_deferred_ns_tool()),
                ResponsesTool(
                    name="flat_tool",
                    defer_loading=True,
                ),
            ],
            deferred=pool,
            tool_search_enabled=enabled,
        )
        assert {tool["function"]["name"] for tool in active} == {
            "planning_center__calendar_events",
            "flat_tool",
        }
        assert not pool

    def test_namespace_deferred_members_excluded_and_indexed(self):
        from omlx.api.responses_tool_search import DeferredToolSet

        deferred = DeferredToolSet()
        tools = [
            {"type": "function", "name": "shell", "parameters": {}},
            _deferred_ns_tool(),
            {"type": "tool_search"},
        ]
        aliases: dict = {}
        active = convert_responses_tools(
            [ResponsesTool(**t) for t in tools],
            aliases,
            deferred=deferred,
            tool_search_enabled=True,
        )
        names = {t["function"]["name"] for t in active}
        assert names == {"shell", "tool_search"}
        assert deferred  # indexed for tool_search
        tool = deferred.get("planning_center__calendar_events")
        assert tool is not None and tool.namespace == "planning_center"
        # Return-path alias still resolves the deferred wire name.
        assert split_namespace_tool_name(
            "planning_center__calendar_events", aliases
        ) == ("planning_center", "calendar_events")

    def test_flat_deferred_function_excluded(self):
        from omlx.api.responses_tool_search import DeferredToolSet

        deferred = DeferredToolSet()
        active = convert_responses_tools(
            [
                ResponsesTool(
                    **{
                        "type": "function",
                        "name": "secret_tool",
                        "defer_loading": True,
                    }
                ),
                ResponsesTool(**{"type": "function", "name": "visible"}),
                ResponsesTool(type="tool_search"),
            ],
            deferred=deferred,
            tool_search_enabled=True,
        )
        assert [t["function"]["name"] for t in active] == ["visible", "tool_search"]
        assert "secret_tool" in deferred

    def test_mixed_namespace_keeps_active_members(self):
        from omlx.api.responses_tool_search import DeferredToolSet

        ns = _deferred_ns_tool("deferred_one")
        ns["tools"].append({"type": "function", "name": "eager_two", "parameters": {}})
        deferred = DeferredToolSet()
        active = convert_responses_tools(
            [ResponsesTool(**ns), ResponsesTool(type="tool_search")],
            deferred=deferred,
            tool_search_enabled=True,
        )
        assert [t["function"]["name"] for t in active] == [
            "planning_center__eager_two",
            "tool_search",
        ]
        assert "planning_center__deferred_one" in deferred

    def test_tool_search_declaration_rendered_when_enabled(self):
        declared = {
            "type": "tool_search",
            "execution": "client",
            "description": "# Tool discovery",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
        active = convert_responses_tools(
            [ResponsesTool(**declared)], tool_search_enabled=True
        )
        assert active and active[0]["function"]["name"] == "tool_search"
        assert active[0]["function"]["description"] == "# Tool discovery"

    def test_tool_search_declaration_skipped_when_disabled(self):
        active = convert_responses_tools(
            [ResponsesTool(**{"type": "tool_search", "execution": "client"})],
            tool_search_enabled=False,
        )
        assert active is None


class TestToolSearchBM25:
    def _set(self):
        from omlx.api.responses_tool_search import DeferredToolSet

        return DeferredToolSet(
            [
                _tool("calendar_events", "planning_center", "Search calendar events"),
                _tool("people_search", "planning_center", "Search people profiles"),
                _tool("give_fireball", "mage", "Cast a fireball spell"),
            ]
        )

    def test_ranks_name_matches_first(self):
        hits = self._set().search("calendar events")
        assert [h.name for h in hits][0] == "calendar_events"

    def test_limit_respected_and_default_eight(self):
        from omlx.api.responses_tool_search import DeferredTool, DeferredToolSet

        tools = [
            DeferredTool(f"ns__t{i}", "ns", f"t{i}", f"tool number {i} search")
            for i in range(12)
        ]
        result = DeferredToolSet(tools).search("tool number search")
        assert len(result) == 8

    def test_empty_query_returns_nothing(self):
        assert self._set().search("   ") == []

    def test_parameter_descriptions_are_searchable(self):
        from omlx.api.responses_tool_search import DeferredTool, DeferredToolSet

        tool = DeferredTool(
            "get_records",
            None,
            "get_records",
            parameters={
                "type": "object",
                "properties": {
                    "filter": {"type": "string", "description": "anniversary"}
                },
            },
        )
        assert DeferredToolSet([tool]).search("anniversary") == [tool]

    def test_results_coalesce_namespaces(self):
        from omlx.api.responses_tool_search import build_tool_search_results

        hits = self._set().search("search")
        results = build_tool_search_results(hits)
        assert len(results) == 1
        spec = results[0]
        assert spec["type"] == "namespace"
        assert spec["name"] == "planning_center"
        assert all(t["defer_loading"] is True for t in spec["tools"])

    def test_parse_loadable_specs_roundtrip(self):
        from omlx.api.responses_tool_search import (
            build_tool_search_results,
            parse_loadable_specs,
        )

        hits = self._set().search("fireball")
        specs = build_tool_search_results(hits)
        parsed = parse_loadable_specs(specs)
        assert [t.wire_name for t in parsed] == ["mage__give_fireball"]

    def test_extend_specs_adds_new_only(self):
        from omlx.api.responses_tool_search import build_tool_search_results

        ts = self._set()
        # A spec whose tool is already registered adds nothing new.
        spec = build_tool_search_results([ts.get("planning_center__people_search")])
        assert ts.extend_specs(spec) == []
        # A previously unseen spec (Codex-only shape, e.g. from history) is
        # indexed and reported as newly loaded.
        fresh = [
            {
                "type": "namespace",
                "name": "planning_center",
                "description": "Tools in the planning_center namespace.",
                "tools": [
                    {"type": "function", "name": "giving_donations", "parameters": {}}
                ],
            }
        ]
        added = ts.extend_specs(fresh)
        assert [t.name for t in added] == ["giving_donations"]


def _tool(name, namespace, description):
    from omlx.api.responses_tool_search import DeferredTool

    return DeferredTool(
        wire_name=f"{namespace}__{name}",
        namespace=namespace,
        name=name,
        description=description,
        parameters={"type": "object", "properties": {}},
    )


class TestToolSearchHistory:
    @pytest.mark.parametrize("eager", [False, True])
    def test_restored_flat_identity_survives_a_namespace_wire_collision(self, eager):
        from omlx.api.responses_tool_search import DeferredToolSet, ToolSearchState
        from omlx.api.responses_utils import apply_namespace_tool_aliases

        pool, aliases = DeferredToolSet(), {}
        namespace = _deferred_ns_tool()
        namespace["tools"][0]["defer_loading"] = not eager
        active = convert_responses_tools(
            [ResponsesTool(**namespace), ResponsesTool(type="tool_search")],
            aliases,
            deferred=pool,
            tool_search_enabled=True,
        )
        flat_name = "planning_center__calendar_events"
        parameters = {"type": "object", "properties": {"flat_arg": {"type": "string"}}}
        state = ToolSearchState(enabled=True, pool=pool)
        state.restore_loaded(
            [{"type": "function", "name": flat_name, "parameters": parameters}],
            aliases,
            active,
        )
        loaded = state.loaded_tool_defs()
        assert len(loaded) == 1
        wire = loaded[0]["function"]["name"]
        assert wire != flat_name
        assert loaded[0]["function"]["parameters"] == parameters
        assert split_namespace_tool_name(wire, aliases) == (None, flat_name)
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": flat_name}},
                    {
                        "function": {
                            "namespace": "planning_center",
                            "name": "calendar_events",
                        }
                    },
                ],
            }
        ]
        apply_namespace_tool_aliases(messages, aliases)
        calls = messages[0]["tool_calls"]
        assert calls[0]["function"]["name"] == wire
        assert calls[1]["function"]["name"] == flat_name

    @pytest.mark.parametrize("namespace", [None, "functions"])
    def test_restored_flat_schema_does_not_override_an_eager_identity(self, namespace):
        from omlx.api.responses_tool_search import DeferredToolSet, ToolSearchState

        parameters = {"type": "object", "properties": {"current": {"type": "string"}}}
        function = {"type": "function", "name": "lookup", "parameters": parameters}
        tool = (
            {"type": "namespace", "name": namespace, "tools": [function]}
            if namespace
            else function
        )
        aliases, pool = {}, DeferredToolSet()
        eager = convert_responses_tools(
            [ResponsesTool(**tool)], aliases, deferred=pool, tool_search_enabled=True
        )
        state = ToolSearchState(enabled=True, pool=pool)
        state.restore_loaded(
            [{"type": "function", "name": "lookup", "parameters": {}}],
            aliases,
            eager,
        )
        assert state.loaded_tool_defs() == []
        assert eager[0]["function"]["parameters"] == parameters

    def test_restored_flat_tool_search_cannot_replace_discovery(self):
        from omlx.api.responses_tool_search import DeferredToolSet, ToolSearchState
        from omlx.exceptions import InvalidRequestError

        state = ToolSearchState(enabled=True, pool=DeferredToolSet())
        with pytest.raises(InvalidRequestError, match="reserved"):
            state.restore_loaded(
                [{"type": "function", "name": "tool_search", "parameters": {}}],
                {},
                [{"type": "function", "function": {"name": "tool_search"}}],
            )

    @pytest.mark.parametrize("answer_reasoning", ["", "Prepare the answer."])
    def test_hosted_search_reasoning_stays_with_its_assistant_turn(
        self, answer_reasoning
    ):
        items = [
            {
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": "Find the calendar tool."}
                ],
            },
            {
                "type": "tool_search_call",
                "id": "search1",
                "call_id": None,
                "execution": "server",
                "arguments": {"query": "calendar"},
            },
            {
                "type": "tool_search_output",
                "call_id": None,
                "execution": "server",
                "tools": [_deferred_ns_tool()],
            },
        ]
        if answer_reasoning:
            items.append(
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": answer_reasoning}],
                }
            )
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Found it."}],
            }
        )
        stored = normalize_response_output_to_messages(items)
        replayed = convert_responses_input_to_messages(
            [InputItem(**item) for item in items]
        )
        assert stored == replayed
        assert stored[0]["reasoning_content"] == "Find the calendar tool."
        assert stored[-1].get("reasoning_content", "") == answer_reasoning

    def test_hosted_search_with_null_call_ids_replays_as_paired_history(self):
        items = [
            {
                "type": "tool_search_call",
                "id": "search1",
                "call_id": None,
                "execution": "server",
                "arguments": {"query": "calendar"},
            },
            {
                "type": "tool_search_output",
                "id": "result1",
                "call_id": None,
                "execution": "server",
                "tools": [_deferred_ns_tool()],
            },
        ]
        stored = normalize_response_output_to_messages(items)
        replayed = convert_responses_input_to_messages(
            [InputItem(**item) for item in items]
        )
        assert stored == replayed
        assert stored[0]["tool_calls"][0]["id"] == "search1"
        assert stored[1]["tool_call_id"] == "search1"

    def test_loaded_snapshot_preserves_schemas_and_current_collision_aliases(self):
        from omlx.api.responses_tool_search import DeferredToolSet, ToolSearchState

        pool, aliases = DeferredToolSet(), {}
        tool = _deferred_ns_tool()
        tool["tools"][0]["strict"] = True
        eager = convert_responses_tools(
            [
                ResponsesTool(name="planning_center__calendar_events"),
                ResponsesTool(**tool),
                ResponsesTool(type="tool_search"),
            ],
            aliases,
            deferred=pool,
            tool_search_enabled=True,
        )
        state = ToolSearchState(enabled=True, pool=pool)
        state.run_search({"query": "calendar"})
        restored = ToolSearchState(enabled=True, pool=pool)
        restored.restore_loaded(state.serialize_loaded_specs(), aliases, eager)
        loaded = restored.loaded_tool_defs()
        assert len(loaded) == 1
        assert loaded[0]["function"]["name"] == "planning_center__calendar_events_2"
        assert loaded[0]["function"]["parameters"] == tool["tools"][0]["parameters"]
        assert loaded[0]["function"]["strict"] is True
        assert aliases["planning_center__calendar_events_2"] == (
            "planning_center",
            "calendar_events",
        )

    def test_restored_schema_uses_current_request_definition(self):
        from omlx.api.responses_tool_search import DeferredToolSet, ToolSearchState

        current = _tool("calendar_events", "planning_center", "current description")
        state = ToolSearchState(enabled=True, pool=DeferredToolSet([current]))
        state.restore_loaded(
            [_deferred_ns_tool(description="stale description")], {}, []
        )
        assert (
            state.loaded_tool_defs()[0]["function"]["description"]
            == "current description"
        )

    def test_flat_tools_roundtrip_through_default_namespace(self):
        from omlx.api.responses_tool_search import (
            DeferredTool,
            DeferredToolSet,
            ToolSearchState,
        )

        tool = DeferredTool("lookup", None, "lookup", "Look up calendars", strict=False)
        state = ToolSearchState(
            enabled=True, pool=DeferredToolSet([tool]), loaded={"lookup"}
        )
        restored = ToolSearchState(enabled=True, pool=DeferredToolSet())
        restored.restore_loaded(state.serialize_loaded_specs(), {}, [])
        assert restored.loaded_tool_defs() == state.loaded_tool_defs()

    def test_explicit_default_namespace_restores_to_its_original_wire_name(self):
        from omlx.api.responses_tool_search import DeferredToolSet, ToolSearchState

        pool = DeferredToolSet()
        namespace = _deferred_ns_tool()
        namespace["name"] = "functions"
        aliases = {}
        eager = convert_responses_tools(
            [
                ResponsesTool(**namespace),
                ResponsesTool(type="tool_search"),
            ],
            aliases,
            deferred=pool,
            tool_search_enabled=True,
        )
        state = ToolSearchState(enabled=True, pool=pool)
        state.run_search({"query": "calendar"})
        restored = ToolSearchState(enabled=True, pool=pool)
        restored.restore_loaded(state.serialize_loaded_specs(), aliases, eager)
        assert [tool["function"]["name"] for tool in restored.loaded_tool_defs()] == [
            "calendar_events"
        ]

    def test_call_and_output_items_become_messages(self):
        items = [
            InputItem(type="message", role="user", content="find events"),
            InputItem(
                **{
                    "type": "tool_search_call",
                    "call_id": "call_1",
                    "execution": "client",
                    "arguments": {"query": "calendar", "limit": 5},
                }
            ),
            InputItem(
                **{
                    "type": "tool_search_output",
                    "call_id": "call_1",
                    "status": "completed",
                    "execution": "client",
                    "tools": [{"type": "namespace", "name": "ns", "tools": []}],
                }
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        assert messages[0] == {"role": "user", "content": "find events"}
        assert messages[1]["role"] == "assistant"
        call = messages[1]["tool_calls"][0]
        assert call["function"]["name"] == "tool_search"
        # Arguments arrive as an object and must survive as parsed JSON.
        assert call["function"]["arguments"] == {"query": "calendar", "limit": 5}
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call_1"
        assert json.loads(messages[2]["content"])["tools"]

    def test_stored_output_items_normalize(self):
        output_items = [
            {
                "type": "tool_search_call",
                "call_id": "c1",
                "execution": "server",
                "arguments": {"query": "people"},
            },
            {
                "type": "tool_search_output",
                "call_id": "c1",
                "execution": "server",
                "tools": [{"type": "function", "name": "people_search"}],
            },
        ]
        messages = normalize_response_output_to_messages(output_items)
        assert messages[0]["role"] == "assistant"
        assert messages[0]["tool_calls"][0]["function"]["name"] == "tool_search"
        assert messages[1]["role"] == "tool"
        assert "people_search" in messages[1]["content"]

    def test_collect_tool_search_results(self):
        from omlx.api.responses_tool_search import collect_tool_search_results

        items = [
            InputItem(type="message", role="user", content="x"),
            InputItem(
                **{
                    "type": "tool_search_output",
                    "tools": [{"type": "function", "name": "a"}],
                }
            ),
            {
                "type": "tool_search_output",
                "tools": [{"type": "function", "name": "b"}],
            },
        ]
        specs = collect_tool_search_results(items)
        assert [s["name"] for s in specs] == ["a", "b"]

    def test_collect_tool_search_results_does_not_scan_plain_text(self):
        from omlx.api.responses_tool_search import collect_tool_search_results

        class PlainText(str):
            def __iter__(self):
                raise AssertionError("Plain text must not be scanned as response items")

        assert collect_tool_search_results(PlainText("Find calendar events")) == []

    def test_store_record_preserves_loaded_specs(self):
        record = build_response_store_record(
            {"id": "resp_1"},
            input_messages=[],
            output_messages=[],
            loaded_tool_specs=[{"wire_name": "a", "name": "a", "namespace": None}],
        )
        assert record["loaded_tool_specs"][0]["wire_name"] == "a"
