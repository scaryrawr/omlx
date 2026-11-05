# SPDX-License-Identifier: Apache-2.0
"""Template contracts complementing tool-loop behavior in the JS suite."""

from pathlib import Path

import pytest

CHAT_TEMPLATE = Path(__file__).parents[1] / "omlx_web" / "templates" / "chat.html"


@pytest.fixture(scope="module")
def stream_response_source():
    source = CHAT_TEMPLATE.read_text(encoding="utf-8")
    start = source.index("async streamResponse(streamContext = null, depth = 0)")
    end = source.index("    stopStreaming()", start)
    return source[start:end]


def test_limit_is_checked_before_executing_an_extra_tool_round(stream_response_source):
    tool_branch = stream_response_source[
        stream_response_source.index("if (toolCalls.length > 0) {") :
    ]
    assert tool_branch.index("if (depth >= maxToolRounds)") < tool_branch.index(
        "const results = await Promise.all"
    )
    assert "MAX_TOOL_DEPTH" not in stream_response_source


def test_final_answer_is_still_allowed_after_the_last_tool_round(stream_response_source):
    assert stream_response_source.index(
        "if (toolCalls.length > 0) {"
    ) < stream_response_source.index("if (depth >= maxToolRounds)")


def test_root_request_owns_timing_and_stream_cleanup(stream_response_source):
    assert "context._requestStartedAt = Date.now();" in stream_response_source
    assert "Date.now() - context._requestStartedAt" in stream_response_source
    assert stream_response_source.count("this.resetStreamSession(stream") == 2
    finally_body = stream_response_source[stream_response_source.rindex("} finally {") :]
    assert finally_body.index("if (depth === 0) {") < finally_body.index(
        "this.resetStreamSession(stream, { preserveFinalContent: true });"
    )


def test_chat_setting_exposes_a_bounded_tool_round_limit():
    source = CHAT_TEMPLATE.read_text(encoding="utf-8")
    assert "maxToolRounds: 10" in source
    assert 'id="max-tool-rounds"' in source
    assert 'min="1" max="100"' in source
    assert "normalizeMaxToolRounds(value)" in source
