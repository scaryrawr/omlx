# SPDX-License-Identifier: Apache-2.0
"""Check a live model's tool calls and tool-result turns, including SSE and usage.

Run against an isolated server:
  OMLX_BASE_URL=http://127.0.0.1:18080 OMLX_TEST_MODEL=Qwen3.5-9B-oQ4e-mtp \\
  uv run --python python3.12 pytest tests/integration/test_live_tool_calling.py -m integration

OMLX_API_KEY is optional. The selected model must be listed by /v1/models.
"""

import json
import os

import httpx
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def live_server():
    model = os.environ.get("OMLX_TEST_MODEL")
    if not model:
        pytest.skip("Set OMLX_TEST_MODEL to a model ID served by the live server")
    headers = {}
    if api_key := os.environ.get("OMLX_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    with httpx.Client(
        base_url=os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8000"),
        headers=headers,
        timeout=180,
        trust_env=False,
    ) as client:
        response = client.get("/v1/models")
        response.raise_for_status()
        assert model in {entry["id"] for entry in response.json()["data"]}
        yield client, model


def _completion(client, payload):
    if not payload["stream"]:
        response = client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    message = {"role": "assistant", "content": "", "tool_calls": []}
    finish_reason = None
    usage = None
    done = False
    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)
            assert "error" not in chunk, chunk
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                assert choice["index"] == 0
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice["delta"]
                message["content"] += delta.get("content") or ""
                for call in delta.get("tool_calls", []):
                    index = call["index"]
                    assert index <= len(message["tool_calls"])
                    if index == len(message["tool_calls"]):
                        message["tool_calls"].append(
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        )
                    target = message["tool_calls"][index]
                    if call.get("id"):
                        assert not target["id"] or target["id"] == call["id"]
                        target["id"] = call["id"]
                    function = call.get("function", {})
                    target["function"]["name"] += function.get("name") or ""
                    target["function"]["arguments"] += function.get("arguments") or ""
    assert done, "Stream ended without [DONE]"
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


@pytest.mark.parametrize("stream", [False, True])
def test_live_tool_call_round_trip(live_server, stream):
    client, model = live_server
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Use multiply for arithmetic. After receiving the tool result, "
                    "answer with only the numeric result."
                ),
            },
            {"role": "user", "content": "What is 6 multiplied by 7?"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "multiply",
                    "description": "Multiply two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "required",
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
        payload["return_progress"] = True
    response = _completion(client, payload)
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls", response
    message = choice["message"]
    assert len(message["tool_calls"]) == 1, response
    call = message["tool_calls"][0]
    assert call["id"]
    assert call["type"] == "function"
    assert call["function"]["name"] == "multiply"
    assert json.loads(call["function"]["arguments"]) == {"a": 6, "b": 7}
    assert response["usage"]["prompt_tokens"] > 0
    assert response["usage"]["completion_tokens"] > 0
    assert response["usage"]["total_tokens"] == (
        response["usage"]["prompt_tokens"] + response["usage"]["completion_tokens"]
    )

    payload["messages"].extend(
        [
            message,
            {"role": "tool", "tool_call_id": call["id"], "content": "42"},
        ]
    )
    payload["tool_choice"] = "none"
    response = _completion(client, payload)
    choice = response["choices"][0]
    assert choice["finish_reason"] == "stop", response
    assert "42" in choice["message"]["content"], response
    assert not choice["message"].get("tool_calls"), response
    assert response["usage"]["completion_tokens"] > 0
