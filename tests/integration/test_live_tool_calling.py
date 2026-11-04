# SPDX-License-Identifier: Apache-2.0
"""Check live tool turns, SSE, usage, and chat-backed decision readouts.

Run against an isolated server:
  OMLX_BASE_URL=http://127.0.0.1:18080 OMLX_TEST_MODEL=Qwen3.5-9B-oQ4e-mtp \\
  uv run --python python3.12 pytest tests/integration/test_live_tool_calling.py -m integration

OMLX_API_KEY is optional. The selected model must be listed by /v1/models.
"""

import json
import math
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
    logprobs = []
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
                if choice.get("logprobs"):
                    logprobs.extend(choice["logprobs"]["content"] or [])
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
        "choices": [
            {
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": {"content": logprobs},
            }
        ],
        "usage": usage,
    }


@pytest.mark.parametrize("stream", [False, True])
def test_live_chat_logprobs(live_server, stream):
    client, model = live_server
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with only the word hello."}],
        "temperature": 0,
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
        "logprobs": True,
        "top_logprobs": 3,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    response = _completion(client, payload)
    choice = response["choices"][0]
    assert choice["finish_reason"] == "stop", response
    assert choice["message"]["content"], response
    entries = choice["logprobs"]["content"]
    assert entries, response
    for entry in entries:
        assert isinstance(entry["token"], str)
        assert math.isfinite(entry["logprob"]) and entry["logprob"] <= 0
        assert all(isinstance(byte, int) and 0 <= byte <= 255 for byte in entry["bytes"])
        assert len(entry["top_logprobs"]) == 3
        for alternative in entry["top_logprobs"]:
            assert math.isfinite(alternative["logprob"])
            assert alternative["logprob"] <= 0
    assert response["usage"]["completion_tokens"] > 0


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


@pytest.mark.parametrize("endpoint", ["/v1/decisions", "/v1/systemone"])
def test_live_chat_decision_readout(live_server, endpoint):
    client, model = live_server
    question = {
        "type": "choice",
        "instructions": "What color is the stop sign?",
    }
    if endpoint == "/v1/systemone":
        payload = {
            "state": "The stop sign is red.",
            "questions": {
                "color": {**question, "criteria": {"red": None, "blue": None}}
            },
        }
    else:
        payload = {
            "input": "The stop sign is red.",
            "questions": [{"name": "color", **question, "choices": ["red", "blue"]}],
        }
    response = client.post(endpoint, json={"model": model, **payload})
    response.raise_for_status()
    body = response.json()
    if endpoint == "/v1/systemone":
        answer = body["answers"]["color"]
        probabilities = list(answer["probabilities"].values())
    else:
        (answer,) = body["answers"]
        assert answer["name"] == "color"
        probabilities = [entry["probability"] for entry in answer["probabilities"]]
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] == 0
    assert answer["choice"] == "red", body
    assert len(probabilities) == 2
    assert all(0 <= probability <= 1 for probability in probabilities)
    assert sum(probabilities) == pytest.approx(1, abs=0.0002)


def test_live_anthropic_tool_call_round_trip(live_server):
    client, model = live_server
    payload = {
        "model": model,
        "system": (
            "Use multiply for arithmetic. After receiving the tool result, "
            "answer with only the numeric result."
        ),
        "messages": [{"role": "user", "content": "What is 6 multiplied by 7?"}],
        "tools": [
            {
                "name": "multiply",
                "description": "Multiply two integers.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            }
        ],
        "tool_choice": {"type": "any"},
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low"},
        "temperature": 0,
        "max_tokens": 128,
    }
    count_response = client.post(
        "/v1/messages/count_tokens",
        json={key: value for key, value in payload.items() if key != "max_tokens"},
    )
    count_response.raise_for_status()
    response = client.post("/v1/messages", json=payload)
    response.raise_for_status()
    body = response.json()
    assert body["stop_reason"] == "tool_use", body
    assert body["usage"]["input_tokens"] == count_response.json()["input_tokens"]
    assert body["usage"]["output_tokens"] > 0
    (call,) = [block for block in body["content"] if block["type"] == "tool_use"]
    assert call["id"]
    assert call["name"] == "multiply"
    assert call["input"] == {"a": 6, "b": 7}

    payload["messages"].extend(
        [
            {"role": "assistant", "content": body["content"]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": call["id"], "content": "42"}
                ],
            },
        ]
    )
    payload["tool_choice"] = {"type": "none"}
    response = client.post("/v1/messages", json=payload)
    response.raise_for_status()
    body = response.json()
    assert body["stop_reason"] == "end_turn", body
    assert any(
        block["type"] == "text" and "42" in block["text"] for block in body["content"]
    ), body
    assert not any(block["type"] == "tool_use" for block in body["content"]), body
