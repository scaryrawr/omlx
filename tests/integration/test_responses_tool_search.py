# SPDX-License-Identifier: Apache-2.0
"""Integration tests for Codex-style deferred tool loading + tool_search.

Replays the Codex request sequence against /v1/responses with a scripted
mock engine:

1. Namespaced ``defer_loading`` tools + a ``tool_search`` declaration —
   deferred schemas must never reach the model prompt, and the declared
   tool_search tool must.
2. The model calls ``tool_search`` — non-streaming requests execute it
   server-side (BM25) inside the response, emit typed
   ``tool_search_call``/``tool_search_output`` items, and continue
   generation with the matched tools promoted into the active tool set.
3. Explicit client-executed search gets a ``tool_search_call`` with
   ``execution: "client"`` so Codex's own handler runs the round trip.
4. History replay: ``tool_search_output`` items in the input promote the
   matched tools for subsequent turns (Codex resends full history).
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from omlx.engine.base import BaseEngine


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 2

    def encode(self, text: str) -> list[int]:
        return [100 + i for i, _ in enumerate(text.split())]

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        return f"<decoded:{len(tokens)} tokens>"

    def apply_chat_template(
        self, messages: list[dict], tokenize: bool = False, **kwargs
    ) -> str:
        parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
        return "\n".join(parts)


class MockOutput:
    def __init__(self, text="", tool_calls=None, **kwargs):
        self.text = text
        self.new_text = kwargs.get("new_text", text)
        self.finished = kwargs.get("finished", True)
        self.finish_reason = kwargs.get("finish_reason", "stop")
        self.tool_calls = tool_calls
        self.prompt_tokens = kwargs.get("prompt_tokens", 10)
        self.completion_tokens = kwargs.get("completion_tokens", 5)
        self.cached_tokens = kwargs.get("cached_tokens", 0)
        self.first_token_at = None


class ScriptedEngine(BaseEngine):
    """Engine returning scripted chat() rounds; records every call."""

    def __init__(self, rounds):
        self._model_name = "test-model"
        self._tokenizer = MockTokenizer()
        self._model_type = "llama"
        self.rounds = list(rounds)
        self.recorded: list[dict[str, Any]] = []
        self.started = False
        self.preflights = []

    async def preflight_chat(self, messages, **kwargs):
        self.preflights.append({"messages": list(messages), "kwargs": kwargs})

    @property
    def model_name(self):
        return self._model_name

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_type(self):
        return self._model_type

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        pass

    async def generate(self, prompt, **kwargs):
        return MockOutput(text="Generated response.")

    async def stream_generate(self, prompt, **kwargs):
        yield MockOutput(text="Hello world.")

    def count_chat_tokens(
        self, messages, tools=None, chat_template_kwargs=None, **kwargs
    ):
        template_kwargs = dict(chat_template_kwargs or {})
        template_kwargs["tokenize"] = False
        if tools:
            template_kwargs["tools"] = tools
        prompt = self._tokenizer.apply_chat_template(messages, **template_kwargs)
        return len(self._tokenizer.encode(prompt))

    async def chat(self, messages, **kwargs):
        self.recorded.append({"messages": list(messages), "kwargs": kwargs})
        if self.rounds:
            return self.rounds.pop(0)
        return MockOutput(text="done")

    async def stream_chat(self, messages, **kwargs):
        self.recorded.append({"messages": list(messages), "kwargs": kwargs})
        round_output = (
            self.rounds.pop(0)
            if self.rounds
            else MockOutput(text="done", new_text="done")
        )
        yield round_output

    def get_stats(self) -> dict[str, Any]:
        return {}

    def get_cache_stats(self):
        return None


class MockEnginePool:
    def __init__(self, engine):
        self._engine = engine
        self._models = [
            {"id": "test-model", "loaded": True, "pinned": False, "size": 1}
        ]

    @property
    def model_count(self):
        return len(self._models)

    @property
    def loaded_model_count(self):
        return 1

    @property
    def max_model_memory(self):
        return 32 * 1024 * 1024 * 1024

    @property
    def current_model_memory(self):
        return 1

    def get_entry(self, model_id):
        return None

    def resolve_model_id(self, model_id_or_alias, settings_manager=None):
        return model_id_or_alias

    def get_model_ids(self):
        return [m["id"] for m in self._models]

    def get_status(self):
        return {
            "models": self._models,
            "loaded_count": self.loaded_model_count,
            "max_model_memory": self.max_model_memory,
        }

    async def get_engine(self, model_id, **kwargs):
        return self._engine

    async def release_engine(self, model_id):
        pass


CALENDAR_DEFERRED = {
    "type": "namespace",
    "name": "planning_center",
    "description": "Tools in the planning_center namespace.",
    "tools": [
        {
            "type": "function",
            "name": "calendar_events",
            "description": "Search calendar events for the church",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
            "defer_loading": True,
        },
        {
            "type": "function",
            "name": "people_search",
            "description": "Search people in the database",
            "parameters": {
                "type": "object",
                "properties": {"first_name": {"type": "string"}},
            },
            "defer_loading": True,
        },
    ],
}

TOOL_SEARCH_DECL = {
    "type": "tool_search",
    "execution": "client",
    "description": "# Tool discovery\n\nSearches deferred tools with BM25.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "number"},
        },
        "required": ["query"],
    },
}

CORE_TOOL = {
    "type": "function",
    "name": "shell",
    "description": "Run a shell command",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
}


def _tool_names(tools):
    return {t["function"]["name"] for t in tools or []}


def _events(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def _body(response, stream):
    assert response.status_code == 200, response.text
    if not stream:
        return response.json()
    events = _events(response)
    return next(
        event["response"]
        for event in events
        if event["type"] in ("response.completed", "response.incomplete")
    )


@pytest.fixture()
def make_client():
    """Build a TestClient over a scripted engine with the feature flag set."""
    from omlx.api.responses_utils import ResponseStore
    from omlx.server import _server_state, app

    originals = {}

    def _make(rounds, *, enabled=True):
        engine = ScriptedEngine(rounds)
        originals.update(
            engine_pool=_server_state.engine_pool,
            default_model=_server_state.default_model,
            mcp_manager=_server_state.mcp_manager,
            settings_manager=_server_state.settings_manager,
            global_settings=_server_state.global_settings,
            responses_store=_server_state.responses_store,
        )
        _server_state.engine_pool = MockEnginePool(engine)
        _server_state.default_model = "test-model"
        _server_state.mcp_manager = None
        _server_state.settings_manager = None
        _server_state.responses_store = ResponseStore()
        _server_state.global_settings = SimpleNamespace(
            server=SimpleNamespace(
                responses_tool_search=enabled,
                preserve_mid_system_cache=True,
                sse_keepalive_mode="chunk",
            ),
            auth=SimpleNamespace(allow_unauthenticated_inference=True),
        )
        return TestClient(app), engine

    yield _make

    for key, value in originals.items():
        setattr(_server_state, key, value)


class TestServerSideToolSearch:
    """Non-streaming: oMLX executes tool_search and continues in-request."""

    @pytest.mark.parametrize("stream", [False, True])
    def test_search_round_trip_loads_tools(self, make_client, stream):
        client, engine = make_client(
            [
                MockOutput(
                    tool_calls=[
                        {
                            "id": "call_s1",
                            "name": "tool_search",
                            "arguments": '{"query": "calendar events"}',
                        }
                    ]
                ),
                MockOutput(
                    text="Found the calendar tool.",
                    tool_calls=[
                        {
                            "id": "call_s2",
                            "name": "planning_center__calendar_events",
                            "arguments": '{"name": "Youth Night"}',
                        }
                    ],
                ),
            ]
        )
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Find youth night on the calendar",
                "store": False,
                "stream": stream,
                "tools": [CORE_TOOL, CALENDAR_DEFERRED, {"type": "tool_search"}],
            },
        )
        assert resp.status_code == 200, resp.text
        body = _body(resp, stream)
        types = [item["type"] for item in body["output"]]
        assert "tool_search_call" in types
        assert "tool_search_output" in types

        call_item = next(i for i in body["output"] if i["type"] == "tool_search_call")
        assert call_item["execution"] == "server"
        assert call_item["arguments"]["query"] == "calendar events"
        assert call_item["call_id"] is None

        out_item = next(i for i in body["output"] if i["type"] == "tool_search_output")
        assert out_item["execution"] == "server"
        ns = out_item["tools"][0]
        assert ns["type"] == "namespace" and ns["name"] == "planning_center"
        assert [t["name"] for t in ns["tools"]] == ["calendar_events"]
        assert ns["tools"][0]["parameters"]["properties"]["name"]["type"] == "string"

        # The matched tool is invoked as a namespaced function call.
        fc = next(i for i in body["output"] if i["type"] == "function_call")
        assert fc["name"] == "calendar_events"
        assert fc["namespace"] == "planning_center"

        # Prompt hygiene: round 1 saw only the core tool + tool_search; the
        # continuation saw the loaded tool with its full schema.
        assert len(engine.recorded) == 2
        first_tools = _tool_names(engine.recorded[0]["kwargs"].get("tools"))
        assert first_tools == {"shell", "tool_search"}
        second_tools = _tool_names(engine.recorded[1]["kwargs"].get("tools"))
        assert "planning_center__calendar_events" in second_tools
        loaded = next(
            t
            for t in engine.recorded[1]["kwargs"]["tools"]
            if t["function"]["name"] == "planning_center__calendar_events"
        )
        assert "parameters" in loaded["function"]

        # The synthesized round-trip is in the continuation messages so the
        # model actually sees its search result.
        second_messages = engine.recorded[1]["messages"]
        assert any(
            m.get("role") == "assistant"
            and (m.get("tool_calls") or [{}])[0]["function"]["name"] == "tool_search"
            for m in second_messages
        )
        tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert tool_msgs and "calendar_events" in tool_msgs[-1]["content"]
        assert len(engine.preflights) == 2
        assert body["usage"]["input_tokens"] == 20
        assert body["usage"]["output_tokens"] == 10
        if stream:
            events = _events(resp)
            assert sum(e["type"] == "response.created" for e in events) == 1
            assert sum(e["type"] == "response.completed" for e in events) == 1
            sequence = [e["sequence_number"] for e in events]
            assert sequence == sorted(set(sequence))
            done = [e for e in events if e["type"] == "response.output_item.done"]
            assert [e["output_index"] for e in done] == list(range(len(done)))
            assert [e["item"] for e in done] == body["output"]

    def test_deferred_tools_remain_callable_when_disabled(self, make_client):
        client, engine = make_client([MockOutput(text="ok")], enabled=False)
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "hi",
                "store": False,
                "tools": [CORE_TOOL, CALENDAR_DEFERRED, TOOL_SEARCH_DECL],
            },
        )
        assert resp.status_code == 200
        names = _tool_names(engine.recorded[0]["kwargs"].get("tools"))
        assert names == {
            "shell",
            "planning_center__calendar_events",
            "planning_center__people_search",
        }


class TestClientHandoff:
    """Without a server-side corpus, tool_search is handed to the client."""

    def test_non_stream_client_execution(self, make_client):
        client, engine = make_client(
            [
                MockOutput(
                    tool_calls=[
                        {
                            "id": "call_h1",
                            "name": "tool_search",
                            "arguments": '{"query": "calendar"}',
                        }
                    ]
                )
            ]
        )
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "hi",
                "store": False,
                # Declared tool_search but no deferred tools: nothing for the
                # server to search over.
                "tools": [CORE_TOOL, TOOL_SEARCH_DECL],
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        items = [i for i in body["output"] if i["type"] == "tool_search_call"]
        assert items and items[0]["execution"] == "client"
        assert items[0]["arguments"] == {"query": "calendar"}
        # Never surfaced as a plain function_call.
        assert not [i for i in body["output"] if i["type"] == "function_call"]
        assert len(engine.recorded) == 1

    def test_stream_client_execution(self, make_client):
        client, engine = make_client(
            [
                MockOutput(
                    text="",
                    new_text="",
                    tool_calls=[
                        {
                            "id": "call_h2",
                            "name": "tool_search",
                            "arguments": '{"query": "people"}',
                        }
                    ],
                )
            ]
        )
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "hi",
                "stream": True,
                "store": False,
                "tools": [CORE_TOOL, CALENDAR_DEFERRED, TOOL_SEARCH_DECL],
            },
        )
        assert resp.status_code == 200, resp.text
        events = []
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        completed = [e for e in events if e.get("type") == "response.completed"]
        assert completed, events[-3:] if events else "no terminal event"
        output = completed[0]["response"]["output"]
        ts_calls = [i for i in output if i["type"] == "tool_search_call"]
        assert ts_calls and ts_calls[0]["execution"] == "client"
        assert not [i for i in output if i["type"] == "function_call"]

    @pytest.mark.parametrize("stream", [False, True])
    def test_client_execution_is_respected_with_a_deferred_corpus(
        self, make_client, stream
    ):
        client, engine = make_client(
            [
                MockOutput(
                    tool_calls=[
                        {
                            "id": "client_search",
                            "name": "tool_search",
                            "arguments": '{"goal":"find calendars"}',
                        }
                    ]
                ),
            ]
        )
        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "find calendars",
                "stream": stream,
                "tools": [CORE_TOOL, CALENDAR_DEFERRED, TOOL_SEARCH_DECL],
            },
        )
        body = _body(response, stream)
        search = next(i for i in body["output"] if i["type"] == "tool_search_call")
        assert search["execution"] == "client"
        assert search["arguments"] == {"goal": "find calendars"}
        assert search["call_id"] == "client_search"
        assert len(engine.recorded) == 1
        assert _tool_names(engine.recorded[0]["kwargs"]["tools"]) == {
            "shell",
            "tool_search",
        }

    @pytest.mark.parametrize("stream", [False, True])
    def test_codex_search_then_function_roundtrip(self, make_client, stream):
        client, engine = make_client(
            [
                MockOutput(
                    tool_calls=[
                        {
                            "id": "discover",
                            "name": "tool_search",
                            "arguments": '{"query":"calendar"}',
                        }
                    ]
                ),
                MockOutput(
                    tool_calls=[
                        {
                            "id": "invoke",
                            "name": "planning_center__calendar_events",
                            "arguments": '{"name":"Youth Night"}',
                        }
                    ]
                ),
                MockOutput(text="Youth Night is booked."),
            ]
        )
        tools = [CORE_TOOL, CALENDAR_DEFERRED, TOOL_SEARCH_DECL]
        history = [{"role": "user", "content": "find Youth Night"}]
        first = _body(
            client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "store": False,
                    "stream": stream,
                    "input": history,
                    "tools": tools,
                },
            ),
            stream,
        )
        history.extend(first["output"])
        loaded_namespace = {
            **CALENDAR_DEFERRED,
            "tools": CALENDAR_DEFERRED["tools"][:1],
        }
        history.append(
            {
                "type": "tool_search_output",
                "call_id": "discover",
                "execution": "client",
                "status": "completed",
                "tools": [loaded_namespace],
            }
        )
        second = _body(
            client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "store": False,
                    "stream": stream,
                    "input": history,
                    "tools": tools,
                },
            ),
            stream,
        )
        call = next(
            item for item in second["output"] if item["type"] == "function_call"
        )
        assert (call["namespace"], call["name"], call["call_id"]) == (
            "planning_center",
            "calendar_events",
            "invoke",
        )
        history.extend(second["output"])
        history.append(
            {
                "type": "function_call_output",
                "call_id": "invoke",
                "output": "booked",
            }
        )
        third = _body(
            client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "store": False,
                    "stream": stream,
                    "input": history,
                    "tools": tools,
                },
            ),
            stream,
        )
        assert any(
            part.get("text") == "Youth Night is booked."
            for item in third["output"]
            for part in item.get("content") or []
        )
        assert _tool_names(engine.recorded[0]["kwargs"]["tools"]) == {
            "shell",
            "tool_search",
        }
        for recorded in engine.recorded[1:]:
            assert _tool_names(recorded["kwargs"]["tools"]) == {
                "shell",
                "tool_search",
                "planning_center__calendar_events",
            }
        call_messages = [
            message
            for message in engine.recorded[2]["messages"]
            if message.get("tool_calls")
        ]
        assert call_messages[-1]["tool_calls"][0]["function"]["name"] == (
            "planning_center__calendar_events"
        )


class TestHostedSearchContinuation:
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("model_type", ["gpt_oss", "llama"])
    def test_continuation_keeps_reasoning_out_of_harmony_prompts(
        self, make_client, monkeypatch, stream, model_type
    ):
        reasoning = "Need to discover the calendar tool."
        preamble = "Let me check."
        client, engine = make_client(
            [
                MockOutput(
                    text=f"<think>{reasoning}</think>{preamble}",
                    tool_calls=[
                        {
                            "id": "search1",
                            "name": "tool_search",
                            "arguments": '{"query":"calendar"}',
                        }
                    ],
                ),
                MockOutput(text="Found it."),
            ]
        )
        engine._model_type = model_type
        tokenizer = engine.tokenizer
        monkeypatch.setattr(tokenizer, "tool_call_start", "<tool_call>", raising=False)
        monkeypatch.setattr(tokenizer, "tool_call_end", "</tool_call>", raising=False)
        original_template = tokenizer.apply_chat_template

        def apply_chat_template(messages, **kwargs):
            if model_type == "gpt_oss" and any(
                message.get("tool_calls")
                and message.get("content")
                and message.get("reasoning_content")
                for message in messages
            ):
                raise ValueError("Harmony rejects content and reasoning on a tool turn")
            return original_template(messages, **kwargs)

        monkeypatch.setattr(tokenizer, "apply_chat_template", apply_chat_template)
        body = _body(
            client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "find calendar",
                    "stream": stream,
                    "tools": [CALENDAR_DEFERRED, {"type": "tool_search"}],
                },
            ),
            stream,
        )
        assert body["status"] == "completed"
        assert len(engine.recorded) == 2
        assistant = next(
            message
            for message in engine.recorded[1]["messages"]
            if message.get("tool_calls")
        )
        assert assistant["content"] == preamble
        if model_type == "gpt_oss":
            assert all(
                "reasoning_content" not in message
                for message in engine.recorded[1]["messages"]
            )
        else:
            assert assistant["reasoning_content"] == reasoning
        assert any(
            part["text"] == reasoning
            for item in body["output"]
            if item["type"] == "reasoning"
            for part in item["summary"]
        )

    @pytest.mark.parametrize("stream", [False, True])
    def test_loaded_state_survives_disk_reload_and_stays_chain_local(
        self, make_client, tmp_path, stream
    ):
        from omlx.api.responses_utils import ResponseStore
        from omlx.server import _server_state

        client, engine = make_client(
            [
                MockOutput(
                    tool_calls=[
                        {
                            "id": "search1",
                            "name": "tool_search",
                            "arguments": '{"query":"calendar"}',
                        }
                    ]
                ),
                MockOutput(
                    tool_calls=[
                        {
                            "id": "calendar1",
                            "name": "planning_center__calendar_events",
                            "arguments": '{"name":"Youth Night"}',
                        }
                    ]
                ),
                MockOutput(text="Youth Night is booked."),
            ]
        )
        _server_state.responses_store = ResponseStore(state_dir=tmp_path)
        first = _body(
            client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "find Youth Night",
                    "stream": stream,
                    "tools": [CORE_TOOL, CALENDAR_DEFERRED, {"type": "tool_search"}],
                },
            ),
            stream,
        )
        _server_state.responses_store = ResponseStore(state_dir=tmp_path)
        second = _body(
            client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "stream": stream,
                    "previous_response_id": first["id"],
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "calendar1",
                            "output": "booked",
                        }
                    ],
                },
            ),
            stream,
        )
        assert second["status"] == "completed"
        assert _tool_names(engine.recorded[2]["kwargs"]["tools"]) == {
            "planning_center__calendar_events"
        }
        assert not any(
            "planning_center__people_search" in str(message)
            for message in engine.recorded[2]["messages"]
        )
        _body(
            client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "unrelated request",
                    "stream": stream,
                },
            ),
            stream,
        )
        assert not engine.recorded[3]["kwargs"].get("tools")

    @pytest.mark.parametrize("stream", [False, True])
    def test_mixed_search_and_function_calls_do_not_drop_client_work(
        self, make_client, stream
    ):
        client, engine = make_client(
            [
                MockOutput(
                    tool_calls=[
                        {
                            "id": "s1",
                            "name": "tool_search",
                            "arguments": '{"query":"calendar"}',
                        },
                        {"id": "exec1", "name": "shell", "arguments": '{"cmd":"pwd"}'},
                    ]
                ),
            ]
        )
        body = _body(
            client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "find calendar and cwd",
                    "stream": stream,
                    "tools": [CORE_TOOL, CALENDAR_DEFERRED, {"type": "tool_search"}],
                },
            ),
            stream,
        )
        assert len(engine.recorded) == 1
        assert [i["type"] for i in body["output"] if i["type"] != "message"] == [
            "tool_search_call",
            "tool_search_output",
            "function_call",
        ]
        call = next(i for i in body["output"] if i["type"] == "function_call")
        assert call["name"] == "shell" and call["call_id"] == "exec1"

    @pytest.mark.parametrize("stream", [False, True])
    def test_search_consumes_the_request_output_budget(self, make_client, stream):
        client, engine = make_client(
            [
                MockOutput(
                    completion_tokens=5,
                    tool_calls=[
                        {
                            "id": "s1",
                            "name": "tool_search",
                            "arguments": '{"query":"calendar"}',
                        }
                    ],
                ),
            ]
        )
        body = _body(
            client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "find calendar",
                    "stream": stream,
                    "max_output_tokens": 5,
                    "tools": [CALENDAR_DEFERRED, {"type": "tool_search"}],
                },
            ),
            stream,
        )
        assert len(engine.recorded) == 1
        assert body["status"] == "incomplete"
        assert body["incomplete_details"] == {"reason": "max_output_tokens"}
        assert body["usage"]["output_tokens"] == 5

    @pytest.mark.parametrize("stream", [False, True])
    def test_continuation_reduces_budget_and_rebuilds_prepared_prompt(
        self, make_client, monkeypatch, stream
    ):
        from unittest.mock import AsyncMock

        from omlx import server

        initial, continued = object(), object()
        prepare = AsyncMock(side_effect=[(1, initial), (2, continued)])
        monkeypatch.setattr(server, "_prepare_chat_prompt_for_request", prepare)
        client, engine = make_client(
            [
                MockOutput(
                    tool_calls=[
                        {
                            "id": "s1",
                            "name": "tool_search",
                            "arguments": '{"query":"calendar"}',
                        }
                    ]
                ),
                MockOutput(text="found it"),
            ]
        )
        body = _body(
            client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "find calendar",
                    "stream": stream,
                    "max_output_tokens": 16,
                    "tools": [CALENDAR_DEFERRED, {"type": "tool_search"}],
                },
            ),
            stream,
        )
        assert body["status"] == "completed"
        assert engine.recorded[0]["kwargs"]["_prepared_prompt"] is initial
        assert engine.recorded[1]["kwargs"]["_prepared_prompt"] is continued
        assert engine.recorded[1]["kwargs"]["max_tokens"] == 11
        assert engine.recorded[1]["kwargs"]["_request_id"] != (
            engine.recorded[0]["kwargs"]["_request_id"]
        )

    @pytest.mark.parametrize("stream", [False, True])
    def test_tool_choice_none_does_not_expose_search_or_tools(
        self, make_client, stream
    ):
        client, engine = make_client([MockOutput(text="ok")])
        _body(
            client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "hi",
                    "stream": stream,
                    "tool_choice": "none",
                    "tools": [CORE_TOOL, CALENDAR_DEFERRED, {"type": "tool_search"}],
                },
            ),
            stream,
        )
        assert not engine.recorded[0]["kwargs"].get("tools")

    @pytest.mark.asyncio
    async def test_disconnect_during_continuation_closes_engine_and_does_not_store(
        self, make_client, monkeypatch
    ):
        from omlx import server
        from omlx.api.responses_models import ResponsesRequest
        from omlx.api.responses_tool_search import DeferredToolSet, ToolSearchState
        from omlx.api.responses_utils import convert_responses_tools

        _, engine = make_client([])
        closed = asyncio.Event()
        continuation_started = asyncio.Event()
        rounds = 0
        request_ids = []
        stream_state = server._ResponsesStreamState()

        async def stream_chat(messages, **kwargs):
            nonlocal rounds
            rounds += 1
            request_ids.append(kwargs["_request_id"])
            try:
                if rounds == 1:
                    yield MockOutput(
                        tool_calls=[
                            {
                                "id": "s1",
                                "name": "tool_search",
                                "arguments": '{"query":"calendar"}',
                            }
                        ]
                    )
                else:
                    continuation_started.set()
                    yield MockOutput(text="found", new_text="found", finished=False)
                    await asyncio.Event().wait()
            finally:
                if rounds == 2:
                    closed.set()

        monkeypatch.setattr(engine, "stream_chat", stream_chat)
        request = ResponsesRequest(
            model="test-model",
            input="find calendar",
            stream=True,
            tools=[CALENDAR_DEFERRED, {"type": "tool_search"}],
        )
        pool = DeferredToolSet()
        tools = convert_responses_tools(
            request.tools, deferred=pool, tool_search_enabled=True
        )
        generator = server.stream_responses_api(
            engine,
            [{"role": "user", "content": "find calendar"}],
            request,
            tool_search_state=ToolSearchState(enabled=True, pool=pool),
            tools=tools,
            max_tokens=64,
            stream_state=stream_state,
            _request_id="initial-search",
        )

        async def consume():
            async for _event in generator:
                pass

        consumer = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(continuation_started.wait(), timeout=5)
            assert request_ids[0] != request_ids[1]
            assert stream_state.request_id == request_ids[1]
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await consumer
        finally:
            if not consumer.done():
                consumer.cancel()
            await generator.aclose()
        assert closed.is_set()
        assert rounds == 2
        assert len(server._server_state.responses_store) == 0

    @pytest.mark.asyncio
    async def test_stream_state_is_persisted_before_terminal_event(self, make_client):
        from omlx import server
        from omlx.api.responses_models import ResponsesRequest

        _, engine = make_client([MockOutput(text="done")])
        request = ResponsesRequest(model="test-model", input="hello", stream=True)
        generator = server.stream_responses_api(
            engine,
            [{"role": "user", "content": "hello"}],
            request,
            max_tokens=64,
        )
        try:
            async for event in generator:
                if event.startswith("event: response.completed"):
                    response = json.loads(event.split("data: ", 1)[1])["response"]
                    assert (
                        server._server_state.responses_store.get(response["id"])
                        is not None
                    )
                    break
        finally:
            await generator.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("enabled", [False, True])
    async def test_codex_catalog_reads_capability_from_server_status(
        self, make_client, enabled
    ):
        from omlx import server
        from omlx.integrations.base import IntegrationContext
        from omlx.integrations.codex import codex_model_catalog

        make_client([], enabled=enabled)
        status = await server.list_models_status()
        catalog = codex_model_catalog(
            IntegrationContext(
                host="127.0.0.1",
                port=8000,
                model="test-model",
                models_status_map={model["id"]: model for model in status["models"]},
            )
        )
        assert catalog["models"][0]["supports_search_tool"] is enabled

    @pytest.mark.parametrize("stream", [False, True])
    def test_hosted_search_rounds_are_bounded(self, make_client, stream):
        client, engine = make_client(
            [
                MockOutput(
                    tool_calls=[
                        {
                            "id": f"s{index}",
                            "name": "tool_search",
                            "arguments": '{"query":"calendar"}',
                        }
                    ]
                )
                for index in range(5)
            ]
        )
        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "find calendar",
                "stream": stream,
                "tools": [CALENDAR_DEFERRED, {"type": "tool_search"}],
            },
        )
        assert len(engine.recorded) == 5
        assert "tool_search" not in _tool_names(engine.recorded[-1]["kwargs"]["tools"])
        if stream:
            terminal = _events(response)[-1]
            assert terminal["type"] == "response.failed"
            assert terminal["response"]["error"]["code"] == "tool_search_limit_exceeded"
        else:
            assert response.status_code == 500
            assert "search round limit" in response.text

    def test_continuation_preflight_failure_emits_failed_not_completed(
        self, make_client, monkeypatch
    ):
        from unittest.mock import AsyncMock

        client, engine = make_client(
            [
                MockOutput(
                    tool_calls=[
                        {
                            "id": "s1",
                            "name": "tool_search",
                            "arguments": '{"query":"calendar"}',
                        }
                    ]
                ),
            ]
        )
        monkeypatch.setattr(
            engine,
            "preflight_chat",
            AsyncMock(side_effect=[None, ValueError("continuation exceeds context")]),
        )
        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "find calendar",
                "stream": True,
                "tools": [CALENDAR_DEFERRED, {"type": "tool_search"}],
            },
        )
        events = _events(response)
        assert not any(e["type"] == "response.completed" for e in events)
        assert events[-1]["type"] == "response.failed"
        assert "exceeds context" in events[-1]["response"]["error"]["message"]
        assert len(engine.recorded) == 1

    def test_continuation_engine_failure_retains_completed_output(
        self, make_client, monkeypatch
    ):
        from omlx import server

        client, engine = make_client([])

        async def stream_chat(messages, **kwargs):
            engine.recorded.append({"messages": list(messages), "kwargs": kwargs})
            if len(engine.recorded) == 2:
                raise RuntimeError("continuation generation failed")
            yield MockOutput(
                text="<think>Find the calendar.</think>Searching.",
                tool_calls=[
                    {
                        "id": "s1",
                        "name": "tool_search",
                        "arguments": '{"query":"calendar"}',
                    }
                ],
            )

        monkeypatch.setattr(engine, "stream_chat", stream_chat)
        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "find calendar",
                "stream": True,
                "store": True,
                "tools": [CALENDAR_DEFERRED, {"type": "tool_search"}],
            },
        )
        events = _events(response)
        completed_items = [
            event["item"]
            for event in events
            if event["type"] == "response.output_item.done"
        ]
        assert any(item["type"] == "tool_search_output" for item in completed_items)
        assert len(engine.recorded) == 2
        assert events[-1]["type"] == "response.failed"
        assert events[-1]["response"]["output"] == completed_items
        assert events[-1]["response"]["error"] == {
            "code": "server_error",
            "message": "continuation generation failed",
        }
        assert len(server._server_state.responses_store) == 0

    @pytest.mark.parametrize("stream", [False, True])
    def test_native_response_format_survives_search_continuation(
        self, make_client, monkeypatch, stream
    ):
        from omlx import server

        monkeypatch.setattr(server, "_native_response_format", lambda engine, fmt: fmt)
        client, engine = make_client(
            [
                MockOutput(
                    tool_calls=[
                        {
                            "id": "s1",
                            "name": "tool_search",
                            "arguments": '{"query":"calendar"}',
                        }
                    ]
                ),
                MockOutput(text='{"found":true}'),
            ]
        )
        body = _body(
            client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "find calendar",
                    "stream": stream,
                    "tools": [CALENDAR_DEFERRED, {"type": "tool_search"}],
                    "text": {"format": {"type": "json_object"}},
                },
            ),
            stream,
        )
        assert body["status"] == "completed"
        assert len(engine.recorded) == 2
        assert all(
            call["kwargs"]["response_format"] == {"type": "json_object"}
            for call in engine.recorded
        )


def test_responses_route_is_registered_once():
    from omlx import server

    routes = [
        route
        for route in server.app.routes
        if getattr(route, "path", None) == "/v1/responses" and "POST" in route.methods
    ]
    assert len(routes) == 1
    assert routes[0].endpoint is server.create_response


class TestHistoryReplayPromotesTools:
    """Full-history replay (the Codex pattern) keeps loaded tools active."""

    @pytest.mark.parametrize("stream", [False, True])
    def test_tool_choice_none_preserves_loaded_tools_for_the_next_turn(
        self, make_client, stream
    ):
        client, engine = make_client(
            [MockOutput(text="No tool call needed."), MockOutput(text="Ready.")]
        )
        loaded_spec = {
            "type": "namespace",
            "name": "planning_center",
            "tools": [CALENDAR_DEFERRED["tools"][0]],
        }
        first = _body(
            client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": [
                        {"role": "user", "content": "find calendar"},
                        {
                            "type": "tool_search_call",
                            "call_id": "s1",
                            "execution": "client",
                            "arguments": {"query": "calendar"},
                        },
                        {
                            "type": "tool_search_output",
                            "call_id": "s1",
                            "execution": "client",
                            "tools": [loaded_spec],
                        },
                    ],
                    "tools": [TOOL_SEARCH_DECL],
                    "tool_choice": "none",
                    "store": True,
                    "stream": stream,
                },
            ),
            stream,
        )
        assert not engine.recorded[0]["kwargs"].get("tools")
        _body(
            client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "Use the discovered calendar tool.",
                    "previous_response_id": first["id"],
                    "tools": [TOOL_SEARCH_DECL],
                    "stream": stream,
                },
            ),
            stream,
        )
        assert "planning_center__calendar_events" in _tool_names(
            engine.recorded[1]["kwargs"].get("tools")
        )

    def test_history_tool_search_output_loads_tools(self, make_client):
        client, engine = make_client([MockOutput(text="Youth Night is booked.")])
        matched_spec = {
            "type": "namespace",
            "name": "planning_center",
            "description": "Tools in the planning_center namespace.",
            "tools": [
                {
                    "type": "function",
                    "name": "calendar_events",
                    "description": "Search calendar events for the church",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                    "defer_loading": True,
                }
            ],
        }
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "store": False,
                "input": [
                    {"role": "user", "content": "find youth night"},
                    {
                        "type": "tool_search_call",
                        "call_id": "call_p1",
                        "execution": "client",
                        "arguments": {"query": "calendar"},
                    },
                    {
                        "type": "tool_search_output",
                        "call_id": "call_p1",
                        "status": "completed",
                        "execution": "client",
                        "tools": [matched_spec],
                    },
                    {"role": "user", "content": "now check it"},
                ],
                "tools": [CORE_TOOL, CALENDAR_DEFERRED, TOOL_SEARCH_DECL],
            },
        )
        assert resp.status_code == 200, resp.text
        names = _tool_names(engine.recorded[0]["kwargs"].get("tools"))
        # Loaded for this and every later turn of the chain.
        assert "planning_center__calendar_events" in names
        assert "planning_center__people_search" not in names

        # The model's history shows the discovery round trip.
        messages = engine.recorded[0]["messages"]
        assistant_calls = [
            m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert assistant_calls
        assert assistant_calls[-1]["tool_calls"][0]["function"]["name"] == "tool_search"
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert tool_msgs and "calendar_events" in tool_msgs[-1]["content"]
