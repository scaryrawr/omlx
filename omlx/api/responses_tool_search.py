# SPDX-License-Identifier: Apache-2.0
"""Deferred tool loading + ``tool_search`` support for the Responses API.

Codex (and other Responses clients that follow the OpenAI deferred-tool
contract) send function tools flagged ``defer_loading: true`` — usually
inside ``namespace`` containers, one per MCP server — plus a
``tool_search`` tool declaration. Deferred tools must stay out of the
model's active prompt while remaining discoverable: the model calls
``tool_search`` with a query, matching tools are returned with their full
schemas, and from the next model call onward they become active.

Search ranking mirrors Codex's own handler (crates ``core/src/tools/
handlers/tool_search.rs``): BM25 over name + description + parameter
names/descriptions of the deferred corpus, default limit 8.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

TOOL_SEARCH_TOOL_NAME = "tool_search"
TOOL_SEARCH_DEFAULT_LIMIT = 8
TOOL_SEARCH_MAX_LIMIT = 32
TOOL_SEARCH_MAX_ROUNDS = 4
# Namespace Codex uses for tools that have no explicit namespace.
DEFAULT_FUNCTION_NAMESPACE = "functions"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# BM25 parameters (Okapi defaults, matching the bm25 crate's English preset).
_BM25_K1 = 1.2
_BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _append_schema_search_text(schema: Any, parts: list[str]) -> None:
    """Mirror codex-tools' append_schema_search_text for JSON schemas."""
    if not isinstance(schema, dict):
        return
    description = schema.get("description")
    if isinstance(description, str) and description.strip():
        parts.append(description.strip())
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, sub in properties.items():
            if name.strip():
                parts.append(name.strip())
            _append_schema_search_text(sub, parts)
    _append_schema_search_text(schema.get("items"), parts)
    for variant in schema.get("anyOf") or schema.get("oneOf") or []:
        _append_schema_search_text(variant, parts)


@dataclass(frozen=True)
class DeferredTool:
    """A function tool withheld from the model prompt until it is loaded."""

    wire_name: str
    # None for flat tools; the namespace name for namespace members.
    namespace: str | None
    name: str
    description: str = ""
    parameters: dict[str, Any] | None = None
    strict: bool | None = None

    @property
    def is_namespace_member(self) -> bool:
        return bool(self.namespace) and self.namespace != DEFAULT_FUNCTION_NAMESPACE

    def search_text(self) -> str:
        parts = [self.name, self.name.replace("_", " ")]
        if self.is_namespace_member:
            parts.append(self.namespace or "")
        if self.description:
            parts.append(self.description)
        if self.parameters:
            _append_schema_search_text(self.parameters, parts)
        return " ".join(p.strip() for p in parts if p and p.strip())

    def to_openai_tool(self) -> dict[str, Any]:
        """Chat-Completions nested form used for template/grammar rendering."""
        function: dict[str, Any] = {"name": self.wire_name}
        if self.description:
            function["description"] = self.description
        function["parameters"] = self.parameters or {"type": "object", "properties": {}}
        if self.strict is not None:
            function["strict"] = self.strict
        return {"type": "function", "function": function}

    def to_responses_spec(self) -> dict[str, Any]:
        """Responses-API flat form (with the defer flag carried along)."""
        spec: dict[str, Any] = {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters or {"type": "object", "properties": {}},
            "defer_loading": True,
        }
        if self.strict is not None:
            spec["strict"] = self.strict
        return spec


def allocate_tool_wire_name(namespace: str | None, name: str, taken: set[str]) -> str:
    """Allocate one collision-safe model name for a Responses tool identity."""
    base = (
        f"{namespace.rstrip('_')}__{name.lstrip('_')}"
        if namespace and namespace != DEFAULT_FUNCTION_NAMESPACE
        else name
    )
    wire, suffix = base, 2
    while wire in taken:
        wire = f"{base}_{suffix}"
        suffix += 1
    return wire


class DeferredToolSet:
    """Ordered registry of deferred tools with BM25 search."""

    def __init__(self, tools: Iterable[DeferredTool] | None = None):
        self._tools: dict[str, DeferredTool] = {}
        self._order: list[str] = []
        if tools:
            for tool in tools:
                self.add(tool)

    def add(self, tool: DeferredTool) -> None:
        if tool.wire_name not in self._tools:
            self._order.append(tool.wire_name)
        self._tools[tool.wire_name] = tool

    def extend_specs(self, specs: Iterable[dict[str, Any]]) -> list[DeferredTool]:
        """Index LoadableToolSpec shapes (from tool_search results or store).

        Returns the tools newly added to the set (already-known wire names
        are not reported again).
        """
        added: list[DeferredTool] = []
        for tool in parse_loadable_specs(specs):
            if tool.wire_name not in self._tools:
                added.append(tool)
            self.add(tool)
        return added

    def get(self, wire_name: str) -> DeferredTool | None:
        return self._tools.get(wire_name)

    def __contains__(self, wire_name: str) -> bool:
        return wire_name in self._tools

    def __bool__(self) -> bool:
        return bool(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def tools(self) -> list[DeferredTool]:
        return [self._tools[name] for name in self._order]

    def view(self, loaded: Iterable[str]) -> list[dict[str, Any]]:
        """Nested tool defs for the loaded wire names, in registration order."""
        loaded_set = set(loaded)
        return [
            tool.to_openai_tool()
            for tool in self.tools()
            if tool.wire_name in loaded_set
        ]

    def search(self, query: str, limit: int | None = None) -> list[DeferredTool]:
        """BM25 over the deferred corpus; stable order on score ties."""
        limit = (
            TOOL_SEARCH_DEFAULT_LIMIT
            if limit is None
            else min(TOOL_SEARCH_MAX_LIMIT, max(1, int(limit)))
        )
        docs = [
            (name, _tokenize(tool.search_text())) for name, tool in self._tools.items()
        ]
        query_tokens = _tokenize(query)
        if not docs or not query_tokens:
            return []

        n_docs = len(docs)
        doc_len = {name: max(1, len(tokens)) for name, tokens in docs}
        avg_len = sum(doc_len.values()) / n_docs
        df: dict[str, int] = {}
        for _name, tokens in docs:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        scores: dict[str, float] = {}
        for name, tokens in docs:
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            score = 0.0
            length = doc_len[name]
            for token in set(query_tokens):
                term_freq = tf.get(token, 0)
                if not term_freq:
                    continue
                df_t = df[token]
                idf = math.log(1.0 + (n_docs - df_t + 0.5) / (df_t + 0.5))
                norm = 1.0 - _BM25_B + _BM25_B * (length / avg_len)
                score += (
                    idf * (term_freq * (_BM25_K1 + 1.0)) / (term_freq + _BM25_K1 * norm)
                )
            if score > 0:
                scores[name] = score

        ranked = sorted(self._order, key=lambda name: -scores.get(name, 0.0))
        return [self._tools[name] for name in ranked[:limit] if name in scores]


def parse_loadable_specs(specs: Iterable[Any]) -> list[DeferredTool]:
    """Parse LoadableToolSpec entries (namespace or flat function shapes)."""
    tools: list[DeferredTool] = []
    for spec in specs or []:
        if not isinstance(spec, dict):
            continue
        spec_type = spec.get("type")
        if spec_type == "namespace":
            namespace = spec.get("name")
            if not isinstance(namespace, str) or not namespace:
                continue
            for member in spec.get("tools") or []:
                tool = _parse_function_member(namespace, member)
                if tool is not None:
                    tools.append(tool)
        elif spec_type == "function":
            tool = _parse_function_member(None, spec)
            if tool is not None:
                tools.append(tool)
    return tools


def _parse_function_member(namespace: str | None, member: Any) -> DeferredTool | None:
    if not isinstance(member, dict):
        return None
    if member.get("type") != "function":
        return None
    if namespace == DEFAULT_FUNCTION_NAMESPACE:
        namespace = None
    name = member.get("name")
    if not isinstance(name, str) or not name:
        return None
    parameters = member.get("parameters")
    return DeferredTool(
        wire_name=allocate_tool_wire_name(namespace, name, set()),
        namespace=namespace,
        name=name,
        description=member.get("description") or "",
        parameters=parameters if isinstance(parameters, dict) else None,
        strict=member.get("strict"),
    )


def collect_tool_search_results(
    input_items: Iterable[Any] | None,
) -> list[dict[str, Any]]:
    """Gather tool specs returned by tool_search_output items in history.

    Accepts pydantic ``InputItem``s or raw dicts (whichever layer hands the
    items over) and returns the LoadableToolSpec dicts to re-index.
    """
    specs: list[dict[str, Any]] = []
    for item in input_items or []:
        item_type = _get(item, "type")
        if item_type != "tool_search_output":
            continue
        tools = _get(item, "tools")
        if isinstance(tools, list):
            for spec in tools:
                if isinstance(spec, dict):
                    specs.append(spec)
    return specs


def _get(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    value = getattr(item, key, None)
    if value is None:
        extra = getattr(item, "model_extra", None) or {}
        value = extra.get(key)
    return value


def build_tool_search_results(matched: list[DeferredTool]) -> list[dict[str, Any]]:
    """Coalesce hits into LoadableToolSpec namespace shapes (Codex format)."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for tool in matched:
        namespace = tool.namespace or DEFAULT_FUNCTION_NAMESPACE
        if namespace not in grouped:
            grouped[namespace] = []
            order.append(namespace)
        grouped[namespace].append(tool.to_responses_spec())
    return [
        {
            "type": "namespace",
            "name": namespace,
            "description": f"Tools in the {namespace} namespace.",
            "tools": grouped[namespace],
        }
        for namespace in order
    ]


def tool_search_call_item(
    call_id: str | None,
    arguments: dict[str, Any],
    *,
    execution: str,
    item_id: str | None = None,
) -> dict[str, Any]:
    """Responses-API ``tool_search_call`` output item."""
    return {
        "type": "tool_search_call",
        "id": item_id or f"tsc_{uuid.uuid4().hex[:24]}",
        "call_id": call_id,
        "status": "completed",
        "execution": execution,
        "arguments": arguments,
    }


def tool_search_output_item(
    call_id: str | None,
    results: list[dict[str, Any]],
    *,
    execution: str,
    item_id: str | None = None,
) -> dict[str, Any]:
    """Responses-API ``tool_search_output`` output item."""
    return {
        "type": "tool_search_output",
        "id": item_id or f"tso_{uuid.uuid4().hex[:24]}",
        "call_id": call_id,
        "status": "completed",
        "execution": execution,
        "tools": results,
    }


def parse_tool_search_arguments(raw: Any) -> tuple[str, int]:
    """Extract (query, limit) from a tool_search call's arguments."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            raw = {}
    if not isinstance(raw, dict):
        raw = {}
    query = raw.get("query")
    query = query if isinstance(query, str) else ""
    limit = raw.get("limit")
    try:
        limit = int(limit) if limit is not None else TOOL_SEARCH_DEFAULT_LIMIT
    except (TypeError, ValueError, OverflowError):
        limit = TOOL_SEARCH_DEFAULT_LIMIT
    return query.strip(), min(TOOL_SEARCH_MAX_LIMIT, max(1, limit))


class ToolSearchState:
    """Per-request tool-search context for the /v1/responses handler."""

    def __init__(
        self,
        *,
        enabled: bool,
        pool: DeferredToolSet,
        loaded: set[str] | None = None,
        execution: str = "server",
    ):
        self.enabled = enabled
        self.execution = execution
        self.pool = pool
        # Wire names promoted into the model's active tool set.
        self.loaded: set[str] = set(loaded or ())

    def restore_loaded(
        self,
        specs: Iterable[dict[str, Any]],
        aliases: dict[str, tuple[str, str]],
        eager_tools: list[dict[str, Any]],
    ) -> None:
        """Reconcile replayed schemas with current request bindings."""
        taken = {tool["function"]["name"] for tool in eager_tools} | set(aliases)
        taken.update(tool.wire_name for tool in self.pool.tools())
        declared = {tool.wire_name for tool in self.pool.tools()}
        for restored in parse_loadable_specs(specs):
            current = next(
                (
                    tool
                    for tool in self.pool.tools()
                    if (tool.namespace, tool.name)
                    == (restored.namespace, restored.name)
                ),
                None,
            )
            if current is None:
                wire = next(
                    (
                        name
                        for name, identity in aliases.items()
                        if identity == (restored.namespace, restored.name)
                    ),
                    None,
                )
                if wire is not None or (
                    restored.namespace is None and restored.name in taken
                ):
                    continue  # An eager request tool already supplies this identity.
                wire = allocate_tool_wire_name(restored.namespace, restored.name, taken)
                current = replace(restored, wire_name=wire)
                self.pool.add(current)
                taken.add(wire)
            elif current.wire_name not in declared:
                current = replace(restored, wire_name=current.wire_name)
                self.pool.add(current)
            self.loaded.add(current.wire_name)
            if current.is_namespace_member:
                aliases[current.wire_name] = (current.namespace, current.name)

    def mark_loaded(self, tools: Iterable[DeferredTool]) -> None:
        for tool in tools:
            self.loaded.add(tool.wire_name)

    def loaded_tool_defs(self) -> list[dict[str, Any]]:
        """Nested tool defs for every loaded deferred tool."""
        return self.pool.view(self.loaded)

    def run_search(self, raw_arguments: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute one server-side search; returns (call_item, output_item)."""
        query, limit = parse_tool_search_arguments(raw_arguments)
        hits = self.pool.search(query, limit) if query else []
        results = build_tool_search_results(hits)
        self.mark_loaded(hits)
        arguments: dict[str, Any] = {"query": query, "limit": limit}
        item_suffix = uuid.uuid4().hex[:24]
        return (
            tool_search_call_item(
                None, arguments, execution="server", item_id=f"tsc_{item_suffix}"
            ),
            tool_search_output_item(
                None, results, execution="server", item_id=f"tso_{item_suffix}"
            ),
        )

    def serialize_loaded_specs(self) -> list[dict[str, Any]]:
        return build_tool_search_results(
            [tool for tool in self.pool.tools() if tool.wire_name in self.loaded]
        )
