# SPDX-License-Identifier: Apache-2.0
"""Claude Code discovery IDs scoped to the Claude gateway endpoints."""

from typing import TypedDict

CLAUDE_GATEWAY_PATH = "/claude"
_MODEL_PREFIX = "omlx-claude/"


class ClaudeGatewayModelInfo(TypedDict):
    id: str
    display_name: str
    max_model_len: int | None


def claude_gateway_model_id(model_id: str) -> str:
    # Claude Code only discovers IDs containing "claude" or "anthropic".
    return f"{_MODEL_PREFIX}{model_id}"


def claude_gateway_served_id(model_id: str) -> str:
    return model_id.removeprefix(_MODEL_PREFIX)
