"""Executable contracts embedded in user-facing documentation."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mcp_example_is_valid_and_matches_the_spec():
    example_text = (ROOT / ".mcp.json.example").read_text(encoding="utf-8")
    example = json.loads(example_text)
    server = example["mcpServers"]["evalmine"]

    assert server["type"] == "stdio"
    assert server["command"] == "evalmine-mcp"
    assert server["env"]["EVALMINE_MCP_SUITE_ROOT"] == "${CLAUDE_PROJECT_DIR:-.}"
    assert not any(name.endswith("_API_KEY") for name in server["env"])

    spec = (ROOT / "docs" / "spec.md").read_text(encoding="utf-8")
    section = spec.split("### 11.5 `.mcp.json.example`", 1)[1]
    embedded = section.split("```json\n", 1)[1].split("```", 1)[0]
    assert embedded == example_text
