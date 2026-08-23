"""The Anthropic Messages API adapter. Spec: docs/spec.md S10.

httpx only, no SDK. Structured output is enforced by forcing a single tool
call whose ``input_schema`` is the task's JSON Schema - the real mechanism the
Messages API offers for getting schema-shaped JSON back, so a schema request
gets ``schema_mode: native``. A request with no schema gets a plain text
reply and ``schema_mode: prompted`` (evalmine appends nothing to the prompt
in that case; "prompted" just means "not enforced by the provider").
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from .base import AdapterError, Request, Response, post_json

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

#: The forced tool the model must call when a schema is requested. The name is
#: arbitrary; only the shape of ``input`` matters to the caller.
_SCHEMA_TOOL_NAME = "emit_result"


class AnthropicAdapter:
    name = "anthropic"
    version = 1

    def __init__(
        self,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        #: Read once at construction, spec S10: "Keys come from ... the
        #: environment and from nowhere else." ``transport`` exists only so
        #: tests can substitute httpx.MockTransport; production code never
        #: passes it.
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        self._transport = transport

    def schema_mode_for(self, schema: dict[str, Any] | None) -> str:
        return "native" if schema is not None else "prompted"

    def complete(self, req: Request) -> Response:
        if not self.api_key:
            raise AdapterError(
                "ANTHROPIC_API_KEY is not set; evalmine reads keys from the environment "
                "and from nowhere else",
                retryable=False,
                status=401,
            )

        body: dict[str, Any] = {
            "model": req.model_id,
            "max_tokens": req.max_tokens,
            "messages": [{"role": "user", "content": req.prompt}],
            "temperature": req.temperature,
        }
        if req.system:
            body["system"] = req.system
        if req.top_p is not None:
            body["top_p"] = req.top_p
        if req.stop:
            body["stop_sequences"] = list(req.stop)
        if req.schema is not None:
            body["tools"] = [
                {
                    "name": _SCHEMA_TOOL_NAME,
                    "description": "Return the answer matching the required JSON structure.",
                    "input_schema": req.schema,
                }
            ]
            body["tool_choice"] = {"type": "tool", "name": _SCHEMA_TOOL_NAME}

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

        data, latency_ms = post_json(
            API_URL, headers, body, req.timeout_s, self._transport, "anthropic"
        )
        text, finish_reason = _extract(data, wants_schema=req.schema is not None)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}

        return Response(
            text=text,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cached_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
            reasoning_tokens=int(usage.get("thinking_tokens") or 0),
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            schema_mode=self.schema_mode_for(req.schema),
        )


def _extract(data: dict[str, Any], wants_schema: bool) -> tuple[str, str]:
    blocks = data.get("content")
    if not isinstance(blocks, list) or not blocks:
        raise AdapterError("anthropic response has no content blocks", retryable=False)

    finish_reason = str(data.get("stop_reason") or "stop")

    if wants_schema:
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return json.dumps(block.get("input", {}), ensure_ascii=False), finish_reason
        raise AdapterError(
            "anthropic response carries a schema request but no tool_use block", retryable=False
        )

    texts = [
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    if not texts:
        raise AdapterError("anthropic response has no text block", retryable=False)
    return "".join(texts), finish_reason
