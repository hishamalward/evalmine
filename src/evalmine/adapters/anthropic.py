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
import re
from typing import Any

import httpx

from .base import AdapterError, Request, Response, post_json

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

#: The forced tool the model must call when a schema is requested. The name is
#: arbitrary; only the shape of ``input`` matters to the caller.
_SCHEMA_TOOL_NAME = "emit_result"

_MODEL_ID = re.compile(r"^claude-(?P<family>opus|sonnet|haiku|fable|mythos)-(?P<major>\d+)(?:-(?P<minor>\d+))?")


def sampling_params_supported(model_id: str) -> bool:
    """Whether the Messages API still accepts ``temperature``/``top_p`` for ``model_id``.

    The API removed the sampling parameters on Opus 4.7 and everything after it
    (Opus 4.8, Opus 5, Sonnet 5, Fable 5, Mythos 5): sending them is a 400,
    ``"temperature is deprecated for this model"``, not a warning. Older models
    (Opus/Sonnet 4.6, Haiku 4.5, the 3.x line) accept them. For a model that
    rejects them the adapter omits the fields, which means the suite's
    ``temperature`` is recorded in the report but had no effect on that model -
    there is no equivalent knob to translate it to.
    """
    m = _MODEL_ID.match(model_id)
    if m is None:
        return True
    family = m.group("family")
    major = int(m.group("major"))
    minor = int(m.group("minor") or 0)
    if family in ("fable", "mythos"):
        return False
    if major >= 5:
        return False
    return not (family == "opus" and (major, minor) >= (4, 7))


def thinking_defaults_on(model_id: str) -> bool:
    """Whether omitting ``thinking`` turns it ON for ``model_id``.

    From Opus 5 / Sonnet 5 onward a request without a ``thinking`` field runs
    with adaptive thinking, and thinking tokens count against ``max_tokens`` -
    on a 900-token answer budget the model can spend the whole budget thinking
    and return no text at all. evalmine's params are an answer budget, so the
    adapter sends ``thinking: {type: "disabled"}`` for these models, which the
    API accepts at the default effort. Fable/Mythos cannot disable thinking
    (that is a 400) and are left alone; older models default to off already.
    """
    m = _MODEL_ID.match(model_id)
    if m is None:
        return False
    return m.group("family") in ("opus", "sonnet", "haiku") and int(m.group("major")) >= 5


class AnthropicAdapter:
    name = "anthropic"
    version = 3

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
        }
        if sampling_params_supported(req.model_id):
            body["temperature"] = req.temperature
            if req.top_p is not None:
                body["top_p"] = req.top_p
        if thinking_defaults_on(req.model_id):
            body["thinking"] = {"type": "disabled"}
        if req.system:
            body["system"] = req.system
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
        if finish_reason == "max_tokens":
            return "", finish_reason
        raise AdapterError(
            "anthropic response carries a schema request but no tool_use block", retryable=False
        )

    texts = [
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    if not texts:
        #: A budget exhausted before the first text token (thinking-only
        #: content) is a measurable outcome - an empty answer with
        #: ``finish_reason: max_tokens`` - not a reason to abort the run.
        if finish_reason == "max_tokens":
            return "", finish_reason
        raise AdapterError("anthropic response has no text block", retryable=False)
    return "".join(texts), finish_reason
