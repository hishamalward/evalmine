"""The OpenAI Chat Completions API adapter. Spec: docs/spec.md S10.

httpx only, no SDK. Structured output is enforced with ``response_format:
json_schema`` (OpenAI's native Structured Outputs), so a schema request gets
``schema_mode: native``.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import AdapterError, Request, Response, post_json

API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIAdapter:
    name = "openai"
    version = 1

    def __init__(
        self,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self._transport = transport

    def schema_mode_for(self, schema: dict[str, Any] | None) -> str:
        return "native" if schema is not None else "prompted"

    def complete(self, req: Request) -> Response:
        if not self.api_key:
            raise AdapterError(
                "OPENAI_API_KEY is not set; evalmine reads keys from the environment "
                "and from nowhere else",
                retryable=False,
                status=401,
            )

        messages: list[dict[str, str]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})

        body: dict[str, Any] = {
            "model": req.model_id,
            "messages": messages,
            "temperature": req.temperature,
            # The unified token-limit parameter across chat and reasoning
            # models; "max_tokens" is the deprecated name for the same knob.
            "max_completion_tokens": req.max_tokens,
        }
        if req.top_p is not None:
            body["top_p"] = req.top_p
        if req.stop:
            body["stop"] = list(req.stop)
        if req.schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "evalmine_response",
                    "schema": req.schema,
                    "strict": True,
                },
            }

        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

        data, latency_ms = post_json(
            API_URL, headers, body, req.timeout_s, self._transport, "openai"
        )
        text, finish_reason = _extract(data)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}

        return Response(
            text=text,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            cached_input_tokens=int(prompt_details.get("cached_tokens") or 0),
            reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            schema_mode=self.schema_mode_for(req.schema),
        )


def _extract(data: dict[str, Any]) -> tuple[str, str]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AdapterError("openai response has no choices", retryable=False)
    choice = choices[0]
    if not isinstance(choice, dict):
        raise AdapterError("openai response has a malformed choice", retryable=False)
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise AdapterError("openai response has no message content", retryable=False)
    finish_reason = str(choice.get("finish_reason") or "stop")
    return message["content"], finish_reason
