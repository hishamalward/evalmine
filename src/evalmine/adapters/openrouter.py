"""The OpenRouter Chat Completions API adapter.

OpenRouter speaks the OpenAI-compatible chat-completions shape, but it has its
own endpoint, credential, attribution headers, routing controls, and usage
accounting. Keeping it as a dedicated adapter prevents an OpenRouter model from
silently depending on ``OPENAI_API_KEY`` or being reported as an OpenAI call.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import AdapterError, Request, Response, post_json

API_URL = "https://openrouter.ai/api/v1/chat/completions"
APP_URL = "https://github.com/hishamalward/evalmine"
APP_TITLE = "evalmine"


class OpenRouterAdapter:
    name = "openrouter"
    version = 1

    def __init__(
        self,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        *,
        http_referer: str | None = APP_URL,
        app_title: str | None = APP_TITLE,
    ) -> None:
        self.api_key = (
            api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        )
        self.http_referer = http_referer
        self.app_title = app_title
        self._transport = transport

    def schema_mode_for(self, schema: dict[str, Any] | None) -> str:
        return "native" if schema is not None else "prompted"

    def complete(self, req: Request) -> Response:
        if not self.api_key:
            raise AdapterError(
                "OPENROUTER_API_KEY is not set; evalmine reads keys from the environment "
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
            # OpenRouter can route one model slug across several upstreams.
            # Refuse a route that would drop the requested schema parameter.
            body["provider"] = {"require_parameters": True}

        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title

        data, latency_ms = post_json(
            API_URL, headers, body, req.timeout_s, self._transport, "openrouter"
        )
        text, finish_reason = _extract(data)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        prompt_details = usage.get("prompt_tokens_details")
        completion_details = usage.get("completion_tokens_details")
        prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
        completion_details = (
            completion_details if isinstance(completion_details, dict) else {}
        )

        return Response(
            text=text,
            input_tokens=usage.get("prompt_tokens"),
            # OpenRouter's completion count already includes reasoning tokens.
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
        raise AdapterError("openrouter response has no choices", retryable=False)
    choice = choices[0]
    if not isinstance(choice, dict):
        raise AdapterError("openrouter response has a malformed choice", retryable=False)
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise AdapterError("openrouter response has no message content", retryable=False)
    finish_reason = str(choice.get("finish_reason") or "stop")
    return message["content"], finish_reason
