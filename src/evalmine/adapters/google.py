"""The Google Gemini generateContent API adapter. Spec: docs/spec.md S10.

httpx only, no SDK. Structured output is enforced with
``generationConfig.responseSchema`` + ``responseMimeType: application/json``
(Gemini's native structured output), so a schema request gets
``schema_mode: native``.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import AdapterError, Request, Response, post_json

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GoogleAdapter:
    name = "google"
    version = 1

    def __init__(
        self,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        #: Spec S10: keys come from ANTHROPIC_API_KEY, OPENAI_API_KEY and
        #: GOOGLE_API_KEY only - "nothing else is read". GEMINI_API_KEY is a
        #: real convention elsewhere but is deliberately not honoured here.
        self.api_key = api_key if api_key is not None else os.environ.get("GOOGLE_API_KEY")
        self._transport = transport

    def schema_mode_for(self, schema: dict[str, Any] | None) -> str:
        return "native" if schema is not None else "prompted"

    def complete(self, req: Request) -> Response:
        if not self.api_key:
            raise AdapterError(
                "GOOGLE_API_KEY is not set; evalmine reads keys from the environment "
                "and from nowhere else",
                retryable=False,
                status=401,
            )

        generation_config: dict[str, Any] = {
            "temperature": req.temperature,
            "maxOutputTokens": req.max_tokens,
        }
        if req.top_p is not None:
            generation_config["topP"] = req.top_p
        if req.stop:
            generation_config["stopSequences"] = list(req.stop)
        if req.schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = req.schema

        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": req.prompt}]}],
            "generationConfig": generation_config,
        }
        if req.system:
            body["systemInstruction"] = {"parts": [{"text": req.system}]}

        url = f"{API_BASE}/{req.model_id}:generateContent"
        headers = {"content-type": "application/json", "x-goog-api-key": self.api_key}

        data, latency_ms = post_json(url, headers, body, req.timeout_s, self._transport, "google")
        text, finish_reason = _extract(data)
        usage = data.get("usageMetadata") if isinstance(data.get("usageMetadata"), dict) else {}
        candidate_tokens = usage.get("candidatesTokenCount")
        thought_tokens = int(usage.get("thoughtsTokenCount") or 0)
        # Gemini reports thoughts separately from candidate tokens but bills both
        # at the output rate. Response.output_tokens is the billed output count;
        # reasoning_tokens remains available as the inspectable subset.
        billed_output_tokens = (
            int(candidate_tokens) + thought_tokens
            if isinstance(candidate_tokens, (int, float)) and not isinstance(candidate_tokens, bool)
            else None
        )

        return Response(
            text=text,
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=billed_output_tokens,
            cached_input_tokens=int(usage.get("cachedContentTokenCount") or 0),
            reasoning_tokens=thought_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            schema_mode=self.schema_mode_for(req.schema),
        )


def _extract(data: dict[str, Any]) -> tuple[str, str]:
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise AdapterError("google response has no candidates", retryable=False)
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise AdapterError("google response has a malformed candidate", retryable=False)
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or not parts:
        raise AdapterError("google response has no content parts", retryable=False)
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
    if not texts:
        raise AdapterError("google response has no text part", retryable=False)
    finish_reason = str(candidate.get("finishReason") or "STOP").lower()
    return "".join(texts), finish_reason
