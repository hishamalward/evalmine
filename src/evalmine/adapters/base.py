"""The adapter interface, and the retry policy every adapter shares.

Spec: docs/spec.md S10. One Protocol, five implementations, no framework and no
provider SDK - the stated point of this layer is that it is small enough to read
in one sitting.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

#: Retries, spec S10: timeouts, 429 and 5xx only. A non-retryable error (401,
#: 400) fails the run immediately - a missing key should stop you in the first
#: two seconds, not after forty calls.
MAX_RETRIES = 2


class AdapterError(Exception):
    def __init__(self, message: str, retryable: bool = False, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class UnsupportedProviderError(Exception):
    """A model string whose provider prefix has no adapter in this build."""


@dataclass(frozen=True)
class Request:
    model_id: str
    prompt: str
    system: str | None = None
    max_tokens: int = 700
    temperature: float = 0.0
    top_p: float | None = None
    stop: tuple[str, ...] | None = None
    schema: dict[str, Any] | None = None
    #: Provider-specific routing controls that affect which backend serves the
    #: request. They are part of the suite contract and cache key; adapters
    #: that do not support routing leave this unset.
    provider_options: dict[str, Any] | None = None
    timeout_s: int = 60


@dataclass(frozen=True)
class Response:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    #: A metered amount returned by the provider for this exact request. When
    #: present it is stronger cost evidence than a token-count estimate.
    reported_cost_usd: float | None = None
    latency_ms: int = 0
    finish_reason: str = "stop"
    #: "native" - the provider enforced the schema for us; "prompted" - the
    #: schema was appended to the prompt as an instruction. Comparing the two
    #: is comparing two different things, so the mode travels with the answer.
    schema_mode: str = "prompted"


@runtime_checkable
class Adapter(Protocol):
    name: str
    version: int

    def complete(self, req: Request) -> Response: ...

    def schema_mode_for(self, schema: dict[str, Any] | None) -> str:
        """Which mode this adapter will use for a request carrying ``schema``.

        Declared up front rather than reported afterwards because the mode is
        part of the cache key (S6.5), and the key has to exist before the call.
        """
        ...


def status_is_retryable(status: int) -> bool:
    """429 and 5xx are worth retrying; everything else fails fast (spec S10)."""
    return status == 429 or status >= 500


def post_json(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout_s: int,
    transport: httpx.BaseTransport | None,
    provider: str,
) -> tuple[dict[str, Any], int]:
    """POST ``body`` as JSON, mapping every transport/HTTP failure onto ``AdapterError``.

    Shared by all four real adapters - the one bit of HTTP boilerplate common
    to a hand-written POST against a documented JSON endpoint. ``transport``
    exists only so tests can substitute ``httpx.MockTransport``; production
    code never passes it. Returns the parsed body and the wall-clock latency
    of the call in milliseconds.
    """
    t0 = time.perf_counter()
    try:
        with httpx.Client(transport=transport, timeout=timeout_s) as client:
            resp = client.post(url, headers=headers, json=body)
    except httpx.TimeoutException as exc:
        raise AdapterError(f"{provider} request timed out: {exc}", retryable=True) from exc
    except httpx.TransportError as exc:
        raise AdapterError(f"{provider} transport error: {exc}", retryable=True) from exc
    latency_ms = int((time.perf_counter() - t0) * 1000)

    if resp.status_code >= 400:
        raise AdapterError(
            f"{provider} error {resp.status_code}: {resp.text[:300]}",
            retryable=status_is_retryable(resp.status_code),
            status=resp.status_code,
        )
    try:
        data = resp.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise AdapterError(
            f"{provider} returned a body that is not valid JSON: {exc}", retryable=False
        ) from exc
    if not isinstance(data, dict):
        raise AdapterError(f"{provider} returned a non-object JSON body", retryable=False)
    return data, latency_ms


def split_model(model: str) -> tuple[str, str]:
    """``anthropic/claude-sonnet-4-6`` -> ``("anthropic", "claude-sonnet-4-6")``."""
    provider, sep, model_id = model.partition("/")
    if not sep or not provider or not model_id:
        raise UnsupportedProviderError(
            f"model string {model!r} must be 'provider/model-id', e.g. "
            "'anthropic/claude-sonnet-4-6'"
        )
    return provider, model_id


def call_with_retries(
    adapter: Adapter,
    req: Request,
    max_retries: int = MAX_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> Response:
    """Call the adapter, retrying only what is worth retrying (spec S10)."""
    rng = rng or random.Random()
    attempt = 0
    while True:
        try:
            return adapter.complete(req)
        except AdapterError as exc:
            if not exc.retryable or attempt >= max_retries:
                raise
            sleep(2**attempt + rng.uniform(0, 1))
            attempt += 1
