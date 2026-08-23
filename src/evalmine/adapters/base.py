"""The adapter interface, and the retry policy every adapter shares.

Spec: docs/spec.md S10. One Protocol, four implementations, no framework and no
provider SDK - the stated point of this layer is that it is small enough to read
in one sitting.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

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
    timeout_s: int = 60


@dataclass(frozen=True)
class Response:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
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
