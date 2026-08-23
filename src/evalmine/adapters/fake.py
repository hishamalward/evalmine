"""A deterministic adapter that contacts nothing. Spec: docs/spec.md S10.

Everything the test suite asserts on runs through here, and ``--fake`` exposes
it to anyone who wants to see what a report looks like before spending money.

Determinism: every value - the text, the token counts, the latency, and which
branch of a schema is taken - is derived from a SHA-256 of the request. The same
request always produces the same answer, on any machine, in any order.

Failures are injectable rather than random, because a test that reproduces a 429
only sometimes is not a test.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from ..suite import canonical_bytes
from .base import AdapterError, Request, Response

#: Modes a :class:`FakeFailure` can inject.
RETRYABLE_MODES = ("rate_limit", "server_error", "timeout")
FAILURE_MODES = RETRYABLE_MODES + ("auth_error", "malformed_json", "schema_violation", "empty")

_WORDS = (
    "cache", "rubric", "latency", "token", "schema", "judge", "baseline", "candidate",
    "receipt", "changelog", "ticket", "query", "answer", "column", "budget", "swap",
)


@dataclass
class FakeFailure:
    """One injected failure, optionally scoped and optionally exhaustible.

    ``times=None`` means "always"; ``times=2`` means the next two matching calls
    fail and the third succeeds, which is how the retry policy gets tested.
    """

    mode: str
    model: str | None = None
    prompt_contains: str | None = None
    times: int | None = None
    used: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.mode not in FAILURE_MODES:
            raise ValueError(f"unknown failure mode {self.mode!r}; expected one of {FAILURE_MODES}")

    def matches(self, req: Request) -> bool:
        if self.times is not None and self.used >= self.times:
            return False
        if self.model is not None and self.model not in (req.model_id, f"fake/{req.model_id}"):
            return False
        if self.prompt_contains is not None and self.prompt_contains not in req.prompt:
            return False
        return True


def _digest(req: Request) -> str:
    payload = {
        "model_id": req.model_id,
        "prompt": req.prompt,
        "system": req.system,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "stop": list(req.stop) if req.stop else None,
        "schema": req.schema,
    }
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _int_at(digest: str, offset: int, width: int = 8) -> int:
    start = (offset * width) % (len(digest) - width)
    return int(digest[start : start + width], 16)


def instance_for_schema(schema: dict[str, Any] | None, digest: str, depth: int = 0) -> Any:
    """A deterministic instance that satisfies ``schema``.

    Small on purpose: it covers ``enum``, ``const``, objects with ``required``,
    arrays, and the union types the example suite uses. It is a fake, not a
    fuzzer.
    """
    if not isinstance(schema, dict):
        return f"fake-{digest[:6]}"
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        options = schema["enum"]
        return options[_int_at(digest, depth + 1) % len(options)]

    types = schema.get("type")
    if isinstance(types, list):
        types = types[_int_at(digest, depth + 2) % len(types)]
    if types is None:
        types = "object" if "properties" in schema else "string"

    if types == "object":
        properties: dict[str, Any] = schema.get("properties") or {}
        required = schema.get("required")
        names = list(required) if required else list(properties)
        out: dict[str, Any] = {}
        for i, name in enumerate(names):
            out[name] = instance_for_schema(properties.get(name), digest, depth + 3 + i)
        return out
    if types == "array":
        min_items = int(schema.get("minItems", 1))
        count = max(min_items, 1 + _int_at(digest, depth + 4) % 2)
        items = schema.get("items")
        return [instance_for_schema(items, digest, depth + 5 + i) for i in range(count)]
    if types == "boolean":
        return bool(_int_at(digest, depth + 6) % 2)
    if types == "integer":
        return _int_at(digest, depth + 7) % 100
    if types == "number":
        return round((_int_at(digest, depth + 8) % 100_000) / 100.0, 2)
    if types == "null":
        return None
    return f"fake-{digest[depth % 32 : depth % 32 + 6]}"


def _prose(digest: str, model_id: str) -> str:
    count = 12 + _int_at(digest, 1) % 20
    words = [_WORDS[_int_at(digest, i + 2) % len(_WORDS)] for i in range(count)]
    return f"[fake:{model_id}] " + " ".join(words) + "."


class FakeAdapter:
    """Deterministic, offline, injectable. ``name`` is always ``"fake"``."""

    name = "fake"
    version = 1

    def __init__(
        self,
        failures: list[FakeFailure] | None = None,
        schema_mode: str = "prompted",
        drop_usage: bool = False,
    ) -> None:
        self.failures = list(failures or [])
        self.schema_mode = schema_mode
        #: Simulates a provider that returns no token counts, so the caller has
        #: to carry a null cost rather than a convenient zero.
        self.drop_usage = drop_usage
        self.calls = 0
        self.seen: list[Request] = []

    # -- injection ---------------------------------------------------------

    def _next_failure(self, req: Request) -> FakeFailure | None:
        for failure in self.failures:
            if failure.matches(req):
                failure.used += 1
                return failure
        return None

    @staticmethod
    def _raise(mode: str) -> None:
        if mode == "rate_limit":
            raise AdapterError("fake: rate limited", retryable=True, status=429)
        if mode == "server_error":
            raise AdapterError("fake: upstream error", retryable=True, status=503)
        if mode == "timeout":
            raise AdapterError("fake: request timed out", retryable=True, status=None)
        if mode == "auth_error":
            raise AdapterError("fake: invalid api key", retryable=False, status=401)

    # -- the interface -----------------------------------------------------

    def complete(self, req: Request) -> Response:
        self.calls += 1
        self.seen.append(req)
        digest = _digest(req)

        failure = self._next_failure(req)
        mode = failure.mode if failure else None
        if mode in ("rate_limit", "server_error", "timeout", "auth_error"):
            self._raise(mode)

        if mode == "malformed_json":
            text = "Sure! Here is the JSON you asked for: {merchant: 'The Second Cup', total:"
        elif mode == "schema_violation":
            text = json.dumps({"not_a_field_in_your_schema": digest[:8]})
        elif mode == "empty":
            text = ""
        elif req.schema is not None:
            text = json.dumps(instance_for_schema(req.schema, digest), ensure_ascii=False)
        else:
            text = _prose(digest, req.model_id)

        input_tokens = None if self.drop_usage else _estimate(req.prompt) + _estimate(req.system)
        output_tokens = None if self.drop_usage else max(1, _estimate(text))
        return Response(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=0,
            reasoning_tokens=0,
            latency_ms=200 + _int_at(digest, 3) % 1800,
            finish_reason="stop",
            schema_mode=self.schema_mode if req.schema is not None else "prompted",
        )


def _estimate(text: str | None) -> int:
    return 0 if not text else max(1, len(text) // 4)
