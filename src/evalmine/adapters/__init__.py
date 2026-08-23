"""Adapter registry.

The three real HTTP adapters are not in this build; the fake one is, and
``--fake`` routes every model string to it regardless of prefix (spec S4).
"""

from __future__ import annotations

from .base import (
    Adapter,
    AdapterError,
    Request,
    Response,
    UnsupportedProviderError,
    call_with_retries,
    split_model,
)
from .fake import FakeAdapter, FakeFailure

REAL_PROVIDERS = ("anthropic", "openai", "google")

__all__ = [
    "Adapter",
    "AdapterError",
    "FakeAdapter",
    "FakeFailure",
    "REAL_PROVIDERS",
    "Request",
    "Response",
    "UnsupportedProviderError",
    "build_adapter",
    "call_with_retries",
    "split_model",
]


def build_adapter(provider: str, fake: bool = False) -> Adapter:
    if fake or provider == "fake":
        return FakeAdapter()
    if provider in REAL_PROVIDERS:
        raise UnsupportedProviderError(
            f"the {provider} adapter is not built in this version of evalmine. "
            "Run with --fake to exercise the harness against the deterministic fake "
            "adapter, or wait for the round that adds the provider adapters."
        )
    raise UnsupportedProviderError(
        f"no adapter for provider {provider!r}; expected one of "
        f"{REAL_PROVIDERS + ('fake',)}"
    )
