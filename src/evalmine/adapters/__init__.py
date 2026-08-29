"""Adapter registry.

``--fake`` routes every model string to the fake adapter regardless of prefix
(spec S4); otherwise the provider prefix selects one of the four real HTTP
adapters, each reading its key from the environment on construction.
"""

from __future__ import annotations

from .anthropic import AnthropicAdapter
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
from .google import GoogleAdapter
from .openai import OpenAIAdapter
from .openrouter import OpenRouterAdapter

REAL_PROVIDERS = ("anthropic", "openai", "google", "openrouter")

_REAL_ADAPTERS: dict[str, type] = {
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
    "google": GoogleAdapter,
    "openrouter": OpenRouterAdapter,
}

__all__ = [
    "Adapter",
    "AdapterError",
    "AnthropicAdapter",
    "FakeAdapter",
    "FakeFailure",
    "GoogleAdapter",
    "OpenAIAdapter",
    "OpenRouterAdapter",
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
    try:
        return _REAL_ADAPTERS[provider]()
    except KeyError:
        raise UnsupportedProviderError(
            f"no adapter for provider {provider!r}; expected one of "
            f"{REAL_PROVIDERS + ('fake',)}"
        ) from None
