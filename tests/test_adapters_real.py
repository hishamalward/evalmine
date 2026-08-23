"""The three real HTTP adapters. Spec S10.

No test in this file makes a real network call: every request is served by
``httpx.MockTransport``, so there is nothing here that needs a key, contacts
a provider, or costs anything.
"""

from __future__ import annotations

import json

import httpx
import pytest

from evalmine.adapters.anthropic import (
    AnthropicAdapter,
    sampling_params_supported,
    thinking_defaults_on,
)
from evalmine.adapters.base import AdapterError, Request
from evalmine.adapters.google import GoogleAdapter
from evalmine.adapters.openai import OpenAIAdapter

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["winner", "reason"],
    "properties": {
        "winner": {"type": "string", "enum": ["1", "2", "tie"]},
        "reason": {"type": "string"},
    },
}


def req(**overrides) -> Request:
    return Request(**{"model_id": "m", "prompt": "hello", **overrides})


def transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def json_response(status: int, body: dict) -> httpx.Response:
    return httpx.Response(status, json=body)


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


ANTHROPIC_SUCCESS = {
    "content": [{"type": "text", "text": "hello there"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 12, "output_tokens": 5, "cache_read_input_tokens": 3},
}

ANTHROPIC_SCHEMA_SUCCESS = {
    "content": [
        {
            "type": "tool_use",
            "id": "t1",
            "name": "emit_result",
            "input": {"winner": "1", "reason": "ok"},
        }
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 20, "output_tokens": 8},
}


def test_anthropic_success_with_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "sk-test"
        assert request.headers["anthropic-version"]
        body = json.loads(request.content)
        assert body["model"] == "claude-haiku-4-5"
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        return json_response(200, ANTHROPIC_SUCCESS)

    adapter = AnthropicAdapter(api_key="sk-test", transport=transport(handler))
    resp = adapter.complete(req(model_id="claude-haiku-4-5"))
    assert resp.text == "hello there"
    assert resp.input_tokens == 12
    assert resp.output_tokens == 5
    assert resp.cached_input_tokens == 3
    assert resp.finish_reason == "end_turn"
    assert resp.schema_mode == "prompted"
    assert resp.latency_ms >= 0


def test_anthropic_schema_request_forces_the_tool_and_is_native():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tool_choice"] == {"type": "tool", "name": "emit_result"}
        assert body["tools"][0]["input_schema"] == SCHEMA
        return json_response(200, ANTHROPIC_SCHEMA_SUCCESS)

    adapter = AnthropicAdapter(api_key="sk-test", transport=transport(handler))
    resp = adapter.complete(req(schema=SCHEMA))
    assert json.loads(resp.text) == {"winner": "1", "reason": "ok"}
    assert resp.schema_mode == "native"
    assert adapter.schema_mode_for(SCHEMA) == "native"
    assert adapter.schema_mode_for(None) == "prompted"


@pytest.mark.parametrize(
    "model_id, supported",
    [
        ("claude-haiku-4-5", True),
        ("claude-sonnet-4-6", True),
        ("claude-opus-4-6", True),
        ("claude-3-5-sonnet-20241022", True),
        ("claude-opus-4-7", False),
        ("claude-opus-4-8", False),
        ("claude-opus-5", False),
        ("claude-sonnet-5", False),
        ("claude-fable-5", False),
        ("claude-mythos-5", False),
    ],
)
def test_anthropic_sampling_support_table(model_id, supported):
    assert sampling_params_supported(model_id) is supported


def test_anthropic_omits_sampling_params_where_the_api_rejects_them():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "temperature" not in body
        assert "top_p" not in body
        assert body["max_tokens"] == 900
        return json_response(200, ANTHROPIC_SUCCESS)

    adapter = AnthropicAdapter(api_key="sk-test", transport=transport(handler))
    resp = adapter.complete(
        req(model_id="claude-opus-5", temperature=0.0, top_p=0.9, max_tokens=900)
    )
    assert resp.text == "hello there"


def test_anthropic_keeps_sampling_params_on_models_that_accept_them():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["temperature"] == 0.2
        assert body["top_p"] == 0.9
        return json_response(200, ANTHROPIC_SUCCESS)

    adapter = AnthropicAdapter(api_key="sk-test", transport=transport(handler))
    adapter.complete(req(model_id="claude-haiku-4-5", temperature=0.2, top_p=0.9))


@pytest.mark.parametrize(
    "model_id, defaults_on",
    [
        ("claude-haiku-4-5", False),
        ("claude-sonnet-4-6", False),
        ("claude-opus-4-8", False),
        ("claude-opus-5", True),
        ("claude-sonnet-5", True),
        ("claude-fable-5", False),
        ("claude-mythos-5", False),
    ],
)
def test_anthropic_thinking_default_table(model_id, defaults_on):
    assert thinking_defaults_on(model_id) is defaults_on


def test_anthropic_disables_thinking_where_it_would_default_on():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen[body["model"]] = body.get("thinking")
        return json_response(200, ANTHROPIC_SUCCESS)

    adapter = AnthropicAdapter(api_key="sk-test", transport=transport(handler))
    for model_id in ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5", "claude-fable-5"):
        adapter.complete(req(model_id=model_id))
    assert seen["claude-opus-5"] == {"type": "disabled"}
    assert seen["claude-sonnet-5"] == {"type": "disabled"}
    assert seen["claude-haiku-4-5"] is None
    assert seen["claude-fable-5"] is None


def test_anthropic_budget_exhausted_before_text_is_an_empty_answer():
    body = {
        "content": [{"type": "thinking", "thinking": "", "signature": "x"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 10, "output_tokens": 900},
    }
    adapter = AnthropicAdapter(
        api_key="sk-test", transport=transport(lambda r: json_response(200, body))
    )
    resp = adapter.complete(req(model_id="claude-opus-5"))
    assert resp.text == ""
    assert resp.finish_reason == "max_tokens"
    assert resp.output_tokens == 900

    schema_resp = adapter.complete(req(model_id="claude-opus-5", schema=SCHEMA))
    assert schema_resp.text == ""
    assert schema_resp.finish_reason == "max_tokens"


def test_anthropic_no_text_on_a_normal_stop_is_still_an_error():
    body = {
        "content": [{"type": "thinking", "thinking": "", "signature": "x"}],
        "stop_reason": "end_turn",
    }
    adapter = AnthropicAdapter(
        api_key="sk-test", transport=transport(lambda r: json_response(200, body))
    )
    with pytest.raises(AdapterError):
        adapter.complete(req())


def test_anthropic_missing_usage_is_none_not_zero():
    body = {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn"}
    adapter = AnthropicAdapter(
        api_key="sk-test", transport=transport(lambda r: json_response(200, body))
    )
    resp = adapter.complete(req())
    assert resp.input_tokens is None
    assert resp.output_tokens is None
    assert resp.cached_input_tokens == 0


def test_anthropic_429_is_retryable():
    adapter = AnthropicAdapter(
        api_key="sk-test",
        transport=transport(lambda r: json_response(429, {"error": {"message": "slow down"}})),
    )
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is True
    assert exc.value.status == 429


def test_anthropic_5xx_is_retryable():
    adapter = AnthropicAdapter(
        api_key="sk-test",
        transport=transport(lambda r: json_response(503, {"error": {"message": "down"}})),
    )
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is True
    assert exc.value.status == 503


def test_anthropic_timeout_is_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    adapter = AnthropicAdapter(api_key="sk-test", transport=transport(handler))
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is True


def test_anthropic_auth_error_is_not_retryable():
    adapter = AnthropicAdapter(
        api_key="sk-test",
        transport=transport(lambda r: json_response(401, {"error": {"message": "bad key"}})),
    )
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is False
    assert exc.value.status == 401


def test_anthropic_missing_key_fails_fast_without_a_network_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return json_response(200, ANTHROPIC_SUCCESS)

    adapter = AnthropicAdapter(transport=transport(handler))
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is False
    assert "ANTHROPIC_API_KEY" in str(exc.value)
    assert calls == []


def test_anthropic_malformed_body_is_not_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json at all")

    adapter = AnthropicAdapter(api_key="sk-test", transport=transport(handler))
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is False


def test_anthropic_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    adapter = AnthropicAdapter()
    assert adapter.api_key == "sk-from-env"


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------


OPENAI_SUCCESS = {
    "choices": [
        {"message": {"role": "assistant", "content": "hello there"}, "finish_reason": "stop"}
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 6,
        "prompt_tokens_details": {"cached_tokens": 2},
        "completion_tokens_details": {"reasoning_tokens": 4},
    },
}


def test_openai_success_with_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer sk-test"
        body = json.loads(request.content)
        assert body["model"] == "gpt-x"
        assert body["messages"][-1] == {"role": "user", "content": "hello"}
        return json_response(200, OPENAI_SUCCESS)

    adapter = OpenAIAdapter(api_key="sk-test", transport=transport(handler))
    resp = adapter.complete(req(model_id="gpt-x", system="be terse"))
    assert resp.text == "hello there"
    assert resp.input_tokens == 10
    assert resp.output_tokens == 6
    assert resp.cached_input_tokens == 2
    assert resp.reasoning_tokens == 4
    assert resp.finish_reason == "stop"


def test_openai_schema_request_uses_json_schema_response_format():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["schema"] == SCHEMA
        return json_response(200, OPENAI_SUCCESS)

    adapter = OpenAIAdapter(api_key="sk-test", transport=transport(handler))
    resp = adapter.complete(req(schema=SCHEMA))
    assert resp.schema_mode == "native"
    assert adapter.schema_mode_for(None) == "prompted"


def test_openai_missing_usage_is_none_not_zero():
    body = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]
    }
    adapter = OpenAIAdapter(
        api_key="sk-test", transport=transport(lambda r: json_response(200, body))
    )
    resp = adapter.complete(req())
    assert resp.input_tokens is None
    assert resp.output_tokens is None
    assert resp.cached_input_tokens == 0
    assert resp.reasoning_tokens == 0


def test_openai_429_is_retryable():
    adapter = OpenAIAdapter(
        api_key="sk-test",
        transport=transport(lambda r: json_response(429, {"error": {"message": "slow down"}})),
    )
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is True


def test_openai_5xx_is_retryable():
    adapter = OpenAIAdapter(
        api_key="sk-test",
        transport=transport(lambda r: json_response(502, {"error": {"message": "down"}})),
    )
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is True


def test_openai_timeout_is_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    adapter = OpenAIAdapter(api_key="sk-test", transport=transport(handler))
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is True


def test_openai_auth_error_is_not_retryable():
    adapter = OpenAIAdapter(
        api_key="sk-test",
        transport=transport(lambda r: json_response(401, {"error": {"message": "bad key"}})),
    )
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is False
    assert exc.value.status == 401


def test_openai_missing_key_fails_fast_without_a_network_call(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = []
    adapter = OpenAIAdapter(
        transport=transport(lambda r: calls.append(r) or json_response(200, OPENAI_SUCCESS))
    )
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert "OPENAI_API_KEY" in str(exc.value)
    assert calls == []


def test_openai_malformed_body_is_not_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>nope</html>")

    adapter = OpenAIAdapter(api_key="sk-test", transport=transport(handler))
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is False


def test_openai_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    adapter = OpenAIAdapter()
    assert adapter.api_key == "sk-from-env"


# --------------------------------------------------------------------------
# Google
# --------------------------------------------------------------------------


GOOGLE_SUCCESS = {
    "candidates": [
        {
            "content": {"parts": [{"text": "hello there"}], "role": "model"},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 9,
        "candidatesTokenCount": 4,
        "cachedContentTokenCount": 1,
        "thoughtsTokenCount": 2,
    },
}


def test_google_success_with_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta/models/gemini-x:generateContent"
        assert request.headers["x-goog-api-key"] == "key-test"
        body = json.loads(request.content)
        assert body["contents"][0]["parts"][0]["text"] == "hello"
        return json_response(200, GOOGLE_SUCCESS)

    adapter = GoogleAdapter(api_key="key-test", transport=transport(handler))
    resp = adapter.complete(req(model_id="gemini-x"))
    assert resp.text == "hello there"
    assert resp.input_tokens == 9
    assert resp.output_tokens == 4
    assert resp.cached_input_tokens == 1
    assert resp.reasoning_tokens == 2
    assert resp.finish_reason == "stop"


def test_google_schema_request_sets_response_schema():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert body["generationConfig"]["responseSchema"] == SCHEMA
        return json_response(200, GOOGLE_SUCCESS)

    adapter = GoogleAdapter(api_key="key-test", transport=transport(handler))
    resp = adapter.complete(req(schema=SCHEMA))
    assert resp.schema_mode == "native"
    assert adapter.schema_mode_for(None) == "prompted"


def test_google_missing_usage_is_none_not_zero():
    body = {"candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}]}
    adapter = GoogleAdapter(
        api_key="key-test", transport=transport(lambda r: json_response(200, body))
    )
    resp = adapter.complete(req())
    assert resp.input_tokens is None
    assert resp.output_tokens is None
    assert resp.cached_input_tokens == 0
    assert resp.reasoning_tokens == 0


def test_google_429_is_retryable():
    adapter = GoogleAdapter(
        api_key="key-test",
        transport=transport(lambda r: json_response(429, {"error": {"message": "slow down"}})),
    )
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is True


def test_google_5xx_is_retryable():
    adapter = GoogleAdapter(
        api_key="key-test",
        transport=transport(lambda r: json_response(500, {"error": {"message": "down"}})),
    )
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is True


def test_google_timeout_is_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    adapter = GoogleAdapter(api_key="key-test", transport=transport(handler))
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is True


def test_google_auth_error_is_not_retryable():
    # Google commonly reports a bad key as 400 INVALID_ARGUMENT rather than
    # 401; either way it must not be retried.
    adapter = GoogleAdapter(
        api_key="key-test",
        transport=transport(
            lambda r: json_response(400, {"error": {"message": "API key not valid"}})
        ),
    )
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is False
    assert exc.value.status == 400


def test_google_missing_key_fails_fast_without_a_network_call(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    calls = []
    adapter = GoogleAdapter(
        transport=transport(lambda r: calls.append(r) or json_response(200, GOOGLE_SUCCESS))
    )
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert "GOOGLE_API_KEY" in str(exc.value)
    assert calls == []


def test_google_malformed_body_is_not_retryable():
    adapter = GoogleAdapter(
        api_key="key-test", transport=transport(lambda r: httpx.Response(200, text="oops"))
    )
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is False


def test_google_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "key-from-env")
    adapter = GoogleAdapter()
    assert adapter.api_key == "key-from-env"


def test_google_ignores_gemini_api_key_env_var(monkeypatch):
    """Spec S10: keys come from GOOGLE_API_KEY and nowhere else."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "should-not-be-read")
    adapter = GoogleAdapter()
    assert adapter.api_key is None
