"""The fake adapter and the shared retry policy. Spec S10.

No test in this file touches the network, and none of them can: the fake adapter
has no HTTP client in it.
"""

from __future__ import annotations

import jsonschema
import pytest

from evalmine.adapters import (
    AdapterError,
    FakeAdapter,
    FakeFailure,
    Request,
    UnsupportedProviderError,
    build_adapter,
    call_with_retries,
    split_model,
)
from evalmine.adapters.fake import instance_for_schema
from evalmine.metrics import schema_verdict

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["merchant", "total", "line_items"],
    "properties": {
        "merchant": {"type": ["string", "null"]},
        "total": {"type": ["number", "null"]},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["description", "amount"],
                "properties": {
                    "description": {"type": "string"},
                    "amount": {"type": "number"},
                },
            },
        },
    },
}


def req(**overrides) -> Request:
    return Request(**{"model_id": "a", "prompt": "Summarise the receipt.", **overrides})


def test_split_model():
    assert split_model("anthropic/claude-sonnet-4-6") == ("anthropic", "claude-sonnet-4-6")
    assert split_model("openrouter/qwen/qwen3.7-plus") == (
        "openrouter",
        "qwen/qwen3.7-plus",
    )
    with pytest.raises(UnsupportedProviderError):
        split_model("no-slash")


def test_the_same_request_always_gives_the_same_answer():
    first = FakeAdapter().complete(req())
    second = FakeAdapter().complete(req())
    assert first == second


def test_different_models_give_different_answers():
    a = FakeAdapter().complete(req(model_id="a"))
    b = FakeAdapter().complete(req(model_id="b"))
    assert a.text != b.text


def test_usage_and_latency_are_populated_and_deterministic():
    response = FakeAdapter().complete(req())
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert 200 <= response.latency_ms < 2000
    assert response.finish_reason == "stop"


def test_drop_usage_reports_no_token_counts():
    response = FakeAdapter(drop_usage=True).complete(req())
    assert response.input_tokens is None
    assert response.output_tokens is None


def test_a_schema_request_produces_schema_valid_json():
    response = FakeAdapter().complete(req(schema=SCHEMA))
    assert schema_verdict(response.text, SCHEMA).status == "pass"


def test_generated_instances_validate_for_a_range_of_schemas():
    schemas = [
        {"type": "string"},
        {"type": "integer"},
        {"type": ["string", "null"]},
        {"type": "string", "enum": ["low", "high"]},
        {"type": "array", "minItems": 2, "items": {"type": "boolean"}},
        SCHEMA,
    ]
    for i, schema in enumerate(schemas):
        instance = instance_for_schema(schema, "ab" * 32, depth=i)
        jsonschema.Draft202012Validator(schema).validate(instance)


@pytest.mark.parametrize("mode", ["rate_limit", "server_error", "timeout"])
def test_retryable_failures_are_retryable(mode):
    adapter = FakeAdapter(failures=[FakeFailure(mode)])
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is True


def test_auth_failure_is_not_retryable():
    adapter = FakeAdapter(failures=[FakeFailure("auth_error")])
    with pytest.raises(AdapterError) as exc:
        adapter.complete(req())
    assert exc.value.retryable is False
    assert exc.value.status == 401


def test_malformed_json_is_a_parse_fail():
    adapter = FakeAdapter(failures=[FakeFailure("malformed_json")])
    response = adapter.complete(req(schema=SCHEMA))
    assert schema_verdict(response.text, SCHEMA).status == "parse_fail"


def test_schema_violating_output_is_a_schema_fail():
    adapter = FakeAdapter(failures=[FakeFailure("schema_violation")])
    response = adapter.complete(req(schema=SCHEMA))
    assert schema_verdict(response.text, SCHEMA).status == "schema_fail"


def test_failures_can_be_scoped_to_a_model_and_a_prompt():
    adapter = FakeAdapter(
        failures=[FakeFailure("rate_limit", model="b", prompt_contains="receipt")]
    )
    adapter.complete(req(model_id="a"))  # different model: fine
    adapter.complete(req(model_id="b", prompt="something else"))  # different prompt: fine
    with pytest.raises(AdapterError):
        adapter.complete(req(model_id="b", prompt="Summarise the receipt."))


def test_a_failure_can_be_exhausted_so_the_retry_succeeds():
    adapter = FakeAdapter(failures=[FakeFailure("rate_limit", times=2)])
    delays: list[float] = []
    response = call_with_retries(adapter, req(), sleep=delays.append)
    assert response.text
    assert adapter.calls == 3
    assert len(delays) == 2
    assert 1.0 <= delays[0] < 2.0 and 2.0 <= delays[1] < 3.0


def test_retries_give_up_after_two_and_re_raise():
    adapter = FakeAdapter(failures=[FakeFailure("server_error")])
    with pytest.raises(AdapterError):
        call_with_retries(adapter, req(), sleep=lambda _: None)
    assert adapter.calls == 3


def test_a_non_retryable_error_is_not_retried():
    adapter = FakeAdapter(failures=[FakeFailure("auth_error")])
    with pytest.raises(AdapterError):
        call_with_retries(adapter, req(), sleep=lambda _: None)
    assert adapter.calls == 1


def test_unknown_failure_mode_is_rejected_at_construction():
    with pytest.raises(ValueError):
        FakeFailure("explode")


def test_build_adapter_routes_everything_to_the_fake_when_asked():
    assert isinstance(build_adapter("anthropic", fake=True), FakeAdapter)
    assert isinstance(build_adapter("fake"), FakeAdapter)


def test_build_adapter_constructs_the_real_adapters():
    from evalmine.adapters import (
        AnthropicAdapter,
        GoogleAdapter,
        OpenAIAdapter,
        OpenRouterAdapter,
    )

    assert isinstance(build_adapter("anthropic"), AnthropicAdapter)
    assert isinstance(build_adapter("openai"), OpenAIAdapter)
    assert isinstance(build_adapter("google"), GoogleAdapter)
    assert isinstance(build_adapter("openrouter"), OpenRouterAdapter)


def test_build_adapter_rejects_an_unknown_provider():
    with pytest.raises(UnsupportedProviderError):
        build_adapter("mystery")
