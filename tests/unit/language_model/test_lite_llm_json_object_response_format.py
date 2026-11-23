# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Regression tests for response_format_json_object reaching LiteLLM."""

import asyncio
from typing import Any

import pytest
from graphrag_cache import create_cache_key
from graphrag_llm.completion.lite_llm_completion import LiteLLMCompletion
from graphrag_llm.config import ModelConfig
from graphrag_llm.metrics.noop_metrics_store import NoopMetricsStore
from graphrag_llm.tokenizer.tokenizer import Tokenizer
from graphrag_llm.types import LLMCompletionResponse
from litellm import ModelResponse
from pydantic import BaseModel


class _StubTokenizer(Tokenizer):
    """Tokenizer that is never invoked by these tests."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def encode(self, text: str) -> list[int]:
        return []

    def decode(self, tokens: list[int]) -> str:
        return ""


class _Answer(BaseModel):
    """Explicit response schema used to confirm caller formats win."""

    ok: bool


_MESSAGES = [{"role": "user", "content": "Answer in JSON."}]
_JSON_OBJECT = {"type": "json_object"}


def _model_response() -> ModelResponse:
    return ModelResponse(
        id="completion-id",
        choices=[
            {
                "message": {"role": "assistant", "content": '{"ok": true}'},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        model="openai/gpt-4o-mini",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def _completion() -> LiteLLMCompletion:
    return LiteLLMCompletion(
        model_id="openai/gpt-4o-mini",
        model_config=ModelConfig(
            model_provider="openai",
            model="gpt-4o-mini",
            api_key="not-used",
        ),
        tokenizer=_StubTokenizer(),
        metrics_store=NoopMetricsStore(),
        cache_key_creator=create_cache_key,
    )


@pytest.fixture
def captured_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record the kwargs LiteLLM would send, without a network call."""
    captured: dict[str, Any] = {}

    def _fake_completion(**kwargs: Any) -> ModelResponse:
        captured.clear()
        captured.update(kwargs)
        return _model_response()

    async def _fake_acompletion(**kwargs: Any) -> ModelResponse:
        captured.clear()
        captured.update(kwargs)
        await asyncio.sleep(0)
        return _model_response()

    monkeypatch.setattr(
        "graphrag_llm.completion.lite_llm_completion.litellm.completion",
        _fake_completion,
    )
    monkeypatch.setattr(
        "graphrag_llm.completion.lite_llm_completion.litellm.acompletion",
        _fake_acompletion,
    )
    return captured


@pytest.mark.parametrize("use_async", [False, True])
async def test_json_object_flag_sets_response_format(
    captured_kwargs: dict[str, Any],
    use_async: bool,
) -> None:
    """response_format_json_object=True must reach LiteLLM as json_object mode.

    The public completion methods always forward response_format, using None
    when the caller omitted it. JSON mode has to treat that None as unset.
    """
    completion = _completion()
    request = {
        "messages": _MESSAGES,
        "response_format_json_object": True,
    }

    if use_async:
        response = await completion.completion_async(**request)
    else:
        response = completion.completion(**request)

    assert isinstance(response, LLMCompletionResponse)
    assert response.content == '{"ok": true}'
    assert captured_kwargs["response_format"] == _JSON_OBJECT
    assert "response_format_json_object" not in captured_kwargs


@pytest.mark.parametrize("use_async", [False, True])
async def test_explicit_response_format_is_not_replaced(
    captured_kwargs: dict[str, Any],
    use_async: bool,
) -> None:
    """An explicit response_format wins over the JSON object flag."""
    completion = _completion()
    request = {
        "messages": _MESSAGES,
        "response_format": _Answer,
        "response_format_json_object": True,
    }

    if use_async:
        response = await completion.completion_async(**request)
    else:
        response = completion.completion(**request)

    assert isinstance(response, LLMCompletionResponse)
    assert response.formatted_response == _Answer(ok=True)
    assert captured_kwargs["response_format"] is _Answer


@pytest.mark.parametrize("use_async", [False, True])
async def test_json_object_flag_off_leaves_response_format_unset(
    captured_kwargs: dict[str, Any],
    use_async: bool,
) -> None:
    """Omitting the flag must not invent a response_format."""
    completion = _completion()

    if use_async:
        await completion.completion_async(messages=_MESSAGES)
    else:
        completion.completion(messages=_MESSAGES)

    assert captured_kwargs["response_format"] is None
