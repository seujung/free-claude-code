"""Tests for generic OpenAI-compatible provider."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.openai import GenericOpenAIProvider
from providers.openai.request import (
    OPENAI_COMPAT_DEFAULT_MAX_TOKENS,
    build_request_body,
)


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "gpt-4o"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.7
        self.top_p = 0.9
        self.system = "You are a helpful assistant."
        self.stop_sequences = None
        self.tools = []
        self.tool_choice = None
        self.extra_body = {}
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def openai_config():
    return ProviderConfig(
        api_key="sk-test-key",
        base_url="https://api.openai.com/v1",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
        instance = mock.get_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.fixture
def openai_provider(openai_config):
    return GenericOpenAIProvider(openai_config)


def test_init(openai_config):
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = GenericOpenAIProvider(openai_config)
        assert provider._api_key == "sk-test-key"
        assert provider._base_url == "https://api.openai.com/v1"
        mock_openai.assert_called_once()


def test_init_default_base_url():
    config = ProviderConfig(api_key="sk-test-key")
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = GenericOpenAIProvider(config)
        assert provider._base_url == "https://api.openai.com/v1"


def test_init_custom_base_url():
    config = ProviderConfig(
        api_key="gsk_test", base_url="https://api.groq.com/openai/v1"
    )
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = GenericOpenAIProvider(config)
        assert provider._base_url == "https://api.groq.com/openai/v1"


def test_build_request_body_basic():
    req = MockRequest()
    body = build_request_body(req)
    assert body["model"] == "gpt-4o"
    assert body["temperature"] == 0.7
    assert len(body["messages"]) == 2  # system + user
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"


def test_build_request_body_default_max_tokens():
    req = MockRequest(max_tokens=None)
    body = build_request_body(req)
    assert body["max_tokens"] == OPENAI_COMPAT_DEFAULT_MAX_TOKENS


def test_build_request_body_extra_body_forwarded():
    req = MockRequest(extra_body={"logprobs": True, "top_logprobs": 5})
    body = build_request_body(req)
    assert body["extra_body"]["logprobs"] is True
    assert body["extra_body"]["top_logprobs"] == 5


def test_build_request_body_no_extra_body_when_empty():
    req = MockRequest(extra_body={})
    body = build_request_body(req)
    assert "extra_body" not in body


def test_init_configurable_timeouts():
    config = ProviderConfig(
        api_key="sk-test",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        GenericOpenAIProvider(config)
        call_kwargs = mock_openai.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


@pytest.mark.asyncio
async def test_stream_response_text(openai_provider):
    req = MockRequest()

    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [
        MagicMock(
            delta=MagicMock(content="Hello", reasoning_content=None, tool_calls=None),
            finish_reason=None,
        )
    ]
    mock_chunk1.usage = None

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [
        MagicMock(
            delta=MagicMock(content=" World", reasoning_content=None, tool_calls=None),
            finish_reason="stop",
        )
    ]
    mock_chunk2.usage = MagicMock(completion_tokens=10)

    async def mock_stream():
        yield mock_chunk1
        yield mock_chunk2

    with patch.object(
        openai_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()
        events = [e async for e in openai_provider.stream_response(req)]

        assert any("event: message_start" in e for e in events)

        text_content = ""
        for e in events:
            if "event: content_block_delta" in e and '"text_delta"' in e:
                for line in e.splitlines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "delta" in data and "text" in data["delta"]:
                            text_content += data["delta"]["text"]

        assert "Hello World" in text_content


@pytest.mark.asyncio
async def test_stream_response_error(openai_provider):
    req = MockRequest()

    async def mock_stream():
        raise RuntimeError("Connection refused")
        yield  # make it a generator

    with patch.object(
        openai_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()
        events = [e async for e in openai_provider.stream_response(req)]
        assert any("Connection refused" in e for e in events)
        assert any("message_stop" in e for e in events)


@pytest.mark.asyncio
async def test_stream_response_completes_with_message_stop(openai_provider):
    req = MockRequest()

    async def mock_stream():
        yield MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(
                        content="ok", reasoning_content=None, tool_calls=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=MagicMock(completion_tokens=1),
        )

    with patch.object(
        openai_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()
        events = [e async for e in openai_provider.stream_response(req)]
        assert any("message_delta" in e for e in events)
        assert any("message_stop" in e for e in events)
