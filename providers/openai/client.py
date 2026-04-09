"""Generic OpenAI-compatible provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class GenericOpenAIProvider(OpenAICompatibleProvider):
    """Generic provider for any OpenAI API-compatible endpoint.

    Works with OpenAI, Groq, Together AI, Ollama, and any other service
    that implements the OpenAI chat completions API.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="OPENAI",
            base_url=config.base_url or OPENAI_DEFAULT_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        return build_request_body(request)
