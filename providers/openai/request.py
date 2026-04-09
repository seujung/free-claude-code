"""Request builder for generic OpenAI-compatible provider."""

from typing import Any

from loguru import logger

from providers.common.message_converter import build_base_request_body

OPENAI_COMPAT_DEFAULT_MAX_TOKENS = 81920


def build_request_body(request_data: Any) -> dict:
    """Build OpenAI-format request body from Anthropic request."""
    logger.debug(
        "OPENAI_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    body = build_base_request_body(
        request_data,
        default_max_tokens=OPENAI_COMPAT_DEFAULT_MAX_TOKENS,
    )

    extra_body: dict[str, Any] = {}
    request_extra = getattr(request_data, "extra_body", None)
    if request_extra:
        extra_body.update(request_extra)
    if extra_body:
        body["extra_body"] = extra_body

    logger.debug(
        "OPENAI_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
