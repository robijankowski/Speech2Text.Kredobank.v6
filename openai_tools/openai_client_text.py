# openai_client_text.py (router)

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from core.config import settings

from openai.types.chat import ChatCompletion

from openai_tools.openai_client_text_azure import (
    chat_completion_azure,
    async_chat_completion_azure,
    chat_completion_with_format_azure,
    async_chat_completion_with_format_azure,
)
from openai_tools.openai_client_text_native import (
    chat_completion_native,
    async_chat_completion_native,
    chat_completion_with_format_native,
    async_chat_completion_with_format_native,
)



def chat_completion(
    *,
    messages: Sequence[dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    **kwargs: Any,
) -> ChatCompletion:
    if settings.USE_AZURE_OPENAI == "Y":
        return chat_completion_azure(
            model=model,
            temperature=temperature,
            messages=list(messages),
            timeout=timeout,
            **kwargs,
        )
    return chat_completion_native(
        model=model,
        temperature=temperature,
        messages=list(messages),
        timeout=timeout,
        **kwargs,
    )


async def async_chat_completion(
    *,
    messages: Sequence[dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    **kwargs: Any,
) -> ChatCompletion:
    if settings.USE_AZURE_OPENAI == "Y":
        return await async_chat_completion_azure(
            model=model,
            temperature=temperature,
            messages=list(messages),
            timeout=timeout,
            **kwargs,
        )
    return await async_chat_completion_native(
        model=model,
        temperature=temperature,
        messages=list(messages),
        timeout=timeout,
        **kwargs,
    )


def chat_completion_with_format(
    *,
    messages: Sequence[dict[str, Any]],
    format_schema: Union[str, dict[str, Any]],
    schema_name: str = "response_schema",
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    **kwargs: Any,
) -> ChatCompletion:
    if settings.USE_AZURE_OPENAI == "Y":
        return chat_completion_with_format_azure(
            messages=messages,
            format_schema=format_schema,
            schema_name=schema_name,
            model=model,
            temperature=temperature,
            timeout=timeout,
            **kwargs,
        )
    return chat_completion_with_format_native(
        messages=messages,
        format_schema=format_schema,
        schema_name=schema_name,
        model=model,
        temperature=temperature,
        timeout=timeout,
        **kwargs,
    )


async def async_chat_completion_with_format(
    *,
    messages: Sequence[dict[str, Any]],
    format_schema: str | dict[str, Any],
    schema_name: str = "response_schema",
    model: str | None = None,
    temperature: float = 0,
    timeout: float = 120,
    **kwargs: Any,
) -> ChatCompletion:
    if settings.USE_AZURE_OPENAI == "Y":
        return await async_chat_completion_with_format_azure(
            messages=messages,
            format_schema=format_schema,
            schema_name=schema_name,
            model=model,
            temperature=temperature,
            timeout=timeout,
            **kwargs,
        )
    return await async_chat_completion_with_format_native(
        messages=messages,
        format_schema=format_schema,
        schema_name=schema_name,
        model=model,
        temperature=temperature,
        timeout=timeout,
        **kwargs,
    )

