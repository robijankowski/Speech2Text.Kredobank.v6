from __future__ import annotations

import json
from typing import Any, Optional, Sequence, Union

import httpx
from tenacity import retry, wait_random_exponential, stop_after_attempt

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

from core.config import settings


# --- clients (same style as openai_client_transcribe.py) ---
openai_chat_client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    timeout=httpx.Timeout(120.0, connect=10.0),
)

_async_openai_chat_client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    http_client=httpx.AsyncClient(
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        timeout=httpx.Timeout(120.0, connect=10.0),
    ),
)


def chat_completion_native(
    *,
    messages: Sequence[dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    **kwargs: Any,
) -> ChatCompletion:
    """
    Sync chat.completions wrapper for scenario_tools.
    Returns the full ChatCompletion so callers can read `.usage`, `.choices`, etc.
    """
    # Prefer a dedicated setting if you have it; otherwise fall back safely.

    if model.startswith("gpt-5"):
        # Special handling for gpt-5 models if needed
        return openai_chat_client.chat.completions.create(
            model=model,
            messages=list(messages),
            timeout=timeout,
            **kwargs,
        )
    return openai_chat_client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=list(messages),
        timeout=timeout,
        **kwargs,
    )




@retry(
    wait=wait_random_exponential(multiplier=1, min=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def async_chat_completion_native(
    *,
    messages: Sequence[dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    **kwargs: Any,
) -> ChatCompletion:
    """
    Async chat.completions wrapper for scenario_tools (with retries).
    Returns the full ChatCompletion so callers can read `.usage`, `.choices`, etc.
    """

    if model.startswith("gpt-5"):
        # Special handling for gpt-5 models if needed
        return await _async_openai_chat_client.chat.completions.create(
            model=model,
            messages=list(messages),
            timeout=timeout,
            **kwargs,
        )
    return await _async_openai_chat_client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=list(messages),
        timeout=timeout,
        **kwargs,
    )






def chat_completion_with_format_native(
    *,
    messages: Sequence[dict[str, Any]],
    format_schema: Union[str, dict[str, Any]],
    schema_name: str = "response_schema",
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    **kwargs: Any,
) -> ChatCompletion:
    """
    Sync chat.completions wrapper with structured output format.
    
    Args:
        messages: Chat messages
        format_schema: JSON schema as string or dict defining the required output format
        schema_name: Name for the schema (default: "response_schema")
        model: Model to use
        temperature: Sampling temperature
        timeout: Request timeout
        **kwargs: Additional parameters
        
    Returns:
        ChatCompletion with structured output
    """
    # Parse schema if it's a string
    if isinstance(format_schema, str):
        schema = json.loads(format_schema)
    else:
        schema = format_schema
    
    # Construct response_format for structured outputs
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": True,
        }
    }
    
    if model.startswith("gpt-5"):
        # Special handling for gpt-5 models if needed
        return openai_chat_client.chat.completions.create(
            model=model,
            messages=list(messages),
            response_format=response_format,
            timeout=timeout,
            **kwargs,
        )
    
    return openai_chat_client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=list(messages),
        response_format=response_format,
        timeout=timeout,
        **kwargs,
    )

@retry(
    wait=wait_random_exponential(multiplier=1, min=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def async_chat_completion_with_format_native(
    *,
    messages: Sequence[dict[str, Any]],
    format_schema: Union[str, dict[str, Any]],
    schema_name: str = "response_schema",
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    **kwargs: Any,
) -> ChatCompletion:
    """
    Async chat.completions wrapper with structured output format (with retries).
    
    Args:
        messages: Chat messages
        format_schema: JSON schema as string or dict defining the required output format
        schema_name: Name for the schema (default: "response_schema")
        model: Model to use
        temperature: Sampling temperature
        timeout: Request timeout
        **kwargs: Additional parameters
        
    Returns:
        ChatCompletion with structured output
    """
    # Parse schema if it's a string
    if isinstance(format_schema, str):
        schema = json.loads(format_schema)
    else:
        schema = format_schema
    
    # Construct response_format for structured outputs
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": True,
        }
    }
    
    if model.startswith("gpt-5"):
        # Special handling for gpt-5 models if needed
        return await _async_openai_chat_client.chat.completions.create(
            model=model,
            messages=list(messages),
            response_format=response_format,
            timeout=timeout,
            **kwargs,
        )

    return await _async_openai_chat_client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=list(messages),
        response_format=response_format,
        timeout=timeout,
        **kwargs,
    )


