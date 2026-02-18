from __future__ import annotations

import json
from typing import Any, Optional, Sequence, Union

import httpx
from tenacity import retry, wait_random_exponential, stop_after_attempt

from openai import AzureOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion

from core.config import settings
from core.logger import log

# Expected settings (add to your settings / env):
# - AZURE_OPENAI_API_KEY
# - AZURE_OPENAI_ENDPOINT   e.g. "https://<resource-name>.openai.azure.com"
# - AZURE_OPENAI_API_VERSION e.g. "2024-02-01" (pick what your resource supports)
#
# IMPORTANT: In Azure, `model=` must be your *deployment name* (often same as model, but not required).
# See: https://learn.microsoft.com/.../switching-endpoints :contentReference[oaicite:2]{index=2}


# --- clients (same style as openai_client_text.py) ---
azure_chat_client = AzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
    timeout=httpx.Timeout(120.0, connect=10.0),
)

_async_azure_chat_client = AsyncAzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
    http_client=httpx.AsyncClient(
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        timeout=httpx.Timeout(120.0, connect=10.0),
    ),
)


@retry(
    wait=wait_random_exponential(multiplier=1, min=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
)
def chat_completion_azure(
    *,
    messages: Sequence[dict[str, Any]],
    model: Optional[str] = None,          # Azure deployment name
    temperature: float = 0.0,
    timeout: float = 120.0,
    **kwargs: Any,
) -> ChatCompletion:
    """
    Sync wrapper for Azure chat.completions.
    In Azure, `model` is the *deployment name*.
    """
    if not model:
        raise ValueError("model must be set to your Azure deployment name")

    log.info(f"Calling AZURE model: {model}")
    # Keep your GPT-5 rule: don't send temperature for gpt-5 deployments (if your deployment is named 'gpt-5-*')
    if model.startswith("gpt-5"):
        return azure_chat_client.chat.completions.create(
            model=model,
            messages=list(messages),
            timeout=timeout,
            **kwargs,
        )

    return azure_chat_client.chat.completions.create(
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
async def async_chat_completion_azure(
    *,
    messages: Sequence[dict[str, Any]],
    model: Optional[str] = None,          # Azure deployment name
    temperature: float = 0.0,
    timeout: float = 120.0,
    **kwargs: Any,
) -> ChatCompletion:
    """
    Async wrapper for Azure chat.completions (with retries).
    """
    if not model:
        raise ValueError("model must be set to your Azure deployment name")

    log.info(f"Calling async AZURE model: {model}")
    if model.startswith("gpt-5"):
        return await _async_azure_chat_client.chat.completions.create(
            model=model,
            messages=list(messages),
            timeout=timeout,
            **kwargs,
        )

    return await _async_azure_chat_client.chat.completions.create(
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
def chat_completion_with_format_azure(
    *,
    messages: Sequence[dict[str, Any]],
    format_schema: Union[str, dict[str, Any]],
    schema_name: str = "response_schema",
    model: Optional[str] = None,          # Azure deployment name
    temperature: float = 0.0,
    timeout: float = 120.0,
    **kwargs: Any,
) -> ChatCompletion:
    """
    Sync wrapper with structured output response_format=json_schema.
    """
    if not model:
        raise ValueError("model must be set to your Azure deployment name")

    schema = json.loads(format_schema) if isinstance(format_schema, str) else format_schema

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": True,
        },
    }

    log.info(f"Calling AZURE model with format: {model}")
    if model.startswith("gpt-5"):
        return azure_chat_client.chat.completions.create(
            model=model,
            messages=list(messages),
            response_format=response_format,
            timeout=timeout,
            **kwargs,
        )

    return azure_chat_client.chat.completions.create(
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
async def async_chat_completion_with_format_azure(
    *,
    messages: Sequence[dict[str, Any]],
    format_schema: Union[str, dict[str, Any]],
    schema_name: str = "response_schema",
    model: Optional[str] = None,          # Azure deployment name
    temperature: float = 0.0,
    timeout: float = 120.0,
    **kwargs: Any,
) -> ChatCompletion:
    """
    Async wrapper with structured output response_format=json_schema (with retries).
    """
    if not model:
        raise ValueError("model must be set to your Azure deployment name")

    schema = json.loads(format_schema) if isinstance(format_schema, str) else format_schema

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": True,
        },
    }

    log.info(f"Calling async AZURE model with format: {model}")

    if model.startswith("gpt-5"):
        return await _async_azure_chat_client.chat.completions.create(
            model=model,
            messages=list(messages),
            response_format=response_format,
            timeout=timeout,
            **kwargs,
        )

    return await _async_azure_chat_client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=list(messages),
        response_format=response_format,
        timeout=timeout,
        **kwargs,
    )
