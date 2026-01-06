from __future__ import annotations

from typing import Iterable, Optional, Union
import httpx
from tenacity import retry, wait_random_exponential, stop_after_attempt

from openai import OpenAI, AsyncOpenAI
from openai.types.audio import Transcription

from core.config import settings


# --- clients (same style as openai_client_utilities.py) ---
_transcribe_client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    timeout=httpx.Timeout(120.0, connect=10.0),
)

_async_transcribe_client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    http_client=httpx.AsyncClient(
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        timeout=httpx.Timeout(120.0, connect=10.0),
    ),
)


def _normalize_timestamp_granularities(
    timestamp_granularities: Optional[Union[str, Iterable[str]]]
) -> Optional[list[str]]:
    if timestamp_granularities is None:
        return None
    if isinstance(timestamp_granularities, str):
        return [timestamp_granularities]
    return list(timestamp_granularities)


def transcribe_audio_file(
    file_path: str,
    *,
    model: Optional[str] = None,
    prompt: str = "",
    temperature: float = 0.0,
    response_format: str = "json",
    timestamp_granularities: Optional[Union[str, Iterable[str]]] = None,
    timeout: float = 120.0,
) -> Transcription:
    """
    Sync transcription tool call.

    Note:
      - If you pass timestamp_granularities, response_format MUST be "verbose_json".
        (OpenAI docs requirement)
    """
    model = model or settings.OPENAI_MODEL_TRANSCRIBE_STEREO

    ts = _normalize_timestamp_granularities(timestamp_granularities)
    if ts and response_format != "json":
        response_format = "json"  # required when timestamps requested

    with open(file_path, "rb") as audio_file:
        return _transcribe_client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            response_format=response_format,
            temperature=temperature,
            prompt=prompt,
            timestamp_granularities=ts,
            timeout=timeout,
        )


@retry(
    wait=wait_random_exponential(multiplier=1, min=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def async_transcribe_audio_file(
    file_path: str,
    *,
    model: Optional[str] = None,
    prompt: str = "",
    temperature: float = 0.0,
    response_format: str = "json",
    timestamp_granularities: Optional[Union[str, Iterable[str]]] = None,
    timeout: float = 120.0,
) -> Transcription:
    """
    Async transcription tool call (with retries, same style as your async chat utilities).
    """
    model = model or settings.OPENAI_MODEL_TRANSCRIBE_STEREO

    ts = _normalize_timestamp_granularities(timestamp_granularities)
    if ts and response_format != "json":
        response_format = "json"

    with open(file_path, "rb") as audio_file:
        return await _async_transcribe_client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            response_format=response_format,
            temperature=temperature,
            prompt=prompt,
            timestamp_granularities=ts,
            timeout=timeout,
        )
