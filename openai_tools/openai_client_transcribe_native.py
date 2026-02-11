from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Optional, Union

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential

from openai import AsyncOpenAI, OpenAI
from openai.types.audio import Transcription

from core.config import settings


AudioInput = Union[str, Path, BinaryIO]


# --- clients (same style as openai_client_text_native.py) ---
_openai_transcribe_client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    timeout=httpx.Timeout(120.0, connect=10.0),
)

_async_openai_transcribe_client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    http_client=httpx.AsyncClient(
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        timeout=httpx.Timeout(120.0, connect=10.0),
    ),
)


def _default_transcribe_model() -> str:
    # Use your existing config name if present; fall back safely.
    return getattr(settings, "OPENAI_MODEL_TRANSCRIBE_STEREO", "whisper-1")


def _normalize_timestamp_granularities(
    timestamp_granularities: Optional[Union[str, Iterable[str]]],
) -> Optional[list[str]]:
    if timestamp_granularities is None:
        return None
    if isinstance(timestamp_granularities, str):
        return [timestamp_granularities]
    return list(timestamp_granularities)


@contextmanager
def _open_audio(audio: AudioInput) -> Iterator[BinaryIO]:
    """
    Open path-like inputs as binary files; pass through file-like objects.
    """
    if isinstance(audio, (str, Path)):
        with open(str(audio), "rb") as f:
            yield f
    else:
        # Assume file-like opened in "rb"
        yield audio


def transcribe_audio_native(
    *,
    audio: AudioInput,
    model: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    response_format: str = "json",
    timestamp_granularities: Optional[Union[str, Iterable[str]]] = None,
    timeout: float = 120.0,
    **kwargs: Any,
) -> Transcription:
    """
    Native OpenAI transcription wrapper.

    Notes:
      - `file` must be a file object (not a filename). :contentReference[oaicite:1]{index=1}
      - If `timestamp_granularities` is provided, `response_format` must be "verbose_json". :contentReference[oaicite:2]{index=2}
    """
    model = model or _default_transcribe_model()

    ts = _normalize_timestamp_granularities(timestamp_granularities)
    if ts and response_format != "json":
        response_format = "json"

    with _open_audio(audio) as audio_file:
        return _openai_transcribe_client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format=response_format,
            timestamp_granularities=ts,
            timeout=timeout,
            **kwargs,
        )


@retry(
    wait=wait_random_exponential(multiplier=1, min=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def async_transcribe_audio_native(
    *,
    audio: AudioInput,
    model: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    response_format: str = "json",
    timestamp_granularities: Optional[Union[str, Iterable[str]]] = None,
    timeout: float = 120.0,
    **kwargs: Any,
) -> Transcription:
    """
    Async native transcription wrapper (with retries).
    """
    model = model or _default_transcribe_model()

    ts = _normalize_timestamp_granularities(timestamp_granularities)
    if ts and response_format != "json":
        response_format = "json"

    with _open_audio(audio) as audio_file:
        return await _async_openai_transcribe_client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format=response_format,
            timestamp_granularities=ts,
            timeout=timeout,
            **kwargs,
        )






# --- diarize defaults ---
def _default_transcribe_diarize_model() -> str:
    # Prefer a dedicated setting if you have one; fall back safely.
    return settings.OPENAI_MODEL_TRANSCRIBE_DIARIZE or "gpt-4o-transcribe-diarize"


@retry(
    wait=wait_random_exponential(multiplier=1, min=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
)
def transcribe_audio_native_diarized(
    *,
    audio: AudioInput,
    model: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    chunking_strategy: str = "auto",
    timeout: float = 120.0,
    **kwargs: Any,
) -> Any:
    """
    Sync native diarized transcription wrapper (with retries).
    Uses response_format="diarized_json".
    """
    model = model or _default_transcribe_diarize_model()

    print(f"Transcribe '{str(audio)}' using native diarize model: {model}")

    with _open_audio(audio) as audio_file:
        return _openai_transcribe_client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format="diarized_json",
            chunking_strategy=chunking_strategy,
            timeout=timeout,
            **kwargs,
        )


@retry(
    wait=wait_random_exponential(multiplier=1, min=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def async_transcribe_audio_native_diarized(
    *,
    audio: AudioInput,
    model: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    chunking_strategy: str = "auto",
    timeout: float = 120.0,
    **kwargs: Any,
) -> Any:
    """
    Async native diarized transcription wrapper (with retries).
    Uses response_format="diarized_json".
    """
    model = model or _default_transcribe_diarize_model()

    with _open_audio(audio) as audio_file:
        return await _async_openai_transcribe_client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format="diarized_json",
            chunking_strategy=chunking_strategy,
            timeout=timeout,
            **kwargs,
        )
