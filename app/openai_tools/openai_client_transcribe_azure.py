from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Optional, Union

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential

from openai import AsyncAzureOpenAI, AzureOpenAI
from openai.types.audio import Transcription, TranscriptionDiarized
from app.core.config import settings


AudioInput = Union[str, Path, BinaryIO]


# Expected settings (like your text azure client):
# - AZURE_OPENAI_API_KEY
# - AZURE_OPENAI_ENDPOINT      e.g. "https://<resource>.openai.azure.com"
# - AZURE_OPENAI_API_VERSION   must support audio/transcriptions on your resource
#
# IMPORTANT: For Azure, `model=` is your *deployment name*. :contentReference[oaicite:3]{index=3}


# --- clients (same style as openai_client_text_azure.py) ---
_azure_transcribe_client = AzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
    timeout=httpx.Timeout(120.0, connect=10.0),
)

_async_azure_transcribe_client = AsyncAzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
    http_client=httpx.AsyncClient(
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        timeout=httpx.Timeout(120.0, connect=10.0),
    ),
)


def _default_azure_transcribe_deployment() -> str:
    # Prefer a dedicated Azure deployment setting if you have one.
    return settings.AZURE_MODEL_TRANSCRIBE_STEREO or "gpt-4o-transcribe-azure-test"

def _default_azure_transcribe_diarize_deployment() -> str:
    # For Azure, model= is deployment name
    return settings.AZURE_MODEL_TRANSCRIBE_DIARIZE or "gpt-4o-transcribe-diarize-azure-test"


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
    if isinstance(audio, (str, Path)):
        with open(str(audio), "rb") as f:
            yield f
    else:
        yield audio




@retry(
    wait=wait_random_exponential(multiplier=1, min=1, max=40),
    stop=stop_after_attempt(3),
    reraise=True,
)
def transcribe_audio_azure(
    *,
    audio: AudioInput,
    model: Optional[str] = None,  # Azure deployment name
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    response_format: str = "json",
    timestamp_granularities: Optional[Union[str, Iterable[str]]] = None,
    timeout: float = 120.0,
    **kwargs: Any,
) -> Transcription:
    """
    Azure OpenAI transcription wrapper.

    Notes:
      - In Azure, `model` is your *deployment name*. :contentReference[oaicite:4]{index=4}
      - If `timestamp_granularities` is provided, `response_format` must be "verbose_json". :contentReference[oaicite:5]{index=5}
    """
    deployment = model or _default_azure_transcribe_deployment()
    if not deployment:
        raise ValueError("model must be set to your Azure Whisper (or transcribe) deployment name")

    ts = _normalize_timestamp_granularities(timestamp_granularities)
    if ts and response_format != "json":
        response_format = "json"

    with _open_audio(audio) as audio_file:
        return _azure_transcribe_client.audio.transcriptions.create(
            file=audio_file,
            model=deployment,
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
async def async_transcribe_audio_azure(
    *,
    audio: AudioInput,
    model: Optional[str] = None,  # Azure deployment name
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    response_format: str = "json",
    timestamp_granularities: Optional[Union[str, Iterable[str]]] = None,
    timeout: float = 120.0,
    **kwargs: Any,
) -> Transcription:
    """
    Async Azure transcription wrapper (with retries).
    """
    deployment = model or _default_azure_transcribe_deployment()
    if not deployment:
        raise ValueError("model must be set to your Azure Whisper (or transcribe) deployment name")

    ts = _normalize_timestamp_granularities(timestamp_granularities)
    if ts and response_format != "json":
        response_format = "json"

    with _open_audio(audio) as audio_file:
        return await _async_azure_transcribe_client.audio.transcriptions.create(
            file=audio_file,
            model=deployment,
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
def transcribe_audio_azure_diarized(
    *,
    audio: AudioInput,
    model: Optional[str] = None,  # Azure deployment name
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    chunking_strategy: str = "auto",
    timeout: float = 120.0,
    **kwargs: Any,
) -> TranscriptionDiarized:
    """
    Sync Azure diarized transcription wrapper (with retries).
    Uses response_format="diarized_json".

    Notes:
      - In Azure, `model` is your *deployment name*.
    """
    deployment = model or _default_azure_transcribe_diarize_deployment()
    if not deployment:
        raise ValueError("model must be set to your Azure diarize transcribe deployment name")
    
    print(f"Transcribe '{str(audio)}' using Azure diarize deployment: {deployment}")
    
    with _open_audio(audio) as audio_file:
        return _azure_transcribe_client.audio.transcriptions.create(
            file=audio_file,
            model=deployment,
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
async def async_transcribe_audio_azure_diarized(
    *,
    audio: AudioInput,
    model: Optional[str] = None,  # Azure deployment name
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    chunking_strategy: str = "auto",
    timeout: float = 120.0,
    **kwargs: Any,
) -> TranscriptionDiarized:
    """
    Async Azure diarized transcription wrapper (with retries).
    Uses response_format="diarized_json".
    """
    deployment = model or _default_azure_transcribe_diarize_deployment()
    if not deployment:
        raise ValueError("model must be set to your Azure diarize transcribe deployment name")

    with _open_audio(audio) as audio_file:
        return await _async_azure_transcribe_client.audio.transcriptions.create(
            file=audio_file,
            model=deployment,
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format="diarized_json",
            chunking_strategy=chunking_strategy,
            timeout=timeout,
            **kwargs,
        )
