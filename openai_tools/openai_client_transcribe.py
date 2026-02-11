# openai_client_transcribe.py (router)

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Optional, Union, TypeAlias

from core.config import settings

# Import your two implementations (you already do this pattern for text)
from openai_tools.openai_client_transcribe_azure import (
    transcribe_audio_azure,
    async_transcribe_audio_azure,
    transcribe_audio_azure_diarized,
    async_transcribe_audio_azure_diarized,
)

from openai_tools.openai_client_transcribe_native import (
    transcribe_audio_native,
    async_transcribe_audio_native,
    transcribe_audio_native_diarized,
    async_transcribe_audio_native_diarized,
    Transcription,
)


# Re-export a common type for callers (so `from ... import Transcription` works)


AudioInput = Union[str, Path, BinaryIO]


def transcribe_audio(
    *,
    audio: AudioInput,
    model: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    response_format: Optional[str] = None,
    **kwargs: Any,
) -> Transcription:
    """
    Router wrapper:
      - Azure when settings.USE_AZURE_OPENAI == "Y"
      - Otherwise native OpenAI

    Parameters are explicit and passed through 1:1 to the underlying implementation.
    """
    if settings.USE_AZURE_OPENAI == "Y":
        return transcribe_audio_azure(
            audio=audio,
            model=model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            timeout=timeout,
            response_format=response_format,
            **kwargs,
        )

    return transcribe_audio_native(
        audio=audio,
        model=model,
        language=language,
        prompt=prompt,
        temperature=temperature,
        timeout=timeout,
        response_format=response_format,
        **kwargs,
    )


async def async_transcribe_audio(
    *,
    audio: AudioInput,
    model: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    response_format: Optional[str] = None,
    **kwargs: Any,
) -> Transcription:
    """
    Async router wrapper (same logic as transcribe_audio).
    """
    if settings.USE_AZURE_OPENAI == "Y":
        return await async_transcribe_audio_azure(
            audio=audio,
            model=model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            timeout=timeout,
            response_format=response_format,
            **kwargs,
        )

    return await async_transcribe_audio_native(
        audio=audio,
        model=model,
        language=language,
        prompt=prompt,
        temperature=temperature,
        timeout=timeout,
        response_format=response_format,
        **kwargs,
    )




def transcribe_audio_diarized(
    *,
    audio: AudioInput,
    model: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    chunking_strategy: str = "auto",
    timeout: float = 120.0,
    **kwargs: Any,
) -> Transcription:
    """
    Router wrapper for diarized transcription:
      - Azure when settings.USE_AZURE_OPENAI == "Y"
      - Otherwise native OpenAI

    Uses response_format="diarized_json" in the underlying implementations.
    """
    if settings.USE_AZURE_OPENAI == "Y":
        return transcribe_audio_azure_diarized(
            audio=audio,
            model=model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            chunking_strategy=chunking_strategy,
            timeout=timeout,
            **kwargs,
        )

    return transcribe_audio_native_diarized(
        audio=audio,
        model=model,
        language=language,
        prompt=prompt,
        temperature=temperature,
        chunking_strategy=chunking_strategy,
        timeout=timeout,
        **kwargs,
    )


async def async_transcribe_audio_diarized(
    *,
    audio: AudioInput,
    model: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    chunking_strategy: str = "auto",
    timeout: float = 120.0,
    **kwargs: Any,
) -> Transcription:
    """
    Async router wrapper for diarized transcription (same logic as transcribe_audio_diarized).
    """
    if settings.USE_AZURE_OPENAI == "Y":
        return await async_transcribe_audio_azure_diarized(
            audio=audio,
            model=model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            chunking_strategy=chunking_strategy,
            timeout=timeout,
            **kwargs,
        )

    return await async_transcribe_audio_native_diarized(
        audio=audio,
        model=model,
        language=language,
        prompt=prompt,
        temperature=temperature,
        chunking_strategy=chunking_strategy,
        timeout=timeout,
        **kwargs,
    )


