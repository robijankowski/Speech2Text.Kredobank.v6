from __future__ import annotations
from typing import Any, BinaryIO, Iterable, Iterator, Optional, Union
from pathlib import Path
import httpx
from openai import AzureOpenAI
from contextlib import contextmanager

from core.config import settings

AudioInput = Union[str, Path, BinaryIO]

AZURE_MODEL_TRANSCRIBE="gpt-4o-transcribe-azure-test"
AZURE_MODEL_CHAT_LARGE="gpt-5.2-chat-azure-test"
AZURE_MODEL_CHAT_MINI="gpt-5-mini-azure-test"

azure_subscription_key = settings.AZURE_OPENAI_API_KEY
azure_endpoint = "https://testkbazure.cognitiveservices.azure.com/"

azure_api_version = "2024-12-01-preview"

azure_client = AzureOpenAI(
    api_version=azure_api_version,
    azure_endpoint=azure_endpoint,
    api_key=azure_subscription_key,
)


@contextmanager
def open_audio(audio: AudioInput) -> Iterator[BinaryIO]:
    """
    Opens a local audio path safely and keeps it open during the API call.
    If a file-like object is provided, it is used as-is.
    """
    if isinstance(audio, (str, Path)):
        f = open(str(audio), "rb")
        try:
            yield f
        finally:
            f.close()
    else:
        # already a file-like object
        yield audio


# --- Transcription ---
audio_path = Path("./test/test_call.wav")

with open_audio(audio_path) as audio_file:
    tr = azure_client.audio.transcriptions.create(
        file=audio_file,
        model=AZURE_MODEL_TRANSCRIBE,
        language="uk",
        response_format="json",
    )
print("\n\n"+"-"*60)
print("Transcription results:")
print(f"model: {tr.model_config}\nEndpoint: {azure_endpoint}\nApi version: {azure_api_version}\nDeployment: {AZURE_MODEL_TRANSCRIBE}\n")
print(tr.text)

# --- Chat with o4 (Azure deployment) ---
resp_large = azure_client.chat.completions.create(
    model=AZURE_MODEL_CHAT_LARGE,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Show me fibanacii 10"},
    ],
)
print("\n\n"+"-"*60)
print("Request results:")
print(f"model: {resp_large.model}\nEndpoint: {azure_endpoint}\nApi version: {azure_api_version}\nDeployment: {AZURE_MODEL_CHAT_LARGE}\n")
print(resp_large.choices[0].message.content)

# --- Chat with o4-mini (Azure deployment) ---
resp_mini = azure_client.chat.completions.create(
    model=AZURE_MODEL_CHAT_MINI,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Show me fibanacii 10"},
    ],
)
print("\n\n"+"-"*60)
print("Request results:")
print(f"model: {resp_mini.model}\nEndpoint: {azure_endpoint}\nApi version: {azure_api_version}\nDeployment: {AZURE_MODEL_CHAT_MINI}\n")
print(resp_mini.choices[0].message.content)