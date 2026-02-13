from __future__ import annotations
from typing import Any, BinaryIO, Iterable, Iterator, Optional, Union
from pathlib import Path
import httpx
from openai import AzureOpenAI
from contextlib import contextmanager

from core.config import settings
from core.logger import get_logger, shutdown_logger
log = get_logger(__name__)


AudioInput = Union[str, Path, BinaryIO]

azure_subscription_key = settings.AZURE_OPENAI_API_KEY
azure_endpoint = settings.AZURE_OPENAI_ENDPOINT
azure_api_version = settings.AZURE_OPENAI_API_VERSION

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

try:
    model = settings.AZURE_MODEL_TRANSCRIBE_STEREO
    print("\n\n"+"-"*60)
    print(f"Sending data to model: {model} @ {azure_endpoint}")
    with open_audio(audio_path) as audio_file:
        tr = azure_client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            language="uk",
            response_format="json",
        )
    print("Transcription results:")
    print(f"model: {tr.model_config}\nEndpoint: {azure_endpoint}\nApi version: {azure_api_version}\nDeployment: {model}\n")
    print(tr)
except httpx.HTTPStatusError as e:
    print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}") 
    print(f"Request details: Endpoint: {azure_endpoint}\nApi version: {azure_api_version}\nDeployment: {model}\n")



try:
    model = settings.AZURE_MODEL_TRANSCRIBE_DIARIZE  
    print("\n\n"+"-"*60)
    print(f"Sending data to model: {model} @ {azure_endpoint}")
    with open_audio(audio_path) as audio_file:
        tr = azure_client.audio.transcriptions.create(
            file=audio_file,
            model= model,
            language="uk",
            response_format="diarized_json",
            chunking_strategy = "auto",
        )
    print("Transcription results:")
    print(f"model: {tr.model_config}\nEndpoint: {azure_endpoint}\nApi version: {azure_api_version}\nDeployment: {model}\n")
    print(tr)
except httpx.HTTPStatusError as e:
    print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}") 
    print(f"Request details: Endpoint: {azure_endpoint}\nApi version: {azure_api_version}\nDeployment: {model}\n")




# --- Chat with o4 (Azure deployment) ---
try:
    model = settings.AZURE_MODEL_CHAT_TRS_SPLIT_INTO_ROLES
    print("\n\n"+"-"*60)
    print(f"Sending data to model: {model} @ {azure_endpoint}")
    resp_large = azure_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Show me fibanacii 10"},
        ],
    )
    print("Request results:")
    print(f"model: {resp_large.model}\nEndpoint: {azure_endpoint}\nApi version: {azure_api_version}\nDeployment: {model}\n")
    print(resp_large)
except httpx.HTTPStatusError as e:
    print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}") 
    print(f"Request details: Endpoint: {azure_endpoint}\nApi version: {azure_api_version}\nDeployment: {model}\n") 



# --- Chat with o4-mini (Azure deployment) ---
try:
    model = settings.AZURE_MODEL_CHAT_SUMMARY   
    print("\n\n"+"-"*60)
    print(f"Sending data to model: {model} @ {azure_endpoint}")
    resp_mini = azure_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Show me fibanacii 10"},
        ],
    )
    print("Request results:")
    print(f"model: {resp_mini.model}\nEndpoint: {azure_endpoint}\nApi version: {azure_api_version}\nDeployment: {model}\n")
    print(resp_mini)
except httpx.HTTPStatusError as e:
    print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}") 
    print(f"Request details: Endpoint: {azure_endpoint}\nApi version: {azure_api_version}\nDeployment: {model}\n")


shutdown_logger()