from core.logger import get_logger, shutdown_logger
log = get_logger(__name__)

from core.config import settings
from transcribe.utilities.o4_transcribe_diarized import transcript_audio_file_verbose_o4_diarize
from transcribe.utilities.scenario_tools import async_detect_speaker_roles
import re


def transcribe_mono_with_diarization(
    file_name: str,
    metadata_text: str = "",
    temperature: float = 0.0,
    language: str = "uk",
    chunking_strategy: str = "auto"
) -> str:
    """
    Transcribe a mono audio file with speaker diarization.
    Uses OpenAI's gpt-4o-transcribe-diarize model to identify speakers,
    then maps speaker IDs to Agent/Client roles.

    Args:
        file_name: Path to mono audio file
        metadata_text: Optional metadata for context
        temperature: Temperature for transcription model
        language: Language code (default: "uk" for Ukrainian)
        chunking_strategy: Diarization chunking strategy

    Returns:
        Formatted dialogue text with AG:/CL: prefixes
    """
    log.info("\n" + "="*30 + " Starting mono file diarization " + "="*30)

    diarized_text = transcript_audio_file_verbose_o4_diarize(
        file_name=file_name,
        temperature=temperature,
        language=language,
        chunking_strategy=chunking_strategy
    )

    log.info(f"\nRaw diarized output:\n{diarized_text}\n")

    speaker_texts = _split_diarized_by_speaker(diarized_text)

    if len(speaker_texts) < 2:
        log.warning(f"Only {len(speaker_texts)} speaker(s) detected. Cannot split into roles.")
        return _format_single_speaker_output(diarized_text)

    log.info("\n" + "="*30 + " Detecting speaker roles (AG/CL) " + "="*30)
    speaker_0_text = speaker_texts.get("speaker_0", "")
    speaker_1_text = speaker_texts.get("speaker_1", "")

    if speaker_0_text and speaker_1_text:
        agent_text, client_text = async_detect_speaker_roles(speaker_0_text, speaker_1_text)
        log.info(f"\nSpeaker role mapping complete")

        role_map = {}
        if agent_text == speaker_0_text:
            role_map = {"speaker_0": "AG", "speaker_1": "CL"}
        else:
            role_map = {"speaker_0": "CL", "speaker_1": "AG"}

        formatted_text = _apply_role_mapping(diarized_text, role_map)

        formatted_text = _consolidate_consecutive_speakers(formatted_text)

        return formatted_text

    log.warning("Could not detect speaker roles properly")
    return diarized_text


def _split_diarized_by_speaker(diarized_text: str) -> dict:
    """
    Split diarized transcript into separate texts per speaker.

    Args:
        diarized_text: Raw diarized transcript with speaker labels

    Returns:
        Dictionary mapping speaker IDs to their combined text
    """
    speaker_texts = {}

    lines = diarized_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if ':' in line:
            speaker, text = line.split(':', 1)
            speaker = speaker.strip().lower()
            text = text.strip()

            if speaker not in speaker_texts:
                speaker_texts[speaker] = []
            speaker_texts[speaker].append(text)

    return {k: ' '.join(v) for k, v in speaker_texts.items()}


def _apply_role_mapping(diarized_text: str, role_map: dict) -> str:
    """
    Replace speaker IDs with Agent/Client role labels.

    Args:
        diarized_text: Raw diarized transcript
        role_map: Mapping from speaker_ID to AG/CL

    Returns:
        Text with AG:/CL: prefixes
    """
    result = []
    lines = diarized_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if ':' in line:
            speaker, text = line.split(':', 1)
            speaker = speaker.strip().lower()
            text = text.strip()

            role = role_map.get(speaker, speaker.upper())
            result.append(f"{role}: {text}")

    return '\n'.join(result)


def _consolidate_consecutive_speakers(text: str) -> str:
    """
    Merge consecutive lines from the same speaker into single lines.

    Args:
        text: Dialogue text with AG:/CL: prefixes

    Returns:
        Consolidated dialogue text
    """
    lines = text.strip().split('\n')
    consolidated = []
    current_speaker = None
    current_content = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if ':' in line:
            speaker, content = line.split(':', 1)
            speaker = speaker.strip()
            content = content.strip()

            if speaker == current_speaker:
                current_content.append(content)
            else:
                if current_speaker and current_content:
                    consolidated.append(f"{current_speaker}: {' '.join(current_content)}")

                current_speaker = speaker
                current_content = [content]

    if current_speaker and current_content:
        consolidated.append(f"{current_speaker}: {' '.join(current_content)}")

    return '\n'.join(consolidated)


def _format_single_speaker_output(diarized_text: str) -> str:
    """
    Format output when only one speaker is detected.

    Args:
        diarized_text: Raw diarized transcript

    Returns:
        Formatted text with UNKNOWN: prefix
    """
    lines = diarized_text.strip().split('\n')
    result = []

    for line in lines:
        line = line.strip()
        if line and ':' in line:
            speaker, text = line.split(':', 1)
            result.append(f"UNKNOWN: {text.strip()}")

    return '\n'.join(result)
