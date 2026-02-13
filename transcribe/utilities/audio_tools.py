from __future__ import annotations

from typing import Literal, Optional

from pydub import AudioSegment
from pydub.effects import normalize
import os

import logging
from core.config import settings

log = logging.getLogger(settings.TR_LOGGER_NAME)


from transcribe.utilities.noise_reduction import reduce_audio_noise
from transcribe.utilities.silence_removal import remove_long_silences


from pathlib import Path
from datetime import datetime
import shutil


def prepare_audio_for_transcription(source_file: str, temp_dir: str):
    """
    1) bierze: plik źródłowy + katalog tymczasowy
    2) tworzy w temp_dir podkatalog YYYY-MM-DD (jeśli nie istnieje)
    3) kopiuje tam plik źródłowy
    4) robi preprocessing: optimize_audio_files + remove_long_silences_in_audio
    Zwraca: (left_cleaned, right_cleaned, org_cleaned)
    """
    src = Path(source_file)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    day_dir = Path(temp_dir) / datetime.now().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    dst = day_dir / src.name

    shutil.copy2(src, dst)
    log.info("Copied source file to temp: %s -> %s", str(src), str(dst))


    l_file, r_file = split_stereo_to_lr_and_clean(str(dst))
    o_file = clean_audio_file(str(dst))

    left_file_cleaned = ""
    right_file_cleaned = ""
    org_file_cleaned = ""

    if l_file:
         left_file_cleaned = remove_long_silences_in_audio(l_file)
    if r_file:
         right_file_cleaned = remove_long_silences_in_audio(r_file)
    org_file_cleaned = remove_long_silences_in_audio(o_file)

    return left_file_cleaned, right_file_cleaned, org_file_cleaned




def split_stereo_to_lr_and_clean(stereo_file_path):
    """
    Split a stereo WAV file into two mono files (left and right channels)
    Returns paths to the created mono files
    """
    # Load the stereo audio file
    stereo_audio = AudioSegment.from_wav(stereo_file_path)
    
    # Check if it's actually stereo
    if stereo_audio.channels != 2:
        log.warning(f"Warning: {stereo_file_path} is not stereo (has {stereo_audio.channels} channels)")
        return None, None
    
    # Split into left and right channels
    left_channel = stereo_audio.split_to_mono()[0]
    left_channel = normalize(left_channel)
    left_channel = left_channel.set_frame_rate(16000)

    right_channel = stereo_audio.split_to_mono()[1]
    right_channel = normalize(right_channel)
    right_channel = right_channel.set_frame_rate(16000)
    
    # Create output file names
    base_name = os.path.splitext(stereo_file_path)[0]
    left_file = f"{base_name}_lc.wav"
    right_file = f"{base_name}_rc.wav"

    # Export mono files
    left_channel.export(left_file, format="wav")
    right_channel.export(right_file, format="wav")

    left_file = reduce_audio_noise(left_file, method="gentle")
    right_file = reduce_audio_noise(right_file, method="gentle")

    # analyze_silence_patterns(left_file)
    # analyze_silence_patterns(right_file)

    log.info(f"Created: {left_file}")
    log.info(f"Created: {right_file}")
    
    return left_file, right_file


def clean_audio_file(input_file) -> str:
    """
    Quick and simple cleanup for immediate use
    """
    audio = AudioSegment.from_file(input_file)
    
    audio = normalize(audio)
    audio = audio.set_frame_rate(16000)

    # Create output filename
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_oc.wav"

    audio.export(output_file, format="wav")
    
    output_file = reduce_audio_noise(output_file, method="gentle")

    log.info(f"Created: {output_file}")
    return output_file



def remove_long_silences_in_audio( file_name ) -> str:
    new_file = remove_long_silences(   file_name,
                                        method='pydub',           # Best for phone audio quality
                                        silence_thresh=-55,       # Phone calls have higher noise floor
                                        min_silence_len=2000,     # Remove pauses longer than 1.2 seconds
                                        keep_silence=500,         # Keep 200ms for natural speech flow
                                        pause_duration=600  )      # 400ms pause between segments
    return new_file




def trim_audio_to_secs(in_file: str, out_file: str, secs: float) -> str:
    """
    Ucina plik audio do pierwszych `secs` sekund i zapisuje do `out_file`.
    Params: in_file, out_file, secs
    Returns: out_file
    """
    if secs is None or secs <= 0:
        raise ValueError("secs musi być > 0")

    audio = AudioSegment.from_file(in_file)
    target_ms = int(secs * 1000)

    trimmed = audio[:target_ms]  # jeśli secs > długość, pydub po prostu zwróci całe audio

    ext = os.path.splitext(out_file)[1].lower().lstrip(".")
    out_format = ext if ext else "wav"

    trimmed.export(out_file, format=out_format)
    return out_file





MonoMode = Literal["left", "right", "mix"]


def stereo_to_mono(
    stereo_file_path: str,
    out_file: Optional[str] = None,
    *,
    mode: MonoMode = "mix",
    frame_rate: int = 16000,
) -> str:
    """
    Create a mono WAV from a stereo audio file.

    mode:
      - "left": keep left channel only
      - "right": keep right channel only
      - "mix": average left+right into one mono channel
    """
    audio = AudioSegment.from_file(stereo_file_path)

    if out_file is None:
        base, _ = os.path.splitext(stereo_file_path)
        out_file = f"{base}_mono.wav"

    if audio.channels == 1:
        mono = audio
    elif audio.channels == 2:
        if mode == "left":
            mono = audio.split_to_mono()[0]
        elif mode == "right":
            mono = audio.split_to_mono()[1]
        elif mode == "mix":
            mono = audio.set_channels(1)  # downmix
        else:
            raise ValueError(f"Unknown mode: {mode}")
    else:
        # fallback: downmix multi-channel to mono
        mono = audio.set_channels(1)

    mono = mono.set_frame_rate(frame_rate)
    mono.export(out_file, format="wav")
    return out_file
