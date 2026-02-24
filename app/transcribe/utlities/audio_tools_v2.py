from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


from typing import Literal, Optional, Tuple

from pydub import AudioSegment
from pydub.effects import normalize
import os

from app.core.config import settings
from app.core.logger import log


from app.transcribe.utlities.noise_reduction import reduce_audio_noise
from app.transcribe.utlities.silence_removal import remove_long_silences


from pathlib import Path
from datetime import datetime
import shutil

ASR_AUDIO_PRESET = {
    # Bandpass
    "hp_hz": 100,
    "lp_hz": 7800,

    # Compression
    "compress_threshold": -22.0,
    "compress_ratio": 2.0,

    # Peak normalize (safe)
    "target_peak_dbfs": -1.0,
    "max_gain_db": 8.0,

    # Crosstalk ducking
    "activity_dbfs": -35.0,
    "dominance_db": 8.0,
    "duck_db": 12.0,   # reduce to 6–9 if overlaps get lost
}

ASR_AUDIO_PRESET_HARD = {
    # Narrower phone band (removes more rumble + hiss)
    "hp_hz": 140,
    "lp_hz": 6500,

    # Slightly stronger compression (helps digits/quiet consonants)
    "compress_threshold": -24.0,
    "compress_ratio": 2.6,

    # Normalize safer (avoid amplifying noise too much)
    "target_peak_dbfs": -2.0,
    "max_gain_db": 5.0,

    # Crosstalk ducking (often helps on bad calls)
    "activity_dbfs": -33.0,
    "dominance_db": 7.0,
    "duck_db": 14.0,

    # Extra: clip protect
    "if_clipped_reduce_db": 4.0,

    # Extra: hum notch (Europe often 50Hz)
    "hum_hz": 50,
}


@dataclass
class ClippingStats:
    sample_width: int
    clipped_samples: int
    total_samples: int
    clipped_ratio: float
    max_abs: int
    max_possible: int

    @property
    def is_clipped(self) -> bool:
        # treat as clipped if >=0.05% samples hit near-full scale
        return self.clipped_ratio >= 0.0005




def _make_temp_audio_file(source_file: str, temp_root_dir: str = "") -> str:
    """
    1) bierze: plik źródłowy + katalog tymczasowy
    2) tworzy w temp_dir podkatalog YYYY-MM-DD (jeśli nie istnieje)
    3) kopiuje tam plik źródłowy
    4) robi preprocessing: optimize_audio_files + remove_long_silences_in_audio
    Zwraca: (left_cleaned, right_cleaned, org_cleaned)
    """
    if not temp_root_dir:
        temp_root_dir = settings.TR_TEMP_ROOT_DIR
    src = Path(source_file)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    day_dir = Path(temp_root_dir) / datetime.now().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    dst = day_dir / src.name

    shutil.copy2(src, dst)
    log.info("Copied source file to temp: %s -> %s", str(src), str(dst))
    return str(dst)






def clean_audio_file(source_file: str, temp_root_dir: str = None) -> str:
    """
    Quick and simple cleanup for immediate use
    """
    tmp_file = _make_temp_audio_file(source_file, temp_root_dir=temp_root_dir)

    audio = AudioSegment.from_file(tmp_file)
    
    audio = normalize(audio)
    audio = audio.set_frame_rate(16000)

    # Create output filename
    base_name = os.path.splitext(tmp_file)[0]
    output_file = f"{base_name}_orgch.wav"

    audio.export(output_file, format="wav")
    
    output_file = reduce_audio_noise(output_file, method="gentle")

    log.info(f"Created: {output_file}")
    return output_file



def clean_audio_file_with_silence_removal(source_file: str, temp_root_dir: str = None) -> str:
    """
    Quick and simple cleanup for immediate use
    """
    tmp_file = _make_temp_audio_file(source_file, temp_root_dir=temp_root_dir)

    audio = AudioSegment.from_file(tmp_file)
    
    audio = normalize(audio)
    audio = audio.set_frame_rate(16000)

    # Create output filename
    base_name = os.path.splitext(tmp_file)[0]
    output_file = f"{base_name}_orgch.wav"

    audio.export(output_file, format="wav")
    
    output_file = reduce_audio_noise(output_file, method="gentle")
    output_file_sr = remove_long_silences_in_audio(output_file)

    log.info(f"Created: {output_file_sr}")
    return output_file_sr



def clean_audio_file_with_silence_removal_asr(source_file: str, temp_root_dir: str = None) -> str:

    tmp_file = _make_temp_audio_file(source_file, temp_root_dir=temp_root_dir)

    log.info(f"Cleaning audio file: {tmp_file}")
    audio = AudioSegment.from_file(tmp_file)
    log.info(f"Loaded audio: {tmp_file}, duration: {len(audio)/1000:.2f}s, channels: {audio.channels}, frame_rate: {audio.frame_rate}Hz")

    log.info(f"Normalizing audio {tmp_file}")
    audio = process_mono_for_asr(
        audio,
        frame_rate=16000,
        hp_hz=ASR_AUDIO_PRESET["hp_hz"],
        lp_hz=ASR_AUDIO_PRESET["lp_hz"],
        compress_threshold=ASR_AUDIO_PRESET["compress_threshold"],
        compress_ratio=ASR_AUDIO_PRESET["compress_ratio"],
        target_peak_dbfs=ASR_AUDIO_PRESET["target_peak_dbfs"],
        max_gain_db=ASR_AUDIO_PRESET["max_gain_db"],
        )

    base_name = os.path.splitext(tmp_file)[0]
    output_file = f"{base_name}_orgch.wav"
    audio.export(output_file, format="wav")
    log.info(f"Audio mono processed for ASR: {output_file}")

    log.info(f"Applying noise reduction (gentle) to cleaned audio {output_file}")
    output_file = reduce_audio_noise(output_file, method="gentle")
    log.info(f"Applied noise reduction to cleaned audio, saved to {output_file}")
    return output_file

    # org_file_cleaned = remove_long_silences_in_audio(output_file)
    # return org_file_cleaned









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







def _np_samples(seg: AudioSegment) -> np.ndarray:
    """Mono only. Returns numpy array of correct dtype."""
    arr = np.array(seg.get_array_of_samples())
    return arr


def detect_clipping(seg: AudioSegment, *, near_full_scale: float = 0.995) -> ClippingStats:
    """
    Detects clipping-like behavior: many samples at/near full-scale.
    near_full_scale=0.995 means 99.5% of max int range.
    """
    x = _np_samples(seg)
    dtype = x.dtype
    info = np.iinfo(dtype)
    max_possible = int(max(abs(info.min), info.max))
    thr = int(max_possible * near_full_scale)

    max_abs = int(np.max(np.abs(x))) if x.size else 0
    clipped = int(np.sum(np.abs(x) >= thr))
    total = int(x.size)
    ratio = (clipped / total) if total else 0.0

    return ClippingStats(
        sample_width=seg.sample_width,
        clipped_samples=clipped,
        total_samples=total,
        clipped_ratio=ratio,
        max_abs=max_abs,
        max_possible=max_possible,
    )


def peak_normalize_safe(
    seg: AudioSegment,
    *,
    target_peak_dbfs: float = -1.0,
    max_gain_db: float = 10.0,
) -> AudioSegment:
    """
    Peak-normalize to target (e.g. -1dBFS), but clamp gain change.
    This avoids aggressive boosts that can amplify noise/artifacts.
    """
    if len(seg) == 0:
        return seg

    # max_dBFS is peak-based; can be -inf for silence
    if seg.max_dBFS == float("-inf"):
        return seg

    gain_needed = target_peak_dbfs - seg.max_dBFS
    gain_needed = max(-max_gain_db, min(max_gain_db, gain_needed))
    return seg.apply_gain(gain_needed)


def apply_phone_bandpass(
    seg: AudioSegment,
    *,
    hp_hz: int = 100,
    lp_hz: int = 7800,
) -> AudioSegment:
    """
    Simple phone-style bandpass: removes rumble + hiss.
    Uses pydub filters (fast, decent).
    """
    out = seg
    if hp_hz and hp_hz > 0:
        out = out.high_pass_filter(hp_hz)
    if lp_hz and lp_hz > 0:
        out = out.low_pass_filter(lp_hz)
    return out


def gentle_compress(
    seg: AudioSegment,
    *,
    threshold: float = -22.0,
    ratio: float = 2.0,
    attack: int = 5,
    release: int = 50,
) -> AudioSegment:
    """
    Compression helps bring up quieter phonemes/digits without boosting noise as much as normalize().
    """
    return seg.compress_dynamic_range(
        threshold=threshold,
        ratio=ratio,
        attack=attack,
        release=release,
    )





def process_mono_for_asr(
    seg: AudioSegment,
    *,
    frame_rate: int = 16000,
    do_bandpass: bool = True,
    hp_hz: int = 100,
    lp_hz: int = 7800,
    do_compress: bool = True,
    compress_threshold: float = -22.0,
    compress_ratio: float = 2.0,
    compress_attack: int = 5,
    compress_release: int = 50,
    do_peak_normalize: bool = True,
    target_peak_dbfs: float = -1.0,
    max_gain_db: float = 8.0,
    if_clipped_reduce_db: float = 3.0,
) -> AudioSegment:
    """
    Accuracy-first conditioning pipeline (in-memory):
    - mono, 16k
    - bandpass for phone speech
    - if clipped: reduce a bit (don’t normalize up)
    - gentle compression
    - safe peak normalize to -1 dBFS (clamped gain)
    """
    out = seg.set_channels(1).set_frame_rate(frame_rate)

    if do_bandpass:
        out = apply_phone_bandpass(out, hp_hz=hp_hz, lp_hz=lp_hz)

    clip = detect_clipping(out)
    if clip.is_clipped and if_clipped_reduce_db > 0:
        out = out.apply_gain(-abs(if_clipped_reduce_db))

    if do_compress:
        out = gentle_compress(
            out,
            threshold=compress_threshold,
            ratio=compress_ratio,
            attack=compress_attack,
            release=compress_release,
        )

    if do_peak_normalize:
        out = peak_normalize_safe(out, target_peak_dbfs=target_peak_dbfs, max_gain_db=max_gain_db)

    return out
