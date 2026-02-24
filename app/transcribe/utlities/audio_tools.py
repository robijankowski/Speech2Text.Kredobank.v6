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
    left_file = f"{base_name}_lch.wav"
    right_file = f"{base_name}_rch.wav"

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




def split_stereo_to_lr_and_clean_lr(stereo_file_path: str):
    # ASR_AUDIO_PRESET = ASR_AUDIO_PRESET_HARD
    log.info(f"Processing stereo file: {stereo_file_path}")
    stereo_audio = AudioSegment.from_wav(stereo_file_path)
    log.info(f"Loaded stereo audio: {stereo_file_path}, duration: {len(stereo_audio)/1000:.2f}s, channels: {stereo_audio.channels}, frame_rate: {stereo_audio.frame_rate}Hz")
    
    if stereo_audio.channels != 2:
        log.warning(f"Warning: {stereo_file_path} is not stereo (has {stereo_audio.channels} channels)")
        return None, None
    
    log.info(f"Splitting stereo into left and right channels: {stereo_file_path}")
    left_raw, right_raw = stereo_audio.split_to_mono()
    log.info(f"Left channel: duration {len(left_raw)/1000:.2f}s, frame_rate: {left_raw.frame_rate}Hz")
    log.info(f"Right channel: duration {len(right_raw)/1000:.2f}s, frame_rate: {right_raw.frame_rate}Hz")

    log.info(f"Processing left channel for ASR: {stereo_file_path}")
    left = process_mono_for_asr(
        left_raw,
        frame_rate=16000,
        hp_hz=ASR_AUDIO_PRESET["hp_hz"],
        lp_hz=ASR_AUDIO_PRESET["lp_hz"],
        compress_threshold=ASR_AUDIO_PRESET["compress_threshold"],
        compress_ratio=ASR_AUDIO_PRESET["compress_ratio"],
        target_peak_dbfs=ASR_AUDIO_PRESET["target_peak_dbfs"],
        max_gain_db=ASR_AUDIO_PRESET["max_gain_db"],
    )
    log.info(f"Left channel processed for ASR:{stereo_file_path}")
    
    log.info(f"Processing right channel for ASR: {stereo_file_path}")
    right = process_mono_for_asr(
        right_raw,
        frame_rate=16000,
        hp_hz=ASR_AUDIO_PRESET["hp_hz"],
        lp_hz=ASR_AUDIO_PRESET["lp_hz"],
        compress_threshold=ASR_AUDIO_PRESET["compress_threshold"],
        compress_ratio=ASR_AUDIO_PRESET["compress_ratio"],
        target_peak_dbfs=ASR_AUDIO_PRESET["target_peak_dbfs"],
        max_gain_db=ASR_AUDIO_PRESET["max_gain_db"],
    )
    log.info(f"Right channel processed for ASR:{stereo_file_path}")

    log.info(f"Applying crosstalk suppression {stereo_file_path}")
    # Crosstalk suppression (this is where you tune duck_db / dominance_db)
    left, right = duck_crosstalk_lr(
        left,
        right,
        frame_ms=20,
        activity_dbfs=ASR_AUDIO_PRESET["activity_dbfs"],
        dominance_db=ASR_AUDIO_PRESET["dominance_db"],
        duck_db=ASR_AUDIO_PRESET["duck_db"],
    )
    log.info(f"Applied crosstalk suppression to left abd right channels: {stereo_file_path}")

    base_name = os.path.splitext(stereo_file_path)[0]
    left_file = f"{base_name}_lch.wav"
    right_file = f"{base_name}_rch.wav"

    left.export(left_file, format="wav")
    right.export(right_file, format="wav")

    # OPTIONAL NR: keep it gentle; consider skipping if it harms clarity
    log.info(f"Applying noise reduction (gentle) to left channel {left_file}")
    left_file = reduce_audio_noise(left_file, method="gentle")
    log.info(f"Applied noise reduction to left channel, saved to {left_file}")
    
    log.info(f"Applying noise reduction (gentle) to right channel {right_file}")
    right_file = reduce_audio_noise(right_file, method="gentle")
    log.info(f"Applied noise reduction to right channel, saved to {right_file}")

    log.info(f"\nCreated mono files: {left_file}, {right_file}\nfrom stereo: {stereo_file_path}")
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
    output_file = f"{base_name}_orgch.wav"

    audio.export(output_file, format="wav")
    
    output_file = reduce_audio_noise(output_file, method="gentle")

    log.info(f"Created: {output_file}")
    return output_file


def clean_audio_file_lr(input_file: str) -> str:
    log.info(f"Cleaning audio file: {input_file}")
    audio = AudioSegment.from_file(input_file)
    log.info(f"Loaded audio: {input_file}, duration: {len(audio)/1000:.2f}s, channels: {audio.channels}, frame_rate: {audio.frame_rate}Hz")

    log.info(f"Normalizing audio mono {input_file}")
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

    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_orgch.wav"
    audio.export(output_file, format="wav")
    log.info(f"Audio mono processed for ASR: {output_file}")

    log.info(f"Applying noise reduction (gentle) to cleaned audio {output_file}")
    output_file = reduce_audio_noise(output_file, method="gentle")
    log.info(f"Applied noise reduction to cleaned audio, saved to {output_file}")
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





def _apply_gain_to_frames(
    x: np.ndarray,
    *,
    frame_len: int,
    gains: np.ndarray,
) -> np.ndarray:
    """
    Apply per-frame linear gain to samples.
    gains: shape (num_frames,), linear multipliers.
    """
    if x.size == 0:
        return x

    dtype = x.dtype
    info = np.iinfo(dtype)
    out = x.astype(np.float32, copy=True)

    n = x.size
    num_frames = int(math.ceil(n / frame_len))
    gains = gains[:num_frames]

    for i in range(num_frames):
        st = i * frame_len
        en = min(n, st + frame_len)
        out[st:en] *= float(gains[i])

    out = np.clip(out, info.min, info.max).astype(dtype)
    return out


def duck_crosstalk_lr(
    left: AudioSegment,
    right: AudioSegment,
    *,
    frame_ms: int = 20,
    activity_dbfs: float = -35.0,
    dominance_db: float = 8.0,
    duck_db: float = 12.0,
) -> Tuple[AudioSegment, AudioSegment]:
    """
    Crosstalk suppression via stereo 'ducking':
    - If Right is active and dominates Left by dominance_db, attenuate Left by duck_db (per-frame).
    - Symmetric for Right.

    This reduces 'ghost words' leaking into the other channel, improving per-speaker accuracy.
    """
    # Ensure same rate/width
    if left.frame_rate != right.frame_rate:
        right = right.set_frame_rate(left.frame_rate)
    if left.sample_width != right.sample_width:
        # pydub typically keeps same; if not, standardize by re-spawning (rare)
        right = right.set_sample_width(left.sample_width)

    # Mono only
    if left.channels != 1:
        left = left.set_channels(1)
    if right.channels != 1:
        right = right.set_channels(1)

    xL = _np_samples(left)
    xR = _np_samples(right)

    sr = left.frame_rate
    frame_len = max(1, int(sr * frame_ms / 1000))

    def frame_dbfs(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.array([], dtype=np.float32)
        dtype = x.dtype
        info = np.iinfo(dtype)
        full_scale = float(max(abs(info.min), info.max))

        n = x.size
        num_frames = int(math.ceil(n / frame_len))
        out = np.empty(num_frames, dtype=np.float32)

        for i in range(num_frames):
            st = i * frame_len
            en = min(n, st + frame_len)
            frame = x[st:en].astype(np.float32)
            rms = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
            if rms <= 1e-9:
                out[i] = -120.0
            else:
                out[i] = 20.0 * math.log10(rms / full_scale + 1e-12)
        return out

    dL = frame_dbfs(xL)
    dR = frame_dbfs(xR)
    nf = min(dL.size, dR.size)
    dL = dL[:nf]
    dR = dR[:nf]

    # Activity + dominance masks
    L_active = dL >= activity_dbfs
    R_active = dR >= activity_dbfs

    # If R dominates, duck L; if L dominates, duck R
    duck_L = R_active & ((dR - dL) >= dominance_db)
    duck_R = L_active & ((dL - dR) >= dominance_db)

    # Convert duck_db to linear gain
    duck_gain = 10.0 ** (-duck_db / 20.0)

    gains_L = np.ones(nf, dtype=np.float32)
    gains_R = np.ones(nf, dtype=np.float32)
    gains_L[duck_L] = duck_gain
    gains_R[duck_R] = duck_gain

    yL = _apply_gain_to_frames(xL, frame_len=frame_len, gains=gains_L)
    yR = _apply_gain_to_frames(xR, frame_len=frame_len, gains=gains_R)

    left2 = left._spawn(yL.tobytes())
    right2 = right._spawn(yR.tobytes())
    return left2, right2


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
