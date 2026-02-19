from typing import Optional, Tuple
import os
from pydub import AudioSegment
from core.logger import log


def _bandpass_phone(seg: AudioSegment, hp_hz: int = 100, lp_hz: int = 7800) -> AudioSegment:
    # pydub filters keep length/timeline intact
    return seg.high_pass_filter(hp_hz).low_pass_filter(lp_hz)


def _peak_normalize_safe(seg: AudioSegment, target_peak_dbfs: float = -3.0, max_gain_db: float = 4.0) -> AudioSegment:
    """
    Move peak toward target, but clamp *amplification* to max_gain_db.
    If audio is too loud, we attenuate freely (negative gain) to reach target.
    """
    peak = seg.max_dBFS  # e.g. -10.2
    if peak == float("-inf"):
        return seg  # silence / empty

    gain = target_peak_dbfs - peak
    if gain > max_gain_db:
        gain = max_gain_db
    return seg.apply_gain(gain)


def split_stereo_to_lr_for_segments_lr(
    in_audio_path: str,
    *,
    out_dir: Optional[str] = None,
    frame_rate: int = 16000,
    hp_hz: int = 100,
    lp_hz: int = 7800,
    target_peak_dbfs: float = -3.0,
    max_gain_db: float = 4.0,
) -> Tuple[str, str]:
    """
    Create two processed mono files (left/right) for speech-segmentation / diarize timestamping.

    If input is mono, produces TWO identical outputs (L and R) and logs a warning.
    Returns: (left_path, right_path)
    """
    audio = AudioSegment.from_file(in_audio_path)
    base = os.path.splitext(os.path.basename(in_audio_path))[0]
    out_dir = out_dir or os.path.dirname(in_audio_path) or "."

    left_path = os.path.join(out_dir, f"{base}_lc_seg.wav")
    right_path = os.path.join(out_dir, f"{base}_rc_seg.wav")

    if audio.channels == 2:
        left_raw, right_raw = audio.split_to_mono()
    else:
        log.warning(f"{in_audio_path} is not stereo (channels={audio.channels}). Duplicating mono to L/R outputs.")
        mono = audio.set_channels(1)
        left_raw, right_raw = mono, mono

    def prep(m: AudioSegment) -> AudioSegment:
        m = m.set_channels(1).set_frame_rate(frame_rate)
        m = _bandpass_phone(m, hp_hz=hp_hz, lp_hz=lp_hz)
        m = _peak_normalize_safe(m, target_peak_dbfs=target_peak_dbfs, max_gain_db=max_gain_db)
        return m

    left = prep(left_raw)
    right = prep(right_raw)

    left.export(left_path, format="wav")
    right.export(right_path, format="wav")

    log.info(f"Prepared segments-only L/R: {left_path} | {right_path}")
    return left_path, right_path
