from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, BinaryIO, Union

import os
import shutil
import json

from pydub import AudioSegment

from core.config import settings
from core.logger import log

from openai_tools.openai_client_transcribe import (
    async_transcribe_audio,
    async_transcribe_audio_diarized,
)

from transcribe.utilities.audio_tools import split_stereo_to_lr_and_clean_lr
from transcribe.utilities.scenario_tools import async_classify_agent_or_client_prefix, Turn

AudioInput = Union[str, Path, BinaryIO]



# =========================
# Formatting helpers
# =========================

def _fmt_ts(sec: float) -> str:
    """Format seconds as MM:SS.mmm."""
    ms_total = int(round(sec * 1000))
    mm, rem = divmod(ms_total, 60_000)
    ss, ms = divmod(rem, 1000)
    return f"{mm:02d}:{ss:02d}.{ms:03d}"


def format_mmss_ms(sec: float) -> str:
    """Backwards-compatible alias used by older code."""
    return _fmt_ts(sec)


def render_timestamped_script(turns: Sequence[Turn], timestamp_on: bool = True) -> str:
    lines: List[str] = []
    for t in turns:
        if timestamp_on:
            lines.append(f"[{_fmt_ts(t.start)}–{_fmt_ts(t.end)}] {t.role}: {t.text}")
        else:
            lines.append(f"{t.role}: {t.text}")
    return "\n".join(lines)

def render_timestamped_script_o4(turns: Sequence[Turn]) -> str:
    lines: List[str] = []
    for t in turns:
        lines.append(f"[{_fmt_ts(t.start)}–{_fmt_ts(t.end)}]: {t.text}")
        # lines.append(f"{t.role}: {t.text}")
    return "\n".join(lines)

def merge_adjacent_turns(turns: List[Turn], *, gap_sec: float = 0.8) -> List[Turn]:
    """
    Merge adjacent turns if:
      - same role
      - gap between them <= gap_sec
    Keeps start/end timestamps spanning the merged turns.
    """
    if not turns:
        return []

    turns_sorted = sorted(turns, key=lambda x: (x.start, x.end))
    merged: List[Turn] = [turns_sorted[0]]

    for t in turns_sorted[1:]:
        last = merged[-1]
        gap = t.start - last.end
        if t.role == last.role and gap <= gap_sec:
            text = (last.text.rstrip() + " " + t.text.lstrip()).strip()
            merged[-1] = Turn(role=last.role, start=last.start, end=max(last.end, t.end), text=text)
        else:
            merged.append(t)

    return merged


# =========================
# Audio slicing primitives
# =========================

def slice_wav(in_wav: str, out_wav: str, start_s: float, end_s: float, pad_ms: int = 250) -> str:
    audio = AudioSegment.from_file(in_wav)
    start_ms = max(0, int(start_s * 1000) - pad_ms)
    end_ms = min(len(audio), int(end_s * 1000) + pad_ms)
    chunk = audio[start_ms:end_ms]
    chunk.export(out_wav, format="wav")
    return out_wav


def extract_diarize_bounds(diarized: Any) -> List[Tuple[float, float]]:
    """Extract (start,end) from diarized result."""
    segs = getattr(diarized, "segments", None) or (diarized.get("segments", []) if isinstance(diarized, dict) else [])
    bounds: List[Tuple[float, float]] = []
    for s in segs:
        st = getattr(s, "start", None) if not isinstance(s, dict) else s.get("start")
        en = getattr(s, "end", None) if not isinstance(s, dict) else s.get("end")
        if st is None or en is None:
            continue
        st_f = float(st)
        en_f = float(en)
        if en_f > st_f:
            bounds.append((st_f, en_f))
    bounds.sort(key=lambda x: x[0])
    return bounds


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    for m in ("model_dump", "dict", "to_dict"):
        if hasattr(obj, m):
            try:
                return getattr(obj, m)()
            except Exception:
                pass
    # last resort: try vars()
    try:
        return vars(obj)
    except Exception:
        return {}

def _extract_segments(obj: Any) -> List[Any]:
    """
    Tries to pull segment list from multiple possible response shapes.
    Works for both json and diarized_json-like responses.
    """
    if obj is None:
        return []
    # pydantic-like / class-like
    segs = getattr(obj, "segments", None)
    if isinstance(segs, list):
        return segs
    d = _as_dict(obj)
    segs = d.get("segments")
    if isinstance(segs, list):
        return segs

    # Some APIs nest under "output" or similar
    out = d.get("output")
    if isinstance(out, dict) and isinstance(out.get("segments"), list):
        return out["segments"]

    return []




async def async_o4_bounds_from_o4_words(
    wav_path: str,
    language: str = "uk",
    temperature: float = 0.0,
    *,
    # segment cleanup (these are SEGMENTS, not words)
    min_seg_s: float = 0.10,
    max_seg_s: float = 300.0,

    # phrase building / merging
    join_gap_s: float = 0.65,         # merge if gap <= this
    hard_break_gap_s: float = 1.10,   # force break if gap >= this
    break_on_punct: bool = False,     # usually unnecessary for diarized segments

    # phrase constraints
    min_phrase_s: float = 0.30,
    max_phrase_s: float = 15.0,

) -> List[Tuple[float, float]]:
    """
    Build slicing bounds from O4 diarized SEGMENTS (not word timestamps).

    Output bounds are "tight" (no padding). All padding is handled later in slice_wav(... pad_ms=...).
    """
    res = await async_transcribe_audio_diarized(
        audio=wav_path,
        language=language,
        temperature=temperature,
    )

    segs = _extract_segments(res)

    script = render_timestamped_script_o4(segs)
    log.info(f"\n=== O4 diarized segments ===\n{str(script)}")

    def s_get(s: Any, k: str) -> Any:
        return s.get(k) if isinstance(s, dict) else getattr(s, k, None)

    # normalize + clean segments
    cleaned: List[Tuple[float, float, str]] = []
    for s in segs:
        st = s_get(s, "start")
        en = s_get(s, "end")
        txt = (s_get(s, "text") or "").strip()

        if st is None or en is None:
            continue
        st_f = float(st)
        en_f = float(en)
        if en_f <= st_f:
            continue

        dur = en_f - st_f
        if dur < min_seg_s or dur > max_seg_s:
            continue
        if not txt:
            continue

        cleaned.append((st_f, en_f, txt))

    if not cleaned:
        return []

    cleaned.sort(key=lambda x: (x[0], x[1]))

    STRONG_PUNCT = {".", "!", "?", "…"}

    def ends_with_strong_punct(text: str) -> bool:
        t = (text or "").strip()
        return bool(t) and t[-1] in STRONG_PUNCT

    # merge into phrases (tight unions, no padding)
    bounds: List[Tuple[float, float]] = []

    cur_st, cur_en, cur_txt = cleaned[0]
    for st, en, txt in cleaned[1:]:
        gap = st - cur_en

        # force break on huge gap
        if gap >= hard_break_gap_s:
            bounds.append((cur_st, cur_en))
            cur_st, cur_en, cur_txt = st, en, txt
            continue

        # optional punctuation break (usually off for diarized segments)
        if break_on_punct and ends_with_strong_punct(cur_txt):
            bounds.append((cur_st, cur_en))
            cur_st, cur_en, cur_txt = st, en, txt
            continue

        # join if close enough
        if gap <= join_gap_s:
            cur_en = max(cur_en, en)
            cur_txt = txt

            # cap phrase length
            if (cur_en - cur_st) >= max_phrase_s:
                bounds.append((cur_st, cur_en))
                cur_st, cur_en, cur_txt = st, en, txt
            continue

        # otherwise new phrase
        bounds.append((cur_st, cur_en))
        cur_st, cur_en, cur_txt = st, en, txt

    bounds.append((cur_st, cur_en))

    # filter very short phrases (still tight)
    out = [(st, en) for st, en in bounds if (en - st) >= min_phrase_s]

    
    bounds_info = [f"\no4_bounds_from_o4_words: segs={len(segs)} cleaned={len(cleaned)} bounds={len(out)}"]
    for st, en in out[:20]:
        bounds_info.append(f"  {st:.2f}–{en:.2f} ({en - st:.2f}s)")
    log.info("\n".join(bounds_info))        

    
    return out




def extend_chunk_ends(
    bounds: List[Tuple[float, float]],
    *,
    extend_s: float = 4.0,
    pad_s: float = 0.25,      # same pad you use in slice_wav (converted from ms)
    safety_s: float = 0.5,    # your "-0.5 sec"
    max_total_s: Optional[float] = None,  # e.g., wav duration, if available
) -> List[Tuple[float, float]]:
    """
    Extend each chunk end by extend_s, but never beyond (next_start - pad_s - safety_s).
    Keeps starts unchanged. Guaranteed non-overlap with next chunk's padded region.
    """
    out: List[Tuple[float, float]] = []
    n = len(bounds)

    for i, (st, en) in enumerate(bounds):
        # Default: extend by extend_s
        target_end = en + extend_s

        # Cap by next chunk start (if exists)
        if i + 1 < n:
            next_start = bounds[i + 1][0]
            cap = next_start - pad_s - safety_s
            if cap > en:
                target_end = min(target_end, cap)
            else:
                target_end = en  # cannot extend safely

        # Optional cap by total duration
        if max_total_s is not None:
            target_end = min(target_end, max_total_s)

        out.append((st, target_end))

    return out


async def async_transcribe_channel_by_bounds(
    channel_wav: str,
    role: str,
    language: str = "uk",
    chunk_pad_ms: int = 250,
    min_seg_s: float = 0.25,
    metadata: Any = None
) -> List[Turn]:

    # Tight bounds (these are the timestamps you want in the final output)
    bounds = await async_o4_bounds_from_o4_words(channel_wav)

    # Extended bounds only for slicing audio (extra context to avoid cut sentences)
    pad_s = chunk_pad_ms / 1000.0
    bounds_ext = extend_chunk_ends(bounds, extend_s=3.0, pad_s=pad_s, safety_s=1)

    # Log first N for sanity
    bounds_info = ["bounds (orig -> ext)"]
    for (st, en), (st2, en2) in list(zip(bounds, bounds_ext))[:20]:
        bounds_info.append(
            f"  {st:.2f}–{en:.2f} ({en-st:.2f}s)  ->  {st2:.2f}–{en2:.2f} ({en2-st2:.2f}s)"
        )
    log.info("\n".join(bounds_info))

    out: List[Turn] = []
    prev_context = ""
    base = os.path.splitext(channel_wav)[0]

    # IMPORTANT: output timestamps come from `bounds`, slicing comes from `bounds_ext`
    for i, ((st, en), (st_ext, en_ext)) in enumerate(zip(bounds, bounds_ext)):
        if (en - st) < min_seg_s:
            continue

        day_dir = Path(settings.TR_TEMP_ROOT_DIR) / datetime.now().strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(base).stem
        chunk_path = str(day_dir / f"{base_name}_seg_{i:04d}.wav")

        # Slice with extended bounds (extra context)
        slice_wav(channel_wav, chunk_path, st_ext, en_ext, pad_ms=chunk_pad_ms)
        meta = metadata if isinstance(metadata, str) else (json.dumps(metadata, ensure_ascii=False) if metadata else "")
        ctx = prev_context[-150:]  # small window, not 1500 chars

        prompt = (
            "Transcribe this call slice verbatim in the original language.\n"
            "Include EVERY spoken word (even repeats, fillers, stutters, corrections, partial words).\n"
            "Do NOT paraphrase or summarize. Preserve numbers, names, banking terms.\n"
            "Output ONLY the transcript text (no tags, no punctuation rules beyond what you hear).\n"
            "If no voice is present, output an empty string.\n"
            f"Known entities canonical names: {meta}"
        )
        # if prev_context:
        #     prompt += f"Previous context (for continuity; it may repeat naturally): {prev_context}\n"

        tr = await async_transcribe_audio(
            audio=chunk_path,
            model="gpt-4o-transcribe",
            language=language,
            temperature=0.0,
            response_format="json",
            prompt=prompt,
        )
        log.info(tr.text)
        text = (getattr(tr, "text", None) or (tr.get("text") if isinstance(tr, dict) else "") or "").strip()
        if text:
            # Return ORIGINAL timestamps (tight)
            out.append(Turn(role=role, start=st, end=en, text=text))
            prev_context = (prev_context + " " + text).strip()

    return out


async def async_transcribe_stereo_high_accuracy_with_timestamps(
    left_wav: str,
    right_wav: str,
    language: str = "uk",
    chunk_pad_ms: int = 250,
    min_seg_s: float = 0.25,
    merge_same_role_gap_s: float = 0.8,
    metadata: Any = None
) -> Tuple[List[Turn], str]:
    """
    Transcribe two mono channels (L=AG, R=CL), return (turns, script).
    bounds_method controls how speech segments are detected (diarize/VAD/whisper-segments).
    """

    a_turns = await async_transcribe_channel_by_bounds(
        channel_wav=left_wav,
        role="A",
        language=language,
        chunk_pad_ms=chunk_pad_ms,
        min_seg_s=min_seg_s,
        metadata=metadata
    )

    b_turns = await async_transcribe_channel_by_bounds(
        right_wav,
        role="B",
        language=language,
        chunk_pad_ms=chunk_pad_ms,
        min_seg_s=min_seg_s,
        metadata=metadata
    )
    
    a_text = render_timestamped_script(a_turns, timestamp_on=False)
    a_role_res = await async_classify_agent_or_client_prefix(a_text)
    log.info(f"\n=== Detected roles ===\n{a_role_res}") 
    if a_role_res == "AGENT":
        a_role, b_role = "AG", "CL"
    elif a_role_res:
        a_role, b_role = "CL", "AG"

    a_turns = [Turn(role=a_role, start=t.start, end=t.end, text=t.text) for t in a_turns]
    b_turns = [Turn(role=b_role, start=t.start, end=t.end, text=t.text) for t in b_turns]
    a_text = render_timestamped_script(a_turns)
    b_text = render_timestamped_script(b_turns)
    log.info(f"\n=== AG script ===\n{a_text}")
    log.info(f"\n=== CL script ===\n{b_text}")   

    turns = sorted(a_turns + b_turns, key=lambda x: (x.start, x.end))
    turns = merge_adjacent_turns(turns, gap_sec=merge_same_role_gap_s)
    script = render_timestamped_script(turns)

    log.info(f"\n=== Final script ===\n{script}")

    return turns, script


async def async_transcribe_stereo_lr_timestamped(
    *,
    source_file: str,
    temp_root_dir: Optional[str] = None,
    language: str = "uk",
    metadata: Any = None
) -> Tuple[List[Turn], str]:
    """
    Stereo -> (turns, script) with timestamps.
    AG = left channel, CL = right channel.
    """
    if not temp_root_dir:
        temp_root_dir = settings.TR_TEMP_ROOT_DIR

    # 1) copy to dated temp dir (preserve timeline)
    day_dir = Path(temp_root_dir) / datetime.now().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    src = Path(source_file)
    dst = day_dir / src.name
    shutil.copy2(src, dst)

    # 2) split + clean (normalize + 16k + NR) WITHOUT cutting silences
    left_wav, right_wav = split_stereo_to_lr_and_clean_lr(str(dst))
    # left_wav, right_wav = process_stereo_recording2(str(dst))

    turns, script = await async_transcribe_stereo_high_accuracy_with_timestamps(
        left_wav,
        right_wav,
        language=language,
        chunk_pad_ms=200,   
        metadata=metadata,
        )

    return turns, script
