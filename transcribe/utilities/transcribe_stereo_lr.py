from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, BinaryIO, Union, Tuple

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
from transcribe.utilities.audio_tools import split_stereo_to_lr_and_clean
from transcribe.utilities.scenario_tools import (async_classify_agent_or_client_prefix, 
                                                 Turn, 
                                                 DiarSeg,
                                                 render_timestamped_script_from_turns,
                                                 render_timestamped_script_from_diar_segs,
                                                 render_turns_tight_vs_ext
                                                 )
from transcribe.utilities.audio_tools_lr import split_stereo_to_lr_for_segments_lr

AudioInput = Union[str, Path, BinaryIO]





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




async def async_gen_o4_diar_segments(
    *,
    wav_seg_path: str,
    language: str = "uk",
    temperature: float = 0.0,
    min_seg_s: float = 0.10,
    max_seg_s: float = 300.0,
) -> List[DiarSeg]:
    """
    Run O4 diarize on wav_seg_path (conditioning-safe audio) to obtain diarized segments.
    Returns cleaned segments: (start, end, text, speaker), sorted by time.
    """
    res = await async_transcribe_audio_diarized(
        audio=wav_seg_path,
        language=language,
        temperature=temperature,
    )

    segs = _extract_segments(res)

    script = render_timestamped_script_from_diar_segs(segs)
    log.info(f"\n=== O4 diarized segments ===\n{str(script)}")

    def s_get(s: Any, k: str) -> Any:
        return s.get(k) if isinstance(s, dict) else getattr(s, k, None)

    cleaned: List[DiarSeg] = []
    for s in segs:
        st = s_get(s, "start")
        en = s_get(s, "end")
        txt = (s_get(s, "text") or "").strip()
        spk = (
            s_get(s, "speaker")
            or s_get(s, "speaker_id")
            or s_get(s, "spk")
            or "S0"
        )
        spk = str(spk).strip() if spk is not None else "S0"

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

        cleaned.append((st_f, en_f, txt, spk))

    cleaned.sort(key=lambda x: (x[0], x[1]))
    return cleaned


def build_turns_from_diar_segments(
    cleaned: List[DiarSeg],
    *,
    wav_path: str,
    join_gap_s: float = 0.65,
    hard_break_gap_s: float = 1.10,
    break_on_punct: bool = False,
    min_phrase_s: float = 0.30,
    max_phrase_s: float = 15.0,
) -> List[Turn]:
    """
    Merge cleaned diarized segments into phrase-level Turns.
    - Break on speaker change (keeps role meaningful)
    - Break on hard gaps
    - Optionally break on strong punctuation
    - Join close segments
    - Enforce max_phrase_s without duplicating a segment
    """
    if not cleaned:
        return []

    STRONG_PUNCT = {".", "!", "?", "…"}

    def ends_with_strong_punct(text: str) -> bool:
        t = (text or "").strip()
        return bool(t) and t[-1] in STRONG_PUNCT

    turns: List[Turn] = []

    cur_st, cur_en, cur_txt, cur_spk = cleaned[0]
    cur_parts: List[str] = [cur_txt]
    cur_last_txt: str = cur_txt

    def flush():
        phrase_text = " ".join(p for p in cur_parts if p).strip()
        if phrase_text and (cur_en - cur_st) >= min_phrase_s:
            turns.append(
                Turn(
                    role=cur_spk,
                    start=cur_st,
                    end=cur_en,
                    text=phrase_text,         # diar text (cheap transcript)
                    text_diar=phrase_text,
                    file=wav_path,
                )
            )

    for st, en, txt, spk in cleaned[1:]:
        gap = st - cur_en

        # speaker change -> flush, start new
        if spk != cur_spk:
            flush()
            cur_st, cur_en, cur_spk = st, en, spk
            cur_parts = [txt]
            cur_last_txt = txt
            continue

        # hard break gap
        if gap >= hard_break_gap_s:
            flush()
            cur_st, cur_en, cur_spk = st, en, spk
            cur_parts = [txt]
            cur_last_txt = txt
            continue

        # optional punctuation break
        if break_on_punct and ends_with_strong_punct(cur_last_txt):
            flush()
            cur_st, cur_en, cur_spk = st, en, spk
            cur_parts = [txt]
            cur_last_txt = txt
            continue

        # join if close enough
        if gap <= join_gap_s:
            prospective_end = max(cur_en, en)

            # cap BEFORE adding -> avoids duplicating current segment in two phrases
            if (prospective_end - cur_st) >= max_phrase_s:
                flush()
                cur_st, cur_en, cur_spk = st, en, spk
                cur_parts = [txt]
                cur_last_txt = txt
                continue

            cur_en = prospective_end
            cur_parts.append(txt)
            cur_last_txt = txt
            continue

        # otherwise new phrase
        flush()
        cur_st, cur_en, cur_spk = st, en, spk
        cur_parts = [txt]
        cur_last_txt = txt

    # final flush
    flush()

    return turns




def extend_chunk_ends_turns(
    turns: List[Turn],
    extend_s: float = 4.0,
    pad_s: float = 0.25,      # same pad you use in slice_wav (converted from ms)
    safety_s: float = 0.5,    # your "-0.5 sec"
    max_total_s: Optional[float] = None,  # e.g., wav duration, if available
) -> List[Turn]:
    """
    Populate start_ext/end_ext for each Turn:
      - start_ext = start (unchanged)
      - end_ext   = min(end + extend_s, next_start - pad_s - safety_s, max_total_s)

    Tight timestamps remain in start/end. Extended timestamps are for slicing only.
    Guaranteed non-overlap with next chunk's padded region.
    """
    if not turns:
        return []

    turns_sorted = sorted(turns, key=lambda t: (t.start, t.end))

    out: List[Turn] = []
    n = len(turns_sorted)

    for i, t in enumerate(turns_sorted):
        st = t.start
        en = t.end

        # Default: extend by extend_s
        target_end_ext = en + extend_s

        # Cap by next chunk start (if exists)
        if i + 1 < n:
            next_start = turns_sorted[i + 1].start
            cap = next_start - pad_s - safety_s
            if cap > en:
                target_end_ext = min(target_end_ext, cap)
            else:
                target_end_ext = en  # cannot extend safely

        # Optional cap by total duration
        if max_total_s is not None:
            target_end_ext = min(target_end_ext, max_total_s)

        # start_ext stays tight start (this function extends only end)
        start_ext = st
        end_ext = target_end_ext

        out.append(
            Turn(
                role=t.role,
                start=st,
                end=en,
                text=t.text,
                text_diar=t.text_diar,
                file=t.file,
                start_ext=start_ext,
                end_ext=end_ext,
            )
        )

    return out




async def async_transcribe_channel_by_turns(
    turns: List[Turn],          # Turns with tight start/end, and populated start_ext/end_ext for slicing
    language: str = "uk",
    chunk_pad_ms: int = 250,
    min_seg_s: float = 0.25,
    metadata: Any = None,
) -> List[Turn]:
    """
    IMPORTANT:
    - output timestamps come from Turn.start/Turn.end (tight)
    - slicing uses Turn.start_ext/Turn.end_ext (extended)
    - wav filename is taken from Turn.file
    - preserves your prompt/context/metadata logic
    """
    out: List[Turn] = []
    prev_context = ""

    if not turns:
        return []

    # deterministic order
    turns_sorted = sorted(turns, key=lambda t: (t.start, t.end))

    day_dir = Path(settings.TR_TEMP_ROOT_DIR) / datetime.now().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    
    meta = metadata if isinstance(metadata, str) else (json.dumps(metadata, ensure_ascii=False) if metadata else "")

    for i, t in enumerate(turns_sorted):
        st, en = t.start, t.end
        st_ext, en_ext = t.start_ext, t.end_ext

        if (en - st) < min_seg_s:
            continue

        channel_wav = t.file.strip()
        if not channel_wav:
            raise ValueError("Turn.file is empty; cannot slice audio.")
        base_name = Path(channel_wav).stem
        chunk_path = str(day_dir / f"{base_name}_seg_{i:04d}.wav")

        # Slice with extended bounds (extra context)
        slice_wav(channel_wav, chunk_path, st_ext, en_ext, pad_ms=chunk_pad_ms)

        ctx = (prev_context or "")[-250:].strip()

        # prompt = (
        #     f"This is part of conversation between Bank AGENT and Bank CLIENT."
        #     "Transcribe verbatim. Keep original spoken language (UA/RU mixed).\n"
        #     "Do not paraphrase. Keep names, amounts, dates exactly. Output only transcript text.\n"
        #     "If word is unclear make a guess and new one with similiar sound/tone\n"
        #     f"Known entities canonical names: {meta}\n"
        # )
        # if prev_context:
        #     prompt += f"Previous context (do not repeat): {ctx}\n"
        # prompt += "Recognize what {t.role} says."

        # prompt = (
        #     f"This is part of conversation between Bank AGENT and Bank CLIENT."
        #     "Transcribe verbatim. Keep original spoken language (UA/RU mixed).\n"
        #     "Do not paraphrase. Keep names, amounts, dates exactly. Output only transcript text.\n"
        #     f"Known entities canonical names: {meta}\n"
        # )
        # if ctx:
        #     prompt += f"Previous context (do not repeat): {ctx}\n"
        # prompt += f"Recognize what {t.role} says."

        # prompt = (
        #     f"This is part of conversation between Bank AGENT and Bank CLIENT."
        #     "Transcribe verbatim. Keep original spoken language (UA/RU mixed).\n"
        #     "Do not paraphrase. Keep names, amounts, dates exactly. Output only transcript text.\n"
        #     f"Known entities canonical names: {meta}\n"
        # )
        # if ctx:
        #     prompt += f"Previous context (do not repeat): {ctx}\n"
        # if t.text_diar:
        #     prompt += f"You can use this text as clue when transcribing: '{t.text_diar}'\n"
        # prompt += f"Recognize very carefullu what {t.role} says."


        # prompt = (
        #     "Transcribe THIS audio slice only. Output ONLY transcript text.\n"
        #     "Keep UA/RU as spoken. Verbatim (repeats/fillers/stutters). No paraphrase/translation.\n"
        #     "Keep names/amounts/dates exactly. If NO speech (noise/silence) -> output EMPTY.\n"
        #     "Never copy from context/draft unless heard in audio.\n"
        #     "If there is no speach or you can not recognize any word then return empty string"
        #     f"Known entities canonical names: <META>{meta}</META>\n"
        # )

        # if ctx:
        #     prompt += f"Context (continuity only, don't repeat): <CONTEXT>{ctx}</CONTEXT>\n"

        # if t.text_diar:
        #     prompt += (
        #         "Use your transcription and DRAFT. Select better in context. "
        #         f"<DRAFT>{t.text_diar}</DRAFT>\n"
        #     )

        # prompt += f"Speaker: {t.role}\n"


        prompt = (
            "Transcribe THIS audio slice only. Output ONLY transcript text.\n"
            "First decide: is there CLEAR speech? If not (noise/silence/too unclear) output EMPTY string.\n"
            "Keep UA/RU as spoken. Verbatim (repeats/fillers/stutters). No paraphrase/translation.\n"
            "Keep names/amounts/dates exactly.\n"
            "Never invent. Never copy from context/draft unless you clearly hear it in the audio.\n"
            f"Known entities: <META>{meta}</META>\n"
        )

        if ctx:
            prompt += f"Context (continuity only; do NOT repeat/copy): <CONTEXT>{ctx}</CONTEXT>\n"

        if t.text_diar:
            prompt += (
                "Draft for this slice: use ONLY if audio clearly matches; otherwise ignore.\n"
                f"<DRAFT>{t.text_diar}</DRAFT>\n"
            )

        prompt += f"Speaker: {t.role}\n"



        tr = await async_transcribe_audio(
            audio=chunk_path,
            model="gpt-4o-transcribe",
            language=language,
            temperature=0.0,
            response_format="json",
            prompt=prompt,
        )


        text = (getattr(tr, "text", None) or (tr.get("text") if isinstance(tr, dict) else "") or "").strip()
        log.info(
            f"\n\n{chunk_path}"
            f"\n\nCONTEXT ---> {ctx}"
            f"\n\n   ROLE ---> {t.role}" 
            f"\n  MODEL ---> {text}" 
            f"\n   DIAR ---> {t.text_diar}\n"
            )
        
        if text:
            # Return tight timestamps + keep ext fields (useful for debugging / later stages)
            out.append(
                Turn(
                    role=t.role,
                    start=st,
                    end=en,
                    text=text,
                    file=channel_wav,
                    start_ext=st_ext,
                    end_ext=en_ext,
                    text_diar=t.text_diar
                )
            )
            prev_context = (prev_context + f" {t.role}: {text}").strip()

    return out




async def async_transcribe_stereo_timestamped_lr(
    source_file: str,
    temp_root_dir: Optional[str] = None,
    language: str = "uk",
    metadata: Any = None,
    chunk_pad_ms: int = 200,
    min_seg_s: float = 0.25,
) -> Tuple[List[Turn], str]:
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
    l_seg_wav, r_seg_wav = split_stereo_to_lr_for_segments_lr(str(dst), max_gain_db=4.0, target_peak_dbfs=-3.0)

    cleaned_a = await async_gen_o4_diar_segments(
        wav_seg_path=l_seg_wav,
        language=language,
        temperature=0.0,
        min_seg_s=min_seg_s,
    )

    cleaned_b = await async_gen_o4_diar_segments(
        wav_seg_path=r_seg_wav,
        language=language,
        temperature=0.0,
        min_seg_s=min_seg_s,
    )

    a_turns = build_turns_from_diar_segments(
        cleaned_a,
        wav_path=left_wav,          # IMPORTANT: original file for slicing later
        join_gap_s=0.65,
        hard_break_gap_s=1.10,
        break_on_punct=False,
        min_phrase_s=0.30,
        max_phrase_s=15.0,
    )

    b_turns = build_turns_from_diar_segments(
        cleaned_b,
        wav_path=right_wav,          # IMPORTANT: original file for slicing later
        join_gap_s=0.65,
        hard_break_gap_s=1.10,
        break_on_punct=False,
        min_phrase_s=0.30,
        max_phrase_s=15.0,
    )

    a_scenario_text = render_timestamped_script_from_turns(a_turns, timestamp_on=False)
    a_role_res = await async_classify_agent_or_client_prefix(a_scenario_text)
    log.info(f"\n=== Detected role for left channel: {a_role_res} ===") 
    if a_role_res == "AGENT":
        a_role, b_role = "AGENT", "CLIENT"
    elif a_role_res:
        a_role, b_role = "CLIENT", "AGENT"
    a_turns = [Turn(role=a_role, start=t.start, end=t.end, text=t.text, text_diar=t.text_diar, file=t.file) for t in a_turns]
    b_turns = [Turn(role=b_role, start=t.start, end=t.end, text=t.text, text_diar=t.text_diar, file=t.file) for t in b_turns]

    pad_s = chunk_pad_ms / 1000.0
    a_turns_ext = extend_chunk_ends_turns(a_turns, extend_s=3.0, pad_s=pad_s, safety_s=1)
    b_turns_ext = extend_chunk_ends_turns(b_turns, extend_s=3.0, pad_s=pad_s, safety_s=1)

    turns = sorted(a_turns_ext + b_turns_ext, key=lambda x: (x.start, x.end))

    turns_info = render_turns_tight_vs_ext(turns=turns)
    log.info(turns_info)


    res_turns = await async_transcribe_channel_by_turns(turns = turns, 
                                                        language = language,
                                                        chunk_pad_ms = chunk_pad_ms,
                                                        min_seg_s = min_seg_s,
                                                        metadata = metadata,
                                                        )

    script = render_timestamped_script_from_turns(res_turns)
    log.info(f"\n=== Final script ===\n{script}")

    return res_turns, script


