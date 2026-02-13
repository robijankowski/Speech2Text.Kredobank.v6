# transcribe/utilities/transcribe_mono.py

from __future__ import annotations

import json
import logging
import os
import shutil
import re

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydub import AudioSegment
from pydub.effects import normalize

from core.config import settings
from transcribe.core.tr_config import tr_settings

from transcribe.utilities.transcribe_stereo import (
    transcript_audio_file_verbose_o4_single_channel,
    transcript_audio_file_verbose_o4_stereo,
)
from transcribe.utilities.scenario_tools import (
    split_transcription_into_roles_4o,
    consolidate_dialogue,
)

from openai_tools.openai_client_text import (
    chat_completion_with_format,
    async_chat_completion_with_format,
)

from openai_tools.openai_client_transcribe import (
    transcribe_audio,
    transcribe_audio_diarized,
    async_transcribe_audio,
    async_transcribe_audio_diarized,
    Transcription,
)

from transcribe.utilities.audio_tools import (
    clean_audio_file,
    remove_long_silences_in_audio,
    stereo_to_mono,
    prepare_audio_for_transcription,  # for stereo wrapper
)

# If you already use these in your pipeline, we reuse them to stay consistent
from transcribe.utilities.scenario_tools import (
    detect_speaker_roles,
    add_prefix_to_sentences,
)

from transcribe.utilities.transcribe_mono_tools import (
    classify_all_speakers_agent_or_client
)

log = logging.getLogger(tr_settings.TR_LOGGER_NAME)


def _default_asr_model() -> str:
    # normal transcription model (NOT diarize)
    return settings.AZURE_MODEL_TRANSCRIBE_STEREO if settings.USE_AZURE_OPENAI == "Y" else settings.OPENAI_MODEL_TRANSCRIBE_STEREO

def _default_diarize_model() -> str:
    # diarization model
    return settings.AZURE_MODEL_TRANSCRIBE_DIARIZE if settings.USE_AZURE_OPENAI == "Y" else settings.OPENAI_MODEL_TRANSCRIBE_DIARIZE

def _default_chat_model() -> str:
    # normal transcription model (NOT diarize)
    return settings.AZURE_MODEL_CHAT_TRS_SPLIT_INTO_ROLES if settings.USE_AZURE_OPENAI == "Y" else settings.OPENAI_MODEL_CHAT_TRS_SPLIT_INTO_ROLES


# --- add below your DiarizedSeg helpers in transcribe_mono.py ---

REPAIR_DIARIZATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["start", "end", "speaker", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["segments"],
    "additionalProperties": False,
}


def _count_speaker_flips(segs: List[DiarizedSeg]) -> int:
    flips = 0
    prev = None
    for s in segs:
        if prev is not None and s.speaker != prev:
            flips += 1
        prev = s.speaker
    return flips


def _words_count(text: str) -> int:
    return len([w for w in (text or "").strip().split() if w])


_BACKCHANNEL = {
    "так", "да", "угу", "ага", "добре", "хорошо", "дякую", "спасибо",
    "алло", "алло.", "ок", "окей", "окей.", "зрозуміло", "понятно",
}


def heuristic_repair_micro_flips(
    segs: List[DiarizedSeg],
    *,
    max_short_sec: float = 0.65,
    max_words: int = 2,
    max_gap_sec: float = 0.40,
) -> List[DiarizedSeg]:
    """
    Cheap deterministic smoothing:
      - if a very short backchannel is sandwiched between same speaker => flip it to that speaker
      - if tiny segments cause A-B-A within small gap => collapse middle to A
      - merges adjacent same-speaker segments if close
    """
    if not segs:
        return []

    segs = sorted(segs, key=lambda x: (x.start, x.end))

    # 1) fix sandwiched micro segments
    fixed: List[DiarizedSeg] = []
    for i, s in enumerate(segs):
        dur = max(0.0, s.end - s.start)
        wc = _words_count(s.text)
        txt_lc = (s.text or "").strip().lower()

        prev_s = segs[i - 1] if i > 0 else None
        next_s = segs[i + 1] if i + 1 < len(segs) else None

        if prev_s and next_s:
            sandwich = (prev_s.speaker == next_s.speaker) and (s.speaker != prev_s.speaker)
            close = (s.start - prev_s.end) <= max_gap_sec and (next_s.start - s.end) <= max_gap_sec
            shorty = (dur <= max_short_sec and wc <= max_words) or (txt_lc in _BACKCHANNEL and dur <= 1.0)

            if sandwich and close and shorty:
                # re-assign to surrounding speaker
                fixed.append(DiarizedSeg(start=s.start, end=s.end, speaker=prev_s.speaker, text=s.text))
                continue

        fixed.append(s)

    # 2) merge adjacent
    merged: List[DiarizedSeg] = []
    for s in fixed:
        if not s.text.strip():
            continue
        if not merged:
            merged.append(s)
            continue
        last = merged[-1]
        if s.speaker == last.speaker and (s.start - last.end) <= 0.80:
            merged[-1] = DiarizedSeg(
                start=last.start,
                end=max(last.end, s.end),
                speaker=last.speaker,
                text=(last.text + " " + s.text).strip(),
            )
        else:
            merged.append(s)

    return merged


def _format_segments_for_llm(segs: List[DiarizedSeg], *, max_chars: int = 12000) -> str:
    """
    Compact representation to keep prompt size sane.
    """
    lines: List[str] = []
    for i, s in enumerate(segs):
        lines.append(
            f"{i:03d} | {s.start:.2f}-{s.end:.2f} | {s.speaker} | {s.text}"
        )
    txt = "\n".join(lines)
    if len(txt) > max_chars:
        return txt[:max_chars] + "\n...[truncated]..."
    return txt


def _validate_repaired_segments(
    repaired: List[DiarizedSeg],
    *,
    allowed_speakers: set[str],
) -> bool:
    """
    Basic sanity checks: time order, non-empty text, speaker in allowed set.
    We allow small overlaps; scenario rendering will still be OK.
    """
    if not repaired:
        return False

    prev_start = -1e9
    for s in repaired:
        if s.speaker not in allowed_speakers:
            return False
        if s.end <= s.start:
            return False
        if s.start < prev_start - 0.25:  # big backward jump
            return False
        if not (s.text or "").strip():
            return False
        prev_start = s.start
    return True


def repair_diarized_segments_llm(
    segs: List[DiarizedSeg],
    *,
    metadata: Any = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    chunk_size: int = 90,
    overlap: int = 12,
) -> List[DiarizedSeg]:
    """
    LLM repair pass (gpt-5.2 recommended):
      - fix micro speaker flips
      - fix overlap artifacts where possible using context
      - merge fragments that belong together
    Works in chunked mode for long calls.
    """
    if not segs:
        return []

    segs = sorted(segs, key=lambda x: (x.start, x.end))
    allowed_speakers = sorted({s.speaker for s in segs})

    system = (
        "You are repairing diarized transcript segments from a TWO-PERSON bank call.\n"
        "Goal: fix diarization artifacts (micro speaker flips, backchannel mislabels, overlap confusion) "
        "WITHOUT changing what was said.\n\n"
        "STRICT RULES:\n"
        "- DO NOT translate or normalize language. Keep Ukrainian/Russian words exactly as spoken.\n"
        "- DO NOT invent new words.\n"
        "- You MAY: reassign the 'speaker' of an existing segment, merge adjacent segments, "
        "and lightly adjust boundaries ONLY if needed for consistency.\n"
        "- Keep speakers limited to the existing labels only.\n"
        "- Output must be valid JSON matching the schema.\n"
    )
    system = (
        "You are repairing diarized transcript segments from a TWO-PERSON bank call: CLIENT and AGENT.\n"
        "Goal: fix diarization artifacts (micro speaker flips, backchannel mislabels, overlap confusion).\n\n"
        "STRICT RULES:\n"
        "- DO NOT translate language. Keep Ukrainian/Russian words as originally spoken.\n"
        "- DO NOT rewrite sentences or improve style.\n"
        "- DO NOT add new content.\n"
        "- You MAY ONLY change 'text' in one special case: "
        "correct obvious ASR misspellings of the KNOWN ENTITIES listed below, "
        "replacing them with the canonical spellings exactly.\n"
        "- If you are not confident a token refers to a known entity, leave it unchanged.\n"
        "- You MAY reassign 'speaker', merge adjacent segments, and lightly adjust boundaries.\n"
        "- Keep speakers limited to the existing labels only.\n"
        "- Output must be valid JSON matching the schema.\n\n"
        "Known entities canonical names: {metadata}"
    )

    def _call_one(chunk: List[DiarizedSeg]) -> List[DiarizedSeg]:
        user = (
            f"Speakers: {allowed_speakers}\n"
            f"Metadata: {_to_str_metadata(metadata)}\n\n"
            "Here are diarized segments (index | start-end | speaker | text):\n"
            "-----\n"
            f"{_format_segments_for_llm(chunk)}\n"
            "-----\n\n"
            "Return repaired segments as JSON."
        )

        print("PROMPT TO LLM:\n", system + "\n\n" + user)

        model = _default_chat_model()
        completion = chat_completion_with_format(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model,
            temperature=temperature,
            timeout=timeout,
            format_schema=REPAIR_DIARIZATION_SCHEMA,
            schema_name="repair_diarization_segments",
        )

        payload = json.loads(completion.choices[0].message.content)
        out = []
        for r in payload.get("segments", []):
            out.append(
                DiarizedSeg(
                    start=float(r["start"]),
                    end=float(r["end"]),
                    speaker=str(r["speaker"]),
                    text=str(r["text"]).strip(),
                )
            )
        out.sort(key=lambda x: (x.start, x.end))
        if not _validate_repaired_segments(out, allowed_speakers=set(allowed_speakers)):
            # fallback to original chunk
            return chunk
        return out

    # Chunked repair for long calls
    if len(segs) <= chunk_size:
        return _call_one(segs)

    repaired_all: List[DiarizedSeg] = []
    i = 0
    while i < len(segs):
        j = min(len(segs), i + chunk_size)
        chunk = segs[i:j]
        repaired_chunk = _call_one(chunk)

        # avoid duplicating overlap region from previous chunk
        if repaired_all and overlap > 0:
            # drop first N segments of chunk to reduce boundary duplication
            repaired_chunk = repaired_chunk[overlap:] if len(repaired_chunk) > overlap else []

        repaired_all.extend(repaired_chunk)
        if j >= len(segs):
            break
        i = j - overlap  # overlap for context

    repaired_all.sort(key=lambda x: (x.start, x.end))
    return repaired_all


def repair_diarized_segments(
    segs: List[DiarizedSeg],
    *,
    metadata: Any = None,
    timeout: float = 120.0,
    force_llm: bool = False,
) -> List[DiarizedSeg]:
    """
    Combined repair:
      1) heuristic smoothing (cheap)
      2) LLM repair (gpt-5.2) when flips are still high or force_llm=True
    """
    if not segs:
        return []

    pre = heuristic_repair_micro_flips(segs)
    flips_before = _count_speaker_flips(segs)
    flips_after = _count_speaker_flips(pre)

    # run LLM if still noisy
    noisy = flips_after >= max(10, int(len(pre) * 0.18))
    if force_llm or noisy or (flips_after > flips_before * 0.9):
        try:
            return repair_diarized_segments_llm(pre, 
                                                metadata=metadata, 
                                                timeout=timeout)
        except Exception:
            log.exception("LLM diarization repair failed; using heuristic-only output.")
            return pre

    return pre


# -----------------------------
# Prompts (aligned with your stereo prompt rules)
# -----------------------------

SINGLE_CHANNEL_UNKNOWN_ROLE_PROMPT_EN = """
This is an isolated single-channel recording from a phone call between a KredoBank Ukraine collections AGENT and a CLIENT.
Only ONE participant is audible in this file, but you DO NOT know whether it is the AGENT or the CLIENT.

Your task: produce a verbatim transcript of the audible speaker ONLY.

CRITICAL: PRESERVE THE ORIGINAL SPOKEN LANGUAGE (NO TRANSLATION)
- Transcribe EXACTLY what is said, without translating or "correcting" language.
- If a word is spoken in Ukrainian → write it in Ukrainian.
- If a word is spoken in Russian → write it in Russian.
- If speech is mixed (code-switching / surzhyk) → preserve each word in its original language.
- NEVER replace Russian words with Ukrainian equivalents (e.g., do not change "да" to "так").

SINGLE-SPEAKER RULES
- Do NOT invent the other speaker’s lines.
- Do NOT add placeholders like “(other speaker)” or imagined replies.
- If the speaker repeats themselves, hesitates, or uses fillers (“ну”, “ммм”, “это”, “вот”, etc.), keep them.

PUNCTUATION & STYLE
- Add natural punctuation and capitalization suitable for spoken dialogue.
- Keep numbers, names, dates, amounts exactly as spoken.
- If a word is unclear, keep your best guess; if truly unintelligible, use [unclear].

Context metadata to help recognition (names, bank, etc.):
Known entities canonical names: {metadata}
"""

MONO_CALL_PROMPT_UA = """
Це МОНО запис телефонної розмови між клієнтом та оператором KredoBank Україна (відділ стягнення заборгованості).

🔸 КРИТИЧНО ВАЖЛИВО - ЗБЕРЕЖЕННЯ ОРИГІНАЛЬНОЇ МОВИ:
- Транскрибуй ТОЧНО ТАК, ЯК СКАЗАНО, без перекладів.
- Якщо слово українською → записуй українською
- Якщо слово російською → записуй російською
- Якщо змішано (суржик) → зберігай кожне слово в оригінальній мові
- НІКОЛИ не замінюй російські слова українськими еквівалентами (не змінюй "да" на "так")
- НІКОЛИ не "виправляй" мову

📌 ВАЖЛИВО:
- Додай природну пунктуацію для усного мовлення.
- Зберігай слова-паразити та повтори.
- Імена/суми/дати — як сказано.

🔹 Контекстні дані (метадані):
Known entities canonical names: {metadata}
"""


# -----------------------------
# Data helpers
# -----------------------------

@dataclass(frozen=True)
class DiarizedSeg:
    start: float
    end: float
    speaker: str
    text: str


def _to_str_metadata(metadata: Any) -> str:
    if metadata is None:
        return ""
    if isinstance(metadata, str):
        return metadata
    try:
        return json.dumps(metadata, ensure_ascii=False)
    except Exception:
        return str(metadata)


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


def _get_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, dict):
        return str(obj.get("text") or "")
    return str(getattr(obj, "text", "") or "")


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


def _seg_get(seg: Any, key: str, default: Any = None) -> Any:
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def parse_diarized_segments(transcription: Any, speaker_map: Optional[Dict[str, str]] = None) -> List[DiarizedSeg]:
    segs = _extract_segments(transcription)
    out: List[DiarizedSeg] = []

    # normalize mapping keys once
    norm_map = {str(k).strip(): str(v).strip() for k, v in (speaker_map or {}).items()}

    for s in segs:
        text = (_seg_get(s, "text", "") or "").strip()
        if not text:
            continue

        start = _seg_get(s, "start", None)
        end = _seg_get(s, "end", None)
        if start is None or end is None:
            continue

        speaker = (
            _seg_get(s, "speaker", None)
            or _seg_get(s, "speaker_id", None)
            or _seg_get(s, "spk", None)
            or "S0"
        )

        spk = str(speaker).strip()
        new_spk = norm_map.get(spk, spk)

        out.append(DiarizedSeg(start=float(start), end=float(end), speaker=str(new_spk), text=text))

    out.sort(key=lambda x: (x.start, x.end))
    return out














def group_by_speaker(segs: List[DiarizedSeg]) -> Dict[str, List[DiarizedSeg]]:
    by: Dict[str, List[DiarizedSeg]] = {}
    for s in segs:
        by.setdefault(s.speaker, []).append(s)
    return by


def speaker_total_duration(segs: List[DiarizedSeg]) -> float:
    return sum(max(0.0, s.end - s.start) for s in segs)


def choose_top_speakers(by_speaker: Dict[str, List[DiarizedSeg]], max_speakers: int = 2) -> List[str]:
    items = [(spk, speaker_total_duration(segs)) for spk, segs in by_speaker.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return [spk for spk, _dur in items[:max_speakers]]


# -----------------------------
# Preprocessing (mono)
# -----------------------------

def prepare_audio_for_transcription_mono(
    *,
    source_file: str,
    temp_dir: str,
    frame_rate: int = 16000,
) -> str:
    """
    Mirrors prepare_audio_for_transcription(), but for mono:
      - copies source into dated temp dir
      - if file is stereo/multi-channel -> downmix to mono
      - normalize + gentle noise reduction (via clean_audio_file())
      - remove long silences (remove_long_silences_in_audio())
    Returns cleaned mono wav path.
    """
    src = Path(source_file)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    day_dir = Path(temp_dir) / datetime.now().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    dst = day_dir / src.name
    shutil.copy2(src, dst)
    log.info("Copied source file to temp: %s -> %s", str(src), str(dst))

    # Ensure mono
    audio = AudioSegment.from_file(str(dst))
    if audio.channels != 1:
        mono_path = stereo_to_mono(str(dst), mode="mix", frame_rate=frame_rate)
        work_path = mono_path
    else:
        # still re-export to ensure frame_rate consistency
        base, _ = os.path.splitext(str(dst))
        work_path = f"{base}_mono.wav"
        audio = normalize(audio).set_frame_rate(frame_rate).set_channels(1)
        audio.export(work_path, format="wav")

    # Clean (normalize + NR) + remove long silences
    cleaned = clean_audio_file(work_path)
    cleaned = remove_long_silences_in_audio(cleaned)
    return cleaned


# -----------------------------
# Transcription steps (mono)
# -----------------------------

def transcribe_mono_full(
    *,
    mono_file_cleaned: str,
    metadata: Any = None,
    model: str = "",
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> Transcription:
    """
    Full mono transcription (not diarized) – useful for fallback / debugging.
    """
    prompt = MONO_CALL_PROMPT_UA.format(metadata=_to_str_metadata(metadata))
    model = _default_asr_model() 

    tr = transcribe_audio(
        audio=mono_file_cleaned,
        model=model,
        prompt=prompt,
        temperature=temperature,
        timeout=timeout,
        response_format="json",
        timestamp_granularities=["segment"],
    )
    log.info("Mono full transcription done with model: %s", model)
    return tr


def transcribe_mono_diarized(
    *,
    mono_file_cleaned: str,
    temperature: float = 0.0,
    timeout: float = 120.0,
    chunking_strategy: str = "auto",
) -> Transcription:
    """
    Diarized mono transcription (speaker turns).
    """
    model = _default_diarize_model()
    tr = transcribe_audio_diarized(
        audio=mono_file_cleaned,
        model=model,
        temperature=temperature,
        timeout=timeout,
        chunking_strategy=chunking_strategy,
    )
    log.info("Mono diarized transcription done with model: %s", model)
    return tr


# -----------------------------
# Virtual channels (optional accuracy boost)
# -----------------------------

MappingItem = Tuple[float, float, float, float]  # (concat_start, concat_end, orig_start, orig_end)


def build_speaker_stream(
    *,
    mono_file_cleaned: str,
    speaker_segments: List[DiarizedSeg],
    out_file: str,
    keep_silence_ms: int = 300,
    pad_ms: int = 60,
    min_segment_sec: float = 0.25,
) -> Tuple[str, List[MappingItem]]:
    """
    Concatenate only one speaker's diarization segments into a new wav.
    Returns (out_file, mapping) where mapping converts concatenated timeline -> original timeline.
    """
    audio = AudioSegment.from_file(mono_file_cleaned)
    silence = AudioSegment.silent(duration=max(0, keep_silence_ms))

    cursor_s = 0.0
    mapping: List[MappingItem] = []
    out_audio = AudioSegment.silent(duration=0)

    for seg in speaker_segments:
        dur = seg.end - seg.start
        if dur < min_segment_sec:
            continue

        start_ms = max(0, int(seg.start * 1000) - pad_ms)
        end_ms = min(len(audio), int(seg.end * 1000) + pad_ms)
        if end_ms <= start_ms:
            continue

        chunk = audio[start_ms:end_ms]
        chunk_dur_s = len(chunk) / 1000.0

        # map chunk interval
        mapping.append((cursor_s, cursor_s + chunk_dur_s, seg.start, seg.end))
        out_audio += chunk
        cursor_s += chunk_dur_s

        # map inserted silence to a fixed original time (end boundary)
        if keep_silence_ms > 0:
            sil_dur_s = keep_silence_ms / 1000.0
            mapping.append((cursor_s, cursor_s + sil_dur_s, seg.end, seg.end))
            out_audio += silence
            cursor_s += sil_dur_s

    out_audio = normalize(out_audio).set_frame_rate(16000).set_channels(1)
    out_audio.export(out_file, format="wav")
    return out_file, mapping


def _map_time(concat_t: float, mapping: List[MappingItem]) -> float:
    """
    Convert time in concatenated stream into original time using piecewise linear mapping.
    """
    if not mapping:
        return concat_t

    # mapping is monotonic; linear scan is fine for typical segment counts
    for c0, c1, o0, o1 in mapping:
        if concat_t < c0:
            return o0
        if c0 <= concat_t <= c1:
            if c1 <= c0:
                return o0
            if o1 == o0:
                return o0
            ratio = (concat_t - c0) / (c1 - c0)
            return o0 + ratio * (o1 - o0)

    # past the end
    return mapping[-1][3]


def parse_timestamped_segments(transcription: Any, speaker_id: str) -> List[DiarizedSeg]:
    """
    Parse timestamped segments from non-diarized ASR output (json with segment timestamps).
    Speaker label is forced to speaker_id.
    """
    segs = _extract_segments(transcription)
    out: List[DiarizedSeg] = []
    for s in segs:
        text = (_seg_get(s, "text", "") or "").strip()
        if not text:
            continue
        start = _seg_get(s, "start", None)
        end = _seg_get(s, "end", None)
        if start is None or end is None:
            continue
        out.append(DiarizedSeg(start=float(start), end=float(end), speaker=speaker_id, text=text))
    out.sort(key=lambda x: (x.start, x.end))
    return out


def transcribe_speaker_stream(
    *,
    speaker_wav: str,
    metadata: Any = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> Transcription:
    """
    Single-speaker ASR on the concatenated stream.
    """
    prompt = SINGLE_CHANNEL_UNKNOWN_ROLE_PROMPT_EN.format(metadata=_to_str_metadata(metadata))
    
    model = _default_asr_model()
    tr = transcribe_audio(
        audio=speaker_wav,
        model=model,
        prompt=prompt,
        temperature=temperature,
        timeout=timeout,
        response_format="json",
        timestamp_granularities=["segment"],
    )
    return tr


def enhance_segments_with_virtual_channels(
    *,
    mono_file_cleaned: str,
    diarized_segments: List[DiarizedSeg],
    temp_dir: str,
    metadata: Any = None,
    model: str = "",
    keep_silence_ms: int = 300,
) -> List[DiarizedSeg]:
    """
    Optional accuracy boost:
      1) Use diarization to isolate each speaker stream
      2) Re-transcribe each stream with single-speaker prompt
      3) Map segments back to original timeline
      4) Merge into one timeline
    """
    by = group_by_speaker(diarized_segments)
    top = choose_top_speakers(by, max_speakers=2)
    if len(top) < 2:
        return diarized_segments  # nothing to enhance

    enhanced: List[DiarizedSeg] = []

    base = Path(mono_file_cleaned)
    for spk in top:
        out_wav = str(base.with_suffix("").as_posix()) + f"_{spk}_stream.wav"
        out_wav = str(Path(temp_dir) / Path(out_wav).name)

        speaker_wav, mapping = build_speaker_stream(
            mono_file_cleaned=mono_file_cleaned,
            speaker_segments=by[spk],
            out_file=out_wav,
            keep_silence_ms=keep_silence_ms,
        )

        tr = transcribe_speaker_stream(
            speaker_wav=speaker_wav,
            metadata=metadata,
        )
        spk_segs = parse_timestamped_segments(tr, speaker_id=spk)

        # remap times back to original timeline
        for s in spk_segs:
            enhanced.append(
                DiarizedSeg(
                    start=_map_time(s.start, mapping),
                    end=_map_time(s.end, mapping),
                    speaker=spk,
                    text=s.text,
                )
            )

    # Add any “extra” diarization speakers back using original diarized text
    for spk, segs in by.items():
        if spk not in top:
            enhanced.extend(segs)

    if not enhanced:
        return diarized_segments  # ✅ fallback safety

    enhanced.sort(key=lambda x: (x.start, x.end))
    return enhanced


# -----------------------------
# Role mapping + scenario formatting
# -----------------------------

def _normalize_ws(s: str) -> str:
    return " ".join((s or "").split())


def map_speaker_ids_to_roles(
    *,
    diarized_segments: List[DiarizedSeg],
) -> Dict[str, str]:
    """
    Uses your existing detect_speaker_roles() heuristic/LLM to decide which speaker is AG vs CL,
    based on concatenated text blocks per speaker.
    Returns mapping like {"S0": "AG", "S1": "CL"}.
    """
    by = group_by_speaker(diarized_segments)
    speakers = choose_top_speakers(by, max_speakers=2)
    if not speakers:
        return {}

    if len(speakers) == 1:
        # single-speaker edge case: guess AG if it looks like scripted agent talk
        txt = " ".join(s.text for s in by[speakers[0]])
        agentish = any(k in txt.lower() for k in ["мене звати", "меня зовут", "розмова записується", "разговор записывается", "кредобанк", "kredobank"])
        return {speakers[0]: "AG" if agentish else "CL"}

    s0, s1 = speakers[0], speakers[1]
    t0 = " ".join(s.text for s in by[s0])
    t1 = " ".join(s.text for s in by[s1])

    agent_text, client_text = detect_speaker_roles(t0, t1)

    # detect_speaker_roles usually returns one of inputs (or near-identical).
    # We map by best whitespace-normalized containment.
    n0 = _normalize_ws(t0)
    n1 = _normalize_ws(t1)
    na = _normalize_ws(agent_text)

    if na and (na in n0 or n0 in na):
        return {s0: "AG", s1: "CL"}
    if na and (na in n1 or n1 in na):
        return {s1: "AG", s0: "CL"}

    # fallback: agent text longer often indicates scripted agent (not always true, but ok fallback)
    return {s0: "AG", s1: "CL"} if len(t0) >= len(t1) else {s1: "AG", s0: "CL"}


def render_scenario_from_diarized(
    *,
    diarized_segments: List[DiarizedSeg],
    speaker_to_role: Dict[str, str],
    merge_gap_sec: float = 0.8,
) -> str:
    """
    Turns diarized segments into final scenario:
      AG: ...
      CL: ...
    Merges consecutive same-role segments if close in time.
    """
    if not diarized_segments:
        return ""

    lines: List[Tuple[str, float, float, str]] = []  # (role, start, end, text)

    for s in diarized_segments:
        role = speaker_to_role.get(s.speaker, "CL")  # default CL for extras
        lines.append((role, s.start, s.end, s.text.strip()))

    # merge adjacent
    merged: List[Tuple[str, float, float, str]] = []
    for role, st, en, text in lines:
        if not text:
            continue
        if not merged:
            merged.append((role, st, en, text))
            continue

        last_role, last_st, last_en, last_text = merged[-1]
        if role == last_role and (st - last_en) <= merge_gap_sec:
            merged[-1] = (last_role, last_st, max(last_en, en), (last_text + " " + text).strip())
        else:
            merged.append((role, st, en, text))

    # final formatting
    out_lines = [f"{role}: {txt}" for role, _st, _en, txt in merged]
    return "\n".join(out_lines)



def consolidate_consecutive_roles_text(
    scenario: str,
    *,
    roles: List[str] = ["AG", "CL"],
    join_with: str = " ",
) -> str:
    """
    Consolidate consecutive lines with the same role prefix.
    Input lines must look like: 'AG: ...' or 'CL: ...'

    Example:
      AG: a
      AG: b
      CL: c
    -> AG: a b
       CL: c
    """
    if not scenario or not scenario.strip():
        return ""

    role_set = set(r.strip().upper() for r in roles)
    lines = [ln.strip() for ln in scenario.splitlines() if ln.strip()]

    out: List[str] = []
    cur_role = None
    cur_text_parts: List[str] = []

    for ln in lines:
        m = re.match(r"^([A-Za-z]+)\s*:\s*(.*)$", ln)
        if not m:
            # if line doesn't match role format, flush current and keep raw
            if cur_role is not None:
                out.append(f"{cur_role}: {join_with.join(cur_text_parts).strip()}")
                cur_role, cur_text_parts = None, []
            out.append(ln)
            continue

        role = m.group(1).upper().strip()
        text = (m.group(2) or "").strip()

        if role not in role_set:
            # unknown role -> flush previous, keep as-is
            if cur_role is not None:
                out.append(f"{cur_role}: {join_with.join(cur_text_parts).strip()}")
                cur_role, cur_text_parts = None, []
            out.append(ln)
            continue

        if cur_role is None:
            cur_role = role
            cur_text_parts = [text] if text else []
            continue

        if role == cur_role:
            if text:
                cur_text_parts.append(text)
        else:
            out.append(f"{cur_role}: {join_with.join(cur_text_parts).strip()}")
            cur_role = role
            cur_text_parts = [text] if text else []

    if cur_role is not None:
        out.append(f"{cur_role}: {join_with.join(cur_text_parts).strip()}")

    return "\n".join(out)


# -----------------------------
# Main wrapper: MONO -> scenario
# -----------------------------

def transcribe_mono_to_scenario(
    *,
    source_file: str,
    temp_dir: str,
    metadata: Any = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    enhance_with_virtual_channels: bool = True,
    keep_silence_ms: int = 300,
    repair_with_llm: bool = True,
) -> str:
    """
    End-to-end MONO pipeline:
      preprocess -> diarized transcription -> (optional refine) -> role map -> scenario
    """
    mono_cleaned = prepare_audio_for_transcription_mono(source_file=source_file, temp_dir=temp_dir)

    diar = transcribe_mono_diarized(
        mono_file_cleaned=mono_cleaned,
        temperature=temperature,
        timeout=timeout,
    )

    print("\n\nDiarization result:", diar)

    speaker_map = classify_all_speakers_agent_or_client(diar)
    print("\n\nRole mapping:", speaker_map)

    # diar = relabel_diarized_speakers(diar, speaker_map=speaker_map, default_role="Agent")

    diar_segs = parse_diarized_segments(diar, speaker_map=speaker_map)
    print("\n\nParsed diarized segments:", diar_segs)

    if enhance_with_virtual_channels:
        diar_segs = enhance_segments_with_virtual_channels(
            mono_file_cleaned=mono_cleaned,
            diarized_segments=diar_segs,
            temp_dir=temp_dir,
            metadata=metadata,
            keep_silence_ms=keep_silence_ms,
        )
        print("Enhanced diarized segments with virtual channels:", diar_segs)   

    if repair_with_llm:
        diar_segs = repair_diarized_segments(
            diar_segs,
            metadata=metadata,
            timeout=timeout,
        )
        print("Repaired diarized segments with LLM:", diar_segs)

    # spk_map = map_speaker_ids_to_roles(diarized_segments=diar_segs)
    # print("Final speaker to role map:", spk_map)
    spk_map = {'AGENT': 'AG', 'CLIENT': 'CL'}
    print("Final speaker to role map:", spk_map)
    scenario = render_scenario_from_diarized(diarized_segments=diar_segs, speaker_to_role=spk_map)
    scenario = consolidate_consecutive_roles_text(scenario)
    return scenario


# -----------------------------
# Universal wrapper: AUTO stereo/mono -> scenario
# -----------------------------

def transcribe_file_to_scenario(
    *,
    source_file: str,
    temp_dir: str,
    metadata: Any = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
    force_mono: bool = False,
) -> str:
    """
    One entry point for your app:
      - If stereo and not force_mono -> use your existing stereo pipeline
      - Else -> mono pipeline above
    """
    audio = AudioSegment.from_file(source_file)
    is_stereo = (audio.channels == 2)

    if is_stereo and not force_mono:
        # Existing stereo pipeline (same as main_transcribe_openai)
        l_file, r_file, org_file = prepare_audio_for_transcription(source_file, temp_dir)

        o4_left = transcript_audio_file_verbose_o4_single_channel(
            l_file,
            o4_metadata_text=_to_str_metadata(metadata),
            temperature=temperature,
        )
        o4_right = transcript_audio_file_verbose_o4_single_channel(
            r_file,
            o4_metadata_text=_to_str_metadata(metadata),
            temperature=temperature,
        )
        o4_full = transcript_audio_file_verbose_o4_stereo(
            org_file,
            o4_metadata_text=_to_str_metadata(metadata),
            temperature=temperature,
        )

        agent_text, client_text = detect_speaker_roles(o4_left.text, o4_right.text)
        agent_text = add_prefix_to_sentences(agent_text, "AG:")
        client_text = add_prefix_to_sentences(client_text, "CL:")

        scenario_granular = split_transcription_into_roles_4o(
            agent_text=agent_text,
            client_text=client_text,
            stereo_text=o4_full.text,
        )
        return consolidate_dialogue(scenario_granular)

    # mono path
    return transcribe_mono_to_scenario(
        source_file=source_file,
        temp_dir=temp_dir,
        metadata=metadata,
        temperature=temperature,
        timeout=timeout,
        enhance_with_virtual_channels=True,
    )



O4_METADATA = [
    {"clientName": "Ivolo Olena Volodymyrivna", "agentName": "Ulyana", "bankName": "KredoBank Ukraine"},
    {"name": "Lukashchuk Serhii Mykolayivych", "agentName": "Sviatoslav", "bankName": "KredoBank Ukraine"},
    {},
    {"agentName": "Ivanova", "bankName": "KredoBank Ukraine"},
    {}
]









scenario = transcribe_file_to_scenario(
    source_file="./test/test_call_mono.wav",
    temp_dir=tr_settings.TR_TEMP_ROOT_DIR,
    metadata=O4_METADATA[0]
)

print(scenario)