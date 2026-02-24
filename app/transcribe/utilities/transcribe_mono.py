from __future__ import annotations

import os
import shutil


from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, BinaryIO, Union, Tuple, Iterable


from pydub import AudioSegment
from pydub.effects import normalize

from app.core.config import settings
from app.core.logger import log

from app.openai_tools.openai_client_transcribe import (
    async_transcribe_audio_diarized,
    Transcription,
)

from app.transcribe.utilities.audio_tools import (
    clean_audio_file,
    remove_long_silences_in_audio,
    stereo_to_mono,
)

from app.transcribe.utilities.scenario_tools import (Turn, 
                                                 DiarSeg,
                                                 render_timestamped_script_from_turns,
                                                 render_timestamped_script_from_diar_segs,
                                                 add_prefix_to_sentences,
                                                 consolidate_dialogue
                                                 )

from app.transcribe.utilities.transcribe_mono_tools import (async_split_transcription_into_roles_4o,
                                                            async_transcript_audio_file_verbose_o4_single_channel,
                                                            async_transcript_audio_file_verbose_o4_stereo,
                                                            async_classify_all_speakers_agent_or_client
                                                            )


AudioInput = Union[str, Path, BinaryIO]



def slice_wav111111(in_wav: str, out_wav: str, start_s: float, end_s: float, pad_ms: int = 250) -> str:
    audio = AudioSegment.from_file(in_wav)
    start_ms = max(0, int(start_s * 1000) - pad_ms)
    end_ms = min(len(audio), int(end_s * 1000) + pad_ms)
    chunk = audio[start_ms:end_ms]
    chunk.export(out_wav, format="wav")
    return out_wav




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


def prepare_audio_for_transcription_mono(
    source_file: str,
    temp_dir: str,
    frame_rate: int = 16000,
) -> str:
    """
    Mirrors prepare_audio_for_transcription(), but for mono:
      - copies source into dated temp dir
      - if file is stereo/multi-channel -> downmix to mono
      - normalize + gentle noise reduction (via clean_audio_file_lr())
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


async def async_generate_diarize_segments(
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
        
        # don't drop empty text segments.
        # if not txt:
        #     continue

        cleaned.append((st_f, en_f, txt, spk))

    cleaned.sort(key=lambda x: (x[0], x[1]))
    return cleaned


def build_turns_from_diarize_segments(
    cleaned_diar_segs: List[DiarSeg],
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
    if not cleaned_diar_segs:
        return []

    STRONG_PUNCT = {".", "!", "?", "…"}

    def ends_with_strong_punct(text: str) -> bool:
        t = (text or "").strip()
        return bool(t) and t[-1] in STRONG_PUNCT

    turns: List[Turn] = []

    cur_st, cur_en, cur_txt, cur_spk = cleaned_diar_segs[0]
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

    for st, en, txt, spk in cleaned_diar_segs[1:]:
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



def extend_chunk_ends_turns11111111(
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




def remap_diar_speakers(
    segs: List[DiarSeg],
    speaker_map: Dict[str, str],
    default_keep: bool = True,
    normalize: bool = True,
) -> List[DiarSeg]:
    """
    Remap speaker labels in diar segments using mapping like {"A":"AG","B":"CL"}.

    - If default_keep=True, unknown speakers are kept as-is.
      If default_keep=False, unknown speakers become "UNK".
    - normalize=True applies .strip().upper() before mapping.
    """
    out: List[DiarSeg] = []
    for st, en, txt, spk in (segs or []):
        key = spk
        if normalize and isinstance(key, str):
            key = key.strip().upper()

        new_spk = speaker_map.get(key)
        if new_spk is None:
            new_spk = spk if default_keep else "UNK"

        out.append((st, en, txt, new_spk))
    return out





def export_role_audio_from_turns_mono(
    mono_cleaned_path: str,
    turns: Iterable[Turn],
    out_dir: Optional[str] = None,
    # Silence between concatenated slices (requested 1–2s)
    silence_between_ms: int = 1000,
    # Small padding around each slice to avoid hard cuts
    pad_ms: int = 120,
    # Use extended bounds for slicing if available (recommended when you use extend_chunk_ends_turns)
    prefer_ext_bounds: bool = True,
    # Keep output naming consistent with audio_tools: _lc.wav and _rc.wav :contentReference[oaicite:3]{index=3}
    ag_suffix: str = "AG",
    cl_suffix: str = "CL",
    # Ensure ASR-friendly settings
    frame_rate: int = 16000,
    normalize_output: bool = False,
) -> Tuple[str, str]:
    """
    Build two audio files from mono_cleaned by concatenating slices for each role:
      - AGENT -> <base>_<ag_suffix>.wav
      - CLIENT -> <base>_<cl_suffix>.wav

    Slices are taken from mono_cleaned with (optional) pad_ms on both ends,
    and separated by 1–2 seconds of inserted silence (configurable).
    """

    audio = AudioSegment.from_file(mono_cleaned_path)
    # Ensure mono + desired frame rate
    audio = audio.set_channels(1).set_frame_rate(frame_rate)

    turns_sorted = sorted(list(turns or []), key=lambda t: (float(t.start), float(t.end)))
    if not turns_sorted:
        raise ValueError("No turns provided")

    def _pick_bounds(t: Turn) -> Tuple[float, float]:
        # st = float(t.start_ext if (prefer_ext_bounds and getattr(t, "start_ext", None) is not None) else t.start)
        # en = float(t.end_ext if (prefer_ext_bounds and getattr(t, "end_ext", None) is not None) else t.end)
        st = float(t.start)
        en = float(t.end)
        if en < st:
            st, en = st, st
        return st, en

    def _slice_ms(st_s: float, en_s: float) -> AudioSegment:
        # st_ms = max(0, int(round(st_s * 1000)) - pad_ms)
        st_ms = max(0, int(round(st_s * 1000)))
        en_ms = min(len(audio), int(round(en_s * 1000)) + pad_ms)
        if en_ms <= st_ms:
            return AudioSegment.silent(duration=0, frame_rate=frame_rate)
        return audio[st_ms:en_ms]

    # Split turns into two streams
    ag_parts: list[AudioSegment] = []
    cl_parts: list[AudioSegment] = []

    for t in turns_sorted:
        role = (t.role or "").strip().upper()
        st_s, en_s = _pick_bounds(t)
        piece = _slice_ms(st_s, en_s)
        if len(piece) == 0:
            continue

        if role == "AGENT":
            ag_parts.append(piece)
        elif role == "CLIENT":
            cl_parts.append(piece)
        else:
            # If you still have raw diar speaker labels here, you can decide where they go.
            # For now: ignore unknown roles.
            continue

    if not ag_parts and not cl_parts:
        raise ValueError("No AG/CL turns found in provided turns list")

    def _concat_with_silence(parts: list[AudioSegment]) -> AudioSegment:
        out = AudioSegment.silent(duration=0, frame_rate=frame_rate)
        for i, p in enumerate(parts):
            out += p
            if i < len(parts) - 1:
                gap = silence_between_ms
                out += AudioSegment.silent(duration=gap, frame_rate=frame_rate)
        return out

    ag_audio = _concat_with_silence(ag_parts)
    cl_audio = _concat_with_silence(cl_parts)

    if normalize_output:
        # Optional; mono_cleaned is usually already normalized/cleaned, so default is False.
        if len(ag_audio) > 0:
            ag_audio = normalize(ag_audio)
        if len(cl_audio) > 0:
            cl_audio = normalize(cl_audio)

    # Output paths: keep convention consistent with audio_tools.py :contentReference[oaicite:4]{index=4}
    base_no_ext = os.path.splitext(mono_cleaned_path)[0]
    out_dir = out_dir or os.path.dirname(mono_cleaned_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    ag_out = os.path.join(out_dir, f"{os.path.basename(base_no_ext)}_{ag_suffix}.wav")
    cl_out = os.path.join(out_dir, f"{os.path.basename(base_no_ext)}_{cl_suffix}.wav")

    ag_audio.export(ag_out, format="wav")
    cl_audio.export(cl_out, format="wav")

    log.info(f"Exported AG concatenated audio: {ag_out} (parts={len(ag_parts)})")
    log.info(f"Exported CL concatenated audio: {cl_out} (parts={len(cl_parts)})")

    return ag_out, cl_out



async def async_transcribe_mono_audio_file_to_scenario(
    source_file: str,
    temp_root_dir: Optional[str] = None,
    language: str = "uk",
    metadata: Any = None,
    chunk_pad_ms: int = 200,
    min_seg_s: float = 0.2,
) -> Tuple[List[Turn], str]:

    turns, mono_cleaned_file = await async_transcribe_mono_audio_file_to_segments( source_file=source_file,
                                                                                  temp_root_dir=temp_root_dir,
                                                                                  language=language,
                                                                                  metadata=metadata,
                                                                                  chunk_pad_ms=chunk_pad_ms,
                                                                                  min_seg_s=min_seg_s)

    info_diar_segs = render_timestamped_script_from_turns(turns, timestamp_on=False)
    log.info(f"\n\n=== Parsed remapped turns ===\n{info_diar_segs}")

    ag_wav_file, cl_wav_file = export_role_audio_from_turns_mono(   mono_cleaned_path=mono_cleaned_file,
                                                                    turns=turns)

    log.info("\n\n" + "="*30 + f" Transcribe O4 cleaned AGENT virtual channel wav " + "="*30)
    o4_ag_trans = await async_transcript_audio_file_verbose_o4_single_channel(ag_wav_file, metadata)
    log.info(f"\n{o4_ag_trans.text}")

    log.info("\n\n" + "="*30 + f" Transcribe O4 cleaned CLIENT virtual channel wav as " + "="*30)
    o4_cl_trans = await async_transcript_audio_file_verbose_o4_single_channel(cl_wav_file, metadata)
    log.info(f"\n{o4_cl_trans.text}")

    log.info("\n\n" + "="*30 + f" Transcribe O4 cleaned mono wav " + "="*30)
    o4_mono_trans = await async_transcript_audio_file_verbose_o4_stereo(mono_cleaned_file, metadata)
    log.info(f"\n{o4_mono_trans.text}")
    

    agent_text = add_prefix_to_sentences(o4_ag_trans.text, "AG:")
    client_text = add_prefix_to_sentences(o4_cl_trans.text, "CL:")

    log.info("\n\n" + "="*30 + " Modified AGENT for o4 " + "="*30 + "\n" + agent_text.replace("AG:", "\nAG:"))
    log.info("\n\n" + "="*30 + " Modified CLIENT for o4 " + "="*30 + "\n" + client_text.replace("CL:", "\nCL:"))

    
    log.info("\n\n\n" + "="*30 + " Generating roles/scenario with LLM " + "="*30)
    scenario_granular = await async_split_transcription_into_roles_4o( agent_text = agent_text, 
                                                            client_text = client_text, 
                                                            stereo_text = o4_mono_trans.text )
    scenario = consolidate_dialogue(scenario_granular)
    log.info("\n" + "="*30 + " Consolidated scenario for mono file "  + "="*30 + "\n" + scenario)

    return turns, scenario





async def async_transcribe_mono_audio_file_to_segments(
    source_file: str,
    temp_root_dir: Optional[str] = None,
    language: str = "uk",
    metadata: Any = None,
    chunk_pad_ms: int = 200,
    min_seg_s: float = 0.2,
) -> List[Turn]:
    if not temp_root_dir:
        temp_root_dir = settings.TR_TEMP_ROOT_DIR

    mono_cleaned_file = prepare_audio_for_transcription_mono(source_file=source_file, temp_dir=temp_root_dir)

    cleaned_segs = await async_generate_diarize_segments(
        wav_seg_path=mono_cleaned_file,
        language=language,
        temperature=0.0,
        min_seg_s=min_seg_s,
    )

    speaker_map = await async_classify_all_speakers_agent_or_client(cleaned_segs)
    log.info(f"\n\nRole mapping for mono file\n{speaker_map}")
    cleaned_segs_remapped = remap_diar_speakers(cleaned_segs, speaker_map)

    turns = build_turns_from_diarize_segments(
        cleaned_diar_segs=cleaned_segs_remapped,
        wav_path=mono_cleaned_file,          
        join_gap_s=0.65,
        hard_break_gap_s=1.10,
        break_on_punct=False,
        min_phrase_s=0.30,
        max_phrase_s=15.0,
    )

    return turns, mono_cleaned_file