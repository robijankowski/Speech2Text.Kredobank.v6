from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, BinaryIO, Union, Tuple

import os
import shutil
import json

from pydub import AudioSegment
from pydub.effects import normalize

from core.config import settings
from core.logger import log

from openai_tools.openai_client_transcribe import (
    async_transcribe_audio,
    async_transcribe_audio_diarized,
    Transcription,
)

from transcribe.utilities.audio_tools import (
    clean_audio_file_lr,
    clean_audio_file,
    remove_long_silences_in_audio,
    stereo_to_mono,
)

from transcribe.utilities.scenario_tools import (Turn, 
                                                 DiarSeg,
                                                 render_timestamped_script_from_turns,
                                                 render_timestamped_script_from_diar_segs,
                                                 render_turns_tight_vs_ext,
                                                 )
from transcribe.utilities.transcribe_mono_tools import async_classify_all_speakers_agent_or_client


AudioInput = Union[str, Path, BinaryIO]

def _default_transcription_model() -> str:
    return settings.AZURE_MODEL_TRANSCRIBE_STEREO if settings.USE_AZURE_OPENAI == "Y" else settings.OPENAI_MODEL_TRANSCRIBE_STEREO


STEREO_PROMPT_UA = """
Це запис розмови між клієнтом та оператором KredoBank Україна.

🔸 КРИТИЧНО ВАЖЛИВО - ЗБЕРЕЖЕННЯ ОРИГІНАЛЬНОЇ МОВИ:
- Транскрибуй ТОЧНО ТАК, ЯК СКАЗАНО, без жодних перекладів. Наприклад, не замінюйте слово «да» на «так».
- Якщо слово вимовлено українською → записуй українською
- Якщо слово вимовлено російською → записуй російською  
- Якщо фраза змішана (суржик) → зберігай кожне слово в оригінальній мові
- НІКОЛИ не замінюй російські слова українськими еквівалентами
- НІКОЛИ не "виправляй" мову - записуй автентично

Цей виклик, ймовірно, пов'язаний із роботою відділу стягнення заборгованості банку, 
і може містити згадки про заборгованість, від'ємний баланс, арешт рахунків, виконавчу службу тощо.



🔹 У розмові можуть зустрічатися ОБОМА МОВАМИ:
- Банківські терміни: "від'ємний баланс" / "отрицательный баланс", "рахунок" / "счет", "арешт" / "арест"
- Звернення: "добрий день" / "добрый день", "дякую" / "спасибо"
- Назви банківських додатків: "KredoBank", "Кредобанк", "KredoMobile"
- Службові фрази: "Алло", "Мене звати" / "Меня зовут", "Наша розмова записується" / "Наш разговор записывается"

🔹 ПРИКЛАДИ правильної транскрипції змішаного мовлення:
- "Добрий день, у мене отрицательный баланс на счету" ✓
- "Алло, можете сказать, чому в мене від'ємний баланс?" ✓  
- "Спасибо, до побачення" ✓

🔹 Інструкції для транскрипції:
- Зберігай всі слова, включно зі словами-паразитами ("ну", "добре", "може", "трошки", "ммм", "это", "вот")
- Не пропускай повторення або вагання
- Використовуй правильну пунктуацію для природного усного мовлення
- Мовна автентичність важливіша за мовну "чистоту". 
- ЗАВЖДИ ЗБЕРІГАЙТЕ РОСІЙСЬКІ СЛОВА, ЯКЩО ВОНИ З'ЯВИЛИСЯ В РОЗМОВІ!

🔹 Контекстні дані (метадані) для покращення розпізнавання:
Known entities canonical names: {metadata}

ПАМ'ЯТАЙ: Твоє завдання - точно відтворити ТЕ, ЩО БУЛО СКАЗАНО, а не те, що "повинно було б" бути сказано!
"""

async def async_transcript_audio_file_verbose_o4_stereo(file_name: str, 
                                            o4_metadata_text="", 
                                            temperature=0.0,
                                            model: str = ""
                                            ) -> Transcription:
    
    prompt = STEREO_PROMPT_UA.format(metadata=o4_metadata_text or "")

    if not model:
        model = _default_transcription_model()

    transcription = await async_transcribe_audio(
        audio=file_name,
        model=model,
        prompt=prompt,
        temperature=temperature,
        response_format="json",
        timestamp_granularities=["segment"],
    )
    log.info(f"File transcription done with model: {model}:")
    log.info("\n" + str(transcription.usage))   

    return transcription




def slice_wav(in_wav: str, out_wav: str, start_s: float, end_s: float, pad_ms: int = 250) -> str:
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
        if not txt:
            continue

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
            "Correct misspeling and grammar errors.\n"
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



async def async_transcribe_mono_timestamped_lr2(
    source_file: str,
    temp_root_dir: Optional[str] = None,
    language: str = "uk",
    metadata: Any = None,
    chunk_pad_ms: int = 200,
    min_seg_s: float = 0.25,
) -> Tuple[List[Turn], str]:
    if not temp_root_dir:
        temp_root_dir = settings.TR_TEMP_ROOT_DIR

    mono_cleaned = prepare_audio_for_transcription_mono(source_file=source_file, temp_dir=temp_root_dir)

    cleaned_segs = await async_generate_diarize_segments(
        wav_seg_path=mono_cleaned,
        language=language,
        temperature=0.0,
        min_seg_s=min_seg_s,
    )

    speaker_map = await async_classify_all_speakers_agent_or_client(cleaned_segs)
    log.info(f"\n\nRole mapping\n{speaker_map}")
    cleaned_segs_remapped = remap_diar_speakers(cleaned_segs, speaker_map)

    turns = build_turns_from_diarize_segments(
        cleaned_diar_segs=cleaned_segs_remapped,
        wav_path=mono_cleaned,          
        join_gap_s=0.65,
        hard_break_gap_s=1.10,
        break_on_punct=False,
        min_phrase_s=0.30,
        max_phrase_s=15.0,
    )

    info_diar_segs = render_timestamped_script_from_turns(turns, timestamp_on=False)
    log.info(f"\n\n=== Parsed remapped turns ===\n{info_diar_segs}")

    tr = await async_transcript_audio_file_verbose_o4_stereo( file_name= mono_cleaned,
                                                             o4_metadata_text=json.dumps(metadata, ensure_ascii=False))

    log.info(tr.text)

    # scenario2 = await async_correct_scenario_using_full_text(scenario_text=info_diar_segs, full_text=tr.text)
    scenario2 = await async_split_transcription_into_roles_4o(scenario_text= info_diar_segs,
                                            stereo_text= tr.text,
                                            metadata_text=json.dumps(metadata, ensure_ascii=False)
                                            )
    log.info(scenario2)

    return turns, info_diar_segs
    pad_s = chunk_pad_ms / 1000.0
    turns = extend_chunk_ends_turns(turns, extend_s=3.0, pad_s=pad_s, safety_s=1)
    turns = sorted(turns, key=lambda x: (x.start, x.end))
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


SYSTEM_PROMPT = (
    "You are correcting a diarized scenario of a phone call (bank AGENT vs CLIENT).\n"
    "You must only correct wording inside each scenario turn using the provided FULL transcript.\n"
    "Do NOT invent new content.\n"
)

USER_PROMPT_TEMPLATE = """\
We have:

1) SCENARIO (timestamps+roles are correct, wording may be noisy):
<SCENARIO>
{scenario}
</SCENARIO>

2) FULL TRANSCRIPT (best transcription of the whole call; use as the source of truth):
<FULL_TEXT>
{full_text}
</FULL_TEXT>

TASK:
- For EACH scenario line, replace its spoken text with the best matching phrase from FULL_TEXT.
- Keep EXACTLY the same number of lines, same order, same timestamps, same speaker labels.
- Do NOT merge or split scenario lines.
- Use only words that appear in FULL_TEXT (you may choose the best matching contiguous fragment).
- Keep UA/RU exactly as in FULL_TEXT (no translation).
- If a scenario line has no matching words in FULL_TEXT, keep it unchanged.

OUTPUT:
Return ONLY the corrected scenario, same format as input scenario.
"""


import json
from typing import Any, Dict

from core.config import settings
from openai_tools.openai_client_text import async_chat_completion_with_format, async_chat_completion  # :contentReference[oaicite:1]{index=1}

SCHEMA_CORRECTED_SCENARIO = {
    "type": "object",
    "properties": {
        "scenario": {"type": "string"},
    },
    "required": ["scenario"],
    "additionalProperties": False,
}

async def async_correct_scenario_using_full_text(
    scenario_text: str,
    full_text: str,
    model: str = "",
) -> str:
    """
    Ask LLM to correct each scenario turn using the full transcript as source-of-truth.
    Returns corrected scenario string.
    """
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT_TEMPLATE.format(scenario=scenario_text.strip(), full_text=full_text.strip())

    if not model:
        # pick a chat model you already use for text tasks
        model = (
            settings.AZURE_MODEL_CHAT_TRS_DETECT_PLAYER
            if settings.USE_AZURE_OPENAI == "Y"
            else settings.OPENAI_MODEL_CHAT_TRS_DETECT_PLAYER
        )

    resp = await async_chat_completion_with_format(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format_schema=SCHEMA_CORRECTED_SCENARIO,
        schema_name="corrected_scenario",
        model=model,
        temperature=0.0,
    )

    data = json.loads(resp.choices[0].message.content)
    return (data.get("scenario") or "").strip()






async def async_split_transcription_into_roles_4o(scenario_text, stereo_text, 
                                      metadata_text="", 
                                      model: str = "") -> str:
    SYSTEM_PROMPT = """
You are a highly accurate transcription and dialogue reconstruction assistant with specialized expertise in speaker identification.

You receive separate transcriptions of a bilingual phone call between two speakers — a bank **agent** and a **client** — from different audio channels (agent, client, stereo). Your task is to **merge them into a complete, speaker-labeled script** with PERFECT speaker assignment accuracy.

This conversation takes place between a **KredoBank Ukraine collections agent** and a **KredoBank client**, most likely concerning **outstanding debt, negative balance, account restrictions, or enforcement actions**.

CRITICAL REQUIREMENTS:
1. **COMPLETE PRESERVATION**: You MUST preserve EVERY SINGLE PHRASE, WORD, NAMES of people and UTTERANCE from the STEREO transcript
2. **ACCURATE SPEAKER ASSIGNMENT**: You MUST verify each utterance's speaker by cross-referencing with individual channel transcripts
3. **ORDER OF SENTENCES**: You MUST preserve order of sentences in STEREO transcript.

Your methodology:
* Use the **STEREO channel transcript as the complete foundation** — every word from it must appear in your output
* Use the **AGENT channel** to identify which parts were spoken by the agent
* Use the **CLIENT channel** to identify which parts were spoken by the client
* When in doubt, use contextual clues (professional language, banking terms, formal address patterns)
* Format as a script with clear speaker labels: `AG:` for agent, `CL:` for client

NEVER omit content from the stereo transcript! Your output should be a COMPLETE transcription with 100% accurate speaker assignments.
"""

    USER_PROMPT = f"""
You are given three transcripts of a phone call between a KredoBank Ukraine agent and a KredoBank client. The conversation is likely from the bank's collections department and concerns an account debt or negative balance.

Your task is to create a COMPLETE speaker-labeled conversation script that includes EVERY SINGLE PHRASE from the STEREO transcript with PERFECT speaker identification.

Conversation Metadata. Known entities canonical names:
{metadata_text}

Input Sources:
AGENT AND CLIENT CHANNEL (AG): <AGENT_CHANNEL>{scenario_text}</AGENT_CHANNEL>  
STEREO CHANNEL (combined): <STEREO_CHANNEL>{stereo_text}</STEREO_CHANNEL>

MANDATORY SPEAKER VERIFICATION PROCESS:

1. **PRIMARY RULE**: Every word from STEREO transcript MUST appear in your final output

2. **SPEAKER IDENTIFICATION METHODOLOGY**:
   - For each phrase/utterance in the STEREO transcript, CHECK if it appears in:
     * AGENT channel → Label as AG:
     * CLIENT channel → Label as CL:
     * Both channels → Use contextual analysis (see below)
     * Neither channel clearly → Use contextual analysis

3. **CONTEXTUAL SPEAKER CLUES**:
   - **AGENT typically says**: Banking terminology, policy explanations, formal procedures, account details, payment instructions, "KredoBank", professional phrases
   - **CLIENT typically says**: Personal explanations, questions about their account, emotional responses, informal language, requests for help
   - **Formal address patterns**: "Пані/Пан [Name]" usually from agent to client
   - **Ukrainian politeness markers**: Note who uses formal vs informal speech

4. **VERIFICATION CROSS-CHECK**:
   - After assigning speakers to ALL stereo content, verify each assignment by asking:
     * "Does this phrase match the tone/content pattern in the AGENT channel?"
     * "Does this phrase match the tone/content pattern in the CLIENT channel?"
     * "Is this consistent with bank agent vs client behavior?"

5. **QUALITY CONTROL REQUIREMENTS**:
   - NO phrase from stereo transcript should be omitted
   - NO speaker should be assigned randomly - each assignment must be evidence-based
   - Maintain conversation flow and logical turn-taking
   - Preserve all natural speech elements (hesitations, fillers, repetitions)

6. **ENHANCEMENT GUIDELINES**: Use AGENT and CLIENT transcripts to:
   - Clarify unclear words from stereo (but keep original if unclear)
   - Confirm speaker identity through content matching
   - Resolve ambiguous speaker assignments
   - BUT NEVER remove or skip content from stereo

Output Format:
AG: [complete agent statement - verified against agent channel]
CL: [complete client statement - verified against client channel]
AG: [next agent statement - cross-referenced for accuracy]
CL: [next client statement - cross-referenced for accuracy]
...

FINAL VERIFICATION CHECKLIST before submitting:
✓ Every phrase from STEREO transcript appears in output
✓ Each speaker assignment is supported by channel evidence or strong contextual clues
✓ Conversation flow is logical and natural
✓ No content has been omitted, condensed, or summarized

Remember: Your accuracy in speaker identification is critical. Take time to cross-reference each utterance with the individual channel transcripts before making speaker assignments.

Do not include timestamps, metadata, or section headers. Output only the final reconstructed script with verified speaker labels.
"""

    if not model:
        model = _default_split_into_roles_model()

    response = await async_chat_completion(
        model=model,    
        temperature=0,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, 
                  {"role": "user", "content": USER_PROMPT}]       
        )
    
    log.info(f"Split into roles done with model: {model}:")
    log.info("\n" + str(response.usage))   

    result = response.choices[0].message.content
    result = result.replace("```plaintext", "").replace("```", "").replace("\n\n","\n").strip()
    return result

def _default_split_into_roles_model() -> str:
    # normal transcription model (NOT diarize)
    return settings.AZURE_MODEL_CHAT_TRS_SPLIT_INTO_ROLES if settings.USE_AZURE_OPENAI == "Y" else settings.OPENAI_MODEL_CHAT_TRS_SPLIT_INTO_ROLES




