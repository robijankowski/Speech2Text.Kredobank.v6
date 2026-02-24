from __future__ import annotations

import json
import asyncio 
from typing import Any, Dict, List, Literal, Tuple, Optional
import re
import os
from concurrent.futures import ProcessPoolExecutor

from app.core.config import settings
from app.core.logger import log

from app.openai_tools.openai_client_text import async_chat_completion_with_format

from app.openai_tools.openai_client_transcribe import ( async_transcribe_audio,
                                                    async_transcribe_audio_diarized,
                                                    TranscriptionDiarizedSegment
                                                    )

from app.transcribe.utlities.audio_tools_v2 import (clean_audio_file_with_silence_removal, 
                                                 clean_audio_file_with_silence_removal_asr
                                                 )

from app.transcribe.utlities.scenario_tools_v2 import (Turn, 
                                                    render_timestamped_script_from_turns
                                                    )




def _default_detect_speaker_role_model() -> str:
    return settings.AZURE_MODEL_CHAT_TRS_DETECT_PLAYER if settings.USE_AZURE_OPENAI == "Y" else settings.OPENAI_MODEL_CHAT_TRS_DETECT_PLAYER

def _default_split_into_roles_model() -> str:
    return settings.AZURE_MODEL_CHAT_TRS_SPLIT_INTO_ROLES if settings.USE_AZURE_OPENAI == "Y" else settings.OPENAI_MODEL_CHAT_TRS_SPLIT_INTO_ROLES






def _seg_get(seg, name: str, default=None):
    # supports both dict-like and attribute objects
    if isinstance(seg, dict):
        return seg.get(name, default)
    return getattr(seg, name, default)


def remap_diar_speakers_to_turns(
    segs: List["TranscriptionDiarizedSegment"],
    speaker_map: Dict[str, str],
    default_keep: bool = True,
    normalize: bool = True,
    file: str = "",
    keep_original_in_text_diar: bool = True,
) -> List["Turn"]:
    """
    Convert diarized segments -> Turn list, remapping speaker labels via speaker_map.

    Input:  List[TranscriptionDiarizedSegment] with fields: start, end, text, speaker
    Output: List[Turn] with:
      - role = mapped label (e.g. "AGENT"/"CLIENT") or original speaker / "UNK"
      - start/end copied
      - text = segment text
      - text_diar optionally keeps the same text for debugging
      - file optionally set

    - If default_keep=True, unknown speakers are kept as-is.
      If default_keep=False, unknown speakers become "UNK".
    - normalize=True applies .strip().upper() before mapping.
    """
    out: List["Turn"] = []
    for seg in (segs or []):
        st = float(_seg_get(seg, "start", 0.0) or 0.0)
        en = float(_seg_get(seg, "end", 0.0) or 0.0)
        if en <= st:
            continue

        txt = str(_seg_get(seg, "text", "") or "").strip()
        spk = str(_seg_get(seg, "speaker", "") or "S0")

        key = spk
        if normalize and isinstance(key, str):
            key = key.strip().upper()

        mapped = speaker_map.get(key)
        if mapped is None:
            mapped = spk if default_keep else "UNK"

        out.append(
            Turn(
                role=str(mapped),
                start=st,
                end=en,
                text=txt,
                text_diar=txt if keep_original_in_text_diar else "",
                file=file,
            )
        )

    return out


async def async_transcribe_stereo_hq_roles_to_turns_v2(
    *,
    source_file: str,
    temp_root_dir: str = "",
    language: str = "uk",
    metadata: Any = None,
) -> Tuple[List[Turn], str]:
    """
    1) diarize whole file -> diar turns (Speaker A/B -> roles)
    2) HQ transcribe whole file with word timestamps
    3) assign HQ words by time -> Turn.text
    4) merge + render scenario
    """

    # Preprocess once and use the SAME cleaned file for both calls (important for timestamp consistency)
    # cleaned_source_file = clean_audio_file_with_silence_removal_asr(source_file, temp_root_dir=temp_root_dir)
    cleaned_source_file = await asyncio.to_thread(
        clean_audio_file_with_silence_removal_asr,
        source_file, 
        temp_root_dir=temp_root_dir
    )

    meta = json.dumps(metadata, ensure_ascii=False)
    # prompt = (
    #     "This is a phone call between bank AGENT and CLIENT.\n"
    #     "Transcribe verbatim. Keep original spoken language (UA/RU mixed). No translation.\n"
    #     "CRITICAL: Keep names/amounts/numbers/dates/codes exactly. \n"
    #     "If unclear: [unclear].\n"
    #     f"Known entities canonical names: {meta}\n"
    # )

    prompt = (
        "This is a phone call between bank AGENT and CLIENT.\n"
        "Transcribe verbatim. Keep original spoken language (UA/RU mixed). No translation.\n"
        "Keep namess exactly. If unclear: [unclear].\n"
        "NUMBERS POLICY (critical):"
            "- Output ALL numbers, amounts, dates as DIGITS (not words)."
            "- For money: use format '8,25' (comma) and include currency if spoken."
            "- If any number is uncertain, write it as '[num?]' and keep surrounding words."
        f"Known entities canonical names: {meta}\n"
    )

    log.info(f"\n\nPrompt for transcribe:\n{prompt}\n")

    # 1) diarize whole file
    diar, hq = await asyncio.gather(
        async_transcribe_audio_diarized(
            audio=cleaned_source_file,
            language=language,
            temperature=0.0,
            chunking_strategy="auto",
        ),
        async_transcribe_audio(
            audio=cleaned_source_file,
            model=settings.OPENAI_MODEL_TRANSCRIBE_STEREO,
            language=language,
            temperature=0.0,
            response_format="json",              # <-- important
            timestamp_granularities=["segment"], # <-- works with verbose_json
            prompt=prompt,
        ),
    )

    log.info(f"\n\nResult of transcribe:\n{str(hq)}\n")
    log.info(f"\n\nResult of transcribe diarize:\n{str(diar)}\n")

    diar_segs = diar.segments
    hq_text = hq.text

    diar_segs_speaker_map = await async_classify_all_speakers_agent_or_client(diar_segs)
    log.info(f"\n\nSpeakers mapping on diarized segs:\n{diar_segs_speaker_map}\n")

    diar_segs_turns = remap_diar_speakers_to_turns(segs=diar_segs, 
                                                   speaker_map=diar_segs_speaker_map, 
                                                   file=cleaned_source_file)
    diar_segs_turns_log_info = render_timestamped_script_from_turns(diar_segs_turns)
    log.info(f"\n\nFormatted turns from diarized segs:\n{diar_segs_turns_log_info}\n")  

    
    diar_segs_scenario = render_timestamped_script_from_turns(diar_segs_turns, timestamp_on=False)
    log.info(f"\n\nFormatted scenario based on diarized segs to be corrected by LLM:\n{diar_segs_scenario}\n")  
    log.info(f"\n\nHQ Text to be used by LLM to correct scenario based on diarized segs:\n{hq_text}\n")  

    scenario_corrected = await async_correct_diarized_scenario_with_hq_text(
        scenario_with_roles = diar_segs_scenario,
        hq_text = hq_text
    )
    log.info(f"\n\nScenario corrected by LLM:\n{scenario_corrected}\n")

    diar_segs_turns_log_info = render_timestamped_script_from_turns(diar_segs_turns)
    log.info(f"\n\nFormatted turns from diarized segs before pre-process for interrupts:\n{diar_segs_turns_log_info}\n")  
    turns_for_interrupts = preprocess_turns_for_interrupts_analysis(
        diar_segs_turns,
        merge_gap_ms_ag=500,
        merge_gap_ms_cl=500,
        drop_backchannels=True,
    )
    interrupts_turns_log_info = render_timestamped_script_from_turns(turns_for_interrupts)
    log.info(f"\n\nFormatted turns from diarized segs for interrupts:\n{interrupts_turns_log_info}\n")  

    return turns_for_interrupts, scenario_corrected




















ROLE = Literal["AG", "CL"]

_CORRECT_SCENARIO_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "role": {"type": "string", "enum": ["AG", "CL"]},
                    "text": {"type": "string", "minLength": 1},
                },
                "required": ["role", "text"],
            },
            "minItems": 1,
        }
    },
    "required": ["turns"],
}


def _merge_adjacent_same_role(turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for t in turns:
        role = t["role"]
        text = t["text"].strip()
        if not text:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["text"] = (out[-1]["text"].rstrip() + " " + text).strip()
        else:
            out.append({"role": role, "text": text})
    return out


def _turns_to_scenario(turns: List[Dict[str, str]]) -> str:
    # Format like:
    # AG: ...
    # CL: ...
    lines = [f'{t["role"]}: {t["text"].strip()}' for t in turns if t["text"].strip()]
    return "\n".join(lines).strip()


async def async_correct_diarized_scenario_with_hq_text(
    scenario_with_roles: str,
    hq_text: str,
    temperature: float = 0.0,
) -> str:
    """
    Use a single LLM call to "correct" the role-labeled scenario using the better HQ transcript.

    Inputs:
      - scenario_with_roles: existing AG/CL scenario (may contain errors, duplicates, mistranscriptions)
      - hq_text: better continuous transcript (no roles)
      - model: chat model name for text reasoning

    Outputs:
      - List[Turn] with role mapped to "AGENT"/"CLIENT"
      - scenario string in AG:/CL: format

    Assumptions:
      - AG == agent, CL == client.
      - Use hq_text as ground truth wording; use scenario_with_roles mostly to infer who said what.
    """

    system = (
        "You are a conversation editor. You will be given:\n"
        "1) A role-labeled scenario (AG/CL) that may have ASR errors.\n"
        "2) A higher-quality transcript (HQ) of the same conversation with better wording but no roles.\n\n"
        "Task:\n"
        "- Produce a corrected role-labeled scenario.\n"
        "- Use the HQ transcript as the ground truth for wording.\n"
        "- Use the role scenario to infer speaker attribution.\n"
        "- Keep the original spoken language (Ukrainian/Russian mix). Do NOT translate.\n"
        "- Do NOT invent new content.\n"
        "- Split into short turns (natural back-and-forth). If uncertain, keep longer turns rather than guessing.\n"
        "- Remove duplicated phrases caused by ASR.\n"
        "- Preserve key details (names, numbers, amounts, dates).\n\n"
        "Output JSON only in the given schema."
    )

    user = (
        "ROLE-LABELED SCENARIO (noisy):\n"
        f"{scenario_with_roles}\n\n"
        "HQ TRANSCRIPT (better text, no roles):\n"
        f"{hq_text}\n\n"
        "Return corrected turns with roles AG/CL."
    )

    resp = await async_chat_completion_with_format(
        model= _default_split_into_roles_model(),
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format_schema=_CORRECT_SCENARIO_SCHEMA,
        schema_name="corrected_scenario",
    )

    # Parse structured response (works with OpenAI ChatCompletion objects)
    content = resp.choices[0].message.content
    log.info(f"\n\nResult of LLM fix:\n{resp}\n")

    data = json.loads(content) if isinstance(content, str) else content

    turns_raw: List[Dict[str, str]] = data["turns"]
    turns_raw = _merge_adjacent_same_role(turns_raw)

    scenario_corrected = _turns_to_scenario(turns_raw)
    return scenario_corrected












SCHEMA_MULTI_SPEAKER_ROLES = {
    "type": "object",
    "properties": {
        "roles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "role": {"type": "string", "enum": ["AGENT", "CLIENT"]},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reason": {"type": "string"},
                },
                "required": ["speaker", "role", "confidence", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["roles"],
    "additionalProperties": False,
}


async def async_classify_all_speakers_agent_or_client(
    segs: List["TranscriptionDiarizedSegment"],
    max_chars_per_speaker: int = 2400,
) -> Dict[str, str]:
    """
    Classify each diarized speaker as AGENT or CLIENT.

    Input: List[TranscriptionDiarizedSegment] (with fields: start, end, text, speaker)
    Output: mapping like {"A":"AGENT", "B":"CLIENT", ...}
    """
    if not segs:
        return {}

    # group by speaker
    by: Dict[str, List["TranscriptionDiarizedSegment"]] = {}
    for seg in segs:
        txt = str(_seg_get(seg, "text", "") or "").strip()
        if not txt:
            continue
        spk = str(_seg_get(seg, "speaker", "") or "S0").strip()
        by.setdefault(spk, []).append(seg)

    speakers = sorted(by.keys())
    if not speakers:
        return {}

    def speaker_stats(items: List["TranscriptionDiarizedSegment"]) -> Tuple[float, int]:
        dur = 0.0
        words = 0
        for seg in items:
            st = float(_seg_get(seg, "start", 0.0) or 0.0)
            en = float(_seg_get(seg, "end", 0.0) or 0.0)
            dur += max(0.0, en - st)
            words += len((str(_seg_get(seg, "text", "") or "")).split())
        return dur, words

    def speaker_excerpt(items: List["TranscriptionDiarizedSegment"], max_chars: int) -> str:
        # chronological, concatenate until max chars
        items_sorted = sorted(
            items,
            key=lambda s: (float(_seg_get(s, "start", 0.0) or 0.0), float(_seg_get(s, "end", 0.0) or 0.0)),
        )
        out: List[str] = []
        total = 0
        for seg in items_sorted:
            piece = str(_seg_get(seg, "text", "") or "").strip()
            if not piece:
                continue
            add = piece + "\n"
            if total + len(add) > max_chars:
                remaining = max_chars - total
                if remaining > 0:
                    out.append(add[:remaining])
                break
            out.append(add)
            total += len(add)
        return "".join(out).strip()

    blocks: List[str] = []
    for spk in speakers:
        dur, words = speaker_stats(by[spk])
        excerpt = speaker_excerpt(by[spk], max_chars=max_chars_per_speaker)
        blocks.append(
            f"SPEAKER {spk}\n"
            f"- total_duration_sec: {dur:.2f}\n"
            f"- approx_words: {words}\n"
            f"- excerpt:\n{excerpt}\n"
        )

    system_prompt = (
        "You classify speakers in bank calls.\n"
        "For EACH speaker, output role AGENT or CLIENT.\n"
        "Return JSON only."
    )
    user_prompt = (
        "Diarized speakers from one call.\n"
        f"Valid speaker ids are ONLY: {', '.join(speakers)}.\n"
        "In the JSON, 'speaker' MUST be exactly one of those ids (e.g., 'A', not 'SPEAKER A').\n\n"
        + "\n".join(blocks)
        + "\nReturn JSON: {\"roles\":[{\"speaker\":\"A\",\"role\":\"AGENT\",\"confidence\":\"high\",\"reason\":\"...\"},...]}"
    )

    model = _default_detect_speaker_role_model()
    resp = await async_chat_completion_with_format(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format_schema=SCHEMA_MULTI_SPEAKER_ROLES,
        schema_name="multi_speaker_roles",
        model=model,
        temperature=0.0,
    )

    content = resp.choices[0].message.content
    data = json.loads(content) if isinstance(content, str) else content
    log.info("\n\nRoles classification analysis result:\n" + str(data) +"\n")

    mapping: Dict[str, str] = {}
    for row in data.get("roles", []):
        spk = str(row.get("speaker", "")).strip()
        role = str(row.get("role", "")).strip().upper()
        if not spk:
            continue
        if role not in ("AGENT", "CLIENT"):
            continue
        mapping[spk] = role

    # ensure all speakers covered (fallback CLIENT)
    for spk in speakers:
        mapping.setdefault(spk, "CLIENT")

    return mapping








_BACKCHANNEL_RE = re.compile(
    r"^\s*(?:угу|ага|так|ок|окей|mhm+|мм+|угу-угу|так-так|ясно|зрозуміло)\s*[\.\!\?\,]*\s*$",
    re.IGNORECASE,
)

def _wc(s: str) -> int:
    return len([w for w in (s or "").strip().split() if w])

def _norm_role(role: str) -> Optional[str]:
    r = (role or "").strip().upper()
    # accept common variants from your pipelines (AGENT/CLIENT) :contentReference[oaicite:3]{index=3}
    if r in {"AG", "AGENT"}:
        return "AG"
    if r in {"CL", "CLIENT"}:
        return "CL"
    # if you sometimes keep diar speaker ids A/B, map if you want:
    if r == "A":
        return "AG"
    if r == "B":
        return "CL"
    return None

def preprocess_turns_for_interrupts_analysis(
    turns: List[Turn],
    *,
    # merge settings (continuous speech consolidation)
    merge_gap_ms_ag: int = 350,
    merge_gap_ms_cl: int = 350,
    overlap_tol_ms: int = 60,   # allow tiny negative gaps due to jitter
    join_with: str = " ",

    # cleanup / noise control
    min_turn_ms: int = 80,      # drop ultra-tiny fragments
    drop_backchannels: bool = True,
    backchannel_max_ms: int = 450,
    backchannel_max_words: int = 2,
    backchannel_re=_BACKCHANNEL_RE,
) -> List[Turn]:
    """
    Preprocess transcription turns for better overlap/interrupt detection.

    - Normalizes role to "AG"/"CL" so evaluation_interrupts.analyze_turn_overlaps works. :contentReference[oaicite:4]{index=4}
    - Optionally drops short backchannels (угу/так/ага/etc.) to reduce false positives.
    - Merges only ADJACENT same-role turns (in global time order) if gap is small.
      This avoids the problematic per-role merge that can span across the other speaker’s speech. :contentReference[oaicite:5]{index=5}
    """
    if not turns:
        return []

    # 1) normalize + basic filtering
    cleaned: List[Turn] = []
    for t in turns:
        role = _norm_role(getattr(t, "role", ""))
        if not role:
            continue

        st = float(getattr(t, "start", 0.0) or 0.0)
        en = float(getattr(t, "end", 0.0) or 0.0)
        if en <= st:
            continue

        txt = (getattr(t, "text", "") or "").strip()
        if not txt:
            continue

        dur_ms = int(round((en - st) * 1000.0))
        if dur_ms < min_turn_ms:
            continue

        # drop backchannels (optional)
        if drop_backchannels:
            wc = _wc(txt)
            if dur_ms <= backchannel_max_ms and wc <= backchannel_max_words and backchannel_re.match(txt):
                continue

        cleaned.append(
            Turn(
                role=role,
                start=st,
                end=en,
                text=txt,
                text_diar=getattr(t, "text_diar", ""),
                file=getattr(t, "file", ""),
            )
        )

    if not cleaned:
        return []

    # 2) sort globally
    cleaned.sort(key=lambda x: (float(x.start), float(x.end)))

    # 3) merge adjacent same-role turns when close enough
    out: List[Turn] = [cleaned[0]]
    for cur in cleaned[1:]:
        prev = out[-1]
        if cur.role != prev.role:
            out.append(cur)
            continue

        gap_ms = (float(cur.start) - float(prev.end)) * 1000.0
        role_gap = merge_gap_ms_ag if prev.role == "AG" else merge_gap_ms_cl

        # allow tiny overlap jitter (gap_ms can be slightly negative)
        if gap_ms <= role_gap and gap_ms >= -overlap_tol_ms:
            prev.end = max(float(prev.end), float(cur.end))
            prev.text = (prev.text.rstrip() + join_with + cur.text.lstrip()).strip()
            # keep diar text if you want (optional)
            if getattr(prev, "text_diar", None) is not None:
                prev.text_diar = (str(prev.text_diar or "").rstrip() + join_with + str(cur.text_diar or "").lstrip()).strip()
        else:
            out.append(cur)

    return out