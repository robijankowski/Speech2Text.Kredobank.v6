import json
import re

from core.config import settings

import logging
from core.config import settings

log = logging.getLogger(settings.TR_LOGGER_NAME)

from openai_tools.openai_client_text import  chat_completion_with_format
from typing import Any, Dict, List, Tuple, Optional

# Schema: classify EACH detected speaker as AGENT or CLIENT
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


def _extract_diarized_segments_any(diarized: Any) -> List[Dict[str, Any]]:
    """
    Returns normalized list of dicts: {speaker, start, end, text}
    Supports:
      - object with .segments
      - dict with ["segments"]
      - string repr: "TranscriptionDiarized(... segments=[TranscriptionDiarizedSegment(...), ...])"
    """
    # 1) object with .segments
    segs = getattr(diarized, "segments", None)
    if isinstance(segs, list):
        out = []
        for s in segs:
            out.append(
                {
                    "speaker": getattr(s, "speaker", None) or getattr(s, "speaker_id", None) or "S0",
                    "start": getattr(s, "start", None),
                    "end": getattr(s, "end", None),
                    "text": (getattr(s, "text", "") or "").strip(),
                }
            )
        return [x for x in out if x["text"]]

    # 2) dict-like
    if isinstance(diarized, dict) and isinstance(diarized.get("segments"), list):
        out = []
        for s in diarized["segments"]:
            if not isinstance(s, dict):
                continue
            out.append(
                {
                    "speaker": s.get("speaker") or s.get("speaker_id") or "S0",
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "text": (s.get("text") or "").strip(),
                }
            )
        return [x for x in out if x["text"]]

    # 3) string repr (best-effort)
    if isinstance(diarized, str):
        import re

        seg_chunks = re.findall(r"TranscriptionDiarizedSegment\((.*?)\)", diarized, flags=re.DOTALL)
        out: List[Dict[str, Any]] = []
        for ch in seg_chunks:
            # speaker='A'
            m_spk = re.search(r"speaker='([^']+)'", ch)
            spk = m_spk.group(1) if m_spk else "S0"

            # start=..., end=...
            m_start = re.search(r"start=([0-9.]+)", ch)
            m_end = re.search(r"end=([0-9.]+)", ch)
            start = float(m_start.group(1)) if m_start else None
            end = float(m_end.group(1)) if m_end else None

            # text='...' or text="..."
            m_txt1 = re.search(r"text='([^']*)'", ch)
            m_txt2 = re.search(r'text="([^"]*)"', ch)
            txt = (m_txt1.group(1) if m_txt1 else (m_txt2.group(1) if m_txt2 else "")).strip()
            if txt:
                out.append({"speaker": spk, "start": start, "end": end, "text": txt})
        return out

    return []


def _speaker_stats(segs: List[Dict[str, Any]]) -> Tuple[float, int]:
    dur = 0.0
    words = 0
    for s in segs:
        try:
            st = float(s.get("start") or 0.0)
            en = float(s.get("end") or 0.0)
            if en > st:
                dur += (en - st)
        except Exception:
            pass
        words += len((s.get("text") or "").split())
    return dur, words


def _build_speaker_excerpt(
    speaker_segs: List[Dict[str, Any]],
    *,
    max_chars: int = 2400,
    head_n: int = 8,
    tail_n: int = 6,
) -> str:
    """
    Good excerpt for role detection:
      - first N turns (often greeting + intro)
      - last N turns (often closure / payment / procedure)
      - then clipped to max_chars
    """
    speaker_segs = [s for s in speaker_segs if (s.get("text") or "").strip()]
    if not speaker_segs:
        return ""

    head = speaker_segs[:head_n]
    tail = speaker_segs[-tail_n:] if len(speaker_segs) > head_n else []
    merged = head + ([{"text": "…"}] if tail else []) + tail

    txt = " ".join((s.get("text") or "").strip() for s in merged if (s.get("text") or "").strip())
    txt = " ".join(txt.split())
    return txt[:max_chars]



def _normalize_speaker_id(spk: str) -> str:
    s = (spk or "").strip()
    # "SPEAKER A" / "Speaker A" -> "A"
    s = re.sub(r"^(speaker)\s+", "", s, flags=re.IGNORECASE).strip()
    # "A:" -> "A"
    s = s.rstrip(":").strip()
    return s



def classify_all_speakers_agent_or_client(
    diarized: Any,
    max_chars_per_speaker: int = 2400,
) -> Dict[str, str]:
    """
    Identify ALL speakers in diarized transcription and classify each as AGENT or CLIENT.
    Returns mapping: { "A": "AG", "B": "CL", ... }

    Uses ONE LLM call with structured output (same style as classify_agent_or_client_prefix).
    """
    segments = _extract_diarized_segments_any(diarized)
    if not segments:
        return {}

    # group by speaker
    by: Dict[str, List[Dict[str, Any]]] = {}
    for s in segments:
        spk = str(s.get("speaker") or "S0")
        by.setdefault(spk, []).append(s)

    speakers = sorted(by.keys())

    # create per-speaker summary for the prompt
    blocks: List[str] = []
    for spk in speakers:
        dur, words = _speaker_stats(by[spk])
        excerpt = _build_speaker_excerpt(by[spk], max_chars=max_chars_per_speaker)
        blocks.append(
            f"SPEAKER {spk}\n"
            f"- total_duration_sec: {dur:.2f}\n"
            f"- approx_words: {words}\n"
            f"- excerpt:\n{excerpt}\n"
        )

    system_prompt = (
        "You are an expert at identifying speaker roles in bank collection calls.\n"
        "You will be given multiple diarized speakers (A/B/S0/S1/etc.).\n"
        "Classify EACH speaker as either AGENT (bank employee) or CLIENT (customer).\n\n"
        "CRITICAL:\n"
        "- Do NOT translate the text.\n"
        "- Do NOT invent missing content.\n"
        "- Base the decision on tone and typical bank call behavior.\n"
        "- If there are multiple bank staff speakers, it is OK to label multiple as AGENT.\n"
    )

    user_prompt = (
        "Diarized speakers from one phone call:\n\n"
        + "\n".join(blocks)
        + "\nReturn structured JSON with a role for every speaker listed."
    )


    model = (
            settings.AZURE_MODEL_CHAT_TRS_DETECT_PLAYER
            if settings.USE_AZURE_OPENAI == "Y"
            else settings.OPENAI_MODEL_CHAT_TRS_DETECT_PLAYER
        )

    resp = chat_completion_with_format(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        format_schema=SCHEMA_MULTI_SPEAKER_ROLES,
        schema_name="multi_speaker_roles",
        model=model,
        temperature=0.0,
    )


    data = json.loads(resp.choices[0].message.content)

    mapping: Dict[str, str] = {}
    for row in data.get("roles", []):
        spk_raw = str(row.get("speaker", ""))
        spk = _normalize_speaker_id(spk_raw)

        role_raw = str(row.get("role", "")).strip().upper()
        role = "AGENT" if role_raw == "AGENT" else "CLIENT"

        if spk in by:
            mapping[spk] = role
        else:
            log.warning(
                f"LLM returned unknown speaker id: '{spk_raw}' (normalized='{spk}'). Known speakers: {list(by.keys())}"
            )

    # ensure all speakers covered (fallback: CLIENT)
    for spk in speakers:
        mapping.setdefault(spk, "CLIENT")

    log.info(f"Role mapping: {mapping}")
    return mapping









