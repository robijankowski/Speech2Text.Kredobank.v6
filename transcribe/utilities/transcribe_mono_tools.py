import json
import re

from core.config import settings
from core.logger import log

from openai_tools.openai_client_text import  async_chat_completion_with_format
from typing import Any, Dict, List, Tuple, Optional

from transcribe.utilities.scenario_tools import DiarSeg, _default_detect_speaker_role_model

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


# def _extract_diarized_segments_any(diarized: Any) -> List[Dict[str, Any]]:
#     """
#     Returns normalized list of dicts: {speaker, start, end, text}
#     Supports:
#       - object with .segments
#       - dict with ["segments"]
#       - string repr: "TranscriptionDiarized(... segments=[TranscriptionDiarizedSegment(...), ...])"
#     """
#     # 1) object with .segments
#     segs = getattr(diarized, "segments", None)
#     if isinstance(segs, list):
#         out = []
#         for s in segs:
#             out.append(
#                 {
#                     "speaker": getattr(s, "speaker", None) or getattr(s, "speaker_id", None) or "S0",
#                     "start": getattr(s, "start", None),
#                     "end": getattr(s, "end", None),
#                     "text": (getattr(s, "text", "") or "").strip(),
#                 }
#             )
#         return [x for x in out if x["text"]]

#     # 2) dict-like
#     if isinstance(diarized, dict) and isinstance(diarized.get("segments"), list):
#         out = []
#         for s in diarized["segments"]:
#             if not isinstance(s, dict):
#                 continue
#             out.append(
#                 {
#                     "speaker": s.get("speaker") or s.get("speaker_id") or "S0",
#                     "start": s.get("start"),
#                     "end": s.get("end"),
#                     "text": (s.get("text") or "").strip(),
#                 }
#             )
#         return [x for x in out if x["text"]]

#     # 3) string repr (best-effort)
#     if isinstance(diarized, str):
#         import re

#         seg_chunks = re.findall(r"TranscriptionDiarizedSegment\((.*?)\)", diarized, flags=re.DOTALL)
#         out: List[Dict[str, Any]] = []
#         for ch in seg_chunks:
#             # speaker='A'
#             m_spk = re.search(r"speaker='([^']+)'", ch)
#             spk = m_spk.group(1) if m_spk else "S0"

#             # start=..., end=...
#             m_start = re.search(r"start=([0-9.]+)", ch)
#             m_end = re.search(r"end=([0-9.]+)", ch)
#             start = float(m_start.group(1)) if m_start else None
#             end = float(m_end.group(1)) if m_end else None

#             # text='...' or text="..."
#             m_txt1 = re.search(r"text='([^']*)'", ch)
#             m_txt2 = re.search(r'text="([^"]*)"', ch)
#             txt = (m_txt1.group(1) if m_txt1 else (m_txt2.group(1) if m_txt2 else "")).strip()
#             if txt:
#                 out.append({"speaker": spk, "start": start, "end": end, "text": txt})
#         return out

#     return []


# def _speaker_stats(segs: List[Dict[str, Any]]) -> Tuple[float, int]:
#     dur = 0.0
#     words = 0
#     for s in segs:
#         try:
#             st = float(s.get("start") or 0.0)
#             en = float(s.get("end") or 0.0)
#             if en > st:
#                 dur += (en - st)
#         except Exception:
#             pass
#         words += len((s.get("text") or "").split())
#     return dur, words


# def _build_speaker_excerpt(
#     speaker_segs: List[Dict[str, Any]],
#     *,
#     max_chars: int = 2400,
#     head_n: int = 8,
#     tail_n: int = 6,
# ) -> str:
#     """
#     Good excerpt for role detection:
#       - first N turns (often greeting + intro)
#       - last N turns (often closure / payment / procedure)
#       - then clipped to max_chars
#     """
#     speaker_segs = [s for s in speaker_segs if (s.get("text") or "").strip()]
#     if not speaker_segs:
#         return ""

#     head = speaker_segs[:head_n]
#     tail = speaker_segs[-tail_n:] if len(speaker_segs) > head_n else []
#     merged = head + ([{"text": "…"}] if tail else []) + tail

#     txt = " ".join((s.get("text") or "").strip() for s in merged if (s.get("text") or "").strip())
#     txt = " ".join(txt.split())
#     return txt[:max_chars]



# def _normalize_speaker_id(spk: str) -> str:
#     s = (spk or "").strip()
#     # "SPEAKER A" / "Speaker A" -> "A"
#     s = re.sub(r"^(speaker)\s+", "", s, flags=re.IGNORECASE).strip()
#     # "A:" -> "A"
#     s = s.rstrip(":").strip()
#     return s



async def async_classify_all_speakers_agent_or_client(
    segs: List[DiarSeg],
    max_chars_per_speaker: int = 2400,
) -> Dict[str, str]:
    """
    Classify each diarized speaker as AG (agent) or CL (client).
    Input: List[DiarSeg] = (start, end, text, speaker)
    Output: mapping like {"A":"AGENT", "B":"CLIENT", ...}
    """
    if not segs:
        return {}

    # group by speaker
    by: Dict[str, List[DiarSeg]] = {}
    for st, en, txt, spk in segs:
        txt = (txt or "").strip()
        if not txt:
            continue
        spk = str(spk or "S0").strip()
        by.setdefault(spk, []).append((float(st), float(en), txt, spk))

    speakers = sorted(by.keys())
    if not speakers:
        return {}

    def speaker_stats(items: List[DiarSeg]) -> Tuple[float, int]:
        dur = 0.0
        words = 0
        for st, en, txt, _ in items:
            dur += max(0.0, float(en) - float(st))
            words += len((txt or "").split())
        return dur, words

    def speaker_excerpt(items: List[DiarSeg], max_chars: int) -> str:
        # chronological, concatenate until max chars
        items = sorted(items, key=lambda x: (x[0], x[1]))
        out = []
        total = 0
        for _, _, txt, _ in items:
            if not txt:
                continue
            piece = txt.strip()
            if not piece:
                continue
            add = (piece + "\n")
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
        "Diarized speakers from one call:\n\n"
        + "\n".join(blocks)
        + "\nReturn JSON: {\"roles\":[{\"speaker\":\"A\",\"role\":\"AGENT\"},...]}"
    )

    # keep your existing schema + call helper
    model = _default_detect_speaker_role_model()
    resp = await async_chat_completion_with_format(
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        format_schema=SCHEMA_MULTI_SPEAKER_ROLES,
        schema_name="multi_speaker_roles",
        model=model,
        temperature=0.0,
    )

    data = json.loads(resp.choices[0].message.content)

    mapping: Dict[str, str] = {}
    for row in data.get("roles", []):
        spk = str(row.get("speaker", "")).strip()
        role = str(row.get("role", "")).strip().upper()
        if not spk:
            continue
        mapping[spk] = "AGENT" if role == "AGENT" else "CLIENT"

    # ensure all speakers covered (fallback CL)
    for spk in speakers:
        mapping.setdefault(spk, "CLIENT")

    return mapping




