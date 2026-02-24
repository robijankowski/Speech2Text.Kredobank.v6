from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from openai import OpenAI

from app.core.config import settings    

openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)



def transcript_audio_file_verbose_o4_diarize(
    file_name,
    temperature=0.0,
    language="uk",
    chunking_strategy="auto",
):
    """
    Uses gpt-4o-transcribe-diarize and prints diarization markers (speaker, start, end) per segment.
    Note: diarize model returns speaker annotations only with response_format="diarized_json"
    and does NOT support prompt or timestamp_granularities. :contentReference[oaicite:2]{index=2}
    """
    model = "gpt-4o-transcribe-diarize"
    print(f"Transcribing file: {file_name} using '{model}'")

    def _fmt_ts(seconds: float) -> str:
        # 00:00.00 formatting
        if seconds is None:
            return "??:??.??"
        m, s = divmod(float(seconds), 60.0)
        return f"{int(m):02d}:{s:05.2f}"

    def _get(obj, key, default=None):
        # supports dicts or objects
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)


    with open(file_name, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            response_format="diarized_json",
            # language=language,
            temperature=temperature,
            chunking_strategy=chunking_strategy
        )

    # Print usage if present (guarded)
    usage = _get(transcript, "usage")
    if usage is not None:
        print("Usage:", usage)

    # Print ALL diarization markers (speaker/start/end/text)
    segments = _get(transcript, "segments", []) or []
    print(f"Segments: {len(segments)}")
    print(str(segments))
    for seg in segments:
        print(str(seg))
    speakers_seen = set()

    for i, seg in enumerate(segments, start=1):
        speaker = _get(seg, "speaker", "unknown")
        start = _get(seg, "start", None)
        end = _get(seg, "end", None)
        text = _get(seg, "text", "")

        speakers_seen.add(speaker)
        print(f"[{i:04d}] {speaker} {_fmt_ts(start)}–{_fmt_ts(end)}: {text}")

    print("Speakers found:", ", ".join(map(str, sorted(speakers_seen))))
    
    return segments




@dataclass
class InterruptionEvent:
    agent_start: float
    agent_end: float
    client_start: float
    client_end: float
    overlap_sec: float
    client_lead_sec: float
    agent_text: str
    client_text: str
    agent_seg_id: Optional[str] = None
    client_seg_id: Optional[str] = None


def _seg_get(seg: Any, key: str, default: Any = None) -> Any:
    """Obsługuje seg jako obiekt (attrs) albo dict."""
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def _normalize_diarized_segments(segments: List[Any]) -> List[Dict[str, Any]]:
    """
    Normalizuje diarized segmenty do listy dictów:
    {"id": str|None, "speaker": str, "start": float, "end": float, "text": str}
    """
    out: List[Dict[str, Any]] = []
    for s in segments:
        start = float(_seg_get(s, "start", 0.0))
        end = float(_seg_get(s, "end", 0.0))
        speaker = str(_seg_get(s, "speaker", "") or "")
        text = str(_seg_get(s, "text", "") or "").strip()
        seg_id = _seg_get(s, "id", None)
        if seg_id is not None:
            seg_id = str(seg_id)

        if not speaker:
            continue
        if end <= start:
            continue

        out.append(
            {
                "id": seg_id,
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
            }
        )

    out.sort(key=lambda x: x["start"])
    return out


def detect_agent_interruptions_from_diarized_segments(
    diarized_segments: List[Any],
    *,
    agent_speaker: str = "A",
    min_overlap_ms: int = 250,
    min_client_lead_ms: int = 300,
    min_agent_segment_ms: int = 200,
    ignore_short_agent_backchannels: bool = False,
    backchannel_words: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Wykrywa "wchodzenie w słowo klientowi" na podstawie diarized segmentów.

    Definicja zdarzenia:
      - agent startuje w trakcie segmentu klienta (client.start + min_client_lead <= agent.start < client.end)
      - overlap = min(agent.end, client.end) - agent.start >= min_overlap
      - agent segment ma długość >= min_agent_segment

    Parametry opcjonalne:
      - ignore_short_agent_backchannels: jeśli True, to dodatkowo odfiltruje krótkie wtrącenia na podstawie słów (np. "tak", "mhm")
      - backchannel_words: lista słów (lowercase), domyślna jeśli None

    Zwraca:
      {
        "events": [ ... ],
        "stats": { ... }
      }
    """
    segs = _normalize_diarized_segments(diarized_segments)

    min_overlap_sec = min_overlap_ms / 1000.0
    min_client_lead_sec = min_client_lead_ms / 1000.0
    min_agent_seg_sec = min_agent_segment_ms / 1000.0

    if backchannel_words is None:
        backchannel_words = ["mhm", "aha", "tak", "no", "угу", "угу.", "да", "ок", "okay", "добре", "дякую"]

    def is_backchannel(text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return True
        # bardzo prosta heurystyka: 1-2 krótkie tokeny i w znanym słowniku
        tokens = [x for x in t.replace(",", " ").replace(".", " ").split() if x]
        if len(tokens) <= 2 and (" ".join(tokens) in backchannel_words or any(tok in backchannel_words for tok in tokens)):
            return True
        return False

    # rozdziel listy
    agent_segs = [s for s in segs if s["speaker"] == agent_speaker]
    client_segs = [s for s in segs if s["speaker"] != agent_speaker]

    events: List[InterruptionEvent] = []

    j = 0
    nA = len(agent_segs)

    for c in client_segs:
        c_s, c_e, c_text = c["start"], c["end"], c["text"]

        # przesuń indeks agenta do możliwego przecięcia
        while j < nA and agent_segs[j]["end"] <= c_s:
            j += 1

        k = j
        while k < nA and agent_segs[k]["start"] < c_e:
            a = agent_segs[k]
            a_s, a_e, a_text = a["start"], a["end"], a["text"]

            # filtr długości segmentu agenta
            if (a_e - a_s) < min_agent_seg_sec:
                k += 1
                continue

            # opcjonalny filtr backchannel
            if ignore_short_agent_backchannels and is_backchannel(a_text):
                k += 1
                continue

            # agent startuje w trakcie wypowiedzi klienta (po minimalnym leadzie klienta)
            if (c_s + min_client_lead_sec) <= a_s < c_e:
                overlap = min(a_e, c_e) - a_s
                if overlap >= min_overlap_sec:
                    events.append(
                        InterruptionEvent(
                            agent_start=a_s,
                            agent_end=a_e,
                            client_start=c_s,
                            client_end=c_e,
                            overlap_sec=round(overlap, 3),
                            client_lead_sec=round(a_s - c_s, 3),
                            agent_text=a_text,
                            client_text=c_text,
                            agent_seg_id=a.get("id"),
                            client_seg_id=c.get("id"),
                        )
                    )

            k += 1

    total_overlap = sum(e.overlap_sec for e in events)

    stats = {
        "agent_speaker": agent_speaker,
        "events_count": len(events),
        "total_overlap_sec": round(total_overlap, 3),
        "avg_overlap_sec": round(total_overlap / len(events), 3) if events else 0.0,
        "min_overlap_ms": min_overlap_ms,
        "min_client_lead_ms": min_client_lead_ms,
        "min_agent_segment_ms": min_agent_segment_ms,
        "ignore_short_agent_backchannels": ignore_short_agent_backchannels,
    }

    return {"events": [asdict(e) for e in events], "stats": stats}


# segments = transcript_audio_file_verbose_o4_diarize("./test/test_call.wav")
# result = detect_agent_interruptions_from_diarized_segments(
#     diarized_segments=segments,
#     agent_speaker="A",
#     min_overlap_ms=300,
#     min_client_lead_ms=300,
#     min_agent_segment_ms=200,
#     ignore_short_agent_backchannels=True,  # opcjonalnie
# )

# print(result["stats"])
# for e in result["events"][:5]:
#     print(e["client_start"], e["agent_start"], e["overlap_sec"], e["client_text"], "||", e["agent_text"])


segments = transcript_audio_file_verbose_o4_diarize("./test/Одночасна розмова фахівця і клієнта.wav")
result = detect_agent_interruptions_from_diarized_segments(
    diarized_segments=segments,
    agent_speaker="A",
    min_overlap_ms=300,
    min_client_lead_ms=300,
    min_agent_segment_ms=200,
    ignore_short_agent_backchannels=True,  # opcjonalnie
)

print(result["stats"])
for e in result["events"][:5]:
    print(e["client_start"], e["agent_start"], e["overlap_sec"], e["client_text"], "||", e["agent_text"])


# segments = transcript_audio_file_verbose_o4_diarize("./test/Святослав одночасна розмова з клієнтом , деколи перебиває.wav")
# result = detect_agent_interruptions_from_diarized_segments(
#     diarized_segments=segments,
#     agent_speaker="A",
#     min_overlap_ms=300,
#     min_client_lead_ms=300,
#     min_agent_segment_ms=200,
#     ignore_short_agent_backchannels=True,  # opcjonalnie
# )

# print(result["stats"])
# for e in result["events"][:5]:
#     print(e["client_start"], e["agent_start"], e["overlap_sec"], e["client_text"], "||", e["agent_text"])



