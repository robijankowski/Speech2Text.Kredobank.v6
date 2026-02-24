from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple

# Twoje moduły z projektu:
from app.transcribe.utlities.audio_tools import split_stereo_to_lr_and_clean
from app.transcribe.utlities.transcribe_stereo_tools import async_transcript_audio_file_verbose_o4_single_channel
from openai import OpenAI


ChannelSide = Literal["L", "R"]

from app.core.config import settings    

openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

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


def _get_segments(transcription: Any) -> List[Dict[str, Any]]:
    """
    Wyciąga listę segmentów w formacie: [{"start": float, "end": float, "text": str}, ...]
    Obsługuje transkrypcję jako obiekt (np. pydantic) albo dict.
    """
    segs = None

    # obiekt (np. transcription.segments)
    if hasattr(transcription, "segments"):
        segs = getattr(transcription, "segments")
    # dict
    elif isinstance(transcription, dict) and "segments" in transcription:
        segs = transcription["segments"]

    if not segs:
        return []

    out: List[Dict[str, Any]] = []
    for s in segs:
        # s może być dict albo obiektem
        if isinstance(s, dict):
            start = float(s.get("start", 0.0))
            end = float(s.get("end", 0.0))
            text = str(s.get("text", "") or "").strip()
        else:
            start = float(getattr(s, "start", 0.0))
            end = float(getattr(s, "end", 0.0))
            text = str(getattr(s, "text", "") or "").strip()

        if end > start:  # odsiecz śmieci
            out.append({"start": start, "end": end, "text": text})

    out.sort(key=lambda x: x["start"])
    return out


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



def detect_agent_interruptions_from_stereo_wav(
    audio_file: str,
    *,
    agent_channel: ChannelSide,   # "L" albo "R"
    min_overlap_ms: int = 250,     # minimalny overlap, żeby uznać zdarzenie
    min_client_lead_ms: int = 300, # klient musi mówić co najmniej tyle zanim agent wejdzie
    min_agent_segment_ms: int = 200,  # ignoruj ultra-krótkie segmenty agenta (np. pojedyncze "mhm")
    o4_metadata_text: str = "",
    temperature: float = 0.0,
    model: str = "",
) -> Dict[str, Any]:
    """
    Wykrywa sytuacje, gdy agent zaczyna mówić w trakcie wypowiedzi klienta (overlap).

    Zwraca:
      {
        "events": [InterruptionEvent as dict...],
        "stats": {...},
        "files": {"left": "...", "right": "..."}
      }
    """
    # 1) Split stereo -> mono L/R (tu NIE ma wycinania ciszy, timeline zostaje)
    left_wav, right_wav = split_stereo_to_lr_and_clean(audio_file)
    if not left_wav or not right_wav:
        raise ValueError("Plik nie jest stereo albo nie udało się rozdzielić kanałów.")

    agent_wav = left_wav if agent_channel == "L" else right_wav
    client_wav = right_wav if agent_channel == "L" else left_wav

    # 2) Transkrypcja segmentowa obu kanałów
    # tr_agent = transcript_audio_file_verbose_o4_single_channel(
    #     file_name=agent_wav,
    #     o4_metadata_text=o4_metadata_text,
    #     temperature=temperature,
    #     model=model,
    # )
    tr_agent = transcript_audio_file_verbose_o4_diarize(file_name=agent_wav)
    print(f"Agent transcription done. {tr_agent}")


    # tr_client = transcript_audio_file_verbose_o4_single_channel(
    #     file_name=client_wav,
    #     o4_metadata_text=o4_metadata_text,
    #     temperature=temperature,
    #     model=model,
    # )
    tr_client = transcript_audio_file_verbose_o4_diarize(file_name=client_wav)
    print(f"Client transcription done. {tr_client}")
    
    agent_segs = _get_segments(tr_agent)
    client_segs = _get_segments(tr_client)

    # 3) Wykrywanie overlapów "agent wchodzi na klienta"
    min_overlap_sec = min_overlap_ms / 1000.0
    min_client_lead_sec = min_client_lead_ms / 1000.0
    min_agent_seg_sec = min_agent_segment_ms / 1000.0

    events: List[InterruptionEvent] = []

    # indeks do agent_segs, żeby nie skanować od zera za każdym razem
    j = 0
    nA = len(agent_segs)

    for c in client_segs:
        c_s, c_e, c_text = c["start"], c["end"], c["text"]

        # przesuń j do pierwszego segmentu agenta, który może się przecinać z tym segmentem klienta
        while j < nA and agent_segs[j]["end"] <= c_s:
            j += 1

        k = j
        # sprawdzaj segmenty agenta, które zaczynają się zanim skończy się segment klienta
        while k < nA and agent_segs[k]["start"] < c_e:
            a = agent_segs[k]
            a_s, a_e, a_text = a["start"], a["end"], a["text"]

            # filtr: segment agenta musi być sensownej długości
            if (a_e - a_s) < min_agent_seg_sec:
                k += 1
                continue

            # kluczowa definicja "przerwania": agent STARTUJE w trakcie wypowiedzi klienta
            # oraz klient mówił już minimalnie długo przed wejściem agenta
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
                        )
                    )

            k += 1

    # 4) Statystyki
    total_overlap = sum(e.overlap_sec for e in events)
    stats = {
        "events_count": len(events),
        "total_overlap_sec": round(total_overlap, 3),
        "avg_overlap_sec": round(total_overlap / len(events), 3) if events else 0.0,
        "min_overlap_ms": min_overlap_ms,
        "min_client_lead_ms": min_client_lead_ms,
        "min_agent_segment_ms": min_agent_segment_ms,
        "agent_channel": agent_channel,
    }

    return {
        "events": [asdict(e) for e in events],
        "stats": stats,
        "files": {"left": left_wav, "right": right_wav},
    }


# result = detect_agent_interruptions_from_stereo_wav(
#     "./test/test_call.wav",
#     agent_channel="L",   # jeśli agent jest na prawym kanale
#     min_overlap_ms=300,
#     min_client_lead_ms=400,
# )

# print(result["stats"])
# print(result["events"][:3])

result = detect_agent_interruptions_from_stereo_wav(
    "./test/Одночасна розмова фахівця і клієнта.wav",
    agent_channel="L",   # jeśli agent jest na prawym kanale
    min_overlap_ms=300,
    min_client_lead_ms=400,
)

print(result["stats"])
print(result["events"][:3])


# result = detect_agent_interruptions_from_stereo_wav(
#     "./test/Святослав одночасна розмова з клієнтом , деколи перебиває.wav",
#     agent_channel="L",   # jeśli agent jest na prawym kanale
#     min_overlap_ms=300,
#     min_client_lead_ms=400,
# )

# print(result["stats"])
# print(result["events"][:3])