from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from openai import OpenAI

from app.core.config import settings
from app.transcribe.utilities.audio_tools import split_stereo_to_lr_and_clean

ChannelSide = Literal["L", "R"]

openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)


@dataclass
class InterruptionEvent:
    starter_start: float
    starter_end: float
    other_start: float
    other_end: float
    overlap_sec: float
    other_lead_sec: float
    starter_text: str
    other_text: str


def fmt_ts(seconds: float) -> str:
    if seconds is None:
        return "??:??.??"
    m, s = divmod(float(seconds), 60.0)
    return f"{int(m):02d}:{s:05.2f}"

def merged_dialogue_text(
    left_segs: List[Dict[str, Any]],
    right_segs: List[Dict[str, Any]],
    *,
    left_label: str = "AG",
    right_label: str = "CL",
    include_timestamps: bool = False,
) -> str:
    """
    Merge left/right segments into one time-ordered dialogue string.
    Output example:
      AG: ...
      CL: ...
    """
    items: List[Dict[str, Any]] = []

    for s in left_segs:
        items.append({"start": float(s["start"]), "end": float(s["end"]), "text": s["text"], "label": left_label})
    for s in right_segs:
        items.append({"start": float(s["start"]), "end": float(s["end"]), "text": s["text"], "label": right_label})

    items.sort(key=lambda x: (x["start"], x["end"]))

    lines: List[str] = []
    for it in items:
        txt = str(it["text"]).strip()
        if not txt:
            continue
        if include_timestamps:
            lines.append(f"{it['label']} {fmt_ts(it['start'])}–{fmt_ts(it['end'])}: {txt}")
        else:
            lines.append(f"{it['label']}: {txt}")

    return "\n".join(lines)


def format_channel_segments(segs: List[Dict[str, Any]], label: str) -> str:
    lines = [f"\n=== {label} channel segments ({len(segs)}) ==="]
    for s in segs:
        lines.append(f"{fmt_ts(s['start'])}–{fmt_ts(s['end'])}  {s['text']}")
    return "\n".join(lines)


def _get_segments(transcription: Any) -> List[Dict[str, Any]]:
    """
    Normalize segments to:
      [{"start": float, "end": float, "text": str}, ...]
    Accepts:
      - list of segment dicts/objs  (your diarize function returns a list)
      - dict with "segments"
      - object with .segments
    """
    segs = None

    if isinstance(transcription, list):
        segs = transcription
    elif hasattr(transcription, "segments"):
        segs = getattr(transcription, "segments")
    elif isinstance(transcription, dict) and "segments" in transcription:
        segs = transcription["segments"]

    if not segs:
        return []

    out: List[Dict[str, Any]] = []
    for s in segs:
        if isinstance(s, dict):
            start = float(s.get("start", 0.0))
            end = float(s.get("end", 0.0))
            text = str(s.get("text", "") or "").strip()
        else:
            start = float(getattr(s, "start", 0.0))
            end = float(getattr(s, "end", 0.0))
            text = str(getattr(s, "text", "") or "").strip()

        if end > start and text:
            out.append({"start": start, "end": end, "text": text})

    out.sort(key=lambda x: x["start"])
    return out


def transcript_audio_file_verbose_o4_diarize(
    file_name: str,
    temperature: float = 0.0,
    chunking_strategy: str = "auto",
) -> List[Dict[str, Any]]:
    """
    Uses gpt-4o-transcribe-diarize.
    Returns LIST of segments (dict-like) with fields including: speaker, start, end, text.
    """
    model = settings.OPENAI_MODEL_TRANSCRIBE_DIARIZE
    print(f"Transcribing file: {file_name} using '{model}'")

    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    with open(file_name, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            response_format="diarized_json",
            temperature=temperature,
            chunking_strategy=chunking_strategy,
        )

    usage = _get(transcript, "usage")
    if usage is not None:
        print("Usage:", usage)

    segments = _get(transcript, "segments", []) or []
    print(f"Segments: {len(segments)}")

    # Debug print (optional; comment out if too noisy)
    # for seg in segments:
    #     print(str(seg))

    speakers_seen = set()
    for i, seg in enumerate(segments, start=1):
        speaker = _get(seg, "speaker", "unknown")
        start = _get(seg, "start", None)
        end = _get(seg, "end", None)
        text = _get(seg, "text", "")
        speakers_seen.add(speaker)
        # print(f"[{i:04d}] {speaker} {fmt_ts(start)}–{fmt_ts(end)}: {text}")

    print("Speakers found:", ", ".join(map(str, sorted(speakers_seen))))
    return segments


def detect_starts_while_other_speaks(
    starter_segs: List[Dict[str, Any]],
    other_segs: List[Dict[str, Any]],
    *,
    min_overlap_ms: int = 250,
    min_starter_segment_ms: int = 200,
    min_other_lead_ms: int = 0,
    eps_ms: int = 30,
) -> List[Dict[str, Any]]:
    """
    Events where STARTER starts while OTHER is ongoing:
      other.start + lead <= starter.start < other.end
    and overlap >= threshold.
    """
    min_overlap_sec = min_overlap_ms / 1000.0
    min_starter_seg_sec = min_starter_segment_ms / 1000.0
    min_other_lead_sec = min_other_lead_ms / 1000.0
    eps = eps_ms / 1000.0

    events: List[Dict[str, Any]] = []
    j = 0
    nS = len(starter_segs)

    for o in other_segs:
        o_s, o_e, o_text = o["start"], o["end"], o["text"]

        while j < nS and starter_segs[j]["end"] <= o_s + eps:
            j += 1

        k = j
        while k < nS and starter_segs[k]["start"] < o_e + eps:
            st = starter_segs[k]
            st_s, st_e, st_text = st["start"], st["end"], st["text"]

            if (st_e - st_s) < min_starter_seg_sec:
                k += 1
                continue

            if (o_s + min_other_lead_sec) <= st_s < (o_e + eps):
                overlap = min(st_e, o_e) - st_s
                if overlap + eps >= min_overlap_sec:
                    events.append(
                        {
                            "starter_start": st_s,
                            "starter_end": st_e,
                            "other_start": o_s,
                            "other_end": o_e,
                            "overlap_sec": round(overlap, 3),
                            "other_lead_sec": round(st_s - o_s, 3),
                            "starter_text": st_text,
                            "other_text": o_text,
                        }
                    )
            k += 1

    return events


def _word_count(text: str) -> int:
    # proste liczenie słów: split po whitespace
    return len([w for w in (text or "").strip().split() if w])


    
def detect_any_overlaps(
    left: List[Dict[str, Any]],
    right: List[Dict[str, Any]],
    *,
    min_overlap_ms: int = 200,
    eps_ms: int = 30,
    min_segment_ms_left: int = 0,
    min_segment_ms_right: int = 0,
    min_words_left: int = 0,
    min_words_right: int = 0,
) -> List[Dict[str, Any]]:
    """
    Wykrywa *wszystkie* nakładania się mowy (overlaps) pomiędzy segmentami z dwóch kanałów:
    - `left`  (np. AG)
    - `right` (np. CL)

    Overlap rozumiany jest jako wspólna część przedziałów czasowych segmentów:
        overlap = min(L.end, R.end) - max(L.start, R.start)

    Algorytm:
      - Zakłada, że segmenty są posortowane po czasie (start rosnąco).
      - Używa "two-pointer sweep": porównuje bieżący segment z lewej i prawej,
        zapisuje overlap jeśli spełnia progi, a następnie przesuwa wskaźnik tego
        segmentu, który kończy się wcześniej.
      - Dzięki temu działa w czasie O(n+m) zamiast O(n*m).

    Zwraca listę zdarzeń overlap, gdzie każde zdarzenie ma m.in.:
      - overlap_start / overlap_end (sekundy)
      - overlap_sec (float)
      - left_seg / right_seg (oryginalne segmenty wejściowe)

    Parametry:

    left, right:
      Listy segmentów w formacie:
        {"start": float_seconds, "end": float_seconds, "text": str}
      W praktyce: wynik `_get_segments()`.

    min_overlap_ms (int, domyślnie 200):
      Minimalna długość nakładania (w milisekundach), żeby uznać to za overlap.
      - Podnieś (np. 400–600), aby ignorować krótkie wejścia typu "tak/mhm/aha".
      - Obniż (np. 150–250), aby łapać bardzo krótkie, ale realne nakładania.

    eps_ms (int, domyślnie 30):
      Tolerancja (w milisekundach) na niedokładność timestampów z ASR/diaryzacji.
      W praktyce sprawdzamy warunek:
        overlap + eps >= min_overlap
      - Zwiększ (np. 50–80), jeśli widzisz, że model "przesuwa" czasy i overlap
        bywa o kilkadziesiąt ms krótszy niż w rzeczywistości.
      - Zmniejsz (np. 10–20), jeśli chcesz bardziej "twarde" decyzje.

    min_segment_ms_left / min_segment_ms_right (int, domyślnie 0):
      Minimalna długość segmentu (w ms) po danej stronie, aby segment brał udział
      w wykrywaniu overlapów.
      Przykład:
        min_segment_ms_right=350
      spowoduje, że krótkie segmenty klienta (np. "tak") nie będą generować overlapów,
      nawet jeśli czasowo nachodzą na segmenty AG.
      UWAGA: to filtruje po *całkowitej długości segmentu*, a nie po długości overlapu.

    min_words_left / min_words_right (int, domyślnie 0):
      Minimalna liczba słów w tekście segmentu po danej stronie, aby segment był brany pod uwagę.
      Pomaga ignorować "backchannel":
        "tak", "mhm", "aha", "ok"
      Typowe ustawienia:
        - 2 (ignoruje pojedyncze słowa)
        - 3 (bardziej agresywne, może ucinać krótkie ale istotne zdania)

    Jak stroić (praktyka):
      - Jeśli chcesz ignorować krótkie wtrącenia:
          min_overlap_ms=300..500
          min_segment_ms_right=250..400 (często bardziej sensowne niż min_words)
          min_words_right=2
      - Jeśli chcesz łapać prawie wszystko:
          min_overlap_ms=150..250
          min_segment_ms_left/right=0
          min_words_left/right=0
      - Jeśli overlapy "znikają" przez jitter czasów:
          zwiększ eps_ms (np. 50)

    Ważne:
      Ta funkcja wykrywa *dowolne* nakładania (nie tylko "kto zaczął w trakcie").
      Do stricte "przerywania" (starter startuje podczas gdy drugi mówi) używaj
      `detect_starts_while_other_speaks()`.
    """    
    
    
    min_overlap = min_overlap_ms / 1000.0
    eps = eps_ms / 1000.0

    min_seg_left = min_segment_ms_left / 1000.0
    min_seg_right = min_segment_ms_right / 1000.0

    i = j = 0
    out: List[Dict[str, Any]] = []

    def left_ok(seg: Dict[str, Any]) -> bool:
        dur = float(seg["end"]) - float(seg["start"])
        if dur + eps < min_seg_left:
            return False
        if _word_count(str(seg.get("text", ""))) < min_words_left:
            return False
        return True

    def right_ok(seg: Dict[str, Any]) -> bool:
        dur = float(seg["end"]) - float(seg["start"])
        if dur + eps < min_seg_right:
            return False
        if _word_count(str(seg.get("text", ""))) < min_words_right:
            return False
        return True

    while i < len(left) and j < len(right):
        L = left[i]
        R = right[j]

        # Overlap math
        s = max(float(L["start"]), float(R["start"]))
        e = min(float(L["end"]), float(R["end"]))
        overlap = e - s

        # Apply filters BEFORE accepting overlap
        if overlap + eps >= min_overlap:
            if left_ok(L) and right_ok(R):
                out.append(
                    {
                        "overlap_start": s,
                        "overlap_end": e,
                        "overlap_sec": round(overlap, 3),
                        "left_seg": L,
                        "right_seg": R,
                    }
                )

        # advance whichever ends first
        if float(L["end"]) < float(R["end"]):
            i += 1
        else:
            j += 1

    return out


def analyze_stereo_lr_diarized(
    audio_file: str,
    *,
    min_overlap_ms: int = 250,
    eps_ms: int = 30,
    min_left_segment_ms: int = 200,
    min_right_segment_ms: int = 200,
    min_other_lead_ms: int = 0,
) -> Dict[str, Any]:
    left_wav, right_wav = split_stereo_to_lr_and_clean(audio_file)
    if not left_wav or not right_wav:
        raise ValueError("File is not stereo or channel split failed.")

    # diarized transcription per channel
    left_tr = transcript_audio_file_verbose_o4_diarize(file_name=left_wav)
    right_tr = transcript_audio_file_verbose_o4_diarize(file_name=right_wav)

    left_segs = _get_segments(left_tr)
    right_segs = _get_segments(right_tr)

    # print channel segments
    print("\n\n\n" + format_channel_segments(left_segs, "LEFT"))
    print("\n\n\n" + format_channel_segments(right_segs, "RIGHT"))

    # ANY overlaps (main thing you asked to fix)
    overlaps = detect_any_overlaps(
        left_segs, right_segs, min_overlap_ms=min_overlap_ms, eps_ms=eps_ms
    )

    overlaps = detect_any_overlaps(
        left_segs,
        right_segs,
        min_overlap_ms=min_overlap_ms,
        eps_ms=eps_ms,
        min_segment_ms_left=250,    # np. ignoruj AG segmenty krótsze niż 350ms
        min_segment_ms_right=250,   # ignoruj CL segmenty krótsze niż 350ms
        min_words_left=2,           # ignoruj AG wypowiedzi < 2 słów
        min_words_right=2,          # ignoruj CL wypowiedzi < 2 słów
    )

    print(f"\n\n\n=== Any overlaps >= {min_overlap_ms}ms ({len(overlaps)}) ===")
    for idx, o in enumerate(overlaps, 1):
        print(
            f"[{idx:03d}] {fmt_ts(o['overlap_start'])}–{fmt_ts(o['overlap_end'])} "
            f"({o['overlap_sec']}s) | "
            f"L: {o['left_seg']['text']} | R: {o['right_seg']['text']}"
        )

    # optional: interruptions (start-inside) in both directions
    left_starts_while_right = detect_starts_while_other_speaks(
        starter_segs=left_segs,
        other_segs=right_segs,
        min_overlap_ms=min_overlap_ms,
        min_starter_segment_ms=min_left_segment_ms,
        min_other_lead_ms=min_other_lead_ms,
        eps_ms=eps_ms,
    )
    right_starts_while_left = detect_starts_while_other_speaks(
        starter_segs=right_segs,
        other_segs=left_segs,
        min_overlap_ms=min_overlap_ms,
        min_starter_segment_ms=min_right_segment_ms,
        min_other_lead_ms=min_other_lead_ms,
        eps_ms=eps_ms,
    )


    def _print_events(title: str, events: List[Dict[str, Any]]):
        print(f"\n\n\n=== {title} ({len(events)}) ===")
        for i, e in enumerate(events, 1):
            print(
                f"[{i:03d}] START {fmt_ts(e['starter_start'])}  "
                f"overlap={e['overlap_sec']}s lead={e['other_lead_sec']}s | "
                f"starter: {e['starter_text']}  | other: {e['other_text']}"
            )

    _print_events("Left starts while Right is speaking", left_starts_while_right)
    _print_events("Right starts while Left is speaking", right_starts_while_left)

    dialogue = merged_dialogue_text(left_segs, right_segs, include_timestamps=False)
    print(f"\n\n\n=== Merged Dialogue ({len(dialogue)} chars) ===\n{dialogue}")

    return {
        "files": {"left": left_wav, "right": right_wav},
        "segments": {"left": left_segs, "right": right_segs},
        "overlaps": overlaps,
        "events": {
            "left_starts_while_right": left_starts_while_right,
            "right_starts_while_left": right_starts_while_left,
        },
        "stats": {
            "left_segments": len(left_segs),
            "right_segments": len(right_segs),
            "any_overlaps": len(overlaps),
            "left_interruptions": len(left_starts_while_right),
            "right_interruptions": len(right_starts_while_left),
            "min_overlap_ms": min_overlap_ms,
            "eps_ms": eps_ms,
            "min_other_lead_ms": min_other_lead_ms,
        },
    }

def check_audio_for_overlapping(audio_file: str) -> Optional[str]:
    try:
        result = analyze_stereo_lr_diarized(
                audio_file,
                min_overlap_ms=1000,      # lower this if you still miss short overlaps
                eps_ms=20,               # jitter tolerance
                min_left_segment_ms=200,
                min_right_segment_ms=200,
                min_other_lead_ms=0,     # set >0 only if you want "true interruptions"
            )
        print("\n\n\nSTATS:", result["stats"])
        if result["stats"]["any_overlaps"] > 1:
            return "YES"
        return "NO"
    except Exception as e:
        print(f"Error processing file {audio_file}: {e}")   
        return None


files = [
        "./test/test_call.wav",
        "./test/Одночасна розмова фахівця і клієнта.wav",
        "./test/Святослав одночасна розмова з клієнтом , деколи перебиває.wav",
        "./test/sources/AUTO-2025-06-30-10-08-380988442847-1087-1751267274.1529139-stereo1.wav",
        "./test/sources/AUTO-2025-06-30-09-05-380963799218-1096-1751263515.1528148-stereo1.wav",
        "./test/sources/AUTO-2025-06-30-10-17-380639093150-1006-1751267854.1529303-stereo1.wav",
        "./test/sources/AUTO-2025-06-30-12-05-380990805468-1098-1751274275.1530761-stereo1.wav",
        "./test/sources/OUT-2025-06-30-09-34-1099-0500814269-1751265256.1528626-stereo1.wav" 
    ]

if __name__ == "__main__":

    for file_name in files:
        print(f"\n\n\n#########################\nFILE: '{file_name}'\n#########################")
        result = check_audio_for_overlapping(file_name)
        print(f"File: {file_name} -> Overlapping speech detected? {result}\n\n")


